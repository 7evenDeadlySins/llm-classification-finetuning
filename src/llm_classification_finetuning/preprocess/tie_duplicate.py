"""Generate audited train-partition or full-data real tie duplicate supplements."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from ..config import AppConfig
from ..errors import DataPreparationError
from ..model_storage import load_pretrained
from ..models.common import (
	BuildSupplementLoader,
	CacheStorage,
	ConversationSerializer,
	TokenizationCache,
	TrainingInputLoader,
	TrainingInputs,
	staged_output_directory,
)


class TieDuplicateGenerator:
	"""Duplicate prioritized distinct-response real tie rows once per prompt."""

	ALGORITHM_VERSION = "tie-duplicate-v1"
	SOURCE_LABEL = 2
	REPEAT = 1
	EXCLUDE_IDENTICAL_AB = True
	MAX_PER_PROMPT = 1
	DEFAULT_TARGET_COUNT = 7343

	def __init__(
		self,
		config: AppConfig,
		*,
		scope: str = "train",
		target_count: int = DEFAULT_TARGET_COUNT,
	) -> None:
		if scope not in {"train", "build"}:
			raise DataPreparationError("Supplement scope must be 'train' or 'build'.")
		if (
			isinstance(target_count, bool)
			or not isinstance(target_count, int)
			or target_count <= 0
		):
			raise DataPreparationError("target_count must be a positive integer.")
		target_path = getattr(config.data, f"{scope}_supplement_path")
		if target_path is None:
			raise DataPreparationError(
				f"data.{scope}_supplement_path must be configured for {scope} supplement generation."
			)
		self.config = config
		self.scope = scope
		self.target_count = target_count
		self.target = target_path.resolve()
		self.output = self.target.parent
		self.filename = self.target.name
		self._require(
			not config.data.processed_path.is_relative_to(self.output)
			and not Path.cwd().resolve().is_relative_to(self.output),
			"The supplement output directory must not contain the original dataset or project root.",
		)
		if self.output.exists() and any(self.output.iterdir()):
			summary = self.output / "summary.json"
			self._require(
				summary.is_file()
				and json.loads(summary.read_text(encoding="utf-8")).get("kind")
				== BuildSupplementLoader.TIE_DUPLICATE_KIND,
				"Refusing to replace an output directory not owned by this generator.",
			)
		self.report: dict[str, Any] = {
			"algorithm_version": self.ALGORITHM_VERSION,
			"kind": BuildSupplementLoader.TIE_DUPLICATE_KIND,
			"schema_version": BuildSupplementLoader.DUPLICATE_SCHEMA_VERSION,
			"scope": self.scope,
			"source_label": self.SOURCE_LABEL,
			"repeat": self.REPEAT,
			"exclude_identical_ab": self.EXCLUDE_IDENTICAL_AB,
			"max_per_prompt": self.MAX_PER_PROMPT,
			"target_count": self.target_count,
			"selection_policy": {
				"order": [
					"prompt_count_in_scope_asc",
					"is_multiturn_desc",
					"tokens_a_plus_b_asc",
					"original_id_asc",
				],
				"max_per_prompt": self.MAX_PER_PROMPT,
				"exclude_identical_ab": self.EXCLUDE_IDENTICAL_AB,
			},
			"status": "started",
		}

	@staticmethod
	def _require(condition: bool, message: str) -> None:
		if not condition:
			raise DataPreparationError(message)

	@staticmethod
	def _digest(path: Path) -> str:
		digest = hashlib.sha256()
		with path.open("rb") as source:
			for chunk in iter(lambda: source.read(1024 * 1024), b""):
				digest.update(chunk)
		return digest.hexdigest()

	@staticmethod
	def _json(path: Path, value: object) -> None:
		path.write_text(
			json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
		)

	def run(self) -> Path:
		"""Build, verify, and atomically publish the prioritized tie duplicate supplement."""
		with staged_output_directory(self.output) as stage:
			original, source_sha256 = self._load_original()
			supplement = self._construct(original, source_sha256)
			self._write(stage, supplement, source_sha256)
			self._verify(stage, supplement, original, source_sha256)
			self._finalize(stage)
		print(
			f"Published {self.scope} tie duplicate supplement: {self.target}",
			flush=True,
		)
		return self.output

	def _load_original(self) -> tuple[pd.DataFrame, str]:
		with CacheStorage(self.config.data.processed_path).open("r") as session:
			original = pd.read_parquet(session.path)
			source_sha256 = self._digest(session.path)
		TrainingInputLoader.validate(
			original, validation_fraction=self.config.validation.validation_fraction
		)
		return original, source_sha256

	def _attach_token_totals(
		self, candidates: pd.DataFrame, source_sha256: str
	) -> pd.Series:
		config = self.config.validation
		tokenizer = load_pretrained(
			AutoTokenizer.from_pretrained,
			config.tokenizer_name,
			revision=config.tokenizer_revision,
			local_files_only=True,
		)
		serializer = ConversationSerializer(tokenizer)
		cache = TokenizationCache(self.config.cache_dir / "token_cache", tokenizer)
		self.report["tokenizer"] = {
			"name": config.tokenizer_name,
			"revision": config.tokenizer_revision,
			"class": type(tokenizer).__name__,
			"tokenizer_fingerprint": cache.tokenizer_key,
			"serializer_version": serializer.FORMAT_VERSION,
			"chat_template_sha256": hashlib.sha256(
				tokenizer.get_chat_template().encode("utf-8")
			).hexdigest(),
			"add_special_tokens": False,
			"truncation": False,
			"padding": False,
		}
		reused = self._reuse_token_stats(candidates["id"], source_sha256)
		if reused is not None:
			return reused
		totals: list[int] = []
		batch_size = 4096
		rows = list(
			candidates.loc[:, ["id", "prompt", "response_a", "response_b"]].itertuples(
				index=False
			)
		)
		for start in range(0, len(rows), batch_size):
			batch = rows[start : start + batch_size]
			pairs = [
				serializer.serialize_pair(
					int(row.id), row.prompt, row.response_a, row.response_b
				)
				for row in batch
			]
			lengths_a = cache.count_tokens([pair.response_a for pair in pairs])
			lengths_b = cache.count_tokens([pair.response_b for pair in pairs])
			for len_a, len_b in zip(lengths_a, lengths_b, strict=True):
				total = int(len_a) + int(len_b)
				self._require(
					int(len_a) > 0 and int(len_b) > 0,
					"Complete branch token counts must be positive.",
				)
				totals.append(total)
		return pd.Series(totals, index=candidates.index, dtype="int64")

	def _reuse_token_stats(
		self, candidate_ids: pd.Series, source_sha256: str
	) -> pd.Series | None:
		"""Reuse pinned complete branch token sums from the real-data merger."""
		root = self.config.data.processed_path.parent
		summary_path, stats_path = root / "summary.json", root / "token_stats.parquet"
		if not summary_path.is_file() or not stats_path.is_file():
			return None
		summary = json.loads(summary_path.read_text(encoding="utf-8"))
		identity = self.report["tokenizer"]
		tokenizer = summary.get("tokenizer", {})
		if summary.get("processed_reconstructed_sha256") != source_sha256 or any(
			tokenizer.get(name) != identity.get(name)
			for name in (
				"name",
				"revision",
				"tokenizer_fingerprint",
				"serializer_version",
				"chat_template_sha256",
				"add_special_tokens",
				"truncation",
				"padding",
			)
		):
			return None
		expected = (
			summary.get("artifacts", {}).get("token_stats.parquet", {}).get("sha256")
		)
		actual = self._digest(stats_path)
		self._require(
			actual == expected, "Pinned real-data token statistics checksum mismatch."
		)
		stats = pd.read_parquet(
			stats_path, columns=["id", "tokens_a", "tokens_b", "retained"]
		)
		stats = stats.loc[stats["retained"]].set_index("id")
		if not stats.index.is_unique or not set(candidate_ids).issubset(stats.index):
			return None
		matched = stats.loc[candidate_ids]
		self._require(
			bool(matched["tokens_a"].gt(0).all() and matched["tokens_b"].gt(0).all()),
			"Complete branch token counts must be positive.",
		)
		totals = (matched["tokens_a"] + matched["tokens_b"]).astype("int64")
		totals.index = candidate_ids.index
		self.report["token_statistics_sha256"] = actual
		print(
			f"Reused {len(candidate_ids):,} complete branch token totals from pinned real-data statistics.",
			flush=True,
		)
		return totals

	def _construct(self, original: pd.DataFrame, source_sha256: str) -> pd.DataFrame:
		scope_data = (
			original.loc[original["split"].eq("train")]
			if self.scope == "train"
			else original
		)
		prompt_counts = scope_data["group_id"].value_counts()
		all_ties = scope_data.loc[scope_data["label"].eq(self.SOURCE_LABEL)]
		identical_ab = all_ties["response_a"].eq(all_ties["response_b"])
		excluded_identical_ab_rows = int(identical_ab.sum())
		candidates = all_ties.loc[~identical_ab].copy()
		eligible_count = len(candidates)
		self._require(
			eligible_count > 0,
			f"No eligible non-identical-A/B tie rows found for scope '{self.scope}'.",
		)
		candidates["_prompt_count"] = (
			candidates["group_id"].map(prompt_counts).astype("int64")
		)
		candidates["_is_multiturn"] = candidates["prompt"].map(
			lambda value: len(json.loads(value)) > 1
		)
		candidates["_tokens_total"] = self._attach_token_totals(
			candidates, source_sha256
		)

		ranked = candidates.sort_values(
			by=["_prompt_count", "_is_multiturn", "_tokens_total", "id"],
			ascending=[True, False, True, True],
			kind="mergesort",
		)
		per_prompt = ranked.drop_duplicates(subset=["group_id"], keep="first")
		eligible_unique_prompts = len(per_prompt)
		self._require(
			eligible_unique_prompts >= self.target_count,
			f"Insufficient unique-prompt eligible tie rows ({eligible_unique_prompts:,}) for target_count ({self.target_count:,}).",
		)
		sources = per_prompt.iloc[: self.target_count].copy()
		self._require(
			bool(
				sources["winner_model_a"].eq(0).all()
				and sources["winner_model_b"].eq(0).all()
				and sources["winner_tie"].eq(1).all()
				and (~sources["is_swapped"]).all()
			),
			"Eligible tie rows have inconsistent winner flags or orientations.",
		)

		prompt_count_dist = {
			str(k): int(v)
			for k, v in sources["_prompt_count"].value_counts().sort_index().items()
		}
		multiturn_rows = int(sources["_is_multiturn"].sum())
		token_values = sources["_tokens_total"]
		token_summary = {
			"min": int(token_values.min()),
			"max": int(token_values.max()),
			"mean": float(token_values.mean()),
			"sum": int(token_values.sum()),
		}
		source_dataset_counts = (
			dict(Counter(sources["source"].astype(str)))
			if "source" in sources.columns
			else {"unknown": len(sources)}
		)

		sources = sources.drop(
			columns=["_prompt_count", "_is_multiturn", "_tokens_total"]
		)
		start = int(original["id"].max()) + 1
		count = len(sources)
		self._require(
			start + count - 1 <= np.iinfo(np.int64).max,
			"Duplicate IDs exceed int64.",
		)
		supplement = sources.reset_index(drop=True).copy()
		supplement["source_original_id"] = supplement["id"].astype("int64")
		supplement["source_split"] = supplement["split"].astype(str)
		supplement["source_dataset"] = (
			supplement["source"].astype(str)
			if "source" in supplement.columns
			else "unknown"
		)
		supplement["id"] = np.arange(start, start + count, dtype="int64")
		supplement["split"] = self.scope
		supplement["is_swapped"] = False
		supplement["source"] = BuildSupplementLoader.TIE_DUPLICATE_KIND

		before_ties = int(scope_data["label"].eq(2).sum())
		after_ties = before_ties + count
		before_total = len(scope_data)
		after_total = before_total + count
		before_share = Fraction(before_ties, before_total)
		after_share = Fraction(after_ties, after_total)
		self.report.update(
			source_path=str(self.config.data.processed_path),
			source_sha256=source_sha256,
			target_path=str(self.target),
			original_rows=before_total,
			original_splits=dict(Counter(scope_data["split"])),
			class_counts_before={
				str(i): int(scope_data["label"].eq(i).sum()) for i in range(3)
			},
			tie_share_before=str(before_share),
			tie_pct_before=float(before_share * 100),
			total_tie_rows=len(all_ties),
			excluded_identical_ab_rows=excluded_identical_ab_rows,
			eligible_tie_rows=eligible_count,
			eligible_unique_prompts=eligible_unique_prompts,
			added_rows=count,
			new_ids={"first": start, "last": start + count - 1},
			source_datasets=source_dataset_counts,
			source_splits=dict(Counter(supplement["source_split"])),
			selected_prompt_count_distribution=prompt_count_dist,
			selected_multiturn_rows=multiturn_rows,
			selected_single_turn_rows=count - multiturn_rows,
			selected_tokens_total=token_summary,
			class_counts_after={
				str(i): int(scope_data["label"].eq(i).sum()) + (count if i == 2 else 0)
				for i in range(3)
			},
			total_rows_after=after_total,
			tie_share_after=str(after_share),
			tie_pct_after=float(after_share * 100),
		)
		return supplement

	def _write(self, stage: Path, supplement: pd.DataFrame, source_sha256: str) -> None:
		identity = {
			"schema_version": BuildSupplementLoader.DUPLICATE_SCHEMA_VERSION,
			"kind": BuildSupplementLoader.TIE_DUPLICATE_KIND,
			"scope": self.scope,
			"source_sha256": source_sha256,
			"source_label": self.SOURCE_LABEL,
			"repeat": self.REPEAT,
			"exclude_identical_ab": self.EXCLUDE_IDENTICAL_AB,
			"max_per_prompt": self.MAX_PER_PROMPT,
			"target_count": len(supplement),
			"algorithm_version": self.ALGORITHM_VERSION,
		}
		table = pa.Table.from_pandas(supplement, preserve_index=False)
		table = table.replace_schema_metadata(
			{
				**(table.schema.metadata or {}),
				BuildSupplementLoader.DUPLICATE_METADATA_KEY: json.dumps(
					identity, sort_keys=True
				).encode("utf-8"),
			}
		)
		with CacheStorage(stage / self.filename).open("w") as session:
			pq.write_table(table, session.path, compression="zstd")

	def _verify(
		self,
		stage: Path,
		expected: pd.DataFrame,
		original: pd.DataFrame,
		source_sha256: str,
	) -> None:
		loaded = BuildSupplementLoader(stage / self.filename, scope=self.scope).load(
			TrainingInputs(original, source_sha256)
		)
		self._require(
			loaded.data.equals(expected),
			"Supplement Parquet roundtrip changed values, order, or dtypes.",
		)
		with CacheStorage(self.config.data.processed_path).open("r") as session:
			self._require(
				self._digest(session.path) == source_sha256,
				"Original dataset changed during generation; publication cancelled.",
			)
		self.report["supplement_sha256"] = loaded.fingerprint
		self.report["training_source_sha256"] = (
			BuildSupplementLoader.combined_fingerprint(
				source_sha256, loaded.fingerprint
			)
		)
		self.report["checks"] = {
			"original_file_values_order_and_partitions_unchanged": True,
			"supplement_roundtrip_values_order_dtypes_preserved": True,
			"unique_ids_without_original_collisions": True,
			"unique_prompts_at_most_one_per_prompt": True,
			"non_identical_ab_ties_only": True,
			"original_orientation_ties_only": True,
			"scope_source_membership_and_validation_isolation": True,
		}

	def _finalize(self, stage: Path) -> None:
		self.report.update(
			status="completed", completed_at=datetime.now(UTC).isoformat()
		)
		files = [p for p in sorted(stage.rglob("*")) if p.is_file()]
		self.report["artifacts"] = {
			p.relative_to(stage).as_posix(): {
				"bytes": p.stat().st_size,
				"sha256": self._digest(p),
			}
			for p in files
		}
		self._json(stage / "summary.json", self.report)
		files.append(stage / "summary.json")
		(stage / "checksums.sha256").write_text(
			"".join(
				f"{self._digest(p)}  {p.relative_to(stage).as_posix()}\n" for p in files
			),
			encoding="utf-8",
		)
