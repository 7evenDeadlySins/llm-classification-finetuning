"""Build a frozen-holdout, real-only dataset from pinned human preference releases."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoConfig

from .config import AppConfig
from .data import EXTERNAL_SOURCES, ExternalSourceDownloader, file_sha256
from .errors import DataPreparationError
from .model_storage import load_pretrained
from .models.common import (
	CacheStorage,
	ConversationSerializer,
	TokenizationCache,
	TrainingInputLoader,
	staged_output_directory,
)
from .preprocess import (
	TARGET_COLUMNS,
	ConversationExclusion,
	ExternalBattleDeduplicator,
	ExternalConversationConverter,
	ExternalVoteConflictError,
)


class MergeExternalDataPipeline:
	"""Stage, audit, and publish original hard-label rows without rebuilding the holdout."""

	ALGORITHM = "external-hard-v1"
	TOKENIZER = "Skywork/Skywork-Reward-V2-Qwen3-4B"
	TOKENIZER_REVISION = "fd958fef475f323f4e6b195930e3dd918485c668"
	PREVIOUS_MAX_LENGTH = 13866
	CHUNK_ROWS = 8192

	def __init__(
		self, config: AppConfig, *, baseline_path: Path, output_dir: Path
	) -> None:
		self.config = config
		self.baseline_path = baseline_path.resolve()
		self.output = output_dir.resolve()
		if self.baseline_path.is_relative_to(
			self.output
		) or Path.cwd().resolve().is_relative_to(self.output):
			raise DataPreparationError(
				"The output directory must not contain the baseline or project root."
			)
		if self.output.exists():
			summary = self.output / "summary.json"
			if (
				not summary.is_file()
				or json.loads(summary.read_text(encoding="utf-8")).get(
					"algorithm_version"
				)
				!= self.ALGORITHM
			):
				raise DataPreparationError(
					"Refusing to replace a directory not owned by this merger."
				)
		if (
			config.validation.tokenizer_name != self.TOKENIZER
			or config.validation.tokenizer_revision != self.TOKENIZER_REVISION
		):
			raise DataPreparationError(
				"External merging requires the pinned SR2 tokenizer and revision."
			)
		self.report = {
			"algorithm_version": self.ALGORITHM,
			"status": "started",
			"hard_labels_only": True,
			"synthetic_rows": 0,
		}

	@staticmethod
	def _require(condition: bool, message: str) -> None:
		if not condition:
			raise DataPreparationError(message)

	@staticmethod
	def _json(path: Path, value: object) -> None:
		path.write_text(
			json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
		)

	@staticmethod
	def _read(path: Path) -> pd.DataFrame:
		with CacheStorage(path).open("r") as session:
			return pd.read_parquet(session.path)

	@staticmethod
	def _write(path: Path, frame: pd.DataFrame) -> str:
		with CacheStorage(path).open("w") as session:
			frame.to_parquet(session.path, index=False, compression="zstd")
			return file_sha256(session.path)

	def run(self) -> Path:
		with staged_output_directory(self.output) as stage:
			manifest = ExternalSourceDownloader(
				self.config.data.raw_dir
			).ensure_available()
			self._json(stage / "sources.json", manifest)
			baseline = self._load_baseline(stage)
			external, conversion_audit = self._convert(manifest)
			self._write(stage / "conversion_audit.parquet", conversion_audit)
			print(
				f"Merging independent hard votes from {len(external):,} valid external records.",
				flush=True,
			)
			try:
				result = ExternalBattleDeduplicator().deduplicate(
					external, baseline, merge_votes=True
				)
			except ExternalVoteConflictError as error:
				conflicts_path = self.output.with_name("vote_conflicts.parquet")
				self._write(conflicts_path, pd.DataFrame(error.conflicts))
				print(f"Wrote vote conflicts diagnostic: {conflicts_path}", flush=True)
				raise
			candidates = pd.concat([baseline, result.data], ignore_index=True)
			stats, identity = self._lengths(candidates)
			self._require(
				bool(stats[["tokens_a", "tokens_b"]].gt(0).all().all()),
				"An encoded branch is empty.",
			)
			stats["exceeds_previous_limit"] = (
				stats[["tokens_a", "tokens_b"]].max(axis=1).gt(self.PREVIOUS_MAX_LENGTH)
			)
			audit = result.audit.copy()
			votes = result.vote_counts.copy()
			data = self._filter_context(candidates, stats, audit, votes, stage)
			identity.update(
				max_length=int(
					stats.loc[stats["retained"], ["tokens_a", "tokens_b"]].max().max()
				),
				length_filter=True,
				model_context_limit=self.report["model_context"][
					"max_position_embeddings"
				],
			)
			data["id"] = data["id"].astype("int64")
			data["label"] = data["label"].astype("int8")
			data["is_swapped"] = data["is_swapped"].astype(bool)
			for name in TARGET_COLUMNS:
				data[name] = data[name].astype("int64")
			fraction = len(baseline.loc[baseline["split"].eq("validation")]) / len(data)
			self._verify(data, baseline, fraction)
			self._write(stage / "dedup_audit.parquet", audit)
			self._write(stage / "vote_counts.parquet", votes)
			self._write(stage / "token_stats.parquet", stats)
			print(
				f"Publishing {len(data):,} rows ({len(data) - 11495:,} train, 11,495 validation).",
				flush=True,
			)
			checksum = self._write(stage / "train_validation.parquet", data)
			loaded = TrainingInputLoader(
				stage / "train_validation.parquet", validation_fraction=fraction
			).load()
			self._require(
				loaded.fingerprint == checksum,
				"Published Parquet checksum differs from the staged data.",
			)
			full_readback = self._read(stage / "train_validation.parquet")
			self._require(
				full_readback.equals(data),
				"Published Parquet values, order, or dtypes changed on readback.",
			)
			self._verify(full_readback, baseline, fraction)
			row_audit = conversion_audit.merge(
				audit,
				on="id",
				how="left",
				suffixes=("", "_dedup"),
				validate="one_to_one",
			)
			row_audit["final_status"] = row_audit["status_dedup"].fillna(
				row_audit["status"]
			)
			self._write(stage / "row_audit.parquet", row_audit)
			self.report.update(
				status="completed",
				completed_at=datetime.now(UTC).isoformat(),
				logical_processed_path=str(self.output / "train_validation.parquet"),
				processed_rows=len(data),
				training_rows=len(data) - 11495,
				validation_rows=11495,
				effective_full_table_validation_fraction=fraction,
				processed_reconstructed_sha256=checksum,
				validation_ids_sha256=file_sha256(stage / "validation_ids.csv"),
				tokenizer=identity,
				class_counts={str(i): int(data["label"].eq(i).sum()) for i in range(3)},
				source_split_statistics=self._statistics(data, stats),
				input_dispositions={
					str(k): int(v)
					for k, v in row_audit["final_status"].value_counts().items()
				},
				vote_tied_battles=int(votes["status"].eq("excluded_vote_tie").sum()),
				length_excluded_battles=int((~stats["retained"]).sum()),
				retained_rows_above_previous_limit=int(
					(stats["exceeds_previous_limit"] & stats["retained"]).sum()
				),
				max_branch_tokens=int(
					stats.loc[stats["retained"], ["tokens_a", "tokens_b"]].max().max()
				),
				checks={
					"frozen_kaggle_rows_preserved": True,
					"frozen_validation_ids_preserved": True,
					"unique_original_ids": True,
					"hard_one_hot_labels": True,
					"no_cross_partition_prompt_overlap": True,
					"unique_external_battles": True,
					"all_retained_branches_nonempty": True,
					"all_retained_branches_within_model_context": True,
					"no_truncation": True,
					"parquet_readback_identical": True,
					"training_input_loader_succeeded": True,
				},
			)
			self._write_readme(stage)
			files = [p for p in sorted(stage.rglob("*")) if p.is_file()]
			self.report["artifacts"] = {
				p.relative_to(stage).as_posix(): {
					"bytes": p.stat().st_size,
					"sha256": file_sha256(p),
				}
				for p in files
			}
			self._json(stage / "summary.json", self.report)
			files.append(stage / "summary.json")
			(stage / "checksums.sha256").write_text(
				"".join(
					f"{file_sha256(p)}  {p.relative_to(stage).as_posix()}\n"
					for p in files
				),
				encoding="utf-8",
			)
			for p in files:
				expected = (
					self.report["artifacts"]
					.get(p.relative_to(stage).as_posix(), {})
					.get("sha256")
				)
				if expected is not None:
					self._require(
						file_sha256(p) == expected,
						f"Artifact checksum changed before publication: {p.name}",
					)
		print(f"Published merged dataset: {self.output}", flush=True)
		return self.output

	def _filter_context(
		self,
		data: pd.DataFrame,
		stats: pd.DataFrame,
		audit: pd.DataFrame,
		votes: pd.DataFrame,
		stage: Path,
	) -> pd.DataFrame:
		"""Exclude whole battles exceeding the pinned backbone's per-branch limit."""
		model = load_pretrained(
			AutoConfig.from_pretrained, self.TOKENIZER, revision=self.TOKENIZER_REVISION
		)
		limit = model.max_position_embeddings
		self._require(
			isinstance(limit, int) and not isinstance(limit, bool) and limit > 0,
			"The pinned model must declare a positive max_position_embeddings.",
		)
		self._require(
			stats["id"].tolist() == data["id"].tolist(),
			"Token statistics do not align with the candidate rows.",
		)
		excluded = stats[["tokens_a", "tokens_b"]].max(axis=1).gt(limit)
		self._require(
			not bool((excluded & data["source"].eq("kaggle")).any()),
			"A frozen Kaggle row exceeds the model context; refusing to alter the baseline.",
		)
		stats["retained"] = ~excluded
		stats["exceeds_model_context"] = excluded
		excluded_ids = set(data.loc[excluded, "id"])
		# Preserve the vote outcome and deduplication disposition for traceability.
		audit["dedup_status"] = audit["status"]
		audit["exceeds_model_context"] = audit["retained_id"].isin(excluded_ids)
		audit.loc[audit["exceeds_model_context"], "status"] = "excluded_model_context"
		votes.loc[votes["representative_id"].isin(excluded_ids), "status"] = (
			"excluded_model_context"
		)
		self._write(stage / "excluded_model_context.parquet", data.loc[excluded])
		self.report.update(
			model_context={
				"name": self.TOKENIZER,
				"revision": self.TOKENIZER_REVISION,
				"max_position_embeddings": limit,
				"policy": "exclude_whole_battle_if_either_complete_branch_exceeds_limit",
			},
			candidate_max_branch_tokens=int(
				stats[["tokens_a", "tokens_b"]].max().max()
			),
		)
		return data.loc[~excluded].reset_index(drop=True)

	def _load_baseline(self, stage: Path) -> pd.DataFrame:
		with CacheStorage(self.baseline_path).open("r") as session:
			self.report["baseline_reconstructed_sha256"] = file_sha256(session.path)
			data = pd.read_parquet(session.path)
		baseline = data.loc[data["source"].eq("kaggle")].copy().reset_index(drop=True)
		self._require(len(baseline) == 57477, "Expected all 57,477 frozen Kaggle rows.")
		TrainingInputLoader.validate(baseline)
		frozen = self.baseline_path.parent / "validation_ids.csv"
		ids = pd.read_csv(frozen)["id"].astype("int64")
		actual = baseline.loc[baseline["split"].eq("validation"), "id"]
		self._require(
			len(ids) == 11495 and ids.is_unique and set(ids) == set(actual),
			"Frozen validation IDs differ from the saved Kaggle baseline.",
		)
		shutil.copyfile(frozen, stage / "validation_ids.csv")
		return baseline

	def _convert(self, manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
		rows, audits = [], []
		root = self.config.data.raw_dir.resolve() / "external"
		for source in EXTERNAL_SOURCES:
			converter = ExternalConversationConverter(source)
			index, reasons = 0, Counter()
			for item in (f for f in manifest["files"] if f["source"] == source.name):
				path = root / item["local_path"]
				split = "validation" if "validation" in item["filename"] else "train"
				for raw in self._source_rows(path, source.name):
					entry = {
						"id": source.id_segment * 2**32 + index,
						"source": source.name,
						"source_row_index": index,
						"source_split": split,
						"source_id": raw.get("id", raw.get("question_id")),
						"source_winner": str(raw.get("winner")),
						"status": "converted",
						"detail": "",
					}
					try:
						row = converter.convert(raw, index, split)
						rows.append(row)
						entry.update(
							source_id=row["source_id"],
							original_label=row["original_label"],
							corrected_label=row["label"],
							tie_corrected=row["tie_corrected"],
						)
					except ConversationExclusion as error:
						entry.update(status=error.reason, detail=str(error)[:500])
						reasons[error.reason] += 1
					audits.append(entry)
					index += 1
					if index % 10000 == 0:
						print(
							f"Converted {source.name}: {index:,}/{source.expected_rows:,}; exclusions={dict(reasons)}",
							flush=True,
						)
			self._require(
				index == source.expected_rows,
				f"Unexpected input row count for {source.name}: {index:,}.",
			)
			print(
				f"Completed {source.name}: {index:,} input rows; exclusions={dict(reasons)}",
				flush=True,
			)
		return pd.DataFrame(rows), pd.DataFrame(audits)

	@staticmethod
	def _source_rows(path: Path, source: str):
		columns = [
			"model_a",
			"model_b",
			"winner",
			"conversation_a",
			"conversation_b",
			"language",
		]
		columns += (
			["question_id", "judge", "tstamp"]
			if source == "lmsys"
			else ["id", "evaluation_order", "full_conversation"]
		)
		for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=columns):
			yield from batch.to_pylist()

	def _lengths(self, data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
		cache = TokenizationCache.from_pretrained(
			self.config.cache_dir / "token_cache",
			self.TOKENIZER,
			revision=self.TOKENIZER_REVISION,
		)
		serializer = ConversationSerializer(cache.tokenizer)
		stats = data[["id", "source", "split", "label", "group_id"]].copy()
		lengths = []
		for start in range(0, len(data), self.CHUNK_ROWS):
			chunk = data.iloc[start : start + self.CHUNK_ROWS]
			pairs = [
				serializer.serialize_pair(
					int(r.id), r.prompt, r.response_a, r.response_b
				)
				for r in chunk.itertuples(index=False)
			]
			encoded_lengths = cache.count_tokens(
				[text for pair in pairs for text in (pair.response_a, pair.response_b)],
				add_special_tokens=False,
			)
			lengths.append(encoded_lengths.reshape(-1, 2))
			print(
				f"Counted complete branch tokens: {min(start + self.CHUNK_ROWS, len(data)):,}/{len(data):,}",
				flush=True,
			)
		values = np.concatenate(lengths)
		stats["tokens_a"], stats["tokens_b"] = values[:, 0], values[:, 1]
		stats["turns"] = data["prompt"].map(lambda value: len(json.loads(value)))
		return stats, {
			"name": self.TOKENIZER,
			"revision": self.TOKENIZER_REVISION,
			"tokenizer_fingerprint": cache.tokenizer_key,
			"serializer_version": serializer.FORMAT_VERSION,
			"chat_template_sha256": hashlib.sha256(
				cache.tokenizer.get_chat_template().encode("utf-8")
			).hexdigest(),
			"add_special_tokens": False,
			"truncation": False,
			"padding": False,
			"max_length": int(stats[["tokens_a", "tokens_b"]].max().max()),
			"previous_max_length": self.PREVIOUS_MAX_LENGTH,
			"length_filter": False,
			"length_counter": "cached_complete_counts_v1",
		}

	def _verify(
		self, data: pd.DataFrame, baseline: pd.DataFrame, fraction: float
	) -> None:
		TrainingInputLoader.validate(data, validation_fraction=fraction)
		actual = data.loc[data["source"].eq("kaggle"), baseline.columns].reset_index(
			drop=True
		)
		self._require(
			actual.equals(baseline), "Frozen Kaggle fields, order, or dtypes changed."
		)
		self._require(
			bool(data["split"].eq("train").loc[~data["source"].eq("kaggle")].all()),
			"An external row entered validation.",
		)
		self._require(
			bool(data[list(TARGET_COLUMNS)].isin((0, 1)).all().all())
			and bool(data[list(TARGET_COLUMNS)].sum(axis=1).eq(1).all()),
			"Winner fields are not hard one-hot labels.",
		)
		for i, name in enumerate(TARGET_COLUMNS):
			self._require(
				bool(data[name].eq(data["label"].eq(i).astype(int)).all()),
				"Winner fields and labels disagree.",
			)
		identities = [
			ExternalBattleDeduplicator.battle_identity(
				r.prompt, r.response_a, r.response_b
			)
			for r in data.itertuples(index=False)
		]
		self._require(
			all(
				group == saved
				for (group, _, _), saved in zip(
					identities, data["group_id"], strict=True
				)
			),
			"A saved prompt hash does not match complete prompt JSON.",
		)
		battles = pd.Series([b for _, b, _ in identities])
		external = ~data["source"].eq("kaggle")
		self._require(
			not battles.loc[external].duplicated().any(),
			"External duplicate battles remain.",
		)
		self._require(
			not battles.loc[external].isin(set(battles.loc[~external])).any(),
			"An external battle duplicates Kaggle.",
		)

	@staticmethod
	def _statistics(data: pd.DataFrame, stats: pd.DataFrame) -> list[dict]:
		result = []
		for (source, split), group in data.groupby(["source", "split"], sort=False):
			tokens = stats.loc[stats["id"].isin(group["id"])]
			result.append(
				{
					"source": source,
					"split": split,
					"rows": len(group),
					"prompt_groups": int(group["group_id"].nunique()),
					"class_counts": [int(group["label"].eq(i).sum()) for i in range(3)],
					"tokens_a": int(tokens["tokens_a"].sum()),
					"tokens_b": int(tokens["tokens_b"].sum()),
					"max_branch_tokens": int(
						tokens[["tokens_a", "tokens_b"]].max().max()
					),
					"multi_turn_rows": int(tokens["turns"].gt(1).sum()),
				}
			)
		return result

	def _write_readme(self, stage: Path) -> None:
		(stage / "README.md").write_text(
			"# Merged real human preference data\n\n"
			"Kaggle + complete LMSYS 33K + Arena human + Arena expert. "
			"The 11,495 Kaggle validation IDs and all Kaggle rows are frozen. External rows are train-only. "
			"All labels are hard one-hot targets; rows have equal weight. Independent votes use the unique "
			"highest-count class after A/B alignment; highest-count ties are excluded. Kaggle duplicate battles "
			"take precedence. No synthetic supplement is used.\n\n"
			"Battles exceeding the pinned model's max_position_embeddings on either complete branch "
			"are excluded without truncation and saved in excluded_model_context.parquet. "
			"token_stats.parquet retains all candidate lengths and their retained status.\n\n"
			"Use the project model commands with the current presets. Rebuild from the project root with "
			"`uv run python data/merge_external_data.py`; do not run init/preprocess with these consumption configs.\n\n"
			"Keep train_validation.parquet or its manifest and all parts together. summary.json records "
			"counts and identities; sources.json pins the original releases and checksums. row_audit.parquet "
			"accounts for every raw external row; vote_counts.parquet records merged vote counts in representative "
			"A/B order. conversion_audit.parquet, dedup_audit.parquet, and token_stats.parquet retain stage details.\n",
			encoding="utf-8",
		)
