"""Generate audited train-partition or full-data Kaggle tie duplicate supplements."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import AppConfig
from ..errors import DataPreparationError
from ..models.common import (
	BuildSupplementLoader,
	CacheStorage,
	TrainingInputLoader,
	TrainingInputs,
)


class KaggleTieDuplicateGenerator:
	"""Duplicate distinct-response Kaggle tie rows once as a scope-specific supplement."""

	ALGORITHM_VERSION = "kaggle-tie-duplicate-v3"
	SOURCE_DATASET = "kaggle"
	SOURCE_LABEL = 2
	REPEAT = 1
	EXCLUDE_IDENTICAL_AB = True
	DEFAULT_REMOVE_COUNT = 6654

	def __init__(
		self,
		config: AppConfig,
		*,
		scope: str = "train",
		remove_count: int = DEFAULT_REMOVE_COUNT,
	) -> None:
		if scope not in {"train", "build"}:
			raise DataPreparationError("Supplement scope must be 'train' or 'build'.")
		if (
			isinstance(remove_count, bool)
			or not isinstance(remove_count, int)
			or remove_count < 0
		):
			raise DataPreparationError("remove_count must be a non-negative integer.")
		target_path = getattr(config.data, f"{scope}_supplement_path")
		if target_path is None:
			raise DataPreparationError(
				f"data.{scope}_supplement_path must be configured for {scope} supplement generation."
			)
		self.config = config
		self.scope = scope
		self.remove_count = remove_count
		self.sample_seed = config.validation.random_state
		self.target = target_path.resolve()
		self.output = self.target.parent
		self.filename = self.target.name
		if Path.cwd().resolve().is_relative_to(self.output):
			raise DataPreparationError(
				"The output directory must not contain the project root."
			)
		self.report: dict = {
			"algorithm_version": self.ALGORITHM_VERSION,
			"kind": BuildSupplementLoader.KAGGLE_TIE_DUPLICATE_KIND,
			"schema_version": BuildSupplementLoader.DUPLICATE_SCHEMA_VERSION,
			"scope": self.scope,
			"source_dataset": self.SOURCE_DATASET,
			"source_label": self.SOURCE_LABEL,
			"repeat": self.REPEAT,
			"exclude_identical_ab": self.EXCLUDE_IDENTICAL_AB,
			"remove_count": self.remove_count,
			"sample_seed": self.sample_seed,
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
		"""Build, verify, and atomically publish the Kaggle tie duplicate supplement."""
		work = self.output.parent / ".cache-work"
		work.mkdir(parents=True, exist_ok=True)
		# Inherit the project ACL on Windows instead of using private mkdtemp modes.
		stage = work / f"kaggle-tie-duplicate-{uuid4().hex}"
		stage.mkdir()
		stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
		try:
			original, source_sha256 = self._load_original()
			supplement = self._construct(original, source_sha256)
			self._write(stage, supplement, source_sha256)
			self._verify(stage, supplement, original, source_sha256)
			self._finalize(stage)
			backup = self.output.with_name(f"{self.output.name}.backup-{stamp}")
			if self.output.exists():
				self.output.rename(backup)
			try:
				stage.rename(self.output)
			except BaseException:
				if backup.exists() and not self.output.exists():
					backup.rename(self.output)
				raise
			print(
				f"Published {self.scope} Kaggle tie duplicate supplement: {self.target}",
				flush=True,
			)
			return self.output
		except BaseException as error:
			self.report.update(
				status="failed", error=f"{type(error).__name__}: {error}"
			)
			if stage.exists():
				self._json(stage / "summary.json", self.report)
				failure = self.output.with_name(f"{self.output.name}.failed-{stamp}")
				stage.rename(failure)
				print(f"Generation failed; diagnostic audit: {failure}", flush=True)
			raise

	def _load_original(self) -> tuple[pd.DataFrame, str]:
		with CacheStorage(self.config.data.processed_path).open("r") as session:
			original = pd.read_parquet(session.path)
			source_sha256 = self._digest(session.path)
		TrainingInputLoader.validate(
			original, validation_fraction=self.config.validation.validation_fraction
		)
		return original, source_sha256

	def _construct(self, original: pd.DataFrame, source_sha256: str) -> pd.DataFrame:
		scope_data = (
			original.loc[original["split"].eq("train")]
			if self.scope == "train"
			else original
		)
		kaggle_mask = (
			scope_data["source"].eq(self.SOURCE_DATASET)
			if "source" in scope_data.columns
			else scope_data["id"].lt(1 << 32)
		)
		kaggle_ties = scope_data.loc[
			kaggle_mask & scope_data["label"].eq(self.SOURCE_LABEL)
		]
		identical_ab = kaggle_ties["response_a"].eq(kaggle_ties["response_b"])
		excluded_identical_ab_rows = int(identical_ab.sum())
		sources = kaggle_ties.loc[~identical_ab].copy()
		eligible_count = len(sources)
		self._require(
			eligible_count > 0,
			f"No eligible non-identical-A/B Kaggle tie rows found for scope '{self.scope}'.",
		)
		self._require(
			self.remove_count < eligible_count,
			f"remove_count ({self.remove_count:,}) must be smaller than eligible Kaggle tie rows ({eligible_count:,}).",
		)
		if self.remove_count > 0:
			rng = np.random.default_rng(self.sample_seed)
			keep_count = eligible_count - self.remove_count
			keep_positions = np.sort(
				rng.choice(eligible_count, size=keep_count, replace=False)
			)
			sources = sources.iloc[keep_positions].copy()
		self._require(
			bool(
				sources["winner_model_a"].eq(0).all()
				and sources["winner_model_b"].eq(0).all()
				and sources["winner_tie"].eq(1).all()
				and (~sources["is_swapped"]).all()
			),
			"Eligible Kaggle tie rows have inconsistent winner flags or orientations.",
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
		supplement["source_dataset"] = self.SOURCE_DATASET
		supplement["id"] = np.arange(start, start + count, dtype="int64")
		supplement["split"] = self.scope
		supplement["is_swapped"] = False
		supplement["source"] = "kaggle_tie_duplicate"

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
			total_kaggle_tie_rows=len(kaggle_ties),
			excluded_identical_ab_rows=excluded_identical_ab_rows,
			eligible_kaggle_tie_rows=eligible_count,
			removed_random_rows=self.remove_count,
			added_rows=count,
			new_ids={"first": start, "last": start + count - 1},
			source_splits=dict(Counter(supplement["source_split"])),
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
			"kind": BuildSupplementLoader.KAGGLE_TIE_DUPLICATE_KIND,
			"scope": self.scope,
			"source_sha256": source_sha256,
			"source_dataset": self.SOURCE_DATASET,
			"source_label": self.SOURCE_LABEL,
			"repeat": self.REPEAT,
			"exclude_identical_ab": self.EXCLUDE_IDENTICAL_AB,
			"eligible_count": self.report["eligible_kaggle_tie_rows"],
			"remove_count": self.remove_count,
			"sample_seed": self.sample_seed,
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
			"exact_non_identical_ab_kaggle_tie_coverage_once": True,
			"original_orientation_ties_only": True,
			"scope_source_membership_and_validation_isolation": True,
		}

	def _finalize(self, stage: Path) -> None:
		self.report.update(
			status="completed", completed_at=datetime.now(UTC).isoformat()
		)
		files = [
			p
			for p in sorted(stage.rglob("*"))
			if p.is_file() and ".cache-work" not in p.relative_to(stage).parts
		]
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
