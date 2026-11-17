"""Validate independent synthetic ties for train-partition and full-data inputs."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ...errors import ModelExecutionError
from .cache_storage import CacheStorage
from .inputs import TrainingInputs, TrainingInputLoader, _sha256


class BuildSupplementLoader:
	"""Load a scope-specific supplement without changing the saved real-data split."""

	COUNT = 1012
	SCHEMA_VERSION = 2
	TARGET_TIE_SHARE = Fraction(17788, 57477)
	METADATA_KEY = b"llm_classification_finetuning.synthetic_tie"
	PROVENANCE_COLUMNS = (
		"source_original_id",
		"source_side",
		"source_split",
		"historical_wins",
		"historical_occurrences",
	)

	def __init__(self, path: Path, *, scope: str = "build") -> None:
		if scope not in {"train", "build"}:
			raise ValueError("Supplement scope must be 'train' or 'build'.")
		self.path = path.resolve()
		self.scope = scope

	@classmethod
	def size_policy(cls, original: pd.DataFrame, scope: str) -> dict:
		"""Compare adjacent integer additions using exact resulting share errors."""
		if scope not in {"train", "build"}:
			raise ValueError("Supplement scope must be 'train' or 'build'.")
		base = (
			original.loc[original.split.eq("train")] if scope == "train" else original
		)
		if base.empty:
			raise ValueError("Supplement scope has no original rows.")
		ties = int(base.label.eq(2).sum())
		ideal = (cls.TARGET_TIE_SHARE * len(base) - ties) / (1 - cls.TARGET_TIE_SHARE)
		lower = ideal.numerator // ideal.denominator
		options = sorted({max(0, lower), max(0, lower + 1)})
		errors = {
			count: abs(Fraction(ties + count, len(base) + count) - cls.TARGET_TIE_SHARE)
			for count in options
		}
		nearest = min(options, key=lambda count: (errors[count], count))
		return {
			"ideal_count": str(ideal),
			"integer_options": [
				{"count": count, "absolute_share_error": str(errors[count])}
				for count in options
			],
			"nearest_count": nearest,
			"target_count": nearest,
		}

	@staticmethod
	def canonical(value: str) -> str:
		"""Normalize JSON formatting without altering conversation content."""
		return json.dumps(
			json.loads(value),
			ensure_ascii=False,
			sort_keys=True,
			separators=(",", ":"),
			allow_nan=False,
		)

	@staticmethod
	def combined_fingerprint(original: str, supplement: str) -> str:
		payload = json.dumps(
			{
				"version": 1,
				"original_sha256": original,
				"supplement_sha256": supplement,
			},
			sort_keys=True,
			separators=(",", ":"),
		)
		return hashlib.sha256(payload.encode("utf-8")).hexdigest()

	def load(self, original: TrainingInputs) -> TrainingInputs:
		"""Read checked single/multipart data and require matching source provenance."""
		try:
			storage = CacheStorage(self.path)
			if not storage.exists():
				raise ValueError(
					"configured supplement is missing; run data/generate_synthetic_tie.py"
				)
			with storage.open("r") as session:
				metadata = pq.read_schema(session.path).metadata or {}
				identity = json.loads(metadata[self.METADATA_KEY])
				data = pd.read_parquet(session.path)
				fingerprint = _sha256(session.path)
			version = identity.get("schema_version")
			if version not in {1, self.SCHEMA_VERSION}:
				raise ValueError("unsupported synthetic-tie schema")
			if identity.get("source_sha256") != original.fingerprint:
				raise ValueError(
					"supplement source SHA-256 does not match the processed dataset"
				)
			count = self.size_policy(original.data, self.scope)["target_count"]
			if version == 1:
				if self.scope != "build" or count != self.COUNT:
					raise ValueError(
						"legacy supplement is only valid for its historical build"
					)
			elif identity.get("scope") != self.scope:
				raise ValueError(
					"supplement scope does not match the requested training mode"
				)
			if identity.get("target_tie_share") != str(self.TARGET_TIE_SHARE):
				raise ValueError("supplement target tie share is inconsistent")
			if identity.get("target_count") != count:
				raise ValueError(f"supplement target count must be {count:,}")
			if identity.get("historical_score_sum") != str(Fraction(count, 2)):
				raise ValueError(
					f"supplement metadata historical score must be exactly {Fraction(count, 2)}"
				)
			self.validate(data, original.data, scope=self.scope)
			return TrainingInputs(data, fingerprint)
		except Exception as error:
			raise ModelExecutionError(
				f"Invalid {self.scope} supplement {self.path}: {error}"
			) from error

	@classmethod
	def validate(
		cls, data: pd.DataFrame, original: pd.DataFrame, *, scope: str = "build"
	) -> None:
		"""Check exact count/score, original orientation, unique content and sources."""
		required = [
			*TrainingInputLoader.REQUIRED_COLUMNS,
			*cls.PROVENANCE_COLUMNS,
			"model_a",
			"model_b",
			"winner_model_a",
			"winner_model_b",
			"winner_tie",
		]
		count = cls.size_policy(original, scope)["target_count"]
		if len(data) != count or data.loc[:, required].isna().any().any():
			raise ValueError(f"supplement must contain {count:,} complete rows")
		for name in (
			"id",
			"source_original_id",
			"historical_wins",
			"historical_occurrences",
		):
			if not pd.api.types.is_integer_dtype(data[name]):
				raise ValueError(f"{name} must contain integers")
		if data.id.duplicated().any() or data.id.isin(original.id).any():
			raise ValueError(
				"supplement IDs are duplicated or collide with original IDs"
			)
		if not pd.api.types.is_bool_dtype(data.is_swapped) or data.is_swapped.any():
			raise ValueError("supplement must contain unswapped rows only")
		if not (
			data.split.eq(scope).all()
			and data.label.eq(2).all()
			and data.winner_model_a.eq(0).all()
			and data.winner_model_b.eq(0).all()
			and data.winner_tie.eq(1).all()
			and data.model_a.eq(data.model_b).all()
			and data.response_a.eq(data.response_b).all()
			and data.source_side.isin(("a", "b")).all()
		):
			raise ValueError(f"supplement must contain {scope} self-response ties")
		base = (
			original.loc[original.split.eq("train")] if scope == "train" else original
		)
		score = Fraction()
		keys: set[tuple[str, str]] = set()
		prompts = {cls.canonical(value) for value in data.prompt}
		tie_prompts = {
			cls.canonical(value) for value in base.loc[base.label.eq(2), "prompt"]
		}
		if prompts & tie_prompts:
			raise ValueError("supplement includes a prompt with an original tie")
		if scope == "train":
			validation = original.loc[original.split.eq("validation")]
			if prompts & {cls.canonical(value) for value in validation.prompt} or set(
				data.group_id
			) & set(validation.group_id):
				raise ValueError("train supplement overlaps frozen validation prompts")
		sources = base.set_index("id")
		for row in data.itertuples(index=False):
			wins, occurrences = (
				int(row.historical_wins),
				int(row.historical_occurrences),
			)
			if occurrences <= 0 or not 0 <= wins <= occurrences:
				raise ValueError(f"invalid historical score for ID {row.id}")
			score += Fraction(wins, occurrences)
			key = (cls.canonical(row.prompt), cls.canonical(row.response_a))
			if key in keys:
				raise ValueError("duplicate normalized synthetic (prompt, response)")
			keys.add(key)
			source = sources.loc[row.source_original_id]
			if not (
				row.prompt == source.prompt
				and row.source_split == source.split
				and row.group_id == source.group_id
				and row.response_a == source[f"response_{row.source_side}"]
			):
				raise ValueError(f"source provenance mismatch for ID {row.id}")
		if score != Fraction(count, 2):
			raise ValueError(
				f"historical score must be exactly {Fraction(count, 2)}, found {score}"
			)
		history = {key: [0, 0] for key in keys}
		for row in base.itertuples(index=False):
			prompt = cls.canonical(row.prompt)
			if prompt in prompts:
				a, b = cls.canonical(row.response_a), cls.canonical(row.response_b)
				if a == b and (prompt, a) in keys:
					raise ValueError(
						"supplement duplicates an original self-response battle"
					)
				for side, response in enumerate((a, b)):
					key = (prompt, response)
					if key in history:
						history[key][0] += int(row.label == side)
						history[key][1] += 1
		for row in data.itertuples(index=False):
			key = (cls.canonical(row.prompt), cls.canonical(row.response_a))
			if history[key] != [row.historical_wins, row.historical_occurrences]:
				raise ValueError(f"historical source counts mismatch for ID {row.id}")
