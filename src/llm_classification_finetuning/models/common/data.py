"""Load and serialize processed pairwise training records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerBase

from ...errors import ModelExecutionError
from .external_selection import RankedExternalSelector
from .inputs import TrainingInputLoader
from .supplement import BuildSupplementLoader

if TYPE_CHECKING:
	from .token_cache import TokenizationCache


@dataclass(frozen=True, slots=True)
class PairwiseText:
	"""One canonical source record serialized into two model inputs."""

	row_id: int
	response_a: str
	response_b: str


@dataclass(frozen=True, slots=True)
class ModelData:
	"""Validated real/supplement references and canonical model inputs."""

	reference: pd.DataFrame
	canonical_texts: tuple[PairwiseText, ...]
	fingerprint: str
	supplement_fingerprint: str | None = None


class ConversationSerializer:
	"""Render aligned user/assistant turns with the supplied model's chat template."""

	FORMAT_VERSION = "sr2-qwen3-chat-template-v1"

	def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
		self._tokenizer = tokenizer
		try:
			self._chat_template = tokenizer.get_chat_template()
			if not self._chat_template:
				raise ValueError("the tokenizer has no chat template")
		except Exception as error:
			raise ModelExecutionError(
				f"Could not prepare the conversation chat template: {error}"
			) from error

	@property
	def chat_template_sha256(self) -> str:
		return hashlib.sha256(self._chat_template.encode("utf-8")).hexdigest()

	def count_tokens(self, texts: Sequence[str]) -> np.ndarray:
		"""Count complete serialized inputs when no shared token cache is supplied."""
		encoded = self._tokenizer(
			list(texts),
			padding=False,
			truncation=False,
			add_special_tokens=False,
			return_attention_mask=False,
			return_token_type_ids=False,
			verbose=False,
		)["input_ids"]
		return np.fromiter(map(len, encoded), dtype=np.int64, count=len(texts))

	def serialize_pair(
		self,
		row_id: int,
		prompt_value: str,
		response_a_value: str,
		response_b_value: str,
	) -> PairwiseText:
		"""Build the A and B branch text for one canonical row."""
		prompts = self._load_array(prompt_value, row_id, "prompt")
		response_a = self._load_array(response_a_value, row_id, "response_a")
		response_b = self._load_array(response_b_value, row_id, "response_b")

		if not prompts:
			raise ModelExecutionError(
				f"Row {row_id} contains an empty prompt conversation."
			)
		if len(prompts) != len(response_a) or len(prompts) != len(response_b):
			raise ModelExecutionError(
				f"Row {row_id} has misaligned prompt and response turn counts."
			)

		for turn, prompt in enumerate(prompts):
			if not isinstance(prompt, str):
				raise ModelExecutionError(
					f"Row {row_id} prompt turn {turn} must be a string."
				)

		text_a = self._serialize_branch(row_id, prompts, response_a, "response_a")
		text_b = self._serialize_branch(row_id, prompts, response_b, "response_b")
		return PairwiseText(row_id=row_id, response_a=text_a, response_b=text_b)

	@staticmethod
	def _load_array(value: str, row_id: int, column: str) -> list[Any]:
		if not isinstance(value, str):
			raise ModelExecutionError(f"Row {row_id} {column} must be JSON text.")
		try:
			parsed = json.loads(value)
		except (TypeError, json.JSONDecodeError) as error:
			raise ModelExecutionError(
				f"Row {row_id} {column} is not a valid JSON array: {error}"
			) from error
		if not isinstance(parsed, list):
			raise ModelExecutionError(f"Row {row_id} {column} must be a JSON array.")
		return parsed

	def _serialize_branch(
		self,
		row_id: int,
		prompts: list[Any],
		responses: list[Any],
		column: str,
	) -> str:
		messages: list[dict[str, str]] = []
		for turn, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
			if response is not None and not isinstance(response, str):
				raise ModelExecutionError(
					f"Row {row_id} {column} turn {turn} must be a string or null."
				)
			messages.extend(
				(
					{"role": "user", "content": prompt},
					{"role": "assistant", "content": response or ""},
				)
			)
		try:
			return self._tokenizer.apply_chat_template(
				messages,
				chat_template=self._chat_template,
				tokenize=False,
				add_generation_prompt=False,
				continue_final_message=False,
			)
		except Exception as error:
			raise ModelExecutionError(
				f"Could not apply the chat template to row {row_id} {column}: {error}"
			) from error


class ModelDataRepository:
	"""Read and validate the processed holdout artifact for model training."""

	REQUIRED_COLUMNS = TrainingInputLoader.REQUIRED_COLUMNS

	def __init__(
		self,
		processed_path: Path,
		*,
		serializer: ConversationSerializer,
		validation_fraction: float | None = None,
		build_supplement_path: Path | None = None,
		train_supplement_path: Path | None = None,
		external_rank_path: Path | None = None,
		token_cache: TokenizationCache | None = None,
	) -> None:
		self._loader = TrainingInputLoader(
			processed_path, validation_fraction=validation_fraction
		)
		self._serializer = serializer
		self._build_supplement_path = build_supplement_path
		self._train_supplement_path = train_supplement_path
		self._external_rank_path = external_rank_path
		self._token_cache = token_cache

	@property
	def serializer_version(self) -> str:
		"""Return the cache-relevant conversation format identifier."""
		return self._serializer.FORMAT_VERSION

	def load(self, *, include_build_supplement: bool = False) -> ModelData:
		"""Return validated references, canonical texts, and a training-data fingerprint."""
		inputs = self._loader.load(include_source=self._external_rank_path is not None)
		selected = inputs
		if self._external_rank_path is not None:
			selected = RankedExternalSelector(self._external_rank_path).select(
				inputs,
				full_data=include_build_supplement,
				load_token_totals=self._external_token_totals,
			)
		data = selected.data
		fingerprint = selected.fingerprint
		supplement_fingerprint = None
		scope = "build" if include_build_supplement else "train"
		supplement_path = (
			self._build_supplement_path
			if include_build_supplement
			else self._train_supplement_path
		)
		if supplement_path is not None:
			supplement = BuildSupplementLoader(supplement_path, scope=scope).load(
				inputs
			)
			data = pd.concat(
				[data, supplement.data.loc[:, list(self.REQUIRED_COLUMNS)]],
				ignore_index=True,
			)
			supplement_fingerprint = supplement.fingerprint
			fingerprint = BuildSupplementLoader.combined_fingerprint(
				fingerprint, supplement.fingerprint
			)
		del inputs, selected
		return ModelData(
			reference=data.loc[:, ["id", "label", "split", "is_swapped"]]
			.copy()
			.reset_index(drop=True),
			canonical_texts=self._serialize(data),
			fingerprint=fingerprint,
			supplement_fingerprint=supplement_fingerprint,
		)

	def _external_token_totals(self, rows: pd.DataFrame, fingerprint: str) -> pd.Series:
		cache = self._token_cache
		if cache is not None:
			stored = self._loader.load_token_totals(
				fingerprint,
				tokenizer_fingerprint=cache.tokenizer_key,
				serializer_version=self.serializer_version,
				chat_template_sha256=self._serializer.chat_template_sha256,
			)
			if stored is not None:
				matched = stored.reindex(rows["id"])
				if matched.notna().all():
					return matched

		count_tokens = (
			cache.count_tokens if cache is not None else self._serializer.count_tokens
		)
		batch_size = 4096 if cache is not None else 256
		totals = np.empty(len(rows), dtype=np.int64)
		for start in range(0, len(rows), batch_size):
			pairs = self._serialize(rows.iloc[start : start + batch_size])
			texts = [
				text for pair in pairs for text in (pair.response_a, pair.response_b)
			]
			lengths = count_tokens(texts).reshape(-1, 2)
			if (lengths <= 0).any():
				raise ModelExecutionError(
					"Complete branch token counts must be positive."
				)
			totals[start : start + len(pairs)] = lengths.sum(axis=1, dtype=np.int64)
		return pd.Series(totals, index=rows["id"])

	def _serialize(self, data: pd.DataFrame) -> tuple[PairwiseText, ...]:
		return tuple(
			self._serializer.serialize_pair(
				row_id=int(row.id),
				prompt_value=row.prompt,
				response_a_value=row.response_a,
				response_b_value=row.response_b,
			)
			for row in data.itertuples(index=False)
		)
