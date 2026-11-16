"""Generate audited, independent self-response ties for train and full-data scopes."""

from __future__ import annotations

import gc
import hashlib
import heapq
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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
	TrainingInputLoader,
	TrainingInputs,
	TokenizationCache,
)


class _CandidateAuditSink:
	"""Stream a staged Parquet into bounded volumes before publishing its manifest."""

	def __init__(self, path: Path) -> None:
		self.path = path
		self.parts_path = path.with_name(f"{path.name}.parts")
		self.parts_path.mkdir()
		self.parts: list[dict[str, Any]] = []
		self.size = 0
		self.closed = False
		self._digest = hashlib.sha256()
		self._part_digest = hashlib.sha256()
		self._part_size = 0
		self._stream = None
		self._working = self.parts_path / ".writing.part"

	def writable(self) -> bool:
		return True

	def tell(self) -> int:
		return self.size

	def write(self, value: bytes) -> int:
		if self.closed:
			raise ValueError("Candidate audit stream is closed.")
		view = memoryview(value)
		length = len(view)
		self._digest.update(view)
		while view:
			if self._stream is None:
				self._stream = self._working.open("wb")
			take = min(len(view), CacheStorage.PART_SIZE - self._part_size)
			block = view[:take]
			self._stream.write(block)
			self._part_digest.update(block)
			self._part_size += take
			self.size += take
			view = view[take:]
			if self._part_size == CacheStorage.PART_SIZE:
				self._finish_part()
		return length

	def _finish_part(self) -> None:
		if self._stream is None:
			return
		self._stream.close()
		checksum = self._part_digest.hexdigest()
		name = f"{len(self.parts) + 1:06d}-{checksum}.part"
		self._working.rename(self.parts_path / name)
		self.parts.append({"name": name, "size": self._part_size, "sha256": checksum})
		self._stream = None
		self._part_digest = hashlib.sha256()
		self._part_size = 0

	def flush(self) -> None:
		if self._stream is not None:
			self._stream.flush()

	def close(self) -> None:
		self._finish_part()
		self.closed = True

	def publish(self) -> None:
		"""Publish only the completed staged stream, using the shared storage schema."""
		self.close()
		if self.size < CacheStorage.SPLIT_THRESHOLD:
			with self.path.open("wb") as output:
				for part in self.parts:
					with (self.parts_path / part["name"]).open("rb") as source:
						shutil.copyfileobj(
							source, output, length=CacheStorage.COPY_SIZE
						)
			for part in self.parts:
				(self.parts_path / part["name"]).unlink()
			self.parts_path.rmdir()
		else:
			manifest = {
				"version": CacheStorage.FORMAT_VERSION,
				"sha256": self._digest.hexdigest(),
				"size": self.size,
				"parts": self.parts,
			}
			self.path.with_name(f"{self.path.name}.manifest.json").write_text(
				json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
			)


@dataclass(frozen=True, slots=True)
class CandidateSource:
	row_index: int
	original_id: int
	side: str
	model: str
	source: str
	source_id: str
	source_row_index: int
	split: str
	won: int


@dataclass(slots=True)
class TieCandidate:
	prompt: str
	response: str
	candidate_id: str
	sources: list[CandidateSource] = field(default_factory=list)
	wins: int = 0
	tokens: int = 0
	new_id: int | None = None

	@property
	def score(self) -> Fraction:
		return Fraction(self.wins, len(self.sources))

	@property
	def representative(self) -> CandidateSource:
		return min(self.sources, key=lambda source: (source.original_id, source.side))

	@property
	def order(self) -> tuple[int, int, str]:
		source = self.representative
		return self.tokens, source.original_id, source.side


class SyntheticTieGenerator:
	"""Publish a reproducible scope-specific supplement matching the Kaggle tie share."""

	TARGET_TIE_SHARE = BuildSupplementLoader.TARGET_TIE_SHARE
	ALGORITHM_VERSION = "prompt_count_full_branch_scoped_v2"

	def __init__(self, config: AppConfig, *, scope: str = "build") -> None:
		self.config = config
		self.scope = scope
		self._require(scope in {"train", "build"}, "Scope must be train or build.")
		path = getattr(config.data, f"{scope}_supplement_path")
		if path is None:
			raise DataPreparationError(
				f"Configure data.{scope}_supplement_path before generation."
			)
		self.output = path.parent
		self._require(
			not config.data.processed_path.is_relative_to(self.output)
			and not Path.cwd().resolve().is_relative_to(self.output),
			"The supplement output directory must not contain the original dataset or project root.",
		)
		if self.output.exists() and any(self.output.iterdir()):
			summary = self.output / "summary.json"
			self._require(
				summary.is_file()
				and json.loads(summary.read_text(encoding="utf-8")).get(
					"algorithm_version"
				)
				in {"prompt_count_full_branch_v1", self.ALGORITHM_VERSION},
				"Refusing to replace an output directory not owned by this generator.",
			)
		self.filename = path.name
		self.candidates: list[TieCandidate] = []
		self.selected: set[int] = set()
		self.selections: list[dict[str, Any]] = []
		self.replacements: list[dict[str, Any]] = []
		self.report: dict[str, Any] = {
			"status": "started",
			"algorithm_version": self.ALGORITHM_VERSION,
			"scope": self.scope,
		}

	@staticmethod
	def _require(condition: bool, message: str) -> None:
		if not condition:
			raise DataPreparationError(message)

	@staticmethod
	def _digest(path: Path) -> str:
		digest = hashlib.sha256()
		with path.open("rb") as stream:
			for block in iter(lambda: stream.read(1024 * 1024), b""):
				digest.update(block)
		return digest.hexdigest()

	@staticmethod
	def _json(path: Path, value: Any) -> None:
		path.write_text(
			json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
		)

	@staticmethod
	def _jsonl(path: Path, records: list[dict[str, Any]]) -> None:
		with path.open("w", encoding="utf-8", newline="\n") as stream:
			for record in records:
				stream.write(json.dumps(record, ensure_ascii=False) + "\n")

	def run(self) -> Path:
		"""Stage all files, validate readback, and promote only a complete run."""
		work_root = self.output.parent / ".cache-work"
		work_root.mkdir(parents=True, exist_ok=True)
		stage = Path(tempfile.mkdtemp(prefix="synthetic-tie-", dir=work_root))
		stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
		try:
			self._load()
			self._collect()
			self._count_tokens()
			self._select()
			supplement = self._construct()
			self._write_audit(stage, supplement)
			# Candidate text and source graphs are now persisted in the audit.
			# Release them before reconstructing and checking the published files.
			self.candidates.clear()
			self.rows.clear()
			self.eligible.clear()
			self.original = self.original.loc[
				:, list(TrainingInputLoader.REQUIRED_COLUMNS)
			]
			del self.data
			gc.collect()
			self._verify(stage, supplement)
			self.report["status"] = "completed"
			self.report["completed_at"] = datetime.now(timezone.utc).isoformat()
			self.report["artifacts"] = {
				str(path.relative_to(stage)): {
					"bytes": path.stat().st_size,
					"sha256": self._digest(path),
				}
				for path in sorted(stage.rglob("*"))
				if path.is_file() and ".cache-work" not in path.relative_to(stage).parts
			}
			self._json(stage / "summary.json", self.report)
			files = [
				path
				for path in sorted(stage.rglob("*"))
				if path.is_file() and ".cache-work" not in path.relative_to(stage).parts
			]
			(stage / "checksums.sha256").write_text(
				"".join(
					f"{self._digest(path)}  {path.relative_to(stage).as_posix()}\n"
					for path in files
				),
				encoding="utf-8",
			)
			# Keep the previous successful run recoverable if replacement is requested.
			backup = self.output.with_name(f"{self.output.name}.backup-{stamp}")
			if self.output.exists():
				self.output.rename(backup)
			try:
				stage.rename(self.output)
			except BaseException:
				if backup.exists() and not self.output.exists():
					backup.rename(self.output)
				raise
			print(f"Published synthetic ties: {self.output}", flush=True)
			return self.output
		except BaseException as error:
			self.report.update(
				status="failed", error=f"{type(error).__name__}: {error}"
			)
			if stage.exists():
				self._json(stage / "summary.json", self.report)
				self._jsonl(stage / "selection.jsonl", self.selections)
				self._jsonl(stage / "replacements.jsonl", self.replacements)
				failure = self.output.with_name(f"{self.output.name}.failed-{stamp}")
				stage.rename(failure)
				print(f"Generation failed; diagnostic audit: {failure}", flush=True)
			raise

	def _load(self) -> None:
		with CacheStorage(self.config.data.processed_path).open("r") as session:
			self.data = pd.read_parquet(session.path)
			self.source_sha256 = self._digest(session.path)
		TrainingInputLoader.validate(
			self.data, validation_fraction=self.config.validation.validation_fraction
		)
		self.original = self.data
		if self.scope == "train":
			self.data = self.original.loc[self.original.split.eq("train")].copy()
		winners = self.data.loc[
			:, ["winner_model_a", "winner_model_b", "winner_tie"]
		].to_numpy()
		self._require(
			bool(
				np.isin(winners, (0, 1)).all()
				and (winners.sum(axis=1) == 1).all()
				and (winners.argmax(axis=1) == self.data.label.to_numpy()).all()
			),
			"Original winner flags and labels are inconsistent.",
		)
		self.rows = list(self.data.itertuples(index=False))
		ties = int(self.data.label.eq(2).sum())
		share = Fraction(ties, len(self.data))
		size_policy = BuildSupplementLoader.size_policy(self.original, self.scope)
		self.count = size_policy["target_count"]
		self.report.update(
			source_path=str(self.config.data.processed_path),
			source_sha256=self.source_sha256,
			original_rows=len(self.data),
			original_splits=dict(Counter(self.data.split)),
			class_counts_before={
				str(i): int(self.data.label.eq(i).sum()) for i in range(3)
			},
			tie_share_before=str(share),
			tie_pct_before=float(share * 100),
			tie_share_target=str(self.TARGET_TIE_SHARE),
			tie_pct_target=float(self.TARGET_TIE_SHARE * 100),
			supplement_size_policy=size_policy,
		)

	def _collect(self) -> None:
		canonical = BuildSupplementLoader.canonical
		groups: dict[str, list[int]] = defaultdict(list)
		for position, row in enumerate(self.rows):
			groups[canonical(row.prompt)].append(position)
		self.eligible = {
			prompt: indices
			for prompt, indices in groups.items()
			if not any(self.rows[i].label == 2 for i in indices)
		}
		combinations: dict[tuple[str, str], TieCandidate] = {}
		for prompt, indices in self.eligible.items():
			for position in indices:
				row = self.rows[position]
				for side in ("a", "b"):
					response = canonical(getattr(row, f"response_{side}"))
					key = (prompt, response)
					if key not in combinations:
						identity = json.dumps(
							key, ensure_ascii=False, separators=(",", ":")
						).encode("utf-8")
						combinations[key] = TieCandidate(
							prompt, response, hashlib.sha256(identity).hexdigest()
						)
					candidate = combinations[key]
					won = int(getattr(row, f"winner_model_{side}"))
					candidate.wins += won
					candidate.sources.append(
						CandidateSource(
							position,
							int(row.id),
							side,
							getattr(row, f"model_{side}"),
							row.source,
							str(row.source_id),
							int(row.source_row_index),
							row.split,
							won,
						)
					)
		self.candidates = list(combinations.values())
		self._require(
			len(self.candidates) >= self.count,
			f"Insufficient unique candidates for {self.count:,} ties.",
		)
		self.report["candidate_counts"] = {
			"all_prompt_groups": len(groups),
			"eligible_prompts": len(self.eligible),
			"eligible_battles": sum(map(len, self.eligible.values())),
			"unique_candidates": len(self.candidates),
			"occurrences": sum(len(c.sources) for c in self.candidates),
			"score_distribution": dict(
				sorted(Counter(str(c.score) for c in self.candidates).items())
			),
		}
		print(
			f"Collected {len(self.candidates):,} candidates from {len(self.eligible):,} tie-free prompts.",
			flush=True,
		)

	def _count_tokens(self) -> None:
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
		if self._reuse_token_stats():
			return
		batch_size = 4096
		for start in range(0, len(self.candidates), batch_size):
			batch = self.candidates[start : start + batch_size]
			texts = []
			for candidate in batch:
				source = candidate.representative
				row = self.rows[source.row_index]
				response = getattr(row, f"response_{source.side}")
				texts.append(
					serializer.serialize_pair(
						int(row.id), row.prompt, response, response
					).response_a
				)
			lengths = cache.count_tokens(texts)
			for candidate, length in zip(batch, lengths, strict=True):
				candidate.tokens = int(length)
			if (start // batch_size + 1) % 10 == 0:
				print(
					f"Counted complete branch tokens: {min(start + batch_size, len(self.candidates)):,}/{len(self.candidates):,}.",
					flush=True,
				)
		print("Complete branch token counting finished.", flush=True)

	def _reuse_token_stats(self) -> bool:
		"""Reuse pinned complete branch statistics from the real-data merger."""
		root = self.config.data.processed_path.parent
		summary_path, stats_path = root / "summary.json", root / "token_stats.parquet"
		if not summary_path.is_file() or not stats_path.is_file():
			return False
		summary = json.loads(summary_path.read_text(encoding="utf-8"))
		identity = self.report["tokenizer"]
		tokenizer = summary.get("tokenizer", {})
		if summary.get("processed_reconstructed_sha256") != self.source_sha256 or any(
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
			return False
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
		stats = stats.loc[stats.retained].set_index("id")
		if not stats.index.is_unique or not set(self.data.id).issubset(stats.index):
			return False
		for candidate in self.candidates:
			source = candidate.representative
			candidate.tokens = int(
				stats.at[source.original_id, f"tokens_{source.side}"]
			)
			self._require(
				candidate.tokens > 0, "Complete branch token counts must be positive."
			)
		self.report["token_statistics_sha256"] = actual
		print(
			f"Reused {len(self.candidates):,} complete branch lengths from pinned real-data statistics.",
			flush=True,
		)
		return True

	def _statistics(self) -> dict[str, Any]:
		chosen = [self.candidates[i] for i in self.selected]
		score = sum((c.score for c in chosen), Fraction())
		return {
			"count": len(chosen),
			"prompts": len({c.prompt for c in chosen}),
			"score_sum": str(score),
			"mean_historical_win_rate": str(score / len(chosen)) if chosen else None,
			"score_distribution": dict(
				sorted(Counter(str(c.score) for c in chosen).items())
			),
			"total_branch_tokens": sum(c.tokens for c in chosen),
			"max_branch_tokens": max((c.tokens for c in chosen), default=0),
		}

	def _select(self) -> None:
		self.ranked = sorted(
			range(len(self.candidates)), key=lambda i: self.candidates[i].order
		)
		by_prompt: dict[str, list[int]] = defaultdict(list)
		for i in self.ranked:
			by_prompt[self.candidates[i].prompt].append(i)
		heap = [
			(0, *self.candidates[indices[0]].order, indices[0], 0)
			for indices in by_prompt.values()
		]
		heapq.heapify(heap)
		counts: Counter[str] = Counter()
		while len(self.selected) < self.count:
			before, *_, index, position = heapq.heappop(heap)
			candidate = self.candidates[index]
			self.selected.add(index)
			counts[candidate.prompt] += 1
			self.selections.append(
				{
					"step": len(self.selected),
					"candidate_id": candidate.candidate_id,
					"prompt_selected_before": before,
					"prompt_selected_after": counts[candidate.prompt],
					"tokens": candidate.tokens,
					"source_id": candidate.representative.original_id,
					"source_side": candidate.representative.side,
					"score": str(candidate.score),
				}
			)
			indices = by_prompt[candidate.prompt]
			if position + 1 < len(indices):
				next_index = indices[position + 1]
				heapq.heappush(
					heap,
					(
						counts[candidate.prompt],
						*self.candidates[next_index].order,
						next_index,
						position + 1,
					),
				)
		self.initial_selected = self.selected.copy()
		covered = set(counts)
		self.report["initial"] = self._statistics()
		score = sum((self.candidates[i].score for i in self.selected), Fraction())
		target = Fraction(self.count, 2)
		self._require(
			score <= target,
			f"Initial score {score} exceeds {target}; legacy upward-only replacements cannot satisfy 50%.",
		)
		winners = [i for i in self.ranked if self.candidates[i].score == 1]
		winners_by_prompt = {
			p: [i for i in indices if self.candidates[i].score == 1]
			for p, indices in by_prompt.items()
		}

		def allowed(old: int, new: int) -> bool:
			left, right = self.candidates[old], self.candidates[new]
			return right.prompt in covered and (
				counts[left.prompt] > 1 or left.prompt == right.prompt
			)

		def replace(old: int, new: int, kind: str) -> None:
			nonlocal score
			left, right = self.candidates[old], self.candidates[new]
			before = {p: counts[p] for p in sorted({left.prompt, right.prompt})}
			previous_score = score
			self.selected.remove(old)
			self.selected.add(new)
			counts[left.prompt] -= 1
			counts[right.prompt] += 1
			score += right.score - left.score
			self.replacements.append(
				{
					"step": len(self.replacements) + 1,
					"kind": kind,
					"removed_candidate_id": left.candidate_id,
					"added_candidate_id": right.candidate_id,
					"removed_source_id": left.representative.original_id,
					"removed_side": left.representative.side,
					"added_source_id": right.representative.original_id,
					"added_side": right.representative.side,
					"old_score": str(left.score),
					"new_score": str(right.score),
					"score_before": str(previous_score),
					"score_after": str(score),
					"token_delta": right.tokens - left.tokens,
					"prompt_counts_before": before,
					"prompt_counts_after": {p: counts[p] for p in before},
				}
			)
			self._require(
				all(counts[p] >= 1 for p in covered) and score <= target,
				"Replacement lost coverage or exceeded target.",
			)

		while score + 1 <= target:
			pair = None
			for old in self.ranked:
				left = self.candidates[old]
				if old not in self.selected or left.score != 0:
					continue
				pool = (
					winners_by_prompt[left.prompt]
					if counts[left.prompt] == 1
					else winners
				)
				new = next(
					(
						i
						for i in pool
						if i not in self.selected
						and self.candidates[i].tokens > left.tokens
						and allowed(old, i)
					),
					None,
				)
				if new is not None:
					pair = (old, new)
					break
			self._require(
				pair is not None,
				f"No coverage-preserving full-win replacement at exact score {score}; target {target}.",
			)
			replace(*pair, "full_win")
		if score < target:
			options = []
			for old in self.ranked:
				if old not in self.selected:
					continue
				left = self.candidates[old]
				pool = (
					by_prompt[left.prompt] if counts[left.prompt] == 1 else self.ranked
				)
				for new in pool:
					right = self.candidates[new]
					if (
						new not in self.selected
						and allowed(old, new)
						and right.score - left.score == target - score
						and Fraction(1, 2) in (left.score, right.score)
					):
						options.append(
							(
								right.tokens - left.tokens,
								left.order,
								right.order,
								old,
								new,
							)
						)
			if options:
				*_, old, new = min(options)
				replace(old, new, "half_win")
			else:
				# Fractional histories such as 1/3 can need a downward correction
				# followed by an upward one. Same-prompt swaps retain coverage.
				adjustments: dict[Fraction, list[tuple]] = defaultdict(list)
				for old in self.ranked:
					if old not in self.selected:
						continue
					left = self.candidates[old]
					for new in by_prompt[left.prompt]:
						if new in self.selected:
							continue
						right = self.candidates[new]
						delta = right.score - left.score
						if delta:
							adjustments[delta].append(
								(
									right.tokens - left.tokens,
									left.order,
									right.order,
									old,
									new,
								)
							)
				# For a two-swap correction, only the cheapest options from two
				# distinct prompts per delta can affect the optimum.
				for delta, choices in adjustments.items():
					best = []
					seen = set()
					for choice in sorted(choices):
						prompt = self.candidates[choice[-2]].prompt
						if prompt not in seen:
							best.append(choice)
							seen.add(prompt)
						if len(best) == 2:
							break
					adjustments[delta] = best
				gap = target - score
				plans = []
				for choice in adjustments.get(gap, ()):
					plans.append((1, choice[0], (choice,)))
				for delta, choices in adjustments.items():
					for first in choices:
						for second in adjustments.get(gap - delta, ()):
							if (
								self.candidates[first[-2]].prompt
								== self.candidates[second[-2]].prompt
							):
								continue
							ordered = tuple(sorted((first, second)))
							plans.append((2, first[0] + second[0], ordered))
				self._require(
					bool(plans),
					f"No exact coverage-preserving fractional correction at score {score}; remaining gap {gap}.",
				)
				*_, chosen = min(plans)
				# Apply negative changes first, so no intermediate score overshoots.
				for choice in sorted(
					chosen,
					key=lambda item: (
						self.candidates[item[-1]].score
						- self.candidates[item[-2]].score,
						item,
					),
				):
					replace(choice[-2], choice[-1], "exact_fraction")
		self._require(
			score == target and len(self.selected) == self.count,
			"Final exact score or count mismatch.",
		)
		self._require(
			{self.candidates[i].prompt for i in self.selected} == covered,
			"Initial prompt coverage changed.",
		)
		self.counts = counts
		self.report.update(
			final=self._statistics(),
			replacements=dict(Counter(r["kind"] for r in self.replacements)),
			prompt_count_distribution=dict(
				sorted(Counter(counts.get(p, 0) for p in self.eligible).items())
			),
			selection_policy={
				"order": [
					"current_selected_prompt_count",
					"complete_branch_tokens",
					"minimum_source_id",
					"side_a_before_b",
				],
				"initial_prompt_count": 0,
				"minimum_original_battles_per_prompt": 1,
				"coverage": "preserve_initial_prompt_set",
				"target_count": self.count,
				"target_score_sum": str(target),
			},
		)
		print(
			f"Selected {self.count:,} candidates; exact mean win rate {score / self.count if self.count else None}; replacements {self.report['replacements']}.",
			flush=True,
		)

	def _construct(self) -> pd.DataFrame:
		start = int(self.original.id.max()) + 1
		self._require(
			start + self.count - 1 <= np.iinfo(np.int64).max,
			"Synthetic IDs exceed int64.",
		)
		rows = []
		for offset, index in enumerate(i for i in self.ranked if i in self.selected):
			candidate = self.candidates[index]
			source = candidate.representative
			row = self.rows[source.row_index]._asdict()
			response = row[f"response_{source.side}"]
			candidate.new_id = start + offset
			row.update(
				id=candidate.new_id,
				model_a=source.model,
				model_b=source.model,
				response_a=response,
				response_b=response,
				winner_model_a=0,
				winner_model_b=0,
				winner_tie=1,
				label=2,
				split=self.scope,
				is_swapped=False,
				source="synthetic_tie",
				source_id=candidate.candidate_id,
				source_row_index=offset,
				source_winner="tie",
				candidate_id=candidate.candidate_id,
				source_original_id=source.original_id,
				source_side=source.side,
				source_split=source.split,
				source_dataset=source.source,
				source_dataset_id=source.source_id,
				source_dataset_row_index=source.source_row_index,
				historical_wins=candidate.wins,
				historical_occurrences=len(candidate.sources),
				branch_tokens=candidate.tokens,
			)
			rows.append(row)
		columns = [
			*self.data.columns,
			"candidate_id",
			"source_original_id",
			"source_side",
			"source_split",
			"source_dataset",
			"source_dataset_id",
			"source_dataset_row_index",
			"historical_wins",
			"historical_occurrences",
			"branch_tokens",
		]
		data = pd.DataFrame(rows, columns=list(dict.fromkeys(columns))).astype(
			self.data.dtypes.to_dict()
		)
		for name in (
			"source_original_id",
			"historical_wins",
			"historical_occurrences",
			"branch_tokens",
		):
			data[name] = data[name].astype("int64")
		total_rows = len(self.data) + len(data)
		tie_rows = int(self.data.label.eq(2).sum()) + len(data)
		share = Fraction(tie_rows, total_rows)
		self.report.update(
			new_ids={"first": start, "last": start + self.count - 1},
			source_splits=dict(Counter(data.source_split)),
			source_datasets=dict(Counter(data.source_dataset)),
			class_counts_after={
				str(i): self.report["class_counts_before"][str(i)]
				+ (self.count if i == 2 else 0)
				for i in range(3)
			},
			total_rows_after=total_rows,
			tie_share_after=str(share),
			tie_pct_after=float(share * 100),
			tie_share_error=str(share - self.TARGET_TIE_SHARE),
		)
		return data

	def _write_audit(self, stage: Path, supplement: pd.DataFrame) -> None:
		identity = {
			"schema_version": BuildSupplementLoader.SCHEMA_VERSION,
			"scope": self.scope,
			"source_sha256": self.source_sha256,
			"target_count": self.count,
			"algorithm_version": self.ALGORITHM_VERSION,
			"historical_score_sum": str(Fraction(self.count, 2)),
			"target_tie_share": str(self.TARGET_TIE_SHARE),
			"tokenizer": self.report["tokenizer"],
		}
		table = pa.Table.from_pandas(supplement, preserve_index=False)
		table = table.replace_schema_metadata(
			{
				**(table.schema.metadata or {}),
				BuildSupplementLoader.METADATA_KEY: json.dumps(
					identity, sort_keys=True
				).encode("utf-8"),
			}
		)
		with CacheStorage(stage / self.filename).open("w") as session:
			pq.write_table(table, session.path, compression="zstd")
		# Row-grouped audit writing bounds the memory needed to retain all sources.
		sink = _CandidateAuditSink(stage / "candidates.parquet")
		try:
			with pq.ParquetWriter(
				sink,
				pa.schema(
					[
						("candidate_id", pa.string()),
						("prompt", pa.large_string()),
						("response", pa.large_string()),
						("wins", pa.int64()),
						("occurrences", pa.int64()),
						("score", pa.string()),
						("tokens", pa.int64()),
						("representative_id", pa.int64()),
						("representative_side", pa.string()),
						("initially_selected", pa.bool_()),
						("selected", pa.bool_()),
						("new_id", pa.int64()),
						("sources_json", pa.large_string()),
					]
				),
				compression="zstd",
			) as writer:
				for start in range(0, len(self.candidates), 1024):
					records = []
					for i in range(start, min(start + 1024, len(self.candidates))):
						c = self.candidates[i]
						records.append(
							{
								"candidate_id": c.candidate_id,
								"prompt": c.prompt,
								"response": c.response,
								"wins": c.wins,
								"occurrences": len(c.sources),
								"score": str(c.score),
								"tokens": c.tokens,
								"representative_id": c.representative.original_id,
								"representative_side": c.representative.side,
								"initially_selected": i in self.initial_selected,
								"selected": i in self.selected,
								"new_id": c.new_id,
								"sources_json": json.dumps(
									[asdict(source) for source in c.sources],
									ensure_ascii=False,
								),
							}
						)
					writer.write_table(
						pa.Table.from_pylist(records, schema=writer.schema)
					)
					if start % 32768 == 0:
						print(
							f"Candidate audit: {min(start + 1024, len(self.candidates)):,}/{len(self.candidates):,}; "
							f"{sink.size / 1024**2:.1f} MiB written.",
							flush=True,
						)
			sink.publish()
		finally:
			sink.close()
		self._jsonl(stage / "selection.jsonl", self.selections)
		self._jsonl(stage / "replacements.jsonl", self.replacements)
		pd.DataFrame(
			[
				{
					"prompt": p,
					"group_id": self.rows[self.eligible[p][0]].group_id,
					"final_selected_count": self.counts.get(p, 0),
				}
				for p in sorted(self.eligible)
			],
			columns=["prompt", "group_id", "final_selected_count"],
		).to_csv(stage / "prompt_counts.csv", index=False, lineterminator="\n")

	def _verify(self, stage: Path, expected: pd.DataFrame) -> None:
		original = TrainingInputs(self.original, self.source_sha256)
		loaded = BuildSupplementLoader(stage / self.filename, scope=self.scope).load(
			original
		)
		self._require(
			loaded.data.equals(expected),
			"Supplement Parquet roundtrip changed values, order or dtypes.",
		)
		with CacheStorage(stage / "candidates.parquet").open("r") as session:
			audit = pd.read_parquet(
				session.path,
				columns=["selected", "new_id", "score", "sources_json", "occurrences"],
			)
		self._require(
			len(audit) == self.report["candidate_counts"]["unique_candidates"]
			and int(audit.selected.sum()) == self.count,
			"Candidate audit count mismatch.",
		)
		self._require(
			set(audit.loc[audit.selected, "new_id"].astype("int64"))
			== set(expected.id),
			"Candidate audit synthetic IDs mismatch.",
		)
		self._require(
			all(
				len(json.loads(row.sources_json)) == row.occurrences
				for row in audit.itertuples(index=False)
			),
			"Candidate audit lost source occurrences.",
		)
		self._require(
			sum(
				(
					Fraction(row.score)
					for row in audit.loc[audit.selected].itertuples(index=False)
				),
				Fraction(),
			)
			== Fraction(self.count, 2),
			"Candidate audit score mismatch.",
		)
		with CacheStorage(self.config.data.processed_path).open("r") as session:
			self._require(
				self._digest(session.path) == self.source_sha256,
				"Original dataset changed during generation; publication cancelled.",
			)
		self.report["supplement_sha256"] = loaded.fingerprint
		self.report["training_source_sha256"] = (
			BuildSupplementLoader.combined_fingerprint(
				self.source_sha256, loaded.fingerprint
			)
		)
		self.report["checks"] = {
			"original_file_values_order_and_partitions_unchanged": True,
			"supplement_roundtrip_values_order_dtypes_preserved": True,
			"unique_ids_without_original_collisions": True,
			"unique_synthetic_combinations_without_original_duplicates": True,
			"original_orientation_self_response_ties_only": True,
			"initial_prompt_coverage_preserved": True,
			"exact_count_and_half_mean_win_rate": True,
			"all_candidate_sources_retained": True,
			"historical_counts_recomputed_from_real_sources": True,
			"scope_source_membership_and_validation_isolation": True,
		}
