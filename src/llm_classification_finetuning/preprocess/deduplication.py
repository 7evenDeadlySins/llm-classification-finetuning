"""Deduplicate external battles while isolating frozen validation prompts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import pandas as pd

from ..errors import DataPreparationError


@dataclass(frozen=True, slots=True)
class ExternalDeduplicationResult:
	"""Retained source rows and one audit entry for every external input row."""

	data: pd.DataFrame
	audit: pd.DataFrame
	vote_counts: pd.DataFrame = field(default_factory=pd.DataFrame)


class ExternalVoteConflictError(DataPreparationError):
	"""One purported vote has contradictory battle content or corrected labels."""

	def __init__(self, conflicts: list[dict]) -> None:
		super().__init__(
			f"Conflicting copies of {len(conflicts):,} external votes; see vote_conflicts.parquet."
		)
		self.conflicts = conflicts


class ExternalBattleDeduplicator:
	"""Compare complete battles, preserving distinct pairs for training prompts."""

	COLUMNS = ("id", "prompt", "response_a", "response_b")
	AUDIT_COLUMNS = (
		"id",
		"group_id",
		"battle_id",
		"status",
		"external_representative_id",
		"retained_id",
		"validation_representative_id",
		"overlaps_training_prompt",
		"overlaps_validation_prompt",
		"duplicates_training_battle",
	)

	def deduplicate(
		self,
		external: pd.DataFrame,
		existing: pd.DataFrame,
		*,
		merge_votes: bool = False,
	) -> ExternalDeduplicationResult:
		"""Keep first distinct external battles after validation/training matching.

		Inputs contain original rows with JSON-array conversation fields. Existing
		rows include their frozen train/validation membership and are never removed.
		Response-pair identity ignores A/B order, labels, model names and source IDs.
		"""
		if merge_votes:
			return self._merge_votes(external, existing)
		for name, data, columns in (
			("External", external, self.COLUMNS),
			("Existing", existing, (*self.COLUMNS, "split")),
		):
			missing = set(columns) - set(data.columns)
			if missing:
				raise DataPreparationError(
					f"{name} battle data is missing columns: {', '.join(sorted(missing))}."
				)
		if not existing["split"].isin(("train", "validation")).all():
			raise DataPreparationError(
				"Existing battles must belong to train or validation."
			)

		training_prompts: set[str] = set()
		training_battles: dict[str, int] = {}
		validation_prompts: dict[str, int] = {}
		for row_id, split, (group_id, battle_id) in zip(
			existing["id"], existing["split"], self._identities(existing), strict=True
		):
			if split == "validation":
				validation_prompts.setdefault(group_id, int(row_id))
			else:
				training_prompts.add(group_id)
				training_battles.setdefault(battle_id, int(row_id))

		representatives: dict[str, int] = {}
		kept: dict[str, int] = {}
		positions: list[int] = []
		entries: list[dict[str, object]] = []
		for position, (row_id, (group_id, battle_id)) in enumerate(
			zip(external["id"], self._identities(external), strict=True)
		):
			row_id = int(row_id)
			representative = representatives.setdefault(battle_id, row_id)
			validation_id = validation_prompts.get(group_id)
			training_id = training_battles.get(battle_id)
			if validation_id is not None:
				status, retained_id = "excluded_validation_prompt", pd.NA
			elif training_id is not None:
				status, retained_id = "duplicate_training", training_id
			elif battle_id in kept:
				status, retained_id = "duplicate_external", kept[battle_id]
			else:
				status, retained_id = "retained", row_id
				kept[battle_id] = row_id
				positions.append(position)
			entries.append(
				{
					"id": row_id,
					"group_id": group_id,
					"battle_id": battle_id,
					"status": status,
					"external_representative_id": representative,
					"retained_id": retained_id,
					"validation_representative_id": (
						pd.NA if validation_id is None else validation_id
					),
					"overlaps_training_prompt": group_id in training_prompts,
					"overlaps_validation_prompt": validation_id is not None,
					"duplicates_training_battle": training_id is not None,
				}
			)
		audit = pd.DataFrame(entries, columns=self.AUDIT_COLUMNS)
		for column in (
			"id",
			"external_representative_id",
			"retained_id",
			"validation_representative_id",
		):
			audit[column] = audit[column].astype("Int64")
		return ExternalDeduplicationResult(
			data=external.iloc[positions].copy().reset_index(drop=True), audit=audit
		)

	def _merge_votes(
		self, external: pd.DataFrame, existing: pd.DataFrame
	) -> ExternalDeduplicationResult:
		"""Count independent hard votes, dropping battles with tied highest counts."""
		for name, data, columns in (
			("External", external, (*self.COLUMNS, "label", "vote_id")),
			("Kaggle", existing, (*self.COLUMNS, "split")),
		):
			missing = set(columns) - set(data.columns)
			if missing:
				raise DataPreparationError(
					f"{name} battle data is missing columns: {', '.join(sorted(missing))}."
				)
		if not existing["split"].isin(("train", "validation")).all() or (
			"source" in existing and not existing["source"].eq("kaggle").all()
		):
			raise DataPreparationError(
				"Vote merging requires only frozen Kaggle train/validation rows as existing data."
			)
		if (
			external["id"].duplicated().any()
			or not external["label"].isin((0, 1, 2)).all()
			or not external["vote_id"]
			.map(lambda v: isinstance(v, str) and bool(v))
			.all()
		):
			raise DataPreparationError(
				"External vote IDs, row IDs, or hard labels are invalid."
			)
		training, validation = {}, {}
		training_prompts = set()
		for row in existing.itertuples(index=False):
			group, battle, _ = self.battle_identity(
				row.prompt, row.response_a, row.response_b
			)
			if row.split == "validation":
				validation.setdefault(group, int(row.id))
			else:
				training.setdefault(battle, int(row.id))
				training_prompts.add(group)
		rows = external.sort_values("id", kind="stable").reset_index(drop=True)
		entries, battles, votes, conflicts = [], {}, {}, []
		for position, row in enumerate(rows.itertuples(index=False)):
			group, battle, flipped = self.battle_identity(
				row.prompt, row.response_a, row.response_b
			)
			label = self._orient_label(int(row.label), flipped)
			prior = votes.get(row.vote_id)
			if prior is not None and (battle, label) != prior[:2]:
				conflicts.append(
					{
						"vote_id": row.vote_id,
						"id": int(row.id),
						"first_id": prior[2],
						"battle_id": battle,
						"first_battle_id": prior[0],
						"canonical_label": label,
						"first_canonical_label": prior[1],
					}
				)
			if prior is None:
				votes[row.vote_id] = (battle, label, int(row.id))
			entry = {
				"id": int(row.id),
				"group_id": group,
				"battle_id": battle,
				"vote_id": row.vote_id,
				"input_label": int(row.label),
				"canonical_label": label,
				"flipped_to_canonical": flipped,
				"vote_representative_id": int(row.id) if prior is None else prior[2],
				"duplicate_vote": prior is not None,
				"overlaps_training_prompt": group in training_prompts,
				"validation_representative_id": validation.get(group),
				"training_representative_id": training.get(battle),
			}
			entries.append(entry)
			if group in validation or battle in training:
				continue
			state = battles.setdefault(
				battle,
				{
					"position": position,
					"id": int(row.id),
					"flipped": flipped,
					"counts": [0, 0, 0],
					"sources": set(),
				},
			)
			if prior is None:
				state["counts"][label] += 1
				if hasattr(row, "source"):
					state["sources"].add(row.source)
		if conflicts:
			raise ExternalVoteConflictError(conflicts)
		positions, final_labels, vote_counts = [], [], []
		for battle, state in battles.items():
			counts = state["counts"]
			winners = [i for i, count in enumerate(counts) if count == max(counts)]
			final = (
				self._orient_label(winners[0], state["flipped"])
				if len(winners) == 1
				else None
			)
			state["label"] = final
			if final is not None:
				positions.append(state["position"])
				final_labels.append(final)
			representative_counts = (
				[counts[1], counts[0], counts[2]] if state["flipped"] else counts
			)
			vote_counts.append(
				{
					"battle_id": battle,
					"representative_id": state["id"],
					"votes_a": representative_counts[0],
					"votes_b": representative_counts[1],
					"votes_tie": representative_counts[2],
					"independent_votes": sum(counts),
					"final_label": final,
					"status": "excluded_vote_tie" if final is None else "retained",
					"sources_json": json.dumps(sorted(state["sources"])),
				}
			)
		for entry in entries:
			state = battles.get(entry["battle_id"])
			entry["final_label"] = None
			entry["counted_vote"] = False
			if entry["validation_representative_id"] is not None:
				status, retained = "excluded_validation_prompt", None
			elif entry["training_representative_id"] is not None:
				status, retained = (
					"duplicate_training",
					entry["training_representative_id"],
				)
			else:
				entry["final_label"] = state["label"]
				entry["counted_vote"] = not entry["duplicate_vote"]
				retained = state["id"] if state["label"] is not None else None
				status = (
					"excluded_vote_tie"
					if retained is None
					else "duplicate_vote"
					if entry["duplicate_vote"]
					else "retained"
					if entry["id"] == retained
					else "merged_vote"
				)
			entry.update(
				status=status,
				retained_id=retained,
				external_representative_id=None if state is None else state["id"],
			)
		data = rows.iloc[positions].copy().reset_index(drop=True)
		data["label"] = pd.Series(final_labels, dtype="int8")
		for label, name in enumerate(
			("winner_model_a", "winner_model_b", "winner_tie")
		):
			data[name] = data["label"].eq(label).astype("int64")
		audit = pd.DataFrame(entries)
		counts = pd.DataFrame(vote_counts)
		for frame in (audit, counts):
			for column in (
				"final_label",
				"retained_id",
				"external_representative_id",
				"validation_representative_id",
				"training_representative_id",
			):
				if column in frame:
					frame[column] = frame[column].astype("Int64")
		return ExternalDeduplicationResult(data=data, audit=audit, vote_counts=counts)

	@staticmethod
	def _orient_label(label: int, flipped: bool) -> int:
		return 1 - label if flipped and label != 2 else label

	@classmethod
	def battle_identity(
		cls, prompt: str, response_a: str, response_b: str
	) -> tuple[str, str, bool]:
		"""Return canonical prompt/battle hashes and whether A/B order was reversed."""
		prompt = cls._canonical(prompt)
		a, b = cls._canonical(response_a), cls._canonical(response_b)
		battle = json.dumps(
			[prompt, *sorted((a, b))], ensure_ascii=False, separators=(",", ":")
		)
		return (
			hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
			hashlib.sha256(battle.encode("utf-8")).hexdigest(),
			a > b,
		)

	@classmethod
	def _identities(cls, data: pd.DataFrame) -> list[tuple[str, str]]:
		identities = []
		for row in data.loc[:, list(cls.COLUMNS)].itertuples(index=False):
			try:
				prompt = cls._canonical(row.prompt)
				answers = sorted(
					(cls._canonical(row.response_a), cls._canonical(row.response_b))
				)
				battle = json.dumps(
					[prompt, *answers], ensure_ascii=False, separators=(",", ":")
				)
			except (TypeError, ValueError) as error:
				raise DataPreparationError(
					f"Invalid conversation JSON for battle ID {row.id}: {error}"
				) from error
			identities.append(
				(
					hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
					hashlib.sha256(battle.encode("utf-8")).hexdigest(),
				)
			)
		return identities

	@staticmethod
	def _canonical(value: str) -> str:
		return json.dumps(
			json.loads(value),
			ensure_ascii=False,
			sort_keys=True,
			separators=(",", ":"),
			allow_nan=False,
		)
