"""Match ranked external training rows to the Kaggle rows in each training scope."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from ...errors import ModelExecutionError
from .cache_storage import CacheStorage
from .inputs import TrainingInputs


class RankedExternalSelector:
	"""Keep the Kaggle split and take the highest-ranked external rows within each quota."""

	def __init__(self, path: Path) -> None:
		self._storage = CacheStorage(path)

	def select(self, inputs: TrainingInputs, *, full_data: bool) -> TrainingInputs:
		data = inputs.data
		train = data["split"].eq("train")
		kaggle = data["source"].eq("kaggle")
		if data["source"].isna().any() or not kaggle.loc[~train].all():
			raise ModelExecutionError(
				"Ranked external selection requires known sources and a Kaggle-only validation split."
			)
		kaggle_labels = data.loc[kaggle if full_data else kaggle & train, "label"]
		if kaggle_labels.empty:
			raise ModelExecutionError(
				"No Kaggle training rows for ranked external selection."
			)

		tie_count = int(kaggle_labels.eq(2).sum())
		non_tie_count = len(kaggle_labels) - tie_count
		ranking = self._load_ranking()
		candidates = ranking.merge(
			data.loc[~kaggle & train, ["id", "source", "label"]],
			on="id",
			suffixes=("_ranked", ""),
			validate="one_to_one",
		).sort_values("rank")
		if (
			candidates["source_ranked"].ne(candidates["source"])
			| candidates["label_ranked"].ne(candidates["label"])
		).any():
			raise ModelExecutionError(
				f"Ranked external data {self._storage.path} disagrees with the processed "
				"dataset on the source or label of at least one ID."
			)
		is_tie = candidates["label"].eq(2)
		ties = candidates.loc[is_tie, "id"]
		non_ties = candidates.loc[~is_tie, "id"]
		if len(ties) < tie_count or len(non_ties) < non_tie_count:
			raise ModelExecutionError(
				f"Insufficient eligible rows in {self._storage.path}: "
				f"need {tie_count:,} ties and {non_tie_count:,} non-ties, "
				f"found {len(ties):,} ties and {len(non_ties):,} non-ties."
			)

		selected_ids = pd.concat([ties.head(tie_count), non_ties.head(non_tie_count)])
		digest = hashlib.sha256(b"ranked-external-selection-v1\0")
		digest.update(inputs.fingerprint.encode("ascii"))
		digest.update(selected_ids.sort_values().to_numpy(dtype="<i8").tobytes())
		return TrainingInputs(
			data=data.loc[kaggle | data["id"].isin(selected_ids)],
			fingerprint=digest.hexdigest(),
		)

	def _load_ranking(self) -> pd.DataFrame:
		types = {"rank": "int64", "id": "int64", "source": "string", "label": "int64"}
		try:
			with self._storage.open("r") as session:
				ranking = pd.read_csv(session.path, usecols=list(types), dtype=types)
			if (
				ranking["id"].duplicated().any()
				or ranking["rank"].duplicated().any()
				or ranking["rank"].le(0).any()
				or ranking["source"].isna().any()
				or not ranking["label"].isin((0, 1, 2)).all()
			):
				raise ValueError(
					"Expected unique IDs, unique positive ranks, sources, and labels 0–2."
				)
		except (OSError, ValueError) as error:
			raise ModelExecutionError(
				f"Could not read external ranking {self._storage.path}: {error}"
			) from error
		return ranking
