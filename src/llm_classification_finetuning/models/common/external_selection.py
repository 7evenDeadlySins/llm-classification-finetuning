"""Match ranked external training tokens to the Kaggle budget in each scope."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from ...errors import ModelExecutionError
from .cache_storage import CacheStorage
from .inputs import TrainingInputs


class RankedExternalSelector:
	"""Keep ranked tie and non-tie prefixes at the Kaggle training row ratio."""

	def __init__(self, path: Path) -> None:
		self._storage = CacheStorage(path)

	def select(
		self,
		inputs: TrainingInputs,
		*,
		full_data: bool,
		load_token_totals: Callable[[pd.DataFrame, str], pd.Series],
	) -> TrainingInputs:
		data = inputs.data
		train = data["split"].eq("train")
		kaggle = data["source"].eq("kaggle")
		if data["source"].isna().any() or not kaggle.loc[~train].all():
			raise ModelExecutionError(
				"Ranked external selection requires known sources and a Kaggle-only validation split."
			)
		kaggle_training = kaggle if full_data else kaggle & train
		kaggle_rows = data.loc[kaggle_training, ["id", "label"]]
		if kaggle_rows.empty:
			raise ModelExecutionError(
				"No Kaggle training rows for ranked external selection."
			)

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
		token_totals = load_token_totals(
			data.loc[
				kaggle_training | data["id"].isin(candidates["id"]),
				["id", "prompt", "response_a", "response_b"],
			],
			inputs.fingerprint,
		)
		selected_ids = self._select_ids(kaggle_rows, candidates, token_totals)
		digest = hashlib.sha256(b"ranked-external-token-selection-v2\0")
		digest.update(inputs.fingerprint.encode("ascii"))
		digest.update(selected_ids.sort_values().to_numpy(dtype="<i8").tobytes())
		return TrainingInputs(
			data=data.loc[kaggle | data["id"].isin(selected_ids)],
			fingerprint=digest.hexdigest(),
		)

	def _select_ids(
		self, kaggle: pd.DataFrame, candidates: pd.DataFrame, token_totals: pd.Series
	) -> pd.Series:
		target = int(token_totals.loc[kaggle["id"]].sum())
		is_tie = candidates["label"].eq(2)
		ties = candidates.loc[is_tie, "id"]
		non_ties = candidates.loc[~is_tie, "id"]
		tie_prefix = np.concatenate(([0], token_totals.loc[ties].cumsum().to_numpy()))
		non_tie_prefix = np.concatenate(
			([0], token_totals.loc[non_ties].cumsum().to_numpy())
		)
		sizes = np.arange(1, len(candidates) + 1, dtype=np.int64)
		kaggle_tie_count = int(kaggle["label"].eq(2).sum())
		# Round the Kaggle tie share to the nearest whole row at each possible size.
		tie_counts = (sizes * kaggle_tie_count + len(kaggle) // 2) // len(kaggle)
		non_tie_counts = sizes - tie_counts
		eligible = (tie_counts <= len(ties)) & (non_tie_counts <= len(non_ties))
		tie_counts = tie_counts[eligible]
		non_tie_counts = non_tie_counts[eligible]
		totals = tie_prefix[tie_counts] + non_tie_prefix[non_tie_counts]
		available = int(totals[-1]) if totals.size else 0
		if available < target:
			raise ModelExecutionError(
				f"Insufficient eligible rows in {self._storage.path}: "
				f"the Kaggle budget is {target:,} tokens, but ranked rows at the "
				f"Kaggle tie share can supply only {available:,} tokens."
			)
		closest = int(np.abs(totals - target).argmin())
		return pd.concat(
			[
				ties.head(int(tie_counts[closest])),
				non_ties.head(int(non_tie_counts[closest])),
			]
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
