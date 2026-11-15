"""Extract pooled CPU features for head-only warmup and current-model inference."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ...errors import ModelExecutionError
from ..common import ExtractionBatchSampler, TrainingProgress
from .data import BranchBatch, PairwiseFeatureStore
from .model import LoRAQwenClassifier


class LoRAEmbeddingExtractor:
	"""Keep paired features from the current backbone in temporary CPU memory."""

	def __init__(self, model_label: str = "LoRA") -> None:
		self._model_label = model_label

	@torch.no_grad()
	def extract(
		self,
		model: LoRAQwenClassifier,
		loader: DataLoader[BranchBatch],
		row_ids: tuple[int, ...],
		*,
		stage: str,
		use_cross_attention: bool,
		target: torch.Tensor | None = None,
		progress: TrainingProgress | None = None,
		description: str = "",
	) -> PairwiseFeatureStore:
		sampler = loader.batch_sampler
		if not isinstance(sampler, ExtractionBatchSampler):
			raise TypeError(
				"Feature extraction requires a fixed ExtractionBatchSampler."
			)
		model.eval()
		feature_dtype = torch.float32 if use_cross_attention else model.feature_dtype
		shape = (len(loader.dataset), model.backbone_hidden_size)
		if target is None:
			features = torch.empty(shape, dtype=feature_dtype, device="cpu")
		else:
			if (
				target.shape != shape
				or target.dtype != feature_dtype
				or target.device.type != "cpu"
			):
				raise ValueError(
					"Feature extraction target must match the CPU feature layout."
				)
			features = target
		for index, (indices, batch) in enumerate(zip(sampler, loader, strict=True)):
			pooled = model.encode_branch(
				batch, use_cross_attention=use_cross_attention
			).to(device="cpu", dtype=features.dtype)
			if not torch.isfinite(pooled).all():
				raise ModelExecutionError(
					f"{self._model_label} {stage} extraction produced non-finite features."
				)
			features[indices] = pooled
			del pooled
			if progress is not None:
				progress.advance(
					f"{description} branch batch {index + 1:,}/{len(loader):,}",
					stage=f"{stage} feature extraction",
				)
			elif index == 0 or (index + 1) % 100 == 0:
				print(
					f"{self._model_label} {stage} extraction branch batch {index + 1:,}/{len(loader):,}.",
					flush=True,
				)
		return PairwiseFeatureStore(
			row_ids, features.view(len(row_ids), 2, model.backbone_hidden_size)
		)
