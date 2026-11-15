"""Shared last-token-query attention with FP32 normalization and residuals."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LastTokenQueryCrossAttention(nn.Module):
	"""Attend over real tokens and share residual dropout across a response pair."""

	ARCHITECTURE = "last_token_query_cross_attention_v1"

	def __init__(
		self,
		backbone_hidden_size: int,
		attention_dim: int,
		heads: int,
		dropout: float,
		*,
		compute_dtype: torch.dtype,
	) -> None:
		super().__init__()
		if attention_dim <= 0 or heads <= 0 or attention_dim % heads:
			raise ValueError(
				"Attention dimension must be positive and divisible by heads."
			)
		self.heads = heads
		self.head_dim = attention_dim // heads
		self._compute_dtype = compute_dtype
		self.norm_q = nn.LayerNorm(
			backbone_hidden_size, eps=1e-5, elementwise_affine=True
		)
		self.norm_kv = nn.LayerNorm(
			backbone_hidden_size, eps=1e-5, elementwise_affine=True
		)
		self.q_proj = nn.Linear(backbone_hidden_size, attention_dim, bias=False)
		self.k_proj = nn.Linear(backbone_hidden_size, attention_dim, bias=False)
		self.v_proj = nn.Linear(backbone_hidden_size, attention_dim, bias=False)
		self.o_proj = nn.Linear(attention_dim, backbone_hidden_size, bias=False)
		self.dropout = nn.Dropout(dropout)
		for projection in (self.q_proj, self.k_proj, self.v_proj):
			nn.init.xavier_uniform_(projection.weight)
		nn.init.zeros_(self.o_proj.weight)
		for norm in (self.norm_q, self.norm_kv):
			nn.init.ones_(norm.weight)
			nn.init.zeros_(norm.bias)

	def make_dropout_mask(self, batch_size: int) -> Tensor | None:
		"""Draw one FP32 [B, D] mask for both branches of the same pair batch."""
		if not self.training or self.dropout.p == 0.0:
			return None
		return self.dropout(
			self.o_proj.weight.new_ones((batch_size, self.o_proj.out_features))
		)

	def forward(
		self,
		hidden: Tensor,
		last_positions: Tensor,
		residual_dropout_mask: Tensor | None = None,
	) -> Tensor:
		"""Return FP32 features; the caller supplies paired training dropout masks."""
		if self.training and self.dropout.p and residual_dropout_mask is None:
			raise ValueError(
				"Training attention pooling requires a shared A/B dropout mask."
			)
		batch_size, length, _ = hidden.shape
		rows = torch.arange(batch_size, device=hidden.device)
		last = hidden[rows, last_positions]
		device_type = hidden.device.type
		with torch.autocast(device_type=device_type, enabled=False):
			query_input = self.norm_q(last.float())
			kv_input = self.norm_kv(hidden.float())
		with torch.autocast(
			device_type=device_type,
			dtype=self._compute_dtype,
			enabled=self._compute_dtype != torch.float32,
		):
			query = self.q_proj(query_input).view(
				batch_size, self.heads, 1, self.head_dim
			)
			key = self.k_proj(kv_input).view(
				batch_size, length, self.heads, self.head_dim
			)
			value = self.v_proj(kv_input).view(
				batch_size, length, self.heads, self.head_dim
			)
		del query_input, kv_input
		key, value = key.transpose(1, 2), value.transpose(1, 2)
		with torch.autocast(device_type=device_type, enabled=False):
			# Promote before QK matmul, so FP16 logits cannot overflow before softmax.
			scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
			scores = scores * (self.head_dim**-0.5)
			valid = (
				torch.arange(length, device=hidden.device)[None, :]
				<= last_positions[:, None]
			)
			scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
			probabilities = scores.softmax(dim=-1)
		with torch.autocast(
			device_type=device_type,
			dtype=self._compute_dtype,
			enabled=self._compute_dtype != torch.float32,
		):
			# Only the probability/value product and projections use the compute dtype.
			attended = torch.matmul(probabilities.to(value.dtype), value)
			output = self.o_proj(attended.reshape(batch_size, -1))
		with torch.autocast(device_type=device_type, enabled=False):
			output = output.float()
			if residual_dropout_mask is not None:
				output = output * residual_dropout_mask
			return last.float() + output
