# basemodel.py
"""
===============================================================================
BASE MODEL
-------------------------------------
Supports:
  - Rotary embeddings
  - GQA (Group Query Attention) (set (gqa_num_heads < num_heads) and (gqa_num_heads % num_heads == 0))
  - RMSNorm
  - (Optional) fused cross-entropy that does not materialize logits
  - Segment-aware block mask
  - FAN-in (as seen in JAX, similar to OLMO 2) param inits
===============================================================================
"""

import math
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

import torch.nn.attention.flex_attention as flex_attention
from torch.nn.attention.flex_attention import create_block_mask
from modeling import rotary, softcap


################################################################################
# RMSNorm
################################################################################

class RMSNorm(nn.Module):
	"""
	Root Mean Square Layer Normalization.

	Unlike standard LayerNorm, RMSNorm does not subtract the mean, but rather
	normalizes by the RMS (sqrt of mean of squares) along the channel dimension.
	"""
	def __init__(
		self,
		hidden_size: int,
		eps: float = 1e-5,
		dtype: torch.dtype = torch.float32,
		dropout_p: float = 0.0,
		device: Optional[str] = None,
		**kwargs
	):
		super().__init__()
		# Use PyTorch's built-in nn.RMSNorm
		self.norm = nn.RMSNorm(hidden_size, eps=eps, elementwise_affine=True)
		self.dropout_p = dropout_p  # If you want dropout after norm

	def forward(self, x: torch.Tensor, *args, **kwargs,) -> torch.Tensor:
		"""
		Apply RMSNorm (and optionally dropout if configured).
		"""
		x = self.norm(x)
		# x = F.dropout(x, p=self.dropout_p, training=self.training)
		return x


################################################################################
# Tanh Softcap
################################################################################
# tanh_softcap = softcap.generate_tanh_softcap(soft_cap=50, approx=True)
# tanh_softcap_last_layer = softcap.generate_tanh_softcap(soft_cap=30, approx=True)


################################################################################
# LinearProjection
################################################################################

class LinearProjection(nn.Module):
	"""
	A single linear layer with optional HPC alignment (dims multiple of 128).
	Uses truncated normal initialization for weights.
	"""
	def __init__(self, in_dim: int = None, out_dim: int = None, **kwargs: Any):
		super().__init__()

		# Grab input/output dims
		if in_dim is None:
			in_dim = kwargs["hidden_size"]
		if out_dim is None:
			out_dim = kwargs.get("out_dim", in_dim)

		is_vocab_head = kwargs.get("is_vocab_head", False)
		vocab_size = kwargs.get("vocab_size", 32000)
		if is_vocab_head:
			out_dim = vocab_size

		param_dtype = kwargs.get("dtype", torch.float32)

		# Make in_dim and out_dim multiples of 128 for performance
		in_dim = utils.make_divisible_by(in_dim, 128)
		out_dim = utils.make_divisible_by(out_dim, 128)

		param_dtype = utils.str_to_dtype(param_dtype)

		# Build linear layer
		self.layer = nn.Linear(
			in_dim,
			out_dim,
			bias=kwargs.get("projection_layers_use_bias", False),
			dtype=param_dtype,
		)

		# Truncated normal init
		fan_in = out_dim
		std = 1.0 / math.sqrt(fan_in)
		nn.init.trunc_normal_(
			self.layer.weight,
			mean=0.0,
			std=std,
			a=-2 * std,
			b=2 * std
		)

	def forward(self, x: torch.Tensor, *args, **kwargs,) -> torch.Tensor:
		return self.layer(x)


################################################################################
# MHA (Multi-Head Attention) using flex_attention
################################################################################
class MHA(nn.Module):
	"""
	Custom MHA that:
	  - Splits Q,K,V
	  - Applies flex_attention
	  - Optionally uses post-attention RMSNorm
	  - Applies rotary embeddings if provided
	  - Supports GQA via gqa_num_heads
	"""
	def __init__(self, **kwargs):
		super().__init__()
		self.hidden_size = kwargs["hidden_size"]
		self.num_heads   = kwargs["num_heads"]
		self.head_dim    = kwargs.get("head_dim", self.hidden_size // self.num_heads)
		self.gqa_num_heads = int(kwargs.get("gqa_num_heads", self.num_heads))

		self.freqs_cis = kwargs.get("freqs_cis", None)

		# Q/K dimension for Q,K
		qk_dims = self.num_heads * self.head_dim
		# For GQA, V can differ
		v_dims = self.gqa_num_heads * self.head_dim

		# Q,K,V projections
		self.q_proj = LinearProjection(self.hidden_size, qk_dims, **kwargs)
		self.k_proj = LinearProjection(self.hidden_size, v_dims, **kwargs)
		self.v_proj = LinearProjection(self.hidden_size, v_dims, **kwargs)

		# Final out projection
		self.o_proj = LinearProjection(qk_dims, self.hidden_size, **kwargs)

		# Post-RMSNorm
		self.post_rms_norm = RMSNorm(
			hidden_size=self.hidden_size,
			eps=kwargs.get("eps", 1e-5),
			dtype=kwargs.get("dtype", torch.float32),
			device=kwargs.get("device", None),
		)

	def forward(self, x: torch.Tensor, block_mask=None, *args, **kwargs,) -> torch.Tensor:
		"""
		Forward pass for the MHA block.
		"""
		residual = x

		# Project to Q,K,V
		q = self.q_proj(x)
		k = self.k_proj(x)
		v = self.v_proj(x)

		B, S, _ = q.shape

		# Reshape to (B, heads, S, head_dim)
		q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
		k = k.view(B, S, self.gqa_num_heads, self.head_dim).transpose(1, 2)
		v = v.view(B, S, self.gqa_num_heads, self.head_dim).transpose(1, 2)

		(q, k) = rotary.apply_rotary_emb(q, k, self.freqs_cis)

		# flex_attention
		attn_out = flex_attention.flex_attention(
			query=q,
			key=k,
			value=v,
			block_mask=block_mask,
			enable_gqa=(self.num_heads != self.gqa_num_heads)
		)
		# Shape -> (B, num_heads, S, head_dim)

		# Reshape back
		attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)

		# Output projection
		out = self.o_proj(attn_out)

		# Optional norm
		if self.post_rms_norm:
			out = self.post_rms_norm(out)

		# Residual
		return (residual + out).to(x.dtype)


################################################################################
# SwiGLU Feed-Forward
################################################################################

class SwiGLU(nn.Module):
	"""
	A feed-forward block using the 'SwiGLU' pattern:
	  - gate = sigmoid(Linear(x))
	  - ungated = Linear(x)
	  - multiply gate * ungated
	  - project down
	  - RMSNorm
	  - add residual
	"""
	def __init__(self, **kwargs):
		super().__init__()
		hidden_size = kwargs["hidden_size"]
		mlp_mult = kwargs.get("mlp_dim_mult", 4.0)

		# Round hidden_size to multiple of 16 for HPC alignment if desired
		in_dim = utils.make_divisible_by(hidden_size, 16)
		ffn_dim = utils.make_divisible_by(int(in_dim * mlp_mult), 16)

		# Two parallel input layers
		self.wi_0 = LinearProjection(in_dim, ffn_dim, **kwargs)
		self.wi_1 = LinearProjection(in_dim, ffn_dim, **kwargs)

		# Output projection
		self.wo = LinearProjection(ffn_dim, in_dim, **kwargs)

		# RMSNorm
		self.rms_norm = RMSNorm(
			hidden_size=in_dim,
			eps=kwargs.get("eps", 1e-5),
			dtype=kwargs.get("dtype", torch.float32),
			device=kwargs.get("device", None),
		)

	def forward(self, x: torch.Tensor, *args, **kwargs,) -> torch.Tensor:
		"""
		Forward pass of SwiGLU feed-forward.
		"""
		residual = x
		gated = torch.sigmoid(self.wi_0(x))
		ungated = self.wi_1(x)
		ff_out = gated * ungated

		ff_out = self.wo(ff_out)
		ff_out = self.rms_norm(ff_out)
		return ff_out + residual
	
class Embedding(nn.Embedding):
	"""
	Wrapper to allow for *args and **kwargs.
	"""
	def forward(self, x: torch.Tensor, *args, **kwargs,) -> torch.Tensor:
		return super().forward(x)


################################################################################
# BaseModel
################################################################################

class BaseModel(nn.Module):
	"""
	A flexible Transformer-like model that:
	  1) Has an embedding layer (vocab_size x hidden_size).
	  2) Stacks MHA + MLP layers (with optional RMSNorm, GQA, rotary, etc.).
	  3) Ends with a final RMSNorm, and has a projection to vocab_size (assuming we're doing LM).

	If label_ids is provided, returns cross-entropy loss. Otherwise returns logits.
	"""
	def __init__(self, layer_kwargs):
		super().__init__()

		# Basic config
		self.num_layers      = layer_kwargs["num_layers"]
		self.hidden_size     = layer_kwargs.get("hidden_size", 2048)
		self.vocab_size      = layer_kwargs.get("vocab_size", 32000)
		self.sequence_length = layer_kwargs.get("sequence_length", 512)
		self.batch_size      = layer_kwargs.get("batch_size", 1)

		self.num_heads = layer_kwargs["num_heads"]
		# We'll store head_dim here to ensure freq_cis dimension is consistent
		self.head_dim  = layer_kwargs.get("head_dim", self.hidden_size // self.num_heads)

		self.use_precomputed_block_mask = int(layer_kwargs.get("use_precomputed_block_mask", 0))
		self.use_fusedlce = int(layer_kwargs.get("use_fusedlce", 0))

		# Embedding
		self.embeddings = Embedding(self.vocab_size, self.hidden_size)
		fan_in = self.hidden_size
		std = 1.0 / math.sqrt(fan_in)
		nn.init.trunc_normal_(
			self.embeddings.weight,
			mean=0.0,
			std=std,
			a=-2 * std,
			b=2 * std
		)

		# Precompute rotary embeddings
		# Important: dim must be (head_dim // 2) to match shape
		self.freqs_cis = rotary.precompute_freqs_cis(
			dim=(self.head_dim),
			end=self.sequence_length,
			theta=layer_kwargs.get("base_decay_rate", 500_000),
			use_scaled=False,
			old_context_len=layer_kwargs.get("pretraining_sequence_length", self.sequence_length)
		)

		# Build sub-layers
		layers = []
		seen_layers = 0
		while seen_layers < self.num_layers:
			if layer_kwargs.get("use_attention", True):
				mha = MHA(
					**layer_kwargs,
					freqs_cis=self.freqs_cis,
				)
				layers.append(mha)
				seen_layers += 1

			if layer_kwargs.get("use_mlp", True):
				ffn = SwiGLU(**layer_kwargs)
				layers.append(ffn)
				seen_layers += 1

		# Final RMSNorm
		layers.append(
			RMSNorm(
				hidden_size=self.hidden_size,
				eps=layer_kwargs.get("eps", 1e-5),
				dtype=layer_kwargs.get("dtype", torch.float32),
				device=layer_kwargs.get("device", None),
			)
		)

		# Vocab head
		head_in_dim = utils.make_divisible_by(self.hidden_size, 128)
		head_out_dim = self.vocab_size
		self.vocab_head = nn.Parameter(torch.randn(head_in_dim, head_out_dim), requires_grad=True)
		fan_in_head = head_out_dim
		std_head = 1.0 / math.sqrt(fan_in_head)
		nn.init.trunc_normal_(
			self.vocab_head,
			mean=0.0,
			std=std_head,
			a=-2 * std_head,
			b=2 * std_head
		)

		# Optionally import fused LCE
		if self.use_fusedlce:
			from cut_cross_entropy import LinearCrossEntropy
			self.LCE = LinearCrossEntropy()

		# Construct layers
		self.layers = nn.ModuleList()
		self.layers.append(self.embeddings)  # first is embedding
		self.layers.extend(layers)

		# Possibly build block_mask once
		if self.use_precomputed_block_mask:
			def mask_mod(b, h, q_idx, kv_idx):
				# Strictly causal
				return (q_idx >= kv_idx)
			self.block_mask = create_block_mask(
				mask_mod,
				B=self.batch_size,
				H=1,
				Q_LEN=self.sequence_length,
				KV_LEN=self.sequence_length,
				device=layer_kwargs.get("device", "cuda"),
				_compile=True
			)
		else:
			self.block_mask = None

		# Simple cross-entropy per sample
		def cross_entropy_per_sample(logits, label_ids):
			"""
			logits: (B, L, vocab_size)
			label_ids: (B, L) with -100 to ignore
			Returns per-sample average, shape (B,)
			"""
			logits_t = logits.permute(0, 2, 1)  # -> (B, vocab_size, L)
			label_ids_ = label_ids.to(torch.long)
			loss_per_pos = F.cross_entropy(logits_t, label_ids_, reduction='none')
			mask = (label_ids_ != -100)
			sum_loss = (loss_per_pos * mask).sum(dim=1)
			count = mask.sum(dim=1).clamp(min=1)
			return sum_loss / count

		self.cross_entropy_per_sample = cross_entropy_per_sample

	def reset_freq_cis(self, seq_len: int, base_decay_rate: float = 500_000, old_context_len: int = None):
		"""
		Recompute rotary embeddings if sequence length changes.
		dim must still be (self.head_dim // 2).
		"""
		if old_context_len is None:
			old_context_len = seq_len
		self.sequence_length = seq_len
		self.freqs_cis = rotary.precompute_freqs_cis(
			dim=(self.head_dim),
			end=seq_len,
			theta=base_decay_rate,
			use_scaled=(old_context_len != seq_len),
			old_context_len=old_context_len,
		)
		return self

	def forward(
		self,
		input_ids: torch.Tensor,
		label_ids: Optional[torch.Tensor] = None,
		segment_ids: Optional[torch.Tensor] = None,
		**kwargs
	) -> torch.Tensor:
		"""
		Forward pass. If label_ids provided, return scalar cross-entropy. Otherwise logits.
		"""
		B, L = input_ids.shape
		if segment_ids is None:
			segment_ids = torch.zeros_like(input_ids, dtype=torch.long, device=input_ids.device)

		# Build block_mask if not precomputed
		if not self.use_precomputed_block_mask:
			def mask_mod(b, h, q_idx, kv_idx):
				# Causal + same segment
				same_segment = (segment_ids[b, q_idx] == segment_ids[b, kv_idx])
				return (q_idx >= kv_idx) & same_segment

			block_mask = create_block_mask(
				mask_mod,
				B=B,
				H=1,
				Q_LEN=L,
				KV_LEN=L,
				device=input_ids.device,
				_compile=True
			)
		else:
			block_mask = self.block_mask

		# Pass through the layers
		x = input_ids
		for layer in self.layers:
			x = layer(x, block_mask=block_mask)


		# If label_ids were not provided, return logits
		if label_ids is None:
			B, L, D = x.shape
			logits = torch.matmul(x.view(-1, D), self.vocab_head.to(x.dtype))
			logits = logits.view(B, L, self.vocab_size)
			return logits

		# Else compute cross-entropy
		# 1) If fused LCE is enabled:
		if self.use_fusedlce:
			logits_16 = x.to(torch.float16)
			w_16 = self.vocab_head.transpose(0, 1).to(torch.float16)
			return self.LCE(logits_16, w_16, label_ids).mean()
		# 2) Otherwise standard cross-entropy:
		logits = torch.matmul(x.view(-1, x.shape[-1]), self.vocab_head.to(x.dtype))
		logits = logits.view(B, L, self.vocab_size)
		per_sample_loss = self.cross_entropy_per_sample(logits, label_ids)
		return per_sample_loss.mean()
