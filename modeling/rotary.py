# modeling/rotary.py
"""
=============================================================================
ROTARY POSITIONAL EMBEDDINGS
----------------------------
Modified so that 'dim' is the *total embedding dimension per head*, not half.
We do NOT slice to [:dim//2]. Instead, we produce (end, dim).
apply_rotary_emb then automatically splits real & imag by factor of 2.
=============================================================================
"""

import math
from typing import Optional, Tuple

import torch

def apply_scaling(freqs: torch.Tensor, old_context_len: int) -> torch.Tensor:
    """
    (Optional) Example function for frequency scaling.
    """
    scale_factor = 8.0
    low_freq_factor = 1
    high_freq_factor = 4

    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < (old_context_len / high_freq_factor):
            new_freqs.append(freq)
        elif wavelen > (old_context_len / low_freq_factor):
            new_freqs.append(freq / scale_factor)
        else:
            new_freqs.append(freq / (scale_factor * 0.5))

    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 500000.0,
    use_scaled: bool = False,
    old_context_len: int = 32768,
    accelerator=None,
):
    """
    Produces a tensor of shape (end, dim) that encodes the cos/sin phases used
    for rotary embeddings. We do NOT slice to (dim//2), so if dim=64,
    the output shape is (end, 64).

    Inside this function:
      - We generate 'dim/2' distinct frequency values (because we step by +2).
      - The final shape is (end, dim/2) in frequency space. Then we convert
        to complex representation, effectively doubling to 'dim'.

    But to keep it simpler for assertion checks, we directly arrange it as
    shape (end, dim). That matches x.shape[-1] if x has dimension 64.

    Steps:
      1) We create a range [0, 2, 4, ..., dim) => length = dim//2
      2) Then do outer product => (end, dim//2)
      3) Convert to a complex representation => shape still (end, dim//2),
         but each entry is complex => effectively 2 * (dim//2) = dim real floats.
      4) We'll store it as a 'real dimension' of 'dim', so we flatten real+imag
         in apply_rotary_emb. This means the final .shape is (end, dim).

    Alternatively, you can store it as shape (end, dim//2) in a complex dtype.
    For clarity, we'll keep it as is and do an assertion in reshape_for_broadcast.
    """
    if accelerator is not None:
        accelerator.print(
            f"precompute_freqs_cis(dim={dim}, end={end}, theta={theta},"
            f" use_scaled={use_scaled}, old_context_len={old_context_len})"
        )

    # Step 1) half_dim = dim//2
    half_dim = dim // 2  # e.g. if dim=64, half_dim=32
    # Create the frequencies for the half-dim
    # for i in [0, 2, 4, ..., 2*(half_dim-1)] => we get exactly half_dim steps
    freq_indices = torch.arange(0, dim, 2, dtype=torch.float32)[:half_dim]
    # e.g. freq_indices.shape => (32,) if dim=64

    # Convert to actual freq
    freqs = 1.0 / (theta ** (freq_indices / dim))  # shape => (half_dim,)

    # If scaling:
    if use_scaled:
        freqs = apply_scaling(freqs, old_context_len=old_context_len)

    # Step 2) Outer product with positions
    t = torch.arange(end, dtype=torch.float32)  # shape => (end,)
    # => shape => (end, half_dim)
    freqs = torch.outer(t, freqs)

    # Step 3) Convert to complex: each entry is cos(...) + i*sin(...)
    # => shape => (end, half_dim) in complex
    freqs_cis = torch.polar(
        torch.ones_like(freqs, dtype=freqs.dtype),
        freqs
    )  # complex64 => shape => (end, half_dim)

    # We keep it in complex form with shape (end, half_dim).
    # In apply_rotary_emb, we interpret that as effectively (end, dim).
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    x shape: (batch, heads, seq_len, dim).
    freqs_cis shape: (seq_len, dim//2) in complex dtype,
      effectively 'dim' if you consider real+imag.

    We check:
      freqs_cis.shape[0] == x.shape[2]  (both are seq_len)
      freqs_cis.shape[1] == (x.shape[3] // 2) in complex sense.

    But for simpler coding, we can do an assertion that x.shape[3] == some multiple of 2,
    and freqs_cis.shape[1] matches x.shape[3]//2 in complex dimension.

    Then we broadcast it to (1, 1, seq_len, dim//2).
    """
    seq_len = x.shape[2]     # e.g. 128
    total_dim = x.shape[3]   # e.g. 64

    assert freqs_cis.shape[0] == seq_len, (
        f"freqs_cis.shape[0] = {freqs_cis.shape[0]} must match x.shape[2] = {seq_len}"
    )
    assert freqs_cis.shape[1] == (total_dim // 2), (
        f"freqs_cis.shape[1] = {freqs_cis.shape[1]} must be total_dim//2 = {total_dim//2} "
        f"for x.shape[-1] = {total_dim}"
    )
    # => shape => (1, 1, seq_len, dim//2)
    return freqs_cis.unsqueeze(0).unsqueeze(0).to(x.device)

@torch._dynamo.disable  # sometimes helpful
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary embedding to query & key. xq, xk: [B, H, S, D].
    We interpret D as real+imag => half for real, half for imag.

    Steps:
      1) We view xq as shape [B,H,S,D/2,2] in real/imag pairs => a complex64 array.
      2) We multiply by freqs_cis broadcasted to shape [1,1,S,D/2].
      3) We convert back to real => shape [B,H,S,D].
    """
    B, H, S, D = xq.shape
    half_d = D // 2  # e.g. 64//2=32

    # 1) Convert xq, xk to complex
    # => shape => [B,H,S,half_d,2] => we interpret last dim as real,imag
    xq_complex = torch.view_as_complex(xq.float().reshape(B, H, S, half_d, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(B, H, S, half_d, 2))

    # 2) reshape freqs_cis => shape => [1,1,S,half_d]
    fc = reshape_for_broadcast(freqs_cis, xq)
    # => shape => [1,1,S,half_d], complex => we must also do .reshape(1,1,S,half_d)
    # We'll interpret the real part & imag part inside polars
    fc_complex = fc  # It's already complex => shape [1,1,S,half_d]

    # 3) Multiply
    xq_out = xq_complex * fc_complex
    xk_out = xk_complex * fc_complex

    # 4) Convert back to real [B,H,S, D]
    xq_real = torch.view_as_real(xq_out).reshape(B, H, S, D)
    xk_real = torch.view_as_real(xk_out).reshape(B, H, S, D)

    # 5) Return same dtype as input
    return xq_real.type_as(xq), xk_real.type_as(xk)
