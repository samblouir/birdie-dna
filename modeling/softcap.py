# modeling/softcap.py
"""
=============================================================================
TANH SOFTCAPPING LOGIT MOD
--------------------------
Copied from Gemma 2's repo.

Implements a 'tanh softcap' function to limit extreme attention scores,
inspired by Gemma2 and other experiments.

Used in your flex_attention if desired. By default, we simply provide
utility to generate a function that clamps scores via tanh.
=============================================================================
"""

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _score_mod_signature
from torch._inductor.lowering import make_pointwise, register_lowering
from torch._inductor.virtualized import ops
from functools import partial

@torch.library.custom_op("approx::tanh", mutates_args=())
def _tanh_approx(inp: Tensor) -> Tensor:
    """
    Custom approximate tanh op. 
    Actual usage might rely on GPU-specific instructions, 
    or we can fallback to normal torch.tanh.
    """
    return torch.tanh(inp)

@_tanh_approx.register_fake
def _(inp: torch.Tensor) -> torch.Tensor:
    # Fallback for "fake" mode
    return torch.tanh(inp)

def _tanh_approx_lowering(inp):
    # Basic partial that returns normal tanh, but you could do an inline_asm
    fn = partial(ops.inline_asm_elementwise, asm="tanh.approx.f32 $0, $1;")
    return make_pointwise(fn)(inp)

register_lowering(torch.ops.approx.tanh)(_tanh_approx_lowering)

class _TanhApprox(torch.autograd.Function):
    """
    Demonstration of a custom approximate tanh function
    for potential hardware acceleration or specialized instructions.
    """
    @staticmethod
    def forward(ctx, x):
        return torch.ops.approx.tanh(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs
        result = output
        ctx.save_for_backward(result)

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_tensors
        return grad_output * (1 - result * result)

    @staticmethod
    def vmap(info, in_dims, x):
        # fallback to normal tanh in vmap scenario
        return torch.tanh(x), 0

_tanh_approx = _TanhApprox.apply

def generate_tanh_softcap(soft_cap: int, approx: bool = False) -> _score_mod_signature:
    """
    Factory function that returns a "score_mod" function 
    to be used in attention, bounding logits by soft_cap via tanh.

    Args:
        soft_cap (int): The maximum absolute value we want to allow.
        approx (bool): If True, use the approximate tanh kernel.

    Returns:
        A function "score_mod" that you can pass to your attention code.
    """
    tanh_fn = _tanh_approx if approx else torch.tanh

    def tanh_softcap_score_mod(score, b, h, q_idx, kv_idx):
        """
        Score mod for limiting attention logits. 
        score is the raw attention logit, 
        we scale by 1/soft_cap, apply tanh, then multiply back.
        """
        return soft_cap * tanh_fn(score / soft_cap)

    prefix = "tanh_softcap_approx" if approx else "tanh_softcap"
    tanh_softcap_score_mod.__name__ = f"{prefix}_{soft_cap}"
    return tanh_softcap_score_mod