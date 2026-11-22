"""Fast exact counterfactual backpropagation for hard n-bit QKV-ROSA.

Quick start::

    import torch
    from qkv_rosa import NBitQKVRosa

    layer = NBitQKVRosa(channels=128, n_bits=8)
    x = torch.randn(2, 32, 128)
    loss = layer(x).square().mean()
    loss.backward()
"""
from __future__ import annotations

from .core import (
    COUNTER_NAMES,
    build,
    credits_numpy,
    exact_stream_bit_credits,
    matching_stats,
    pack_bits,
)
from .layer import NBitQKVRosa
from .native import (
    APPROXIMATION_COUNTER_NAMES,
    TRAINING_COUNTER_NAMES,
    ApproximationOptions,
    TrainingCache,
    TrainingOptions,
    make_approximation_options,
    make_options,
    training_backward_numpy,
    training_forward_numpy,
)
from .reference import (
    backward_bruteforce,
    forward_naive,
    matching_stats_naive,
)
from .transformer import ConvRosaBlock, RosaTransformer

__all__ = [
    "NBitQKVRosa",
    "ConvRosaBlock",
    "RosaTransformer",
    "build",
    "pack_bits",
    "matching_stats",
    "credits_numpy",
    "exact_stream_bit_credits",
    "COUNTER_NAMES",
    "TrainingOptions",
    "ApproximationOptions",
    "TrainingCache",
    "make_options",
    "make_approximation_options",
    "training_forward_numpy",
    "training_backward_numpy",
    "TRAINING_COUNTER_NAMES",
    "APPROXIMATION_COUNTER_NAMES",
    "matching_stats_naive",
    "forward_naive",
    "backward_bruteforce",
]
