"""Thin NumPy bindings for the standalone batched C++ training interface.

All per-stream routing, trust metric, block proposals/replays and gradient
arithmetic live in csrc/training_impl.h. Arrays use [B,G,T,D] order. This module
validates shapes/options, owns the forward cache, and translates return values.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
import operator

import numpy as np

if __package__:
    from . import rosa_core
else:
    import rosa_core


class TrainingOptions(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
        ("credit_order", ctypes.c_int32), ("randomized", ctypes.c_int32),
        ("logit_mode", ctypes.c_int32), ("joint_pairs", ctypes.c_int32),
        ("joint_rows", ctypes.c_int32), ("active_rows", ctypes.c_int32),
        ("repair_budget", ctypes.c_int32), ("reserved", ctypes.c_int32),
        ("tau", ctypes.c_double), ("trust_diagonal", ctypes.c_double),
        ("proposal_seed", ctypes.c_uint64), ("repair_seed", ctypes.c_uint64),
    ]


TRAINING_COUNTER_NAMES = rosa_core.COUNTER_NAMES + (
    "selected_pairs", "joint_matching_calls", "active_proposal_rows",
    "candidate_events", "supported_repair_lanes", "selected_repair_lanes",
    "exact_repair_streams", "gradient_elements", "randomized_root_streams",
    "proposal_rows", "joint_dot_components", "credit_engine_calls",
)


def make_options(*, credit_order="second", randomized=False, tau=1.,
                 credit_to_logit="bernoulli", joint_pairs=None, joint_rows=32,
                 trust_diagonal=None, joint_row_selection="active",
                 repair_samples=None, proposal_seed=0, repair_seed=0):
    if credit_order not in ("first", "second"):
        raise ValueError("credit_order must be first or second")
    if not isinstance(randomized, (bool, np.bool_)):
        raise TypeError("randomized must be bool")
    if credit_to_logit not in ("bernoulli", "identity"):
        raise ValueError("credit_to_logit must be bernoulli or identity")
    if joint_row_selection not in ("input", "active"):
        raise ValueError("joint_row_selection must be input or active")
    tau=float(tau)
    trust_diagonal=float((.5 if credit_order=="second" else 0.)
                         if trust_diagonal is None else trust_diagonal)
    joint_pairs=operator.index((1 if credit_order=="second" else 0)
                               if joint_pairs is None else joint_pairs)
    joint_rows=operator.index(joint_rows)
    repair_budget=0 if repair_samples is None else operator.index(repair_samples)
    if not math.isfinite(tau) or tau<=0:
        raise ValueError("tau must be finite and positive")
    if not math.isfinite(trust_diagonal) or trust_diagonal<0:
        raise ValueError("trust_diagonal must be finite and nonnegative")
    if not 0<=joint_pairs<=(2**31-1)//8 or not 0<joint_rows<2**31:
        raise ValueError("joint budgets are outside the native int32 domain")
    if repair_samples is not None and not 1<=repair_budget<=64:
        raise ValueError("repair_samples must be positive and at most 64")
    if credit_order=="first" and (joint_pairs or trust_diagonal):
        raise ValueError("first order requires joint_pairs=0 and trust_diagonal=0")
    proposal_seed,repair_seed=operator.index(proposal_seed),operator.index(repair_seed)
    if not 0<=proposal_seed<2**64 or not 0<=repair_seed<2**64:
        raise ValueError("native seeds must fit uint64")
    return TrainingOptions(1, ctypes.sizeof(TrainingOptions),
        1 if credit_order=="first" else 2, int(randomized),
        int(credit_to_logit=="identity"), joint_pairs, joint_rows,
        int(joint_row_selection=="active"), repair_budget, 0,
        tau, trust_diagonal, proposal_seed, repair_seed)


def _library():
    library=rosa_core._library()
    if not getattr(library, "_training_api_bound", False):
        library.rosa_training_defaults.argtypes=[ctypes.POINTER(TrainingOptions)]
        library.rosa_training_defaults.restype=None
        prefix=[ctypes.POINTER(TrainingOptions)]+[ctypes.c_int]*4
        library.rosa_training_forward.argtypes=prefix+[ctypes.c_void_p]*12
        library.rosa_training_forward.restype=ctypes.c_int
        library.rosa_training_backward.argtypes=prefix+[ctypes.c_void_p]*17
        library.rosa_training_backward.restype=ctypes.c_int
        library._training_api_bound=True
    return library


def _pointers(values):
    return [ctypes.c_void_p() if value is None else value.ctypes.data_as(ctypes.c_void_p)
            for value in values]


@dataclass
class TrainingCache:
    """Forward-owned native arrays; do not mutate between forward/backward."""
    q_logits: np.ndarray
    k_logits: np.ndarray
    v_logits: np.ndarray
    emb0: np.ndarray
    emb1: np.ndarray
    q: np.ndarray
    k: np.ndarray
    v_bits: np.ndarray
    lengths: np.ndarray
    routes: np.ndarray
    options: TrainingOptions


def training_forward_numpy(q_logits, k_logits, v_logits, emb0, emb1,
                           *, options=None, **kwargs):
    """One batched C++ hard forward; return output, owned cache and counters.

    Logits have [B,G,T,D], embeddings [G,D]. No PyTorch dependency. The cache
    snapshots logits, embeddings and options independently of caller storage.
    Pass it unchanged to training_backward_numpy for the same model evaluation.
    The C ABI in csrc/training.h exposes these operations directly.
    """
    if options is not None and kwargs:
        raise ValueError("pass options or keyword settings, not both")
    options=make_options(**kwargs) if options is None else options
    if not isinstance(options,TrainingOptions):
        raise TypeError("options must be TrainingOptions")
    options=TrainingOptions.from_buffer_copy(options)
    qz,kz,vz=[rosa_core._array(x,np.float64) for x in (q_logits,k_logits,v_logits)]
    if qz.ndim!=4 or kz.shape!=qz.shape or vz.shape!=qz.shape:
        raise ValueError("logits require a common [B,G,T,D] shape")
    b,groups,n,d=qz.shape
    if groups<=0 or not 1<=d<=64 or n>=rosa_core._MAX_T or n*d>rosa_core._MAX_ELEMENTS:
        raise ValueError("invalid native training shape")
    if b*groups>rosa_core._MAX_ELEMENTS or options.repair_budget>d:
        raise ValueError("batch/group count or repair budget is outside the domain")
    e0,e1=[rosa_core._array(x,np.float64) for x in (emb0,emb1)]
    if e0.shape!=(groups,d) or e1.shape!=(groups,d):
        raise ValueError("embeddings require [G,D]")
    # _array may borrow contiguous float64 caller storage. Snapshot every
    # retained floating input before forward so backward sees that same state.
    qz,kz,vz,e0,e1=[a.copy() for a in (qz,kz,vz,e0,e1)]
    q,k=[np.zeros((b,groups,n),dtype=np.uint64) for _ in range(2)]
    bits=np.zeros(qz.shape,dtype=np.uint8)
    lengths=np.zeros(q.shape,dtype=np.int32);routes=np.full(q.shape,-1,dtype=np.int32)
    y=np.empty(qz.shape,dtype=np.float64);counts=np.zeros((b*groups,4),dtype=np.uint64)
    arrays=(qz,kz,vz,e0,e1,q,k,bits,lengths,routes,y,counts)
    code=_library().rosa_training_forward(ctypes.byref(options),b,groups,n,d,*_pointers(arrays))
    if code:
        raise ValueError("native training forward requires finite inputs") if code==4 else RuntimeError(f"native training forward failed: {code}")
    cache=TrainingCache(qz,kz,vz,e0,e1,q,k,bits,lengths,routes,options)
    totals=counts.sum(axis=0,dtype=np.uint64)
    stats=dict(backend="native_batched",baseline_matching_calls=int(totals[0]),
        transition_probes=int(totals[1]),hash_probe_steps=int(totals[2]),
        endpoint_queries=int(totals[3]),streams=b*groups)
    return y,cache,stats


def training_backward_numpy(cache, grad_y, *, trace=False):
    """One batched C++ backward, returning Q/K/V logits and embedding grads.

    The selected rule and random seeds are frozen in the forward cache.
    Exact acceleration never changes the credit vector or proposal seeds.
    Optional trace contains the selected pair codes and four frozen costs.
    """
    if not isinstance(cache,TrainingCache):
        raise TypeError("cache must come from training_forward_numpy")
    g=rosa_core._array(grad_y,np.float64)
    if g.shape!=cache.q_logits.shape:
        raise ValueError("grad_y must match cached [B,G,T,D]")
    b,groups,n,d=g.shape;options=cache.options
    gq,gk,gv=[np.empty(g.shape,dtype=np.float64) for _ in range(3)]
    ge0,ge1=[np.zeros((groups,d),dtype=np.float64) for _ in range(2)]
    counts=np.zeros((b*groups,len(TRAINING_COUNTER_NAMES)),dtype=np.uint64)
    pairs=np.full((b*groups,options.joint_pairs,2),2**64-1,dtype=np.uint64) if trace else None
    costs=np.full((b*groups,options.joint_pairs,4),np.nan,dtype=np.float64) if trace else None
    arrays=(cache.q,cache.k,cache.v_bits,cache.q_logits,cache.k_logits,cache.v_logits,
            g,cache.emb0,cache.emb1,gq,gk,gv,ge0,ge1,counts,pairs,costs)
    code=_library().rosa_training_backward(ctypes.byref(options),b,groups,n,d,*_pointers(arrays))
    if code:
        raise ValueError("native training backward requires finite inputs") if code==4 else RuntimeError(f"native training backward failed: {code}")
    totals=counts.sum(axis=0,dtype=np.uint64)
    stats=dict(zip(TRAINING_COUNTER_NAMES,map(int,totals)))
    stats.update(backend="native_batched",streams=b*groups,
                 repair_sampling_exact=stats["exact_repair_streams"]==b*groups)
    if trace:stats.update(pair_codes=pairs,pair_costs=costs,stream_counters=counts)
    return gq,gk,gv,ge0,ge1,stats


__all__=["TrainingOptions","TrainingCache","make_options",
         "training_forward_numpy","training_backward_numpy","TRAINING_COUNTER_NAMES"]
