"""Ordinary PyTorch adapter to the one batched C++ QKV-ROSA implementation.

Dense projections stay on the tensor device. Packing, exact hard forward,
trust construction, block proposals/replays and all credit/logit/embedding
arithmetic use the standalone C++ API in csrc/training.h. CUDA tensors incur
CPU transfers; this is not a GPU matcher or a promise of GPU throughput.
"""
from __future__ import annotations

import math
import operator
import numpy as np
import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

if __package__:
    from .native_training import make_options, training_forward_numpy, training_backward_numpy
else:
    from native_training import make_options, training_forward_numpy, training_backward_numpy


def _native_layout(value, b, t, groups, d):
    return value.detach().reshape(b,t,groups,d).permute(0,2,1,3).to(
        device="cpu",dtype=torch.float64).contiguous().numpy()


def _tensor_layout(value, b, t, groups, d, *, dtype, device):
    return torch.from_numpy(value).permute(0,2,1,3).reshape(b,t,groups*d).to(
        device=device,dtype=dtype)


class _HardQKVRosaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,emb0,emb1,n_bits,config,work):
        if q.ndim!=3 or k.shape!=q.shape or v.shape!=q.shape:
            raise ValueError("q, k, v must share [B,T,C]")
        if not all(z.is_floating_point() for z in (q,k,v)):
            raise TypeError("q, k, v must be floating tensors")
        if any(z.device!=q.device or z.dtype!=q.dtype for z in (k,v)):
            raise ValueError("q, k, v must share device and dtype")
        b,t,c=q.shape;d=int(n_bits)
        if not 1<=d<=64 or c==0 or c%d:
            raise ValueError("n_bits must divide positive channels and lie in [1,64]")
        if emb0.shape!=(1,1,c) or emb1.shape!=(1,1,c):
            raise ValueError("embeddings must have shape [1,1,C]")
        if emb0.device!=q.device or emb1.device!=q.device:
            raise ValueError("embeddings must share the logits device")
        groups=c//d
        # On/off exact acceleration never changes framework RNG consumption.
        # First-order/no-MC has no draws, including when acceleration is on.
        seeds=torch.randint(0,2**63-1,(2,),device="cpu").tolist() if (
            config["joint_pairs"]>0 or (config["repair_samples"] is not None
                                       and config["repair_samples"]<d)) else (0,0)
        options=make_options(**config,proposal_seed=seeds[0],repair_seed=seeds[1])
        arrays=[_native_layout(z,b,t,groups,d) for z in (q,k,v)]
        e0=emb0.detach().to(device="cpu",dtype=torch.float64).reshape(groups,d).contiguous().numpy()
        e1=emb1.detach().to(device="cpu",dtype=torch.float64).reshape(groups,d).contiguous().numpy()
        y,cache,stats=training_forward_numpy(*arrays,e0,e1,options=options)
        ctx.cache,ctx.work,ctx.shape=cache,work,(b,t,groups,d)
        ctx.save_for_backward(q,k,v,emb0,emb1)
        work.update(stats)
        return _tensor_layout(y,b,t,groups,d,dtype=emb1.dtype,device=q.device)

    @staticmethod
    @once_differentiable
    def backward(ctx,grad_y):
        q,k,v,emb0,emb1=ctx.saved_tensors
        b,t,groups,d=ctx.shape
        gy=_native_layout(grad_y,b,t,groups,d)
        if not np.isfinite(gy).all():
            # Let a normal GradScaler detect overflow and skip the update.
            return tuple(torch.full_like(z,float("nan")) for z in (q,k,v,emb0,emb1))+(None,)*3
        gq,gk,gv,ge0,ge1,stats=training_backward_numpy(ctx.cache,gy)
        ctx.work["backward"]=stats
        gradients=[_tensor_layout(a,b,t,groups,d,dtype=z.dtype,device=z.device)
                   for a,z in zip((gq,gk,gv),(q,k,v))]
        gradients += [torch.from_numpy(a).reshape(1,1,groups*d).to(dtype=z.dtype,device=z.device)
                      for a,z in zip((ge0,ge1),(emb0,emb1))]
        return tuple(gradients)+(None,)*3


class NBitQKVRosa(nn.Module):
    """Hard QKV-ROSA, with first/second credit and exact acceleration switches.

    ``credit_order="first"`` is the original single-bit linear finite credit.
    ``credit_order="second"`` (default) adds an output-diameter trust penalty
    and at most one independent joint-bit block per stream. The diagonal is a
    trust model, not an automatically recovered loss Hessian. The interface
    remains one model forward and one loss.backward(), with no special optimizer.

    ``randomized=True`` enables answer-exact randomized root preparation. It
    does not thin gradients or change joint proposals. Detailed legacy block
    options remain explicit for reproducible studies. ``repair_samples`` is a
    separate noisy research option and is never implied by ``randomized``.

    All arithmetic after dense projections uses the batched C++ engine.
    Parameters/state names are unchanged. Save configuration with weights.
    Gradients in second-order mode depend on incoming g and are generally not
    additive across separately backpropagated losses; positive scaling works.
    The hard-credit path explicitly does not support double autograd.
    """
    def __init__(self,channels:int,n_bits:int,init_emb1:float=1.0,bias:bool=True,
                 tau:float=1.0,credit_to_logit:str="bernoulli",*,
                 credit_order:str="second",randomized:bool=False,
                 joint_pairs:int|None=None,joint_rows:int=32,
                 trust_diagonal:float|None=None,joint_row_selection:str="active",
                 repair_samples:int|None=None,stochastic_values:bool=False,
                 training_rule:str="hard_credit"):
        super().__init__()
        channels,n_bits=operator.index(channels),operator.index(n_bits)
        if channels<=0 or not 1<=n_bits<=64 or channels%n_bits:
            raise ValueError("n_bits must lie in [1,64] and divide positive channels")
        if not isinstance(stochastic_values,bool):
            raise TypeError("stochastic_values must be bool")
        if stochastic_values or training_rule!="hard_credit":
            raise ValueError("experimental stochastic V/output mixtures are not production training rules")
        options=make_options(credit_order=credit_order,randomized=randomized,tau=tau,
            credit_to_logit=credit_to_logit,joint_pairs=joint_pairs,joint_rows=joint_rows,
            trust_diagonal=trust_diagonal,joint_row_selection=joint_row_selection,
            repair_samples=repair_samples)
        if options.repair_budget>n_bits:
            raise ValueError("repair_samples must not exceed n_bits")
        self.channels,self.n_bits=channels,n_bits;self.groups=channels//n_bits
        self.credit_order,self.randomized=credit_order,bool(randomized)
        self.tau,self.credit_to_logit=float(options.tau),credit_to_logit
        self.joint_pairs,self.joint_rows=int(options.joint_pairs),int(options.joint_rows)
        self.trust_diagonal=float(options.trust_diagonal)
        self.joint_row_selection=joint_row_selection
        self.repair_samples=repair_samples
        self.stochastic_values=False;self.training_rule="hard_credit"
        self.last_forward_work=None
        self.q_proj=nn.Linear(channels,channels,bias=bias)
        self.k_proj=nn.Linear(channels,channels,bias=bias)
        self.v_proj=nn.Linear(channels,channels,bias=bias)
        self.o_proj=nn.Linear(channels,channels,bias=bias)
        self.emb0=nn.Parameter(torch.zeros(1,1,channels))
        self.emb1=nn.Parameter(torch.full((1,1,channels),float(init_emb1)))

    def _hard_rosa_from_qkv(self,q,k,v):
        config=dict(credit_order=self.credit_order if self.training else "first",
            randomized=self.randomized,tau=self.tau,credit_to_logit=self.credit_to_logit,
            joint_pairs=self.joint_pairs if self.training else 0,joint_rows=self.joint_rows,
            trust_diagonal=self.trust_diagonal if self.training else 0.,
            joint_row_selection=self.joint_row_selection,
            repair_samples=self.repair_samples if self.training else None)
        self.last_forward_work={}
        return _HardQKVRosaFn.apply(q,k,v,self.emb0,self.emb1,self.n_bits,config,self.last_forward_work)

    def forward(self,x):
        if x.ndim!=3 or x.size(-1)!=self.channels:
            raise ValueError("x must have shape [B,T,C]")
        return self.o_proj(self._hard_rosa_from_qkv(self.q_proj(x),self.k_proj(x),self.v_proj(x)))


NBitQKVRosaCpp=NBitQKVRosa
NBitQKVRosaForward=NBitQKVRosa
NBitQKVRosaForwardCpp=NBitQKVRosa
__all__=["NBitQKVRosa","NBitQKVRosaCpp","NBitQKVRosaForward","NBitQKVRosaForwardCpp"]
