"""Small ordinary models for the exact/approximate native-credit study.

Data generators use private Torch generators. The same input, initial state,
optimizer and framework RNG stream can be shared by every method. No production
implementation is selected here; callers inject the layer class and settings.
"""
from __future__ import annotations

import torch
from torch import nn

from native_random_toy import Retrieval, ClassificationRetrieval, make_data as short_data


class DenseRetrieval(nn.Module):
    """A bank of key/value pairs followed by many query rows.

    K initially maps query rows to a reserved word, so they cannot overwrite
    bank entries. Q and K are both learned, together with V/O and a task head.
    Every query contributes to the loss; many output adjoints are nonzero.
    """
    def __init__(self,layer_class,config,keys=4,payload=8,channels=16,classification=False):
        super().__init__();self.keys=keys;self.payload=payload;self.bank=2*keys
        self.channels=channels;self.rosa=layer_class(channels,payload,bias=False,**config)
        self.head=nn.Linear(channels,payload,bias=False)
        # Distinct balanced 8-bit words, deliberately perturbed independently
        # in Q and K to leave a genuine routing-learning problem.
        code=torch.tensor([[1,1,1,1,-1,-1,-1,-1],
                           [1,1,-1,-1,1,1,-1,-1],
                           [1,-1,1,-1,1,-1,1,-1],
                           [1,-1,-1,1,-1,1,1,-1]],dtype=torch.float32).T
        groups=channels//payload
        with torch.no_grad():
            for projection in (self.rosa.q_proj,self.rosa.k_proj):
                projection.weight.zero_()
                projection.weight[:,:keys]=.30*code.repeat(channels//8,1)+( .03 if payload>=32 else .25)*torch.randn(channels,keys)
            if payload>=32:
                # A bounded initial routing error, rather than an exponentially
                # unlikely random agreement between unrelated wide hard words.
                for key in range(keys):
                    self.rosa.q_proj.weight[(3*key+1)%payload,key]*=-1
                    self.rosa.q_proj.weight[(3*key+2)%payload,key]*=-1
            self.rosa.k_proj.weight[:,keys+payload]=3.
            self.rosa.v_proj.weight.zero_()
            self.rosa.v_proj.weight[:,keys:keys+payload]=.45*torch.eye(payload).repeat(groups,1)
            self.rosa.o_proj.weight.copy_(torch.eye(channels))
            self.head.weight.copy_(torch.cat([torch.eye(payload)]*groups,dim=1)/groups)
    def forward(self,x):return self.head(self.rosa(x)[:,self.bank:])


class PeriodicCopy(nn.Module):
    """Dense-output copy task whose initial Q/K route has long periodic matches.

    All projections are trainable. Initial Q/K codes follow a period-four
    key sequence; V starts imperfect. Hard teacher semantics give output at
    t>=4 from payload t-3. The first four outputs are the zero empty value.
    """
    def __init__(self,layer_class,config):
        super().__init__();self.rosa=layer_class(16,8,bias=False,**config)
        self.head=nn.Linear(16,8,bias=False)
        code=torch.tensor([[1,1,1,1,-1,-1,-1,-1],
                           [1,1,-1,-1,1,1,-1,-1],
                           [1,-1,1,-1,1,-1,1,-1],
                           [1,-1,-1,1,-1,1,1,-1]],dtype=torch.float32).T
        with torch.no_grad():
            for projection in (self.rosa.q_proj,self.rosa.k_proj):
                projection.weight.zero_()
                projection.weight[:,:4]=.35*code.repeat(2,1)+.08*torch.randn(16,4)
            self.rosa.v_proj.weight.zero_()
            self.rosa.v_proj.weight[:,4:12]=.20*torch.eye(8).repeat(2,1)+.16*torch.randn(16,8)
            self.rosa.o_proj.weight.copy_(torch.eye(16))
            self.head.weight.copy_(torch.cat([torch.eye(8)]*2,dim=1)/2)
    def forward(self,x):return self.head(self.rosa(x))


class ResidualInputStem(nn.Module):
    """Small trainable nonlinear component before ROSA for an auxiliary check."""
    def __init__(self,channels=8):
        super().__init__();self.inner=nn.Linear(channels,2*channels,bias=False)
        self.outer=nn.Linear(2*channels,channels,bias=False)
        with torch.no_grad():
            self.inner.weight.copy_(.15*torch.randn_like(self.inner.weight))
            self.outer.weight.copy_(.02*torch.randn_like(self.outer.weight))
    def forward(self,x):return x+self.outer(torch.tanh(self.inner(x)))


class ContextRecall(nn.Module):
    """Dense contextual recall over all four two-bit key words.

    Random key sequences have repeated short contexts and many one-bit neighbor
    repairs. The two-bit width deliberately prevents the no-near-pair
    certificate from making almost every output row exact for free.
    """
    def __init__(self,layer_class,config,stem=False):
        super().__init__();self.rosa=layer_class(8,2,bias=False,**config)
        self.head=nn.Linear(8,2,bias=False)
        code=torch.tensor([[-1,1,-1,1],[-1,-1,1,1]],dtype=torch.float32)
        with torch.no_grad():
            for projection in (self.rosa.q_proj,self.rosa.k_proj):
                projection.weight.zero_()
                projection.weight[:,:4]=.35*code.repeat(4,1)+.08*torch.randn(8,4)
            # Every group has one deliberate routing defect. Different groups
            # fail on different symbols, so the initial task is not merely
            # fitting an output scale with an already-correct router.
            self.rosa.q_proj.weight[0,1]*=-1
            self.rosa.q_proj.weight[3,2]*=-1
            self.rosa.k_proj.weight[4,0]*=-1
            self.rosa.k_proj.weight[7,3]*=-1
            self.rosa.v_proj.weight.zero_()
            self.rosa.v_proj.weight[:,4:6]=.20*torch.eye(2).repeat(4,1)+.16*torch.randn(8,2)
            self.rosa.o_proj.weight.copy_(torch.eye(8))
            self.head.weight.copy_(torch.cat([torch.eye(2)]*4,dim=1)/4)
        self.stem=ResidualInputStem() if stem else nn.Identity()
    def forward(self,x):return self.head(self.rosa(self.stem(x)))


class PrefixDefectCopy(nn.Module):
    """A structural stress model with a single mismatched key prefix.

    Constant initial Q and all but the first K create long counterfactual
    repair corridors. Dense payload reconstruction supplies real output
    adjoints. This deliberately structured input is a performance stress
    case, not a representative general-language training distribution.
    """
    def __init__(self,layer_class,config,payload=8,margin=.35):
        super().__init__();channels=2*payload;self.payload=payload
        self.rosa=layer_class(channels,payload,bias=False,**config)
        self.head=nn.Linear(channels,payload,bias=False)
        with torch.no_grad():
            for projection in (self.rosa.q_proj,self.rosa.k_proj):
                projection.weight.zero_();projection.weight[:,0]=-margin
            self.rosa.k_proj.weight[payload-1,1]=.70
            self.rosa.k_proj.weight[2*payload-1,1]=.70
            self.rosa.v_proj.weight.zero_()
            self.rosa.v_proj.weight[:,2:2+payload]=.20*torch.eye(payload).repeat(2,1)+.16*torch.randn(channels,payload)/(payload/8)**.5
            self.rosa.o_proj.weight.copy_(torch.eye(channels))
            self.head.weight.copy_(torch.cat([torch.eye(payload)]*2,dim=1)/2)
    def forward(self,x):return self.head(self.rosa(x))


def prefix_data(seed,n=1,length=2048,split=0,payload=8):
    gen=torch.Generator().manual_seed(seed+1237913+split*104729)
    x=torch.zeros(n,length,2*payload);x[:,:,0]=1;x[:,0,1]=1
    x[:,:,2:2+payload]=(torch.rand(n,length,payload,generator=gen)>.5).float()*2-1
    target=x[:,:,2:2+payload].clone();target[:,:2]=0
    return x,target


def context_data(seed,n=4,length=128,split=0):
    # The task's hard teacher uses only the exact forward matcher. It does not
    # use a backward rule, student parameters, or the approximation RNG.
    from conv_rosa_transformer.rosa_core import matching_stats
    gen=torch.Generator().manual_seed(seed+947921+split*104729)
    keys=torch.randint(4,(n,length),generator=gen)
    x=torch.zeros(n,length,8)
    x[torch.arange(n)[:,None],torch.arange(length)[None,:],keys]=1
    x[:,:,4:6]=(torch.rand(n,length,2,generator=gen)>.5).float()*2-1
    target=torch.zeros(n,length,2)
    for batch in range(n):
        _,routes=matching_stats(keys[batch].tolist(),keys[batch].tolist())
        indices=torch.tensor(routes);keep=indices>=0
        target[batch,keep]=x[batch,indices[keep]+1,4:6]
    return x,target


def periodic_data(seed,n=4,length=128,split=0):
    gen=torch.Generator().manual_seed(seed+617921+split*104729)
    x=torch.zeros(n,length,16)
    phase=torch.randint(4,(n,),generator=gen)
    x[torch.arange(n)[:,None],torch.arange(length)[None,:],(torch.arange(length)[None,:]+phase[:,None])%4]=1
    x[:,:,4:12]=(torch.rand(n,length,8,generator=gen)>.5).float()*2-1
    target=torch.zeros(n,length,8)
    if length>4:target[:,4:]=x[:,1:length-3,4:12]
    return x,target


def dense_data(seed,n=8,queries=64,split=0,keys=4,payload=8,channels=16):
    gen=torch.Generator().manual_seed(seed+527921+split*104729)
    x=torch.zeros(n,2*keys+queries,channels)
    target=torch.zeros(n,queries,payload)
    for sample in range(n):
        values=(torch.rand(keys,payload,generator=gen)>.5).float()*2-1
        order=torch.randperm(keys,generator=gen)
        for i,key in enumerate(order.tolist()):
            x[sample,2*i,key]=1
            x[sample,2*i+1,keys:keys+payload]=values[key]
        query=torch.randint(keys,(queries,),generator=gen)
        x[sample,2*keys+torch.arange(queries),query]=1
        x[sample,2*keys:,keys+payload]=1
        target[sample]=values[query]
    return x,target


def make_model(layer_class,config,task):
    if task.startswith('prefix_copy64'):return PrefixDefectCopy(layer_class,config,payload=64,margin=0. if 'zero' in task else .35)
    if task.startswith('prefix_copy'):return PrefixDefectCopy(layer_class,config,margin=0. if 'zero' in task else .35)
    if task.startswith('context_recall'):return ContextRecall(layer_class,config,stem='stem' in task)
    if task.startswith('periodic_copy'):return PeriodicCopy(layer_class,config)
    if task.startswith('dense_retrieval64'):return DenseRetrieval(layer_class,config,payload=64,channels=128)
    if task.startswith('dense_retrieval'):return DenseRetrieval(layer_class,config)
    if task.startswith('retrieval_classification'):return ClassificationRetrieval(layer_class,config)
    return Retrieval(layer_class,config)


def make_data(seed,task,n,split=0,queries=64):
    if task.startswith('prefix_copy64'):return prefix_data(seed,n,queries,split,payload=64)
    if task.startswith('prefix_copy'):return prefix_data(seed,n,queries,split)
    if task.startswith('context_recall'):return context_data(seed,n,queries,split)
    if task.startswith('periodic_copy'):return periodic_data(seed,n,queries,split)
    if task.startswith('dense_retrieval64'):return dense_data(seed,n,queries,split,payload=64,channels=128)
    if task.startswith('dense_retrieval'):return dense_data(seed,n,queries,split)
    return short_data(seed,task,n,split)


def loss_value(pred,target,task):
    if task.startswith('retrieval_classification'):
        ids=((target>0).long()*(1<<torch.arange(target.shape[-1]))).sum(-1)
        return nn.functional.cross_entropy(pred,ids)
    per_row=.5*(pred-target).square().sum(-1)
    if task.endswith('_weighted'):
        weight=torch.ones(per_row.shape[-1],dtype=per_row.dtype,device=per_row.device)
        weight[per_row.shape[-1]//2]=128.
        return (per_row*weight).sum()/(per_row.shape[0]*weight.sum())
    return per_row.mean()


def metrics(pred,target,task):
    if task.startswith('retrieval_classification'):
        ids=pred.argmax(-1);bits=((ids[...,None]>>torch.arange(target.shape[-1]))&1).bool()
    else:bits=pred>0
    correct=bits==(target>0)
    nonempty=target.abs().any(-1)
    return {'loss':float(loss_value(pred,target,task)),
            'bit_accuracy':float(correct.float().mean()),
            'payload_accuracy':float(correct.all(-1).float().mean()),
            'nonempty_bit_accuracy':float(correct[nonempty].float().mean()) if nonempty.any() else None,
            'nonempty_payload_accuracy':float(correct.all(-1)[nonempty].float().mean()) if nonempty.any() else None}
