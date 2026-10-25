"""Small composed models shared by native/frozen training comparisons.

The layer class is injected by the caller. There is no method registry or
alternate ROSA implementation in this file.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn


class Regression(nn.Module):
    def __init__(self,layer_class,config):
        super().__init__();self.rosa=layer_class(8,4,**config)
        with torch.no_grad():
            for p in (self.rosa.q_proj,self.rosa.k_proj):p.weight.zero_();p.bias.zero_()
        for p in (*self.rosa.q_proj.parameters(),*self.rosa.k_proj.parameters()):p.requires_grad_(False)
        self.head=nn.Sequential(nn.Linear(8,12),nn.Tanh(),nn.Linear(12,4))
    def forward(self,x):return self.head(self.rosa(x)[:,-1])


class Retrieval(nn.Module):
    def __init__(self,layer_class,config):
        super().__init__();self.rosa=layer_class(8,4,bias=False,**config)
        self.head=nn.Linear(8,4,bias=False)
        codes=torch.tensor([[1,1,-1,-1],[1,-1,1,-1],[-1,1,1,-1],[-1,-1,-1,1]],dtype=torch.float64).T
        with torch.no_grad():
            for p in (self.rosa.q_proj,self.rosa.k_proj):
                p.weight.zero_()
                p.weight[:,:4]=(.30*codes.repeat(2,1)+.35*torch.randn(8,4)).to(p.weight.dtype)
            self.rosa.v_proj.weight.zero_()
            self.rosa.v_proj.weight[:,4:]=torch.eye(4).repeat(2,1)*.45
            self.rosa.o_proj.weight.copy_(torch.eye(8))
            self.head.weight.copy_(torch.cat((torch.eye(4),torch.eye(4)),dim=1)/2)
    def forward(self,x):return self.head(self.rosa(x)[:,-1])


class ClassificationRetrieval(Retrieval):
    def __init__(self,layer_class,config):
        super().__init__(layer_class,config)
        self.head=nn.Linear(8,16,bias=False)
        codes=torch.tensor([[2*((word>>j)&1)-1 for j in range(4)] for word in range(16)],dtype=torch.float32)
        with torch.no_grad():self.head.weight.copy_(torch.cat((codes,codes),dim=1)/2)


def loss_value(pred,target,classification):
    if classification:
        ids=((target>0).long()*(1<<torch.arange(4))).sum(-1)
        return nn.functional.cross_entropy(pred,ids)
    return .5*(pred-target).square().sum(-1).mean()


def metrics(pred,target,classification):
    if classification:
        ids=pred.argmax(-1);bits=((ids[:,None]>>torch.arange(4))&1).bool()
    else:bits=pred>0
    correct=bits==(target>0)
    return dict(loss=float(loss_value(pred,target,classification)),
        bit_accuracy=float(correct.float().mean()),payload_accuracy=float(correct.all(-1).float().mean()))


def make_data(seed,task,n,split=0):
    gen=torch.Generator().manual_seed(seed+771003+split*104729)
    if task=='nonlinear_regression':
        features=torch.randn(n,8,generator=gen)
        teacher_gen=torch.Generator().manual_seed(seed+991013)
        teacher=torch.randn(8,4,generator=teacher_gen)/np.sqrt(8)
        x=torch.zeros(n,2,8);x[:,1]=features;y=.55*torch.tanh(features@teacher)
    else:
        x=torch.zeros(n,9,8);y=torch.zeros(n,4)
        for row in range(n):
            order=torch.randperm(4,generator=gen)
            values=(torch.rand(4,4,generator=gen)>.5).float()*2-1
            key=int(torch.randint(4,(),generator=gen))
            for p in range(4):x[row,2*p,order[p]]=1;x[row,2*p+1,4:]=values[p]
            x[row,8,order[key]]=1;y[row]=values[key]
    return x,y
