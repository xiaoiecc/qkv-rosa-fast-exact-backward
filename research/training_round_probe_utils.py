"""Pure NumPy helpers for the final public-API gradient diagnostics.

Function bodies are copied unchanged from the measured research probes.
Only import placement changed when preparing the reproducibility package.
"""
import numpy as np


def adjoint(arrays,profile):
    g=arrays['grad_y'].copy();d=g.shape[-1]
    if profile in ('single_dominant','nine_dominant','nine_tangent'):
        scale=np.full(d,1e-6);scale[:1 if profile=='single_dominant' else 9]=1;g*=scale
    if profile in ('tangent','nine_tangent'):
        y=np.empty_like(g)
        for b in range(g.shape[0]):
          for group in range(g.shape[1]):
            routes=arrays['routes'][b,group];valid=routes>=0;y[b,group]=arrays['emb0'][group]
            y[b,group,valid]=(2*arrays['v_bits'][b,group,routes[valid]+1].astype(np.float64)-1)*arrays['emb1'][group]
        direction=y.copy()
        if profile=='nine_tangent':direction[...,9:]=0
        norm=(direction*direction).sum(-1);dot=(g*y).sum(-1)
        g-=np.divide(dot,norm,out=np.zeros_like(dot),where=norm!=0)[...,None]*direction
    return np.ascontiguousarray(g)


def summary(errors):
    result={}
    for space in errors[0]:
      result[space]={}
      for metric in errors[0][space]:
        values=[e[space][metric] for e in errors if e[space][metric] is not None]
        if not values:continue
        a=np.asarray(values,dtype=np.float64)
        result[space][metric]={'mean':float(a.mean()),'median':float(np.median(a)),
            'p05':float(np.quantile(a,.05)),'p95':float(np.quantile(a,.95)),'minimum':float(a.min()),'maximum':float(a.max())}
      values=[e[space]['relative_descent'] for e in errors if e[space]['relative_descent'] is not None]
      result[space]['negative_descent_fraction']=float(np.mean(np.asarray(values)<0)) if values else None
    return result
