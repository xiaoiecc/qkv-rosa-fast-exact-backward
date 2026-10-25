"""Oracle importance ceiling for complete-lane repair sampling.

This is a diagnostic, not a production proposal: it first pays for all exact
credits and then asks how much the best inclusion probabilities could help.
The frozen previous delivery supplies the numerical credits. No training
benefit or runtime speedup is inferred from these analytical errors.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def make_input(family,n,d,seed):
    rng=np.random.default_rng(seed)
    if family=='random':
        q=rng.integers(0,1<<d,n,dtype=np.uint64)
        k=rng.integers(0,1<<d,n,dtype=np.uint64)
    elif family=='narrow':
        q,k=rng.integers(0,256,(2,n),dtype=np.uint64)
    else:
        q,k=rng.bit_generator.random_raw((2,n))
        rows=np.arange(1,n,4);bits=rng.integers(0,64,len(rows),dtype=np.uint64)
        q[rows]=k[rows-1]^(np.uint64(1)<<bits)
    v=rng.integers(0,2,(n,d),dtype=np.uint8)
    g=rng.normal(size=(n,d));h=rng.uniform(.1,1.5,(n,d))
    e0,e1=rng.normal(size=(2,d))
    return q,k,v,g,h,e0,e1


def optimal_probabilities(energy, budget):
    energy=np.asarray(energy,dtype=np.float64)
    active=energy>0
    if active.sum()<=budget:return active.astype(np.float64)
    scale=np.sqrt(energy)
    lo=0.;hi=float(scale.sum()/budget)
    for _ in range(90):
        mid=(lo+hi)/2
        count=np.minimum(1.,scale/max(mid,np.finfo(float).tiny)).sum()
        if count>budget:lo=mid
        else:hi=mid
    return np.minimum(1.,scale/hi)


def mse(energy, p):
    return float(np.sum(energy[p>0]*(1/p[p>0]-1)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path,default=ROOT.parent/'qkv-rosa-train')
    parser.add_argument('--cpu',type=int,default=2)
    parser.add_argument('--output',type=Path,default=ROOT/'research/results/native_random_noise_ceiling.json')
    a=parser.parse_args()
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{a.cpu})
    source=a.baseline_root/'conv_rosa_transformer/rosa_core.py'
    spec=importlib.util.spec_from_file_location('frozen_credit_core',source)
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    rows=[]
    cases=[('random',1024,8),('random',8192,8),('random',8192,16),
           ('narrow',8192,64),('wide',8192,64)]
    for index,(family,n,d) in enumerate(cases):
        seed=391104+index
        q,k,v,g,h,e0,e1=make_input(family,n,d,seed)
        args=q,k,v,g,e0,e1
        full=core.credits_numpy(*args)
        selected=core.selected_credits_numpy(*args,[])
        C=np.stack(full[:2])
        B=np.stack([np.broadcast_to(selected[6+s][:,None],C[s].shape) for s in range(2)])
        R=C-B
        energy=np.sum(R*R,axis=(0,1))
        support=np.flatnonzero(energy>0)
        budget=max(1,len(support)//2)
        norm_sq=float(np.sum(C*C))
        p_uniform=np.zeros(d);p_uniform[support]=min(1.,budget/len(support)) if len(support) else 1.
        p_oracle=optimal_probabilities(energy,budget)
        uniform_mse=mse(energy,p_uniform);oracle_mse=mse(energy,p_oracle)
        # A stronger, also unavailable-for-free common baseline is the exact
        # mean of lane credits per owner. Include only lanes with residuals,
        # since certified zero-residual lanes should remain exact in all rules.
        mean=np.mean(C[...,support],axis=-1,keepdims=True)
        centered=C[...,support]-mean
        centered_energy=np.sum(centered*centered,axis=(0,1))
        p_center=optimal_probabilities(centered_energy,budget)
        center_mse=mse(centered_energy,p_center)
        required={}
        for threshold in (.10,.25):
            required[str(threshold)]=next((r for r in range(1,len(support)+1)
                if mse(energy,optimal_probabilities(energy,r))<=threshold**2*norm_sq),0)
        row=dict(family=family,T=n,D=d,seed=seed,supported_nonzero_residual_lanes=len(support),
            budget=budget,full_qk_norm_sq=norm_sq,
            uniform_relative_rms=math.sqrt(uniform_mse/norm_sq) if norm_sq else 0.,
            oracle_importance_relative_rms=math.sqrt(oracle_mse/norm_sq) if norm_sq else 0.,
            oracle_importance_variance_ratio=oracle_mse/uniform_mse if uniform_mse else 0.,
            oracle_center_and_importance_relative_rms=math.sqrt(center_mse/norm_sq) if norm_sq else 0.,
            minimum_integer_budget_for_oracle_relative_rms=required,
            oracle_probabilities=p_oracle.tolist(),residual_lane_energy=energy.tolist())
        rows.append(row)
        print(json.dumps(row),flush=True)
    report=dict(status='COMPLETE',scope=__doc__,
        frozen_baseline_revision='f6d7c4b73d7040b1b33860f5a2649c3cd3036cc5',
        baseline_core_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        loaded_native_sha256=hashlib.sha256(Path(core._library()._name).read_bytes()).hexdigest(),
        probability_solution='minimize sum_j E_j(1/p_j-1), sum p_j <= budget; p_j=min(1,sqrt(E_j)/lambda)',
        centered_control='Unimplementable-for-free oracle: exact mean of all supported lane credits at each owner',
        rows=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
