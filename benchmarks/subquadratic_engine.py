"""Interleaved complete backward timings and per-mode numerical errors.

Includes input validation, allocation, index construction, all singleton
credits, diagonal/joint work, V/embedding gradients and logit mapping.
Excludes compilation, dense projections, loss and optimizer. All runtime
methods use the one current engine. An optional historical JSON receipt can
supply an explicitly labeled old timing; no old implementation is loaded.
All inputs and random seeds are generated here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from conv_rosa_transformer import native_training as native


def source_hashes():
    paths=sorted((ROOT/"conv_rosa_transformer").glob("*.py"))
    paths+=sorted((ROOT/"conv_rosa_transformer/csrc").rglob("*.h"))
    paths+=sorted((ROOT/"conv_rosa_transformer/csrc").glob("*.cpp"))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def make_case(family,n,d,seed):
    rng=np.random.default_rng(seed)
    if family=="skyline":
        m=n;k=[]
        for i in range(1,m+1):k += [0]*(3*m-i)+[1]+[0]*i+[1]
        n=2*len(k)+2*m;k += [0]*(n-len(k));q=np.zeros(n,np.uint64);k=np.array(k,np.uint64)
    elif family in ("unary","shadow"):
        q=np.zeros(n,np.uint64);k=q.copy()
        if family=="shadow":k[0]=1
    elif family=="periodic":
        q=np.arange(n,dtype=np.uint64)%4;k=q.copy();k[n//2]^=np.uint64(1)
    else:
        q,k=rng.bit_generator.random_raw((2,n))
        if d<64:q&=np.uint64((1<<d)-1);k&=np.uint64((1<<d)-1)
    shifts=np.arange(d,dtype=np.uint64)
    arrays=[(2*((word[:,None]>>shifts)&1).astype(float)-1)[None,None]*.35 for word in (q,k)]
    arrays.append((2*rng.integers(0,2,(1,1,n,d)).astype(float)-1)*.35)
    arrays.extend([rng.normal(size=(1,d)),rng.normal(size=(1,d))])
    g=rng.normal(size=(1,1,n,d))/np.sqrt(d)
    return arrays,g


def errors(reference,result):
    out={}
    for name,axes in (("q",(0,)),("k",(1,)),("qk",(0,1))):
        a=np.concatenate([reference[i].ravel() for i in axes])
        b=np.concatenate([result[i].ravel() for i in axes])
        out[name+"_relative_l2"]=float(np.linalg.norm(b-a)/max(np.linalg.norm(a),1e-300))
        out[name+"_max_abs"]=float(np.max(np.abs(b-a),initial=0))
        out[name+"_cosine"]=float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-300))
    out["v_embeddings_bitwise_equal"]=all(a.tobytes()==b.tobytes() for a,b in zip(reference[2:5],result[2:5]))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--cpu",type=int)
    p.add_argument("--quick",action="store_true")
    p.add_argument("--expanded",action="store_true")
    p.add_argument("--output",type=Path,default=ROOT/"benchmarks/results/subquadratic_engine.json")
    p.add_argument("--historical-baseline",type=Path)
    a=p.parse_args()
    if a.repeats<3:p.error("use at least three interleaved repetitions")
    historical={};historical_receipt=None
    if a.historical_baseline is not None:
        if a.output.resolve()==a.historical_baseline.resolve():p.error("historical receipt is read-only")
        raw=a.historical_baseline.read_bytes();receipt=json.loads(raw)
        historical_receipt=dict(path=str(a.historical_baseline),sha256=hashlib.sha256(raw).hexdigest(),
                                platform=receipt.get("platform"),affinity=receipt.get("affinity"),
                                note="Historical timings are not a freshly interleaved comparison.")
        for row in receipt.get("rows",[]):
            old=row.get("median_ms",{}).get("legacy_exact")
            if old is not None:historical[(row["family"],row["T"],row["D"],row["order"])]=old
    if hasattr(os,"sched_setaffinity"):
        cpu=a.cpu if a.cpu is not None else min(os.sched_getaffinity(0))
        os.sched_setaffinity(0,{cpu})
    before=source_hashes();library=native._library();built=source_hashes()
    cases=[("random",512,1),("random",2048,8),("random",8192,16),
           ("random",4096,64),("unary",4096,8),("periodic",4096,4),
           ("shadow",4096,8),("shadow",8192,8),("skyline",16,8),("skyline",32,8)]
    if a.expanded:
        cases += [("random",n,d) for n,d in
                  ((1024,1),(2048,1),(4096,1),(1024,8),(4096,8),
                   (8192,8),(1024,16),(4096,16),(8192,64))]
        cases += [("unary",1024,8),("periodic",1024,4),("shadow",1024,8)]
    if a.quick:cases=[("random",256,4),("shadow",512,8),("skyline",6,4)]
    methods={
        "deterministic":dict(randomized=False),
        "randomized_exact":dict(randomized=True),
        "residual32":dict(randomized=True,backward_mode="approximate",approximation_rows=32),
        "channels8":dict(randomized=True,backward_mode="approximate",approximation_channels=8),
    }
    rows=[]
    for case,(family,n,d) in enumerate(cases):
      arrays,g=make_case(family,n,d,2026110600+case)
      for order in ("first","second"):
        caches={};outputs={};timings={name:[] for name in methods};quality={name:[] for name in methods}
        for name,options in methods.items():
            _,caches[name],_=native.training_forward_numpy(*arrays,credit_order=order,
                proposal_seed=719,repair_seed=817,**options)
            outputs[name]=native.training_backward_numpy(caches[name],g,trace=True)
        reference=outputs["deterministic"]
        for x,y in zip(outputs["deterministic"][:5],outputs["randomized_exact"][:5]):
            assert x.tobytes()==y.tobytes(),(family,order,"cache changed answer")
        for key in ("pair_codes","pair_costs"):
            assert outputs["deterministic"][-1][key].tobytes()==outputs["randomized_exact"][-1][key].tobytes()
        for rep in range(a.repeats):
            names=list(methods)
            if rep&1:names.reverse()
            for name in names:
                cache=caches[name]
                if cache.approximation is not None:cache.approximation.seed=91117+rep*104729
                start=time.perf_counter_ns();result=native.training_backward_numpy(cache,g,trace=True)
                timings[name].append((time.perf_counter_ns()-start)/1e6)
                quality[name].append(errors(reference,result));outputs[name]=result
        med={name:statistics.median(values) for name,values in timings.items()}
        row=dict(family=family,T=g.shape[2],D=d,order=order,median_ms=med,samples_ms=timings,
                 seed=2026110600+case,
                 input_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in (*arrays,g)],
                 errors=quality,exact_randomized_bytewise_equal=True,
                 counters={name:{k:v for k,v in r[-1].items() if isinstance(v,(str,int,float,bool))} for name,r in outputs.items()},
                 exact_cache_speedup=med["deterministic"]/med["randomized_exact"],
                 residual_speedup_vs_new_exact=med["randomized_exact"]/med["residual32"])
        key=(family,row["T"],d,order)
        if key in historical:
            row["historical_legacy_ms"]=historical[key]
            row["speedup_vs_historical_legacy"]=historical[key]/med["randomized_exact"]
        rows.append(row)
        print(json.dumps(dict(family=family,T=row["T"],D=d,order=order,ms=med,
            exact_cache_speedup=row["exact_cache_speedup"],residual_error=statistics.median(x["qk_relative_l2"] for x in quality["residual32"]))),flush=True)
    report=dict(status="PASS" if before==built==source_hashes() else "SOURCE_CHANGED",
        scope=__doc__,historical_baseline=historical_receipt,
        source_sha256=before,library_sha256=hashlib.sha256(Path(library._name).read_bytes()).hexdigest(),
        platform=platform.platform(),numpy=np.__version__,repeats=a.repeats,
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else [],rows=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+"\n")
    return 0 if report["status"]=="PASS" else 1


if __name__=="__main__":raise SystemExit(main())
