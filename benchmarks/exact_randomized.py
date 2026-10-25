"""Exact random root join and sparse root storage against the frozen last delivery.

Core timings reuse native buffers and include all per-call indexing/contraction.
SAM preparation is also measured separately. Neither includes projections,
training, Python input validation, compilation, or warmup. All modes return the
same full credits, with no repair-lane sampling.
"""
from pathlib import Path
import argparse,ctypes,hashlib,importlib.util,json,os,statistics,sys,time
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tests.test_randomized_join import RandomizedJoinTests
from benchmarks.randomized_backward import make_input,native_call


def hashes(root):
    files=sorted((root/'conv_rosa_transformer/csrc').glob('*.h'))+sorted((root/'conv_rosa_transformer/csrc').glob('*.cpp'))
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def current_call(q,k,v,g,e0,e1,h=None):
    n,d=v.shape
    result=[np.zeros((n,d)) for _ in range(3)]+[np.zeros(n,np.int32),np.full(n,-1,np.int32),np.zeros(23,np.uint64)]
    arrays=[q,k,v,g,h,e0,e1]+result
    ptr=[ctypes.c_void_p() if a is None else a.ctypes.data_as(ctypes.c_void_p) for a in arrays]
    def run(on,seed):assert RandomizedJoinTests.credit(n,d,on,seed,*ptr)==0
    run.arrays=result
    return run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',type=Path,default=ROOT.parent/'qkv-rosa-train')
    p.add_argument('--repeats',type=int,default=11);p.add_argument('--cpu',type=int,default=8)
    p.add_argument('--output',type=Path,default=ROOT/'benchmarks/results/exact_randomized.json');a=p.parse_args()
    os.sched_setaffinity(0,{a.cpu})
    spec=importlib.util.spec_from_file_location('last_delivered_core',a.baseline_root/'conv_rosa_transformer/rosa_core.py')
    previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous);previous._library()
    RandomizedJoinTests.setUpClass();before=hashes(ROOT);old=hashes(a.baseline_root)
    rows=[];started=datetime.now(timezone.utc).isoformat()
    cases=[('random',1024,8),('random',8192,8),('random',2048,16),('random',8192,16),('random',8192,32),
           ('wide',8192,64),('narrow',8192,64),('correlated',8192,64),('asymmetric',8192,16),
           ('shadow',2048,64),('periodic',4096,4),('all_same',8192,8)]
    for case,(family,n,d) in enumerate(cases):
        q,k,v,g,h,e0,e1=make_input('random' if family in ('correlated','asymmetric') else family,n,d,981725+case)
        rng=np.random.default_rng(18926+case)
        if family=='correlated':
            q,k=rng.integers(0,256,(2,n),dtype=np.uint64)*np.uint64(0x0101010101010101)
        if family=='asymmetric':
            dictionary=rng.choice(1<<d,2048,replace=False).astype(np.uint64)
            k=dictionary[rng.integers(0,len(dictionary),n)].copy()
        args=q,k,v,g,e0,e1
        for model in (('linear','diagonal') if case in (2,5,9) else ('linear',)):
            diagonal=h if model=='diagonal' else None
            prior=native_call(previous,args,diagonal)
            fresh=current_call(*args,diagonal)
            fresh(0,31721);reference=[x.copy() for x in fresh.arrays]
            prior(0)
            for x,y in zip(reference,prior.arrays[-6:]):np.testing.assert_array_equal(x.view(np.uint8),y.view(np.uint8))
            for seed in (0,1,2**64-1):
                fresh(1,seed)
                for x,y in zip(reference,fresh.arrays):np.testing.assert_array_equal(x.view(np.uint8),y.view(np.uint8))
            samples={name:[] for name in ('baseline','native_off','native_on')}
            prepare={0:[],1:[]};counts=[]
            for rep in range(a.repeats):
                seed=719263+rep
                names=['baseline','native_off','native_on'];names=names[rep%3:]+names[:rep%3]
                if rep%2:names.reverse()
                for name in names:
                    begin=time.perf_counter_ns()
                    if name=='baseline':prior(seed)
                    else:fresh(name=='native_on',seed)
                    samples[name].append((time.perf_counter_ns()-begin)/1e6)
                per=[]
                for on in ((0,1) if rep%2==0 else (1,0)):
                    cc=np.zeros(12,np.uint64);ptr=[x.ctypes.data_as(ctypes.c_void_p) for x in (q,k,cc)]
                    begin=time.perf_counter_ns();assert RandomizedJoinTests.prepare(n,d,on,seed,*ptr)==0
                    prepare[on].append((time.perf_counter_ns()-begin)/1e6)
                    per.append(dict(randomized=bool(on),seed=seed,counters=cc.tolist()))
                counts.extend(per)
            med={name:statistics.median(v) for name,v in samples.items()}
            row=dict(family=family,T=n,D=d,model=model,bitwise_equal_baseline=True,bitwise_equal_on_off=True,
                median_native_ms=med,speedup_on_vs_last=med['baseline']/med['native_on'],speedup_on_vs_off=med['native_off']/med['native_on'],
                sam_preparation_median_ms={str(k):statistics.median(v) for k,v in prepare.items()},
                native_samples_ms=samples,sam_preparation_samples_ms=prepare,root_counts=counts)
            rows.append(row);print(json.dumps({k:row[k] for k in ('family','T','D','model','median_native_ms','speedup_on_vs_last','speedup_on_vs_off')}),flush=True)
    after=hashes(ROOT)
    result=dict(status='PASS' if before==after and old==hashes(a.baseline_root) else 'SOURCE_CHANGED',scope=__doc__,
        started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),baseline_commit='f6d7c4b73d7040b1b33860f5a2649c3cd3036cc5',
        source_sha256=before,baseline_source_sha256=old,source_unchanged=before==after,baseline_source_unchanged=old==hashes(a.baseline_root),
        test_bridge_sha256=hashlib.sha256((ROOT/'tests/test_randomized_join.py').read_bytes()).hexdigest(),
        loaded_binary_sha256=hashlib.sha256((ROOT/'build/tests/randomized_join_probe.so').read_bytes()).hexdigest(),
        compiler='c++ -O3 -DNDEBUG -std=c++17 -ffp-contract=off',cpu_affinity=[a.cpu],repeats=a.repeats,
        root_counter_names=['enabled','used','direct_root_queries','bucket_queries','bucket_members','bucket_build_entries','empty_buckets','verified_neighbors','query_classes','supported_mask','blocks','stored_root_entries'],rows=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],str(a.output),flush=True)
if __name__=='__main__':main()
