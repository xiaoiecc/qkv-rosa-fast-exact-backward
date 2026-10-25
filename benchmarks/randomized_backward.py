"""Exact hash dictionaries and support-certified repair sampling, CPU benchmark.

Reference: the previous completed research archive. Public calls include NumPy
validation/allocation, native computation and estimator weighting. Native calls
reuse output buffers; their sampled values exclude Python weighting. Neither
sampler calls an extra forward. Exact conditional estimator MSE is computed
from the complete credit vector, and is not a training or loss-error claim.
"""
from pathlib import Path
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from conv_rosa_transformer import rosa_core as core
from benchmarks.curvature import make_case


def hashes(root):
    paths=[root/'conv_rosa_transformer/rosa_core.py']
    paths+=sorted((root/'conv_rosa_transformer/csrc').glob('*.h'))
    paths+=sorted((root/'conv_rosa_transformer/csrc').glob('*.cpp'))
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def array_hash(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256(repr((a.dtype.str,a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()


def make_input(family,n,d,seed):
    if family not in {'wide','narrow'}:
        return make_case(family,n,d,seed)
    rng=np.random.default_rng(seed)
    if family=='wide':
        q,k=rng.bit_generator.random_raw((2,n))
        rows=np.arange(1,n,4);bits=rng.integers(0,64,len(rows),dtype=np.uint64)
        q[rows]=k[rows-1]^(np.uint64(1)<<bits)
    else:
        q,k=rng.integers(0,256,(2,n),dtype=np.uint64)
    v=rng.integers(0,2,(n,d),dtype=np.uint8)
    g=rng.normal(size=(n,d));h=rng.uniform(.1,1.5,(n,d))
    e0,e1=rng.normal(size=(2,d))
    return q,k,v,g,h,e0,e1


def native_call(module,args,diagonal=None,budget=None):
    q,k,v,g,e0,e1=args;n,d=v.shape
    outputs=[np.zeros((n,d)) for _ in range(3)]+[np.zeros(n,np.int32),np.full(n,-1,np.int32),np.zeros(23,np.uint64)]
    if budget is None:
        arrays=[q,k,v,g]+([] if diagonal is None else [diagonal])+[e0,e1]+outputs
        entry=module._library().rosa_uniform if diagonal is None else module._library().rosa_quadratic
        pointers=[a.ctypes.data_as(ctypes.c_void_p) for a in arrays]
        def run(seed):
            assert entry(n,d,*pointers)==0
    else:
        arrays=[q,k,v,g,diagonal,e0,e1]+outputs+[np.zeros(n),np.zeros(n),np.zeros(5,np.uint64)]
        pointers=[ctypes.c_void_p() if a is None else a.ctypes.data_as(ctypes.c_void_p) for a in arrays]
        entry=module._library().rosa_budgeted
        def run(seed):
            assert entry(n,d,budget,seed,0,*pointers)==0
    run.arrays=arrays
    return run


def timed(fn,*args):
    start=time.perf_counter_ns();out=fn(*args)
    return out,(time.perf_counter_ns()-start)/1e6


def summarize(values):
    return dict(median_ms=statistics.median(values),mean_ms=statistics.mean(values),samples_ms=values)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path,default=ROOT.parent/'qkv-rosa-next')
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/results/randomized_backward.json')
    parser.add_argument('--cpu',type=int,default=8)
    parser.add_argument('--repeats',type=int,default=7)
    opts=parser.parse_args()
    if opts.repeats<3:parser.error('At least three repeats required')
    os.sched_setaffinity(0,{opts.cpu})
    spec=importlib.util.spec_from_file_location('previous_research_core',opts.baseline_root/'conv_rosa_transformer/rosa_core.py')
    previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
    previous._library();core._library()
    before,old_before=hashes(ROOT),hashes(opts.baseline_root)
    report=dict(status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=before,baseline_source_sha256=old_before,
        baseline='previous completed diagonal/joint/forward research archive',
        seed=2710930,repeats=opts.repeats,affinity=sorted(os.sched_getaffinity(0)),platform=platform.platform(),
        cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown'),
        public_scope='NumPy validation/allocation + full native core + inverse-probability weighting',
        native_scope='Full native core with reused buffers; sampled native outputs unweighted',
        excluded=['compilation/warmup','g/h acquisition','projections','optimizer','GPU transfers'],
        sampling='uniform without replacement over certified support; exact deletion and V',
        noise_scope='exact-real conditional expected squared credit error, not training/loss error',
        hash_scope='full keys checked, seeds affect work only; no universal hash-family bound claimed',
        extra_forward_calls_per_backward=0,rows=[])
    cases=[('random',1024,8),('random',8192,8),('random',8192,16),('wide',8192,64),
           ('narrow',8192,64),('all_same',8192,8),('shadow',2048,64),('periodic',4096,4),('random',8192,1)]
    for index,(family,n,d) in enumerate(cases):
        seed=2710930+index;q,k,v,g,h,e0,e1=make_input(family,n,d,seed);args=q,k,v,g,e0,e1
        input_hashes=[array_hash(a) for a in (q,k,v,g,h,e0,e1)]
        for model in (('linear','diagonal') if index in (1,2,6) else ('linear',)):
            diagonal=h if model=='diagonal' else None
            def exact(module):
                return (module.credits_numpy(*args) if diagonal is None else
                        module.quadratic_credits_numpy(q,k,v,g,diagonal,e0,e1))
            old,full=exact(previous),exact(core)
            assert old[3:5]==full[3:5]
            assert all(np.array_equal(a.view(np.uint64),b.view(np.uint64)) for a,b in zip(old[:3],full[:3]))
            assert all(old[5][key]==full[5][key] for key in core.COUNTER_NAMES)
            base=core.selected_credits_numpy(*args,[],diag_y=diagonal)
            dense=np.stack(full[:2]);baseline=np.stack([np.broadcast_to(base[6+s][:,None],dense[s].shape) for s in range(2)])
            residual=dense-baseline;norm_sq=float(np.sum(dense*dense));residual_sq=float(np.sum(residual*residual))
            pub={key:[] for key in ('previous_exact','current_exact')};native={key:[] for key in pub}
            calls={'previous_exact':native_call(previous,args,diagonal),'current_exact':native_call(core,args,diagonal)}
            for repeat in range(opts.repeats):
                order=[('previous_exact',previous),('current_exact',core)]
                if repeat%2:order.reverse()
                for label,module in order:
                    _,ms=timed(exact,module);pub[label].append(ms)
                    _,ms=timed(calls[label],seed+repeat);native[label].append(ms)
            sampled=[]
            for budget in sorted(set((min(d,4),max(1,d//2)))):
                native_budget=native_call(core,args,diagonal,budget);ms_public=[];ms_native=[];realizations=[]
                for repeat in range(opts.repeats):
                    sample_seed=seed*100+repeat
                    fn=lambda:core.adaptive_repair_sampled_credits_numpy(*args,budget,sample_seed,diag_y=diagonal)
                    got,ms=timed(fn);ms_public.append(ms)
                    _,ms=timed(native_budget,sample_seed);ms_native.append(ms)
                    assert got[3:5]==full[3:5] and np.array_equal(got[2].view(np.uint64),full[2].view(np.uint64))
                    selected,support=got[5]['repair_bits'],got[5]['supported_repair_bits']
                    unsupported=[bit for bit in range(d) if bit not in support]
                    for stream in range(2):
                        assert np.array_equal(got[stream][:,unsupported].view(np.uint64),full[stream][:,unsupported].view(np.uint64))
                    p=len(selected)/len(support) if support else 1.
                    expected_mse=(1-p)/p*residual_sq
                    naive_support=(1-p)/p*float(np.sum(dense[...,support]**2))
                    naive_p=min(1.,budget/d);naive_all=(1-naive_p)/naive_p*norm_sq
                    error_sq=float(np.sum((np.stack(got[:2])-dense)**2))
                    realizations.append(dict(seed=sample_seed,selected=selected,supported=support,pi_on_support=p,
                        actual_squared_error=error_sq,sam_active_advances=got[5]['sam_active_advances'],visited_alignment_pairs=got[5]['visited_alignment_pairs']))
                    theory=dict(expected_squared_error=expected_mse,
                        expected_relative_rms_qk=math.sqrt(expected_mse/norm_sq) if norm_sq else 0.,
                        naive_same_support_expected_squared_error=naive_support,naive_all_d_expected_squared_error=naive_all,
                        control_variate_ratio_same_support=expected_mse/naive_support if naive_support else 0.,
                        active_lane_count=len(support),actual_lane_count=len(selected),exact_on_this_case=got[5]['repair_sampling_exact'],
                        root_repair_entries=got[5]['root_repair_entries'],query_symbol_classes=got[5]['query_symbol_classes'],
                        transition_working_capacity_bytes=got[5]['transition_working_capacity_bytes'])
                psummary,nsummary=summarize(ms_public),summarize(ms_native)
                sampled.append(dict(target_budget=budget,public=psummary,native_unweighted=nsummary,noise=theory,realizations=realizations,
                    speedup_vs_current_exact_public=statistics.median(pub['current_exact'])/psummary['median_ms'],
                    mean_speedup_vs_current_exact_public=statistics.mean(pub['current_exact'])/psummary['mean_ms']))
            public={key:summarize(values) for key,values in pub.items()};native_summary={key:summarize(values) for key,values in native.items()}
            row=dict(family=family,T=n,D=d,model=model,input_seed=seed,input_sha256=input_hashes,
                bitwise_equal_previous_exact=True,structural_counters_equal_previous=True,public=public,native=native_summary,
                exact_public_speedup=public['previous_exact']['median_ms']/public['current_exact']['median_ms'],
                exact_native_speedup=native_summary['previous_exact']['median_ms']/native_summary['current_exact']['median_ms'],
                qk_credit_norm_sq=norm_sq,repair_residual_norm_sq=residual_sq,all_lane_control_variate_ratio=residual_sq/norm_sq if norm_sq else 0.,
                full_counters=full[5],sampling=sampled)
            report['rows'].append(row)
            print(json.dumps(dict(family=family,T=n,D=d,model=model,exact_ms=public['current_exact']['median_ms'],exact_speedup=row['exact_public_speedup'],
                sampled=[(x['target_budget'],x['public']['median_ms'],x['noise']['expected_relative_rms_qk'],x['noise']['active_lane_count']) for x in sampled])),flush=True)
        assert input_hashes==[array_hash(a) for a in (q,k,v,g,h,e0,e1)]
    report['source_unchanged']=before==hashes(ROOT);report['baseline_source_unchanged']=old_before==hashes(opts.baseline_root)
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    report['status']='PASS' if report['source_unchanged'] and report['baseline_source_unchanged'] else 'INVALID_SOURCE_CHANGED'
    opts.output.parent.mkdir(parents=True,exist_ok=True);opts.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],output=str(opts.output))),flush=True)


if __name__=='__main__':main()
