"""Frozen-adjoint channel controls through the actual public native API.

Each exact reference and estimate use the same native library. The source
capture is held fixed; only the upstream adjoint profile and RNG seed vary.
Call times are diagnostics, not the formal benchmark.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'research'))
from training_round_capture import restore,comparison,source_hashes,measure_error
from training_round_probe_utils import adjoint,summary
from conv_rosa_transformer import native_training as native


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,default=ROOT/'research/results/training_round_captures_prefix64_2048/prefix_copy64_fresh_second_s1101_p0.npz')
    p.add_argument('--profiles',nargs='+',default=['balanced','single_dominant','nine_dominant','tangent','nine_tangent'])
    p.add_argument('--orders',nargs='+',default=['first','second'])
    p.add_argument('--budgets',type=int,nargs='+',default=[0,8])
    p.add_argument('--draws',type=int,default=16);p.add_argument('--cpu',type=int,default=2)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();os.sched_setaffinity(0,{a.cpu});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    library=native._library();before=source_hashes()
    captured=dict(np.load(a.capture));meta=json.loads(a.capture.with_suffix('.json').read_text());records=[]
    for profile in a.profiles:
      for order in a.orders:
        arrays=dict(captured);arrays['grad_y']=adjoint(captured,profile)
        cache,g=restore(arrays,meta)
        cache.options=native.make_options(credit_order=order,randomized=True,
            proposal_seed=cache.options.proposal_seed,repair_seed=cache.options.repair_seed)
        reference=native.training_backward_numpy(cache,g,trace=True)
        for i,value in enumerate(reference[:5]):arrays[f'reference_{i}']=value
        for budget in a.budgets:
            cache.approximation=native.make_approximation_options(backward_mode='approximate',
                approximation_rows=32,approximation_channels=budget)
            errors=[];counters=[];elapsed=[];pairs=[];costs=[];exact_v=[];exact_emb=[]
            cost_errors=[];totals=[np.zeros_like(v) for v in reference[:5]]
            for draw in range(a.draws):
                cache.approximation.seed=0x91a7111+draw*104729
                start=time.perf_counter();result=native.training_backward_numpy(cache,g,trace=True)
                elapsed.append(time.perf_counter()-start);errors.append(comparison(arrays,result[:5]))
                counters.append({k:v for k,v in result[-1].items() if isinstance(v,(int,str,float,bool))})
                pairs.append(np.array_equal(reference[-1]['pair_codes'],result[-1]['pair_codes']))
                rc,ec=reference[-1]['pair_costs'],result[-1]['pair_costs'];valid=np.isfinite(rc)&np.isfinite(ec)
                costs.append(np.allclose(rc,ec,rtol=1e-12,atol=1e-12,equal_nan=True))
                cost_errors.append(measure_error([rc[valid]],[ec[valid]]))
                exact_v.append(np.array_equal(reference[2].view(np.uint8),result[2].view(np.uint8)))
                exact_emb.append(all(np.array_equal(reference[i].view(np.uint8),result[i].view(np.uint8)) for i in (3,4)))
                for total,value in zip(totals,result[:5]):total+=value/a.draws
            row=dict(profile=profile,order=order,channels=budget,draws=a.draws,
                adjoint_sha256=hashlib.sha256(g.tobytes()).hexdigest(),summary=summary(errors),
                draw_errors=errors,finite_sample_mean_error=comparison(arrays,totals),
                native_counters=counters,diagnostic_seconds=elapsed,pair_codes_equal=all(pairs),
                pair_costs_allclose_1e12=all(costs),pair_cost_errors=cost_errors,
                value_gradient_bitwise_equal=all(exact_v),embedding_gradients_bitwise_equal=all(exact_emb))
            records.append(row);s=row['summary']
            print(profile,order,budget,'Q',s['q_parameters']['relative_l2']['median'],
                'QK',s['qk_parameters']['relative_l2']['median'],'input',s['incoming_input']['relative_l2']['median'],flush=True)
            report=dict(status='RUNNING',scope=__doc__,capture=str(a.capture),source_before=before,
                native_library=dict(path=library._name,sha256=hashlib.sha256(Path(library._name).read_bytes()).hexdigest()),
                blas_threads={k:os.environ.get(k,'unset') for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=records)
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps(report,indent=2)+'\n')
    report['source_after']=source_hashes()
    report['status']='COMPLETE' if report['source_after']==before else 'SOURCE_CHANGED'
    a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
