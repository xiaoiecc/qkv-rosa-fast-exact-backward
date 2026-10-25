"""Paired native training costs, with hard-component and full-layer scopes.

The full-layer scope includes Q/K/V/O dense projections and their gradients.
Both scopes exclude a downstream model and optimizer. The frozen Python
component is imported into a separate module namespace with its own old core.
All library compilation and warmup are outside timing.
"""
from __future__ import annotations
import argparse, hashlib, importlib, json, os, platform, statistics, sys, time, types
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from conv_rosa_transformer import rosa_core as core
from conv_rosa_transformer.rosa_layer import NBitQKVRosa


def load_baseline(root):
    name='frozen_qkv_training_baseline'
    package=types.ModuleType(name);package.__path__=[str(root/'conv_rosa_transformer')]
    sys.modules[name]=package
    layer=importlib.import_module(name+'.rosa_layer')
    old_core=importlib.import_module(name+'.rosa_core')
    return layer.NBitQKVRosa,old_core


def hashes(root):
    paths=sorted((root/'conv_rosa_transformer').glob('*.py'))
    paths+=sorted((root/'conv_rosa_transformer/csrc').glob('*.h'))
    paths+=sorted((root/'conv_rosa_transformer/csrc').glob('*.cpp'))
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',type=Path,default=ROOT.parent/'qkv-rosa-train')
    p.add_argument('--cpu',type=int,default=8);p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--output',type=Path,default=ROOT/'benchmarks/results/native_component.json')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{a.cpu})
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    Baseline,oldcore=load_baseline(a.baseline_root)
    core._library();oldcore._library()
    before=hashes(ROOT);oldbefore=hashes(a.baseline_root)
    rows=[]
    cases=[(128,8,4,2),(1024,8,1,1),(1024,16,2,4),
           (8192,8,1,1),(8192,16,1,1),(8192,64,1,1)]
    if a.smoke:cases=[(17,4,2,2)]
    for case,(t,d,b,g) in enumerate(cases):
        c=d*g;torch.manual_seed(501271+case)
        models={
            'baseline_second':Baseline(c,d,repair_samples=None,joint_pairs=1,joint_rows=32,
                stochastic_values=False,trust_diagonal=.5,training_rule='hard_credit',joint_row_selection='active')}
        for order in ('first','second'):
            for random in (False,True):
                name=order+('_random' if random else '_exact')
                models[name]=NBitQKVRosa(c,d,credit_order=order,randomized=random)
                models[name].load_state_dict(models['baseline_second'].state_dict(),strict=True)
        rng=np.random.default_rng(417201+case)
        logits=[torch.tensor((2*rng.integers(0,2,(b,t,c)).astype(np.float32)-1)*.35,requires_grad=True)
                for _ in range(3)]
        x=torch.tensor(rng.normal(size=(b,t,c)).astype(np.float32),requires_grad=True)
        gy=torch.tensor(rng.normal(size=(b,t,c)).astype(np.float32)/np.sqrt(d))
        for scope in ('hard_component','full_layer'):
            def execute(name,rep,return_values=False):
                model=models[name];model.train();model.zero_grad(set_to_none=True)
                x.grad=None
                for z in logits:z.grad=None
                torch.manual_seed(181141+rep)
                start=time.perf_counter_ns()
                y=model(x) if scope=='full_layer' else model._hard_rosa_from_qkv(*logits)
                fwd=time.perf_counter_ns()
                objective=(y*gy).sum()
                back=time.perf_counter_ns();objective.backward();end=time.perf_counter_ns()
                if return_values:
                    gradients=([x.grad] if scope=='full_layer' else [z.grad for z in logits])
                    gradients += [p.grad for p in model.parameters() if p.grad is not None]
                    return y.detach().clone(),[z.detach().clone() for z in gradients]
                return [(fwd-start)/1e6,(end-back)/1e6]
            # Equality is an answer/gradient gate, outside the timer. Both
            # exact-acceleration settings use the same proposal realization.
            for order in ('first','second'):
                off,goff=execute(order+'_exact',-1,True)
                on,gon=execute(order+'_random',-1,True)
                assert torch.equal(off,on) and len(goff)==len(gon)
                assert all(torch.equal(a,b) for a,b in zip(goff,gon)),(scope,order,t,d)
            baseline,_=execute('baseline_second',-1,True)
            assert torch.equal(baseline,off),(scope,t,d)
            samples={name:[] for name in models}
            names=list(models)
            for rep in range(a.repeats):
                order=names if rep%2==0 else names[::-1]
                for name in order:samples[name].append(execute(name,rep))
            methods={}
            for name,data in samples.items():
                arr=np.array(data)
                methods[name]=dict(forward_median_ms=float(np.median(arr[:,0])),
                    backward_median_ms=float(np.median(arr[:,1])),
                    forward_backward_median_ms=float(np.median(arr.sum(axis=1))),
                    samples_forward_backward_ms=data)
            ratios={}
            for order in ('first','second'):
                ratios[order+'_random_speedup']=methods[order+'_exact']['forward_backward_median_ms']/methods[order+'_random']['forward_backward_median_ms']
            ratios['native_second_speedup_vs_frozen']=methods['baseline_second']['forward_backward_median_ms']/methods['second_random']['forward_backward_median_ms']
            row=dict(T=t,D=d,B=b,G=g,scope=scope,methods=methods,ratios=ratios,
                random_on_off_outputs_and_gradients_bitwise_equal=True,baseline_forward_bitwise_equal=True)
            rows.append(row);print(json.dumps(dict(T=t,D=d,B=b,G=g,scope=scope,ratios=ratios)),flush=True)
    after=hashes(ROOT);oldafter=hashes(a.baseline_root)
    report=dict(status='PASS' if before==after and oldbefore==oldafter else 'INVALID_SOURCE_CHANGED',scope=__doc__,
        source_before=before,source_after=after,source_unchanged=before==after,
        baseline_source_before=oldbefore,baseline_source_after=oldafter,baseline_source_unchanged=oldbefore==oldafter,
        native_library_sha256=hashlib.sha256(Path(core._library()._name).read_bytes()).hexdigest(),
        baseline_library_sha256=hashlib.sha256(Path(oldcore._library()._name).read_bytes()).hexdigest(),
        baseline_revision='f6d7c4b73d7040b1b33860f5a2649c3cd3036cc5',
        torch_version=torch.__version__,platform=platform.platform(),affinity=sorted(os.sched_getaffinity(0)),
        repeats=a.repeats,smoke=a.smoke,rows=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
