"""Paired ordinary-training study for research backward callables.

A candidate module exports backward(cache, grad_y, seed=<int>, **settings) and
returns the native six-result contract. Only the component backward binding
is replaced; the model forward, task loss, optimizer, and outer loop are shared.
"""
from __future__ import annotations
import argparse,contextlib,hashlib,importlib,json,os,sys,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
IMPLEMENTATION_ROOT=Path(os.environ.get('QKV_STUDY_ROOT',ROOT))
sys.path.insert(0,str(IMPLEMENTATION_ROOT))
import training_round_tasks as tasks
from conv_rosa_transformer import rosa_layer as layer_module
from conv_rosa_transformer import rosa_core as core
from conv_rosa_transformer.native_training import training_backward_numpy


def tensor_hash(values):
    digest=hashlib.sha256()
    for x in values:
        a=x.detach().cpu().contiguous().numpy()
        digest.update(str((a.shape,a.dtype.str)).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def source_hashes():
    out={}
    for p in sorted((IMPLEMENTATION_ROOT/'conv_rosa_transformer').glob('*.py'))+sorted((IMPLEMENTATION_ROOT/'conv_rosa_transformer/csrc').glob('*')):
        if p.is_file():out[str(p.relative_to(IMPLEMENTATION_ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def cache_hash(cache):
    digest=hashlib.sha256()
    for name in ('q','k','v_bits','lengths','routes'):
        value=getattr(cache,name)
        digest.update(name.encode());digest.update(str((value.shape,value.dtype.str)).encode())
        digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


def candidate(spec):
    if 'module' not in spec:return None
    return getattr(importlib.import_module(spec['module']),spec.get('callable','backward'))


def run(spec,seed,task,steps=100,lr=.03,examples=8,queries=64,seconds=None,eval_every=10):
    torch.manual_seed(seed)
    model=tasks.make_model(layer_module.NBitQKVRosa,
        dict(credit_order=spec.get('credit_order','second'),randomized=True,**spec.get('layer_config',{})),task)
    # Save the ordinary framework stream independently from candidate randomness.
    framework=torch.get_rng_state().clone()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    x,y=tasks.make_data(seed,task,examples,queries=queries)
    xt,yt=tasks.make_data(seed,task,max(32,examples*4),split=1,queries=queries)
    fresh=task.endswith('_fresh') or task=='periodic_copy_weighted'
    initial=tensor_hash(model.state_dict().values())
    data_hash=tensor_hash((x,y,xt,yt))
    call=candidate(spec)
    histories=[];batch_losses=[];update_hashes=[];forward_hashes=[];step_times=[];elapsed=0.;native_counts=[];last_cache=None
    counters={'forward':0,'backward':0};candidate_calls=0
    def forward_count(*unused):counters['forward']+=1
    handle=model.rosa.register_forward_hook(forward_count)
    def backward(cache,g):
        nonlocal candidate_calls,last_cache
        last_cache=cache
        counters['backward']+=1
        if call is None:result=training_backward_numpy(cache,g)
        else:
            result=call(cache,g,seed=(seed*1000003+candidate_calls*104729+1729)%2**64,**spec.get('settings',{}))
        candidate_calls+=1
        native_counts.append({k:v for k,v in result[-1].items() if isinstance(v,(int,float,bool,str))})
        return result
    def evaluation(step):
        model.eval()
        with torch.no_grad():fit=tasks.metrics(model(x),y,task);test=tasks.metrics(model(xt),yt,task)
        histories.append(dict(step=step,training_seconds=elapsed,fit=fit,test=test,
            recent_loss=float(np.mean(batch_losses[-10:])) if batch_losses else None,
            parameters_sha256=tensor_hash(model.state_dict().values())))
    with patch.object(layer_module,'training_backward_numpy',backward):
        evaluation(0)
        for step in range(steps):
            model.train();torch.set_rng_state(framework)
            xb,yb=tasks.make_data(seed,task,examples,split=step+2,queries=queries) if fresh else (x,y)
            before=counters.copy();start=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss=tasks.loss_value(model(xb),yb,task)
            loss.backward();optimizer.step()
            duration=time.perf_counter()-start;elapsed+=duration;step_times.append(duration)
            # Hash the actual hard forward cache outside the training timer.
            # This catches routing changes without adding a model evaluation.
            forward_hashes.append(cache_hash(last_cache))
            framework=torch.get_rng_state().clone()
            assert counters['forward']-before['forward']==1 and counters['backward']-before['backward']==1
            batch_losses.append(float(loss.detach()));update_hashes.append(tensor_hash(model.state_dict().values()))
            if (step+1)%eval_every==0:evaluation(step+1)
            if seconds is not None and elapsed>=seconds:break
        completed=step+1
        if histories[-1]['step']!=completed:evaluation(completed)
    handle.remove()
    return dict(spec=spec,seed=seed,task=task,steps=completed,lr=lr,examples=examples,queries=queries,
        data_protocol='fresh_data_each_update' if fresh else 'fixed_training_set',
        requested_time_seconds=seconds,training_seconds=elapsed,initial_state_sha256=initial,
        input_sha256=data_hash,initial=histories[0],final=histories[-1],history=histories,
        batch_losses=batch_losses,every_update_parameter_sha256=update_hashes,
        every_training_hard_forward_sha256=forward_hashes,
        final_framework_rng_sha256=hashlib.sha256(framework.numpy().tobytes()).hexdigest(),
        training_model_forwards=completed,training_backward_calls=completed,
        step_seconds=step_times,
        step_work=[{k:v for k,v in c.items() if k in ('singleton_corridors','singleton_represented_rows',
            'singleton_dot_components','credit_engine_calls','approx_population_rows',
            'approx_selected_rows','approx_certified_zero_rows','approx_extra_joint_replays') or 'fft' in k or 'spectral' in k}
            for c in native_counts],
        native_counter_first=native_counts[0],native_counter_last=native_counts[-1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--specs',type=Path);p.add_argument('--seeds',type=int,nargs='+',default=[1101])
    p.add_argument('--tasks',nargs='+',default=['dense_retrieval_fresh'])
    p.add_argument('--steps',type=int,default=100);p.add_argument('--lr',type=float,default=.03)
    p.add_argument('--examples',type=int,default=8);p.add_argument('--queries',type=int,default=64)
    p.add_argument('--seconds',type=float);p.add_argument('--cpu',type=int,default=2)
    p.add_argument('--equal-time',action='store_true',help='Also restart each approximate method for the same measured time as the exact fixed-step run')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    os.sched_setaffinity(0,{a.cpu});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    library=core._library();before=source_hashes()
    native_library=dict(path=library._name,sha256=hashlib.sha256(Path(library._name).read_bytes()).hexdigest())
    research_paths=(Path(__file__),ROOT/'research/training_round_tasks.py')
    research_before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in research_paths}
    specs=json.loads(a.specs.read_text()) if a.specs else [dict(method='exact',credit_order='second')]
    results=[]
    def record(row,protocol):
        row['protocol']=protocol;results.append(row)
        print(json.dumps({k:row[k] for k in ['spec','protocol','seed','task','steps','training_seconds','initial','final']}),flush=True)
    for task in a.tasks:
      for index,seed in enumerate(a.seeds):
        budget=None
        if a.equal_time:
            exact_spec=next(spec for spec in specs if spec['method']=='exact')
            exact=run(exact_spec,seed,task,a.steps,a.lr,a.examples,a.queries,a.seconds)
            budget=exact['training_seconds'];record(exact,'equal_steps')
        for spec in (specs if index%2==0 else specs[::-1]):
            if a.equal_time and spec['method']=='exact':continue
            row=run(spec,seed,task,a.steps,a.lr,a.examples,a.queries,a.seconds)
            record(row,'equal_steps')
            if a.equal_time:
                timed=run(spec,seed,task,max(10000,a.steps),a.lr,a.examples,a.queries,budget)
                record(timed,'equal_time')
    after=source_hashes()
    research_after={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in research_paths}
    result=dict(status='COMPLETE' if before==after and research_before==research_after else 'SOURCE_CHANGED',
        source_before=before,source_after=after,source_unchanged=before==after,
        numpy=np.__version__,torch=torch.__version__,affinity=sorted(os.sched_getaffinity(0)),
        blas_thread_environment={k:os.environ.get(k,'unset') for k in
            ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        native_library=native_library,implementation_root=str(IMPLEMENTATION_ROOT),research_source_hashes_before=research_before,
        research_source_hashes=research_after,records=results)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
