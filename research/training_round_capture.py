"""Capture actual component adjoints and assess projection-aware VJP errors.

The exact reference comes from the same native C++ training engine. Candidate
errors are reported at logits and after the shared Q/K/V projections; a large
per-token error must not automatically be called a large parameter error.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
IMPLEMENTATION_ROOT=Path(os.environ.get('QKV_CAPTURE_ROOT',ROOT))
sys.path.insert(0,str(IMPLEMENTATION_ROOT))
import training_round_tasks as tasks
from conv_rosa_transformer import rosa_layer as layer_module
from conv_rosa_transformer import native_training as native
from conv_rosa_transformer import rosa_core as core
def source_hashes():
    paths=sorted((IMPLEMENTATION_ROOT/'conv_rosa_transformer').glob('*.py'))+sorted((IMPLEMENTATION_ROOT/'conv_rosa_transformer/csrc').glob('*'))
    return {str(p.relative_to(IMPLEMENTATION_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def capture(seed,task,phase,examples,queries,order='second',lr=.03):
    torch.manual_seed(seed)
    model=tasks.make_model(layer_module.NBitQKVRosa,dict(credit_order=order,randomized=True),task)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    for step in range(phase):
        x,y=tasks.make_data(seed,task,examples,split=step+2,queries=queries)
        optimizer.zero_grad(set_to_none=True)
        loss=tasks.loss_value(model(x),y,task);loss.backward();optimizer.step()
    x,y=tasks.make_data(seed,task,examples,split=991,queries=queries)
    snapshot={}
    def capture_input(module,args):snapshot['component_input']=args[0].detach().cpu().numpy().copy()
    input_handle=model.rosa.register_forward_pre_hook(capture_input)
    def observe(cache,g):
        result=native.training_backward_numpy(cache,g,trace=True)
        snapshot.update(cache=cache,g=g.copy(),reference=result)
        return result
    optimizer.zero_grad(set_to_none=True)
    with patch.object(layer_module,'training_backward_numpy',observe):
        loss=tasks.loss_value(model(x),y,task);loss.backward()
    input_handle.remove()
    cache=snapshot['cache'];reference=snapshot['reference'];g=snapshot['g']
    arrays={name:getattr(cache,name) for name in ('q_logits','k_logits','v_logits','emb0','emb1','q','k','v_bits','lengths','routes')}
    arrays.update(grad_y=g,input=x.numpy(),component_input=snapshot['component_input'],target=y.numpy(),
        q_weight=model.rosa.q_proj.weight.detach().numpy(),
        k_weight=model.rosa.k_proj.weight.detach().numpy(),
        v_weight=model.rosa.v_proj.weight.detach().numpy(),
        pair_codes=reference[-1]['pair_codes'],pair_costs=reference[-1]['pair_costs'])
    arrays.update({f'reference_{i}':a for i,a in enumerate(reference[:5])})
    for key,p in model.named_parameters():
        arrays['parameter__'+key.replace('.','__')]=p.detach().numpy()
    row_norm=np.linalg.norm(g,axis=-1);active=row_norm>0
    meta=dict(seed=seed,task=task,phase=phase,examples=examples,queries=queries,lr=lr,
        phase_protocol='fresh_data_updates_before_capture',implementation_root=str(IMPLEMENTATION_ROOT),
        order=order,loss=float(loss.detach()),shape=list(g.shape),
        active_rows=int(active.sum()),total_rows=int(active.size),
        active_rows_per_stream=active.sum(-1).tolist(),
        row_l2_sum=float(row_norm.sum()),row_l2_squared_sum=float((row_norm**2).sum()),
        options={name:getattr(cache.options,name) for name,_ in cache.options._fields_},
        native_stats={k:v for k,v in reference[-1].items() if isinstance(v,(str,int,float,bool))})
    return arrays,meta


def restore(arrays,meta):
    options=native.make_options(credit_order='first' if meta['options']['credit_order']==1 else 'second')
    for key,value in meta['options'].items():
        if key not in ('abi_version','struct_size') and hasattr(options,key):setattr(options,key,value)
    cache=native.TrainingCache(*(arrays[k] for k in ('q_logits','k_logits','v_logits','emb0','emb1','q','k','v_bits','lengths','routes')),options)
    return cache,arrays['grad_y']


def measure_error(reference,estimate):
    a=np.concatenate([np.asarray(x,dtype=np.float64).ravel() for x in reference])
    b=np.concatenate([np.asarray(x,dtype=np.float64).ravel() for x in estimate])
    norm=float(np.linalg.norm(a));other=float(np.linalg.norm(b));error=float(np.linalg.norm(b-a))
    threshold=max(1e-14,norm/max(1,np.sqrt(a.size))*.1)
    salient=np.abs(a)>=threshold
    return dict(exact_norm=norm,estimate_norm=other,error_norm=error,
        relative_l2=error/norm if norm else (0. if error==0 else None),
        cosine=float(a@b/(norm*other)) if norm and other else None,
        relative_descent=float(a@b/(norm*norm)) if norm else None,
        maximum_absolute_error=float(np.abs(a-b).max(initial=0)),
        salient_sign_mismatch=float(np.mean(np.signbit(a[salient])!=np.signbit(b[salient]))) if salient.any() else None,
        exact_zero_estimate_nonzero=int(np.count_nonzero((a==0)&(b!=0))),elements=int(a.size))


def projected(arrays,gradients):
    b,groups,t,d=arrays['grad_y'].shape
    # Float32 is intentional: compare the actual shared projection gradients
    # used by the ordinary toy optimizer, including their rounding.
    x=torch.from_numpy(arrays.get('component_input',arrays['input'])).reshape(b*t,groups*d)
    gradient=[]
    for a in gradients[:3]:
        z=torch.from_numpy(a.transpose(0,2,1,3).reshape(b*t,groups*d).copy()).to(x.dtype)
        gradient.append(z)
    weights=[torch.from_numpy(arrays[k]) for k in ('q_weight','k_weight','v_weight')]
    params=[(g.T@x).numpy() for g in gradient]
    incoming=sum(g@w for g,w in zip(gradient,weights)).numpy()
    return params,incoming


def comparison(arrays,estimate):
    reference=[arrays[f'reference_{i}'] for i in range(5)]
    rp,rx=projected(arrays,reference);ep,ex=projected(arrays,estimate)
    return dict(q_logits=measure_error(reference[:1],estimate[:1]),
        k_logits=measure_error(reference[1:2],estimate[1:2]),
        qk_logits=measure_error(reference[:2],estimate[:2]),
        v_logits=measure_error(reference[2:3],estimate[2:3]),
        embeddings=measure_error(reference[3:],estimate[3:5]),
        q_parameters=measure_error(rp[:1],ep[:1]),
        k_parameters=measure_error(rp[1:2],ep[1:2]),
        qk_parameters=measure_error(rp[:2],ep[:2]),
        full_component_parameters=measure_error(rp+reference[3:],ep+list(estimate[3:5])),
        incoming_input=measure_error([rx],[ex]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',nargs='+',default=['retrieval','dense_retrieval_fresh','periodic_copy_fresh'])
    p.add_argument('--phases',type=int,nargs='+',default=[0,20])
    p.add_argument('--seeds',type=int,nargs='+',default=[1101])
    p.add_argument('--queries',type=int,default=128);p.add_argument('--examples',type=int,default=4)
    p.add_argument('--order',default='second',choices=['first','second'])
    p.add_argument('--lr',type=float,default=.03)
    p.add_argument('--cpu',type=int,default=2)
    p.add_argument('--output-dir',type=Path,default=ROOT/'research/results/training_round_captures')
    a=p.parse_args();os.sched_setaffinity(0,{a.cpu});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    library=core._library();before=source_hashes();rows=[];a.output_dir.mkdir(parents=True,exist_ok=True)
    native_library=dict(path=library._name,sha256=hashlib.sha256(Path(library._name).read_bytes()).hexdigest())
    for task in a.tasks:
      for seed in a.seeds:
       for phase in a.phases:
        arrays,meta=capture(seed,task,phase,a.examples,a.queries,a.order,a.lr)
        name=f'{task}_{a.order}_s{seed}_p{phase}'
        np.savez_compressed(a.output_dir/(name+'.npz'),**arrays)
        (a.output_dir/(name+'.json')).write_text(json.dumps(meta,indent=2)+'\n')
        rows.append(dict(name=name,**meta));print(name,meta['loss'],meta['active_rows'],meta['total_rows'],flush=True)
    after=source_hashes()
    report=dict(status='COMPLETE' if before==after else 'SOURCE_CHANGED',source_before=before,source_after=after,native_library=native_library,
        source_unchanged=before==after,rows=rows)
    (a.output_dir/'index.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
