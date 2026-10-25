"""Ordinary composed-model training for native first/second and random off/on.

The frozen baseline and native implementation run in separate invocations so
their Python modules and loaded libraries cannot collide. Loss, data, optimizer,
initialization, and evaluation are shared with the preceding frozen study.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time, warnings
from pathlib import Path

import numpy as np
import torch
import native_random_toy as shared

HERE=Path(__file__).resolve()
ROOT=HERE.parents[1]


def tensor_hash(values):
    h=hashlib.sha256()
    for value in values:
        a=value.detach().cpu().contiguous().numpy()
        h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
    return h.hexdigest()


def hashes(root, core):
    paths=sorted((root/'conv_rosa_transformer').glob('*.py'))
    paths+=sorted((root/'conv_rosa_transformer/csrc').glob('*.h'))
    paths+=sorted((root/'conv_rosa_transformer/csrc').glob('*.cpp'))
    out={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    out['experiment_driver']=hashlib.sha256(HERE.read_bytes()).hexdigest()
    out['shared_toy']=hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest()
    out['loaded_native_binary']=hashlib.sha256(Path(core._library()._name).read_bytes()).hexdigest()
    return out


def configuration(method, implementation):
    if implementation=='baseline':
        assert method=='baseline_second'
        return dict(repair_samples=None,joint_pairs=1,joint_rows=32,
                    stochastic_values=False,trust_diagonal=.5,
                    training_rule='hard_credit',joint_row_selection='active')
    order,random=method.split('_')
    return dict(credit_order=order,randomized=random=='random')


def run(layer_class, implementation, method, seed, task, steps, lr, n):
    config=configuration(method,implementation)
    torch.manual_seed(seed)
    classification=task.startswith('retrieval_classification')
    fresh=task.endswith('_fresh')
    cls=shared.ClassificationRetrieval if classification else shared.Retrieval
    model=cls(layer_class,config)
    x,target=shared.make_data(seed,task,n)
    xt,yt=shared.make_data(seed,task,96,split=1)
    batches=([(x,target)]+[shared.make_data(seed,task,n,split=i+2)
        for i in range(1,steps)]) if fresh else []
    data_hash=tensor_hash((x,target,xt,yt)+tuple(t for pair in batches for t in pair))
    initial_hash=tensor_hash(model.state_dict().values())
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    counters=dict(forward=0,backward=0)
    def forward_hook(*args):counters['forward']+=1
    def backward_hook(*args):counters['backward']+=1
    fh=model.rosa.register_forward_hook(forward_hook)
    bh=model.rosa.register_full_backward_hook(backward_hook)
    history=[];training_time=0.;training_forwards=training_backwards=0
    batch_losses=[];update_hashes=[];begin=time.perf_counter()
    for step in range(steps+1):
        if step%10==0 or step==steps:
            model.eval()
            with torch.no_grad():
                fit=shared.metrics(model(x),target,classification)
                test=shared.metrics(model(xt),yt,classification)
            history.append(dict(step=step,**fit,test_loss=test['loss'],
                test_bit_accuracy=test['bit_accuracy'],test_payload_accuracy=test['payload_accuracy'],
                recent_batch_training_loss=float(np.mean(batch_losses[-10:])) if batch_losses else None,
                training_seconds=training_time,parameter_sha256=tensor_hash(model.state_dict().values())))
        if step==steps:break
        model.train();before=counters.copy();start=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        xb,yb=batches[step] if fresh else (x,target)
        loss=shared.loss_value(model(xb),yb,classification)
        batch_losses.append(float(loss.detach()))
        loss.backward();optimizer.step()
        training_time+=time.perf_counter()-start
        assert counters['forward']-before['forward']==1
        assert counters['backward']-before['backward']==1
        training_forwards+=1;training_backwards+=1
        update_hashes.append(tensor_hash(model.state_dict().values()))
    fh.remove();bh.remove()
    return dict(implementation=implementation,method=method,seed=seed,task=task,steps=steps,
        lr=lr,examples=n,config=config,training_seconds=training_time,
        elapsed_seconds=time.perf_counter()-begin,initial_state_sha256=initial_hash,
        data_sha256=data_hash,final_state_sha256=tensor_hash(model.state_dict().values()),
        final_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        every_update_parameter_sha256=update_hashes,training_batch_losses=batch_losses,
        training_model_forwards=training_forwards,training_backward_calls=training_backwards,
        initial=history[0],final=history[-1],history=history)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation',choices=['baseline','native'],required=True)
    p.add_argument('--baseline-root',type=Path,default=ROOT.parent/'qkv-rosa-train')
    p.add_argument('--seeds',type=int,nargs='+',default=[901,902,903])
    p.add_argument('--tasks',nargs='+',default=['retrieval','retrieval_classification_fresh'])
    p.add_argument('--methods',nargs='+')
    p.add_argument('--steps',type=int,default=100);p.add_argument('--lr',type=float,default=.03)
    p.add_argument('--examples',type=int,default=24);p.add_argument('--cpu',type=int,default=2)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{args.cpu})
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    warnings.filterwarnings('ignore',message='Full backward hook is firing when gradients are computed')
    root=args.baseline_root if args.implementation=='baseline' else ROOT
    sys.path.insert(0,str(root))
    from conv_rosa_transformer.rosa_layer import NBitQKVRosa
    from conv_rosa_transformer import rosa_core as core
    core._library()
    methods=args.methods or (['baseline_second'] if args.implementation=='baseline'
        else ['first_exact','first_random','second_exact','second_random'])
    before=hashes(root,core);records=[];begin=time.perf_counter()
    for task in args.tasks:
      for index,seed in enumerate(args.seeds):
        order=methods if index%2==0 else methods[::-1]
        for method in order:
            r=run(NBitQKVRosa,args.implementation,method,seed,task,args.steps,args.lr,args.examples)
            records.append(r)
            print(json.dumps({k:r[k] for k in ('method','seed','task','training_seconds','final')}),flush=True)
    after=hashes(root,core)
    out=dict(status='COMPLETE' if before==after else 'INVALID_SOURCE_CHANGED',scope=__doc__,
        implementation=args.implementation,implementation_root=str(root),torch_version=torch.__version__,
        platform=platform.platform(),affinity=sorted(os.sched_getaffinity(0)),source_before=before,
        source_after=after,source_unchanged=before==after,elapsed_seconds=time.perf_counter()-begin,records=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
