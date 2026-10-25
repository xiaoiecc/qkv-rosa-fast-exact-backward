"""Trace the worst held-out MSE seed to its first hard-bit disagreement.

This diagnostic preserves the already measured negative result. It runs the
same frozen/native training rules with equal initial weights and saved RNG
streams, recording numerical differences and exact proposal identities.
"""
from __future__ import annotations
import argparse,hashlib,importlib,json,os,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmarks.native_component import load_baseline,hashes
from conv_rosa_transformer import rosa_layer as native_layer
from conv_rosa_transformer import rosa_core as native_core
import native_random_toy as toy


def differences(a,b):
    result={}
    for name,x in a.items():
        x=x.detach();y=b[name].detach();delta=(x-y).abs();flat=int(delta.argmax()) if delta.numel() else 0
        result[name]=dict(maximum_absolute_difference=float(delta.max()) if delta.numel() else 0.,
            l2_difference=float(torch.linalg.vector_norm(x-y)),
            baseline_l2=float(torch.linalg.vector_norm(x)),
            worst_flat_index=flat,
            baseline_value=float(x.reshape(-1)[flat]) if x.numel() else None,
            native_value=float(y.reshape(-1)[flat]) if y.numel() else None)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',type=Path,default=ROOT.parent/'qkv-rosa-train')
    p.add_argument('--seed',type=int,default=903);p.add_argument('--steps',type=int,default=100)
    p.add_argument('--cpu',type=int,default=2)
    p.add_argument('--output',type=Path,default=ROOT/'research/results/native_random_roundoff_trace.json')
    a=p.parse_args()
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{a.cpu})
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    Frozen,frozen_core=load_baseline(a.baseline_root)
    frozen_joint=importlib.import_module(Frozen.__module__.rsplit('.',1)[0]+'.joint_credit')
    oldconfig=dict(repair_samples=None,joint_pairs=1,joint_rows=32,stochastic_values=False,
        trust_diagonal=.5,training_rule='hard_credit',joint_row_selection='active')
    torch.manual_seed(a.seed);old=toy.Retrieval(Frozen,oldconfig);old_rng=torch.get_rng_state().clone()
    torch.manual_seed(a.seed);new=toy.Retrieval(native_layer.NBitQKVRosa,dict(credit_order='second',randomized=False));new_rng=torch.get_rng_state().clone()
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    assert torch.equal(old_rng,new_rng)
    x,target=toy.make_data(a.seed,'retrieval',24)
    optimizers=[torch.optim.Adam(m.parameters(),lr=.03) for m in (old,new)]
    before=hashes(ROOT);oldbefore=hashes(a.baseline_root)
    records=[];first_bit_difference=None
    original_old_credit=frozen_core.quadratic_credits_numpy
    original_old_pair=frozen_joint.evaluate_pairs
    original_new_backward=native_layer.training_backward_numpy
    for step in range(a.steps+1):
        with torch.no_grad():
            logits=[{s:getattr(m.rosa,s.lower()+'_proj')(x) for s in ('Q','K','V')} for m in (old,new)]
        differing=[]
        for stream in ('Q','K','V'):
            positions=torch.nonzero((logits[0][stream]>0)!=(logits[1][stream]>0),as_tuple=False)
            for index in positions.tolist():
                loc=tuple(index);u=float(logits[0][stream][loc]);v=float(logits[1][stream][loc])
                differing.append(dict(stream=stream,batch=index[0],position=index[1],channel=index[2],
                    group=index[2]//4,bit=index[2]%4,baseline_logit=u,native_logit=v,
                    absolute_logit_difference=abs(u-v)))
        frozen_pairs={};stream_index=[-1]
        def old_credit(*args,**kwargs):
            stream_index[0]+=1
            return original_old_credit(*args,**kwargs)
        def old_pair(*args,**kwargs):
            result,work=original_old_pair(*args,**kwargs)
            def code(bit):return (0 if bit.stream=='K' else 9*4)+bit.position*4+bit.bit
            frozen_pairs[stream_index[0]]=dict(codes=[[code(r.left),code(r.right)] for r in result],
                costs=[[r.single_left,r.single_right,r.joint,r.interaction] for r in result])
            return result,work
        def new_backward(cache,g):return original_new_backward(cache,g,trace=True)
        parameter_difference=differences(dict(old.named_parameters()),dict(new.named_parameters()))
        prediction=[];gradient=[];losses=[];rngs=[]
        for which,(model,opt,rng) in enumerate(zip((old,new),optimizers,(old_rng,new_rng))):
            torch.set_rng_state(rng);model.train();opt.zero_grad(set_to_none=True)
            if which==0:
                with patch.object(frozen_core,'quadratic_credits_numpy',old_credit),patch.object(frozen_joint,'evaluate_pairs',old_pair):
                    pred=model(x);loss=toy.loss_value(pred,target,False);loss.backward()
            else:
                with patch.object(native_layer,'training_backward_numpy',new_backward):
                    pred=model(x);loss=toy.loss_value(pred,target,False);loss.backward()
            prediction.append(pred.detach());losses.append(float(loss.detach()))
            gradient.append({name:v.grad.detach().clone() for name,v in model.named_parameters()})
            rngs.append(torch.get_rng_state().clone())
        trace=new.rosa.last_forward_work['backward']
        codes=trace['pair_codes'];costs=trace['pair_costs'];pair_equal=True;cost_difference=0.
        for stream in range(48):
            nc=codes[stream];valid=nc[:,0]!=np.uint64(2**64-1)
            nc=nc[valid].tolist();ns=costs[stream][valid]
            prior=frozen_pairs.get(stream,dict(codes=[],costs=[]))
            pair_equal&=prior['codes']==nc
            if prior['codes']==nc and len(nc):cost_difference=max(cost_difference,float(np.max(np.abs(np.array(prior['costs'])-ns))))
        row=dict(step=step,hard_bit_differences=differing,
            forward_maximum_absolute_difference=float(torch.max(torch.abs(prediction[0]-prediction[1]))),
            loss=dict(baseline=losses[0],native=losses[1]),
            parameter_difference=parameter_difference,gradient_difference=differences(*gradient),
            selected_pair_identities_equal=bool(pair_equal),
            pair_cost_maximum_absolute_difference_when_same_pair=cost_difference,
            framework_rng_equal=torch.equal(*rngs))
        if step==0:
            row['initial_near_zero_adam_witnesses']=[dict(parameter=name,flat_index=index,
                baseline_gradient=float(gradient[0][name].reshape(-1)[index]),
                native_gradient=float(gradient[1][name].reshape(-1)[index]),
                adam_epsilon=1e-8,learning_rate=.03,
                baseline_first_adam_update=-.03*float(gradient[0][name].reshape(-1)[index])/(abs(float(gradient[0][name].reshape(-1)[index]))+1e-8),
                native_first_adam_update=-.03*float(gradient[1][name].reshape(-1)[index])/(abs(float(gradient[1][name].reshape(-1)[index]))+1e-8))
                for name,index in [('rosa.emb0',1),('rosa.v_proj.weight',38)]]
        records.append(row)
        if differing:
            first_bit_difference=step;break
        for optimizer in optimizers:optimizer.step()
        old_rng,new_rng=rngs
    assert before==hashes(ROOT) and oldbefore==hashes(a.baseline_root)
    out=dict(status='COMPLETE',scope=__doc__,seed=a.seed,first_hard_bit_divergence_before_update=first_bit_difference,
        source_unchanged=True,baseline_source_unchanged=True,source_sha256=before,baseline_source_sha256=oldbefore,
        previous_negative_results_unchanged=True,records=records)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
    last=records[-1]
    print(json.dumps(dict(seed=a.seed,first_hard_bit_divergence=first_bit_difference,
        differing=last['hard_bit_differences'],forward_maxdiff=last['forward_maximum_absolute_difference'],
        initial_gradient_maxdiff=max(v['maximum_absolute_difference'] for v in records[0]['gradient_difference'].values()),
        all_prior_pair_identities_equal=all(r['selected_pair_identities_equal'] for r in records[:-1]))))


if __name__=='__main__':main()
