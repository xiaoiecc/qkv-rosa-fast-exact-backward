"""Exact acceleration must not perturb a normal model's Dropout RNG/updates."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from conv_rosa_transformer.rosa_layer import NBitQKVRosa
from conv_rosa_transformer import rosa_core as core


def digest(tensors):
    h=hashlib.sha256()
    for x in tensors:h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def hashes():
    paths=sorted((ROOT/'conv_rosa_transformer').glob('*.py'))
    paths+=sorted((ROOT/'conv_rosa_transformer/csrc').glob('*.h'))
    paths+=sorted((ROOT/'conv_rosa_transformer/csrc').glob('*.cpp'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu',type=int,default=2)
    p.add_argument('--output',type=Path,default=ROOT/'research/results/native_random_mode_contract.json')
    a=p.parse_args()
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{a.cpu})
    torch.set_num_threads(1);torch.set_num_interop_threads(1);core._library()
    before=hashes();rows=[]
    for order in ('first','second'):
      for t,d in ((9,4),(65,16)):
        torch.manual_seed(472901+d)
        off=nn.Sequential(NBitQKVRosa(2*d,d,credit_order=order,randomized=False),
                          nn.Dropout(.3),nn.Linear(2*d,7))
        on=nn.Sequential(NBitQKVRosa(2*d,d,credit_order=order,randomized=True),
                         nn.Dropout(.3),nn.Linear(2*d,7))
        on.load_state_dict(off.state_dict(),strict=True)
        x=torch.randn(2,t,2*d);target=torch.randn(2,t,7)
        optimizers=[torch.optim.AdamW(m.parameters(),lr=.003) for m in (off,on)]
        updates=[]
        for step in range(3):
            states=[]
            for model,opt in zip((off,on),optimizers):
                torch.manual_seed(77721+step)
                model.train();opt.zero_grad(set_to_none=True)
                prediction=model(x);loss=.5*(prediction-target).square().mean()
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                gradient=digest([p.grad for p in model.parameters()])
                opt.step()
                states.append(dict(prediction_sha256=digest([prediction]),gradient_sha256=gradient,
                    parameter_sha256=digest(model.state_dict().values()),rng_sha256=digest([torch.get_rng_state()])))
            assert states[0]==states[1],(order,t,d,step,states)
            updates.append(states[0])
        off.eval();on.eval();rng=torch.get_rng_state().clone()
        with torch.no_grad():assert torch.equal(off(x),on(x))
        assert torch.equal(rng,torch.get_rng_state())
        rows.append(dict(credit_order=order,T=t,D=d,B=2,G=2,updates=updates,
            all_training_outputs_gradients_updates_rng_bitwise_equal=True,evaluation_rng_unchanged=True))
    after=hashes();assert before==after
    report=dict(status='PASS',scope=__doc__,torch_version=torch.__version__,source_sha256=before,
        source_unchanged=True,loaded_native_sha256=hashlib.sha256(Path(core._library()._name).read_bytes()).hexdigest(),rows=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',cases=len(rows),updates_per_case=3,output=str(a.output))))


if __name__=='__main__':main()
