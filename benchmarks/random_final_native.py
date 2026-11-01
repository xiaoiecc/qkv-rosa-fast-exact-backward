"""Final paired native backward study: online, exact, row sampling, auto contraction.

Timers include validation/allocation, metric/proposals, all credit work, joint
replays, logit mapping and V/embedding gradients. They exclude projections,
model loss/optimizer and compilation. Captures contain actual model adjoints.
"""
from pathlib import Path
import argparse,hashlib,importlib,json,os,platform,statistics,sys,time,types
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from conv_rosa_transformer import native_training as final
from benchmarks.random_round_native import hashes,quality,summary
from benchmarks.randomized_backward import make_input


def load(root,name):
    package=types.ModuleType(name);package.__path__=[str(root/'conv_rosa_transformer')]
    sys.modules[name]=package
    return importlib.import_module(name+'.native_training')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--online-root',type=Path,default=ROOT.parent/'qkv-rosa-random-baseline')
    p.add_argument('--phase1-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu',type=int,default=8);p.add_argument('--repeats',type=int,default=9)
    a=p.parse_args();os.sched_setaffinity(0,{a.cpu})
    online=load(a.online_root,'final_online');phase1=load(a.phase1_root,'final_phase1')
    modules={'online':online,'phase1':phase1,'final':final}
    roots={'online':a.online_root,'phase1':a.phase1_root,'final':ROOT}
    libraries={key:api._library() for key,api in modules.items()}
    before={key:hashes(root) for key,root in roots.items()}
    cases=[]
    specs=[('random',128,8),('random',8192,8),('random',8192,16),('wide',8192,64),
           ('narrow',8192,64),('all_same',8192,8),('periodic',4096,4),
           ('shadow',8192,1),('shadow',2048,64),('sparse_periodic',4096,4)]
    for index,(family,n,d) in enumerate(specs):
        q,k,v,g,h,e0,e1=make_input('periodic' if family=='sparse_periodic' else family,n,d,2710930+index)
        if family=='sparse_periodic':g[:-1]=0
        bits=np.arange(d,dtype=np.uint64)
        arrays=[((2*((word[:,None]>>bits)&1).astype(float)-1)*.35)[None,None] for word in (q,k)]
        arrays+=[((2*v.astype(float)-1)*.35)[None,None],e0[None],e1[None]]
        cases.append((family,arrays,g[None,None],None))
    for folder_name in ('training_round_captures_context512','training_round_captures_prefix4096',
                        'training_round_captures_prefix64_2048'):
        folder=ROOT/'research/results'/folder_name
        meta=json.loads((folder/'index.json').read_text())['rows'][0]
        z=dict(np.load(folder/(meta['name']+'.npz')))
        cases.append((meta['task'],[z[k] for k in ('q_logits','k_logits','v_logits','emb0','emb1')],
                      z['grad_y'],meta))
    records=[]
    for name,arrays,g,metadata in cases:
      b,groups,n,d=g.shape
      for order in ('first','second'):
        methods={'online_exact':(online,None,0),'exact':(final,None,0),
                 'phase1_rows32':(phase1,32,0),'final32':(final,32,0),'final_channels8':(final,32,8)}
        caches={};samples={key:[] for key in methods};errors={key:[] for key in methods}
        for key,(api,budget,channels) in methods.items():
            options=dict(credit_order=order,randomized=True,proposal_seed=719,repair_seed=817)
            if budget is not None:
                options.update(backward_mode='approximate',approximation_rows=budget)
                if api is final:options['approximation_channels']=channels
            y,cache,_=api.training_forward_numpy(*arrays,**options);caches[key]=cache
            api.training_backward_numpy(cache,g)
        reference=final.training_backward_numpy(caches['exact'],g,trace=True)
        old=online.training_backward_numpy(caches['online_exact'],g,trace=True)
        for x,y in zip(reference[:5],old[:5]):assert x.tobytes()==y.tobytes(),(name,order,'exact parity')
        for key in ('pair_codes','pair_costs'):assert reference[-1][key].tobytes()==old[-1][key].tobytes()
        counters={};pair_equal={};pair_close={};keys=list(methods)
        for repeat in range(a.repeats):
          for key in (keys if repeat%2==0 else keys[::-1]):
            api,budget,channels=methods[key];cache=caches[key]
            if budget is not None:cache.approximation.seed=91117+repeat*104729
            start=time.perf_counter_ns();result=api.training_backward_numpy(cache,g,trace=True)
            samples[key].append((time.perf_counter_ns()-start)/1e6)
            errors[key].append(quality(reference,result))
            counters[key]={k:v for k,v in result[-1].items() if isinstance(v,(str,int,float,bool))}
            pair_equal[key]=bool(np.array_equal(reference[-1]['pair_codes'],result[-1]['pair_codes']))
            pair_close[key]=bool(np.allclose(reference[-1]['pair_costs'],result[-1]['pair_costs'],rtol=1e-12,atol=1e-12,equal_nan=True))
        timings={key:summary(values) for key,values in samples.items()}
        med={key:value['median_ms'] for key,value in timings.items()}
        row=dict(name=name,B=b,G=groups,T=n,D=d,order=order,capture=metadata,
                 backward=timings,errors=errors,counters=counters,pair_codes_equal=pair_equal,
                 pair_costs_close=pair_close,exact_bytewise_equal=True,
                 final_speedup_vs_exact=med['exact']/med['final32'],
                 final_speedup_vs_phase1=med['phase1_rows32']/med['final32'],
                 channels8_speedup_vs_exact=med['exact']/med['final_channels8'],
                 channels8_speedup_vs_full_fft=med['final32']/med['final_channels8'],
                 exact_speedup_vs_online=med['online_exact']/med['exact'])
        records.append(row)
        print(json.dumps(dict(name=name,T=n,D=d,order=order,ms=med,
            final_speedup=row['final_speedup_vs_exact'],improvement=row['final_speedup_vs_phase1'],
            relative_l2=statistics.median(z['qk_relative_l2'] for z in errors['final32']),
            spectral_groups=counters['final32']['approx_spectral_groups'])),flush=True)
    after={key:hashes(root) for key,root in roots.items()}
    result=dict(status='PASS' if before==after else 'SOURCE_CHANGED',scope=__doc__,
                source_before=before,source_after=after,source_unchanged=before==after,
                online_revision='b510fd9479946f10e46ced1a6265695cbc131d0a',
                phase1_revision='5e3e9f81965242492ae5a3870efd3cfbbca4bf9e',
                libraries={k:dict(path=v._name,sha256=hashlib.sha256(Path(v._name).read_bytes()).hexdigest()) for k,v in libraries.items()},
                numpy=np.__version__,platform=platform.platform(),repeats=a.repeats,
                affinity=sorted(os.sched_getaffinity(0)),rows=records)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
