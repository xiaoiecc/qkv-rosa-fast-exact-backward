"""Paired complete native backward costs and approximation errors.

Includes NumPy validation/allocation, native local metric, proposals, complete
control, certificates, sampled row corrections, exact joint corner replays,
logit mapping, V and embedding gradients. Excludes projections, the downstream
model/loss/optimizer, compilation and the separately reported hard forward.
"""
from pathlib import Path
import argparse,hashlib,importlib,json,os,platform,statistics,sys,time,types
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from conv_rosa_transformer import native_training as current
from benchmarks.randomized_backward import make_input


def hashes(root):
    paths=sorted((root/'conv_rosa_transformer').glob('*.py'))+sorted((root/'conv_rosa_transformer/csrc').glob('*'))
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def baseline(root):
    name='round_online_baseline';package=types.ModuleType(name)
    package.__path__=[str(root/'conv_rosa_transformer')];sys.modules[name]=package
    return importlib.import_module(name+'.native_training')


def quality(reference,result):
    x=np.concatenate([a.ravel() for a in reference[:2]]);y=np.concatenate([a.ravel() for a in result[:2]])
    nx,ny=np.linalg.norm(x),np.linalg.norm(y);err=np.linalg.norm(y-x)
    return dict(qk_relative_l2=float(err/nx) if nx else (0. if not err else None),
        qk_absolute_l2=float(err),qk_cosine=float(x@y/(nx*ny)) if nx and ny else float(not nx and not ny),
        value_bitwise_equal=reference[2].tobytes()==result[2].tobytes(),
        embedding_bitwise_equal=all(reference[i].tobytes()==result[i].tobytes() for i in (3,4)),
        q_relative_l2=float(np.linalg.norm(reference[0]-result[0])/max(np.linalg.norm(reference[0]),np.finfo(float).tiny)),
        k_relative_l2=float(np.linalg.norm(reference[1]-result[1])/max(np.linalg.norm(reference[1]),np.finfo(float).tiny)))


def summary(values):
    return dict(median_ms=float(np.median(values)),q25_ms=float(np.quantile(values,.25)),
        q75_ms=float(np.quantile(values,.75)),samples_ms=values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',type=Path,default=ROOT.parent/'qkv-rosa-random-baseline')
    p.add_argument('--cpu',type=int,default=8);p.add_argument('--repeats',type=int,default=9)
    p.add_argument('--budgets',type=int,nargs='+',default=[8,32,128])
    p.add_argument('--quick',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();os.sched_setaffinity(0,{a.cpu});old=baseline(a.baseline_root)
    current._library();old._library();before=hashes(ROOT);old_before=hashes(a.baseline_root)
    cases=[('random',128,8),('random',1024,8),('random',8192,8),('random',8192,16),
        ('wide',8192,64),('narrow',8192,64),('all_same',8192,8),('periodic',4096,4),
        ('shadow',8192,1),('shadow',2048,64),('sparse_shadow',8192,1)]
    if a.quick:cases=[('random',128,8),('wide',1024,64),('shadow',1024,1)]
    records=[]
    for case,(family,n,d) in enumerate(cases):
      q,k,v,g,h,e0,e1=make_input('shadow' if family=='sparse_shadow' else family,n,d,2710930+case)
      if family=='sparse_shadow':g[:-1]=0
      shifts=np.arange(d,dtype=np.uint64)
      qz=(2*((q[:,None]>>shifts)&np.uint64(1)).astype(np.float64)-1)*.35
      kz=(2*((k[:,None]>>shifts)&np.uint64(1)).astype(np.float64)-1)*.35
      vz=(2*v.astype(np.float64)-1)*.35
      arrays=[z[None,None] for z in (qz,kz,vz)]+[e0[None],e1[None]];gg=g[None,None]
      digest=hashlib.sha256()
      for array in arrays+[gg]:digest.update(array.tobytes())
      for order in ('first','second'):
        methods={'online_off':(old,False,None),'online_on':(old,True,None),
            'exact_off':(current,False,None),'exact':(current,True,None)}
        methods.update({f'approx_{budget}':(current,True,budget) for budget in a.budgets})
        caches={};outputs={};forward={};samples={name:[] for name in methods};errors={name:[] for name in methods}
        for name,(api,randomized,budget) in methods.items():
            options=dict(credit_order=order,randomized=randomized,proposal_seed=817,repair_seed=919)
            if budget is not None:options.update(backward_mode='approximate',approximation_rows=budget)
            start=time.perf_counter_ns();y,cache,counts=api.training_forward_numpy(*arrays,**options)
            forward[name]=(time.perf_counter_ns()-start)/1e6;caches[name]=cache;outputs[name]=y
            api.training_backward_numpy(cache,gg)
        reference=current.training_backward_numpy(caches['exact'],gg)
        for name in ('online_off','online_on','exact_off'):
            result=methods[name][0].training_backward_numpy(caches[name],gg)
            assert outputs[name].tobytes()==outputs['exact'].tobytes(),(family,n,d,order,name,'forward')
            for x,y in zip(result[:5],reference[:5]):assert x.tobytes()==y.tobytes(),(family,n,d,order,name)
        names=list(methods);last={}
        for rep in range(a.repeats):
          order_names=names if rep%2==0 else names[::-1]
          for name in order_names:
            api,randomized,budget=methods[name];cache=caches[name]
            if budget is not None:cache.approximation.seed=87178291199+rep*104729
            start=time.perf_counter_ns();result=api.training_backward_numpy(cache,gg)
            samples[name].append((time.perf_counter_ns()-start)/1e6)
            if budget is not None:errors[name].append(quality(reference,result))
            last[name]={key:value for key,value in result[-1].items() if isinstance(value,(str,bool,int,float))}
        med={name:summary(values) for name,values in samples.items()}
        row=dict(family=family,T=n,D=d,order=order,B=1,G=1,input_sha256=digest.hexdigest(),
            backward=med,speedup_vs_current_exact={name:med['exact']['median_ms']/value['median_ms'] for name,value in med.items()},
            current_exact_speedup_vs_online=med['online_on']['median_ms']/med['exact']['median_ms'],
            forward_setup_ms=forward,errors=errors,last_counters=last,all_exact_paths_bytewise_equal=True)
        records.append(row)
        print(json.dumps(dict(family=family,T=n,D=d,order=order,
            median_ms={name:value['median_ms'] for name,value in med.items()},
            speedup=row['speedup_vs_current_exact'],approx32_error=errors.get('approx_32',[]))),flush=True)
    after=hashes(ROOT);old_after=hashes(a.baseline_root)
    report=dict(status='PASS' if before==after and old_before==old_after else 'SOURCE_CHANGED',
        scope=__doc__,source_before=before,source_after=after,baseline_source_before=old_before,
        baseline_source_after=old_after,source_unchanged=before==after and old_before==old_after,
        baseline_revision='b510fd9479946f10e46ced1a6265695cbc131d0a',numpy=np.__version__,
        platform=platform.platform(),affinity=sorted(os.sched_getaffinity(0)),repeats=a.repeats,
        cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown'),
        compiler=os.popen('c++ --version').read().splitlines()[0],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=records)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
