"""Sequential ordinary-model comparison across frozen native runtime packages.

Each method gets a fresh Python process and one native library. Parameters,
data, the outer training loop and optimizer are paired. The timer remains the
complete ordinary train step; cache hashing is outside it and never performs
an extra model evaluation.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def first_difference(a,b):
    return next((i+1 for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase1-root',type=Path,default=Path('/tmp/qkv-rosa-row-phase1'))
    p.add_argument('--final-root',type=Path,default=ROOT)
    p.add_argument('--tasks',nargs='+',default=['prefix_copy64_fresh'])
    p.add_argument('--seeds',type=int,nargs='+',default=[1101])
    p.add_argument('--steps',type=int,default=20);p.add_argument('--queries',type=int,default=2048)
    p.add_argument('--examples',type=int,default=1);p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--rows',type=int,default=32);p.add_argument('--cpu',type=int,default=2)
    p.add_argument('--channels',type=int,default=8,
        help='Optional channel budget, used only by the final_channels method.')
    p.add_argument('--order',choices=['first','second'],default='second')
    p.add_argument('--methods',nargs='+',default=['exact','row_phase1','final'])
    p.add_argument('--equal-time',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    folder=a.output.with_suffix('');folder.mkdir(exist_ok=True)
    records=[];runs=[];comparisons=[]

    def execute(method,task,seed,protocol='equal_steps',seconds=None):
        runtime=a.phase1_root if method=='row_phase1' else a.final_root
        config={} if method=='exact' else dict(backward_mode='approximate',approximation_rows=a.rows)
        if method=='final_channels':config['approximation_channels']=a.channels
        spec=dict(method=method,credit_order=a.order,layer_config=config)
        name=f'{task}_{a.order}_s{seed}_{method}_{protocol}'
        specfile=folder/(name+'_spec.json');resultfile=folder/(name+'.json');logfile=folder/(name+'.log')
        specfile.write_text(json.dumps([spec],indent=2)+'\n')
        command=[sys.executable,str(ROOT/'research/training_round_study.py'),
            '--specs',str(specfile.resolve()),'--tasks',task,'--seeds',str(seed),
            '--steps',str(a.steps if seconds is None else max(10000,a.steps)),
            '--queries',str(a.queries),'--examples',str(a.examples),'--lr',str(a.lr),
            '--cpu',str(a.cpu),'--output',str(resultfile.resolve())]
        if seconds is not None:command+=['--seconds',str(seconds)]
        environment=dict(os.environ,QKV_STUDY_ROOT=str(runtime.resolve()))
        environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with logfile.open('w') as log:
            subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads(resultfile.read_text());assert len(report['records'])==1
        row=report['records'][0];row['protocol']=protocol
        row.update(runtime=str(runtime.resolve()),native_library=report['native_library'],run_report=str(resultfile))
        records.append(row);runs.append(dict(path=str(resultfile),status=report['status'],
            implementation_root=report['implementation_root'],native_library=report['native_library'],
            source_before=report['source_before'],source_after=report['source_after'],
            research_source_hashes=report['research_source_hashes']))
        print(task,seed,method,protocol,'steps',row['steps'],'seconds',row['training_seconds'],
            'loss',row['final']['test']['loss'],report['status'],flush=True)
        return row

    for task in a.tasks:
      for index,seed in enumerate(a.seeds):
        fixed={}
        for method in (a.methods if index%2==0 else a.methods[::-1]):
            fixed[method]=execute(method,task,seed)
        baseline=fixed['exact']
        for method,row in fixed.items():
            assert row['initial_state_sha256']==baseline['initial_state_sha256']
            assert row['input_sha256']==baseline['input_sha256']
            def compare(other,label):
                n=min(row['steps'],other['steps'])
                return dict(reference=label,first_parameter_difference=first_difference(
                        row['every_update_parameter_sha256'][:n],other['every_update_parameter_sha256'][:n]),
                    first_hard_forward_difference=first_difference(row['every_training_hard_forward_sha256'][:n],
                        other['every_training_hard_forward_sha256'][:n]),
                    parameter_updates_equal=sum(x==y for x,y in zip(row['every_update_parameter_sha256'][:n],other['every_update_parameter_sha256'][:n])),
                    hard_forwards_equal=sum(x==y for x,y in zip(row['every_training_hard_forward_sha256'][:n],other['every_training_hard_forward_sha256'][:n])),
                    common_steps=n,maximum_batch_loss_difference=max(abs(x-y) for x,y in zip(row['batch_losses'][:n],other['batch_losses'][:n])),
                    heldout_loss_difference=row['final']['test']['loss']-other['final']['test']['loss'],
                    train_step_speedup=other['training_seconds']/row['training_seconds'])
            comparison=dict(task=task,seed=seed,method=method,against_exact=compare(baseline,'exact'))
            if 'row_phase1' in fixed:comparison['against_phase1']=compare(fixed['row_phase1'],'row_phase1')
            comparisons.append(comparison)
        if a.equal_time:
            for method in a.methods:
                if method!='exact':execute(method,task,seed,'equal_time',baseline['training_seconds'])
        # Checkpoint every complete paired case, retaining any source changes.
        report=dict(status='COMPLETE' if all(r['status']=='COMPLETE' for r in runs) else 'SOURCE_CHANGED',
            thread_controls=dict(OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,
                torch_intraop_threads=1,torch_interop_threads=1,cpu=a.cpu),
            phase1_source_commit='5e3e9f81965242492ae5a3870efd3cfbbca4bf9e',scope=__doc__,runs=runs,
            records=records,comparisons=comparisons)
        a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
