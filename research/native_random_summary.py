"""Summarize the frozen/native study without dropping negative seeds."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];RESULTS=ROOT/'research/results'


def summarize():
    paths=[RESULTS/'native_random_training_baseline.json',RESULTS/'native_random_training_final.json']
    raw=[json.loads(p.read_text()) for p in paths]
    assert all(x['source_unchanged'] for x in raw)
    baseline={(r['task'],r['seed']):r for r in raw[0]['records']}
    native={(r['task'],r['seed'],r['method']):r for r in raw[1]['records']}
    comparisons=[];pairs=[]
    for key,r in native.items():
        b=baseline[key[:2]]
        assert r['initial_state_sha256']==b['initial_state_sha256']
        assert r['data_sha256']==b['data_sha256']
        divergence=next((i+1 for i,(x,y) in enumerate(zip(b['every_update_parameter_sha256'],r['every_update_parameter_sha256'])) if x!=y),None)
        comparisons.append(dict(task=r['task'],seed=r['seed'],method=r['method'],
            initial_state_equal=True,data_equal=True,first_parameter_hash_divergence_vs_frozen=divergence,
            loss_vs_frozen=dict(baseline=b['final']['loss'],native=r['final']['loss']),
            test_loss_vs_frozen=dict(baseline=b['final']['test_loss'],native=r['final']['test_loss']),
            runtime_speedup_vs_frozen=b['training_seconds']/r['training_seconds']))
        if r['method'].endswith('_exact'):
            on=native.get((r['task'],r['seed'],r['method'].replace('_exact','_random')))
            if on:
                same_parameters=r['every_update_parameter_sha256']==on['every_update_parameter_sha256']
                same_losses=r['training_batch_losses']==on['training_batch_losses']
                same_rng=r['final_rng_sha256']==on['final_rng_sha256']
                assert same_parameters and same_losses and same_rng
                pairs.append(dict(task=r['task'],seed=r['seed'],credit_order=r['method'].split('_')[0],
                    every_update_parameters_bitwise_equal=same_parameters,training_losses_bitwise_equal=same_losses,
                    final_rng_bitwise_equal=same_rng,updates_compared=len(r['every_update_parameter_sha256']),
                    exact_training_seconds=r['training_seconds'],random_training_seconds=on['training_seconds'],
                    random_speedup=r['training_seconds']/on['training_seconds']))
    summaries=[]
    records=raw[0]['records']+raw[1]['records']
    for task,method in sorted({(r['task'],r['method']) for r in records}):
        rr=[r for r in records if r['task']==task and r['method']==method]
        row=dict(task=task,method=method,seeds=[r['seed'] for r in rr],n=len(rr),
            mean_training_seconds=float(np.mean([r['training_seconds'] for r in rr])),
            training_seconds_per_seed=[r['training_seconds'] for r in rr])
        for metric in ('loss','test_loss','bit_accuracy','test_bit_accuracy','test_payload_accuracy'):
            values=[r['final'][metric] for r in rr]
            row[metric]=dict(mean=float(np.mean(values)),median=float(np.median(values)),
                sample_std=float(np.std(values,ddof=1)) if len(values)>1 else 0.,per_seed=values)
        summaries.append(row)
    out=dict(status='PASS',scope=__doc__,raw_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        all_sources_unchanged=True,initialization_and_data_equal=True,
        one_model_forward_and_backward_per_step=all(r['steps']==r['training_model_forwards']==r['training_backward_calls'] for r in records),
        all_random_on_off_parameter_trajectories_bitwise_equal=all(r['every_update_parameters_bitwise_equal'] for r in pairs),
        note='Native-vs-frozen arithmetic may differ by rounding; numerical/quality differences are retained. Exact random acceleration itself must preserve every update.',
        summaries=summaries,random_on_off_pairs=pairs,native_vs_frozen_per_seed=comparisons)
    path=RESULTS/'native_random_training_summary.json';path.write_text(json.dumps(out,indent=2)+'\n')
    for row in summaries:print(row['task'],row['method'],row['loss']['mean'],row['test_loss']['mean'],row['mean_training_seconds'])
    print('all random on/off trajectories identical',out['all_random_on_off_parameter_trajectories_bitwise_equal'])
    return out


if __name__=='__main__':summarize()
