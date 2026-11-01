"""Summarize the final frozen native stress runs without dropping failures."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'research/results'
NAMES=(
    'training_round_final_channels_prefix1101_second',
    'training_round_final_channels_prefix_fresh_second',
    'training_round_final_channels_prefix1101_first',
    'training_round_final_channels_prefix4096_second',
)


def main():
    rows=[];sources=[]
    for name in NAMES:
        path=RESULTS/(name+'.json');report=json.loads(path.read_text())
        assert report['status']=='COMPLETE'
        sources.append(dict(path=str(path.relative_to(ROOT)),runs=report['runs']))
        for run,comp in zip(report['records'],report['comparisons']):
            assert run['training_model_forwards']==run['steps']==run['training_backward_calls']
            assert run['spec']['method']==comp['method'] and run['seed']==comp['seed']
            rows.append(dict(task=run['task'],order=run['spec']['credit_order'],T=run['queries'],
                D=64,B=run['examples'],seed=run['seed'],steps=run['steps'],method=run['spec']['method'],
                seconds=run['training_seconds'],initial_loss=run['initial']['test']['loss'],
                final_loss=run['final']['test']['loss'],**comp['against_exact'],
                native_library=run['native_library'],source=str(path.relative_to(ROOT))))
    report=dict(status='COMPLETE',scope='ordinary full train-step structural stress',
        models=len(rows),updates=sum(r['steps'] for r in rows),sources=sources,rows=rows,
        final_native_hashes=sorted({r['native_library']['sha256'] for r in rows if r['method']!='row_phase1'}))
    assert len(report['final_native_hashes'])==1
    (RESULTS/'training_round_final_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Final native training checks','',
        'These are deliberately structured prefix-copy stress tasks. Each run uses B1, D64, Adam at learning rate 0.003 and 20 ordinary updates. The timer covers forward, loss, backward and Adam; data generation, evaluation, compilation and logging are excluded. Every update has exactly one model forward and one component backward. OpenBLAS, OMP, MKL and Torch threads were all limited to one, with process affinity CPU2.','',
        'The regression objective is the mean over rows of `0.5 * sum(error**2)`. Seed 1101 was used in discovery; seeds 2401 and 2402 were held out from it. Twenty updates test early training and the lifetime of the costly structure, not convergence or generalization to arbitrary networks.','',
        f"Final native SHA256: `{report['final_native_hashes'][0]}`. Every report below is marked `COMPLETE`; the historical row implementation has its own frozen binary. There are {report['models']} runs and {report['updates']} updates.",'',
        '| T | Rule | Seed | Method | Train-step seconds | Speedup vs exact | Initial loss | Final loss | First changed parameter update | First changed hard forward |',
        '| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['T']} | {r['order']} | {r['seed']} | {r['method']} | {r['seconds']:.6f} | {r['train_step_speedup']:.3f} | {r['initial_loss']:.7f} | {r['final_loss']:.7f} | {r['first_parameter_difference'] or 'none'} | {r['first_hard_forward_difference'] or 'none'} |")
    lines+=['',
        '`exact` is the default exact randomized mode. `final` is approximate mode with 32 row samples and full spectral channels. `final_channels` sets the optional channel budget to 8. `row_phase1` is the archived row-only implementation.','',
        'At T4096, the optional eight-channel mode took 1.53027 seconds against 19.5010 seconds for exact: 12.7436 times less complete train-step time. The 20 training losses and hard forwards matched exact on this run; its Q/K parameter updates did not. The same discovery seed at T2048 gave 8.2474 times less time. This is a size trend on a particular structural stress case, not a universal complexity benchmark.','',
        'Failures and regressions remain visible. With the first-order rule, all three final-runtime runs increased loss from about 27.06 to 43–44 under this learning-rate configuration. On second-order seed 2402, training made little progress, and eight-channel sampling was both slower and slightly worse in loss than full channels. Do not use the successful second-order seed to claim that first-order learning or every seed is equally reliable.','',
        'The broader 300-update tasks, equal-time comparisons, and nonlinear-stem failure are archived separately in `training_round_frozen_results.md`; they used the earlier frozen row runtime. The final spectral path was assessed here, and the unchanged row path had an additional 600-update equivalence check in `training_round_phase2_findings.md`.','',
        'Raw source and binary hashes, per-step timings, losses, hard-forward hashes, parameter hashes and native work counters remain in the JSON reports.']
    (RESULTS/'training_round_final_summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
