"""Run a paired NoMAD reconstruction/alignment sweep and select on validation only.

Example:
  python run_reconstruction_sweep.py --ckpt0 /content/drive/MyDrive/nomad/model_day0.pt \
      --target Jango_20150908=/content/drive/MyDrive/nomad/target_0908.npz \
      --out_dir /content/drive/MyDrive/nomad/reconstruction_sweep
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent


def run(cmd):
    print('+', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=ROOT, check=True)


def eval_force_r2(report_path):
    report = json.loads(Path(report_path).read_text())
    candidates = [r for r in report['results']
                  if r['method'] == 'Wiener on generator'
                  and r['split'].startswith('day-k ')
                  and ('unsupervised alignment' in r['split'])]
    if len(candidates) != 1:
        raise ValueError(f'Could not identify aligned generator force result in {report_path}')
    return float(candidates[0]['scores']['force'])


def run_eval(data, ckpt0, ckptk, split, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    run([sys.executable, ROOT / 'evaluate.py', '--data', data, '--ckpt0', ckpt0,
         '--ckptk', ckptk, '--split', split, '--out', out])
    return eval_force_r2(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt0', required=True, help='Frozen Day 0 model checkpoint')
    p.add_argument('--target', action='append', required=True, metavar='NAME=PATH',
                   help='Target-session NPZ; repeat once per session')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--weights', default='0,1,10,100',
                   help='Reconstruction weights with alignment weight 1')
    p.add_argument('--include_reconstruction_only', action='store_true',
                   help='Also run align_w=0, rec_w=10')
    p.add_argument('--seeds', default='0,1,2', help='Paired model seeds')
    p.add_argument('--split_seed', type=int, default=0,
                   help='Fixed target train/validation/test split shared by all runs')
    p.add_argument('--epochs', type=int, default=100,
                   help='Common optimizer update budget in epochs')
    p.add_argument('--bs', type=int, default=300)
    p.add_argument('--lr', type=float, default=4e-4)
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=0)
    p.add_argument('--hist', type=int, default=4)
    a = p.parse_args()

    weights = [float(x) for x in a.weights.split(',')]
    seeds = [int(x) for x in a.seeds.split(',')]
    if not weights or any(not math.isfinite(x) or x < 0 for x in weights):
        p.error('--weights must be a comma-separated list of finite nonnegative numbers')
    if len(set(seeds)) != len(seeds) or not seeds:
        p.error('--seeds must contain unique integer seeds')
    targets = []
    for spec in a.target:
        if '=' not in spec:
            p.error(f'Invalid target {spec!r}; expected NAME=PATH')
        name, path = spec.split('=', 1)
        if not name or not Path(path).is_file():
            p.error(f'Target name must be nonempty and file must exist: {spec}')
        targets.append((name, str(Path(path).resolve())))
    if not Path(a.ckpt0).is_file():
        p.error(f'Day 0 checkpoint does not exist: {a.ckpt0}')
    if a.epochs < 1:
        p.error('--epochs must be positive')

    conditions = [(1.0, w, f'align1_rec{w:g}') for w in weights]
    if a.include_reconstruction_only:
        conditions.append((0.0, 10.0, 'align0_rec10'))
    outroot = Path(a.out_dir).resolve()
    outroot.mkdir(parents=True, exist_ok=True)
    rows = []

    # Same seed and exact trial split are reused across all objective conditions.
    for session, data in targets:
        for seed in seeds:
            for align_w, rec_w, condition in conditions:
                run_dir = outroot / session / f'seed_{seed}' / condition
                ckpt_dir = run_dir / 'checkpoints'
                run([sys.executable, ROOT / 'train_phase2.py', '--data', data,
                     '--ckpt', Path(a.ckpt0).resolve(), '--out_dir', run_dir,
                     '--epochs', a.epochs, '--patience', a.epochs + 1,
                     '--fixed_epochs',
                     '--bs', a.bs, '--lr', a.lr, '--align_w', align_w, '--rec_w', rec_w,
                     '--seed', seed, '--target_split_seed', a.split_seed,
                     '--device', a.device, '--threads', a.threads, '--save_checkpoints'])
                for checkpoint in sorted(ckpt_dir.glob('epoch_*.pt')):
                    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
                    neural_score = float(state['metrics']['rec_unweighted'])
                    val_report = run_dir / 'validation' / f'{checkpoint.stem}.json'
                    behavior_score = run_eval(data, a.ckpt0, checkpoint, 'validation', val_report)
                    rows.append(dict(session=session, seed=seed, condition=condition,
                                     align_w=align_w, rec_w=rec_w,
                                     epoch=int(state['epoch']), checkpoint=str(checkpoint),
                                     validation_nll=neural_score,
                                     validation_objective=float(state['metrics']['total']),
                                     validation_force_r2=behavior_score))
                    del state

    # Select independently within each seed, then compare paired seed outcomes
    # within session. Test data is touched only for these choices.
    chosen = []
    for session, _ in targets:
        for seed in seeds:
            group = [r for r in rows if r['session'] == session and r['seed'] == seed]
            picks = [('neural_selected', min(group, key=lambda r: r['validation_nll'])),
                     ('behavior_selected', max(group, key=lambda r: r['validation_force_r2']))]
            default_rows = [r for r in group if r['align_w'] == 1 and r['rec_w'] == 10]
            if default_rows:
                picks.append(('nomad_default', min(default_rows, key=lambda r: r['validation_objective'])))
            for selection, picked in picks:
                test_report = outroot / session / 'selected_test' / f'seed_{seed}_{selection}.json'
                test_r2 = run_eval(dict(targets)[session], a.ckpt0, picked['checkpoint'], 'test', test_report)
                chosen.append(dict(session=session, seed=seed, selection=selection,
                                   selected_candidate=picked, test_force_r2=test_r2,
                                   test_report=str(test_report)))

    summary = []
    for session, _ in targets:
        vals = {name: [x['test_force_r2'] for x in chosen
                       if x['session'] == session and x['selection'] == name]
                for name in ('neural_selected', 'behavior_selected', 'nomad_default')}
        if not vals['nomad_default']:
            continue
        base = sum(vals['nomad_default']) / len(vals['nomad_default'])
        for name, scores in vals.items():
            if scores:
                mean = sum(scores) / len(scores)
                sd = (sum((x - mean) ** 2 for x in scores) / (len(scores) - 1)) ** 0.5 if len(scores) > 1 else None
                summary.append(dict(session=session, selection=name, mean_test_force_r2=mean,
                                    sample_sd_test_force_r2=sd,
                                    paired_difference_vs_default=mean - base))

    report = dict(protocol=dict(conditions=[dict(align_w=x, rec_w=y, name=n)
                                               for x, y, n in conditions],
                                seeds=seeds, split_seed=a.split_seed, epochs=a.epochs,
                                validation_selection_only=True,
                                test_scored_only_after_selection=True),
                   candidates=rows, selected_test=chosen, session_summary=summary)
    path = outroot / 'sweep_results.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Saved', path)


if __name__ == '__main__':
    main()
