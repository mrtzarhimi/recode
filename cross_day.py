"""Train one reference model, align multiple targets, and compare on reserved trials."""
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from dataloader import load_npz
from evaluate import read_checkpoint


def run(script, args):
    command = [sys.executable, str(Path(__file__).with_name(script)), *map(str, args)]
    subprocess.run(command, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', help='Reference-only archive or a pair archive with the same reference')
    p.add_argument('--targets', nargs='+', required=True, help='Pair archives, each containing the same day 0 and a different target')
    p.add_argument('--ckpt0', help='Reuse a version-5 reference checkpoint instead of training it')
    p.add_argument('--out_dir', default='runs/cross_day')
    p.add_argument('--epochs0', type=int, default=10000)
    p.add_argument('--epochsk', type=int, default=10000)
    p.add_argument('--bs0', type=int, default=1000)
    p.add_argument('--bsk', type=int, default=300)
    p.add_argument('--eval_bs', type=int, default=256)
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=0)
    p.add_argument('--amp', choices=('off', 'bf16'), default='off')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split_seed', type=int, default=None)
    p.add_argument('--val_frac', type=float, default=None)
    p.add_argument('--test_frac', type=float, default=None)
    p.add_argument('--kl_warmup', type=int, default=100)
    p.add_argument('--kl_start', type=int, default=0)
    p.add_argument('--l2_warmup', type=int, default=100)
    p.add_argument('--l2_start', type=int, default=0)
    p.add_argument('--beh_warmup', type=int, default=100)
    p.add_argument('--beh_start', type=int, default=0)
    p.add_argument('--nll_warmupk', type=int, default=100)
    p.add_argument('--nll_startk', type=int, default=0)
    p.add_argument('--kl_warmupk', type=int, default=10)
    p.add_argument('--kl_startk', type=int, default=0)
    p.add_argument('--hist', type=int, default=4)
    p.add_argument('--folds', type=int, default=10)
    p.add_argument('--smooth_ms', type=float, default=40.0)
    p.add_argument('--split', choices=('test', 'validation'), default='test')
    for name in ('readin_dim', 'enc_dim', 'ic_dim', 'ci_dim', 'con_dim', 'co_dim', 'gen_dim', 'fac_dim'):
        p.add_argument('--' + name, type=int, default=None)
    for name in ('dropout_rate', 'cd_rate', 'cd_pass_rate', 'ic_prior_var', 'co_prior_tau',
                 'co_prior_nvar', 'state_clip'):
        p.add_argument('--' + name, type=float, default=None)
    for name in ('controller_lag', 'align_dim'):
        p.add_argument('--' + name, type=int, default=None)
    a = p.parse_args()
    if not a.reference and not a.ckpt0:
        p.error('Provide --reference to train day 0, or --ckpt0 to reuse it.')
    if min(a.epochs0, a.epochsk, a.bs0, a.bsk, a.eval_bs, a.hist) <= 0 or a.folds < 2 or a.threads < 0:
        p.error('Invalid epoch, batch, history, fold, or thread setting.')
    root = Path(a.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    reference_path = a.reference or a.targets[0]
    ck = read_checkpoint(a.ckpt0) if a.ckpt0 else None
    if ck is not None and any(getattr(a, name) is not None for name in
            ('readin_dim', 'enc_dim', 'ic_dim', 'ci_dim', 'con_dim', 'co_dim', 'gen_dim', 'fac_dim',
             'dropout_rate', 'cd_rate', 'cd_pass_rate', 'ic_prior_var', 'co_prior_tau', 'co_prior_nvar',
             'state_clip', 'controller_lag', 'align_dim')):
        p.error('Model options cannot override an existing --ckpt0 architecture.')
    if ck is not None and ck.get('aligned'):
        p.error('--ckpt0 must be an unaligned reference checkpoint.')
    saved = ck['data'] if ck else None
    seed = (saved['split_seed'] if saved else a.seed) if a.split_seed is None else a.split_seed
    val = (saved['val_frac'] if saved else 0.2) if a.val_frac is None else a.val_frac
    test = (saved['test_frac'] if saved else 0.2) if a.test_frac is None else a.test_frac
    ref = load_npz(reference_path, seed=seed, val_frac=val, test_frac=test,
                   reference=saved, load_target=False, materialize_segments=False)
    reference_meta = ref['metadata']
    del ref
    prepared, seen = [], set()
    for path in a.targets:
        day = load_npz(path, seed=seed, val_frac=val, test_frac=test,
                       reference=reference_meta, materialize_segments=False)
        if 'dayk' not in day:
            p.error(f'Target archive has no target recording: {path}')
        meta = day['metadata']['dayk']
        if meta['sha256'] in seen or meta['sha256'] == reference_meta['day0']['sha256']:
            p.error(f'Duplicate target or target identical to reference: {path}')
        seen.add(meta['sha256'])
        label = re.sub(r'[^A-Za-z0-9_.-]+', '_', Path(meta['id']).stem or Path(path).stem)
        target_dir = root / (label + '_' + meta['sha256'][:10])
        target_seed = (a.seed + int(hashlib.sha256(meta['id'].encode()).hexdigest()[:8], 16)) % (2 ** 32)
        prepared.append((path, target_dir, target_seed, meta['id']))
        del day
    common = ['--eval_bs', a.eval_bs, '--device', a.device,
              '--threads', a.threads, '--amp', a.amp, '--seed', a.seed]
    if ck is None:
        model_args = []
        for name in ('readin_dim', 'enc_dim', 'ic_dim', 'ci_dim', 'con_dim', 'co_dim',
                     'gen_dim', 'fac_dim', 'dropout_rate', 'cd_rate', 'cd_pass_rate', 'ic_prior_var',
                     'co_prior_tau', 'co_prior_nvar', 'state_clip', 'controller_lag', 'align_dim'):
            if getattr(a, name) is not None:
                model_args.extend(['--' + name, getattr(a, name)])
        ref_dir = root / 'reference'
        run('train_phase1.py', ['--data', reference_path, '--out_dir', ref_dir,
            '--epochs', a.epochs0, '--bs', a.bs0, '--kl_warmup', a.kl_warmup,
            '--kl_start', a.kl_start, '--l2_warmup', a.l2_warmup,
            '--l2_start', a.l2_start,
            '--beh_warmup', a.beh_warmup, '--beh_start', a.beh_start,
            '--split_seed', seed, '--val_frac', val, '--test_frac', test, *common, *model_args])
        ckpt0 = ref_dir / 'model_day0.pt'
    else:
        ckpt0 = Path(a.ckpt0).resolve()
    reports, table = [], []
    for path, target_dir, target_seed, label in prepared:
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f'\nAligning {label}: independent target split seed={target_seed}', flush=True)
        run('train_phase2.py', ['--data', path, '--ckpt', ckpt0, '--out_dir', target_dir,
            '--epochs', a.epochsk, '--bs', a.bsk, '--target_split_seed', target_seed,
            '--nll_warmup', a.nll_warmupk, '--nll_start', a.nll_startk,
            '--kl_warmup', a.kl_warmupk, '--kl_start', a.kl_startk, *common])
        report_path = target_dir / 'evaluation.json'
        run('evaluate.py', ['--data', path, '--ckpt0', ckpt0,
            '--ckptk', target_dir / 'model_dayk.pt', '--bs', a.eval_bs,
            '--device', a.device, '--threads', a.threads, '--hist', a.hist,
            '--folds', a.folds, '--smooth_ms', a.smooth_ms, '--seed', a.seed,
            '--split', a.split, '--out', report_path])
        report = json.loads(report_path.read_text())
        reports.append(dict(target=label, archive=str(Path(path).resolve()), report=report))
        for row in report['results']:
            table.append(dict(target=label, method=row['method'], split=row['split'], **row['scores']))
        summary = dict(reference_checkpoint=str(ckpt0), split=a.split, days=reports,
                       note='Separate target models share one reference; no target behavior labels train the default alignment or decoder.')
        temporary = root / 'cross_day_summary.json.tmp'
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        temporary.replace(root / 'cross_day_summary.json')
    metric_names = sorted({k for row in table for k in row} - {'target', 'method', 'split'})
    with (root / 'cross_day_scores.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['target', 'method', 'split', *metric_names])
        writer.writeheader()
        writer.writerows(table)
    print('\nsaved', root / 'cross_day_summary.json')
    print('saved', root / 'cross_day_scores.csv')


if __name__ == '__main__':
    main()
