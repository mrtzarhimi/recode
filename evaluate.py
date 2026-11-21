"""Trial-contained validation and cross-day comparisons using training metadata."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve

from dataloader import SEG_LEN, load_npz
from lfads import LFADS
from training_utils import model_fingerprint


NOTICE = ('Normalization-only transfer and full alignment use training-only day-k calibration; unchanged transfer '
          'retains day-0 normalization. This is calibrated transfer, not zero-shot. '
          'Decoder tuning uses day-0 training trials only. '
          'Test trials must not be used to tune models or pipeline settings.')

def r2vw(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if y.shape != prediction.shape or y.ndim < 2 or not y.size:
        raise ValueError('Targets and predictions must have matching nonempty shapes.')
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError('Nonfinite targets or predictions; refusing to drop samples selectively.')
    y = y.reshape(-1, y.shape[-1]).astype(np.float64)
    prediction = prediction.reshape(y.shape).astype(np.float64)
    if len(y) < 2:
        raise ValueError('R2 requires at least two observations.')
    error = ((prediction - y) ** 2).sum(0)
    variance = ((y - y.mean(0)) ** 2).sum(0)
    scores = np.ones_like(variance)
    variable = variance > 0
    scores[variable] = 1 - error[variable] / variance[variable]
    scores[~variable] = (error[~variable] == 0).astype(np.float64)
    if variable.any():
        return float(np.average(scores[variable], weights=variance[variable]))
    return float(scores.mean())


def read_checkpoint(path):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck.get('data', {}).get('version') != 6:
        raise ValueError('Regenerate data and retrain with pipeline version 6 for behavior anti-alias filtering.')
    if ck.get('model_id') != model_fingerprint(ck['cfg'], ck['state']):
        raise ValueError('Checkpoint model identity is invalid.')
    return ck


def model_from_checkpoint(ck, device):
    model = LFADS(**ck['cfg'])
    if ck.get('aligned', False):
        model.add_align()
    model.load_state_dict(ck['state'], strict=True)
    model.deterministic = True
    return model.to(device).eval()


def load_model(path, dev):
    return model_from_checkpoint(read_checkpoint(path), dev)


def validate_pair(base, aligned):
    if base.get('aligned') or not aligned.get('aligned'):
        raise ValueError('Expected a day-0 checkpoint and an aligned target checkpoint.')
    if base['cfg'] != aligned['cfg']:
        raise ValueError('Reference and target architectures differ.')
    if aligned.get('source_model_id') != base['model_id']:
        raise ValueError('Target alignment was trained from a different reference checkpoint.')
    m0, mk = base['data'], aligned['data']
    for key in ('version', 'split_seed', 'val_frac', 'test_frac', 'bin_size', 'n_force',
                'emg_names', 'units', 'selected_units', 'smooth_ms', 'corr_thresh',
                'behavior_limits', 'derivative_definition', 'behavior_mask_policy',
                'behavior_resampling', 'behavior_filter_order', 'behavior_filter_ripple_db',
                'behavior_filter_passband_nyquist_fraction', 'behavior_filter_padlen'):
        if m0[key] != mk[key]:
            raise ValueError(f'Checkpoint preprocessing mismatch: {key}.')
    for key in ('channel_idx', 'behavior_lo', 'behavior_span'):
        if not torch.equal(m0[key], mk[key]):
            raise ValueError(f'Checkpoint preprocessing mismatch: {key}.')
    for key, value in m0['day0'].items():
        same = torch.equal(value, mk['day0'][key]) if isinstance(value, torch.Tensor) else value == mk['day0'][key]
        if not same:
            raise ValueError(f'Reference preprocessing mismatch: {key}.')
    if 'dayk' not in mk:
        raise ValueError('Missing target-day metadata.')
    trainable = ('readin', 'align', 'fac', 'rate')
    for key, value in base['state'].items():
        if key.split('.')[0] not in trainable and not torch.equal(value, aligned['state'][key]):
            raise ValueError(f'Frozen parameter differs: {key}.')


def load_evaluation_data(path, base, aligned):
    reference = base['data']
    expected = aligned['data']['dayk']
    D = load_npz(path, seed=reference['split_seed'], val_frac=reference['val_frac'],
                 test_frac=reference['test_frac'], reference=reference,
                 target_seed=expected['split_seed'], materialize_segments=False)
    if 'dayk' not in D:
        raise ValueError('Evaluation requires a target day.')
    actual = D['metadata']['dayk']
    for key, value in expected.items():
        same = torch.equal(value, actual[key]) if isinstance(value, torch.Tensor) else value == actual[key]
        if not same:
            raise ValueError(f'Target data or preprocessing differs from alignment training: {key}.')
    for name in ('day0', 'dayk'):
        day = D[name]
        for key in ('x', 'y', 'b', 'mu', 'sd'):
            day[key] = day[key].cpu().numpy()
        day['bounds'] = day['trial_bounds']
    return D


def trial_index(onsets, bounds, trials, bin_size, hist, N=SEG_LEN):
    """Require the target window and every feature's full history inside its trial."""
    pre, post = int(round(0.25 / bin_size)), int(round(0.5 / bin_size))
    if min(pre, post, hist, N) < 1:
        raise ValueError('Bin size, history, and window lengths are incompatible.')
    onsets = np.asarray(onsets)
    if onsets.ndim != 1 or not np.issubdtype(onsets.dtype, np.integer):
        raise ValueError('Onsets must be a vector of integer bin indices.')
    owners = np.searchsorted(bounds[:, 0], onsets, side='right') - 1
    indices, ids = [], []
    allowed = set(np.asarray(trials).tolist())
    for onset, owner in zip(onsets, owners):
        if owner < 0 or owner not in allowed:
            continue
        start, end = bounds[owner]
        idx = onset + np.arange(-pre, post)
        if idx[0] - (hist - 1) - (N - 1) >= start and idx[-1] < end:
            indices.append(idx)
            ids.append(owner)
    return np.asarray(indices, dtype=np.int64).reshape(-1, pre + post), np.asarray(ids, dtype=np.int64)


def continuous_trial_index(bounds, trials, hist, N=SEG_LEN):
    """Use every eligible causal time bin, keeping model and decoder histories in-trial."""
    if min(hist, N) < 1:
        raise ValueError('History and model window lengths must be positive.')
    indices, ids = [], []
    for trial in np.asarray(trials, dtype=np.int64):
        start, end = bounds[trial]
        first = start + N + hist - 2
        if first >= end:
            continue
        times = np.arange(first, end, dtype=np.int64)
        indices.append(times[:, None])
        ids.append(np.full(len(times), trial, dtype=np.int64))
    if not indices:
        return np.empty((0, 1), dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(indices), np.concatenate(ids)


def feature_times(idx, hist):
    requested = idx[..., None] - np.arange(hist - 1, -1, -1)
    times, inverse = np.unique(requested, return_inverse=True)
    return times, inverse.reshape(requested.shape)


def validate_times(times, bounds, T, N):
    if np.asarray(times).ndim != 1 or not len(times):
        raise ValueError('Feature times must be nonempty.')
    owners = np.searchsorted(bounds[:, 0], times, side='right') - 1
    if (owners < 0).any() or (times >= T).any():
        raise ValueError('Feature time outside recording.')
    if (times - N + 1 < bounds[owners, 0]).any() or (times >= bounds[owners, 1]).any():
        raise ValueError('A feature window crosses a trial boundary.')


@torch.no_grad()
def causal_infer(model, d, times, N=SEG_LEN, bs=256, normalization_stats=None):
    validate_times(times, d['bounds'], len(d['x']), N)
    device = next(model.parameters()).device
    mu, sd = normalization_stats if normalization_stats is not None else (d['mu'], d['sd'])
    mu = torch.as_tensor(mu, device=device)
    sd = torch.as_tensor(sd, device=device)
    G = np.empty((len(times), model.gen.hidden_dim), np.float32)
    P = np.empty((len(times), model.beh.out_features), np.float32)
    model.eval()
    offsets = np.arange(1 - N, 1)
    for i in range(0, len(times), bs):
        indices = times[i:i + bs, None] + offsets
        raw = torch.from_numpy(d['x'][indices])
        if device.type == 'cuda':
            raw = raw.pin_memory()
        x = raw.to(device, non_blocking=device.type == 'cuda')
        out = model((x - mu) / sd, sample=False)
        n = len(x)
        G[i:i + n] = out['gen'][:, -1].cpu().numpy()
        P[i:i + n] = out['beh'][:, -1].cpu().numpy()
    return G, P


def spike_baseline(d, times, bin_size, smooth_ms=40.0, N=SEG_LEN, bs=256):
    validate_times(times, d['bounds'], len(d['x']), N)
    sigma = smooth_ms / (1000 * bin_size)
    lag = np.arange(N)
    weights = np.exp(-0.5 * (lag / sigma) ** 2) if sigma else (lag == 0).astype(float)
    weights /= weights.sum()
    result = np.empty((len(times), d['x'].shape[1]), np.float32)
    for i in range(0, len(times), bs):
        windows = d['x'][times[i:i + bs, None] - lag]
        result[i:i + bs] = (np.einsum('btc,t->bc', windows, weights) - d['mu']) / d['sd']
    return result


def wiener_features(values, lookup):
    gathered = values[lookup]
    return gathered.reshape(*lookup.shape[:2], -1)


def ridge_system(X, Y):
    X, Y = X.reshape(-1, X.shape[-1]).astype(np.float64), Y.reshape(-1, Y.shape[-1]).astype(np.float64)
    mean, scale, target_mean = X.mean(0), np.ones(X.shape[1]), Y.mean(0)
    Z = (X - mean) / scale
    return dict(mean=mean, scale=scale, target_mean=target_mean), Z.T @ Z, Z.T @ (Y - target_mean)


def solve_ridge(system, lam):
    decoder, gram, cross = system
    gram = gram.copy()
    gram.flat[::len(gram) + 1] += lam
    weights = cho_solve(cho_factor(gram, lower=True, check_finite=False), cross, check_finite=False)
    return dict(**decoder, weights=weights)


def fit_ridge(X, Y, lam):
    return solve_ridge(ridge_system(X, Y), lam)


def apply_w(decoder, X):
    return ((X - decoder['mean']) / decoder['scale']) @ decoder['weights'] + decoder['target_mean']


def behavior_units(prediction, lo, span):
    return (prediction + 1) * span / 2 + lo


def ridge_cv(X, Y, lams, groups, k=10, seed=0):
    """Tune only on day-0 training trials; fit scaling separately in each fold."""
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError('Nonfinite decoder training data.')
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError('At least two eligible day-0 training trials are needed for decoder tuning.')
    folds = np.array_split(np.random.default_rng(seed).permutation(unique), min(k, len(unique)))
    systems = []
    for fold in folds:
        validation = np.isin(groups, fold)
        systems.append((validation, ridge_system(X[~validation], Y[~validation])))
    best_score, best_lam = -np.inf, None
    for lam in lams:
        predictions = np.empty_like(Y, dtype=np.float64)
        for validation, system in systems:
            decoder = solve_ridge(system, float(lam))
            predictions[validation] = apply_w(decoder, X[validation])
        score = r2vw(Y, predictions)
        if score > best_score:
            best_score, best_lam = score, float(lam)
    return best_lam, fit_ridge(X, Y, best_lam)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--ckpt0', default='runs/model_day0.pt')
    p.add_argument('--ckptk', default='runs/model_dayk.pt')
    p.add_argument('--hist', type=int, default=4)
    p.add_argument('--window_mode', choices=('movement_onset', 'continuous'),
                   default='movement_onset',
                   help='Score movement windows or all valid causal bins contained within each trial.')
    p.add_argument('--split', choices=('test', 'validation'), default='test')
    p.add_argument('--bs', type=int, default=256)
    p.add_argument('--folds', type=int, default=10)
    p.add_argument('--seed', type=int, default=0, help='Decoder CV seed; checkpoint trial splits are unchanged')
    p.add_argument('--smooth_ms', type=float, default=40.0)
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=0)
    p.add_argument('--out', help='Optional JSON results path')
    a = p.parse_args()
    if min(a.hist, a.bs) < 1 or a.folds < 2 or a.threads < 0 or not math.isfinite(a.smooth_ms) or a.smooth_ms < 0:
        p.error('Require hist/bs >= 1, folds >= 2, threads >= 0, and finite smooth_ms >= 0.')
    if a.threads:
        torch.set_num_threads(a.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else torch.device(a.device)
    torch.manual_seed(a.seed)
    notice = NOTICE + ('Movement-onset selection is retrospective.' if a.window_mode == 'movement_onset'
                       else ' Scores use all behavior-valid causal bins inside each trial.')
    ck0, ckk = read_checkpoint(a.ckpt0), read_checkpoint(a.ckptk)
    validate_pair(ck0, ckk)
    D = load_evaluation_data(a.data, ck0, ckk)
    d0, dk = D['day0'], D['dayk']
    sections = {}
    split_key = 'te_trials' if a.split == 'test' else 'va_trials'
    for label, day, trials in (('train0', d0, d0['tr_trials']), ('eval0', d0, d0[split_key]), ('evalk', dk, dk[split_key])):
        trials = trials[day['behavior_valid_trials'][trials]]
        if not len(trials):
            raise ValueError(f'No inlier behavior trials remain in {label}.')
        if a.window_mode == 'movement_onset':
            idx, ids = trial_index(day['onset'], day['bounds'], trials, D['bin_size'], a.hist)
        else:
            idx, ids = continuous_trial_index(day['bounds'], trials, a.hist)
        if not len(idx):
            detail = ('Movement-onset windows require a valid go-cue movement onset and enough trial-contained history.'
                      if a.window_mode == 'movement_onset' else
                      'Continuous windows require at least one behavior-valid trial long enough for model and decoder history.')
            raise ValueError(f'No eligible {label} windows. {detail}')
        times, lookup = feature_times(idx, a.hist)
        sections[label] = dict(idx=idx, ids=ids, times=times, lookup=lookup, day=day)
        print(f'{label}: {len(np.unique(ids))}/{len(trials)} eligible trials, {len(idx)} {a.window_mode} windows')
    if len(np.unique(sections['train0']['ids'])) < 2:
        raise ValueError('Need at least two eligible day-0 training trials.')
    base = model_from_checkpoint(ck0, device)
    for section in sections.values():
        G, P = causal_infer(base, section['day'], section['times'], bs=a.bs)
        section['generator'] = wiener_features(G, section['lookup'])
        section['behavior'] = behavior_units(P[section['lookup'][..., 0]],
                                             D['behavior_lo'], D['behavior_span'])
        spikes = spike_baseline(section['day'], section['times'], D['bin_size'], a.smooth_ms, bs=a.bs)
        section['spikes'] = wiener_features(spikes, section['lookup'])
    target = sections['evalk']
    reference_norm = (d0['mu'], d0['sd'])
    G, P = causal_infer(base, target['day'], target['times'], bs=a.bs,
                        normalization_stats=reference_norm)
    target['day0_norm_generator'] = wiener_features(G, target['lookup'])
    target['day0_norm_behavior'] = behavior_units(P[target['lookup'][..., 0]],
                                                  D['behavior_lo'], D['behavior_span'])
    del base
    aligned = model_from_checkpoint(ckk, device)
    G, P = causal_infer(aligned, dk, target['times'], bs=a.bs)
    target['aligned_generator'] = wiener_features(G, target['lookup'])
    target['aligned_behavior'] = behavior_units(P[target['lookup'][..., 0]],
                                                D['behavior_lo'], D['behavior_span'])
    del aligned
    nf = D['n_force']
    targets = {'force': list(range(nf))}
    if D['emg_names']:
        targets['EMG'] = list(range(2 * nf, d0['y'].shape[1]))
    rows, lambdas = [], {}
    train, val0, valk = (sections[key] for key in ('train0', 'eval0', 'evalk'))
    lams = np.logspace(-5, 5, 20)
    alignment_label = 'supervised alignment' if ckk.get('args', {}).get('sup_w', 0) > 0 else 'unsupervised alignment'
    for feature, title in (('generator', 'Wiener on generator'), ('spikes', 'Wiener on causal spikes')):
        scores0, scores_unchanged, scores_norm, scoresa = {}, {}, {}, {}
        lambdas[feature] = {}
        for name, cols in targets.items():
            truth = lambda section: section['day']['b'][section['idx']][..., cols]
            lam, decoder = ridge_cv(train[feature], truth(train), lams, train['ids'], a.folds, a.seed)
            lambdas[feature][name] = lam
            scores0[name] = r2vw(truth(val0), apply_w(decoder, val0[feature]))
            scores_norm[name] = r2vw(truth(valk), apply_w(decoder, valk[feature]))
            if feature == 'generator':
                scores_unchanged[name] = r2vw(
                    truth(valk), apply_w(decoder, valk['day0_norm_generator']))
                scoresa[name] = r2vw(truth(valk), apply_w(decoder, valk['aligned_generator']))
        if feature == 'generator':
            rows.extend([dict(method=title, split=f'day-0 {a.split}', scores=scores0),
                         dict(method=title, split=f'day-k {a.split}, unchanged Day-0 transfer', scores=scores_unchanged),
                         dict(method=title, split=f'day-k {a.split}, normalization-only', scores=scores_norm),
                         dict(method=title, split=f'day-k {a.split}, {alignment_label}', scores=scoresa)])
        else:
            rows.append(dict(method=title, split=f'day-k {a.split}, normalization-only', scores=scores_norm))
    for section, feature, label in ((val0, 'behavior', f'day-0 {a.split}'),
                                     (valk, 'day0_norm_behavior', f'day-k {a.split}, unchanged Day-0 transfer'),
                                     (valk, 'behavior', f'day-k {a.split}, normalization-only'),
                                     (valk, 'aligned_behavior', f'day-k {a.split}, {alignment_label}')):
        scores = {name: r2vw(section['day']['b'][section['idx']][..., cols], section[feature][..., cols])
                  for name, cols in targets.items()}
        rows.append(dict(method='LFADS behavior readout', split=label, scores=scores))
    print('\n' + notice)
    if a.split == 'validation':
        print('Validation influenced checkpoint selection; these are not independent test estimates.')
    print('Targets: raw force and EMG units, without clipping; identical bins for every method.')
    print(f'Window mode: {a.window_mode}. Baseline: one-sided Gaussian, sigma={a.smooth_ms:g} ms, limited to {SEG_LEN} bins; model history={SEG_LEN} bins; Wiener history={a.hist} bins.')
    print(f'\n{"method":26s} {"split":30s}' + ''.join(f'{name + " R2":>12s}' for name in targets))
    for row in rows:
        print(f'{row["method"]:26s} {row["split"]:30s}' + ''.join(f'{row["scores"][name]:12.4f}' for name in targets))
    if a.out:
        report = dict(notice=notice, args=vars(a), reference_sha256=ck0['data']['day0']['sha256'],
                      target_sha256=ckk['data']['dayk']['sha256'], reference_model_id=ck0['model_id'],
                      target_model_id=ckk['model_id'], split=a.split,
                      reference_epoch=ck0['epoch'], alignment_epoch=ckk['epoch'],
                      alignment_retained_initial=ckk['epoch'] < 0,
                      trial_counts={key: len(np.unique(value['ids'])) for key, value in sections.items()},
                      trial_ids={key: np.unique(value['ids']).tolist() for key, value in sections.items()},
                      target_mode='raw physical units, unclipped', alignment=alignment_label,
                      lambdas=lambdas, results=rows)
        path = Path(a.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print('saved', path)


if __name__ == '__main__':
    main()
