"""Continuous-session spike preprocessing and LFADS data loading."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import cheby1, sosfiltfilt

SEG_LEN, OVERLAP = 30, 6  # 600 ms segments, 120 ms overlap at 20 ms bins
BEHAVIOR_LIMITS = np.asarray([[-1500, 800], [-1100, 600]], dtype=np.float32)
DERIVATIVE_DEFINITION = 'backward_difference_per_second_reset_at_trial_owned_interval_start'
BEHAVIOR_MASK_POLICY = 'force_bounds_and_difference_dependency_per_bin_whole_force_trial_evaluation'
BEHAVIOR_RESAMPLING = 'native_cheby1_sos_zero_phase_then_decimate_per_trial_owned_interval'
BEHAVIOR_FILTER_ORDER = 500
BEHAVIOR_FILTER_RIPPLE_DB = 0.05
BEHAVIOR_FILTER_PASSBAND_NYQUIST_FRACTION = 0.8
BEHAVIOR_FILTER_PADLEN = 500


def _str(r):
    r = np.asarray(r).ravel()[0]
    return r.decode() if isinstance(r, bytes) else str(r)


def _need(d, names):
    miss = [n for n in names if not hasattr(d, n)]
    if miss:
        raise AttributeError(f'xds object has no {miss}. Available: {[a for a in dir(d) if not a.startswith("_")]}')


def _matlab_string(value):
    codes = np.asarray(value).ravel()
    if codes.dtype.kind == 'U':
        return ''.join(codes.tolist()).rstrip('\x00')
    if codes.dtype.kind == 'S':
        return b''.join(codes.tolist()).decode('utf-8').rstrip('\x00')
    return codes.astype('<u2').tobytes().decode('utf-16-le').rstrip('\x00')


def _parse_xds_hdf5(base_path, filename):
    import h5py

    with h5py.File(Path(base_path) / filename, 'r') as f:
        x = f['xds']
        array = lambda key: np.asarray(x[key])
        scalar = lambda key: array(key).item()

        def cells(key):
            return [np.asarray(f[ref]) for ref in array(key).ravel()]

        parsed = {key: scalar(key) for key in ('has_EMG', 'has_force', 'sorted', 'bin_width')}
        for key in ('has_kin', 'has_cursor'):
            parsed[key] = scalar(key) if key in x else int('curs_p' in x or 'kin_p' in x)
        for key in ('has_raw_EMG', 'has_raw_force'):
            parsed[key] = scalar(key) if key in x else 0
        parsed.update(spikes=[value.T for value in cells('spikes')],
                      spike_counts=array('spike_counts').T,
                      unit_names=[_matlab_string(value) for value in cells('unit_names')],
                      time_frame=array('time_frame').ravel())
        for key in ('thresholds', 'trial_start_time', 'trial_end_time', 'trial_target_dir'):
            if key in x:
                parsed[key] = array(key).ravel()
        for key in ('EMG', 'force', 'raw_EMG', 'raw_force', 'trial_target_corners'):
            if key in x:
                parsed[key] = array(key).T
        for key in ('raw_EMG_time_frame', 'raw_force_time_frame'):
            if key in x:
                parsed[key] = array(key).ravel()
        if 'EMG_names' in x:
            parsed['EMG_names'] = [_matlab_string(value) for value in cells('EMG_names')]
        if 'spike_waveforms' in x:
            parsed['spike_waveforms'] = [value.T for value in cells('spike_waveforms')]
        prefix = 'curs' if 'curs_p' in x else 'kin'
        for suffix in ('p', 'v', 'a'):
            key = f'{prefix}_{suffix}'
            if key in x:
                parsed[f'curs_{suffix}'] = array(key).T
        go = array('trial_gocue_time')
        parsed['trial_gocue_time'] = go.ravel() if 1 in go.shape else go.T
        parsed['trial_result'] = np.asarray(list(_matlab_string(array('trial_result'))))
        parsed['trial_info_table_header'] = [_matlab_string(value) for value in cells('trial_info_table_header')]
        table = array('trial_info_table')
        parsed['trial_info_table'] = [[np.asarray(f[ref]) for ref in row] for row in table]
        parsed.setdefault('trial_target_corners', np.empty((len(parsed['trial_start_time']), 0)))
        meta = {}
        text_keys = {'ranBy', 'monkey', 'array', 'dateTime', 'processedTime', 'rawFileName', 'cdsName', 'task'}
        for key, dataset in x['meta'].items():
            if not isinstance(dataset, h5py.Dataset):
                continue
            value = np.asarray(dataset)
            meta[key] = _matlab_string(value) if key in text_keys else (value.item() if value.size == 1 else value)
        if 'rawFileName' not in meta and 'cdsName' in meta:
            meta['rawFileName'] = meta['cdsName']
        parsed['meta'] = meta
    return parsed


def _bin_xds(d, bin_size, use_emg):
    ratio = bin_size / float(d.bin_width)
    if ratio < 1 or not np.isclose(ratio, round(ratio)):
        raise ValueError('bin_size must be an integer multiple of the native XDS bin_width.')
    times, counts = d.bin_spikes(bin_size, mode='left')
    times = np.asarray(times).ravel()
    streams = [d.resample_force(bin_size)]
    if use_emg:
        streams.append(d.resample_EMG(bin_size))
    keep = np.ones(len(times), bool)
    aligned = []
    for sample_times, values in streams:
        sample_times = np.asarray(sample_times).ravel()
        values = np.asarray(values)
        if not len(sample_times) or len(sample_times) != len(values):
            raise ValueError('XDS returned inconsistent behavior sample times.')
        right = np.searchsorted(sample_times, times).clip(0, len(sample_times) - 1)
        left = np.maximum(right - 1, 0)
        ids = np.where(abs(sample_times[left] - times) < abs(sample_times[right] - times), left, right)
        keep &= np.isclose(sample_times[ids], times, rtol=0, atol=max(1e-9, bin_size * 1e-6))
        aligned.append(values[ids])
    if keep.sum() < 2:
        raise ValueError('Neural and behavior timestamps have insufficient common bins.')
    if not np.allclose(np.diff(times[keep]), bin_size, rtol=0, atol=max(1e-9, bin_size * 1e-6)):
        raise ValueError('Common timestamps have internal gaps; continuous windows require a uniform time grid.')
    force = aligned[0][keep].astype(np.float32)
    emg = aligned[1][keep].astype(np.float32) if use_emg else np.empty((keep.sum(), 0), np.float32)
    return times[keep], np.asarray(counts, np.float32)[keep], force, emg


def antialias_resample(values, native_times, target_times, intervals, native_bin, target_bin):
    """Filter native continuous signals independently inside owned intervals."""
    values = np.asarray(values, np.float64)
    native_times = np.asarray(native_times, np.float64).ravel()
    target_times = np.asarray(target_times, np.float64).ravel()
    if values.ndim != 2 or values.shape[0] != len(native_times):
        raise ValueError('Native behavior values and timestamps must have matching (time, channel) shapes.')
    intervals = validate_bounds(intervals, len(target_times))
    if len(intervals) == 0 or not len(target_times):
        raise ValueError('Anti-alias filtering requires nonempty target data and trial bounds.')
    if not np.isfinite(values).all():
        raise ValueError('Native behavior values must be finite before anti-alias filtering.')
    tolerance = max(1e-9, native_bin * 1e-6)
    if (not np.isfinite(native_times).all() or not np.isfinite(target_times).all() or
            np.any(np.diff(native_times) <= 0) or np.any(np.diff(target_times) <= 0)):
        raise ValueError('Behavior timestamps must be finite and strictly increasing.')
    if not np.allclose(np.diff(native_times), native_bin, rtol=0, atol=tolerance):
        raise ValueError('Native behavior timestamps must form a uniform grid for anti-alias filtering.')
    ratio = target_bin / native_bin
    if ratio < 1 or not np.isclose(ratio, round(ratio)):
        raise ValueError('target_bin must be an integer multiple of native_bin.')
    ratio = int(round(ratio))
    result = np.empty((len(target_times), values.shape[1]), np.float32)
    starts = np.r_[0, intervals[1:, 0], len(target_times)]
    sos = None
    if ratio > 1:
        cutoff = BEHAVIOR_FILTER_PASSBAND_NYQUIST_FRACTION / ratio
        sos = cheby1(BEHAVIOR_FILTER_ORDER, BEHAVIOR_FILTER_RIPPLE_DB,
                     cutoff, output='sos')
    for left, right in zip(starts[:-1], starts[1:]):
        t0, t1 = target_times[left], target_times[right - 1] + target_bin
        raw_lo = int(np.searchsorted(native_times, t0 - tolerance, side='left'))
        raw_hi = int(np.searchsorted(native_times, t1 - tolerance, side='left'))
        source = values[raw_lo:raw_hi]
        if sos is not None:
            if len(source) <= BEHAVIOR_FILTER_PADLEN:
                raise ValueError('Owned behavior interval is too short for the order-500 zero-phase anti-alias filter.')
            source = sosfiltfilt(sos, source, axis=0, padlen=BEHAVIOR_FILTER_PADLEN)
        sample_idx = np.rint((target_times[left:right] - native_times[raw_lo]) / native_bin).astype(int)
        if (sample_idx < 0).any() or (sample_idx >= len(source)).any():
            raise ValueError('Target behavior timestamps fall outside their native owned interval.')
        if not np.allclose(native_times[raw_lo + sample_idx], target_times[left:right],
                           rtol=0, atol=tolerance):
            raise ValueError('Native and target behavior timestamps do not share an exact decimation grid.')
        result[left:right] = source[sample_idx].astype(np.float32)
    return result


def _load_day(xds_path, data_dir, fname, bin_size, use_emg=True):
    root = Path(xds_path).resolve()
    if (root / 'xds_python' / 'xds.py').is_file():
        root = root / 'xds_python'
    if not (root / 'xds.py').is_file():
        raise FileNotFoundError(f'--xds_path must point to the limblab/xds repository or its xds_python directory: {root}')
    sys.path.insert(0, str(root))
    try:
        from xds import lab_data
    finally:
        sys.path.pop(0)
    module = sys.modules[lab_data.__module__]
    if Path(module.__file__).resolve().parent != root:
        raise ImportError(f'A different xds module is already loaded from {module.__file__}. Restart Python with the requested XDS path.')
    namespace = lab_data.parse_file.__globals__
    original_parser = namespace['parse_h5py']
    namespace['parse_h5py'] = _parse_xds_hdf5
    try:
        d = lab_data(str(Path(data_dir)), fname)
    finally:
        namespace['parse_h5py'] = original_parser
    _need(d, ['time_frame', 'spike_counts', 'force', 'unit_names',
              'trial_gocue_time', 'trial_result', 'trial_start_time', 'trial_end_time'])
    if use_emg:
        _need(d, ['EMG', 'EMG_names'])
    t, spikes, _, _ = _bin_xds(d, bin_size, use_emg)
    bounds = trial_bounds(t, d.trial_start_time, d.trial_end_time)
    force = antialias_resample(d.force, d.time_frame, t, bounds, d.bin_width, bin_size)
    emg = (antialias_resample(d.EMG, d.time_frame, t, bounds, d.bin_width, bin_size)
           if use_emg else np.empty((len(t), 0), np.float32))
    return dict(t=t, spikes=spikes, units=[str(n) for n in d.unit_names], force=force, emg=emg,
                emg_names=[str(n) for n in d.EMG_names] if use_emg else [],
                gocue=np.asarray(d.trial_gocue_time, np.float64).ravel(),
                result=[_str(r) for r in np.asarray(d.trial_result, dtype=object).ravel()],
                bounds=bounds, native_bin=float(d.bin_width))


def _match_units(spikes, names, ref_names):
    idx = {n: i for i, n in enumerate(names)}
    out = np.zeros((len(spikes), len(ref_names)), np.float32)
    for j, n in enumerate(ref_names):
        if n in idx:
            out[:, j] = spikes[:, idx[n]]
    return out


def trial_bounds(t, starts, ends):
    t = np.asarray(t).ravel()
    starts, ends = np.asarray(starts).ravel(), np.asarray(ends).ravel()
    if len(starts) != len(ends) or not len(starts):
        raise ValueError('Trial starts and ends must be nonempty and paired.')
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Bin times must be finite and strictly increasing.')
    if not np.isfinite(starts).all() or not np.isfinite(ends).all():
        raise ValueError('Trial boundaries must be finite.')
    bounds = np.column_stack([np.searchsorted(t, starts), np.searchsorted(t, ends)])
    return validate_bounds(bounds, len(t))


def validate_bounds(bounds, T):
    bounds = np.asarray(bounds)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or not len(bounds):
        raise ValueError('Expected nonempty trial bounds with shape (n, 2).')
    if not np.issubdtype(bounds.dtype, np.integer):
        raise ValueError('Trial bounds must be integer bin indices.')
    if (bounds[:, 0] < 0).any() or (bounds[:, 1] > T).any() or (np.diff(bounds, axis=1) <= 0).any():
        raise ValueError('Trial bounds must satisfy 0 <= start < end <= T.')
    if (bounds[1:, 0] < bounds[:-1, 1]).any():
        raise ValueError('Trials must be ordered and nonoverlapping; ends are exclusive.')
    return bounds.astype(np.int64)


def smooth_trials(x, bounds, sigma):
    if sigma < 0 or not np.isfinite(sigma):
        raise ValueError('Smoothing sigma must be finite and nonnegative.')
    out = np.full(x.shape, np.nan, np.float64)
    for lo, hi in validate_bounds(bounds, len(x)):
        out[lo:hi] = gaussian_filter1d(x[lo:hi].astype(np.float64), sigma, axis=0, mode='reflect') if sigma else x[lo:hi]
    return out


def trial_gradient(x, bounds):
    out = np.zeros_like(x)
    for lo, hi in bounds:
        if hi - lo > 1:
            out[lo + 1:hi] = np.diff(x[lo:hi], axis=0)
    return out


def force_derivative(force, bounds, bin_size):
    """Per-second differences within independently owned continuous intervals."""
    if not np.isfinite(bin_size) or bin_size <= 0:
        raise ValueError('bin_size must be finite and positive.')
    bounds = validate_bounds(bounds, len(force))
    starts = np.r_[0, bounds[1:, 0], len(force)]
    return trial_gradient(force, np.column_stack((starts[:-1], starts[1:]))) / bin_size


def behavior_masks(behavior, bounds):
    bounds = validate_bounds(bounds, len(behavior))
    force = behavior[:, :2]
    force_valid = ((force >= BEHAVIOR_LIMITS[:, 0]) & (force <= BEHAVIOR_LIMITS[:, 1])).all(1)
    valid = np.isfinite(behavior).all(1) & force_valid
    previous_valid = np.r_[True, force_valid[:-1]]
    previous_valid[np.r_[0, bounds[1:, 0]]] = True
    trials = np.asarray([valid[lo:hi].all() for lo, hi in bounds], dtype=bool)
    return valid & previous_valid, trials


def movement_onsets(force, t, gocue, result, bin_size, bounds, tol_bins=2):
    if len(gocue) != len(bounds) or len(result) != len(bounds):
        raise ValueError('Trial boundaries, go cues and results must align.')
    sm = smooth_trials(force, bounds, 1.0)
    speed = np.linalg.norm(trial_gradient(sm, bounds), axis=1)
    pre, post = int(round(0.25 / bin_size)), int(round(0.75 / bin_size))
    onsets = []
    for (start, end), g, r in zip(bounds, gocue, result):
        if r != 'R' or not np.isfinite(g):
            continue
        gi = int(np.searchsorted(t, g))
        lo, hi = max(start, gi - pre), min(end, gi + post)
        if not start <= gi < end or hi <= lo:
            continue
        pk = lo + int(np.argmax(speed[lo:hi]))
        if speed[pk] <= 0:
            continue
        thr = 0.2 * speed[pk]
        b = pk
        while b > lo and speed[b] >= thr:
            b -= 1
        f = gi
        while f < hi and speed[f] < thr:
            f += 1
        if f < hi and abs(f - b) <= tol_bins and b >= gi:
            onsets.append(b)
    return np.asarray(onsets, np.int64)


def _day_arrays(d, units, emg_keep, bin_size, use_emg):
    x = _match_units(d['spikes'], d['units'], units)
    force, bounds = d['force'], d['bounds']
    cols = [force, force_derivative(force, bounds, bin_size)]
    if use_emg:
        cols.append(d['emg'][:, [d['emg_names'].index(e) for e in emg_keep]])
    b = np.concatenate(cols, 1).astype(np.float32)
    onsets = movement_onsets(force, d['t'], d['gocue'], d['result'], bin_size, bounds)
    return dict(x=x, b=b, onsets=onsets, bounds=bounds, time=np.asarray(d['t'], np.float64),
                native_bin=d['native_bin'])


def prepare(xds_path, data_dir, day0, dayk, bin_size, corr_thresh, use_emg, out):
    d0 = _load_day(xds_path, data_dir, day0, bin_size, use_emg)
    dk = _load_day(xds_path, data_dir, dayk, bin_size, use_emg) if dayk else None
    units = d0['units']
    emg_keep = d0['emg_names'] if use_emg else []
    if dk is not None and any(e not in dk['emg_names'] for e in emg_keep):
        raise ValueError('Target EMG channels must contain the reference EMG schema. Use --no_emg for all days if unavailable.')
    D0 = _day_arrays(d0, units, emg_keep, bin_size, use_emg)
    DK = _day_arrays(dk, units, emg_keep, bin_size, use_emg) if dk else None
    if DK is not None:
        DK['present'] = np.asarray([u in dk['units'] for u in units], bool)
    finalize(out, D0, DK, bin_size, d0['force'].shape[1], emg_keep,
             units=units, corr_thresh=corr_thresh, reference_id=day0, target_id=dayk or '')


def finalize(out, D0, DK, bin_size, n_force, emg_names, smooth_ms=20.0,
             units=None, corr_thresh=0.0, reference_id='day0', target_id='dayk'):
    if not np.isfinite(bin_size) or bin_size <= 0 or not np.isfinite(smooth_ms) or smooth_ms < 0:
        raise ValueError('Require positive bin_size and nonnegative smooth_ms.')
    if not 0 <= corr_thresh <= 1:
        raise ValueError('corr_thresh must lie in [0,1]; 0 disables training-only filtering.')
    units = list(units) if units is not None else [f'unit_{i}' for i in range(D0['x'].shape[1])]
    if len(units) != D0['x'].shape[1] or len(set(units)) != len(units):
        raise ValueError('Reference unit names must be unique and match the spike dimension.')
    if n_force != 2:
        raise ValueError('The pinned NoMAD monkey force outlier rules require two force dimensions.')
    arrays = {}
    for tag, D in (('0', D0), ('k', DK)):
        if D is None:
            continue
        bounds = validate_bounds(D['bounds'], len(D['x']))
        if D['b'].shape[1] < 4:
            raise ValueError('Behavior data must include two force and two force-derivative dimensions.')
        valid_bins, valid_trials = behavior_masks(D['b'], bounds)
        time = np.asarray(D.get('time', np.arange(len(D['x'])) * bin_size), np.float64)
        if (time.shape != (len(D['x']),) or not np.isfinite(time).all() or
                not np.allclose(np.diff(time), bin_size, rtol=0, atol=max(1e-9, bin_size * 1e-6))):
            raise ValueError('Each day must have finite timestamps on a uniform bin_size time grid.')
        arrays.update({f'x_{tag}': D['x'].astype(np.float32), f'b_{tag}': D['b'].astype(np.float32),
                       f'onset_{tag}': np.asarray(D['onsets'], np.int64), f'trial_bounds_{tag}': bounds,
                       f'time_{tag}': time,
                       f'behavior_native_bin_{tag}': float(D.get('native_bin', bin_size)),
                       f'behavior_valid_{tag}': valid_bins,
                       f'behavior_valid_trials_{tag}': valid_trials,
                       f'present_{tag}': np.asarray(D.get('present', np.ones(len(units), bool)), bool)})
        print(f'day{tag}: behavior-valid bins {valid_bins.sum()}/{len(valid_bins)}, '
              f'evaluation-inlier trials {valid_trials.sum()}/{len(valid_trials)}')
    np.savez_compressed(out, format_version=6, bin_size=bin_size, smooth_ms=smooth_ms,
                        corr_thresh=corr_thresh, units=np.asarray(units, dtype=str),
                        behavior_limits=BEHAVIOR_LIMITS,
                        derivative_definition=DERIVATIVE_DEFINITION,
                        behavior_mask_policy=BEHAVIOR_MASK_POLICY,
                        behavior_resampling=BEHAVIOR_RESAMPLING,
                        behavior_filter_order=BEHAVIOR_FILTER_ORDER,
                        behavior_filter_ripple_db=BEHAVIOR_FILTER_RIPPLE_DB,
                        behavior_filter_passband_nyquist_fraction=BEHAVIOR_FILTER_PASSBAND_NYQUIST_FRACTION,
                        behavior_filter_padlen=BEHAVIOR_FILTER_PADLEN,
                        reference_id=reference_id, target_id=target_id, n_force=n_force,
                        emg_names=np.asarray(emg_names, dtype=str), **arrays)
    print('saved', out)


def make_synthetic(out, n_trials=400, trial_len=75, C=96, seed=0, bin_size=0.02):
    rng = np.random.default_rng(seed)
    K = 6
    A = np.zeros((K, K))
    for i in range(0, K, 2):
        th = 0.15 * (1 + i / 2)
        A[i:i + 2, i:i + 2] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
    Wn = rng.normal(0, 0.6, (K, C))
    Wb = rng.normal(0, 0.5, (K, 5))  # 2 force + 3 EMG

    def day(Wmix, gain, dead):
        T = n_trials * trial_len
        z = np.zeros((T, K))
        for i in range(n_trials):
            s = i * trial_len
            z[s] = rng.normal(0, 1, K)
            for t in range(1, trial_len):
                z[s + t] = z[s + t - 1] @ A.T
        rate = np.exp(-1.5 + (z @ Wn) @ Wmix * gain)
        rate[:, dead] = 0
        x = rng.poisson(rate).astype(np.float32)
        beh = z @ Wb
        force, emg = beh[:, :2], beh[:, 2:]
        bounds = np.column_stack([np.arange(n_trials) * trial_len, (np.arange(n_trials) + 1) * trial_len])
        b = np.concatenate([force, force_derivative(force, bounds, bin_size), emg], 1).astype(np.float32)
        return dict(x=x, b=b, onsets=np.arange(n_trials) * trial_len + int(round(0.25 / bin_size)) + SEG_LEN + 2,
                    bounds=bounds, time=np.arange(T, dtype=np.float64) * bin_size)

    D0 = day(np.eye(C), 1.0, [])
    Q = np.eye(C) + 0.3 * rng.normal(0, 1, (C, C)) / np.sqrt(C)
    DK = day(Q, 1.2, list(range(0, C, 17)))
    finalize(out, D0, DK, bin_size, 2, ['emg1', 'emg2', 'emg3'])


def segment_index(T, seg_len=SEG_LEN, overlap=OVERLAP):
    if seg_len <= 0 or not 0 <= overlap < seg_len:
        raise ValueError('Require seg_len > 0 and 0 <= overlap < seg_len.')
    starts = np.arange(0, T - seg_len + 1, seg_len - overlap)
    return starts[:, None] + np.arange(seg_len)[None, :]


def trial_segments(bounds, seg_len=SEG_LEN, overlap=OVERLAP):
    indices, owners = [], []
    for trial, (lo, hi) in enumerate(bounds):
        idx = segment_index(hi - lo, seg_len, overlap) + lo
        indices.append(idx)
        owners.append(np.full(len(idx), trial, np.int64))
    return np.concatenate(indices), np.concatenate(owners)


def session_segments(T, bounds, seg_len=SEG_LEN, overlap=OVERLAP):
    """Chop continuous data; assign each intertrial gap to its preceding trial."""
    bounds = validate_bounds(bounds, T)
    if seg_len <= 0 or not 0 <= overlap < seg_len:
        raise ValueError('Require seg_len > 0 and 0 <= overlap < seg_len.')
    starts = np.r_[0, bounds[1:, 0], T]
    step = seg_len - overlap
    indices, owners = [], []
    for trial, (lo, hi) in enumerate(zip(starts[:-1], starts[1:])):
        begin = np.arange(lo, hi - seg_len + 1, step)
        indices.extend(begin[:, None] + np.arange(seg_len)[None, :])
        owners.extend(np.full(len(begin), trial, np.int64))
    if not len(indices):
        return np.empty((0, seg_len), np.int64), np.empty(0, np.int64)
    return np.asarray(indices, dtype=np.int64), np.asarray(owners, dtype=np.int64)


def split_intervals(bounds, trials, T):
    """Return continuous intervals owned by selected whole-trial partitions."""
    bounds = validate_bounds(bounds, T)
    starts = np.r_[0, bounds[1:, 0], T]
    chosen = np.zeros(len(bounds), bool)
    chosen[np.asarray(trials, dtype=np.int64)] = True
    intervals = []
    for trial in np.flatnonzero(chosen):
        lo, hi = int(starts[trial]), int(starts[trial + 1])
        if intervals and intervals[-1][1] == lo:
            intervals[-1][1] = hi
        else:
            intervals.append([lo, hi])
    return np.asarray(intervals, dtype=np.int64).reshape(-1, 2)


def split_trials(bounds, val_frac=0.2, seed=0, test_frac=0.2):
    if not 0 < val_frac < 1 or not 0 < test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError('Validation and test fractions must be positive and sum to less than 1.')
    eligible = np.flatnonzero(bounds[:, 1] - bounds[:, 0] >= SEG_LEN)
    if len(eligible) < 4:
        raise ValueError('At least four trials must contain 30 or more bins.')
    order = np.random.default_rng(seed).permutation(eligible)
    nt = max(1, int(round(test_frac * len(order))))
    nv = max(1, int(round(val_frac * len(order))))
    if nt + nv > len(order) - 2:
        raise ValueError('Split leaves fewer than two training trials; reduce fractions or add trials.')
    return np.sort(order[nt + nv:]), np.sort(order[nt:nt + nv]), np.sort(order[:nt])


def trial_mask(T, bounds, trials):
    mask = np.zeros(T, bool)
    for lo, hi in bounds[trials]:
        mask[lo:hi] = True
    return mask


def read_archive(path):
    with np.load(path, allow_pickle=False) as z:
        if 'format_version' not in z or int(z['format_version']) != 6:
            raise ValueError('Regenerate NPZ files with dataloader format 6; behavior anti-alias filtering is required.')
        conventions = dict(derivative_definition=DERIVATIVE_DEFINITION,
                           behavior_mask_policy=BEHAVIOR_MASK_POLICY,
                           behavior_resampling=BEHAVIOR_RESAMPLING,
                           behavior_filter_order=BEHAVIOR_FILTER_ORDER,
                           behavior_filter_ripple_db=BEHAVIOR_FILTER_RIPPLE_DB,
                           behavior_filter_passband_nyquist_fraction=BEHAVIOR_FILTER_PASSBAND_NYQUIST_FRACTION,
                           behavior_filter_padlen=BEHAVIOR_FILTER_PADLEN)
        for key, expected in conventions.items():
            if key not in z:
                raise ValueError(f'Archive preprocessing convention differs: {key}; regenerate the archive.')
            stored = z[key].item()
            matches = str(stored) == expected if isinstance(expected, str) else float(stored) == float(expected)
            if not matches:
                raise ValueError(f'Archive preprocessing convention differs: {key}; regenerate the archive.')
        limits = z['behavior_limits'].astype(np.float32)
        if not np.array_equal(limits, BEHAVIOR_LIMITS):
            raise ValueError('Behavior outlier limits differ from the pinned source rules; regenerate the archive.')
        schema = dict(bin_size=float(z['bin_size']), n_force=int(z['n_force']),
                      smooth_ms=float(z['smooth_ms']), corr_thresh=float(z['corr_thresh']),
                      units=z['units'].tolist(), emg_names=z['emg_names'].tolist(),
                      behavior_limits=limits.tolist(), **conventions)
        days = {}
        for tag, name, id_key in (('0', 'day0', 'reference_id'), ('k', 'dayk', 'target_id')):
            if f'x_{tag}' not in z:
                continue
            x, b, time = z[f'x_{tag}'].copy(), z[f'b_{tag}'].copy(), z[f'time_{tag}'].copy()
            native_bin = float(z[f'behavior_native_bin_{tag}'])
            if not np.isfinite(native_bin) or native_bin <= 0 or not np.isclose(
                    schema['bin_size'] / native_bin,
                    round(schema['bin_size'] / native_bin)):
                raise ValueError('Stored native behavior bin is incompatible with the output bin size.')
            bounds = validate_bounds(z[f'trial_bounds_{tag}'], len(x))
            if x.ndim != 2 or x.shape[1] != len(schema['units']) or b.shape != (len(x), 2 * schema['n_force'] + len(schema['emg_names'])):
                raise ValueError('Inconsistent spike or behavior dimensions.')
            if not np.isfinite(x).all() or (x < 0).any() or (x != np.round(x)).any():
                raise ValueError('Spikes must be finite nonnegative integer counts.')
            if not np.isfinite(b).all():
                raise ValueError('Continuous-session behavior must be finite in every modeled bin.')
            present = z[f'present_{tag}'].astype(bool)
            valid_bins = z[f'behavior_valid_{tag}'].astype(bool)
            valid_trials = z[f'behavior_valid_trials_{tag}'].astype(bool)
            if present.shape != (x.shape[1],):
                raise ValueError('Invalid channel presence mask.')
            if (time.shape != (len(x),) or not np.isfinite(time).all() or
                    not np.allclose(np.diff(time), schema['bin_size'], rtol=0,
                                    atol=max(1e-9, schema['bin_size'] * 1e-6))):
                raise ValueError('Each day must have finite timestamps on a uniform bin_size time grid.')
            if valid_bins.shape != (len(x),) or valid_trials.shape != (len(bounds),):
                raise ValueError('Invalid behavioral outlier masks.')
            expected_bins, expected_trials = behavior_masks(b, bounds)
            if not np.array_equal(valid_trials, expected_trials) or not np.array_equal(valid_bins, expected_bins):
                raise ValueError('Stored behavior outlier masks do not match continuous behavior values.')
            onsets = z[f'onset_{tag}'].copy()
            fingerprint = hashlib.sha256(json.dumps(schema, sort_keys=True).encode())
            for arr in (x, b, time, bounds, onsets, present, valid_bins, valid_trials):
                fingerprint.update(str((arr.shape, arr.dtype.str)).encode())
                fingerprint.update(np.ascontiguousarray(arr).tobytes())
            fingerprint.update(np.float64(native_bin).tobytes())
            days[name] = dict(x=x, b=b, time=time, bounds=bounds, onset=onsets, present=present,
                              behavior_valid=valid_bins, behavior_valid_trials=valid_trials,
                              native_bin=native_bin, sha256=fingerprint.hexdigest(), id=str(z[id_key]))
    if 'day0' not in days:
        raise ValueError('A reference day is required.')
    return schema, days


def normalization_stats(x, intervals, sigma):
    n, mean, scatter = 0, None, None
    for lo, hi in intervals:
        values = x[lo:hi].astype(np.float64)
        if sigma:
            values = gaussian_filter1d(values, sigma, axis=0, mode='reflect')
        count, center = len(values), values.mean(0)
        ss = ((values - center) ** 2).sum(0)
        if n:
            delta = center - mean
            scatter += ss + delta ** 2 * (n * count / (n + count))
            mean += delta * (count / (n + count))
        else:
            mean, scatter = center, ss
        n += count
    sd = np.sqrt(np.maximum(scatter / n, 0))
    sd[sd < 1e-6] = 1.0
    return mean.astype(np.float32), sd.astype(np.float32)


def load_npz(path, device='cpu', seed=0, val_frac=0.2, test_frac=0.2,
             reference=None, target_seed=None, load_target=True, materialize_segments=True):
    import torch
    import copy

    schema, raw = read_archive(path)
    if not load_target:
        raw.pop('dayk', None)
    ref = raw['day0']
    sigma = schema['smooth_ms'] / 1000 / schema['bin_size']
    tensor = lambda a: torch.as_tensor(a, device=device)
    if reference is None:
        tr, va, te = split_trials(ref['bounds'], val_frac, seed, test_frac)
        intervals = split_intervals(ref['bounds'], tr, len(ref['x']))
        mask = np.zeros(len(ref['x']), bool)
        for lo, hi in intervals:
            mask[lo:hi] = True
        channels = np.arange(ref['x'].shape[1])
        if schema['corr_thresh']:
            with np.errstate(invalid='ignore', divide='ignore'):
                corr = np.nan_to_num(np.corrcoef(ref['x'][mask].astype(np.float64).T))
            corr = np.atleast_2d(corr)
            np.fill_diagonal(corr, 0)
            channels = np.flatnonzero(~(corr > schema['corr_thresh']).any(1))
            if not len(channels):
                raise ValueError('Training-only correlation filtering removed all channels. Disable it or raise corr_thresh.')
        behavior_mask = mask & ref['behavior_valid']
        if not behavior_mask.any():
            raise ValueError('Day-0 training partitions contain no behavior within the source outlier limits.')
        lo, hi = np.percentile(ref['b'][behavior_mask], [0.1, 99.9], axis=0)
        meta = dict(version=6, **schema, split_seed=seed, val_frac=val_frac, test_frac=test_frac,
                    channel_idx=torch.as_tensor(channels), selected_units=[schema['units'][i] for i in channels],
                    behavior_lo=torch.as_tensor(lo), behavior_span=torch.as_tensor(np.maximum(hi - lo, 1e-6)))
    else:
        if reference.get('version') != 6:
            raise ValueError('Regenerate data and retrain the reference model with pipeline version 6.')
        meta = copy.deepcopy(reference)
        meta.pop('dayk', None)
        for key, value in schema.items():
            if meta[key] != value:
                raise ValueError(f'Reference schema changed: {key}.')
        if ref['sha256'] != meta['day0']['sha256']:
            raise ValueError('Reference day changed; refusing to reuse the day-0 model.')
        if (seed, val_frac, test_frac) != (meta['split_seed'], meta['val_frac'], meta['test_frac']):
            raise ValueError('Reference split settings must match the day-0 checkpoint.')
        channels = meta['channel_idx'].cpu().numpy()
        if channels.ndim != 1 or not len(channels) or len(np.unique(channels)) != len(channels) or (channels < 0).any() or (channels >= len(schema['units'])).any():
            raise ValueError('Invalid reference channel selection.')
        if meta['selected_units'] != [schema['units'][i] for i in channels]:
            raise ValueError('Reference channel names do not match the selected indices.')
    lo, span = meta['behavior_lo'].cpu().numpy(), meta['behavior_span'].cpu().numpy()
    if lo.shape != (ref['b'].shape[1],) or span.shape != lo.shape or not np.isfinite(lo).all() or not np.isfinite(span).all() or (span <= 0).any():
        raise ValueError('Invalid reference behavior scaling.')
    out = {k: schema[k] for k in ('bin_size', 'n_force', 'emg_names')}
    out.update(behavior_lo=lo, behavior_span=span)
    for name in ('day0', 'dayk'):
        if name not in raw or (name == 'dayk' and not load_target):
            continue
        day = raw[name]
        x, bounds = day['x'][:, channels], day['bounds']
        split_seed = seed if name == 'day0' or target_seed is None else target_seed
        if name == 'day0' and reference is not None:
            saved = meta['day0']
            tr, va, te = (saved[k].cpu().numpy() for k in ('tr_trials', 'va_trials', 'te_trials'))
            expected = split_trials(bounds, val_frac, seed, test_frac)
            if not np.array_equal(bounds, saved['trial_bounds'].cpu().numpy()) or any(not np.array_equal(a, b) for a, b in zip((tr, va, te), expected)):
                raise ValueError('Reference trial bounds or split metadata are inconsistent.')
            mu, sd = saved['mu'].cpu().numpy(), saved['sd'].cpu().numpy()
            if mu.shape != (len(channels),) or sd.shape != mu.shape or not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd <= 0).any():
                raise ValueError('Invalid reference neural normalization.')
        else:
            tr, va, te = split_trials(bounds, val_frac, split_seed, test_frac)
            intervals = split_intervals(bounds, tr, len(x))
            mu, sd = normalization_stats(x, intervals, sigma)
        if name == 'dayk' and (not day['present'][channels].any() or
                               not any((x[lo:hi] > 0).any() for lo, hi in
                                       split_intervals(bounds, tr, len(x)))):
            raise ValueError('Target has no usable reference channels in training trials. Check the channel mapping.')
        meta[name] = dict(sha256=day['sha256'], id=day['id'], split_seed=split_seed,
                          trial_bounds=torch.as_tensor(bounds.copy()), tr_trials=torch.as_tensor(tr.copy()),
                          va_trials=torch.as_tensor(va.copy()), te_trials=torch.as_tensor(te.copy()),
                          mu=torch.as_tensor(mu.copy()), sd=torch.as_tensor(sd.copy()),
                          present=torch.as_tensor(day['present'][channels].copy()))
        y = (2 * (day['b'] - lo) / span - 1).astype(np.float32)
        d = dict(x=tensor(x), y=tensor(y), b=tensor(day['b']),
                 behavior_valid=tensor(day['behavior_valid']),
                 behavior_valid_trials=day['behavior_valid_trials'],
                 mu=tensor(mu), sd=tensor(sd), onset=day['onset'],
                 time=day['time'],
                 trial_bounds=bounds, tr_trials=tr, va_trials=va, te_trials=te)
        if materialize_segments:
            idx, owners = session_segments(len(x), bounds)
            d.update(seg_idx=idx, seg_trial=owners, seg_x=tensor(x[idx]), seg_y=tensor(y[idx]),
                     seg_behavior_valid=tensor(day['behavior_valid'][idx]))
            for key, trials in (('tr', tr), ('va', va), ('te', te)):
                d[key] = tensor(np.flatnonzero(np.isin(owners, trials)))
        out[name] = d
    out['metadata'] = meta
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--xds_path', default='/content/xds/xds_python')
    p.add_argument('--data_dir', default='/content/drive/MyDrive/jango')
    p.add_argument('--day0', default='Jango_20150730_001.mat')
    p.add_argument('--dayk', default=None, help='Optional target day; omit for the reference-only archive')
    p.add_argument('--bin_size', type=float, default=0.02)
    p.add_argument('--corr_thresh', type=float, default=0.0, help='Training-only correlation threshold on binned counts; 0 disables filtering')
    p.add_argument('--no_emg', action='store_true', help='behavior = force + dforce only (as in the paper)')
    p.add_argument('--out', default='jango.npz')
    p.add_argument('--synthetic', action='store_true')
    a = p.parse_args()
    if a.synthetic:
        make_synthetic(a.out)
    else:
        prepare(a.xds_path, a.data_dir, a.day0, a.dayk, a.bin_size, a.corr_thresh, not a.no_emg, a.out)
