"""Shared batching, validation statistics, and checkpoint utilities."""
import hashlib
import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import torch


def runtime_args(p, eval_bs=256):
    p.add_argument('--device', default='auto')
    p.add_argument('--eval_bs', type=int, default=eval_bs)
    p.add_argument('--threads', type=int, default=0, help='0 keeps the PyTorch default')
    p.add_argument('--amp', choices=('off', 'bf16'), default='off')
    p.add_argument('--log_every', type=int, default=5)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--split_seed', type=int, default=None)
    p.add_argument('--val_frac', type=float, default=None)
    p.add_argument('--test_frac', type=float, default=None)


def setup(a):
    for name in ('epochs', 'bs', 'eval_bs', 'patience', 'log_every'):
        if getattr(a, name) <= 0:
            raise ValueError(f'{name} must be positive.')
    if a.threads < 0 or a.lr_patience < 0:
        raise ValueError('threads and lr_patience must be nonnegative.')
    for name in ('lr', 'clip', 'min_lr'):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            raise ValueError(f'{name} must be finite and positive.')
    if not 0 < a.lr_decay < 1:
        raise ValueError('lr_decay must be in (0,1).')
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else torch.device(a.device)
    if a.amp != 'off' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('bf16 AMP requires a supported CUDA device.')
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    return device


def autocast(a, device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=a.amp == 'bf16')


def batches(d, indices, bs, device, shuffle=False):
    order = indices.cpu()
    if shuffle:
        order = order[torch.randperm(len(order))]
    for ids in order.split(bs):
        values = [d['seg_x'][ids], d['seg_y'][ids], d['seg_behavior_valid'][ids]]
        if device.type == 'cuda':
            values = [v.pin_memory() for v in values]
        yield tuple(v.to(device, non_blocking=device.type == 'cuda') for v in values)


def normalization(d, device):
    return d['mu'].to(device), d['sd'].to(device)


def require_behavior(d):
    for split, label in (('tr', 'training'), ('va', 'validation')):
        if not d['seg_behavior_valid'][d[split]].any():
            raise ValueError(f'{label} windows contain no behavior observations within force limits; '
                             'supervised optimization and validation require valid behavior bins.')


class Moments:
    def __init__(self):
        self.n, self.mean, self.scatter = 0, None, None

    def update(self, x):
        x = x.detach().flatten(0, 1).double()
        n = len(x)
        mean = x.mean(0)
        centered = x - mean
        scatter = centered.T @ centered
        if self.n:
            delta = mean - self.mean
            total = self.n + n
            self.scatter += scatter + torch.outer(delta, delta) * (self.n * n / total)
            self.mean += delta * (n / total)
        else:
            self.mean, self.scatter = mean, scatter
        self.n += n

    def result(self, eps=1e-6):
        if self.n < 2:
            raise ValueError('At least two state samples are required.')
        cov = self.scatter / (self.n - 1)
        cov = (cov + cov.T) * 0.5
        return self.mean, cov + eps * torch.eye(len(self.mean), device=cov.device, dtype=cov.dtype)


def data_metadata(path, D, split_seed, val_frac):
    return copy.deepcopy(D['metadata'])


def compact_training_data(D):
    for name in ('day0', 'dayk'):
        if name not in D:
            continue
        for key in ('x', 'y', 'b', 'time', 'behavior_valid', 'behavior_valid_trials',
                    'seg_xin', 'seg_idx', 'seg_trial', 'onset'):
            D[name].pop(key, None)
    D.pop('metadata', None)


def model_fingerprint(cfg, state):
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode())
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(str((key, tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_checkpoint(path, model, cfg, meta, a, epoch, metrics, aligned=False, source_model_id=None):
    payload = dict(cfg=cfg, state=cpu_state(model), data=meta, args=vars(a),
                   epoch=epoch, metrics=metrics, aligned=aligned)
    payload['model_id'] = model_fingerprint(cfg, payload['state'])
    payload['source_model_id'] = source_model_id
    temporary = str(path) + '.tmp'
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def finite_loss(loss):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite objective; check inputs and training settings.')
