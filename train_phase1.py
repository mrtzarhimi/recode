"""Train LFADS on day-0 training trials; validate on held-out trials."""
import argparse
import math
from pathlib import Path

import torch

from dataloader import load_npz
from lfads import LFADS, coordinated_dropout, lfads_loss, spike_jitter
from training_utils import (autocast, batches, compact_training_data, data_metadata, finite_loss,
                            normalization, require_behavior, runtime_args, save_checkpoint, setup)


def ramp(epoch, increase, start=0):
    if increase:
        return min(1.0, max(0.0, (epoch - start) / (increase + 1)))
    return float(epoch >= start)


@torch.no_grad()
def evaluate(model, d, idx, kl_w, beh_w, l2_gen, n_force, bs=256, a=None, l2_w=1.0):
    model.eval()
    device = next(model.parameters()).device
    mu, sd = normalization(d, device)
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    count, target_mean, target_scatter, residual = 0, None, None, None
    behavior_count = 0
    for xr, y, behavior_mask in batches(d, idx, bs, device):
        if a is not None and a.jitter:
            xr = spike_jitter(xr, a.jitter)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=a is not None and a.amp == 'bf16'):
            out = model((xr - mu) / sd)
            loss, parts = lfads_loss(model, out, xr, y, kl_w, beh_w, l2_gen,
                                     a.l2_con, a.readin_l2, l2_w,
                                     a.kl_ic_w, a.kl_co_w, a.nll_mean,
                                     a.l2_ic_enc, a.l2_ci_enc, behavior_mask)
        finite_loss(loss)
        valid_count = int(behavior_mask.sum().item())
        totals[:3] += torch.stack((loss - beh_w * parts['sse'], parts['rec'],
                                    parts['kl'])).double() * len(xr)
        totals[3] += parts['sse'].double() * valid_count
        behavior_count += valid_count
        valid = behavior_mask.reshape(-1)
        if not valid.any():
            continue
        truth = y[..., :n_force].flatten(0, 1).double()[valid]
        pred = out['beh'][..., :n_force].flatten(0, 1).double()[valid]
        n, mean = len(truth), truth.mean(0)
        scatter = (truth - mean).square().sum()
        error = (truth - pred).square().sum()
        if count:
            target_scatter += scatter + (mean - target_mean).square().sum() * (count * n / (count + n))
            target_mean += (mean - target_mean) * (n / (count + n))
            residual += error
        else:
            target_mean, target_scatter, residual = mean, scatter, error
        count += n
    if not count:
        raise ValueError('Validation split has no behavior observations within the source outlier limits.')
    result = dict(zip(('total', 'rec', 'kl', 'sse'), (totals[:3] / len(idx)).tolist()))
    result['sse'] = (totals[3] / max(1, behavior_count)).item()
    result['total'] += beh_w * result['sse']
    variance, error = target_scatter.item(), residual.item()
    result['force_r2'] = 1 - error / variance if variance > 0 else float(error == 0)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out_dir', default='runs')
    p.add_argument('--epochs', type=int, default=10000)
    p.add_argument('--bs', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--beh_lr', type=float, default=1e-3)
    p.add_argument('--lr_decay', type=float, default=0.95)
    p.add_argument('--lr_patience', type=int, default=6)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--clip', type=float, default=200.0)
    p.add_argument('--kl_ic_w', type=float, default=1e-4)
    p.add_argument('--kl_co_w', type=float, default=1e-4)
    p.add_argument('--kl_warmup', type=int, default=100)
    p.add_argument('--kl_start', type=int, default=0)
    p.add_argument('--beh_w', type=float, default=0.1)
    p.add_argument('--beh_warmup', type=int, default=100)
    p.add_argument('--beh_start', type=int, default=0)
    p.add_argument('--l2_gen', type=float, default=1e-6)
    p.add_argument('--l2_con', type=float, default=1e-6)
    p.add_argument('--l2_ic_enc', type=float, default=0.0)
    p.add_argument('--l2_ci_enc', type=float, default=0.0)
    p.add_argument('--readin_l2', type=float, default=5e-3)
    p.add_argument('--l2_warmup', type=int, default=100)
    p.add_argument('--l2_start', type=int, default=0)
    p.add_argument('--loss_scale', type=float, default=1e4)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    p.add_argument('--nll_mean', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--jitter', type=int, default=2)
    for name, default in (('readin_dim', 50), ('enc_dim', 100), ('ic_dim', 100),
                          ('ci_dim', 100), ('con_dim', 100), ('co_dim', 4),
                          ('gen_dim', 100), ('fac_dim', 30)):
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--dropout_rate', type=float, default=0.05)
    p.add_argument('--cd_rate', type=float, default=0.3)
    p.add_argument('--cd_pass_rate', type=float, default=0.0)
    p.add_argument('--ic_prior_var', type=float, default=0.1)
    p.add_argument('--co_prior_tau', type=float, default=10.0)
    p.add_argument('--co_prior_nvar', type=float, default=0.1)
    p.add_argument('--controller_lag', type=int, default=1)
    p.add_argument('--align_dim', type=int, default=None, help='Defaults to readin_dim; must match it')
    p.add_argument('--state_clip', type=float, default=5.0)
    p.add_argument('--seed', type=int, default=0)
    runtime_args(p)
    a = p.parse_args()
    dev = setup(a)
    if min(a.kl_warmup, a.beh_warmup, a.l2_warmup, a.kl_start, a.beh_start,
           a.l2_start, a.jitter) < 0:
        p.error('Warmup values and jitter must be nonnegative.')
    if any(not math.isfinite(w) or w < 0 for w in (a.kl_ic_w, a.kl_co_w, a.beh_w,
            a.l2_gen, a.l2_con, a.l2_ic_enc, a.l2_ci_enc,
            a.readin_l2)) or not math.isfinite(a.beh_lr) or a.beh_lr <= 0:
        p.error('Loss weights must be nonnegative and beh_lr positive, all finite.')
    if not 0 <= a.cd_pass_rate <= 1:
        p.error('cd_pass_rate must be in [0, 1].')
    split_seed = a.seed if a.split_seed is None else a.split_seed
    val_frac = 0.2 if a.val_frac is None else a.val_frac
    test_frac = 0.2 if a.test_frac is None else a.test_frac
    D = load_npz(a.data, 'cpu', seed=split_seed, val_frac=val_frac, test_frac=test_frac, load_target=False)
    d, nf = D['day0'], D['n_force']
    require_behavior(d)
    meta = data_metadata(a.data, D, split_seed, val_frac)
    compact_training_data(D)
    del D
    cfg = {k: getattr(a, k) for k in ('readin_dim', 'enc_dim', 'ic_dim', 'ci_dim', 'con_dim',
           'co_dim', 'gen_dim', 'fac_dim', 'dropout_rate', 'cd_rate', 'ic_prior_var',
           'co_prior_tau', 'co_prior_nvar', 'state_clip', 'controller_lag')}
    cfg['align_dim'] = a.readin_dim if a.align_dim is None else a.align_dim
    cfg.update(n_in=d['seg_x'].shape[-1], n_beh=d['seg_y'].shape[-1])
    model = LFADS(**cfg).to(dev)
    main_params = [v for n, v in model.named_parameters() if not n.startswith('beh.')]
    beh_params = list(model.beh.parameters())
    opt = torch.optim.Adam(main_params, lr=a.lr, eps=a.adam_epsilon)
    beh_opt = torch.optim.Adam(beh_params, lr=a.beh_lr, eps=a.adam_epsilon)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=a.lr_decay, patience=a.lr_patience)
    beh_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        beh_opt, factor=a.lr_decay, patience=a.lr_patience)
    mu, sd = normalization(d, dev)
    path = Path(a.out_dir) / 'model_day0.pt'
    best, wait = float('inf'), 0
    print(f'device={dev} | train={len(d["tr"])} valid={len(d["va"])} segments | data stays on CPU')
    for epoch in range(a.epochs):
        model.train()
        kl_w = ramp(epoch + 1, a.kl_warmup, a.kl_start)
        beh_w = a.beh_w * ramp(epoch + 1, a.beh_warmup, a.beh_start)
        l2_w = ramp(epoch + 1, a.l2_warmup, a.l2_start)
        for xr, y, behavior_mask in batches(d, d['tr'], a.bs, dev, shuffle=True):
            opt.zero_grad(set_to_none=True)
            beh_opt.zero_grad(set_to_none=True)
            x_aug = spike_jitter(xr, a.jitter) if a.jitter else xr
            x_cd, gradient_mask = coordinated_dropout(x_aug, a.cd_rate, a.cd_pass_rate)
            xin = (x_cd - mu) / sd
            with autocast(a, dev):
                out = model(xin)
                rates = out['log_rate']
                out['log_rate'] = rates.detach() + (rates - rates.detach()) * gradient_mask
                loss, _ = lfads_loss(model, out, x_aug, y, kl_w, beh_w,
                                     a.l2_gen, a.l2_con, a.readin_l2, l2_w,
                                     a.kl_ic_w, a.kl_co_w, a.nll_mean,
                                     a.l2_ic_enc, a.l2_ci_enc, behavior_mask)
            finite_loss(loss)
            (loss * a.loss_scale).backward()
            torch.nn.utils.clip_grad_norm_(main_params + beh_params, a.clip,
                                           error_if_nonfinite=True)
            opt.step()
            beh_opt.step()
        v = evaluate(model, d, d['va'], ramp(epoch + 1, a.kl_warmup, a.kl_start),
                     a.beh_w * ramp(epoch + 1, a.beh_warmup, a.beh_start), a.l2_gen,
                     nf, a.eval_bs, a, ramp(epoch + 1, a.l2_warmup, a.l2_start))
        ramp_ends = (a.kl_start + (a.kl_warmup + 1 if a.kl_warmup else 0),
                     a.beh_start + (a.beh_warmup + 1 if a.beh_warmup else 0),
                     a.l2_start + (a.l2_warmup + 1 if a.l2_warmup else 0))
        ready = epoch + 1 >= max(ramp_ends)
        if ready:
            scheduler.step(v['total'])
            beh_scheduler.step(v['sse'])
        if ready and v['total'] < best:
            best, wait = v['total'], 0
            save_checkpoint(path, model, cfg, meta, a, epoch, v)
        elif ready:
            wait += 1
        if epoch % a.log_every == 0 or epoch == a.epochs - 1:
            print(f'epoch {epoch + 1} | KL ramp {kl_w:.3f} | valid {v["total"]:.5f} | '
                  f'NLL {v["rec"]:.5f} | KL {v["kl"]:.5f} | behavior MSE {v["sse"]:.5f} | '
                  f'force R2 {v["force_r2"]:.3f} | lr {opt.param_groups[0]["lr"]:.2g} | '
                  f'behavior lr {beh_opt.param_groups[0]["lr"]:.2g}')
        if ready and (wait >= a.patience or opt.param_groups[0]['lr'] < a.min_lr):
            break
    if not math.isfinite(best):
        raise RuntimeError('Training ended before all regularization ramps completed; no reference checkpoint was selected.')
    print('saved', path)


if __name__ == '__main__':
    main()
