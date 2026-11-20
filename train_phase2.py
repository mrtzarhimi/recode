"""Align day-k training trials to the frozen day-0 generator distribution."""
import argparse
import math
from pathlib import Path

import torch

from dataloader import load_npz
from lfads import LFADS, lfads_kl, poisson_nll
from nomad import mvn_kl, mvn_stats
from training_utils import (Moments, autocast, batches, compact_training_data, data_metadata, finite_loss,
                            normalization, require_behavior, runtime_args, save_checkpoint, setup, model_fingerprint)

TRAINABLE = ('readin', 'align', 'fac', 'rate')


def ramp(epoch, increase, start=0):
    if increase:
        return min(1.0, max(0.0, (epoch - start) / (increase + 1)))
    return float(epoch >= start)


def objective(model, xin, x_raw, y, behavior_mask, stats0, a):
    out = model(xin)
    kl = mvn_kl(*stats0, *mvn_stats(out['gen'].flatten(0, 1)))
    lr = out['log_rate'].float()
    rec = poisson_nll(lr, x_raw).mean()
    ic_kl, co_kl = lfads_kl(model, out)
    posterior_kl = a.kl_ic_w * ic_kl + a.kl_co_w * co_kl
    if a.sup_w:
        error = (out['beh'].float() - y.float()).square()
        mask = behavior_mask.to(device=error.device, dtype=error.dtype).unsqueeze(-1)
        sup = (error * mask).sum() / (mask.sum() * y.shape[-1]).clamp_min(1)
    else:
        sup = rec.new_zeros(())
    weighted_align = a.align_w * kl
    weighted_rec = a.rec_w * ramp(a.epoch, a.nll_warmup, a.nll_start) * rec
    weighted_kl = ramp(a.epoch, a.kl_warmup, a.kl_start) * posterior_kl
    readin_l2 = a.readin_l2 * model.readin.weight.float().square().sum() * 0.5
    loss = weighted_align + a.lfads_w * (weighted_rec + weighted_kl + readin_l2)
    weighted_sup = a.sup_w * sup
    loss = loss + weighted_sup
    return loss, dict(align=weighted_align, rec=a.lfads_w * weighted_rec,
                      posterior_kl=a.lfads_w * weighted_kl,
                      ic_kl=ic_kl.detach(), co_kl=co_kl.detach(),
                      readin_l2=a.lfads_w * readin_l2, sup=weighted_sup)


@torch.no_grad()
def reference_stats(model, d, a, device):
    model.eval()
    moments = Moments()
    mu, sd = normalization(d, device)
    for xr, _, _ in batches(d, d['tr'], a.eval_bs, device):
        with autocast(a, device):
            moments.update(model((xr - mu) / sd)['gen'])
    return moments.result()


@torch.no_grad()
def evaluate(model, d, stats0, a, device):
    model.eval()
    moments = Moments()
    mu, sd = normalization(d, device)
    sums = torch.zeros(2, dtype=torch.float64, device=device)
    count = 0
    behavior_count = 0
    for xr, y, behavior_mask in batches(d, d['va'], a.eval_bs, device):
        with autocast(a, device):
            out = model((xr - mu) / sd)
            moments.update(out['gen'])
            lr = out['log_rate'].float()
            rec = poisson_nll(lr, xr).mean()
            if a.sup_w:
                error = (out['beh'].float() - y.float()).square()
                mask = behavior_mask.to(device=error.device, dtype=error.dtype).unsqueeze(-1)
                sup = (error * mask).sum() / (mask.sum() * y.shape[-1]).clamp_min(1)
            else:
                sup = rec.new_zeros(())
        valid_count = int(behavior_mask.sum().item())
        sums += torch.stack((rec * len(xr), sup * valid_count)).double()
        count += len(xr)
        behavior_count += valid_count
    if not count:
        raise ValueError('Validation split is empty.')
    kl = mvn_kl(*stats0, *moments.result())
    rec, sup = sums[0] / count, sums[1] / max(1, behavior_count)
    weighted_rec = a.lfads_w * a.rec_w * ramp(a.epoch, a.nll_warmup, a.nll_start) * rec
    total = a.align_w * kl + weighted_rec + a.sup_w * sup
    finite_loss(total)
    return dict(total=total.item(), align=(a.align_w * kl).item(), rec=weighted_rec.item(),
                rec_unweighted=rec.item(), sup=sup.item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--ckpt', default='runs/model_day0.pt')
    p.add_argument('--out_dir', default='runs')
    p.add_argument('--save_checkpoints', action='store_true',
                   help='Save each epoch checkpoint for matched downstream model selection.')
    p.add_argument('--fixed_epochs', action='store_true',
                   help='Run the full epoch budget without early stopping.')
    p.add_argument('--epochs', type=int, default=10000)
    p.add_argument('--bs', type=int, default=300)
    p.add_argument('--lr', type=float, default=4e-4)
    p.add_argument('--lr_decay', type=float, default=0.95)
    p.add_argument('--lr_patience', type=int, default=6)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--clip', type=float, default=2000.0)
    p.add_argument('--align_w', type=float, default=1.0)
    p.add_argument('--rec_w', type=float, default=10.0)
    p.add_argument('--nll_warmup', type=int, default=100)
    p.add_argument('--kl_warmup', type=int, default=10)
    p.add_argument('--nll_start', type=int, default=0)
    p.add_argument('--kl_start', type=int, default=0)
    p.add_argument('--kl_ic_w', type=float, default=1e-4)
    p.add_argument('--kl_co_w', type=float, default=1e-4)
    p.add_argument('--lfads_w', type=float, default=1.0)
    p.add_argument('--readin_l2', type=float, default=5e-3)
    p.add_argument('--loss_scale', type=float, default=1e4)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    p.add_argument('--sup_w', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--target_split_seed', type=int, default=None)
    runtime_args(p)
    p.set_defaults(min_lr=1e-9)
    a = p.parse_args()
    dev = setup(a)
    if any(not math.isfinite(w) or w < 0 for w in (a.align_w, a.rec_w, a.sup_w,
            a.kl_ic_w, a.kl_co_w, a.lfads_w, a.readin_l2)) or not any((a.align_w, a.rec_w, a.sup_w)):
        p.error('Objective weights must be finite, nonnegative, and not all zero.')
    if min(a.nll_warmup, a.kl_warmup, a.nll_start, a.kl_start) < 0:
        p.error('Annealing durations must be nonnegative.')
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=True)
    if ck.get('aligned'):
        p.error('Expected an unaligned day-0 checkpoint.')
    saved = ck.get('data')
    if saved is None or saved.get('version') != 6:
        p.error('Regenerate data and retrain phase 1 with pipeline version 6 (behavior anti-alias filtering).')
    source_model_id = model_fingerprint(ck['cfg'], ck['state'])
    if ck.get('model_id') != source_model_id:
        p.error('Reference checkpoint identity is invalid.')
    split_seed = saved['split_seed'] if a.split_seed is None else a.split_seed
    val_frac = saved['val_frac'] if a.val_frac is None else a.val_frac
    test_frac = saved['test_frac'] if a.test_frac is None else a.test_frac
    target_seed = a.seed if a.target_split_seed is None else a.target_split_seed
    D = load_npz(a.data, 'cpu', seed=split_seed, val_frac=val_frac, test_frac=test_frac,
                 reference=saved, target_seed=target_seed)
    if 'dayk' not in D:
        p.error('Phase 2 requires an archive containing a target day.')
    if a.sup_w:
        require_behavior(D['dayk'])
    if D['metadata']['day0']['sha256'] == D['metadata']['dayk']['sha256']:
        p.error('Reference and target recordings must be different.')
    meta = data_metadata(a.data, D, split_seed, val_frac)
    compact_training_data(D)
    model = LFADS(**ck['cfg']).to(dev)
    model.load_state_dict(ck['state'])
    model.requires_grad_(False)
    model.deterministic = False
    model.dropout_enabled = False
    a.epoch = 0
    stats0 = reference_stats(model, D['day0'], a, dev)
    model.add_align()
    for n, param in model.named_parameters():
        param.requires_grad_(n.split('.')[0] in TRAINABLE)
    params = [v for v in model.parameters() if v.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr, eps=a.adam_epsilon)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=a.lr_decay, patience=a.lr_patience)
    d = D['dayk']
    mu, sd = normalization(d, dev)
    path = Path(a.out_dir) / 'model_dayk.pt'
    checkpoint_dir = Path(a.out_dir) / 'checkpoints'
    if a.save_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    v = evaluate(model, d, stats0, a, dev)
    best, wait, best_epoch = v['total'], 0, -1
    save_checkpoint(path, model, ck['cfg'], meta, a, -1, v, aligned=True, source_model_id=source_model_id)
    if a.save_checkpoints:
        save_checkpoint(checkpoint_dir / 'epoch_00000.pt', model, ck['cfg'], meta, a, -1, v,
                        aligned=True, source_model_id=source_model_id)
    print(f'device={dev} | train={len(d["tr"])} valid={len(d["va"])} segments | trainable={sum(p.numel() for p in params):,}')
    print(f'before alignment | valid {best:.5f} | weighted alignment {v["align"]:.5f} | '
          f'weighted reconstruction {v["rec"]:.5f}')
    model_cfg = ck['cfg']
    del ck, D
    for epoch in range(a.epochs):
        a.epoch = epoch + 1
        model.train()
        model.dropout_enabled = False
        for xr, y, behavior_mask in batches(d, d['tr'], a.bs, dev, shuffle=True):
            opt.zero_grad(set_to_none=True)
            with autocast(a, dev):
                loss, parts = objective(model, (xr - mu) / sd, xr, y, behavior_mask, stats0, a)
            finite_loss(loss)
            norm = lambda grads: torch.sqrt(sum(g.float().square().sum()
                                                for g in grads if g is not None))
            align_grad = norm(torch.autograd.grad(parts['align'], params,
                                                   retain_graph=True, allow_unused=True))
            rec_grad = norm(torch.autograd.grad(parts['rec'], params,
                                                 retain_graph=True, allow_unused=True))
            (loss * a.loss_scale).backward()
            torch.nn.utils.clip_grad_norm_(params, a.clip, error_if_nonfinite=True)
            opt.step()
        v = evaluate(model, d, stats0, a, dev)
        if a.save_checkpoints:
            save_checkpoint(checkpoint_dir / f'epoch_{epoch + 1:05d}.pt', model, model_cfg,
                            meta, a, epoch, v, aligned=True, source_model_id=source_model_id)
        scheduler.step(v['total'])
        if v['total'] < best:
            best, wait, best_epoch = v['total'], 0, epoch
            save_checkpoint(path, model, model_cfg, meta, a, epoch, v, aligned=True, source_model_id=source_model_id)
        else:
            wait += 1
        if epoch % a.log_every == 0 or epoch == a.epochs - 1:
            print(f'epoch {epoch + 1} | valid {v["total"]:.5f} | weighted align {v["align"]:.5f} | '
                  f'weighted reconstruction {parts["rec"].item():.5f} | '
                  f'validation weighted reconstruction {v["rec"]:.5f} | '
                  f'weighted KL {parts["posterior_kl"].item():.5f} | '
                  f'alignment-objective grad {align_grad.item():.5g} | '
                  f'reconstruction-objective grad {rec_grad.item():.5g} | '
                  f'weighted readin L2 {parts["readin_l2"].item():.5g} | '
                  f'weighted supervised {parts["sup"].item():.5g} | '
                  f'lr {opt.param_groups[0]["lr"]:.2g}')
        if not a.fixed_epochs and (wait >= a.patience or opt.param_groups[0]['lr'] < a.min_lr):
            break
    selection = 'initial alignment retained' if best_epoch < 0 else f'best validation epoch {best_epoch + 1}'
    print('saved', path, f'({selection})')


if __name__ == '__main__':
    main()
