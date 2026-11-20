"""Identity-initialized alignment and Gaussian generator-state matching."""
import math
from numbers import Integral

import torch
from torch import nn


class AlignmentNet(nn.Module):
    def __init__(self, d, hidden_dim=None):
        super().__init__()
        hidden_dim = d if hidden_dim is None else hidden_dim
        if any(not isinstance(value, Integral) or isinstance(value, bool) or value <= 0
               for value in (d, hidden_dim)):
            raise ValueError('Alignment dimensions must be positive integers.')
        if hidden_dim != d:
            raise ValueError('Alignment width must match its encoder input dimension.')
        self.l1 = nn.Linear(d, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.GroupNorm(1, hidden_dim, eps=1e-12)
        with torch.no_grad():
            for layer in (self.l1, self.l2, self.l3):
                nn.init.eye_(layer.weight)
                layer.bias.zero_()

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def _validate_eps(eps):
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('Covariance regularization must be finite and positive.')


def _covariance(scatter, n, eps):
    cov = scatter / (n - 1)
    cov = (cov + cov.T) * 0.5
    return cov + eps * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)


def mvn_stats(G, eps=1e-6):
    """Differentiable sample mean and regularized unbiased covariance in float64."""
    _validate_eps(eps)
    if G.ndim != 2 or G.shape[0] < 2 or G.shape[1] == 0:
        raise ValueError('Expected states with shape (N,D), N >= 2 and D >= 1.')
    G = G.double()
    mu = G.mean(0)
    centered = G - mu
    return mu, _covariance(centered.T @ centered, len(G), eps)


def mvn_kl(mu0, S0, muK, SK):
    """KL(N0 || NK) via float64 Cholesky solves; covariances must be SPD."""
    if mu0.ndim != 1 or muK.shape != mu0.shape or not mu0.numel():
        raise ValueError('Gaussian means must be matching nonempty vectors.')
    m = mu0.numel()
    if S0.shape != (m, m) or SK.shape != (m, m):
        raise ValueError('Covariance shapes must match the Gaussian dimension.')
    mu0, S0, muK, SK = (v.double() for v in (mu0, S0, muK, SK))
    L0 = torch.linalg.cholesky((S0 + S0.T) * 0.5)
    LK = torch.linalg.cholesky((SK + SK.T) * 0.5)
    A = torch.linalg.solve_triangular(LK, L0, upper=False)
    delta = torch.linalg.solve_triangular(LK, (muK - mu0).unsqueeze(-1), upper=False)
    logdet = 2 * (LK.diagonal().log().sum() - L0.diagonal().log().sum())
    kl = 0.5 * (A.square().sum() + delta.square().sum() - m + logdet)
    return kl.float()


@torch.no_grad()
def day0_stats(base, seg_xin, bs=512, eps=1e-6):
    """Pool generator-state moments without retaining all state trajectories."""
    _validate_eps(eps)
    if not isinstance(bs, Integral) or isinstance(bs, bool) or bs <= 0:
        raise ValueError('bs must be a positive integer.')
    if seg_xin.ndim != 3 or min(seg_xin.shape) == 0:
        raise ValueError('Expected nonempty segments with shape (B,T,C).')
    modes = [(module, module.training) for module in base.modules()]
    deterministic = base.deterministic
    n, mean, scatter = 0, None, None
    try:
        base.eval()
        base.deterministic = True
        for i in range(0, len(seg_xin), bs):
            states = base(seg_xin[i:i + bs])['gen'].flatten(0, 1).double()
            count = len(states)
            batch_mean = states.mean(0)
            centered = states - batch_mean
            batch_scatter = centered.T @ centered
            if n == 0:
                mean, scatter = batch_mean, batch_scatter
            else:
                delta = batch_mean - mean
                total = n + count
                scatter = scatter + batch_scatter + torch.outer(delta, delta) * (n * count / total)
                mean = mean + delta * (count / total)
            n += count
        if n < 2:
            raise ValueError('At least two generator states are required.')
        return mean, _covariance(scatter, n, eps)
    finally:
        base.deterministic = deterministic
        for module, training in modes:
            module.training = training


def mr2(y, yh):
    """Pooled variance-weighted R², with finite results for constant targets."""
    if y.ndim != 2 or y.shape != yh.shape or min(y.shape) == 0:
        raise ValueError('Targets and predictions must have matching nonempty (N,D) shapes.')
    if isinstance(y, torch.Tensor):
        y, yh = y.double(), yh.double()
        residual = (yh - y).square().sum()
        variance = (y - y.mean(0)).square().sum()
        score = 1 - residual / variance.clamp_min(torch.finfo(variance.dtype).tiny)
        constant_score = (residual == 0).to(score.dtype)
        return torch.where(variance > 0, score, constant_score).float()
    import numpy as np

    y, yh = np.asarray(y, dtype=np.float64), np.asarray(yh, dtype=np.float64)
    residual = ((yh - y) ** 2).sum()
    variance = ((y - y.mean(0)) ** 2).sum()
    return 1 - residual / variance if variance > 0 else float(residual == 0)
