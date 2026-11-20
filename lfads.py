"""PyTorch port of the pinned NoMAD BehaviorLFADS executable model."""
import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


class GRUCell(nn.Module):
    """Keras GRUCell equations with reset_after=False and LFADS clipping."""
    def __init__(self, input_dim, hidden_dim, clip=5.0, train_h0=False,
                 init_scale_dim=None):
        super().__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.clip, self.train_h0 = clip, train_h0
        self.init_scale_dim = init_scale_dim
        self.kernel = nn.Parameter(torch.empty(3 * hidden_dim, input_dim))
        self.recurrent_kernel = nn.Parameter(torch.empty(3 * hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(3 * hidden_dim))
        self.h0 = nn.Parameter(torch.zeros(hidden_dim)) if train_h0 else None
        self.reset_parameters()

    def reset_parameters(self):
        input_scale = self.init_scale_dim or self.input_dim
        recurrent_scale = self.init_scale_dim or self.hidden_dim
        nn.init.normal_(self.kernel, std=1 / math.sqrt(input_scale))
        nn.init.normal_(self.recurrent_kernel, std=1 / math.sqrt(recurrent_scale))
        with torch.no_grad():
            self.bias[:2 * self.hidden_dim].fill_(1)
            if self.h0 is not None:
                self.h0.zero_()

    def initial(self, batch, ref):
        return ref.new_zeros(batch, self.hidden_dim) if self.h0 is None else self.h0.expand(batch, -1)

    def forward(self, x, h):
        x_z, x_r, x_h = F.linear(x, self.kernel, self.bias).chunk(3, dim=-1)
        h_z, h_r, h_h = F.linear(h, self.recurrent_kernel).chunk(3, dim=-1)
        z = torch.sigmoid(x_z + h_z)
        r = torch.sigmoid(x_r + h_r)
        candidate = torch.tanh(x_h + r * h_h)
        h = (z * h + (1 - z) * candidate).clamp(-self.clip, self.clip)
        return h


class BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, clip):
        super().__init__()
        init_scale_dim = input_dim + hidden_dim
        self.forward_cell = GRUCell(input_dim, hidden_dim, clip,
                                    init_scale_dim=init_scale_dim)
        self.backward_cell = GRUCell(input_dim, hidden_dim, clip,
                                     init_scale_dim=init_scale_dim)

    def forward(self, x):
        b, t, _ = x.shape
        hf = self.forward_cell.initial(b, x)
        hb = self.backward_cell.initial(b, x)
        forward, backward = [], []
        for i in range(t):
            hf = self.forward_cell(x[:, i], hf)
            forward.append(hf)
            j = t - i - 1
            hb = self.backward_cell(x[:, j], hb)
            backward.append(hb)
        return torch.stack(forward, 1), torch.stack(backward[::-1], 1)


class LFADS(nn.Module):
    def __init__(self, n_in, n_beh, readin_dim=50, enc_dim=100, ic_dim=100,
                 ci_dim=100, con_dim=100, co_dim=4, gen_dim=100, fac_dim=30,
                 dropout_rate=0.05, cd_rate=0.3, ic_prior_var=0.1,
                 co_prior_tau=10.0, co_prior_nvar=0.1, state_clip=5.0,
                 controller_lag=1, align_dim=None):
        super().__init__()
        dims = (n_in, n_beh, readin_dim, enc_dim, ic_dim, ci_dim, con_dim,
                co_dim, gen_dim, fac_dim)
        if any(not isinstance(d, Integral) or isinstance(d, bool) or d <= 0 for d in dims):
            raise ValueError('All model dimensions must be positive integers.')
        if not 0 <= dropout_rate < 1 or not 0 <= cd_rate < 1:
            raise ValueError('Dropout and CD rates must be in [0, 1).')
        positive = (ic_prior_var, co_prior_tau, co_prior_nvar, state_clip)
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError('Prior parameters and state clipping must be finite and positive.')
        if not isinstance(controller_lag, Integral) or isinstance(controller_lag, bool) or controller_lag < 0:
            raise ValueError('controller_lag must be a nonnegative integer.')
        align_dim = readin_dim if align_dim is None else align_dim
        if not isinstance(align_dim, Integral) or isinstance(align_dim, bool) or align_dim <= 0:
            raise ValueError('align_dim must be a positive integer.')
        if align_dim != readin_dim:
            raise ValueError('align_dim must equal readin_dim because the frozen encoders consume the alignment output.')
        self.dropout_rate, self.cd_rate = dropout_rate, cd_rate
        self.state_clip, self.controller_lag, self.align_dim = state_clip, controller_lag, align_dim
        self.posterior_sampling = True
        self.dropout_enabled = True
        self.deterministic = False
        self.readin = nn.Linear(n_in, readin_dim)
        self.readin_norm = nn.GroupNorm(1, readin_dim, eps=1e-12)
        self.align = None
        self.ic_enc = BiGRU(readin_dim, enc_dim, state_clip)
        self.ic_mu, self.ic_lv = nn.Linear(2 * enc_dim, ic_dim), nn.Linear(2 * enc_dim, ic_dim)
        self.ic_to_g0 = nn.Linear(ic_dim, gen_dim)
        self.ci_enc = BiGRU(readin_dim, ci_dim, state_clip)
        self.con = GRUCell(2 * ci_dim + fac_dim, con_dim, state_clip, train_h0=True)
        self.co_mu, self.co_lv = nn.Linear(con_dim, co_dim), nn.Linear(con_dim, co_dim)
        self.gen = GRUCell(co_dim, gen_dim, state_clip)
        self.fac = nn.Linear(gen_dim, fac_dim, bias=False)
        self.rate = nn.Linear(fac_dim, n_in)
        self.beh = nn.Linear(gen_dim, n_beh)
        self.ic_prior_mean = nn.Parameter(torch.zeros(ic_dim))
        self.register_buffer('ic_prior_logvar', torch.full((ic_dim,), math.log(ic_prior_var)))
        self.co_prior_logtau = nn.Parameter(torch.full((co_dim,), math.log(co_prior_tau)))
        self.co_prior_lognvar = nn.Parameter(torch.full((co_dim,), math.log(co_prior_nvar)))
        self.ic_prior_var = ic_prior_var
        self.co_prior_tau, self.co_prior_nvar = co_prior_tau, co_prior_nvar
        self._initialize_dense(self.readin)
        self._initialize_dense(self.ic_mu)
        self._initialize_dense(self.ic_lv)
        self._initialize_dense(self.ic_to_g0)
        self._initialize_dense(self.co_mu)
        self._initialize_dense(self.co_lv)
        self._initialize_dense(self.rate)
        self._initialize_dense(self.beh)
        self._initialize_dense(self.fac)

    @staticmethod
    def _initialize_dense(layer):
        nn.init.normal_(layer.weight, std=1 / math.sqrt(layer.in_features))
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def add_align(self, hidden_dim=None):
        from nomad import AlignmentNet
        if hidden_dim is not None and hidden_dim != self.readin.out_features:
            raise ValueError('align_dim must equal readin_dim because the frozen encoders consume the alignment output.')
        if self.align is None:
            self.align = AlignmentNet(self.readin.out_features, hidden_dim or self.align_dim).to(
                device=self.readin.weight.device, dtype=self.readin.weight.dtype)
            self.align.train(self.training)
        return self.align

    def _dropout(self, x):
        return F.dropout(x, p=self.dropout_rate,
                         training=self.training and self.dropout_enabled)

    def _sample(self, mu, lv, sample):
        return mu + (0.5 * lv).exp() * torch.randn_like(mu) if sample else mu

    def forward(self, x, *, sample=None):
        if x.ndim != 3 or min(x.shape[:2]) == 0 or x.shape[-1] != self.readin.in_features:
            raise ValueError('Expected nonempty x with shape (batch, time, n_in).')
        if not x.is_floating_point():
            raise TypeError('x must be floating point, normalized spike counts.')
        b, t, _ = x.shape
        stochastic = self.posterior_sampling and not self.deterministic
        sample = stochastic if sample is None else sample

        h = self.readin(x)
        h = self.readin_norm(h.transpose(1, 2)).transpose(1, 2)
        if self.align is not None:
            h = self.align(h)
        h = self._dropout(h)

        ic_f, ic_b = self.ic_enc(h)
        ic_summary = self._dropout(torch.cat((ic_f[:, -1], ic_b[:, 0]), dim=-1))
        ic_mu = self.ic_mu(ic_summary)
        ic_raw_lv = self.ic_lv(ic_summary)
        ic_lv = torch.log(ic_raw_lv.exp() + 1e-4)
        g0 = self.ic_to_g0(self._sample(ic_mu, ic_lv, sample))
        g0_drop = self._dropout(g0)

        ci_f, ci_b = self.ci_enc(h)
        lag = self.controller_lag
        if lag:
            ci_f = torch.cat((ci_f.new_zeros(b, lag, ci_f.shape[-1]), ci_f[:, :-lag]), 1)
            ci_b = torch.cat((ci_b[:, lag:], ci_b.new_zeros(b, lag, ci_b.shape[-1])), 1)
        ci = torch.cat((ci_f, ci_b), -1)

        con = self.con.initial(b, h)
        gen = g0
        factor = F.linear(g0_drop, self._normalized_factor_weight())
        states, factors, means, logvars, inputs = [], [], [], [], []
        for step in range(t):
            con_input = self._dropout(torch.cat((ci[:, step], factor), -1))
            con = self.con(con_input, con)
            um, ul = self.co_mu(con), self.co_lv(con)
            u = self._sample(um, ul, sample)
            gen = self.gen(u, gen)
            factor = F.linear(self._dropout(gen), self._normalized_factor_weight())
            states.append(gen)
            factors.append(factor)
            means.append(um)
            logvars.append(ul)
            inputs.append(u)
        gen, factors = torch.stack(states, 1), torch.stack(factors, 1)
        return dict(gen=gen, factors=factors,
                    log_rate=self.rate(factors), beh=self.beh(gen),
                    ic_mu=ic_mu, ic_lv=ic_lv, u_mu=torch.stack(means, 1),
                    u_lv=torch.stack(logvars, 1), u_sample=torch.stack(inputs, 1))

    def _normalized_factor_weight(self):
        return F.normalize(self.fac.weight, p=2, dim=0)


def _normal_kl(mu, logvar, prior_mu, prior_logvar):
    return 0.5 * (prior_logvar - logvar
                  + (logvar.exp() + (mu - prior_mu).square()) * (-prior_logvar).exp() - 1)


def lfads_kl(model, out):
    ic = _normal_kl(out['ic_mu'], out['ic_lv'], model.ic_prior_mean,
                    model.ic_prior_logvar).sum(-1).mean()
    u = out['u_sample']
    log_q = -0.5 * (math.log(2 * math.pi) + out['u_lv']
                    + (u - out['u_mu']).square() * (-out['u_lv']).exp()).sum(-1)
    alpha = torch.exp(-torch.exp(-model.co_prior_logtau))
    innovation_var = model.co_prior_lognvar.exp()
    stationary_var = innovation_var / (1 - alpha.square())
    prior_mean = torch.cat((torch.zeros_like(u[:, :1]), alpha * u[:, :-1]), dim=1)
    prior_var = torch.cat((stationary_var.expand_as(u[:, :1]),
                           innovation_var.expand_as(u[:, 1:])), dim=1)
    log_p = -0.5 * (math.log(2 * math.pi) + prior_var.log()
                    + (u - prior_mean).square() / prior_var).sum(-1)
    return ic, (log_q - log_p).sum(1).mean()


def recurrent_l2(model, l2_gen, l2_con, l2_ic_enc, l2_ci_enc):
    groups = (([model.ic_enc.forward_cell, model.ic_enc.backward_cell], l2_ic_enc),
              ([model.ci_enc.forward_cell, model.ci_enc.backward_cell], l2_ci_enc),
              ([model.con], l2_con), ([model.gen], l2_gen))
    value = sum((scale * 0.5 * cell.recurrent_kernel.square().sum()
                 for cells, scale in groups if scale > 0 for cell in cells),
                model.gen.recurrent_kernel.new_zeros(()))
    recurrent_size = sum(3 * cell.hidden_dim ** 2 for cells, scale in groups
                         if scale > 0 for cell in cells)
    return value / max(1, recurrent_size)


def poisson_nll(log_rate, counts):
    """Exact TensorFlow log_poisson_loss with compute_full_loss=True."""
    log_rate, counts = log_rate.float(), counts.float()
    return log_rate.exp() - counts * log_rate + torch.lgamma(counts + 1)


def lfads_loss(model, out, x_raw, y, kl_w, beh_w, l2_gen=1e-6,
               l2_con=1e-6, readin_l2=5e-3, l2_w=1.0,
               kl_ic_w=1e-4, kl_co_w=1e-4, nll_mean=True,
               l2_ic_enc=0.0, l2_ci_enc=0.0, behavior_mask=None):
    """Original mean-entry NLL/behavior loss and LFADS KL/L2 conventions."""
    lr = out['log_rate'].float()
    counts = x_raw.float()
    if lr.shape != counts.shape or counts.ndim != 3 or counts.shape[0] == 0:
        raise ValueError('Raw counts and log rates must have matching nonempty (B,T,C) shapes.')
    nll = poisson_nll(lr, counts)
    rec = nll.mean() if nll_mean else nll.sum((1, 2)).mean()
    ic, co = lfads_kl(model, out)
    kl = kl_ic_w * ic + kl_co_w * co
    if y is None:
        if beh_w:
            raise ValueError('Behavior targets are required when beh_w is positive.')
        sse = rec.new_zeros(())
    else:
        if y.shape != out['beh'].shape:
            raise ValueError('Behavior targets and predictions must have matching shapes.')
        error = (out['beh'].float() - y.float()).square()
        if behavior_mask is None:
            sse = error.mean()
        else:
            if behavior_mask.shape != y.shape[:-1]:
                raise ValueError('Behavior mask must match the batch and time dimensions.')
            mask = behavior_mask.to(device=error.device, dtype=error.dtype).unsqueeze(-1)
            denominator = (mask.sum() * y.shape[-1]).clamp_min(1)
            sse = (error * mask).sum() / denominator
    l2 = l2_w * recurrent_l2(model, l2_gen, l2_con, l2_ic_enc, l2_ci_enc)
    readin = l2_w * readin_l2 * model.readin.weight.square().sum() * 0.5 / model.readin.weight.numel()
    total = rec + kl_w * kl + beh_w * sse + l2 + readin
    return total, dict(rec=rec.detach(), kl=kl.detach(), ic_kl=ic.detach(),
                       co_kl=co.detach(), sse=sse.detach(), l2=l2.detach(),
                       readin_l2=readin.detach())


@torch.no_grad()
def coordinated_dropout(x, rate, pass_rate=0.0):
    if not 0 <= rate < 1 or not 0 <= pass_rate <= 1:
        raise ValueError('Coordinated dropout rate must be in [0, 1) and pass rate in [0, 1].')
    keep = torch.rand_like(x) >= rate if rate else torch.ones_like(x, dtype=torch.bool)
    pass_mask = torch.rand_like(x) < pass_rate if pass_rate else torch.zeros_like(keep)
    gradient_mask = torch.ones_like(keep) if rate == 0 else (~keep | pass_mask)
    return x * keep / (1 - rate), gradient_mask


@torch.no_grad()
def spike_jitter(x, max_shift=2):
    """Jitter input/target counts independently with reflected segment edges."""
    if not isinstance(max_shift, Integral) or isinstance(max_shift, bool) or max_shift < 0:
        raise ValueError('max_shift must be a nonnegative integer.')
    if x.ndim != 3 or min(x.shape) == 0:
        raise ValueError('Expected nonempty counts with shape (B,T,C).')
    if not torch.isfinite(x).all() or (x < 0).any() or (x != x.round()).any():
        raise ValueError('Spike jitter requires finite, nonnegative integer counts.')
    if max_shift == 0:
        return x.clone()
    b, t, c = x.shape
    out = torch.zeros_like(x)
    for batch in range(b):
        for unit in range(c):
            counts = x[batch, :, unit].long()
            bins = torch.repeat_interleave(torch.arange(t, device=x.device), counts)
            shifts = torch.randint(-max_shift, max_shift + 1, bins.shape, device=x.device)
            destination = bins + shifts
            destination = torch.where(destination < 0, -destination, destination)
            destination = torch.where(destination >= t, 2 * (t - 1) - destination, destination)
            destination = destination.clamp(0, t - 1)
            out[batch, :, unit].scatter_add_(0, destination, torch.ones_like(destination, dtype=x.dtype))
    return out
