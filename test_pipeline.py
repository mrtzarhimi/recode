"""Run with: python -m unittest test_pipeline -v"""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dataloader import (_bin_xds, antialias_resample, behavior_masks, force_derivative,
                        load_npz, make_synthetic, session_segments, split_intervals)
from evaluate import (causal_infer, feature_times, model_from_checkpoint,
                      r2vw, spike_baseline, trial_index, validate_pair)
from generate_report import split_condition
from lfads import GRUCell, LFADS, coordinated_dropout, lfads_loss, poisson_nll, spike_jitter
from nomad import mvn_kl
from train_phase1 import ramp as phase1_ramp
from train_phase2 import ramp as phase2_ramp
from training_utils import model_fingerprint, require_behavior


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        make_synthetic(cls.root / 'a.npz', n_trials=20, trial_len=100, C=8)
        make_synthetic(cls.root / 'b.npz', n_trials=20, trial_len=100, C=8, seed=9)
        with np.load(cls.root / 'a.npz') as z:
            cls.a = {k: z[k] for k in z.files}
        with np.load(cls.root / 'b.npz') as z:
            b = {k: z[k] for k in z.files}
        cls.b = copy.deepcopy(cls.a)
        for key in ('x_k', 'b_k', 'time_k', 'onset_k', 'trial_bounds_k', 'present_k',
                    'behavior_valid_k', 'behavior_valid_trials_k'):
            cls.b[key] = b[key]
        cls.b['target_id'] = np.array('target_b')
        np.savez_compressed(cls.root / 'b.npz', **cls.b)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_force_derivative_units_and_partition_boundaries(self):
        force = np.array([[0, 0], [10, -5], [30, 5], [40, 15],
                          [100, 100], [110, 90]], dtype=np.float32)
        bounds = np.array([[0, 3], [4, 6]])
        expected = [[0, 0], [500, -250], [1000, 500], [500, 500],
                    [0, 0], [500, -500]]
        np.testing.assert_allclose(force_derivative(force, bounds, .02), expected)
        changed = force.copy()
        changed[4:] += 10000
        np.testing.assert_array_equal(force_derivative(force, bounds, .02)[:4],
                                      force_derivative(changed, bounds, .02)[:4])

    def test_force_outliers_mask_samples_without_derivative_cutoffs(self):
        behavior = np.zeros((6, 4), np.float32)
        behavior[0, 2:] = [5000, -5000]
        behavior[1, 0] = 900
        bins, trials = behavior_masks(behavior, np.array([[0, 3], [4, 6]]))
        np.testing.assert_array_equal(bins, [True, False, False, True, True, True])
        np.testing.assert_array_equal(trials, [False, True])

    def test_session_segment_owner_shape_and_empty_intervals(self):
        idx, owners = session_segments(75, np.array([[0, 35], [40, 75]]))
        np.testing.assert_array_equal(idx[:, 0], [0, 40])
        np.testing.assert_array_equal(owners, [0, 1])
        empty, owners = session_segments(5, np.array([[0, 5]]))
        self.assertEqual(empty.shape, (0, 30))
        self.assertEqual(owners.shape, (0,))

    def test_target_alignment_loading_does_not_require_behavior_inliers(self):
        archive = copy.deepcopy(self.a)
        archive['b_k'][:, 0] = 900
        archive['behavior_valid_k'], archive['behavior_valid_trials_k'] = behavior_masks(
            archive['b_k'], archive['trial_bounds_k'])
        path = self.root / 'target_outliers.npz'
        np.savez_compressed(path, **archive)
        loaded = load_npz(path)
        self.assertFalse(loaded['dayk']['behavior_valid'].any())
        self.assertTrue(loaded['dayk']['x'].numel() > 0)
        with self.assertRaisesRegex(ValueError, 'training.*behavior'):
            require_behavior(loaded['dayk'])

    def test_training_keeps_valid_bins_in_trials_with_one_outlier(self):
        archive = copy.deepcopy(self.a)
        archive['b_0'][archive['trial_bounds_0'][:, 0], 0] = 900
        archive['behavior_valid_0'], archive['behavior_valid_trials_0'] = behavior_masks(
            archive['b_0'], archive['trial_bounds_0'])
        path = self.root / 'partial_behavior.npz'
        np.savez_compressed(path, **archive)
        loaded = load_npz(path, load_target=False)
        self.assertFalse(loaded['day0']['behavior_valid_trials'].any())
        self.assertTrue(loaded['day0']['seg_behavior_valid'].any())
        require_behavior(loaded['day0'])

    def test_legacy_derivative_and_mask_archives_require_regeneration(self):
        archive = copy.deepcopy(self.a)
        archive['format_version'] = np.array(4)
        path = self.root / 'legacy.npz'
        np.savez_compressed(path, **archive)
        with self.assertRaisesRegex(ValueError, 'format 7'):
            load_npz(path)

    def test_behavior_antialias_filter_attenuates_above_target_nyquist(self):
        native_bin, target_bin = .001, .02
        # Four seconds between reflection endpoints, so both tones fit exactly.
        native_times = np.arange(4001) * native_bin
        target_times = native_times[::20]
        bounds = np.array([[0, len(target_times)]])
        low = np.sin(2 * np.pi * 3 * native_times)
        high = np.sin(2 * np.pi * 35 * native_times)
        filtered_low = antialias_resample(low[:, None], native_times, target_times,
                                          bounds, native_bin, target_bin)[:, 0]
        filtered_high = antialias_resample(high[:, None], native_times, target_times,
                                           bounds, native_bin, target_bin)[:, 0]
        ripple = np.expm1(.05 * np.log(10) / 10)
        warped = np.tan(np.pi * 3 * native_bin) / np.tan(np.pi * 20 * native_bin)
        gain = 1 / (1 + ripple * np.cos(500 * np.arccos(warped)) ** 2)
        np.testing.assert_allclose(filtered_low, gain * low[::20], atol=1e-6, rtol=0)
        self.assertLess(np.max(np.abs(filtered_high)), 1e-6)

    def test_behavior_filter_never_uses_samples_from_another_owned_interval(self):
        native_bin, target_bin = .001, .02
        native_times = np.arange(8000) * native_bin
        target_times = native_times[::20]
        bounds = np.array([[0, 180], [200, 400]])
        values = np.sin(2 * np.pi * 3 * native_times)[:, None]
        changed = values.copy()
        changed[4000:] += 1000
        before = antialias_resample(values, native_times, target_times, bounds,
                                    native_bin, target_bin)
        after = antialias_resample(changed, native_times, target_times, bounds,
                                   native_bin, target_bin)
        np.testing.assert_array_equal(before[:200], after[:200])
        self.assertGreater(np.std(before[:200]), .6)
        self.assertGreater(np.mean(after[200:] - before[200:]), 900)

    def test_archive_rejects_constant_force_with_varying_other_behavior(self):
        archive = copy.deepcopy(self.a)
        archive['b_k'][:, :2] = [17, -21]
        path = self.root / 'constant_target.npz'
        np.savez_compressed(path, **archive)
        with self.assertRaisesRegex(ValueError, 'dayk.*constant force'):
            load_npz(path)

    def test_report_distinguishes_unsupervised_and_supervised_alignment(self):
        self.assertEqual(split_condition('day-k test, unsupervised alignment'), 'alignment_unsupervised')
        self.assertEqual(split_condition('day-k test, supervised alignment'), 'alignment_supervised')

    def test_behavior_resampling_at_native_rate_is_exact(self):
        values = np.arange(40, dtype=np.float64).reshape(20, 2)
        times = np.arange(20) * .02
        np.testing.assert_array_equal(
            antialias_resample(values, times, times, np.array([[0, 20]]), .02, .02),
            values.astype(np.float32))

    def test_disjoint_trials_and_bins(self):
        D = load_npz(self.root / 'a.npz')
        for day in ('day0', 'dayk'):
            d = D[day]
            masks = []
            for key in ('tr_trials', 'va_trials', 'te_trials'):
                mask = np.zeros(len(d['x']), bool)
                for lo, hi in split_intervals(d['trial_bounds'], d[key], len(d['x'])):
                    mask[lo:hi] = True
                masks.append(mask)
            self.assertFalse((masks[0] & masks[1]).any())
            self.assertFalse((masks[0] & masks[2]).any())
            self.assertFalse((masks[1] & masks[2]).any())
            bin_split = np.full(len(d['x']), -1)
            for split_id, mask in enumerate(masks):
                bin_split[mask] = split_id
            for idx, owner in zip(d['seg_idx'], d['seg_trial']):
                lo, hi = split_intervals(d['trial_bounds'], [owner], len(d['x']))[0]
                self.assertTrue(idx[0] >= lo and idx[-1] < hi)
                self.assertTrue((bin_split[idx] == bin_split[idx[0]]).all())
                self.assertGreaterEqual(bin_split[idx[0]], 0)

    def test_holdouts_cannot_change_preprocessing(self):
        a = copy.deepcopy(self.a)
        a['corr_thresh'] = np.array(.95)
        np.savez_compressed(self.root / 'filter.npz', **a)
        before = load_npz(self.root / 'filter.npz')
        for tag, name in (('0', 'day0'), ('k', 'dayk')):
            d = before[name]
            trials = np.r_[d['va_trials'], d['te_trials']]
            intervals = split_intervals(d['trial_bounds'], trials, len(d['x']))
            mask = np.zeros(len(d['x']), bool)
            for lo, hi in intervals:
                mask[lo:hi] = True
            a['x_' + tag][mask] += 1000
            a['b_' + tag][mask] += 1
        np.savez_compressed(self.root / 'changed.npz', **a)
        after = load_npz(self.root / 'changed.npz')
        for key in ('channel_idx', 'behavior_lo', 'behavior_span'):
            torch.testing.assert_close(before['metadata'][key], after['metadata'][key], rtol=0, atol=0)
        for name in ('day0', 'dayk'):
            for key in ('mu', 'sd'):
                torch.testing.assert_close(before[name][key], after[name][key], rtol=0, atol=0)
            tr = before[name]['tr']
            torch.testing.assert_close(before[name]['seg_x'][tr], after[name]['seg_x'][tr])
            torch.testing.assert_close(before[name]['seg_y'][tr], after[name]['seg_y'][tr])

    def test_reference_reuse_and_schema_rejection(self):
        first = load_npz(self.root / 'a.npz', load_target=False, materialize_segments=False)
        self.assertNotIn('dayk', first['metadata'])
        second = load_npz(self.root / 'b.npz', reference=first['metadata'], target_seed=22,
                          materialize_segments=False)
        self.assertEqual(first['metadata']['day0']['sha256'], second['metadata']['day0']['sha256'])
        torch.testing.assert_close(first['day0']['mu'], second['day0']['mu'], rtol=0, atol=0)
        bad = copy.deepcopy(self.b)
        bad['x_0'][0, 0] += 1
        np.savez_compressed(self.root / 'bad.npz', **bad)
        with self.assertRaisesRegex(ValueError, 'Reference day changed'):
            load_npz(self.root / 'bad.npz', reference=first['metadata'])

    def test_multiple_targets_share_one_reference(self):
        first = load_npz(self.root / 'a.npz', load_target=False, materialize_segments=False)
        changed = copy.deepcopy(self.b)
        changed['x_k'][0, 0] += 1
        changed['target_id'] = np.array('target_c')
        np.savez_compressed(self.root / 'c.npz', **changed)
        target_b = load_npz(self.root / 'b.npz', reference=first['metadata'], target_seed=22,
                            materialize_segments=False)
        target_c = load_npz(self.root / 'c.npz', reference=first['metadata'], target_seed=23,
                            materialize_segments=False)
        self.assertEqual(target_b['metadata']['day0']['sha256'], target_c['metadata']['day0']['sha256'])
        self.assertNotEqual(target_b['metadata']['dayk']['sha256'], target_c['metadata']['dayk']['sha256'])
        self.assertNotEqual(target_b['metadata']['dayk']['split_seed'],
                            target_c['metadata']['dayk']['split_seed'])

    def test_xds_instance_api_and_timestamps(self):
        class FakeXDS:
            bin_width = 0.01

            def bin_spikes(self, size, mode):
                self.mode = mode
                return np.array([0., .02, .04]), np.array([[1], [0], [2]])

            def resample_force(self, size):
                return np.array([0., .02, .04]), np.ones((3, 2))

            def resample_EMG(self, size):
                return np.array([0., .02, .04]), np.ones((3, 1))

        fake = FakeXDS()
        times, spikes, force, emg = _bin_xds(fake, .02, True)
        np.testing.assert_array_equal(times, [0., .02, .04])
        np.testing.assert_array_equal(spikes[:, 0], [1, 0, 2])
        self.assertEqual(fake.mode, 'left')
        self.assertEqual(force.shape, (3, 2))
        self.assertEqual(emg.shape, (3, 1))

        class ShiftedXDS(FakeXDS):
            def resample_force(self, size):
                return np.array([.01, .03, .05]), np.ones((3, 2))

        with self.assertRaisesRegex(ValueError, 'insufficient common bins'):
            _bin_xds(ShiftedXDS(), .02, False)

        class MissingBinXDS(FakeXDS):
            def resample_force(self, size):
                return np.array([0., .04]), np.ones((2, 2))

        with self.assertRaisesRegex(ValueError, 'internal gaps'):
            _bin_xds(MissingBinXDS(), .02, False)

    def test_objective_reductions_and_annealing(self):
        model = LFADS(n_in=2, n_beh=1, readin_dim=3, enc_dim=2, ic_dim=2,
                      ci_dim=2, con_dim=2, co_dim=1, gen_dim=3, fac_dim=2)
        x = torch.tensor([[[1., 0.], [0., 2.]], [[2., 1.], [1., 0.]]])
        y = torch.zeros(2, 2, 1)
        mask = torch.tensor([[True, False], [True, True]])
        out = model(x.float(), sample=False)
        loss, parts = lfads_loss(model, out, x, y, 0., 2., l2_gen=0., l2_con=0.,
                                 readin_l2=0., behavior_mask=mask)
        nll = poisson_nll(out['log_rate'], x).mean()
        expected_mse = ((out['beh'].square() * mask.unsqueeze(-1)).sum() / mask.sum())
        torch.testing.assert_close(parts['rec'], nll)
        torch.testing.assert_close(parts['sse'], expected_mse)
        torch.testing.assert_close(loss, nll + 2 * expected_mse)
        _, summed = lfads_loss(model, out, x, y, 0., 0., l2_gen=0., l2_con=0.,
                               readin_l2=0., nll_mean=False)
        expected_sum = poisson_nll(out['log_rate'], x).sum((1, 2)).mean()
        torch.testing.assert_close(summed['rec'], expected_sum)
        self.assertAlmostEqual(phase1_ramp(1, 100), 1 / 101)
        self.assertAlmostEqual(phase2_ramp(4, 2, start=2), 2 / 3)
        self.assertEqual(phase1_ramp(3, 0, start=4), 0.)
        self.assertEqual(phase2_ramp(4, 0, start=4), 1.)

    def test_gaussian_kl_direction_reference(self):
        mu0, cov0 = torch.tensor([0.]), torch.tensor([[1.]])
        muk, covk = torch.tensor([1.]), torch.tensor([[4.]])
        forward = mvn_kl(mu0, cov0, muk, covk)
        reverse = mvn_kl(muk, covk, mu0, cov0)
        torch.testing.assert_close(forward, torch.tensor(0.4431471805599453, dtype=torch.float32), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(reverse, torch.tensor(1.3068528194400546, dtype=torch.float32), atol=1e-6, rtol=1e-6)

    def test_alignment_rejects_incompatible_encoder_width(self):
        with self.assertRaisesRegex(ValueError, 'align_dim must equal readin_dim'):
            LFADS(n_in=2, n_beh=4, readin_dim=3, align_dim=50)

    def test_gru_equation_lag_projection_and_factor_norm(self):
        cell = GRUCell(1, 1, clip=100.)
        with torch.no_grad():
            cell.kernel.copy_(torch.tensor([[0.], [0.], [1.]]))
            cell.recurrent_kernel.copy_(torch.tensor([[1.], [2.], [3.]]))
            cell.bias.zero_()
        x, h = torch.zeros(1, 1), torch.tensor([[.5]])
        z, r = torch.sigmoid(h), torch.sigmoid(2 * h)
        expected = z * h + (1 - z) * torch.tanh(r * 3 * h)
        torch.testing.assert_close(cell(x, h), expected)

        model = LFADS(n_in=2, n_beh=1, readin_dim=3, enc_dim=2, ic_dim=2,
                      ci_dim=2, con_dim=2, co_dim=1, gen_dim=3, fac_dim=2,
                      controller_lag=1).eval()
        self.assertNotIn('ci_enc.h0', model.state_dict())
        self.assertIn('con.h0', model.state_dict())
        states = model._normalized_factor_weight().norm(dim=0)
        torch.testing.assert_close(states, torch.ones_like(states))
        seen = []
        handle = model.con.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
        inputs = torch.randn(1, 4, 2)
        hread = model.readin_norm(model.readin(inputs).transpose(1, 2)).transpose(1, 2)
        forward, backward = model.ci_enc(hread)
        with torch.no_grad():
            model.ic_to_g0.weight.zero_()
            model.ic_to_g0.bias.fill_(2.)
        init = []
        gen_hook = model.gen.register_forward_pre_hook(lambda module, args: init.append(args[1].detach()))
        model(inputs, sample=False)
        handle.remove()
        gen_hook.remove()
        self.assertTrue(torch.equal(seen[0][:, :2], torch.zeros_like(seen[0][:, :2])))
        torch.testing.assert_close(seen[1][:, :2], forward[:, 0])
        torch.testing.assert_close(seen[0][:, 2:4], backward[:, 1])
        torch.testing.assert_close(init[0], torch.full_like(init[0], 2.))

    def test_alignment_freezing_and_three_normalization_conditions(self):
        cfg = dict(n_in=2, n_beh=4, readin_dim=3, enc_dim=2, ic_dim=2,
                   ci_dim=2, con_dim=2, co_dim=1, gen_dim=3, fac_dim=2)
        model = LFADS(**cfg)
        model.add_align()
        model.requires_grad_(False)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.split('.')[0] in ('readin', 'align', 'fac', 'rate'))
        model.dropout_enabled = False
        model.deterministic = True
        out = model(torch.randn(2, 30, 2))
        (out['log_rate'].square().mean() + out['gen'].square().mean()).backward()
        for name, parameter in model.named_parameters():
            if name.split('.')[0] in ('readin', 'align', 'fac', 'rate'):
                self.assertIsNotNone(parameter.grad, name)
            else:
                self.assertIsNone(parameter.grad, name)

        base = LFADS(**cfg).eval()
        base.deterministic = True
        day = dict(x=np.random.default_rng(1).normal(size=(40, 2)).astype(np.float32),
                   mu=np.zeros(2, np.float32), sd=np.ones(2, np.float32),
                   bounds=np.array([[0, 40]], dtype=np.int64))
        times = np.array([35, 36], dtype=np.int64)
        unchanged, _ = causal_infer(base, day, times, normalization_stats=(day['mu'], day['sd']))
        target_mu, target_sd = np.array([1., -1.], np.float32), np.array([2., .5], np.float32)
        normalized, _ = causal_infer(base, day, times, normalization_stats=(target_mu, target_sd))
        base.add_align()
        aligned, _ = causal_infer(base, day, times, normalization_stats=(target_mu, target_sd))
        self.assertFalse(np.allclose(unchanged, normalized))
        self.assertFalse(np.allclose(normalized, aligned))

    def test_spike_jitter_reflects_without_losing_counts(self):
        x = torch.zeros(2, 30, 2)
        x[:, 0, 0], x[:, -1, 1] = 3, 2
        jittered = spike_jitter(x, 2)
        torch.testing.assert_close(jittered.sum((1, 2)), x.sum((1, 2)))
        self.assertTrue((jittered >= 0).all())

    def test_coordinated_dropout_masks_inputs_but_not_reconstruction_targets(self):
        x = torch.ones(4, 30, 8)
        masked, gradient_mask = coordinated_dropout(x, .3, 0.)
        self.assertTrue(torch.equal(masked == 0, gradient_mask))
        torch.testing.assert_close(masked[masked > 0], torch.full_like(masked[masked > 0], 1 / .7))
        self.assertEqual(x.sum().item(), 4 * 30 * 8)

    def test_causal_trial_contained_features(self):
        D = load_npz(self.root / 'a.npz', materialize_segments=False)
        d = D['dayk']
        idx, _ = trial_index(d['onset'], d['trial_bounds'], d['te_trials'], D['bin_size'], 4)
        times, lookup = feature_times(idx, 4)
        np.testing.assert_array_equal(times[lookup], idx[..., None] - np.arange(3, -1, -1))
        cfg = dict(n_in=8, n_beh=7, readin_dim=4, enc_dim=4, ic_dim=4, ci_dim=4,
                   con_dim=4, co_dim=2, gen_dim=5, fac_dim=3)
        model = LFADS(**cfg).eval()
        for key in ('x', 'y', 'mu', 'sd'):
            d[key] = d[key].numpy()
        d['bounds'] = d['trial_bounds']
        end = np.array([idx[0, 15]])
        g, p = causal_infer(model, d, end)
        s = spike_baseline(d, end, D['bin_size'])
        changed = dict(d, x=d['x'].copy())
        changed['x'][:end[0] - 29] += 100
        changed['x'][end[0] + 1:] += 100
        gg, pp = causal_infer(model, changed, end)
        np.testing.assert_array_equal(g, gg)
        np.testing.assert_array_equal(p, pp)
        np.testing.assert_array_equal(s, spike_baseline(changed, end, D['bin_size']))
        with self.assertRaisesRegex(ValueError, 'crosses a trial'):
            causal_infer(model, d, np.array([d['bounds'][1, 0] + 5]))

    def test_model_provenance(self):
        D = load_npz(self.root / 'a.npz', materialize_segments=False)
        cfg = dict(n_in=8, n_beh=7, gen_dim=5, fac_dim=3)
        model = LFADS(**cfg)
        state = model.state_dict()
        base = dict(cfg=cfg, data=copy.deepcopy(D['metadata']), state=copy.deepcopy(state),
                    model_id=model_fingerprint(cfg, state), aligned=False)
        base['data'].pop('dayk')
        model.add_align()
        aligned = dict(cfg=cfg, data=D['metadata'], state=model.state_dict(),
                       source_model_id=base['model_id'], aligned=True)
        validate_pair(base, aligned)
        aligned['source_model_id'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'different reference'):
            validate_pair(base, aligned)

    def test_constant_target_r2(self):
        y = np.array([[0., 1.], [1., 1.], [2., 1.]])
        p = y.copy()
        p[:, 1] = 100
        self.assertEqual(r2vw(y, p), 1.)
        self.assertEqual(r2vw(np.ones((3, 2)), np.zeros((3, 2))), 0.)


if __name__ == '__main__':
    unittest.main()
