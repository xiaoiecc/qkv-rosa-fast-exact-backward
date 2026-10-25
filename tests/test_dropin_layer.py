"""Ordinary autograd contracts for block credits and repair sampling."""
from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from conv_rosa_transformer import rosa_core
from conv_rosa_transformer.joint_credit import Bit
from conv_rosa_transformer.rosa_layer import NBitQKVRosa
from tests.frozen_training_oracle import (NBitQKVRosa as PythonOracle,
    joint_rows as _joint_rows, payload_radius as _payload_radius)


def _layer(*args, **overrides):
    # Every ablation is explicit, independent of the selected shipping recipe.
    options = dict(repair_samples=None, joint_pairs=0, joint_rows=32,
                   stochastic_values=False, trust_diagonal=0.,
                   training_rule="hard_credit", joint_row_selection="input")
    use_oracle=overrides.pop("_python_oracle",False)
    options.update(overrides)
    # Python mocking checks use the frozen mathematical oracle. Normal
    # training/AMP/checkpoint/state checks below use the native layer.
    if use_oracle:
        return PythonOracle(*args, **options)
    return NBitQKVRosa(*args, **options)


class DropInLayerTests(unittest.TestCase):
    def data(self, d=4, n=7):
        generator = torch.Generator().manual_seed(719)
        tensors = [torch.randn(1, n, d, generator=generator, dtype=torch.float64,
                               requires_grad=True) for _ in range(3)]
        g1 = torch.randn(1, n, d, generator=generator, dtype=torch.float64)
        g2 = torch.randn(1, n, d, generator=generator, dtype=torch.float64)
        return tensors, g1, g2

    def test_loss_addition_scaling_and_repeated_backward(self):
        for options in ({"repair_samples": 2}, {"joint_pairs": 2},
                        {"repair_samples": 2, "joint_pairs": 2}):
            with self.subTest(options=options):
                layer = _layer(4, 4, **options).double()
                tensors, g1, g2 = self.data()
                y = layer._hard_rosa_from_qkv(*tensors)
                parameters = tensors + [layer.emb0, layer.emb1]
                def grad(g):
                    return torch.autograd.grad(y, parameters, g, retain_graph=True)
                a, b, total, scaled, repeated = grad(g1), grad(g2), grad(g1+g2), grad(8*g1), grad(g1)
                for x, z, t, s, r in zip(a, b, total, scaled, repeated):
                    torch.testing.assert_close(t, x+z, atol=2e-12, rtol=2e-12)
                    torch.testing.assert_close(s, 8*x, atol=2e-12, rtol=2e-12)
                    torch.testing.assert_close(r, x, atol=0, rtol=0)

    def test_mask_average_matches_full_independent_block(self):
        # Pair lanes 0/1 are forced. Sampling one of 2/3 has pi=1/2;
        # averaging both complete realizations recovers the full surrogate.
        tensors, g, _ = self.data()
        pair = [(Bit("Q", 6, 0), Bit("Q", 6, 1))]
        options = dict(joint_pairs=1, joint_rows=7)
        full = _layer(4, 4, _python_oracle=True, **options).double()
        sampled = _layer(4, 4, repair_samples=3, _python_oracle=True, **options).double()
        sampled.load_state_dict(full.state_dict())
        with patch("conv_rosa_transformer.joint_credit.propose_pairs", return_value=(pair, {})):
            expected = torch.autograd.grad(full._hard_rosa_from_qkv(*tensors), tensors, g)
            values = []
            for lane in (2, 3):
                def sample(q, k, v, g, e0, e1, count, seed, *, forced_bits, D, diag_y):
                    self.assertEqual(forced_bits, [0, 1])
                    return rosa_core.repair_sampled_credits_numpy(q, k, v, g, e0, e1,
                        [0, 1, lane], np.array([1., 1., .5, .5]), D, diag_y)
                with patch.object(rosa_core, "adaptive_repair_sampled_credits_numpy", side_effect=sample):
                    values.append(torch.autograd.grad(sampled._hard_rosa_from_qkv(*tensors), tensors, g))
        for i in range(3):
            torch.testing.assert_close((values[0][i]+values[1][i])/2, expected[i],
                                       atol=2e-12, rtol=2e-12)

    def test_joint_replays_stay_inside_one_layer_forward_backward(self):
        tensors, g, _ = self.data()
        pair = [(Bit("Q", 6, 0), Bit("K", 0, 1))]
        layer = _layer(4, 4, joint_pairs=1, _python_oracle=True).double()
        with patch("conv_rosa_transformer.joint_credit.propose_pairs", return_value=(pair, {})), \
             patch.object(rosa_core, "matching_stats", wraps=rosa_core.matching_stats) as matcher, \
             patch.object(rosa_core, "credits_numpy", wraps=rosa_core.credits_numpy) as credits:
            y = layer._hard_rosa_from_qkv(*tensors)
            self.assertEqual(matcher.call_count, 1)
            (y*g).sum().backward()
            self.assertEqual(credits.call_count, 1)
            self.assertEqual(matcher.call_count, 2)  # baseline plus one joint replay

    def test_normal_optimizer_step_and_state_dict(self):
        torch.manual_seed(95)
        rosa = _layer(8, 4, repair_samples=2, joint_pairs=2)
        model = nn.Sequential(nn.Linear(8, 8), rosa, nn.Tanh(), nn.Linear(8, 3))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        x = torch.randn(2, 7, 8)
        target = torch.randn(2, 7, 3)
        calls = []
        handle = rosa.register_forward_hook(lambda *args: calls.append(1))
        for _ in range(3):
            optimizer.zero_grad()
            loss = (model(x)-target).square().mean()
            loss.backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            optimizer.step()
        handle.remove()
        self.assertEqual(len(calls), 3)
        legacy = _layer(8, 4)
        self.assertEqual(set(legacy.state_dict()), set(rosa.state_dict()))
        legacy.load_state_dict(rosa.state_dict(), strict=True)


    def test_no_silent_double_backward(self):
        layer = _layer(4, 4, repair_samples=2, joint_pairs=2).double()
        tensors, g, _ = self.data()
        y = layer._hard_rosa_from_qkv(*tensors)
        first = torch.autograd.grad(y, tensors, g, create_graph=True)
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(sum(z.sum() for z in first), tensors)

    def test_grad_scaler_can_skip_an_overflow(self):
        torch.manual_seed(291)
        layer = _layer(4, 4, repair_samples=2, joint_pairs=1)
        optimizer = torch.optim.SGD(layer.parameters(), lr=.1)
        scaler = torch.amp.GradScaler("cpu")
        before = {name: value.detach().clone() for name, value in layer.named_parameters()}
        loss = layer(torch.randn(1, 5, 4)).sum() * float("inf")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        for name, parameter in layer.named_parameters():
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
        self.assertLess(scaler.get_scale(), 65536)




    def test_trust_metric_preserves_linear_value_credit_and_positive_scaling(self):
        tensors, g, _ = self.data()
        base = _layer(4, 4).double()
        trust = _layer(4, 4, trust_diagonal=1., joint_pairs=2).double()
        trust.load_state_dict(base.state_dict())
        plain_v = torch.autograd.grad(base._hard_rosa_from_qkv(*tensors), tensors, g)[2]
        y = trust._hard_rosa_from_qkv(*tensors)
        first = torch.autograd.grad(y, tensors, g, retain_graph=True)
        scaled = torch.autograd.grad(y, tensors, 8*g)
        torch.testing.assert_close(first[2], plain_v, atol=2e-12, rtol=2e-12)
        for a, b in zip(first, scaled):
            torch.testing.assert_close(b, 8*a, atol=2e-12, rtol=2e-12)

    def test_active_rows_use_only_nonzero_support_and_scale_consistently(self):
        g = np.zeros((20, 4))
        g[[2, 4, 7, 11, 17]] = np.arange(1, 21).reshape(5, 4)
        selected = _joint_rows(20, 3, g, seed=18)
        self.assertEqual(len(selected), 3)
        self.assertTrue(set(selected) <= {2, 4, 7, 11, 17})
        self.assertEqual(selected, _joint_rows(20, 3, 8*g, seed=18))
        self.assertEqual(_joint_rows(20, 10, g), [2, 4, 7, 11, 17])
        self.assertEqual(_joint_rows(20, 10, np.zeros_like(g)), [])

    def test_active_joint_training_accepts_standard_scaled_backward(self):
        tensors, g, _ = self.data()
        g[:, :4] = 0
        layer = _layer(4, 4, joint_pairs=2, joint_rows=2,
                           joint_row_selection="active", trust_diagonal=1.,
                           repair_samples=2).double()
        y = layer._hard_rosa_from_qkv(*tensors)
        a = torch.autograd.grad(y, tensors, g, retain_graph=True)
        b = torch.autograd.grad(y, tensors, 8*g)
        for x, z in zip(a, b):
            torch.testing.assert_close(z, 8*x, atol=2e-12, rtol=2e-12)

    def test_half_trust_keeps_aligned_antipodal_and_empty_jumps_descending(self):
        # These two actual output-set diameters are 10 and 55 respectively.
        # A trust coefficient 1 would flatten the first aligned antipodal
        # linear gain. A coefficient 1/2 retains at least half of it, and the
        # far-empty radius guard prevents overpenalizing the second example.
        cases = ((np.array([0., 0.]), np.array([-3., -4.]), np.array([3., 4.]), 5.),
                 (np.array([30., -40.]), np.array([30., -40.]), np.array([-3., 4.]), 27.5))
        for empty, baseline, changed, radius in cases:
            actual = float(_payload_radius(empty, np.array([3., 4.])))
            self.assertEqual(actual, radius)
            delta = changed-baseline
            gradient = -delta/np.linalg.norm(delta)
            metric = .5*np.linalg.norm(gradient)/actual
            linear = gradient@delta
            finite = linear+.5*metric*(delta@delta)
            self.assertLess(finite, 0)
            self.assertAlmostEqual(finite, .5*linear, places=12)

    def test_half_trust_restores_an_actual_two_bit_route_gain(self):
        # Q=[2,2,0], K=[0,3,0]. Either single Q flip stays at (-1,-1),
        # while their joint flip retrieves (+1,+1), reducing true MSE by 4.
        pair = [(Bit("Q", 2, 0), Bit("Q", 2, 1))]
        observed = []
        for coefficient in (1., .5):
            q = torch.tensor([[[-.2,.2],[-.2,.2],[-.2,-.2]]],
                             dtype=torch.float64, requires_grad=True)
            k = torch.tensor([[[-.2,-.2],[.2,.2],[-.2,-.2]]],
                             dtype=torch.float64, requires_grad=True)
            v = torch.tensor([[[-.2,-.2],[-.2,-.2],[.2,.2]]],
                             dtype=torch.float64, requires_grad=True)
            layer = _layer(2, 2, trust_diagonal=coefficient, joint_pairs=1, _python_oracle=True,
                           joint_row_selection="active").double()
            with torch.no_grad():
                layer.emb0.fill_(-1)
            with patch("conv_rosa_transformer.joint_credit.propose_pairs", return_value=(pair, {})):
                y = layer._hard_rosa_from_qkv(q, k, v)
                (.5*(y[:, -1]-1).square().sum()).backward()
            observed.append(q.grad[0, -1])
        torch.testing.assert_close(observed[0], torch.zeros(2, dtype=torch.float64), atol=0, rtol=0)
        p = torch.sigmoid(torch.tensor(-.2, dtype=torch.float64))
        expected = torch.full((2,), -4*p*p*(1-p), dtype=torch.float64)
        torch.testing.assert_close(observed[1], expected, atol=1e-14, rtol=1e-14)

    def test_checkpoint_recomputation_preserves_rng_and_gradients(self):
        from torch.utils.checkpoint import checkpoint
        for reentrant in (False, True):
            with self.subTest(use_reentrant=reentrant):
                options = dict(repair_samples=2, joint_pairs=2,
                    joint_row_selection="active", trust_diagonal=1.)
                torch.manual_seed(715)
                plain = _layer(8, 4, **options).double()
                recomputed = _layer(8, 4, **options).double()
                recomputed.load_state_dict(plain.state_dict())
                x = torch.randn(2, 7, 8, dtype=torch.float64, requires_grad=True)
                xc = x.detach().clone().requires_grad_()
                torch.manual_seed(811)
                plain(x).square().mean().backward()
                expected_rng = torch.get_rng_state().clone()
                torch.manual_seed(811)
                checkpoint(recomputed, xc, use_reentrant=reentrant,
                           preserve_rng_state=True).square().mean().backward()
                torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
                torch.testing.assert_close(xc.grad, x.grad, atol=1e-12, rtol=1e-12)
                for a, b in zip(plain.parameters(), recomputed.parameters()):
                    torch.testing.assert_close(a.grad, b.grad, atol=1e-12, rtol=1e-12)

    def test_cpu_autocast_finite_loss_scaling(self):
        options = dict(repair_samples=2, joint_pairs=2,
                       joint_row_selection="active", trust_diagonal=1.)
        torch.manual_seed(725)
        plain, scaled = _layer(8, 4, **options), _layer(8, 4, **options)
        scaled.load_state_dict(plain.state_dict())
        optimizer = torch.optim.SGD(scaled.parameters(), lr=.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=128.)
        x = torch.randn(2, 7, 8)
        torch.manual_seed(183)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = plain(x).square().mean()
        loss.backward()
        torch.manual_seed(183)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            other_loss = scaled(x).square().mean()
        scaler.scale(other_loss).backward()
        scaler.unscale_(optimizer)
        for a, b in zip(plain.parameters(), scaled.parameters()):
            self.assertTrue(bool(torch.isfinite(b.grad).all()))
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-6)

    def test_config_and_state_dict_restore_training_and_deterministic_eval(self):
        config = dict(channels=8, n_bits=4, repair_samples=2, joint_pairs=2,
                      joint_row_selection="active", trust_diagonal=1.)
        torch.manual_seed(171)
        original = _layer(**config).double()
        buffer = BytesIO()
        torch.save(dict(config=config, state_dict=original.state_dict()), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        restored = _layer(**saved["config"]).double()
        restored.load_state_dict(saved["state_dict"], strict=True)
        x = torch.randn(2, 7, 8, dtype=torch.float64)
        torch.manual_seed(134); a = original(x)
        torch.manual_seed(134); b = restored(x)
        a.square().mean().backward(); b.square().mean().backward()
        for p, q in zip(original.parameters(), restored.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
        original.eval(); restored.eval()
        state = torch.get_rng_state().clone()
        torch.testing.assert_close(original(x), restored(x), atol=0, rtol=0)
        torch.testing.assert_close(torch.get_rng_state(), state, atol=0, rtol=0)

    def test_invalid_options(self):
        for budget in (0, 5, -2):
            with self.assertRaises(ValueError):
                _layer(4, 4, repair_samples=budget)
        for options in ({"joint_pairs": -1}, {"joint_rows": 0}):
            with self.assertRaises(ValueError):
                _layer(4, 4, **options)
        with self.assertRaises(TypeError):
            _layer(4, 4, stochastic_values="yes")


if __name__ == "__main__":
    unittest.main()
