"""Ordinary model integration for the sole native training engine.

Finite-credit correctness is checked independently in test_subquadratic_engine;
these checks cover ownership, autograd, checkpointing, RNG and public options.
"""
from __future__ import annotations

import unittest

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from conv_rosa_transformer.rosa_layer import NBitQKVRosa
from conv_rosa_transformer.native_training import (
    make_options, training_forward_numpy, training_backward_numpy,
)


class NativeTrainingTests(unittest.TestCase):
    def test_forward_cache_owns_inputs_and_options(self):
        for order in ("first", "second"):
            rng = np.random.default_rng(382)
            logits = [rng.normal(size=(1, 2, 7, 4)) for _ in range(3)]
            embeddings = [rng.normal(size=(2, 4)) for _ in range(2)]
            options = make_options(credit_order=order, randomized=True,
                                   proposal_seed=17, repair_seed=23)
            _, cache, _ = training_forward_numpy(*logits, *embeddings, options=options)
            gradient = rng.normal(size=logits[0].shape)
            before = training_backward_numpy(cache, gradient, trace=True)
            cached = (cache.q_logits, cache.k_logits, cache.v_logits, cache.emb0, cache.emb1)
            for original, saved in zip(logits + embeddings, cached):
                self.assertFalse(np.shares_memory(original, saved))
            self.assertIsNot(options, cache.options)
            saved_options = bytes(cache.options)
            for original in logits + embeddings:
                original.fill(np.nan)
            for field, _ in options._fields_:
                setattr(options, field, 0)
            self.assertEqual(bytes(cache.options), saved_options)
            after = training_backward_numpy(cache, gradient, trace=True)
            for a, b in zip(before[:5], after[:5]):
                self.assertEqual(a.tobytes(), b.tobytes())
            self.assertEqual(before[-1].keys(), after[-1].keys())
            for key in before[-1]:
                np.testing.assert_array_equal(before[-1][key], after[-1][key])

    def test_native_exact_cache_preserves_all_arithmetic(self):
        rng = np.random.default_rng(817)
        for n, d in ((0, 4), (9, 4), (70, 8), (15, 64)):
            for order in ("first", "second"):
                logits = [rng.normal(size=(2, 2, n, d)) for _ in range(3)]
                e0, e1 = rng.normal(size=(2, d)), rng.normal(size=(2, d))
                g = rng.normal(size=(2, 2, n, d))
                outputs = []
                for randomized in (False, True):
                    y, cache, _ = training_forward_numpy(*logits, e0, e1, credit_order=order,
                                                        randomized=randomized, proposal_seed=778,
                                                        repair_seed=889, joint_rows=4)
                    outputs.append((y, training_backward_numpy(cache, g, trace=True)))
                self.assertEqual(outputs[0][0].tobytes(), outputs[1][0].tobytes())
                for a, b in zip(outputs[0][1][:5], outputs[1][1][:5]):
                    self.assertEqual(a.tobytes(), b.tobytes())
                for key in ("pair_codes", "pair_costs"):
                    self.assertEqual(outputs[0][1][-1][key].tobytes(), outputs[1][1][-1][key].tobytes())

    def test_one_model_forward_and_backward(self):
        for order in ("first", "second"):
            for mode in ("exact", "approximate"):
                torch.manual_seed(25)
                layer = NBitQKVRosa(8, 4, credit_order=order, backward_mode=mode,
                                    approximation_rows=4).double()
                x = torch.randn(3, 17, 8, dtype=torch.float64, requires_grad=True)
                calls = [0, 0]
                forward = layer.register_forward_hook(lambda *args: calls.__setitem__(0, calls[0] + 1))
                backward = layer.register_full_backward_hook(lambda *args: calls.__setitem__(1, calls[1] + 1))
                layer(x).square().mean().backward()
                forward.remove(); backward.remove()
                self.assertEqual(calls, [1, 1])
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in layer.parameters()))
                self.assertEqual(layer.last_forward_work["baseline_matching_calls"], 6)

    def test_framework_rng_is_forward_only_and_cache_switch_invariant(self):
        for order in ("first", "second"):
            states, gradients = [], []
            for randomized in (False, True):
                torch.manual_seed(814)
                layer = NBitQKVRosa(8, 4, credit_order=order, randomized=randomized).double()
                x = torch.randn(2, 11, 8, dtype=torch.float64, requires_grad=True)
                before = torch.get_rng_state().clone()
                y = layer(x)
                after = torch.get_rng_state().clone()
                y.square().sum().backward()
                self.assertTrue(torch.equal(after, torch.get_rng_state()))
                self.assertEqual(torch.equal(before, after), order == "first")
                states.append(after)
                gradients.append([x.grad] + [p.grad for p in layer.parameters()])
            self.assertTrue(torch.equal(*states))
            for a, b in zip(*gradients):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_checkpoint_preserves_normal_gradients(self):
        for mode in ("exact", "approximate"):
            torch.manual_seed(511)
            plain = NBitQKVRosa(8, 4, backward_mode=mode, approximation_rows=3).double()
            recomputed = NBitQKVRosa(8, 4, backward_mode=mode, approximation_rows=3).double()
            recomputed.load_state_dict(plain.state_dict())
            x = torch.randn(2, 13, 8, dtype=torch.float64, requires_grad=True)
            other = x.detach().clone().requires_grad_()
            torch.manual_seed(907)
            a = plain(x)
            a.square().mean().backward()
            torch.manual_seed(907)
            b = checkpoint(recomputed, other, use_reentrant=False, preserve_rng_state=True)
            b.square().mean().backward()
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            torch.testing.assert_close(x.grad, other.grad, atol=0, rtol=0)
            for p, q in zip(plain.parameters(), recomputed.parameters()):
                torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)

    def test_saved_configuration_is_used_by_backward(self):
        torch.manual_seed(51)
        original = NBitQKVRosa(8, 4, backward_mode="approximate", approximation_rows=3).double()
        changed = NBitQKVRosa(8, 4, backward_mode="approximate", approximation_rows=3).double()
        changed.load_state_dict(original.state_dict())
        x = torch.randn(2, 23, 8, dtype=torch.float64, requires_grad=True)
        other = x.detach().clone().requires_grad_()
        torch.manual_seed(818)
        a = original(x)
        a.square().sum().backward()
        torch.manual_seed(818)
        b = changed(other)
        changed.credit_order = "first"
        changed.joint_pairs = 0
        changed.trust_diagonal = 0
        changed.randomized = False
        changed.backward_mode = "exact"
        changed.approximation_rows = 1
        b.square().sum().backward()
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(x.grad, other.grad, atol=0, rtol=0)
        for p, q in zip(original.parameters(), changed.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
        self.assertEqual(changed.last_forward_work["backward"]["backward_mode"], "approximate")

    def test_cpu_autocast_and_nonfinite_adjoint(self):
        layer = NBitQKVRosa(8, 4)
        x = torch.randn(2, 9, 8, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = layer(x)
            loss = y.float().square().mean()
        loss.backward()
        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in layer.parameters()))
        x = torch.randn(1, 5, 8, requires_grad=True)
        y = layer(x)
        y.backward(torch.full_like(y, float("nan")))
        self.assertTrue(torch.isnan(x.grad).any())

    def test_public_options_and_removed_selectors(self):
        first = NBitQKVRosa(8, 4, credit_order="first", randomized=True)
        self.assertEqual((first.joint_pairs, first.trust_diagonal), (0, 0.))
        second = NBitQKVRosa(8, 4, credit_order="second")
        self.assertEqual((second.joint_pairs, second.trust_diagonal), (1, .5))
        for kwargs in ({"credit_order": "invalid"}, {"credit_order": "first", "joint_pairs": 1},
                       {"training_rule": "local_mixture"}, {"stochastic_values": True}):
            with self.assertRaises(ValueError):
                NBitQKVRosa(8, 4, **kwargs)
        for kwargs in ({"credit_engine": "legacy"}, {"repair_samples": 1}):
            with self.assertRaises(TypeError):
                NBitQKVRosa(8, 4, **kwargs)
        with self.assertRaises(TypeError):
            NBitQKVRosa(8, 4, randomized="yes")

    def test_native_joint_corner_crosses_a_singleton_barrier(self):
        q = np.array([[[[-.2, .2], [-.2, .2], [-.2, -.2]]]])
        k = np.array([[[[-.2, -.2], [.2, .2], [-.2, -.2]]]])
        v = np.array([[[[-.2, -.2], [-.2, -.2], [.2, .2]]]])
        e0, e1 = -np.ones((1, 2)), np.ones((1, 2))
        g = np.zeros_like(q); g[:, :, 2] = -2
        chosen = None
        for seed in range(30):
            _, cache, _ = training_forward_numpy(q, k, v, e0, e1, proposal_seed=seed)
            result = training_backward_numpy(cache, g, trace=True)
            if tuple(result[-1]["pair_codes"][0, 0]) == (10, 11):
                chosen = seed
                break
        self.assertIsNotNone(chosen)
        probability = 1 / (1 + np.exp(.2))
        np.testing.assert_allclose(result[0][0, 0, 2], [-4 * probability**2 * (1 - probability)] * 2,
                                   rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result[-1]["pair_costs"][0, 0], [0, 0, -4, -4], rtol=0, atol=1e-13)
        _, cache, _ = training_forward_numpy(q, k, v, e0, e1, proposal_seed=chosen, trust_diagonal=1.)
        penalized = training_backward_numpy(cache, g, trace=True)
        np.testing.assert_array_equal(penalized[0][0, 0, 2], np.zeros(2))


if __name__ == "__main__":
    unittest.main()
