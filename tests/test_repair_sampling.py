"""Selective exact repair lanes and their deletion-control-variate estimator."""
import itertools
import unittest

import numpy as np

from conv_rosa_transformer import rosa_core as core


class RepairSamplingTests(unittest.TestCase):
    @staticmethod
    def case(n, d, rng, family="random"):
        q = rng.bit_generator.random_raw(n)
        k = rng.bit_generator.random_raw(n)
        if d < 64:
            q &= np.uint64((1 << d) - 1)
            k &= np.uint64((1 << d) - 1)
        if family in {"same", "shadow"}:
            q[:] = 0
            k[:] = 0
            if family == "shadow" and n:
                k[0] = np.uint64(1) << np.uint64(d - 1)
        elif family == "paired":
            k = q ^ (np.uint64(1) << np.uint64(d - 1))
        v = rng.integers(0, 2, (n, d), dtype=np.uint8)
        g = rng.normal(size=(n, d))
        e0, e1 = rng.normal(size=(2, d))
        return q, k, v, g, e0, e1

    def test_exact_selected_and_baseline_invariant(self):
        rng = np.random.default_rng(2710801)
        for trial in range(48):
            d = (1, 3, 8, 64)[trial % 4]
            n = (0, 1, 7, 23)[(trial // 4) % 4]
            family = ("random", "same", "shadow", "paired")[(trial // 3) % 4]
            args = self.case(n, d, rng, family)
            h = rng.uniform(0.01, 2, (n, d)) if trial % 2 else None
            full = (core.credits_numpy(*args) if h is None else
                    core.quadratic_credits_numpy(*args[:4], h, *args[4:]))
            # Quadratic baselines may differ, so compare masks under same model.
            zero = core.selected_credits_numpy(*args, [], diag_y=h)
            for selected in ([], list(range(d)), sorted(set((0, d - 1))),
                             rng.choice(d, max(1, d // 2), replace=False).tolist()):
                got = core.selected_credits_numpy(*args, selected, diag_y=h)
                self.assertEqual(got[3:5], full[3:5])
                np.testing.assert_array_equal(got[2].view(np.uint64), full[2].view(np.uint64))
                for s in range(2):
                    np.testing.assert_array_equal(got[6+s].view(np.uint64), zero[6+s].view(np.uint64))
                    for j in range(d):
                        target = full[s][:, j] if j in selected else zero[6+s]
                        np.testing.assert_array_equal(got[s][:, j].view(np.uint64), target.view(np.uint64))

    def test_enumerated_expectation_and_variance(self):
        rng = np.random.default_rng(2710802)
        args = self.case(19, 4, rng)
        full = core.credits_numpy(*args)
        base = core.selected_credits_numpy(*args, [])
        c = np.stack(full[:2])
        b = np.stack([np.broadcast_to(base[s+6][:, None], c[s].shape) for s in range(2)])
        for r in (1, 2, 3, 4):
            p = r / 4
            samples = [np.stack(core.repair_sampled_credits_numpy(*args, subset, p)[:2])
                       for subset in itertools.combinations(range(4), r)]
            np.testing.assert_allclose(np.mean(samples, axis=0), c, rtol=1e-13, atol=1e-13)
            empirical = float(np.mean([np.sum((sample-c)**2) for sample in samples]))
            expected = (1-p)/p * float(np.sum((c-b)**2))
            self.assertAlmostEqual(empirical, expected, delta=max(1e-10, expected*1e-13))
        # Nonuniform marginals: force lane 0 and sample one of the rest.
        prob = np.array([1., 1/3, 1/3, 1/3])
        values = [np.stack(core.repair_sampled_credits_numpy(*args, [0, j], prob)[:2])
                  for j in (1, 2, 3)]
        np.testing.assert_allclose(np.mean(values, axis=0), c, rtol=1e-13, atol=1e-13)

    def test_no_repair_has_zero_sampling_noise(self):
        rng = np.random.default_rng(2710803)
        args = self.case(37, 8, rng, "same")
        full = core.credits_numpy(*args)
        self.assertTrue(any(np.any(x) for x in full[:2]))
        for selected in ([0], [7], [1, 2, 5]):
            got = core.repair_sampled_credits_numpy(*args, selected, len(selected)/8)
            for actual, exact in zip(got[:3], full[:3]):
                np.testing.assert_array_equal(actual.view(np.uint64), exact.view(np.uint64))
            self.assertEqual(got[5]["sam_active_advances"], 0)

    def test_selected_work_is_actually_skipped(self):
        rng = np.random.default_rng(2710804)
        args = self.case(101, 4, rng)
        full = core.selected_credits_numpy(*args, range(4))
        subset = core.selected_credits_numpy(*args, [0])
        self.assertEqual(subset[5]["root_repair_entries"], len(np.unique(args[0])))
        self.assertEqual(full[5]["root_repair_entries"], 4 * len(np.unique(args[0])))
        self.assertLess(subset[5]["sam_active_advances"], full[5]["sam_active_advances"])
        self.assertLess(subset[5]["visited_alignment_pairs"], full[5]["visited_alignment_pairs"])

    def test_causal_support_certificates_and_forced_lanes(self):
        rng = np.random.default_rng(2710806)
        for trial in range(28):
            d = (3, 8, 64)[trial % 3]
            n = (0, 1, 13, 29)[trial % 4]
            args = self.case(n, d, rng, ("same", "shadow", "paired", "random")[trial % 4])
            q, k = args[:2]
            expected_support = set()
            for t in range(n):
                for r in range(t):
                    xor = int(q[t]) ^ int(k[r])
                    if xor and xor & (xor - 1) == 0:
                        expected_support.add(xor.bit_length() - 1)
            exact = core.credits_numpy(*args)
            forced = [d - 1]
            got = core.adaptive_repair_sampled_credits_numpy(*args, 1, 9281, forced)
            self.assertEqual(set(got[5]["supported_repair_bits"]), expected_support)
            self.assertTrue((set(forced) & expected_support) <= set(got[5]["repair_bits"]))
            self.assertEqual(got[3:5], exact[3:5])
            np.testing.assert_array_equal(got[2].view(np.uint64), exact[2].view(np.uint64))
            p = got[5]["repair_inclusion_probabilities"]
            for bit in range(d):
                if bit not in expected_support or bit in forced or p[bit] == 1:
                    for x, y in zip(got[:2], exact[:2]):
                        np.testing.assert_array_equal(x[:, bit].view(np.uint64), y[:, bit].view(np.uint64))
            repeat = core.adaptive_repair_sampled_credits_numpy(*args, 1, 9281, forced)
            self.assertEqual(got[5], repeat[5])
            for a, b in zip(got[:3], repeat[:3]):
                np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
        # The only one-bit match is in K's future: it cannot support a repair.
        args = ([0, 0], [0, 128], np.ones((2, 8), np.uint8), np.ones((2, 8)),
                np.zeros(8), np.ones(8))
        got = core.adaptive_repair_sampled_credits_numpy(*args, 1, 0)
        self.assertEqual(got[5]["supported_repair_bits"], [])
        self.assertTrue(got[5]["repair_sampling_exact"])

    def test_native_support_sampling_expectation(self):
        rng = np.random.default_rng(2710807)
        args = self.case(37, 3, rng)
        exact = core.credits_numpy(*args)
        # Force one lane, let a target budget of one sample one extra optional
        # lane. Each of the two remaining lanes has probability one half.
        outcomes = {}
        for seed in range(128):
            got = core.adaptive_repair_sampled_credits_numpy(*args, 1, seed, [0])
            self.assertEqual(got[5]["supported_repair_bits"], [0, 1, 2])
            self.assertEqual(got[5]["repair_inclusion_probabilities"], [1., .5, .5])
            self.assertEqual(got[5]["repair_lane_count"], 2)
            outcomes[tuple(got[5]["repair_bits"])] = np.stack(got[:2])
        self.assertEqual(set(outcomes), {(0, 1), (0, 2)})
        np.testing.assert_allclose(np.mean(list(outcomes.values()), axis=0),
                                   np.stack(exact[:2]), atol=1e-13, rtol=1e-13)
        for bad in ((0, 0, []), (4, 0, []), (1, -1, []), (1, 2**64, []),
                    (1, 0, [3]), (1, 0, [0, 0])):
            with self.assertRaises(ValueError):
                core.adaptive_repair_sampled_credits_numpy(*args, *bad)

    def test_validation_and_high_bit(self):
        rng = np.random.default_rng(2710805)
        args = self.case(7, 64, rng, "paired")
        full = core.credits_numpy(*args)
        got = core.selected_credits_numpy(*args, [63])
        self.assertEqual(got[5]["repair_bits"], [63])
        for x, y in zip(got[:2], full[:2]):
            np.testing.assert_array_equal(x[:, 63], y[:, 63])
        for bad in ([64], [-1], [0, 0], [0.5]):
            with self.assertRaises(ValueError):
                core.selected_credits_numpy(*args, bad)
        for prob in (0., -1., 1.1, np.nan, [0.2, 0.2], 1.0):
            with self.assertRaises(ValueError):
                core.repair_sampled_credits_numpy(*args, [63], prob)


if __name__ == "__main__":
    unittest.main()
