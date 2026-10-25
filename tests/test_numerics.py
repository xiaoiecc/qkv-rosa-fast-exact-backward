"""Structural zero identities and independently computed cancellation witnesses.

The saved nonconstant cases assert their documented target Q owner, not every
ordinary Q/K/V sum at a condition number near machine precision. Expansion
tests separately check exact binary64 event accumulation.
"""
from pathlib import Path
import json
import unittest

import numpy as np

from conv_rosa_transformer import rosa_core
from tests._support import COVERAGE
from tests.oracle import hard_outputs


class NumericalTests(unittest.TestCase):
    def test_minimal_constant_witnesses(self):
        # A huge unmatched row must not contaminate later corridor intervals.
        q = np.zeros(12, dtype=np.uint64)
        k = q.copy()
        k[0] = 1
        g = np.ones((12, 1))
        g[0] = 1e16
        inputs = [(q, k, np.ones((12, 1), dtype=np.uint8), g, np.ones(1), np.ones(1), 1)]
        # Two-bucket owner contraction preserves a structural zero through
        # cancelling large coordinates and later small gradient coordinates.
        q = np.zeros(6, dtype=np.uint64)
        g = np.zeros((6, 2))
        g[4], g[5] = [1e16, -1e16], [1., 1.]
        inputs.append((q, q.copy(), np.ones((6, 2), dtype=np.uint8), g, np.ones(2), np.ones(2), 2))
        for case in inputs:
            for value in rosa_core.credits_numpy(*case)[:2]:
                self.assertTrue(np.all(value == 0))
        COVERAGE["minimal_constant_witnesses"] = {"inputs": len(inputs)}

    def test_constant_forward_has_zero_qk_credit(self):
        rng = np.random.default_rng(202610183173)
        cases = elements = 0
        scales = np.array([0., 1., -1., 1e8, -1e8, 1e16, -1e16,
                           1e32, -1e32, 1e64, -1e64, 1e100, -1e100])
        for n in (0, 1, 2, 3, 5, 6, 12, 31, 64):
            for d in (1, 2, 3, 8, 64):
                positions = list(dict.fromkeys((0, d - 1, d // 2)))
                words = np.array([sum(((i >> j) & 1) << p
                                      for j, p in enumerate(positions))
                                  for i in range(1 << len(positions))], dtype=np.uint64)
                for family in ("same", "shadow", "periodic", "random"):
                    q = np.zeros(n, dtype=np.uint64)
                    k = q.copy()
                    if family == "shadow" and n:
                        k[0] = np.uint64(1) << np.uint64(d - 1)
                    elif family == "periodic":
                        q = words[np.arange(n) % min(len(words), 4)].copy()
                        k = q.copy()
                        if n:
                            k[n // 2] ^= np.uint64(1) << np.uint64(d - 1)
                    elif family == "random":
                        q = words[rng.integers(len(words), size=n)]
                        k = words[rng.integers(len(words), size=n)]
                    bits = rng.integers(0, 2, size=d, dtype=np.uint8)
                    v = np.broadcast_to(bits, (n, d)).copy()
                    e1 = rng.uniform(.125, 2., size=d)
                    e0 = np.where(bits != 0, e1, -e1)
                    for mode in ("ordinary", "mixed", "outside"):
                        if mode == "ordinary":
                            g = rng.normal(size=(n, d))
                        elif mode == "mixed":
                            g = rng.choice(scales, size=(n, d))
                        else:
                            g = np.ones((n, d))
                            if n:
                                g[0] = rng.choice((-1e100, 1e100), size=d)
                        out = rosa_core.credits_numpy(q, k, v, g, e0, e1, d)
                        for array in out[:2]:
                            self.assertTrue(np.all(array == 0), (n, d, family, mode))
                            elements += array.size
                        cases += 1
        COVERAGE["constant_forward"] = {"inputs": cases, "strict_zero_qk_elements": elements}

    def test_snapshot_and_corridor_cancellation(self):
        fixtures = []
        q = np.zeros(6, dtype=np.uint64)
        k = q.copy()
        k[0] = 1
        fixtures.append((q, k, [1, 1, 0, 1, 0, 1],
                         [0., 0., 1e16, -1., 1., 1e16], 1., 2, 4.))
        for exponent in (54, 100, 300, 600, 1000):
            for tiny_exponent in (0, -100, -500, -1000):
                n = 8
                q = np.zeros(n, dtype=np.uint64)
                k = q.copy()
                k[0] = 1
                v = np.ones(n, dtype=np.uint8)
                v[-1] = 0
                g = np.zeros(n)
                g[3] = -np.ldexp(1., exponent)
                tiny = np.ldexp(1., tiny_exponent)
                g[-2], g[-1] = tiny, -tiny * .5
                fixtures.append((q, k, v, g, 0., n - 2, -tiny))
        for powers in ((1000, 700, 400, 100, -200),
                       (-200, 100, 400, 700, 1000), (1000, -1000, 500, -500, 0)):
            n = 16
            q = np.zeros(n, dtype=np.uint64)
            k = q.copy()
            k[0] = 1
            v = np.ones(n, dtype=np.uint8)
            v[-1] = 0
            g = np.zeros(n)
            tiny = np.ldexp(1., -900)
            for row, power in zip((3, 5, 7, 9, 11), powers):
                g[row] = -np.ldexp(1., power)
            g[-2], g[-1] = tiny, -tiny * .5
            fixtures.append((q, k, v, g, 0., n - 2, -tiny))
        for q, k, v, g, e0, owner, exact in fixtures:
            n = len(q)
            actual = rosa_core.credits_numpy(
                q, k, np.asarray(v, dtype=np.uint8).reshape(n, 1),
                np.asarray(g, dtype=np.float64).reshape(n, 1),
                np.array([e0]), np.ones(1), 1)[0][owner, 0]
            self.assertEqual(actual, exact)
        COVERAGE["snapshot_cancellation"] = {"exact_target_cases": len(fixtures)}

    def test_saved_nonconstant_targets(self):
        path = Path(__file__).parent / "fixtures" / "numeric_credits.json"
        fixtures = json.loads(path.read_text())
        for case in fixtures:
            q, k, v, g = (case[key] for key in ("q", "k", "v", "g"))
            changed = q.copy()
            changed[case["owner"]] ^= 1
            vb = [[bit] for bit in v]
            _, base = hard_outputs(q, k, vb, [case["emb0"]], [1])
            _, flipped = hard_outputs(changed, k, vb, [case["emb0"]], [1])
            exact = sum(gg * (ff[0] - bb[0]) for gg, ff, bb in zip(g, flipped, base))
            self.assertEqual(exact, case["expected"])
            n = len(q)
            actual = rosa_core.credits_numpy(
                q, k, np.asarray(v, dtype=np.uint8).reshape(n, 1),
                np.asarray(g, dtype=np.float64).reshape(n, 1),
                np.array([float(case["emb0"])]), np.ones(1), 1)[0][case["owner"], 0]
            self.assertEqual(actual, exact, case["name"])
        COVERAGE["nonconstant_cancellation"] = {"exact_target_cases": len(fixtures)}


if __name__ == "__main__":
    unittest.main()
