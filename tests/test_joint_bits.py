"""Joint conditional credits: exact corners and bounded structural proposals."""
import itertools
import unittest

import numpy as np

from research.joint_bits import (Bit, PairCredit, bits_from_streams,
    block_logit_gradient, block_objective, correlated_block_logit_gradient,
    correlated_block_objective, correlated_pair_logit_gradient, evaluate_pairs,
    forward_payload, pair_logit_gradient, propose_pairs)


def brute_routes(q, k):
    lengths, routes = [], []
    for t in range(len(q)):
        best = (0, -1)
        for r in range(t):
            length = 0
            while length <= r and q[t - length] == k[r - length]:
                length += 1
            if length:
                best = max(best, (length, r))
        lengths.append(best[0])
        routes.append(best[1])
    return lengths, routes


def payload(q, k, v, e0, e1):
    _, routes = brute_routes(q, k)
    return np.array([e0 if r < 0 else (2 * v[r + 1].astype(float) - 1) * e1
                     for r in routes])


class JointBitTests(unittest.TestCase):
    def test_zero_singletons_receive_pair_signal(self):
        q, k = [0, 3], [0, 0]
        v = np.array([[0, 0], [1, 1]])
        e0, e1 = np.zeros(2), np.ones(2)
        target = np.array([[0., 0.], [1., 1.]])
        y = payload(q, k, v, e0, e1)
        pairs, _ = propose_pairs(q, k, 2, rows=[1], allowed_streams=("Q",),
            allowed_owners={"Q": {1}}, max_pairs=1, seed=81)
        self.assertEqual(pairs, [(Bit("Q", 1, 0), Bit("Q", 1, 1))])
        evaluations, stats = evaluate_pairs(q, k, v, y - target, np.ones_like(y),
            e0, e1, pairs, matcher=brute_routes)
        result = evaluations[0]
        self.assertEqual(result.single_left, 0)
        self.assertEqual(result.single_right, 0)
        self.assertEqual(result.joint, -1)
        self.assertEqual(stats["joint_forwards"], 1)
        logits = np.full((3, 2, 2), 0.2)
        bits = bits_from_streams(q, k, v)
        grad = block_logit_gradient(logits, bits, np.zeros((3, 2, 2)), evaluations)
        self.assertGreater(grad[0, 1, 0], 0)
        self.assertGreater(grad[0, 1, 1], 0)
        reused, reused_stats = evaluate_pairs(q, k, v, y - target, np.ones_like(y),
            e0, e1, pairs, baseline_y=y, matcher=brute_routes)
        self.assertEqual(reused, evaluations)
        self.assertEqual(reused_stats["baseline_forwards"], 0)

    def test_two_mismatches_at_different_positions(self):
        pairs, _ = propose_pairs([0, 1, 1], [0, 0, 0], 1, rows=[2],
            allowed_streams=("Q",), allowed_owners={"Q": {1, 2}},
            max_pairs=2, bucket_cap=3, occurrence_cap=3, scan=3)
        self.assertIn((Bit("Q", 1, 0), Bit("Q", 2, 0)), pairs)

    def test_joint_credit_equals_true_diagonal_quadratic_loss(self):
        rng = np.random.default_rng(7263)
        for _ in range(35):
            n, d = 5, 3
            q, k = rng.integers(0, 1 << d, (2, n)).tolist()
            v = rng.integers(0, 2, (n, d))
            e0, e1 = rng.normal(size=(2, d))
            target, h = rng.normal(size=(n, d)), rng.uniform(0.1, 2, size=(n, d))
            y = payload(q, k, v, e0, e1)
            bits = [Bit(s, t, j) for s in ("Q", "K", "V")
                    for t in range(n) for j in range(d)]
            selected = rng.choice(len(bits), size=8, replace=False)
            pairs = [(bits[selected[j]], bits[selected[j + 1]]) for j in range(0, 8, 2)]
            evaluations, _ = evaluate_pairs(q, k, v, h * (y - target), h,
                e0, e1, pairs, matcher=brute_routes)
            base_loss = 0.5 * np.sum(h * (y - target) ** 2)
            for result in evaluations:
                qq, kk, vv = q.copy(), k.copy(), v.copy()
                for b in (result.left, result.right):
                    if b.stream == "Q": qq[b.position] ^= 1 << b.bit
                    elif b.stream == "K": kk[b.position] ^= 1 << b.bit
                    else: vv[b.position, b.bit] ^= 1
                yy = payload(qq, kk, vv, e0, e1)
                actual = 0.5 * np.sum(h * (yy - target) ** 2) - base_loss
                self.assertAlmostEqual(result.joint, actual, places=11)
                self.assertAlmostEqual(result.interaction,
                    result.joint - result.single_left - result.single_right, places=12)

    def test_block_gradient_matches_four_corner_expectation(self):
        rng = np.random.default_rng(901)
        for bit_left, bit_right in itertools.product((0, 1), repeat=2):
            bits = np.zeros((3, 1, 2), dtype=np.uint8)
            bits[0, 0] = (bit_left, bit_right)
            z = rng.normal(size=bits.shape)
            ci, cj, cij = rng.normal(size=3)
            singles = np.zeros_like(z); singles[0, 0] = (ci, cj)
            records = [PairCredit(Bit("Q", 0, 0), Bit("Q", 0, 1),
                                 cij, cij - ci - cj, ci, cj)]
            tau = 0.7

            def enumerated(zz):
                p = 1 / (1 + np.exp(-zz[0, 0] / tau))
                total = 0.
                for a, b in itertools.product((0, 1), repeat=2):
                    weight = (p[0] if a else 1 - p[0]) * (p[1] if b else 1 - p[1])
                    flips = (a != bit_left, b != bit_right)
                    value = {(False, False): 0., (True, False): ci,
                             (False, True): cj, (True, True): cij}[flips]
                    total += weight * value
                return total

            self.assertAlmostEqual(block_objective(z, bits, singles, records, tau=tau),
                                   enumerated(z), places=12)
            grad = block_logit_gradient(z, bits, singles, records, tau=tau)
            for j in range(2):
                plus, minus = z.copy(), z.copy()
                plus[0, 0, j] += 1e-6; minus[0, 0, j] -= 1e-6
                finite_difference = (enumerated(plus) - enumerated(minus)) / 2e-6
                self.assertAlmostEqual(grad[0, 0, j], finite_difference, places=8)

    def test_seeded_bounded_proposals_and_64bit_words(self):
        rng = np.random.default_rng(289)
        n, d, budget, capacity, occurrences, scan = 70, 64, 4, 3, 2, 5
        k = [int(x) for x in rng.integers(0, 1 << 63, n, dtype=np.uint64)]
        q = [x ^ (1 << 63) ^ 1 for x in k]
        args = dict(max_pairs=budget, seed=761, bucket_cap=capacity,
                    occurrence_cap=occurrences, scan=scan)
        first, stats = propose_pairs(q, k, d, **args)
        second, other = propose_pairs(q, k, d, **args)
        self.assertEqual(first, second)
        self.assertEqual({key: value for key, value in stats.items() if not key.endswith("seconds")},
                         {key: value for key, value in other.items() if not key.endswith("seconds")})
        self.assertLessEqual(len(first), budget)
        self.assertEqual(len({b for pair in first for b in pair}), 2 * len(first))
        self.assertLessEqual(stats["examined_alignments"], n * 3 * capacity * (occurrences + 1))
        self.assertLessEqual(stats["symbol_comparisons"], stats["examined_alignments"] * scan)
        self.assertLessEqual(stats["retained_candidates"], 8 * budget)

    def test_overlapping_blocks_are_rejected(self):
        records = [PairCredit(Bit("Q", 0, 0), Bit("Q", 0, 1), 0, 1, 0, 0),
                   PairCredit(Bit("Q", 0, 1), Bit("K", 0, 1), 0, 1, 0, 0)]
        z = np.zeros((3, 1, 2))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            block_logit_gradient(z, z, z, records)

    def test_correlated_envelope_equals_four_corner_probability_lp(self):
        # Enumerate both extreme feasible distributions, using their four
        # probabilities explicitly rather than the implementation's formula.
        left, right = Bit("Q", 0, 0), Bit("Q", 0, 1)
        for ci, cj, cij in ((1., 2., -2.), (-2., -1., 3.),
                           (2., -3., 1.), (1., -2., -1.)):
            singles = np.zeros((3, 1, 2)); singles[0, 0] = (ci, cj)
            records = [PairCredit(left, right, cij, cij-ci-cj, ci, cj)]
            for baseline in itertools.product((0, 1), repeat=2):
                bits = np.zeros_like(singles, dtype=np.uint8); bits[0, 0] = baseline
                for a, b in itertools.product((0., .13, .5, .81, 1.), repeat=2):
                    z = np.zeros_like(singles)
                    for j, alpha in enumerate((a, b)):
                        p = alpha if baseline[j] == 0 else 1-alpha
                        z[0, 0, j] = -1000 if p == 0 else 1000 if p == 1 else np.log(p/(1-p))
                    costs = []
                    for both in (max(0., a+b-1.), min(a, b)):
                        probabilities = np.array([1-a-b+both, a-both, b-both, both])
                        self.assertGreaterEqual(float(probabilities.min()), -1e-14)
                        costs.append(float(probabilities @ np.array([0., ci, cj, cij])))
                    actual = correlated_block_objective(z, bits, singles, records)
                    self.assertAlmostEqual(actual, min(costs), places=11)
                    self.assertLessEqual(actual, block_objective(z, bits, singles, records)+1e-12)

    def test_correlated_gradient_matches_smooth_finite_differences(self):
        rng = np.random.default_rng(71083)
        left, right = Bit("Q", 0, 0), Bit("K", 0, 1)
        for repeat in range(24):
            z, singles = rng.normal(size=(2, 3, 1, 2))
            bits = rng.integers(0, 2, size=z.shape)
            ci, cj = singles[left.index()], singles[right.index()]
            interaction = (-1 if repeat % 2 else 1) * rng.uniform(.5, 3.)
            records = [PairCredit(left, right, ci+cj+interaction, interaction, ci, cj)]
            grad = correlated_block_logit_gradient(z, bits, singles, records, tau=.63)
            for index in np.ndindex(z.shape):
                plus, minus = z.copy(), z.copy()
                plus[index] += 1e-6; minus[index] -= 1e-6
                fd = (correlated_block_objective(plus, bits, singles, records, tau=.63)
                      - correlated_block_objective(minus, bits, singles, records, tau=.63)) / 2e-6
                self.assertAlmostEqual(grad[index], fd, places=8)

    def test_correlated_ties_are_legal_minimum_norm_subgradients(self):
        rng = np.random.default_rng(90203)
        left, right = Bit("Q", 0, 0), Bit("Q", 0, 1)
        # Include interior and endpoint minima of each subgradient segment.
        for ci, cj, interaction in ((10., 1., -12.), (-10., 1., -2.),
                (3., -8., -2.), (-3., -1., 6.), (3., 5., 2.), (-8., -4., 2.)):
            singles = np.zeros((3, 1, 2)); singles[0, 0] = (ci, cj)
            cij = ci+cj+interaction
            records = [PairCredit(left, right, cij, interaction, ci, cj)]
            z = np.zeros_like(singles)
            marginal_grad = correlated_block_logit_gradient(z, z, singles, records)[0, 0] / .25

            def explicit_lp(a, b):
                return min(np.array([1-a-b+rho, a-rho, b-rho, rho])
                           @ np.array([0., ci, cj, cij])
                           for rho in (max(0., a+b-1.), min(a, b)))

            center = explicit_lp(.5, .5)
            for point in rng.uniform(size=(40, 2)):
                self.assertGreaterEqual(explicit_lp(*point)-center
                    - float(marginal_grad @ (point-.5)), -1e-12)
            edges = (np.array([[ci+interaction, cj], [ci, cj+interaction]])
                     if interaction < 0 else np.array([[ci, cj], [ci+interaction, cj+interaction]]))
            segment = edges[0]+np.linspace(0., 1., 10001)[:, None]*(edges[1]-edges[0])
            self.assertLessEqual(float(marginal_grad @ marginal_grad),
                float(np.min(np.sum(segment*segment, axis=1)))+1e-12)

    def test_correlated_small_marginals_escape_positive_singleton_barrier(self):
        # Each individual edit costs loss; doing both improves it. The costs
        # are intentionally asymmetric: an arbitrary half-interaction rule
        # would still push the expensive bit the wrong way.
        ci, cj, cij, alpha = 10., 1., -1., .001
        left, right = Bit("Q", 0, 0), Bit("Q", 0, 1)
        singles = np.zeros((3, 1, 2)); singles[0, 0] = (ci, cj)
        records = [PairCredit(left, right, cij, cij-ci-cj, ci, cj)]
        bits = np.zeros_like(singles)
        z = np.zeros_like(singles); z[0, 0] = np.log(alpha/(1-alpha))
        independent = block_logit_gradient(z, bits, singles, records)
        correlated = correlated_block_logit_gradient(z, bits, singles, records)
        self.assertTrue(np.all(independent[0, 0] > 0))
        self.assertTrue(np.all(correlated[0, 0] < 0))
        np.testing.assert_allclose(correlated[0, 0], .5*cij*alpha*(1-alpha), atol=1e-15)
        self.assertLess(correlated_block_objective(z-correlated, bits, singles, records),
                        correlated_block_objective(z, bits, singles, records))
        for function in (correlated_block_objective, correlated_block_logit_gradient):
            with self.assertRaisesRegex(ValueError, "disjoint"):
                function(z, bits, singles, records+records)

    def test_vectorized_unpack_preserves_unsigned_bit_63(self):
        rng = np.random.default_rng(39127)
        for d in (1,2,4,8,16,32,64):
            mask = (1 << d)-1
            q = [0,mask]+[int(x)&mask for x in rng.bit_generator.random_raw(8)]
            k = [mask,0]+[int(x)&mask for x in rng.bit_generator.random_raw(8)]
            v = rng.integers(0,2,size=(len(q),d),dtype=np.uint8)
            expected = np.asarray([[[((word >> j)&1) for j in range(d)] for word in stream]
                                   for stream in (q,k)],dtype=np.uint8)
            actual = bits_from_streams(q,k,v)
            np.testing.assert_array_equal(actual[:2],expected)
            np.testing.assert_array_equal(actual[2],v)

    def test_sparse_pair_replacements_are_bitwise_equal_to_full_reference(self):
        rng = np.random.default_rng(50631)
        checked = 0
        for d in (2,4,64):
            n = 3
            owners = [Bit(s,t,j) for s in ("Q","K","V") for t in range(n) for j in range(d)]
            for repetition in range(20):
                z = rng.normal(size=(3,n,d))
                if repetition%3 == 0:
                    z.fill(0.)  # both negative/positive-interaction kink branches
                bits = rng.integers(0,2,size=z.shape)
                singles = rng.normal(size=z.shape)
                selected = rng.choice(len(owners),size=8,replace=False)
                records = []
                for offset in range(0,8,2):
                    left,right = owners[selected[offset]],owners[selected[offset+1]]
                    ci,cj = float(singles[left.index()]),float(singles[right.index()])
                    interaction = float(rng.normal())
                    records.append(PairCredit(left,right,ci+cj+interaction,interaction,ci,cj))
                pair_z = np.asarray([[z[r.left.index()],z[r.right.index()]] for r in records])
                pair_bits = np.asarray([[bits[r.left.index()],bits[r.right.index()]] for r in records])
                tau = (.4,1.,2.)[repetition%3]
                for full_function,sparse_function in (
                        (block_logit_gradient,pair_logit_gradient),
                        (correlated_block_logit_gradient,correlated_pair_logit_gradient)):
                    full = full_function(z,bits,singles,records,tau=tau)
                    expected = np.asarray([[full[r.left.index()],full[r.right.index()]] for r in records])
                    actual = sparse_function(pair_z,pair_bits,records,tau=tau)
                    # Compare IEEE representations, including signed exact zero.
                    np.testing.assert_array_equal(actual.view(np.uint64),expected.view(np.uint64))
                    checked += actual.size
        self.assertEqual(checked,960)
        for function in (pair_logit_gradient,correlated_pair_logit_gradient):
            self.assertEqual(function(np.zeros((0,2)),np.zeros((0,2)),[]).shape,(0,2))
            with self.assertRaisesRegex(ValueError,"disjoint"):
                function(np.vstack((pair_z,pair_z)),np.vstack((pair_bits,pair_bits)),records+records)


if __name__ == "__main__":
    unittest.main()
