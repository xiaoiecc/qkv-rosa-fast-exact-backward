"""Independent full-credit checks for the structural engine (NumPy only).

The oracle edits the packed inputs, recomputes literal causal matching with a
quadratic LCS table, and evaluates the frozen output loss. It shares no route,
deletion, repair, feature-lift or contraction helpers with the implementation.
The proposal trace only chooses which two coordinates the oracle should edit;
all three corner costs and the resulting mixture are recomputed independently.
"""
from __future__ import annotations

import math
import json
import subprocess
import ctypes
import unittest

import numpy as np

from conv_rosa_transformer.native_training import (
    training_forward_numpy, training_backward_numpy,
)


def literal_routes(q, k):
    """Longest equal suffix ending at each causal endpoint; latest wins ties."""
    n = len(q)
    lengths = np.zeros(n, dtype=np.int32)
    routes = np.full(n, -1, dtype=np.int32)
    previous = np.zeros(n + 1, dtype=np.int32)
    for t in range(n):
        current = np.zeros(n + 1, dtype=np.int32)
        current[1:] = (previous[:-1] + 1) * (q[t] == k)
        if t:
            length = int(current[1:t + 1].max())
            if length:
                lengths[t] = length
                routes[t] = int(np.flatnonzero(current[1:t + 1] == length)[-1])
        previous = current
    return lengths, routes


def literal_output(q, k, v, e0, e1):
    lengths, routes = literal_routes(q, k)
    result = np.broadcast_to(e0, v.shape).copy()
    active = routes >= 0
    result[active] = (2 * v[routes[active] + 1].astype(np.int64) - 1) * e1
    return result, lengths, routes


def sigmoid(z, tau):
    exponential = np.exp(-np.abs(z / tau))
    return np.where(z >= 0, 1 / (1 + exponential), exponential / (1 + exponential))


def literal_stream(cache, gradient, stream, pair_codes=(), coordinates=None):
    """Return five gradients and independent costs for one cached stream."""
    batch, group = stream
    q, k = cache.q[batch, group].copy(), cache.k[batch, group].copy()
    v = cache.v_bits[batch, group]
    g = gradient[batch, group]
    n, d = v.shape
    nd = n * d
    e0, e1 = cache.emb0[group], cache.emb1[group]
    options = cache.options
    y, lengths, routes = literal_output(q, k, v, e0, e1)
    radius = max(float(np.sqrt(np.sum(e1 * e1))),
                 float(.5 * np.sqrt(np.sum((np.abs(e0) + np.abs(e1)) ** 2))), 1e-12)
    diagonal = options.trust_diagonal * np.sqrt(np.sum(g * g, axis=1)) / radius

    def corner(codes):
        edited_q, edited_k = q.copy(), k.copy()
        for code in codes:
            code = int(code)
            words = edited_q if code >= nd else edited_k
            at = code % nd
            words[at // d] ^= np.uint64(1) << np.uint64(at % d)
        value, _, _ = literal_output(edited_q, edited_k, v, e0, e1)
        delta = value - y
        terms = g * delta + .5 * diagonal[:, None] * delta * delta
        return math.fsum(float(x) for x in terms.flat)

    cq, ck = np.full((n, d), np.nan), np.full((n, d), np.nan)
    if coordinates is None:
        coordinates = range(2 * nd)
    for code in coordinates:
        code = int(code)
        (cq if code >= nd else ck).flat[code % nd] = corner([code])
    cv, ge0, ge1 = np.zeros((n, d)), np.zeros(d), np.zeros(d)
    for t in range(n):
        if routes[t] < 0:
            ge0 += g[t]
        else:
            source = routes[t] + 1
            sign = 2 * v[source].astype(np.int64) - 1
            ge1 += g[t] * sign
            cv[source] += g[t] * (-2 * sign * e1)

    logits = (cache.q_logits[batch, group], cache.k_logits[batch, group],
              cache.v_logits[batch, group])
    bit_arrays = tuple(x > 0 for x in logits)
    factors = tuple(np.ones_like(z) if options.logit_mode else
                    sigmoid(z, options.tau) * (1 - sigmoid(z, options.tau)) / options.tau
                    for z in logits)
    result = [factor * (1 - 2 * bits.astype(np.int64)) * credit
              for factor, bits, credit in zip(factors, bit_arrays, (cq, ck, cv))]
    costs = []
    for codes in pair_codes:
        if int(codes[0]) >= 2 * nd:
            continue
        a, b = map(int, codes)
        ca, cb, cij = corner([a]), corner([b]), corner([a, b])
        interaction = cij - ca - cb
        costs.append((ca, cb, cij, interaction))
        for code, other, singleton in ((a, b, ca), (b, a, cb)):
            kind = 0 if code >= nd else 1
            other_kind = 0 if other >= nd else 1
            position = np.unravel_index(code % nd, (n, d))
            other_position = np.unravel_index(other % nd, (n, d))
            probability = float(sigmoid(logits[other_kind][other_position], options.tau))
            if bit_arrays[other_kind][other_position]:
                probability = 1 - probability
            orientation = -1 if bit_arrays[kind][position] else 1
            result[kind][position] = (factors[kind][position] * orientation
                                      * (singleton + probability * interaction))
    return (*result, ge0, ge1), np.asarray(costs), y, lengths, routes


def make_case(n, d, seed, family="random", batches=1, groups=1):
    rng = np.random.default_rng(seed)
    shape = (batches, groups, n, d)
    logits = [rng.normal(size=shape) for _ in range(3)]
    if family == "unary":
        logits[0].fill(-.3)
        logits[1].fill(-.3)
        if n:
            logits[1][:, :, 0, -1] = .3
    elif family == "periodic":
        word = rng.integers(0, 2, (3, d))
        logits[0][:] = .3 * (2 * np.resize(word, (n, d)) - 1)
        logits[1][:] = logits[0]
        if n > 3:
            logits[1][:, :, n // 3, -1] *= -1
    e0, e1 = rng.normal(size=(2, groups, d))
    gradient = rng.normal(size=shape)
    return (*logits, e0, e1), gradient


def evaluate(arguments, gradient, **options):
    output, cache, _ = training_forward_numpy(*arguments, **options)
    result = training_backward_numpy(cache, gradient, trace=True)
    return output, cache, result


class SubquadraticEngineTests(unittest.TestCase):
    def assert_literal(self, arguments, gradient, **options):
        y, cache, actual = evaluate(arguments, gradient, **options)
        b, groups, n, d = gradient.shape
        ge0, ge1 = np.zeros((groups, d)), np.zeros((groups, d))
        for batch in range(b):
            for group in range(groups):
                pair_codes = actual[-1]["pair_codes"][batch * groups + group]
                expected, costs, old, lengths, routes = literal_stream(
                    cache, gradient, (batch, group), pair_codes)
                np.testing.assert_array_equal(y[batch, group], old)
                np.testing.assert_array_equal(cache.lengths[batch, group], lengths)
                np.testing.assert_array_equal(cache.routes[batch, group], routes)
                for i in range(3):
                    np.testing.assert_allclose(actual[i][batch, group], expected[i],
                                               rtol=2e-11, atol=2e-11)
                ge0[group] += expected[3]
                ge1[group] += expected[4]
                if costs.size:
                    trace = actual[-1]["pair_costs"][batch * groups + group, :len(costs)]
                    np.testing.assert_allclose(trace, costs, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(actual[3], ge0, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(actual[4], ge1, rtol=2e-13, atol=2e-13)
        return actual

    def test_literal_singleton_and_joint_credits(self):
        for n, d, family in ((0, 1, "random"), (1, 64, "random"),
                             (7, 1, "random"), (9, 3, "random"),
                             (7, 64, "periodic"), (17, 4, "unary"),
                             (128, 1, "periodic"), (129, 2, "unary")):
            args, gradient = make_case(n, d, 971 + n + d, family)
            for order in ("first", "second"):
                with self.subTest(n=n, d=d, family=family, order=order):
                    self.assert_literal(args, gradient, credit_order=order, tau=.7,
                                        joint_rows=7, proposal_seed=53, repair_seed=71)

    def test_batch_groups_identity_and_zero_adjoint(self):
        args, gradient = make_case(11, 5, 7102, "periodic", batches=2, groups=2)
        gradient[0, 0] = 0
        self.assert_literal(args, gradient, credit_to_logit="identity", joint_pairs=2,
                            proposal_seed=151, repair_seed=661)

    def test_every_packed_width(self):
        for d in range(1, 65):
            args, gradient = make_case(5, d, 4912 + d, "periodic")
            with self.subTest(d=d):
                self.assert_literal(args, gradient, credit_order="second", joint_pairs=0)

    def test_large_structured_selected_coordinates_match_literal(self):
        for n, d, family in ((129, 64, "unary"), (257, 4, "periodic"),
                             (513, 1, "unary")):
            args, gradient = make_case(n, d, 193 + n, family)
            for order in ("first", "second"):
                with self.subTest(n=n, d=d, family=family, order=order):
                    y, cache, actual = evaluate(args, gradient, credit_order=order,
                                                proposal_seed=557, repair_seed=719)
                    coordinates = {kind * n * d + owner * d + bit
                                   for kind in (0, 1)
                                   for owner in (0, 1, n // 3, n // 2, n - 2, n - 1)
                                   for bit in (0, d - 1)}
                    random = np.random.default_rng(917 + n + d)
                    coordinates.update(map(int, random.integers(0, 2 * n * d, 16)))
                    expected, costs, old, lengths, routes = literal_stream(
                        cache, gradient, (0, 0), actual[-1]["pair_codes"][0], coordinates)
                    np.testing.assert_array_equal(y[0, 0], old)
                    np.testing.assert_array_equal(cache.lengths[0, 0], lengths)
                    np.testing.assert_array_equal(cache.routes[0, 0], routes)
                    for i in (0, 1):
                        checked = np.isfinite(expected[i])
                        np.testing.assert_allclose(actual[i][0, 0][checked], expected[i][checked],
                                                   rtol=4e-10, atol=4e-10)
                    for i in (2, 3, 4):
                        observed = actual[i][0, 0] if i == 2 else actual[i][0]
                        np.testing.assert_allclose(observed, expected[i], rtol=2e-12, atol=2e-12)
                    if costs.size:
                        np.testing.assert_allclose(actual[-1]["pair_costs"][0, :len(costs)], costs,
                                                   rtol=4e-10, atol=4e-10)

    def test_exact_randomized_is_bitwise_and_seed_independent(self):
        args, gradient = make_case(193, 4, 1772, "periodic")
        for order in ("first", "second"):
            reference = evaluate(args, gradient, credit_order=order, randomized=False,
                                 proposal_seed=118, repair_seed=0)[2]
            for seed in (0, 1, 2**64 - 1):
                actual = evaluate(args, gradient, credit_order=order, randomized=True,
                                  proposal_seed=118, repair_seed=seed)[2]
                for a, b in zip(reference[:5], actual[:5]):
                    self.assertEqual(a.tobytes(), b.tobytes())
                for name in ("pair_codes", "pair_costs"):
                    self.assertEqual(reference[-1][name].tobytes(), actual[-1][name].tobytes())

    def test_full_approximate_coverage_is_complete(self):
        for n, d, family in ((0, 4, "random"), (1, 64, "random"),
                             (73, 3, "unary"), (129, 8, "periodic")):
            args, gradient = make_case(n, d, 771 + n, family)
            for order in ("first", "second"):
                exact = evaluate(args, gradient, credit_order=order, repair_seed=19)[2]
                full = evaluate(args, gradient, credit_order=order, repair_seed=19,
                                backward_mode="approximate", approximation_rows=max(1, n))[2]
                for a, b in zip(exact[:5], full[:5]):
                    np.testing.assert_allclose(a, b, rtol=3e-11, atol=3e-11)
                np.testing.assert_array_equal(exact[-1]["pair_codes"], full[-1]["pair_codes"])
                self.assertTrue(full[-1]["approximation_complete_residual"])

    def test_partial_rows_and_channels_restore_selected_singletons(self):
        args, gradient = make_case(73, 5, 9127, "periodic")
        for channels in (0, 2):
            for order in ("first", "second"):
                options = dict(credit_order=order, proposal_seed=127, repair_seed=61)
                exact = evaluate(args, gradient, **options)[2]
                _, cache, partial = evaluate(args, gradient, backward_mode="approximate",
                                             approximation_rows=4, approximation_channels=channels,
                                             **options)
                for i in (2, 3, 4):
                    self.assertEqual(exact[i].tobytes(), partial[i].tobytes())
                np.testing.assert_array_equal(exact[-1]["pair_codes"], partial[-1]["pair_codes"])
                np.testing.assert_allclose(exact[-1]["pair_costs"], partial[-1]["pair_costs"],
                                           rtol=3e-11, atol=3e-11)
                if order == "second":
                    for code in partial[-1]["pair_codes"].flat:
                        code = int(code)
                        if code >= 2 * 73 * 5:
                            continue
                        at = np.unravel_index(code % (73 * 5), (1, 1, 73, 5))
                        i = 0 if code >= 73 * 5 else 1
                        self.assertAlmostEqual(exact[i][at], partial[i][at], delta=3e-11)
                    self.assertGreater(partial[-1]["approx_extra_joint_replays"], 0)
                self.assertTrue(np.isfinite(partial[0]).all())
                self.assertTrue(np.isfinite(partial[1]).all())

    def test_partial_estimator_mean(self):
        args, gradient = make_case(41, 3, 6271, "unary")
        for channels in (0, 1):
            exact = evaluate(args, gradient, credit_order="second", joint_pairs=0,
                             credit_to_logit="identity")[2]
            target = np.concatenate([x.ravel() for x in exact[:2]])
            samples = []
            for seed in range(256):
                result = evaluate(args, gradient, credit_order="second", joint_pairs=0,
                                  credit_to_logit="identity", backward_mode="approximate",
                                  approximation_rows=3, approximation_channels=channels,
                                  repair_seed=seed)[2]
                samples.append(np.concatenate([x.ravel() for x in result[:2]]))
            samples = np.asarray(samples)
            bias = np.linalg.norm(samples.mean(axis=0) - target)
            standard_error = np.sqrt(np.sum(samples.var(axis=0, ddof=1)) / len(samples))
            # Aggregate mean-square error, rather than a fragile per-coordinate
            # z test; the deterministic seed set makes this regression stable.
            self.assertLess(bias, 4 * standard_error + 3e-10)
            self.assertGreater(float(np.linalg.norm(samples[0] - target)), 1e-8)

    def test_empty_batch_and_validation(self):
        args, gradient = make_case(7, 64, 211, batches=0, groups=2)
        _, _, result = evaluate(args, gradient)
        self.assertEqual(result[0].shape, gradient.shape)
        np.testing.assert_array_equal(result[3], np.zeros((2, 64)))
        np.testing.assert_array_equal(result[4], np.zeros((2, 64)))
        args, gradient = make_case(3, 2, 19)
        for removed in ({"repair_samples": 1}, {"credit_engine": "legacy"}):
            with self.assertRaises((TypeError, ValueError)):
                evaluate(args, gradient, **removed)
        bad = gradient.copy()
        bad[0, 0, 1, 0] = np.nan
        with self.assertRaises(ValueError):
            evaluate(args, bad)

    def test_c_api_matches_numpy_binding(self):
        from conv_rosa_transformer.native_training import (
            _library, _pointers, TRAINING_COUNTER_NAMES,
        )
        args, gradient = make_case(9, 3, 813, "unary")
        _, cache, reference = evaluate(args, gradient, credit_order="first")
        gq, gk, gv = [np.empty_like(gradient) for _ in range(3)]
        ge0, ge1 = [np.zeros((1, 3)) for _ in range(2)]
        counts = np.zeros((1, len(TRAINING_COUNTER_NAMES)), dtype=np.uint64)
        arrays = (cache.q, cache.k, cache.v_bits, cache.q_logits, cache.k_logits,
                  cache.v_logits, gradient, cache.emb0, cache.emb1, gq, gk, gv,
                  ge0, ge1, counts, None, None)
        code = _library().rosa_training_backward(
            ctypes.byref(cache.options), 1, 1, 9, 3, *_pointers(arrays))
        self.assertEqual(code, 0)
        for a, b in zip((gq, gk, gv, ge0, ge1), reference[:5]):
            self.assertEqual(a.tobytes(), b.tobytes())

    def test_constant_payload_has_no_route_credit(self):
        args, gradient = make_case(257, 64, 1892, "periodic")
        q, k, v, e0, e1 = args
        v.fill(.3)
        e0.fill(1.)
        e1.fill(1.)
        gradient *= 1e100
        # Every route, including empty, returns exactly the same output.
        # Centering should retain this algebraic zero even at extreme scale.
        for order in ("first", "second"):
            result = evaluate(args, gradient, credit_order=order)[2]
            for i in (0, 1):
                np.testing.assert_array_equal(result[i], np.zeros_like(result[i]))

    def test_channel_sampling_ignores_unreachable_reference_payload(self):
        args, gradient = make_case(41, 3, 3, "unary")
        q, k, v, e0, e1 = args
        v.fill(.3)
        v[:, :, 0] = -.3
        e0.fill(1.)
        e1.fill(1.)
        # V[0] is never a returned payload: a causal endpoint r reads V[r+1].
        # Changing this unreachable value must not create sampling variance
        # when all reachable outputs (including empty) are identical.
        for order in ("first", "second"):
            for channels in (0, 1):
                result = evaluate(args, gradient, credit_order=order,
                                  backward_mode="approximate",
                                  approximation_rows=4, approximation_channels=channels)[2]
                for i in (0, 1):
                    np.testing.assert_array_equal(result[i], np.zeros_like(result[i]))

    def test_correlation_interval_and_cancellation_oracle(self):
        from tests._support import compile_test
        binary = compile_test("subquadratic_correlation")
        completed = subprocess.run([str(binary)], check=True, text=True, capture_output=True)
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "PASS")
        self.assertGreater(report["queries"], 200000)
        self.assertLess(report["max_error_over_l1"], 3e-12)


if __name__ == "__main__":
    unittest.main()
