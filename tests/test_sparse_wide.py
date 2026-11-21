"""Exercise sparse wide-word credits through the public saved-route entry."""
import unittest

import numpy as np

from tests.test_subquadratic_engine import make_case, evaluate, literal_stream


class SparseWideTests(unittest.TestCase):
    def test_complete_finite_changes_with_ties_and_zero_credit_signs(self):
        rng = np.random.default_rng(7184)
        for bits in (16, 25, 64):
            high = np.uint64(1) << np.uint64(bits - 1)
            for matched in (False, True):
                # An edit to the high bit can create several equal-length
                # endpoints with different payloads. It can also affect
                # several output rows for a single edited key.
                q = np.array([2, 1, 2, 2, 1, 2, 1, 2], dtype=np.uint64) ^ high
                k = np.array([1, 2, 2, 7, 1, 7, 1, 2], dtype=np.uint64)
                if matched:
                    k[1:4] = q[2:5]
                arrays, g = make_case(len(q), bits, 17)
                g[0, 0, 3] = 0
                for logits, packed in zip(arrays[:2], (q, k)):
                    bit = (packed[:, None] >> np.arange(bits, dtype=np.uint64)) & 1
                    logits[:] = np.where(bit, 1., -1.) * (.1 + rng.random((len(q), bits)))
                for order in ("first", "second"):
                    y, cache, actual = evaluate(arrays, g, credit_order=order, joint_pairs=0)
                    expected, _, output, lengths, _ = literal_stream(cache, g, (0, 0))
                    self.assertEqual(bool(lengths.max()), matched)
                    np.testing.assert_array_equal(y[0, 0], output)
                    for j, want in enumerate(expected):
                        got = actual[j][0, 0] if j < 3 else actual[j][0]
                        np.testing.assert_allclose(got, want, rtol=2e-11, atol=1e-12)
                        if not matched and j < 3:
                            zero = want == 0.
                            np.testing.assert_array_equal(np.signbit(got[zero]), np.signbit(want[zero]))


if __name__ == "__main__":
    unittest.main()
