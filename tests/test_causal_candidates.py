"""Static proposals checked against independent causal alignment enumeration."""
import itertools
import unittest

import numpy as np

from research.causal_candidates import propose_pairs
from research.joint_bits import Bit


def structural_oracle(q, k, d, *, rows=None, scan=8,
                      allowed_streams=("Q", "K"), allowed_owners=None):
    """Enumerate every causal endpoint, with no bands, sampling, or heap."""
    pool = set()
    for t in range(len(q)) if rows is None else sorted(set(rows)):
        for r in range(t):
            choices = []
            for offset in range(min(scan, r + 1)):
                u, s = t - offset, r - offset
                differing = [j for j in range(d)
                             if ((q[u] >> j) & 1) != ((k[s] >> j) & 1)]
                if len(choices) + len(differing) > 2:
                    break
                for j in differing:
                    choices.append((("Q", u, j), ("K", s, j)))
                if len(choices) == 2:
                    for pair in itertools.product(*choices):
                        if all(stream in allowed_streams and
                               (allowed_owners is None or
                                position in allowed_owners.get(stream, ()))
                               for stream, position, bit in pair):
                            pool.add(tuple(sorted(pair)))
                    break
    return pool


def traced_pool(stats):
    return {tuple(tuple(bit) for bit in pair)
            for pair in stats.get("candidate_pairs", [])}


class CausalCandidateTests(unittest.TestCase):
    def check_full_capacity(self, q, k, d, *, rows=None, seed=0,
                            allowed_streams=("Q", "K"), allowed_owners=None):
        n = len(q)
        scan = max(1, n)
        options = dict(rows=rows, scan=scan, allowed_streams=allowed_streams,
                       allowed_owners=allowed_owners)
        pairs, stats = propose_pairs(q, k, d, max_pairs=8, seed=seed,
            bucket_cap=max(1, n), occurrence_cap=max(1, n),
            candidate_cap=10000, candidate_trace=True, **options)
        expected = structural_oracle(q, k, d, **options)
        self.assertEqual(traced_pool(stats), expected)
        self.assertTrue(all(tuple((b.stream, b.position, b.bit) for b in pair)
                            in expected for pair in pairs))
        self.assertEqual(len({b for pair in pairs for b in pair}), 2 * len(pairs))
        return pairs, stats

    def test_exhaustive_full_capacity_structural_pool(self):
        # 341 binary Q/K streams through length four; each alignment is checked.
        for n in range(5):
            for values in itertools.product(range(2), repeat=2 * n):
                self.check_full_capacity(values[:n], values[n:], 1)
        # 256 two-bit Q/K streams of length two exercise two bits in one word.
        for values in itertools.product(range(4), repeat=4):
            self.check_full_capacity(values[:2], values[2:], 2, seed=19)

    def test_future_words_occurrences_and_latest(self):
        # The only mismatching K word first appears at the query's own endpoint.
        pairs, stats = propose_pairs([0] * 4, [0, 0, 0, 3], 2,
            bucket_cap=1, occurrence_cap=1, candidate_trace=True)
        self.assertEqual(pairs, [])
        self.assertEqual(traced_pool(stats), set())
        # An earlier first occurrence must not expose that word's later ones.
        n, t = 9, 6
        _, stats = propose_pairs([3] * n, [0] * n, 2, rows=[t],
            bucket_cap=1, occurrence_cap=1, max_pairs=8,
            candidate_cap=1000, candidate_trace=True, seed=17)
        pool = traced_pool(stats)
        self.assertTrue(pool)
        self.assertTrue(all(position < t for pair in pool for stream, position, _ in pair
                            if stream == "K"))
        self.assertIn((("K", t - 1, 0), ("K", t - 1, 1)), pool)

    def test_full_unsigned_words(self):
        maximum = (1 << 64) - 1
        q, k = [maximum, maximum ^ (1 << 63) ^ 1], [maximum, maximum]
        pairs, _ = self.check_full_capacity(q, k, 64, rows=[1],
                                           allowed_streams=("Q",))
        self.assertEqual(pairs, [(Bit("Q", 1, 0), Bit("Q", 1, 63))])
        # Share band keys without permitting a full-word distance-three match.
        self.check_full_capacity([0, 0, 3], [1 << 63, 0, 3], 64, seed=182)

    def test_cross_position_and_owner_filters(self):
        pairs, _ = self.check_full_capacity([0, 1, 1], [0, 0, 0], 1,
            rows=[2], allowed_streams=("Q",), allowed_owners={"Q": {1, 2}})
        self.assertEqual(pairs, [(Bit("Q", 1, 0), Bit("Q", 2, 0))])
        for streams in (("Q",), ("K",), ("Q", "K")):
            self.check_full_capacity([0, 3, 1, 3, 0], [0, 2, 0, 3, 1], 2,
                rows=[4, 2, 4, 0], allowed_streams=streams,
                allowed_owners={"Q": {1, 2, 4}, "K": {0, 1, 3}}, seed=710)

    def test_reproducible_bounded_sampling_is_sound(self):
        rng = np.random.default_rng(289)
        q, k = rng.integers(0, 256, (2, 70)).tolist()
        budget, buckets, occurrences, scan = 4, 3, 2, 5
        options = dict(max_pairs=budget, seed=761, bucket_cap=buckets,
                       occurrence_cap=occurrences, scan=scan, candidate_trace=True)
        first, stats = propose_pairs(q, k, 8, **options)
        second, other = propose_pairs(q, k, 8, **options)
        self.assertTrue(first)
        self.assertEqual(first, second)
        self.assertEqual({key: value for key, value in stats.items() if not key.endswith("seconds")},
                         {key: value for key, value in other.items() if not key.endswith("seconds")})
        self.assertTrue(traced_pool(stats) <= structural_oracle(q, k, 8, scan=scan))
        self.assertLessEqual(stats["word_rng_calls"], 3 * len(q))
        self.assertLessEqual(stats["sampled_word_events"], 3 * len(q) * buckets)
        self.assertLessEqual(stats["examined_alignments"], len(q) * 3 * buckets * (occurrences + 1))
        self.assertLessEqual(stats["symbol_comparisons"], stats["examined_alignments"] * scan)
        self.assertLessEqual(stats["candidate_events"], 4 * stats["examined_alignments"])
        self.assertLessEqual(stats["retained_candidates"], 8 * budget)
        self.assertLessEqual(len(first), budget)
        self.assertEqual(len({b for pair in first for b in pair}), 2 * len(first))

    def test_validation_and_public_lazy_entry(self):
        from research.joint_bits import propose_pairs as public
        options = dict(rows=[1], seed=25, candidate_trace=True)
        a, sa = public([0, 3], [0, 0], 2, **options)
        b, sb = propose_pairs([0, 3], [0, 0], 2, **options)
        self.assertEqual(a, b)
        self.assertEqual(sa["candidate_pairs"], sb["candidate_pairs"])
        with self.assertRaises(ValueError):
            propose_pairs([0], [0], 1, rows=[1 << 80])
        with self.assertRaises(ValueError):
            propose_pairs([1 << 64], [0], 64)
        self.assertEqual(propose_pairs([], [], 64)[0], [])


if __name__ == "__main__":
    unittest.main()
