"""Exact dyadic representation and round-to-nearest tests for expansions."""
from fractions import Fraction
import math
import random
import subprocess
import unittest

from tests._support import COVERAGE, EXHAUSTIVE, compile_test


class ExpansionTests(unittest.TestCase):
    def test_fraction_and_snapshot_oracle(self):
        binary = compile_test("expansion_driver")
        rng = random.Random(20261018)
        cases = []

        def add(values, snapshot=-1):
            cases.append((snapshot, values))

        for values in ([-1., -1e16, 1e16], [1e100, 1e50, 1., -1e100, -1e50],
                       [1e300, 1e150, 1., 1e-150, 1e-300, -1e300, -1e150, -1., -1e-150],
                       [1., 2. ** -53, 2. ** -1074], [1., 2. ** -53, -2. ** -1074]):
            add(values)
            for index in range(len(values) - 1):
                add(values, index)
        for trial in range(2000 if EXHAUSTIVE else 200):
            values = [math.ldexp(rng.choice((-1., 1.)) * rng.randrange(1, 2 ** 30),
                                 rng.randrange(-1074, 900))
                      for _ in range(rng.randrange(1, 110))]
            values = [x for x in values if math.isfinite(x)]
            if trial % 3 == 0:
                root = math.ldexp(rng.uniform(1, 2), rng.randrange(100, 950))
                values = [root] + values + [-root]
            add(values)
            if len(values) > 2:
                add(values, rng.randrange(len(values) - 1))
        for exponent in (-1000, -500, -1, 0, 1, 500, 950):
            value = math.ldexp(1., exponent)
            ulp = math.ulp(value)
            for direction in (-1, 1):
                for tiny in (2. ** -1074, math.ldexp(ulp, -54)):
                    add([value, direction * ulp / 2, tiny])
                    add([value, direction * ulp / 2, -tiny])
        data = "".join(f"{s} {len(xs)} " + " ".join(x.hex() for x in xs) + "\n"
                       for s, xs in cases)
        run = subprocess.run([str(binary)], input=data, text=True,
                             capture_output=True, check=True)
        lines = run.stdout.splitlines()
        self.assertEqual(len(lines), len(cases))
        maximum = spill_results = 0
        for index, ((snapshot, values), line) in enumerate(zip(cases, lines)):
            words = line.split()
            actual, count = float.fromhex(words[0]), int(words[1])
            parts = list(map(float.fromhex, words[2:]))
            self.assertEqual(count, len(parts))
            terms = values[snapshot + 1:] if snapshot >= 0 else values
            exact = sum(map(Fraction.from_float, terms), Fraction())
            represented = sum(map(Fraction.from_float, parts), Fraction())
            self.assertEqual(exact, represented, index)
            self.assertEqual(actual, float(exact), index)
            maximum = max(maximum, count)
            spill_results += count > 2
        COVERAGE["expansion_fraction"] = {"exact_representation_checks": len(cases),
                                           "correct_rounding_checks": len(cases),
                                           "max_components": maximum,
                                           "results_with_more_than_two_components": spill_results}

    def test_ordinary_accumulation_storage(self):
        binary = compile_test("expansion_normal")
        run = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
        spills, updates, maximum = map(int, run.stdout.split())
        self.assertEqual(updates, 1000000)
        self.assertEqual(spills, 0)
        self.assertLessEqual(maximum, 2)
        COVERAGE["expansion_ordinary"] = dict(updates=updates, spills=spills,
                                               max_components=maximum)


if __name__ == "__main__":
    unittest.main()
