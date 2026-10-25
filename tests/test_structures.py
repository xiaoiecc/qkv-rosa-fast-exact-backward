"""Exhaustive structural oracles use the production C++ headers directly."""
import json
import subprocess
import unittest

from tests._support import COVERAGE, EXHAUSTIVE, compile_test


@unittest.skipUnless(EXHAUSTIVE, "pass --exhaustive to run C++ structural oracles")
class StructureTests(unittest.TestCase):
    def check_oracle(self, stem):
        binary = compile_test(stem)
        run = subprocess.run([str(binary)], check=True, text=True, capture_output=True)
        result = json.loads(run.stdout)
        self.assertEqual(result["status"], "PASS")
        COVERAGE[stem] = result

    def test_delete_routes_and_numeric_ranges(self):
        self.check_oracle("delete_routes")

    def test_causal_occurrences_and_latest(self):
        self.check_oracle("causal_occurrences")

    def test_corridor_lifetimes_and_winners(self):
        self.check_oracle("corridor_routes")


if __name__ == "__main__":
    unittest.main()
