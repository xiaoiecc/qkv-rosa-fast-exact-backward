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

    def test_interval_credit_and_sparse_time_coordinates(self):
        self.check_oracle("interval_credit")

    def test_deletion_and_owner_time_ranges(self):
        self.check_oracle("subquadratic_delete")

    def test_indexed_row_residual(self):
        self.check_oracle("subquadratic_indexed_row")

    def test_c_training_interface(self):
        self.check_oracle("subquadratic_training")


if __name__ == "__main__":
    unittest.main()
