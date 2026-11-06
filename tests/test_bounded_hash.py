"""Full-key collision fixtures force each bounded dictionary's fallback."""
import json
import subprocess
import unittest

from tests._support import COVERAGE, compile_test


class BoundedHashTests(unittest.TestCase):
    def test_adversarial_full_key_collisions(self):
        run = subprocess.run([str(compile_test("bounded_hash"))], check=True,
                             text=True, capture_output=True)
        result = json.loads(run.stdout)
        self.assertEqual(result["status"], "PASS")
        COVERAGE["bounded_hash"] = result


if __name__ == "__main__":
    unittest.main()
