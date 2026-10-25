"""Run the current implementation's correctness gates.

Examples:
    python tests/run.py
    python tests/run.py --exhaustive --output build/validation.json

No historical optimized implementation is imported. The optional exhaustive
mode adds longer integer enumeration and standalone C++ structural oracles.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "validation.json")
    args = parser.parse_args()
    os.environ["ROSA_EXHAUSTIVE"] = "1" if args.exhaustive else "0"
    from tests._support import COVERAGE, source_hashes
    before = source_hashes()
    begin = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py",
                                               top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    unchanged = before == source_hashes()
    passed = result.wasSuccessful() and unchanged
    report = {"status": "PASS" if passed else "FAIL", "exhaustive": args.exhaustive,
              "seconds": time.perf_counter() - begin, "tests_run": result.testsRun,
              "skipped": len(result.skipped), "source_unchanged": unchanged,
              "source_sha256": before, "coverage": COVERAGE}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "exhaustive", "tests_run", "skipped", "seconds")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
