"""Compare the same exact backward on structured inputs at four lengths."""
import argparse
import json
import os
from pathlib import Path

from compare_checkout import ROOT, current, hashes, load_reference, revision
from random_qkv import input_digest, measure
from benchmarks.subquadratic_engine import make_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkout", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = args.reference_checkout.resolve()
    if reference == ROOT or args.repeats < 3:
        parser.error("use a separate reference checkout and at least three repeats")
    os.sched_setaffinity(0, {args.cpu})
    roots = {"reference": reference, "current": ROOT}
    snapshots = {k: {"revision": revision(v), "source_sha256": hashes(v)} for k, v in roots.items()}
    modules = {"reference": load_reference(reference), "current": current}
    rows = []
    seed = 813663
    for family, bits in (("unary", 8), ("periodic", 4), ("lagged_copy", 64), ("near_fanout", 8)):
        for n in (256, 512, 1024, 2048):
            inputs, gradient = make_case(family, n, bits, seed)
            digest = input_digest([*inputs, gradient])
            for order in ("first", "second"):
                row = measure(modules, inputs, gradient, order, args.repeats, seed)
                row.update(family=family, bits=bits, T=n, order=order, seed=seed, input_sha256=digest)
                rows.append(row)
    unchanged = all(record == {"revision": revision(roots[k]), "source_sha256": hashes(roots[k])}
                    for k, record in snapshots.items())
    passed = unchanged and all(r["forward_equal"] and r["gradient_close"] and r["pair_codes_equal"]
                               and r["pair_costs_close"] and r["errors"]["v_embeddings_bitwise_equal"]
                               for r in rows)
    report = dict(status="PASS" if passed else "FAIL", cpu=args.cpu, repeats=args.repeats,
                  source_unchanged=unchanged, checkouts=snapshots, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
