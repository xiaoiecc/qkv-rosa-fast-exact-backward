"""Run independent native oracles with ASan and UBSan, recording exact dependencies.

Example: python tests/sanitize.py --jobs 2 --no-leaks --output build/sanitizers.json
Use --no-leaks only where the process environment cannot run LeakSanitizer.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(paths):
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", nargs="+", default=["subquadratic_training",
                        "subquadratic_q_frontier", "subquadratic_short", "subquadratic_indexed_row",
                        "subquadratic_fingerprint_k"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--no-leaks", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "build/sanitizers.json")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    flags = ["-O1", "-g", "-std=c++17", "-ffp-contract=off",
             "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
             "-fno-omit-frame-pointer"]
    build = ROOT / "build/sanitizers"
    build.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ASAN_OPTIONS=f"detect_leaks={0 if args.no_leaks else 1}:halt_on_error=1",
               UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")

    def run(stem):
        source = ROOT / "tests/cpp" / (stem + ".cpp")
        dependencies = subprocess.check_output(compiler + ["-std=c++17", "-MM", str(source)],
                                               cwd=ROOT, text=True)
        paths = [Path(p).resolve() for p in shlex.split(dependencies.replace("\\\n", " "))[1:]]
        before = digest(paths)
        binary = build / stem
        command = compiler + flags + [str(source), "-o", str(binary)]
        start = time.perf_counter()
        compiled = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        compile_seconds = time.perf_counter() - start
        start = time.perf_counter()
        executed = (subprocess.run([str(binary)], cwd=ROOT, env=env, text=True, capture_output=True)
                    if compiled.returncode == 0 else None)
        unchanged = before == digest(paths)
        passed = compiled.returncode == 0 and executed.returncode == 0 and unchanged
        row = dict(test=stem, status="PASS" if passed else "FAIL", compile_command=command,
                   compile_seconds=compile_seconds, run_seconds=time.perf_counter() - start,
                   compile_returncode=compiled.returncode, compile_stderr=compiled.stderr,
                   returncode=executed.returncode if executed else None,
                   stdout=executed.stdout if executed else "", stderr=executed.stderr if executed else "",
                   source_sha256=before, source_unchanged=unchanged,
                   sanitizer_environment={key: env[key] for key in ("ASAN_OPTIONS", "UBSAN_OPTIONS")})
        if executed and executed.stdout.strip():
            try:
                row["oracle"] = json.loads(executed.stdout)
            except json.JSONDecodeError:
                pass
        print(json.dumps({key: row[key] for key in ("test", "status", "compile_seconds", "run_seconds")}), flush=True)
        return row

    rows = json.loads(args.output.read_text())["tests"] if args.append and args.output.exists() else []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        new_rows = list(pool.map(run, args.tests))
    rows = [row for row in rows if row["test"] not in args.tests] + new_rows
    for row in rows:
        current = digest([ROOT / p for p in row["source_sha256"]])
        row["source_matches_final"] = current == row["source_sha256"]
    passed = all(row["status"] == "PASS" and row["source_matches_final"] for row in rows)
    report = dict(status="PASS" if passed else "FAIL", compiler=subprocess.check_output(
                  compiler + ["--version"], text=True).splitlines()[0],
                  scope="Independent literal native oracles; address and undefined-behavior sanitizers",
                  leak_detection=not args.no_leaks, tests=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
