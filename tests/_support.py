"""Shared test build support; test binaries are never runtime backends."""
from pathlib import Path
import hashlib
import os
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "conv_rosa_transformer" / "csrc"
BUILD = ROOT / "build" / "tests"
EXHAUSTIVE = os.environ.get("ROSA_EXHAUSTIVE") == "1"
COVERAGE = {}


def source_hashes():
    # The ordinary layer now delegates to production joint/candidate helpers.
    # A validation snapshot must cover those files, not just the C++ wrapper.
    paths = sorted((ROOT / "conv_rosa_transformer").glob("*.py"))
    paths += sorted(CSRC.glob("*.cpp")) + sorted(CSRC.glob("*.h"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}


def compile_test(stem):
    """Compile a standalone oracle with the production headers.

    Requires a GCC/Clang-compatible C++17 compiler (Linux, macOS, or WSL).
    The compiler can be selected with CXX; no fast-math flags are used.
    """
    BUILD.mkdir(parents=True, exist_ok=True)
    source = ROOT / "tests" / "cpp" / (stem + ".cpp")
    binary = BUILD / (stem + (".exe" if os.name == "nt" else ""))
    flags = ["-O3", "-std=c++17", "-ffp-contract=off", "-I", str(CSRC)]
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    subprocess.run(compiler + flags + [str(source), "-o", str(binary)], check=True)
    return binary
