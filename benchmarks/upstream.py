"""Build a selected historical upstream revision outside the working tree.

This is benchmark setup only. No old runtime is bundled or used by the current
library. On Linux, one standard dependent-template keyword is added so GCC
accepts the original MSVC-oriented source; its arithmetic is unchanged.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import sysconfig
import tempfile

import torch

UPSTREAM_REVISION = "32e90e272929942a764d696b639409b783c61bb1"


def _git(directory, *arguments):
    return subprocess.run(["git", "-C", str(directory), *arguments],
                          capture_output=True, check=True).stdout


def load_upstream(directory, revision=UPSTREAM_REVISION):
    if sys.platform != "linux":
        raise RuntimeError("Historical benchmark build currently supports Linux/WSL")
    from torch.utils.cpp_extension import include_paths, library_paths
    directory = Path(directory).resolve()
    commit = _git(directory, "rev-parse", "--verify", revision + "^{commit}").decode().strip()
    prefix = "conv_rosa_transformer/csrc/"
    paths = _git(directory, "ls-tree", "-r", "--name-only", commit, "--", prefix).decode().splitlines()
    files = {path: _git(directory, "show", f"{commit}:{path}") for path in paths
             if path.endswith((".h", ".cpp"))}
    if not files or prefix + "pipeline.cpp" not in files:
        raise ValueError("Selected revision does not contain the original upstream extension")
    original = {p: hashlib.sha256(v).hexdigest() for p, v in files.items()}
    key = prefix + "contract.cpp"
    needle = b"prodt.sum().item<scalar_t>()"
    if files[key].count(needle) != 1:
        raise ValueError("Expected original dependent-template call was not found exactly once")
    files[key] = files[key].replace(needle, b"prodt.sum().template item<scalar_t>()")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    compiler_version = subprocess.run(compiler + ["--version"], capture_output=True,
                                      text=True, check=True).stdout.splitlines()[0]
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    settings = {"commit": commit, "source_sha256": original,
                "compiled_source_sha256": {p: hashlib.sha256(v).hexdigest() for p, v in files.items()},
                "torch": torch.__version__, "python": sys.version,
                "compiler": compiler_version, "cxx11_abi": abi,
                "compile_flags": ["-O3", "-DNDEBUG", "-std=c++20", "-fPIC",
                                  "-fvisibility=hidden", "-ffp-contract=off"],
                "source_portability_edit": "contract.cpp: add template before dependent item<scalar_t>()"}
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    module_name = "rosa_upstream_" + signature
    cache = Path(tempfile.gettempdir()) / "qkv_rosa_upstream_benchmark" / signature
    cache.mkdir(parents=True, exist_ok=True)
    extension = cache / (module_name + sysconfig.get_config_var("EXT_SUFFIX"))
    manifest = cache / "manifest.json"
    if not extension.exists() or not manifest.exists() or json.loads(manifest.read_text()) != settings:
        source_dir = cache / "src"
        source_dir.mkdir(exist_ok=True)
        for path, value in files.items():
            (source_dir / Path(path).name).write_bytes(value)
        flags = settings["compile_flags"] + [f"-DTORCH_EXTENSION_NAME={module_name}",
                                            f"-D_GLIBCXX_USE_CXX11_ABI={abi}"]
        for path in include_paths() + [sysconfig.get_paths()["include"]]:
            flags += ["-isystem", str(path)]
        objects = []
        for source in sorted(source_dir.glob("*.cpp")):
            obj = cache / (source.stem + ".o")
            print(f"Build original upstream: {source.name}", flush=True)
            subprocess.run(compiler + flags + ["-c", str(source), "-o", str(obj)], check=True)
            objects.append(str(obj))
        link = compiler + ["-shared", "-o", str(extension)] + objects
        for path in library_paths():
            link += ["-L" + str(path), "-Wl,-rpath," + str(path)]
        link += ["-ltorch", "-ltorch_cpu", "-lc10", "-ltorch_python"]
        subprocess.run(link, check=True)
        manifest.write_text(json.dumps(settings, indent=2) + "\n")
    spec = importlib.util.spec_from_file_location(module_name, extension)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result, settings
