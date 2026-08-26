"""JIT build for the rosa C++ extension. Run under vcvars64 (see build.bat)."""
import os
import glob
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
os.makedirs(BUILD, exist_ok=True)

def build(verbose=False):
    sources = sorted(glob.glob(os.path.join(HERE, "csrc", "*.cpp")))
    return load(
        name="rosa_cpp",
        sources=sources,
        build_directory=BUILD,
        extra_cflags=["/O2", "/std:c++17", "/utf-8"],
        verbose=verbose,
    )

if __name__ == "__main__":
    ext = build(verbose=True)
    print("BUILD OK:", ext.__file__)
