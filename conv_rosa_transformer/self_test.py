"""Self-test: C++ extension bitwise parity against the bundled Python reference (rosa_reference.py)."""
import importlib.util
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import build

ext = build()
ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rosa_reference.py")
spec = importlib.util.spec_from_file_location("rosa_ref", ref_path)
py = importlib.util.module_from_spec(spec)
sys.modules["rosa_ref"] = py
spec.loader.exec_module(py)

rng = random.Random(43)
torch.manual_seed(43)
backends_seen = {}

def check(tag, q, k, D):
    T = len(q)
    vb = torch.randint(0, 2, (T, D), dtype=torch.uint8)
    gy = torch.randn(T, D)
    e0 = torch.randn(D); e1 = torch.randn(D)
    ra = py.exact_stream_bit_credits(q, k, vb, gy, e0, e1, D)
    rb = ext.exact_stream_bit_credits(q, k, vb, gy, e0, e1, D)
    backend_py = ra[3].repair_backend
    backend_cc = ext.repair_backend(q, k, D)
    assert backend_py == backend_cc, (tag, backend_py, backend_cc)
    backends_seen[backend_py] = backends_seen.get(backend_py, 0) + 1
    for i in range(3):
        assert torch.equal(ra[i], rb[i]), (tag, i, backend_py)
    assert list(ra[4]) == list(rb[3]) and list(ra[5]) == list(rb[4]), (tag, "ell/route")

for trial in range(20):
    T = rng.choice([2, 5, 30, 120, 400, 900])
    D = rng.choice([1, 2, 4, 6])
    alpha = rng.choice([2, 4, 16, 256])
    q = [rng.randrange(alpha) for _ in range(T)]
    k = [rng.randrange(alpha) for _ in range(T)]
    check(f"rand{trial}", q, k, D)

check("none", [0] * 100, [3] * 100, 2)
q = [rng.randrange(4) for _ in range(300)]
k = [100 + rng.randrange(4) for _ in range(300)]
check("zeromatch", q, k, 2)
base = [rng.randrange(8) for _ in range(40)]
check("periodic", (base * 30)[:1200], (base * 30)[:1200], 4)
s = [rng.randrange(4) for _ in range(600)]
check("dup", s * 2, s * 2, 2)
mut = (base * 30)[:1200]
for i in range(0, 1200, 137):
    mut[i] = (mut[i] + 1) % 8
check("mutperiodic", mut, (base * 30)[:1200], 4)

print("backend coverage:", backends_seen)
print("SELF-TEST OK: C++ bitwise identical to bundled Python reference")
