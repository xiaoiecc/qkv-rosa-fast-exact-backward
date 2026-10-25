"""Exact causal forward and hash-seed invariance against independent scans."""
import ctypes
import itertools
import os
from pathlib import Path
import shlex
import subprocess
import unittest

import numpy as np

from tests._support import BUILD, COVERAGE, CSRC, EXHAUSTIVE


def build_forward_probe():
    BUILD.mkdir(parents=True, exist_ok=True)
    source = BUILD / "forward_probe.cpp"
    source.write_text('''#include "forward.h"
extern "C" int forward_probe(int n,const uint64_t*q,const uint64_t*k,
 int32_t*l,int32_t*r,uint64_t seed,uint64_t*c) {
 try {rosa_exact::SamForward f(q,k,n,seed);f.run(l,r);
 c[0]=f.counters.transition_probes;c[1]=f.counters.suffix_fallbacks;
 c[2]=f.counters.endpoint_queries;c[3]=f.counters.endpoint_updates;
 c[4]=f.counters.hash_probe_steps;c[5]=f.retained_capacity_bytes();
 c[6]=f.state_count();c[7]=f.edge_count();c[8]=f.hash_slots();
 c[9]=f.endpoint_index_materialized();return 0;
 }catch(...){return 1;}}
''')
    library = BUILD / "forward_probe.so"
    subprocess.run(shlex.split(os.environ.get("CXX", "c++")) +
                   ["-O3", "-std=c++17", "-shared", "-fPIC", "-I", str(CSRC),
                    str(source), "-o", str(library)], check=True)
    loaded = ctypes.CDLL(str(library))
    function = loaded.forward_probe
    function.argtypes = ([ctypes.c_int] + [ctypes.c_void_p] * 4 +
                         [ctypes.c_uint64, ctypes.c_void_p])
    function.restype = ctypes.c_int
    return loaded, function


def independent_forward(q, k):
    result = []
    for t in range(len(q)):
        best = (0, -1)
        for r in range(t):
            length = 0
            while length <= r and q[t-length] == k[r-length]:
                length += 1
            if length:
                best = max(best, (length, r))
        result.append(best)
    return result


class LeanForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library, function = build_forward_probe()
        cls.function = staticmethod(function)

    def check_one(self, q, k, seed=17):
        q = np.ascontiguousarray(q, dtype=np.uint64)
        k = np.ascontiguousarray(k, dtype=np.uint64)
        n = len(q)
        lengths = np.zeros(n, dtype=np.int32)
        routes = np.full(n, -1, dtype=np.int32)
        counts = np.zeros(10, dtype=np.uint64)
        arrays = [q, k, lengths, routes]
        code = self.function(n, *(a.ctypes.data_as(ctypes.c_void_p) for a in arrays),
                             seed, counts.ctypes.data_as(ctypes.c_void_p))
        self.assertEqual(code, 0)
        self.assertEqual(list(zip(lengths.tolist(), routes.tolist())),
                         independent_forward(q, k))
        self.assertLessEqual(int(counts[0]), 2*n)
        self.assertLessEqual(int(counts[1]), n)
        return counts

    def test_binary_enumeration(self):
        limit = 7 if EXHAUSTIVE else 5
        cases = 0
        for n in range(limit+1):
            words = list(itertools.product(range(2), repeat=n))
            for q in words:
                for k in words:
                    self.check_one(q, k)
                    cases += 1
        COVERAGE["lean_forward_binary_pairs"] = cases

    def test_seeds_uint64_and_clones(self):
        rng = np.random.default_rng(20261025)
        cases = []
        for n in (1, 2, 17, 65, 129):
            for alphabet in (2, 3, 16):
                cases.append((rng.integers(alphabet, size=n, dtype=np.uint64),
                              rng.integers(alphabet, size=n, dtype=np.uint64)))
        cases += [([0, 2**63, 2**64-1, 0, 2**64-1],
                   [2**63, 2**64-1, 0, 2**64-1, 0]),
                  ([0]*129, [1]+[0]*128)]
        for seed in (0, 1, 2**64-1, 0x123456789abcdef0):
            for q, k in cases:
                self.check_one(q, k, seed)
        COVERAGE["lean_forward_hash_seeds"] = 4
        COVERAGE["lean_forward_seed_cases"] = len(cases)*4

    def test_exact_certificates_avoid_endpoint_tree(self):
        counts = self.check_one([0]*257, [0]*257)
        self.assertEqual(int(counts[9]), 0)
        self.assertEqual(int(counts[3]), 0)
        counts = self.check_one(list(range(257)), list(range(257)))
        self.assertEqual(int(counts[9]), 0)


if __name__ == "__main__":
    unittest.main()
