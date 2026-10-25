"""Backward dictionary seeds change work placement, never routing answers."""
import ctypes
import os
import shlex
import subprocess
import unittest

import numpy as np

from tests._support import BUILD, CSRC
from tests.test_forward_lean import independent_forward


class BackwardHashTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        BUILD.mkdir(parents=True, exist_ok=True)
        source = BUILD / 'backward_hash_probe.cpp'
        source.write_text('''#include "sam_q.h"
extern "C" int backward_hash_probe(int n,int d,const uint64_t*q,const uint64_t*k,
 uint64_t seed,int32_t*out) {
 try {
  rosa_exact::SamQRepairs sam(q,k,n,d,~uint64_t(0),-1,0,0,seed);
  std::vector<rosa_exact::SamQRepairs::Repair> detached;int count=0;
  auto append=[&](const auto&list) {
   out[count++]=int(list.size());
   for(const auto&r:list) {out[count++]=r.owner;out[count++]=r.bit;
    out[count++]=r.length;out[count++]=r.end;out[count++]=r.state;out[count++]=r.short_state;}
  };
  for(int t=0;t<n;++t) {sam.advance(t);sam.extract_singleton_continuations(k,detached);
   out[count++]=sam.baseline().length;out[count++]=sam.baseline().end;
   append(sam.repairs());append(detached);}
  return count;
 }catch(...){return -1;}}
''')
        binary = BUILD / 'backward_hash_probe.so'
        subprocess.run(shlex.split(os.environ.get('CXX', 'c++')) +
                       ['-O3', '-std=c++17', '-shared', '-fPIC', '-I', str(CSRC),
                        str(source), '-o', str(binary)], check=True)
        cls.library = ctypes.CDLL(str(binary))
        function = cls.library.backward_hash_probe
        function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_uint64, ctypes.c_void_p]
        function.restype = ctypes.c_int
        cls.function = staticmethod(function)

    def test_all_repair_states_and_baselines_seed_invariant(self):
        rng = np.random.default_rng(2710811)
        cases = []
        for n in (0, 1, 17, 65):
            for d in (1, 3, 8):
                cases.append((rng.integers(1 << d, size=n, dtype=np.uint64),
                              rng.integers(1 << d, size=n, dtype=np.uint64), d))
        q = rng.bit_generator.random_raw(17)
        cases.extend([(q, q ^ (np.uint64(1) << np.uint64(63)), 64),
                      (np.zeros(65, np.uint64), np.r_[np.uint64(1), np.zeros(64, np.uint64)], 8)])
        for q, k, d in cases:
            n = len(q)
            previous = None
            for seed in (0, 1, 2**64 - 1, 0x123456789abcdef0):
                output = np.empty(max(1, n * (4 + 12 * max(1, n) * d)), np.int32)
                size = self.function(n, d, q.ctypes.data_as(ctypes.c_void_p),
                                     k.ctypes.data_as(ctypes.c_void_p), seed,
                                     output.ctypes.data_as(ctypes.c_void_p))
                self.assertGreaterEqual(size, 0)
                result = output[:size].copy()
                if previous is not None:
                    np.testing.assert_array_equal(result, previous)
                previous = result
                pos, baseline = 0, []
                for _ in range(n):
                    baseline.append(tuple(map(int, result[pos:pos + 2])))
                    pos += 2
                    for _ in range(2):
                        count = int(result[pos])
                        pos += 1 + count * 6
                self.assertEqual(pos, size)
                self.assertEqual(baseline, independent_forward(q, k))


if __name__ == '__main__':
    unittest.main()
