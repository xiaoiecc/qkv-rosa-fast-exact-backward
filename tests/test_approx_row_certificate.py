"""Independent proof witnesses for exact zero-residual row certification."""
import ctypes,itertools,os,shlex,subprocess,unittest
import numpy as np
from tests.oracle import hard_routes
from tests._support import BUILD,CSRC

_library=None
def uncertified(q,k,d,active=None,lengths=None,seed=20261101):
 global _library
 if _library is None:
  BUILD.mkdir(parents=True,exist_ok=True)
  source=BUILD/'row_certificate_probe.cpp';binary=BUILD/'row_certificate_probe.so'
  source.write_text('''#include "approx_row_certificate.h"
extern "C" int row_certificate_probe(const uint64_t*q,const uint64_t*k,int n,int d,
 const int32_t*active,int count,const int32_t*lengths,uint64_t seed,int32_t*out,uint64_t*stats) {
 try {auto rows=rosa_random::uncertified_rows(q,k,n,d,active,count,lengths,seed,stats);
 std::copy(rows.begin(),rows.end(),out);return int(rows.size());}catch(...){return -1;}}
''')
  subprocess.run(shlex.split(os.environ.get('CXX','c++'))+['-O3','-std=c++17','-ffp-contract=off',
   '-fPIC','-shared','-I',str(CSRC),str(source),'-o',str(binary)],check=True)
  _library=ctypes.CDLL(str(binary));function=_library.row_certificate_probe
  function.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,
   ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
  function.restype=ctypes.c_int
 q=np.ascontiguousarray(q,dtype=np.uint64);k=np.ascontiguousarray(k,dtype=np.uint64);n=len(q)
 active=np.ascontiguousarray(np.arange(n) if active is None else active,dtype=np.int32)
 lengths=None if lengths is None else np.ascontiguousarray(lengths,dtype=np.int32)
 out=np.empty(len(active),dtype=np.int32);stats=np.zeros(8,dtype=np.uint64)
 pointer=lambda x:None if x is None else ctypes.c_void_p(x.ctypes.data)
 count=_library.row_certificate_probe(pointer(q),pointer(k),n,d,pointer(active),len(active),
  pointer(lengths),seed,pointer(out),pointer(stats))
 if count<0:raise RuntimeError('certificate probe failed')
 return out[:count],stats

class ApproxRowCertificateTests(unittest.TestCase):
 def test_exhaustive_single_bit_routes(self):
  def gram(q,k):
   return [(0,-1) if not (rs:=[r for r in range(t) if q[t]==k[r]]) else (1,max(rs)) for t in range(len(q))]
  for n in range(6):
   words=list(itertools.product((0,1),repeat=n))
   for q in words:
    for k in words:
     bad,_=uncertified(q,k,1);good=sorted(set(range(n))-set(bad.tolist()))
     for stream in range(2):
      for owner in range(-1,n):
       qq=list(q);kk=list(k)
       if owner>=0:(qq if stream==0 else kk)[owner]^=1
       full,local=hard_routes(qq,kk),gram(qq,kk)
       for t in good:self.assertEqual(full[t],local[t],(q,k,stream,owner,t))
 def test_naive_pair_criterion_and_bit63(self):
  rng=np.random.default_rng(2026110101)
  for d in [1,3,8,16,64]:
   for trial in range(40):
    n=int(rng.integers(3,65));q=rng.bit_generator.random_raw(n);k=rng.bit_generator.random_raw(n)
    if d<64:q&=np.uint64((1<<d)-1);k&=np.uint64((1<<d)-1)
    if trial%3==0:q[1:]=k[:-1]
    if trial%4==0:q[n//2]^=np.uint64(1)<<np.uint64(d-1)
    expected=[t for t in range(2,n) if any((int(q[t-1]^k[r-1]).bit_count()+int(q[t]^k[r]).bit_count())<=1 for r in range(1,t))]
    lengths=np.asarray([l for l,r in hard_routes(q.tolist(),k.tolist())],dtype=np.int32)
    for seed in [7,2**64-1]:
     for ls in [None,lengths]:
      got,_=uncertified(q,k,d,lengths=ls,seed=seed)
      self.assertEqual(got.tolist(),expected)
 def test_certified_control_is_complete(self):
  from conv_rosa_transformer.native_training import training_forward_numpy,training_backward_numpy
  rng=np.random.default_rng(2026110102)
  for d in [1,4,8,64]:
   for trial in range(12):
    n=24;q=rng.bit_generator.random_raw(n);k=rng.bit_generator.random_raw(n)
    if d<64:q&=np.uint64((1<<d)-1);k&=np.uint64((1<<d)-1)
    if trial%2:q[1:]=k[:-1]
    v=rng.integers(0,2,(n,d),dtype=np.uint8);g=rng.integers(-3,4,(n,d)).astype(float)
    e0,e1=rng.integers(-3,4,(2,d)).astype(float)
    rows,_=uncertified(q,k,d)
    g[rows]=0
    shifts=np.arange(d,dtype=np.uint64)
    logits=[(2*((words[:,None]>>shifts)&1).astype(float)-1)[None,None]*.2 for words in (q,k)]
    logits.append((2*v.astype(float)-1)[None,None]*.2)
    for order in ('first','second'):
     results=[]
     for mode in ('exact','approximate'):
      _,cache,_=training_forward_numpy(*logits,e0[None],e1[None],credit_order=order,
       joint_pairs=0,backward_mode=mode,approximation_rows=1)
      results.append(training_backward_numpy(cache,g[None,None]))
     for a,b in zip(results[0][:5],results[1][:5]):np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-10)
     self.assertEqual(results[1][-1]['approx_selected_rows'],0)

if __name__=='__main__':unittest.main()
