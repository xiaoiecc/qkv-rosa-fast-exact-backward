"""Independent proof witnesses for exact zero-residual row certification."""
import itertools,unittest
import numpy as np
from research.approx_round_certificate import uncertified
from tests.oracle import hard_routes

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
  from research.approx_round_binding import credits
  rng=np.random.default_rng(2026110102)
  for d in [1,4,8,64]:
   for trial in range(12):
    n=24;q=rng.bit_generator.random_raw(n);k=rng.bit_generator.random_raw(n)
    if d<64:q&=np.uint64((1<<d)-1);k&=np.uint64((1<<d)-1)
    if trial%2:q[1:]=k[:-1]
    v=rng.integers(0,2,(n,d),dtype=np.uint8);g=rng.integers(-3,4,(n,d)).astype(float)
    h=2*rng.integers(0,3,(n,d)).astype(float);e0,e1=rng.integers(-3,4,(2,d)).astype(float)
    rows,_=uncertified(q,k,d)
    actual=credits(q,k,v,g,e0,e1,rows=rows,h=h,control_variate=True)
    expected=credits(q,k,v,g,e0,e1,h=h)
    for a,b in zip(actual[:3],expected[:3]):np.testing.assert_array_equal(a,b)

if __name__=='__main__':unittest.main()
