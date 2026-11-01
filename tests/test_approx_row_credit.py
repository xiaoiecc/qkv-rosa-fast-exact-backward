"""Independent selected-row finite-credit checks, never a runtime backend."""
import itertools,unittest
import numpy as np
from research.approx_round_binding import credits
from tests.test_curvature import quadratic_oracle
from conv_rosa_transformer import rosa_core

class ApproxRowCreditTests(unittest.TestCase):
 def test_binary_integer_and_selected_rows(self):
  rng=np.random.default_rng(441991)
  for n in range(5):
   words=list(itertools.product((0,1),repeat=n))
   for q in words:
    for k in words:
     v=rng.integers(0,2,(n,1),dtype=np.uint8);g=rng.integers(-3,4,(n,1)).astype(float)
     h=2*rng.integers(0,3,(n,1)).astype(float);e0,e1=rng.integers(-3,4,(2,1)).astype(float)
     rows=np.arange(n,dtype=np.int32)[::2];weights=np.full(len(rows),2.)
     gg=np.zeros_like(g);hh=np.zeros_like(h);gg[rows]=2*g[rows];hh[rows]=2*h[rows]
     got=credits(q,k,v,g,e0,e1,rows,weights,h,False)
     want=quadratic_oracle(q,k,v,gg,hh,e0,e1)
     for a,b in zip(got[:3],want[:3]):np.testing.assert_array_equal(a,b)
     for a,b in zip(got[3:5],want[3:5]):np.testing.assert_array_equal(a,b)
 def test_full_sparse_floating_and_bit63(self):
  rng=np.random.default_rng(441992)
  for n,d in [(1,64),(8,64),(17,3),(37,1),(48,8)]:
   for family in ['random','same','shadow','periodic']:
    q=rng.bit_generator.random_raw(n);k=rng.bit_generator.random_raw(n)
    if d<64:q&=np.uint64((1<<d)-1);k&=np.uint64((1<<d)-1)
    if family in ('same','shadow'):q[:]=0;k[:]=0
    if family=='shadow':k[0]=np.uint64(1)<<np.uint64(d-1)
    if family=='periodic':q=np.arange(n,dtype=np.uint64)%min(3,1<<d);k=q.copy();k[n//2]^=np.uint64(1)<<np.uint64(d-1)
    v=rng.integers(0,2,(n,d),dtype=np.uint8);g=rng.normal(size=(n,d));h=rng.uniform(size=(n,d));e0,e1=rng.normal(size=(2,d))
    for second in [False,True]:
     hh=h if second else None
     got=credits(q,k,v,g,e0,e1,h=hh)
     want=rosa_core.quadratic_credits_numpy(q,k,v,g,h,e0,e1) if second else rosa_core.credits_numpy(q,k,v,g,e0,e1)
     for a,b in zip(got[:3],want[:3]):np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-10)
     for a,b in zip(got[3:5],want[3:5]):np.testing.assert_array_equal(a,b)
 def test_all_pairs_of_sampled_rows_expectation(self):
  rng=np.random.default_rng(441993);n=8;d=3
  q=rng.integers(0,8,n,dtype=np.uint64);k=rng.integers(0,8,n,dtype=np.uint64)
  v=rng.integers(0,2,(n,d),dtype=np.uint8);g=rng.normal(size=(n,d));h=rng.uniform(size=(n,d));e0,e1=rng.normal(size=(2,d))
  full=credits(q,k,v,g,e0,e1,h=h)
  estimates=[]
  for rows in itertools.combinations(range(n),2):estimates.append(credits(q,k,v,g,e0,e1,rows,np.array([4.,4.]),h,True)[:3])
  for mean,want in zip(np.mean(estimates,axis=0),full[:3]):np.testing.assert_allclose(mean,want,rtol=2e-13,atol=2e-13)

 def test_onegram_control_independent_oracle(self):
  from tests.oracle import hard_outputs
  rng=np.random.default_rng(441994)
  def gram(q,k,v,e0,e1):
   out=[]
   for t in range(len(q)):
    ends=[r for r in range(t) if q[t]==k[r]]
    out.append(e0 if not ends else (2*v[max(ends)+1].astype(int)-1)*e1)
   return np.asarray(out)
  for trial in range(70):
   n=int(rng.integers(1,9));d=int(rng.integers(1,5))
   q=rng.integers(0,1<<d,n,dtype=np.uint64);k=rng.integers(0,1<<d,n,dtype=np.uint64)
   v=rng.integers(0,2,(n,d),dtype=np.uint8);g=rng.integers(-3,4,(n,d)).astype(float)
   h=2*rng.integers(0,3,(n,d)).astype(float);e0,e1=rng.integers(-3,4,(2,d)).astype(float)
   _,yy=hard_outputs(q,k,v,e0,e1);actual=np.asarray(yy);base=gram(q,k,v,e0,e1)
   def score(y):
    delta=y-actual;return np.sum(g*delta+.5*h*delta*delta)
   baseline_cost=score(base);expected=[np.zeros((n,d)),np.zeros((n,d))]
   for stream in range(2):
    for owner in range(n):
     for bit in range(d):
      qq=q.copy();kk=k.copy();(qq if stream==0 else kk)[owner]^=np.uint64(1<<bit)
      expected[stream][owner,bit]=score(gram(qq,kk,v,e0,e1))-baseline_cost
   got=credits(q,k,v,g,e0,e1,rows=[],h=h,control_variate=True)
   for a,b in zip(got[:2],expected):np.testing.assert_array_equal(a,b)
   full=credits(q,k,v,g,e0,e1,h=h,control_variate=True)
   want=quadratic_oracle(q,k,v,g,h,e0,e1)
   for a,b in zip(full[:3],want[:3]):np.testing.assert_array_equal(a,b)
 def test_control_variate_expectation_and_zero_residual(self):
  rng=np.random.default_rng(441995);n=8;d=3
  q=rng.integers(0,8,n,dtype=np.uint64);k=rng.integers(0,8,n,dtype=np.uint64)
  v=rng.integers(0,2,(n,d),dtype=np.uint8);g=rng.normal(size=(n,d));h=rng.uniform(size=(n,d));e0,e1=rng.normal(size=(2,d))
  full=credits(q,k,v,g,e0,e1,h=h);estimates=[]
  for rows in itertools.combinations(range(n),2):estimates.append(credits(q,k,v,g,e0,e1,rows,np.array([4.,4.]),h,True,True)[:3])
  for mean,want in zip(np.mean(estimates,axis=0),full[:3]):np.testing.assert_allclose(mean,want,rtol=2e-12,atol=2e-12)
  n=48;d=64;k=rng.bit_generator.random_raw(n);q=k.copy();q[1:]=k[:-1]^np.uint64(1<<63)
  v=rng.integers(0,2,(n,d),dtype=np.uint8);g=rng.integers(-3,4,(n,d)).astype(float);h=2*rng.integers(0,3,(n,d)).astype(float);e0,e1=rng.integers(-3,4,(2,d)).astype(float)
  full=credits(q,k,v,g,e0,e1,h=h)
  for rows in [[],[0],[23],[47],[3,19]]:
   got=credits(q,k,v,g,e0,e1,rows,np.full(len(rows),n/max(1,len(rows))),h,True,True)
   for a,b in zip(got[:3],full[:3]):np.testing.assert_array_equal(a,b)

 def test_saved_baseline_zero_and_nonzero_rows(self):
  rng=np.random.default_rng(441996);n=41;d=6
  q=np.arange(n,dtype=np.uint64)%3;k=q.copy();k[3]^=np.uint64(1)
  v=rng.integers(0,2,(n,d),dtype=np.uint8);g=rng.normal(size=(n,d));h=rng.uniform(size=(n,d));e0,e1=rng.normal(size=(2,d))
  full=credits(q,k,v,g,e0,e1,h=h)
  for rows in [[],[0],[12,29,40]]:
   for second in [False,True]:
    hh=h if second else None
    built=credits(q,k,v,g,e0,e1,rows,h=hh,control_variate=True)
    saved=credits(q,k,v,g,e0,e1,rows,h=hh,control_variate=True,
      saved_lengths=full[3],saved_routes=full[4])
    for a,b in zip(saved[:5],built[:5]):np.testing.assert_array_equal(a,b)
    self.assertEqual(int(saved[-1][8]),0)
    if not rows:self.assertEqual(int(saved[-1][15]),0)

 def test_repair_birth_prefix_and_exact_zero_certificate(self):
  rng=np.random.default_rng(441997)
  for n in range(6):
   words=list(itertools.product((0,1),repeat=n))
   for q in words:
    for k in words:
     v=rng.integers(0,2,(n,1),dtype=np.uint8);g=rng.integers(-3,4,(n,1)).astype(float)
     h=2*rng.integers(0,3,(n,1)).astype(float);e0,e1=rng.integers(-3,4,(2,1)).astype(float)
     prefix=np.zeros(n+1,dtype=np.int32)
     control=credits(q,k,v,g,e0,e1,rows=[],h=h,control_variate=True,
       deletion_control=True,repair_prefix=prefix)
     expected=np.r_[0,np.cumsum([any((q[t]^k[s]).bit_count()==1 for s in range(t)) for t in range(n)])]
     np.testing.assert_array_equal(prefix,expected)
     lengths=control[3];certified=np.array([prefix[t+1]==prefix[max(0,t-int(lengths[t]))] for t in range(n)],dtype=bool)
     gg=g.copy();hh=h.copy();gg[~certified]=0;hh[~certified]=0
     got=credits(q,k,v,gg,e0,e1,rows=[],h=hh,control_variate=True,deletion_control=True)
     want=quadratic_oracle(q,k,v,gg,hh,e0,e1)
     for a,b in zip(got[:3],want[:3]):np.testing.assert_array_equal(a,b)

 def test_full_deletion_control_reconstructs_both_orders(self):
  rng=np.random.default_rng(441998)
  for n,d in [(9,3),(30,5),(41,1),(29,64)]:
   for family in ['random','periodic','shadow']:
    q=rng.bit_generator.random_raw(n);k=rng.bit_generator.random_raw(n)
    if d<64:q&=np.uint64((1<<d)-1);k&=np.uint64((1<<d)-1)
    if family=='periodic':q=np.arange(n,dtype=np.uint64)%min(4,1<<d);k=q.copy()
    if family=='shadow':q[:]=0;k[:]=0;k[0]=1
    v=rng.integers(0,2,(n,d),dtype=np.uint8);g=rng.normal(size=(n,d));h=rng.uniform(size=(n,d));e0,e1=rng.normal(size=(2,d))
    for hh in [None,h]:
     full=credits(q,k,v,g,e0,e1,h=hh)
     got=credits(q,k,v,g,e0,e1,h=hh,control_variate=True,deletion_control=True)
     for a,b in zip(got[:3],full[:3]):np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-10)

 def test_short_active_matches_have_onegram_deletion(self):
  from tests.oracle import hard_outputs,hard_routes
  rng=np.random.default_rng(441999)
  cases=[([0,0,2],[1,2,3],2)]
  for trial in range(100):
   n=int(rng.integers(1,13));d=int(rng.integers(1,6))
   cases.append((rng.integers(0,1<<d,n).tolist(),rng.integers(0,1<<d,n).tolist(),d))
  for q,k,d in cases:
   n=len(q);v=rng.integers(0,2,(n,d),dtype=np.uint8)
   g=rng.integers(-3,4,(n,d)).astype(float);h=2*rng.integers(0,3,(n,d)).astype(float)
   e0,e1=rng.integers(-3,4,(2,d)).astype(float)
   routes,values=hard_outputs(q,k,v,e0,e1);actual=np.asarray(values)
   active=np.asarray([length<=1 for length,_ in routes]);g[~active]=0;h[~active]=0
   for hh in [None,h]:
    full=rosa_core.selected_credits_numpy(q,k,v,g,e0,e1,[],diag_y=hh)
    def output_onegram(qq,kk):
     result=[]
     for t in range(n):
      matches=[r for r in range(t) if qq[t]==kk[r]]
      result.append(e0 if not matches else (2*v[max(matches)+1].astype(int)-1)*e1)
     return np.asarray(result)
    base=output_onegram(q,k)
    for stream in range(2):
     expected=[]
     for owner in range(n):
      qq=q.copy();kk=k.copy();(qq if stream==0 else kk)[owner]=-1
      delta=output_onegram(qq,kk)-base
      expected.append(np.sum(g*delta+(0 if hh is None else .5*hh*delta*delta)))
     np.testing.assert_array_equal(full[-2+stream],expected)

 def test_repair_birth_wide_bits_and_empty_numeric_control(self):
  rng=np.random.default_rng(442000);n=47;d=64
  q=rng.bit_generator.random_raw(n);k=rng.bit_generator.random_raw(n)
  q[1::3]=k[::3][:len(q[1::3])]^(np.uint64(1)<<np.uint64(63))
  v=rng.integers(0,2,(n,d),dtype=np.uint8);g=rng.normal(size=(n,d));h=rng.uniform(size=(n,d));e0,e1=rng.normal(size=(2,d))
  prefix=np.zeros(n+1,dtype=np.int32)
  credits(q,k,v,g,e0,e1,rows=[],h=h,control_variate=True,deletion_control=True,repair_prefix=prefix)
  expected=np.r_[0,np.cumsum([any((int(q[t])^int(k[s])).bit_count()==1 for s in range(t)) for t in range(n)])]
  np.testing.assert_array_equal(prefix,expected)
  q=np.arange(n,dtype=np.uint64)%2;q*=np.uint64(3);k=q.copy()
  got=credits(q,k,v,g,e0,e1,rows=[],h=h,control_variate=True,deletion_control=True,repair_prefix=prefix)
  np.testing.assert_array_equal(prefix,np.zeros(n+1,dtype=np.int32))
  self.assertEqual(int(got[-1][12]),0)

 def test_quadratic_control_near_equal_large_payloads(self):
  # A square-difference control loses the entire tiny positive quadratic
  # credit here: the old algebra returned -0.4901 instead of 2**-53.
  for sign in [-1.,1.]:
   e1=np.array([1e8]);e0=np.nextafter(sign*e1,np.inf)
   v=np.full((2,1),int(sign>0),dtype=np.uint8)
   for q,k in [([1,0],[0,1]),([0,1],[0,0])]:
    g=np.zeros((2,1));h=np.ones((2,1))
    for gg in [g,np.ones_like(g)]:
     exact=credits(q,k,v,gg,e0,e1,h=h)
     control=credits(q,k,v,gg,e0,e1,rows=[],h=h,control_variate=True)
     for a,b in zip(control[:3],exact[:3]):np.testing.assert_array_equal(a,b)

if __name__=='__main__':unittest.main()
