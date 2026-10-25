"""Exact random covering join: complete credits remain bitwise unchanged."""
import ctypes
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import unittest
import numpy as np
from tests._support import BUILD,CSRC


class RandomizedJoinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        BUILD.mkdir(parents=True,exist_ok=True)
        source=BUILD/'randomized_join_probe.cpp'
        source.write_text(r'''#include "unified_core.cpp"
extern "C" int randomized_join_credit_probe(int n,int d,int accelerated,uint64_t seed,
const uint64_t*q,const uint64_t*k,const uint8_t*v,const double*g,const double*h,
const double*e0,const double*e1,double*cq,double*ck,double*cv,int32_t*lengths,int32_t*routes,uint64_t*counts) {
 if(h)return uniform_impl<true>(n,d,q,k,v,g,h,e0,e1,cq,ck,cv,lengths,routes,counts,
  ~uint64_t(0),nullptr,nullptr,nullptr,-1,0,0,accelerated,seed);
 return uniform_impl<false>(n,d,q,k,v,g,nullptr,e0,e1,cq,ck,cv,lengths,routes,counts,
  ~uint64_t(0),nullptr,nullptr,nullptr,-1,0,0,accelerated,seed);
}
extern "C" int randomized_join_prepare_probe(int n,int d,int accelerated,uint64_t seed,
const uint64_t*q,const uint64_t*k,uint64_t*counts) {
 try {rosa_exact::SamQRepairs sam(q,k,n,d,~uint64_t(0),-1,0,0,seed,accelerated);
 const auto&c=sam.root_join_counters;
 counts[0]=c.enabled;counts[1]=c.used;counts[2]=c.direct_queries;counts[3]=c.bucket_queries;
 counts[4]=c.bucket_members;counts[5]=c.bucket_build_entries;counts[6]=c.empty_buckets;counts[7]=c.verified_neighbors;
 counts[8]=sam.query_symbol_count();counts[9]=sam.supported_repair_mask();counts[10]=c.blocks;counts[11]=sam.stored_root_entries();return 0;
 }catch(...){return 1;}}

extern "C" int randomized_join_trace_probe(int n,int d,int accelerated,uint64_t root_seed,
uint64_t repair_mask,int budget,uint64_t sample_seed,uint64_t forced_mask,
const uint64_t*q,const uint64_t*k,int32_t*out) {
 try {
  rosa_exact::SamQRepairs sam(q,k,n,d,repair_mask,budget,sample_seed,forced_mask,root_seed,accelerated);
  std::vector<rosa_exact::SamQRepairs::Repair> detached;int count=0;
  auto append=[&](const auto&list) {out[count++]=int(list.size());
   for(const auto&r:list){out[count++]=r.owner;out[count++]=r.bit;out[count++]=r.length;
    out[count++]=r.end;out[count++]=r.state;out[count++]=r.short_state;}};
  for(int t=0;t<n;++t){sam.advance(t);sam.extract_singleton_continuations(k,detached);
   out[count++]=sam.baseline().length;out[count++]=sam.baseline().end;
   append(sam.repairs());append(detached);}
  return count;
 }catch(...){return -1;}}
''')
        binary=BUILD/'randomized_join_probe.so'
        subprocess.run(shlex.split(os.environ.get('CXX','c++'))+['-O3','-DNDEBUG','-std=c++17','-ffp-contract=off','-shared','-fPIC','-I',str(CSRC),str(source),'-o',str(binary)],check=True)
        cls.library=ctypes.CDLL(str(binary))
        f=cls.library.randomized_join_credit_probe
        f.argtypes=[ctypes.c_int]*3+[ctypes.c_uint64]+[ctypes.c_void_p]*13
        f.restype=ctypes.c_int;cls.credit=staticmethod(f)
        f=cls.library.randomized_join_prepare_probe
        f.argtypes=[ctypes.c_int]*3+[ctypes.c_uint64]+[ctypes.c_void_p]*3
        f.restype=ctypes.c_int;cls.prepare=staticmethod(f)
        f=cls.library.randomized_join_trace_probe
        f.argtypes=[ctypes.c_int]*3+[ctypes.c_uint64]*2+[ctypes.c_int]+[ctypes.c_uint64]*2+[ctypes.c_void_p]*3
        f.restype=ctypes.c_int;cls.trace=staticmethod(f)


    @classmethod
    def call(cls,q,k,v,g,e0,e1,seed,accelerated,h=None):
        n,d=v.shape
        result=[np.zeros((n,d)) for _ in range(3)]+[np.zeros(n,np.int32),np.zeros(n,np.int32),np.zeros(23,np.uint64)]
        values=[q,k,v,g,h,e0,e1]+result
        ptr=[ctypes.c_void_p() if x is None else x.ctypes.data_as(ctypes.c_void_p) for x in values]
        assert cls.credit(n,d,accelerated,seed,*ptr)==0
        return result

    def test_bitwise_full_credits_across_layouts(self):
        rng=np.random.default_rng(981742)
        cases=[]
        for n in (0,1,17,97,257):
          for d in (1,4,8,16,32,64):
            mask=(1<<d)-1
            q=(rng.bit_generator.random_raw(n)&np.uint64(mask)).copy()
            k=(rng.bit_generator.random_raw(n)&np.uint64(mask)).copy()
            for row in range(1,n,3):q[row]=k[row-1]^(np.uint64(1)<<np.uint64(row%d))
            cases.append((q,k,d))
        for n,d in ((65,64),(257,32)):
            q,k=rng.integers(0,64,size=(2,n),dtype=np.uint64)
            cases.append((q,k,d))
            cases.append((np.zeros(n,np.uint64),np.r_[np.uint64(1),np.zeros(n-1,np.uint64)],d))
        q,k=rng.integers(0,1<<16,size=(2,2048),dtype=np.uint64)
        cases.append((q,k,16))
        used=0;seen_blocks=set()
        for q,k,d in cases:
            n=len(q);v=rng.integers(0,2,size=(n,d),dtype=np.uint8)
            g=rng.normal(size=(n,d));e0,e1=rng.normal(size=(2,d));h=rng.random((n,d))
            for seed in (0,1,2**64-1):
              for diagonal in (None,h):
                off=self.call(q,k,v,g,e0,e1,seed,0,diagonal)
                on=self.call(q,k,v,g,e0,e1,seed,1,diagonal)
                for a,b in zip(off,on):np.testing.assert_array_equal(a.view(np.uint8),b.view(np.uint8))
              stats=[]
              for accelerated in (0,1):
                cc=np.zeros(12,np.uint64)
                args=[q.ctypes.data_as(ctypes.c_void_p),k.ctypes.data_as(ctypes.c_void_p),cc.ctypes.data_as(ctypes.c_void_p)]
                self.assertEqual(self.prepare(n,d,accelerated,seed,*args),0);stats.append(cc)
              self.assertEqual(stats[0][7],stats[1][7])
              self.assertEqual(stats[0][9],stats[1][9])
              self.assertLessEqual(int(stats[1][2]+stats[1][4]),int(stats[0][2]))
              used+=int(stats[1][1]);seen_blocks.add(int(stats[1][10]))
        self.assertGreater(used,0)
        self.assertTrue({0,2,4}.issubset(seen_blocks),seen_blocks)

    def test_selected_and_sampled_repair_trajectories(self):
        rng=np.random.default_rng(196271)
        n,d=65,64
        cases=[rng.bit_generator.random_raw((2,n)),rng.integers(0,16,(2,n),dtype=np.uint64)]
        for q,k in cases:
            q=q.copy();k=k.copy()
            for row in range(1,n,3):q[row]=k[row-1]^(np.uint64(1)<<np.uint64(row%64))
            for mask,budget,forced in (((1<<64)-1,-1,0),((1<<64)-1,2,1<<63),((1<<63)|(1<<9)|1,-1,0)):
              for seed in (0,113,2**64-1):
                outputs=[]
                for on in (0,1):
                    out=np.empty(n*(4+12*n*d),np.int32)
                    size=self.trace(n,d,on,seed,mask,budget,8171,forced,
                        q.ctypes.data_as(ctypes.c_void_p),k.ctypes.data_as(ctypes.c_void_p),out.ctypes.data_as(ctypes.c_void_p))
                    self.assertGreaterEqual(size,0);outputs.append(out[:size].copy())
                np.testing.assert_array_equal(*outputs)

if __name__=='__main__':unittest.main()
