"""Exact adjacent-pair certificate for zero one-gram correction rows."""
from pathlib import Path
import ctypes,hashlib,os,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];CSRC=ROOT/'conv_rosa_transformer/csrc'
BUILD=ROOT/'build/approx_round'
_SOURCE=r'''
#include "approx_row_certificate.h"
extern "C" int rosa_residual_rows(const uint64_t*q,const uint64_t*k,int n,int d,
const int32_t*active,int count,const int32_t*lengths,uint64_t seed,int32_t*out,uint64_t*stats) {
 if(n<0||d<1||d>64||count<0||count>n)return -1;
 for(int i=0;i<count;++i)if(active[i]<0||active[i]>=n)return -1;
 try {auto rows=rosa_random::uncertified_rows(q,k,n,d,active,count,lengths,seed,stats);
 std::copy(rows.begin(),rows.end(),out);return int(rows.size());}catch(...){return -2;}
}
'''
_lib=None
def library():
 global _lib
 if _lib is not None:return _lib
 BUILD.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256(_SOURCE.encode())
 for name in ['approx_row_certificate.h','full_word_map.h']:digest.update((CSRC/name).read_bytes())
 name='certificate_'+digest.hexdigest()[:16];source=BUILD/(name+'.cpp');binary=BUILD/(name+'.so')
 if not binary.exists():
  source.write_text(_SOURCE)
  subprocess.run([os.environ.get('CXX','c++'),'-O3','-std=c++17','-ffp-contract=off','-fPIC','-shared','-I',str(CSRC),str(source),'-o',str(binary)],check=True)
 _lib=ctypes.CDLL(str(binary));_lib.rosa_residual_rows.restype=ctypes.c_int
 _lib.rosa_residual_rows.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
 return _lib

def uncertified(q,k,d,active=None,lengths=None,seed=20261101):
 q=np.ascontiguousarray(q,dtype=np.uint64);k=np.ascontiguousarray(k,dtype=np.uint64);n=len(q)
 if k.shape!=q.shape or q.ndim!=1:raise ValueError('Q/K need identical 1-D shapes')
 active=np.ascontiguousarray(np.arange(n) if active is None else active,dtype=np.int32)
 if lengths is not None:
  lengths=np.ascontiguousarray(lengths,dtype=np.int32)
  if lengths.shape!=(n,):raise ValueError('lengths shape')
 out=np.empty(len(active),dtype=np.int32);stats=np.zeros(8,dtype=np.uint64)
 ptr=lambda a:None if a is None else ctypes.c_void_p(a.ctypes.data)
 result=library().rosa_residual_rows(ptr(q),ptr(k),n,d,ptr(active),len(active),ptr(lengths),seed,ptr(out),ptr(stats))
 if result<0:raise RuntimeError(f'certificate status {result}')
 return out[:result],stats
