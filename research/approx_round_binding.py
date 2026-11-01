"""Research-only binding for complete selected-row credit evaluation."""
from pathlib import Path
import ctypes,hashlib,os,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CSRC=ROOT/'conv_rosa_transformer/csrc'
BUILD=ROOT/'build/approx_round'
_BINDING=r'''
#include "approx_row_credit.h"
#include "approx_onegram.h"
extern "C" int rosa_selected_rows(int n,int d,const uint64_t*q,const uint64_t*k,
const uint8_t*v,const double*g,const double*h,const double*e0,const double*e1,
const int32_t*rows,const double*weights,int count,int exact_v,int control,
double*cq,double*ck,double*cv,int32_t*lengths,int32_t*routes,uint64_t*stats,const int32_t*saved_lengths,const int32_t*saved_routes,int repair_only,int32_t*repair_prefix) {
int code=h?rosa_random::selected_row_credits<true>(n,d,q,k,v,g,h,e0,e1,rows,weights,count,exact_v,cq,ck,cv,lengths,routes,stats,control!=0,saved_lengths,saved_routes,repair_only!=0)
        :rosa_random::selected_row_credits<false>(n,d,q,k,v,g,h,e0,e1,rows,weights,count,exact_v,cq,ck,cv,lengths,routes,stats,control!=0,saved_lengths,saved_routes,repair_only!=0);
if(!code&&control) {
 if(h)rosa_random::onegram_credit<true>(n,d,q,k,v,g,h,e0,e1,routes,cq,ck,stats,!repair_only,repair_prefix);
 else rosa_random::onegram_credit<false>(n,d,q,k,v,g,h,e0,e1,routes,cq,ck,stats,!repair_only,repair_prefix);
}
return code;
}
'''
_lib=None
def library():
 global _lib
 if _lib is not None:return _lib
 BUILD.mkdir(parents=True,exist_ok=True)
 digest=hashlib.sha256(_BINDING.encode())
 for p in sorted(CSRC.glob('*.h')):digest.update(p.read_bytes())
 name='rows_'+digest.hexdigest()[:16]
 source=BUILD/(name+'.cpp');binary=BUILD/(name+'.so')
 if not binary.exists():
  source.write_text(_BINDING)
  subprocess.run([os.environ.get('CXX','c++'),'-O3','-std=c++17','-ffp-contract=off','-fPIC','-shared','-I',str(CSRC),str(source),'-o',str(binary)],check=True)
 _lib=ctypes.CDLL(str(binary));_lib.rosa_selected_rows.restype=ctypes.c_int
 _lib.rosa_selected_rows.argtypes=[ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*9+[ctypes.c_int,ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*8+[ctypes.c_int,ctypes.c_void_p]
 return _lib

def credits(q,k,v,g,e0,e1,rows=None,weights=None,h=None,exact_v=True,control_variate=False,saved_lengths=None,saved_routes=None,deletion_control=False,repair_prefix=None,include_full_deletion=True):
 q=np.ascontiguousarray(q,dtype=np.uint64);k=np.ascontiguousarray(k,dtype=np.uint64)
 v=np.ascontiguousarray(v,dtype=np.uint8);g=np.ascontiguousarray(g,dtype=np.float64)
 e0=np.ascontiguousarray(e0,dtype=np.float64);e1=np.ascontiguousarray(e1,dtype=np.float64)
 n,d=v.shape
 if h is not None:h=np.ascontiguousarray(h,dtype=np.float64)
 if rows is None:rows=np.flatnonzero(np.any(g!=0,axis=1)|(np.any(h!=0,axis=1) if h is not None else False))
 rows=np.ascontiguousarray(rows,dtype=np.int32)
 if weights is None:weights=np.ones(len(rows),dtype=np.float64)
 weights=np.ascontiguousarray(weights,dtype=np.float64)
 out=[np.empty((n,d),dtype=np.float64) for _ in range(3)]
 route=[np.empty(n,dtype=np.int32) for _ in range(2)];stats=np.zeros(16,dtype=np.uint64)
 ptr=lambda x:None if x is None else ctypes.c_void_p(x.ctypes.data)
 saved_lengths=None if saved_lengths is None else np.ascontiguousarray(saved_lengths,dtype=np.int32)
 saved_routes=None if saved_routes is None else np.ascontiguousarray(saved_routes,dtype=np.int32)
 status=library().rosa_selected_rows(n,d,*[ptr(x) for x in [q,k,v,g,h,e0,e1,rows,weights]],len(rows),int(exact_v),int(control_variate),*[ptr(x) for x in out+route+[stats,saved_lengths,saved_routes]],int(deletion_control),ptr(repair_prefix))
 if status:raise RuntimeError(f'selected row status {status}')
 if deletion_control and include_full_deletion:
  from conv_rosa_transformer import rosa_core
  deletion=rosa_core.selected_credits_numpy(q,k,v,g,e0,e1,[],diag_y=h)
  out[0]+=deletion[-2][:,None];out[1]+=deletion[-1][:,None]
 return (*out,*route,stats)
