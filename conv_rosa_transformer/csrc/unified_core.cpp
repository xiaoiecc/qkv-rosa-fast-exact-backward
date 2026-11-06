// One C++17 engine: exact routing, implicit edit intervals and batched correlations.
// No input-family dispatch and no retained quadratic backward implementation.
#if defined(_WIN32)
#define ROSA_EXPORT extern "C" __declspec(dllexport)
#else
#define ROSA_EXPORT extern "C" __attribute__((visibility("default")))
#endif
#include "training_impl.h"

static int stream_credits(int n,int d,const uint64_t*q,const uint64_t*k,
    const uint8_t*v,const double*g,const double*h,const double*e0,const double*e1,
    double*cq,double*ck,double*cv,int32_t*lengths,int32_t*routes,uint64_t*counts) {
  try {
    if(!rosa_exact::training_shape(1,1,n,d)||!counts)return 1;
    const size_t nd=size_t(n)*d;
    if((n&&(!q||!k||!lengths||!routes))||(nd&&(!v||!cq||!ck||!cv)))return 1;
    if(!rosa_exact::finite_values(g,nd)||!rosa_exact::finite_values(e0,d)
        ||!rosa_exact::finite_values(e1,d)||(h&&!rosa_exact::finite_values(h,nd)))return 4;
    if(h)for(size_t i=0;i<nd;++i)if(h[i]<0)return 1;
    for(int t=0;t<n;++t)if(d<64&&((q[t]>>d)||(k[t]>>d)))return 1;
    for(size_t i=0;i<nd;++i)if(v[i]>1)return 1;
    std::fill(counts,counts+ROSA_SUBQUADRATIC_COUNTERS,uint64_t(0));
    if(!n)return 0;
    rosa_subq::Stats work;std::vector<double> qcredit(nd,0.),kcredit(nd,0.);
    const int code=rosa_subq::run(n,d,q,k,v,g,h,e0,e1,true,0,nullptr,
        qcredit,kcredit,lengths,routes,work,nullptr);
    if(code)return code;
    std::copy(qcredit.begin(),qcredit.end(),cq);std::copy(kcredit.begin(),kcredit.end(),ck);
    std::fill(cv,cv+nd,0.);
    for(int t=0;t<n;++t)if(routes[t]>=0)for(int j=0;j<d;++j) {
        const size_t source=size_t(routes[t]+1)*d+j;
        cv[source]+=-2*g[size_t(t)*d+j]*((2*int(v[source])-1)*e1[j]);
    }
    rosa_exact::store_subquadratic_counters(work,counts);
    return 0;
  }catch(const std::bad_alloc&){return 2;}
   catch(const std::exception&){return 3;}
   catch(...){return 3;}
}

// Low-level adapters of the same engine. The diagonal protects Q/K routes;
// V is always the original linear finite credit, as in ordinary training.
ROSA_EXPORT int rosa_uniform(int n,int d,const uint64_t*q,const uint64_t*k,
    const uint8_t*v,const double*g,const double*e0,const double*e1,
    double*cq,double*ck,double*cv,int32_t*lengths,int32_t*routes,uint64_t*counts) {
    return stream_credits(n,d,q,k,v,g,nullptr,e0,e1,cq,ck,cv,lengths,routes,counts);
}
ROSA_EXPORT int rosa_quadratic(int n,int d,const uint64_t*q,const uint64_t*k,
    const uint8_t*v,const double*g,const double*h,const double*e0,const double*e1,
    double*cq,double*ck,double*cv,int32_t*lengths,int32_t*routes,uint64_t*counts) {
    if(!h&&n)return 1;
    return stream_credits(n,d,q,k,v,g,h,e0,e1,cq,ck,cv,lengths,routes,counts);
}
ROSA_EXPORT int rosa_matching_stats(int n,const uint64_t*q,const uint64_t*k,
    int32_t*lengths,int32_t*routes) {
  try {
    if(n<0||n>=(1<<28)||(n&&(!q||!k||!lengths||!routes)))return 1;
    if(n){rosa_exact::SamForward scan(q,k,n);scan.run(lengths,routes);}
    return 0;
  }catch(const std::bad_alloc&){return 2;}
   catch(const std::exception&){return 3;}
   catch(...){return 3;}
}
