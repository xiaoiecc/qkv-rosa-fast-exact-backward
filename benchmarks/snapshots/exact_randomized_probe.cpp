#include "unified_core.cpp"
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
