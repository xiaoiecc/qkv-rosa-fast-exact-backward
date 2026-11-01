#pragma once
// Reusable full-key class structure for a one-symbol credit control.
// apply() adds into caller-owned Q/K arrays.
// Each apply reports construction counters once; callers reusing the plan for
// multiple applies should count join_work once in their aggregate statistics.
#include "sam_q.h"
#include "full_word_map.h"
#include "exact_hamming_join.h"
#include "approx_route_helpers.h"
#ifndef ROSA_ONEGRAM_PLAN_DEFINED
#define ROSA_ONEGRAM_PLAN_DEFINED
namespace rosa_random {
struct GramNeighbor {int bit,other;};
struct OnegramPlan {
    int n,d,nq=0,nk=0;
    const uint64_t*q;const uint64_t*k;
    rosa_exact::FullWordMap qmap,kmap;
    std::vector<uint64_t> qwords,kwords;
    std::vector<int> qclass,kclass,qo,ko,qrows,krows,qsame,ksame,first_neighbor;
    std::vector<std::vector<GramNeighbor>> qneighbors,kneighbors;
    std::vector<int32_t> birth_prefix;
    uint64_t births=0,join_work=0;
    std::vector<int> first_use_[2],numeric_begin_[2];
    std::vector<size_t> prefix_offset_[2];

    OnegramPlan(const uint64_t*q_,const uint64_t*k_,int n_,int d_,
        uint64_t seed=rosa_exact::SamQRepairs::process_hash_seed())
      : n(n_),d(d_),q(q_),k(k_),qmap(size_t(n_),seed),
        kmap(size_t(n_),seed^UINT64_C(0xd6e8feb86659fd93)),
        qclass(n_),kclass(n_),qrows(n_),krows(n_),birth_prefix(size_t(n_)+1,0) {
    if(!n)return;
    std::vector<int> qcounts,kcounts;
    for(int t=0;t<n;++t) {
        auto a=qmap.insert(q[t],int(qwords.size()));
        if(a.second){qwords.push_back(q[t]);qcounts.push_back(0);}
        qclass[t]=a.first;++qcounts[a.first];
        auto b=kmap.insert(k[t],int(kwords.size()));
        if(b.second){kwords.push_back(k[t]);kcounts.push_back(0);}
        kclass[t]=b.first;++kcounts[b.first];
    }
    nq=int(qwords.size());nk=int(kwords.size());
    qo.assign(nq+1,0);ko.assign(nk+1,0);
    for(int i=0;i<nq;++i)qo[i+1]=qo[i]+qcounts[i];
    for(int i=0;i<nk;++i)ko[i+1]=ko[i]+kcounts[i];
    std::vector<int> cursor=qo;
    for(int t=0;t<n;++t)qrows[cursor[qclass[t]]++]=t;
    cursor=ko;
    for(int t=0;t<n;++t)krows[cursor[kclass[t]]++]=t;
    // Build Hamming-one class edges once and use the same edges both ways.
    qneighbors.resize(nq);kneighbors.resize(nk);
    qsame.assign(nq,-1);ksame.assign(nk,-1);
    uint64_t varying=0;
    for(uint64_t word:kwords)varying|=word^kwords.front();
    const uint64_t allowed=d==64?~uint64_t(0):(uint64_t(1)<<d)-1;
    for(int id=0;id<nq;++id) {
        const int same=kmap.find(qwords[id]);
        if(same>=0){qsame[id]=same;ksame[same]=id;}
    }
    rosa_exact::HammingJoinCounters join_counts;
    rosa_exact::exact_hamming_one_join(qwords,qsame,
        rosa_exact::HammingDomain{size_t(nk),kwords.front(),varying},allowed,seed,true,
        [&](uint64_t word){return kmap.find(word);},
        [&](auto emit){for(int id=0;id<nk;++id)emit(kwords[id],id);},
        [&](size_t id,uint64_t mask,const int*targets) {
            for(;mask;mask&=mask-1) {
                const int bit=row_bit(mask&(~mask+1)),other=targets[bit];
                qneighbors[id].push_back({bit,other});
                kneighbors[other].push_back({bit,int(id)});
            }
        },join_counts);
    join_work=join_counts.direct_queries+join_counts.bucket_queries+join_counts.bucket_members;
    // A repair ending at t has a Q mismatch owner u=t-F[r], so u belongs
    // to [t-L[t],t]. Its K mismatch owner s is strictly before u. Mark
    // whether each Q owner has ANY causal Hamming-one K neighbor. A zero
    // prefix-count difference over that interval certifies all full repairs
    // absent, including K repairs; the one-symbol repairs are absent as well.
    // Class first occurrences make this O(number of edges + T), with no
    // occurrence-pair enumeration and no probabilistic equality test.
    first_neighbor.assign(nq,n);
    for(int id=0;id<nq;++id)for(const auto&neighbor:qneighbors[id])
        first_neighbor[id]=std::min(first_neighbor[id],krows[ko[neighbor.other]]);
    births=0;
    for(int t=0;t<n;++t) {
        births+=int(first_neighbor[qclass[t]]<t);
        birth_prefix[t+1]=int32_t(births);
    }


    for(int mode=0;mode<2;++mode) {
    // A class and its early rows with no causal exact/neighbor K word have
    // identically zero onegram credit. Do not allocate or evaluate numeric
    // prefix rows for them. This is support elimination, independent of any
    // input family, and also bounds the memory of wide sparse controls.
    auto&first_use=first_use_[mode];first_use=first_neighbor;
    auto&numeric_begin=numeric_begin_[mode];numeric_begin.resize(nq);
    auto&prefix_offset=prefix_offset_[mode];prefix_offset.assign(nq+1,0);
    for(int id=0;id<nq;++id) {
        if(mode&&qsame[id]>=0)
            first_use[id]=std::min(first_use[id],krows[ko[qsame[id]]]);
        numeric_begin[id]=int(std::upper_bound(qrows.begin()+qo[id],qrows.begin()+qo[id+1],first_use[id])-qrows.begin());
        const int count=qo[id+1]-numeric_begin[id];
        prefix_offset[id+1]=prefix_offset[id]+(count?size_t(count+1):0);
    }

    }
    }

    template<bool Quadratic>
    void apply(const uint8_t*v,const double*g,const double*h,
      const double*e0,const double*e1,const int32_t*actual_routes,
      double*cq,double*ck,uint64_t*stats,bool include_deletion=true,
      const double*row_weight=nullptr) const {
    if(!n)return;
    stats[11]+=join_work;stats[14]=births;
    if(!include_deletion&&!births)return;
    const int stride=Quadratic?3*d:d,mode=int(include_deletion);
    const auto&first_use=first_use_[mode];
    const auto&numeric_begin=numeric_begin_[mode];
    const auto&prefix_offset=prefix_offset_[mode];
    std::vector<double> prefix(prefix_offset.back()*stride,0.);
    for(int id=0;id<nq;++id) {
        if(numeric_begin[id]==qo[id+1])continue;
        double*start=prefix.data()+prefix_offset[id]*stride;
        for(int index=numeric_begin[id];index<qo[id+1];++index) {
            const int t=qrows[index];
            double*dst=start+size_t(index-numeric_begin[id]+1)*stride;
            const double*prev=dst-stride;
            for(int j=0;j<d;++j) {
                const size_t x=size_t(t)*d+j;dst[j]=prev[j]+(row_weight?g[x]*row_weight[t]:g[x]);
                if constexpr(Quadratic) {
                    // Empty-baseline rows can only appear in an interval
                    // whose onegram `from` is empty. Only total H is used
                    // there, so place it in the positive bucket.
                    const bool positive=actual_routes[t]<0||v[size_t(actual_routes[t]+1)*d+j];
                    dst[d+j]=prev[d+j]+(positive?(row_weight?h[x]*row_weight[t]:h[x]):0.);
                    dst[2*d+j]=prev[2*d+j]+(positive?0.:(row_weight?h[x]*row_weight[t]:h[x]));
                }
            }
        }
    }
    auto row_score=[&](int t,int to,int from)->double {
        if(to==from)return 0.;double result=0.;
        for(int j=0;j<d;++j) {
            const double a=to<0?e0[j]:(2*int(v[size_t(to+1)*d+j])-1)*e1[j];
            const double b=from<0?e0[j]:(2*int(v[size_t(from+1)*d+j])-1)*e1[j];
            const double delta=a-b;const size_t x=size_t(t)*d+j;
            result+=(row_weight?g[x]*row_weight[t]:g[x])*delta;
            if constexpr(Quadratic) {
                const double y=actual_routes[t]<0?e0[j]:(2*int(v[size_t(actual_routes[t]+1)*d+j])-1)*e1[j];
                result+=(.5*(row_weight?h[x]*row_weight[t]:h[x])*delta)*((a-y)+(b-y));
            }
        }
        stats[12]+=d;return result;
    };
    std::vector<int> latest(nk,-1);
    for(int t=0;t<n;++t) {
        if(t)latest[kclass[t-1]]=t-1;
        const int id=qclass[t],from=qsame[id]<0?-1:latest[qsame[id]];
        if(t<=first_use[id])continue;
        const double missing=include_deletion?row_score(t,-1,from):0.;
        if(missing!=0.)for(int j=0;j<d;++j)cq[size_t(t)*d+j]+=missing;
        for(const auto&neighbor:qneighbors[id]) {
            const int to=latest[neighbor.other];
            if(to>=0)cq[size_t(t)*d+neighbor.bit]+=row_score(t,to,-1);
        }
    }
    auto interval_score=[&](int id,int left,int right,int to,int from)->double {
        if(to==from||left>=right)return 0.;
        const auto begin=qrows.begin()+numeric_begin[id],end=qrows.begin()+qo[id+1];
        const auto lo=std::lower_bound(begin,end,left),hi=std::lower_bound(lo,end,right);
        if(lo==hi)return 0.;
        const double*start=prefix.data()+prefix_offset[id]*stride;
        const double*upper=start+size_t(hi-begin)*stride,*lower=start+size_t(lo-begin)*stride;
        double result=0.;
        for(int j=0;j<d;++j) {
            const double a=to<0?e0[j]:(2*int(v[size_t(to+1)*d+j])-1)*e1[j];
            const double b=from<0?e0[j]:(2*int(v[size_t(from+1)*d+j])-1)*e1[j];
            const double delta=a-b;
            result+=(upper[j]-lower[j])*delta;
            if constexpr(Quadratic) {
                const double hp=upper[d+j]-lower[d+j],hn=upper[2*d+j]-lower[2*d+j];
                result+=(.5*delta*delta)*(hp+hn);
                // Every interval's `from` is its unedited onegram route.
                // If it is empty the full baseline is empty as well. If
                // matched, full-baseline values are exactly +/- e1; form
                // their differences first, before multiplying the metric.
                if(from>=0)result+=delta*((b-e1[j])*hp+(b+e1[j])*hn);
            }
        }
        ++stats[13];stats[12]+=d;return result;
    };
    for(int id=0;id<nk;++id)for(int index=ko[id];index<ko[id+1];++index) {
        const int s=krows[index];if(s+1>=n)continue;
        const int previous=index>ko[id]?krows[index-1]:-1;
        const int next=index+1<ko[id+1]?krows[index+1]:n;
        if(ksame[id]>=0&&include_deletion) {
            const double deletion=interval_score(ksame[id],s+1,std::min(n,next+1),previous,s);
            if(deletion!=0.)for(int j=0;j<d;++j)ck[size_t(s)*d+j]+=deletion;
        }
        for(const auto&neighbor:kneighbors[id]) {
            const int qid=neighbor.other,kid=qsame[qid];
            int before=-1,after=n;
            if(kid>=0) {
                const auto begin=krows.begin()+ko[kid],end=krows.begin()+ko[kid+1];
                const auto it=std::lower_bound(begin,end,s);
                if(it!=begin)before=*(it-1);
                if(it!=end)after=*it;
            }
            ck[size_t(s)*d+neighbor.bit]+=interval_score(qid,s+1,std::min(n,after+1),s,before);
        }
    }
    }
};
} // namespace rosa_random
#endif

// Compatibility entry point: one structure implementation and one numeric
// implementation. Native training keeps the plan for its certificates/rows.
namespace rosa_random {
template<bool Quadratic>
inline void onegram_credit(int n,int d,const uint64_t*q,const uint64_t*k,
    const uint8_t*v,const double*g,const double*h,const double*e0,const double*e1,
    const int32_t*actual_routes,double*cq,double*ck,uint64_t*stats,
    bool include_deletion=true,int32_t*repair_prefix=nullptr) {
    if(repair_prefix)repair_prefix[0]=0;
    if(!n)return;
    const OnegramPlan plan(q,k,n,d);
    if(repair_prefix)std::copy(plan.birth_prefix.begin(),plan.birth_prefix.end(),repair_prefix);
    plan.apply<Quadratic>(v,g,h,e0,e1,actual_routes,cq,ck,stats,include_deletion);
}
} // namespace rosa_random
