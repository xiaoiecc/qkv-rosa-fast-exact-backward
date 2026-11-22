#pragma once
// Reusable full-key class structure for a one-symbol credit control.
// apply() adds into caller-owned Q/K arrays.
// Each apply reports construction counters once; callers reusing the plan for
// multiple applies should count join_work once in their aggregate statistics.
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
#include "exact_expansion.h"
#include "full_word_map.h"
#include "exact_hamming_join.h"
#include "approx_route_helpers.h"
#ifndef ROSA_ONEGRAM_PLAN_DEFINED
#define ROSA_ONEGRAM_PLAN_DEFINED
namespace rosa_random {
struct GramNeighbor {int bit,other;};
struct GramGroups {
    std::vector<int> offsets;
    std::vector<GramNeighbor> edges;
    struct Range {
        std::vector<GramNeighbor>::const_iterator first,last;
        auto begin()const{return first;}
        auto end()const{return last;}
    };
    Range operator[](int group)const {
        return {edges.begin()+offsets[group],edges.begin()+offsets[group+1]};
    }
};
struct OnegramPlan {
    int n,d,nq=0,nk=0;
    const uint64_t*q;const uint64_t*k;
    rosa_exact::FullWordMap qmap,kmap;
    std::vector<uint64_t> qwords,kwords;
    std::vector<int> qclass,kclass,qo,ko,qrows,krows,qsame,ksame,first_neighbor;
    GramGroups qneighbors,kneighbors;
    std::vector<int32_t> birth_prefix;
    uint64_t births=0,join_work=0;
    std::vector<int> first_use_[2],numeric_begin_[2];
    std::vector<size_t> prefix_offset_[2];

    OnegramPlan(const uint64_t*q_,const uint64_t*k_,int n_,int d_,
        uint64_t seed=UINT64_C(0x243f6a8885a308d3))
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
    qneighbors.offsets.assign(nq+1,0);kneighbors.offsets.assign(nk+1,0);
    qneighbors.edges.reserve(nq);
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
                qneighbors.edges.push_back({bit,other});
                ++kneighbors.offsets[other+1];
            }
            qneighbors.offsets[id+1]=int(qneighbors.edges.size());
        },join_counts);
    // One contiguous adjacency array per direction avoids a separate heap
    // allocation for each symbol. Stable transposition preserves the exact
    // class/bit iteration order used by the numerical control.
    for(int id=0;id<nk;++id)kneighbors.offsets[id+1]+=kneighbors.offsets[id];
    kneighbors.edges.resize(qneighbors.edges.size());cursor=kneighbors.offsets;
    for(int id=0;id<nq;++id)for(const auto&edge:qneighbors[id])
        kneighbors.edges[cursor[edge.other]++]={edge.bit,id};
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

    // Retained vector capacities and dictionary payloads, excluding allocator
    // bookkeeping and the caller-owned Q/K streams.
    size_t workspace_bytes() const {
        size_t bytes=qmap.workspace_bytes()+kmap.workspace_bytes();
        bytes+=(qwords.capacity()+kwords.capacity())*sizeof(uint64_t);
        bytes+=(qclass.capacity()+kclass.capacity()+qo.capacity()+ko.capacity()
            +qrows.capacity()+krows.capacity()+qsame.capacity()+ksame.capacity()
            +first_neighbor.capacity())*sizeof(int);
        bytes+=birth_prefix.capacity()*sizeof(int32_t);
        bytes+=(qneighbors.offsets.capacity()+kneighbors.offsets.capacity())*sizeof(int)
            +(qneighbors.edges.capacity()+kneighbors.edges.capacity())*sizeof(GramNeighbor);
        for(int mode=0;mode<2;++mode) {
            bytes+=(first_use_[mode].capacity()+numeric_begin_[mode].capacity())*sizeof(int);
            bytes+=prefix_offset_[mode].capacity()*sizeof(size_t);
        }
        return bytes;
    }

    template<bool Quadratic>
    void apply(const uint8_t*v,const double*g,const double*h,
      const double*e0,const double*e1,const int32_t*actual_routes,
      double*cq,double*ck,uint64_t*stats,bool include_deletion=true,
      const double*row_weight=nullptr,size_t* numeric_workspace=nullptr,
      bool baseline_is_onegram=false) const {
    if(numeric_workspace)*numeric_workspace=0;
    if(!n)return;
    stats[11]+=join_work;stats[14]=births;
    if(!include_deletion&&!births)return;
    // When the caller has certified every contributing baseline length <=1,
    // each K interval's `from` is that actual baseline. Its quadratic cost
    // needs only sum(G) and sum(H), not separate +/- baseline H buckets.
    const int stride=Quadratic?(baseline_is_onegram?2*d:3*d):d,mode=int(include_deletion);
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
                    if(baseline_is_onegram) {
                        dst[d+j]=prev[d+j]+(row_weight?h[x]*row_weight[t]:h[x]);
                        continue;
                    }
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
                if(from==actual_routes[t])
                    result+=(.5*(row_weight?h[x]*row_weight[t]:h[x])*delta)*delta;
                else {
                    const double y=actual_routes[t]<0?e0[j]:(2*int(v[size_t(actual_routes[t]+1)*d+j])-1)*e1[j];
                    result+=(.5*(row_weight?h[x]*row_weight[t]:h[x])*delta)*((a-y)+(b-y));
                }
            }
        }
        stats[12]+=d;return result;
    };
    std::vector<int> latest(nk,-1);
    if(numeric_workspace)*numeric_workspace=prefix.capacity()*sizeof(double)+latest.capacity()*sizeof(int);
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
    auto interval_score=[&](int id,int lo,int hi,int to,int from)->double {
        if(to==from||lo>=hi)return 0.;
        const double*start=prefix.data()+prefix_offset[id]*stride;
        const double*upper=start+size_t(hi-numeric_begin[id])*stride;
        const double*lower=start+size_t(lo-numeric_begin[id])*stride;
        double result=0.;
        for(int j=0;j<d;++j) {
            const double a=to<0?e0[j]:(2*int(v[size_t(to+1)*d+j])-1)*e1[j];
            const double b=from<0?e0[j]:(2*int(v[size_t(from+1)*d+j])-1)*e1[j];
            const double delta=a-b;
            result+=(upper[j]-lower[j])*delta;
            if constexpr(Quadratic) {
                if(baseline_is_onegram)result+=(.5*delta*delta)*(upper[d+j]-lower[d+j]);
                else {
                    const double hp=upper[d+j]-lower[d+j],hn=upper[2*d+j]-lower[2*d+j];
                    result+=(.5*delta*delta)*(hp+hn);
                    // Every interval's `from` is its unedited onegram route.
                    // If it is empty the full baseline is empty as well. If
                    // matched, full-baseline values are exactly +/- e1; form
                    // their differences first, before multiplying the metric.
                    if(from>=0)result+=delta*((b-e1[j])*hp+(b+e1[j])*hn);
                }
            }
        }
        ++stats[13];stats[12]+=d;return result;
    };
    // For one owner and bit there is at most one neighboring Q class.
    // Delete first, then visit each class edge in increasing owner order:
    // this preserves the arithmetic order of every output coordinate while
    // merging sorted occurrence lists instead of repeating binary searches.
    if(include_deletion)for(int id=0;id<nk;++id)if(ksame[id]>=0) {
        const int qid=ksame[id],qend=qo[qid+1];
        int lo=numeric_begin[qid],hi=lo;
        for(int index=ko[id];index<ko[id+1];++index) {
            const int s=krows[index];if(s+1>=n)break;
            const int previous=index>ko[id]?krows[index-1]:-1;
            const int next=index+1<ko[id+1]?krows[index+1]:n;
            while(lo<qend && qrows[lo]<=s)++lo;
            hi=std::max(hi,lo);
            while(hi<qend && qrows[hi]<=next)++hi;
            const double deletion=interval_score(qid,lo,hi,previous,s);
            if(deletion!=0.)for(int j=0;j<d;++j)ck[size_t(s)*d+j]+=deletion;
        }
    }
    for(int id=0;id<nk;++id)for(const auto&neighbor:kneighbors[id]) {
        const int qid=neighbor.other,kid=qsame[qid],qend=qo[qid+1];
        int lo=numeric_begin[qid],hi=lo;
        int next_match=kid<0?0:ko[kid];
        const int match_end=kid<0?0:ko[kid+1];
        for(int index=ko[id];index<ko[id+1];++index) {
            const int s=krows[index];if(s+1>=n)break;
            int before=-1,after=n;
            if(kid>=0) {
                while(next_match<match_end && krows[next_match]<s)++next_match;
                if(next_match>ko[kid])before=krows[next_match-1];
                if(next_match<match_end)after=krows[next_match];
            }
            while(lo<qend && qrows[lo]<=s)++lo;
            hi=std::max(hi,lo);
            while(hi<qend && qrows[hi]<=after)++hi;
            ck[size_t(s)*d+neighbor.bit]+=interval_score(qid,lo,hi,s,before);
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
