#pragma once
// Complete selected-row repair evaluator. Route maxima are
// unchanged; class indexes remove endpoints that provably cannot contribute.
#include "approx_onegram.h"
#include "sam_routes.h"
#include <memory>
#include <array>

namespace rosa_random {
struct IndexedRowStep {int lo,hi,end;};
struct IndexedNeighborSpan {int bit;const int*begin;const int*end;};

// Adds sum(weight * full repair), optionally subtracting the identical
// one-symbol repair for those same rows. The caller owns exact full deletion,
// value/embedding derivatives, and the full one-symbol control. No baseline
// routes or output arrays are cleared or rebuilt by this function.
template<bool Quadratic>
inline int indexed_row_repairs(const OnegramPlan&plan,const uint8_t*v,
    const double*gradient,const double*diagonal,const double*empty,const double*scale,
    const int32_t*selected,const double*weights,int count,
    const int32_t*lengths,const int32_t*routes,double*cq,double*ck,uint64_t*stats,
    bool subtract_onegram=true) {
  try {
    const int n=plan.n,d=plan.d;const uint64_t*q=plan.q,*k=plan.k;
    if(count<0)return 1;
    std::fill(stats,stats+16,uint64_t(0));stats[0]=count;
    if(!n)return count?1:0;
    if(!count)return 0;
    const size_t nd=size_t(n)*d;
    std::unique_ptr<rosa_exact::SamQRepairs> sam;
    using Index=rosa_exact::SamRoutes<rosa_exact::SamQRepairs>;
    std::unique_ptr<Index> index;
    auto lcs=[&](int t,int r)->int {
        if(t<0||r<0)return 0;
        for(int matched=0;matched<4;++matched)
            if(r<matched||q[t-matched]!=k[r-matched])return matched;
        if(!index) {
            sam=std::make_unique<rosa_exact::SamQRepairs>(q,k,n,0);
            index=std::make_unique<Index>(*sam,q,k);
            for(int row=0;row<n;++row){sam->advance(row);index->observe();}
            stats[15]=n;
        }
        return index->lcs(t,r);
    };
    std::vector<int> f;
    std::vector<RowRoute> prefix,qbest(nd,0),kbest(nd,0);
    std::vector<IndexedRowStep> steps;
    std::vector<size_t> qtouch,ktouch;
    std::vector<double> cost(n+1);
    std::vector<int> cost_stamp(n+1,-1);
    for(int sample=0;sample<count;++sample) {
        const int t=selected[sample];const double weight=weights?weights[sample]:1.;
        if(t<0||t>=n||!std::isfinite(weight)||weight<0)return 1;
        if(!weight)continue;
        const int L=lengths[t],r0=routes[t],first=r0-L+1;
        const RowRoute baseline=row_route(L,r0);
        const double*g=gradient+size_t(t)*d;
        const double*h=Quadratic?diagonal+size_t(t)*d:nullptr;
        bool nonzero=false;
        for(int j=0;j<d;++j)nonzero|=g[j]!=0||(Quadratic&&h[j]!=0);
        if(!nonzero)continue;
        ++stats[10];

        // Only exact last-word matches have positive F[r]. Their causal
        // endpoint order supplies A(s)=max_{r<s}(F[r],r) without any sort.
        const int qid=plan.qclass[t],kid=plan.qsame[qid];
        const int*same=nullptr;int same_count=0;
        if(kid>=0) {
            same=plan.krows.data()+plan.ko[kid];
            same_count=int(std::lower_bound(same,plan.krows.data()+plan.ko[kid+1],t)-same);
        }
        f.resize(same_count);prefix.resize(size_t(same_count)+1);prefix[0]=0;
        for(int at=0;at<same_count;++at) {
            const int r=same[at];f[at]=lcs(t,r);++stats[2];
            prefix[at+1]=std::max(prefix[at],row_route(f[at],r));
        }
        stats[1]+=same_count;
        // R(h)=latest endpoint with F[r]>=h. A reverse scan appends a
        // staircase interval only when the maximum observed F increases.
        steps.clear();int covered=0;
        for(int at=same_count-1;at>=0&&covered<L;--at)if(f[at]>covered) {
            steps.push_back({covered+1,f[at],same[at]});covered=f[at];
        }
        if(covered<L)return 1; // Invalid externally supplied forward cache.
        auto qdel=[&](int owner)->RowRoute {
            const int cap=t-owner;
            if(cap>=L)return baseline;
            if(cap<=0)return 0;
            int lo=0,hi=int(steps.size());
            while(lo<hi){const int mid=(lo+hi)/2;if(steps[mid].hi<cap)lo=mid+1;else hi=mid;}
            return row_route(cap,steps[lo].end);
        };
        auto kdel=[&](int owner)->RowRoute {
            if(!L||owner<first||owner>r0)return baseline;
            const int before=int(std::lower_bound(same,same+same_count,owner)-same);
            RowRoute result=prefix[before];
            // A suffix candidate supports length h iff R(h)-h>=owner.
            // end-lo decreases strictly across the R staircase. The final
            // feasible step gives h=min(hi,end-owner), and its latest end.
            int lo=0,hi=int(steps.size());
            while(lo<hi) {
                const int mid=(lo+hi)/2;
                if(steps[mid].end-steps[mid].lo>=owner)lo=mid+1;else hi=mid;
            }
            if(lo) {
                const auto&s=steps[lo-1];
                result=std::max(result,row_route(std::min(s.hi,s.end-owner),s.end));
            }
            return result;
        };
        auto score=[&](int endpoint)->double {
            if(endpoint==r0)return 0.;
            const int cell=endpoint+1;
            if(cost_stamp[cell]!=sample) {
                double result=0.;
                for(int j=0;j<d;++j) {
                    const double old=r0<0?empty[j]:(2*int(v[size_t(r0+1)*d+j])-1)*scale[j];
                    const double now=endpoint<0?empty[j]:(2*int(v[size_t(endpoint+1)*d+j])-1)*scale[j];
                    const double delta=now-old;
                    result+=Quadratic?g[j]*delta+(.5*h[j]*delta)*delta:g[j]*delta;
                }
                cost[cell]=result*weight;cost_stamp[cell]=sample;stats[6]+=d;
            }
            return cost[cell];
        };
        qtouch.clear();ktouch.clear();
        auto consider=[&](int r,int a,int bit) {
            const int u=t-a,s=r-a;if(s<0)return;
            const int b=s?lcs(u-1,s-1):0;
            stats[2]+=s>0;++stats[3];
            const RowRoute candidate=row_route(a+1+b,r);
            const size_t qi=size_t(u)*d+bit,ki=size_t(s)*d+bit;
            if(candidate>qdel(u)) {
                if(!qbest[qi])qtouch.push_back(qi);
                qbest[qi]=std::max(qbest[qi],candidate);
            }
            if(candidate>kdel(s)) {
                if(!kbest[ki])ktouch.push_back(ki);
                kbest[ki]=std::max(kbest[ki],candidate);
            }
        };
        for(int at=0;at<same_count;++at) {
            const int a=f[at],u=t-a,s=same[at]-a;if(s<0)continue;
            const uint64_t difference=q[u]^k[s];
            if(!difference||(difference&(difference-1)))continue;
            const int bit=row_bit(difference);if(bit<d)consider(same[at],a,bit);
        }
        // Every F[r]==0 repair has K[r] at Hamming distance one from Q[t].
        // All other last-word classes are impossible and never need an LCE.
        std::array<IndexedNeighborSpan,64> neighbors;int neighbor_count=0;
        for(const auto&neighbor:plan.qneighbors[qid]) {
            const int*begin=plan.krows.data()+plan.ko[neighbor.other];
            const int*end=std::lower_bound(begin,plan.krows.data()+plan.ko[neighbor.other+1],t);
            if(begin==end)continue;
            neighbors[neighbor_count++]={neighbor.bit,begin,end};stats[1]+=end-begin;
            for(const int*at=begin;at!=end;++at)consider(*at,0,neighbor.bit);
        }
        for(size_t key:qtouch){cq[key]+=score(row_end(qbest[key]))-score(row_end(qdel(int(key/d))));qbest[key]=0;}
        for(size_t key:ktouch){ck[key]+=score(row_end(kbest[key]))-score(row_end(kdel(int(key/d))));kbest[key]=0;}
        stats[4]+=qtouch.size();stats[5]+=ktouch.size();
        if(subtract_onegram) {
            const int gram_last=same_count?same[same_count-1]:-1;
            const double original=score(gram_last),deleted=score(-1);
            for(int at=0;at<neighbor_count;++at) {
                const auto&neighbor=neighbors[at];
                cq[size_t(t)*d+neighbor.bit]-=score(*(neighbor.end-1))-deleted;
            }
            for(int at=0;at<neighbor_count;++at) {
                const auto&neighbor=neighbors[at];
                const int*begin=std::upper_bound(neighbor.begin,neighbor.end,gram_last);
                for(const int*r=begin;r!=neighbor.end;++r)
                    ck[size_t(*r)*d+neighbor.bit]-=score(*r)-original;
            }
        }
    }
    stats[9]=index?index->materialized_index_flags():0;
    return 0;
  }catch(const std::bad_alloc&){return 2;}catch(const std::exception&){return 3;}
}
} // namespace rosa_random
