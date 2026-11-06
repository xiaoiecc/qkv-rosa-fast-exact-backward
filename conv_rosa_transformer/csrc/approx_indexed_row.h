#pragma once
// Complete selected-row repair evaluator. Route maxima are
// unchanged; class indexes remove endpoints that provably cannot contribute.
#include "approx_onegram.h"
#include <array>
#include <cmath>

namespace rosa_random {
struct IndexedRowStep {int lo,hi,end;};
struct IndexedNeighborSpan {int bit;const int*begin;const int*end;};

// Adds sum(weight * full repair), optionally subtracting the identical
// one-symbol repair for those same rows. The caller owns exact full deletion,
// value/embedding derivatives, and the full one-symbol control. No baseline
// routes or output arrays are cleared or rebuilt by this function.
// The caller supplies LCS queries from the same suffix index used by the
// complete engine; selected rows never construct another matcher.
template<bool Quadratic>
inline int indexed_row_repairs(const OnegramPlan&plan,const uint8_t*v,
    const double*gradient,const double*diagonal,const double*empty,const double*scale,
    const int32_t*selected,const double*weights,int count,
    const int32_t*lengths,const int32_t*routes,double*cq,double*ck,uint64_t*stats,
    bool subtract_onegram,
    int(*external_lcs)(const void*,int,int),const void*external_context,
    size_t* workspace=nullptr) {
    if(workspace)*workspace=0;
  try {
    const int n=plan.n,d=plan.d;const uint64_t*q=plan.q,*k=plan.k;
    if(count<0 || !external_lcs)return 1;
    std::fill(stats,stats+16,uint64_t(0));stats[0]=count;
    if(!n)return count?1:0;
    if(!count)return 0;
    const size_t nd=size_t(n)*d;
    // Selected adjacent rows share the exact recurrence F[t,r]=F[t-1,r-1]+1.
    // Keep a bounded row cache so the endpoint join can reuse those answers
    // and its nearby left-extension queries. Sparse/gapped samples still use
    // the same shared suffix index; no quadratic table is ever constructed.
    constexpr int history_limit=32;
    bool adjacent=false;
    for(int i=1;i<count;++i)adjacent|=selected[i]==selected[i-1]+1;
    std::array<int,history_limit> history_time;history_time.fill(-2);
    std::vector<int> history(adjacent?size_t(n)*history_limit:0);
    auto lcs=[&](int t,int r)->int {
        if(t<0||r<0)return 0;
        if(adjacent) {
            if(q[t]!=k[r])return 0;
            if(history_time[t%history_limit]==t)return history[size_t(t%history_limit)*n+r];
            if(t&&history_time[(t-1)%history_limit]==t-1)
                return 1+(r&&q[t-1]==k[r-1]?history[size_t((t-1)%history_limit)*n+r-1]:0);
        }
        return external_lcs(external_context,t,r);
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
        const int gram_last=same_count?same[same_count-1]:-1;
        f.resize(same_count);prefix.resize(size_t(same_count)+1);prefix[0]=0;
        for(int at=0;at<same_count;++at) {
            const int r=same[at];f[at]=lcs(t,r);++stats[2];
            prefix[at+1]=std::max(prefix[at],row_route(f[at],r));
        }
        if(adjacent) {
            int*saved=history.data()+size_t(t%history_limit)*n;
            for(int at=0;at<same_count;++at)saved[same[at]]=f[at];
            history_time[t%history_limit]=t;++stats[15];
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
            // The deleted Q route has length min(L,a), whereas this repair
            // already has a+1 matched symbols. It wins without another
            // staircase lookup; compute the deleted score only at emission.
            if(!qbest[qi])qtouch.push_back(qi);
            qbest[qi]=std::max(qbest[qi],candidate);
            // A repair beating the unedited winner also beats its deletion.
            // Losing candidates need no repeated owner lookup after a better
            // repair has already been retained for the same owner and bit.
            // With a length-zero/one baseline, a new last-word match after
            // gram_last has exactly the same returned endpoint as its
            // one-symbol control, regardless of its left extension. No
            // positive-a candidate can touch that owner: its endpoint is
            // an original last-word match, hence at most gram_last.
            if(!(subtract_onegram&&a==0&&L<=1&&s>gram_last)
                &&candidate>kbest[ki] && (candidate>baseline||candidate>kdel(s))) {
                if(!kbest[ki])ktouch.push_back(ki);
                kbest[ki]=candidate;
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
        std::array<int,64> neighbor_last;
        for(const auto&neighbor:plan.qneighbors[qid]) {
            const int*begin=plan.krows.data()+plan.ko[neighbor.other];
            const int*end=std::lower_bound(begin,plan.krows.data()+plan.ko[neighbor.other+1],t);
            if(begin==end)continue;
            neighbors[neighbor_count++]={neighbor.bit,begin,end};stats[1]+=end-begin;
            neighbor_last[neighbor.bit]=*(end-1);
            for(const int*at=begin;at!=end;++at) {
                // The zero-left-extension Q candidate is already in the
                // one-symbol reference. For K it either cancels that same
                // reference (L<=1), or loses to the surviving original
                // suffix (L>=2). Only a genuine left extension remains.
                if(!subtract_onegram||(t&&*at&&q[t-1]==k[*at-1]))
                    consider(*at,0,neighbor.bit);
            }
        }
        for(size_t key:qtouch) {
            const int owner=int(key/d);
            // At owner=t, full repair minus the one-symbol repair is a
            // single endpoint difference. Evaluating it directly also
            // avoids cancelling two large empty-route scores numerically.
            const int before=subtract_onegram&&owner==t
                ?neighbor_last[key%d]:row_end(qdel(owner));
            cq[key]+=score(row_end(qbest[key]))-score(before);qbest[key]=0;
        }
        for(size_t key:ktouch){ck[key]+=score(row_end(kbest[key]))-score(row_end(kdel(int(key/d))));kbest[key]=0;}
        stats[4]+=qtouch.size();stats[5]+=ktouch.size();
        if(subtract_onegram&&L>=2) {
            const double original=score(gram_last);
            for(int at=0;at<neighbor_count;++at) {
                const auto&neighbor=neighbors[at];
                const int*begin=std::upper_bound(neighbor.begin,neighbor.end,gram_last);
                for(const int*r=begin;r!=neighbor.end;++r)
                    ck[size_t(*r)*d+neighbor.bit]-=score(*r)-original;
            }
        }
    }
    if(workspace)*workspace=(f.capacity()+history.capacity()+cost_stamp.capacity())*sizeof(int)
        +(prefix.capacity()+qbest.capacity()+kbest.capacity())*sizeof(RowRoute)
        +steps.capacity()*sizeof(IndexedRowStep)+(qtouch.capacity()+ktouch.capacity())*sizeof(size_t)
        +cost.capacity()*sizeof(double);
    return 0;
  }catch(const std::bad_alloc&){return 2;}catch(const std::exception&){return 3;}
}
} // namespace rosa_random
