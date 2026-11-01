#pragma once
// Dense-row reference for independent tests/research; native training does not
// include this evaluator.
// Exact per-output-row counterfactual credits, with external row weights.
// An approximate estimator may select and weight rows; every selected row's
// routing maximum is computed completely. This code never samples candidates
// before taking a maximum. It uses a different floating summation order from
// the streaming engine, so real-arithmetic equality is not a bitwise promise.
#include "sam_q.h"
#include "sam_routes.h"
#include "forward.h"
#include <memory>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "approx_route_helpers.h"

namespace rosa_random {
// counters: selected rows, summed endpoint count, LCE queries, one-bit pairs,
// Q touched keys, K touched keys, score components, deletion owner updates,
// live baseline rows, LCA/wavelet materialization flags, actual row evaluations.
template<bool Quadratic>
inline int selected_row_credits(int n,int d,const uint64_t*q,const uint64_t*k,
    const uint8_t*v,const double*gradient,const double*diagonal,
    const double*empty,const double*scale,const int32_t*selected,
    const double*weights,int count,bool exact_value,
    double*cq,double*ck,double*cv,int32_t*lengths,int32_t*routes,uint64_t*stats,
    bool subtract_onegram=false,const int32_t*saved_lengths=nullptr,
    const int32_t*saved_routes=nullptr,bool repair_only=false,
    bool accumulate=false,bool compute_value=true) {
  try {
    if(n<0||n>=(1<<28)||d<1||d>64||count<0)return 1;
    const size_t nd=size_t(n)*d;
    if(nd>size_t(std::numeric_limits<int>::max()))return 1;
    if(!accumulate) {
        std::fill(cq,cq+nd,0.);std::fill(ck,ck+nd,0.);std::fill(cv,cv+nd,0.);
    }
    std::fill(stats,stats+16,uint64_t(0));stats[0]=count;
    if(!n)return count?1:0;
    // Native training already owns the forward routes. In particular, an
    // all-certified control-variate call has no need to build a SAM or LCE.
    if(saved_lengths&&saved_routes) {
        if(lengths!=saved_lengths)std::copy(saved_lengths,saved_lengths+n,lengths);
        if(routes!=saved_routes)std::copy(saved_routes,saved_routes+n,routes);
    } else {
        rosa_exact::SamForward scan(q,k,n);scan.run(lengths,routes);stats[8]=n;
    }
    if(!count) {
        if(compute_value&&exact_value)for(int t=0;t<n;++t)if(routes[t]>=0)for(int j=0;j<d;++j) {
            const size_t x=size_t(routes[t]+1)*d+j,y=size_t(t)*d+j;
            const double delta=-2*(2*int(v[x])-1)*scale[j];
            cv[x]+=Quadratic?gradient[y]*delta+(.5*diagonal[y]*delta)*delta:gradient[y]*delta;
        }
        return 0;
    }
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
    std::vector<int> f(n),latest(n+1),flat_head(n),flat_next(n);
    std::vector<RowRoute> prefix(n+1),kdelete(n),qbest(nd,0),kbest(nd,0);
    std::vector<size_t> qtouch,ktouch;
    std::vector<double> cost(n+1);
    std::vector<int> cost_stamp(n+1,-1);
    for(int sample=0;sample<count;++sample) {
        const int t=selected[sample];const double weight=weights?weights[sample]:1.;
        if(t<0||t>=n||!std::isfinite(weight)||weight<0)return 1;
        if(!weight)continue;
        const int L=lengths[t],r0=routes[t];const RowRoute baseline=row_route(L,r0);
        const int first=r0-L+1;
        const double*g=gradient+size_t(t)*d;
        const double*h=Quadratic?diagonal+size_t(t)*d:nullptr;
        bool nonzero=false;for(int j=0;j<d;++j)nonzero|=g[j]!=0||(Quadratic&&h[j]!=0);
        if(!nonzero)continue;
        ++stats[10];stats[1]+=t;
        std::fill(latest.begin(),latest.begin()+L+1,-1);
        std::fill(flat_head.begin(),flat_head.begin()+t,-1);
        prefix[0]=0;
        int gram_last=-1,gram_previous=-1;int gram_neighbors[64];
        std::fill(gram_neighbors,gram_neighbors+d,-1);
        std::vector<std::pair<int,int>> gram_repairs;
        for(int r=0;r<t;++r) {
            f[r]=lcs(t,r);++stats[2];
            if(subtract_onegram) {
                const uint64_t diff=q[t]^k[r];
                if(!diff){gram_previous=gram_last;gram_last=r;}
                else if(!(diff&(diff-1))) {const int bit=row_bit(diff);if(bit<d){
                    gram_neighbors[bit]=r;gram_repairs.emplace_back(r,bit);}}
            }
            latest[f[r]]=r;
            prefix[r+1]=std::max(prefix[r],row_route(f[r],r));
            const int deadline=r-f[r];
            if(deadline>=0){flat_next[r]=flat_head[deadline];flat_head[deadline]=r;}
        }
        for(int h0=L-1;h0>=1;--h0)latest[h0]=std::max(latest[h0],latest[h0+1]);
        // K deletion is a prefix winner versus a capped suffix winner.
        // A suffix candidate is growing until s reaches r-F[r], then flat.
        // Flat candidates only enter. The maximal growing r moves only left.
        RowRoute flat=0;int growing=t-1;
        for(int s=t-1;s>=first&&L;--s) {
            for(int r=flat_head[s];r>=0;r=flat_next[r])flat=std::max(flat,row_route(f[r],r));
            while(growing>=s&&growing-f[growing]>=s)--growing;
            if(s<=r0)kdelete[s]=std::max({prefix[s],flat,
                growing>=s?row_route(growing-s,growing):RowRoute(0)});
        }
        auto qdel=[&](int owner)->RowRoute {
            const int cap=t-owner;
            return cap>=L?baseline:(cap>0?row_route(cap,latest[cap]):0);
        };
        auto kdel=[&](int owner)->RowRoute {
            return owner<first||owner>r0?baseline:kdelete[owner];
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
        if(!repair_only)for(int owner=t-L+1;owner<=t&&L;++owner) {
            const double value=score(row_end(qdel(owner)));
            for(int j=0;j<d;++j)cq[size_t(owner)*d+j]+=value;
            ++stats[7];
        }
        if(!repair_only)for(int owner=first;owner<=r0&&L;++owner) {
            const double value=score(row_end(kdelete[owner]));
            for(int j=0;j<d;++j)ck[size_t(owner)*d+j]+=value;
            ++stats[7];
        }
        qtouch.clear();ktouch.clear();
        for(int r=0;r<t;++r) {
            const int a=f[r],u=t-a,s=r-a;
            if(s<0)continue;
            const uint64_t difference=q[u]^k[s];
            if(!difference||(difference&(difference-1)))continue;
            const int bit=row_bit(difference);if(bit>=d)continue;
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
        }
        for(size_t key:qtouch) {cq[key]+=score(row_end(qbest[key]))-score(row_end(qdel(int(key/d))));qbest[key]=0;}
        for(size_t key:ktouch) {ck[key]+=score(row_end(kbest[key]))-score(row_end(kdel(int(key/d))));kbest[key]=0;}
        stats[4]+=qtouch.size();stats[5]+=ktouch.size();
        if(subtract_onegram) {
            const double original=score(gram_last);
            const double q_reference=repair_only?score(-1):original;
            for(int j=0;j<d;++j)cq[size_t(t)*d+j]-=score(gram_neighbors[j])-q_reference;
            if(gram_last>=0&&!repair_only) {
                const double deletion=score(gram_previous)-original;
                for(int j=0;j<d;++j)ck[size_t(gram_last)*d+j]-=deletion;
            }
            for(const auto&item:gram_repairs)if(item.first>gram_last)
                ck[size_t(item.first)*d+item.second]-=score(item.first)-original;
        }
        if(compute_value&&!exact_value&&r0>=0)for(int j=0;j<d;++j) {
            const size_t x=size_t(r0+1)*d+j;
            const double delta=-2*(2*int(v[x])-1)*scale[j];
            cv[x]+=weight*(Quadratic?g[j]*delta+(.5*h[j]*delta)*delta:g[j]*delta);
        }
    }
    if(compute_value&&exact_value)for(int t=0;t<n;++t)if(routes[t]>=0)for(int j=0;j<d;++j) {
        const size_t x=size_t(routes[t]+1)*d+j,y=size_t(t)*d+j;
        const double delta=-2*(2*int(v[x])-1)*scale[j];
        cv[x]+=Quadratic?gradient[y]*delta+(.5*diagonal[y]*delta)*delta:gradient[y]*delta;
    }
    stats[9]=index?index->materialized_index_flags():0;
    return 0;
  }catch(const std::bad_alloc&){return 2;}catch(const std::exception&){return 3;}
}
} // namespace rosa_random
