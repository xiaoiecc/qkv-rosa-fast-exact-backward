#pragma once
// One structural engine for all packed alphabets: deletion, short repair,
// anchored long repair and batched numerical correlations. No family dispatch.
#include "index.h"
#include "delete.h"
#include "repair.h"
#include "short_k.h"
#include "correlation.h"
#include "../training.h"
#include "../approx_indexed_row.h"
#include "../approx_row_certificate.h"
#include "../approx_sampling.h"
#include "../approx_channel_pivotal.h"
#include <cmath>
#include <limits>
#include <numeric>

namespace rosa_subq {

// A row-dependent constant can be removed from every route score. Center on
// the first signed payload, then represent the empty route by one indicator.
// This avoids subtracting large squared embedding norms in the trust lift.
class Features {
    int n_=0,d_=0;
    std::vector<double> weights_,payload_,empty_;
public:
    bool sampled=false;
    Features(int n,int bits,const uint8_t*v,const double*g,const double*h,
             const double*e0,const double*e1,const std::vector<Route>&baseline,
             int channel_budget,uint64_t seed,Stats&stats):n_(n) {
        std::vector<double> reference(bits),extra(n),raw(size_t(n)*bits);
        std::vector<int> columns;
        std::vector<double> magnitude;
        // V[0] is never returned by a causal endpoint. Center on the first
        // reachable payload and keep that unused slot constant as well.
        const int reference_row=n>1?1:0;
        for(int j=0;j<bits;++j)reference[j]=(2*int(v[size_t(reference_row)*bits+j])-1)*e1[j];
        for(int j=0;j<bits;++j) {
            double maximum=0,span=0;
            for(int t=0;t<n;++t) {
                const size_t at=size_t(t)*bits+j;
                const int r=baseline[t].end;
                const double old=r<0?e0[j]:(2*int(v[size_t(r+1)*bits+j])-1)*e1[j];
                const double diagonal=h?h[at]:0.;
                const double value=g[at]-diagonal*old;
                raw[at]=value;maximum=std::max(maximum,std::abs(value));
                if(t)span=std::max(span,std::abs((2*int(v[at])-1)*e1[j]-reference[j]));
                const double delta=e0[j]-reference[j];
                extra[t]+=g[at]*delta+.5*diagonal*delta*((e0[j]-old)+(reference[j]-old));
            }
            if(maximum!=0 && span!=0) {
                columns.push_back(j);
                const long double bound=static_cast<long double>(maximum)*span;
                magnitude.push_back(double(std::min(bound,static_cast<long double>(std::numeric_limits<double>::max()))));
            }
        }
        std::vector<double> inclusion(columns.size(),1.);
        if(channel_budget>0 && channel_budget<int(columns.size())) {
            inclusion=rosa_random::pivotal_channel_weights(magnitude,channel_budget,seed);
            sampled=true;
        }
        const bool has_empty=std::any_of(extra.begin(),extra.end(),[](double x){return x!=0.;});
        std::vector<int> selected;
        for(int at=0;at<int(columns.size());++at)if(inclusion[at])selected.push_back(at);
        d_=std::max(1,int(selected.size())+int(has_empty));
        weights_.assign(size_t(n)*d_,0.);payload_.assign(size_t(n)*d_,0.);empty_.assign(d_,0.);
        for(int c=0;c<int(selected.size());++c) {
            const int at=selected[c],j=columns[at];
            for(int t=0;t<n;++t) {
                weights_[size_t(t)*d_+c]=raw[size_t(t)*bits+j]*inclusion[at];
                payload_[size_t(t)*d_+c]=t?(2*int(v[size_t(t)*bits+j])-1)*e1[j]-reference[j]:0.;
            }
        }
        if(has_empty) {
            const int c=d_-1;empty_[c]=1.;
            for(int t=0;t<n;++t)weights_[size_t(t)*d_+c]=extra[t];
        }
        for(double x:weights_)if(!std::isfinite(x))throw std::runtime_error("nonfinite lifted credit");
        for(double x:payload_)if(!std::isfinite(x))throw std::runtime_error("nonfinite centered payload");
        stats.sampled_channels=sampled?selected.size():0;
        stats.peak_workspace_bytes=std::max(stats.peak_workspace_bytes,
            uint64_t(sizeof(double)*(raw.size()+weights_.size()+payload_.size()+extra.size())));
    }
    Scores scores()const {return {n_,d_,weights_.data(),payload_.data(),empty_.data()};}
};

inline void deletion(Index&index,const Scores&score,int bits,
                     std::vector<double>&cq,std::vector<double>&ck,Stats&stats) {
    const int n=index.n,A=std::max(1,int(std::sqrt(double(n))));
    const auto qd=q_deletion(index,score,A,stats);
    const auto kd=k_deletion(index,score,A,stats);
    for(int t=0;t<n;++t)for(int j=0;j<bits;++j) {
        cq[size_t(t)*bits+j]=qd[t];ck[size_t(t)*bits+j]=kd[t];
    }
}

inline void correlations(const Scores&score,int bits,const std::vector<Segment>&segments,
                         std::vector<double>&credit,Stats&stats) {
    if(segments.empty())return;
    std::vector<CorrelationQuery> query;query.reserve(segments.size());
    for(const auto&s:segments)query.push_back({s.lo,s.hi,s.beta+1});
    CorrelationStats cs;
    const auto value=BlockCorrelations::solve(score.n,score.d,score.weights,score.payload,
        query,correlation_block(score.n,score.d,query),cs);
    for(size_t i=0;i<segments.size();++i) {
        const auto&s=segments[i];credit[size_t(s.owner)*bits+s.bit]+=value[i];
    }
    stats.correlation_queries+=query.size();stats.correlation_direct+=cs.tail_products;
    stats.fft_transforms+=cs.transforms;
    stats.peak_workspace_bytes=std::max(stats.peak_workspace_bytes,cs.workspace_bytes);
}

inline bool repair_supported(const Index&index,int bits) {
    // A repair must contain a causal one-bit center. Deduplicate symbols
    // before checking this necessary condition; full uint64 keys are compared.
    std::vector<std::pair<uint64_t,int>> query,key;
    query.reserve(index.n);key.reserve(index.n);
    for(int t=0;t<index.n;++t){query.emplace_back(index.q[t],t);key.emplace_back(index.k[t],t);}
    std::sort(query.begin(),query.end());std::sort(key.begin(),key.end());
    for(size_t at=0;at<query.size();) {
        size_t after=at+1;while(after<query.size()&&query[after].first==query[at].first)++after;
        const int latest=query[after-1].second;
        for(int j=0;j<bits;++j) {
            const uint64_t symbol=query[at].first^(uint64_t(1)<<j);
            const auto found=std::lower_bound(key.begin(),key.end(),std::make_pair(symbol,-1));
            if(found!=key.end()&&found->first==symbol&&found->second<latest)return true;
        }
        at=after;
    }
    return false;
}

inline void complete(Index&index,const Scores&score,int bits,
                     std::vector<double>&cq,std::vector<double>&ck,Stats&stats) {
    const int n=index.n;
    deletion(index,score,bits,cq,ck,stats);
    if(!repair_supported(index,bits))return;
    const double ratio=std::max(1.,double(n)/bits);
    const int threshold=std::max(1,int(std::cbrt(ratio)));
    const int anchor=std::max(1,int(std::pow(ratio,.75)/4));
    // Each side of a one-mismatch match is an ordinary causal exact
    // match somewhere in the stream. Their global maximum is a valid cap.
    int longest=0;for(const auto&r:index.baseline())longest=std::max(longest,r.length);
    const int cutoff=std::min({n,4*anchor,2*longest+1});
    std::vector<double> heavy(size_t(n)*bits,0.);
    std::vector<Segment> light;
    std::vector<BitRange> exist;
    q_repair(index,score,bits,threshold,heavy,light,exist,stats);
    for(size_t i=0;i<cq.size();++i)cq[i]+=heavy[i];
    correlations(score,bits,light,cq,stats);
    if(!exist.empty()) {
        std::vector<OwnerRange> ranges;ranges.reserve(exist.size());
        for(const auto&r:exist)ranges.push_back({r.owner,r.lo,r.hi});
        const int block=std::min(n,std::max(1,int(std::ceil(double(n)/std::sqrt(double(ranges.size()))))));
        const auto negative=q_delete_ranges(index,score,ranges,block,stats);
        for(size_t i=0;i<exist.size();++i)cq[size_t(exist[i].owner)*bits+exist[i].bit]-=negative[i];
    }
    short_k(index,score,bits,cutoff,ck,stats);
    auto segments=4*anchor>=2*longest+1?std::vector<Segment>{}:long_k(index,anchor,stats);
    segments.erase(std::remove_if(segments.begin(),segments.end(),
        [bits](const Segment&s){return s.bit>=bits;}),segments.end());
    correlations(score,bits,segments,ck,stats);
    if(!segments.empty()) {
        // The envelope is disjoint in time. Adjacent payload runs share one
        // negative deletion projection, even when their payload shifts differ.
        std::sort(segments.begin(),segments.end(),[](const Segment&a,const Segment&b){
            if(a.owner!=b.owner)return a.owner<b.owner;
            if(a.bit!=b.bit)return a.bit<b.bit;
            return a.lo<b.lo;
        });
        std::vector<BitRange> united;
        for(const auto&s:segments) {
            if(!united.empty()&&united.back().owner==s.owner&&united.back().bit==s.bit&&united.back().hi+1>=s.lo)
                united.back().hi=std::max(united.back().hi,s.hi);
            else united.push_back({s.owner,s.bit,s.lo,s.hi});
        }
        std::vector<OwnerRange> ranges;ranges.reserve(united.size());
        for(const auto&r:united)ranges.push_back({r.owner,r.lo,r.hi});
        // Balance against the ACTUAL number of owner/time projections.
        // At the worst-case M=O(n^2/B) this is B^(2/3); sparse envelopes
        // use larger blocks instead of paying that pessimistic global bound.
        const double balanced=std::pow(double(n)*double(n)/ranges.size(),2./3);
        const int block=std::max(1,int(std::ceil(std::min(double(n),balanced))));
        const auto negative=k_delete_ranges(index,score,ranges,block,stats);
        for(size_t i=0;i<united.size();++i)ck[size_t(united[i].owner)*bits+united[i].bit]-=negative[i];
    }
}

inline int external_lcs(const void*context,int t,int r) {
    return static_cast<const Index*>(context)->lcs(t,r);
}

// The aggressive estimator keeps every deletion and one-symbol repair. Only
// the remaining complete row correction is sampled with its exact HT weight.
inline int run(int n,int bits,const uint64_t*q,const uint64_t*k,const uint8_t*v,
    const double*g,const double*h,const double*e0,const double*e1,bool randomized,
    uint64_t seed,const RosaApproximationOptions*approximation,
    std::vector<double>&cq,std::vector<double>&ck,int32_t*lengths,int32_t*routes,
    Stats&stats,uint64_t*approx_stats) {
    if(!n)return 0;
    Index index(q,k,n,randomized,seed);
    const auto&base=index.baseline();
    for(int t=0;t<n;++t){lengths[t]=base[t].length;routes[t]=base[t].end;}
    const uint64_t random_seed=seed^(approximation?approximation->seed:0);
    const int channels=approximation?approximation->sample_channels:0;
    Features features(n,bits,v,g,h,e0,e1,base,channels,random_seed,stats);
    const Scores score=features.scores();
    if(std::none_of(score.weights,score.weights+size_t(n)*score.d,[](double x){return x!=0.;})) {
        if(approx_stats)approx_stats[27]=1;
        stats.peak_workspace_bytes+=index.workspace_bytes();
        return 0;
    }
    int result=0;
    if(!approximation || channels) {
        complete(index,score,bits,cq,ck,stats);
        if(approximation && approx_stats) {
            approx_stats[26]=approximation->sample_rows;
            approx_stats[27]=!features.sampled;
            approx_stats[33]=features.sampled;
            approx_stats[37]=stats.sampled_channels;
        }
    } else {
        const rosa_random::OnegramPlan plan(q,k,n,bits,seed);
        std::vector<int32_t> active,repair_active;
        for(int t=0;t<n;++t) {
            bool nonzero=false;for(int j=0;j<bits;++j)nonzero|=g[size_t(t)*bits+j]!=0;
            if(!nonzero)continue;
            active.push_back(t);
            if(plan.birth_prefix[t+1]!=plan.birth_prefix[t-lengths[t]])repair_active.push_back(t);
        }
        const auto population=rosa_random::uncertified_rows(q,k,n,bits,repair_active.data(),
            int(repair_active.size()),lengths,random_seed,approx_stats+16);
        // Large budgets use the complete structural algorithm. This avoids
        // turning the approximate entry point back into an all-rows scan.
        const int cap=std::max(1,int(std::ceil(std::pow(double(n),.75))));
        if(!population.empty() && (approximation->sample_rows>=int(population.size())||approximation->sample_rows>cap)) {
            complete(index,score,bits,cq,ck,stats);
            approx_stats[25]=population.size();approx_stats[27]=1;
        } else {
            deletion(index,score,bits,cq,ck,stats);approx_stats[31]=1;
            uint64_t control[16]={};
            if(h)plan.apply<true>(v,g,h,e0,e1,routes,cq.data(),ck.data(),control,false);
            else plan.apply<false>(v,g,nullptr,e0,e1,routes,cq.data(),ck.data(),control,false);
            std::vector<int32_t> selected;std::vector<double> weights;
            rosa_random::stratified_rows(population,approximation->sample_rows,g,bits,
                approximation->importance,random_seed,selected,weights);
            if(h)result=rosa_random::indexed_row_repairs<true>(plan,v,g,h,e0,e1,
                selected.data(),weights.data(),int(selected.size()),lengths,routes,
                cq.data(),ck.data(),approx_stats,true,external_lcs,&index);
            else result=rosa_random::indexed_row_repairs<false>(plan,v,g,nullptr,e0,e1,
                selected.data(),weights.data(),int(selected.size()),lengths,routes,
                cq.data(),ck.data(),approx_stats,true,external_lcs,&index);
            for(int i=0;i<16;++i)approx_stats[i]+=control[i];
            approx_stats[25]=selected.size();approx_stats[27]=selected.size()==population.size();
        }
        approx_stats[16]=active.size();approx_stats[24]=population.size();
        approx_stats[26]=approximation->sample_rows;
        approx_stats[29]=active.size()-population.size();approx_stats[30]=approximation->importance;
    }
    stats.hash_probes=index.cache_hits+index.cache_misses;
    stats.hash_verifications=index.cache_hits;
    stats.peak_workspace_bytes+=index.workspace_bytes();
    // Strict causality makes these coordinates identically zero. Interval
    // differences may otherwise leave a floating cancellation residual.
    std::fill(cq.begin(),cq.begin()+bits,0.);
    std::fill(ck.begin()+size_t(n-1)*bits,ck.end(),0.);
    return result;
}
} // namespace rosa_subq
