#pragma once
// One structural engine for all packed alphabets: deletion, short repair,
// anchored long repair and batched numerical correlations. No family dispatch.
#include "index.h"
#include "delete.h"
#include "repair.h"
#include "short_k.h"
#include "near_k.h"
#include "fingerprint_k.h"
#include "q_frontier.h"
#include "correlation.h"
#include "../training.h"
#include "../forward.h"
#include <memory>
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
        std::vector<double> reference(bits),extra(n),raw(h?size_t(n)*bits:0);
        std::vector<int> columns;
        std::vector<double> magnitude;
        // V[0] is never returned by a causal endpoint. Center on the first
        // reachable payload and keep that unused slot constant as well.
        const int reference_row=n>1?1:0;
        for(int j=0;j<bits;++j)reference[j]=(2*int(v[size_t(reference_row)*bits+j])-1)*e1[j];
        std::vector<double> maximum(bits,0.),span(bits,0.);
        for(int t=0;t<n;++t) {
            for(int j=0;j<bits;++j) {
                const size_t at=size_t(t)*bits+j;
                double value=g[at];const double delta=e0[j]-reference[j];
                if(h) {
                    const int r=baseline[t].end;
                    const double old=r<0?e0[j]:(2*int(v[size_t(r+1)*bits+j])-1)*e1[j];
                    value-=h[at]*old;raw[at]=value;
                    extra[t]+=g[at]*delta+.5*h[at]*delta*((e0[j]-old)+(reference[j]-old));
                } else extra[t]+=g[at]*delta;
                maximum[j]=std::max(maximum[j],std::abs(value));
                if(t)span[j]=std::max(span[j],std::abs((2*int(v[at])-1)*e1[j]-reference[j]));
            }
        }
        for(int j=0;j<bits;++j) {
            if(maximum[j]!=0 && span[j]!=0) {
                columns.push_back(j);
                const long double bound=static_cast<long double>(maximum[j])*span[j];
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
        for(int t=0;t<n;++t) {
            for(int c=0;c<int(selected.size());++c) {
                const int at=selected[c],j=columns[at];
                weights_[size_t(t)*d_+c]=(h?raw[size_t(t)*bits+j]:g[size_t(t)*bits+j])*inclusion[at];
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
    size_t workspace_bytes()const {
        return (weights_.capacity()+payload_.capacity()+empty_.capacity())*sizeof(double);
    }
};

inline void deletion(Index&index,const Scores&score,int bits,
                     std::vector<double>&cq,std::vector<double>&ck,Stats&stats) {
    const int n=index.n,A=std::max(1,int(std::sqrt(double(n))));
    const auto qd=q_deletion(index,score,A,stats);
    const auto kd=k_deletion(index,score,A,stats);
    for(int t=0;t<n;++t)for(int j=0;j<bits;++j) {
        cq[size_t(t)*bits+j]=qd[t];ck[size_t(t)*bits+j]=kd[t];
    }
    stats.peak_workspace_bytes+=(qd.capacity()+kd.capacity())*sizeof(double);
}

// The length-one deletion is completely described by two neighboring K
// occurrences. Isolate those rows before constructing general AP profiles
// when only a small minority needs longer contexts. This is the length-one
// leaf of the same deletion formula, with no extra matching query.
inline void sparse_deletion(Index&index,const Scores&score,int bits,
    const rosa_random::OnegramPlan&plan,const std::vector<int32_t>&active,
    std::vector<double>&cq,std::vector<double>&ck,Stats&stats) {
    std::vector<int> long_rows;
    for(int t:active)if(index.baseline()[t].length>1)long_rows.push_back(t);
    if(long_rows.size()*4>active.size()) {
        deletion(index,score,bits,cq,ck,stats);return;
    }
    const int n=index.n;
    const auto qd=q_deletion(index,score,std::max(1,int(std::sqrt(double(n)))),stats,&long_rows);
    const auto kd=k_deletion(index,score,std::max(1,int(std::sqrt(double(long_rows.size())))),stats,&long_rows);
    for(int t=0;t<n;++t)for(int j=0;j<bits;++j) {
        cq[size_t(t)*bits+j]=qd[t];ck[size_t(t)*bits+j]=kd[t];
    }
    std::vector<int> previous(n,-1);
    for(int id=0;id<plan.nk;++id)for(int at=plan.ko[id]+1;at<plan.ko[id+1];++at)
        previous[plan.krows[at]]=plan.krows[at-1];
    for(int t:active) {
        const Route original=index.baseline()[t];
        if(original.length!=1)continue;
        const double*w=score.row(t),*old=score.value(original.end);
        const double*before=score.value(previous[original.end]);
        double qvalue=0,kvalue=0;
        for(int c=0;c<score.d;++c) {
            qvalue+=w[c]*(score.empty[c]-old[c]);
            kvalue+=w[c]*(before[c]-old[c]);
        }
        for(int j=0;j<bits;++j) {
            cq[size_t(t)*bits+j]+=qvalue;
            ck[size_t(original.end)*bits+j]+=kvalue;
        }
    }
    stats.peak_workspace_bytes=std::max(stats.peak_workspace_bytes,
        uint64_t(index.workspace_bytes()+(qd.capacity()+kd.capacity())*sizeof(double)
                 +(previous.capacity()+long_rows.capacity())*sizeof(int)));
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
                     std::vector<double>&cq,std::vector<double>&ck,Stats&stats,uint64_t seed=0) {
    const int n=index.n;
    const int deletion_width=std::max(1,int(std::sqrt(double(n))));
    if(!repair_supported(index,bits)) {
        deletion(index,score,bits,cq,ck,stats);
        return;
    }
    const double ratio=std::max(1.,double(n)/bits);
    const int threshold=std::max(1,int(std::cbrt(ratio)));
    const int anchor=std::max(1,int(std::pow(ratio,.75)/4));
    // Each side of a one-mismatch match is an ordinary causal exact
    // match somewhere in the stream. Their global maximum is a valid cap.
    int longest=0;for(const auto&r:index.baseline())longest=std::max(longest,r.length);
    const int cutoff=std::min({n,4*anchor,2*longest+1});
    // The same near-winner occurrence summary supplies K deletion and K
    // repair when its complete report fits the bounded work budget. Compare
    // that budget with the structural short catalogue before doing the sum.
    const uint64_t near_budget=std::min(uint64_t(4)*n*std::max(1,int(std::pow(double(n),.75))),
                                       uint64_t(8)*n*bits*cutoff);
    const uint64_t window_budget=uint64_t(n)*std::max(1,int(std::pow(double(n),.75)));
    const bool finished_k=fingerprint_k(index,score,bits,ck,stats,window_budget,true,seed)
        ||near_k(index,score,bits,ck,stats,near_budget,true);
    if(!finished_k) {
        const auto kd=k_deletion(index,score,deletion_width,stats);
        for(int t=0;t<n;++t)for(int j=0;j<bits;++j)ck[size_t(t)*bits+j]=kd[t];
    }
    std::vector<double> heavy(size_t(n)*bits,0.);
    uint64_t retained_peak=heavy.capacity()*sizeof(double);
    if(q_frontier(index,score,bits,heavy,stats,seed,true)) {
        // The same bounded barrier frontier now owns full Q deletion and
        // repair. Its accepted result needs no second indexed Q sweep.
        cq.swap(heavy);
    } else {
        std::vector<Segment> light;
        std::vector<BitRange> exist;
        q_repair(index,score,bits,threshold,heavy,light,exist,stats);
        std::vector<OwnerRange> ranges;ranges.reserve(exist.size());
        for(const auto&r:exist)ranges.push_back({r.owner,r.lo,r.hi});
        const int block=ranges.empty()?1:std::min(n,std::max(1,int(std::ceil(double(n)/std::sqrt(double(ranges.size()))))));
        std::vector<double> qd;
        const auto negative=q_deletion_and_ranges(index,score,ranges,block,stats,qd);
        retained_peak+=light.capacity()*sizeof(Segment)+exist.capacity()*sizeof(BitRange)
            +ranges.capacity()*sizeof(OwnerRange)+(qd.capacity()+negative.capacity())*sizeof(double);
        for(int t=0;t<n;++t)for(int j=0;j<bits;++j)cq[size_t(t)*bits+j]=qd[t];
        for(size_t i=0;i<cq.size();++i)cq[i]+=heavy[i];
        correlations(score,bits,light,cq,stats);
        for(size_t i=0;i<exist.size();++i)cq[size_t(exist[i].owner)*bits+exist[i].bit]-=negative[i];
    }
    if(finished_k){stats.peak_workspace_bytes+=retained_peak;return;}
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
        retained_peak=std::max(retained_peak,uint64_t(heavy.capacity()*sizeof(double)
            +segments.capacity()*sizeof(Segment)+united.capacity()*sizeof(BitRange)
            +ranges.capacity()*sizeof(OwnerRange)+negative.capacity()*sizeof(double)));
        for(size_t i=0;i<united.size();++i)ck[size_t(united[i].owner)*bits+united[i].bit]-=negative[i];
    }
    stats.peak_workspace_bytes+=retained_peak;
}

inline int external_lcs(const void*context,int t,int r) {
    return static_cast<const Index*>(context)->lcs(t,r);
}

// Count only endpoints a complete residual-row evaluator can visit. Stop at
// the algebraic work budget before any expensive row calculation. Choosing a
// sparse evaluation of the same repair correction is independent of inputs'
// labels and keeps its cost within the engine's O~(n^(7/4) (d + 1)) bound.
inline bool sparse_residual(const rosa_random::OnegramPlan&plan,
                            const std::vector<int32_t>&rows) {
    const uint64_t budget=uint64_t(plan.n)*std::max(1,int(std::pow(double(plan.n),.75)));
    uint64_t work=0;
    for(int t:rows) {
        const int id=plan.qclass[t],same=plan.qsame[id];
        auto count=[&](int group) {
            const auto first=plan.krows.begin()+plan.ko[group];
            return uint64_t(std::lower_bound(first,plan.krows.begin()+plan.ko[group+1],t)-first);
        };
        if(same>=0)work+=count(same);
        for(const auto&edge:plan.qneighbors[id])work+=count(edge.other);
        if(work>budget)return false;
    }
    return true;
}

// First settle the one-symbol control. Only rows that can have a longer
// original or edited route require a correction. Sparse correction rows use
// the complete endpoint join; dense correction uses the shared interval
// construction. Both are evaluations of the same delete/repair identity.
inline int run(int n,int bits,const uint64_t*q,const uint64_t*k,const uint8_t*v,
    const double*g,const double*h,const double*e0,const double*e1,bool randomized,
    uint64_t seed,const RosaApproximationOptions*approximation,
    std::vector<double>&cq,std::vector<double>&ck,int32_t*lengths,int32_t*routes,
    Stats&stats,uint64_t*approx_stats,const int32_t*known_lengths=nullptr,
    const int32_t*known_routes=nullptr) {
    const size_t nd=size_t(n)*bits;
    cq.assign(nd,0.);ck.assign(nd,0.);
    if(!n)return 0;
    if(bool(known_lengths)!=bool(known_routes))return 1;
    if(known_lengths) {
        for(int t=0;t<n;++t) {
            const int len=known_lengths[t],end=known_routes[t];
            if(len<0||len>t||end < -1||end>=t||(len==0)!=(end<0)||len>end+1)return 1;
            lengths[t]=len;routes[t]=end;
        }
    } else {
        rosa_exact::SamForward forward(q,k,n,seed);
        forward.run(lengths,routes);
    }
    std::vector<Route>base(n);
    std::vector<int32_t>active;
    bool short_deletion=true;
    for(int t=0;t<n;++t) {
        base[t]={lengths[t],routes[t]};
        bool nonzero=false;
        for(int j=0;j<bits;++j){const size_t at=size_t(t)*bits+j;nonzero|=g[at]!=0||(h&&h[at]!=0);}
        if(nonzero){active.push_back(t);short_deletion&=lengths[t]<=1;}
    }
    if(active.empty()){if(approx_stats)approx_stats[27]=1;return 0;}
    const uint64_t random_seed=seed^(approximation?approximation->seed:0);
    const int channels=approximation?approximation->sample_channels:0;
    const rosa_random::OnegramPlan plan(q,k,n,bits,seed);
    std::vector<int32_t>repair_active;
    for(int t:active)if(plan.birth_prefix[t+1]!=plan.birth_prefix[t-lengths[t]])repair_active.push_back(t);
    uint64_t certificate[8]={};
    const auto population=rosa_random::uncertified_rows(q,k,n,bits,repair_active.data(),
        int(repair_active.size()),lengths,random_seed,certificate,plan.kclass.data(),plan.nk,&plan.kmap);
    const bool sparse=sparse_residual(plan,population);
    // A positive channel budget is a numerical approximation opportunity,
    // not an instruction to add noise to an already cheap exact correction.
    const int row_cap=std::max(1,int(std::ceil(std::pow(double(n),.75))));
    const bool sample_rows=approximation&&!channels
        &&approximation->sample_rows<int(population.size())&&approximation->sample_rows<=row_cap;
    const bool settled=short_deletion&&population.empty();
    const bool dense=!settled&&(channels>0||(!sample_rows&&!sparse));
    std::unique_ptr<Index> index;
    std::unique_ptr<Features> features;
    if(!short_deletion||!population.empty()||dense) {
        index.reset(new Index(q,k,n,randomized,seed,&base));
        if(!short_deletion||dense)features.reset(new Features(n,bits,v,g,h,e0,e1,base,
            dense?channels:0,random_seed,stats));
    }
    int result=0;
    if(dense) {
        const Scores score=features->scores();
        if(std::any_of(score.weights,score.weights+size_t(n)*score.d,[](double x){return x!=0.;}))
            complete(*index,score,bits,cq,ck,stats,seed);
        if(approx_stats) {
            approx_stats[27]=!features->sampled;
            approx_stats[33]=features->sampled;
            approx_stats[37]=stats.sampled_channels;
        }
    } else {
        if(!short_deletion)sparse_deletion(*index,features->scores(),bits,plan,active,cq,ck,stats);
        uint64_t control[16]={},row_work[16]={};
        size_t control_workspace=0;
        if(h)plan.apply<true>(v,g,h,e0,e1,routes,cq.data(),ck.data(),control,short_deletion,nullptr,&control_workspace,short_deletion);
        else plan.apply<false>(v,g,nullptr,e0,e1,routes,cq.data(),ck.data(),control,short_deletion,nullptr,&control_workspace);
        stats.peak_workspace_bytes=std::max(stats.peak_workspace_bytes,uint64_t(control_workspace
            +(index?index->workspace_bytes():0)+(features?features->workspace_bytes():0)));
        std::vector<int32_t>selected;
        std::vector<double>weights;
        if(sample_rows)rosa_random::stratified_rows(population,approximation->sample_rows,g,bits,
            approximation->importance,random_seed,selected,weights);
        else selected=population;
        size_t row_workspace=0;
        if(!selected.empty()) {
            if(h)result=rosa_random::indexed_row_repairs<true>(plan,v,g,h,e0,e1,
                selected.data(),weights.empty()?nullptr:weights.data(),int(selected.size()),lengths,routes,
                cq.data(),ck.data(),row_work,true,external_lcs,index.get(),&row_workspace);
            else result=rosa_random::indexed_row_repairs<false>(plan,v,g,nullptr,e0,e1,
                selected.data(),weights.empty()?nullptr:weights.data(),int(selected.size()),lengths,routes,
                cq.data(),ck.data(),row_work,true,external_lcs,index.get(),&row_workspace);
        }
        stats.peak_workspace_bytes=std::max(stats.peak_workspace_bytes,uint64_t(row_workspace
            +selected.capacity()*sizeof(int32_t)+weights.capacity()*sizeof(double)
            +(index?index->workspace_bytes():0)+(features?features->workspace_bytes():0)));
        stats.suffix_queries+=row_work[2];
        if(approx_stats) {
            for(int i=0;i<16;++i)approx_stats[i]=row_work[i]+control[i];
            approx_stats[25]=selected.size();approx_stats[27]=!sample_rows;
            approx_stats[31]=!short_deletion;
        }
    }
    if(approx_stats) {
        std::copy(certificate,certificate+8,approx_stats+16);
        approx_stats[16]=active.size();approx_stats[24]=population.size();
        approx_stats[26]=approximation?approximation->sample_rows:0;
        approx_stats[29]=active.size()-population.size();
        approx_stats[30]=sample_rows?approximation->importance:0;
    }
    if(index) {
        stats.hash_probes=index->cache_hits+index->cache_misses;
        stats.hash_verifications=index->cache_hits;
        // Component diagnostics use mixed local-buffer scopes. Add the
        // retained owners conservatively instead of taking max(), which
        // would lose their coexistence with the largest temporary buffer.
        // Some shared buffers may be counted twice; this is a capacity
        // estimate, not allocator accounting or measured peak RSS.
        stats.peak_workspace_bytes+=index->workspace_bytes()
            +(features?features->workspace_bytes():0);
    }
    stats.peak_workspace_bytes+=plan.workspace_bytes()+base.capacity()*sizeof(Route)
        +(active.capacity()+repair_active.capacity()+population.capacity())*sizeof(int32_t);
    std::fill(cq.begin(),cq.begin()+bits,0.);
    std::fill(ck.begin()+size_t(n-1)*bits,ck.end(),0.);
    return result;
}
} // namespace rosa_subq
