#pragma once
// Complete credit uses one interval reduction. Explicit row or channel
// sampling shares its baseline, numerical projection and deletion machinery.
#include "index.h"
#include "delete.h"
#include "interval_credit.h"
#include "../training.h"
#include "../causal_windows.h"
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

inline int external_lcs(const void*context,int t,int r) {
    return static_cast<const Index*>(context)->lcs(t,r);
}

// Count only endpoints a complete residual-row evaluator can visit. Stop at
// the algebraic work budget before any expensive row calculation. Choosing a
// sparse evaluation of the same repair correction is independent of inputs'
// labels and keeps its cost within the engine's O~(n^(5/3) (d + 1)) bound.
inline bool sparse_residual(const rosa_random::OnegramPlan&plan,
                            const std::vector<int32_t>&rows) {
    const uint64_t budget=uint64_t(plan.n)*std::max(1,int(std::pow(double(plan.n),2./3)));
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

// A training caller can consume credits on the retained coordinates directly.
// Low-level callers still receive the full dense arrays.
struct CreditDomain {
    bool compact=false;
    std::vector<int> row;
    int position(int t)const{return compact?(row.empty()?-1:row[t]):t;}
};

inline void exact_intervals(const uint64_t* q,const uint64_t* k,int n,int b,
        const std::vector<Route>& base,const uint8_t* v,const double* g,const double* h,
        const double* diagonal,const double* e0,const double* e1,
        std::vector<double>& cq,std::vector<double>& ck,Stats& stats,CreditDomain* domain) {
    auto project=[&](const uint64_t* query,const uint64_t* key,const std::vector<Route>& routes,
                     const uint8_t* value,const double* adjoint,const double* metric,
                     std::vector<double>& qout,std::vector<double>& kout) {
        int size=int(routes.size());uint64_t largest=0;
        for(int i=0;i<size;++i)largest=std::max({largest,query[i],key[i]});
        interval_credit::EditAlphabet symbols(query,key,size,b,largest);
        if(symbols.births || std::any_of(routes.begin(),routes.end(),[](Route r){return r.length>0;})) {
            Features features(size,b,value,adjoint,metric,e0,e1,routes,0,0,stats);
            interval_credit::evaluate(symbols,routes,features.scores(),qout,kout,stats,n);
            stats.peak_workspace_bytes+=features.workspace_bytes();
        }
        stats.peak_workspace_bytes+=symbols.workspace_bytes();
    };
    auto keep=interval_credit::live_coordinates(q,k,n,b,base,stats);
    int m=int(keep.size());
    if(m==1 && base[keep.front()].length==0) {
        if(domain)domain->compact=true;
        return;
    }
    if(keep.empty() || m>=n-n/4) {
        std::vector<double> expanded;
        if(diagonal) {
            expanded.resize(size_t(n)*b);
            for(int t=0;t<n;++t)std::fill_n(expanded.data()+size_t(t)*b,b,diagonal[t]);
            h=expanded.data();
        }
        if(domain){cq.assign(size_t(n)*b,0.);ck.assign(cq.size(),0.);}
        project(q,k,base,v,g,h,cq,ck);
        stats.peak_workspace_bytes+=expanded.capacity()*sizeof(double)+keep.capacity()*sizeof(int);
        return;
    }
    std::vector<int> mapped(n,-1);
    std::vector<uint64_t> query(m),key(m);
    std::vector<Route> routes(m);
    const size_t cells=size_t(m)*b;
    std::vector<uint8_t> value(cells);
    std::vector<double> adjoint(cells),metric((h || diagonal)?cells:0);
    for(int i=0;i<m;++i) {
        int t=keep[i];mapped[t]=i;
        query[i]=q[t];key[i]=k[t];
        routes[i]=base[t];
        if(routes[i].end>=0)routes[i].end=mapped[routes[i].end];
        size_t row=size_t(i)*b,source=size_t(t)*b;
        std::copy_n(g+source,b,adjoint.data()+row);
        if(h)std::copy_n(h+source,b,metric.data()+row);
        if(diagonal)std::fill_n(metric.data()+row,b,diagonal[t]);
        // An endpoint returns its ORIGINAL successor's payload.
        if(i+1<m)std::copy_n(v+source+b,b,value.data()+row+b);
    }
    std::vector<double> qout(cells),kout(cells);
    project(query.data(),key.data(),routes,value.data(),adjoint.data(),metric.empty()?nullptr:metric.data(),qout,kout);
    if(domain) {
        domain->compact=true;domain->row.swap(mapped);cq.swap(qout);ck.swap(kout);
    } else for(int i=0;i<m;++i) {
        size_t dst=size_t(keep[i])*b,src=size_t(i)*b;
        std::copy_n(qout.data()+src,b,cq.data()+dst);std::copy_n(kout.data()+src,b,ck.data()+dst);
    }
    stats.peak_workspace_bytes+=cells*(sizeof(uint8_t)+3*sizeof(double))
        +metric.capacity()*sizeof(double)+size_t(m)*(2*sizeof(uint64_t)+sizeof(Route))
        +(keep.capacity()+size_t(n))*sizeof(int);
}

inline int run(int n,int bits,const uint64_t*q,const uint64_t*k,const uint8_t*v,
    const double*g,const double*h,const double*e0,const double*e1,bool randomized,
    uint64_t seed,const RosaApproximationOptions*approximation,
    std::vector<double>&cq,std::vector<double>&ck,int32_t*lengths,int32_t*routes,
    Stats&stats,uint64_t*approx_stats,const int32_t*known_lengths=nullptr,
    const int32_t*known_routes=nullptr,CreditDomain* domain=nullptr,const double* diagonal=nullptr) {
    const size_t nd=size_t(n)*bits;
    if(!domain || approximation){cq.assign(nd,0.);ck.assign(nd,0.);}
    if(!n)return 0;
    if(bool(known_lengths)!=bool(known_routes))return 1;
    if(known_lengths) {
        for(int t=0;t<n;++t) {
            const int len=known_lengths[t],end=known_routes[t];
            if(len<0||len>t||end < -1||end>=t||(len==0)!=(end<0)||len>end+1)return 1;
            if(end>=0 && q[t]!=k[end])return 1;
            lengths[t]=len;routes[t]=end;
        }
    } else {
        rosa_exact::training_routes(q,k,n,bits,lengths,routes);
    }
    std::vector<Route>base(n);
    for(int t=0;t<n;++t)base[t]={lengths[t],routes[t]};
    if(!approximation) {
        exact_intervals(q,k,n,bits,base,v,g,h,diagonal,e0,e1,cq,ck,stats,domain);
        stats.peak_workspace_bytes+=base.capacity()*sizeof(Route);
        if(approx_stats)approx_stats[27]=1;
        return 0;
    }
    std::vector<int32_t>active;
    bool short_deletion=true;
    for(int t=0;t<n;++t) {
        bool nonzero=false;
        for(int j=0;j<bits;++j){const size_t at=size_t(t)*bits+j;nonzero|=g[at]!=0||(h&&h[at]!=0);}
        if(nonzero){active.push_back(t);short_deletion&=lengths[t]<=1;}
    }
    if(active.empty()){if(approx_stats)approx_stats[27]=1;return 0;}
    const uint64_t random_seed=seed^approximation->seed;
    const int channels=approximation->sample_channels;
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
        if(std::any_of(score.weights,score.weights+size_t(n)*score.d,[](double x){return x!=0.;})) {
            uint64_t largest=0;for(int i=0;i<n;++i)largest=std::max({largest,q[i],k[i]});
            interval_credit::EditAlphabet alphabet(q,k,n,bits,largest);
            interval_credit::evaluate(alphabet,base,score,cq,ck,stats);
            stats.peak_workspace_bytes+=alphabet.workspace_bytes();
        }
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
