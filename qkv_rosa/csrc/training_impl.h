#pragma once
// The batched API and low-level credit wrappers share this one engine.
#include "training.h"
#include "causal_windows.h"
#include "pair_candidates.h"
#include "engine/engine.h"
#include <cstring>
#include <random>

namespace rosa_exact {

inline void store_subquadratic_counters(const rosa_subq::Stats&work,uint64_t*out) {
    const uint64_t counters[ROSA_SUBQUADRATIC_COUNTERS]={
        work.suffix_queries,work.delete_pieces,work.delete_events,
        work.q_heavy_rows,work.q_light_sources,work.q_segments,
        work.k_anchor_probes,work.k_centers,work.k_segments,
        work.short_words,work.short_pieces,work.short_queries,work.short_exceptions,
        work.correlation_queries,work.correlation_direct,work.fft_transforms,
        work.range_blocks,work.range_fringes,work.peak_workspace_bytes,
        work.hash_probes,work.hash_verifications,work.sampled_channels};
    std::copy(counters,counters+ROSA_SUBQUADRATIC_COUNTERS,out);
}

inline bool training_shape(int b,int groups,int n,int d) {
    return b>=0 && groups>0 && n>=0 && n<(1<<28) && d>=1 && d<=64
        && uint64_t(n)*uint64_t(d)<=uint64_t(std::numeric_limits<int>::max())
        && uint64_t(b)*uint64_t(groups)<=uint64_t(std::numeric_limits<int>::max());
}
inline bool training_options(const RosaTrainingOptions* o) {
    return o && o->abi_version==ROSA_TRAINING_ABI_VERSION
        && o->struct_size==sizeof(RosaTrainingOptions)
        && (o->credit_order==1 || o->credit_order==2)
        && (o->randomized==0 || o->randomized==1)
        && (o->logit_mode==0 || o->logit_mode==1)
        && o->joint_pairs>=0 && o->joint_pairs<=std::numeric_limits<int>::max()/8
        && o->joint_rows>0
        && (o->active_rows==0 || o->active_rows==1)
        && o->repair_budget==0 && o->reserved==0 && std::isfinite(o->tau) && o->tau>0
        && std::isfinite(o->trust_diagonal) && o->trust_diagonal>=0
        && (o->credit_order==2 || (!o->joint_pairs && !o->trust_diagonal));
}
inline bool approximation_options(const RosaApproximationOptions*o) {
    return o && o->abi_version==ROSA_APPROXIMATION_ABI_VERSION
        && o->struct_size==sizeof(RosaApproximationOptions) && o->sample_rows>0
        && (o->importance==0 || o->importance==1)
        && o->sample_channels>=0 && o->sample_channels<=64;
}
inline bool finite_values(const double* p,size_t count) {
    if(count && !p)return false;
    // The exponent of a finite binary64 is strictly below the all-ones
    // exponent. A maximum reduction lets the compiler check adjacent lanes.
    uint64_t maximum=0;
    for(size_t i=0;i<count;++i) {
        uint64_t word;std::memcpy(&word,p+i,sizeof(word));
        maximum=std::max(maximum,word&UINT64_C(0x7fffffffffffffff));
    }
    return maximum<UINT64_C(0x7ff0000000000000);
}
inline double train_sigmoid(double z,double tau) {
    const double scaled=z/tau;
    const double exponential=std::exp(-std::abs(scaled));
    return scaled>=0 ? 1/(1+exponential) : exponential/(1+exponential);
}
inline double train_chain(double z,int bit,double credit,const RosaTrainingOptions&o) {
    const double oriented=(bit ? -1.0 : 1.0)*credit;
    if(o.logit_mode)return oriented;
    // Multiplying a zero credit by the nonnegative finite sigmoid factor
    // preserves this signed zero. Avoid exp for such coordinates. Extremely
    // tiny temperatures keep the original arithmetic, including overflow.
    if(credit==0. && o.tau>=std::numeric_limits<double>::min())return oriented;
    const double p=train_sigmoid(z,o.tau);
    return (p*(1-p)/o.tau)*oriented;
}
// The derivative factor depends on the exact logit and the stream's fixed
// temperature. One cached value per sign and Q/K/V stream avoids repeated exp calls
// without a dictionary, approximate exponential, or changed multiplication.
struct TrainFactorCache {
    uint64_t key[2]={};double factor[2]={};bool valid[2]={};
    double apply(double z,int bit,double credit,const RosaTrainingOptions&o) {
        const double oriented=(bit?-1.:1.)*credit;
        if(o.logit_mode)return oriented;
        if(credit==0. && o.tau>=std::numeric_limits<double>::min())return oriented;
        uint64_t bits;std::memcpy(&bits,&z,sizeof(bits));
        const size_t slot=size_t(bits>>63);
        if(!valid[slot]||bits!=key[slot]) {
            const double p=train_sigmoid(z,o.tau);factor[slot]=p*(1-p)/o.tau;
            key[slot]=bits;valid[slot]=true;
        }
        return factor[slot]*oriented;
    }
};

// Multiplying by these signs also preserves the orientation of zero credit.
inline const auto& byte_signs() {
    static const auto signs=[] {
        std::array<std::array<double,8>,256> result{};
        for(int byte=0;byte<256;++byte)for(int j=0;j<8;++j)
            result[byte][j]=(byte>>j)&1?-1.:1.;
        return result;
    }();
    return signs;
}

inline void write_word_zero(int d,uint64_t word,double* out) {
    static const auto zeros=[] {
        std::array<std::array<double,8>,256> table{};
        for(int byte=0;byte<256;++byte)for(int j=0;j<8;++j)table[byte][j]=(byte>>j)&1?-0.:0.;
        return table;
    }();
    int j=0;
    for(;j+8<=d;j+=8,word>>=8)std::copy_n(zeros[word&255].data(),8,out+j);
    for(;j<d;++j,word>>=1)out[j]=(word&1)?-0.:0.;
}

inline bool zero_eight(const double* credit) {
    uint64_t combined=0;
    for(int j=0;j<8;++j) {
        uint64_t word;std::memcpy(&word,credit+j,sizeof(word));combined|=word;
    }
    return !(combined&UINT64_C(0x7fffffffffffffff));
}

inline void map_word_credit(int d,uint64_t word,const double* z,const double* credit,double* out,
                            TrainFactorCache& cache,const RosaTrainingOptions& options) {
    int j=0;
    for(;j+8<=d;j+=8,word>>=8) {
        if(options.logit_mode || (options.tau>=std::numeric_limits<double>::min() && zero_eight(credit+j))) {
            const auto& sign=byte_signs()[word&255];
            for(int b=0;b<8;++b)out[j+b]=sign[b]*credit[j+b];
        } else for(int b=0;b<8;++b)out[j+b]=cache.apply(z[j+b],(word>>b)&1,credit[j+b],options);
    }
    for(int b=0;j+b<d;++b)out[j+b]=cache.apply(z[j+b],(word>>b)&1,credit[j+b],options);
}
inline void map_value_credit(int d,const uint8_t* v,const double* z,double* credit,
                             TrainFactorCache& cache,const RosaTrainingOptions& options) {
    int j=0;
    for(;j+8<=d;j+=8) {
        if(options.logit_mode || (options.tau>=std::numeric_limits<double>::min() && zero_eight(credit+j))) {
            for(int b=0;b<8;++b)credit[j+b]*=v[j+b]?-1.:1.;
        } else for(int b=0;b<8;++b)credit[j+b]=cache.apply(z[j+b],v[j+b],credit[j+b],options);
    }
    for(;j<d;++j)credit[j]=cache.apply(z[j],v[j],credit[j],options);
}

inline double train_radius(const double*e0,const double*e1,int d) {
    double a=0,b=0;
    for(int j=0;j<d;++j) {
        a+=e1[j]*e1[j];const double x=std::abs(e0[j])+std::abs(e1[j]);b+=x*x;
    }
    return std::max({std::sqrt(a),.5*std::sqrt(b),1e-12});
}

inline std::vector<int32_t> train_rows(const double*g,int n,int d,
                                      int count,bool active,uint64_t seed) {
    std::vector<int32_t> rows;
    if(!active) {
        if(n<=count){rows.resize(n);std::iota(rows.begin(),rows.end(),0);}
        else if(count==1)rows.push_back(0);
        else for(int i=0;i<count;++i)rows.push_back(int(double(i)*double(n-1)/double(count-1)));
        return rows;
    }
    double scale=0;
    for(int t=0;t<n;++t) {
        bool nonzero=false;
        for(int j=0;j<d;++j) {
            const double value=std::abs(g[size_t(t)*d+j]);
            nonzero|=value!=0;scale=std::max(scale,value);
        }
        if(nonzero)rows.push_back(t);
    }
    if(int(rows.size())<=count)return rows;
    // Exponential races implement weighted sampling without replacement.
    // Scaling all upstream gradients positively leaves the race unchanged.
    std::mt19937_64 random(seed);
    std::vector<std::pair<double,int32_t>> races;races.reserve(rows.size());
    for(const int32_t t:rows) {
        double norm=0;
        for(int j=0;j<d;++j){const double x=g[size_t(t)*d+j]/scale;norm+=x*x;}
        const double u=std::generate_canonical<double,53>(random);
        races.emplace_back(std::log1p(-u)/std::sqrt(norm),t);
    }
    std::nth_element(races.begin(),races.end()-count,races.end());
    rows.clear();for(auto it=races.end()-count;it!=races.end();++it)rows.push_back(it->second);
    std::sort(rows.begin(),rows.end());return rows;
}

// Every admitted proposal starts at a causal endpoint whose last word has
// at most two differing bits. For a bounded number of selected rows, an
// exact negative census avoids building the three full sorted band indexes.
inline bool pair_search_needed(const uint64_t* q,const uint64_t* k,int n,int d,
                               const std::vector<int32_t>& rows) {
    if(d<24 || rows.size()>std::max<size_t>(16,3*size_t(std::log2(double(n))+1)))return true;
    for(int t:rows) {
        const uint64_t word=q[t];
        for(int r=0;r<t;++r) {
            uint64_t diff=word^k[r];diff&=diff-1;diff&=diff-1;
            if(!diff)return true;
        }
    }
    return false;
}

inline int train_stream(const RosaTrainingOptions&o,int n,int d,
    const uint64_t*q,const uint64_t*k,const uint8_t*v,
    const double*qz,const double*kz,const double*vz,const double*g,
    const double*e0,const double*e1,double*gq,double*gk,double*gv,
    double*ge0,double*ge1,uint64_t*stats,uint64_t*trace,double*cost_trace,
    uint64_t proposal_seed,uint64_t repair_seed,
    const RosaApproximationOptions*approximation=nullptr,
    const int32_t*saved_lengths=nullptr,const int32_t*saved_routes=nullptr,
    uint64_t*approximation_stats=nullptr,uint64_t*subquadratic_stats=nullptr) {
    const size_t nd=size_t(n)*d;
    if(subquadratic_stats)std::fill(subquadratic_stats,subquadratic_stats+ROSA_SUBQUADRATIC_COUNTERS,uint64_t(0));
    std::fill(stats,stats+ROSA_TRAINING_COUNTERS,uint64_t(0));
    if(approximation_stats)std::fill(approximation_stats,approximation_stats+ROSA_APPROXIMATION_COUNTERS,uint64_t(0));
    // Zero upstream support implies zero trust, finite credit and embeddings.
    // This is the ordinary zero-adjoint identity, not an input-family kernel.
    if(std::none_of(g,g+nd,[](double value){return value!=0;})) {
        std::fill(gq,gq+nd,0.);std::fill(gk,gk+nd,0.);std::fill(gv,gv+nd,0.);
        stats[29]=1;stats[30]=nd;stats[31]=o.randomized!=0;
        if(approximation_stats)approximation_stats[27]=1;
        return 0;
    }
    std::vector<uint64_t> pairs(size_t(o.joint_pairs)*2);
    std::vector<int32_t> rows;
    int pair_count=0;
    if(o.joint_pairs && n) {
        rows=train_rows(g,n,d,o.joint_rows,o.active_rows!=0,proposal_seed);
        stats[25]=rows.size();
        if(pair_search_needed(q,k,n,d,rows)) {
            std::vector<uint8_t> allowed(n,uint8_t(1));
            uint64_t pc[12]={};double timings[2]={};
            const int capacity=std::max(1,8*o.joint_pairs);
            const int code=propose_pair_candidates(n,d,q,k,rows.data(),int(rows.size()),
                allowed.data(),allowed.data(),o.joint_pairs,proposal_seed,
                8,2,8,capacity,pairs.data(),nullptr,pc,timings);
            if(code)return code;
            pair_count=int(pc[5]);stats[26]=pc[3];
        }
    }
    rosa_subq::Stats work;
    std::vector<double> cq,ck,h,row_diagonal;
    rosa_subq::CreditDomain domain;
    double* cv=gv;
    std::vector<int32_t> lengths(n),routes(n);
    if(o.trust_diagonal) {
        const bool compact=!pair_count && !approximation;
        if(compact)row_diagonal.resize(n);else h.resize(nd);
        const double radius=train_radius(e0,e1,d);
        for(int t=0;t<n;++t) {
            double norm=0;for(int j=0;j<d;++j){double x=g[size_t(t)*d+j];norm+=x*x;}
            const double diagonal=o.trust_diagonal*std::sqrt(norm)/radius;
            if(compact)row_diagonal[t]=diagonal;
            else std::fill_n(h.data()+size_t(t)*d,d,diagonal);
        }
    }
    const auto& metric=row_diagonal.empty()?h:row_diagonal;
    const bool quadratic=std::any_of(metric.begin(),metric.end(),[](double x){return x!=0;});
    const int code=rosa_subq::run(n,d,q,k,v,g,quadratic && !h.empty()?h.data():nullptr,e0,e1,
        o.randomized!=0,repair_seed,approximation,cq,ck,lengths.data(),routes.data(),
        work,approximation_stats,saved_lengths,saved_routes,&domain,
        quadratic && !row_diagonal.empty()?row_diagonal.data():nullptr);
    if(code)return code;
    store_subquadratic_counters(work,stats);
    if(subquadratic_stats)std::copy(stats,stats+ROSA_SUBQUADRATIC_COUNTERS,subquadratic_stats);
    stats[23]=pair_count;stats[24]=pair_count;
    stats[27]=stats[28]=d;
    stats[29]=uint64_t(!approximation || (approximation_stats[27] && !approximation_stats[33]));
    stats[30]=nd;stats[31]=o.randomized!=0;stats[32]=rows.size();
    const bool zero_chain=o.logit_mode || o.tau>=std::numeric_limits<double>::min();
    const bool empty_v=zero_chain && std::none_of(routes.begin(),routes.end(),[](int r){return r>=0;});
    if(!empty_v)std::fill_n(cv,nd,0.);
    // The route penalty protects Q/K. V and embeddings remain linear.
    for(int t=0;t<n;++t) {
        const double* gy=g+size_t(t)*d;
        if(routes[t]<0) {
            for(int j=0;j<d;++j)ge0[j]+=gy[j];
        } else {
            const size_t source=size_t(routes[t]+1)*d;
            for(int j=0;j<d;++j) {
                const int sign=2*int(v[source+j])-1;ge1[j]+=gy[j]*sign;
                if(quadratic)cv[source+j]+=gy[j]*(-2*sign*e1[j]);
                else cv[source+j]+=-2*gy[j]*(sign*e1[j]);
            }
        }
    }
    TrainFactorCache qfactor,kfactor,vfactor;
    const std::array<double,64> zeros{};
    for(int t=0;t<n;++t) {
        const size_t x=size_t(t)*d;
        const int u=domain.position(t);
        if(u<0 && zero_chain) {
            write_word_zero(d,q[t],gq+x);write_word_zero(d,k[t],gk+x);
        } else {
            map_word_credit(d,q[t],qz+x,u<0?zeros.data():cq.data()+size_t(u)*d,gq+x,qfactor,o);
            map_word_credit(d,k[t],kz+x,u<0?zeros.data():ck.data()+size_t(u)*d,gk+x,kfactor,o);
        }
        if(empty_v)for(int j=0;j<d;++j)cv[x+j]=v[x+j]?-0.:0.;
        else map_value_credit(d,v+x,vz+x,cv+x,vfactor,o);
    }
    std::vector<uint64_t> qe,ke;
    std::vector<int32_t> edited_lengths,edited_routes;
    if(pair_count){qe.assign(q,q+n);ke.assign(k,k+n);edited_lengths.resize(n);edited_routes.resize(n);}
    for(int pair=0;pair<pair_count;++pair) {
        const uint64_t a=pairs[2*pair],b=pairs[2*pair+1];
        auto toggle=[&](uint64_t code) {
            auto& words=code>=nd?qe:ke;const size_t x=size_t(code%nd);
            words[x/d]^=uint64_t(1)<<(x%d);
        };
        auto replay_cost=[&]() {
            // Joint corners need only the ordinary forward trajectory.
            // Building deletion/repair suffix indexes here has no consumer.
            training_routes(qe.data(),ke.data(),n,d,edited_lengths.data(),edited_routes.data());
        ExactExpansion joint;
        for(int t=0;t<n;++t) {
            if(edited_routes[t]==routes[t])continue;
            for(int j=0;j<d;++j) {
                const size_t x=size_t(t)*d+j;
                if(g[x]==0 && (!quadratic || h[x]==0))continue;
                const double old=routes[t]<0?e0[j]:(2*int(v[size_t(routes[t]+1)*d+j])-1)*e1[j];
                const double now=edited_routes[t]<0?e0[j]:(2*int(v[size_t(edited_routes[t]+1)*d+j])-1)*e1[j];
                const double delta=now-old;
                const double term=quadratic ? g[x]*delta+.5*h[x]*delta*delta : g[x]*delta;
                joint.add(term);++stats[33];
            }
        }
            return joint.to_double();
        };
        auto singleton=[&](uint64_t code){
            size_t i=size_t(code%nd);int u=domain.position(int(i/d));
            return u<0?0.:(code>=nd?cq:ck)[size_t(u)*d+i%d];
        };
        double ca=singleton(a),cb=singleton(b);
        if(approximation && (!approximation_stats[27] || approximation_stats[33])) {
            // Restore both chosen singleton corners exactly. Sampling cannot
            // erase the very block selected to cross a discrete barrier.
            // Complete/certified residuals already provide those corners.
            toggle(a);ca=replay_cost();toggle(a);
            toggle(b);cb=replay_cost();toggle(b);
            stats[24]+=2;approximation_stats[28]+=2;
        }
        toggle(a);toggle(b);
        const double cij=replay_cost(),interaction=cij-ca-cb;
        const size_t ia=size_t(a%nd),ib=size_t(b%nd);
        const double za=a>=nd?qz[ia]:kz[ia],zb=b>=nd?qz[ib]:kz[ib];
        const int ba=int(((a>=nd?q[ia/d]:k[ia/d])>>(ia%d))&1);
        const int bb=int(((b>=nd?q[ib/d]:k[ib/d])>>(ib%d))&1);
        const double pa=train_sigmoid(za,o.tau),pb=train_sigmoid(zb,o.tau);
        const double fa=ba?1-pa:pa,fb=bb?1-pb:pb;
        (a>=nd?gq:gk)[ia]=train_chain(za,ba,ca+fb*interaction,o);
        (b>=nd?gq:gk)[ib]=train_chain(zb,bb,cb+fa*interaction,o);
        if(trace){trace[2*pair]=a;trace[2*pair+1]=b;}
        if(cost_trace){double*p=cost_trace+4*pair;p[0]=ca;p[1]=cb;p[2]=cij;p[3]=interaction;}
        toggle(a);toggle(b);
    }
    stats[34]=1;return 0;
}
} // namespace rosa_exact

ROSA_EXPORT void rosa_training_defaults(RosaTrainingOptions*o) {
    if(!o)return;
    std::memset(o,0,sizeof(*o));o->abi_version=ROSA_TRAINING_ABI_VERSION;
    o->struct_size=sizeof(*o);o->credit_order=2;o->joint_pairs=1;o->joint_rows=32;
    o->active_rows=1;o->tau=1.;o->trust_diagonal=.5;o->randomized=1;
}

ROSA_EXPORT void rosa_approximation_defaults(RosaApproximationOptions*o) {
    if(!o)return;
    std::memset(o,0,sizeof(*o));o->abi_version=ROSA_APPROXIMATION_ABI_VERSION;
    o->struct_size=sizeof(*o);o->sample_rows=32;o->importance=1;
}

ROSA_EXPORT int rosa_training_forward(const RosaTrainingOptions*o,int b,int groups,int n,int d,
    const double*qz,const double*kz,const double*vz,const double*e0,const double*e1,
    uint64_t*q,uint64_t*k,uint8_t*v,int32_t*lengths,int32_t*routes,double*y,uint64_t*counts) {
  try {
    if(!rosa_exact::training_options(o) || !rosa_exact::training_shape(b,groups,n,d))return 1;
    const size_t streams=size_t(b)*groups,nd=size_t(n)*d,total=streams*nd;
    if(!rosa_exact::finite_values(qz,total)||!rosa_exact::finite_values(kz,total)||!rosa_exact::finite_values(vz,total)
       ||!rosa_exact::finite_values(e0,size_t(groups)*d)||!rosa_exact::finite_values(e1,size_t(groups)*d))return 4;
    if((streams&& !counts)||(streams*n&&(!q||!k||!lengths||!routes))||(total&&(!v||!y)))return 1;
    if(n==0) {
        if(streams)std::fill(counts,counts+streams*ROSA_FORWARD_COUNTERS,uint64_t(0));
        return 0;
    }
    for(size_t s=0;s<streams;++s) {
        const size_t base=s*nd,word=s*n,embedding=(s%groups)*d;
        for(int t=0;t<n;++t) {
            uint64_t a=0,c=0;
            for(int j=0;j<d;++j){const size_t x=base+size_t(t)*d+j;
                a|=uint64_t(qz[x]>0)<<j;c|=uint64_t(kz[x]>0)<<j;v[x]=uint8_t(vz[x]>0);}
            q[word+t]=a;k[word+t]=c;
        }
        const auto work=rosa_exact::training_routes(q+word,k+word,n,d,lengths+word,routes+word);
        for(int t=0;t<n;++t)for(int j=0;j<d;++j) {
            const int r=routes[word+t];
            y[base+size_t(t)*d+j]=r<0?e0[embedding+j]:(2*int(v[base+size_t(r+1)*d+j])-1)*e1[embedding+j];
        }
        uint64_t*c=counts+s*ROSA_FORWARD_COUNTERS;c[0]=1;c[1]=work.transition_probes;
        c[2]=work.hash_probe_steps;c[3]=work.endpoint_queries;
    }
    return 0;
  }catch(const std::bad_alloc&){return 2;}catch(const std::exception&){return 3;}catch(...){return 3;}
}

static int training_backward_impl(const RosaTrainingOptions*o,const RosaApproximationOptions*approximation,
    int b,int groups,int n,int d,
    const uint64_t*q,const uint64_t*k,const uint8_t*v,const double*qz,const double*kz,const double*vz,
    const double*g,const double*e0,const double*e1,double*gq,double*gk,double*gv,
    double*ge0,double*ge1,uint64_t*counts,uint64_t*trace,double*cost_trace,
    const int32_t*saved_lengths,const int32_t*saved_routes,uint64_t*approximation_counts,
    uint64_t*subquadratic_counts=nullptr) {
  try {
    if(!rosa_exact::training_options(o)||!rosa_exact::training_shape(b,groups,n,d))return 1;
    const size_t streams=size_t(b)*groups,nd=size_t(n)*d,total=streams*nd;
    if(!rosa_exact::finite_values(qz,total)||!rosa_exact::finite_values(kz,total)||!rosa_exact::finite_values(vz,total)
       ||!rosa_exact::finite_values(g,total)||!rosa_exact::finite_values(e0,size_t(groups)*d)
       ||!rosa_exact::finite_values(e1,size_t(groups)*d))return 4;
    if(!ge0||!ge1||(streams&&!counts)||(streams*n&&(!q||!k))||(total&&(!v||!gq||!gk||!gv)))return 1;
    if(approximation) {
        if(!rosa_exact::approximation_options(approximation)||o->repair_budget
           ||(streams&&!approximation_counts)||(streams*n&&(!saved_lengths||!saved_routes)))return 1;
        for(size_t s=0;s<streams;++s)for(int t=0;t<n;++t) {
            const size_t at=s*n+t;const int len=saved_lengths[at],route=saved_routes[at];
            if(len<0||len>t||route < -1||route>=t||(len==0)!=(route<0)||len>route+1)return 1;
        }
    }
    std::fill(ge0,ge0+size_t(groups)*d,0.);std::fill(ge1,ge1+size_t(groups)*d,0.);
    if(n==0) {
        if(streams)std::fill(counts,counts+streams*ROSA_TRAINING_COUNTERS,uint64_t(0));
        if(approximation_counts && streams)std::fill(approximation_counts,
            approximation_counts+streams*ROSA_APPROXIMATION_COUNTERS,uint64_t(0));
        if(subquadratic_counts && streams)std::fill(subquadratic_counts,
            subquadratic_counts+streams*ROSA_SUBQUADRATIC_COUNTERS,uint64_t(0));
        for(size_t s=0;s<streams;++s) {
            counts[s*ROSA_TRAINING_COUNTERS+29]=1;
            if(approximation_counts)approximation_counts[s*ROSA_APPROXIMATION_COUNTERS+27]=1;
        }
        return 0;
    }
    for(size_t s=0;s<streams;++s) {
        const size_t base=s*nd,word=s*n,embedding=(s%groups)*d;
        const uint64_t salt=UINT64_C(0x9e3779b97f4a7c15)*(s+1);
        const int code=rosa_exact::train_stream(*o,n,d,q+word,k+word,v+base,qz+base,kz+base,vz+base,
            g+base,e0+embedding,e1+embedding,gq+base,gk+base,gv+base,ge0+embedding,ge1+embedding,
            counts+s*ROSA_TRAINING_COUNTERS,trace?trace+s*size_t(o->joint_pairs)*2:nullptr,
            cost_trace?cost_trace+s*size_t(o->joint_pairs)*4:nullptr,o->proposal_seed+salt,o->repair_seed+salt,
            approximation,saved_lengths?saved_lengths+word:nullptr,saved_routes?saved_routes+word:nullptr,
            approximation_counts?approximation_counts+s*ROSA_APPROXIMATION_COUNTERS:nullptr,
            subquadratic_counts?subquadratic_counts+s*ROSA_SUBQUADRATIC_COUNTERS:nullptr);
        if(code)return code;
    }
    return 0;
  }catch(const std::bad_alloc&){return 2;}catch(const std::exception&){return 3;}catch(...){return 3;}
}

ROSA_EXPORT int rosa_training_backward(const RosaTrainingOptions*o,int b,int groups,int n,int d,
    const uint64_t*q,const uint64_t*k,const uint8_t*v,const double*qz,const double*kz,const double*vz,
    const double*g,const double*e0,const double*e1,double*gq,double*gk,double*gv,
    double*ge0,double*ge1,uint64_t*counts,uint64_t*trace,double*cost_trace) {
    return training_backward_impl(o,nullptr,b,groups,n,d,q,k,v,qz,kz,vz,g,e0,e1,
        gq,gk,gv,ge0,ge1,counts,trace,cost_trace,nullptr,nullptr,nullptr);
}

ROSA_EXPORT int rosa_training_backward_approx(const RosaTrainingOptions*o,
    const RosaApproximationOptions*approximation,int b,int groups,int n,int d,
    const uint64_t*q,const uint64_t*k,const uint8_t*v,const double*qz,const double*kz,const double*vz,
    const double*g,const double*e0,const double*e1,double*gq,double*gk,double*gv,
    double*ge0,double*ge1,uint64_t*counts,uint64_t*trace,double*cost_trace,
    const int32_t*saved_lengths,const int32_t*saved_routes,uint64_t*approximation_counts) {
    if(!approximation)return 1;
    return training_backward_impl(o,approximation,b,groups,n,d,q,k,v,qz,kz,vz,g,e0,e1,
        gq,gk,gv,ge0,ge1,counts,trace,cost_trace,saved_lengths,saved_routes,approximation_counts);
}

// Diagnostic alias of the same engine, with a separate structural counter buffer.
ROSA_EXPORT int rosa_training_backward_subquadratic(const RosaTrainingOptions*o,
    const RosaApproximationOptions*approximation,int b,int groups,int n,int d,
    const uint64_t*q,const uint64_t*k,const uint8_t*v,const double*qz,const double*kz,const double*vz,
    const double*g,const double*e0,const double*e1,double*gq,double*gk,double*gv,
    double*ge0,double*ge1,uint64_t*counts,uint64_t*trace,double*cost_trace,
    const int32_t*saved_lengths,const int32_t*saved_routes,uint64_t*approximation_counts,
    uint64_t*subquadratic_counts) {
    return training_backward_impl(o,approximation,b,groups,n,d,q,k,v,qz,kz,vz,g,e0,e1,
        gq,gk,gv,ge0,ge1,counts,trace,cost_trace,saved_lengths,saved_routes,
        approximation_counts,subquadratic_counts);
}
