#pragma once
// Included after uniform_impl: the public training API reuses that one engine.
#include "training.h"
#include <cstring>
#include <random>

namespace rosa_exact {

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
        && o->repair_budget>=0 && std::isfinite(o->tau) && o->tau>0
        && std::isfinite(o->trust_diagonal) && o->trust_diagonal>=0
        && (o->credit_order==2 || (!o->joint_pairs && !o->trust_diagonal));
}
inline bool finite_values(const double* p,size_t count) {
    if(count && !p)return false;
    for(size_t i=0;i<count;++i)if(!std::isfinite(p[i]))return false;
    return true;
}
inline double train_sigmoid(double z,double tau) {
    const double scaled=z/tau;
    const double exponential=std::exp(-std::abs(scaled));
    return scaled>=0 ? 1/(1+exponential) : exponential/(1+exponential);
}
inline double train_chain(double z,int bit,double credit,const RosaTrainingOptions&o) {
    const double oriented=(bit ? -1.0 : 1.0)*credit;
    if(o.logit_mode)return oriented;
    const double p=train_sigmoid(z,o.tau);
    return (p*(1-p)/o.tau)*oriented;
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

inline int train_stream(const RosaTrainingOptions&o,int n,int d,
    const uint64_t*q,const uint64_t*k,const uint8_t*v,
    const double*qz,const double*kz,const double*vz,const double*g,
    const double*e0,const double*e1,double*gq,double*gk,double*gv,
    double*ge0,double*ge1,uint64_t*stats,uint64_t*trace,double*cost_trace,
    uint64_t proposal_seed,uint64_t repair_seed) {
    const size_t nd=size_t(n)*d;
    std::fill(stats,stats+ROSA_TRAINING_COUNTERS,uint64_t(0));
    // Zero upstream support implies zero trust, finite credit and embeddings.
    // This is the ordinary zero-adjoint identity, not an input-family kernel.
    if(std::none_of(g,g+nd,[](double value){return value!=0;})) {
        std::fill(gq,gq+nd,0.);std::fill(gk,gk+nd,0.);std::fill(gv,gv+nd,0.);
        stats[29]=1;stats[30]=nd;stats[31]=o.randomized!=0;return 0;
    }
    std::vector<double> cq(nd),ck(nd),cv(nd),h;
    std::vector<int32_t> lengths(n),routes(n);
    std::vector<uint64_t> pairs(size_t(o.joint_pairs)*2);
    std::vector<int32_t> rows;
    uint64_t forced=0;int pair_count=0;
    if(o.joint_pairs && n) {
        rows=train_rows(g,n,d,o.joint_rows,o.active_rows!=0,proposal_seed);
        std::vector<uint8_t> allowed(n,uint8_t(1));
        uint64_t pc[12]={};double timings[2]={};
        const int capacity=std::max(1,8*o.joint_pairs);
        const int code=propose_pair_candidates(n,d,q,k,rows.data(),int(rows.size()),
            allowed.data(),allowed.data(),o.joint_pairs,proposal_seed,
            8,2,8,capacity,pairs.data(),nullptr,pc,timings);
        if(code)return code;
        pair_count=int(pc[5]);stats[25]=rows.size();stats[26]=pc[3];
        for(int i=0;i<2*pair_count;++i)forced|=uint64_t(1)<<(pairs[i]%uint64_t(d));
    }
    if(o.trust_diagonal) {
        h.resize(nd);const double radius=train_radius(e0,e1,d);
        for(int t=0;t<n;++t) {
            double norm=0;for(int j=0;j<d;++j){double x=g[size_t(t)*d+j];norm+=x*x;}
            const double diagonal=o.trust_diagonal*std::sqrt(norm)/radius;
            std::fill(h.begin()+size_t(t)*d,h.begin()+size_t(t+1)*d,diagonal);
        }
    }
    const bool quadratic=!h.empty() && std::any_of(h.begin(),h.end(),[](double x){return x!=0;});
    const bool sampled=o.repair_budget>0 && o.repair_budget<d;
    std::vector<double> qb(sampled?n:0),kb(sampled?n:0);
    uint64_t selection[5]={};
    const int budget=sampled ? o.repair_budget : -1;
    const int code=quadratic
        ? uniform_impl<true>(n,d,q,k,v,g,h.data(),e0,e1,cq.data(),ck.data(),cv.data(),
            lengths.data(),routes.data(),stats,~uint64_t(0),sampled?qb.data():nullptr,
            sampled?kb.data():nullptr,selection,budget,repair_seed,forced,
            o.randomized!=0,repair_seed)
        : uniform_impl<false>(n,d,q,k,v,g,nullptr,e0,e1,cq.data(),ck.data(),cv.data(),
            lengths.data(),routes.data(),stats,~uint64_t(0),sampled?qb.data():nullptr,
            sampled?kb.data():nullptr,selection,budget,repair_seed,forced,
            o.randomized!=0,repair_seed);
    if(code)return code;
    const uint64_t support=selection[3],selected=selection[4];
    if(sampled) {
        const uint64_t optional=support&~forced;
        const int total=pair_popcount(optional),kept=pair_popcount(selected&optional);
        const double probability=total ? double(kept)/total : 1.;
        if(total && !kept)return 3;
        if(probability!=1.)for(int j=0;j<d;++j)if((selected&optional)>>j&1) {
            for(int t=0;t<n;++t){const size_t x=size_t(t)*d+j;
                cq[x]=qb[t]+(cq[x]-qb[t])/probability;
                ck[x]=kb[t]+(ck[x]-kb[t])/probability;}
        }
    }
    stats[23]=pair_count;stats[24]=pair_count;
    stats[27]=pair_popcount(support);stats[28]=pair_popcount(selected);
    stats[29]=uint64_t(!sampled || selected==support);stats[30]=nd;
    stats[31]=o.randomized!=0;stats[32]=rows.size();
    // The route penalty protects Q/K. V remains the original linear credit.
    if(quadratic)std::fill(cv.begin(),cv.end(),0.);
    for(int t=0;t<n;++t)for(int j=0;j<d;++j) {
        const size_t x=size_t(t)*d+j;const double gy=g[x];
        if(routes[t]<0)ge0[j]+=gy;
        else {
            const size_t source=size_t(routes[t]+1)*d+j;
            const int sign=2*int(v[source])-1;ge1[j]+=gy*sign;
            if(quadratic)cv[source]+=gy*(-2*sign*e1[j]);
        }
    }
    for(int t=0;t<n;++t)for(int j=0;j<d;++j) {
        const size_t x=size_t(t)*d+j;
        gq[x]=train_chain(qz[x],int((q[t]>>j)&1),cq[x],o);
        gk[x]=train_chain(kz[x],int((k[t]>>j)&1),ck[x],o);
        gv[x]=train_chain(vz[x],int(v[x]),cv[x],o);
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
        toggle(a);toggle(b);
        SamForward scan(qe.data(),ke.data(),n,repair_seed);
        scan.run(edited_lengths.data(),edited_routes.data());
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
        auto singleton=[&](uint64_t code){return code>=nd?cq[size_t(code-nd)]:ck[size_t(code)];};
        const double ca=singleton(a),cb=singleton(b),cij=joint.to_double(),interaction=cij-ca-cb;
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
    o->active_rows=1;o->tau=1.;o->trust_diagonal=.5;
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
    for(size_t s=0;s<streams;++s) {
        const size_t base=s*nd,word=s*n,embedding=(s%groups)*d;
        for(int t=0;t<n;++t) {
            uint64_t a=0,c=0;
            for(int j=0;j<d;++j){const size_t x=base+size_t(t)*d+j;
                a|=uint64_t(qz[x]>0)<<j;c|=uint64_t(kz[x]>0)<<j;v[x]=uint8_t(vz[x]>0);}
            q[word+t]=a;k[word+t]=c;
        }
        rosa_exact::SamForward scan(q+word,k+word,n,o->repair_seed+uint64_t(s));
        scan.run(lengths+word,routes+word);
        for(int t=0;t<n;++t)for(int j=0;j<d;++j) {
            const int r=routes[word+t];
            y[base+size_t(t)*d+j]=r<0?e0[embedding+j]:(2*int(v[base+size_t(r+1)*d+j])-1)*e1[embedding+j];
        }
        uint64_t*c=counts+s*ROSA_FORWARD_COUNTERS;c[0]=1;c[1]=scan.counters.transition_probes;
        c[2]=scan.counters.hash_probe_steps;c[3]=scan.counters.endpoint_queries;
    }
    return 0;
  }catch(const std::bad_alloc&){return 2;}catch(const std::exception&){return 3;}catch(...){return 3;}
}

ROSA_EXPORT int rosa_training_backward(const RosaTrainingOptions*o,int b,int groups,int n,int d,
    const uint64_t*q,const uint64_t*k,const uint8_t*v,const double*qz,const double*kz,const double*vz,
    const double*g,const double*e0,const double*e1,double*gq,double*gk,double*gv,
    double*ge0,double*ge1,uint64_t*counts,uint64_t*trace,double*cost_trace) {
  try {
    if(!rosa_exact::training_options(o)||!rosa_exact::training_shape(b,groups,n,d)||o->repair_budget>d)return 1;
    const size_t streams=size_t(b)*groups,nd=size_t(n)*d,total=streams*nd;
    if(!rosa_exact::finite_values(qz,total)||!rosa_exact::finite_values(kz,total)||!rosa_exact::finite_values(vz,total)
       ||!rosa_exact::finite_values(g,total)||!rosa_exact::finite_values(e0,size_t(groups)*d)
       ||!rosa_exact::finite_values(e1,size_t(groups)*d))return 4;
    if(!ge0||!ge1||(streams&&!counts)||(streams*n&&(!q||!k))||(total&&(!v||!gq||!gk||!gv)))return 1;
    std::fill(ge0,ge0+size_t(groups)*d,0.);std::fill(ge1,ge1+size_t(groups)*d,0.);
    for(size_t s=0;s<streams;++s) {
        const size_t base=s*nd,word=s*n,embedding=(s%groups)*d;
        const uint64_t salt=UINT64_C(0x9e3779b97f4a7c15)*(s+1);
        const int code=rosa_exact::train_stream(*o,n,d,q+word,k+word,v+base,qz+base,kz+base,vz+base,
            g+base,e0+embedding,e1+embedding,gq+base,gk+base,gv+base,ge0+embedding,ge1+embedding,
            counts+s*ROSA_TRAINING_COUNTERS,trace?trace+s*size_t(o->joint_pairs)*2:nullptr,
            cost_trace?cost_trace+s*size_t(o->joint_pairs)*4:nullptr,o->proposal_seed+salt,o->repair_seed+salt);
        if(code)return code;
    }
    return 0;
  }catch(const std::bad_alloc&){return 2;}catch(const std::exception&){return 3;}catch(...){return 3;}
}
