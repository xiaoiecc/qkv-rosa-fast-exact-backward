#pragma once
#include "endpoint_credit.h"
#include "packed_suffix_level.h"
#include <memory>

namespace rosa_subq {
namespace suffix_frontier {

// One deterministic report rule for every exact stream. Short exact levels
// need at most h+1 witnesses: one edited key can cover at most h endpoints.
inline void evaluate(const uint64_t* q,const uint64_t* k,int n,int bits,
        const std::vector<Route>& base,const uint8_t* v,const double* g,const double* h,
        const double* e0,const double* e1,std::vector<double>& cq,std::vector<double>& ck,Stats& stats) {
    if(n<2)return;
    // Standalone native callers may carry immutable tag bits above the
    // editable width. Keep them in equality and exclude them from edits.
    uint64_t occupied=0;
    for(int i=0;i<n;++i)occupied|=q[i]|k[i];
    int symbol_bits=bits;
    while(symbol_bits<64 && (occupied>>symbol_bits))++symbol_bits;
    int depth=1;
    // Avoid shifts of 64 and determine the ceiling without floating point.
    while(depth*symbol_bits<31 && (uint32_t(1)<<(depth*symbol_bits))<uint32_t(n))++depth;
    std::vector<PackedLevel> levels;
    levels.reserve(depth);
    for(int j=1;j<=depth;++j)levels.emplace_back(n,j*symbol_bits);
    const int digits=64/symbol_bits;
    const uint64_t digit_mask=low_mask(digits*symbol_bits);
    std::vector<uint64_t> qw(n),kw(n);
    uint64_t qa=0,ka=0;
    for(int i=0;i<n;++i) {
        qa=(symbol_bits==64?0:qa<<symbol_bits)|q[i];ka=(symbol_bits==64?0:ka<<symbol_bits)|k[i];
        qw[i]=qa&digit_mask;kw[i]=ka&digit_mask;
    }
    std::unique_ptr<Index> longer_arms;
    auto lcs=[&](int t,int r) {
        if(r<0)return 0;
        const uint64_t diff=qw[t]^kw[r];
        if(diff)return std::min(rosa_random::row_bit(diff)/symbol_bits,r+1);
        if(r<digits)return r+1;
        if(q[t-digits]!=k[r-digits])return digits;
        if(!longer_arms)longer_arms=std::make_unique<Index>(q,k,n,false,0,&base);
        return digits+longer_arms->lcs(t-digits,r-digits);
    };
    endpoint_credit::EventSweep sweep(q,k,bits,stats);
    std::vector<int> seen(n,-1),length(n),ends;
    std::vector<double> qdiff(size_t(n)+1),kdiff(size_t(n)+1),costs(size_t(n)+1);
    std::vector<int> stamp(size_t(n)+1,-1),exact_stamp(n,-1);
    for(int t=1;t<n;++t) {
        for(int j=1;j<=depth && j<=t;++j)
            levels[j-1].insert(kw[t-1]&levels[j-1].mask(),t-1);
        const auto original=base[t];
        const int L=original.length,p=std::min({depth,t,L+1});
        auto cost=[&](int end) {
            if(end==original.end)return 0.;
            if(stamp[end+1]!=t) {
                auto value=[&](int r,int c){return r<0?e0[c]:(2*int(v[size_t(r+1)*bits+c])-1)*e1[c];};
                double sum=0.;
                for(int c=0;c<bits;++c) {
                    const size_t at=size_t(t)*bits+c;
                    const double dy=value(end,c)-value(original.end,c);
                    sum+=g[at]*dy;
                    if(h)sum+=(.5*h[at]*dy)*dy;
                }
                costs[end+1]=sum;stamp[end+1]=t;
            }
            return costs[end+1];
        };
        sweep.row_events(t,original,[&](auto exact,auto both,auto q_only,auto k_only) {
            ends.clear();
            auto remember=[&](int r,int len) {
                if(exact_stamp[r]!=t){exact_stamp[r]=t;length[r]=len;ends.push_back(r);}
                else length[r]=std::max(length[r],len);
            };
            const auto inspect=[&](int r,bool long_report) {
                if(seen[r]==t)return;
                seen[r]=t;++stats.suffix_queries;
                const int a=lcs(t,r);
                if(long_report && a>=p)remember(r,a);
                if(r<a)return;
                const uint64_t mask=q[t-a]^k[r-a];
                if(!mask || (mask&(mask-1)))return;
                const int bit=rosa_random::row_bit(mask);
                if(bit>=bits)return;
                const int len=a+1+lcs(t-a-1,r-a-1);
                if(long_report)both(a,bit,len,r);
                else if(len<p)k_only(a,bit,len,r);
            };
            std::array<Route,64> short_q;
            uint64_t short_bits=0;
            for(int j=1;j<=p;++j) {
                auto& level=levels[j-1];
                const uint64_t prefix=qw[t]&level.mask();
                const int last=level.exact(prefix);
                if(j==p)level.endpoints(last,-1,n,[&](int r){inspect(r,true);});
                else level.endpoints(last,-1,j+1,[&](int r){remember(r,j);});
                level.neighbors(prefix,[&](uint64_t mask,int r){
                    const int pos=rosa_random::row_bit(mask);
                    if(pos%symbol_bits>=bits)return;
                    if(j==p){level.endpoints(r,-1,n,[&](int end){inspect(end,true);});return;}
                    short_bits|=mask;short_q[pos]={j,r};
                    if(j==L)level.endpoints(r,original.end,n,[&](int end){k_only(pos/symbol_bits,pos%symbol_bits,j,end);});
                });
            }
            for(;short_bits;short_bits&=short_bits-1) {
                const int pos=rosa_random::row_bit(short_bits);
                q_only(pos/symbol_bits,pos%symbol_bits,short_q[pos].length,short_q[pos].end);
            }
            // A shorter repaired winner must either tie later (handled
            // above), or touch the original winning key interval.
            if(p>1 && L) {
                const int first=original.end-L+1;
                const int last=first+std::min(t-first,L+p-2);
                for(int r=first;r<last;++r)inspect(r,false);
            }
            for(int r:ends)exact(length[r],r);
        },cost,qdiff,kdiff,cq,ck);
    }
    double qsum=0.,ksum=0.;
    for(int i=0;i<n;++i) {
        qsum+=qdiff[i];ksum+=kdiff[i];
        for(int j=0;j<bits;++j){cq[size_t(i)*bits+j]+=qsum;ck[size_t(i)*bits+j]+=ksum;}
    }
    std::fill_n(cq.begin(),bits,0.);std::fill_n(ck.end()-bits,bits,0.);
    size_t space=longer_arms?longer_arms->workspace_bytes():0;
    for(const auto& level:levels)space+=level.bytes();
    stats.peak_workspace_bytes=std::max<uint64_t>(stats.peak_workspace_bytes,space+sweep.bytes()
        +(qw.capacity()+kw.capacity())*sizeof(uint64_t)
        +(seen.capacity()+length.capacity()+ends.capacity()+stamp.capacity()+exact_stamp.capacity())*sizeof(int)
        +(qdiff.capacity()+kdiff.capacity()+costs.capacity())*sizeof(double));
}
} // namespace suffix_frontier
} // namespace rosa_subq
