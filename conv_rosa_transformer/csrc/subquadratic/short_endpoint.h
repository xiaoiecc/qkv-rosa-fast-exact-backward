#pragma once
#include "index.h"
#include "near_k.h"
#include <array>

namespace rosa_subq::endpoint_credit {

// With short baseline matches there are at most L deletion owners per
// side in a row. Accumulate them directly so unrelated coordinates never
// see a floating residual from the subtraction in a difference-array scan.
class ShortEndpointSweep {
    static constexpr int limit=4;
    const Index& index;
    const int bits;
    std::vector<Route> key_best;
    std::array<Route,(limit+1)*64> query_best{};
    std::vector<size_t> keys;
    std::vector<int> queries;
    int suffix(int t,int r) const {
        int length=0;
        while(r>=0 && index.q[t]==index.k[r]){++length;--r;--t;}
        return length;
    }
public:
    static bool applicable(const std::vector<Route>& base) {
        return std::all_of(base.begin(),base.end(),[](Route r){return r.length<=limit;});
    }
    ShortEndpointSweep(const Index& ix,int b,bool enabled)
        :index(ix),bits(b),key_best(enabled?size_t(ix.n)*b:0) {}
    template<class Report,class Cost>
    void row(int t,Report&& report,Cost&& cost,std::vector<double>& qpoint,std::vector<double>& kpoint,
             std::vector<double>& cq,std::vector<double>& ck,Stats& stats) {
        const Route base=index.baseline()[t];
        const int length=base.length,first=base.end-length+1;
        std::array<Route,limit> qdelete{},kdelete{};
        report([&](int r) {
            const int a=suffix(t,r);
            if(a>length)throw std::invalid_argument("inconsistent saved match length");
            if(a)for(int h=0;h<length;++h) {
                const int qlength=std::min(a,h),s=first+h;
                const int klength=r<s?a:std::min(a,r-s);
                if(qlength)qdelete[h]=std::max(qdelete[h],Route{qlength,r});
                if(klength)kdelete[h]=std::max(kdelete[h],Route{klength,r});
            }
            const int u=t-a,s=r-a;
            if(s<0)return;
            const uint64_t diff=index.q[u]^index.k[s];
            if(!diff || (diff&(diff-1)))return;
            const int bit=near_detail::one_bit_index(diff);
            if(bit>=bits)return;
            const Route cand{a+1+suffix(u-1,s-1),r};
            const int qi=a*bits+bit;
            const size_t ki=size_t(s)*bits+bit;
            if(!query_best[qi].length)queries.push_back(qi);
            if(!key_best[ki].length)keys.push_back(ki);
            query_best[qi]=std::max(query_best[qi],cand);
            key_best[ki]=std::max(key_best[ki],cand);
            ++stats.k_centers;
        });
        for(int h=0;h<length;++h) {
            qpoint[t-h]+=cost(qdelete[h].end);
            kpoint[first+h]+=cost(kdelete[h].end);
        }
        for(int qi:queries) {
            const int age=qi/bits;
            const Route other=age<length?qdelete[age]:base,best=query_best[qi];
            if(best>other && best.end!=other.end)
                cq[size_t(t-age)*bits+qi%bits]+=cost(best.end)-cost(other.end);
            query_best[qi]={};
        }
        for(size_t ki:keys) {
            const int s=int(ki/bits);
            const Route other=s<first || s>base.end?base:kdelete[s-first],best=key_best[ki];
            if(best>other && best.end!=other.end)ck[ki]+=cost(best.end)-cost(other.end);
            key_best[ki]={};
        }
        queries.clear();keys.clear();
    }
    size_t bytes() const {
        return key_best.capacity()*sizeof(Route)+keys.capacity()*sizeof(size_t)
            +queries.capacity()*sizeof(int)+sizeof(query_best);
    }
};
}
