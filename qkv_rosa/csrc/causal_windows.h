#pragma once
#include "forward.h"
#include "engine/index.h"
#include <memory>

namespace rosa_exact {
// A row can extend the previous best length by at most one. Short suffixes
// have direct latest-end tables. A long suffix is a refinement of the same
// bucket. A singleton extension, a bounded posting list or a suffix-range
// predecessor query resolves that row without restarting the trajectory.
inline SamForward::Counters training_routes(const uint64_t* q,const uint64_t* k,int n,int bits,
        int32_t* lengths,int32_t* routes) {
    SamForward::Counters work;
    if(!n)return work;
    lengths[0]=0;routes[0]=-1;
    uint64_t used=0;
    for(int i=0;i<n;++i)used|=q[i]|k[i];
    while(bits<64 && (used>>bits))++bits;
    struct Radix {int mask,offset;};
    std::vector<Radix> levels;
    int slots=0;
    const uint64_t capacity=std::min<uint64_t>(1<<20,std::max<uint64_t>(256,uint64_t(n)*16));
    for(int h=1;h*bits<=20 && (UINT64_C(1)<<(h*bits))<=capacity;++h) {
        const int size=1<<(h*bits);
        levels.push_back({size-1,slots});slots+=size;
    }
    const bool hashed=levels.empty();
    if(hashed){slots=1;while(slots<n*2)slots*=2;}
    auto slot=[&](uint64_t x){x^=x>>32;x*=UINT64_C(0xd6e8feb86659fd93);return int((x^(x>>32))&(slots-1));};
    std::vector<int> head(slots,-1),previous(n,-1),chain(n);
    std::unique_ptr<rosa_subq::Index> full;
    auto index=[&]() -> rosa_subq::Index& {
        if(!full) {
            // Only occurrence and LCE queries are used during construction.
            std::vector<rosa_subq::Route> empty(n);
            full=std::make_unique<rosa_subq::Index>(q,k,n,false,0,&empty);
        }
        return *full;
    };
    const int top=std::max(1,int(levels.size()));
    uint64_t qw=0,kw=0;
    for(int t=1;t<n;++t) {
        const int upper=lengths[t-1]+1,r=t-1;
        int h=0,end=-1;
        if(hashed) {
            int& latest=head[slot(k[r])];previous[r]=latest;latest=r;
            chain[r]=previous[r]<0?1:chain[previous[r]]+1;
            end=head[slot(q[t])];h=1;
            ++work.hash_probe_steps;
        } else {
            qw=(qw<<bits)|q[t];kw=(kw<<bits)|k[r];
            for(int j=0;j<std::min(t,top);++j) {
                const auto lv=levels[j];
                int& latest=head[lv.offset+(kw&lv.mask)];
                if(j+1==top){previous[r]=latest;chain[r]=latest<0?1:chain[latest]+1;}
                latest=r;
            }
            for(h=std::min(upper,top);h;--h) {
                const auto lv=levels[h-1];++work.transition_probes;
                end=head[lv.offset+(qw&lv.mask)];if(end>=0)break;
            }
        }
        if(end<0){lengths[t]=0;routes[t]=-1;continue;}
        if(hashed || (h==top && h<upper)) {
            ++work.endpoint_queries;
            // The newest possible end reaches the temporal upper bound.
            if(end==routes[t-1]+1 && k[end]==q[t] && (lengths[t-1] || upper==1))h=upper;
            else if(chain[end]<=12) {
                int best=0,winner=-1;
                for(int e=end;e>=0;e=previous[e]) {
                    if(hashed && k[e]!=q[t])continue;
                    int len=h,stop=std::min(e+1,upper);
                    const int direct=std::min(stop,h+6);
                    while(len<direct && q[t-len]==k[e-len])++len;
                    if(len==direct && len<stop)len+=index().lcs(t-len,e-len);
                    if(len>best){best=len;winner=e;}
                    if(len==upper)break;
                }
                h=best;end=winner;
            } else {
                const int recent=hashed?index().R(t,1):end;
                if(recent<0){h=0;end=-1;}
                else {
                    int low=index().lcs(t,recent),high=upper;end=recent;
                    while(low<high) {
                        const int mid=low+(high-low+1)/2,last=index().R(t,mid);
                        if(last>=0){low=mid;end=last;}else high=mid-1;
                    }
                    h=low;
                }
            }
        }
        lengths[t]=h;routes[t]=end;
    }
    return work;
}
} // namespace rosa_exact
