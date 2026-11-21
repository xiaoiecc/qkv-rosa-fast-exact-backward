#pragma once
#include "forward.h"

namespace rosa_exact {

// Online dictionary of exact causal suffixes. Each table stores the latest
// endpoint of a radix word. The deepest also chains occurrences, so longer
// matches can be resolved by checking their original symbols. Budgeted work
// and a full SAM retry preserve the general forward bound.
inline bool window_forward(const uint64_t* query,const uint64_t* key,int n,int bits,
        int32_t* lengths,int32_t* routes,SamForward::Counters& counters) {
    const uint64_t limit=std::max<uint64_t>(256,uint64_t(n)*8);
    uint64_t probes=0;
    auto charge=[&](){return ++probes<=limit;};
    if(n<=0)return true;
    lengths[0]=0;routes[0]=-1;
    struct Level { int length,mask,base; };
    std::vector<Level> levels;
    size_t slots=0;
    // A few integer directories beat a general keyed map when the actual
    // alphabet fits. Refuse direct addressing above 2^20 slots.
    const uint64_t directory_limit=std::min<uint64_t>(UINT64_C(1)<<20,std::max<uint64_t>(256,uint64_t(n)*32));
    for(int len=1;len*bits<=20 && (UINT64_C(1)<<(len*bits))<=directory_limit;++len) {
        const int size=1<<(len*bits);
        levels.push_back({len,size-1,int(slots)});slots+=size;
    }
    const bool hashed=levels.empty();
    if(hashed) {
        slots=1;while(slots<size_t(n)*2)slots*=2;
    }
    std::vector<int> head(slots,-1),previous(n);
    // Only the hashed fallback needs the actual last-symbol check.
    auto slot=[&](uint64_t word){
        word^=word>>32;word*=UINT64_C(0xd6e8feb86659fd93);word^=word>>32;
        return size_t(word)&(slots-1);
    };
    uint64_t qword=0,kword=0;
    const uint64_t high_mask=bits==64?0:~((UINT64_C(1)<<bits)-1);
    for(int t=1;t<n;++t) {
        const int upper=lengths[t-1]+1;
        const uint64_t q=query[t],k=key[t-1];
        int length=0,end=-1;
        if(hashed) {
            const auto s=slot(k);previous[t-1]=head[s];head[s]=t-1;
            int best=0;
            ++counters.transition_probes;
            counters.hash_probe_steps+=2;
            if(head[slot(q)]>=0)++counters.endpoint_queries;
            for(int r=head[slot(q)];r>=0;r=previous[r]) {
                if(!charge())return false;
                ++counters.hash_probe_steps;
                if(key[r]!=q)continue;
                int h=1;
                while(h<=r && h<upper && key[r-h]==query[t-h]){if(!charge())return false;++h;}
                if(h>best){best=h;end=r;}
                if(best==upper)break;
            }
            length=best;
        } else {
            if((q|k)&high_mask)return false;
            qword=(qword<<bits)|q;kword=(kword<<bits)|k;
            const auto& top=levels.back();
            for(const auto& level:levels) {
                if(level.length>t)break;
                int& latest=head[level.base+(kword&level.mask)];
                if(level.length==top.length)previous[t-1]=latest;
                latest=t-1;
            }
            // Removing the last symbol of a causal match bounds its new
            // length by the preceding row's winner plus one. If a level
            // below top is reached, every longer bucket was empty.
            for(int h=std::min(top.length,upper);h>0;--h) {
                ++counters.transition_probes;
                const auto& level=levels[h-1];
                const int latest=head[level.base+(qword&level.mask)];
                if(latest<0)continue;
                length=h;end=latest;
                if(h==top.length && h<upper) {
                    ++counters.endpoint_queries;
                    for(int r=latest;r>=0;r=previous[r]) {
                        if(!charge())return false;
                        int j=h;
                        while(j<=r && j<upper && key[r-j]==query[t-j]) {
                            if(!charge())return false;
                            ++j;
                        }
                        if(j>length){length=j;end=r;}
                        if(length==upper)break;
                    }
                }
                break;
            }
        }
        lengths[t]=length;routes[t]=end;
    }
    return true;
}

inline SamForward::Counters training_routes(const uint64_t* q,const uint64_t* k,int n,int bits,
        int32_t* lengths,int32_t* routes,uint64_t seed) {
    SamForward::Counters counters;
    if(window_forward(q,k,n,bits,lengths,routes,counters))return counters;
    SamForward scan(q,k,n,seed);
    scan.run(lengths,routes);
    counters.transition_probes+=scan.counters.transition_probes;
    counters.hash_probe_steps+=scan.counters.hash_probe_steps;
    counters.endpoint_queries+=scan.counters.endpoint_queries;
    return counters;
}
} // namespace rosa_exact
