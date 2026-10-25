#pragma once
// One static causal-band proposal algorithm. Randomness selects candidates;
// full words are checked before a candidate is admitted. No payload/loss input.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rosa_exact {

inline uint64_t pair_mix64(uint64_t x) {
    x+=UINT64_C(0x9e3779b97f4a7c15);
    x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
    return x^(x>>31);
}
inline int pair_popcount(uint64_t x) {
    x=x-((x>>1)&UINT64_C(0x5555555555555555));
    x=(x&UINT64_C(0x3333333333333333))+((x>>2)&UINT64_C(0x3333333333333333));
    x=(x+(x>>4))&UINT64_C(0x0f0f0f0f0f0f0f0f);
    return int((x*UINT64_C(0x0101010101010101))>>56);
}
inline int pair_bit_index(uint64_t low) {
    int bit=0;
    if(low>>32){bit+=32;low>>=32;}
    if(low>>16){bit+=16;low>>=16;}
    if(low>>8){bit+=8;low>>=8;}
    if(low>>4){bit+=4;low>>=4;}
    if(low>>2){bit+=2;low>>=2;}
    if(low>>1)++bit;
    return bit;
}

struct PairCandidate {
    uint64_t priority,a,b,flat_a,flat_b;
    bool operator<(const PairCandidate&other) const {
        return std::tie(priority,flat_a,flat_b)<std::tie(other.priority,other.flat_a,other.flat_b);
    }
};
struct PairBandEntry {uint64_t key;int first,id;};

inline int propose_pair_candidates(
    int n,int d,const uint64_t*q,const uint64_t*k,const int32_t*rows,int row_count,
    const uint8_t*q_allowed,const uint8_t*k_allowed,int max_pairs,uint64_t seed,
    int bucket_cap,int occurrence_cap,int scan,int candidate_cap,
    uint64_t*pairs,uint64_t*trace,uint64_t*counts,double*timings) {
    using Clock=std::chrono::steady_clock;
    const auto started=Clock::now();
    std::fill(counts,counts+12,uint64_t(0));timings[0]=timings[1]=0;
    counts[7]=row_count;
    if(!n || !max_pairs)return 0;
    std::mt19937_64 random(seed);
    std::vector<int> bits(d);std::iota(bits.begin(),bits.end(),0);
    for(int i=d-1;i>0;--i) {
        std::uniform_int_distribution<int> choose(0,i);
        std::swap(bits[i],bits[choose(random)]);
    }
    std::array<uint64_t,3> masks{0,0,0};
    for(int i=0;i<d;++i)masks[i%3]|=uint64_t(1)<<bits[i];
    std::vector<int> occurrences(n);std::iota(occurrences.begin(),occurrences.end(),0);
    std::sort(occurrences.begin(),occurrences.end(),[&](int a,int b){
        return k[a]<k[b] || (k[a]==k[b] && a<b);
    });
    std::vector<uint64_t> words;
    std::vector<int> first,offsets;
    words.reserve(n);first.reserve(n);offsets.reserve(n+1);
    for(int p=0;p<n;++p) {
        const int t=occurrences[p];
        if(words.empty() || words.back()!=k[t]) {
            words.push_back(k[t]);first.push_back(t);offsets.push_back(p);
        }
    }
    offsets.push_back(n);
    const int unique=int(words.size());
    std::array<std::vector<PairBandEntry>,3> bands;
    for(int band=0;band<3;++band) {
        auto&entries=bands[band];entries.reserve(unique);
        for(int id=0;id<unique;++id)entries.push_back({words[id]&masks[band],first[id],id});
        std::sort(entries.begin(),entries.end(),[](const auto&a,const auto&b){
            return std::tie(a.key,a.first)<std::tie(b.key,b.first);
        });
    }
    counts[6]=n-1;counts[8]=unique;
    timings[0]=std::chrono::duration<double>(Clock::now()-started).count();
    const uint64_t nd=uint64_t(n)*d;
    std::priority_queue<PairCandidate> heap;
    std::set<std::pair<uint64_t,uint64_t>> retained;
    auto offer=[&](uint64_t a,uint64_t b) {
        if(b<a)std::swap(a,b);
        ++counts[3];
        const std::pair<uint64_t,uint64_t> pair{a,b};
        if(retained.find(pair)!=retained.end())return;
        const uint64_t fa=a<nd ? a+nd : a-nd;
        const uint64_t fb=b<nd ? b+nd : b-nd;
        const uint64_t priority=pair_mix64(pair_mix64(seed^pair_mix64(fa))^pair_mix64(fb));
        const PairCandidate candidate{priority,a,b,fa,fb};
        if(int(heap.size())==candidate_cap) {
            if(!(candidate<heap.top()))return;
            retained.erase({heap.top().a,heap.top().b});heap.pop();
        }
        retained.insert(pair);heap.push(candidate);
    };
    // Floyd's algorithm samples distinct offsets uniformly without scanning
    // the whole bucket. The hash set stores full integers, not fingerprints.
    std::unordered_set<int> sample_set;
    std::vector<int> picks;
    auto sample=[&](int population,int cap) {
        picks.clear();
        if(population<=cap) {
            picks.resize(population);std::iota(picks.begin(),picks.end(),0);
        } else {
            sample_set.clear();sample_set.reserve(size_t(cap)*2);
            for(int j=population-cap;j<population;++j) {
                std::uniform_int_distribution<int> choose(0,j);
                const int draw=choose(random);
                const int value=sample_set.find(draw)==sample_set.end() ? draw : j;
                sample_set.insert(value);picks.push_back(value);
            }
        }
    };
    std::vector<int> inspected,endpoints;
    for(int row=0;row<row_count;++row) {
        const int t=rows[row];inspected.clear();
        for(int band=0;band<3;++band) {
            ++counts[0];const auto&entries=bands[band];const uint64_t key=q[t]&masks[band];
            const auto begin=std::lower_bound(entries.begin(),entries.end(),key,
                [](const PairBandEntry&a,uint64_t b){return a.key<b;});
            const auto end=std::lower_bound(begin,entries.end(),std::pair<uint64_t,int>{key,t},
                [](const PairBandEntry&a,const std::pair<uint64_t,int>&b){return std::tie(a.key,a.first)<std::tie(b.first,b.second);});
            const int population=int(end-begin);sample(population,bucket_cap);
            if(population>bucket_cap)++counts[10];
            counts[9]+=picks.size();
            for(const int pick:picks)inspected.push_back(begin[pick].id);
        }
        std::sort(inspected.begin(),inspected.end());
        inspected.erase(std::unique(inspected.begin(),inspected.end()),inspected.end());
        for(const int id:inspected) {
            if(pair_popcount(q[t]^words[id])>2)continue;
            const auto start=occurrences.begin()+offsets[id],stop=occurrences.begin()+offsets[id+1];
            const int causal=int(std::lower_bound(start,stop,t)-start);
            if(!causal)continue; // Also guaranteed by first<t in the band join.
            sample(causal,occurrence_cap);
            if(causal>occurrence_cap)++counts[11];
            endpoints.clear();for(const int pick:picks)endpoints.push_back(start[pick]);
            endpoints.push_back(start[causal-1]);
            std::sort(endpoints.begin(),endpoints.end(),std::greater<int>());
            endpoints.erase(std::unique(endpoints.begin(),endpoints.end()),endpoints.end());
            for(const int r:endpoints) {
                ++counts[1];int mismatch_count=0;
                uint64_t choices[2][2];int choice_count[2]={0,0};
                for(int offset=0;offset<std::min(scan,r+1);++offset) {
                    const int u=t-offset,s=r-offset;uint64_t difference=q[u]^k[s];++counts[2];
                    if(mismatch_count+pair_popcount(difference)>2)break;
                    while(difference) {
                        const uint64_t low=difference&(~difference+1);
                        const int bit=pair_bit_index(low);
                        if(q_allowed[u])choices[mismatch_count][choice_count[mismatch_count]++]=nd+uint64_t(u)*d+bit;
                        if(k_allowed[s])choices[mismatch_count][choice_count[mismatch_count]++]=uint64_t(s)*d+bit;
                        ++mismatch_count;difference^=low;
                    }
                    if(mismatch_count==2) {
                        for(int a=0;a<choice_count[0];++a)for(int b=0;b<choice_count[1];++b)
                            offer(choices[0][a],choices[1][b]);
                        break;
                    }
                }
            }
        }
    }
    counts[4]=retained.size();
    if(trace) {
        size_t row=0;for(const auto&pair:retained){trace[2*row]=pair.first;trace[2*row+1]=pair.second;++row;}
    }
    std::vector<PairCandidate> ordered;ordered.reserve(heap.size());
    while(!heap.empty()){ordered.push_back(heap.top());heap.pop();}
    std::reverse(ordered.begin(),ordered.end());
    std::unordered_set<uint64_t> used;used.reserve(size_t(max_pairs)*4);
    int selected=0;
    for(const auto&candidate:ordered) {
        if(used.find(candidate.a)!=used.end() || used.find(candidate.b)!=used.end())continue;
        pairs[2*selected]=candidate.a;pairs[2*selected+1]=candidate.b;
        used.insert(candidate.a);used.insert(candidate.b);
        if(++selected==max_pairs)break;
    }
    counts[5]=selected;timings[1]=std::chrono::duration<double>(Clock::now()-started).count();
    return 0;
}
} // namespace rosa_exact
