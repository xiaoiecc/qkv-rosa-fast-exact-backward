#pragma once
// Exact Hamming-one neighbor discovery for distinct packed words.
// One full-key join shared by the exact and sampled one-symbol control
// variates. Randomization changes work and bucket placement, never membership.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>
#include "full_word_map.h"

namespace rosa_exact {
struct HammingJoinCounters {
    uint64_t enabled=0, used=0, direct_queries=0, bucket_queries=0;
    uint64_t bucket_members=0, bucket_build_entries=0, empty_buckets=0, verified_neighbors=0;
    uint64_t blocks=0;
};

struct HammingDomain {
    size_t size=0;
    uint64_t reference=0, varying=0;
};
namespace hamming_join_detail {
    struct HammingRoot {uint64_t word;int target;};
    static uint64_t mix(uint64_t x) {
        x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
        x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
        return x^(x>>31);
    }
    struct RootBuckets {
        struct Group { uint64_t word; int head=-1,count=0; };
        uint64_t mask=0;
        FullWordMap classes;
        std::vector<int> next;
        std::vector<Group> groups;
        RootBuckets(uint64_t bits,const std::vector<HammingRoot>& roots,uint64_t seed)
            :mask(bits),classes(roots.size(),seed),next(roots.size(),-1) {
            groups.reserve(roots.size());
            for(size_t i=0;i<roots.size();++i) {
                const uint64_t word=roots[i].word&mask;
                const auto inserted=classes.insert(word,int(groups.size()));
                const int group=inserted.first;
                if(inserted.second)groups.push_back({word,-1,0});
                next[i]=groups[group].head;groups[group].head=int(i);++groups[group].count;
            }
        }
        const Group* find(uint64_t word,uint64_t) const {
            const int id=classes.find(word&mask);
            return id<0?nullptr:&groups[id];
        }
    };
    static int singleton_bit(uint64_t word) {
        // Called only for a nonzero power of two.
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_ctzll(word);
#else
        int bit=0;
        if(!(word&UINT64_C(0xffffffff))){word>>=32;bit+=32;}
        if(!(word&UINT64_C(0xffff))){word>>=16;bit+=16;}
        if(!(word&UINT64_C(0xff))){word>>=8;bit+=8;}
        if(!(word&UINT64_C(0xf))){word>>=4;bit+=4;}
        if(!(word&UINT64_C(3))){word>>=2;bit+=2;}
        if(!(word&UINT64_C(1)))++bit;
        return bit;
#endif
    }

} // namespace hamming_join_detail

// `queries` contains distinct full words; ordinary_targets[i] is the exact K
// class/state for queries[i], or -1. `lookup(word)` must compare full keys.
// `enumerate(emit)` emits each distinct K word and its target exactly once.
// Every query invokes result(id, mask, targets), with targets[bit] defined
// exactly for bits in mask. The callback may apply further causal pruning.
// K domain.varying is OR(word XOR reference) over that same distinct domain.
template<class Lookup,class Enumerate,class Result>
void exact_hamming_one_join(const std::vector<uint64_t>& symbols,
                           const std::vector<int>& ordinary_targets,
                           HammingDomain domain,uint64_t allowed,
                           uint64_t seed,bool randomized,
                           Lookup&& lookup,Enumerate&& enumerate,Result&& result,
                           HammingJoinCounters& counters) {
    using namespace hamming_join_detail;
    if(ordinary_targets.size()!=symbols.size())throw std::invalid_argument("Hamming join query classes disagree");
    counters={};counters.enabled=randomized;
    const uint64_t varying_allowed=domain.varying&allowed;
    std::vector<int> bits;
    for(uint64_t remaining=varying_allowed;remaining;remaining&=remaining-1)
        bits.push_back(singleton_bit(remaining&(~remaining+1)));
    std::vector<uint64_t> possible(symbols.size(),0);
    size_t varying_queries=0,exact_members=0;
    for(size_t id=0;id<symbols.size();++id) {
        const uint64_t fixed_difference=(symbols[id]^domain.reference)&~domain.varying;
        if(!fixed_difference) {
            possible[id]=varying_allowed;
            varying_queries+=bool(varying_allowed);
            exact_members+=ordinary_targets[id]>=0;
        } else if(!(fixed_difference&(fixed_difference-1))) {
            possible[id]=fixed_difference&allowed;
        }
    }
    auto direct=[&](size_t id,int bit)->int {
        ++counters.direct_queries;
        return lookup(symbols[id]^(uint64_t(1)<<bit));
    };
    const size_t direct_work=varying_queries*bits.size();
    int blocks=0;
    if(randomized && bits.size()>=2 && direct_work>256 &&
       direct_work>2*(varying_queries+domain.size)+bits.size()) {
        int varying_count=0;
        for(uint64_t remaining=domain.varying;remaining;remaining&=remaining-1)++varying_count;
        const double self_rate=varying_queries?double(exact_members)/varying_queries:0.0;
        double best=0.85*double(direct_work);
        for(int count: {2,4}) {
            if(count>int(bits.size()))continue;
            double estimate=256.0+2.0*bits.size()+2.0*count*domain.size+double(count)*varying_queries;
            for(int block=0;block<count;++block) {
                const size_t width=(block+1)*bits.size()/count-block*bits.size()/count;
                const int entropy_bits=varying_count-int(width);
                const double occupancy=std::max(self_rate,std::ldexp(double(domain.size),-entropy_bits));
                estimate+=double(varying_queries)*(occupancy>width?double(width):0.5*occupancy);
            }
            if(estimate<best){best=estimate;blocks=count;}
        }
    }
    counters.used=blocks!=0;counters.blocks=blocks;
    std::vector<HammingRoot> roots;
    std::vector<RootBuckets> indexes;
    std::vector<uint64_t> bucket_seeds,omitted_masks;
    std::vector<size_t> omitted_widths;
    if(blocks) {
        roots.reserve(domain.size);
        enumerate([&](uint64_t word,int target){roots.push_back({word,target});});
        std::mt19937_64 random(seed^UINT64_C(0x72da41f6c385eb09));
        for(int j=int(bits.size())-1;j>0;--j) {
            std::uniform_int_distribution<int> pick(0,j);std::swap(bits[j],bits[pick(random)]);
        }
        indexes.reserve(blocks);bucket_seeds.reserve(blocks);omitted_masks.reserve(blocks);omitted_widths.reserve(blocks);
        for(int block=0;block<blocks;++block) {
            uint64_t omitted=0;
            for(size_t j=block*bits.size()/blocks;j<(block+1)*bits.size()/blocks;++j)
                omitted|=uint64_t(1)<<bits[j];
            const uint64_t bucket_seed=mix(seed+uint64_t(block)+UINT64_C(0xf16a8bd029c735e4));
            indexes.emplace_back(~omitted,roots,bucket_seed);
            omitted_masks.push_back(omitted);bucket_seeds.push_back(bucket_seed);
            omitted_widths.push_back((block+1)*bits.size()/blocks-block*bits.size()/blocks);
            counters.bucket_build_entries+=roots.size();
        }
    }
    for(size_t id=0;id<symbols.size();++id) {
        int targets[64];uint64_t available=0;
        auto install=[&](int bit,int target) {
            if(target<0)return;
            ++counters.verified_neighbors;
            available|=uint64_t(1)<<bit;targets[bit]=target;
        };
        if(!blocks && possible[id]==varying_allowed) {
            for(int bit:bits)install(bit,direct(id,bit));
        } else if(!blocks || possible[id]!=varying_allowed) {
            for(uint64_t remaining=possible[id];remaining;remaining&=remaining-1) {
                const int bit=singleton_bit(remaining&(~remaining+1));install(bit,direct(id,bit));
            }
        } else for(size_t block=0;block<indexes.size();++block) {
            const auto&buckets=indexes[block];const uint64_t omitted=omitted_masks[block];
            const auto*group=buckets.find(symbols[id],bucket_seeds[block]);
            ++counters.bucket_queries;
            if(!group){++counters.empty_buckets;continue;}
            const size_t width=omitted_widths[block];
            if(size_t(group->count)>width) {
                for(uint64_t remaining=omitted;remaining;remaining&=remaining-1) {
                    const int bit=singleton_bit(remaining&(~remaining+1));install(bit,direct(id,bit));
                }
            } else {
                for(int member=group->head;member>=0;member=buckets.next[member]) {
                    ++counters.bucket_members;
                    const auto&root=roots[member];
                    const uint64_t diff=root.word^symbols[id];
                    if(!diff || (diff&(diff-1)))continue;
                    install(singleton_bit(diff),root.target);
                }
            }
        }
        result(id,available,targets);
    }
}
} // namespace rosa_exact
