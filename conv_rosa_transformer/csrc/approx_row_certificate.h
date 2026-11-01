#pragma once
// Exact zero-residual certificate for a one-symbol control variate.
// If no causal K adjacent pair is within total bit distance one of Q[t-1:t],
// no original or single-bit-edited suffix can have length >= 2. All complete
// single-bit row routes then equal the one-symbol routes, for every payload.
// Full keys are always checked. Hash randomness can only change probe counts.
#include "full_word_map.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace rosa_random {
class AdjacentPairCertificate {
    struct Entry {uint64_t left,right;int first,left_class,right_class;};
    std::vector<Entry> entries_;
    std::vector<int> table_;
    std::array<std::vector<int>,2> offsets_,members_;
    rosa_exact::FullWordMap left_,right_;
    uint64_t seed_;size_t mask_;
    static uint64_t mix(uint64_t x) {
        x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
        x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
        return x^(x>>31);
    }
    size_t hash(uint64_t a,uint64_t b)const {
        const uint64_t y=mix(b^(seed_+UINT64_C(0x9e3779b97f4a7c15)));
        return size_t(mix(a^seed_)^((y<<1)|(y>>63)))&mask_;
    }
    void grow() {
        table_.assign(table_.size()*2,-1);mask_=table_.size()-1;
        for(int id=0;id<int(entries_.size());++id) {
            const auto&e=entries_[id];size_t at=hash(e.left,e.right);
            while(table_[at]>=0)at=(at+1)&mask_;
            table_[at]=id;
        }
    }
    int find(uint64_t a,uint64_t b,uint64_t*stats)const {
        size_t at=hash(a,b);++stats[4];
        while(table_[at]>=0) {
            ++stats[5];const auto&e=entries_[table_[at]];
            if(e.left==a&&e.right==b)return table_[at];
            at=(at+1)&mask_;
        }
        return -1;
    }
public:
    AdjacentPairCertificate(const uint64_t*k,int n,uint64_t seed)
        :left_(size_t(n),seed),right_(size_t(n),seed^UINT64_C(0xd6e8feb86659fd93)),
         seed_(seed),mask_(15) {
        table_.assign(16,-1);entries_.reserve(size_t(n));
        std::array<std::vector<int>,2> counts;
        // K endpoint n-1 can never be causal for any output in this stream.
        for(int r=1;r<n-1;++r) {
            const uint64_t a=k[r-1],b=k[r];size_t at=hash(a,b);
            while(table_[at]>=0) {
                const auto&e=entries_[table_[at]];
                if(e.left==a&&e.right==b)break;
                at=(at+1)&mask_;
            }
            if(table_[at]>=0)continue;
            if(entries_.size()*2>=table_.size()) {
                grow();at=hash(a,b);while(table_[at]>=0)at=(at+1)&mask_;
            }
            const auto l=left_.insert(a,int(counts[0].size()));
            const auto h=right_.insert(b,int(counts[1].size()));
            if(l.second)counts[0].push_back(0);
            if(h.second)counts[1].push_back(0);
            ++counts[0][l.first];++counts[1][h.first];
            table_[at]=int(entries_.size());entries_.push_back({a,b,r,l.first,h.first});
        }
        for(int axis=0;axis<2;++axis) {
            auto&offset=offsets_[axis];offset.resize(counts[axis].size()+1,0);
            for(size_t i=0;i<counts[axis].size();++i)offset[i+1]=offset[i]+counts[axis][i];
            members_[axis].resize(entries_.size());std::vector<int> cursor=offset;
            for(int id=0;id<int(entries_.size());++id) {
                const auto&e=entries_[id];const int group=axis?e.right_class:e.left_class;
                members_[axis][cursor[group]++]=id;
            }
        }
    }
    size_t distinct_pairs()const{return entries_.size();}
    bool needs_residual(uint64_t a,uint64_t b,int t,int d,uint64_t*stats)const {
        if(t<=1)return false;
        for(int axis=0;axis<2;++axis) {
            const int group=axis?right_.find(b):left_.find(a);++stats[2];
            if(group<0)continue;
            const auto&members=members_[axis];const auto&offset=offsets_[axis];
            const int start=offset[group],stop=offset[group+1];
            // Members are in increasing earliest-endpoint order. Locate only
            // the causal prefix, then cap work by direct exact bit probes.
            const auto end=std::lower_bound(members.begin()+start,members.begin()+stop,t,
                [&](int id,int time){return entries_[id].first<time;});
            const int causal=int(end-(members.begin()+start));
            if(causal<=d+1) {
                for(int at=start;at<start+causal;++at) {
                    ++stats[3];const auto&e=entries_[members[at]];
                    const uint64_t difference=axis?(a^e.left):(b^e.right);
                    if(!difference||!(difference&(difference-1)))return true;
                }
            } else {
                int id=find(a,b,stats);
                if(id>=0&&entries_[id].first<t)return true;
                for(int bit=0;bit<d;++bit) {
                    const uint64_t flip=uint64_t(1)<<bit;
                    id=find(axis?a^flip:a,axis?b:b^flip,stats);
                    if(id>=0&&entries_[id].first<t)return true;
                }
            }
        }
        return false;
    }
};

// stats: active rows, uncertified rows, bucket queries, verified members,
// full pair probes, actual occupied hash probes, unique K pairs, indexes built.
inline std::vector<int32_t> uncertified_rows(const uint64_t*q,const uint64_t*k,
    int n,int d,const int32_t*active,int count,const int32_t*lengths,
    uint64_t seed,uint64_t*stats) {
    std::fill(stats,stats+8,uint64_t(0));stats[0]=count;
    bool need_index=false;
    for(int i=0;i<count;++i) {
        const int t=active[i];
        if(t>=2&&(!lengths||lengths[t]<2)){need_index=true;break;}
    }
    std::vector<int32_t> result;result.reserve(count);
    if(!need_index) {
        for(int i=0;i<count;++i)if(active[i]>=2&&lengths&&lengths[active[i]]>=2)result.push_back(active[i]);
    } else {
        AdjacentPairCertificate index(k,n,seed);stats[6]=index.distinct_pairs();stats[7]=1;
        for(int i=0;i<count;++i) {
            const int t=active[i];
            if(t>=2&&((lengths&&lengths[t]>=2)||index.needs_residual(q[t-1],q[t],t,d,stats)))result.push_back(t);
        }
    }
    stats[1]=result.size();return result;
}
} // namespace rosa_random
