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
#include <map>
#include <vector>

namespace rosa_random {
class AdjacentPairCertificate {
    struct Entry {uint64_t left,right;int first,left_class,right_class;};
    std::vector<Entry> entries_;
    std::vector<int> table_;
    std::array<std::vector<int>,2> offsets_,members_;
    rosa_exact::FullWordMap left_,right_;
    const rosa_exact::FullWordMap* shared_symbols_=nullptr;
    uint64_t seed_;size_t mask_;
    std::map<std::pair<uint64_t,uint64_t>,int> overflow_;
    static constexpr int probe_limit_=16;
    static uint64_t mix(uint64_t x) {
        x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
        x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
        return x^(x>>31);
    }
    size_t hash(uint64_t a,uint64_t b)const {
        const uint64_t y=mix(b^(seed_+UINT64_C(0x9e3779b97f4a7c15)));
        return size_t(mix(a^seed_)^((y<<1)|(y>>63)))&mask_;
    }
    size_t slot(uint64_t a,uint64_t b,uint64_t*stats=nullptr)const {
        size_t at=hash(a,b);
        for(int probe=0;probe<probe_limit_;++probe,at=(at+1)&mask_) {
            const int id=table_[at];
            if(id<0)return at;
            if(stats)++stats[5];
            const auto&e=entries_[id];
            if(e.left==a&&e.right==b)return at;
        }
        return table_.size();
    }
    int entry(size_t at,uint64_t a,uint64_t b)const {
        if(at<table_.size())return table_[at];
        const auto found=overflow_.find({a,b});
        return found==overflow_.end()?-1:found->second;
    }
    void place(int id) {
        const auto&e=entries_[id];const size_t at=slot(e.left,e.right);
        if(at<table_.size())table_[at]=id;
        else overflow_.emplace(std::make_pair(e.left,e.right),id);
    }
    void grow() {
        table_.assign(table_.size()*2,-1);mask_=table_.size()-1;
        overflow_.clear();
        for(int id=0;id<int(entries_.size());++id)place(id);
    }
    int find(uint64_t a,uint64_t b,uint64_t*stats)const {
        ++stats[4];return entry(slot(a,b,stats),a,b);
    }
public:
    AdjacentPairCertificate(const uint64_t*k,int n,uint64_t seed,
        const int*known_classes=nullptr,int known_count=0,
        const rosa_exact::FullWordMap*known_symbols=nullptr)
        :left_(size_t(n),seed),right_(size_t(n),seed^UINT64_C(0xd6e8feb86659fd93)),
         shared_symbols_(known_symbols),seed_(seed),mask_(15) {
        table_.assign(16,-1);entries_.reserve(size_t(n));
        std::array<std::vector<int>,2> counts;
        if(shared_symbols_){counts[0].assign(known_count,0);counts[1].assign(known_count,0);}
        // K endpoint n-1 can never be causal for any output in this stream.
        for(int r=1;r<n-1;++r) {
            const uint64_t a=k[r-1],b=k[r];
            if(entry(slot(a,b),a,b)>=0)continue;
            if(entries_.size()*2>=table_.size()) {
                grow();
            }
            const auto l=shared_symbols_?std::make_pair(known_classes[r-1],false):left_.insert(a,int(counts[0].size()));
            const auto h=shared_symbols_?std::make_pair(known_classes[r],false):right_.insert(b,int(counts[1].size()));
            if(l.second)counts[0].push_back(0);
            if(h.second)counts[1].push_back(0);
            ++counts[0][l.first];++counts[1][h.first];
            entries_.push_back({a,b,r,l.first,h.first});place(int(entries_.size())-1);
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
            const int group=shared_symbols_?shared_symbols_->find(axis?b:a):(axis?right_.find(b):left_.find(a));++stats[2];
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
    uint64_t seed,uint64_t*stats,const int*known_classes=nullptr,
    int known_count=0,const rosa_exact::FullWordMap*known_symbols=nullptr) {
    std::fill(stats,stats+8,uint64_t(0));stats[0]=count;
    uint64_t direct_work=0;
    for(int i=0;i<count;++i) {
        const int t=active[i];
        if(t>=2&&(!lengths||lengths[t]<2))direct_work+=uint64_t(t-1);
    }
    std::vector<int32_t> result;result.reserve(count);
    if(!direct_work) {
        for(int i=0;i<count;++i)if(active[i]>=2&&lengths&&lengths[active[i]]>=2)result.push_back(active[i]);
    } else if(direct_work<=uint64_t(n)*2) {
        // Few short-prefix queries do not justify a whole-stream index.
        // The counted scan budget is linear for every input, not just for
        // data on which a mismatch happens to be found early.
        for(int i=0;i<count;++i) {
            const int t=active[i];if(t<2)continue;
            bool needed=lengths&&lengths[t]>=2;
            for(int r=1;!needed&&r<t;++r) {
                const uint64_t a=q[t-1]^k[r-1],b=q[t]^k[r];++stats[3];
                needed=(!a&&(!b||!(b&(b-1))))||(!b&&(!a||!(a&(a-1))));
            }
            if(needed)result.push_back(t);
        }
    } else {
        AdjacentPairCertificate index(k,n,seed,known_classes,known_count,known_symbols);stats[6]=index.distinct_pairs();stats[7]=1;
        for(int i=0;i<count;++i) {
            const int t=active[i];
            if(t>=2&&((lengths&&lengths[t]>=2)||index.needs_residual(q[t-1],q[t],t,d,stats)))result.push_back(t);
        }
    }
    stats[1]=result.size();return result;
}
} // namespace rosa_random
