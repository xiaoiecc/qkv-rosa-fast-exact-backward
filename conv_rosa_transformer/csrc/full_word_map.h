#pragma once
// Full-key randomized placement. Values are insertion-order class IDs;
// collisions change probe work only, never symbol equality or class order.
#include <cstdint>
#include <vector>
#include <utility>

namespace rosa_exact {
class FullWordMap {
    std::vector<uint64_t> keys_;
    std::vector<int> values_;
    uint64_t seed_;
    size_t mask_;
    size_t count_=0;
    static uint64_t mix(uint64_t x) {
        x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
        x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
        return x^(x>>31);
    }
    void grow() {
        std::vector<uint64_t> old_keys;old_keys.swap(keys_);
        std::vector<int> old_values;old_values.swap(values_);
        const size_t capacity=old_values.size()*2;
        keys_.resize(capacity);values_.assign(capacity,-1);mask_=capacity-1;
        for(size_t at=0;at<old_values.size();++at)if(old_values[at]>=0) {
            size_t slot=size_t(mix(old_keys[at]^seed_))&mask_;
            while(values_[slot]>=0)slot=(slot+1)&mask_;
            keys_[slot]=old_keys[at];values_[slot]=old_values[at];
        }
    }
public:
    explicit FullWordMap(size_t maximum_keys,uint64_t seed):seed_(seed) {
        size_t capacity=2;
        // Grow with observed diversity; repeated payloads need O(U), not O(T),
        // dictionary storage. This is one load-factor policy for every input.
        const size_t initial=maximum_keys<8?maximum_keys:8;
        while(capacity<2*initial)capacity<<=1;
        keys_.resize(capacity);values_.assign(capacity,-1);mask_=capacity-1;
    }
    std::pair<int,bool> insert(uint64_t key,int next_id) {
        size_t slot=size_t(mix(key^seed_))&mask_;
        while(values_[slot]>=0 && keys_[slot]!=key)slot=(slot+1)&mask_;
        if(values_[slot]>=0)return {values_[slot],false};
        if(2*count_>=values_.size()) {
            grow();slot=size_t(mix(key^seed_))&mask_;
            while(values_[slot]>=0)slot=(slot+1)&mask_;
        }
        keys_[slot]=key;values_[slot]=next_id;++count_;return {next_id,true};
    }
    int find(uint64_t key) const {
        size_t slot=size_t(mix(key^seed_))&mask_;
        while(values_[slot]>=0 && keys_[slot]!=key)slot=(slot+1)&mask_;
        return values_[slot];
    }
};
} // namespace rosa_exact
