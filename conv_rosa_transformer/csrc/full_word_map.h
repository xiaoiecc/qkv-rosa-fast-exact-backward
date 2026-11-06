#pragma once
// Full-key randomized placement. Values are insertion-order class IDs;
// collisions change probe work only, never symbol equality or class order.
#include <cstdint>
#include <map>
#include <vector>
#include <utility>

namespace rosa_exact {
class FullWordMap {
    std::vector<uint64_t> keys_;
    std::vector<int> values_;
    uint64_t seed_;
    size_t mask_;
    size_t count_=0;
    static constexpr int probe_limit_=16;
    std::map<uint64_t,int> overflow_;
    static uint64_t mix(uint64_t x) {
        x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
        x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
        return x^(x>>31);
    }
    size_t slot(uint64_t key) const {
        size_t at=size_t(mix(key^seed_))&mask_;
        for(int probe=0;probe<probe_limit_;++probe,at=(at+1)&mask_)
            if(values_[at]<0||keys_[at]==key)return at;
        return values_.size();
    }
    void place(uint64_t key,int value) {
        const size_t at=slot(key);
        if(at==values_.size())overflow_.emplace(key,value);
        else {keys_[at]=key;values_[at]=value;}
    }
    void grow() {
        std::vector<uint64_t> old_keys;old_keys.swap(keys_);
        std::vector<int> old_values;old_values.swap(values_);
        std::map<uint64_t,int> old_overflow;old_overflow.swap(overflow_);
        const size_t capacity=old_values.size()*2;
        keys_.resize(capacity);values_.assign(capacity,-1);mask_=capacity-1;
        for(size_t at=0;at<old_values.size();++at)if(old_values[at]>=0) {
            place(old_keys[at],old_values[at]);
        }
        for(const auto&entry:old_overflow)place(entry.first,entry.second);
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
        size_t at=slot(key);
        if(at<values_.size()) {
            if(values_[at]>=0)return {values_[at],false};
        } else {
            const auto found=overflow_.find(key);
            if(found!=overflow_.end())return {found->second,false};
        }
        if(4*count_>=values_.size()) {
            grow();at=slot(key);
        }
        if(at==values_.size())overflow_.emplace(key,next_id);
        else {keys_[at]=key;values_[at]=next_id;}
        ++count_;return {next_id,true};
    }
    int find(uint64_t key) const {
        const size_t at=slot(key);
        if(at<values_.size())return values_[at];
        const auto found=overflow_.find(key);
        return found==overflow_.end()?-1:found->second;
    }
    size_t workspace_bytes()const {
        // Retained element storage, excluding allocator bookkeeping and the
        // implementation-specific balanced-tree node links.
        return keys_.capacity()*sizeof(uint64_t)+values_.capacity()*sizeof(int)
            +overflow_.size()*sizeof(std::pair<const uint64_t,int>);
    }
};
} // namespace rosa_exact
