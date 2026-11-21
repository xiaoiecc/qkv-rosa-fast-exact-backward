#pragma once
#include "../full_word_map.h"
#include "types.h"

namespace rosa_subq {
namespace suffix_frontier {

inline uint64_t low_mask(int width) {
    return width==64?UINT64_MAX:(UINT64_C(1)<<width)-1;
}

// Occurrences at one depth. Dense and sparse levels expose identical point
// and Hamming-neighbor queries. Representation is fixed before the sweep.
class PackedLevel {
    struct Group {int head=-1,count=0;};
    struct Band {
        rosa_exact::FullWordMap ids;
        std::vector<Group> groups;
        std::vector<int> next;
        explicit Band(int n):ids(n,47){}
    };
    int width_;
    uint64_t mask_,fixed_[2];
    rosa_exact::FullWordMap ids_;
    Band bands_[2];
    std::vector<uint64_t> words_;
    std::vector<int> direct_,head_,previous_;
public:
    PackedLevel(int n,int width):width_(width),mask_(low_mask(width)),ids_(n,13),
            bands_{Band(n),Band(n)},previous_(n) {
        fixed_[0]=low_mask(width/2);fixed_[1]=mask_^fixed_[0];
        if(width<31 && (UINT64_C(1)<<width)<=2*uint64_t(n))
            direct_.assign(size_t(1)<<width,-1);
    }
    uint64_t mask() const {return mask_;}
    int exact(uint64_t word) const {
        if(!direct_.empty())return direct_[word];
        int id=ids_.find(word);return id<0?-1:head_[id];
    }
    void insert(uint64_t word,int r) {
        if(!direct_.empty()) {
            previous_[r]=direct_[word];direct_[word]=r;return;
        }
        const auto id=ids_.insert(word,int(words_.size()));
        if(id.second) {
            head_.push_back(-1);words_.push_back(word);
            for(int j=0;j<2;++j) {
                auto& b=bands_[j];
                const auto group=b.ids.insert(word&fixed_[j],int(b.groups.size()));
                if(group.second)b.groups.emplace_back();
                auto& list=b.groups[group.first];
                b.next.push_back(list.head);list.head=id.first;++list.count;
            }
        }
        previous_[r]=head_[id.first];head_[id.first]=r;
    }
    template<class F>
    void endpoints(int r,int after,int count,F emit) const {
        while(r>after && count>0){emit(r);r=previous_[r];--count;}
    }
    template<class F>
    void neighbors(uint64_t word,F emit) const {
        auto toggle=[&](uint64_t bits) {
            for(;bits;bits&=bits-1) {
                const uint64_t bit=bits&(~bits+1);
                const int r=exact(word^bit);
                if(r>=0)emit(bit,r);
            }
        };
        if(!direct_.empty()){toggle(mask_);return;}
        for(int j=0;j<2;++j) {
            const auto& b=bands_[j];const int id=b.ids.find(word&fixed_[j]);
            if(id<0)continue;
            const auto& group=b.groups[id];
            const int other_width=j==0?width_-width_/2:width_/2;
            if(group.count>other_width)toggle(fixed_[1-j]);
            else for(int i=group.head;i>=0;i=b.next[i]) {
                const uint64_t diff=word^words_[i];
                if(diff && !(diff&(diff-1)))emit(diff,head_[i]);
            }
        }
    }
    size_t bytes() const {
        size_t z=ids_.workspace_bytes()+words_.capacity()*sizeof(uint64_t)
            +(direct_.capacity()+head_.capacity()+previous_.capacity())*sizeof(int);
        for(const auto& b:bands_)z+=b.ids.workspace_bytes()+b.next.capacity()*sizeof(int)
            +b.groups.capacity()*sizeof(Group);
        return z;
    }
};
} // namespace suffix_frontier
} // namespace rosa_subq
