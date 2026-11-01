#pragma once
// Exact offline identity certificates for finished numerical corridors.
// Full tuple keys give collision-free dyadic substring ranks. Only two rank
// levels are live; each query is answered at floor(log2(length)), using two
// overlapping blocks. Expected build work O(T log T + M), space O(T + M).
#include "full_word_map.h"
#include <algorithm>
#include <cstdint>
#include <vector>
namespace rosa_exact {
class ExactPayloadIdentityBatch {
    std::vector<uint8_t> confirmed_;
public:
    struct Query {int id,baseline_start,payload_start,length;};
    // Answers use input-vector order; caller IDs may be sparse.
    bool confirmed(int query_index)const {
        return query_index>=0&&size_t(query_index)<confirmed_.size()&&confirmed_[query_index];
    }
    void build(const int*payload_class,const int32_t*routes,int n,
               const std::vector<Query>&queries,uint64_t seed) {
        confirmed_.clear();if(!n||queries.empty())return;
        int maximum_level=0;
        std::vector<uint8_t> logarithm(size_t(n)+1,0);
        for(int x=2;x<=n;++x)logarithm[x]=uint8_t(logarithm[x/2]+1);
        std::vector<int> level(queries.size());
        for(size_t i=0;i<queries.size();++i) {
            const int exponent=logarithm[queries[i].length];
            level[i]=exponent;maximum_level=std::max(maximum_level,exponent);
        }
        confirmed_.assign(queries.size(),0);
        std::vector<int> first(maximum_level+1,-1),next(queries.size());
        for(int i=int(queries.size())-1;i>=0;--i) {
            next[i]=first[level[i]];first[level[i]]=i;
        }
        const int total=2*n;
        std::vector<int> previous(total),current(total);
        for(int x=0;x<n;++x) {
            // Empty outputs have a separate symbol. The certificate is
            // sufficient even when emb0 equals a real signed payload.
            previous[x]=routes[x]<0?0:payload_class[routes[x]+1]+1;
            previous[n+x]=payload_class[x]+1;
        }
        for(int exponent=0;exponent<=maximum_level;++exponent) {
            const int length=1<<exponent;
            if(exponent) {
                const int half=length/2;
                FullWordMap tuples(size_t(total),seed+uint64_t(exponent));
                int classes=0;
                for(int x=0;x+length<=total;++x) {
                    const uint64_t key=(uint64_t(uint32_t(previous[x]))<<32)|uint32_t(previous[x+half]);
                    const auto value=tuples.insert(key,classes);
                    if(value.second)++classes;
                    current[x]=value.first;
                }
                current.swap(previous);
            }
            for(int i=first[exponent];i>=0;i=next[i]) {
                const auto&q=queries[i];const int tail=q.length-length;
                confirmed_[i]=uint8_t(previous[q.baseline_start]==previous[n+q.payload_start]&&
                    previous[q.baseline_start+tail]==previous[n+q.payload_start+tail]);
            }
        }
    }
};
} // namespace rosa_exact
