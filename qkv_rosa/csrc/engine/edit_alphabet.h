#pragma once
#include "../approx_onegram.h"
#include "types.h"
#include <array>

namespace rosa_subq {
namespace interval_credit {

// Full-word dictionary with one slot per observed class in the direct case.
// The hashed table keeps an explicit key and a bounded load factor.
class WordTable {
    std::vector<int> ids_;
    std::vector<uint64_t> keys_;
    bool direct_;
    static uint64_t hash(uint64_t w) {
        w^=w>>32;w*=UINT64_C(0xd6e8feb86659fd93);w^=w>>32;return w;
    }
    size_t slot(uint64_t word)const {
        if(direct_)return size_t(word);
        const size_t mask=ids_.size()-1;
        size_t i=hash(word)&mask;
        while(ids_[i]>=0 && keys_[i]!=word)i=(i+1)&mask;
        return i;
    }
public:
    WordTable(int n,uint64_t largest) {
        direct_=largest<std::min<uint64_t>(UINT64_C(1)<<20,std::max<uint64_t>(256,uint64_t(n)*32));
        size_t cap=1;
        if(direct_)cap=size_t(largest)+1;
        else {while(cap<size_t(n)*2)cap*=2;keys_.resize(cap);}
        ids_.assign(cap,-1);
    }
    int find(uint64_t w)const{return direct_ && w>=ids_.size()?-1:ids_[slot(w)];}
    int intern(uint64_t w,std::vector<uint64_t>& words) {
        size_t i=slot(w);
        if(ids_[i]<0) {
            ids_[i]=int(words.size());words.push_back(w);
            if(!direct_)keys_[i]=w;
        }
        return ids_[i];
    }
    bool direct()const{return direct_;}
    size_t bytes()const{return ids_.capacity()*sizeof(int)+keys_.capacity()*sizeof(uint64_t);}
};

// Every causal equal or one-bit pair shares at least one projection. Verify
// light postings with full words. In a heavy bucket retain its participating
// time prefix conservatively, with a single timestamp (including aliases).
// Both cases cost O(1) work per time/band. No candidate graph is built here.
inline std::vector<int> live_coordinates(const uint64_t* q,const uint64_t* k,int n,int bits,
                                         const std::vector<Route>& base,Stats& stats) {
    // With a small physical alphabet, constructing its exact incidence graph
    // is already cheaper than building projection postings. Keep all rows.
    if(bits<=8)return {};
    struct Bucket {int head=-1,count=0,through=-1;};
    const int half=bits/2;
    const uint64_t low=(UINT64_C(1)<<half)-1;
    const uint64_t allowed=bits==64?UINT64_MAX:((UINT64_C(1)<<bits)-1);
    const uint64_t omitted[]={low,allowed^low};
    size_t slots=1;while(slots<size_t(n)*4)slots*=2;
    const size_t mask=slots-1;
    auto slot=[&](uint64_t word){word^=word>>32;word*=UINT64_C(0xd6e8feb86659fd93);return (word^(word>>32))&mask;};
    std::vector<uint8_t> live(n);
    std::vector<Bucket> buckets(slots);
    std::vector<int> previous(n);
    for(int band=0;band<2;++band) {
        if(band)std::fill(buckets.begin(),buckets.end(),Bucket{});
        const uint64_t fixed=~omitted[band];
        for(int t=1;t<n;++t) {
            auto& added=buckets[slot(k[t-1]&fixed)];
            previous[t-1]=added.head;added.head=t-1;++added.count;
            auto& found=buckets[slot(q[t]&fixed)];
            if(found.count>8){live[t]=1;found.through=t-1;continue;}
            for(int r=found.head;r>=0;r=previous[r]) {
                const uint64_t diff=q[t]^k[r];
                if(!(diff&~allowed) && !(diff&(diff-1)))live[t]=live[r]=1;
            }
        }
        for(int r=0;r+1<n;++r)if(buckets[slot(k[r]&fixed)].through>=r)live[r]=1;
    }
    std::vector<int> keep;bool gap=false;
    for(int t=0;t<n;++t) {
        bool inert=!live[t] && !base[t].length;
        if(!inert || !gap)keep.push_back(t);
        gap=inert;
    }
    stats.peak_workspace_bytes=std::max<uint64_t>(stats.peak_workspace_bytes,buckets.capacity()*sizeof(Bucket)
        +(previous.capacity()+keep.capacity())*sizeof(int)+live.capacity());
    return keep;
}

// Class-level one-edit incidence. Two complementary projections suffice:
// the changed bit belongs to exactly one omitted half. Dense buckets are
// queried by bit, so duplicates or projection collisions cannot expand a
// class into a quadratic number of comparisons.
struct EditAlphabet {
    int n,d,nq,nk;
    const uint64_t* q;const uint64_t* k;
    WordTable kmap;
    std::vector<int> qclass,kclass,qcount,kcount,qsame,ksame,first_neighbor,birth_prefix;
    rosa_random::GramGroups qneighbors,kneighbors;
    int births=0;
    EditAlphabet(const uint64_t* query,const uint64_t* key,int size,int bits,uint64_t largest)
        :n(size),d(bits),q(query),k(key),kmap(n,largest),qclass(n),kclass(n),birth_prefix(n+1) {
        WordTable qmap(n,largest);
        std::vector<uint64_t> qwords,kwords;
        for(int i=0;i<n;++i){qclass[i]=qmap.intern(q[i],qwords);kclass[i]=kmap.intern(k[i],kwords);}
        nq=int(qwords.size());nk=int(kwords.size());
        qsame.resize(nq);ksame.assign(nk,-1);first_neighbor.assign(nq,n);
        qcount.resize(nq);kcount.resize(nk);
        std::vector<int> first(nk,n);
        for(int i=0;i<n;++i) {
            ++qcount[qclass[i]];++kcount[kclass[i]];
            first[kclass[i]]=std::min(first[kclass[i]],i);
        }
        qneighbors.offsets.resize(nq+1);kneighbors.offsets.resize(nk+1);
        const int half=d/2;
        const uint64_t allowed=d==64?UINT64_MAX:((UINT64_C(1)<<d)-1);
        const std::array<uint64_t,2> omitted={(UINT64_C(1)<<half)-1,allowed^((UINT64_C(1)<<half)-1)};
        struct Bucket {int head=-1,count=0;};
        std::vector<Bucket> bands[2];std::vector<int> next[2];
        size_t mask=1;while(mask<size_t(nk)*2)mask*=2;--mask;
        auto slot=[&](uint64_t word){word^=word>>32;word*=UINT64_C(0xd6e8feb86659fd93);return (word^(word>>32))&mask;};
        if(!kmap.direct())for(int band=0;band<2;++band) {
            bands[band].resize(mask+1);next[band].resize(nk);
            for(int id=0;id<nk;++id) {
                auto& b=bands[band][slot(kwords[id]&~omitted[band])];
                next[band][id]=b.head;b.head=id;++b.count;
            }
        }
        for(int id=0;id<nq;++id) {
            uint64_t word=qwords[id];
            int same=kmap.find(word);qsame[id]=same;if(same>=0)ksame[same]=id;
            auto add=[&](int bit,int other) {
                qneighbors.edges.push_back({bit,other});++kneighbors.offsets[other+1];
                first_neighbor[id]=std::min(first_neighbor[id],first[other]);
            };
            for(int band=0;band<2;++band) {
                uint64_t remaining=omitted[band];
                if(!remaining)continue;
                if(!kmap.direct()) {
                    const auto b=bands[band][slot(word&~remaining)];
                    if(b.count<=(band?d-half:half)) {
                        for(int i=b.head;i>=0;i=next[band][i]) {
                            uint64_t diff=word^kwords[i];
                            if((diff&remaining) && !(diff&(diff-1)))add(rosa_random::row_bit(diff),i);
                        }
                        continue;
                    }
                }
                for(;remaining;remaining&=remaining-1) {
                    uint64_t bit=remaining&-remaining;
                    int other=kmap.find(word^bit);
                    if(other>=0)add(rosa_random::row_bit(bit),other);
                }
            }
            qneighbors.offsets[id+1]=int(qneighbors.edges.size());
        }
        for(int i=0;i<nk;++i)kneighbors.offsets[i+1]+=kneighbors.offsets[i];
        auto cursor=kneighbors.offsets;kneighbors.edges.resize(qneighbors.edges.size());
        for(int id=0;id<nq;++id)for(auto edge:qneighbors[id])kneighbors.edges[cursor[edge.other]++]={edge.bit,id};
        for(int i=0;i<n;++i){births+=first_neighbor[qclass[i]]<i;birth_prefix[i+1]=births;}
    }
    size_t workspace_bytes()const {
        return kmap.bytes()+(qclass.capacity()+kclass.capacity()+qcount.capacity()+kcount.capacity()
            +qsame.capacity()+ksame.capacity()+first_neighbor.capacity()+birth_prefix.capacity()
            +qneighbors.offsets.capacity()+kneighbors.offsets.capacity())*sizeof(int)
            +(qneighbors.edges.capacity()+kneighbors.edges.capacity())*sizeof(rosa_random::GramNeighbor);
    }
};
} // namespace interval_credit
} // namespace rosa_subq
