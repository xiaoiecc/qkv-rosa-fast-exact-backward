#pragma once
// One causal baseline trajectory over a static suffix automaton of K.
// Unlike the many edited trajectories in the backward, its total suffix-link
// fallback count is <= T, so it needs neither binary lifting nor sorted Q.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace rosa_exact {

class SamForward {
public:
    struct Counters {
        uint64_t transition_probes=0, suffix_fallbacks=0, endpoint_queries=0;
        uint64_t endpoint_updates=0, hash_probe_steps=0;
    } counters;

    // Hashes choose dictionary slots, never decide symbol equality. Every hit
    // compares the complete state id and uint64 word. Seeds affect only work.
    static uint64_t default_seed() {
        static const uint64_t value=[] {
            try {
                std::random_device rd;
                return (uint64_t(rd())<<32)^uint64_t(rd());
            } catch (...) {
                return uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
            }
        }();
        return value;
    }

    SamForward(const uint64_t* q,const uint64_t* k,int n,
               uint64_t seed=default_seed()) : q_(q),k_(k),n_(n),seed_(seed) {
        if(n<0 || n>=(1<<28))throw std::invalid_argument("invalid sequence length");
        nodes_.reserve(size_t(std::max(1,2*n)));
        edges_.reserve(size_t(n)*3);
        nodes_.push_back(Node{});nodes_[0].salt=mix(seed_+UINT64_C(0x9e3779b97f4a7c15));
        size_t capacity=8;
        while(capacity<std::max(size_t(8),size_t(n)*6))capacity*=2;
        table_.assign(capacity,-1);
        occurrence_position_.resize(n);
        int last=0;
        for(int r=0;r<n;++r)last=extend(last,k[r],r);
        build_occurrence_ranges();
    }

    void run(int32_t* lengths,int32_t* routes) {
        int state=0,length=0,previous_end=-1;
        for(int t=0;t<n_;++t) {
            bool direct=true;
            int target=-1;
            const uint64_t word_hash=mix(q_[t]^seed_);
            for(;;) {
                ++counters.transition_probes;
                const int edge=table_[find_slot(state,q_[t],word_hash)];
                target=edge<0 ? -1 : edges_[edge].target;
                if(target>=0 && nodes_[target].first<t)break;
                if(!state){target=-1;length=0;break;}
                state=nodes_[state].link;
                length=nodes_[state].length;
                direct=false;
                ++counters.suffix_fallbacks;
            }
            if(target>=0) {
                state=target;++length;
                previous_end=latest_endpoint(state,t,direct?previous_end:-1,q_[t]);
            } else {state=0;previous_end=-1;}
            lengths[t]=length;routes[t]=previous_end;
        }
    }

    size_t state_count() const {return nodes_.size();}
    size_t edge_count() const {return edges_.size();}
    size_t hash_slots() const {return table_.size();}
    bool endpoint_index_materialized() const {return !tree_.empty();}
    // Allocator overhead is excluded; these are retained vector capacities,
    // not process RSS and not the peak while the temporary DFS tree is built.
    size_t retained_capacity_bytes() const {
        return nodes_.capacity()*sizeof(Node)+edges_.capacity()*sizeof(Edge)
            +(table_.capacity()+occurrence_position_.capacity()+tree_.capacity())*sizeof(int);
    }

private:
    struct Node {
        int length=0,link=-1,first=-1,head=-1;
        int begin=0,end=0,last=-1,terminal=-1;
        uint64_t salt=0;
    };
    struct Edge {uint64_t symbol;int source,target,next;};
    const uint64_t* q_;
    const uint64_t* k_;
    int n_;
    uint64_t seed_;
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
    std::vector<int> table_,occurrence_position_,tree_;
    int tree_size_=1,activated_=0;

    static uint64_t mix(uint64_t x) {
        x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
        x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
        return x^(x>>31);
    }
    size_t hash(int state,uint64_t word_hash) const {
        return size_t(word_hash^nodes_[state].salt)&(table_.size()-1);
    }
    size_t find_slot(int state,uint64_t word,uint64_t word_hash) {
        size_t p=hash(state,word_hash);
        for(;;p=(p+1)&(table_.size()-1)) {
            ++counters.hash_probe_steps;
            const int id=table_[p];
            if(id<0)return p;
            const auto&e=edges_[id];
            if(e.source==state && e.symbol==word)return p;
        }
    }
    void add_edge(size_t p,int state,uint64_t word,int target) {
        table_[p]=int(edges_.size());
        edges_.push_back({word,state,target,nodes_[state].head});
        nodes_[state].head=int(edges_.size())-1;
    }
    int extend(int last,uint64_t symbol,int r) {
        const int cur=int(nodes_.size());nodes_.push_back(Node{});
        nodes_[cur].salt=mix(uint64_t(cur)+seed_+UINT64_C(0x9e3779b97f4a7c15));
        nodes_[cur].length=nodes_[last].length+1;
        nodes_[cur].first=r;nodes_[cur].last=r;nodes_[cur].terminal=r;
        const uint64_t word_hash=mix(symbol^seed_);
        int p=last,existing=-1;
        while(p>=0) {
            const size_t slot=find_slot(p,symbol,word_hash);
            existing=table_[slot];
            if(existing>=0)break;
            add_edge(slot,p,symbol,cur);p=nodes_[p].link;
        }
        if(p<0)nodes_[cur].link=0;
        else {
            const int v=edges_[existing].target;
            if(nodes_[p].length+1==nodes_[v].length)nodes_[cur].link=v;
            else {
                const int clone=int(nodes_.size());
                Node copy=nodes_[v];copy.length=nodes_[p].length+1;
                copy.head=-1;copy.terminal=-1;copy.last=-1;
                copy.salt=mix(uint64_t(clone)+seed_+UINT64_C(0x9e3779b97f4a7c15));
                nodes_.push_back(copy);
                for(int e=nodes_[v].head;e>=0;) {
                    const Edge old=edges_[e];
                    const size_t slot=find_slot(clone,old.symbol,mix(old.symbol^seed_));
                    add_edge(slot,clone,old.symbol,old.target);e=old.next;
                }
                while(p>=0) {
                    const int e=table_[find_slot(p,symbol,word_hash)];
                    if(e<0 || edges_[e].target!=v)break;
                    edges_[e].target=clone;p=nodes_[p].link;
                }
                nodes_[v].link=nodes_[cur].link=clone;
            }
        }
        return cur;
    }
    void build_occurrence_ranges() {
        const int m=int(nodes_.size());
        std::vector<int> child(m,-1),sibling(m,-1),stack;
        for(int v=1;v<m;++v) {
            const int p=nodes_[v].link;sibling[v]=child[p];child[p]=v;
        }
        stack.reserve(m);stack.push_back(0);int timer=0;
        while(!stack.empty()) {
            int v=stack.back();stack.pop_back();
            if(v<0) {
                v=~v;nodes_[v].end=timer;
                if(v)nodes_[nodes_[v].link].last=std::max(nodes_[nodes_[v].link].last,nodes_[v].last);
                continue;
            }
            nodes_[v].begin=timer;
            if(nodes_[v].terminal>=0)occurrence_position_[nodes_[v].terminal]=timer++;
            stack.push_back(~v);
            for(int c=child[v];c>=0;c=sibling[c])stack.push_back(c);
        }
    }
    void activate_before(int t) {
        if(tree_.empty()) {
            while(tree_size_<n_)tree_size_*=2;
            tree_.assign(size_t(tree_size_)*2,-1);
            // First query builds the entire active prefix in linear time.
            for(int r=0;r<t;++r)tree_[tree_size_+occurrence_position_[r]]=r;
            for(int p=tree_size_-1;p;--p)tree_[p]=std::max(tree_[p*2],tree_[p*2+1]);
            counters.endpoint_updates+=uint64_t(t);activated_=t;
        }
        while(activated_<t) {
            const int r=activated_++;
            for(int p=tree_size_+occurrence_position_[r];p;p>>=1)tree_[p]=r;
            ++counters.endpoint_updates;
        }
    }
    int latest_endpoint(int state,int t,int previous_end,uint64_t symbol) {
        const auto&v=nodes_[state];
        if(v.last<t)return v.last;
        const int candidate=previous_end+1;
        if(previous_end>=0 && candidate<t && k_[candidate]==symbol)return candidate;
        if(v.end-v.begin==v.last-v.first+1)return std::min(v.last,t-1);
        // The certificates above establish the exact answer for any input.
        // Only unresolved queries materialize/update the same general index.
        activate_before(t);++counters.endpoint_queries;
        int answer=-1;
        for(int l=v.begin+tree_size_,r=v.end+tree_size_;l<r;l>>=1,r>>=1) {
            if(l&1)answer=std::max(answer,tree_[l++]);
            if(r&1)answer=std::max(answer,tree_[--r]);
        }
        return answer;
    }
};

} // namespace rosa_exact
