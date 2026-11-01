#pragma once
// Uniform Q-repair streaming states over one static suffix automaton of K.
// Does not compute K repair or common Q deletion contributions.
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include "exact_hamming_join.h"

namespace rosa_exact {

class SamQRepairs {
public:
    struct Route { int length=0, end=-1; };
    struct Repair {
        int owner=0, bit=0, length=0, end=-1, state=0;
        int short_state=0; // Exact edited suffix from owner through current t.
    };
    struct Counters {
        uint64_t transition_probes=0, fallback_queries=0, active_advances=0;
        uint64_t born=0, retired=0, endpoint_queries=0;
    };

    using RootJoinCounters=HammingJoinCounters;
    static uint64_t process_hash_seed() { return default_seed(); }
    RootJoinCounters root_join_counters;

    // q and k must remain valid throughout this object's lifetime.
    // Internal d=0 requests baseline matching only, using the same SAM scan.
    SamQRepairs(const uint64_t* q, const uint64_t* k, int n, int d,
                uint64_t repair_mask=~uint64_t(0), int repair_budget=-1,
                uint64_t sampling_seed=0, uint64_t forced_mask=0,
                uint64_t hash_seed=default_seed(), bool randomized_roots=false)
        : q_(q), k_(k), n_(n), d_(d), seed_(hash_seed), prefix_state_(n) {
        if (n<0 || d<0 || d>64) throw std::invalid_argument("invalid T or D");
        for (int bit=0;bit<d;++bit)
            if ((repair_mask>>bit)&1) repair_bits_.push_back(bit);
        builder_.reserve(std::max(1,2*n));
        edges_.reserve(size_t(n)*3);
        size_t capacity=8;
        while(capacity<std::max(size_t(8),size_t(n)*6))capacity*=2;
        transition_table_.assign(capacity,-1);
        builder_.push_back(BuilderNode{});
        state_salts_.push_back(mix(seed_+UINT64_C(0x9e3779b97f4a7c15)));
        int last=0;
        for (int r=0;r<n;++r) last=extend(last,k[r],r);
        build_root_targets(randomized_roots);
        if(repair_budget>=0)sample_repair_lanes(repair_budget,sampling_seed,forced_mask);
        freeze_transitions();
        build_links_and_euler();
        score_stamp_.assign(nodes_.size(),-1);
        endpoint_cache_.assign(nodes_.size(),-1);
        active_.reserve(size_t(n)*std::min(size_t(4),active_lanes_.size()));
    }

    void advance(int t) {
        if (t!=time_+1 || t>=n_) throw std::invalid_argument("advance must be sequential");
        time_=t;
        detached_.clear();
        if (t && !occurrence_position_.empty()) {
            const int p=occurrence_position_[t-1];
            if(!endpoint_tree_.a.empty())endpoint_tree_.update(p,t-1);
            if(!occurrence_prev_.empty()) {
                const int left=occurrence_prev_[p],right=occurrence_next_[p];
                if (left>=0) occurrence_next_[left]=p;
                if (right<n_) occurrence_prev_[right]=p;
            }
        }
        const Match old_base=base_match_;
        const uint64_t word_hash=mix(q_[t]^seed_);
        const int ordinary_root=causal_root(root_original_[t],t);
        base_match_=consume(old_base,q_[t],word_hash,t,ordinary_root);
        baseline_={base_match_.length,base_match_.end};

        size_t kept=0;
        const size_t old_size=active_.size();
        for (size_t i=0;i<old_size;++i) {
            Repair z=active_[i];
            ++counters.active_advances;
            const int short_target=valid_transition(z.short_state,q_[t],word_hash,t);
            if (short_target<0) { ++counters.retired; continue; }
            const Match m=z.short_state==z.state
                ? Match{short_target,z.length+1,direct_endpoint(short_target,z.end,q_[t],t)}
                : consume({z.state,z.length,z.end},q_[t],word_hash,t,ordinary_root);
            z.short_state=short_target;
            z.state=m.state;z.length=m.length;z.end=m.end;
            if(detachable(z,t))detached_.push_back(z);
            else active_[kept++]=z;
        }
        active_.resize(kept);
        const size_t query=size_t(query_class_[t]);
        size_t entry=root_offsets_[query];
        for(uint64_t pending=root_masks_[query];pending;pending&=pending-1) {
            const int bit=singleton_bit(pending&(~pending+1));
            const int target=root_targets_[entry++];
            if(!((selected_mask_>>bit)&1))continue;
            const int root=causal_root(target,t);
            if (root<0) continue;
            const uint64_t edited=q_[t]^(uint64_t(1)<<bit);
            Match m=consume(old_base,edited,mix(edited^seed_),t,root);
            const Repair z{t,bit,m.length,m.end,m.state,root};
            if(detachable(z,t))detached_.push_back(z);
            else active_.push_back(z);
            ++counters.born;
        }
    }

    // Singleton recognition is fused into advance's stable compaction.
    // Swapping recycles the caller's previous detached-vector capacity.
    void extract_singleton_continuations(const uint64_t*,std::vector<Repair>& out) {
        out.swap(detached_);detached_.clear();
    }

    Route baseline() const { return baseline_; }
    int baseline_state() const { return base_match_.state; }
    const std::vector<Repair>& repairs() const { return active_; }
    template<class F> void for_each_repair(F&& f) const { for (const auto& z:active_) f(z); }
    int state_count() const { return int(nodes_.size()); }
    int state_length(int state) const { return nodes_.at(state).length; }
    int suffix_link(int state) const { return nodes_.at(state).link; }
    int first_endpoint(int state) const { return nodes_.at(state).firstpos; }
    // Static maximum over all K occurrences, including future endpoints.
    int last_endpoint(int state) const { return last_endpoint_.at(state); }
    int current_time() const { return time_; }
    int sequence_length() const { return n_; }
    int prefix_state_at(int r) const { return prefix_state_.at(r); }

    // Most specific suffix-link ancestor with an occurrence ending before
    // limit. Static firstpos values are monotone along suffix links.
    int ancestor_before(int state, int limit) const {
        if (state<0 || state>=int(nodes_.size())) throw std::invalid_argument("invalid state");
        if (!state || limit<=0) return 0;
        if (nodes_[state].firstpos<limit) return state;
        int failed=state;
        if(up_.empty())build_lifting();
        for (int level=int(up_.size())-1; level>=0; --level) {
            const int ancestor=up_[level][failed];
            if (ancestor && nodes_[ancestor].firstpos>=limit) failed=ancestor;
        }
        return nodes_[failed].link;
    }

    // Return the unique suffix-link ancestor representing this state's
    // suffix of the requested positive length. Callers with a partial match
    // must additionally ensure length <= that match's actual length.
    int state_for_length(int state, int length) const {
        if (state<=0 || state>=int(nodes_.size()) || length<1 ||
            length>nodes_[state].length) throw std::invalid_argument("invalid suffix length");
        if(nodes_[nodes_[state].link].length<length)return state;
        int v=state;
        if(up_.empty())build_lifting();
        for (int level=int(up_.size())-1; level>=0; --level) {
            const int ancestor=up_[level][v];
            if (ancestor && nodes_[ancestor].length>=length) v=ancestor;
        }
        return v;
    }
    // Endpos occurrences, ordered by Euler order of prefix terminal states.
    // Ranges are half-open. Endpoints are not sorted numerically; filter r<t
    // explicitly when only the current causal dictionary should be visible.
    std::pair<int,int> occurrence_range(int state) const {
        if (state<0 || state>=int(nodes_.size())) throw std::invalid_argument("invalid state");
        return {occurrence_prefix_[tin_[state]],occurrence_prefix_[tout_[state]]};
    }
    const std::vector<int>& occurrence_endpoints() const { return occurrence_endpoints_; }

    // Report exactly the currently causal endpos set. Its earliest endpoint
    // is a known active anchor inside the state's contiguous Euler interval.
    // Walk both directions in the active occurrence chain, stopping at the
    // interval boundary. Cost is O(1 + number of reported occurrences).
    template<class F> void for_each_causal_occurrence(int state,F&& f) const {
        if (state<=0 || state>=int(nodes_.size())) throw std::invalid_argument("invalid state");
        const int first=nodes_[state].firstpos;
        if (first>=time_) return;
        if(first==last_endpoint_[state]){f(first);return;}
        if(occurrence_prev_.empty())build_active_occurrences();
        const auto range=occurrence_range(state);
        const int anchor=occurrence_position_[first];
        for (int p=anchor;p<range.second;p=occurrence_next_[p]) f(occurrence_endpoints_[p]);
        for (int p=occurrence_prev_[anchor];p>=range.first;p=occurrence_prev_[p]) f(occurrence_endpoints_[p]);
    }

    int latest_endpoint(int state) {
        if (!state) return -1;
        if (state<0 || state>=int(nodes_.size()) || time_<0) throw std::invalid_argument("invalid state");
        const int last=last_endpoint_[state];
        if (last<time_) return last;
        if (score_stamp_[state]!=time_) {
            const int first=nodes_[state].firstpos;
            const int lo=occurrence_prefix_[tin_[state]], hi=occurrence_prefix_[tout_[state]];
            if (first>=time_) endpoint_cache_[state]=-1;
            else if (hi-lo==last-first+1) endpoint_cache_[state]=std::min(last,time_-1);
            else {if(endpoint_tree_.a.empty())build_endpoint_tree();endpoint_cache_[state]=endpoint_tree_.query(lo,hi);}
            score_stamp_[state]=time_;
            ++counters.endpoint_queries;
        }
        return endpoint_cache_[state];
    }
    // Logical query count retained for the existing public work-counter ABI.
    size_t root_repair_entries() const {return query_symbols_*repair_bits_.size();}
    size_t stored_root_entries() const {return root_targets_.size();}
    size_t query_symbol_count() const {return query_symbols_;}
    size_t transition_working_bytes() const {
        return edges_.capacity()*sizeof(Edge)+transition_table_.capacity()*sizeof(int)
            +state_salts_.capacity()*sizeof(uint64_t);
    }
    // Exact auxiliary indexes are built only for an unresolved query.
    int materialized_auxiliary_indexes() const {return int(!endpoint_tree_.a.empty()) | (int(!up_.empty())<<1);}
    uint64_t supported_repair_mask() const {return supported_mask_;}
    uint64_t selected_repair_mask() const {return selected_mask_;}
    Counters counters;

private:
    struct RuntimeNode { int length=0,link=-1,firstpos=-1; };
    struct BuilderNode { int length=0,link=-1,firstpos=-1,head=-1; };
    struct Edge { uint64_t symbol; int source,target,next; };
    struct Match { int state=0,length=0,end=-1; };
    struct MaxTree {
        int size=1;
        std::vector<int> a;
        void init(int n) { while(size<n)size<<=1;a.assign(2*size,-1); }
        void update(int p,int value) {
            p+=size;a[p]=value;
            for(p>>=1;p;p>>=1)a[p]=value;
        }
        int query(int l,int r) const {
            int ans=-1;
            for(l+=size,r+=size;l<r;l>>=1,r>>=1) {
                if(l&1)ans=std::max(ans,a[l++]);
                if(r&1)ans=std::max(ans,a[--r]);
            }
            return ans;
        }
    } endpoint_tree_;
    const uint64_t* q_;
    const uint64_t* k_;
    int n_,d_,time_=-1,max_link_depth_=0;
    uint64_t seed_;
    size_t query_symbols_=0;
    size_t root_degree_=0;
    uint64_t root_reference_=0,root_varying_=0;
    uint64_t supported_mask_=0,selected_mask_=0;
    std::vector<size_t> active_lanes_;
    std::vector<uint64_t> state_salts_;
    std::vector<RuntimeNode> nodes_;
    std::vector<BuilderNode> builder_;
    std::vector<Edge> edges_;
    std::vector<int> transition_table_;
    std::vector<int> root_original_,root_targets_,repair_bits_,query_class_;
    std::vector<size_t> root_offsets_;
    std::vector<uint64_t> root_masks_;
    std::vector<int> prefix_state_,tin_,tout_,score_stamp_,endpoint_cache_;
    std::vector<int> last_endpoint_;
    std::vector<int> occurrence_prefix_,occurrence_endpoints_;
    mutable std::vector<int> occurrence_position_,occurrence_prev_,occurrence_next_;
    mutable std::vector<std::vector<int>> up_;
    Match base_match_;
    Route baseline_;
    std::vector<Repair> active_,detached_;
    bool detachable(const Repair& z,int t) const {
        return t+1<n_ && k_[z.end+1]==q_[t+1]
            && nodes_[z.short_state].firstpos==last_endpoint_[z.short_state];
    }

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
    static uint64_t mix(uint64_t x) {
        x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
        x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
        return x^(x>>31);
    }
    size_t find_slot(int state,uint64_t symbol,uint64_t word_hash) const {
        size_t slot=size_t(word_hash^state_salts_[state])&(transition_table_.size()-1);
        for(;;slot=(slot+1)&(transition_table_.size()-1)) {
            const int id=transition_table_[slot];
            if(id<0)return slot;
            const auto&e=edges_[id];
            if(e.source==state && e.symbol==symbol)return slot;
        }
    }
    void add_edge(size_t slot,int state,uint64_t symbol,int target) {
        if(!state) {
            if(root_degree_++)root_varying_|=symbol^root_reference_;
            else root_reference_=symbol;
        }
        transition_table_[slot]=int(edges_.size());
        edges_.push_back({symbol,state,target,builder_[state].head});
        builder_[state].head=int(edges_.size())-1;
    }
    int extend(int last,uint64_t symbol,int position) {
        const int cur=int(builder_.size());builder_.push_back(BuilderNode{});
        state_salts_.push_back(mix(uint64_t(cur)+seed_+UINT64_C(0x9e3779b97f4a7c15)));
        builder_[cur].length=builder_[last].length+1;
        builder_[cur].firstpos=position;prefix_state_[position]=cur;
        const uint64_t word_hash=mix(symbol^seed_);
        int p=last,existing=-1;
        while(p>=0) {
            const size_t slot=find_slot(p,symbol,word_hash);
            existing=transition_table_[slot];
            if(existing>=0)break;
            add_edge(slot,p,symbol,cur);p=builder_[p].link;
        }
        if(p<0)builder_[cur].link=0;
        else {
            const int v=edges_[existing].target;
            if(builder_[p].length+1==builder_[v].length)builder_[cur].link=v;
            else {
                const int clone=int(builder_.size());
                BuilderNode copy=builder_[v];copy.length=builder_[p].length+1;copy.head=-1;
                builder_.push_back(copy);state_salts_.push_back(mix(uint64_t(clone)+seed_+UINT64_C(0x9e3779b97f4a7c15)));
                for(int e=builder_[v].head;e>=0;) {
                    const Edge old=edges_[e];
                    add_edge(find_slot(clone,old.symbol,mix(old.symbol^seed_)),clone,old.symbol,old.target);
                    e=old.next;
                }
                while(p>=0) {
                    const int e=transition_table_[find_slot(p,symbol,word_hash)];
                    if(e<0 || edges_[e].target!=v)break;
                    edges_[e].target=clone;p=builder_[p].link;
                }
                builder_[v].link=builder_[cur].link=clone;
            }
        }
        return cur;
    }

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

    // Equal Q symbols share one static neighbor query. When enabled and large
    // enough to amortize its indexes, use a random balanced bit partition.
    // A one-bit neighbor agrees outside exactly one block; query that bucket.
    // Each bucket costs at most the number of flips in its omitted block:
    // enumerate a small bucket, otherwise do those direct full-key lookups.
    // No root neighbor can be missed and the candidate work is at most U_Q*D.
    void build_root_targets(bool randomized_roots) {
        query_class_.resize(n_);
        root_original_.assign(n_,-1);
        std::vector<uint64_t> symbols;
        std::vector<int> latest_query,ordinary_root;
        symbols.reserve(n_);latest_query.reserve(n_);ordinary_root.reserve(n_);
        size_t capacity=8;
        while(capacity<std::max(size_t(8),size_t(n_)*2))capacity*=2;
        std::vector<int> classes(capacity,-1);
        for(int t=0;t<n_;++t) {
            const uint64_t symbol=q_[t],word_hash=mix(symbol^seed_);
            size_t slot=size_t(word_hash)&(capacity-1);
            while(classes[slot]>=0 && symbols[classes[slot]]!=symbol)
                slot=(slot+1)&(capacity-1);
            int id=classes[slot];
            if(id<0) {
                id=int(symbols.size());classes[slot]=id;
                symbols.push_back(symbol);latest_query.push_back(t);
                // Every K word agrees outside root_varying_. A mismatch
                // there proves absence without touching the root dictionary.
                const int edge=((symbol^root_reference_)&~root_varying_) ? -1
                    : transition_table_[find_slot(0,symbol,word_hash)];
                ordinary_root.push_back(edge<0 ? -1 : edges_[edge].target);
            } else latest_query[id]=t;
            query_class_[t]=id;root_original_[t]=ordinary_root[id];
        }
        query_symbols_=symbols.size();
        root_offsets_.resize(symbols.size()+1);
        root_masks_.assign(symbols.size(),0);
        root_targets_.reserve(symbols.size()*std::min(size_t(4),repair_bits_.size()));
        uint64_t allowed=0;
        for(int bit:repair_bits_)allowed|=uint64_t(1)<<bit;
        exact_hamming_one_join(symbols,ordinary_root,
            HammingDomain{root_degree_,root_reference_,root_varying_},allowed,seed_,randomized_roots,
            [&](uint64_t symbol)->int {
                const int edge=transition_table_[find_slot(0,symbol,mix(symbol^seed_))];
                return edge<0?-1:edges_[edge].target;
            },
            [&](auto&& emit) {
                for(int edge=builder_[0].head;edge>=0;edge=edges_[edge].next)
                    emit(edges_[edge].symbol,edges_[edge].target);
            },
            [&](size_t id,uint64_t available,const int* targets) {
                root_offsets_[id]=root_targets_.size();
                uint64_t causal=0;
                for(uint64_t pending=available;pending;pending&=pending-1) {
                    const int bit=singleton_bit(pending&(~pending+1));
                    const int target=targets[bit];
                    if(builder_[target].firstpos>=latest_query[id])continue;
                    causal|=uint64_t(1)<<bit;root_targets_.push_back(target);
                }
                root_masks_[id]=causal;supported_mask_|=causal;
            },root_join_counters);
        root_offsets_[symbols.size()]=root_targets_.size();
        for(size_t lane=0;lane<repair_bits_.size();++lane)
            if((supported_mask_>>repair_bits_[lane])&1)active_lanes_.push_back(lane);
        selected_mask_=supported_mask_;
    }

    void sample_repair_lanes(int budget,uint64_t seed,uint64_t forced_mask) {
        std::vector<size_t> forced,optional;
        for(const size_t lane:active_lanes_)
            (((forced_mask>>repair_bits_[lane])&1)?forced:optional).push_back(lane);
        // Every nonzero residual keeps a positive inclusion probability.
        // Forced exact lanes may therefore raise a too-small target budget
        // by one, rather than silently dropping every other lane forever.
        const size_t wanted=std::min(optional.size(),size_t(std::max(1,budget-int(forced.size()))));
        if(wanted<optional.size()) {
            std::mt19937_64 random(seed);
            for(size_t i=0;i<wanted;++i) {
                std::uniform_int_distribution<size_t> choose(i,optional.size()-1);
                std::swap(optional[i],optional[choose(random)]);
            }
            optional.resize(wanted);
        }
        active_lanes_=std::move(forced);
        active_lanes_.insert(active_lanes_.end(),optional.begin(),optional.end());
        std::sort(active_lanes_.begin(),active_lanes_.end());
        selected_mask_=0;
        for(const size_t lane:active_lanes_)selected_mask_|=uint64_t(1)<<repair_bits_[lane];
    }

    int causal_root(int target,int t) const {
        return target>=0 && nodes_[target].firstpos<t ? target : -1;
    }

    void freeze_transitions() {
        nodes_.resize(builder_.size());
        for(size_t v=0;v<builder_.size();++v) {
            const auto&built=builder_[v];
            nodes_[v]={built.length,built.link,built.firstpos};
        }
        // The full-key table is also the immutable runtime dictionary. No
        // node maps or tree-shaped transition copies survive construction.
        std::vector<BuilderNode>().swap(builder_);
    }

    void build_links_and_euler() {
        const int m=int(nodes_.size());
        std::vector<int> head(m,-1),next(m,-1);
        for(int v=1;v<m;++v) {
            const int p=nodes_[v].link;
            next[v]=head[p];head[p]=v;
        }
        tin_.resize(m);tout_.resize(m);
        last_endpoint_.assign(m,-1);
        for(int r=0;r<n_;++r)last_endpoint_[prefix_state_[r]]=r;
        std::vector<int> stack{0};int timer=0,depth=-1;
        while(!stack.empty()) {
            int v=stack.back();stack.pop_back();
            if(v<0){
                --depth;v=~v;tout_[v]=timer;
                if(v) {
                    const int p=nodes_[v].link;
                    last_endpoint_[p]=std::max(last_endpoint_[p],last_endpoint_[v]);
                }
                continue;
            }
            ++depth;max_link_depth_=std::max(max_link_depth_,depth);
            tin_[v]=timer++;stack.push_back(~v);
            for(int c=head[v];c>=0;c=next[c])stack.push_back(c);
        }
        // Every input endpoint contributes exactly its prefix terminal,
        // never a clone. Prefix counts map Euler node ranges to these T
        // actual occurrences without retaining holes for nonterminals.
        std::vector<int> endpoint_at_euler(m,-1);
        for(int r=0;r<n_;++r)endpoint_at_euler[tin_[prefix_state_[r]]]=r;
        occurrence_prefix_.assign(m+1,0);
        occurrence_endpoints_.reserve(n_);
        for(int p=0;p<m;++p) {
            occurrence_prefix_[p+1]=occurrence_prefix_[p];
            if(endpoint_at_euler[p]>=0) {
                occurrence_endpoints_.push_back(endpoint_at_euler[p]);
                ++occurrence_prefix_[p+1];
            }
        }
    }

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((noinline))
#elif defined(_MSC_VER)
    __declspec(noinline)
#endif
    void build_active_occurrences() const {
        occurrence_position_.resize(n_);
        occurrence_prev_.resize(n_);occurrence_next_.resize(n_);
        for (int p=0;p<n_;++p) {
            occurrence_position_[occurrence_endpoints_[p]]=p;
            occurrence_prev_[p]=p-1;occurrence_next_[p]=p+1;
        }
        // Reverse deletion records the neighbors that will exist at each
        // forward activation. A deleted slot is no longer modified, so these
        // same two arrays retain insertion instructions for inactive slots
        // while simultaneously linking all currently active slots.
        for (int r=n_-1;r>=0;--r) {
            const int p=occurrence_position_[r];
            const int left=occurrence_prev_[p],right=occurrence_next_[p];
            if (left>=0) occurrence_next_[left]=right;
            if (right<n_) occurrence_prev_[right]=left;
        }
        for(int r=0;r<time_;++r) {
            const int p=occurrence_position_[r];
            const int left=occurrence_prev_[p],right=occurrence_next_[p];
            if(left>=0)occurrence_next_[left]=p;
            if(right<n_)occurrence_prev_[right]=p;
        }
    }

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((noinline))
#elif defined(_MSC_VER)
    __declspec(noinline)
#endif
    void build_endpoint_tree() {
        endpoint_tree_.init(n_);
        if(occurrence_position_.empty()) {
            occurrence_position_.resize(n_);
            for(int p=0;p<n_;++p)occurrence_position_[occurrence_endpoints_[p]]=p;
        }
        for(int r=0;r<time_;++r)endpoint_tree_.a[endpoint_tree_.size+occurrence_position_[r]]=r;
        for(int p=endpoint_tree_.size-1;p>0;--p)
            endpoint_tree_.a[p]=std::max(endpoint_tree_.a[2*p],endpoint_tree_.a[2*p+1]);
    }

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((noinline))
#elif defined(_MSC_VER)
    __declspec(noinline)
#endif
    void build_lifting() const {
        const int m=int(nodes_.size());
        int levels=1;while((uint64_t(1)<<levels)<=uint64_t(std::max(1,max_link_depth_)))++levels;
        up_.assign(levels,std::vector<int>(m,0));
        for(int v=1;v<m;++v)up_[0][v]=nodes_[v].link;
        for(int level=1;level<levels;++level)
            for(int v=0;v<m;++v)up_[level][v]=up_[level-1][up_[level-1][v]];
    }

    int valid_transition(int state,uint64_t symbol,uint64_t word_hash,int t) {
        ++counters.transition_probes;
        const int edge=transition_table_[find_slot(state,symbol,word_hash)];
        if(edge<0)return -1;
        const int target=edges_[edge].target;
        return nodes_[target].firstpos<t ? target : -1;
    }

    int direct_endpoint(int state,int previous_end,uint64_t symbol,int t) {
        const int last=last_endpoint_[state];
        if (last<t) return last;
        const int candidate=previous_end+1;
        // Every occurrence of the direct extension loses its final symbol
        // to become an occurrence of the previous pattern. Thus candidate
        // is an upper bound; if it extends, that bound is attained exactly.
        if (previous_end>=0 && candidate<t && k_[candidate]==symbol) {
            endpoint_cache_[state]=candidate;score_stamp_[state]=t;
            return candidate;
        }
        return latest_endpoint(state);
    }

    Match consume(Match before,uint64_t symbol,uint64_t word_hash,int t,int root_target) {
        int v=before.state;
        if(!v)return root_target>=0 ? Match{root_target,1,latest_endpoint(root_target)} : Match{};
        int target=valid_transition(v,symbol,word_hash,t);
        if(target>=0)return {target,before.length+1,direct_endpoint(target,before.end,symbol,t)};
        // If even the one-symbol suffix is unavailable, no ancestor can
        // accept it. This exact rejection applies to every input.
        if(root_target<0)return {};
        ++counters.fallback_queries;
        const int parent=nodes_[v].link;
        const int parent_target=parent ? valid_transition(parent,symbol,word_hash,t) : root_target;
        if(parent_target>=0)
            return {parent_target,std::min(before.length,nodes_[parent].length)+1,latest_endpoint(parent_target)};
        // Validity is ancestor-closed: every shorter suffix of an occurrence
        // ending before t also occurs before t. The direct parent was checked
        // above; binary lifting is only needed beyond that invalid parent.
        int failed=parent;
        if(up_.empty())build_lifting();
        for(int level=int(up_.size())-1;level>=0;--level) {
            const int ancestor=up_[level][failed];
            if(ancestor && valid_transition(ancestor,symbol,word_hash,t)<0)failed=ancestor;
        }
        const int source=nodes_[failed].link;
        target=source ? valid_transition(source,symbol,word_hash,t) : root_target;
        if(target<0)return {};
        return {target,std::min(before.length,nodes_[source].length)+1,latest_endpoint(target)};
    }
};

} // namespace rosa_exact
