#pragma once
// Uniform Q-repair streaming states over one static suffix automaton of K.
// Does not compute K repair or common Q deletion contributions.
#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

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

    // q and k must remain valid throughout this object's lifetime.
    // Internal d=0 requests baseline matching only, using the same SAM scan.
    SamQRepairs(const uint64_t* q, const uint64_t* k, int n, int d)
        : q_(q), k_(k), n_(n), d_(d), prefix_state_(n) {
        if (n<0 || d<0 || d>64) throw std::invalid_argument("invalid T or D");
        builder_.reserve(std::max(1,2*n));
        builder_.push_back(BuilderNode{});
        int last=0;
        for (int r=0;r<n;++r) last=extend(last,k[r],r);
        build_root_targets();
        freeze_transitions();
        build_links_and_euler();
        score_stamp_.assign(nodes_.size(),-1);
        endpoint_cache_.assign(nodes_.size(),-1);
        active_.reserve(size_t(n)*std::min(d,4));
    }

    void advance(int t) {
        if (t!=time_+1 || t>=n_) throw std::invalid_argument("advance must be sequential");
        time_=t;
        detached_.clear();
        if (t) {
            const int p=occurrence_position_[t-1];
            endpoint_tree_.update(p,t-1);
            const int left=occurrence_prev_[p],right=occurrence_next_[p];
            if (left>=0) occurrence_next_[left]=p;
            if (right<n_) occurrence_prev_[right]=p;
        }
        const Match old_base=base_match_;
        const int ordinary_root=causal_root(root_original_[t],t);
        base_match_=consume(old_base,q_[t],t,ordinary_root);
        baseline_={base_match_.length,base_match_.end};

        size_t kept=0;
        const size_t old_size=active_.size();
        for (size_t i=0;i<old_size;++i) {
            Repair z=active_[i];
            ++counters.active_advances;
            const int short_target=valid_transition(z.short_state,q_[t],t);
            if (short_target<0) { ++counters.retired; continue; }
            const Match m=z.short_state==z.state
                ? Match{short_target,z.length+1,direct_endpoint(short_target,z.end,q_[t],t)}
                : consume({z.state,z.length,z.end},q_[t],t,ordinary_root);
            z.short_state=short_target;
            z.state=m.state;z.length=m.length;z.end=m.end;
            if(detachable(z,t))detached_.push_back(z);
            else active_[kept++]=z;
        }
        active_.resize(kept);
        for (int bit=0;bit<d_;++bit) {
            const int root=causal_root(root_flips_[size_t(t)*d_+bit],t);
            if (root<0) continue;
            Match m=consume(old_base,q_[t]^(uint64_t(1)<<bit),t,root);
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
            else endpoint_cache_[state]=endpoint_tree_.query(lo,hi);
            score_stamp_[state]=time_;
            ++counters.endpoint_queries;
        }
        return endpoint_cache_[state];
    }
    Counters counters;

private:
    struct RuntimeNode { int length=0,link=-1,firstpos=-1; };
    struct BuilderNode { int length=0,link=-1,firstpos=-1; std::map<uint64_t,int> next; };
    struct Edge { uint64_t symbol; int target; };
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
    int n_,d_,time_=-1;
    std::vector<RuntimeNode> nodes_;
    std::vector<BuilderNode> builder_;
    std::vector<Edge> edges_;
    std::vector<int> edge_start_;
    std::vector<int> root_original_,root_flips_;
    std::vector<int> prefix_state_,tin_,tout_,score_stamp_,endpoint_cache_;
    std::vector<int> last_endpoint_;
    std::vector<int> occurrence_prefix_,occurrence_endpoints_;
    std::vector<int> occurrence_position_,occurrence_prev_,occurrence_next_;
    std::vector<std::vector<int>> up_;
    Match base_match_;
    Route baseline_;
    std::vector<Repair> active_,detached_;
    bool detachable(const Repair& z,int t) const {
        return t+1<n_ && k_[z.end+1]==q_[t+1]
            && nodes_[z.short_state].firstpos==last_endpoint_[z.short_state];
    }

    int extend(int last,uint64_t c,int position) {
        int cur=int(builder_.size());builder_.push_back(BuilderNode{});
        builder_[cur].length=builder_[last].length+1;builder_[cur].firstpos=position;
        prefix_state_[position]=cur;
        int p=last;
        while(p>=0 && builder_[p].next.find(c)==builder_[p].next.end()) {
            builder_[p].next[c]=cur;p=builder_[p].link;
        }
        if(p<0)builder_[cur].link=0;
        else {
            const int q=builder_[p].next.find(c)->second;
            if(builder_[p].length+1==builder_[q].length)builder_[cur].link=q;
            else {
                BuilderNode copy=builder_[q];copy.length=builder_[p].length+1;
                const int clone=int(builder_.size());builder_.push_back(std::move(copy));
                while(p>=0) {
                    auto it=builder_[p].next.find(c);
                    if(it==builder_[p].next.end() || it->second!=q)break;
                    it->second=clone;p=builder_[p].link;
                }
                builder_[q].link=builder_[cur].link=clone;
            }
        }
        return cur;
    }

    // Deterministic offline dictionary join for every future root query.
    // Flipping bit j exchanges the two j-bit halves inside each block with
    // identical higher bits. Their lower-bit order is unchanged, so the
    // sorted Q order yields each sorted flipped order in one linear pass.
    void build_root_targets() {
        std::vector<Edge> root;
        root.reserve(builder_[0].next.size());
        for (const auto& edge:builder_[0].next)
            root.push_back({edge.first,edge.second});
        std::vector<int> order(n_);
        for (int t=0;t<n_;++t)order[t]=t;
        std::sort(order.begin(),order.end(),[&](int a,int b){return q_[a]<q_[b];});
        root_original_.assign(n_,-1);
        root_flips_.assign(size_t(n_)*d_,-1);
        size_t cursor=0;
        for (const int t:order) {
            const uint64_t symbol=q_[t];
            while(cursor<root.size() && root[cursor].symbol<symbol)++cursor;
            if(cursor<root.size() && root[cursor].symbol==symbol)
                root_original_[t]=root[cursor].target;
        }
        for (int bit=0;bit<d_;++bit) {
            const uint64_t flip=uint64_t(1)<<bit;
            const uint64_t high_mask=bit==63 ? 0 : (~uint64_t(0)<<(bit+1));
            cursor=0;
            auto join_range=[&](int lo,int hi) {
                for (int p=lo;p<hi;++p) {
                    const int t=order[p];const uint64_t symbol=q_[t]^flip;
                    while(cursor<root.size() && root[cursor].symbol<symbol)++cursor;
                    if(cursor<root.size() && root[cursor].symbol==symbol)
                        root_flips_[size_t(t)*d_+bit]=root[cursor].target;
                }
            };
            int lo=0;
            while(lo<n_) {
                const uint64_t high=q_[order[lo]]&high_mask;
                int hi=lo+1;
                while(hi<n_ && (q_[order[hi]]&high_mask)==high)++hi;
                int mid=lo;
                while(mid<hi && !(q_[order[mid]]&flip))++mid;
                join_range(mid,hi);
                join_range(lo,mid);
                lo=hi;
            }
        }
    }

    int causal_root(int target,int t) const {
        return target>=0 && nodes_[target].firstpos<t ? target : -1;
    }

    void freeze_transitions() {
        size_t count=0;
        for (const auto& node:builder_) count+=node.next.size();
        edges_.resize(count);
        edge_start_.resize(builder_.size()+1);
        nodes_.resize(builder_.size());
        size_t offset=0;
        for (size_t v=0;v<builder_.size();++v) {
            const auto& built=builder_[v];
            nodes_[v]={built.length,built.link,built.firstpos};
            edge_start_[v]=int(offset);
            const size_t degree=builder_[v].next.size();
            auto it=builder_[v].next.begin();
            // One implicit complete binary search tree per node. Inorder
            // traversal consumes the sorted map; heap slot 1 is stored at
            // offset, so no dummy edge or padding is required.
            const auto fill=[&](const auto& self,size_t slot)->void {
                if(slot>degree)return;
                self(self,2*slot);
                edges_[offset+slot-1]={it->first,it->second};
                ++it;
                self(self,2*slot+1);
            };
            fill(fill,1);
            offset+=degree;
            builder_[v].next.clear();
        }
        edge_start_[builder_.size()]=int(offset);
        // No streaming method retains a construction map or its capacity.
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
        std::vector<int> stack{0};int timer=0;
        while(!stack.empty()) {
            int v=stack.back();stack.pop_back();
            if(v<0){
                v=~v;tout_[v]=timer;
                if(v) {
                    const int p=nodes_[v].link;
                    last_endpoint_[p]=std::max(last_endpoint_[p],last_endpoint_[v]);
                }
                continue;
            }
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
        endpoint_tree_.init(n_);
        int levels=1;while((uint64_t(1)<<levels)<=uint64_t(std::max(1,m)))++levels;
        up_.assign(levels,std::vector<int>(m,0));
        for(int v=1;v<m;++v)up_[0][v]=nodes_[v].link;
        for(int level=1;level<levels;++level)
            for(int v=0;v<m;++v)up_[level][v]=up_[level-1][up_[level-1][v]];
    }

    int valid_transition(int state,uint64_t symbol,int t) {
        ++counters.transition_probes;
        const size_t offset=size_t(edge_start_[state]);
        const size_t degree=size_t(edge_start_[state+1])-offset;
        size_t slot=1;
        while(slot<=degree) {
            const auto& edge=edges_[offset+slot-1];
            if(edge.symbol==symbol) {
                const int target=edge.target;
                return nodes_[target].firstpos<t ? target : -1;
            }
            slot=2*slot+size_t(edge.symbol<symbol);
        }
        return -1;
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

    Match consume(Match before,uint64_t symbol,int t,int root_target) {
        int v=before.state;
        if(!v)return root_target>=0 ? Match{root_target,1,latest_endpoint(root_target)} : Match{};
        int target=valid_transition(v,symbol,t);
        if(target>=0)return {target,before.length+1,direct_endpoint(target,before.end,symbol,t)};
        // If even the one-symbol suffix is unavailable, no ancestor can
        // accept it. This exact rejection applies to every input.
        if(root_target<0)return {};
        ++counters.fallback_queries;
        const int parent=nodes_[v].link;
        const int parent_target=parent ? valid_transition(parent,symbol,t) : root_target;
        if(parent_target>=0)
            return {parent_target,std::min(before.length,nodes_[parent].length)+1,latest_endpoint(parent_target)};
        // Validity is ancestor-closed: every shorter suffix of an occurrence
        // ending before t also occurs before t. The direct parent was checked
        // above; binary lifting is only needed beyond that invalid parent.
        int failed=parent;
        for(int level=int(up_.size())-1;level>=0;--level) {
            const int ancestor=up_[level][failed];
            if(ancestor && valid_transition(ancestor,symbol,t)<0)failed=ancestor;
        }
        const int source=nodes_[failed].link;
        target=source ? valid_transition(source,symbol,t) : root_target;
        if(target<0)return {};
        return {target,std::min(before.length,nodes_[source].length)+1,latest_endpoint(target)};
    }
};

} // namespace rosa_exact
