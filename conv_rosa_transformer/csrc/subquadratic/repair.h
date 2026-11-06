#pragma once
// Indexed Q heavy/light repair and anchored long K repair.
// Matching and latest-endpoint ties are exact. These routines emit weighted
// work, not the quadratic owner-by-row route table. See docs/subquadratic.md.
#include "index.h"
#include "types.h"
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace rosa_subq {
namespace repair_detail {

struct Point {
    std::uint64_t symbol;
    int x,y,source;
};

// All inner range trees share contiguous arenas. Points are activated once,
// in increasing source order: every ancestor can be assigned the new source
// directly, rather than recomputing max(children) on each update.
class FrontierIndex {
    struct Node {
        std::size_t ybase=0,tbase=0;
        int count=0,power=0;
    };
    struct Group {
        std::uint64_t symbol;
        int first,count,power,earliest;
        std::size_t node_base;
    };
    struct Update { std::uint32_t node,leaf; };
    std::vector<Group> groups_;
    ExactWordRanks group_ids_;
    std::vector<Node> nodes_;
    std::vector<int> ys_,maximum_;
    std::vector<std::size_t> update_start_;
    std::vector<Update> updates_;

    static int power_for(int count) {
        int power=1;
        while(power<count) {
            if(power>std::numeric_limits<int>::max()/2)
                throw std::length_error("Q range tree exceeds int32 range");
            power*=2;
        }
        return power;
    }
    static std::size_t checked_add(std::size_t a,std::size_t b) {
        if(b>std::numeric_limits<std::size_t>::max()-a)
            throw std::length_error("Q range arena size overflow");
        return a+b;
    }
    int descend_first(const Node& node,int vertex) const {
        const int* tree=maximum_.data()+node.tbase;
        while(vertex<node.power) {
            vertex*=2;
            if(tree[vertex]<0)++vertex;
        }
        return ys_[node.ybase+std::size_t(vertex-node.power)];
    }
    int descend_last(const Node& node,int vertex) const {
        const int* tree=maximum_.data()+node.tbase;
        while(vertex<node.power) {
            vertex=2*vertex+1;
            if(tree[vertex]<0)--vertex;
        }
        return ys_[node.ybase+std::size_t(vertex-node.power)];
    }
    // First/last active y in the given half-open index range. Deferred right
    // (left) blocks are overwritten by blocks closer to the requested edge.
    int first_active(const Node& node,int lo,int hi) const {
        const int* tree=maximum_.data()+node.tbase;
        int deferred=-1;lo+=node.power;hi+=node.power;
        while(lo<hi) {
            if(lo&1) { if(tree[lo]>=0)return descend_first(node,lo); ++lo; }
            if(hi&1) { --hi;if(tree[hi]>=0)deferred=hi; }
            lo/=2;hi/=2;
        }
        return deferred<0?-1:descend_first(node,deferred);
    }
    int last_active(const Node& node,int lo,int hi) const {
        const int* tree=maximum_.data()+node.tbase;
        int deferred=-1;lo+=node.power;hi+=node.power;
        while(lo<hi) {
            if(hi&1) { --hi;if(tree[hi]>=0)return descend_last(node,hi); }
            if(lo&1) { if(tree[lo]>=0)deferred=lo; ++lo; }
            lo/=2;hi/=2;
        }
        return deferred<0?-1:descend_last(node,deferred);
    }

public:
    struct Cover {
        // At most two nodes per level, with n bounded by Index's int domain.
        std::array<std::uint32_t,64> nodes{};
        int size=0;
    };
    std::vector<Point> points;

    explicit FrontierIndex(const Index& index,int light_threshold=0,int requested_bits=0) {
        const int n=index.n;
        if(requested_bits<0 || requested_bits>64)
            throw std::invalid_argument("invalid Q frontier bit count");
        const auto& right=index.forward_suffixes();
        const auto& left=index.reverse_suffixes();
        points.reserve(n);
        std::vector<int> source_by_y(left.size(),-1);
        for(int source=0;source<n;++source) {
            const int x=right.rank[n+source+2];
            const int y=left.rank[2*n+1-source];
            points.push_back(Point{index.k[source],x,y,source});
            source_by_y[y]=source;
        }
        std::sort(points.begin(),points.end(),[](const Point& a,const Point& b){
            return a.symbol<b.symbol || (a.symbol==b.symbol && a.x<b.x);
        });
        update_start_.assign(std::size_t(n)+1,0);
        std::vector<std::uint64_t> symbols;
        ExactWordRanks query_ids;
        std::vector<int> last_query;
        bool queries_ready=false;
        auto queried_after=[&](std::uint64_t symbol,int earliest) {
            if(!requested_bits)return true;
            if(!queries_ready) {
                std::vector<std::uint64_t> words=index.q;
                std::sort(words.begin(),words.end());words.erase(std::unique(words.begin(),words.end()),words.end());
                last_query.assign(words.size(),-1);query_ids.build(std::move(words));
                for(int t=0;t<n;++t)last_query[query_ids.find(index.q[t])]=t;
                queries_ready=true;
            }
            for(int bit=0;bit<requested_bits;++bit) {
                const int id=query_ids.find(symbol^(std::uint64_t(1)<<bit));
                if(id>=0 && last_query[id]>earliest)return true;
            }
            return false;
        };
        std::size_t ysize=0,tsize=0;
        for(int first=0;first<n;) {
            int after=first+1;
            while(after<n && points[after].symbol==points[first].symbol)++after;
            const int count=after-first;
            int earliest=n;
            for(int j=first;j<after;++j)earliest=std::min(earliest,points[j].source);
            symbols.push_back(points[first].symbol);
            // A class with at most k sources can never enter the heavy phase.
            // Its sorted points suffice for every query; no 2D updates exist.
            // A class never queried after its first source also needs no
            // activation structure. The full-word check includes every
            // requested bit and proves this for the whole causal sweep.
            if(count<=light_threshold || !queried_after(points[first].symbol,earliest)) {
                groups_.push_back(Group{points[first].symbol,first,count,0,earliest,0});
                first=after;continue;
            }
            const int power=power_for(count);
            const std::size_t base=nodes_.size();
            const std::size_t after_nodes=checked_add(base,std::size_t(2)*power);
            if(after_nodes>std::numeric_limits<std::uint32_t>::max())
                throw std::length_error("Q range node identifiers exceed uint32");
            nodes_.resize(after_nodes);
            groups_.push_back(Group{points[first].symbol,first,count,power,earliest,base});
            int depth=1;for(int z=power;z>1;z/=2)++depth;
            for(int j=0;j<count;++j) {
                nodes_[base+power+j].count=1;
                update_start_[std::size_t(points[first+j].source)+1]=std::size_t(depth);
            }
            for(int vertex=power-1;vertex;--vertex)
                nodes_[base+vertex].count=nodes_[base+2*vertex].count+nodes_[base+2*vertex+1].count;
            for(int vertex=1;vertex<2*power;++vertex) {
                Node& node=nodes_[base+vertex];
                if(!node.count)continue;
                node.power=power_for(node.count);node.ybase=ysize;node.tbase=tsize;
                ysize=checked_add(ysize,std::size_t(node.count));
                tsize=checked_add(tsize,std::size_t(2)*node.power);
            }
            first=after;
        }
        group_ids_.build(std::move(symbols));
        ys_.resize(ysize);maximum_.assign(tsize,-1);
        for(const Group& group:groups_) {
            if(!group.power)continue;
            const std::size_t base=group.node_base;
            for(int j=0;j<group.count;++j)
                ys_[nodes_[base+group.power+j].ybase]=points[group.first+j].y;
            for(int vertex=group.power-1;vertex;--vertex) {
                const Node& node=nodes_[base+vertex];
                if(!node.count)continue;
                const Node& a=nodes_[base+2*vertex];const Node& b=nodes_[base+2*vertex+1];
                std::merge(ys_.begin()+a.ybase,ys_.begin()+a.ybase+a.count,
                           ys_.begin()+b.ybase,ys_.begin()+b.ybase+b.count,
                           ys_.begin()+node.ybase);
            }
        }
        for(int source=0;source<n;++source)
            update_start_[source+1]=checked_add(update_start_[source+1],update_start_[source]);
        updates_.resize(update_start_.back());
        std::vector<std::size_t> cursor=update_start_;
        for(std::size_t vertex=0;vertex<nodes_.size();++vertex) {
            const Node& node=nodes_[vertex];
            for(int j=0;j<node.count;++j) {
                const int source=source_by_y[ys_[node.ybase+j]];
                assert(source>=0 && cursor[source]<update_start_[source+1]);
                updates_[cursor[source]++]=Update{std::uint32_t(vertex),std::uint32_t(node.power+j)};
            }
        }
    }

    int group(std::uint64_t symbol) const {
        return group_ids_.find(symbol);
    }
    bool active(int id,int owner) const { return groups_[id].earliest<owner; }
    std::pair<int,int> whole(int id) const {
        const Group& group=groups_[id];return {group.first,group.first+group.count};
    }
    std::pair<int,int> range(int id,std::pair<int,int> ranks) const {
        const Group& group=groups_[id];
        const auto begin=points.begin()+group.first,end=begin+group.count;
        const auto a=std::lower_bound(begin,end,ranks.first,[](const Point& p,int rank){return p.x<rank;});
        const auto b=std::lower_bound(a,end,ranks.second,[](const Point& p,int rank){return p.x<rank;});
        return {int(a-points.begin()),int(b-points.begin())};
    }
    void activate(int source) {
        for(std::size_t at=update_start_[source];at<update_start_[source+1];++at) {
            const Update link=updates_[at];const Node& node=nodes_[link.node];
            for(std::uint32_t vertex=link.leaf;vertex;vertex/=2)
                maximum_[node.tbase+vertex]=source;
        }
    }
    Cover cover(int id,std::pair<int,int> point_range) const {
        const Group& group=groups_[id];Cover result;
        assert(group.power>0);
        int lo=point_range.first-group.first+group.power;
        int hi=point_range.second-group.first+group.power;
        while(lo<hi) {
            if(lo&1) { assert(result.size<int(result.nodes.size()));result.nodes[result.size++]=std::uint32_t(group.node_base+lo++); }
            if(hi&1) { assert(result.size<int(result.nodes.size()));result.nodes[result.size++]=std::uint32_t(group.node_base+--hi); }
            lo/=2;hi/=2;
        }
        return result;
    }
    std::pair<int,int> neighbors(const Cover& cover,int target) const {
        int predecessor=-1,successor=-1;
        for(int i=0;i<cover.size;++i) {
            const Node& node=nodes_[cover.nodes[i]];
            if(maximum_[node.tbase+1]<0)continue;
            const int* first=ys_.data()+node.ybase;
            const int lower=int(std::lower_bound(first,first+node.count,target)-first);
            const int upper=int(std::upper_bound(first+lower,first+node.count,target)-first);
            if(upper && first[upper-1]>predecessor)
                predecessor=std::max(predecessor,last_active(node,0,upper));
            if(lower<node.count && (successor<0 || first[lower]<successor)) {
                const int y=first_active(node,lower,node.count);
                if(y>=0 && (successor<0 || y<successor))successor=y;
            }
        }
        return {predecessor,successor};
    }
    int rectangle_max(const Cover& cover,std::pair<int,int> y_ranks) const {
        int result=-1;
        for(int i=0;i<cover.size;++i) {
            const Node& node=nodes_[cover.nodes[i]];
            const int* tree=maximum_.data()+node.tbase;
            if(tree[1]<=result)continue;
            const int* first=ys_.data()+node.ybase;
            int lo=int(std::lower_bound(first,first+node.count,y_ranks.first)-first)+node.power;
            int hi=int(std::lower_bound(first,first+node.count,y_ranks.second)-first)+node.power;
            while(lo<hi) {
                if(lo&1)result=std::max(result,tree[lo++]);
                if(hi&1)result=std::max(result,tree[--hi]);
                lo/=2;hi/=2;
            }
        }
        return result;
    }
    std::size_t workspace_bytes() const {
        return groups_.capacity()*sizeof(Group)+nodes_.capacity()*sizeof(Node)
            +(ys_.capacity()+maximum_.capacity())*sizeof(int)
            +update_start_.capacity()*sizeof(std::size_t)+updates_.capacity()*sizeof(Update)
            +points.capacity()*sizeof(Point)+group_ids_.workspace_bytes();
    }
};

// At one Q branch, source right-LCPs are monotone on each side of the query
// suffix rank. Select the (threshold+1)-th largest from these two sorted lists
// without reporting their elements. The branch first becomes light one row
// after that LCP. This replaces a per-heavy-row count query by O(log k) RMQs.
inline int first_light_age(const SuffixArray& suffixes,int query_rank,
    const std::vector<Point>& points,std::pair<int,int> whole,int threshold) {
    const int count=whole.second-whole.first;
    if(count<=threshold)return 0;
    const auto first=points.begin()+whole.first,last=points.begin()+whole.second;
    const int middle=int(std::lower_bound(first,last,query_rank,
        [](const Point& p,int rank){return p.x<rank;})-points.begin());
    const int left_count=middle-whole.first,right_count=whole.second-middle;
    const int order=threshold+1;
    const int infinity=std::numeric_limits<int>::max();
    auto left=[&](int offset){return suffixes.lcp_rank(query_rank,points[middle-1-offset].x);};
    auto right=[&](int offset){return suffixes.lcp_rank(query_rank,points[middle+offset].x);};
    int lo=std::max(0,order-right_count),hi=std::min(order,left_count);
    while(lo<=hi) {
        const int a=lo+(hi-lo)/2,b=order-a;
        const int al=a?left(a-1):infinity,ar=a<left_count?left(a):-1;
        const int bl=b?right(b-1):infinity,br=b<right_count?right(b):-1;
        if(al<br)hi=a-1;
        else if(bl<ar)lo=a+1;
        else return std::min(al,bl)+1;
    }
    throw std::logic_error("invalid Q LCP order statistic");
}

inline int one_bit_index(std::uint64_t value) {
    assert(value && !(value&(value-1)));
    int bit=0;
    while(!(value&1)){value>>=1;++bit;}
    return bit;
}

} // namespace repair_detail

inline void q_repair(const Index& index,const Scores& scores,int bits,int threshold,
    std::vector<double>& heavy_positive,std::vector<Segment>& light,
    std::vector<BitRange>& exists,Stats& stats) {
    const int n=index.n;
    if(bits<1 || bits>64 || threshold<1 || scores.n!=n || scores.d<1
       || heavy_positive.size()!=std::size_t(n)*bits)
        throw std::invalid_argument("invalid Q repair dimensions");
    // Small classes are cheaper to resolve as complete envelopes than as
    // dynamic 2D trees. All leaf scans cost O(nb sqrt(n)) structural work;
    // heavy score rows and at most nb*threshold light records are unchanged.
    const int leaf_capacity=std::max(64,int(std::sqrt(double(n))));
    repair_detail::FrontierIndex tree(index,std::max(threshold,leaf_capacity),bits);
    stats.peak_workspace_bytes=std::max(stats.peak_workspace_bytes,std::uint64_t(tree.workspace_bytes()));
    const auto& right=index.forward_suffixes();const auto& left=index.reverse_suffixes();
    struct Candidate {int left,source,age;};
    std::vector<Candidate> candidates;
    candidates.reserve(std::min(std::max(threshold,leaf_capacity),n));
    const std::size_t old_segments=light.size();
    for(int owner=0;owner<n;++owner) {
        if(owner)tree.activate(owner-1);
        const int xq=right.rank[owner+1],yq=left.rank[n-owner];
        for(int bit=0;bit<bits;++bit) {
            const int group=tree.group(index.q[owner]^(std::uint64_t(1)<<bit));
            if(group<0 || !tree.active(group,owner))continue;
            const auto whole=tree.whole(group);
            if(whole.second-whole.first<=leaf_capacity) {
                candidates.clear();
                const int count=whole.second-whole.first;
                const int middle=int(std::lower_bound(tree.points.begin()+whole.first,tree.points.begin()+whole.second,xq,
                    [](const repair_detail::Point& point,int rank){return point.x<rank;})-tree.points.begin());
                int a=middle-1,b=middle,at_tail=-1,before_tail=0;
                int al=a>=whole.first?right.lcp_rank(xq,tree.points[a].x):-1;
                int bl=b<whole.second?right.lcp_rank(xq,tree.points[b].x):-1;
                int light_age=0,light_sources=count<=threshold?count:0;
                Candidate best{-1,-1,-1};
                // Right-LCP lifetimes decrease away from xq on either side.
                // Merge them from longest to shortest. A new priority maximum
                // records exactly one winner of the nested surviving sets.
                for(int seen=0;seen<count;++seen) {
                    const int at=al>=bl?a:b,tail=std::max(al,bl);
                    if(tail!=at_tail){at_tail=tail;before_tail=seen;}
                    if(seen==threshold){light_age=tail+1;light_sources=before_tail;}
                    const auto& point=tree.points[at];
                    if(point.source<owner) {
                        const int ell=left.lcp_rank(yq,point.y);
                        if(ell>best.left || (ell==best.left && point.source>best.source)) {
                            best=Candidate{ell,point.source,tail};candidates.push_back(best);
                        }
                    }
                    if(al>=bl){--a;al=a>=whole.first?right.lcp_rank(xq,tree.points[a].x):-1;}
                    else{++b;bl=b<whole.second?right.lcp_rank(xq,tree.points[b].x):-1;}
                }
                assert(light_age<=n-owner);
                int far=-1;double& total=heavy_positive[std::size_t(owner)*bits+bit];
                for(auto it=candidates.rbegin();it!=candidates.rend();++it)if(it->age>far) {
                    const Candidate& candidate=*it;
                    const int last_heavy=std::min(candidate.age,light_age-1);
                    for(int h=far+1;h<=last_heavy;++h)total+=scores.score(owner+h,candidate.source+h);
                    if(last_heavy>far)stats.q_heavy_rows+=std::uint64_t(last_heavy-far);
                    const int first_light=std::max(far+1,light_age);
                    if(first_light<=candidate.age)
                        light.push_back(Segment{owner,bit,owner+first_light,owner+candidate.age,
                                                candidate.left+1-owner,candidate.source-owner});
                    far=candidate.age;
                }
                if(far+1>=light_age && light_age<n-owner)
                    stats.q_light_sources+=std::uint64_t(light_sources);
                if(far>=0)exists.push_back(BitRange{owner,bit,owner,owner+far});
                continue;
            }
            const int light_age=repair_detail::first_light_age(right,xq,tree.points,whole,threshold);
            assert(light_age<=n-owner);
            int age=0,last=-1;
            // A fixed winner is valid through its right LCE. All of these
            // heavy rows are still charged/projected directly, but the 2D
            // winner lookup is shared by the entire run.
            while(age<light_age) {
                const auto xr=right.prefix_interval(xq,age);
                const auto cover=tree.cover(group,tree.range(group,xr));
                const auto near=tree.neighbors(cover,yq);
                if(near.first<0 && near.second<0)break;
                int ell=0;
                if(near.first>=0)ell=left.lcp_rank(yq,near.first);
                if(near.second>=0)ell=std::max(ell,left.lcp_rank(yq,near.second));
                const int source=tree.rectangle_max(cover,left.prefix_interval(yq,ell));
                assert(source>=0 && source<owner);
                const int finish=std::min(index.lce(owner+1,source+1),light_age-1);
                assert(finish>=age);
                double& total=heavy_positive[std::size_t(owner)*bits+bit];
                for(int h=age;h<=finish;++h)total+=scores.score(owner+h,source+h);
                stats.q_heavy_rows+=std::uint64_t(finish-age+1);
                last=finish;age=finish+1;
            }
            if(age==light_age && age<n-owner) {
                const auto bounds=age?tree.range(group,right.prefix_interval(xq,age)):tree.whole(group);
                assert(bounds.second-bounds.first<=threshold);
                stats.q_light_sources+=std::uint64_t(bounds.second-bounds.first);
                candidates.clear();
                for(int at=bounds.first;at<bounds.second;++at) {
                    const auto& point=tree.points[at];
                    if(point.source>=owner)continue;
                    const int ell=left.lcp_rank(yq,point.y),tail=right.lcp_rank(xq,point.x);
                    assert(tail>=age && tail<n-owner);
                    candidates.push_back(Candidate{ell,point.source,tail});
                }
                std::sort(candidates.begin(),candidates.end(),[](const Candidate& a,const Candidate& b){
                    return a.left>b.left || (a.left==b.left && a.source>b.source);
                });
                int far=age-1;
                for(const Candidate& candidate:candidates)if(candidate.age>far) {
                    light.push_back(Segment{owner,bit,owner+far+1,owner+candidate.age,
                                            candidate.left+1-owner,candidate.source-owner});
                    far=candidate.age;
                }
                last=std::max(last,far);
            }
            if(last>=0)exists.push_back(BitRange{owner,bit,owner,owner+last});
        }
    }
    stats.q_segments+=std::uint64_t(light.size()-old_segments);
}

inline std::vector<Segment> long_k(const Index& index,int anchor,Stats& stats) {
    if(anchor<1)throw std::invalid_argument("long K anchor must be positive");
    const int n=index.n;
    const std::int64_t wide_cutoff=std::int64_t(anchor)*4;
    if(wide_cutoff>=n)return {};
    const int cutoff=int(wide_cutoff);
    std::vector<std::uint64_t> centers;
    std::vector<std::uint64_t> recent_centers;
    centers.reserve(n);
    auto remember=[&](int query,int key){
        if(key<0 || query<=key || query>=n)return;
        const std::uint64_t difference=index.q[query]^index.k[key];
        if(!difference || (difference&(difference-1)))return;
        const std::uint64_t center=(std::uint64_t(std::uint32_t(query))<<32)|std::uint32_t(key);
        if(recent_centers.empty()) {
            std::size_t slots=1;while(slots<std::size_t(n)*2)slots*=2;
            recent_centers.assign(slots,0);
        }
        // Every stored pair has query>key>=0, so its encoded value is nonzero.
        // A single bounded probe suppresses repeated anchors for that exact
        // pair. Collisions merely forget work; the final sort still deduplicates.
        const std::size_t at=std::size_t(std::uint64_t(query)*UINT64_C(0x9e3779b97f4a7c15)
            +std::uint64_t(key)*UINT64_C(0xd6e8feb86659fd93))&(recent_centers.size()-1);
        if(recent_centers[at]==center)return;
        recent_centers[at]=center;
        // The adjacent mismatch can support a long repair only if both exact
        // sides together exceed the cutoff. Reject this structural zero
        // before storing and sorting it, rather than after materialization.
        if(index.lcs(query-1,key-1)+1+index.lce(query+1,key+1)<=cutoff)return;
        centers.push_back(center);
    };
    // Every capacity > 4B has an exact side of length >= 2B, which contains a
    // Q-grid B-anchor. Enumerating its suffix interval directly avoids testing
    // absent K occurrences. Each interval has at most 2n+2 ranks, so this keeps
    // the deterministic O(n^2/B) structural bound for every input. Extending
    // an exact occurrence finds the adjacent mismatch.
    const auto& suffixes=index.forward_suffixes();
    for(int query=0;query<=n-anchor;query+=anchor) {
        const auto ranks=suffixes.prefix_interval(suffixes.rank[query],anchor);
        for(int rank=ranks.first;rank<ranks.second;++rank) {
            const int key=suffixes.sa[rank]-(n+1);
            if(key<0 || key>=query)continue;
            ++stats.k_anchor_probes;
            const int before=index.lcs(query-1,key-1);
            const int after=index.lce(query+anchor,key+anchor);
            remember(query-before-1,key-before-1);
            remember(query+anchor+after,key+anchor+after);
        }
    }
    std::sort(centers.begin(),centers.end());
    centers.erase(std::unique(centers.begin(),centers.end()),centers.end());
    stats.k_centers+=centers.size();
    std::vector<Segment> corridors;corridors.reserve(centers.size());
    for(std::uint64_t center:centers) {
        const int query=int(center>>32),key=int(std::uint32_t(center));
        const int before=index.lcs(query-1,key-1),after=index.lce(query+1,key+1);
        int lo=std::max(query,query+cutoff-before);const int hi=query+after;
        if(lo>hi)continue;
        const int alpha=before+1-query,beta=key-query;
        auto wins=[&](int t){return Route{t+alpha,t+beta}>index.deleted(t,key);};
        if(!wins(hi))continue;
        int end=hi;
        while(lo<end) {
            const int middle=lo+(end-lo)/2;
            if(wins(middle))end=middle;else lo=middle+1;
        }
        const int bit=repair_detail::one_bit_index(index.q[query]^index.k[key]);
        corridors.push_back(Segment{key,bit,lo,hi,alpha,beta});
    }
    std::sort(corridors.begin(),corridors.end(),[](const Segment& a,const Segment& b){
        return std::tie(a.owner,a.bit,a.lo,a.hi,a.alpha,a.beta)
             < std::tie(b.owner,b.bit,b.lo,b.hi,b.alpha,b.beta);
    });
    struct Event {int time,id;bool start;};
    std::vector<Event> events;
    std::vector<int> heap;
    std::vector<unsigned char> active;
    std::vector<Segment> output;output.reserve(corridors.size());
    for(std::size_t first=0;first<corridors.size();) {
        std::size_t last=first+1;
        while(last<corridors.size() && corridors[last].owner==corridors[first].owner
            && corridors[last].bit==corridors[first].bit)++last;
        const std::size_t count=last-first;
        if(count>std::size_t(std::numeric_limits<int>::max()))
            throw std::length_error("long K envelope exceeds int32 ids");
        events.clear();heap.clear();active.assign(count,0);
        for(int id=0;id<int(count);++id) {
            const Segment& segment=corridors[first+id];
            events.push_back(Event{segment.lo,id,true});
            events.push_back(Event{segment.hi+1,id,false});
        }
        std::sort(events.begin(),events.end(),[](const Event& a,const Event& b){
            return std::tie(a.time,a.id,a.start)<std::tie(b.time,b.id,b.start);
        });
        auto less=[&](int a,int b){
            const Segment& x=corridors[first+a];const Segment& y=corridors[first+b];
            return std::tie(x.alpha,x.beta,a)<std::tie(y.alpha,y.beta,b);
        };
        std::size_t at=0;
        while(at<events.size()) {
            const int time=events[at].time;
            do {
                const Event event=events[at++];active[event.id]=event.start;
                if(event.start){heap.push_back(event.id);std::push_heap(heap.begin(),heap.end(),less);}
            }while(at<events.size() && events[at].time==time);
            while(!heap.empty() && !active[heap.front()]) {
                std::pop_heap(heap.begin(),heap.end(),less);heap.pop_back();
            }
            if(at<events.size() && !heap.empty()) {
                Segment current=corridors[first+heap.front()];
                current.lo=time;current.hi=events[at].time-1;
                if(!output.empty()) {
                    Segment& previous=output.back();
                    if(previous.owner==current.owner && previous.bit==current.bit
                        && previous.hi+1==current.lo && previous.alpha==current.alpha
                        && previous.beta==current.beta) {
                        previous.hi=current.hi;continue;
                    }
                }
                output.push_back(current);
            }
        }
        first=last;
    }
    stats.k_segments+=output.size();
    const std::size_t bytes=(centers.capacity()+recent_centers.capacity())*sizeof(std::uint64_t)
        +(corridors.capacity()+output.capacity())*sizeof(Segment)
        +events.capacity()*sizeof(Event)+heap.capacity()*sizeof(int)+active.capacity();
    stats.peak_workspace_bytes=std::max(stats.peak_workspace_bytes,std::uint64_t(bytes));
    return output;
}

} // namespace rosa_subq
