#pragma once
// Compact short K repair. Each dictionary has O(L) virtual edited words;
// payload-constant suffix-rank pieces are summed by offline 3D/4D dominance.
// There is no owner-by-time route table and no L scan per exceptional cell.
#include "index.h"
#include "types.h"
#include <array>
#include <cassert>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace rosa_subq {
namespace short_detail {

struct Word {
    std::uint64_t mask=0;
    int start=0,length=0,edit=-1,end=-1;
};

// Prefix classes of a sorted string list. Adjacent LCPs form a Cartesian
// compact trie; a weighted-ancestor query then returns a whole prefix range
// using integer ancestors, without comparing virtual strings at every step
// of a binary search. All arrays are reused between bounded-size batches.
class PrefixIntervals {
    std::vector<int> depth_,parent_,lo_,hi_,left_,right_,stack_,order_;
    std::vector<std::vector<int>> up_;
    int count_=0;
public:
    template<class Length,class Common>
    void build(int count,const Length& length,const Common& common) {
        count_=count;if(!count)return;
        const int nodes=2*count-1;
        depth_.assign(nodes,0);parent_.assign(nodes,-1);lo_.resize(nodes);hi_.resize(nodes);
        left_.assign(count,-1);right_.assign(count,-1);stack_.clear();
        for(int i=0;i<count;++i){depth_[i]=length(i);lo_[i]=i;hi_[i]=i+1;}
        for(int i=1;i<count;++i) {
            const int current=count+i-1;depth_[current]=common(i-1,i);
            int last=-1;
            while(!stack_.empty() && depth_[current]<depth_[count+stack_.back()-1]) {
                last=stack_.back();stack_.pop_back();
            }
            left_[i]=last;if(!stack_.empty())right_[stack_.back()]=i;stack_.push_back(i);
        }
        const int root=count==1?0:count+stack_.front()-1;
        for(int i=1;i<count;++i) {
            const int node=count+i-1;
            const int a=left_[i]<0?i-1:count+left_[i]-1;
            const int b=right_[i]<0?i:count+right_[i]-1;
            parent_[a]=parent_[b]=node;
        }
        order_.clear();order_.push_back(root);
        for(std::size_t at=0;at<order_.size();++at)if(order_[at]>=count) {
            const int i=order_[at]-count+1;
            order_.push_back(left_[i]<0?i-1:count+left_[i]-1);
            order_.push_back(right_[i]<0?i:count+right_[i]-1);
        }
        for(auto it=order_.rbegin();it!=order_.rend();++it)if(*it>=count) {
            const int i=*it-count+1;
            const int a=left_[i]<0?i-1:count+left_[i]-1;
            const int b=right_[i]<0?i:count+right_[i]-1;
            lo_[*it]=lo_[a];hi_[*it]=hi_[b];
        }
        int levels=1;while((std::size_t(1)<<levels)<std::size_t(nodes))++levels;
        up_.resize(levels);up_[0]=parent_;
        for(int level=1;level<levels;++level) {
            up_[level].resize(nodes);
            for(int i=0;i<nodes;++i){const int p=up_[level-1][i];up_[level][i]=p<0?-1:up_[level-1][p];}
        }
    }
    std::pair<int,int> interval(int leaf,int depth) const {
        assert(leaf>=0 && leaf<count_ && depth_[leaf]>=depth);
        int node=leaf;
        // Most competitive short prefixes need only one or two ancestors.
        // Bound the direct climb so repeated strings still take O(log n).
        for(int step=0;step<4;++step) {
            const int p=parent_[node];
            if(p<0 || depth_[p]<depth)return {lo_[node],hi_[node]};
            node=p;
        }
        for(int level=int(up_.size())-1;level>=0;--level) {
            const int p=up_[level][node];if(p>=0 && depth_[p]>=depth)node=p;
        }
        return {lo_[node],hi_[node]};
    }
    std::size_t bytes() const {
        std::size_t bytes=(depth_.capacity()+parent_.capacity()+lo_.capacity()+hi_.capacity()
            +left_.capacity()+right_.capacity()+stack_.capacity()+order_.capacity())*sizeof(int);
        for(const auto& row:up_)bytes+=row.capacity()*sizeof(int);
        return bytes;
    }
};

// The integer suffix array contains separate sentinels; virtual characters
// are compared as their original uint64 values. In particular, UINT64_MAX
// and bit 63 never pass through an overflowing symbol-plus-two operation.
class VirtualIndex {
    const SuffixArray& suffix_;
    std::vector<std::uint64_t> letters_;
    struct Neighbors {int position,left,right;};
    PrefixIntervals query_prefixes_,word_prefixes_;
    std::vector<Neighbors> word_neighbors_,query_neighbors_;
    std::vector<int> word_ranks_;
    const Word* prepared_words_=nullptr;
    bool query_prefixes_ready_=false;
public:
    int n;
    std::vector<Word> query_words;
    std::vector<int> row_rank;
    explicit VirtualIndex(const Index& index)
      : suffix_(index.reverse_suffixes()),letters_(std::size_t(index.n)*2+2),n(index.n),row_rank(n) {
        for(int i=0;i<n;++i) {
            letters_[i]=index.q[n-1-i];
            letters_[n+1+i]=index.k[n-1-i];
        }
        query_words.reserve(n);
        for(int p:suffix_.sa)if(p<n) {
            const int t=n-1-p;
            row_rank[t]=int(query_words.size());
            query_words.push_back(Word{0,p,t+1,-1,-1});
        }
    }
    Word row(int t) const { return Word{0,n-1-t,t+1,-1,-1}; }
    Word edited(int owner,int bit,int end,int cutoff) const {
        return Word{std::uint64_t(1)<<bit,2*n-end,std::min(cutoff+1,end+1),end-owner,end};
    }
    std::uint64_t letter(const Word& word,int offset) const {
        return letters_[word.start+offset]^(offset==word.edit?word.mask:0);
    }
    int common(const Word& a,const Word& b) const {
        const int cap=std::min(a.length,b.length);
        int at=0;
        while(at<cap) {
            int stop=cap;
            if(a.edit>=at)stop=std::min(stop,a.edit);
            if(b.edit>=at)stop=std::min(stop,b.edit);
            if(stop>at) {
                const int count=a.start==b.start?stop-at:
                    std::min(stop-at,suffix_.lcp_rank(suffix_.rank[a.start+at],suffix_.rank[b.start+at]));
                at+=count;
                if(at<stop)return at;
            }
            if(at==cap)return at;
            if(letter(a,at)!=letter(b,at))return at;
            ++at;
        }
        return at;
    }
    int compare(const Word& a,const Word& b) const {
        const int length=common(a,b),cap=std::min(a.length,b.length);
        if(length==cap)return (a.length>b.length)-(a.length<b.length);
        const auto x=letter(a,length),y=letter(b,length);
        return (x>y)-(x<y);
    }
    int prefix_compare(const Word& a,const Word& b,int depth) const {
        const int length=std::min(depth,common(a,b));
        if(length==depth)return 0;
        if(length==a.length)return -1;
        const auto x=letter(a,length),y=letter(b,length);
        return (x>y)-(x<y);
    }
    template<class Get>
    std::pair<int,int> interval(int count,const Get& get,const Word& prefix,int depth,
                                int begin=0,int finish=-1) const {
        if(depth>prefix.length)return {0,0};
        if(finish<0)finish=count;
        int lo=begin,hi=finish;
        while(lo<hi) {
            const int mid=lo+(hi-lo)/2;
            if(prefix_compare(get(mid),prefix,depth)<0)lo=mid+1;
            else hi=mid;
        }
        const int first=lo;hi=finish;
        while(lo<hi) {
            const int mid=lo+(hi-lo)/2;
            if(prefix_compare(get(mid),prefix,depth)<=0)lo=mid+1;
            else hi=mid;
        }
        return {first,lo};
    }
    void prepare(const std::vector<Word>& words,const std::vector<int>& sorted) {
        const int count=int(words.size());prepared_words_=words.data();
        word_neighbors_.resize(count);query_neighbors_.resize(n);
        word_ranks_.resize(count);for(int at=0;at<count;++at)word_ranks_[sorted[at]]=at;
        if(!query_prefixes_ready_) {
            query_prefixes_.build(n,[&](int i){return query_words[i].length;},
                [&](int a,int b){return common(query_words[a],query_words[b]);});
            query_prefixes_ready_=true;
        }
        word_prefixes_.build(count,[&](int i){return words[sorted[i]].length;},
            [&](int a,int b){return common(words[sorted[a]],words[sorted[b]]);});
        int q=0,w=0;
        while(q<n || w<count) {
            if(w<count && (q==n || compare(words[sorted[w]],query_words[q])<=0)) {
                const int id=sorted[w++];const Word& word=words[id];
                word_neighbors_[id]={q,q?common(word,query_words[q-1]):0,q<n?common(word,query_words[q]):0};
            } else {
                const Word& query=query_words[q];
                query_neighbors_[q++]={w,w?common(query,words[sorted[w-1]]):0,w<count?common(query,words[sorted[w]]):0};
            }
        }
    }
    std::pair<int,int> word_interval(int row,int depth) const {
        if(depth>row+1)return {0,0};
        if(depth<=0)return {0,int(word_neighbors_.size())};
        const Neighbors& at=query_neighbors_[row_rank[row]];
        if(at.right>=depth)return word_prefixes_.interval(at.position,depth);
        if(at.left>=depth)return word_prefixes_.interval(at.position-1,depth);
        return {at.position,at.position};
    }
    bool prepared(const std::vector<Word>& words) const {
        return prepared_words_==words.data() && word_ranks_.size()==words.size();
    }
    int word_rank(int word) const {return word_ranks_[word];}
    int row_word_position(int row) const {return query_neighbors_[row_rank[row]].position;}
    std::pair<int,int> query_interval(const Word& word,int depth,int id=-1) const {
        if(depth>word.length)return {0,0};
        if(depth<=0)return {0,n};
        if(id>=0 && id<int(word_neighbors_.size()) && &word==prepared_words_+id) {
            const Neighbors& at=word_neighbors_[id];
            if(at.right>=depth)return query_prefixes_.interval(at.position,depth);
            if(at.left>=depth)return query_prefixes_.interval(at.position-1,depth);
            return {at.position,at.position};
        }
        return interval(n,[&](int p)->const Word&{return query_words[p];},word,depth);
    }
    int maximum_query_match(const Word& word,int id,int insertion) const {
        if(id>=0 && id<int(word_neighbors_.size()) && &word==prepared_words_+id)
            return std::max(word_neighbors_[id].left,word_neighbors_[id].right);
        int longest=0;
        if(insertion)longest=common(word,query_words[insertion-1]);
        if(insertion<n)longest=std::max(longest,common(word,query_words[insertion]));
        return longest;
    }
    std::size_t bytes() const {
        return letters_.capacity()*sizeof(std::uint64_t)+query_words.capacity()*sizeof(Word)+row_rank.capacity()*sizeof(int)
            +(word_neighbors_.capacity()+query_neighbors_.capacity())*sizeof(Neighbors)
            +word_ranks_.capacity()*sizeof(int)
            +query_prefixes_.bytes()+word_prefixes_.bytes();
    }
};

struct Piece { int lo,hi,word,ceiling; };
struct Paint { int lo,hi,depth,word,ceiling; };

// Output-sensitive sparse/dense decision. A large lexicographic interval
// may contain only a few dictionaries whose endpoints are already causal.
// Report at most seventeen such ranks, in O((reported+1) log n) time; the
// seventeenth switches to the ordinary dense range-sum path.
class CausalWords {
    int base_=1;
    std::vector<int> first_time_;
    void visit(int node,int lo,int hi,int qlo,int qhi,int time,int* out,int limit,int& count) const {
        if(count==limit || lo>=qhi || hi<=qlo || first_time_[node]>time)return;
        if(hi-lo==1){out[count++]=lo;return;}
        const int mid=lo+(hi-lo)/2;
        visit(2*node,lo,mid,qlo,qhi,time,out,limit,count);
        visit(2*node+1,mid,hi,qlo,qhi,time,out,limit,count);
    }
public:
    template<class Get>void reset(int count,const Get& first_time) {
        base_=1;while(base_<count)base_*=2;
        first_time_.assign(std::size_t(base_)*2,std::numeric_limits<int>::max());
        for(int i=0;i<count;++i)first_time_[base_+i]=first_time(i);
        for(int i=base_-1;i>0;--i)first_time_[i]=std::min(first_time_[2*i],first_time_[2*i+1]);
    }
    int report(int lo,int hi,int time,int* out,int limit) const {
        int count=0;visit(1,0,base_,lo,hi,time,out,limit,count);return count;
    }
    std::size_t bytes() const {return first_time_.capacity()*sizeof(int);}
};

// A trie edge ending at depth h can win only with length <=h: any longer
// match would have entered a child edge. A range minimum of the original
// routes therefore rejects an entire noncompetitive piece before dominance.
class MinimumRoutes {
    int base_=1;
    std::vector<Route> tree_;
    static Route lesser(const Route& a,const Route& b){return b<a?b:a;}
public:
    MinimumRoutes(const std::vector<Route>& routes,const std::vector<int>& rank,
                  const std::vector<unsigned char>& active) {
        while(base_<int(routes.size()))base_*=2;
        tree_.assign(std::size_t(base_)*2,Route{std::numeric_limits<int>::max(),std::numeric_limits<int>::max()});
        for(int t=1;t<int(routes.size());++t)if(active[t])tree_[base_+rank[t]]=routes[t];
        for(int i=base_-1;i>0;--i)tree_[i]=lesser(tree_[2*i],tree_[2*i+1]);
    }
    Route query(int lo,int hi) const {
        Route result{std::numeric_limits<int>::max(),std::numeric_limits<int>::max()};
        for(lo+=base_,hi+=base_;lo<hi;lo/=2,hi/=2) {
            if(lo&1)result=lesser(result,tree_[lo++]);
            if(hi&1)result=lesser(result,tree_[--hi]);
        }
        return result;
    }
    std::size_t bytes() const {return tree_.capacity()*sizeof(Route);}
};

// Only right contexts that can actually follow a causal one-bit center can
// make a virtual endpoint useful. Group Q[u+1:] by Q[u], in suffix order.
// The nearest predecessor/successor with u>owner maximizes LCE with K[owner+1:].
// A single flat range-maximum tree supplies those two qualified neighbors;
// no dynamic sets or per-symbol tree allocations are needed.
class SupportedWords {
    struct Center {std::uint64_t symbol;int rank,position;};
    struct Group {std::uint64_t symbol;int lo,hi,latest;};
    const SuffixArray& suffix_;
    int n_,base_=1;
    std::vector<Center> centers_;
    std::vector<Group> groups_;
    std::vector<int> maximum_,lookup_;
    static std::uint64_t hash(std::uint64_t value) {
        value^=value>>30;value*=UINT64_C(0xbf58476d1ce4e5b9);
        value^=value>>27;value*=UINT64_C(0x94d049bb133111eb);
        return value^(value>>31);
    }
    const Group* find_group(std::uint64_t symbol) const {
        const int id=lookup_[hash(symbol)&(lookup_.size()-1)];
        if(id>=0)return groups_[id].symbol==symbol?&groups_[id]:nullptr;
        if(id==-1)return nullptr;
        // Collision buckets retain a sorted-search fallback. An empty or
        // single-key bucket proves absence without probing unrelated groups;
        // a collision can change work but cannot decide equality or absence.
        const auto found=std::lower_bound(groups_.begin(),groups_.end(),symbol,
            [](const Group& a,std::uint64_t b){return a.symbol<b;});
        return found!=groups_.end()&&found->symbol==symbol?&*found:nullptr;
    }
    int last(int node,int lo,int hi,int qlo,int qhi,int owner) const {
        if(lo>=qhi || hi<=qlo || maximum_[node]<=owner)return -1;
        if(qlo<=lo && hi<=qhi) {
            while(node<base_)node=maximum_[2*node+1]>owner?2*node+1:2*node;
            return node-base_;
        }
        const int mid=lo+(hi-lo)/2;
        const int r=last(2*node+1,mid,hi,qlo,qhi,owner);
        return r>=0?r:last(2*node,lo,mid,qlo,qhi,owner);
    }
    int first(int node,int lo,int hi,int qlo,int qhi,int owner) const {
        if(lo>=qhi || hi<=qlo || maximum_[node]<=owner)return -1;
        if(qlo<=lo && hi<=qhi) {
            while(node<base_)node=maximum_[2*node]>owner?2*node:2*node+1;
            return node-base_;
        }
        const int mid=lo+(hi-lo)/2;
        const int l=first(2*node,lo,mid,qlo,qhi,owner);
        return l>=0?l:first(2*node+1,mid,hi,qlo,qhi,owner);
    }
public:
    SupportedWords(const Index& index,int last_active)
      :suffix_(index.forward_suffixes()),n_(index.n) {
        centers_.reserve(last_active+1);
        for(int u=0;u<=last_active;++u)centers_.push_back(Center{index.q[u],suffix_.rank[u+1],u});
        std::sort(centers_.begin(),centers_.end(),[](const Center& a,const Center& b){
            return a.symbol!=b.symbol?a.symbol<b.symbol:a.rank<b.rank;
        });
        for(int i=0;i<int(centers_.size());++i) {
            const Center& c=centers_[i];
            if(groups_.empty() || groups_.back().symbol!=c.symbol)groups_.push_back(Group{c.symbol,i,i+1,c.position});
            else {groups_.back().hi=i+1;groups_.back().latest=std::max(groups_.back().latest,c.position);}
        }
        std::size_t slots=8;while(slots<groups_.size()*4)slots*=2;
        lookup_.assign(slots,-1);
        for(int i=0;i<int(groups_.size());++i) {
            int& slot=lookup_[hash(groups_[i].symbol)&(slots-1)];
            slot=slot==-1?i:-2;
        }
        while(base_<int(centers_.size()))base_*=2;
        maximum_.assign(std::size_t(base_)*2,-1);
        for(int i=0;i<int(centers_.size());++i)maximum_[base_+i]=centers_[i].position;
        for(int i=base_-1;i>0;--i)maximum_[i]=std::max(maximum_[2*i],maximum_[2*i+1]);
    }
    int right_extension(std::uint64_t symbol,int owner) const {
        const Group* group=find_group(symbol);
        if(!group || group->latest<=owner)return -1;
        const int target=suffix_.rank[n_+owner+2];
        const int split=int(std::lower_bound(centers_.begin()+group->lo,centers_.begin()+group->hi,target,
            [](const Center& a,int b){return a.rank<b;})-centers_.begin());
        const int before=last(1,0,base_,group->lo,split,owner);
        const int after=first(1,0,base_,split,group->hi,owner);
        int result=0;
        if(before>=0)result=suffix_.lcp_rank(centers_[before].rank,target);
        if(after>=0)result=std::max(result,suffix_.lcp_rank(centers_[after].rank,target));
        return std::min(result,n_-owner-1);
    }
    int sparse_centers(std::uint64_t symbol,int owner,int* output,int limit) const {
        const Group* group=find_group(symbol);
        if(!group || group->latest<=owner)return 0;
        if(group->hi-group->lo>limit)return -1;
        int count=0;
        for(int i=group->lo;i<group->hi;++i)
            if(centers_[i].position>owner)output[count++]=centers_[i].position;
        return count;
    }
    std::size_t bytes() const {
        return centers_.capacity()*sizeof(Center)+groups_.capacity()*sizeof(Group)
            +(maximum_.capacity()+lookup_.capacity())*sizeof(int);
    }
};

// The Cartesian tree of adjacent LCPs is the binary compact trie. Constructing
// it linearly avoids an RMQ query and a tree allocation for every trie node.
class CompactTrie {
    struct Node { int left=-1,right=-1,parent=-1,depth=0,word=-1; };
    std::vector<Node> nodes_;
    std::vector<int> adjacent_,left_,right_,stack_,order_,heap_;
    std::vector<Paint> paints_;
    std::vector<std::pair<int,int>> bounds_;
public:
    std::vector<Piece> pieces;
    void build(const VirtualIndex& ix,const std::vector<Word>& words,const int* sorted,
               int count,int cutoff,std::vector<int>& mask_lo,std::vector<int>& mask_hi,
               bool crossing_only=false) {
        pieces.clear();paints_.clear();bounds_.clear();heap_.clear();
        if(!count)return;
        adjacent_.assign(count,0);left_.assign(count,-1);right_.assign(count,-1);
        nodes_.assign(std::size_t(count)*2-1,Node{});
        for(int i=0;i<count;++i) {
            nodes_[i].word=sorted[i];nodes_[i].depth=words[sorted[i]].length;
            if(i)adjacent_[i]=ix.common(words[sorted[i-1]],words[sorted[i]]);
        }
        stack_.clear();
        for(int i=1;i<count;++i) {
            int last=-1;
            while(!stack_.empty() && adjacent_[i]<adjacent_[stack_.back()]) {
                last=stack_.back();stack_.pop_back();
            }
            left_[i]=last;
            if(!stack_.empty())right_[stack_.back()]=i;
            stack_.push_back(i);
        }
        const int root=count==1?0:count+stack_.front()-1;
        for(int i=1;i<count;++i) {
            const int p=count+i-1;
            const int l=left_[i]<0?i-1:count+left_[i]-1;
            const int r=right_[i]<0?i:count+right_[i]-1;
            nodes_[p].left=l;nodes_[p].right=r;nodes_[p].depth=adjacent_[i];
            nodes_[l].parent=p;nodes_[r].parent=p;
        }
        order_.clear();order_.push_back(root);
        for(std::size_t i=0;i<order_.size();++i) {
            const Node& node=nodes_[order_[i]];
            if(node.left>=0){order_.push_back(node.left);order_.push_back(node.right);}
        }
        for(auto it=order_.rbegin();it!=order_.rend();++it) {
            Node& node=nodes_[*it];
            if(node.left>=0) {
                const int a=nodes_[node.left].word,b=nodes_[node.right].word;
                node.word=words[a].end>words[b].end?a:b;
            }
            const int parent_depth=node.parent<0?0:nodes_[node.parent].depth;
            // A prefix ending before the edited character is also available
            // to DelK. It cannot contribute repair credit. Omitting that
            // part of an edge is safe even if it exposes another candidate:
            // the omitted unchanged winner still lower-bounds DelK.
            const int first_depth=crossing_only?std::max(parent_depth+1,words[node.word].edit+1):parent_depth+1;
            if(node.depth<first_depth)continue;
            const auto interval=ix.query_interval(words[node.word],first_depth,node.word);
            if(interval.first==interval.second)continue;
            const int id=int(paints_.size());
            paints_.push_back(Paint{interval.first,interval.second,parent_depth+1,node.word,std::min(node.depth,cutoff)});
            bounds_.push_back({interval.first,id});bounds_.push_back({interval.second,-1});
        }
        std::sort(bounds_.begin(),bounds_.end());
        const auto lower_priority=[&](int a,int b) {
            if(paints_[a].depth!=paints_[b].depth)return paints_[a].depth<paints_[b].depth;
            const int x=words[paints_[a].word].end,y=words[paints_[b].word].end;
            return x!=y?x<y:a<b;
        };
        const auto append=[&](int lo,int hi,int word,int ceiling) {
            if(lo>=hi)return;
            if(!pieces.empty() && pieces.back().hi==lo && pieces.back().word==word && pieces.back().ceiling==ceiling)
                pieces.back().hi=hi;
            else pieces.push_back(Piece{lo,hi,word,ceiling});
        };
        std::size_t at=0;
        while(at<bounds_.size()) {
            const int lo=bounds_[at].first;
            do {
                const int id=bounds_[at].second;
                if(id>=0){heap_.push_back(id);std::push_heap(heap_.begin(),heap_.end(),lower_priority);}
                ++at;
            }while(at<bounds_.size() && bounds_[at].first==lo);
            while(!heap_.empty() && paints_[heap_.front()].hi<=lo) {
                std::pop_heap(heap_.begin(),heap_.end(),lower_priority);heap_.pop_back();
            }
            if(at==bounds_.size() || heap_.empty())continue;
            const int hi=bounds_[at].first,word=paints_[heap_.front()].word;
            int ceiling=paints_[heap_.front()].ceiling;
            if(mask_lo[word]<0) {
                const auto mask=words[word].length==cutoff+1?ix.query_interval(words[word],cutoff+1,word):std::pair<int,int>{0,0};
                mask_lo[word]=mask.first;mask_hi[word]=mask.second;
            }
            const int ml=mask_lo[word],mr=mask_hi[word];
            if(words[word].length==cutoff+1 && ml==mr) {
                // The long-mask search also locates the virtual word among
                // all Q suffixes. If the mask is empty, its two neighbors
                // give the actual maximum LCP. Long unused trie tails must
                // not force deletion exceptions out to the nominal cutoff.
                const int longest=ix.maximum_query_match(words[word],word,ml);
                ceiling=std::min(ceiling,longest);
            }
            if(mr<=lo || ml>=hi)append(lo,hi,word,ceiling);
            else {append(lo,std::min(hi,ml),word,ceiling);append(std::max(lo,mr),hi,word,ceiling);}
        }
        assert(pieces.size()<=std::size_t(count)*6);
    }
    int find(int rank) const {
        auto it=std::upper_bound(pieces.begin(),pieces.end(),rank,
                                [](int x,const Piece& p){return x<p.lo;});
        if(it==pieces.begin())return -1;
        --it;return rank<it->hi?it->word:-1;
    }
    std::size_t bytes() const {
        return nodes_.capacity()*sizeof(Node)+(adjacent_.capacity()+left_.capacity()+right_.capacity()+stack_.capacity()+order_.capacity()+heap_.capacity())*sizeof(int)
            +paints_.capacity()*sizeof(Paint)+bounds_.capacity()*sizeof(std::pair<int,int>)+pieces.capacity()*sizeof(Piece);
    }
};

// Exception rows are processed in increasing time. Endpoint versions can
// therefore be installed incrementally in one flat tree instead of retaining
// a persistent O(L log L) node arena. This preserves the same cutoff query.
class ActiveDictionary {
    std::vector<int> tree_,rank_by_end_;
    int base_=1,count_=0,owner_=0,active_=0;
    int last(int node,int lo,int hi,int cutoff) const {
        if(tree_[node]<0 || lo>=cutoff)return -1;
        if(hi<=cutoff) {
            while(node<base_)node=tree_[2*node+1]>=0?2*node+1:2*node;
            return node-base_;
        }
        const int mid=lo+(hi-lo)/2;
        const int r=last(2*node+1,mid,hi,cutoff);
        return r>=0?r:last(2*node,lo,mid,cutoff);
    }
    int first(int node,int lo,int hi,int cutoff) const {
        if(tree_[node]<0 || hi<=cutoff)return -1;
        if(lo>=cutoff) {
            while(node<base_)node=tree_[2*node]>=0?2*node:2*node+1;
            return node-base_;
        }
        const int mid=lo+(hi-lo)/2;
        const int l=first(2*node,lo,mid,cutoff);
        return l>=0?l:first(2*node+1,mid,hi,cutoff);
    }
    int maximum(int lo,int hi) const {
        int result=-1;lo+=base_;hi+=base_;
        while(lo<hi) {
            if(lo&1)result=std::max(result,tree_[lo++]);
            if(hi&1)result=std::max(result,tree_[--hi]);
            lo/=2;hi/=2;
        }
        return result;
    }
public:
    void reset(const std::vector<Word>& words,const int* sorted,int count,int owner) {
        count_=count;owner_=owner;active_=0;base_=1;
        while(base_<count_)base_*=2;
        tree_.assign(std::size_t(base_)*2,-1);rank_by_end_.resize(count_);
        for(int i=0;i<count_;++i)rank_by_end_[words[sorted[i]].end-owner_]=i;
    }
    void advance(int time) {
        const int wanted=std::min(count_,std::max(0,time-owner_));
        assert(wanted>=active_);
        while(active_<wanted) {
            int node=base_+rank_by_end_[active_];tree_[node]=owner_+active_;++active_;
            while((node/=2)!=0)tree_[node]=std::max(tree_[2*node],tree_[2*node+1]);
        }
    }
    Route query(const VirtualIndex& ix,const std::vector<Word>& words,const int* sorted,const Word& row) const {
        if(!active_)return {};
        const bool prepared=ix.prepared(words);
        const auto local_rank=[&](int rank) {
            int lo=0,hi=count_;
            while(lo<hi) {
                const int mid=lo+(hi-lo)/2;
                if(ix.word_rank(sorted[mid])<rank)lo=mid+1;else hi=mid;
            }
            return lo;
        };
        int lo=0,hi=count_;
        if(prepared)lo=local_rank(ix.row_word_position(row.length-1));
        else while(lo<hi) {
            const int mid=lo+(hi-lo)/2;
            if(ix.compare(words[sorted[mid]],row)<0)lo=mid+1;
            else hi=mid;
        }
        const int before=last(1,0,base_,lo),after=first(1,0,base_,lo);
        int length=0;
        if(before>=0)length=ix.common(words[sorted[before]],row);
        if(after>=0)length=std::max(length,ix.common(words[sorted[after]],row));
        if(!length)return {};
        auto range=prepared?ix.word_interval(row.length-1,length):
            ix.interval(count_,[&](int p)->const Word&{return words[sorted[p]];},row,length);
        if(prepared)range={local_rank(range.first),local_rank(range.second)};
        return Route{length,maximum(range.first,range.second)};
    }
    std::size_t bytes() const {return (tree_.capacity()+rank_by_end_.capacity())*sizeof(int);}
};

struct Event {
    std::array<int,4> coordinate{};
    int id=0,sign=1;
    bool query=false;
};

// Vector-weighted CDQ. 3D uses one divide-and-conquer and a Fenwick sweep;
// 4D uses one additional divide-and-conquer. All work buffers are reused.
// Clearing touched Fenwick nodes exactly (rather than subtracting updates)
// prevents accumulated rollback roundoff from leaking into later queries.
class Dominance {
    int n_,width_;
    const std::vector<double>& weights_;
    std::vector<double> tree_;
    std::vector<unsigned char> dirty_;
    std::vector<int> touched_,sweep_;
    std::vector<Event> events_,cross_;
    std::vector<double>* answers_=nullptr;
    static bool earlier(const Event& a,const Event& b,int axis) {
        if(a.coordinate[axis]!=b.coordinate[axis])return a.coordinate[axis]<b.coordinate[axis];
        if(a.query!=b.query)return !a.query;
        if(a.id!=b.id)return a.id<b.id;
        return a.sign<b.sign;
    }
    void add_row(const Event& point,const Event& query) {
        double* out=answers_->data()+std::size_t(query.id)*width_;
        const double* row=weights_.data()+std::size_t(point.id)*width_;
        if(point.sign==query.sign)for(int c=0;c<width_;++c)out[c]+=row[c];
        else for(int c=0;c<width_;++c)out[c]-=row[c];
    }
    void install(const Event& point,int coordinate) {
        const double* row=weights_.data()+std::size_t(point.id)*width_;
        for(int p=coordinate+1;p<=n_;p+=p&-p) {
            if(!dirty_[p]){dirty_[p]=1;touched_.push_back(p);}
            double* out=tree_.data()+std::size_t(p)*width_;
            if(point.sign>0)for(int c=0;c<width_;++c)out[c]+=row[c];
            else for(int c=0;c<width_;++c)out[c]-=row[c];
        }
    }
    void retrieve(const Event& query,int coordinate) {
        double* out=answers_->data()+std::size_t(query.id)*width_;
        for(int p=std::min(coordinate+1,n_);p>0;p-=p&-p) {
            const double* value=tree_.data()+std::size_t(p)*width_;
            if(query.sign>0)for(int c=0;c<width_;++c)out[c]+=value[c];
            else for(int c=0;c<width_;++c)out[c]-=value[c];
        }
    }
    void clear_tree() {
        for(int p:touched_) {
            std::fill_n(tree_.data()+std::size_t(p)*width_,width_,0.0);dirty_[p]=0;
        }
        touched_.clear();
    }
    void direct(const std::vector<Event>& event,int lo,int hi,int axis,int last_axis) {
        for(int i=lo;i<hi;++i)if(event[i].query)for(int j=lo;j<i;++j)if(!event[j].query) {
            bool accepted=true;
            for(int d=axis+1;d<=last_axis;++d)accepted&=event[j].coordinate[d]<=event[i].coordinate[d];
            if(accepted)add_row(event[j],event[i]);
        }
    }
    void recurse3(const std::vector<Event>& event,int lo,int hi,int axis) {
        if(hi-lo<=16){direct(event,lo,hi,axis,axis+2);return;}
        bool points=false,queries=false;
        for(int i=lo;i<hi;++i){points|=!event[i].query;queries|=event[i].query;}
        if(!points || !queries)return;
        const int mid=lo+(hi-lo)/2;
        recurse3(event,lo,mid,axis);recurse3(event,mid,hi,axis);
        sweep_.clear();
        for(int i=lo;i<mid;++i)if(!event[i].query)sweep_.push_back(i);
        const std::size_t point_count=sweep_.size();
        if(!point_count)return;
        for(int i=mid;i<hi;++i)if(event[i].query)sweep_.push_back(i);
        const std::size_t query_count=sweep_.size()-point_count;
        if(!query_count)return;
        if(point_count*query_count<=64) {
            for(std::size_t j=point_count;j<sweep_.size();++j)for(std::size_t i=0;i<point_count;++i) {
                const Event& p=event[sweep_[i]];const Event& q=event[sweep_[j]];
                if(p.coordinate[axis+1]<=q.coordinate[axis+1] && p.coordinate[axis+2]<=q.coordinate[axis+2])add_row(p,q);
            }
            return;
        }
        std::sort(sweep_.begin(),sweep_.end(),[&](int a,int b){return earlier(event[a],event[b],axis+1);});
        for(int id:sweep_) {
            const Event& e=event[id];
            if(e.query)retrieve(e,e.coordinate[axis+2]);
            else install(e,e.coordinate[axis+2]);
        }
        clear_tree();
    }
    void solve3(std::vector<Event>& event,int axis) {
        std::sort(event.begin(),event.end(),[&](const Event& a,const Event& b){return earlier(a,b,axis);});
        recurse3(event,0,int(event.size()),axis);
    }
    void recurse4(int lo,int hi) {
        if(hi-lo<=16){direct(events_,lo,hi,0,3);return;}
        bool points=false,queries=false;
        for(int i=lo;i<hi;++i){points|=!events_[i].query;queries|=events_[i].query;}
        if(!points || !queries)return;
        const int mid=lo+(hi-lo)/2;
        recurse4(lo,mid);recurse4(mid,hi);
        cross_.clear();
        for(int i=lo;i<mid;++i)if(!events_[i].query)cross_.push_back(events_[i]);
        const std::size_t point_count=cross_.size();
        if(!point_count)return;
        for(int i=mid;i<hi;++i)if(events_[i].query)cross_.push_back(events_[i]);
        const std::size_t query_count=cross_.size()-point_count;
        if(!query_count)return;
        if(point_count*query_count<=64) {
            for(std::size_t j=point_count;j<cross_.size();++j)for(std::size_t i=0;i<point_count;++i) {
                const Event& p=cross_[i];const Event& q=cross_[j];
                if(p.coordinate[1]<=q.coordinate[1] && p.coordinate[2]<=q.coordinate[2] && p.coordinate[3]<=q.coordinate[3])add_row(p,q);
            }
        } else solve3(cross_,1);
    }
public:
    Dominance(int n,int width,const std::vector<double>& weights)
      : n_(n),width_(width),weights_(weights) {}
    void add(const std::vector<Event>& points,const std::vector<Event>& queries,int dimensions,std::vector<double>& answers) {
        if(points.empty() || queries.empty())return;
        if(tree_.empty()) {
            tree_.assign((std::size_t(n_)+1)*width_,0.0);
            dirty_.assign(std::size_t(n_)+1,0);
        }
        answers_=&answers;events_.clear();events_.reserve(points.size()+queries.size());
        events_.insert(events_.end(),points.begin(),points.end());
        events_.insert(events_.end(),queries.begin(),queries.end());
        if(dimensions==3)solve3(events_,0);
        else {
            std::sort(events_.begin(),events_.end(),[](const Event& a,const Event& b){return earlier(a,b,0);});
            recurse4(0,int(events_.size()));
        }
        assert(touched_.empty());
    }
    std::size_t bytes() const {
        return tree_.capacity()*sizeof(double)+dirty_.capacity()+(touched_.capacity()+sweep_.capacity())*sizeof(int)
            +(events_.capacity()+cross_.capacity())*sizeof(Event);
    }
};

inline double difference(const Scores& scores,int row,int a,int b) {
    if(a==b)return 0.0;
    const double* g=scores.row(row);const double* x=scores.value(a);const double* y=scores.value(b);
    double value=0;for(int c=0;c<scores.d;++c)value+=g[c]*(x[c]-y[c]);
    return value;
}

} // namespace short_detail

inline void short_k(const Index& index,const Scores& scores,int bits,int cutoff,
                    std::vector<double>& credits,Stats& stats) {
    using namespace short_detail;
    const int n=index.n,d=scores.d;
    if(!n)return;
    if(scores.n!=n || d<1 || bits<1 || bits>64 || cutoff<1 || cutoff>n || credits.size()<std::size_t(n)*bits)
        throw std::invalid_argument("invalid compact K repair dimensions");

    std::vector<unsigned char> active(n,0);int last_active=-1;
    for(int t=0;t<n;++t)for(int c=0;c<d;++c)if(scores.row(t)[c]!=0.0){active[t]=1;last_active=t;break;}
    if(last_active<0)return;

    VirtualIndex ix(index);
    std::vector<Route> baseline(n);
    std::vector<double> row_weights;
    std::vector<int> end_offsets(std::size_t(n)+1,0),end_rows;
    for(int t=0;t<n;++t) {
        baseline[t]=index.best(t);
        if(active[t] && baseline[t].end>=0)++end_offsets[baseline[t].end+1];
    }
    std::partial_sum(end_offsets.begin(),end_offsets.end(),end_offsets.begin());
    end_rows.resize(end_offsets.back());std::vector<int> cursor=end_offsets;
    for(int t=0;t<n;++t)if(active[t] && baseline[t].end>=0)end_rows[cursor[baseline[t].end]++]=t;

    SupportedWords support(index,last_active);
    MinimumRoutes minimum(baseline,ix.row_rank,active);

    struct Dictionary {int owner,bit,first,count;int piece_first=0,piece_count=0;};
    struct Job {int owner,bit,lo,hi,word,time;};
    std::vector<Word> words;words.reserve(n);
    std::vector<Dictionary> dictionaries;
    std::vector<int> word_dictionary,lexical,rank,local_order,placed,mask_lo,mask_hi;
    std::vector<Event> points_a,points_b,queries_a,queries_b;
    std::vector<Piece> batch_pieces;
    std::vector<Job> jobs;std::vector<double> sums;
    std::vector<int> exception_rows;int exception_owner=-1,exception_length=-1;
    std::vector<Route> sparse_winners;
    std::vector<int> sparse_rows;
    CompactTrie trie;ActiveDictionary dictionary;CausalWords causal_words;
    Dominance dominance(n,d+1,row_weights);

    const auto record_workspace=[&]() {
        const std::size_t bytes=ix.bytes()+(baseline.capacity()+sparse_winners.capacity())*sizeof(Route)
            +row_weights.capacity()*sizeof(double)
            +(end_offsets.capacity()+end_rows.capacity()+cursor.capacity()+word_dictionary.capacity()+lexical.capacity()+rank.capacity()+local_order.capacity()+placed.capacity()+mask_lo.capacity()+mask_hi.capacity()+exception_rows.capacity()+sparse_rows.capacity())*sizeof(int)
            +active.capacity()+support.bytes()+words.capacity()*sizeof(Word)
            +dictionaries.capacity()*sizeof(Dictionary)+jobs.capacity()*sizeof(Job)+sums.capacity()*sizeof(double)
            +(points_a.capacity()+points_b.capacity()+queries_a.capacity()+queries_b.capacity())*sizeof(Event)
            +batch_pieces.capacity()*sizeof(Piece)+trie.bytes()+dictionary.bytes()+dominance.bytes()+minimum.bytes()+causal_words.bytes();
        stats.peak_workspace_bytes=std::max<std::uint64_t>(stats.peak_workspace_bytes,bytes);
    };

    const auto collect_exceptions=[&](int owner,int maximum_length) {
        if(exception_owner==owner && exception_length==maximum_length)return;
        exception_owner=owner;exception_length=maximum_length;exception_rows.clear();
        if(!maximum_length)return;
        // A baseline endpoint farther than maximum_length past the edit
        // retains a longer untouched suffix after deletion. No short winner
        // in this dictionary can improve it, so those rows are not exceptions.
        const int finish=end_offsets[std::min(n,owner+maximum_length+1)];
        for(int p=end_offsets[owner];p<finish;++p) {
            const int t=end_rows[p];
            if(owner>=baseline[t].end-baseline[t].length+1)exception_rows.push_back(t);
        }
        std::sort(exception_rows.begin(),exception_rows.end());
    };

    const auto process=[&]() {
        const int count=int(words.size());if(!count)return;
        lexical.resize(count);std::iota(lexical.begin(),lexical.end(),0);
        std::sort(lexical.begin(),lexical.end(),[&](int a,int b) {
            const int order=ix.compare(words[a],words[b]);return order?order<0:a<b;
        });
        ix.prepare(words,lexical);
        rank.resize(count);local_order.resize(count);placed.assign(dictionaries.size(),0);
        for(int p=0;p<count;++p) {
            const int word=lexical[p],di=word_dictionary[word];rank[word]=p;
            local_order[dictionaries[di].first+placed[di]++]=word;
        }
        causal_words.reset(count,[&](int r){const Dictionary& dict=dictionaries[word_dictionary[lexical[r]]];return dict.owner+dict.count;});
        mask_lo.assign(count,-1);mask_hi.assign(count,-1);
        points_a.clear();points_b.clear();jobs.clear();batch_pieces.clear();
        int earliest_late=n;
        for(const Dictionary& dict:dictionaries)earliest_late=std::min(earliest_late,dict.owner+dict.count);
        const auto signed_interval=[&](std::vector<Event>& output,int lo,int hi,int t,bool tie) {
            if(lo>=hi)return;
            // Small prefix intervals are the sparse base case of this same
            // batched range sum. Enumerate at most sixteen virtual words per
            // row interval, then keep only its dictionary's selected winner.
            // Large intervals retain the signed 3D/4D dominance formulation.
            std::array<int,17> sparse{};int sparse_count;
            if(hi-lo<=16) {
                sparse_count=hi-lo;for(int i=0;i<sparse_count;++i)sparse[i]=lo+i;
            } else sparse_count=causal_words.report(lo,hi,t,sparse.data(),17);
            if(sparse_count<=16) {
                const int qr=ix.row_rank[t];
                for(int i=0;i<sparse_count;++i) {
                    const int r=sparse[i];
                    const int word=lexical[r];const Dictionary& target=dictionaries[word_dictionary[word]];
                    if(t<target.owner+target.count || !target.piece_count || (tie && words[word].end<=baseline[t].end))continue;
                    const auto begin=batch_pieces.begin()+target.piece_first,finish=begin+target.piece_count;
                    auto piece=std::upper_bound(begin,finish,qr,[](int x,const Piece& p){return x<p.lo;});
                    if(piece==begin)continue;
                    --piece;
                    if(qr<piece->hi && piece->word==word)
                        credits[std::size_t(target.owner)*bits+target.bit]+=difference(scores,t,words[word].end,baseline[t].end);
                }
                return;
            }
            output.push_back(Event{{lo,ix.row_rank[t],n-1-t,baseline[t].end},t,1,false});
            // No representative rank equals count, so this last cancellation
            // event cannot be consumed by any query in the current batch.
            if(hi<count)output.push_back(Event{{hi,ix.row_rank[t],n-1-t,baseline[t].end},t,-1,false});
        };
        const auto build_points=[&]() {
          for(int t=earliest_late;t<n;++t)if(active[t] && baseline[t].length<=cutoff) {
            const int length=baseline[t].length;
            std::pair<int,int> exact{0,0},strict;
            if(length) {
                exact=ix.word_interval(t,length);
                if(exact.first==exact.second)continue;
                strict=ix.word_interval(t,length+1);
            } else strict=ix.word_interval(t,1);
            signed_interval(points_a,strict.first,strict.second,t,false);
            if(length) {
                if(strict.first==strict.second)signed_interval(points_b,exact.first,exact.second,t,true);
                else {
                    signed_interval(points_b,exact.first,strict.first,t,true);
                    signed_interval(points_b,strict.second,exact.second,t,true);
                }
            }
          }
        };

        for(Dictionary& dict:dictionaries) {
            const int owner=dict.owner,bit=dict.bit;
            const int* sorted=local_order.data()+dict.first;
            const int last_early=owner+dict.count-1;
            const bool has_late=last_early<n-1;
            double correction=0.0;
            dict.piece_first=int(batch_pieces.size());dict.piece_count=0;
            stats.short_words+=dict.count;
            if(has_late) {
                trie.build(ix,words,sorted,dict.count,cutoff,mask_lo,mask_hi,true);
                stats.short_pieces+=trie.pieces.size();
                for(const Piece& piece:trie.pieces) {
                    if(!(Route{piece.ceiling,words[piece.word].end}>minimum.query(piece.lo,piece.hi)))continue;
                    // The two rank axes need symmetric sparse base cases.
                    // A word-prefix interval can be large while its winning
                    // Q-rank piece contains only a handful of rows. Evaluate
                    // at most sixteen here, before constructing CDQ events.
                    // The affected-baseline correction below remains shared.
                    if(piece.hi-piece.lo<=16) {
                        const Word& word=words[piece.word];
                        for(int qr=piece.lo;qr<piece.hi;++qr) {
                            const Word& row=ix.query_words[qr];const int t=row.length-1;
                            if(t<=last_early || !active[t])continue;
                            const Route original=baseline[t];
                            if(!(Route{piece.ceiling,word.end}>original))continue;
                            const Route candidate{ix.common(word,row),word.end};
                            if(candidate>original)
                                correction+=difference(scores,t,candidate.end,original.end);
                        }
                        continue;
                    }
                    batch_pieces.push_back(piece);
                    jobs.push_back(Job{owner,bit,piece.lo,piece.hi,piece.word,n-last_early-2});
                }
                dict.piece_count=int(batch_pieces.size())-dict.piece_first;
            } else trie.pieces.clear();
            int maximum_length=0;
            for(const Piece& piece:trie.pieces)maximum_length=std::max(maximum_length,piece.ceiling);
            collect_exceptions(owner,maximum_length);bool prepared=false;
            int early=owner+1;std::size_t exceptional=0;
            while(early<=last_early || exceptional<exception_rows.size()) {
                while(early<=last_early && !active[early])++early;
                const int t=std::min(early<=last_early?early:n,
                    exceptional<exception_rows.size()?exception_rows[exceptional]:n);
                if(t==n)break;
                if(t==early)++early;
                if(exceptional<exception_rows.size() && t==exception_rows[exceptional])++exceptional;
                ++stats.short_exceptions;
                const Route original=baseline[t];
                const bool affected=original.length && owner<=original.end && owner>=original.end-original.length+1;
                if(t>last_early) {
                    // Late dictionaries are fully causal. The rank piece
                    // supplies the candidate that can beat deletion; omitted
                    // unchanged prefixes are themselves available to DelK.
                    if(!affected)continue;
                    const int word=trie.find(ix.row_rank[t]);if(word<0)continue;
                    const Route candidate{ix.common(words[word],ix.row(t)),words[word].end};
                    const Route surviving=original.end>owner?Route{original.end-owner,original.end}:Route{};
                    if(!(candidate>surviving))continue;
                    const Route deleted=index.deleted(t,owner);++stats.suffix_queries;
                    if(candidate>original)correction+=difference(scores,t,original.end,deleted.end);
                    else if(candidate>deleted)correction+=difference(scores,t,candidate.end,deleted.end);
                } else {
                    const Route surviving=affected?(original.end>owner?Route{original.end-owner,original.end}:Route{}):original;
                    if(surviving.length>cutoff)continue;
                    if(!prepared){dictionary.reset(words,sorted,dict.count,owner);prepared=true;}
                    dictionary.advance(t);
                    const Route candidate=dictionary.query(ix,words,sorted,ix.row(t));
                    if(candidate.length>cutoff || !(candidate>surviving))continue;
                    const Route deleted=affected?index.deleted(t,owner):original;
                    if(affected)++stats.suffix_queries;
                    if(candidate>deleted)correction+=difference(scores,t,candidate.end,deleted.end);
                }
            }
            credits[std::size_t(owner)*bits+bit]+=correction;
        }
        if(!jobs.empty())build_points();

        if((!points_a.empty() || !points_b.empty()) && row_weights.empty()) {
            row_weights.resize(std::size_t(n)*(d+1));
            for(int t=0;t<n;++t) {
                std::copy_n(scores.row(t),d,row_weights.data()+std::size_t(t)*(d+1));
                row_weights[std::size_t(t)*(d+1)+d]=scores.score(t,baseline[t].end);
            }
        }

        for(std::size_t start=0;start<jobs.size() && (!points_a.empty() || !points_b.empty());start+=n) {
            const int size=int(std::min<std::size_t>(n,jobs.size()-start));
            queries_a.clear();queries_b.clear();sums.assign(std::size_t(size)*(d+1),0.0);
            for(int i=0;i<size;++i) {
                const Job& job=jobs[start+i];const int x=rank[job.word],time=job.time;
                const int end=words[job.word].end;
                for(int which=0;which<2;++which) {
                    const int qr=which?job.lo-1:job.hi-1,sign=which?-1:1;
                    if(qr<0)continue;
                    if(!points_a.empty())queries_a.push_back(Event{{x,qr,time,0},i,sign,true});
                    if(end>0 && !points_b.empty())queries_b.push_back(Event{{x,qr,time,end-1},i,sign,true});
                }
            }
            stats.short_queries+=queries_a.size()+queries_b.size();
            dominance.add(points_a,queries_a,3,sums);dominance.add(points_b,queries_b,4,sums);
            for(int i=0;i<size;++i) {
                const Job& job=jobs[start+i];const double* aggregated=sums.data()+std::size_t(i)*(d+1);
                const double* payload=scores.value(words[job.word].end);
                double value=-aggregated[d];for(int c=0;c<d;++c)value+=aggregated[c]*payload[c];
                credits[std::size_t(job.owner)*bits+job.bit]+=value;
            }
        }
        record_workspace();
    };

    for(int owner=0;owner<last_active;++owner)for(int bit=0;bit<bits;++bit) {
        const std::uint64_t symbol=index.k[owner]^(std::uint64_t(1)<<bit);
        std::array<int,16> centers;
        const int centers_count=support.sparse_centers(symbol,owner,centers.data(),int(centers.size()));
        if(centers_count>=0) {
            if(!centers_count)continue;
            if(sparse_winners.empty())sparse_winners.resize(n);
            int extension=0;sparse_rows.clear();
            // This is the sparse base case of the same edited dictionary.
            // Each eligible center generates one affine candidate interval.
            // With at most sixteen centers, direct winner accumulation costs
            // O(cutoff) per dictionary and needs one reusable row buffer.
            for(int i=0;i<centers_count;++i) {
                const int u=centers[i];
                const int left=owner?index.lcs(u-1,owner-1):0;
                const int right=index.lce(u+1,owner+1);
                const int until=std::min(cutoff,right);extension=std::max(extension,until);
                for(int offset=0;offset<=until;++offset) {
                    const int t=u+offset;if(!active[t])continue;
                    const Route candidate{left+offset+1,owner+offset};
                    Route& best=sparse_winners[t];
                    if(!best.length)sparse_rows.push_back(t);
                    if(candidate>best)best=candidate;
                }
            }
            std::sort(sparse_rows.begin(),sparse_rows.end());
            double correction=0.;
            for(int t:sparse_rows) {
                const Route candidate=sparse_winners[t];sparse_winners[t]={};
                ++stats.short_exceptions;
                if(candidate.length>cutoff)continue;
                const Route original=baseline[t];
                const bool affected=original.length && owner<=original.end && owner>=original.end-original.length+1;
                const Route surviving=affected?(original.end>owner?Route{original.end-owner,original.end}:Route{}):original;
                if(!(candidate>surviving))continue;
                const Route deleted=affected?index.deleted(t,owner):original;
                if(affected)++stats.suffix_queries;
                if(candidate>deleted)correction+=difference(scores,t,candidate.end,deleted.end);
            }
            credits[std::size_t(owner)*bits+bit]+=correction;
            stats.short_words+=extension+1;
            continue;
        }
        const int extension=support.right_extension(symbol,owner);
        if(extension<0)continue;
        // Endpoints beyond this bound cannot reach the changed bit. Any
        // prefix they still match is unchanged and is already in DelK.
        // Keep the original L for long masks and deletion exceptions.
        const int count=std::min(cutoff,extension)+1;
        if(!words.empty() && words.size()+std::size_t(count)>std::size_t(n)) {
            process();words.clear();word_dictionary.clear();dictionaries.clear();
        }
        const int dict=int(dictionaries.size()),first=int(words.size());
        dictionaries.push_back(Dictionary{owner,bit,first,count});
        for(int end=owner;end<owner+count;++end) {
            words.push_back(ix.edited(owner,bit,end,cutoff));word_dictionary.push_back(dict);
        }
    }
    process();
    record_workspace();
}

} // namespace rosa_subq
