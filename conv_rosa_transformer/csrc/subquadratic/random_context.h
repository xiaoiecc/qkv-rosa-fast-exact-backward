#pragma once
#include "types.h"
#include "../full_word_map.h"
#include <cmath>

namespace rosa_subq {
namespace random_context {

// Each node is an exact reversed K word. Its occurrence list is in decreasing
// endpoint order. All queries see just the causal prefix inserted so far.
class Directory {
    struct Node {int latest,level;};
    rosa_exact::FullWordMap symbols,edges;
    std::vector<Node> nodes;
    std::vector<int> previous;
    std::vector<int> key_class;
    int depth_;
    static uint64_t edge(int node,int symbol) {
        return (uint64_t(uint32_t(node))<<32)|uint32_t(symbol);
    }
public:
    Directory(const uint64_t* k,int n,int depth):symbols(n,17),edges(n,39),
        nodes(1,Node{-1,0}),previous(size_t(n)*depth),key_class(n),depth_(depth) {
        int count=0;
        for(int i=0;i<n;++i) {
            const auto item=symbols.insert(k[i],count);
            key_class[i]=item.first;count+=item.second;
        }
    }
    int symbol(uint64_t word) const {return symbols.find(word);}
    int child(int node,int symbol) const {
        return symbol<0 || node<0?-1:edges.find(edge(node,symbol));
    }
    bool insert(int r) {
        int node=0;
        for(int age=0;age<depth_ && age<=r;++age) {
            if(nodes.size()>=size_t(INT32_MAX))return false;
            const auto item=edges.insert(edge(node,key_class[r-age]),int(nodes.size()));
            if(item.second)nodes.push_back({-1,age});
            node=item.first;
            previous[size_t(age)*key_class.size()+r]=nodes[node].latest;
            nodes[node].latest=r;
        }
        return true;
    }
    int latest(int node) const {return nodes[node].latest;}
    int survivor(int node,int h,int s) const {
        int end=nodes[node].latest;
        if(end-h>=s || end<s)return end;
        // A covering latest occurrence ends in [s,s+h); skipping at most h
        // entries reaches the best occurrence strictly before the owner.
        const int* chain=previous.data()+size_t(nodes[node].level)*key_class.size();
        while(end>=s)end=chain[end];
        return end;
    }
    template<class Emit>
    bool visit(int node,int after,uint64_t& remaining,Emit emit) const {
        if(node<0)return true;
        const int* chain=previous.data()+size_t(nodes[node].level)*key_class.size();
        for(int end=nodes[node].latest;end>after;end=chain[end]) {
            if(!remaining)return false;
            --remaining;emit(end);
        }
        return true;
    }
    uint64_t bytes() const {
        return symbols.workspace_bytes()+edges.workspace_bytes()+
            (previous.capacity()+key_class.capacity())*sizeof(int)+nodes.capacity()*sizeof(Node);
    }
};

inline int max_length(int n,int bits) {
    return std::max(1,int(std::ceil(5*std::log2(double(n)+1)/bits)));
}

// A complete, bounded alternative for any packed width. All numerical
// updates are private until the candidate census finishes within budget.
inline bool credit(const uint64_t* q,const uint64_t* k,const std::vector<Route>& base,
                   int bits,int longest,const Scores& score,std::vector<double>& cq,
                   std::vector<double>& ck,Stats& stats,uint64_t budget) {
    const int n=score.n;
    if(n<2 || longest>max_length(n,bits))return false;
    const int depth=std::max(1,int(std::ceil(std::log2(double(n))/bits)));
    Directory trie(k,n,depth);
    std::vector<int> query(n),path(depth+1),seen(n,-1);
    for(int i=0;i<n;++i)query[i]=trie.symbol(q[i]);
    std::vector<Route> removed(longest),best(size_t(n)*bits),deep;
    std::vector<size_t> touched;
    std::vector<double> qsum(cq.size()),ksum(ck.size()),deleted(n);
    uint64_t left=budget;
    auto lcs=[&](int t,int r){
        int h=0;while(r>=0 && q[t]==k[r]){--r;--t;++h;}
        return h;
    };
    for(int t=1;t<n;++t) {
        if(!trie.insert(t-1))return false;
        const double* w=score.row(t);
        if(std::none_of(w,w+score.d,[](double x){return x!=0.;}))continue;
        const Route original=base[t];
        const int L=original.length,first=original.end-L+1,known=std::min(L,depth);
        path[0]=0;
        for(int h=1;h<=known;++h) {
            path[h]=trie.child(path[h-1],query[t-h+1]);
            if(path[h]<0)return false;
        }
        auto delta=[&](int end,int before) {
            const double* a=score.value(end);const double* b=score.value(before);
            double sum=0;for(int c=0;c<score.d;++c)sum+=w[c]*(a[c]-b[c]);
            return sum;
        };
        for(int age=0;age<=L && age<t;++age)for(int bit=0;bit<bits;++bit) {
            int h=std::min(age,depth),node=path[h];
            while(h<depth && h<t) {
                const int id=h==age?trie.symbol(q[t-h]^(UINT64_C(1)<<bit)):query[t-h];
                const int next=trie.child(node,id);
                if(next<0)break;
                node=next;++h;
            }
            Route winner=h?Route{h,trie.latest(node)}:Route{};
            if(h==depth && h<t && !trie.visit(node,-1,left,[&](int r){
                int j=h;
                while(j<=r && (q[t-j]^(j==age?(UINT64_C(1)<<bit):0))==k[r-j])++j;
                winner=std::max(winner,Route{j,r});
            }))return false;
            if(winner.end!=original.end)qsum[size_t(t-age)*bits+bit]+=delta(winner.end,original.end);
        }
        deep.clear();
        if(L>depth && !trie.visit(path[depth],-1,left,[&](int r){
            const int len=lcs(t,r);if(len>depth)deep.push_back({len,r});
        }))return false;
        for(int s=first;L && s<=original.end;++s) {
            Route other;
            for(int h=known;h>0;--h) {
                const int end=trie.survivor(path[h],h,s);
                if(end>=0){other={h,end};break;}
            }
            for(auto item:deep) {
                if(item.end>=s)item.length=std::min(item.length,item.end-s);
                if(item.length>0)other=std::max(other,item);
            }
            removed[s-first]=other;
            if(other.end!=original.end)deleted[s]+=delta(other.end,original.end);
        }
        touched.clear();
        auto consider=[&](int r) {
            if(seen[r]==t)return;
            seen[r]=t;
            const int a=lcs(t,r),s=r-a,u=t-a;
            if(s<0)return;
            const uint64_t mask=q[u]^k[s];
            if(!mask || (mask&(mask-1)))return;
            int bit=0;for(auto x=mask;x>1;x>>=1)++bit;
            if(bit>=bits)return;
            const Route candidate{a+1+lcs(u-1,s-1),r};
            const Route other=s>=first && s<=original.end?removed[s-first]:original;
            if(!(candidate>other))return;
            const size_t at=size_t(s)*bits+bit;
            if(!best[at].length)touched.push_back(at);
            best[at]=std::max(best[at],candidate);
        };
        auto neighbors=[&](int length,int after) {
            if(length>t || after>=t-1)return true;
            const int h=std::min(length,depth);
            if(h<=known && !trie.visit(path[h],after,left,consider))return false;
            for(int age=0;age<h && age<=known;++age)for(int bit=0;bit<bits;++bit) {
                int node=trie.child(path[age],trie.symbol(q[t-age]^(UINT64_C(1)<<bit)));
                for(int j=age+1;j<h && node>=0;++j)node=trie.child(node,query[t-j]);
                if(!trie.visit(node,after,left,consider))return false;
            }
            return true;
        };
        // Outside the damaged occurrence, either beat the length or win
        // a tie at a later end. Inside it, shorter winners are also possible.
        if(!neighbors(L+1,-1) || (L && !neighbors(L,original.end)))return false;
        if(L)for(int r=first;r<std::min(t,original.end+L);++r)consider(r);
        for(size_t at:touched) {
            const int s=int(at/bits);
            const Route other=s>=first && s<=original.end?removed[s-first]:original;
            if(best[at].end!=other.end)ksum[at]+=delta(best[at].end,other.end);
            best[at]={};
        }
    }
    for(int s=0;s<n;++s)for(int j=0;j<bits;++j)ksum[size_t(s)*bits+j]+=deleted[s];
    cq.swap(qsum);ck.swap(ksum);
    stats.suffix_queries+=budget-left;
    stats.peak_workspace_bytes=std::max<uint64_t>(stats.peak_workspace_bytes,trie.bytes()
        +(path.capacity()+query.capacity()+seen.capacity())*sizeof(int)
        +(removed.capacity()+best.capacity()+deep.capacity())*sizeof(Route)
        +touched.capacity()*sizeof(size_t)+(qsum.capacity()+ksum.capacity()+deleted.capacity())*sizeof(double));
    return true;
}
} // namespace random_context
} // namespace rosa_subq
