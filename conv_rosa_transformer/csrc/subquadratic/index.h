#pragma once
// Exact suffix and occurrence queries for the subquadratic credit engine.
// Packed symbols are compressed by their full uint64 value: no fingerprints,
// collisions, or reserved packed-symbol values enter route decisions.
#include "types.h"
#include <cassert>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace rosa_subq {

class SuffixArray {
    int size_=0;
    std::vector<unsigned char> log_;
    std::vector<std::vector<int>> minimum_;
public:
    std::vector<int> sa,rank;
    SuffixArray()=default;
    explicit SuffixArray(const std::vector<int>& text) { build(text); }
    int size() const { return size_; }
    void build(const std::vector<int>& text) {
        if(text.size()>std::size_t(std::numeric_limits<int>::max()/2))
            throw std::length_error("suffix index exceeds int32 range");
        size_=int(text.size());const int n=size_;
        sa.resize(n);rank.resize(n);minimum_.clear();log_.assign(std::size_t(n)+1,0);
        if(!n)return;
        std::iota(sa.begin(),sa.end(),0);
        std::sort(sa.begin(),sa.end(),[&](int a,int b){return text[a]<text[b];});
        int classes=1;rank[sa[0]]=0;
        for(int z=1;z<n;++z) {classes+=text[sa[z-1]]!=text[sa[z]];rank[sa[z]]=classes-1;}
        std::vector<int> work(n),next_rank(n),count(std::size_t(n)+1);
        // Two stable counting passes per doubling level, O(n log n).
        for(int step=1;classes<n;) {
            std::fill(count.begin(),count.begin()+classes+1,0);
            for(int i=0;i<n;++i)++count[i<n-step?rank[i+step]+1:0];
            int sum=0;
            for(int key=0;key<=classes;++key){const int z=count[key];count[key]=sum;sum+=z;}
            for(int i=0;i<n;++i)work[count[i<n-step?rank[i+step]+1:0]++]=i;
            std::fill(count.begin(),count.begin()+classes+1,0);
            for(int i:work)++count[rank[i]+1];
            sum=0;
            for(int key=0;key<=classes;++key){const int z=count[key];count[key]=sum;sum+=z;}
            for(int i:work)sa[count[rank[i]+1]++]=i;
            int next_classes=1;next_rank[sa[0]]=0;
            for(int z=1;z<n;++z) {
                const int a=sa[z-1],b=sa[z];
                const int ar=a<n-step?rank[a+step]:-1;
                const int br=b<n-step?rank[b+step]:-1;
                next_classes+=(rank[a]!=rank[b] || ar!=br);
                next_rank[b]=next_classes-1;
            }
            rank.swap(next_rank);classes=next_classes;
            if(step>n/2)step=n;else step*=2;
        }
        std::vector<int> lcp(n,0);int h=0;
        for(int i=0;i<n;++i) {
            const int z=rank[i];
            if(!z){h=0;continue;}
            const int j=sa[z-1];
            while(h<n-i && h<n-j && text[i+h]==text[j+h])++h;
            lcp[z]=h;if(h)--h;
        }
        minimum_.push_back(std::move(lcp));
        for(int i=2;i<=n;++i)log_[i]=static_cast<unsigned char>(log_[i/2]+1);
        for(int width=2;width<=n;) {
            const int half=width/2;const auto& previous=minimum_.back();
            std::vector<int> level(n-width+1);
            for(int i=0;i<int(level.size());++i)level[i]=std::min(previous[i],previous[i+half]);
            minimum_.push_back(std::move(level));
            if(width>n/2)break;
            width*=2;
        }
    }
    int lcp_rank(int a,int b) const {
        assert(a>=0 && a<size_ && b>=0 && b<size_);
        if(a==b)return size_-sa[a];
        if(a>b)std::swap(a,b);
        ++a;const int level=log_[b-a+1],width=1<<level;
        return std::min(minimum_[level][a],minimum_[level][b-width+1]);
    }
    std::pair<int,int> prefix_interval(int at,int length) const {
        assert(at>=0 && at<size_ && length>=0);
        if(!length)return {0,size_};
        if(length>size_-sa[at])return {at,at};
        int lo=0,hi=at;
        while(lo<hi){const int mid=lo+(hi-lo)/2;if(lcp_rank(mid,at)>=length)hi=mid;else lo=mid+1;}
        const int left=lo;lo=at;hi=size_-1;
        while(lo<hi){const int mid=lo+(hi-lo+1)/2;if(lcp_rank(at,mid)>=length)lo=mid;else hi=mid-1;}
        return {left,lo+1};
    }
    std::size_t workspace_bytes() const {
        std::size_t bytes=(sa.capacity()+rank.capacity())*sizeof(int)+log_.capacity();
        for(const auto& v:minimum_)bytes+=v.capacity()*sizeof(int);
        return bytes;
    }
};

// A wavelet matrix over suffix ranks. Non-K suffixes have value0; a K start s
// has value s+1. One quantile after a rank query gives predecessor/successor.
// Its contiguous level arrays replace a pointer-heavy tree of sorted vectors.
class OccurrenceWavelet {
    int maximum_=0,levels_=0;
    std::vector<int> zeros_;
    std::vector<std::vector<int>> zero_prefix_;
public:
    OccurrenceWavelet()=default;
    OccurrenceWavelet(std::vector<int> values,int maximum) { build(std::move(values),maximum); }
    void build(std::vector<int> values,int maximum) {
        maximum_=maximum;levels_=0;zero_prefix_.clear();zeros_.clear();
        for(unsigned x=static_cast<unsigned>(maximum);x;x>>=1)++levels_;
        if(!levels_)levels_=1;
        const int n=int(values.size());std::vector<int> work(n);
        zero_prefix_.reserve(levels_);zeros_.reserve(levels_);
        for(int level=0;level<levels_;++level) {
            const int bit=levels_-1-level;std::vector<int> prefix(std::size_t(n)+1,0);
            for(int i=0;i<n;++i)prefix[i+1]=prefix[i]+(((values[i]>>bit)&1)==0);
            const int zero=prefix[n];int a=0,b=zero;
            for(int x:values){if((x>>bit)&1)work[b++]=x;else work[a++]=x;}
            values.swap(work);zeros_.push_back(zero);zero_prefix_.push_back(std::move(prefix));
        }
    }
    int count_less(int lo,int hi,int value) const {
        if(value<=0 || lo==hi)return 0;
        if(value>maximum_)return hi-lo;
        int count=0;
        for(int level=0;level<levels_;++level) {
            const auto& prefix=zero_prefix_[level];const int zl=prefix[lo],zr=prefix[hi];
            if((value>>(levels_-1-level))&1){count+=zr-zl;lo=zeros_[level]+lo-zl;hi=zeros_[level]+hi-zr;}
            else{lo=zl;hi=zr;}
        }
        return count;
    }
    int quantile(int lo,int hi,int order) const {
        assert(order>=0 && order<hi-lo);int result=0;
        for(int level=0;level<levels_;++level) {
            const auto& prefix=zero_prefix_[level];const int zl=prefix[lo],zr=prefix[hi],zero=zr-zl;
            if(order<zero){lo=zl;hi=zr;}
            else {order-=zero;result|=1<<(levels_-1-level);lo=zeros_[level]+lo-zl;hi=zeros_[level]+hi-zr;}
        }
        return result;
    }
    int greatest_below(int lo,int hi,int cutoff) const {
        if(lo>=hi || cutoff<=0)return -1;
        const int count=count_less(lo,hi,cutoff+1);
        if(!count)return -1;
        return quantile(lo,hi,count-1)-1;
    }
    int position(int lo,int hi,int lower,int upper,bool successor) const {
        if(lo>=hi || lower>upper)return -1;
        const int below=count_less(lo,hi,lower+1),after=count_less(lo,hi,upper+2);
        if(below==after)return -1;
        return quantile(lo,hi,successor?below:after-1)-1;
    }
    std::size_t workspace_bytes() const {
        std::size_t bytes=zeros_.capacity()*sizeof(int);
        for(const auto& v:zero_prefix_)bytes+=v.capacity()*sizeof(int);
        return bytes;
    }
};

class Index {
    struct RCacheEntry { int t=-1,length=0,cutoff=0,result=-1; };
    SuffixArray forward_,reverse_;
    OccurrenceWavelet positions_,prefix_ranks_;
    std::vector<Route> baseline_;
    mutable std::vector<RCacheEntry> r_cache_;
    mutable unsigned cache_window_=0,cache_good_=0,cache_skip_=0;
    std::uint64_t cache_seed_=0;
    static std::uint64_t mix(std::uint64_t x) {
        x^=x>>30;x*=UINT64_C(0xbf58476d1ce4e5b9);
        x^=x>>27;x*=UINT64_C(0x94d049bb133111eb);return x^(x>>31);
    }
    void observe_cache(bool hit) const {
        cache_good_+=hit;
        if(++cache_window_==512) {
            // Low reuse should cost a bounded probe window, not a hash at
            // every query. Resume periodically because later phases can reuse
            // different keys. Bypassing only changes work, never the answer.
            if(cache_good_<32)cache_skip_=2048;
            cache_window_=cache_good_=0;
        }
    }
    Route best_search(int t,int cutoff) const {
        if(cutoff<=0)return {};
        // Among an arbitrary subset of suffixes, the longest common prefix
        // with a query occurs at its lexicographic predecessor or successor.
        // prefix_ranks_ is time -> reversed suffix rank, so one range-rank
        // query finds both neighbours among endpoints < cutoff. This removes
        // binary search over match lengths from every deletion prefix query.
        const int rank=reverse_.rank[n-1-t];
        const int below=prefix_ranks_.count_less(0,cutoff,rank+1);int length=0;
        if(below)length=reverse_.lcp_rank(rank,prefix_ranks_.quantile(0,cutoff,below-1)-1);
        if(below<cutoff)length=std::max(length,reverse_.lcp_rank(rank,prefix_ranks_.quantile(0,cutoff,below)-1));
        return length?Route{length,R(t,length,cutoff)}:Route{};
    }
public:
    int n=0;
    std::vector<std::uint64_t> q,k;
    // A stream owns its Index. Optional cache mutation must not be shared by
    // simultaneous calls; independent streams already have independent indices.
    mutable std::uint64_t cache_hits=0,cache_misses=0,cache_evictions=0,cache_bypasses=0;
    Index(const std::uint64_t* query,const std::uint64_t* key,int length,
          bool randomized=false,std::uint64_t seed=0):cache_seed_(seed),n(length) {
        if(n<1 || n>(std::numeric_limits<int>::max()-2)/4 || !query || !key)
            throw std::invalid_argument("invalid subquadratic index input");
        q.assign(query,query+n);k.assign(key,key+n);
        std::vector<std::uint64_t> alphabet=q;alphabet.insert(alphabet.end(),k.begin(),k.end());
        std::sort(alphabet.begin(),alphabet.end());alphabet.erase(std::unique(alphabet.begin(),alphabet.end()),alphabet.end());
        std::vector<int> f(std::size_t(2)*n+2),r(f.size());
        for(int i=0;i<n;++i) {
            f[i]=2+int(std::lower_bound(alphabet.begin(),alphabet.end(),q[i])-alphabet.begin());
            f[n+1+i]=2+int(std::lower_bound(alphabet.begin(),alphabet.end(),k[i])-alphabet.begin());
            r[n-1-i]=f[i];r[2*n-i]=f[n+1+i];
        }
        f[n]=r[n]=0;f[2*n+1]=r[2*n+1]=1;
        forward_.build(f);reverse_.build(r);
        std::vector<int> values(f.size(),0);
        for(int s=0;s<n;++s)values[forward_.rank[n+1+s]]=s+1;
        positions_.build(std::move(values),n);
        std::vector<int> reversed_ranks(n);
        for(int end=0;end<n;++end)reversed_ranks[end]=reverse_.rank[2*n-end]+1;
        prefix_ranks_.build(std::move(reversed_ranks),reverse_.size());
        if(randomized) {
            std::size_t slots=1;
            while(slots<std::size_t(n)*2)slots*=2;
            r_cache_.resize(slots);
        }
        baseline_.reserve(n);for(int t=0;t<n;++t)baseline_.push_back(best_search(t,t));
    }
    const SuffixArray& forward_suffixes() const { return forward_; }
    const SuffixArray& reverse_suffixes() const { return reverse_; }
    const std::vector<Route>& baseline() const { return baseline_; }
    int lce(int qstart,int kstart) const {
        if(qstart>=n || kstart>=n)return 0;
        assert(qstart>=0 && kstart>=0);
        return std::min({n-qstart,n-kstart,forward_.lcp_rank(forward_.rank[qstart],forward_.rank[n+1+kstart])});
    }
    int lcs(int qend,int kend) const {
        if(qend<0 || kend<0)return 0;
        assert(qend<n && kend<n);
        return std::min({qend+1,kend+1,reverse_.lcp_rank(reverse_.rank[n-1-qend],reverse_.rank[2*n-kend])});
    }
    int same_lcs(bool query_string,int aend,int bend) const {
        if(aend<0 || bend<0)return 0;
        assert(aend<n && bend<n);
        const int offset=query_string?0:n+1;
        return std::min({aend+1,bend+1,reverse_.lcp_rank(reverse_.rank[offset+n-1-aend],reverse_.rank[offset+n-1-bend])});
    }
    std::pair<int,int> interval(int qstart,int length) const {
        assert(qstart>=0 && qstart<n && length>0 && length<=n-qstart);
        return forward_.prefix_interval(forward_.rank[qstart],length);
    }
    int position(std::pair<int,int> ranks,int lower,int upper,bool successor) const {
        lower=std::max(0,lower);upper=std::min(n-1,upper);
        return positions_.position(ranks.first,ranks.second,lower,upper,successor);
    }
    int R(int t,int length,int cutoff=-1) const {
        assert(t>=0 && t<n);
        if(cutoff<0)cutoff=t;else cutoff=std::min(cutoff,t);
        if(length<=0 || length>t+1 || length>cutoff)return -1;
        RCacheEntry* cached=nullptr;
        if(cache_skip_){--cache_skip_;++cache_bypasses;}
        else if(!r_cache_.empty()) {
            const std::uint64_t hash=mix(cache_seed_^std::uint64_t(t)*UINT64_C(0x9e3779b97f4a7c15)
                ^std::uint64_t(length)*UINT64_C(0xd6e8feb86659fd93)^std::uint64_t(cutoff)*UINT64_C(0xa0761d6478bd642f));
            cached=&r_cache_[std::size_t(hash)&(r_cache_.size()-1)];
            if(cached->t==t && cached->length==length && cached->cutoff==cutoff) {
                ++cache_hits;observe_cache(true);return cached->result;
            }
            ++cache_misses;cache_evictions+=cached->t>=0;observe_cache(false);
        }
        const auto ranks=interval(t-length+1,length);
        const int start=positions_.greatest_below(ranks.first,ranks.second,cutoff-length+1);
        const int result=start<0?-1:start+length-1;
        if(cached)*cached=RCacheEntry{t,length,cutoff,result};
        return result;
    }
    Route best(int t,int cutoff=-1) const {
        assert(t>=0 && t<n);
        if(cutoff<0 || cutoff>=t)return baseline_[t];
        return best_search(t,cutoff);
    }
    Route deleted(int t,int owner) const {
        const Route original=best(t);
        if(!original.length || owner<original.end-original.length+1 || owner>original.end)return original;
        const Route left=best(t,owner);
        int lo=0,hi=std::min(original.length,t-owner-1);
        if(left.length>hi)return left;
        if(left.length) {
            const int end=R(t,left.length);
            if(end<owner+left.length)return left;
            lo=left.length;
            if(lo==hi)return {lo,end};
        }
        while(lo<hi){const int mid=lo+(hi-lo+1)/2;if(R(t,mid)>=owner+mid)lo=mid;else hi=mid-1;}
        const Route right=lo?Route{lo,R(t,lo)}:Route{};
        return std::max(left,right);
    }
    std::size_t workspace_bytes() const {
        return (q.capacity()+k.capacity())*sizeof(std::uint64_t)+baseline_.capacity()*sizeof(Route)
            +r_cache_.capacity()*sizeof(RCacheEntry)+forward_.workspace_bytes()+reverse_.workspace_bytes()
            +positions_.workspace_bytes()+prefix_ranks_.workspace_bytes();
    }
};
} // namespace rosa_subq
