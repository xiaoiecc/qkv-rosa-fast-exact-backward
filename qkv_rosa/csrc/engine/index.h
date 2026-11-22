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

// A bounded-probe accelerator for an ordered set of complete packed words.
// Four unsuccessful probes fall back to binary search, so even adversarial
// collisions preserve O(log n) lookup and never alter symbol equality.
class ExactWordRanks {
    std::vector<std::uint64_t> words_;
    std::vector<int> slots_;
    static std::uint64_t mix(std::uint64_t x) {
        x^=x>>30;x*=UINT64_C(0xbf58476d1ce4e5b9);
        x^=x>>27;x*=UINT64_C(0x94d049bb133111eb);return x^(x>>31);
    }
public:
    ExactWordRanks()=default;
    explicit ExactWordRanks(std::vector<std::uint64_t> sorted_unique) { build(std::move(sorted_unique)); }
    void build(std::vector<std::uint64_t> sorted_unique) {
        words_=std::move(sorted_unique);
        std::size_t size=2;
        while(size<words_.size()*2)size*=2;
        slots_.assign(size,-1);
        for(int id=0;id<int(words_.size());++id) {
            std::size_t at=std::size_t(mix(words_[id]))&(size-1);
            for(int probe=0;probe<4;++probe,at=(at+1)&(size-1))
                if(slots_[at]<0){slots_[at]=id;break;}
        }
    }
    int find(std::uint64_t word) const {
        if(slots_.empty())return -1;
        const std::size_t mask=slots_.size()-1;
        std::size_t at=std::size_t(mix(word))&mask;
        for(int probe=0;probe<4;++probe,at=(at+1)&mask) {
            const int id=slots_[at];
            if(id<0)return -1;
            if(words_[id]==word)return id;
        }
        const auto it=std::lower_bound(words_.begin(),words_.end(),word);
        return it==words_.end() || *it!=word?-1:int(it-words_.begin());
    }
    std::size_t workspace_bytes() const {
        return words_.capacity()*sizeof(std::uint64_t)+slots_.capacity()*sizeof(int);
    }
};

class SuffixArray {
    int size_=0;
    std::vector<unsigned char> log_;
    std::vector<std::vector<int>> minimum_;
public:
    std::vector<int> sa,rank;
    SuffixArray()=default;
    explicit SuffixArray(const std::vector<int>& text) { build(text); }
    int size() const { return size_; }
    void build(const std::vector<int>& text,int dense_alphabet=0) {
        if(text.size()>std::size_t(std::numeric_limits<int>::max()/2))
            throw std::length_error("suffix index exceeds int32 range");
        size_=int(text.size());const int n=size_;
        sa.resize(n);rank.resize(n);minimum_.clear();log_.assign(std::size_t(n)+1,0);
        if(!n)return;
        std::vector<int> work(n),next_rank(n),count(std::size_t(n)+1);
        if(dense_alphabet>0 && dense_alphabet<=n) {
            for(int x:text) {
                if(x<0 || x>=dense_alphabet)throw std::invalid_argument("invalid compressed symbol");
                ++count[x];
            }
            int sum=0;
            for(int x=0;x<dense_alphabet;++x){const int z=count[x];count[x]=sum;sum+=z;}
            for(int i=0;i<n;++i)sa[count[text[i]]++]=i;
        } else {
            std::iota(sa.begin(),sa.end(),0);
            std::sort(sa.begin(),sa.end(),[&](int a,int b){return text[a]<text[b];});
        }
        int classes=1;rank[sa[0]]=0;
        for(int z=1;z<n;++z) {classes+=text[sa[z-1]]!=text[sa[z]];rank[sa[z]]=classes-1;}
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
    mutable SuffixArray forward_,reverse_;
    mutable OccurrenceWavelet positions_,prefix_ranks_;
    mutable std::vector<int> forward_text_,reverse_text_;
    mutable bool symbols_ready_=false,forward_ready_=false,reverse_ready_=false;
    mutable bool positions_ready_=false,prefix_ready_=false;
    mutable int alphabet_size_=0;
    mutable std::vector<int> query_class_,end_offsets_,ends_,latest_equal_;
    std::vector<int> q_left_,k_left_,q_right_,k_right_;
    std::vector<Route> baseline_;
    mutable std::vector<RCacheEntry> r_cache_;
    mutable unsigned cache_window_=0,cache_good_=0,cache_skip_=0;
    bool cache_enabled_=false;
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
    void ensure_symbols() const {
        if(symbols_ready_)return;
        std::vector<std::uint64_t> alphabet=q;alphabet.insert(alphabet.end(),k.begin(),k.end());
        std::sort(alphabet.begin(),alphabet.end());alphabet.erase(std::unique(alphabet.begin(),alphabet.end()),alphabet.end());
        alphabet_size_=int(alphabet.size());
        const ExactWordRanks symbol_ranks(std::move(alphabet));
        forward_text_.resize(std::size_t(2)*n+2);reverse_text_.resize(forward_text_.size());
        auto& f=forward_text_;auto& r=reverse_text_;
        query_class_.resize(n);end_offsets_.assign(std::size_t(alphabet_size_)+1,0);
        for(int i=0;i<n;++i) {
            const int a=symbol_ranks.find(q[i]),b=symbol_ranks.find(k[i]);
            assert(a>=0 && b>=0);query_class_[i]=a;++end_offsets_[b+1];
            f[i]=2+a;f[n+1+i]=2+b;
            r[n-1-i]=f[i];r[2*n-i]=f[n+1+i];
        }
        f[n]=r[n]=0;f[2*n+1]=r[2*n+1]=1;
        for(int c=0;c<alphabet_size_;++c)end_offsets_[c+1]+=end_offsets_[c];
        std::vector<int> cursor=end_offsets_,latest(alphabet_size_,-1);
        ends_.resize(n);latest_equal_.resize(n);
        for(int t=0;t<n;++t) {
            const int group=f[n+1+t]-2;
            ends_[cursor[group]++]=t;latest_equal_[t]=latest[query_class_[t]];
            latest[group]=t;
        }
        symbols_ready_=true;
    }
    void ensure_forward() const {
        if(forward_ready_)return;
        ensure_symbols();
        forward_.build(forward_text_,alphabet_size_+2);
        std::vector<int>().swap(forward_text_);forward_ready_=true;
    }
    void ensure_reverse() const {
        if(reverse_ready_)return;
        ensure_symbols();
        reverse_.build(reverse_text_,alphabet_size_+2);
        std::vector<int>().swap(reverse_text_);reverse_ready_=true;
    }
    void ensure_positions() const {
        if(positions_ready_)return;
        ensure_forward();std::vector<int> values(std::size_t(2)*n+2,0);
        for(int s=0;s<n;++s)values[forward_.rank[n+1+s]]=s+1;
        positions_.build(std::move(values),n);positions_ready_=true;
    }
    void ensure_prefix_ranks() const {
        if(prefix_ready_)return;
        ensure_reverse();
        std::vector<int> values(n);
        for(int end=0;end<n;++end)values[end]=reverse_.rank[2*n-end]+1;
        prefix_ranks_.build(std::move(values),reverse_.size());prefix_ready_=true;
    }
    std::pair<int,int> last_two_equal(int t,int cutoff) const {
        if(cutoff>0 && k[cutoff-1]==q[t]) {
            if(cutoff==1)return {0,-1};
            if(k[cutoff-2]==q[t])return {cutoff-1,cutoff-2};
        }
        ensure_symbols();
        const int group=query_class_[t];
        const auto begin=ends_.begin()+end_offsets_[group],end=ends_.begin()+end_offsets_[group+1];
        const auto stop=std::lower_bound(begin,end,cutoff);
        if(stop==begin)return {-1,-1};
        return {*(stop-1),stop-begin>1?*(stop-2):-1};
    }
    int latest_before(int t,int cutoff) const {
        if(cutoff>0 && k[cutoff-1]==q[t])return cutoff-1;
        ensure_symbols();
        return cutoff==t?latest_equal_[t]:last_two_equal(t,cutoff).first;
    }
    // A run comparison either proves the complete LCP or says "use the index".
    // Only two boundaries are inspected, even for adversarial alternating
    // strings. This is O(1) per call, not a scan hidden inside an LCE query.
    static int run_match(const std::vector<std::uint64_t>& a,const std::vector<int>& ar,
        int ai,const std::vector<std::uint64_t>& b,const std::vector<int>& br,int bi,
        int limit,int step) {
        int matched=0;
        for(int boundary=0;boundary<2;++boundary) {
            if(a[ai]!=b[bi])return matched;
            const int al=ar[ai],bl=br[bi],take=std::min(al,bl);
            matched+=take;
            if(al!=bl || matched==limit)return matched;
            ai+=step*take;bi+=step*take;
        }
        return a[ai]!=b[bi]?matched:-1;
    }
    Route best_search(int t,int cutoff) const {
        if(cutoff<=0)return {};
        const auto last=last_two_equal(t,cutoff);
        if(last.first<0)return {};
        const int recent_length=lcs(t,last.first);
        // All other matches end no later than the previous occurrence of the
        // last symbol. Their lengths cannot exceed previous_end+1. The known
        // full-prefix maximum gives another exact upper bound after baseline.
        if(recent_length>=last.second+1 ||
            (baseline_.size()>std::size_t(t) && recent_length>=baseline_[t].length))
            return {recent_length,last.first};
        // Among an arbitrary subset of suffixes, the longest common prefix
        // with a query occurs at its lexicographic predecessor or successor.
        // prefix_ranks_ is time -> reversed suffix rank, so one range-rank
        // query finds both neighbours among endpoints < cutoff. This removes
        // binary search over match lengths from every deletion prefix query.
        ensure_prefix_ranks();const int rank=reverse_.rank[n-1-t];
        const int below=prefix_ranks_.count_less(0,cutoff,rank+1);int length=0;
        if(below)length=reverse_.lcp_rank(rank,prefix_ranks_.quantile(0,cutoff,below-1)-1);
        if(below<cutoff)length=std::max(length,reverse_.lcp_rank(rank,prefix_ranks_.quantile(0,cutoff,below)-1));
        return length?Route{length,recent_length==length?last.first:R(t,length,cutoff)}:Route{};
    }
public:
    int n=0;
    std::vector<std::uint64_t> q,k;
    // A stream owns its Index. Optional cache mutation must not be shared by
    // simultaneous calls; independent streams already have independent indices.
    mutable std::uint64_t cache_hits=0,cache_misses=0,cache_evictions=0,cache_bypasses=0;
    Index(const std::uint64_t* query,const std::uint64_t* key,int length,
          bool randomized=false,std::uint64_t seed=0,
          const std::vector<Route>* known_baseline=nullptr):cache_enabled_(randomized),cache_seed_(seed),n(length) {
        if(n<1 || n>(std::numeric_limits<int>::max()-2)/4 || !query || !key)
            throw std::invalid_argument("invalid subquadratic index input");
        q.assign(query,query+n);k.assign(key,key+n);
        q_left_.resize(n);k_left_.resize(n);q_right_.resize(n);k_right_.resize(n);
        for(int i=0;i<n;++i) {
            q_left_[i]=i && q[i-1]==q[i]?q_left_[i-1]+1:1;
            k_left_[i]=i && k[i-1]==k[i]?k_left_[i-1]+1:1;
        }
        for(int i=n-1;i>=0;--i) {
            q_right_[i]=i+1<n && q[i+1]==q[i]?q_right_[i+1]+1:1;
            k_right_[i]=i+1<n && k[i+1]==k[i]?k_right_[i+1]+1:1;
        }
        if(known_baseline) {
            if(known_baseline->size()!=std::size_t(n))throw std::invalid_argument("invalid saved baseline size");
            for(int t=0;t<n;++t) {
                const Route route=(*known_baseline)[t];
                if(route.length<0 || route.length>t || route.end<-1 || route.end>=t ||
                    (route.length==0)!=(route.end<0) || route.length>route.end+1)
                    throw std::invalid_argument("invalid saved baseline route");
            }
            baseline_=*known_baseline;
        } else {
            baseline_.reserve(n);for(int t=0;t<n;++t)baseline_.push_back(best_search(t,t));
        }
    }
    const SuffixArray& forward_suffixes() const { ensure_forward();return forward_; }
    const SuffixArray& reverse_suffixes() const { ensure_reverse();return reverse_; }
    const std::vector<Route>& baseline() const { return baseline_; }
    int lce(int qstart,int kstart) const {
        if(qstart>=n || kstart>=n)return 0;
        assert(qstart>=0 && kstart>=0);
        const int direct=run_match(q,q_right_,qstart,k,k_right_,kstart,std::min(n-qstart,n-kstart),1);
        if(direct>=0)return direct;
        ensure_forward();
        return std::min({n-qstart,n-kstart,forward_.lcp_rank(forward_.rank[qstart],forward_.rank[n+1+kstart])});
    }
    int lcs(int qend,int kend) const {
        if(qend<0 || kend<0)return 0;
        assert(qend<n && kend<n);
        const int direct=run_match(q,q_left_,qend,k,k_left_,kend,std::min(qend+1,kend+1),-1);
        if(direct>=0)return direct;
        ensure_reverse();
        return std::min({qend+1,kend+1,reverse_.lcp_rank(reverse_.rank[n-1-qend],reverse_.rank[2*n-kend])});
    }
    int same_lcs(bool query_string,int aend,int bend) const {
        if(aend<0 || bend<0)return 0;
        assert(aend<n && bend<n);
        if(aend==bend)return aend+1;
        const auto& words=query_string?q:k;const auto& runs=query_string?q_left_:k_left_;
        const int direct=run_match(words,runs,aend,words,runs,bend,std::min(aend+1,bend+1),-1);
        if(direct>=0)return direct;
        ensure_reverse();
        const int offset=query_string?0:n+1;
        return std::min({aend+1,bend+1,reverse_.lcp_rank(reverse_.rank[offset+n-1-aend],reverse_.rank[offset+n-1-bend])});
    }
    int same_lce(bool query_string,int astart,int bstart) const {
        if(astart>=n || bstart>=n)return 0;
        assert(astart>=0 && bstart>=0);
        if(astart==bstart)return n-astart;
        const auto& words=query_string?q:k;const auto& runs=query_string?q_right_:k_right_;
        const int direct=run_match(words,runs,astart,words,runs,bstart,std::min(n-astart,n-bstart),1);
        if(direct>=0)return direct;
        ensure_forward();const int offset=query_string?0:n+1;
        return std::min({n-astart,n-bstart,forward_.lcp_rank(forward_.rank[offset+astart],forward_.rank[offset+bstart])});
    }
    std::pair<int,int> interval(int qstart,int length) const {
        assert(qstart>=0 && qstart<n && length>0 && length<=n-qstart);
        ensure_forward();
        return forward_.prefix_interval(forward_.rank[qstart],length);
    }
    int position(std::pair<int,int> ranks,int lower,int upper,bool successor) const {
        lower=std::max(0,lower);upper=std::min(n-1,upper);
        ensure_positions();
        return positions_.position(ranks.first,ranks.second,lower,upper,successor);
    }
    int occurrence(int qstart,int length,int lower,int upper,bool successor,
        std::pair<int,int>* cached_ranks=nullptr) const {
        assert(qstart>=0 && qstart<n && length>0 && length<=n-qstart);
        lower=std::max(0,lower);upper=std::min(n-length,upper);
        if(lower>upper)return -1;
        // If the requested boundary itself matches, it is already the first
        // (last) occurrence. No suffix interval or wavelet query is needed.
        const int candidate=successor?lower:upper;
        if(lce(qstart,candidate)>=length)return candidate;
        if(!cached_ranks)return position(interval(qstart,length),lower,upper,successor);
        if(cached_ranks->first<0)*cached_ranks=interval(qstart,length);
        return position(*cached_ranks,lower,upper,successor);
    }
    int R(int t,int length,int cutoff=-1) const {
        assert(t>=0 && t<n);
        if(cutoff<0)cutoff=t;else cutoff=std::min(cutoff,t);
        if(length<=0 || length>t+1 || length>cutoff)return -1;
        if(length==1)return latest_before(t,cutoff);
        const int recent=latest_before(t,cutoff);
        if(recent<length-1)return -1;
        if(lcs(t,recent)>=length)return recent;
        // Cache only genuine interval queries. Exact boundary certificates
        // need neither a hash probe nor an allocated cache, even when optional
        // randomized memoization is enabled for the enclosing backward call.
        RCacheEntry* cached=nullptr;
        if(cache_skip_){--cache_skip_;++cache_bypasses;}
        else if(cache_enabled_) {
            if(r_cache_.empty()) {
                std::size_t slots=1;while(slots<std::size_t(n)*2)slots*=2;
                r_cache_.resize(slots);
            }
            const std::uint64_t hash=mix(cache_seed_^std::uint64_t(t)*UINT64_C(0x9e3779b97f4a7c15)
                ^std::uint64_t(length)*UINT64_C(0xd6e8feb86659fd93)^std::uint64_t(cutoff)*UINT64_C(0xa0761d6478bd642f));
            cached=&r_cache_[std::size_t(hash)&(r_cache_.size()-1)];
            if(cached->t==t && cached->length==length && cached->cutoff==cutoff) {
                ++cache_hits;observe_cache(true);return cached->result;
            }
            ++cache_misses;cache_evictions+=cached->t>=0;observe_cache(false);
        }
        const auto ranks=interval(t-length+1,length);ensure_positions();
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
            +positions_.workspace_bytes()+prefix_ranks_.workspace_bytes()
            +(forward_text_.capacity()+reverse_text_.capacity()+query_class_.capacity()+end_offsets_.capacity()
                +ends_.capacity()+latest_equal_.capacity()+q_left_.capacity()+k_left_.capacity()
                +q_right_.capacity()+k_right_.capacity())*sizeof(int);
    }
};
} // namespace rosa_subq
