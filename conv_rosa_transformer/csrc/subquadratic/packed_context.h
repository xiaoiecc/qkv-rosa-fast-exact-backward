#pragma once
#include "types.h"
#include "near_k.h"

namespace rosa_subq {

// All exact arms are at most L, so even a singly edited match has at most
// 2L+1 symbols. If these fit, the least differing bit gives an exact LCS.
class PackedContext {
    struct Level {
        uint64_t mask=0;
        std::vector<int> offsets,ends,active;
        int group(uint64_t code) const {return int(code&mask);}
    };
    int n,bits,log_bits,depth;
    int cutoff_=0;
    std::vector<uint64_t> query_,key_;
    std::vector<Level> levels_;
    uint64_t work_=0,budget_;
    bool charge(int count) {work_+=uint64_t(count);return work_<=budget_;}
public:
    static bool fits(int bits,int longest) {
        return (bits==1 || bits==2 || bits==4 || bits==8) && 2LL*longest+1<=64/bits;
    }
    PackedContext(const uint64_t* q,const uint64_t* k,int count,int width,int longest,uint64_t budget)
        :n(count),bits(width),log_bits(near_detail::one_bit_index(uint64_t(width))),depth(1),
         query_(n),key_(n),budget_(budget) {
        // Keep the top directory linear in T even for a small alphabet.
        while(depth<std::max(1,2*longest+1) && (depth+1)*bits<29
              && (UINT64_C(1)<<((depth+1)*bits))<=uint64_t(n)*2)++depth;
        uint64_t a=0,b=0;
        for(int t=0;t<n;++t){a=(a<<bits)|q[t];b=(b<<bits)|k[t];query_[t]=a;key_[t]=b;}
        levels_.resize(depth+1);
        for(int h=1;h<=depth;++h) {
            auto& level=levels_[h];
            const int groups=1<<(bits*h);
            level.mask=groups-1;
            level.offsets.assign(size_t(groups)+1,0);
            for(int r=h-1;r<n-1;++r)++level.offsets[level.group(key_[r])+1];
            for(int c=0;c<groups;++c)level.offsets[c+1]+=level.offsets[c];
            level.active=level.offsets;
            level.ends.resize(level.offsets.back());
            for(int r=h-1;r<n-1;++r)level.ends[level.active[level.group(key_[r])]++]=r;
            level.active=level.offsets;
        }
    }
    void advance(int t) {
        for(;cutoff_<t;++cutoff_)
            for(int h=1;h<=std::min(depth,cutoff_+1);++h)
                ++levels_[h].active[levels_[h].group(key_[cutoff_])];
    }
    uint64_t query(int t) const {return query_[t];}
    int match(uint64_t word,int r) const {
        const uint64_t diff=word^key_[r];
        const int length=diff?(near_detail::one_bit_index(diff&(-diff))>>log_bits):64/bits;
        return std::min(length,r+1);
    }
    int lcs(int t,int r) const {return r<0?0:match(query_[t],r);}
    bool best(uint64_t word,int cutoff,Route& out) {
        out={};
        for(int h=std::min(depth,cutoff);h>=1;--h) {
            const auto& level=levels_[h];const int c=level.group(word);
            const int first=level.offsets[c],end=level.active[c];
            if(first==end)continue;
            const int stop=cutoff==cutoff_?end:int(std::lower_bound(level.ends.begin()+first,
                level.ends.begin()+end,cutoff)-level.ends.begin());
            if(stop==first)continue;
            if(h<depth){out={h,level.ends[stop-1]};return true;}
            if(!charge(stop-first))return false;
            for(int i=first;i<stop;++i) {
                const int r=level.ends[i];out=std::max(out,Route{match(word,r),r});
            }
            return true;
        }
        return true;
    }
    int latest(uint64_t word,int h) {
        const auto& level=levels_[std::min(h,depth)];
        const int c=level.group(word),first=level.offsets[c],stop=level.active[c];
        if(h<=depth)return stop==first?-1:level.ends[stop-1];
        for(int i=stop;i>first;) {
            if(!charge(1))return -2;
            const int r=level.ends[--i];
            if(match(word,r)>=h)return r;
        }
        return -1;
    }
    template<class Emit>
    bool neighbors(uint64_t word,int h,Emit emit) {
        h=std::min(h,depth);
        const auto& level=levels_[h];
        const int original=level.group(word);
        for(int j=-1;j<h*bits;++j) {
            const int c=j<0?original:original^(1<<j);
            const int first=level.offsets[c],stop=level.active[c];
            if(!charge(stop-first+1))return false;
            for(int i=first;i<stop;++i)emit(level.ends[i]);
        }
        return true;
    }
    size_t bytes() const {
        size_t size=(query_.capacity()+key_.capacity())*sizeof(uint64_t);
        for(const auto& l:levels_)size+=(l.ends.capacity()+l.offsets.capacity()+l.active.capacity())*sizeof(int);
        return size;
    }
    uint64_t probes() const {return work_;}
};

// Use the directory for both directions. For K repairs beating an intact
// winner, a prefix of the baseline-length suffix contains at most one bit
// mismatch. Shorter repairs can win only near the damaged winning occurrence.
inline bool packed_credit(const uint64_t* q,const uint64_t* k,const std::vector<Route>& base,
        int bits,int longest,const Scores& score,std::vector<double>& cq,std::vector<double>& ck,
        Stats& stats,uint64_t budget) {
    const int n=score.n;
    if(!PackedContext::fits(bits,longest))return false;
    for(int i=0;i<n;++i)if((q[i]|k[i])>>bits)return false;
    PackedContext table(q,k,n,bits,longest,budget);
    std::vector<int> latest(size_t(longest)+1),seen(n,-1);
    std::vector<Route> removed(longest),winners(size_t(n)*bits);
    std::vector<size_t> touched;
    std::vector<double> qsum(cq.size()),ksum(ck.size()),deleted(n);
    for(int t=1;t<n;++t) {
        table.advance(t);
        const double* w=score.row(t);
        if(std::none_of(w,w+score.d,[](double x){return x!=0.;}))continue;
        const Route original=base[t];const int L=original.length,first=original.end-L+1;
        const uint64_t word=table.query(t);
        auto delta=[&](int end,int before) {
            const double* a=score.value(end);const double* b=score.value(before);
            double sum=0;for(int c=0;c<score.d;++c)sum+=w[c]*(a[c]-b[c]);
            return sum;
        };
        for(int age=0;age<=L && age<t;++age)for(int bit=0;bit<bits;++bit) {
            Route winner;
            if(!table.best(word^(UINT64_C(1)<<(age*bits+bit)),t,winner))return false;
            if(winner.end!=original.end)qsum[size_t(t-age)*bits+bit]+=delta(winner.end,original.end);
        }
        if(L) {
            for(int h=1;h<=L;) {
                const int end=table.latest(word,h);
                if(end<0)return false;
                const int high=end==original.end?L:table.lcs(t,end);
                if(high<h || high>L)return false;
                std::fill(latest.begin()+h,latest.begin()+high+1,end);h=high+1;
            }
            Route prefix;
            if(!table.best(word,first,prefix))return false;
            int h=L;
            for(int s=first;s<=original.end;++s) {
                if(s>first) {
                    const int len=table.lcs(t,s-1);
                    if(len)prefix=std::max(prefix,Route{len,s-1});
                }
                while(h && latest[h]-h<s)--h;
                removed[s-first]=std::max(prefix,h?Route{h,latest[h]}:Route{});
                if(removed[s-first].end!=original.end)deleted[s]+=delta(removed[s-first].end,original.end);
            }
        }
        touched.clear();
        auto consider=[&](int end) {
            if(seen[end]==t)return;
            seen[end]=t;
            const int a=table.lcs(t,end),s=end-a,u=t-a;
            if(s<0)return;
            const auto mask=q[u]^k[s];
            if(!mask || (mask&(mask-1)))return;
            const int bit=near_detail::one_bit_index(mask);
            if(bit>=bits)return;
            const Route candidate{a+1+table.lcs(u-1,s-1),end};
            const Route other=s>=first && s<=original.end?removed[s-first]:original;
            if(!(candidate>other))return;
            const size_t at=size_t(s)*bits+bit;
            if(!winners[at].length)touched.push_back(at);
            winners[at]=std::max(winners[at],candidate);
        };
        if(!table.neighbors(word,std::max(1,L),consider))return false;
        if(L)for(int r=first;r<std::min(t,original.end+L);++r)consider(r);
        for(size_t at:touched) {
            const int s=int(at/bits);
            const Route other=s>=first && s<=original.end?removed[s-first]:original;
            if(winners[at].end!=other.end)ksum[at]+=delta(winners[at].end,other.end);
            winners[at]={};
        }
    }
    for(int s=0;s<n;++s)for(int bit=0;bit<bits;++bit)ksum[size_t(s)*bits+bit]+=deleted[s];
    cq.swap(qsum);ck.swap(ksum);
    stats.suffix_queries+=table.probes();
    stats.peak_workspace_bytes=std::max<uint64_t>(stats.peak_workspace_bytes,table.bytes()
        +(latest.capacity()+seen.capacity())*sizeof(int)+(removed.capacity()+winners.capacity())*sizeof(Route)
        +touched.capacity()*sizeof(size_t)+(qsum.capacity()+ksum.capacity()+deleted.capacity())*sizeof(double));
    return true;
}
} // namespace rosa_subq
