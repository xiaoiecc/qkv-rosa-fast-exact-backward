#pragma once
#include "types.h"
#include "../approx_route_helpers.h"

namespace rosa_subq {

// A complete two-symbol neighborhood suffices beyond the one-symbol
// control. Bound the exact arms to keep exceptional enumerations cheap.
inline bool bigram_eligible(int n,int bits,const std::vector<Route>& base) {
    if(n<2 || bits<4 || bits>16)return false;
    return std::all_of(base.begin(),base.end(),[](Route r){return r.length<=12;});
}

namespace bigram_detail {
inline Route single(int end) {return end<0?Route{}:Route{1,end};}

inline double difference(const Scores& score,const double* weight,int to,int from) {
    if(to==from)return 0.;
    const double* a=score.value(to);
    const double* b=score.value(from);
    double sum=0.;
    for(int c=0;c<score.d;++c)sum+=weight[c]*(a[c]-b[c]);
    return sum;
}

// Exact candidate lists for adjacent packed words. A bounded directory may
// alias keys; the full packed pair is always compared on the report path.
class PairDirectory {
    std::vector<int> heads,next;
    std::vector<uint32_t> pairs;
    int shift=0;
    bool direct;
    int bucket(uint32_t code) const {
        return direct?int(code):int((code*UINT32_C(2654435761))>>shift);
    }
public:
    PairDirectory(int n,int bits):next(n,-1),pairs(n) {
        const uint64_t alphabet=UINT64_C(1)<<(bits*2);
        int cap=1,log=0;
        while(log<24 && uint64_t(cap)<alphabet && uint64_t(cap)<uint64_t(n)*16){cap*=2;++log;}
        heads.assign(cap,-1);direct=uint64_t(cap)==alphabet;shift=32-log;
    }
    // Count the very same bucket links visited by the reduction, aliases
    // included. Each head stores frequency minus one until admission.
    bool prepare(const uint64_t* q,const uint64_t* k,int bits,uint64_t limit) {
        const int n=int(pairs.size());
        const uint64_t alphabet=UINT64_C(1)<<bits;
        int largest=0;
        for(int i=0;i<n;++i) {
            if((q[i]|k[i])>=alphabet)return false;
            if(i>0 && i<n-1) {
                pairs[i]=uint32_t((k[i-1]<<bits)|k[i]);
                const int count=++heads[bucket(pairs[i])]+1;
                largest=std::max(largest,count);
            }
        }
        const uint64_t probes=uint64_t(n-2)*(1+2*bits);
        if(!largest || probes<=limit/uint64_t(largest)) {
            std::fill(heads.begin(),heads.end(),-1);
            return true;
        }
        uint64_t visits=0;
        // At t the frequencies cover precisely 1 <= r < t. Removing
        // r=t-1 also restores all heads to -1 on an admitted stream.
        for(int t=n-1;t>=2;--t) {
            const uint32_t pair=uint32_t((q[t-1]<<bits)|q[t]);
            uint64_t row=heads[bucket(pair)]+1;
            for(int j=0;j<2*bits;++j)row+=heads[bucket(pair^(UINT32_C(1)<<j))]+1;
            if(row>limit-visits)return false;
            visits+=row;
            --heads[bucket(pairs[t-1])];
        }
        return true;
    }
    void add(int end) {
        auto& head=heads[bucket(pairs[end])];next[end]=head;head=end;
    }
    template<class F>
    void visit(uint32_t code,uint64_t& visits,F&& emit) const {
        for(int r=heads[bucket(code)];r>=0;r=next[r]) {
            ++visits;
            if(direct || pairs[r]==code)emit(r);
        }
    }
    size_t bytes() const {return (heads.capacity()+next.capacity())*sizeof(int)+pairs.capacity()*sizeof(uint32_t);}
};
} // namespace bigram_detail

inline bool bigram_credit(const uint64_t* q,const uint64_t* k,const std::vector<Route>& base,
        int bits,const Scores& score,std::vector<double>& cq,std::vector<double>& ck,Stats& stats,
        bigram_detail::PairDirectory& directory) {
    using bigram_detail::difference;
    using bigram_detail::single;
    const int n=score.n;
    const int sigma=1<<bits;
    int longest=0;
    for(const Route r:base)longest=std::max(longest,r.length);
    const size_t nd=size_t(n)*bits;
    std::vector<double> qsum(nd),ksum(nd),deleted(n);
    std::vector<int> latest(sigma,-1),previous(n,-1);
    std::vector<Route> qbest(size_t(longest+1)*bits),kbest(nd),survives(longest);
    std::vector<size_t> touched;
    std::vector<int> end_at_length(size_t(longest)+1);
    uint64_t visits=0;
    // LCS for a particular diagonal. Every inspected exact arm is bounded
    // by the certified longest original causal match.
    const auto lcs=[&](int t,int r){
        int a=0;while(r>=0 && q[t]==k[r]){--t;--r;++a;}return a;
    };
    for(int t=1;t<n;++t) {
        const int last=t-1,word=int(q[t]);
        previous[last]=latest[k[last]];latest[k[last]]=last;
        if(last)directory.add(last);
        const Route old=base[t];
        const int L=old.length,first=old.end-L+1;
        const int gram=latest[word];
        if((L==0)!=(gram<0))return false;
        auto del=[&](int s){return s>=first && s<=old.end?survives[s-first]:old;};
        std::fill_n(qbest.begin(),size_t(L+1)*bits,Route{});
        std::fill_n(end_at_length.begin(),L+1,-1);
        if(L) {
            end_at_length[1]=gram;
            for(int s=first;s<=old.end;++s)
                survives[s-first]=single(s==gram?previous[s]:gram);
        }
        // A flip at the query's last symbol can always use the latest
        // one-symbol occurrence. Older flips are seeded by their survivors.
        for(int j=0;j<bits;++j)qbest[j]=single(latest[word^(1<<j)]);
        touched.clear();
        const auto repair=[&](int r,int a,int bit,int len){
            const int s=r-a;
            const Route route{len,r};
            auto& qb=qbest[size_t(a)*bits+bit];qb=std::max(qb,route);
            // A candidate equal to or below an unaffected winner adds no
            // K credit. Within its damage window the survivor is settled later.
            if((s<first || s>old.end) && !(route>old))return;
            const size_t idx=size_t(s)*bits+bit;
            if(!kbest[idx].length)touched.push_back(idx);
            kbest[idx]=std::max(kbest[idx],route);
        };
        // r<t and r>=1 wherever a two-word occurrence is reported.
        const uint32_t pair=uint32_t((q[t-1]<<bits)|q[t]);
        bool valid=true;
        directory.visit(pair,visits,[&](int r){
            const int a=2+lcs(t-2,r-2);
            if(a>L){valid=false;return;}
            end_at_length[a]=std::max(end_at_length[a],r);
            for(int s=first;s<=old.end;++s) {
                const int len=s<=r && s>r-a? r-s:a;
                if(len)survives[s-first]=std::max(survives[s-first],Route{len,r});
            }
            if(r>=a) {
                const uint64_t mask=q[t-a]^k[r-a];
                if(mask && !(mask&(mask-1)))
                    repair(r,a,rosa_random::row_bit(mask),a+1+lcs(t-a-1,r-a-1));
            }
        });
        if(!valid)return false;
        for(int j=0;j<2*bits;++j) {
            directory.visit(pair^(uint32_t(1)<<j),visits,[&](int r){
                const int age=j/bits;
                repair(r,age,j-age*bits,2+lcs(t-2,r-2));
            });
        }
        const double* wt=score.row(t);
        for(int h=L-1;h>=1;--h)end_at_length[h]=std::max(end_at_length[h],end_at_length[h+1]);
        for(int a=0;a<=L && a<t;++a) {
            const Route survivor=a>=L?old:(a?Route{a,end_at_length[a]}:Route{});
            for(int j=0;j<bits;++j) {
                const int end=std::max(survivor,qbest[size_t(a)*bits+j]).end;
                if(end!=old.end)qsum[size_t(t-a)*bits+j]+=difference(score,wt,end,old.end);
            }
        }
        for(int s=first;s<=old.end;++s)if(survives[s-first].end!=old.end)
            deleted[s]+=difference(score,wt,survives[s-first].end,old.end);
        for(size_t i:touched) {
            const int s=int(i/bits),j=int(i%bits);
            Route before=del(s);
            // The reverse interval sum below already supplies this seed.
            if(L<=1 && s>old.end && (q[t]^k[s])==(UINT64_C(1)<<j))before=single(s);
            if(kbest[i]>before)ksum[i]+=difference(score,wt,kbest[i].end,before.end);
            kbest[i]={};
        }
    }
    // For K[s] -> x, length one wins exactly on rows with L<=1 between
    // s and the next occurrence of x. Throughout this interval the previous
    // occurrence of x supplies the same baseline endpoint. Accumulate the
    // row weights in reverse until an occurrence of x resets the class.
    // Allocate a row of numerical accumulation only for observed Q words
    // once the nominal alphabet is larger than the input.
    std::vector<int> compact;
    int classes=sigma;
    if(sigma>n) {
        compact.assign(sigma,-1);classes=0;
        for(int t=1;t<n;++t)if(compact[q[t]]<0)compact[q[t]]=classes++;
    }
    const auto id=[&](int word){return compact.empty()?word:compact[word];};
    std::vector<double> weights(size_t(classes)*score.d);
    std::vector<int> count(classes);
    for(int s=n-2;s>=0;--s) {
        const int x=int(k[s]),t=s+1,query=id(int(q[t]));
        if(base[t].length<=1) {
            double* dst=weights.data()+size_t(query)*score.d;
            for(int c=0;c<score.d;++c)dst[c]+=score.row(t)[c];
            ++count[query];
        }
        for(int j=0;j<bits;++j) {
            const int target=x^(1<<j),group=id(target);
            double delta=deleted[s];
            if(group>=0 && count[group])
                delta+=difference(score,weights.data()+size_t(group)*score.d,s,latest[target]);
            ksum[size_t(s)*bits+j]+=delta;
        }
        latest[x]=previous[s];
        const int group=id(x);
        if(group>=0 && count[group]) {
            std::fill_n(weights.begin()+size_t(group)*score.d,score.d,0.);
            count[group]=0;
        }
    }
    stats.suffix_queries+=visits;
    stats.peak_workspace_bytes=std::max<uint64_t>(stats.peak_workspace_bytes,directory.bytes()
        +(latest.capacity()+previous.capacity()+end_at_length.capacity()+compact.capacity()+count.capacity())*sizeof(int)
        +(qbest.capacity()+kbest.capacity()+survives.capacity())*sizeof(Route)
        +touched.capacity()*sizeof(size_t)+(qsum.capacity()+ksum.capacity()+deleted.capacity()+weights.capacity())*sizeof(double));
    cq.swap(qsum);ck.swap(ksum);
    return true;
}
// Build a work certificate before materializing any row weights or credits.
// The fixed visit cap preserves near-linear work on all admitted streams.
template<class Evaluate>
inline bool with_bigram_plan(const uint64_t* q,const uint64_t* k,const std::vector<Route>& base,
        int bits,Evaluate evaluate) {
    const int n=int(base.size());
    if(!bigram_eligible(n,bits,base))return false;
    bigram_detail::PairDirectory directory(n,bits);
    if(!directory.prepare(q,k,bits,uint64_t(n)*128))return false;
    return evaluate(directory);
}

inline bool bigram_credit(const uint64_t* q,const uint64_t* k,const std::vector<Route>& base,
        int bits,const Scores& score,std::vector<double>& cq,std::vector<double>& ck,Stats& stats) {
    if(score.n!=int(base.size()))return false;
    return with_bigram_plan(q,k,base,bits,[&](auto& directory) {
        return bigram_credit(q,k,base,bits,score,cq,ck,stats,directory);
    });
}
} // namespace rosa_subq
