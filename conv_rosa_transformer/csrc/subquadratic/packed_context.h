#pragma once
#include "near_k.h"
#include "types.h"
#include <array>

namespace rosa_subq {

// Index short reversed contexts at all radix depths in one append-only pass.
// A prefix class contains two disjoint matches of length h exactly when its
// oldest and latest occurrences end at least h apart.
class PackedContext {
    int depth_,bits_;
    std::vector<int> latest_,first_,previous_;
public:
    static bool fits(int bits,int longest) {
        return bits>=1 && bits<=8 && longest>=0 && longest<64/bits;
    }
    PackedContext(int n,int bits,int longest):depth_(1),bits_(bits) {
        while(depth_<std::max(1,longest) && (depth_+1)*bits<29
              && (UINT64_C(1)<<((depth_+1)*bits))<=uint64_t(n)*4)++depth_;
        latest_.assign(size_t(2)<<(depth_*bits),-1);
        first_.assign(latest_.size(),n);
        previous_.resize(size_t(n)*depth_);
    }
    int slot(uint64_t word,int h) const {
        const int size=1<<(h*bits_);
        return size+(int(word)&(size-1));
    }
    void insert(uint64_t word,int end) {
        int* chain=previous_.data()+size_t(end)*depth_;
        const int levels=std::min(depth_,end+1);
        for(int h=1;h<=levels;++h) {
            const int c=slot(word,h);
            chain[h-1]=latest_[c];
            if(latest_[c]<0)first_[c]=end;
            latest_[c]=end;
        }
    }
    int disjoint(uint64_t word,int longest) const {
        for(int h=std::min(longest,depth_);h>0;--h) {
            const int c=slot(word,h);
            if(latest_[c]-first_[c]>=h)return h;
        }
        return 0;
    }
    int depth() const {return depth_;}
    int head(int slot) const {return latest_[slot];}
    int previous(int end,int h) const {return previous_[size_t(end)*depth_+h-1];}
    int best_short(uint64_t word,int maximum) const {
        for(int h=maximum;h>0;--h) {
            const int r=latest_[slot(word,h)];
            if(r>=0)return r;
        }
        return -1;
    }
    size_t bytes() const {
        return (latest_.capacity()+first_.capacity()+previous_.capacity())*sizeof(int);
    }
};

namespace packed_detail {
inline int low_bit(uint64_t mask) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(mask);
#else
    return near_detail::one_bit_index(mask&(-mask));
#endif
}

template<int Bits>
inline bool sweep(const uint64_t* q,const uint64_t* k,const std::vector<Route>& base,
        int longest,const Scores& score,std::vector<double>& cq,std::vector<double>& ck,
        Stats& stats,uint64_t budget) {
    const int n=score.n;
    PackedContext table(n,Bits,longest);
    constexpr int digits=64/Bits;
    // Packed words contain an integral number of symbols. The high spare
    // bits in a 3/5/6/7-bit context belong to the previous word.
    constexpr uint64_t word_mask=UINT64_MAX>>(64-digits*Bits);
    // Only a completely matching repaired first word can require the second.
    // Longest < digits bounds each separate arm, hence also bounds this tail.
    std::vector<uint64_t> key(n),older_query(2*longest+1>digits?n:0);
    std::vector<double> qsum(cq.size()),ksum(ck.size()),deleted(n);
    std::vector<Route> krepaired(ck.size()),overlap;
    std::vector<size_t> touched;
    std::array<Route,65> qrepaired,removed;
    std::array<int,65> latest;
    uint64_t word=q[0],kw=0,work=0;
    if(!older_query.empty())older_query[0]=word;
    for(int t=1;t<n;++t) {
        word=((word<<Bits)|q[t])&word_mask;
        if(!older_query.empty())older_query[t]=word;
        kw=((kw<<Bits)|k[t-1])&word_mask;key[t-1]=kw;table.insert(kw,t-1);
        const double* w=score.row(t);
        if(std::none_of(w,w+score.d,[](double x){return x!=0.;}))continue;
        const Route original=base[t];
        const int L=original.length,first=original.end-L+1;
        const int depth=table.disjoint(word,L);
        const int nd=std::min(t,L+1)*Bits;
        std::fill_n(qrepaired.begin(),nd,Route{});
        touched.clear();overlap.clear();
        Route intact;
        int high=0;
        auto exact=[&](Route target) {
            // Newest-first reports imply that the first endpoint covering a
            // given age is the latest one; each age is written once per row.
            for(;high<target.length;++high)latest[high+1]=target.end;
            if(target.end<first || target.end-target.length>=original.end)
                intact=std::max(intact,target);
            else if(target.end!=original.end)overlap.push_back(target);
        };
        auto repaired_length=[&](uint64_t edited,int end) {
            if(edited)return std::min(low_bit(edited)/Bits,end+1);
            if(end<digits || older_query.empty())return std::min(digits,end+1);
            const uint64_t tail=older_query[t-digits]^key[end-digits];
            return digits+std::min(tail?low_bit(tail)/Bits:digits,end+1-digits);
        };
        auto repair=[&](int r,int age,int bit,int length) {
            const Route target{length,r};
            const int s=r-age;
            qrepaired[age*Bits+bit]=std::max(qrepaired[age*Bits+bit],target);
            if((s<first || s>original.end) && !(target>original))return;
            const size_t at=size_t(s)*Bits+bit;
            if(!krepaired[at].length)touched.push_back(at);
            krepaired[at]=std::max(krepaired[at],target);
        };
        const int pivot=std::min(table.depth(),L+1);
        const int group=depth?table.slot(word,depth):0;
        const int near_group=table.slot(word,pivot);
        // The exact disjoint-depth bucket supplies all survivors and every
        // Q repair whose mismatch is older than that prefix.
        for(int r=depth?table.head(group):t-1;r>=0;r=depth?table.previous(r,depth):r-1) {
            ++work;
            const uint64_t diff=word^key[r];
            const int pos=diff?low_bit(diff):64;
            const int a=std::min(pos/Bits,r+1);
            if(a>L)return false;
            if(a)exact({a,r});
            if(a>r)continue;
            const uint64_t mask=(diff>>(a*Bits))&((1U<<Bits)-1);
            if(mask&(mask-1))continue;
            const uint64_t edited=diff&(diff-1);
            repair(r,a,pos%Bits,repaired_length(edited,r));
        }
        // Younger mismatches can be numerous at disjoint depth. Ask for the
        // longer pivot context instead; each nonempty bucket has a known
        // edited age and bit and one visit supplies both Q and K.
        for(int j=0;j<depth*Bits;++j) {
            for(int r=table.head(near_group^(1<<j));r>=0;r=table.previous(r,pivot)) {
                ++work;
                const uint64_t edited=(word^key[r])^(UINT64_C(1)<<j);
                repair(r,j/Bits,j%Bits,repaired_length(edited,r));
            }
        }
        // A length-L repair outside the damaged window can only win at
        // a later endpoint. Q ties for absent long repairs are recovered by
        // the longest-short lookup below.
        if(pivot>L && L>0) {
            const int tie_group=table.slot(word,L);
            for(int j=0;j<depth*Bits;++j) {
                for(int r=table.head(tie_group^(1<<j));r>original.end;r=table.previous(r,L)) {
                    ++work;
                    const uint64_t diff=(word^key[r])^(UINT64_C(1)<<j);
                    if(!((diff>>(L*Bits))&((1U<<Bits)-1)) && r>=L)continue;
                    repair(r,j/Bits,j%Bits,L);
                }
            }
        }
        if(high!=L)return false;
        if((work+=nd+1)>budget)return false;
        auto delta=[&](int end,int before) {
            const double* a=score.value(end);const double* b=score.value(before);
            if(score.d==Bits+1) {
                double fixed=0;
                for(int c=0;c<=Bits;++c)fixed+=w[c]*(a[c]-b[c]);
                return fixed;
            }
            double sum=0;for(int c=0;c<score.d;++c)sum+=w[c]*(a[c]-b[c]);
            return sum;
        };
        for(int age=0;age<nd/Bits;++age)for(int j=0;j<Bits;++j) {
            const int keybit=age*Bits+j;
            int end=qrepaired[keybit].end;
            if(end<0) {
                if(age<depth)end=table.best_short(word^(UINT64_C(1)<<keybit),pivot-1);
                else end=age>=L?original.end:(age?latest[age]:-1);
            }
            if(end!=original.end)qsum[size_t(t-age)*Bits+j]+=delta(end,original.end);
        }
        // At least one certified original occurrence survives every K edit.
        // Disjoint candidates win throughout the window. Intersecting ones
        // are clipped to the suffix strictly to the right of the edited key.
        for(int s=first;s<=original.end;++s) {
            const int h=original.end-s;
            removed[s-first]=h?std::max(intact,Route{h,original.end}):intact;
        }
        for(Route target:overlap)for(int s=first;s<=original.end;++s) {
            int length=target.length;
            if(s<=target.end)length=std::min(length,target.end-s);
            if(length)removed[s-first]=std::max(removed[s-first],Route{length,target.end});
        }
        for(int s=first;s<=original.end;++s)if(removed[s-first].end!=original.end)
            deleted[s]+=delta(removed[s-first].end,original.end);
        // A shorter unreported repair can win only inside the damaged
        // winning interval and for a mismatch age < depth. This bounds its
        // endpoint inside one contiguous window, independent of n.
        const uint64_t low_mask=(UINT64_C(1)<<(depth*Bits))-1;
        if(pivot>depth && depth)for(int r=first;r<std::min(t,original.end+depth);++r) {
            const uint64_t diff=word^key[r],low=diff&low_mask;
            // The disjoint witness survives. A new, younger repair must
            // therefore have exactly one differing bit in these low bits.
            if(!low || (low&(low-1)))continue;
            const int pos=low_bit(low),age=pos/Bits;
            const int owner=r-age;
            if(owner<first || owner>original.end)continue;
            const uint64_t edited=diff&(diff-1);
            const int length=repaired_length(edited,r);
            if(!(Route{length,r}>removed[owner-first]))continue;
            const size_t at=size_t(owner)*Bits+pos%Bits;
            if(!krepaired[at].length)touched.push_back(at);
            krepaired[at]=std::max(krepaired[at],Route{length,r});
        }
        for(size_t at:touched) {
            const int s=int(at/Bits);
            const Route other=s>=first && s<=original.end?removed[s-first]:original;
            if(krepaired[at]>other && krepaired[at].end!=other.end)
                ksum[at]+=delta(krepaired[at].end,other.end);
            krepaired[at]={};
        }
    }
    for(int s=0;s<n;++s)for(int j=0;j<Bits;++j)ksum[size_t(s)*Bits+j]+=deleted[s];
    cq.swap(qsum);ck.swap(ksum);
    stats.suffix_queries+=work;
    stats.peak_workspace_bytes=std::max<uint64_t>(stats.peak_workspace_bytes,table.bytes()
        +(key.capacity()+older_query.capacity())*sizeof(uint64_t)+(overlap.capacity()+krepaired.capacity())*sizeof(Route)
        +touched.capacity()*sizeof(size_t)+(qsum.capacity()+ksum.capacity()+deleted.capacity())*sizeof(double));
    return true;
}
} // namespace packed_detail

inline bool packed_credit(const uint64_t* q,const uint64_t* k,const std::vector<Route>& base,
        int bits,int longest,const Scores& score,std::vector<double>& cq,std::vector<double>& ck,
        Stats& stats,uint64_t budget) {
    if(!score.n || !PackedContext::fits(bits,longest) || !budget)return false;
    for(int i=0;i<score.n;++i)if((q[i]|k[i])>>bits)return false;
    using Kernel=bool(*)(const uint64_t*,const uint64_t*,const std::vector<Route>&,int,
                         const Scores&,std::vector<double>&,std::vector<double>&,Stats&,uint64_t);
    static constexpr Kernel kernels[]={
        packed_detail::sweep<1>,packed_detail::sweep<2>,packed_detail::sweep<3>,packed_detail::sweep<4>,
        packed_detail::sweep<5>,packed_detail::sweep<6>,packed_detail::sweep<7>,packed_detail::sweep<8>
    };
    return kernels[bits-1](q,k,base,longest,score,cq,ck,stats,budget);
}
} // namespace rosa_subq
