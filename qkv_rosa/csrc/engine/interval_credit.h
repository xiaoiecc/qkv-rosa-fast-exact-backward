#pragma once
#include "delete.h"
#include "edit_alphabet.h"
#include <memory>

namespace rosa_subq {
namespace interval_credit {

// Causal postings for one suffix length. The sparse directory compares full
// keys. Both representations return the very same list of exact endpoints.
class Level {
    uint64_t mask_;
    std::vector<int> heads_,first_,previous_;
    std::unique_ptr<rosa_exact::FullWordMap> ids_;
public:
    Level(int n,int width):mask_(width==64?UINT64_MAX:((UINT64_C(1)<<width)-1)),previous_(n,-1) {
        if(width<31 && (UINT64_C(1)<<width)<=uint64_t(n)*8)
            heads_.assign(size_t(1)<<width,-1);
        else ids_=std::make_unique<rosa_exact::FullWordMap>(n,59);
        first_.resize(heads_.size(),n);
    }
    uint64_t mask()const{return mask_;}
    int last(uint64_t code)const {
        code&=mask_;
        if(!ids_)return heads_[code];
        int i=ids_->find(code);return i<0?-1:heads_[i];
    }
    bool disjoint(uint64_t code,int length)const {
        code&=mask_;int id=ids_?ids_->find(code):int(code);
        return id>=0 && heads_[id]-first_[id]>=length;
    }
    int previous(int end)const{return previous_[end];}
    void append(uint64_t code,int end) {
        code&=mask_;int id;
        if(ids_) {
            auto slot=ids_->insert(code,int(heads_.size()));
            if(slot.second){heads_.push_back(-1);first_.push_back(end); }
            id=slot.first;
        } else id=int(code);
        if(heads_[id]<0)first_[id]=end;
        previous_[end]=heads_[id];heads_[id]=end;
    }
    size_t bytes()const{return (heads_.capacity()+first_.capacity()+previous_.capacity())*sizeof(int)
        +(ids_?ids_->workspace_bytes():0);}
};

inline Route singleton(int end){return end<0?Route{}:Route{1,end};}
inline double change(const Scores& score,const double* w,int to,int from) {
    if(to==from)return 0.;
    const double* a=score.value(to),*b=score.value(from);
    double value=0.;
    for(int c=0;c<score.d;++c)value+=w[c]*(a[c]-b[c]);
    return value;
}

// A row's cost telescopes through its surviving exact suffix, its one-word
// repairs, then its longer repairs. All three terms have disjoint ownership.
template<int Width>
inline void sweep(const EditAlphabet& plan,int rank_width,int horizon,const std::vector<Route>& base,
        const Scores& score,std::vector<double>& cq,std::vector<double>& ck,Stats& stats) {
    const int n=plan.n,bits=Width?Width:plan.d,digit=Width?Width:rank_width;
    const auto* q=plan.q;const auto* k=plan.k;
    const int digits=64/digit;
    const uint64_t digit_mask=(UINT64_C(1)<<digit)-1;
    double collision=0.;
    for(int id=0;id<plan.nq;++id)if(plan.qsame[id]>=0) {
        const int key=plan.qsame[id];
        collision+=double(plan.qcount[id])*plan.kcount[key];
    }
    collision/=double(n)*n;
    const int uniform_depth=int(std::ceil(std::log2(double(horizon))/bits));
    const int collision_depth=int(std::ceil(std::log(double(n))/
                                    -std::log(std::clamp(collision,1./(double(n)*n),.75))));
    const int depth=std::min(digits,std::max({2,uniform_depth,collision_depth}));
    std::vector<uint64_t> qw(n),kw(n);
    std::vector<Level> levels;
    if(plan.births)for(int j=2;j<=depth && j<n;++j)levels.emplace_back(n,j*digit);
    for(int i=0;i<n;++i) {
        int id=plan.qsame[plan.qclass[i]];
        qw[i]=((i?qw[i-1]:0)<<digit)|uint64_t(id<0?plan.nk:id);
        kw[i]=((i?kw[i-1]:0)<<digit)|uint64_t(plan.kclass[i]);
    }

    std::unique_ptr<Index> suffix;
    auto index=[&]() -> Index& {
        if(!suffix)suffix=std::make_unique<Index>(q,k,n,false,0,&base);
        return *suffix;
    };
    auto packed_arm=[&](uint64_t word,int t,int r) {
        if(r<0)return 0;
        uint64_t delta=word^kw[r];
        if(digits*digit<64)delta&=(UINT64_C(1)<<(digits*digit))-1;
        if(delta)return std::min(rosa_random::row_bit(delta)/digit,r+1);
        if(r<digits)return r+1;
        if(q[t-digits]!=k[r-digits])return digits;
        return digits+index().lcs(t-digits,r-digits);
    };
    auto lcs=[&](int t,int r){return r<0?0:packed_arm(qw[t],t,r);};
    std::vector<int> latest(plan.nk,-1),previous(n,-1),count(plan.nk);
    const int local_limit=std::max({depth,48,int(std::ceil(3*std::log2(double(horizon))))});
    std::vector<int> key_run(n),long_rows,ends(local_limit+1),seen(n,-1);
    int longest=0;for(Route route:base)longest=std::max(longest,route.length);
    const bool has_pairs=plan.births && longest;
    std::vector<Route> survivors(local_limit),qbest(has_pairs?size_t(longest+1)*bits:0),
                       kbest(has_pairs?size_t(n)*bits:0);
    std::vector<size_t> qtouched,ktouched;
    std::vector<double> qdel(n),kdel(n),affine,offset;
    int qrun=1;
    for(int t=1;t<n;++t) {
        const int r=t-1,kid=plan.kclass[r],qid=plan.qclass[t],same=plan.qsame[qid];
        previous[r]=latest[kid];latest[kid]=r;++count[kid];
        key_run[r]=(r && k[r]==k[r-1])?key_run[r-1]+1:1;
        qrun=q[t]==q[t-1]?qrun+1:1;
        for(int h=2;h<=int(levels.size())+1 && h<=t;++h)levels[h-2].append(kw[r],r);
        const Route old=base[t];const int L=old.length,first=old.end-L+1;
        const double* w=score.row(t);
        // For a single K run the right piece retains the old endpoint. On
        // the left-winning interval the returned payload is precisely V[s].
        const bool one_run=L>1 && qrun>=L && old.end==latest[same]
            && key_run[old.end]==count[same];
        const int start=one_run?old.end-key_run[old.end]+1:0;
        const bool indexed=L>1 && !one_run && L>local_limit;
        if(L) {
            if(one_run) {
                qdel[t]+=change(score,w,-1,old.end);
                const int lo=std::max(first,start+(old.end-start)/2+1),hi=old.end+1;
                if(affine.empty()){affine.resize(size_t(n+1)*score.d);offset.resize(n+1);}
                for(int c=0;c<score.d;++c) {
                    affine[size_t(lo)*score.d+c]+=w[c];affine[size_t(hi)*score.d+c]-=w[c];
                }
                const double val=score.score(t,old.end);
                offset[lo]-=val;offset[hi]+=val;
            } else if(indexed)long_rows.push_back(t);
            else if(L>1 && levels.empty()) {
                // A nonempty baseline can survive even with no possible
                // repairs. Query bounded barriers directly for these rows.
                for(int a=0;a<L;++a)
                    qdel[t-a]+=change(score,w,a?index().R(t,a):-1,old.end);
                for(int s=first;s<=old.end;++s)
                    kdel[s]+=change(score,w,index().deleted(t,s).end,old.end);
            }
            else {
                ends[1]=latest[same];
                const int local_depth=std::min(L,depth);
                for(int h=2;h<=local_depth;++h)ends[h]=levels[h-2].last(qw[t]);
                // Longer witnesses share one packed suffix. Clip each exact
                // occurrence once over the small damaged interval.
                if(L>depth) {
                    std::fill(ends.begin()+depth+1,ends.begin()+L+1,-1);
                    std::fill_n(survivors.begin(),L,Route{});
                    auto& top=levels[depth-2];
                    for(int end=ends[depth];end>=0;end=top.previous(end)) {
                        const int len=lcs(t,end);
                        if(len>L)throw std::invalid_argument("inconsistent saved match length");
                        ends[len]=std::max(ends[len],end);
                        for(int s=first;s<=old.end;++s) {
                            int left=s<=end?std::min(len,end-s):len;
                            if(left)survivors[s-first]=std::max(survivors[s-first],Route{left,end});
                        }
                    }
                    for(int h=L-1;h>=depth;--h)ends[h]=std::max(ends[h],ends[h+1]);
                }
                for(int a=0;a<L;++a)qdel[t-a]+=change(score,w,a?ends[a]:-1,old.end);
                for(int s=first;s<=old.end;++s) {
                    Route winner=L>depth?survivors[s-first]:Route{};
                    for(int h=local_depth;h>winner.length;--h) {
                        int end=ends[h];
                        if(end>=s && end<s+h) {
                            if(h==1)end=previous[end];
                            else do {end=levels[h-2].previous(end);}while(end>=s);
                        }
                        if(end>=0){winner={h,end};break;}
                    }
                    // At equal length a later short witness wins.
                    if(L>depth && winner.length && winner.length<=local_depth) {
                        int h=winner.length,end=ends[h];
                        if(end>=s && end<s+h) {
                            if(h==1)end=previous[end];
                            else do {end=levels[h-2].previous(end);}while(end>=s);
                        }
                        winner.end=std::max(winner.end,end);
                    }
                    survivors[s-first]=winner;
                    kdel[s]+=change(score,w,winner.end,old.end);
                }
            }
        }
        if(plan.birth_prefix[t+1]==plan.birth_prefix[t-L])continue;
        // The Q singleton seed exists at every row, including long baselines.
        for(const auto edge:plan.qneighbors[qid])if(latest[edge.other]>=0)
            cq[size_t(t)*bits+edge.bit]+=change(score,w,latest[edge.other],-1);
        if(!has_pairs || t<2)continue;
        const int p=std::min(t,std::max(2,std::min(depth,L+1)));
        const auto note=[&](std::vector<Route>& best,std::vector<size_t>& touched,size_t i,Route route) {
            if(!best[i].length)touched.push_back(i);
            best[i]=std::max(best[i],route);
        };
        auto report=[&](int a,int bit,int len,int end,bool qside,bool kside) {
            if(a>L)throw std::invalid_argument("inconsistent saved match length");
            const Route route{len,end};const int s=end-a;
            if(qside && (a<L || route>old))note(qbest,qtouched,size_t(a)*bits+bit,route);
            if(kside && ((s>=first && s<=old.end)||route>old))note(kbest,ktouched,size_t(s)*bits+bit,route);
        };
        auto inspect=[&](int end,bool local) {
            if(seen[end]==t)return;
            seen[end]=t;++stats.suffix_queries;
            const int a=lcs(t,end);
            if(a>end)return;
            const uint64_t diff=q[t-a]^k[end-a];
            if(!diff || (diff&(diff-1)))return;
            const int bit=rosa_random::row_bit(diff);
            if(bit>=bits)return;
            const int len=a+1+lcs(t-a-1,end-a-1);
            if(len>=2)report(a,bit,len,end,!local,true);
        };
        auto& pivot=levels[p-2];
        if(L>=p && plan.birth_prefix[t-p+1]!=plan.birth_prefix[t-L])
            for(int end=pivot.last(qw[t]);end>=0;end=pivot.previous(end))inspect(end,false);
        // For one edited digit, suffix occurrence is monotone in its depth.
        // A single predecessor search replaces visiting that edit at every
        // shorter level. Only K ties later than the old winner need posting.
        for(int a=0;a<p;++a) {
            const int shift=a*digit;
            const uint64_t cleared=qw[t]&~(digit_mask<<shift);
            for(auto edge:plan.qneighbors[plan.qclass[t-a]]) {
                const uint64_t word=cleared|(uint64_t(edge.other)<<shift);
                const int last=pivot.last(word);
                if(last>=0)for(int end=last;end>=0;end=pivot.previous(end)) {
                    // The posting key already proves this exact edit. Only
                    // the part older than the pivot can extend its length.
                    seen[end]=t;++stats.suffix_queries;
                    report(a,edge.bit,packed_arm(word,t,end),end,true,true);
                }
                else if(p>2 && a+1<p) {
                    const int minimum=std::max(2,a+1);
                    int lo=minimum,hi=p-1,best=-1,len=0,stride=1;
                    while(hi>=minimum) {
                        int probe=std::max(minimum,p-stride);
                        int end=levels[probe-2].last(word);
                        if(end>=0){best=end;len=probe;lo=probe+1;break;}
                        hi=probe-1;stride*=2;
                    }
                    while(lo<=hi) {
                        const int mid=(lo+hi)/2,end=levels[mid-2].last(word);
                        if(end<0)hi=mid-1;
                        else {best=end;len=mid;lo=mid+1;}
                    }
                    if(best>=0)report(a,edge.bit,len,best,true,false);
                }
                if(L>=2 && L<p && a<L) {
                    auto& tie=levels[L-2];
                    for(int end=tie.last(word);end>old.end;end=tie.previous(end))
                        report(a,edge.bit,L,end,false,true);
                }
            }
        }
        if(p>2 && L && !pivot.disjoint(qw[t],p))for(int end=first;end<std::min(t,old.end+p-1);++end)inspect(end,true);
        for(size_t i:qtouched) {
            int a=int(i/bits),bit=int(i%bits);
            Route before=old;
            if(a==0) {
                int id=plan.kmap.find(q[t]^(UINT64_C(1)<<bit));
                before=singleton(id<0?-1:latest[id]);
            } else if(a<L)before={a,one_run?old.end:(indexed?index().R(t,a):ends[a])};
            if(qbest[i]>before)cq[size_t(t-a)*bits+bit]+=change(score,w,qbest[i].end,before.end);
            qbest[i]={};
        }
        qtouched.clear();
        for(size_t i:ktouched) {
            int s=int(i/bits),bit=int(i%bits);Route before=old;
            if(s>=first && s<=old.end) {
                if(one_run) {
                    before=Route{old.end-s,old.end};
                    if(!before.length)before={};
                    if(s>start)before=std::max(before,Route{std::min(L,s-start),s-1});
                } else before=indexed?index().deleted(t,s):survivors[s-first];
            } else if(L<=1 && s>old.end && (q[t]^k[s])==(UINT64_C(1)<<bit))before={1,s};
            if(kbest[i]>before)ck[i]+=change(score,w,kbest[i].end,before.end);
            kbest[i]={};
        }
        ktouched.clear();
    }
    if(!long_rows.empty()) {
        int width=std::max(1,int(std::sqrt(double(n))));
        auto qd=q_deletion(index(),score,width,stats,&long_rows);
        auto kd=k_deletion(index(),score,width,stats,&long_rows);
        for(int s=0;s<n;++s){qdel[s]+=qd[s];kdel[s]+=kd[s];}
    }
    if(!affine.empty()) {
        double subtract=0.;std::vector<double> moving(score.d);
        for(int s=0;s<n;++s) {
            subtract+=offset[s];double val=subtract;
            for(int c=0;c<score.d;++c) {
                moving[c]+=affine[size_t(s)*score.d+c];
                val+=moving[c]*score.payload[size_t(s)*score.d+c];
            }
            kdel[s]+=val;
        }
    }
    // For K[s] -> x the one-word repair beats the baseline iff L<=1
    // and s is later than the preceding x. Reverse class sums cover exactly
    // these rows until the next x. Their original endpoint is constant.
    std::vector<double> weights(plan.births?size_t(plan.nq)*score.d:0);
    std::vector<int> pending(plan.nq);
    for(int s=n-2;s>=0;--s) {
        int t=s+1,id=plan.qclass[t],kid=plan.kclass[s];
        if(plan.births && base[t].length<=1 && plan.first_neighbor[id]<t) {
            double* dst=weights.data()+size_t(id)*score.d;
            for(int c=0;c<score.d;++c)dst[c]+=score.row(t)[c];
            ++pending[id];
        }
        for(auto edge:plan.kneighbors[kid])if(pending[edge.other]) {
            int ktarget=plan.qsame[edge.other];
            ck[size_t(s)*bits+edge.bit]+=change(score,weights.data()+size_t(edge.other)*score.d,
                                                s,ktarget<0?-1:latest[ktarget]);
        }
        latest[kid]=previous[s];id=plan.ksame[kid];
        if(id>=0 && pending[id]) {
            std::fill_n(weights.data()+size_t(id)*score.d,score.d,0.);pending[id]=0;
        }
        for(int j=0;j<bits;++j){ck[size_t(s)*bits+j]+=kdel[s];cq[size_t(t)*bits+j]+=qdel[t];}
    }
    size_t bytes=(qw.capacity()+kw.capacity())*sizeof(uint64_t)
        +(latest.capacity()+previous.capacity()+count.capacity()+key_run.capacity()+long_rows.capacity()
        +ends.capacity()+seen.capacity()+pending.capacity())*sizeof(int)
        +(qbest.capacity()+kbest.capacity()+survivors.capacity())*sizeof(Route)
        +(qtouched.capacity()+ktouched.capacity())*sizeof(size_t)
        +(qdel.capacity()+kdel.capacity()+affine.capacity()+offset.capacity()+weights.capacity())*sizeof(double);
    for(const auto& level:levels)bytes+=level.bytes();
    stats.peak_workspace_bytes+=bytes+(suffix?suffix->workspace_bytes():0);
}
inline void evaluate(const EditAlphabet& plan,const std::vector<Route>& base,
        const Scores& score,std::vector<double>& cq,std::vector<double>& ck,Stats& stats,int horizon=0) {
    if(plan.n<2)return;
    if(!horizon)horizon=plan.n;
    const bool missing=std::find(plan.qsame.begin(),plan.qsame.end(),-1)!=plan.qsame.end();
    int digit=1;while((UINT64_C(1)<<digit)<uint64_t(plan.nk+int(missing)))++digit;
    // All instantiations use the same interval proof and posting order.
    // Constant lane sizes let the compiler fold division in the inner loop.
    if(digit==plan.d && digit==1)sweep<1>(plan,digit,horizon,base,score,cq,ck,stats);
    else if(digit==plan.d && digit==2)sweep<2>(plan,digit,horizon,base,score,cq,ck,stats);
    else if(digit==plan.d && digit==4)sweep<4>(plan,digit,horizon,base,score,cq,ck,stats);
    else sweep<0>(plan,digit,horizon,base,score,cq,ck,stats);
}
} // namespace interval_credit
} // namespace rosa_subq
