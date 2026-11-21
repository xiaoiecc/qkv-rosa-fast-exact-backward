#pragma once
#include "types.h"
#include "correlation.h"
#include <cmath>

namespace rosa_subq {
namespace run_credit {

// -2 means use this row's baseline. The empty route retains its usual -1.
constexpr int baseline=-2;
struct Change { int age,before,after; };
struct Epoch { int owner,first; };

// Prefixes of [row weights, baseline score, score at causal endpoint t-1].
// The first d+1 channels are also the signal for the moving Q age kernel.
class Signals {
public:
    const Scores& scores;
    int n,d;
    std::vector<double> signal,prefix,aligned;
    Signals(const Scores& s,const std::vector<Route>& base):scores(s),n(s.n),d(s.d+1),
        signal(size_t(n)*d),prefix(size_t(n+1)*d),aligned(n+1) {
        for(int t=0;t<n;++t) {
            double* row=signal.data()+size_t(t)*d;
            std::copy_n(s.row(t),s.d,row);
            row[d-1]=s.score(t,base[t].end);
            for(int c=0;c<d;++c)prefix[size_t(t+1)*d+c]=prefix[size_t(t)*d+c]+row[c];
            aligned[t+1]=aligned[t]+s.score(t,t-1);
        }
    }
    double fixed(int lo,int hi,int end) const {
        const double* a=prefix.data()+size_t(lo)*d;
        const double* b=prefix.data()+size_t(hi)*d;
        if(end==baseline)return b[d-1]-a[d-1];
        const double* payload=scores.value(end);
        double sum=0;
        for(int c=0;c<d-1;++c)sum+=(b[c]-a[c])*payload[c];
        return sum;
    }
    size_t bytes() const {return (signal.capacity()+prefix.capacity()+aligned.capacity())*sizeof(double);}
};

// Each numerical change takes effect for every later owner. A binary split
// applies left-side changes to all right-side outputs in one middle product.
// The terminal epoch is a sentinel. No owner-by-age table is retained.
class AgeContraction {
    const Signals& data;
    const std::vector<Change>& changes;
    const std::vector<Epoch>& epochs;
    std::vector<double>& out;
    Stats& stats;
    std::vector<double> kernel;
    std::vector<int> plateaus;
    void apply(int first,int stop,int lo,int hi) {
        const int d=data.d;
        int length=0;
        for(int i=first;i<stop;++i)if(changes[i].age<data.n-lo)
            length=std::max(length,changes[i].age+1);
        if(!length)return;
        kernel.assign(size_t(length)*d,0.);
        for(int i=first;i<stop;++i) {
            const auto& change=changes[i];
            if(change.age>=length)continue;
            double* dst=kernel.data()+size_t(change.age)*d;
            for(int side=0;side<2;++side) {
                const int end=side?change.after:change.before,sign=side?1:-1;
                if(end==baseline)dst[d-1]+=sign;
                else {
                    const double* val=data.scores.value(end);
                    for(int c=0;c<d-1;++c)dst[c]+=sign*val[c];
                }
            }
        }
        plateaus.clear();plateaus.push_back(0);
        for(int h=1;h<length;++h) {
            const double* at=kernel.data()+size_t(h)*d;
            if(!std::equal(at,at+d,at-d))plateaus.push_back(h);
        }
        const int owners=hi-lo;
        if(double(plateaus.size())*owners<=8.*(length+owners)*std::log2(2.+length+owners)) {
            plateaus.push_back(length);
            for(size_t i=1;i<plateaus.size();++i) {
                const int a=plateaus[i-1],b=plateaus[i];
                const double* coeff=kernel.data()+size_t(a)*d;
                if(std::none_of(coeff,coeff+d,[](double x){return x!=0.;}))continue;
                for(int u=lo;u<hi && u+a<data.n;++u) {
                    const double* left=data.prefix.data()+size_t(u+a)*d;
                    const double* right=data.prefix.data()+size_t(std::min(data.n,u+b))*d;
                    double sum=0;
                    if(b==a+1) {
                        const double* row=data.signal.data()+size_t(u+a)*d;
                        for(int c=0;c<d;++c)sum+=row[c]*coeff[c];
                    } else for(int c=0;c<d;++c)sum+=(right[c]-left[c])*coeff[c];
                    out[u]+=sum;
                }
            }
        } else {
            CorrelationStats cs;
            BlockCorrelations::middle_product(data.signal.data()+size_t(lo)*d,data.n-lo,d,
                kernel.data(),length,owners,out.data()+lo,cs);
            stats.fft_transforms+=cs.transforms;
            scratch=std::max<size_t>(scratch,cs.workspace_bytes+kernel.capacity()*sizeof(double)
                +plateaus.capacity()*sizeof(int));
        }
    }
    void split(int a,int b) {
        if(b-a==1) {
            apply(epochs[a].first,epochs[b].first,epochs[a].owner,epochs[b].owner);
            return;
        }
        const int mid=a+(b-a)/2;
        apply(epochs[a].first,epochs[mid].first,epochs[mid].owner,epochs[b].owner);
        split(a,mid);split(mid,b);
    }
public:
    size_t scratch=0;
    AgeContraction(const Signals& s,const std::vector<Change>& c,const std::vector<Epoch>& e,
        std::vector<double>& output,Stats& work):data(s),changes(c),epochs(e),out(output),stats(work) {
        if(e.size()>1)split(0,int(e.size()-1));
        scratch=std::max(scratch,kernel.capacity()*sizeof(double)+plateaus.capacity()*sizeof(int));
    }
};

class RunGeometry {
    const uint64_t* key;
    uint64_t word;
    const std::vector<Route>& base;
    const Signals& data;
    int n;
    // lengths of equal-word runs starting at, and immediately before, a position
    std::vector<int> right,left;
public:
    RunGeometry(const uint64_t* k,uint64_t x,const std::vector<Route>& original,const Signals& s)
        :key(k),word(x),base(original),data(s),n(s.n),right(n),left(n) {
        for(int i=n-2;i>=0;--i)if(k[i]==x)right[i]=right[i+1]+1;
        for(int i=1;i<n;++i)if(k[i-1]==x)left[i]=left[i-1]+1;
    }
    size_t bytes() const {return (right.capacity()+left.capacity())*sizeof(int);}

    // Across a winning run [f,r], deletion picks the right fragment (same
    // payload), the best intact rival, or the prefix ending at owner-1.
    void keys(int bits,std::vector<double>& ck,Stats& stats) const {
        const int d=data.d;
        std::vector<double> diff(size_t(n+1)*d),active(d);
        Route longest,second;
        for(int t=1;t<n;++t) {
            const int r=t-1;
            if(!left[t] && left[r]) {
                const Route finished{left[r],r-1};
                if(finished>longest){second=longest;longest=finished;}
                else second=std::max(second,finished);
            }
            const Route current=left[t]?Route{left[t],r}:Route{};
            const Route rival=current>longest?longest:std::max(current,second);
            const Route win=base[t];
            if(!win.length)continue;
            const int f=win.end-win.length+1;
            const int from_left=std::max({f+1,(f+win.end)/2+1,
                f+rival.length+int(rival.end>=f)});
            const int right_need=std::max(1,rival.length+int(rival.end>win.end));
            const int from_rival=std::max(f,win.end-right_need+1);
            const int after_rival=std::min(from_left,win.end+1);
            const double old=data.signal[size_t(t)*d+d-1];
            if(from_rival<after_rival) {
                const double delta=data.scores.score(t,rival.end)-old;
                diff[size_t(from_rival)*d+d-1]+=delta;diff[size_t(after_rival)*d+d-1]-=delta;
            }
            if(from_left<=win.end) {
                double* start=diff.data()+size_t(from_left)*d;
                double* stop=diff.data()+size_t(win.end+1)*d;
                const double* weights=data.scores.row(t);
                for(int c=0;c<d-1;++c){start[c]+=weights[c];stop[c]-=weights[c];}
                start[d-1]-=old;stop[d-1]+=old;
            }
        }
        for(int s=0;s<n-1;++s) {
            for(int c=0;c<d;++c)active[c]+=diff[size_t(s)*d+c];
            double* credit=ck.data()+size_t(s)*bits;
            if(key[s]==word) {
                double sum=active[d-1];
                const double* payload=data.scores.value(s-1);
                for(int c=0;c<d-1;++c)sum+=active[c]*payload[c];
                std::fill_n(credit,bits,sum);
                continue;
            }
            const auto mask=key[s]^word;
            if(mask&(mask-1))continue;
            int bit=0;while((UINT64_C(1)<<bit)!=mask)++bit;
            if(bit>=bits)continue;
            const int end=s+right[s+1],start=s-left[s];
            // The merged run grows up to end, then stays fixed. Its winner
            // set is a suffix of the growing phase and a prefix of the rest.
            int lo=s+1,hi=end+2;
            while(lo<hi) {
                const int t=lo+(hi-lo)/2;
                if(Route{t-start,t-1}>base[t])hi=t;else lo=t+1;
            }
            double sum=data.aligned[end+2]-data.aligned[lo]-data.fixed(lo,end+2,baseline);
            const Route merged{end-start+1,end};
            const int stop=int(std::lower_bound(base.begin()+end+2,base.end(),merged)-base.begin());
            sum+=data.fixed(end+2,stop,end)-data.fixed(end+2,stop,baseline);
            credit[bit]=sum;
        }
        stats.peak_workspace_bytes=std::max<uint64_t>(stats.peak_workspace_bytes,
            bytes()+data.bytes()+(diff.capacity()+active.capacity())*sizeof(double));
    }

    // Events store changes of the winning endpoint at each age. The right
    // arms after nonmatching keys are disjoint runs, giving O(n) changes.
    void queries(uint64_t mask,std::vector<double>& out,Stats& stats) const {
        std::vector<int> deleted(n,baseline),source(n,-1);
        std::vector<Epoch> epochs;
        std::vector<Change> changes;
        int far=-1;
        auto value=[&](int h){return source[h]<0?deleted[h]:source[h]+h;};
        for(int u=0;u<n;++u) {
            const int first=int(changes.size());
            if(!u){changes.push_back({0,baseline,-1});deleted[0]=-1;}
            if(right[u] && !left[u]) {
                const int end=u+right[u]-1;
                for(int h=1;h<=right[u];++h) {
                    const int old=value(h);deleted[h]=end;
                    if(old!=value(h))changes.push_back({h,old,value(h)});
                }
            }
            const int s=u-1;
            if(s>=0 && mask && (key[s]^word)==mask) {
                far=std::max(far,right[u]);
                for(int h=0;h<=right[u];++h)if(source[h]<0 || left[s]>=left[source[h]]) {
                    const int old=value(h);source[h]=s;
                    if(old!=value(h))changes.push_back({h,old,value(h)});
                }
            }
            if(int(changes.size())>first)epochs.push_back({u,first});
            // The current run's end in the dictionary is its final end.
            // Before it finishes, causality clips it to t-1, unless a repair
            // already wins at that age. The rectangle cuts out a time prefix.
            const int lo=u+std::max(1,far+1),end=u+right[u]-1;
            if(lo<=end)out[u]+=data.aligned[end+1]-data.aligned[lo]-data.fixed(lo,end+1,end);
        }
        epochs.push_back({n,int(changes.size())});
        AgeContraction contraction(data,changes,epochs,out,stats);
        stats.peak_workspace_bytes=std::max<uint64_t>(stats.peak_workspace_bytes,
            data.bytes()+bytes()+contraction.scratch+(deleted.capacity()+source.capacity())*sizeof(int)
            +epochs.capacity()*sizeof(Epoch)+changes.capacity()*sizeof(Change)+out.capacity()*sizeof(double));
    }
};

// A row with a trailing run longer than every causal K substring within
// one bit of x^m cannot match across that run after any single edit.
inline bool residual_rows(const uint64_t* q,const uint64_t* k,int n,uint64_t word,
        const std::vector<int32_t>& active,std::vector<int32_t>& residual) {
    const size_t limit=1+size_t(std::log2(double(n)));
    int trailing=0,exact=0,near=0,maximum=0;
    size_t next=0;
    for(int t=0;t<n;++t) {
        if(t) {
            const auto diff=k[t-1]^word;
            if(!diff){++near;++exact;}
            else {near=(diff&(diff-1))?0:exact+1;exact=0;}
            maximum=std::max(maximum,near);
            trailing=q[t]==word?trailing+1:0;
        }
        if(next<active.size() && active[next]==t) {
            ++next;
            if(trailing<t && trailing<=maximum) {
                residual.push_back(t);
                if(residual.size()>limit)return false;
            }
        }
    }
    return residual.size()<active.size();
}

inline void evaluate(const uint64_t* key,uint64_t word,const std::vector<Route>& base,
        const Scores& scores,int bits,std::vector<double>& cq,std::vector<double>& ck,Stats& stats) {
    Signals data(scores,base);
    RunGeometry runs(key,word,base,data);
    runs.keys(bits,ck,stats);
    uint64_t supported=0;
    for(int s=0;s<scores.n-1;++s) {
        const uint64_t mask=key[s]^word;
        if(!(mask&(mask-1)))supported|=mask;
    }
    std::vector<double> shared(scores.n),out(scores.n);
    const uint64_t allowed=bits==64?UINT64_MAX:(UINT64_C(1)<<bits)-1;
    if((supported&allowed)!=allowed)runs.queries(0,shared,stats);
    for(int bit=0;bit<bits;++bit) {
        const uint64_t mask=UINT64_C(1)<<bit;
        const auto* values=&shared;
        if(supported&mask) {
            std::fill(out.begin(),out.end(),0.);
            runs.queries(mask,out,stats);values=&out;
        }
        for(int u=1;u<scores.n;++u)cq[size_t(u)*bits+bit]=(*values)[u];
    }
    stats.peak_workspace_bytes+=(shared.capacity()+out.capacity())*sizeof(double);
}

} // namespace run_credit
} // namespace rosa_subq
