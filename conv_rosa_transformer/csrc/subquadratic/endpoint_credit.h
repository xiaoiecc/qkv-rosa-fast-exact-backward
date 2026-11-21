#pragma once
#include "index.h"
#include "near_k.h"
#include "short_endpoint.h"
#include "../approx_onegram.h"
#include <array>

namespace rosa_subq {
namespace endpoint_credit {

// For heights 1..high the latest endpoint is a staircase. Its last feasible
// height after a K barrier s lies in the last step with end-low >= s.
struct Step { int low,high,end; };
struct Repair { int owner,bit; Route route; };

class EventSweep {
    const uint64_t *q,*k;
    int bits;
    Stats& stats;
    std::vector<Route> exact,prefix;
    std::vector<Step> steps;
    std::vector<Repair> qr,kr;
    void prepare() {
        std::sort(exact.begin(),exact.end(),[](Route a,Route b){return a.end<b.end;});
        prefix.clear();steps.clear();
        Route best;int high=0;
        for(auto r:exact){best=std::max(r,best);prefix.push_back(best);}
        for(auto it=exact.rbegin();it!=exact.rend();++it)if(it->length>high) {
            steps.push_back({high+1,it->length,it->end});high=it->length;
        }
    }
    Route q_deleted(int t,int owner,Route base) const {
        const int age=t-owner;
        if(age>=base.length)return base;
        if(!age)return {};
        auto it=std::lower_bound(steps.begin(),steps.end(),age,
            [](const Step& a,int h){return a.high<h;});
        return {age,it->end};
    }
    Route k_deleted(int s,Route base) const {
        if(s<=base.end-base.length || s>base.end)return base;
        auto after=std::lower_bound(exact.begin(),exact.end(),s,[](Route a,int owner){return a.end<owner;});
        Route answer;
        if(after!=exact.begin())answer=prefix[size_t(after-exact.begin()-1)];
        auto step=std::partition_point(steps.begin(),steps.end(),[&](const Step& a){return a.end-a.low>=s;});
        if(step!=steps.begin()) {
            --step;answer=std::max(answer,Route{std::min(step->high,step->end-s),step->end});
        }
        return answer;
    }
    template<class Add>
    void k_ranges(Route base,Add&& add) const {
        int s=base.end-base.length+1;
        size_t left=0,right=steps.size();
        while(s<=base.end) {
            while(left<exact.size() && exact[left].end<s)++left;
            while(right && steps[right-1].end-steps[right-1].low<s)--right;
            const Route a=left?prefix[left-1]:Route{};
            int hi=left<exact.size()?std::min(base.end,exact[left].end):base.end;
            if(right) {
                const Step& r=steps[right-1];
                hi=std::min(hi,r.end-r.low);
                // Equal positive length goes to the right endpoint (end>s).
                const int last=a.length>r.high?s-1:std::min(hi,r.end-a.length);
                if(s<=last)add(s,last,r.end);
                if(last<hi)add(std::max(s,last+1),hi,a.end);
            } else add(s,hi,a.end);
            s=hi+1;
        }
    }
    template<class Removed,class Cost>
    void repairs(std::vector<Repair>& candidates,Removed&& removed,Cost&& cost,std::vector<double>& out) {
        std::sort(candidates.begin(),candidates.end(),[](const Repair& a,const Repair& b){
            return a.owner!=b.owner?a.owner<b.owner:a.bit<b.bit;
        });
        for(size_t i=0;i<candidates.size();) {
            const auto edit=candidates[i++];Route best=edit.route;
            while(i<candidates.size() && candidates[i].owner==edit.owner && candidates[i].bit==edit.bit)
                best=std::max(best,candidates[i++].route);
            const Route other=removed(edit.owner,edit.bit);
            if(best>other && best.end!=other.end)
                out[size_t(edit.owner)*bits+edit.bit]+=cost(best.end)-cost(other.end);
        }
    }
public:
    EventSweep(const uint64_t* query,const uint64_t* keys,int b,Stats& work):q(query),k(keys),bits(b),stats(work){}
    template<class Report,class Cost>
    void row_events(int t,Route base,Report&& report,Cost&& cost,
            std::vector<double>& qdiff,std::vector<double>& kdiff,std::vector<double>& cq,
            std::vector<double>& ck,const std::array<int,64>* singleton=nullptr) {
        exact.clear();qr.clear();kr.clear();
        auto emit_both=[&](int tail,int bit,int len,int end){
            qr.push_back({t-tail,bit,{len,end}});
            kr.push_back({end-tail,bit,{len,end}});
            ++stats.k_centers;
        };
        report([&](int len,int end){exact.push_back({len,end});},emit_both,
               [&](int tail,int bit,int len,int end){qr.push_back({t-tail,bit,{len,end}});},
               [&](int tail,int bit,int len,int end){kr.push_back({end-tail,bit,{len,end}});});
        prepare();
        auto range=[&](std::vector<double>& diff,int lo,int hi,int end) {
            if(end==base.end)return;
            const double value=cost(end);
            diff[lo]+=value;diff[hi+1]-=value;
        };
        if(base.length) {
            range(qdiff,t,t,-1);
            for(const Step& step:steps)if(step.low<base.length)
                range(qdiff,t-std::min(step.high,base.length-1),t-step.low,step.end);
            k_ranges(base,[&](int lo,int hi,int end){range(kdiff,lo,hi,end);});
        }
        // A batched one-symbol repair already supplies short rows. For
        // longer rows only a Q edit at the last symbol needs that seed.
        if(singleton && base.length>=2)for(int j=0;j<bits;++j) {
            const int end=(*singleton)[j];
            if(end>=0)cq[size_t(t)*bits+j]+=cost(end)-cost(-1);
        }
        repairs(qr,[&](int u,int bit){
            if(singleton && u==t && (*singleton)[bit]>=0)return Route{1,(*singleton)[bit]};
            return q_deleted(t,u,base);
        },cost,cq);
        repairs(kr,[&](int s,int bit){
            if(singleton && base.length<=1 && s>base.end && (q[t]^k[s])==(UINT64_C(1)<<bit))
                return Route{1,s};
            return k_deleted(s,base);
        },cost,ck);
    }
    size_t bytes() const {
        return (exact.capacity()+prefix.capacity())*sizeof(Route)+steps.capacity()*sizeof(Step)
            +(qr.capacity()+kr.capacity())*sizeof(Repair);
    }
};

// Adapter for the independently indexed structural/approximation routines.
class RowSweep:public EventSweep {
    const Index& index;
    int width;
public:
    RowSweep(const Index& ix,int b,Stats& work):EventSweep(ix.q.data(),ix.k.data(),b,work),index(ix),width(b){}
    template<class Report,class Cost>
    void row(int t,Report&& report,Cost&& cost,std::vector<double>& qdiff,std::vector<double>& kdiff,
             std::vector<double>& cq,std::vector<double>& ck,const std::array<int,64>* singleton=nullptr) {
        row_events(t,index.baseline()[t],[&](auto exact,auto repair,auto,auto) {
            report([&](int r) {
                const int a=index.q[t]==index.k[r]?index.lcs(t,r):0;
                if(a)exact(a,r);
                const int u=t-a,s=r-a;
                if(s<0)return;
                const auto mask=index.q[u]^index.k[s];
                if(!mask || (mask&(mask-1)))return;
                const int bit=near_detail::one_bit_index(mask);
                if(bit<width)repair(a,bit,a+1+index.lcs(u-1,s-1),r);
            });
        },cost,qdiff,kdiff,cq,ck,singleton);
    }
};

// Enumerate full-word matches to projections omitting one disjoint band.
// Unrelated words or hash collisions only consume the bounded search budget.
struct Reports {
    std::vector<size_t> offsets;
    std::vector<int> ends;
    size_t peak_bytes=0;
    bool build(const uint64_t* q,const uint64_t* k,int n,int bits) {
        const int bands=bits>=32?2:4;
        size_t slots=1;while(slots<size_t(n)*2)slots*=2;
        std::vector<int> heads(slots*bands,-1),next(size_t(n)*bands);
        std::array<uint64_t,4> omitted{};
        for(int j=0;j<bands;++j) {
            const int lo=bits*j/bands,width=bits*(j+1)/bands-lo;
            omitted[j]=((UINT64_C(1)<<width)-1)<<lo;
        }
        const auto bucket=[&](uint64_t word,int j) {
            uint64_t x=word&~omitted[j];x^=x>>31;x*=UINT64_C(0x9e3779b97f4a7c15);x^=x>>33;
            return j*slots+(size_t(x)&(slots-1));
        };
        uint64_t work=0,limit=uint64_t(n)*12;
        offsets.resize(size_t(n)+1);
        for(int t=1;t<n;++t) {
            for(int j=0;j<bands;++j) {
                const auto slot=bucket(k[t-1],j);
                next[size_t(t-1)*bands+j]=heads[slot];heads[slot]=t-1;
                for(int r=heads[bucket(q[t],j)];r>=0;r=next[size_t(r)*bands+j]) {
                    if(++work>limit)return false;
                    const auto diff=q[t]^k[r];
                    if(!diff?j==0:(!(diff&~omitted[j]) && !(diff&(diff-1))))ends.push_back(r);
                }
            }
            offsets[t+1]=ends.size();
        }
        peak_bytes=(heads.capacity()+next.capacity()+ends.capacity())*sizeof(int)+offsets.capacity()*sizeof(size_t);
        return true;
    }
};

// Count causal reports before sorting any row. Rebuild the accepted reports
// from occurrence lists so their sum J never appears in the storage bound.
class Occurrences {
    int bits;
    bool direct;
    uint64_t first_word,varying=0,allowed;
    ExactWordRanks ranks;
    std::vector<int> offsets,ends;
    int rank(uint64_t word) const {
        if(direct)return word<(UINT64_C(1)<<bits)?int(word):-1;
        return ranks.find(word);
    }
    template<class Visit> void neighbors(uint64_t word,Visit visit) const {
        // Bits fixed in the entire key dictionary rule out whole groups
        // of probes. One inconsistent fixed bit uniquely determines an edit.
        const uint64_t fixed=(word^first_word)&~varying;
        if(fixed) {
            if(!(fixed&(fixed-1)) && (fixed&allowed)) {
                const int id=rank(word^fixed);
                if(id>=0)visit(id);
            }
            return;
        }
        int id=rank(word);
        if(id>=0)visit(id);
        for(uint64_t mask=varying&allowed;mask;mask&=mask-1) {
            id=rank(word^(mask&(-mask)));
            if(id>=0)visit(id);
        }
    }
public:
    Occurrences(const uint64_t* key,int n,int width):bits(width),direct(bits<=8),first_word(key[0]),
        allowed(bits==64?UINT64_MAX:(UINT64_C(1)<<bits)-1) {
        for(int i=1;i<n;++i)varying|=key[i]^first_word;
        if(direct)for(int r=0;r<n;++r)if(key[r]>>bits){direct=false;break;}
        int count;
        if(direct)count=1<<bits;
        else {
            std::vector<uint64_t> words(key,key+n);
            std::sort(words.begin(),words.end());
            words.erase(std::unique(words.begin(),words.end()),words.end());
            count=int(words.size());ranks.build(std::move(words));
        }
        offsets.resize(size_t(count)+1);
    }
    bool admit(const uint64_t* query,const uint64_t* key,int n,uint64_t limit) {
        std::vector<int> count(offsets.size()-1);
        uint64_t total=0;
        for(int t=1;t<n;++t) {
            ++count[rank(key[t-1])];
            neighbors(query[t],[&](int id){total+=count[id];});
            if(total>limit)return false;
        }
        for(size_t i=0;i<count.size();++i)offsets[i+1]=offsets[i]+count[i];
        ends.resize(offsets.back());
        std::copy_n(offsets.begin(),count.size(),count.begin());
        for(int r=0;r<n-1;++r)ends[count[rank(key[r])]++]=r;
        return true;
    }
    template<class Emit> void row(uint64_t word,int t,Emit emit) const {
        neighbors(word,[&](int id){
            for(int i=offsets[id];i<offsets[id+1] && ends[i]<t;++i)emit(ends[i]);
        });
    }
    size_t bytes() const {
        return ranks.workspace_bytes()+(2*offsets.capacity()+ends.capacity())*sizeof(int);
    }
};

template<class Report>
inline void evaluate(const Index& index,int bits,const uint8_t* v,const double* g,const double* h,
        const double* e0,const double* e1,const std::vector<int32_t>& rows,Report&& report,
        std::vector<double>& cq,std::vector<double>& ck,Stats& stats,
        const rosa_random::OnegramPlan* singleton=nullptr) {
    const int n=index.n;
    std::vector<double> qdiff(size_t(n)+1),kdiff(size_t(n)+1),costs(size_t(n)+1);
    std::vector<int> stamp(size_t(n)+1,-1);
    RowSweep sweep(index,bits,stats);
    const bool short_rows=!singleton && ShortEndpointSweep::applicable(index.baseline());
    ShortEndpointSweep small(index,bits,short_rows);
    for(int t:rows) {
        const int before=index.baseline()[t].end;
        auto cost=[&](int end) {
            if(end==before)return 0.;
            if(stamp[end+1]!=t) {
                const auto value=[&](int r,int c){return r<0?e0[c]:(2*int(v[size_t(r+1)*bits+c])-1)*e1[c];};
                double sum=0;
                for(int c=0;c<bits;++c) {
                    const size_t cell=size_t(t)*bits+c;
                    const double dy=value(end,c)-value(before,c);
                    sum+=g[cell]*dy;
                    if(h)sum+=(.5*h[cell]*dy)*dy;
                }
                stamp[end+1]=t;costs[end+1]=sum;
            }
            return costs[end+1];
        };
        if(short_rows) {
            small.row(t,[&](auto emit){report(t,emit);},cost,qdiff,kdiff,cq,ck,stats);
            continue;
        }
        std::array<int,64> latest;
        if(singleton) {
            latest.fill(-1);
            for(const auto& edge:singleton->qneighbors[singleton->qclass[t]]) {
                const auto first=singleton->krows.begin()+singleton->ko[edge.other];
                const auto stop=std::lower_bound(first,singleton->krows.begin()+singleton->ko[edge.other+1],t);
                if(stop!=first)latest[edge.bit]=stop[-1];
            }
        }
        sweep.row(t,[&](auto emit){report(t,emit);},cost,qdiff,kdiff,cq,ck,singleton?&latest:nullptr);
    }
    double qsum=0,ksum=0;
    for(int s=0;s<n;++s) {
        if(short_rows){qsum=qdiff[s];ksum=kdiff[s];}
        else {qsum+=qdiff[s];ksum+=kdiff[s];}
        for(int j=0;j<bits;++j){cq[size_t(s)*bits+j]+=qsum;ck[size_t(s)*bits+j]+=ksum;}
    }
    std::fill_n(cq.begin(),bits,0.);std::fill_n(ck.end()-bits,bits,0.);
    stats.peak_workspace_bytes=std::max<uint64_t>(stats.peak_workspace_bytes,
        index.workspace_bytes()+sweep.bytes()+small.bytes()+(qdiff.capacity()+kdiff.capacity()+costs.capacity())*sizeof(double)
        +stamp.capacity()*sizeof(int));
}
// Count equal last symbols only. Once the length-one repairs are summed
// by symbol class, every remaining alignment has an exact pair at its last
// or penultimate symbol. These two translated joins contain at most 2E rows.
inline uint64_t equal_pairs(const rosa_random::OnegramPlan& plan,uint64_t limit) {
    std::vector<int> seen(plan.nk);
    uint64_t total=0;
    for(int t=1;t<plan.n;++t) {
        ++seen[plan.kclass[t-1]];
        const int id=plan.qsame[plan.qclass[t]];
        if(id>=0)total+=seen[id];
        if(total>limit)break;
    }
    return total;
}

inline void equal_sweep(const rosa_random::OnegramPlan& plan,uint64_t count,const std::vector<Route>& base,
        const std::vector<int32_t>& active,const uint8_t* v,const double* g,const double* h,
        const double* e0,const double* e1,const int32_t* routes,bool randomized,uint64_t seed,
        std::vector<double>& cq,std::vector<double>& ck,Stats& stats) {
    const int n=plan.n;
    std::vector<double> weight;
    if(std::any_of(base.begin(),base.end(),[](Route r){return r.length>1;})) {
        weight.resize(n);
        for(int t=0;t<n;++t)weight[t]=base[t].length<=1;
    }
    uint64_t work[16]={};size_t workspace=0;
    const double* selected=weight.empty()?nullptr:weight.data();
    if(h)plan.apply<true>(v,g,h,e0,e1,routes,cq.data(),ck.data(),work,false,selected,&workspace,true);
    else plan.apply<false>(v,g,h,e0,e1,routes,cq.data(),ck.data(),work,false,selected,&workspace,true);
    stats.peak_workspace_bytes=std::max<uint64_t>(stats.peak_workspace_bytes,workspace);
    if(count) {
        Index index(plan.q,plan.k,n,randomized,seed,&base);
        const auto exact=[&](int t,auto emit) {
            const int id=plan.qsame[plan.qclass[t]];
            if(id<0)return;
            for(int i=plan.ko[id];i<plan.ko[id+1] && plan.krows[i]<t;++i)emit(plan.krows[i]);
        };
        evaluate(index,plan.d,v,g,h,e0,e1,active,[&](int t,auto emit){
            exact(t,emit);
            if(t>1)exact(t-1,[&](int r){
                if(plan.q[t]!=plan.k[r+1])emit(r+1);
            });
        },cq,ck,stats,&plan);
    }
    stats.peak_workspace_bytes+=weight.capacity()*sizeof(double);
}
} // namespace endpoint_credit
} // namespace rosa_subq
