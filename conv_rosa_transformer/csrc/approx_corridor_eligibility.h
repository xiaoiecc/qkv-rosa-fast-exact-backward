#pragma once
// Cheap sufficient evidence that shared FFT corridor contraction can pay.
// A globally unique K word certifies a singleton repair at its birth. This
// planner never predicts a route from a fingerprint or runs a second SAM.
#include <algorithm>
#include <cstdint>
#include <vector>

namespace rosa_random {
struct CertifiedBirthCorridor {
    int owner,bit,start,stop,payload_start;
};
struct CorridorEligibility {
    bool eligible=false,capped=false;
    uint64_t candidates=0,jump_probes=0,certified_queries=0,groups=0;
    uint64_t represented_rows=0,selected_rows=0,selected_queries=0;
    double direct_work=0,fft_work=0;
    int maximum_length=0;
};

// Plan provides the existing OnegramPlan class/CSR arrays and Hamming-one
// neighbors. active_prefix optionally counts rows with a nonzero numerical
// score; the planner then prices exactly those rows, not their interval span.
// The optional trace is for independent tests and need not be retained by a
// training wrapper. Both work caps are O(T), independent of the input family.
template<class Plan>
CorridorEligibility corridor_fft_eligibility(const Plan&plan,
    const int32_t*baseline_lengths,const int32_t*active_prefix,
    bool quadratic,std::vector<CertifiedBirthCorridor>*trace=nullptr) {
    CorridorEligibility result;
    const int n=plan.n,d=plan.d;
    if(trace)trace->clear();
    if(n<16)return result;
    if(baseline_lengths) {
        const int longest=*std::max_element(baseline_lengths,baseline_lengths+n);
        // An M-row corridor contains an unedited suffix of length M-1.
        if(longest<15)return result;
    }
    const uint64_t cap=8*uint64_t(n);
    std::vector<int> qrun_end,krun_end;
    std::vector<CertifiedBirthCorridor> queries;
    bool exhausted=false;
    for(int id=0;id<plan.nq&&!exhausted;++id)
        for(const auto&neighbor:plan.qneighbors[id]) {
            const int kid=neighbor.other;
            if(plan.ko[kid+1]-plan.ko[kid]!=1)continue;
            const int s=plan.krows[plan.ko[kid]];
            auto first=std::upper_bound(plan.qrows.begin()+plan.qo[id],
                plan.qrows.begin()+plan.qo[id+1],s);
            for(auto it=first;it!=plan.qrows.begin()+plan.qo[id+1];++it) {
                if(++result.candidates>cap){result.capped=true;exhausted=true;break;}
                const int u=*it;
                if(u+1>=n||plan.q[u+1]!=plan.k[s+1])continue;
                if(qrun_end.empty()) {
                    qrun_end.resize(n);krun_end.resize(n);
                    for(int x=n-1;x>=0;--x) {
                        qrun_end[x]=x+1<n&&plan.q[x]==plan.q[x+1]?qrun_end[x+1]:x+1;
                        krun_end[x]=x+1<n&&plan.k[x]==plan.k[x+1]?krun_end[x+1]:x+1;
                    }
                }
                int x=u+1,y=s+1;
                while(x<n&&plan.q[x]==plan.k[y]) {
                    if(++result.jump_probes>cap){result.capped=true;exhausted=true;break;}
                    const int step=std::min(qrun_end[x]-x,krun_end[y]-y);
                    x+=step;y+=step;
                }
                if(exhausted)break; // An unfinished LCE is not a certificate.
                const int length=x-u;
                if(length<16)continue;
                queries.push_back({u,neighbor.bit,u,x-1,s+1});
                ++result.certified_queries;
                result.maximum_length=std::max(result.maximum_length,length);
            }
            if(exhausted)break;
        }
    if(trace)*trace=queries;
    std::sort(queries.begin(),queries.end(),[](const auto&a,const auto&b) {
        return a.stop!=b.stop?a.stop<b.stop:
            (a.payload_start!=b.payload_start?a.payload_start<b.payload_start:a.owner<b.owner);
    });
    for(size_t begin=0;begin<queries.size();) {
        size_t end=begin+1;
        while(end<queries.size()&&queries[end].stop==queries[begin].stop&&
            queries[end].payload_start==queries[begin].payload_start)++end;
        uint64_t work=0;int m=0;
        for(size_t at=begin;at<end;++at) {
            const auto&q=queries[at];const int length=q.stop-q.start+1;
            m=std::max(m,length);
            work+=active_prefix?uint64_t(active_prefix[q.stop+1]-active_prefix[q.start]):uint64_t(length);
        }
        result.represented_rows+=work;++result.groups;
        if(end-begin>=8&&work>=8*uint64_t(m)) {
            int size=1,levels=0;while(size<2*m-1){size*=2;++levels;}
            // Before payload classes are known, price two channels per bit
            // for quadratic credit. The real contraction planner may reuse
            // complementary/equal columns and choose fewer transforms.
            const double direct=double(4+d)*work;
            const double fft=512.+8.*double(quadratic?2*d:d)*size*levels;
            // Require a substantial margin: this selects an entire full
            // contraction engine, not merely one isolated interval query.
            if(direct>2.*fft) {
                result.selected_rows+=work;result.selected_queries+=end-begin;
                result.direct_work+=direct;result.fft_work+=fft;
            }
        }
        begin=end;
    }
    result.eligible=result.selected_queries>=8&&result.direct_work>2.*result.fft_work;
    return result;
}
} // namespace rosa_random
