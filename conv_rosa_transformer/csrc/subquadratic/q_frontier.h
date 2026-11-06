#pragma once
// A bounded live frontier for Q deletion and repair. The current forward
// matcher advances all branches; a failed work certificate publishes nothing.
#include "index.h"
#include "../forward.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace rosa_subq {

inline bool q_frontier(const Index& index,const Scores& scores,int bits,
        std::vector<double>& credit,Stats& stats,std::uint64_t seed=0,
        bool include_deletion=false) {
    const int n=index.n;
    if(bits<1 || bits>64 || scores.n!=n || scores.d<1 || !scores.weights ||
       !scores.payload || !scores.empty || credit.size()!=std::size_t(n)*bits)
        throw std::invalid_argument("invalid Q frontier dimensions");
    int logarithm=0;
    for(unsigned value=static_cast<unsigned>(n);value;value>>=1)++logarithm;
    const int cap=2*std::max(32,2*bits+2*logarithm);
    // The deleted-owner cursors that are still relevant at t are exactly
    // ages 0..L[t]-1. Since L can increase by at most one per row, none of
    // these owners has died earlier. This is a complete early certificate
    // of cap failure, independent of the payload and input alphabet.
    const auto& baseline=index.baseline();
    for(const Route route:baseline)if(route.length>cap)return false;
    // Forking a long original cursor gives each birth fresh suffix-link
    // potential. A live-count cap alone therefore cannot bound fallback work.
    // Count every transition (including the baseline and root certificates).
    // The last rejected step can overshoot by at most n transitions.
    const std::uint64_t probe_budget=2*std::uint64_t(n)*cap;
    rosa_exact::SamForward matcher(index.q.data(),index.k.data(),n,seed);
    using Cursor=rosa_exact::SamForward::Cursor;
    struct Branch { int owner,bit;Cursor cursor; };
    std::vector<Branch> live;live.reserve(std::size_t(cap)+1);
    std::vector<double> delta(credit.size(),0.);
    // Owner indexing avoids a modulo in every branch lookup. Only a suffix
    // of at most cap entries is live, but the retained storage is O(n).
    std::vector<Cursor> barrier(n);
    std::vector<double> deleted_score(n),deletion_delta(include_deletion?n:0,0.);
    Cursor original;
    std::uint64_t evaluated=0;
    for(int t=0;t<n;++t) {
        const Cursor before=original;
        matcher.step(t,index.q[t],original,n+1);
        if(matcher.counters.transition_probes>probe_budget)return false;
        const Route base=baseline[t];
        assert(original.length==base.length);
        original.end=base.end;
        const double* row=scores.row(t);
        const bool nonzero=std::any_of(row,row+scores.d,[](double x){return x!=0.;});
        const double base_score=nonzero?scores.score(t,base.end):0.;
        const int first=t-base.length+1;
        for(int owner=first;owner<=t;++owner) {
            Cursor& removed=barrier[owner];
            // A deletion starts an empty cursor at its owner. While L>age,
            // its complete age-symbol tail matches, so this step is direct.
            // When L<=age, deletion agrees with the original forever; the
            // left boundary first never decreases and discards that cursor.
            if(owner<t) {
                matcher.step(t,index.q[t],removed);
                if(matcher.counters.transition_probes>probe_budget)return false;
                assert(removed.length==t-owner);
            }
            if(nonzero) {
                const double value=removed.end==base.end?base_score:scores.score(t,removed.end);
                deleted_score[owner]=value;
                if(include_deletion)deletion_delta[owner]+=value-base_score;
            }
        }
        auto add=[&](const Branch& branch) {
            if(nonzero) {
                const bool alive=branch.owner>=first;
                const int removed_end=alive?barrier[branch.owner].end:base.end;
                if(branch.cursor.end!=removed_end)
                    delta[std::size_t(branch.owner)*bits+branch.bit]
                        +=scores.score(t,branch.cursor.end)
                           -(alive?deleted_score[branch.owner]:base_score);
                ++evaluated;
            }
        };
        std::size_t retained=0;
        for(std::size_t at=0;at<live.size();++at) {
            Branch branch=live[at];const int age=t-branch.owner;
            matcher.step(t,index.q[t],branch.cursor,age+1);
            if(matcher.counters.transition_probes>probe_budget)return false;
            // Any later crossing match would extend a crossing match now:
            // removing its last symbol also preserves strict causality. Once
            // none exists, this edited owner can never affect a later row.
            if(branch.cursor.length<=age)continue;
            add(branch);
            live[retained++]=branch;
        }
        live.resize(retained);
        if(live.size()+std::size_t(base.length)>std::size_t(cap))return false;
        for(int bit=0;bit<bits;++bit) {
            const auto word=index.q[t]^(std::uint64_t(1)<<bit);
            const bool present=matcher.has_causal_symbol(t,word);
            if(matcher.counters.transition_probes>probe_budget)return false;
            if(!present)continue;
            Branch branch{t,bit,before};
            matcher.step(t,word,branch.cursor);
            if(matcher.counters.transition_probes>probe_budget)return false;
            assert(branch.cursor.length>0);
            if(live.size()+std::size_t(base.length)==std::size_t(cap))return false;
            add(branch);
            live.push_back(branch);
        }
    }
    // A live edited route has length > age; every Q-deleted route has length
    // <= age. It is therefore exactly the winning repair, independent of the
    // numerical payload. Commit only after both work certificates succeed.
    for(int owner=0;owner<n;++owner)for(int bit=0;bit<bits;++bit) {
        const auto at=std::size_t(owner)*bits+bit;
        if(include_deletion)credit[at]+=deletion_delta[owner];
        credit[at]+=delta[at];
    }
    stats.q_heavy_rows+=evaluated;
    stats.suffix_queries+=matcher.counters.endpoint_queries;
    stats.peak_workspace_bytes=std::max<std::uint64_t>(stats.peak_workspace_bytes,
        index.workspace_bytes()+matcher.retained_capacity_bytes()
        +(delta.capacity()+deleted_score.capacity()+deletion_delta.capacity())*sizeof(double)
        +live.capacity()*sizeof(Branch)+barrier.capacity()*sizeof(Cursor));
    return true;
}
} // namespace rosa_subq
