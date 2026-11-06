#pragma once
// Exact K repair from a certified, bounded set of near-winning endpoints.
#include "index.h"
#include "types.h"
#include <limits>
#include <stdexcept>
#include <utility>

namespace rosa_subq {
namespace near_detail {

inline int one_bit_index_portable(std::uint64_t value) {
    int bit=0;
    if(value>>32){value>>=32;bit+=32;}
    if(value>>16){value>>=16;bit+=16;}
    if(value>>8){value>>=8;bit+=8;}
    if(value>>4){value>>=4;bit+=4;}
    if(value>>2){value>>=2;bit+=2;}
    return bit+int(value>>1);
}

inline int one_bit_index(std::uint64_t value) {
    assert(value && !(value&(value-1)));
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(value);
#else
    return one_bit_index_portable(value);
#endif
}

// Only owners in the original winning occurrence can have deletion credit.
// On this contiguous owner interval, each reported exact match contributes
// an unchanged route on two exterior ranges and an affine suffix inside.
// Two bucket scans produce every deletion route in O(reported + owners).
// The upper word orders match lengths; end+1 in the lower word implements
// the latest-end tie and gives the empty route the all-zero representation.
inline std::uint64_t order_key(Route route) {
    return (std::uint64_t(std::uint32_t(route.length))<<32)|std::uint32_t(route.end+1);
}
inline Route route_from_key(std::uint64_t key) {
    return Route{int(key>>32),int(std::uint32_t(key))-1};
}

class Survivors {
    std::vector<std::uint64_t> prefix_,suffix_;
    std::vector<int> crossing_;
    int first_=0;
public:
    void build(const std::vector<Route>& routes,int first,int last) {
        first_=first;const int count=last-first+1;
        prefix_.assign(count,0);suffix_.assign(count,0);crossing_.assign(count,-1);
        for(Route route:routes) {
            const std::uint64_t key=order_key(route);
            const int start=route.end-route.length+1;
            const int before=std::min(count-1,start-first-1);
            if(before>=0)prefix_[before]=std::max(prefix_[before],key);
            const int after=std::max(0,route.end-first+1);
            if(after<count)suffix_[after]=std::max(suffix_[after],key);
            const int crossing=std::max(0,start-first);
            if(crossing<count)crossing_[crossing]=std::max(crossing_[crossing],route.end);
        }
        for(int i=count-2;i>=0;--i)prefix_[i]=std::max(prefix_[i],prefix_[i+1]);
        std::uint64_t best=0;int end=-1;
        for(int i=0;i<count;++i) {
            best=std::max(best,suffix_[i]);end=std::max(end,crossing_[i]);
            prefix_[i]=std::max(prefix_[i],best);
            if(end>first+i)prefix_[i]=std::max(prefix_[i],order_key(Route{end-first-i,end}));
        }
    }
    Route query(int owner) const {return route_from_key(prefix_[owner-first_]);}
    std::size_t bytes() const {
        return (prefix_.capacity()+suffix_.capacity())*sizeof(std::uint64_t)
            +crossing_.capacity()*sizeof(int);
    }
};

} // namespace near_detail

// Return false before writing any credits when the complete witness set
// exceeds work_budget. The caller then uses the structural short/long sum.
// This is an output-sensitive base case of that sum, not an input-family
// classification: every accepted row has an exact completeness certificate.
inline bool near_k(const Index& index,const Scores& scores,int bits,
                   std::vector<double>& credits,Stats& stats,std::uint64_t work_budget,
                   bool include_deletion=false) {
    const int n=index.n,d=scores.d;
    if(scores.n!=n || d<1 || bits<1 || bits>64 || credits.size()<std::size_t(n)*bits)
        throw std::invalid_argument("invalid near K repair dimensions");
    struct Plan {
        int row,split,lo,hi;
        std::pair<int,int> recent,older;
    };
    std::vector<Plan> plans;plans.reserve(n);
    const SuffixArray* suffix=nullptr;
    std::vector<int> key_prefix,key_ends;
    std::uint64_t work=0;
    for(int t=1;t<n;++t) {
        bool active=false;for(int c=0;c<d;++c)if(scores.row(t)[c]!=0.){active=true;break;}
        if(!active)continue;
        const Route original=index.baseline()[t];const int length=original.length;
        Plan plan{t,0,0,t,{0,0},{0,0}};
        if(length>=2) {
            if(!suffix) {
                suffix=&index.reverse_suffixes();
                key_prefix.resize(std::size_t(suffix->size())+1);key_ends.reserve(n);
                for(int rank=0;rank<suffix->size();++rank) {
                    key_prefix[rank]=int(key_ends.size());const int at=suffix->sa[rank];
                    if(at>n && at<=2*n)key_ends.push_back(2*n-at);
                }
                key_prefix.back()=int(key_ends.size());
            }
            plan.split=length/2;
            plan.recent=suffix->prefix_interval(suffix->rank[n-1-t],plan.split);
            plan.older=suffix->prefix_interval(suffix->rank[n-1-(t-plan.split)],length-plan.split);
            plan.recent={key_prefix[plan.recent.first],key_prefix[plan.recent.second]};
            plan.older={key_prefix[plan.older.first],key_prefix[plan.older.second]};
            plan.lo=original.end-length+1;
            plan.hi=std::min(t,original.end+length);
        }
        // Count complete K occurrence intervals, including noncausal ends.
        // Causality is filtered during reporting, so this is a safe bound.
        const std::uint64_t amount=std::uint64_t(plan.hi-plan.lo)
            +std::uint64_t(plan.recent.second-plan.recent.first)
            +std::uint64_t(plan.older.second-plan.older.first);
        if(amount>work_budget-work)return false;
        work+=amount;plans.push_back(plan);
    }
    if(plans.empty())return true;

    std::vector<int> visited(n,-1);
    std::vector<std::uint64_t> winners(std::size_t(n)*bits);
    std::vector<std::size_t> touched;
    std::vector<Route> exact;
    near_detail::Survivors survivors;
    std::vector<Route> deleted_cache(n);
    std::vector<int> deleted_at(n,-1);
    std::vector<double> deletion_credit(include_deletion?n:0,0.);
    const std::uint64_t allowed=bits==64?~UINT64_C(0):(UINT64_C(1)<<bits)-1;
    const auto local_lcs=[&](int a,int b) {
        // A bounded prefix certificate avoids index/run metadata for short
        // mismatching arms. Four equal symbols always continue in the exact
        // index; no input can turn this base case into a long linear scan.
        if(a<3 || b<3)return index.lcs(a,b);
        if(index.q[a]!=index.k[b])return 0;
        if(index.q[a-1]!=index.k[b-1])return 1;
        if(index.q[a-2]!=index.k[b-2])return 2;
        if(index.q[a-3]!=index.k[b-3])return 3;
        return 4+index.lcs(a-4,b-4);
    };
    for(const Plan& plan:plans) {
        const int t=plan.row;const Route original=index.baseline()[t];
        touched.clear();exact.clear();bool survivors_ready=false;
        const auto consider=[&](int end,int confirmed_right=0) {
            assert(end>=0 && end<t);
            // The recent-half interval reports each K endpoint exactly once
            // and is visited first. Only the other half/window need to test
            // its mark; every report still installs the mark for those passes.
            if(!confirmed_right && visited[end]==t)return;
            visited[end]=t;
            const int right=confirmed_right+local_lcs(t-confirmed_right,end-confirmed_right);++stats.suffix_queries;
            if(confirmed_right || (!plan.split && right))exact.push_back(Route{right,end});
            const int owner=end-right,center=t-right;
            if(owner<0)return;
            const std::uint64_t mask=index.q[center]^index.k[owner];
            if(!mask || (mask&(mask-1)) || (mask&~allowed))return;
            const int bit=near_detail::one_bit_index(mask);
            const bool affected=original.length && owner<=original.end
                && owner>=original.end-original.length+1;
            const Route surviving=affected?(original.end>owner?Route{original.end-owner,original.end}:Route{}):original;
            if(right+1<surviving.length) {
                // The left boundary is outside the unique changed symbol.
                // A mismatch here certifies failure before an LCS query.
                const int start=end-surviving.length+1;
                if(start<0 || index.q[t-surviving.length+1]!=index.k[start])return;
            }
            const int left=local_lcs(center-1,owner-1);++stats.suffix_queries;
            const Route candidate{right+1+left,end};
            const std::uint64_t candidate_key=near_detail::order_key(candidate);
            if(candidate_key<=near_detail::order_key(surviving))return;
            const std::size_t key=std::size_t(owner)*bits+bit;
            std::uint64_t& best=winners[key];
            if(!best)touched.push_back(key);
            best=std::max(best,candidate_key);
            ++stats.k_centers;
        };

        // A repair beating an unaffected baseline has length at least L.
        // At most one of its two disjoint baseline-suffix halves contains
        // the edited character; the other half is an exact witness.
        if(plan.split) {
            for(int rank=plan.recent.first;rank<plan.recent.second;++rank) {
                const int end=key_ends[rank];if(end<t)consider(end,plan.split);
            }
            for(int rank=plan.older.first;rank<plan.older.second;++rank) {
                const int end=key_ends[rank]+plan.split;if(end<t)consider(end);
            }
        }
        // If deleting owner s damages the baseline, s lies inside its
        // length-L occurrence. A shorter repair ending at r has r=s+a,
        // where a<L, hence r belongs to this contiguous interval of <2L
        // endpoints. Long repairs were already certified by the halves.
        for(int end=plan.lo;end<plan.hi;++end)consider(end);

        const auto deleted_at_owner=[&](int owner) {
            if(deleted_at[owner]==t)return deleted_cache[owner];
            if(!survivors_ready) {
                survivors.build(exact,original.end-original.length+1,original.end);
                survivors_ready=true;
            }
            Route deleted=survivors.query(owner);
            // Every original match of length >= split was reported by the
            // recent half. A survivor that long is the global winner. With
            // no split, all causal endpoints were explicitly reported.
            if(plan.split && deleted.length<plan.split) {
                deleted=index.deleted(t,owner);++stats.suffix_queries;
            }
            deleted_at[owner]=t;deleted_cache[owner]=deleted;return deleted;
        };

        if(include_deletion && original.length) {
            const double* before=scores.value(original.end);const double* weight=scores.row(t);
            for(int owner=original.end-original.length+1;owner<=original.end;++owner) {
                const Route deleted=deleted_at_owner(owner);
                if(deleted.end==original.end)continue;
                const double* after=scores.value(deleted.end);
                double value=0.;for(int c=0;c<d;++c)value+=weight[c]*(after[c]-before[c]);
                deletion_credit[owner]+=value;
            }
        }

        for(std::size_t key:touched) {
            const std::uint64_t candidate_key=winners[key];winners[key]=0;
            const Route candidate=near_detail::route_from_key(candidate_key);
            const int owner=int(key/std::size_t(bits));
            const bool affected=original.length && owner<=original.end
                && owner>=original.end-original.length+1;
            Route deleted=original;
            if(affected)deleted=deleted_at_owner(owner);
            if(candidate_key<=near_detail::order_key(deleted) || candidate.end==deleted.end)continue;
            const double* a=scores.value(candidate.end);const double* b=scores.value(deleted.end);
            const double* weight=scores.row(t);double value=0.;
            for(int c=0;c<d;++c)value+=weight[c]*(a[c]-b[c]);
            credits[key]+=value;
        }
    }
    if(include_deletion)for(int owner=0;owner<n;++owner)for(int bit=0;bit<bits;++bit)
        credits[std::size_t(owner)*bits+bit]+=deletion_credit[owner];
    const std::size_t bytes=plans.capacity()*sizeof(Plan)+visited.capacity()*sizeof(int)
        +winners.capacity()*sizeof(std::uint64_t)+touched.capacity()*sizeof(std::size_t)
        +(key_prefix.capacity()+key_ends.capacity()+deleted_at.capacity())*sizeof(int)
        +(exact.capacity()+deleted_cache.capacity())*sizeof(Route)+survivors.bytes()
        +deletion_credit.capacity()*sizeof(double);
    stats.peak_workspace_bytes=std::max<std::uint64_t>(stats.peak_workspace_bytes,bytes);
    return true;
}

} // namespace rosa_subq
