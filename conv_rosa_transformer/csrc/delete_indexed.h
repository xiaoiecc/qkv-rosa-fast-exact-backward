// Uniform exact deletion ranges, with implicit constant-symbol run records.
// C++17; instantiate with the shared rosa_exact::SamRoutes index.
// Probes O(log T) adjacent endpoints per record, then skips longer gaps by
// index queries.  Short record gaps take time proportional to their length.
// Never expands a record into all length levels or all covered owners.
// A sliding endpoint is contracted by a global owner-vector difference array.
// The constructor's Index is templated only to allow an independent mock test;
// Index::latest(t,L) must return the latest causal exact length-L endpoint.
// runtime uses one algorithm, without input-type or size-based backend choice.
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace rosa_exact {

struct DeleteRoute { int length = 0, end = -1; };
struct DeleteNumericRange { int lo, hi, end; };
struct DeleteOwnerRange { int lo, hi; };

class IndexedDeleteRow {
    struct RStep { int lo, hi, end; };
    // End(s)=min(s-1,end). A negative length encodes a growing prefix;
    // its origin is end-abs(length)+1, so no fourth field is necessary.
    // Nonnegative lengths are constant. The next record supplies the hi.
    struct ARecord {
        int lo, length, end;
        int cap() const { return length<0 ? -length : length; }
        int origin() const { return end+length+1; } // growth records only
        int at_length(int s) const { return length<0 ? std::min(-length,s-origin()) : length; }
        bool sliding() const { return end>=lo; }
    };
    struct KRouteRun {
        int lo, hi, end, length;
        // end >= -1: literal endpoint, negative length encodes H.
        // end < -1: moving endpoint, encoded last endpoint = -end-2;
        // negative length encodes growth, positive length a fixed cap.
        DeleteRoute route(int s) const {
            if (end<-1) {
                const int cap=length<0 ? -length : length;
                return {length<0 ? std::min(cap,s+end+cap+1) : cap,s-1};
            }
            if (end<0) return {0,-1};
            return {length<0 ? std::min(-length,end-s) : length,end};
        }
    };
    int t_ = -1, L0_ = 0, r0_ = -1, first_ = 0, cross_ = 0, minimum_ = 0;
    int index_size_ = -1, probe_budget_ = 1;
    int minimum_owner_ = 0;
    uint64_t expanded_a_records_ = 0;
    DeleteRoute minimum_route_;
    std::vector<RStep> rsteps_;
    std::vector<ARecord> arecords_;
    std::vector<KRouteRun> kruns_;
    std::vector<DeleteNumericRange> qnumeric_, knumeric_;
    std::vector<DeleteOwnerRange> kowner_;
    std::vector<int> qrun_, kleft_, kright_;


public:
    IndexedDeleteRow(const uint64_t* q, const uint64_t* k, int n) {
        qrun_.resize(n);kleft_.resize(n);kright_.resize(n);
        for (int i=0;i<n;++i) {
            qrun_[i]=(i && q[i]==q[i-1]) ? qrun_[i-1]+1 : 1;
            kleft_[i]=(i && k[i]==k[i-1]) ? kleft_[i-1] : i;
        }
        for (int i=n-1;i>=0;--i)
            kright_[i]=(i+1<n && k[i]==k[i+1]) ? kright_[i+1] : i;
    }

    // Reusing one instance across rows preserves vector capacity. This reset
    // only touches actual records, never T owners or all L0 length levels.
    template<class Index>
    void reset(Index& index, int t, int L0, int r0) {
        t_ = t; L0_ = L0; r0_ = L0 ? r0 : -1;
        first_ = r0_ - L0_ + 1; cross_ = r0_ + 1; minimum_ = L0_;
        minimum_owner_ = first_;
        minimum_route_ = {L0_, r0_};
        rsteps_.clear(); arecords_.clear(); kruns_.clear();
        qnumeric_.clear(); knumeric_.clear(); kowner_.clear();
        expanded_a_records_ = 0;
        if (!L0_) return;
        expanded_a_records_ = 1;
        if (index_size_ != index.size()) {
            index_size_ = index.size();
            probe_budget_ = 1;
            for (int remaining = index_size_; remaining > 1; remaining >>= 1)
                ++probe_budget_;
        }

        // R[L]=latest causal endpoint with exact length >= L. Query one new
        // endpoint per whole constant-R length staircase, then jump to f+1.
        // Every endpoint >= limit has already failed this longer threshold.
        // Probe up to ceil(log2(T+1)) predecessors; only a longer gap needs
        // the index.  r0's length is already known and never queried again.
        int limit = t_;
        for (int L = 1; L <= L0_;) {
            int r = limit - 1;
            int f = 0;
            if (L == L0_) {
                r = r0_;
                f = L0_;
            } else {
                for (int probed = 0; r > r0_ && probed < probe_budget_; --r, ++probed) {
                    f = index.lcs(t_, r);
                    if (f >= L) break;
                }
                if (r == r0_) {
                    f = L0_;
                } else if (f < L) {
                    // All endpoints above r have already failed length L.
                    // The current causal maximum is therefore identical to
                    // the bounded predecessor, and the SAM caches it.
                    r = index.latest(t_, L);
                    if (r < 0) throw std::logic_error("missing deletion R endpoint");
                    f = r == r0_ ? L0_ : index.lcs(t_, r);
                }
            }
            if (f < L || f > L0_) throw std::logic_error("inconsistent deletion R length");
            rsteps_.push_back({L, f, r});
            limit = r;
            L = f + 1;
        }
        if (rsteps_.back().end != r0_)
            throw std::logic_error("deletion baseline endpoint disagrees with index");

        // A records within a constant-symbol K run have an exact affine
        // form, except at its one possible Q-run boundary. Keep the form,
        // not one record per endpoint. Every input uses this representation.
        const auto initial = index.prefix_best(t_, first_);
        int current_length = initial.length;
        int current_end = initial.end;
        int search=first_;
        arecords_.push_back({first_,current_length,current_end});
        const int qh=qrun_[t_];
        while (search < r0_) {
            const int threshold=std::max(1,current_length);
            int next=search, next_length=0;
            for (int probed=0;next<r0_ && probed<probe_budget_;++next,++probed) {
                next_length=index.lcs(t_,next);
                if (next_length>=threshold) break;
            }
            if (next>=r0_) break;
            if (next_length<threshold) {
                const auto iv=index.interval(t_,threshold);
                next=index.succ(iv.first,iv.second,next);
                if (next<0 || next>=r0_) break;
                next_length=index.lcs(t_,next);
            }
            const int right=std::min(r0_-1,kright_[next]);
            int stop=next;
            bool growing=false;
            current_length=next_length;
            if (right>next) {
                const int left=kleft_[next], boundary=left+qh-1;
                if (next<boundary) {
                    stop=std::min(right,boundary-1);
                    current_length=stop-left+1;
                    growing=(stop>next);
                } else if (next>boundary || next_length==qh) {
                    stop=right;current_length=qh;
                }
            }
            arecords_.push_back({next+1,growing ? -current_length : current_length,stop});
            expanded_a_records_ += uint64_t(stop-next+1);
            search=(current_length>qh) ? right+1 : stop+1;
        }

        // A(s) rises, H(s) falls. For positive f=A.length, A wins iff
        // s+f >= R[f]+1. Merge affine A pieces with constant-R steps.
        std::size_t rcursor=0;
        for (std::size_t i=0;i<arecords_.size();++i) {
            const auto& a=arecords_[i];
            if (!a.length) continue;
            const int end=i+1<arecords_.size() ? arecords_[i+1].lo-1 : r0_;
            if (a.length>=0) {
                while (rsteps_[rcursor].hi<a.length) ++rcursor;
                const int lo=std::max(a.lo,rsteps_[rcursor].end-a.length+1);
                if (lo<=end) {cross_=lo;break;}
            } else {
                const int cap=-a.length, origin=a.origin();
                int owner=a.lo;
                while (owner<=end) {
                    const int f=a.at_length(owner);
                    while (rsteps_[rcursor].hi<f) ++rcursor;
                    const auto& r=rsteps_[rcursor];
                    const int hi=cap>r.hi ? std::min(end,origin+r.hi) : end;
                    const int threshold=std::max(r.end+1-cap,(r.end+2+origin)/2);
                    const int lo=std::max(owner,threshold);
                    if (lo<=hi) {cross_=lo;break;}
                    owner=hi+1;
                }
                if (cross_<=r0_) break;
            }
        }

        // If R is r on lengths [a,b], then H's endpoint is r exactly on
        // owners [R[b+1]-b, r-a]. Its length is min(b,r-s) on that interval.
        for (int i = int(rsteps_.size()) - 1; i >= 0; --i) {
            const auto& z = rsteps_[i];
            const int start = i + 1 < int(rsteps_.size()) ? rsteps_[i + 1].end - z.hi : first_;
            const int lo = std::max(first_, start);
            const int hi = std::min({z.end - z.lo, cross_ - 1, r0_});
            if (lo <= hi) kruns_.push_back({lo, hi, z.end, -z.hi});
        }
        const int zero_lo = std::max(first_, rsteps_.front().end);
        const int zero_hi = std::min(r0_, cross_ - 1);
        if (zero_lo <= zero_hi) kruns_.push_back({zero_lo, zero_hi, -1, 0});

        for (std::size_t i=0;i<arecords_.size();++i) {
            const auto& a=arecords_[i];
            const int end=i+1<arecords_.size() ? arecords_[i+1].lo-1 : r0_;
            const int lo=std::max(cross_,a.lo), hi=std::min(end,r0_);
            if (lo>hi) continue;
            const int cap=a.cap();
            if (a.sliding()) {
                const int slide_hi=std::min(hi,a.end+1);
                if (lo<slide_hi) kruns_.push_back({lo,slide_hi,-a.end-2,a.length});
                else if (lo==slide_hi) kruns_.push_back({lo,lo,lo-1,a.at_length(lo)});
                const int flat_lo=std::max(lo,a.end+2);
                if (flat_lo<=hi) kruns_.push_back({flat_lo,hi,a.end,cap});
            } else kruns_.push_back({lo,hi,a.end,cap});
        }

        // Numerical deletion depends only on the payload endpoint. Ranges
        // with the baseline endpoint contribute exactly zero and are omitted.
        for (int i = int(rsteps_.size()) - 1; i >= 0; --i) {
            const auto& z = rsteps_[i];
            const int hi = std::min(z.hi, L0_ - 1);
            if (z.lo <= hi && z.end != r0_)
                qnumeric_.push_back({t_ - hi, t_ - z.lo, z.end});
        }
        qnumeric_.push_back({t_, t_, -1});
        for (const auto& z : kruns_) {
            const int min_owner=z.end<-1 ? z.lo : z.hi;
            const auto candidate = z.route(min_owner);
            if (candidate.length < minimum_route_.length ||
                (candidate.length == minimum_route_.length && candidate.end < minimum_route_.end)) {
                minimum_route_ = candidate;
                minimum_owner_ = min_owner;
            }
            if (z.end<-1) {
                if (z.lo==z.hi) knumeric_.push_back({z.lo,z.hi,z.lo-1});
                else kowner_.push_back({z.lo,z.hi});
            } else if (z.end != r0_) knumeric_.push_back({z.lo, z.hi, z.end});
        }
        minimum_ = minimum_route_.length;
    }

    DeleteRoute baseline() const { return {L0_, r0_}; }

    DeleteRoute qroute(int u) const {
        if (u > t_ || u <= t_ - L0_) return baseline();
        const int h = t_ - u;
        if (!h) return {0, -1};
        const auto it = std::lower_bound(rsteps_.begin(), rsteps_.end(), h,
            [](const RStep& z, int length) { return z.hi < length; });
        return {h, it->end};
    }
    // For nondecreasing Q owners only. Start each row with qcursor();
    // equal owners (different bits) are allowed. The cursor crosses each
    // R step at most once over the complete live-Q numeric loop.
    std::size_t qcursor() const { return rsteps_.size(); }
    DeleteRoute qroute_monotone(int u, std::size_t& cursor) const {
        if (u > t_ || u <= t_ - L0_) return baseline();
        const int h=t_-u;
        if (!h) return {0,-1};
        while (cursor>0 && rsteps_[cursor-1].hi>=h) --cursor;
        return {h,rsteps_[cursor].end};
    }
    DeleteRoute kroute(int s) const {
        if (!L0_ || s < first_ || s > r0_) return baseline();
        auto it = std::upper_bound(kruns_.begin(), kruns_.end(), s,
            [](int owner, const KRouteRun& z) { return owner < z.lo; });
        return (--it)->route(s);
    }

    // Inclusive owner interval.  D(s)=max(A(s),H(s)) is lexicographically
    // nonincreasing before its minimum and nondecreasing after it.  Clamping
    // a cached global minimizer into the interval therefore gives its exact
    // minimum, with at most one route lookup.  Owners outside the affected
    // baseline window have the baseline route.  An empty interval returns
    // that same baseline sentinel and should normally be skipped by callers.
    DeleteRoute min_k_delete_range(int lo, int hi) const {
        if (!L0_ || lo > hi) return baseline();
        if (lo <= minimum_owner_ && minimum_owner_ <= hi) return minimum_route_;
        return kroute(minimum_owner_ < lo ? lo : hi);
    }

    const std::vector<DeleteNumericRange>& qranges() const { return qnumeric_; }
    const std::vector<DeleteNumericRange>& kranges() const { return knumeric_; }
    const std::vector<DeleteOwnerRange>& kowner_ranges() const { return kowner_; }
    int min_k_delete_length() const { return minimum_; }
    std::size_t r_record_count() const { return rsteps_.size(); }
    std::size_t a_record_count() const { return arecords_.size(); }
    uint64_t a_expanded_record_count() const { return expanded_a_records_; }
    std::size_t k_route_run_count() const { return kruns_.size(); }
};

}  // namespace rosa_exact
