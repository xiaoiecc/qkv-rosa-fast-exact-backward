#pragma once

// Shared routes over the K suffix automaton already used for Q repairs.
// Requires Sam's read-only sequence_length(), prefix_state_at(), and
// ancestor_before() APIs in addition to the existing public SAM interface.
// All LCS/interval queries are causal: their K endpoint is below the Q time.
// No second suffix array, alphabet remapping, minimum segment tree, or
// occurrence-reporting index is constructed.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rosa_exact {

template<class Sam>
class SamRoutes {
public:
    struct Route { int length; int end; };

    explicit SamRoutes(Sam& sam, const uint64_t* q, const uint64_t* k)
        : sam_(sam), n_(sam.sequence_length()), q_(q), k_(k) {
        prefix_state_.resize(n_);
        for (int r = 0; r < n_; ++r) prefix_state_[r] = sam_.prefix_state_at(r);
        history_state_.assign(n_, 0);
        history_length_.assign(n_, 0);
        history_end_.assign(n_, -1);
        // Large auxiliary indices are materialized only by a query that
        // needs them; direct bounded queries keep no unused index alive.
    }

    // Call exactly once after each sam.advance(t), before deletion/repair
    // queries. Previous rows remain available for the left side of a repair.
    void observe() {
        const int t = sam_.current_time();
        if (t != time_ + 1 || t >= n_)
            throw std::invalid_argument("SamRoutes::observe requires sequential SAM advances");
        time_ = t;
        const auto base = sam_.baseline();
        history_state_[t] = sam_.baseline_state();
        history_length_[t] = base.length;
        history_end_[t] = base.end;
    }

    int size() const { return n_; }
    int materialized_index_flags() const { return int(!rmq_.empty()) | (int(!prefix_one_.empty())<<1); }

    int lcs(int t, int r) const {
        if (t < 0 || r < 0) return 0;
        assert(t <= time_ && r < t);
        // The same bounded comparison prefix is used on every query. Most
        // failed alignments stop before touching the less-local Euler RMQ.
        // Four successes defer to the unchanged O(1) complete LCS query.
        for (int matched = 0; matched < 4; ++matched) {
            if (r < matched || q_[t - matched] != k_[r - matched])
                return matched;
        }
        const int length = history_length_[t];
        if (length<=4 || r==3) return std::min(length,4);
        if (rmq_.empty()) build_lca();
        int l = first_[history_state_[t]], h = first_[prefix_state_[r]];
        if (l > h) std::swap(l, h);
        const int level = logs_[h - l + 1];
        const int common = std::min(rmq_[level][l], rmq_[level][h - (1 << level) + 1]);
        return std::min(length, common);
    }

    // Half-open Euler-occurrence interval for Q's exact suffix, rather than
    // a lexicographic interval. pred/succ retain their endpoint semantics.
    std::pair<int, int> interval(int t, int length) const {
        if (length <= 0) return {0, n_};
        assert(t >= 0 && t <= time_);
        if (length > history_length_[t]) return {0, 0};
        const int state = sam_.state_for_length(history_state_[t], length);
        return sam_.occurrence_range(state);
    }

    const std::vector<int>& endpoints() const { return sam_.occurrence_endpoints(); }

    int pred(int l, int h, int limit) const {
        if (h-l <= 16) {
            int result = -1;
            const auto& ends = endpoints();
            for (int i=l;i<h;++i) if (ends[i]<limit) result=std::max(result,ends[i]);
            return result;
        }
        if (prefix_one_.empty()) build_wavelet();
        const int count = less(l, h, limit);
        return count ? kth(l, h, count - 1) : -1;
    }

    int succ(int l, int h, int lower) const {
        if (h-l <= 16) {
            int result = n_;
            const auto& ends = endpoints();
            for (int i=l;i<h;++i) if (ends[i]>=lower) result=std::min(result,ends[i]);
            return result<n_ ? result : -1;
        }
        if (prefix_one_.empty()) build_wavelet();
        const int count = less(l, h, lower);
        return count < h - l ? kth(l, h, count) : -1;
    }

    int minimum(int l, int h) const {
        if (h-l <= 16) {
            int result = n_;
            const auto& ends = endpoints();
            for (int i=l;i<h;++i) result=std::min(result,ends[i]);
            return result<n_ ? result : -1;
        }
        if (prefix_one_.empty()) build_wavelet();
        return kth(l,h,0);
    }

    // R[length] can use the existing cached dynamic SAM maximum directly;
    // this avoids the interval+wavelet predecessor used by a generic index.
    int latest(int t, int length) {
        if (length <= 0) return -1;
        assert(t >= 0 && t <= time_ && length <= history_length_[t]);
        const int state = sam_.state_for_length(history_state_[t], length);
        if (t == sam_.current_time()) return sam_.latest_endpoint(state);
        const auto range = sam_.occurrence_range(state);
        return pred(range.first, range.second, t);
    }

    Route prefix_best(int t, int limit) const {
        if (t < 0 || limit <= 0) return {0, -1};
        assert(t <= time_ && limit <= t);
        const int base_length = history_length_[t];
        if (!base_length) return {0, -1};
        if (limit == t) return {base_length, history_end_[t]};
        // The same bounded leaf query handles every tiny endpoint prefix.
        // No ancestor/wavelet structure is needed for at most four points.
        if (limit<=4) {
            Route best{0,-1};
            for (int r=0;r<limit;++r) {
                const int length=lcs(t,r);
                if (length>best.length || (length && length==best.length))
                    best={length,r};
            }
            return best;
        }
        const int state = sam_.ancestor_before(history_state_[t], limit);
        if (!state) return {0, -1};
        const int length = std::min(base_length, sam_.state_length(state));
        const auto range = sam_.occurrence_range(state);
        return {length, pred(range.first, range.second, limit)};
    }

private:
    Sam& sam_;
    int n_, time_ = -1;
    const uint64_t* q_;
    const uint64_t* k_;
    std::vector<int> prefix_state_, history_state_, history_length_, history_end_;
    mutable std::vector<int> first_, logs_;
    mutable std::vector<std::vector<int>> rmq_;
    mutable int levels_ = 1;
    mutable std::vector<int> zeros_;
    // Direct integer ranks avoid generic-target __popcountdi2 library calls.
    // Only two O(n) value buffers are needed while building; no per-level
    // endpoint order is retained because the final core does not report here.
    mutable std::vector<std::vector<int>> prefix_one_;

    void build_lca() const {
        const int count = sam_.state_count();
        std::vector<int> head(count, -1), next(count, -1), state_length(count, 0);
        for (int state = 1; state < count; ++state) {
            const int parent = sam_.suffix_link(state);
            next[state] = head[parent];
            head[parent] = state;
            state_length[state] = sam_.state_length(state);
        }
        first_.assign(count, -1);
        std::vector<int> euler;
        euler.reserve(2 * count - 1);
        std::vector<int> stack;
        stack.reserve(count);
        stack.push_back(0);
        first_[0] = 0;
        euler.push_back(0);
        while (!stack.empty()) {
            const int state = stack.back();
            const int child = head[state];
            if (child < 0) {
                stack.pop_back();
                if (!stack.empty()) euler.push_back(state_length[stack.back()]);
            } else {
                head[state] = next[child];
                first_[child] = static_cast<int>(euler.size());
                euler.push_back(state_length[child]);
                stack.push_back(child);
            }
        }
        // State lengths strictly increase down the suffix-link tree.
        // Therefore the minimum length between the first Euler occurrences
        // is exactly length(LCA); no separate depth/state RMQ is necessary.
        const int entries = static_cast<int>(euler.size());
        logs_.assign(entries + 1, 0);
        for (int i = 2; i <= entries; ++i) logs_[i] = logs_[i / 2] + 1;
        rmq_.resize(logs_[entries] + 1);
        rmq_[0] = std::move(euler);
        for (int level = 1; level < static_cast<int>(rmq_.size()); ++level) {
            const int span = 1 << level, half = span / 2;
            rmq_[level].resize(entries - span + 1);
            for (int i = 0; i + span <= entries; ++i)
                rmq_[level][i] = std::min(rmq_[level - 1][i], rmq_[level - 1][i + half]);
        }
    }

    void build_wavelet() const {
        levels_ = 1;
        while ((uint64_t{1} << levels_) < static_cast<uint64_t>(n_)) ++levels_;
        zeros_.resize(levels_);
        prefix_one_.resize(levels_);
        std::vector<int> current = sam_.occurrence_endpoints(), next(n_);
        for (int level = 0; level < levels_; ++level) {
            const int shift = levels_ - 1 - level;
            auto& prefix = prefix_one_[level];
            prefix.resize(n_ + 1);
            prefix[0] = 0;
            for (int i = 0; i < n_; ++i)
                prefix[i + 1] = prefix[i] + ((current[i] >> shift) & 1);
            zeros_[level] = n_ - prefix[n_];
            int left = 0, right = zeros_[level];
            for (int value : current)
                next[((value >> shift) & 1) ? right++ : left++] = value;
            current.swap(next);
        }
    }

    int less(int l, int h, int upper) const {
        if (l >= h || upper <= 0) return 0;
        if (upper >= n_) return h - l;
        int result = 0;
        for (int level = 0; level < levels_; ++level) {
            const auto& prefix = prefix_one_[level];
            const int one_l = prefix[l], one_h = prefix[h];
            if ((upper >> (levels_ - 1 - level)) & 1) {
                result += (h - l) - (one_h - one_l);
                l = zeros_[level] + one_l;
                h = zeros_[level] + one_h;
            } else {
                l -= one_l;
                h -= one_h;
            }
        }
        return result;
    }

    int kth(int l, int h, int ordinal) const {
        int value = 0;
        for (int level = 0; level < levels_; ++level) {
            const auto& prefix = prefix_one_[level];
            const int one_l = prefix[l], one_h = prefix[h];
            const int zero_count = (h - l) - (one_h - one_l);
            if (ordinal < zero_count) {
                l -= one_l;
                h -= one_h;
            } else {
                ordinal -= zero_count;
                value |= 1 << (levels_ - 1 - level);
                l = zeros_[level] + one_l;
                h = zeros_[level] + one_h;
            }
        }
        return value;
    }
};

template<class Sam> SamRoutes(Sam&, const uint64_t*, const uint64_t*) -> SamRoutes<Sam>;

}  // namespace rosa_exact
