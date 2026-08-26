#pragma once
// Layer 3: bidirectional position-restricted suffix index + causal cut-envelope primitives.
// Step-identical port of BiPositionSuffixIndex / CausalCutSuffixIndex.
// NOTE: stats counters of the Python version are instrumentation-only and intentionally dropped.
#include <vector>
#include <map>
#include <unordered_map>
#include <optional>
#include <tuple>
#include <cstdint>
#include "suffix.h"
#include "ortho.h"

namespace rosa {

struct OneBitOccurrence {
    int start, end, mismatch_owner, bit;
    bool operator<(const OneBitOccurrence& o) const {
        return std::tie(start, end, mismatch_owner, bit) <
               std::tie(o.start, o.end, o.mismatch_owner, o.bit);
    }
};

class CausalCutSuffixIndex {
public:
    int T;
    std::vector<int> q, k;
    int q_off, k_off, rq_off, rk_off;
    SuffixLCE fwd, rev;
    RangePositionIndex fpos, rpos;
    std::unordered_map<int, std::vector<int>> positions_by_symbol;

    CausalCutSuffixIndex(const std::vector<int64_t>& q, const std::vector<int64_t>& k);

    // ----- exact string primitives -----
    int lcp_after_center(int u, int p) const;
    int lcs_before_center(int u, int p) const;
    std::pair<int, int> q_pattern_interval(int t, int L) const;
    std::pair<int, int> q_reversed_prefix_interval(int t, int L) const;

    // ----- direct-Q bidirectional rectangle oracle -----
    std::optional<std::pair<int, int>> direct_q_best_center(int target, int u, int depth);

    // ----- exact causal forward -----
    std::pair<std::vector<int>, std::vector<int>> matching_stats() const {
        return rosa::matching_stats(
            std::vector<int64_t>(q.begin(), q.end()),
            std::vector<int64_t>(k.begin(), k.end()));
    }

    // ----- position-restricted exact one-bit occurrences -----
    std::vector<OneBitOccurrence> one_bit_occurrences_filtered(int t, int L, int S0, int S1, int D);

    // ----- causal cut-envelope primitives -----
    int lcs_end(int t, int e) const;
    std::optional<int> latest_endpoint_for_suffix(int t, int L) const;
    std::optional<int> next_endpoint_at_least(int t, int L, int after_e) const;
    int F_reversed(int t, int p) const;
    std::optional<int> min_p_with_F_at_least(int t, int L, int p_lo, int p_hi) const;

private:
    std::vector<int> report_q_substring_starts(int q_start, int length, int k_start_lo, int k_start_hi) const;
    const SymbolOrthogonalOracle& q_orth_for_symbol(int target);

    std::map<std::tuple<int,int,int,int,int>, std::vector<OneBitOccurrence>> onebit_cache_;
    std::unordered_map<int, SymbolOrthogonalOracle> q_orth_;
};

}  // namespace rosa
