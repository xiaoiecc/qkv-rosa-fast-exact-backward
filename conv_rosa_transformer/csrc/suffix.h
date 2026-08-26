#pragma once
// Layer 1: suffix array + LCE + static rank/position range structures.
// Step-identical port of the Python reference (n_bit_qkv_rosa.py).
#include <vector>
#include <optional>
#include <utility>
#include <cstdint>

namespace rosa {

// prefix-doubling suffix array, stable sort semantics identical to Python list.sort
std::vector<int> suffix_array_int(const std::vector<int>& seq);

struct SuffixLCE {
    std::vector<int> seq, sa, rank, lcp, lg;
    std::vector<std::vector<int>> st;
    int n = 0;
    explicit SuffixLCE(const std::vector<int>& s);
    int rmq_lcp(int lo, int hi) const;              // min lcp[lo:hi), requires lo < hi
    int lcp_pos(int a, int b) const;
    std::pair<int, int> prefix_rank_interval(int pos, int length) const;
};

struct StaticMaxPByRank {
    int n = 0, size = 1;
    std::vector<int> tr;
    explicit StaticMaxPByRank(const std::vector<int>& values);
    std::optional<int> find_last(int lo, int hi, int x) const;   // rightmost rank r in [lo,hi) with value[r] > x
    std::optional<int> find_first(int lo, int hi, int x) const;  // leftmost rank r in [lo,hi) with value[r] > x
};

struct StaticRangeSuccessorP {
    int n = 0, size = 1;
    std::vector<std::vector<int>> rows;
    explicit StaticRangeSuccessorP(const std::vector<int>& values);  // negative values become empty leaves
    std::optional<int> range_successor(int lo, int hi, int x) const; // min p > x among leaves in [lo,hi)
};

struct RangePositionIndex {
    int n = 0, size = 1;
    std::vector<std::vector<int>> rows;              // merge-sort tree over SA rank
    explicit RangePositionIndex(const std::vector<std::optional<int>>& values);
    std::vector<int> nodes(int lo, int hi) const;    // canonical node decomposition
    std::optional<int> successor(int lo, int hi, int x) const;
    std::optional<int> predecessor(int lo, int hi, int x) const;
    std::optional<int> min_in(int lo, int hi, int p_lo, int p_hi) const;
    std::optional<int> max_in(int lo, int hi, int p_lo, int p_hi) const;
    std::vector<int> report(int lo, int hi, int p_lo, int p_hi) const;
};

// ROSA matching statistics (ell, route) for one Q/K symbol stream.
// Symbols are arbitrary int64 ids; remapped internally to >= 2 in first-appearance order.
std::pair<std::vector<int>, std::vector<int>> matching_stats(
    const std::vector<int64_t>& q, const std::vector<int64_t>& k);

}  // namespace rosa
