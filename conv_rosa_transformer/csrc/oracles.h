#pragma once
// Layer 4: affine delete-run machinery + A/H cut oracles + KDeleteCutOracle + head compilers.
// Step-identical port; stats counters dropped (instrumentation only).
#include <vector>
#include <optional>
#include <utility>
#include "index.h"

namespace rosa {

using Route = std::pair<int, int>;   // (match length, K endpoint); endpoint -1 = unmatched

struct AffineDeleteRun {
    int output_t, s_lo, s_hi, len_a, len_b, end_a, end_b;
    Route route(int s) const { return {len_a * s + len_b, end_a * s + end_b}; }
};

struct LatestOccurrenceHead { int output_t, L_lo, L_hi, endpoint; };

std::vector<AffineDeleteRun> merge_affine_runs(const std::vector<AffineDeleteRun>& xs);
std::vector<AffineDeleteRun> compress_singleton_affine(const std::vector<AffineDeleteRun>& xs);

class MostRecentSuffixMatchOracle {   // A_t(s): best route with K endpoint strictly < owner s
public:
    MostRecentSuffixMatchOracle(CausalCutSuffixIndex* index) : index_(index) {}
    const std::vector<AffineDeleteRun>& compile(int t);
    Route route(int t, int s);
private:
    CausalCutSuffixIndex* index_;
    std::unordered_map<int, std::vector<AffineDeleteRun>> cache_;
    std::unordered_map<int, std::vector<int>> starts_;
};

class TruncatedRightMatchOracle {     // H_t(s): best occurrence wholly right of owner s
public:
    TruncatedRightMatchOracle(CausalCutSuffixIndex* index) : index_(index) {}
    const std::vector<AffineDeleteRun>& compile(int t);
private:
    CausalCutSuffixIndex* index_;
    std::unordered_map<int, std::vector<AffineDeleteRun>> cache_;
};

std::vector<std::vector<LatestOccurrenceHead>> compile_q_latest_heads(
    CausalCutSuffixIndex& index, const std::vector<int>& ell);
std::vector<std::vector<AffineDeleteRun>> build_q_delete_from_latest_heads(
    const std::vector<int>& ell, const std::vector<std::vector<LatestOccurrenceHead>>& heads);

}  // namespace rosa
