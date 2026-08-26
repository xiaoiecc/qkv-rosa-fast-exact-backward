#pragma once
// K-delete threshold surfaces and shared one-bit repair bridges.
#include <vector>
#include <string>
#include <tuple>
#include <utility>
#include "oracles.h"
#include "qrepair.h"

namespace rosa {

// owner-axis repair threshold q(s): a repaired route needs length >= threshold_length(s)
struct KRepairThresholdRun {
    int output_t, s_lo, s_hi, len_a, len_b;
    bool strict;   // false: >= baseline wins; true: must be > baseline
    int baseline_length(int s) const { return len_a * s + len_b; }
    int threshold_length(int s) const { return std::max(1, baseline_length(s) + (strict ? 1 : 0)); }
};

std::vector<KRepairThresholdRun> merge_threshold_runs(const std::vector<KRepairThresholdRun>& xs);

// exact max(A,H) over an owner interval, keeping repair-threshold polarity of the winner
std::pair<std::vector<AffineDeleteRun>, std::vector<KRepairThresholdRun>> merge_A_H_surface_runs(
    const std::vector<AffineDeleteRun>& a_runs, const std::vector<AffineDeleteRun>& h_runs,
    int t, int s_lo, int s_hi);

// exact post-delete K baseline route + minimal repair-requirement surfaces
class KDeleteCutOracle {
public:
    KDeleteCutOracle(CausalCutSuffixIndex* index, const std::vector<int>& ell,
                     const std::vector<int>& route);
    Route route(int t, int s) const;
    int route0_at(int t) const { return route0_[t]; }
    const std::vector<KRepairThresholdRun>& repair_regions(int t) const { return repair_runs[t]; }

    std::vector<std::vector<AffineDeleteRun>> runs;
    std::vector<std::vector<int>> starts;
    std::vector<std::vector<KRepairThresholdRun>> repair_runs;
private:
    std::vector<int> ell_, route0_;
};

// one-bit Q/K center pair extended by LCEs both ways: a candidate repair "bridge"
struct SharedRepairBridge {
    int q_pos, k_pos, bit, left, right;
    int end_t() const { return q_pos + right; }
    std::pair<int,int> q_priority() const { return {left, k_pos}; }
    std::pair<int,int> k_priority() const { return {left + 1 - q_pos, k_pos - q_pos}; }
    int shift() const { return k_pos - q_pos + 1; }
    Route route_at(int t) const { int d = t - q_pos; return {left + 1 + d, k_pos + d}; }
};

int64_t count_causal_onebit_pairs(const std::vector<int>& q, const std::vector<int>& k, int D);

using BridgeGrid = std::vector<std::vector<std::vector<SharedRepairBridge>>>;  // [T][D][bridges]
using TermGrid = std::vector<std::vector<std::vector<RepairTrackTerm>>>;       // [T][D][terms]

std::pair<BridgeGrid, BridgeGrid> build_shared_bridges_sparse(
    const std::vector<int>& q, const std::vector<int>& k, int D, CausalCutSuffixIndex& index);
std::pair<BridgeGrid, BridgeGrid> build_shared_bridges_diagonal(
    const std::vector<int>& q, const std::vector<int>& k, int D);

TermGrid q_shared_terms(const BridgeGrid& qg, int D);
TermGrid k_shared_terms(const BridgeGrid& kg, int D, const KDeleteCutOracle& kdel);

}  // namespace rosa
