#pragma once
// Repair-track terms and direct-Q / zero-baseline compilers.
#include <vector>
#include "oracles.h"

namespace rosa {

struct RepairTrackTerm {
    int shift, lo, hi;   // candidate payload on inclusive [lo,hi] is V[t+shift]
};

struct ZeroBaselineSurface { int bit, k_symbol, target_symbol; };

// direct-Q repair terms over one-bit symbol pairs (indexed, output-sensitive)
std::vector<std::vector<std::vector<RepairTrackTerm>>> q_repair_terms_suffix_range(
    const std::vector<int>& q, int D, CausalCutSuffixIndex& index);

// zero-baseline K surfaces for entirely-unmatched output positions
std::vector<ZeroBaselineSurface> zero_baseline_surfaces(
    const std::vector<int>& q, const std::vector<int>& k,
    const std::vector<int>& ell, int D);

}  // namespace rosa
