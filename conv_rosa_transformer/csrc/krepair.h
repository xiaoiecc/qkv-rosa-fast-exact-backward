#pragma once
// K-side repair compiler: one threshold per delete surface, with Equality-Shadow
// H-ramp handling (saturation-boundary repair / run certificates / exact fallback).
#include <vector>
#include "bridges.h"
#include "qrepair.h"

namespace rosa {

std::vector<std::vector<std::vector<RepairTrackTerm>>> compile_k_surface_conditioned(
    int D, CausalCutSuffixIndex& index, const KDeleteCutOracle& kdel,
    const std::vector<int>& ell);

}  // namespace rosa
