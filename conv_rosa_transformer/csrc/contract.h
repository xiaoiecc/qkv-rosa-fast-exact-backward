#pragma once
// Layer 5b: pure numerical contraction + top-level stream credits.
// Float semantics: elementwise ops manual in tensor dtype (IEEE-identical to ATen);
// reductions (.sum/dot) go through ATen to reproduce torch's exact summation order.
// Pure-Python-float arithmetic of the reference is double -- kept double here.
#include <torch/extension.h>
#include "qrepair.h"

namespace rosa {

// numerical contraction over explicit IR fields
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> contract_fields(
    const std::vector<std::vector<std::vector<RepairTrackTerm>>>& q_terms,
    const std::vector<std::vector<std::vector<RepairTrackTerm>>>& k_terms,
    const std::vector<std::vector<AffineDeleteRun>>& q_delete_runs_by_t,
    const std::vector<std::vector<AffineDeleteRun>>& k_delete_runs_by_t,
    const std::vector<ZeroBaselineSurface>& k_zero_surfaces,
    const std::vector<int>& q, const std::vector<int>& k,
    const std::vector<int>& ell, const std::vector<int>& route,
    torch::Tensor v_bits, torch::Tensor grad_y,
    torch::Tensor emb0_g, torch::Tensor emb1_g, int D);


}  // namespace rosa
