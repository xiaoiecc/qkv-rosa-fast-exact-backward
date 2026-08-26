#pragma once
// Pipeline entry: adaptive exact repair compiler + stream-level VJP credits.
#include <vector>
#include <string>
#include <tuple>
#include <torch/extension.h>

namespace rosa {

// diagnostic: which adaptive repair backend a stream would use
std::string repair_backend(const std::vector<int64_t>& q, const std::vector<int64_t>& k, int D);

// backward for one Q/K/V bit stream: exact one-bit counterfactual VJP credits
// returns (q_credits, k_credits, v_credits, match_lengths, route_endpoints)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<int64_t>, std::vector<int64_t>>
exact_stream_bit_credits(
    const std::vector<int64_t>& q, const std::vector<int64_t>& k,
    torch::Tensor v_bits, torch::Tensor grad_y,
    torch::Tensor emb0_g, torch::Tensor emb1_g, int64_t n_bits);

}  // namespace rosa
