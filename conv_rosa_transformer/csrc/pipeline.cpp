#include "pipeline.h"
#include "certificates.h"
#include "krepair.h"
#include "contract.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rosa {

std::string select_repair_backend(int T, int64_t pair_count, int D, int max_match_len) {
    if (pair_count == 0) return "none";
    if (max_match_len == 0) return "surface_run_certified";
    double lg = std::log2((double)std::max(2, T));
    if (T >= 96 && max_match_len >= std::max(32, T / 4)) return "surface_run_certified";
    double sparse_factor = (D == 1) ? 6.0 : 8.0;
    int64_t sparse_limit = std::max<int64_t>(
        1024, (int64_t)(sparse_factor * std::max(1, T) * lg));
    if (pair_count <= sparse_limit) return "shared_sparse";
    int surface_cutoff = (D <= 2) ? 352 : 448;
    if (T < surface_cutoff) return "shared_diagonal";
    return "surface_run_certified";
}

struct RepairIR {
    TermGrid q_terms, k_terms;
    std::vector<std::vector<AffineDeleteRun>> qdel, kdel;
    std::vector<ZeroBaselineSurface> zero;
    std::vector<int> ell, route;
    std::string backend;
};

static RepairIR compile_repair_ir(const std::vector<int64_t>& q64, const std::vector<int64_t>& k64, int D) {
    CausalCutSuffixIndex index(q64, k64);
    auto [ell, route] = index.matching_stats();
    std::vector<int> qi(q64.begin(), q64.end()), ki(k64.begin(), k64.end());
    RepairIR out;
    out.ell = ell; out.route = route;
    auto qheads = compile_q_latest_heads(index, ell);
    out.qdel = build_q_delete_from_latest_heads(ell, qheads);
    KDeleteCutOracle kdel(&index, ell, route);
    out.kdel = kdel.runs;
    int64_t pair_count = count_causal_onebit_pairs(qi, ki, D);
    int max_match = 0;
    for (int e : ell) max_match = std::max(max_match, e);
    out.backend = select_repair_backend((int)qi.size(), pair_count, D, max_match);
    if (out.backend == "none") {
        out.q_terms.assign(qi.size(), std::vector<std::vector<RepairTrackTerm>>(D));
        out.k_terms.assign(qi.size(), std::vector<std::vector<RepairTrackTerm>>(D));
    } else if (out.backend == "shared_sparse" || out.backend == "shared_diagonal") {
        BridgeGrid qg, kg;
        if (out.backend == "shared_sparse") {
            std::tie(qg, kg) = build_shared_bridges_sparse(qi, ki, D, index);
        } else {
            std::tie(qg, kg) = build_shared_bridges_diagonal(qi, ki, D);
        }
        out.q_terms = q_shared_terms(qg, D);
        out.k_terms = k_shared_terms(kg, D, kdel);
    } else if (out.backend == "surface_run_certified") {
        out.q_terms = q_repair_terms_suffix_range(qi, D, index);
        out.k_terms = compile_k_surface_conditioned(D, index, kdel, ell);
        out.zero = zero_baseline_surfaces(qi, ki, ell, D);
    } else {
        throw std::runtime_error("unknown repair backend");
    }
    return out;
}

std::string repair_backend(const std::vector<int64_t>& q, const std::vector<int64_t>& k, int D) {
    return compile_repair_ir(q, k, D).backend;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<int64_t>, std::vector<int64_t>>
exact_stream_bit_credits(
    const std::vector<int64_t>& q, const std::vector<int64_t>& k,
    torch::Tensor v_bits, torch::Tensor grad_y,
    torch::Tensor emb0_g, torch::Tensor emb1_g, int64_t n_bits) {
    int D = (int)n_bits;
    v_bits = v_bits.contiguous();
    grad_y = grad_y.contiguous();
    emb0_g = emb0_g.contiguous();
    emb1_g = emb1_g.contiguous();
    RepairIR ir = compile_repair_ir(q, k, D);
    std::vector<int> qi(q.begin(), q.end()), ki(k.begin(), k.end());
    auto [qcred, kcred, vcred] = contract_fields(
        ir.q_terms, ir.k_terms, ir.qdel, ir.kdel, ir.zero,
        qi, ki, ir.ell, ir.route, v_bits, grad_y, emb0_g, emb1_g, D);
    std::vector<int64_t> ell64(ir.ell.begin(), ir.ell.end()), rt64(ir.route.begin(), ir.route.end());
    return {qcred, kcred, vcred, ell64, rt64};
}


}  // namespace rosa
