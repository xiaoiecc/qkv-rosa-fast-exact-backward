#include "krepair.h"
#include "certificates.h"
#include <algorithm>
#include <map>
#include <optional>
#include <stdexcept>

namespace rosa {

static int onebit_index(int x, int D) {
    if (x <= 0 || (x & (x - 1))) return -1;
    int j = 0, xx = x;
    while (xx >>= 1) ++j;
    return j < D ? j : -1;
}

std::vector<std::vector<std::vector<RepairTrackTerm>>> compile_k_surface_conditioned(
    int D, CausalCutSuffixIndex& index, const KDeleteCutOracle& kdel,
    const std::vector<int>& ell) {
    int T = index.T;
    using TermMap = std::map<int, Route>;
    std::vector<std::vector<TermMap>> best(T, std::vector<TermMap>(D));
    std::optional<KRunRepairCertificateIndex> run_index;

    auto record_route = [&](int t, int s, int j, Route cand) {
        auto it = best[s][j].find(t);
        if (it == best[s][j].end() || cand > it->second) best[s][j][t] = cand;
    };
    auto record_occ = [&](int t, int Lq, const OneBitOccurrence& o) {
        int q0 = t - Lq + 1;
        record_route(t, o.mismatch_owner, o.bit,
                     Route{Lq + index.lcs_before_center(q0, o.start), o.end});
    };
    auto record_shadow_event = [&](int t, int s, int j, int e, int d) {
        int u = t - d;
        if (!(0 <= u && u < T && 0 <= s && s < T && 0 <= e && e < t))
            throw std::runtime_error("shadow repair coordinate");
        int actual_j = onebit_index(index.q[u] ^ index.k[s], D);
        if (actual_j != j) throw std::runtime_error("shadow repair bit");
        int left = index.lcs_before_center(u, s);
        Route cand{d + 1 + left, e};
        if (cand <= kdel.route(t, s)) throw std::runtime_error("shadow repair failed threshold");
        record_route(t, s, j, cand);
    };

    for (int t = 0; t < T; ++t) {
        if (ell[t] == 0) continue;
        int r0 = kdel.route0_at(t);
        for (const auto& reg : kdel.repair_regions(t)) {
            if (reg.len_a == 0) {
                int Lq = reg.threshold_length(reg.s_lo);
                if (!(1 <= Lq && Lq <= t + 1)) continue;
                for (const auto& o : index.one_bit_occurrences_filtered(t, Lq, reg.s_lo, reg.s_hi, D))
                    record_occ(t, Lq, o);
                continue;
            }
            // proven H-ramp family: strict threshold L(s)=M-s
            if (reg.strict && reg.len_a == -1) {
                int M = reg.len_b;
                if (M > r0) {
                    int d = index.lcs_end(t, M);
                    int s = M - d;
                    if (reg.s_lo <= s && s <= reg.s_hi) {
                        int u = t - d;
                        if (0 <= s && s < T && 0 <= u && u < T) {
                            int j = onebit_index(index.q[u] ^ index.k[s], D);
                            if (j >= 0) record_shadow_event(t, s, j, M, d);
                        }
                    }
                    continue;
                }
                if (M == r0) {
                    if (!run_index) run_index.emplace(index, D);
                    for (const auto& c : run_index->query(M, reg.s_lo, reg.s_hi)) {
                        int e = M - c.period;
                        int d = e - c.owner;
                        if (d < 0) throw std::runtime_error("run certificate negative depth");
                        record_shadow_event(t, c.owner, c.bit, e, d);
                    }
                    continue;
                }
            }
            for (int owner = reg.s_lo; owner <= reg.s_hi; ++owner) {
                int Lq = reg.threshold_length(owner);
                if (!(1 <= Lq && Lq <= t + 1)) continue;
                for (const auto& o : index.one_bit_occurrences_filtered(t, Lq, owner, owner, D))
                    record_occ(t, Lq, o);
            }
        }
    }
    std::vector<std::vector<std::vector<RepairTrackTerm>>> out(
        T, std::vector<std::vector<RepairTrackTerm>>(D));
    for (int s = 0; s < T; ++s) {
        for (int j = 0; j < D; ++j) {
            const TermMap& mp = best[s][j];
            if (mp.empty()) continue;
            auto it = mp.begin();
            int lo = it->first, hi = it->first, shift = it->second.second - it->first + 1;
            for (++it; it != mp.end(); ++it) {
                int t = it->first;
                int sh = it->second.second - t + 1;
                if (t == hi + 1 && sh == shift) { hi = t; }
                else {
                    out[s][j].push_back({shift, lo, hi});
                    lo = hi = t; shift = sh;
                }
            }
            out[s][j].push_back({shift, lo, hi});
        }
    }
    return out;
}


}  // namespace rosa
