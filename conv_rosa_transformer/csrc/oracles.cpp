#include "oracles.h"
#include <algorithm>
#include <stdexcept>

namespace rosa {

std::vector<AffineDeleteRun> merge_affine_runs(const std::vector<AffineDeleteRun>& xs) {
    std::vector<AffineDeleteRun> sorted = xs;
    std::stable_sort(sorted.begin(), sorted.end(), [](const AffineDeleteRun& a, const AffineDeleteRun& b) {
        return std::tie(a.s_lo, a.s_hi) < std::tie(b.s_lo, b.s_hi);
    });
    std::vector<AffineDeleteRun> out;
    for (const auto& z : sorted) {
        if (z.s_lo > z.s_hi) continue;
        if (!out.empty()) {
            const AffineDeleteRun& p = out.back();
            if (p.output_t == z.output_t && p.s_hi + 1 == z.s_lo &&
                p.len_a == z.len_a && p.len_b == z.len_b &&
                p.end_a == z.end_a && p.end_b == z.end_b) {
                out.back().s_hi = z.s_hi;
                continue;
            }
        }
        out.push_back(z);
    }
    return out;
}

std::vector<AffineDeleteRun> compress_singleton_affine(const std::vector<AffineDeleteRun>& xs) {
    std::vector<AffineDeleteRun> raw = merge_affine_runs(xs);
    std::vector<AffineDeleteRun> out;
    size_t i = 0;
    while (i < raw.size()) {
        const AffineDeleteRun& z = raw[i];
        if (z.s_lo != z.s_hi || i + 1 >= raw.size() ||
            raw[i + 1].s_lo != raw[i + 1].s_hi || raw[i + 1].s_lo != z.s_hi + 1) {
            out.push_back(z); ++i; continue;
        }
        const AffineDeleteRun& z1 = raw[i + 1];
        int s0 = z.s_lo, s1 = z1.s_lo;
        Route r0 = z.route(s0), r1 = z1.route(s1);
        int la = r1.first - r0.first, ea = r1.second - r0.second;
        int lb = r0.first - la * s0, eb = r0.second - ea * s0;
        size_t j = i + 2;
        int last = s1;
        while (j < raw.size()) {
            const AffineDeleteRun& zj = raw[j];
            if (zj.s_lo != zj.s_hi || zj.s_lo != last + 1) break;
            int s = zj.s_lo;
            Route rs = zj.route(s);
            if (rs.first != la * s + lb || rs.second != ea * s + eb) break;
            last = s; ++j;
        }
        out.push_back({z.output_t, s0, last, la, lb, ea, eb});
        i = j;
    }
    return merge_affine_runs(out);
}

const std::vector<AffineDeleteRun>& MostRecentSuffixMatchOracle::compile(int t) {
    auto it = cache_.find(t);
    if (it != cache_.end()) return it->second;
    if (t <= 0) {
        auto [eit, _] = cache_.emplace(t, std::vector<AffineDeleteRun>{});
        starts_[t] = {};
        return eit->second;
    }
    std::vector<AffineDeleteRun> raw;
    auto u = index_->next_endpoint_at_least(t, 1, -1);
    if (!u) {
        raw = {{t, 0, t - 1, 0, 0, 0, -1}};
    } else {
        raw.push_back({t, 0, std::min(*u, t - 1), 0, 0, 0, -1});
        int cur_e = *u, cur_L = index_->lcs_end(t, *u), cur_s = *u + 1;
        while (cur_s <= t - 1) {
            auto nxt = index_->next_endpoint_at_least(t, cur_L, cur_e);
            int hi = nxt ? std::min(*nxt, t - 1) : t - 1;
            raw.push_back({t, cur_s, hi, 0, cur_L, 0, cur_e});
            if (!nxt || *nxt + 1 > t - 1) break;
            cur_e = *nxt;
            cur_L = index_->lcs_end(t, cur_e);
            cur_s = cur_e + 1;
        }
    }
    std::vector<AffineDeleteRun> out = compress_singleton_affine(raw);
    std::vector<int> st;
    for (auto& z : out) st.push_back(z.s_lo);
    starts_[t] = st;
    auto [eit, _] = cache_.emplace(t, std::move(out));
    return eit->second;
}

Route MostRecentSuffixMatchOracle::route(int t, int s) {
    const auto& xs = compile(t);
    if (xs.empty()) return {0, -1};
    const auto& starts = starts_[t];
    auto it = std::upper_bound(starts.begin(), starts.end(), s);
    int i = (int)(it - starts.begin()) - 1;
    if (i >= 0 && xs[i].s_lo <= s && s <= xs[i].s_hi) return xs[i].route(s);
    return {0, -1};
}

const std::vector<AffineDeleteRun>& TruncatedRightMatchOracle::compile(int t) {
    auto it = cache_.find(t);
    if (it != cache_.end()) return it->second;
    if (t <= 0) {
        auto [eit, _] = cache_.emplace(t, std::vector<AffineDeleteRun>{});
        return eit->second;
    }
    int T = index_->T, x = T - 1 - t;
    struct Bout { int b_lo, b_hi, kind, param, endpoint; };  // kind: 0=unmatched 1=ramp 2=plateau
    std::vector<Bout> bouts;
    auto p = index_->min_p_with_F_at_least(t, 1, x + 1, T - 1);
    if (!p || *p + 1 > T - 1) {
        std::vector<AffineDeleteRun> out = {{t, 0, t - 1, 0, 0, 0, -1}};
        auto [eit, _] = cache_.emplace(t, std::move(out));
        return eit->second;
    }
    if (x + 1 <= *p) bouts.push_back({x + 1, *p, 0, 0, -1});
    int bcur = *p + 1;
    while (bcur <= T - 1) {
        int F = index_->F_reversed(t, *p);
        if (F <= 0) throw std::runtime_error("H winner with zero F");
        int sat = std::min(T - 1, *p + F);
        if (bcur <= sat) {
            int e = T - 1 - *p;
            bouts.push_back({bcur, sat, 1, 0, e});
        }
        if (sat >= T - 1) break;
        auto qn = index_->min_p_with_F_at_least(t, F + 1, *p + 1, T - 1);
        int e = T - 1 - *p;
        if (!qn) {
            if (sat + 1 <= T - 1) bouts.push_back({sat + 1, T - 1, 2, F, e});
            break;
        }
        int bnext = *qn + F + 1;
        if (sat + 1 <= std::min(T - 1, bnext - 1))
            bouts.push_back({sat + 1, std::min(T - 1, bnext - 1), 2, F, e});
        if (bnext > T - 1) break;
        p = qn;
        bcur = bnext;
    }
    std::vector<AffineDeleteRun> out;
    for (auto& [blo, bhi, kind, param, e] : bouts) {
        int slo = std::max(0, T - 1 - bhi), shi = std::min(t - 1, T - 1 - blo);
        if (slo > shi) continue;
        if (kind == 0) out.push_back({t, slo, shi, 0, 0, 0, -1});
        else if (kind == 1) out.push_back({t, slo, shi, -1, e, 0, e});
        else out.push_back({t, slo, shi, 0, param, 0, e});
    }
    out = merge_affine_runs(out);
    int cur = 0;
    for (auto& z : out) {
        if (z.s_lo != cur) throw std::runtime_error("H coverage gap");
        cur = z.s_hi + 1;
    }
    if (cur != t) throw std::runtime_error("H coverage tail");
    auto [eit, _] = cache_.emplace(t, std::move(out));
    return eit->second;
}

std::vector<std::vector<LatestOccurrenceHead>> compile_q_latest_heads(
    CausalCutSuffixIndex& index, const std::vector<int>& ell) {
    std::vector<std::vector<LatestOccurrenceHead>> out;
    for (int t = 0; t < (int)ell.size(); ++t) {
        int maxL = std::max(0, ell[t] - 1);
        std::vector<LatestOccurrenceHead> xs;
        int L = 1;
        while (L <= maxL) {
            auto e = index.latest_endpoint_for_suffix(t, L);
            if (!e) throw std::runtime_error("matched suffix lost latest occurrence");
            int H = std::min(maxL, index.lcs_end(t, *e));
            if (H < L) throw std::runtime_error("latest head lifetime");
            xs.push_back({t, L, H, *e});
            L = H + 1;
        }
        out.push_back(std::move(xs));
    }
    return out;
}

std::vector<std::vector<AffineDeleteRun>> build_q_delete_from_latest_heads(
    const std::vector<int>& ell, const std::vector<std::vector<LatestOccurrenceHead>>& heads) {
    std::vector<std::vector<AffineDeleteRun>> out(ell.size());
    for (int t = 0; t < (int)ell.size(); ++t) {
        int L = ell[t];
        if (L <= 0) continue;
        std::vector<AffineDeleteRun> xs = {{t, t, t, 0, 0, 0, -1}};
        for (const auto& h : heads[t])
            xs.push_back({t, t - h.L_hi, t - h.L_lo, -1, t, 0, h.endpoint});
        out[t] = merge_affine_runs(xs);
    }
    return out;
}

}  // namespace rosa
