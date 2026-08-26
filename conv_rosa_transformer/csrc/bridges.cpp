#include "bridges.h"
#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <unordered_map>

namespace rosa {

std::vector<KRepairThresholdRun> merge_threshold_runs(const std::vector<KRepairThresholdRun>& xs) {
    std::vector<KRepairThresholdRun> sorted = xs;
    std::stable_sort(sorted.begin(), sorted.end(), [](const KRepairThresholdRun& a, const KRepairThresholdRun& b) {
        return std::tie(a.s_lo, a.s_hi) < std::tie(b.s_lo, b.s_hi);
    });
    std::vector<KRepairThresholdRun> out;
    for (const auto& z : sorted) {
        if (z.s_lo > z.s_hi) continue;
        if (!out.empty()) {
            const KRepairThresholdRun& p = out.back();
            if (p.output_t == z.output_t && p.s_hi + 1 == z.s_lo &&
                p.len_a == z.len_a && p.len_b == z.len_b && p.strict == z.strict) {
                out.back().s_hi = z.s_hi;
                continue;
            }
        }
        out.push_back(z);
    }
    return out;
}

std::pair<std::vector<AffineDeleteRun>, std::vector<KRepairThresholdRun>> merge_A_H_surface_runs(
    const std::vector<AffineDeleteRun>& a_runs, const std::vector<AffineDeleteRun>& h_runs,
    int t, int s_lo, int s_hi) {
    std::vector<AffineDeleteRun> A, H;
    for (const auto& x : a_runs) {
        int a = std::max(s_lo, x.s_lo), b = std::min(s_hi, x.s_hi);
        if (a <= b) A.push_back({x.output_t, a, b, x.len_a, x.len_b, x.end_a, x.end_b});
    }
    for (const auto& x : h_runs) {
        int a = std::max(s_lo, x.s_lo), b = std::min(s_hi, x.s_hi);
        if (a <= b) H.push_back({x.output_t, a, b, x.len_a, x.len_b, x.end_a, x.end_b});
    }
    std::vector<AffineDeleteRun> routes;
    std::vector<KRepairThresholdRun> reqs;
    auto emit = [&](const AffineDeleteRun& z, int lo, int hi, bool strict) {
        routes.push_back({t, lo, hi, z.len_a, z.len_b, z.end_a, z.end_b});
        reqs.push_back({t, lo, hi, z.len_a, z.len_b, strict});
    };
    size_t i = 0, j = 0;
    while (i < A.size() && j < H.size()) {
        const AffineDeleteRun& za = A[i];
        const AffineDeleteRun& zh = H[j];
        int lo = std::max(za.s_lo, zh.s_lo), hi = std::min(za.s_hi, zh.s_hi);
        if (lo <= hi) {
            if (za.route(lo) > zh.route(lo)) {
                emit(za, lo, hi, false);
            } else if (za.route(hi) <= zh.route(hi)) {
                emit(zh, lo, hi, true);
            } else {
                int l = lo, r = hi;
                while (l < r) {
                    int m = (l + r) / 2;
                    if (za.route(m) > zh.route(m)) r = m; else l = m + 1;
                }
                int c = l;
                if (lo <= c - 1) emit(zh, lo, c - 1, true);
                emit(za, c, hi, false);
            }
        }
        int ae = za.s_hi, he = zh.s_hi;
        if (ae <= he) ++i;
        if (he <= ae) ++j;
    }
    routes = merge_affine_runs(routes);
    reqs = merge_threshold_runs(reqs);
    {
        int cur = s_lo;
        for (auto& z : routes) {
            if (z.s_lo != cur) throw std::runtime_error("A/H route gap");
            cur = z.s_hi + 1;
        }
        if (cur != s_hi + 1) throw std::runtime_error("A/H route tail");
        cur = s_lo;
        for (auto& z : reqs) {
            if (z.s_lo != cur) throw std::runtime_error("A/H threshold gap");
            cur = z.s_hi + 1;
        }
        if (cur != s_hi + 1) throw std::runtime_error("A/H threshold tail");
    }
    return {routes, reqs};
}

KDeleteCutOracle::KDeleteCutOracle(CausalCutSuffixIndex* index, const std::vector<int>& ell,
                                             const std::vector<int>& route)
    : ell_(ell), route0_(route) {
    MostRecentSuffixMatchOracle A(index);
    TruncatedRightMatchOracle H(index);
    int T = (int)ell.size();
    runs.assign(T, {});
    starts.assign(T, {});
    repair_runs.assign(T, {});
    for (int t = 0; t < T; ++t) {
        int L0 = ell[t], r0 = route[t];
        if (L0 <= 0 || r0 < 0) continue;
        int wlo = r0 - L0 + 1, whi = r0;
        auto [routes, req_inside] = merge_A_H_surface_runs(A.compile(t), H.compile(t), t, wlo, whi);
        runs[t] = routes;
        std::vector<int> st;
        for (auto& z : routes) st.push_back(z.s_lo);
        starts[t] = st;
        std::vector<KRepairThresholdRun> req_full;
        if (wlo > 0) req_full.push_back({t, 0, wlo - 1, 0, L0, true});
        req_full.insert(req_full.end(), req_inside.begin(), req_inside.end());
        if (whi + 1 <= t - 1) req_full.push_back({t, whi + 1, t - 1, 0, L0, false});
        repair_runs[t] = merge_threshold_runs(req_full);
        int cur = 0;
        for (auto& z : repair_runs[t]) {
            if (z.s_lo != cur) throw std::runtime_error("K repair threshold gap");
            cur = z.s_hi + 1;
        }
        if (cur != t) throw std::runtime_error("K repair threshold tail");
    }
}

Route KDeleteCutOracle::route(int t, int s) const {
    const auto& xs = runs[t];
    const auto& st = starts[t];
    auto it = std::upper_bound(st.begin(), st.end(), s);
    int i = (int)(it - st.begin()) - 1;
    if (i >= 0 && xs[i].s_lo <= s && s <= xs[i].s_hi) return xs[i].route(s);
    return {ell_[t], route0_[t]};
}

static int onebit_index(int x, int D) {
    if (x <= 0 || (x & (x - 1))) return -1;
    int j = 0, xx = x;
    while (xx >>= 1) ++j;
    return j < D ? j : -1;
}

int64_t count_causal_onebit_pairs(const std::vector<int>& q, const std::vector<int>& k, int D) {
    std::unordered_map<int, int> seen;
    int64_t total = 0;
    for (int u = 0; u < (int)q.size(); ++u) {
        int qu = q[u];
        for (int j = 0; j < D; ++j) {
            auto it = seen.find(qu ^ (1 << j));
            if (it != seen.end()) total += it->second;
        }
        ++seen[k[u]];
    }
    return total;
}

std::pair<BridgeGrid, BridgeGrid> build_shared_bridges_sparse(
    const std::vector<int>& q, const std::vector<int>& k, int D, CausalCutSuffixIndex& index) {
    int T = (int)q.size();
    BridgeGrid qg(T, std::vector<std::vector<SharedRepairBridge>>(D));
    BridgeGrid kg(T, std::vector<std::vector<SharedRepairBridge>>(D));
    std::unordered_map<int, std::vector<int>> pos;
    for (int p = 0; p < T; ++p) pos[k[p]].push_back(p);
    for (int u = 0; u < T; ++u) {
        int qu = q[u];
        for (int j = 0; j < D; ++j) {
            auto it = pos.find(qu ^ (1 << j));
            if (it == pos.end()) continue;
            const auto& arr = it->second;
            auto z = std::lower_bound(arr.begin(), arr.end(), u);
            for (auto jt = arr.begin(); jt != z; ++jt) {
                int p = *jt;
                int L = index.lcs_before_center(u, p);
                int R = index.lcp_after_center(u, p);
                SharedRepairBridge b{u, p, j, L, R};
                qg[u][j].push_back(b);
                kg[p][j].push_back(b);
            }
        }
    }
    return {qg, kg};
}

std::pair<BridgeGrid, BridgeGrid> build_shared_bridges_diagonal(
    const std::vector<int>& q, const std::vector<int>& k, int D) {
    int T = (int)q.size();
    BridgeGrid qg(T, std::vector<std::vector<SharedRepairBridge>>(D));
    BridgeGrid kg(T, std::vector<std::vector<SharedRepairBridge>>(D));
    for (int h = 1; h < T; ++h) {
        int start_u = h;
        std::vector<std::tuple<int,int,int>> mism;  // (u, p, j or -1)
        for (int p = 0; p + h < T; ++p) {
            int u = p + h;
            int x = q[u] ^ k[p];
            if (x) mism.emplace_back(u, p, onebit_index(x, D));
        }
        for (size_t z = 0; z < mism.size(); ++z) {
            auto [u, p, j] = mism[z];
            if (j < 0) continue;
            int prev_u = z ? std::get<0>(mism[z - 1]) : start_u - 1;
            int next_u = z + 1 < mism.size() ? std::get<0>(mism[z + 1]) : T;
            int L = u - prev_u - 1;
            int R = next_u - u - 1;
            SharedRepairBridge b{u, p, j, L, R};
            qg[u][j].push_back(b);
            kg[p][j].push_back(b);
        }
    }
    return {qg, kg};
}

TermGrid q_shared_terms(const BridgeGrid& qg, int D) {
    int T = (int)qg.size();
    TermGrid out(T, std::vector<std::vector<RepairTrackTerm>>(D));
    for (int u = 0; u < T; ++u) {
        for (int j = 0; j < D; ++j) {
            const auto& bs = qg[u][j];
            if (bs.empty()) continue;
            std::vector<SharedRepairBridge> ordered = bs;
            std::stable_sort(ordered.begin(), ordered.end(), [](const SharedRepairBridge& a, const SharedRepairBridge& b) {
                return a.q_priority() > b.q_priority();  // reverse=True, stable
            });
            int covered = -1;
            for (const auto& b : ordered) {
                if (b.right <= covered) continue;
                int dlo = covered + 1, dhi = b.right;
                out[u][j].push_back({b.shift(), u + dlo, u + dhi});
                covered = dhi;
            }
        }
    }
    return out;
}

struct BridgeInterval {
    int lo, hi;
    std::pair<int,int> priority;
    int shift;
    bool operator==(const BridgeInterval& o) const {
        return lo == o.lo && hi == o.hi && priority == o.priority && shift == o.shift;
    }
};

static int first_win_shared_bridge(const SharedRepairBridge& b, int owner,
                                   const KDeleteCutOracle& kdel) {
    int lo = b.q_pos, hi = b.end_t();
    auto wins = [&](int t) { return b.route_at(t) > kdel.route(t, owner); };
    if (!wins(hi)) return -1;
    if (wins(lo)) return lo;
    int a = lo + 1, c = hi;
    while (a < c) {
        int m = (a + c) / 2;
        if (wins(m)) c = m; else a = m + 1;
    }
    return a;
}

// max constant-priority interval envelope; yields (lo, hi, winner) in t order
static std::vector<std::tuple<int,int,BridgeInterval>> bridge_envelope_segments(
    const std::vector<BridgeInterval>& xs) {
    std::vector<std::tuple<int,int,BridgeInterval>> out;
    if (xs.empty()) return out;
    std::vector<BridgeInterval> items = xs;
    std::stable_sort(items.begin(), items.end(), [](const BridgeInterval& a, const BridgeInterval& b) {
        return std::tie(a.lo, a.priority, a.hi) < std::tie(b.lo, b.priority, b.hi);
    });
    int n = (int)items.size();
    // heap entries comparable on (-a, -b, hi, i) ascending -> top = smallest tuple
    struct Ent {
        int na, nb, hi, i;
        BridgeInterval z;
        bool operator<(const Ent& o) const {  // reversed: priority_queue top = "largest" by <
            return std::tie(na, nb, hi, i) > std::tie(o.na, o.nb, o.hi, o.i);
        }
    };
    std::priority_queue<Ent> heap;
    struct Pending { int lo, hi; BridgeInterval z; };
    std::optional<Pending> pending;
    auto flush = [&](int lo, int hi, const BridgeInterval& z) -> std::optional<Pending> {
        if (pending && pending->z == z && pending->hi + 1 == lo) {
            pending->hi = hi;
            return std::nullopt;
        }
        auto old = pending;
        pending = Pending{lo, hi, z};
        return old;
    };
    int i = 0;
    int t = items[0].lo;
    while (i < n || !heap.empty()) {
        if (heap.empty() && i < n) t = std::max(t, items[i].lo);
        while (!heap.empty() && heap.top().hi < t) heap.pop();
        while (i < n && items[i].lo == t) {
            const BridgeInterval& z = items[i];
            heap.push({-z.priority.first, -z.priority.second, z.hi, i, z});
            ++i;
        }
        while (!heap.empty() && heap.top().hi < t) heap.pop();
        if (heap.empty()) {
            if (i < n) { t = items[i].lo; continue; }
            break;
        }
        const BridgeInterval& win = heap.top().z;
        int win_hi = heap.top().hi;
        int64_t next_start = i < n ? items[i].lo : (int64_t(1) << 60);
        int hi = (int)std::min<int64_t>(win_hi, next_start - 1);
        auto old = flush(t, hi, win);
        if (old) out.emplace_back(old->lo, old->hi, old->z);
        t = hi + 1;
    }
    if (pending) out.emplace_back(pending->lo, pending->hi, pending->z);
    return out;
}

TermGrid k_shared_terms(const BridgeGrid& kg, int D, const KDeleteCutOracle& kdel) {
    int T = (int)kg.size();
    TermGrid out(T, std::vector<std::vector<RepairTrackTerm>>(D));
    for (int s = 0; s < T; ++s) {
        for (int j = 0; j < D; ++j) {
            std::vector<BridgeInterval> clipped;
            for (const auto& b : kg[s][j]) {
                int fw = first_win_shared_bridge(b, s, kdel);
                if (fw >= 0)
                    clipped.push_back({fw, b.end_t(), b.k_priority(), b.shift()});
            }
            for (const auto& [lo, hi, z] : bridge_envelope_segments(clipped))
                out[s][j].push_back({z.shift, lo, hi});
        }
    }
    return out;
}

}  // namespace rosa
