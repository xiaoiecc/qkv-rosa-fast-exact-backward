#include "index.h"
#include <algorithm>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace rosa {

// floor(log2(x)) for x > 0; portable (no compiler builtins)
static int bitlen_minus1(int x) {
    int r = 0;
    while (x >>= 1) ++r;
    return r;
}

static std::pair<std::vector<int>, std::vector<int>> remap_symbols(
    const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    std::unordered_map<int64_t, int> mp;
    int nxt = 2;
    std::vector<int> ra(a.size()), rb(b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        auto [it, nw] = mp.emplace(a[i], nxt);
        if (nw) ++nxt;
        ra[i] = it->second;
    }
    for (size_t i = 0; i < b.size(); ++i) {
        auto [it, nw] = mp.emplace(b[i], nxt);
        if (nw) ++nxt;
        rb[i] = it->second;
    }
    return {ra, rb};
}

CausalCutSuffixIndex::CausalCutSuffixIndex(const std::vector<int64_t>& qq, const std::vector<int64_t>& kk)
    : T((int)qq.size()),
      fwd(std::vector<int>{}),
      rev(std::vector<int>{}),
      fpos(std::vector<std::optional<int>>{}),
      rpos(std::vector<std::optional<int>>{}) {
    if (qq.size() != kk.size()) throw std::invalid_argument("Q/K lengths differ");
    auto [qi, ki] = remap_symbols(qq, kk);
    // members keep ORIGINAL symbols (xor tests use them); qi/ki only build the SA texts
    q.assign(qq.begin(), qq.end());
    k.assign(kk.begin(), kk.end());

    q_off = 0; k_off = T + 1;
    {
        std::vector<int> text;
        text.reserve(2 * T + 2);
        text.insert(text.end(), qi.begin(), qi.end());
        text.push_back(0);
        text.insert(text.end(), ki.begin(), ki.end());
        text.push_back(1);
        fwd = SuffixLCE(text);
    }
    rq_off = 0; rk_off = T + 1;
    {
        std::vector<int> text;
        text.reserve(2 * T + 2);
        text.insert(text.end(), qi.rbegin(), qi.rend());
        text.push_back(0);
        text.insert(text.end(), ki.rbegin(), ki.rend());
        text.push_back(1);
        rev = SuffixLCE(text);
    }
    {
        std::vector<std::optional<int>> fvals(fwd.n, std::nullopt);
        for (int a = 0; a < T; ++a) fvals[fwd.rank[k_off + a]] = a;
        fpos = RangePositionIndex(fvals);
    }
    {
        std::vector<std::optional<int>> rvals(rev.n, std::nullopt);
        for (int p = 0; p < T; ++p) rvals[rev.rank[rk_off + p]] = p;
        rpos = RangePositionIndex(rvals);
    }
    positions_by_symbol.reserve(T * 2);
    for (int a = 0; a < T; ++a) positions_by_symbol[k[a]].push_back(a);
}

int CausalCutSuffixIndex::lcp_after_center(int u, int p) const {
    if (u + 1 >= T || p + 1 >= T) return 0;
    return std::min({fwd.lcp_pos(q_off + u + 1, k_off + p + 1), T - u - 1, T - p - 1});
}

int CausalCutSuffixIndex::lcs_before_center(int u, int p) const {
    if (u <= 0 || p <= 0) return 0;
    return std::min({rev.lcp_pos(rq_off + T - u, rk_off + T - p), u, p});
}

std::pair<int, int> CausalCutSuffixIndex::q_pattern_interval(int t, int L) const {
    if (L <= 0) return {0, fwd.n};
    return fwd.prefix_rank_interval(q_off + t - L + 1, L);
}

std::pair<int, int> CausalCutSuffixIndex::q_reversed_prefix_interval(int t, int L) const {
    if (L <= 0) return {0, rev.n};
    int x = T - 1 - t;
    return rev.prefix_rank_interval(rq_off + x, L);
}

const SymbolOrthogonalOracle& CausalCutSuffixIndex::q_orth_for_symbol(int target) {
    auto it = q_orth_.find(target);
    if (it != q_orth_.end()) return it->second;
    std::vector<std::tuple<int,int,int>> pts;
    for (int p = 0; p < T; ++p) {
        if (k[p] != target) continue;
        int xrank = fwd.rank[k_off + p + 1];
        int yrank = rev.rank[rk_off + (T - p)];
        pts.emplace_back(xrank, yrank, p);
    }
    auto [it2, _] = q_orth_.emplace(target, SymbolOrthogonalOracle(pts));
    return it2->second;
}

std::optional<std::pair<int,int>> CausalCutSuffixIndex::direct_q_best_center(int target, int u, int depth) {
    const SymbolOrthogonalOracle& orth = q_orth_for_symbol(target);
    if (orth.n == 0) return std::nullopt;
    int xl, xr;
    if (depth <= 0 || u + 1 >= T) { xl = 0; xr = fwd.n; }
    else { auto iv = fwd.prefix_rank_interval(q_off + u + 1, depth); xl = iv.first; xr = iv.second; }
    int qrev_pos = rq_off + (T - u);
    int yq = rev.rank[qrev_pos];
    auto [pred, succ] = orth.nearest(xl, xr, yq, u);
    std::vector<std::pair<int,int>> cands;
    for (const auto& z : {pred, succ})
        if (z) cands.emplace_back(lcs_before_center(u, z->second), z->second);
    if (cands.empty()) return std::nullopt;
    int Lstar = 0;
    for (auto& [a, _] : cands) Lstar = std::max(Lstar, a);
    int ylo, yhi;
    if (Lstar <= 0) { ylo = 0; yhi = rev.n; }
    else { auto iv = rev.prefix_rank_interval(qrev_pos, Lstar); ylo = iv.first; yhi = iv.second; }
    auto pbest = orth.max_p(xl, xr, ylo, yhi, u);
    if (!pbest) throw std::runtime_error("direct Q tie query lost candidate");
    int Lbest = lcs_before_center(u, *pbest);
    if (Lbest != Lstar) throw std::runtime_error("direct Q max-L mismatch");
    return std::make_pair(*pbest, Lbest);
}

std::vector<int> CausalCutSuffixIndex::report_q_substring_starts(
    int q_start, int length, int k_start_lo, int k_start_hi) const {
    if (length <= 0) {
        std::vector<int> out;
        if (k_start_lo > k_start_hi) return out;
        for (int a = std::max(0, k_start_lo); a <= std::min(T - 1, k_start_hi); ++a) out.push_back(a);
        return out;
    }
    if (q_start < 0 || q_start + length > T) return {};
    auto [l, r] = fwd.prefix_rank_interval(q_off + q_start, length);
    return fpos.report(l, r, std::max(0, k_start_lo), std::min(T - 1, k_start_hi));
}

std::vector<OneBitOccurrence> CausalCutSuffixIndex::one_bit_occurrences_filtered(
    int t, int L, int S0, int S1, int D) {
    S0 = std::max(0, S0); S1 = std::min(t - 1, S1);
    if (L <= 0 || L > t + 1 || S0 > S1) return {};
    auto key = std::make_tuple(t, L, S0, S1, D);
    auto cit = onebit_cache_.find(key);
    if (cit != onebit_cache_.end()) return cit->second;

    int q0 = t - L + 1;
    std::set<int> starts;
    if (L == 1) {
        int qsym = q[t];
        for (int bit = 0; bit < D; ++bit) {
            int target = qsym ^ (1 << bit);
            auto it = positions_by_symbol.find(target);
            if (it == positions_by_symbol.end()) continue;
            const auto& arr = it->second;
            auto a = std::lower_bound(arr.begin(), arr.end(), S0);
            auto b = std::upper_bound(arr.begin(), arr.end(), S1);
            for (auto jt = a; jt != b; ++jt)
                if (*jt < t) starts.insert(*jt);
        }
    } else {
        int h = L / 2;
        // mismatch in right half: left seed exact
        int lo = std::max(0, S0 - L + 1), hi = std::min({S1 - h, t - L, T - L});
        if (h > 0 && lo <= hi)
            for (int a : report_q_substring_starts(q0, h, lo, hi)) starts.insert(a);
        // mismatch in left half: right seed exact at b=a+h
        int tail = L - h;
        int alo = std::max(0, S0 - h + 1), ahi = std::min({S1, t - L, T - L});
        if (tail > 0 && alo <= ahi) {
            int blo = alo + h, bhi = ahi + h;
            for (int b0 : report_q_substring_starts(q0 + h, tail, blo, bhi)) starts.insert(b0 - h);
        }
    }
    std::vector<OneBitOccurrence> out;
    for (int a : starts) {
        if (!(0 <= a && a + L - 1 < t)) continue;
        int x = std::min(L, fwd.lcp_pos(q_off + q0, k_off + a));
        if (x >= L) continue;
        int s = a + x;
        if (!(S0 <= s && s <= S1)) continue;
        int xo = q[q0 + x] ^ k[s];
        if (xo <= 0 || (xo & (xo - 1)) || bitlen_minus1(xo) >= D) continue;
        int rem = L - x - 1;
        int y = 0;
        if (rem > 0) y = std::min(rem, fwd.lcp_pos(q_off + q0 + x + 1, k_off + a + x + 1));
        if (y != rem) continue;
        out.push_back({a, a + L - 1, s, bitlen_minus1(xo)});
    }
    // deduplicate by full tuple, then sort (Python does dict-dedup + sort)
    std::map<std::tuple<int,int,int,int>, OneBitOccurrence> uniq;
    for (auto& o : out) uniq[std::make_tuple(o.start, o.end, o.mismatch_owner, o.bit)] = o;
    std::vector<OneBitOccurrence> z;
    for (auto& [_, o] : uniq) z.push_back(o);
    std::sort(z.begin(), z.end());
    onebit_cache_[key] = z;
    return z;
}

int CausalCutSuffixIndex::lcs_end(int t, int e) const {
    if (t < 0 || e < 0) return 0;
    int x = T - 1 - t;
    int p = T - 1 - e;
    return std::min({rev.lcp_pos(rq_off + x, rk_off + p), t + 1, e + 1});
}

std::optional<int> CausalCutSuffixIndex::latest_endpoint_for_suffix(int t, int L) const {
    if (L <= 0 || L > t + 1) return std::nullopt;
    auto [l, r] = q_pattern_interval(t, L);
    auto a = fpos.max_in(l, r, 0, t - L);
    if (!a) return std::nullopt;
    return *a + L - 1;
}

std::optional<int> CausalCutSuffixIndex::next_endpoint_at_least(int t, int L, int after_e) const {
    if (L <= 0 || L > t + 1) return std::nullopt;
    auto [l, r] = q_pattern_interval(t, L);
    int lo = std::max(0, after_e - L + 2);
    int hi = t - L;
    if (lo > hi) return std::nullopt;
    auto a = fpos.min_in(l, r, lo, hi);
    if (!a) return std::nullopt;
    return *a + L - 1;
}

int CausalCutSuffixIndex::F_reversed(int t, int p) const {
    int x = T - 1 - t;
    if (!(x < p && p < T)) return 0;
    int e = T - 1 - p;
    return std::min({rev.lcp_pos(rq_off + x, rk_off + p), t + 1, e + 1});
}

std::optional<int> CausalCutSuffixIndex::min_p_with_F_at_least(int t, int L, int p_lo, int p_hi) const {
    if (L <= 0) return p_lo <= p_hi ? std::optional<int>(p_lo) : std::nullopt;
    if (p_lo > p_hi || L > t + 1) return std::nullopt;
    auto [l, r] = q_reversed_prefix_interval(t, L);
    return rpos.min_in(l, r, p_lo, p_hi);
}

}  // namespace rosa
