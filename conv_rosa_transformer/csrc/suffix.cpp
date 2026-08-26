#include "suffix.h"
#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <stdexcept>

namespace rosa {

std::vector<int> suffix_array_int(const std::vector<int>& seq) {
    const int n = (int)seq.size();
    if (n == 0) return {};
    // rank compression over sorted unique values
    std::vector<int> uniq = seq;
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    std::unordered_map<int, int> vals;
    vals.reserve(uniq.size() * 2);
    for (int i = 0; i < (int)uniq.size(); ++i) vals[uniq[i]] = i;
    std::vector<int> rank(n);
    for (int i = 0; i < n; ++i) rank[i] = vals[seq[i]];
    std::vector<int> sa(n);
    for (int i = 0; i < n; ++i) sa[i] = i;
    for (int k = 1; k < n; k <<= 1) {
        // stable sort by (rank[i], rank[i+k] or -1) -- matches Python's stable list.sort
        std::stable_sort(sa.begin(), sa.end(), [&](int a, int b) {
            int ra = rank[a], rb = rank[b];
            if (ra != rb) return ra < rb;
            int ka2 = (a + k < n) ? rank[a + k] : -1;
            int kb2 = (b + k < n) ? rank[b + k] : -1;
            return ka2 < kb2;
        });
        std::vector<int> nr(n, 0);
        int cls = 0;
        for (int z = 1; z < n; ++z) {
            int a = sa[z - 1], b = sa[z];
            int ka1 = rank[a], ka2 = (a + k < n) ? rank[a + k] : -1;
            int kb1 = rank[b], kb2 = (b + k < n) ? rank[b + k] : -1;
            if (ka1 != kb1 || ka2 != kb2) ++cls;
            nr[b] = cls;
        }
        rank = std::move(nr);
        if (cls == n - 1) break;
    }
    return sa;
}

SuffixLCE::SuffixLCE(const std::vector<int>& s) : seq(s), n((int)s.size()) {
    sa = suffix_array_int(seq);
    rank.assign(n, 0);
    for (int r = 0; r < n; ++r) rank[sa[r]] = r;
    // Kasai: lcp[r] = LCP(sa[r-1], sa[r]) for r>=1; lcp[0]=0
    lcp.assign(n, 0);
    int h = 0;
    for (int i = 0; i < n; ++i) {
        int r = rank[i];
        if (r == 0) continue;
        int j = sa[r - 1];
        while (i + h < n && j + h < n && seq[i + h] == seq[j + h]) ++h;
        lcp[r] = h;
        if (h) --h;
    }
    lg.assign(n + 1, 0);
    for (int i = 2; i <= n; ++i) lg[i] = lg[i >> 1] + 1;
    st.push_back(lcp);
    for (int j = 1; (1 << j) <= n; ++j) {
        const std::vector<int>& prev = st.back();
        int half = 1 << (j - 1), width = 1 << j;
        std::vector<int> row(n, 0);
        int lim = n - width + 1;
        for (int i = 0; i < std::max(0, lim); ++i) row[i] = std::min(prev[i], prev[i + half]);
        st.push_back(std::move(row));
    }
}

int SuffixLCE::rmq_lcp(int lo, int hi) const {
    int length = hi - lo;
    int j = lg[length];
    int w = 1 << j;
    return std::min(st[j][lo], st[j][hi - w]);
}

int SuffixLCE::lcp_pos(int a, int b) const {
    if (a == b) return n - a;
    int ra = rank[a], rb = rank[b];
    if (ra > rb) std::swap(ra, rb);
    return rmq_lcp(ra + 1, rb + 1);
}

std::pair<int, int> SuffixLCE::prefix_rank_interval(int pos, int length) const {
    if (length <= 0) return {0, n};
    int r0 = rank[pos];
    int lo = 0, hi = r0;
    while (lo < hi) {
        int m = (lo + hi) / 2;
        if (lcp_pos(pos, sa[m]) >= length) hi = m; else lo = m + 1;
    }
    int left = lo;
    lo = r0; hi = n - 1;
    while (lo < hi) {
        int m = (lo + hi + 1) / 2;
        if (lcp_pos(pos, sa[m]) >= length) lo = m; else hi = m - 1;
    }
    return {left, lo + 1};
}

StaticMaxPByRank::StaticMaxPByRank(const std::vector<int>& values) {
    n = (int)values.size();
    size = 1;
    while (size < std::max(1, n)) size <<= 1;
    tr.assign(2 * size, -1);
    for (int i = 0; i < n; ++i) tr[size + i] = values[i];
    for (int i = size - 1; i > 0; --i) tr[i] = std::max(tr[i << 1], tr[i << 1 | 1]);
}

static std::optional<int> maxp_rec(const std::vector<int>& tr, int n,
                                   int node, int nl, int nr, int lo, int hi, int x, bool last) {
    if (nr <= lo || hi <= nl || tr[node] <= x) return std::nullopt;
    if (nr - nl == 1) {
        if (nl < n) return nl;
        return std::nullopt;
    }
    int m = (nl + nr) >> 1;
    if (last) {
        auto z = maxp_rec(tr, n, node << 1 | 1, m, nr, lo, hi, x, last);
        if (z) return z;
        return maxp_rec(tr, n, node << 1, nl, m, lo, hi, x, last);
    } else {
        auto z = maxp_rec(tr, n, node << 1, nl, m, lo, hi, x, last);
        if (z) return z;
        return maxp_rec(tr, n, node << 1 | 1, m, nr, lo, hi, x, last);
    }
}

std::optional<int> StaticMaxPByRank::find_last(int lo, int hi, int x) const {
    lo = std::max(0, lo); hi = std::min(n, hi);
    if (lo >= hi) return std::nullopt;
    return maxp_rec(tr, n, 1, 0, size, lo, hi, x, true);
}

std::optional<int> StaticMaxPByRank::find_first(int lo, int hi, int x) const {
    lo = std::max(0, lo); hi = std::min(n, hi);
    if (lo >= hi) return std::nullopt;
    return maxp_rec(tr, n, 1, 0, size, lo, hi, x, false);
}

StaticRangeSuccessorP::StaticRangeSuccessorP(const std::vector<int>& values) {
    n = (int)values.size();
    size = 1;
    while (size < std::max(1, n)) size <<= 1;
    rows.assign(2 * size, {});
    for (int i = 0; i < n; ++i)
        if (values[i] >= 0) rows[size + i] = {values[i]};
    for (int i = size - 1; i > 0; --i) {
        const auto& a = rows[i << 1];
        const auto& b = rows[i << 1 | 1];
        std::vector<int> row;
        row.reserve(a.size() + b.size());
        std::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(row));
        rows[i] = std::move(row);
    }
}

std::optional<int> StaticRangeSuccessorP::range_successor(int lo, int hi, int x) const {
    lo = std::max(0, lo); hi = std::min(n, hi);
    if (lo >= hi) return std::nullopt;
    lo += size; hi += size;
    std::optional<int> ans;
    while (lo < hi) {
        if (lo & 1) {
            const auto& row = rows[lo];
            auto it = std::upper_bound(row.begin(), row.end(), x);
            if (it != row.end() && (!ans || *it < *ans)) ans = *it;
            ++lo;
        }
        if (hi & 1) {
            --hi;
            const auto& row = rows[hi];
            auto it = std::upper_bound(row.begin(), row.end(), x);
            if (it != row.end() && (!ans || *it < *ans)) ans = *it;
        }
        lo >>= 1; hi >>= 1;
    }
    return ans;
}

RangePositionIndex::RangePositionIndex(const std::vector<std::optional<int>>& values) {
    n = (int)values.size();
    size = 1;
    while (size < std::max(1, n)) size <<= 1;
    rows.assign(2 * size, {});
    for (int i = 0; i < n; ++i)
        if (values[i]) rows[size + i] = {*values[i]};
    for (int i = size - 1; i > 0; --i) {
        const auto& a = rows[i << 1];
        const auto& b = rows[i << 1 | 1];
        std::vector<int> row;
        row.reserve(a.size() + b.size());
        std::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(row));
        rows[i] = std::move(row);
    }
}

std::vector<int> RangePositionIndex::nodes(int lo, int hi) const {
    lo = std::max(0, lo); hi = std::min(n, hi);
    std::vector<int> out;
    if (lo >= hi) return out;
    lo += size; hi += size;
    while (lo < hi) {
        if (lo & 1) { out.push_back(lo); ++lo; }
        if (hi & 1) { --hi; out.push_back(hi); }
        lo >>= 1; hi >>= 1;
    }
    return out;
}

std::optional<int> RangePositionIndex::successor(int lo, int hi, int x) const {
    std::optional<int> ans;
    for (int node : nodes(lo, hi)) {
        const auto& row = rows[node];
        auto it = std::upper_bound(row.begin(), row.end(), x);
        if (it != row.end() && (!ans || *it < *ans)) ans = *it;
    }
    return ans;
}

std::optional<int> RangePositionIndex::predecessor(int lo, int hi, int x) const {
    std::optional<int> ans;
    for (int node : nodes(lo, hi)) {
        const auto& row = rows[node];
        auto it = std::lower_bound(row.begin(), row.end(), x);
        if (it != row.begin()) {
            --it;
            if (!ans || *it > *ans) ans = *it;
        }
    }
    return ans;
}

std::optional<int> RangePositionIndex::min_in(int lo, int hi, int p_lo, int p_hi) const {
    std::optional<int> ans;
    for (int node : nodes(lo, hi)) {
        const auto& row = rows[node];
        auto it = std::lower_bound(row.begin(), row.end(), p_lo);
        if (it != row.end() && *it <= p_hi && (!ans || *it < *ans)) ans = *it;
    }
    return ans;
}

std::optional<int> RangePositionIndex::max_in(int lo, int hi, int p_lo, int p_hi) const {
    std::optional<int> ans;
    for (int node : nodes(lo, hi)) {
        const auto& row = rows[node];
        auto it = std::upper_bound(row.begin(), row.end(), p_hi);
        if (it != row.begin()) {
            --it;
            if (*it >= p_lo && (!ans || *it > *ans)) ans = *it;
        }
    }
    return ans;
}

std::vector<int> RangePositionIndex::report(int lo, int hi, int p_lo, int p_hi) const {
    std::vector<int> out;
    for (int node : nodes(lo, hi)) {
        const auto& row = rows[node];
        auto a = std::lower_bound(row.begin(), row.end(), p_lo);
        auto b = std::upper_bound(row.begin(), row.end(), p_hi);
        out.insert(out.end(), a, b);
    }
    return out;
}

std::pair<std::vector<int>, std::vector<int>> matching_stats(
    const std::vector<int64_t>& q, const std::vector<int64_t>& k) {
    const int n = (int)q.size();
    if (n != (int)k.size()) throw std::invalid_argument("Q/K lengths differ");
    if (n == 0) return {{}, {}};

    // remap symbols to >= 2 in first-appearance order (q then k), identical to Python
    std::unordered_map<int64_t, int> mp;
    mp.reserve(2 * n);
    int nxt = 2;
    std::vector<int> qi(n), ki(n);
    for (int i = 0; i < n; ++i) {
        auto [itq, nq] = mp.emplace(q[i], nxt);
        if (nq) ++nxt;
        qi[i] = itq->second;
    }
    for (int i = 0; i < n; ++i) {
        auto [itk, nk] = mp.emplace(k[i], nxt);
        if (nk) ++nxt;
        ki[i] = itk->second;
    }
    std::vector<int> qr(qi.rbegin(), qi.rend()), kr(ki.rbegin(), ki.rend());
    const int q_off = 0, k_off = n + 1;
    std::vector<int> text;
    text.reserve(2 * n + 2);
    text.insert(text.end(), qr.begin(), qr.end());
    text.push_back(0);
    text.insert(text.end(), kr.begin(), kr.end());
    text.push_back(1);
    SuffixLCE idx(text);

    const int N = (int)text.size();
    std::vector<int> rank_to_p(N, -1);
    for (int p = 0; p < n; ++p) rank_to_p[idx.rank[k_off + p]] = p;
    StaticMaxPByRank extrema(rank_to_p);
    StaticRangeSuccessorP succ_p(rank_to_p);

    std::vector<int> lens(n, 0), route_end(n, -1);
    for (int t = 0; t < n; ++t) {
        int x = n - 1 - t;
        int qpos = q_off + x;
        int rq = idx.rank[qpos];
        auto rp = extrema.find_last(0, rq, x);
        auto rs = extrema.find_first(rq + 1, N, x);
        int best = 0;
        if (rp) best = std::max(best, idx.lcp_pos(qpos, idx.sa[*rp]));
        if (rs) best = std::max(best, idx.lcp_pos(qpos, idx.sa[*rs]));
        if (best <= 0) continue;
        auto [lo, hi] = idx.prefix_rank_interval(qpos, best);
        auto p_star = succ_p.range_successor(lo, hi, x);
        if (!p_star) throw std::runtime_error("static position-restricted winner interval lost K suffix");
        int e_star = n - 1 - *p_star;
        best = std::min({best, t + 1, e_star + 1});
        if (best > 0) {
            lens[t] = best;
            route_end[t] = e_star;
        }
    }
    return {lens, route_end};
}

}  // namespace rosa
