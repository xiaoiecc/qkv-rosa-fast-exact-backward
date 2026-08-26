#include "ortho.h"
#include <algorithm>
#include <iterator>

namespace rosa {

static constexpr int64_t kInf = int64_t(1) << 60;

MinPRange::MinPRange(const std::vector<int>& values) {
    n = (int)values.size();
    size = 1;
    while (size < std::max(1, n)) size <<= 1;
    tr.assign(2 * size, kInf);
    for (int i = 0; i < n; ++i) tr[size + i] = values[i];
    for (int i = size - 1; i > 0; --i) tr[i] = std::min(tr[2 * i], tr[2 * i + 1]);
}

static std::optional<int> minp_rec(const std::vector<int64_t>& tr, int n,
                                   int node, int nl, int nr, int ql, int qr,
                                   int64_t threshold, bool last) {
    if (nr <= ql || qr <= nl || tr[node] >= threshold) return std::nullopt;
    if (nr - nl == 1) {
        if (nl < n) return nl;
        return std::nullopt;
    }
    int mid = (nl + nr) / 2;
    if (last) {
        auto z = minp_rec(tr, n, node * 2 + 1, mid, nr, ql, qr, threshold, last);
        if (z) return z;
        return minp_rec(tr, n, node * 2, nl, mid, ql, qr, threshold, last);
    } else {
        auto z = minp_rec(tr, n, node * 2, nl, mid, ql, qr, threshold, last);
        if (z) return z;
        return minp_rec(tr, n, node * 2 + 1, mid, nr, ql, qr, threshold, last);
    }
}

std::optional<int> MinPRange::find_last(int right_exclusive, int64_t threshold) const {
    int ql = 0, qr = std::min(n, right_exclusive);
    if (ql >= qr) return std::nullopt;
    return minp_rec(tr, n, 1, 0, size, ql, qr, threshold, true);
}

std::optional<int> MinPRange::find_first(int left, int64_t threshold) const {
    int ql = std::max(0, left), qr = n;
    if (ql >= qr) return std::nullopt;
    return minp_rec(tr, n, 1, 0, size, ql, qr, threshold, false);
}

MergeRangePredecessor::MergeRangePredecessor(const std::vector<std::optional<int>>& values_by_index) {
    int n = (int)values_by_index.size();
    size = 1;
    while (size < std::max(1, n)) size <<= 1;
    tree.assign(2 * size, {});
    for (int i = 0; i < n; ++i)
        if (values_by_index[i]) tree[size + i] = {*values_by_index[i]};
    for (int i = size - 1; i > 0; --i) {
        const auto& a = tree[2 * i];
        const auto& b = tree[2 * i + 1];
        std::vector<int> out;
        out.reserve(a.size() + b.size());
        std::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
        tree[i] = std::move(out);
    }
}

std::optional<int> MergeRangePredecessor::predecessor(int left, int right, int x) const {
    if (left >= right) return std::nullopt;
    left += size; right += size;
    std::optional<int> ans;
    while (left < right) {
        if (left & 1) {
            const auto& arr = tree[left];
            auto it = std::upper_bound(arr.begin(), arr.end(), x);
            if (it != arr.begin()) { --it; if (!ans || *it > *ans) ans = *it; }
            ++left;
        }
        if (right & 1) {
            --right;
            const auto& arr = tree[right];
            auto it = std::upper_bound(arr.begin(), arr.end(), x);
            if (it != arr.begin()) { --it; if (!ans || *it > *ans) ans = *it; }
        }
        left >>= 1; right >>= 1;
    }
    return ans;
}

YBucket::YBucket(const std::vector<std::pair<int,int>>& points)
    : minp(std::vector<int>{}), predp(std::vector<std::optional<int>>{}) {
    std::vector<std::pair<int,int>> xs = points;
    std::sort(xs.begin(), xs.end());
    ys.reserve(xs.size()); ps.reserve(xs.size());
    for (auto& [y, p] : xs) { ys.push_back(y); ps.push_back(p); }
    minp = MinPRange(ps);
    std::vector<std::optional<int>> pv(ps.begin(), ps.end());
    predp = MergeRangePredecessor(pv);
}

std::optional<std::pair<int,int>> YBucket::pred_y(int yq, int owner_u) const {
    int idx = (int)(std::upper_bound(ys.begin(), ys.end(), yq) - ys.begin());
    auto z = minp.find_last(idx, owner_u);
    if (!z) return std::nullopt;
    return std::make_pair(ys[*z], ps[*z]);
}

std::optional<std::pair<int,int>> YBucket::succ_y(int yq, int owner_u) const {
    int idx = (int)(std::lower_bound(ys.begin(), ys.end(), yq) - ys.begin());
    auto z = minp.find_first(idx, owner_u);
    if (!z) return std::nullopt;
    return std::make_pair(ys[*z], ps[*z]);
}

std::optional<int> YBucket::max_p(int ylo, int yhi, int owner_u) const {
    int a = (int)(std::lower_bound(ys.begin(), ys.end(), ylo) - ys.begin());
    int b = (int)(std::lower_bound(ys.begin(), ys.end(), yhi) - ys.begin());
    return predp.predecessor(a, b, owner_u - 1);
}

SymbolOrthogonalOracle::SymbolOrthogonalOracle(const std::vector<std::tuple<int,int,int>>& points) {
    std::vector<std::tuple<int,int,int>> pts = points;
    std::sort(pts.begin(), pts.end());
    n = (int)pts.size();
    xs.reserve(n);
    for (auto& [x, y, p] : pts) xs.push_back(x);
    size = 1;
    while (size < std::max(1, n)) size <<= 1;
    std::vector<std::vector<std::pair<int,int>>> raw(2 * size);
    for (int i = 0; i < n; ++i) raw[size + i] = {{std::get<1>(pts[i]), std::get<2>(pts[i])}};
    for (int i = size - 1; i > 0; --i) {
        raw[i] = raw[2 * i];
        raw[i].insert(raw[i].end(), raw[2 * i + 1].begin(), raw[2 * i + 1].end());
    }
    buckets.assign(2 * size, std::nullopt);
    for (int i = 1; i < 2 * size; ++i)
        if (!raw[i].empty()) buckets[i] = YBucket(raw[i]);
}

std::vector<int> SymbolOrthogonalOracle::nodes(int xlo, int xhi) const {
    int l = (int)(std::lower_bound(xs.begin(), xs.end(), xlo) - xs.begin());
    int r = (int)(std::lower_bound(xs.begin(), xs.end(), xhi) - xs.begin());
    l += size; r += size;
    std::vector<int> out;
    while (l < r) {
        if (l & 1) { out.push_back(l); ++l; }
        if (r & 1) { --r; out.push_back(r); }
        l >>= 1; r >>= 1;
    }
    return out;
}

std::pair<std::optional<std::pair<int,int>>, std::optional<std::pair<int,int>>>
SymbolOrthogonalOracle::nearest(int xlo, int xhi, int yq, int owner_u) const {
    std::optional<std::pair<int,int>> pred, succ;
    for (int node : nodes(xlo, xhi)) {
        const auto& b = buckets[node];
        if (!b) continue;
        auto z = b->pred_y(yq, owner_u);
        if (z && (!pred || z->first > pred->first)) pred = z;
        z = b->succ_y(yq, owner_u);
        if (z && (!succ || z->first < succ->first)) succ = z;
    }
    return {pred, succ};
}

std::optional<int> SymbolOrthogonalOracle::max_p(int xlo, int xhi, int ylo, int yhi, int owner_u) const {
    std::optional<int> ans;
    for (int node : nodes(xlo, xhi)) {
        const auto& b = buckets[node];
        if (!b) continue;
        auto z = b->max_p(ylo, yhi, owner_u);
        if (z && (!ans || *z > *ans)) ans = z;
    }
    return ans;
}

}  // namespace rosa
