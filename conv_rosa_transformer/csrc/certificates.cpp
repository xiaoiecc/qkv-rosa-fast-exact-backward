#include "certificates.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace rosa {

static int onebit_index(int x, int D) {
    if (x <= 0 || (x & (x - 1))) return -1;
    int j = 0, xx = x;
    while (xx >>= 1) ++j;
    return j < D ? j : -1;
}

std::vector<int> next_suffix_rank_index(const std::vector<int>& ranks, bool smaller) {
    int n = (int)ranks.size();
    std::vector<int> out(n, n), st;
    for (int i = n - 1; i >= 0; --i) {
        int ri = ranks[i];
        if (smaller) {
            while (!st.empty() && ranks[st.back()] >= ri) st.pop_back();
        } else {
            while (!st.empty() && ranks[st.back()] <= ri) st.pop_back();
        }
        out[i] = st.empty() ? n : st.back();
        st.push_back(i);
    }
    return out;
}

std::vector<std::tuple<int,int,int>> enumerate_k_runs(const CausalCutSuffixIndex& index) {
    int T = index.T;
    if (T <= 1) return {};
    std::vector<int> ranks(T);
    for (int i = 0; i < T; ++i) ranks[i] = index.fwd.rank[index.k_off + i];
    std::set<std::tuple<int,int,int>> runs;
    for (bool smaller : {true, false}) {
        auto nxt = next_suffix_rank_index(ranks, smaller);
        for (int i = 0; i < T; ++i) {
            int j = nxt[i];
            if (j >= T) continue;
            int p = j - i;
            if (p <= 0) continue;
            int left_ext = 0;
            if (i > 0)
                left_ext = std::min({i, j,
                    index.rev.lcp_pos(index.rk_off + T - i, index.rk_off + T - j)});
            int right_ext = std::min({T - i, T - j,
                index.fwd.lcp_pos(index.k_off + i, index.k_off + j)});
            if (left_ext + right_ext < p) continue;
            int lo = i - left_ext;
            int hi = j + right_ext - 1;
            if (hi - lo + 1 < 2 * p) continue;
            runs.emplace(lo, hi, p);
        }
    }
    return std::vector<std::tuple<int,int,int>>(runs.begin(), runs.end());
}

KRunRepairCertificateIndex::KRunRepairCertificateIndex(const CausalCutSuffixIndex& index, int D) {
    T = index.T;
    std::set<KRunRepairCertificate> cert_set;
    for (const auto& [lo, hi, q] : enumerate_k_runs(index)) {
        int run_len = hi - lo + 1;
        int max_k = run_len / (2 * q);
        if (hi + 1 >= T) continue;
        int right_symbol_pos = hi + 1;
        for (int mult = 1; mult <= max_k; ++mult) {
            int p = mult * q;
            int s = hi - p + 1;
            int j = onebit_index(index.k[s] ^ index.k[right_symbol_pos], D);
            if (j < 0) continue;
            int rho;
            if (right_symbol_pos + 1 >= T) rho = 0;
            else rho = std::min({T - (s + 1), T - (right_symbol_pos + 1),
                index.fwd.lcp_pos(index.k_off + s + 1, index.k_off + right_symbol_pos + 1)});
            int m_lo = right_symbol_pos;
            int m_hi = std::min(T - 1, right_symbol_pos + rho);
            cert_set.insert({s, p, j, m_lo, m_hi});
        }
    }
    certs.assign(cert_set.begin(), cert_set.end());

    size = 1;
    while (size < std::max(1, T)) size <<= 1;
    std::vector<std::vector<KRunRepairCertificate>> raw(2 * size);
    for (const auto& c : certs) {
        int l = c.m_lo + size, r = c.m_hi + 1 + size;
        while (l < r) {
            if (l & 1) { raw[l].push_back(c); ++l; }
            if (r & 1) { --r; raw[r].push_back(c); }
            l >>= 1; r >>= 1;
        }
    }
    rows.resize(2 * size);
    owners.resize(2 * size);
    for (int i = 0; i < 2 * size; ++i) {
        auto& row = raw[i];
        std::sort(row.begin(), row.end(), [](const KRunRepairCertificate& a, const KRunRepairCertificate& b) {
            return std::tie(a.owner, a.bit, a.period, a.m_lo, a.m_hi) <
                   std::tie(b.owner, b.bit, b.period, b.m_lo, b.m_hi);
        });
        rows[i] = row;
        for (auto& c : row) owners[i].push_back(c.owner);
    }
}

std::vector<KRunRepairCertificate> KRunRepairCertificateIndex::query(int M, int s_lo, int s_hi) const {
    std::vector<KRunRepairCertificate> out;
    if (!(0 <= M && M < T) || s_lo > s_hi) return out;
    int node = size + M;
    while (node) {
        const auto& ow = owners[node];
        if (!ow.empty()) {
            auto a = std::lower_bound(ow.begin(), ow.end(), s_lo);
            auto b = std::upper_bound(ow.begin(), ow.end(), s_hi);
            out.insert(out.end(), rows[node].begin() + (a - ow.begin()),
                       rows[node].begin() + (b - ow.begin()));
        }
        node >>= 1;
    }
    return out;
}


}  // namespace rosa
