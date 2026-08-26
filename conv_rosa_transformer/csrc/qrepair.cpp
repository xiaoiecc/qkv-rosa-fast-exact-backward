#include "qrepair.h"
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

namespace rosa {

std::vector<std::vector<std::vector<RepairTrackTerm>>> q_repair_terms_suffix_range(
    const std::vector<int>& q, int D, CausalCutSuffixIndex& index) {
    int T = (int)q.size();
    std::vector<std::vector<std::vector<RepairTrackTerm>>> out(T, std::vector<std::vector<RepairTrackTerm>>(D));
    for (int u = 0; u < T; ++u) {
        int qu = q[u];
        for (int j = 0; j < D; ++j) {
            int target = qu ^ (1 << j);
            int d = 0;
            while (u + d < T) {
                auto best = index.direct_q_best_center(target, u, d);
                if (!best) break;
                int p = best->first, L = best->second;
                int R = index.lcp_after_center(u, p);
                if (R < d) throw std::runtime_error("direct Q right extent");
                out[u][j].push_back({p - u + 1, u + d, std::min(T - 1, u + R)});
                d = R + 1;
            }
        }
    }
    return out;
}

std::vector<ZeroBaselineSurface> zero_baseline_surfaces(
    const std::vector<int>& q, const std::vector<int>& k,
    const std::vector<int>& ell, int D) {
    std::set<int> unmatched;
    for (int t = 0; t < (int)ell.size(); ++t)
        if (ell[t] == 0) unmatched.insert(q[t]);
    std::set<int> ksyms(k.begin(), k.end());
    std::vector<ZeroBaselineSurface> out;
    for (int c : ksyms)
        for (int j = 0; j < D; ++j) {
            int tar = c ^ (1 << j);
            if (unmatched.count(tar)) out.push_back({j, c, tar});
        }
    return out;
}

}  // namespace rosa
