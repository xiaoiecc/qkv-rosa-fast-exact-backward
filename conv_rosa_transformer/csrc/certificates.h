#pragma once
// K-only H-ramp repair certificates: right-boundary squares from maximal runs in K.
#include <vector>
#include <tuple>
#include "index.h"

namespace rosa {

// one maximal-run certificate: owner s repairs anchors M in [m_lo,m_hi] via endpoint M-period
struct KRunRepairCertificate {
    int owner, period, bit, m_lo, m_hi;
    bool operator<(const KRunRepairCertificate& o) const {
        return std::tie(owner, period, bit, m_lo, m_hi) <
               std::tie(o.owner, o.period, o.bit, o.m_lo, o.m_hi);
    }
};

// next position to the right with smaller/greater suffix rank (one monotone stack)
std::vector<int> next_suffix_rank_index(const std::vector<int>& ranks, bool smaller);

// all maximal runs (lo, hi, minimal_period) via the two-Lyndon-orientation characterization
std::vector<std::tuple<int,int,int>> enumerate_k_runs(const CausalCutSuffixIndex& index);

// static (anchor point M, owner interval) reporting index over O(T) certificates
class KRunRepairCertificateIndex {
public:
    KRunRepairCertificateIndex(const CausalCutSuffixIndex& index, int D);
    std::vector<KRunRepairCertificate> query(int M, int s_lo, int s_hi) const;
    std::vector<KRunRepairCertificate> certs;
private:
    int T, size;
    std::vector<std::vector<KRunRepairCertificate>> rows;
    std::vector<std::vector<int>> owners;
};

}  // namespace rosa
