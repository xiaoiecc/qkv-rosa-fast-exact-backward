#include "contract.h"
#include "suffix.h"
#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>

namespace rosa {

// ---------------- Fenwick over double (Python float arithmetic) ----------------
struct Fenwick {
    int n;
    std::vector<double> bit;
    explicit Fenwick(int n_) : n(n_), bit(n_ + 1, 0.0) {}
    void add(int i, double x) {
        for (++i; i <= n; i += i & -i) bit[i] += x;
    }
    double prefix(int i) const {
        double s = 0.0;
        for (; i > 0; i -= i & -i) s += bit[i];
        return s;
    }
    double range_sum(int lo, int hi) const {
        if (lo > hi) return 0.0;
        return prefix(hi + 1) - prefix(lo);
    }
    double total() const { return prefix(n); }
};

// ---------------- Payload LCE (semantic-skip oracle for DifferenceRS) ----------------
struct PayloadLCE {
    int T, v_off;
    SuffixLCE idx;
    PayloadLCE(const std::vector<std::optional<int64_t>>& baseline_labels,
               const std::vector<int64_t>& v_symbols)
        : T((int)v_symbols.size()), v_off((int)v_symbols.size() + 1), idx(build(baseline_labels, v_symbols)) {}
private:
    static std::vector<int> build(const std::vector<std::optional<int64_t>>& baseline_labels,
                                  const std::vector<int64_t>& v_symbols) {
        std::set<int64_t> vals(v_symbols.begin(), v_symbols.end());
        for (auto& x : baseline_labels) if (x) vals.insert(*x);
        std::unordered_map<int64_t, int> mp;
        int i = 2;
        for (int64_t x : vals) mp[x] = i++;
        int T = (int)v_symbols.size();
        std::vector<int> text;
        text.reserve(2 * T + 2);
        for (auto& x : baseline_labels) text.push_back(x ? mp[*x] : 1);
        text.push_back(0);
        for (int64_t x : v_symbols) text.push_back(mp[x]);
        text.push_back(T + (int)vals.size() + 7);
        return text;
    }
public:
    int lce(int t, int p) const {
        if (t >= T || p >= T) return 0;
        return std::min({idx.lcp_pos(t, v_off + p), T - t, T - p});
    }
};

// ---------------- DifferenceRS ----------------
struct DifferenceRS {
    int T;
    std::map<int, std::vector<int>> pos;        // shift -> sorted mismatch positions
    std::map<int, std::vector<double>> pref;    // shift -> prefix sums (double, like Python floats)

    DifferenceRS(const std::vector<RepairTrackTerm>& terms,
                 torch::Tensor grad_y, torch::Tensor v_bits,
                 torch::Tensor emb0_g, torch::Tensor emb1_g,
                 const std::vector<int64_t>& v_symbols,
                 const std::vector<int>& route_end) {
        T = (int)grad_y.size(0);
        std::vector<std::optional<int64_t>> baseline_labels(T);
        for (int t = 0; t < T; ++t) {
            int r = route_end[t];
            if (r >= 0) baseline_labels[t] = v_symbols[r + 1];
        }
        PayloadLCE sem(baseline_labels, v_symbols);

        std::map<int, std::vector<std::pair<int,int>>> grouped;
        for (auto& z : terms) grouped[z.shift].emplace_back(z.lo, z.hi);

        for (auto& [shift, ivs] : grouped) {
            std::sort(ivs.begin(), ivs.end());
            std::vector<std::pair<int,int>> merged;
            for (auto [lo, hi] : ivs) {
                if (merged.empty() || lo > merged.back().second + 1) merged.emplace_back(lo, hi);
                else if (hi > merged.back().second) merged.back().second = hi;
            }
            std::vector<int64_t> mism;
            for (auto [lo, hi] : merged) {
                int t = lo;
                while (t <= hi) {
                    int p = t + shift;
                    if (!(1 <= p && p < T)) throw std::runtime_error("DRS payload OOB");
                    if (baseline_labels[t] && *baseline_labels[t] == v_symbols[p]) {
                        int z = std::max(1, std::min(sem.lce(t, p), hi - t + 1));
                        t += z;
                    } else {
                        mism.push_back(t);
                        t += 1;
                    }
                }
            }
            std::vector<double> pr = {0.0};
            if (!mism.empty()) {
                // ATen replication of the reference torch expression (bitwise-equal reductions)
                auto tt = torch::tensor(mism, torch::kLong);
                auto pp = tt + shift;
                auto gy = grad_y.index_select(0, tt);
                auto cand_v = v_bits.index_select(0, pp).to(grad_y.dtype());
                auto cand = (cand_v * 2.0 - 1.0) * emb1_g;
                auto base = torch::empty_like(cand);
                std::vector<uint8_t> mmask;
                std::vector<int64_t> br;
                for (int t : mism) {
                    mmask.push_back(route_end[t] >= 0 ? 1 : 0);
                    br.push_back(route_end[t] >= 0 ? (int64_t)route_end[t] + 1 : 0);
                }
                auto matched_mask = torch::tensor(mmask, torch::kUInt8).to(torch::kBool);
                if (matched_mask.any().item<bool>()) {
                    auto brt = torch::tensor(br, torch::kLong);
                    auto bv = v_bits.index_select(0, brt).to(grad_y.dtype());
                    auto bmatched = (bv * 2.0 - 1.0) * emb1_g;
                    base.copy_(emb0_g.expand_as(base));
                    base.index_put_({matched_mask}, bmatched.index({matched_mask}));
                } else {
                    base.copy_(emb0_g.expand_as(base));
                }
                auto vals = (gy * (cand - base)).sum(-1).contiguous();
                AT_DISPATCH_FLOATING_TYPES(vals.scalar_type(), "drs_vals", [&] {
                    const scalar_t* vp = vals.data_ptr<scalar_t>();
                    for (int i = 0; i < (int)mism.size(); ++i)
                        pr.push_back(pr.back() + (double)vp[i]);
                });
            }
            pos[shift] = std::vector<int>(mism.begin(), mism.end());
            pref[shift] = std::move(pr);
        }
    }

    double query(int shift, int lo, int hi) const {
        if (lo > hi) return 0.0;
        auto it = pos.find(shift);
        static const std::vector<int> empty_pos;
        static const std::vector<double> empty_pref = {0.0};
        const std::vector<int>& xs = it != pos.end() ? it->second : empty_pos;
        const std::vector<double>& pr = pref.count(shift) ? pref.at(shift) : empty_pref;
        auto a = std::lower_bound(xs.begin(), xs.end(), lo) - xs.begin();
        auto b = std::upper_bound(xs.begin(), xs.end(), hi) - xs.begin();
        return pr[b] - pr[a];
    }
};

// ---------------- score cache ----------------
struct ScoreCache {
    torch::Tensor grad_y, v_bits, emb0_g, emb1_g;
    std::vector<std::unordered_map<int, double>> cache;
    explicit ScoreCache(int T, torch::Tensor gy, torch::Tensor vb, torch::Tensor e0, torch::Tensor e1)
        : grad_y(gy), v_bits(vb), emb0_g(e0), emb1_g(e1), cache(T) {}

    double route_score(int t, int route_end) {
        // replicate _route_score with ATen ops (torch.dot order preserved)
        auto grad_t = grad_y.select(0, t);
        if (route_end < 0) return at::dot(grad_t, emb0_g).item<double>();
        auto vidx = route_end + 1;
        auto sign = v_bits.select(0, vidx).to(grad_t.dtype()) * 2.0 - 1.0;
        return at::dot(grad_t, sign * emb1_g).item<double>();
    }

    double score(int t, int r) {
        auto& d = cache[t];
        auto it = d.find(r);
        if (it != d.end()) return it->second;
        double v = route_score(t, r);
        d[r] = v;
        return v;
    }
};

// ---------------- Q delete events (end_a==0 branch; other branches unreachable in this pipeline) --
static std::vector<std::vector<std::pair<int, double>>> delete_affine_events(
    const std::vector<std::vector<AffineDeleteRun>>& runs_by_t,
    const std::vector<double>& baseline, ScoreCache& sc,
    const std::vector<int>& route_end, const std::vector<int64_t>& v_symbols) {
    int T = (int)runs_by_t.size();
    std::vector<std::vector<std::pair<int, double>>> events(T + 1);
    for (int t = 0; t < T; ++t) {
        int r0 = route_end[t];
        std::optional<int64_t> lab0;
        if (r0 >= 0) lab0 = v_symbols[r0 + 1];
        for (const auto& z : runs_by_t[t]) {
            if (z.end_a != 0) throw std::runtime_error("Q delete run with end_a!=0 (unreachable in this pipeline)");
            int r = z.end_b;
            std::optional<int64_t> lab;
            if (r >= 0) lab = v_symbols[r + 1];
            double delta = (lab == lab0) ? 0.0 : sc.score(t, r) - baseline[t];
            if (delta != 0.0) {
                events[z.s_lo].emplace_back(t, delta);
                if (z.s_hi + 1 <= T) events[z.s_hi + 1].emplace_back(t, -delta);
            }
        }
    }
    return events;
}

// ---------------- K delete sweep: scalar events + diagonal (end_a==1) vector-Fenwick surfaces ----
// PeriodicDeleteRun has no producer in this pipeline -> not ported.
// Diag final reduction uses ATen .sum() on a from_blob view to preserve torch's summation order.
struct KDeleteSurfaceSweep {
    int T, D;
    torch::Tensor grad_y, v_bits, emb0_g, emb1_g;
    std::vector<std::vector<std::pair<int, double>>> scalar_events;
    // diag: c -> events[t] = (t_index, sign); insertion order across c must match Python dict order
    std::vector<int> diag_cs;                                   // first-appearance order
    std::unordered_map<int, int> diag_index;                    // c -> slot
    std::vector<std::vector<std::vector<std::pair<int, int>>>> diag_events;  // slot -> T+1 -> (t,sgn)
    struct VecFW {
        int n, D;
        std::vector<std::vector<float>> bitf;                   // float32 path
        std::vector<std::vector<double>> bitd;                  // float64 path
        VecFW() : n(0), D(0) {}
        VecFW(int n_, int D_, bool is_double) : n(n_), D(D_) {
            if (is_double) bitd.assign(n + 1, std::vector<double>(D + 1, 0.0));
            else bitf.assign(n + 1, std::vector<float>(D + 1, 0.0));
        }
    };
    std::vector<VecFW> diag_fw;                                 // per slot
    bool is_double;
    torch::Tensor vb_long_;
    const int64_t* vbp_;

    KDeleteSurfaceSweep(const std::vector<std::vector<AffineDeleteRun>>& runs_by_t,
                        const std::vector<double>& baseline, ScoreCache& sc,
                        torch::Tensor gy, torch::Tensor vb, torch::Tensor e0, torch::Tensor e1)
        : T((int)runs_by_t.size()), D((int)gy.size(1)),
          grad_y(gy), v_bits(vb), emb0_g(e0), emb1_g(e1),
          scalar_events(T + 1), fw(T),
          is_double(gy.scalar_type() == at::kDouble) {
        baseline_ = baseline;
        vb_long_ = vb.cpu().contiguous().to(torch::kLong);
        vbp_ = vb_long_.data_ptr<int64_t>();
        for (int t = 0; t < T; ++t) {
            for (const auto& z : runs_by_t[t]) {
                if (z.end_a == 0) {
                    double delta = sc.score(t, z.end_b) - baseline[t];
                    if (delta != 0.0) {
                        scalar_events[z.s_lo].emplace_back(t, delta);
                        if (z.s_hi + 1 <= T) scalar_events[z.s_hi + 1].emplace_back(t, -delta);
                    }
                    continue;
                }
                int Llo = z.len_a * z.s_lo + z.len_b, Lhi = z.len_a * z.s_hi + z.len_b;
                int p_lo = z.end_a * z.s_lo + z.end_b + 1, p_hi = z.end_a * z.s_hi + z.end_b + 1;
                if (z.end_a == 1 && std::min(Llo, Lhi) > 0 && 1 <= std::min(p_lo, p_hi) &&
                    std::max(p_lo, p_hi) < T) {
                    int c = z.end_b;
                    auto it = diag_index.find(c);
                    int slot;
                    if (it == diag_index.end()) {
                        slot = (int)diag_cs.size();
                        diag_index[c] = slot;
                        diag_cs.push_back(c);
                        diag_events.emplace_back(T + 1);
                        diag_fw.emplace_back(T, D, is_double);
                    } else slot = it->second;
                    diag_events[slot][z.s_lo].emplace_back(t, +1);
                    if (z.s_hi + 1 <= T) diag_events[slot][z.s_hi + 1].emplace_back(t, -1);
                    continue;
                }
                // uncommon exact affine slope: owner-point fallback
                for (int owner = z.s_lo; owner <= z.s_hi; ++owner) {
                    Route lr = z.route(owner);
                    double delta = sc.score(t, lr.second) - baseline[t];
                    if (delta != 0.0) {
                        scalar_events[owner].emplace_back(t, delta);
                        if (owner + 1 <= T) scalar_events[owner + 1].emplace_back(t, -delta);
                    }
                }
            }
        }
    }

    template <typename scalar_t>
    void fw_add(std::vector<VecFW>& fws, int slot, int i, const scalar_t* v) const {
        auto& bit = [&]() -> auto& { if constexpr (std::is_same_v<scalar_t, double>) return fws[slot].bitd; else return fws[slot].bitf; }();
        for (++i; i <= fws[slot].n; i += i & -i)
            for (int j = 0; j <= fws[slot].D; ++j) bit[i][j] += v[j];
    }

    void advance(int s) {
        for (auto& [t, d] : scalar_events[s]) fw.add(t, d);
        for (int slot = 0; slot < (int)diag_cs.size(); ++slot) {
            for (auto& [t, sgn] : diag_events[slot][s]) {
                if (is_double) {
                    std::vector<double> vec(D + 1);
                    const double* gy = grad_y.data_ptr<double>() + (int64_t)t * D;
                    for (int j = 0; j < D; ++j) vec[j] = gy[j] * (double)sgn;
                    vec[D] = baseline_[t] * (double)sgn;
                    fw_add<double>(diag_fw, slot, t, vec.data());
                } else {
                    std::vector<float> vec(D + 1);
                    const float* gy = grad_y.data_ptr<float>() + (int64_t)t * D;
                    for (int j = 0; j < D; ++j) vec[j] = gy[j] * (float)sgn;
                    // Python assigns a double (baseline[t]*sgn) into a float32 tensor: round after mult
                    vec[D] = (float)(baseline_[t] * (double)sgn);
                    fw_add<float>(diag_fw, slot, t, vec.data());
                }
            }
        }
    }

    // prefix over [0,i) -> out[D+1], elementwise sequential adds in same order as Python
    template <typename scalar_t>
    void fw_prefix(const VecFW& f, int i, scalar_t* out) const {
        for (int j = 0; j <= D; ++j) out[j] = scalar_t(0);
        if constexpr (std::is_same_v<scalar_t, double>) {
            for (; i > 0; i -= i & -i)
                for (int j = 0; j <= D; ++j) out[j] += f.bitd[i][j];
        } else {
            for (; i > 0; i -= i & -i)
                for (int j = 0; j <= D; ++j) out[j] += f.bitf[i][j];
        }
    }

    template <typename scalar_t>
    double range_sum_t(int lo, int hi, int s) const {
        double val = fw.range_sum(lo, hi);
        if (diag_cs.empty()) return val;
        auto opts = grad_y.options();
        std::vector<scalar_t> z1(D + 1), z2(D + 1), z(D + 1);
        std::vector<scalar_t> prod(D);
        for (int slot = 0; slot < (int)diag_cs.size(); ++slot) {
            fw_prefix<scalar_t>(diag_fw[slot], hi + 1, z1.data());
            fw_prefix<scalar_t>(diag_fw[slot], lo, z2.data());
            bool nonzero = false;
            for (int j = 0; j <= D; ++j) {
                z[j] = z1[j] - z2[j];
                if (z[j] != scalar_t(0)) nonzero = true;
            }
            if (!nonzero) continue;
            int c = diag_cs[slot];
            int p = s + c + 1;
            if (!(1 <= p && p < T)) throw std::runtime_error("delete surface payload OOB");
            // embed = (vb[p].to(dtype)*2.0-1.0)*e1 ; prod = z[:D]*embed ; sum via ATen (order!)
            const scalar_t* e1 = emb1_g.data_ptr<scalar_t>();
            for (int j = 0; j < D; ++j) {
                scalar_t bit = (scalar_t)vbp_[p * D + j];
                scalar_t sign = bit * (scalar_t)2.0 - (scalar_t)1.0;
                prod[j] = z[j] * (sign * e1[j]);
            }
            auto prodt = torch::from_blob(prod.data(), {D}, opts);
            // Python: (prod.sum() - z[D]) in float32, THEN float(...) -> double
            scalar_t ssum = prodt.sum().item<scalar_t>();
            val += (double)(scalar_t)(ssum - z[D]);
        }
        return val;
    }

    double range_sum(int lo, int hi, int s) const {
        if (lo > hi) return 0.0;
        if (is_double) return range_sum_t<double>(lo, hi, s);
        return range_sum_t<float>(lo, hi, s);
    }
    double total(int s) const { return range_sum(0, T - 1, s); }

    Fenwick fw;
    std::vector<double> baseline_;

    // init-order fixup: baseline_ must be set before advance(); see constructor below
    void set_baseline(const std::vector<double>& b) { baseline_ = b; }
};

// ---------------- zero-baseline surfaces contract ----------------
static torch::Tensor contract_zero_surfaces(
    const std::vector<int>& q, const std::vector<int>& k, const std::vector<int>& match_len,
    const std::vector<ZeroBaselineSurface>& surfaces,
    torch::Tensor v_bits, torch::Tensor grad_y, torch::Tensor emb0_g, torch::Tensor emb1_g, int D) {
    int T = (int)q.size();
    auto out = torch::zeros({T, D}, grad_y.dtype());
    if (surfaces.empty()) return out;
    std::set<int> targets;
    for (auto& z : surfaces) targets.insert(z.target_symbol);
    std::map<int, torch::Tensor> suffix, running;
    for (int c : targets) {
        suffix[c] = torch::zeros({T + 1, D}, grad_y.dtype());
        running[c] = torch::zeros({D}, grad_y.dtype());
    }
    for (int t = T - 1; t >= 0; --t) {
        int qt = q[t];
        if (match_len[t] == 0 && running.count(qt)) running[qt] = running[qt] + grad_y.select(0, t);
        for (int c : targets) suffix[c].select(0, t).copy_(running[c]);
    }
    std::map<std::pair<int,int>, const ZeroBaselineSurface*> by_key;
    for (auto& z : surfaces) by_key[{z.k_symbol, z.bit}] = &z;
    for (int s = 0; s < T; ++s) {
        int c = k[s];
        if (s + 1 >= T) continue;
        auto payload = (v_bits.select(0, s + 1).to(grad_y.dtype()) * 2.0 - 1.0) * emb1_g;
        auto diff = payload - emb0_g;
        for (int j = 0; j < D; ++j) {
            auto it = by_key.find({c, j});
            if (it == by_key.end()) continue;
            auto gsum = suffix[it->second->target_symbol].select(0, s + 1);
            out.select(0, s).select(0, j).copy_((gsum * diff).sum());
        }
    }
    return out;
}

// ---------------- main contract ----------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> contract_fields(
    const std::vector<std::vector<std::vector<RepairTrackTerm>>>& q_terms,
    const std::vector<std::vector<std::vector<RepairTrackTerm>>>& k_terms,
    const std::vector<std::vector<AffineDeleteRun>>& q_delete_runs_by_t,
    const std::vector<std::vector<AffineDeleteRun>>& k_delete_runs_by_t,
    const std::vector<ZeroBaselineSurface>& k_zero_surfaces,
    const std::vector<int>& q, const std::vector<int>& k,
    const std::vector<int>& ell, const std::vector<int>& route,
    torch::Tensor v_bits, torch::Tensor grad_y,
    torch::Tensor emb0_g, torch::Tensor emb1_g, int D) {
    int T = (int)q.size();
    ScoreCache sc(T, grad_y, v_bits, emb0_g, emb1_g);
    std::vector<double> baseline(T);
    for (int t = 0; t < T; ++t) baseline[t] = sc.score(t, route[t]);

    std::vector<int64_t> v_symbols(T);
    {
        auto vb_cpu = v_bits.cpu().contiguous();
        // generic pack: works for uint8/bool/int64 etc.
        auto vb8 = vb_cpu.to(torch::kLong);
        const int64_t* p = vb8.data_ptr<int64_t>();
        for (int t = 0; t < T; ++t) {
            int64_t x = 0;
            for (int j = 0; j < D; ++j)
                if (p[t * D + j]) x |= int64_t(1) << j;
            v_symbols[t] = x;
        }
    }

    std::vector<RepairTrackTerm> all_terms;
    for (const auto* side : {&q_terms, &k_terms})
        for (const auto& per_pos : *side)
            for (const auto& xs : per_pos)
                for (const auto& z : xs) all_terms.push_back(z);
    DifferenceRS drs(all_terms, grad_y, v_bits, emb0_g, emb1_g, v_symbols, route);

    auto qcred = torch::zeros({T, D}, grad_y.dtype());
    {
        auto events = delete_affine_events(q_delete_runs_by_t, baseline, sc, route, v_symbols);
        Fenwick fw(T);
        AT_DISPATCH_FLOATING_TYPES(qcred.scalar_type(), "qcred", [&] {
            scalar_t* qp = qcred.data_ptr<scalar_t>();
            for (int owner = 0; owner < T; ++owner) {
                for (auto& [t, d] : events[owner]) fw.add(t, d);
                double base = fw.total();
                for (int j = 0; j < D; ++j) {
                    double val = base;
                    for (const auto& z : q_terms[owner][j])
                        val += drs.query(z.shift, z.lo, z.hi) - fw.range_sum(z.lo, z.hi);
                    qp[owner * D + j] = (scalar_t)val;
                }
            }
        });
    }

    auto kcred = torch::zeros({T, D}, grad_y.dtype());
    {
        KDeleteSurfaceSweep ksweep(k_delete_runs_by_t, baseline, sc, grad_y, v_bits, emb0_g, emb1_g);
        AT_DISPATCH_FLOATING_TYPES(kcred.scalar_type(), "kcred", [&] {
            scalar_t* kp = kcred.data_ptr<scalar_t>();
            for (int owner = 0; owner < T; ++owner) {
                ksweep.advance(owner);
                double base = ksweep.total(owner);
                for (int j = 0; j < D; ++j) {
                    double val = base;
                    for (const auto& z : k_terms[owner][j])
                        val += drs.query(z.shift, z.lo, z.hi) - ksweep.range_sum(z.lo, z.hi, owner);
                    kp[owner * D + j] = (scalar_t)val;
                }
            }
        });
    }
    kcred += contract_zero_surfaces(q, k, ell, k_zero_surfaces, v_bits, grad_y, emb0_g, emb1_g, D);

    auto vcred = torch::zeros({T, D}, grad_y.dtype());
    AT_DISPATCH_FLOATING_TYPES(vcred.scalar_type(), "vcred", [&] {
        scalar_t* vp = vcred.data_ptr<scalar_t>();
        const scalar_t* gy = grad_y.data_ptr<scalar_t>();
        const scalar_t* e1 = emb1_g.data_ptr<scalar_t>();
        auto vb_cpu = v_bits.cpu().contiguous().to(torch::kLong);
        const int64_t* vbp = vb_cpu.data_ptr<int64_t>();
        for (int t = 0; t < T; ++t) {
            int r = route[t];
            if (r < 0) continue;
            int owner = r + 1;
            for (int j = 0; j < D; ++j) {
                // sign = v_bits[owner].to(dtype)*2.0-1.0 ; vcred[owner] += grad_y[t]*(-2.0*sign*emb1)
                scalar_t sign = (scalar_t)vbp[owner * D + j] * (scalar_t)2.0 - (scalar_t)1.0;
                scalar_t tmp = ((scalar_t)-2.0 * sign) * e1[j];
                vp[owner * D + j] += gy[t * D + j] * tmp;
            }
        }
    });
    return {qcred, kcred, vcred};
}

}  // namespace rosa
