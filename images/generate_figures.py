#!/usr/bin/env python
"""Regenerate every SVG figure of ALGORITHM.md from real code runs.

Usage (from anywhere):

    python images/generate_figures.py              # generate en + zh figures
    python images/generate_figures.py --lang en    # English only
    python images/generate_figures.py --lang zh    # Chinese only (uses the
                                                   # zh entries of STRINGS)

Outputs (next to this script):

    images/figNN-name.svg          English
    images/figNN-name.zh-CN.svg    Chinese (zh strings; until translated they
                                   intentionally mirror the English text)

Every number drawn in the figures is computed on the spot by importing
`hard_qkv_rosa_explained.py` from the repository root (this script's parent
directory); nothing is transcribed by hand.  All layout is object-bound: see
`images/svg_layout.py` for the layout engine and the validator that runs on
every render (connector/object collisions, enclose containment, overlaps —
any violation aborts the run with an error).

The Chinese translations of the in-figure strings live in the `zh` entries of
the STRINGS table below and are maintained by a translation pass; the switch
mechanism is fully functional today.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

import hard_qkv_rosa_explained as H  # noqa: E402

from svg_layout import (  # noqa: E402
    Scene, PALETTE, LINE,
)

OUT_DIR = Path(__file__).resolve().parent

BLUE = PALETTE["blue"]["stroke"]
RED = PALETTE["red"]["stroke"]
GRAY = PALETTE["gray"]["stroke"]
GREEN = PALETTE["green"]["stroke"]

# --------------------------------------------------------------------------
# String table.  Every in-figure string goes through here.
# {placeholders} are filled with measured values at build time.
# --------------------------------------------------------------------------
STRINGS = {
    # fig00 — legend
    "fig0.title": {"en": "Legend: colors and connectors used in all figures",
                   "zh": "图例：所有图中使用的颜色与连线语义"},
    "fig0.blue": {"en": "forward / baseline data", "zh": "forward / baseline data"},
    "fig0.red": {"en": "credit flow / changed object", "zh": "credit flow / changed object"},
    "fig0.gray": {"en": "frozen content", "zh": "frozen content"},
    "fig0.green": {"en": "match / hit", "zh": "match / hit"},
    "fig0.flow_from": {"en": "from", "zh": "from"},
    "fig0.flow_to": {"en": "to", "zh": "to"},
    "fig0.flow": {"en": "credit flow / dependency", "zh": "credit flow / dependency"},

    # fig01 — running instance
    "fig1.title": {"en": "Figure 1 — the running instance (T = 8, D = 1)",
                   "zh": "Figure 1 — the running instance (T = 8, D = 1)"},
    "fig1.pos": {"en": "position t", "zh": "position t"},
    "fig1.q": {"en": "q[t]  (probe)", "zh": "q[t]  (probe)"},
    "fig1.k": {"en": "k[t]  (memory)", "zh": "k[t]  (memory)"},
    "fig1.v": {"en": "v_bits[t]", "zh": "v_bits[t]"},
    "fig1.ell": {"en": "ell[t]", "zh": "ell[t]"},
    "fig1.route": {"en": "route[t]", "zh": "route[t]"},
    "fig1.payload": {"en": "payload src = V[route[t]+1]", "zh": "payload src = V[route[t]+1]"},
    "fig1.reuse": {"en": "t=7 reuses V[4]:  route[7]+1 = 4", "zh": "t=7 reuses V[4]:  route[7]+1 = 4"},
    "fig1.note": {"en": "the backward asks, for every bit position u of q, k, v_bits:  flip [u] -> how do ell, route, and y change? -> credit[u]",
                  "zh": "the backward asks, for every bit position u of q, k, v_bits:  flip [u] -> how do ell, route, and y change? -> credit[u]"},

    # fig03 — structural zeros
    "fig3.title": {"en": "Figure 3 — three structural zeros of the payload rule (running instance)",
                   "zh": "Figure 3 — three structural zeros of the payload rule (running instance)"},
    "fig3.q": {"en": "q[t]", "zh": "q[t]"},
    "fig3.k": {"en": "k[t]", "zh": "k[t]"},
    "fig3.v": {"en": "v_bits[t]", "zh": "v_bits[t]"},
    "fig3.v0": {"en": "V[0] is never read:  route[t] >= 0  =>  payload index = route[t]+1 >= 1",
                "zh": "V[0] is never read:  route[t] >= 0  =>  payload index = route[t]+1 >= 1"},
    "fig3.kT": {"en": "K[T-1] is never an endpoint:  an endpoint r must satisfy r < t <= T-1",
                "zh": "K[T-1] is never an endpoint:  an endpoint r must satisfy r < t <= T-1"},
    "fig3.q0": {"en": "q[0] never matters:  it could only serve a full-prefix match ending at r = t, which is not a legal endpoint",
                "zh": "q[0] never matters:  it could only serve a full-prefix match ending at r = t, which is not a legal endpoint"},

    # fig06 — naive vs fast
    "fig6.title": {"en": "Figure 6 — same 24 bits, same answers: naive recomputation vs compiled surfaces (running instance)",
                   "zh": "Figure 6 — same 24 bits, same answers: naive recomputation vs compiled surfaces (running instance)"},
    "fig6.naive_title": {"en": "NAIVE (backward_bruteforce)", "zh": "NAIVE (backward_bruteforce)"},
    "fig6.naive_1": {"en": "24 bit flips x full forward O(T^3)", "zh": "24 bit flips x full forward O(T^3)"},
    "fig6.naive_2": {"en": "each forward re-derives ALL of:", "zh": "each forward re-derives ALL of:"},
    "fig6.naive_3": {"en": "  every suffix comparison", "zh": "  every suffix comparison"},
    "fig6.naive_4": {"en": "  every tie-break", "zh": "  every tie-break"},
    "fig6.naive_5": {"en": "  every payload read", "zh": "  every payload read"},
    "fig6.naive_total": {"en": "total: 24 x 8^3 = {n} suffix steps", "zh": "total: 24 x 8^3 = {n} suffix steps"},
    "fig6.fast_title": {"en": "THIS ALGORITHM", "zh": "THIS ALGORITHM"},
    "fig6.fast_1": {"en": "one baseline forward (fast, Part 2)", "zh": "one baseline forward (fast, Part 2)"},
    "fig6.fast_2": {"en": "Q-side deletion surfaces: {runs} affine runs cover all {pairs} owner-output deletion events",
                    "zh": "Q-side deletion surfaces: {runs} affine runs cover all {pairs} owner-output deletion events"},
    "fig6.fast_3": {"en": "K-side deletion surfaces: {runs} affine runs cover all {pairs} owner-output deletion events",
                    "zh": "K-side deletion surfaces: {runs} affine runs cover all {pairs} owner-output deletion events"},
    "fig6.fast_4": {"en": "{p} one-bit (q,k) pairs -> {terms} repair terms on t-intervals",
                    "zh": "{p} one-bit (q,k) pairs -> {terms} repair terms on t-intervals"},
    "fig6.fast_5": {"en": "closed-form V credit ({n} adds)", "zh": "closed-form V credit ({n} adds)"},
    "fig6.fast_total": {"en": "total: {terms} compiled terms, each carrying a grad_y-weighted range sum",
                        "zh": "total: {terms} compiled terms, each carrying a grad_y-weighted range sum"},
    "fig6.vs": {"en": "same 24 answers, exactly", "zh": "same 24 answers, exactly"},

    # fig07 — pipeline
    "fig7.title": {"en": "Figure 7 — the pipeline: from symbol streams to grad_z",
                   "zh": "Figure 7 — the pipeline: from symbol streams to grad_z"},
    "fig7.b0": {"en": "q, k  (symbol streams)", "zh": "q, k  (symbol streams)"},
    "fig7.b1a": {"en": "Part 2 — fast forward", "zh": "Part 2 — fast forward"},
    "fig7.b1b": {"en": "suffix array + LCE + static certificates", "zh": "suffix array + LCE + static certificates"},
    "fig7.b2": {"en": "ell[t], route[t]  (baseline)", "zh": "ell[t], route[t]  (baseline)"},
    "fig7.b3a": {"en": "Part 4 — rebuild workbench", "zh": "Part 4 — rebuild workbench"},
    "fig7.b3b": {"en": "bidirectional suffix indexes -> CausalCutSuffixIndex", "zh": "bidirectional suffix indexes -> CausalCutSuffixIndex"},
    "fig7.s1a": {"en": "Part 6 — Q-side deletion surfaces", "zh": "Part 6 — Q-side deletion surfaces"},
    "fig7.s1b": {"en": "(latest-occurrence heads)", "zh": "(latest-occurrence heads)"},
    "fig7.s2a": {"en": "Part 5 — K-side deletion oracles A/H", "zh": "Part 5 — K-side deletion oracles A/H"},
    "fig7.s2b": {"en": "-> max(A, H) surfaces", "zh": "-> max(A, H) surfaces"},
    "fig7.s3a": {"en": "Part 12/14 — shared one-bit repair bridges", "zh": "Part 12/14 — shared one-bit repair bridges"},
    "fig7.s3b": {"en": "(sparse / diagonal)", "zh": "(sparse / diagonal)"},
    "fig7.s4a": {"en": "Part 10/13 — surface-conditioned repair compiler", "zh": "Part 10/13 — surface-conditioned repair compiler"},
    "fig7.s4b": {"en": "(maximal-run square certificates)", "zh": "(maximal-run square certificates)"},
    "fig7.b5": {"en": "RepairIR:  q_terms, k_terms, q_delete_runs, k_delete_runs, zeros",
                "zh": "RepairIR:  q_terms, k_terms, q_delete_runs, k_delete_runs, zeros"},
    "fig7.b6": {"en": "Part 8 — numerical contraction (no string queries left)",
                "zh": "Part 8 — numerical contraction (no string queries left)"},
    "fig7.b7": {"en": "credit_q, credit_k, credit_v", "zh": "credit_q, credit_k, credit_v"},
    "fig7.b8": {"en": "Part 11 — orient = 1 - 2*bit", "zh": "Part 11 — orient = 1 - 2*bit"},
    "fig7.b9": {"en": "grad_z  (identity / bernoulli)", "zh": "grad_z  (identity / bernoulli)"},
    "fig7.rail": {"en": "baseline ell/route", "zh": "baseline ell/route"},
    "fig7.grady": {"en": "grad_y", "zh": "grad_y"},

    # fig09 — Q-side latest heads
    "fig9.title": {"en": "Figure 9 — heads of t = 6 (ell[6] = 6) and their conversion to owner-axis runs",
                   "zh": "Figure 9 — heads of t = 6 (ell[6] = 6) and their conversion to owner-axis runs"},
    "fig9.hypL": {"en": "hypothetical length L", "zh": "hypothetical length L"},
    "fig9.latest": {"en": "latest endpoint", "zh": "latest endpoint"},
    "fig9.onehead": {"en": "constant endpoint: ONE head covers L_lo={lo}..L_hi={hi}, endpoint={e}",
                     "zh": "constant endpoint: ONE head covers L_lo={lo}..L_hi={hi}, endpoint={e}"},
    "fig9.owner": {"en": "owner s", "zh": "owner s"},
    "fig9.Ls": {"en": "L(s) = t - s", "zh": "L(s) = t - s"},
    "fig9.rs": {"en": "r(s)", "zh": "r(s)"},
    "fig9.run1": {"en": "AffineDeleteRun(t=6, s in [{lo},{hi}], len_a={la}, len_b={lb}, end_a={ea}, end_b={eb})",
                  "zh": "AffineDeleteRun(t=6, s in [{lo},{hi}], len_a={la}, len_b={lb}, end_a={ea}, end_b={eb})"},
    "fig9.unmatched": {"en": "s = t: cut the last bit -> unmatched (r = -1)",
                       "zh": "s = t: cut the last bit -> unmatched (r = -1)"},

    # fig10 — K deletion re-index
    "fig10.title": {"en": "Figure 10 — delete K[2] and re-answer t = 3: re-index and re-match",
                    "zh": "Figure 10 — delete K[2] and re-answer t = 3: re-index and re-match"},
    "fig10.before": {"en": "BEFORE (baseline)", "zh": "BEFORE (baseline)"},
    "fig10.after": {"en": "AFTER (K[2] treated as deleted)", "zh": "AFTER (K[2] treated as deleted)"},
    "fig10.hole": {"en": "hole at s = 2: no window may contain it",
                   "zh": "hole at s = 2: no window may contain it"},
    "fig10.broken": {"en": "baseline window [0,2] contains s = 2 -> broken",
                     "zh": "baseline window [0,2] contains s = 2 -> broken"},
    "fig10.baseinfo": {"en": "Q[:4] = [0,0,0,0];  baseline route(3) = (3, 2)",
                       "zh": "Q[:4] = [0,0,0,0];  baseline route(3) = (3, 2)"},
    "fig10.origidx": {"en": "cells keep their original K indices",
                      "zh": "cells keep their original K indices"},
    "fig10.win": {"en": "window [0,1]: L = 2, endpoint r = 1  <- wins",
                  "zh": "window [0,1]: L = 2, endpoint r = 1  <- wins"},
    "fig10.alt": {"en": "window [3,3]: L = 1, endpoint r = 3",
                  "zh": "window [3,3]: L = 1, endpoint r = 3"},
    "fig10.result": {"en": "post-delete route(3, s=2) = (2, 1)   — matches the brute-force flip (ell 3->2, route 2->1)",
                     "zh": "post-delete route(3, s=2) = (2, 1)   — matches the brute-force flip (ell 3->2, route 2->1)"},

    # fig11 — repair threshold polarity
    "fig11.title": {"en": "Figure 11 — repair threshold surface at t = 7 (ell = 1, route = 3, window = [3,3])",
                    "zh": "Figure 11 — repair threshold surface at t = 7 (ell = 1, route = 3, window = [3,3])"},
    "fig11.owner": {"en": "owner s", "zh": "owner s"},
    "fig11.window": {"en": "baseline window [3,3]", "zh": "baseline window [3,3]"},
    "fig11.strict": {"en": "s in [0,2] (left of window): repair must be len >= 2 (strict) — an equal-length repair ends before r0 = 3 and LOSES the tie",
                     "zh": "s in [0,2] (left of window): repair must be len >= 2 (strict) — an equal-length repair ends before r0 = 3 and LOSES the tie"},
    "fig11.incl": {"en": "s in [3,6] (window + right): len >= 1 suffices (inclusive) — an equal-length repair ends at or after r0 and WINS the tie",
                   "zh": "s in [3,6] (window + right): len >= 1 suffices (inclusive) — an equal-length repair ends at or after r0 and WINS the tie"},

    # fig12 — shared bridge geometry
    "fig12.title": {"en": "Figure 12 — the bridge behind the k[2] flip: pair (q_pos=5, k_pos=2, bit=0)",
                    "zh": "Figure 12 — the bridge behind the k[2] flip: pair (q_pos=5, k_pos=2, bit=0)"},
    "fig12.qrow": {"en": "q (positions 3..7)", "zh": "q (positions 3..7)"},
    "fig12.krow": {"en": "k (positions 0..4)", "zh": "k (positions 0..4)"},
    "fig12.left": {"en": "left = 2: equal context q[3..4] == k[0..1]",
                   "zh": "left = 2: equal context q[3..4] == k[0..1]"},
    "fig12.flip": {"en": "differ in bit 0 — flip me", "zh": "differ in bit 0 — flip me"},
    "fig12.right": {"en": "right = 0: q[6] != k[3]", "zh": "right = 0: q[6] != k[3]"},
    "fig12.window": {"en": "after the flip: created match window q[3..5] == k'[0..2] == [0,0,1]",
                     "zh": "after the flip: created match window q[3..5] == k'[0..2] == [0,0,1]"},
    "fig12.r1": {"en": "d = t - 5;  lifetime t in [5, 5+right] = [5,5];  route_at(5) = (left+1+0, k_pos+0) = (3, 2)",
                 "zh": "d = t - 5;  lifetime t in [5, 5+right] = [5,5];  route_at(5) = (left+1+0, k_pos+0) = (3, 2)"},
    "fig12.r2": {"en": "shift = k_pos - q_pos + 1 = -2  ->  candidate payload V[t-2] = V[3]  (brute force: t=5 ell 5->3, route 4->2)",
                 "zh": "shift = k_pos - q_pos + 1 = -2  ->  candidate payload V[t-2] = V[3]  (brute force: t=5 ell 5->3, route 4->2)"},

    # fig13 — bridge envelope
    "fig13.title": {"en": "Figure 13 — owner s=2, bit 0: two bridges, first-win, and the surviving terms",
                    "zh": "Figure 13 — owner s=2, bit 0: two bridges, first-win, and the surviving terms"},
    "fig13.t": {"en": "t", "zh": "t"},
    "fig13.rowA": {"en": "bridge A (q_pos=5, k_pos=2, left=2, right=0), shift=-2",
                   "zh": "bridge A (q_pos=5, k_pos=2, left=2, right=0), shift=-2"},
    "fig13.rowB": {"en": "bridge B (q_pos=6, k_pos=2, left=0, right=1), shift=-3",
                   "zh": "bridge B (q_pos=6, k_pos=2, left=0, right=1), shift=-3"},
    "fig13.noteA": {"en": "t=5: bridge (3,2) vs delete route (2,4) -> 3 > 2, WINS -> term [5,5]",
                    "zh": "t=5: bridge (3,2) vs delete route (2,4) -> 3 > 2, WINS -> term [5,5]"},
    "fig13.noteB": {"en": "t=6: bridge (1,2) vs delete route (3,4) -> loses, pruned;   t=7: bridge (2,3) vs baseline (1,3) -> 2 > 1, WINS -> term [7,7]",
                    "zh": "t=6: bridge (1,2) vs delete route (3,4) -> loses, pruned;   t=7: bridge (2,3) vs baseline (1,3) -> 2 > 1, WINS -> term [7,7]"},
    "fig13.lifeA": {"en": "lifetime [5,5]", "zh": "lifetime [5,5]"},
    "fig13.lifeB": {"en": "lifetime [6,7]", "zh": "lifetime [6,7]"},
    "fig13.result": {"en": "k_terms[2][0] = [RepairTrackTerm(shift=-2, lo=5, hi=5), RepairTrackTerm(shift=-3, lo=7, hi=7)]   (the [7,7] term is later skipped by the payload LCE: route change, no credit)",
                     "zh": "k_terms[2][0] = [RepairTrackTerm(shift=-2, lo=5, hi=5), RepairTrackTerm(shift=-3, lo=7, hi=7)]   (the [7,7] term is later skipped by the payload LCE: route change, no credit)"},

    # fig14 — run certificates
    "fig14.title": {"en": "Figure 14 — T = 6 certificate instance:  k = [0,0,1,0,0,0],  ell = [0,1,2,2,3,4],  route = [-1,0,1,1,2,3]",
                    "zh": "Figure 14 — T = 6 certificate instance:  k = [0,0,1,0,0,0],  ell = [0,1,2,2,3,4],  route = [-1,0,1,1,2,3]"},
    "fig14.krow": {"en": "k", "zh": "k"},
    "fig14.run1": {"en": "run [0,1], p = 1", "zh": "run [0,1], p = 1"},
    "fig14.run2": {"en": "run [3,5], p = 1:  hi+1 = 6 = T -> reaches the end -> NO certificate",
                   "zh": "run [3,5], p = 1:  hi+1 = 6 = T -> reaches the end -> NO certificate"},
    "fig14.breaker": {"en": "breaker at hi+1 = 2: k[1]=0 vs k[2]=1, xor = 1 = 2^0 -> exactly bit 0",
                      "zh": "breaker at hi+1 = 2: k[1]=0 vs k[2]=1, xor = 1 = 2^0 -> exactly bit 0"},
    "fig14.owner": {"en": "owner s = hi-p+1 = 1", "zh": "owner s = hi-p+1 = 1"},
    "fig14.cert": {"en": "_KRunRepairCertificate(owner=1, period=1, bit=0, m_lo=2, m_hi=2)",
                   "zh": "_KRunRepairCertificate(owner=1, period=1, bit=0, m_lo=2, m_hi=2)"},
    "fig14.stats": {"en": "index: runs={runs}, certs={certs};  compile: certificate queries={cq} (hits={ch}), boundary={bq} (hits={bh}), fallback owner queries={fb}",
                    "zh": "index: runs={runs}, certs={certs};  compile: certificate queries={cq} (hits={ch}), boundary={bq} (hits={bh}), fallback owner queries={fb}"},
    "fig14.hit": {"en": "t=4: baseline route (3,2);  anchor M = 2 == route[4], H-ramp owner interval hits s=1 -> certificate query HITS",
                  "zh": "t=4: baseline route (3,2);  anchor M = 2 == route[4], H-ramp owner interval hits s=1 -> certificate query HITS"},

    # fig15 — shadow repair
    "fig15.title": {"en": "Figure 15 — the repair itself on t = 4: flip k[1] and the match casts a shadow one period left",
                    "zh": "Figure 15 — the repair itself on t = 4: flip k[1] and the match casts a shadow one period left"},
    "fig15.k": {"en": "k (before)", "zh": "k (before)"},
    "fig15.kp": {"en": "k' (after flip)", "zh": "k' (after flip)"},
    "fig15.fliplab": {"en": "flip k[1]: 0 -> 1   (the certificate's owner, bit 0)",
                      "zh": "flip k[1]: 0 -> 1   (the certificate's owner, bit 0)"},
    "fig15.before": {"en": "BEFORE: baseline match", "zh": "BEFORE: baseline match"},
    "fig15.beforeEq": {"en": "Q[2..4] == K[0..2] == [0,0,1]   (ell = 3, r = 2, anchor M = 2)",
                       "zh": "Q[2..4] == K[0..2] == [0,0,1]   (ell = 3, r = 2, anchor M = 2)"},
    "fig15.after": {"en": "AFTER: shadow one period left", "zh": "AFTER: shadow one period left"},
    "fig15.afterEq": {"en": "Q[3..4] == K'[0..1] == [0,1]   new match: length 2, endpoint M-p = 1",
                      "zh": "Q[3..4] == K'[0..1] == [0,1]   new match: length 2, endpoint M-p = 1"},
    "fig15.arrow": {"en": "flip shifts the window", "zh": "flip shifts the window"},
    "fig15.cmp": {"en": "compare:  deletion baseline route(4, s=1) = (1, 2)   vs   repair route (2, 1)  ->  2 > 1, repair WINS",
                  "zh": "compare:  deletion baseline route(4, s=1) = (1, 2)   vs   repair route (2, 1)  ->  2 > 1, repair WINS"},
    "fig15.payload": {"en": "payload:  candidate V[M-p+1] = V[2] = 1  ==  baseline payload V[3] = 1  ->  y[4] unchanged (the term is skipped by the payload LCE of section 9)",
                      "zh": "payload:  candidate V[M-p+1] = V[2] = 1  ==  baseline payload V[3] = 1  ->  y[4] unchanged (the term is skipped by the payload LCE of section 9)"},
    "fig15.term": {"en": "compiled term:  owner 1, bit 0:  RepairTrackTerm(shift=-2, lo=4, hi=4)   (shift -2: candidate payload is V[t-2] = V[2])",
                   "zh": "compiled term:  owner 1, bit 0:  RepairTrackTerm(shift=-2, lo=4, hi=4)   (shift -2: candidate payload is V[t-2] = V[2])"},
}


def S(key, lang, **fmt):
    text = STRINGS[key][lang]
    return text.format(**fmt) if fmt else text


# --------------------------------------------------------------------------
# Measured data — everything below is computed by importing the repository's
# reference implementation, never transcribed by hand.
# --------------------------------------------------------------------------
def compute_data():
    data = {}

    # ---- running instance (T=8, D=1) ----------------------------------
    T, D = 8, 1
    q = [0, 0, 0, 0, 0, 1, 1, 0]
    k = [0, 0, 0, 0, 1, 1, 1, 0]
    torch.manual_seed(7)
    v_bits = torch.randint(0, 2, (T, D), dtype=torch.uint8)
    torch.manual_seed(11)
    grad_y = torch.randn(T, D)
    torch.manual_seed(123)
    emb0 = torch.randn(D)
    emb1 = torch.randn(D)

    y, ell, route = H.forward_naive(q, k, v_bits, emb0, emb1, D)
    assert ell == [0, 1, 2, 3, 4, 5, 6, 1] and route == [-1, 0, 1, 2, 3, 4, 5, 3]
    bq, bk, bv = H.backward_bruteforce(q, k, v_bits, grad_y, emb0, emb1, D)
    fq, fk, fv, stats, ell2, route2, ir = H.exact_stream_bit_credits(
        q, k, v_bits, grad_y, emb0, emb1, D)
    assert ell2 == ell and route2 == route
    assert float((bq - fq).abs().max()) < 1e-5
    assert float((bk - fk).abs().max()) < 1e-5
    assert float((bv - fv).abs().max()) < 1e-5
    assert stats.repair_backend == "shared_sparse"

    index = H.CausalCutSuffixIndex(q, k)
    heads = H._compile_q_latest_heads(index, ell)
    qdel = H._build_q_delete_from_latest_heads(ell, heads)
    kdel = H.KDeleteCutOracle(index, ell, route)

    data.update(T=T, D=D, q=q, k=k, v=[int(x) for x in v_bits.flatten()],
                ell=ell, route=route, stats=stats, ir=ir, heads=heads,
                qdel=qdel, kdel=kdel,
                n_matched=sum(1 for e in ell if e > 0))

    # fig9: heads and owner-axis runs for t=6
    h6 = heads[6]
    assert len(h6) == 1 and (h6[0].L_lo, h6[0].L_hi, h6[0].endpoint) == (1, 5, 5)
    r6 = qdel[6]
    assert len(r6) == 2
    assert (r6[0].s_lo, r6[0].s_hi, r6[0].len_a, r6[0].len_b,
            r6[0].end_a, r6[0].end_b) == (1, 5, -1, 6, 0, 5)
    assert r6[1].end_b == -1

    # fig10/fig11: K-side oracle
    assert kdel.route(3, 2) == (2, 1)
    rr7 = kdel.repair_runs[7]
    assert [(r.s_lo, r.s_hi, r.strict) for r in rr7] == [(0, 2, True), (3, 6, False)]

    # fig13: the two bridges owned by s=2 and the surviving k_terms
    kt2 = ir.k_terms[2][0]
    assert [(t.shift, t.lo, t.hi) for t in kt2] == [(-2, 5, 5), (-3, 7, 7)]

    # ---- T=6 certificate instance (section 8) --------------------------
    T6, D6 = 6, 1
    q6 = [0, 0, 0, 0, 1, 0]
    k6 = [0, 0, 1, 0, 0, 0]
    v6 = torch.tensor([[1], [0], [1], [1], [0], [1]], dtype=torch.uint8)
    torch.manual_seed(5)
    g6 = torch.randn(T6, D6)
    e06 = torch.tensor([-0.4])
    e16 = torch.tensor([0.9])
    y6, ell6, route6 = H.forward_naive(q6, k6, v6, e06, e16, D6)
    assert ell6 == [0, 1, 2, 2, 3, 4] and route6 == [-1, 0, 1, 1, 2, 3]
    index6 = H.CausalCutSuffixIndex(q6, k6)
    runs6 = H._enumerate_k_runs_from_existing_lce(index6)
    assert runs6 == [(0, 1, 1), (3, 5, 1)]
    st6 = H.RepairStats()
    certs6 = H._KRunRepairCertificateIndex(index6, D6, st6)
    assert len(certs6.certs) == 1
    cert = certs6.certs[0]
    assert (cert.owner, cert.period, cert.bit, cert.m_lo, cert.m_hi) == (1, 1, 0, 2, 2)
    kdel6 = H.KDeleteCutOracle(index6, ell6, route6)
    st6b = H.RepairStats()
    kterms6 = H._compile_k_surface_conditioned(D6, index6, kdel6, ell6, st6b)
    assert [(t.shift, t.lo, t.hi) for t in kterms6[1][0]] == [(-2, 4, 4)]
    assert kdel6.route(4, 1) == (1, 2)

    data.update(T6=T6, q6=q6, k6=k6, v6=[int(x) for x in v6.flatten()],
                ell6=ell6, route6=route6, runs6=runs6, cert6=cert,
                cert_stats=st6b)
    return data


# --------------------------------------------------------------------------
# figure builders
# --------------------------------------------------------------------------
CW, CH = 34, 30          # default grid cell size
FS = 13                  # default font size


def _caption(sc, key, lang, **fmt):
    cap = sc.label("caption", S(key, lang, **fmt), font_size=14, bold=True,
                   anchor="start")
    cap.at(0, 0)
    return cap


def _col_labels(sc, name, grid, values, style_fn=None):
    """One label per column, centered on the column's cell."""
    out = []
    for c, v in enumerate(values):
        lab = sc.label(f"{name}[{c}]", str(v),
                       color=(style_fn(c) if style_fn else "#111827"))
        lab.center_x_on(grid.cell(0, c))
        out.append(lab)
    return out


def fig00_legend(data, lang):
    sc = Scene("fig00-legend", title=S("fig0.title", lang))
    cap = _caption(sc, "fig0.title", lang)
    b1 = sc.box("sw.blue", S("fig0.blue", lang), style="blue")
    b1.at(0, 0).below(cap, 12)
    b2 = sc.box("sw.red", S("fig0.red", lang), style="red")
    b2.right_of(b1, 16).align_top(b1)
    b3 = sc.box("sw.gray", S("fig0.gray", lang), style="gray", dashed=True)
    b3.right_of(b2, 16).align_top(b1)
    b4 = sc.box("sw.green", S("fig0.green", lang), style="green")
    b4.right_of(b3, 16).align_top(b1)
    a = sc.box("flow.from", S("fig0.flow_from", lang), style="ink")
    a.below(b1, 26).align_left(b1)
    b = sc.box("flow.to", S("fig0.flow_to", lang), style="ink")
    b.right_of(a, 210).center_y_on(a)
    sc.connect(a, "right", b, "left", color=RED, label=S("fig0.flow", lang))
    return sc


def fig01_running_instance(data, lang):
    T = data["T"]
    sc = Scene("fig01-running-instance", title=S("fig1.title", lang),
               desc="Running instance: q/k/v_bits grids with ell, route and "
                    "payload rows; t=7 reuses V[4].")
    cap = _caption(sc, "fig1.title", lang)

    g = sc.grid("inputs", 3, T, cw=CW, ch=CH)
    g.at(0, 0).below(cap, 52)
    for c in range(T):
        g.cell(0, c, str(data["q"][c]), "blue")
        g.cell(1, c, str(data["k"][c]), "blue")
        g.cell(2, c, str(data["v"][c]), "blue")

    # column indices above the grid
    idx = []
    for c in range(T):
        lab = sc.label(f"idx{c}", str(c), font_size=12, color="#6b7280")
        lab.center_x_on(g.cell(0, c)).above(g.cell(0, c), 4)
        idx.append(lab)
    poslab = sc.label("pos", S("fig1.pos", lang), font_size=12, color="#6b7280")
    poslab.center_x_on([g.cell(0, 0), g.cell(0, T - 1)]).above(idx[0], 2)

    # row tags to the left
    for r, key in enumerate(("fig1.q", "fig1.k", "fig1.v")):
        tag = sc.label(f"tag{r}", S(key, lang), anchor="end")
        tag.left_of(g.cell(r, 0), 8).center_y_on(g.cell(r, 0))

    # ell / route / payload rows below the grid
    rows = [("ell", S("fig1.ell", lang), data["ell"], None),
            ("route", S("fig1.route", lang), data["route"], None),
            ("pay", S("fig1.payload", lang),
             ["emb0"] + [str(data["route"][t] + 1) for t in range(1, T)],
             lambda c: GREEN if c in (4, 7) else ("#6b7280" if c == 0 else "#111827"))]
    prev = g
    pay_labels = None
    for name, tag_text, values, color_fn in rows:
        labs = _col_labels(sc, name, g, values, color_fn)
        for lab in labs:
            lab.below(prev, 6)
        tag = sc.label(f"tag.{name}", tag_text, anchor="end", font_size=12)
        tag.left_of(g.cell(0, 0), 8).center_y_on(labs[0])
        prev = labs[0]
        if name == "pay":
            pay_labels = labs

    # the reuse arrow: t=7 reads the same V[4] as t=4
    lane_gap = 16
    sc.connect(pay_labels[7], "bottom", pay_labels[4], "bottom",
               color=GREEN,
               waypoints=lambda: [(pay_labels[7].bounds.cx,
                                   pay_labels[7].bounds.y1 + lane_gap),
                                  (pay_labels[4].bounds.cx,
                                   pay_labels[4].bounds.y1 + lane_gap)],
               label=S("fig1.reuse", lang), label_side="below")

    note = sc.label("note", S("fig1.note", lang), anchor="start",
                    color="#374151")
    note.at(0, 0).below(pay_labels[0], lane_gap + 30)
    return sc


def fig03_structural_zeros(data, lang):
    T = data["T"]
    sc = Scene("fig03-structural-zeros", title=S("fig3.title", lang),
               desc="q[0], K[T-1] and V[0] are structurally zero-credit.")
    cap = _caption(sc, "fig3.title", lang)

    def row(name, values, dead_col, note_key, note_color=GRAY):
        g = sc.grid(name, 1, T, cw=CW, ch=CH)
        for c in range(T):
            g.cell(0, c, str(values[c]), "gray" if c == dead_col else "blue")
        tag = sc.label(f"tag.{name}", S(f"fig3.{name[-1]}", lang), anchor="end")
        tag.left_of(g.cell(0, 0), 8).center_y_on(g.cell(0, 0))
        for c in range(T):
            lab = sc.label(f"{name}.idx{c}", str(c), font_size=11,
                           color="#9ca3af")
            lab.center_x_on(g.cell(0, c)).above(g.cell(0, c), 3)
        note = sc.label(f"{name}.note", S(note_key, lang), anchor="start",
                        color=note_color, font_size=12)
        note.center_x_on(g).below(g, 6)
        return g, note

    gq, nq = row("rowq", data["q"], 0, "fig3.q0")
    gq.at(0, 0).below(cap, 30)
    gk, nk = row("rowk", data["k"], T - 1, "fig3.kT")
    gk.align_left(gq).below(nq, 26)
    gv, nv = row("rowv", data["v"], 0, "fig3.v0")
    gv.align_left(gq).below(nk, 26)
    return sc


def fig06_naive_vs_fast(data, lang):
    st = data["stats"]
    sc = Scene("fig06-naive-vs-fast", title=S("fig6.title", lang),
               desc="Side-by-side of naive recomputation and compiled surfaces.")
    cap = _caption(sc, "fig6.title", lang)

    nt = sc.box("naive.title", S("fig6.naive_title", lang), style="gray",
                bold_first=True)
    nt.at(0, 0).below(cap, 14)
    n1 = sc.box("naive.1", S("fig6.naive_1", lang), style="ink")
    n1.align_left(nt).below(nt, 8)
    n2 = sc.box("naive.2", [S("fig6.naive_2", lang), S("fig6.naive_3", lang),
                            S("fig6.naive_4", lang), S("fig6.naive_5", lang)],
                style="ink")
    n2.align_left(nt).below(n1, 8)
    n3 = sc.box("naive.total", S("fig6.naive_total", lang, n=24 * 8 ** 3),
                style="gray")
    n3.align_left(nt).below(n2, 8)
    naive = sc.enclose("naive.panel", [nt, n1, n2, n3], padding=10,
                       style="gray", dashed=True)

    ft = sc.box("fast.title", S("fig6.fast_title", lang), style="blue",
                bold_first=True)
    ft.right_of(naive, 70).align_top(nt)
    f1 = sc.box("fast.1", S("fig6.fast_1", lang), style="blue")
    f1.align_left(ft).below(ft, 8)
    f2 = sc.box("fast.2", S("fig6.fast_2", lang, runs=st.q_delete_runs,
                            pairs=st.q_delete_pair_equiv), style="blue")
    f2.align_left(ft).below(f1, 8)
    f3 = sc.box("fast.3", S("fig6.fast_3", lang, runs=st.k_delete_runs,
                            pairs=st.k_delete_pair_equiv), style="blue")
    f3.align_left(ft).below(f2, 8)
    f4 = sc.box("fast.4", S("fig6.fast_4", lang, p=st.onebit_pair_count,
                            terms=st.drs_raw_terms), style="red")
    f4.align_left(ft).below(f3, 8)
    f5 = sc.box("fast.5", S("fig6.fast_5", lang, n=data["n_matched"]),
                style="blue")
    f5.align_left(ft).below(f4, 8)
    f6 = sc.box("fast.total", S("fig6.fast_total", lang,
                                terms=st.final_surface_terms), style="green")
    f6.align_left(ft).below(f5, 8)
    fast = sc.enclose("fast.panel", [ft, f1, f2, f3, f4, f5, f6], padding=10,
                      style="blue")

    sc.connect(naive, "right", fast, "left", color=GREEN)
    return sc


def fig07_pipeline(data, lang):
    sc = Scene("fig07-pipeline", title=S("fig7.title", lang),
               desc="Pipeline from symbol streams to grad_z.")
    cap = _caption(sc, "fig7.title", lang)

    b0 = sc.box("b0", S("fig7.b0", lang), style="blue")
    b0.at(140, 0).below(cap, 14)
    b1 = sc.box("b1", [S("fig7.b1a", lang), S("fig7.b1b", lang)], style="blue",
                bold_first=True)
    b1.center_x_on(b0).below(b0, 18)
    b2 = sc.box("b2", S("fig7.b2", lang), style="blue")
    b2.center_x_on(b0).below(b1, 18)
    b3 = sc.box("b3", [S("fig7.b3a", lang), S("fig7.b3b", lang)], style="blue",
                bold_first=True)
    b3.center_x_on(b0).below(b2, 18)

    s1 = sc.box("s1", [S("fig7.s1a", lang), S("fig7.s1b", lang)], style="blue")
    s1.center_x_on(b0).below(b3, 18)
    s2 = sc.box("s2", [S("fig7.s2a", lang), S("fig7.s2b", lang)], style="blue")
    s2.center_x_on(b0).below(s1, 8)
    s3 = sc.box("s3", [S("fig7.s3a", lang), S("fig7.s3b", lang)], style="red")
    s3.center_x_on(b0).below(s2, 8)
    s4 = sc.box("s4", [S("fig7.s4a", lang), S("fig7.s4b", lang)], style="red")
    s4.center_x_on(b0).below(s3, 8)
    comp = sc.enclose("compile", [s1, s2, s3, s4], padding=10, style="gray",
                      dashed=True)

    b5 = sc.box("b5", S("fig7.b5", lang), style="ink")
    b5.center_x_on(b0).below(comp, 18)
    b6 = sc.box("b6", S("fig7.b6", lang), style="ink")
    b6.center_x_on(b0).below(b5, 18)
    b7 = sc.box("b7", S("fig7.b7", lang), style="red")
    b7.center_x_on(b0).below(b6, 18)
    b8 = sc.box("b8", S("fig7.b8", lang), style="ink")
    b8.center_x_on(b0).below(b7, 18)
    b9 = sc.box("b9", S("fig7.b9", lang), style="ink")
    b9.center_x_on(b0).below(b8, 18)

    for a, b in ((b0, b1), (b1, b2), (b2, b3), (b3, comp), (comp, b5),
                 (b5, b6), (b6, b7), (b7, b8), (b8, b9)):
        sc.connect(a, "bottom", b, "top", color=LINE)

    # baseline rail on the right: ell/route feed the contraction directly
    rail_gap = 40
    rail_x = lambda: max(comp.bounds.x1, b5.bounds.x1, b6.bounds.x1) + rail_gap
    sc.connect(b2, "right", b7, "right", color=BLUE,
               waypoints=lambda: [(rail_x(), b2.bounds.cy),
                                  (rail_x(), b7.bounds.cy)],
               label=S("fig7.rail", lang), label_side="right")

    gy = sc.box("grady", S("fig7.grady", lang), style="red")
    gy.left_of(b7, 40).center_y_on(b7)
    sc.connect(gy, "right", b7, "left", color=RED)
    return sc


def fig09_q_heads(data, lang):
    h = data["heads"][6][0]
    run = data["qdel"][6][0]
    sc = Scene("fig09-q-latest-heads", title=S("fig9.title", lang),
               desc="One latest-occurrence head covers L=1..5 at endpoint 5; "
                    "it converts into one affine owner-axis run.")
    cap = _caption(sc, "fig9.title", lang)

    # upper band: head table (L = 1..5)
    gL = sc.grid("heads", 2, 5, cw=CW, ch=CH)
    gL.at(0, 0).below(cap, 26)
    for c in range(5):
        gL.cell(0, c, str(c + 1), "cell")
        gL.cell(1, c, str(h.endpoint), "green")
    tagL = sc.label("tagL", S("fig9.hypL", lang), anchor="end", font_size=12)
    tagL.left_of(gL.cell(0, 0), 8).center_y_on(gL.cell(0, 0))
    tagE = sc.label("tagE", S("fig9.latest", lang), anchor="end", font_size=12)
    tagE.left_of(gL.cell(1, 0), 8).center_y_on(gL.cell(1, 0))
    head_box = sc.enclose("head", gL.row_cells(1), padding=6, style="green")
    head_lab = sc.label("head.lab", S("fig9.onehead", lang, lo=h.L_lo,
                                      hi=h.L_hi, e=h.endpoint),
                        font_size=12, color=GREEN, anchor="start")
    head_lab.align_left(gL).below(head_box, 6)

    # lower band: owner axis s = 1..6
    gS = sc.grid("owner", 2, 6, cw=CW, ch=CH)
    gS.align_left(gL).below(head_lab, 26)
    Ls = [run.len_a * s + run.len_b for s in range(1, 6)] + [0]
    rs = [run.end_a * s + run.end_b for s in range(1, 6)] + [-1]
    for c in range(6):
        style = "blue" if c < 5 else "gray"
        gS.cell(0, c, str(Ls[c]), style)
        gS.cell(1, c, str(rs[c]), style)
        idx = sc.label(f"s{c}", f"s={c + 1}", font_size=11, color="#9ca3af")
        idx.center_x_on(gS.cell(0, c)).above(gS.cell(0, c), 3)
    tagO = sc.label("tagO", S("fig9.owner", lang), anchor="end",
                    font_size=12, color="#6b7280")
    tagO.left_of(gS.cell(0, 0), 8).above(gS.cell(0, 0), 3)
    tagLs = sc.label("tagLs", S("fig9.Ls", lang), anchor="end", font_size=12)
    tagLs.left_of(gS.cell(0, 0), 8).center_y_on(gS.cell(0, 0))
    tagRs = sc.label("tagRs", S("fig9.rs", lang), anchor="end", font_size=12)
    tagRs.left_of(gS.cell(1, 0), 8).center_y_on(gS.cell(1, 0))

    runbox = sc.enclose("run", gS.row_cells(0, 0, 4) + gS.row_cells(1, 0, 4),
                        padding=6, style="blue")
    runlab = sc.label("run.lab", S("fig9.run1", lang, lo=run.s_lo, hi=run.s_hi,
                                   la=run.len_a, lb=run.len_b, ea=run.end_a,
                                   eb=run.end_b), font_size=12, color=BLUE,
                      anchor="start")
    runlab.align_left(gS).below(runbox, 6)
    unlab = sc.label("unmatched", S("fig9.unmatched", lang), font_size=12,
                     color=GRAY, anchor="start")
    unlab.align_left(gS).below(runlab, 4)
    return sc


def _index_labels(sc, grid, indices, above=True, color="#9ca3af"):
    labs = []
    for c, v in enumerate(indices):
        lab = sc.label(f"{grid.name}.idx{c}", str(v), font_size=11,
                       color=color)
        lab.center_x_on(grid.cell(0, c))
        if above:
            lab.above(grid.cell(0, c), 3)
        else:
            lab.below(grid.cell(0, c), 3)
        labs.append(lab)
    return labs


def fig10_k_delete(data, lang):
    sc = Scene("fig10-k-delete-reindex", title=S("fig10.title", lang),
               desc="Deleting K[2]: windows avoid the hole, positions to the "
                    "right sit one step closer.")
    cap = _caption(sc, "fig10.title", lang)

    # BEFORE panel
    gB = sc.grid("before", 1, 8, cw=CW, ch=CH)
    gB.at(0, 0).below(cap, 46)
    for c in range(8):
        gB.cell(0, c, str(data["k"][c]), "red" if c == 2 else "blue")
    _index_labels(sc, gB, list(range(8)))
    win = sc.enclose("before.window", gB.row_cells(0, 0, 2), padding=5,
                     style="red", dashed=True)
    tagB = sc.box("before.tag", S("fig10.before", lang), style="gray",
                  font_size=12, pad_x=8, pad_y=4)
    tagB.center_x_on(gB).above(gB, 20)
    hole = sc.label("before.hole", S("fig10.hole", lang), font_size=12,
                    color=RED, anchor="start")
    hole.align_left(gB).below(gB, 8)
    broken = sc.label("before.broken", S("fig10.broken", lang), font_size=12,
                      color=RED, anchor="start")
    broken.align_left(gB).below(hole, 3)
    base = sc.label("before.base", S("fig10.baseinfo", lang), font_size=12,
                    anchor="start")
    base.align_left(gB).below(broken, 3)
    panelB = sc.enclose("panel.before", [gB, tagB, hole, broken, base],
                        padding=10, style="gray", dashed=True)

    # AFTER panel
    after_vals = [data["k"][i] for i in (0, 1, 3, 4, 5, 6, 7)]
    gA = sc.grid("after", 1, 7, cw=CW, ch=CH)
    gA.right_of(panelB, 70).below(cap, 46)
    for c in range(7):
        gA.cell(0, c, str(after_vals[c]), "blue")
    _index_labels(sc, gA, [0, 1, 3, 4, 5, 6, 7])
    winA = sc.enclose("after.window", gA.row_cells(0, 0, 1), padding=5,
                      style="green")
    altA = sc.enclose("after.alt", [gA.cell(0, 2)], padding=5, style="blue")
    tagA = sc.box("after.tag", S("fig10.after", lang), style="gray",
                  font_size=12, pad_x=8, pad_y=4)
    tagA.center_x_on(gA).above(gA, 20)
    idx_note = sc.label("after.idxnote", S("fig10.origidx", lang),
                        font_size=11, color="#9ca3af", anchor="start")
    idx_note.align_left(gA).below(gA, 8)
    winlab = sc.label("after.winlab", S("fig10.win", lang), font_size=12,
                      color=GREEN, anchor="start")
    winlab.align_left(gA).below(idx_note, 3)
    altlab = sc.label("after.altlab", S("fig10.alt", lang), font_size=12,
                      color=BLUE, anchor="start")
    altlab.align_left(gA).below(winlab, 3)
    result = sc.label("after.result", S("fig10.result", lang), font_size=12,
                      anchor="start")
    result.align_left(gA).below(altlab, 6)
    panelA = sc.enclose("panel.after",
                        [gA, tagA, idx_note, winlab, altlab, result],
                        padding=10, style="blue")

    sc.connect(panelB, "right", panelA, "left", color=RED)
    return sc


def fig11_thresholds(data, lang):
    rr = data["kdel"].repair_runs[7]
    lo, hi = rr[0], rr[1]
    sc = Scene("fig11-repair-thresholds", title=S("fig11.title", lang),
               desc="Threshold surface at t=7: strict left of the baseline "
                    "window, inclusive at and right of it.")
    cap = _caption(sc, "fig11.title", lang)

    g = sc.grid("axis", 1, 7, cw=40, ch=CH)
    g.at(0, 0).below(cap, 30)
    for c in range(7):
        style = "red" if c <= hi.s_lo - 1 else "green"
        if c == 3:
            style = "blue"
        g.cell(0, c, str(c), style)
    tag = sc.label("tag", S("fig11.owner", lang), anchor="end", font_size=12)
    tag.left_of(g.cell(0, 0), 8).center_y_on(g.cell(0, 0))
    winlab = sc.label("window", S("fig11.window", lang), font_size=12,
                      color=BLUE)
    winlab.center_x_on(g.cell(0, 3)).above(g.cell(0, 3), 4)
    sc.enclose("strict", g.row_cells(0, lo.s_lo, lo.s_hi), padding=5,
               style="red", dashed=True)
    sc.enclose("incl", g.row_cells(0, hi.s_lo, hi.s_hi), padding=5,
               style="green")
    strictlab = sc.label("strict.lab", S("fig11.strict", lang), font_size=12,
                         color=RED, anchor="start")
    strictlab.align_left(g).below(g, 10)
    incllab = sc.label("incl.lab", S("fig11.incl", lang), font_size=12,
                       color=GREEN, anchor="start")
    incllab.align_left(g).below(strictlab, 4)
    return sc


def fig12_bridge(data, lang):
    sc = Scene("fig12-shared-bridge", title=S("fig12.title", lang),
               desc="Bridge geometry: q row and k row paired column-wise, "
                    "left=2 context, the flipped bit, right=0.")
    cap = _caption(sc, "fig12.title", lang)

    qvals = [data["q"][i] for i in range(3, 8)]   # positions 3..7
    kvals = [data["k"][i] for i in range(0, 5)]   # positions 0..4
    gQ = sc.grid("qrow", 1, 5, cw=CW, ch=CH)
    gQ.at(0, 0).below(cap, 30)
    for c in range(5):
        gQ.cell(0, c, str(qvals[c]), "red" if c == 2 else "blue")
    _index_labels(sc, gQ, [3, 4, 5, 6, 7])
    tagQ = sc.label("tagQ", S("fig12.qrow", lang), anchor="end", font_size=12)
    tagQ.left_of(gQ.cell(0, 0), 8).center_y_on(gQ.cell(0, 0))

    gK = sc.grid("krow", 1, 5, cw=CW, ch=CH)
    gK.align_left(gQ).below(gQ, 42)
    for c in range(5):
        gK.cell(0, c, str(kvals[c]), "red" if c == 2 else "blue")
    gQ.cell(0, 3).style = "gray"   # q[6] != k[3]: the right context is empty
    gK.cell(0, 3).style = "gray"
    _index_labels(sc, gK, [0, 1, 2, 3, 4], above=False)
    tagK = sc.label("tagK", S("fig12.krow", lang), anchor="end", font_size=12)
    tagK.left_of(gK.cell(0, 0), 8).center_y_on(gK.cell(0, 0))

    sc.enclose("leftctx", gQ.row_cells(0, 0, 1), padding=5, style="green",
               dashed=True)
    flip = sc.label("flip", S("fig12.flip", lang), font_size=12, color=RED)
    flip.center_x_on(gQ.cell(0, 2))
    flip._y_rule = lambda: (gQ.bounds.y1 + gK.bounds.y0) / 2 - flip.h / 2

    lines = [("leftctx.lab", S("fig12.left", lang), GREEN),
             ("rightctx.lab", S("fig12.right", lang), GRAY),
             ("window.lab", S("fig12.window", lang), GREEN),
             ("r1", S("fig12.r1", lang), "#111827"),
             ("r2", S("fig12.r2", lang), "#111827")]
    prev = gK
    for name, text, color in lines:
        lab = sc.label(name, text, font_size=12, color=color, anchor="start")
        lab.align_left(gQ).below(prev, 26 if prev is gK else 4)
        prev = lab
    return sc


def fig13_envelope(data, lang):
    kt2 = data["ir"].k_terms[2][0]
    assert [(t.shift, t.lo, t.hi) for t in kt2] == [(-2, 5, 5), (-3, 7, 7)]
    sc = Scene("fig13-bridge-envelope", title=S("fig13.title", lang),
               desc="Two bridges owned by s=2 on the t axis; first-win "
                    "decides the surviving terms.")
    cap = _caption(sc, "fig13.title", lang)

    g = sc.grid("taxis", 3, 8, cw=40, ch=CH, font_size=13)
    g.at(0, 0).below(cap, 28)
    for c in range(8):
        g.cell(0, c, str(c), "plain")
    g.cell(1, 5, "A", "green")
    g.cell(2, 6, "B", "gray")
    g.cell(2, 7, "B", "green")
    sc.enclose("lifeA", [g.cell(1, 5)], padding=4, style="green",
               dashed=True)
    sc.enclose("lifeB", g.row_cells(2, 6, 7), padding=4, style="blue",
               dashed=True)

    tagT = sc.label("tagT", S("fig13.t", lang), anchor="end", font_size=12)
    tagT.left_of(g.cell(0, 0), 8).center_y_on(g.cell(0, 0))
    rowA = sc.label("rowA", S("fig13.rowA", lang) + ",  " +
                    S("fig13.lifeA", lang), anchor="end", font_size=12)
    rowA.left_of(g.cell(1, 0), 10).center_y_on(g.cell(1, 0))
    rowB = sc.label("rowB", S("fig13.rowB", lang) + ",  " +
                    S("fig13.lifeB", lang), anchor="end", font_size=12)
    rowB.left_of(g.cell(2, 0), 10).center_y_on(g.cell(2, 0))
    noteA = sc.label("noteA", S("fig13.noteA", lang), font_size=12,
                     color=GREEN, anchor="start")
    noteA.right_of(g.cell(1, 7), 12).center_y_on(g.cell(1, 5))
    noteB = sc.label("noteB", S("fig13.noteB", lang), font_size=12,
                     anchor="start")
    noteB.right_of(g.cell(2, 7), 12).center_y_on(g.cell(2, 6))

    result = sc.box("result", S("fig13.result", lang), style="green",
                    font_size=12, pad_x=8, pad_y=5)
    result.align_left(g).below(g, 14)
    return sc


def fig14_certificates(data, lang):
    st = data["cert_stats"]
    sc = Scene("fig14-run-certificates", title=S("fig14.title", lang),
               desc="k=[0,0,1,0,0,0]: two maximal runs, one breaking bit, "
                    "one square certificate.")
    cap = _caption(sc, "fig14.title", lang)

    g = sc.grid("krow", 1, 6, cw=40, ch=CH)
    g.at(0, 0).below(cap, 28)
    for c in range(6):
        style = "blue"
        if c == 1:
            style = "green"
        elif c == 2:
            style = "red"
        g.cell(0, c, str(data["k6"][c]), style)
    _index_labels(sc, g, list(range(6)))
    tag = sc.label("tag", S("fig14.krow", lang), anchor="end", font_size=12)
    tag.left_of(g.cell(0, 0), 8).center_y_on(g.cell(0, 0))
    sc.enclose("run1", g.row_cells(0, 0, 1), padding=5, style="blue")
    sc.enclose("run2", g.row_cells(0, 3, 5), padding=5, style="gray",
               dashed=True)

    lines = [("run1.lab", S("fig14.run1", lang), BLUE),
             ("run2.lab", S("fig14.run2", lang), GRAY),
             ("breaker.lab", S("fig14.breaker", lang), RED),
             ("owner.lab", S("fig14.owner", lang), GREEN),
             ("cert.lab", S("fig14.cert", lang), GREEN),
             ("stats.lab", S("fig14.stats", lang, runs=len(data["runs6"]),
                             certs=1, cq=st.k_run_certificate_queries,
                             ch=st.k_run_certificate_hits,
                             bq=st.k_h_ramp_boundary_queries,
                             bh=st.k_h_ramp_boundary_hits,
                             fb=st.k_h_ramp_fallback_owner_queries),
              "#111827"),
             ("hit.lab", S("fig14.hit", lang), GREEN)]
    prev = g
    for name, text, color in lines:
        lab = sc.label(name, text, font_size=12, color=color, anchor="start")
        lab.align_left(g).below(prev, 10 if prev is g else 4)
        prev = lab
    return sc


def fig15_shadow(data, lang):
    sc = Scene("fig15-shadow-repair", title=S("fig15.title", lang),
               desc="Flip k[1]: the baseline match casts an equality shadow "
                    "one period left; the repair wins the route but not the "
                    "payload.")
    cap = _caption(sc, "fig15.title", lang)

    # flip band: k and k' rows
    gK = sc.grid("k", 1, 6, cw=CW, ch=CH)
    gK.at(0, 0).below(cap, 28)
    for c in range(6):
        gK.cell(0, c, str(data["k6"][c]), "gray" if c == 1 else "blue")
    _index_labels(sc, gK, list(range(6)))
    tagK = sc.label("tagK", S("fig15.k", lang), anchor="end", font_size=12)
    tagK.left_of(gK.cell(0, 0), 8).center_y_on(gK.cell(0, 0))
    gP = sc.grid("kprime", 1, 6, cw=CW, ch=CH)
    gP.align_left(gK).below(gK, 8)
    for c in range(6):
        v = data["k6"][c] ^ (1 if c == 1 else 0)
        gP.cell(0, c, str(v), "red" if c == 1 else "blue")
    tagP = sc.label("tagP", S("fig15.kp", lang), anchor="end", font_size=12)
    tagP.left_of(gP.cell(0, 0), 8).center_y_on(gP.cell(0, 0))
    flip = sc.label("flip", S("fig15.fliplab", lang), font_size=12, color=RED,
                    anchor="start")
    flip.align_left(gK).below(gP, 6)

    # BEFORE mini diagram: Q[2..4] over K[0..2]
    qB = sc.grid("before.q", 1, 3, cw=CW, ch=CH)
    qB.align_left(gK).below(flip, 22)
    for c, v in enumerate([data["q6"][i] for i in (2, 3, 4)]):
        qB.cell(0, c, str(v), "green")
    _index_labels(sc, qB, [2, 3, 4])
    kB = sc.grid("before.k", 1, 3, cw=CW, ch=CH)
    kB.align_left(qB).below(qB, 8)
    for c, v in enumerate([data["k6"][i] for i in (0, 1, 2)]):
        kB.cell(0, c, str(v), "green")
    _index_labels(sc, kB, [0, 1, 2], above=False)
    contB = sc.enclose("before.cont", [qB, kB], padding=8, style="green")
    titleB = sc.label("before.title", S("fig15.before", lang), font_size=12,
                      bold=True)
    titleB.center_x_on(contB).below(contB, 16)

    # AFTER mini diagram: Q[3..4] over K'[0..1]
    qA = sc.grid("after.q", 1, 2, cw=CW, ch=CH)
    qA.right_of(contB, 190).align_top(qB)
    for c, v in enumerate([data["q6"][i] for i in (3, 4)]):
        qA.cell(0, c, str(v), "green")
    _index_labels(sc, qA, [3, 4])
    kA = sc.grid("after.k", 1, 2, cw=CW, ch=CH)
    kA.align_left(qA).below(qA, 8)
    kA.cell(0, 0, str(data["k6"][0]), "green")
    kA.cell(0, 1, "1", "red")
    _index_labels(sc, kA, [0, 1], above=False)
    contA = sc.enclose("after.cont", [qA, kA], padding=8, style="green")
    titleA = sc.label("after.title", S("fig15.after", lang), font_size=12,
                      bold=True)
    titleA.center_x_on(contA).below(contA, 16)

    sc.connect(contB, "right", contA, "left", color=RED,
               label=S("fig15.arrow", lang))

    # verdict band
    lines = [("beforeEq", S("fig15.beforeEq", lang), GREEN),
             ("afterEq", S("fig15.afterEq", lang), GREEN),
             ("cmp", S("fig15.cmp", lang), "#111827"),
             ("payload", S("fig15.payload", lang), BLUE)]
    prev = [titleB, titleA]
    for name, text, color in lines:
        lab = sc.label(name, text, font_size=12, color=color, anchor="start")
        lab.align_left(gK).below(prev, 14)
        prev = lab
    term = sc.box("term", S("fig15.term", lang), style="ink", font_size=12,
                  pad_x=8, pad_y=5)
    term.align_left(gK).below(prev, 10)
    return sc


FIGURES = {
    "fig00-legend": fig00_legend,
    "fig01-running-instance": fig01_running_instance,
    "fig03-structural-zeros": fig03_structural_zeros,
    "fig06-naive-vs-fast": fig06_naive_vs_fast,
    "fig07-pipeline": fig07_pipeline,
    "fig09-q-latest-heads": fig09_q_heads,
    "fig10-k-delete-reindex": fig10_k_delete,
    "fig11-repair-thresholds": fig11_thresholds,
    "fig12-shared-bridge": fig12_bridge,
    "fig13-bridge-envelope": fig13_envelope,
    "fig14-run-certificates": fig14_certificates,
    "fig15-shadow-repair": fig15_shadow,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", choices=["en", "zh"], default=None,
                    help="generate only one language (default: both)")
    args = ap.parse_args()

    data = compute_data()
    langs = [args.lang] if args.lang else ["en", "zh"]
    for lang in langs:
        for name, builder in FIGURES.items():
            scene = builder(data, lang)
            svg = scene.render()  # validator runs here; violations raise
            suffix = ".zh-CN" if lang == "zh" else ""
            out = OUT_DIR / f"{name}{suffix}.svg"
            out.write_text(svg, encoding="utf-8")
            print(f"[{lang}] wrote {out.name}  (validator: 0 violations)")


if __name__ == "__main__":
    main()
