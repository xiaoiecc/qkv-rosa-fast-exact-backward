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
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

import hard_qkv_rosa_explained as H  # noqa: E402

from svg_layout import (  # noqa: E402
    Scene, PALETTE, LINE, TEXT,
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
    "fig0.blue": {"en": "forward / baseline data", "zh": "前向 / 基线数据"},
    "fig0.red": {"en": "credit flow / changed object", "zh": "credit 流 / 被改变的对象"},
    "fig0.gray": {"en": "frozen content", "zh": "冻结内容"},
    "fig0.green": {"en": "match / hit", "zh": "匹配 / 命中"},
    "fig0.flow_from": {"en": "from", "zh": "来源"},
    "fig0.flow_to": {"en": "to", "zh": "去向"},
    "fig0.flow": {"en": "credit flow / dependency", "zh": "credit 流 / 依赖"},

    # fig01 — running instance
    "fig1.title": {"en": "Figure 1 — the running instance (T = 8, D = 1)",
                   "zh": "图 1 —— 贯穿实例（T = 8, D = 1）"},
    "fig1.pos": {"en": "position t", "zh": "位置 t"},
    "fig1.q": {"en": "q[t]  (probe)", "zh": "q[t]（探针）"},
    "fig1.k": {"en": "k[t]  (memory)", "zh": "k[t]（记忆）"},
    "fig1.v": {"en": "v_bits[t]", "zh": "v_bits[t]"},
    "fig1.ell": {"en": "ell[t]", "zh": "ell[t]"},
    "fig1.route": {"en": "route[t]", "zh": "route[t]"},
    "fig1.payload": {"en": "payload src = V[route[t]+1]", "zh": "负载来源 = V[route[t]+1]"},
    "fig1.reuse": {"en": "t=7 reuses V[4]:  route[7]+1 = 4", "zh": "t=7 复用 V[4]：route[7]+1 = 4"},
    "fig1.note": {"en": "the backward asks, for every bit position u of q, k, v_bits:  flip [u] -> how do ell, route, and y change? -> credit[u]",
                  "zh": "反向对 q、k、v_bits 的每个比特位置 u 发问：翻转 [u] -> ell、route、y 如何变化？-> credit[u]"},

    # fig03 — structural zeros
    "fig3.title": {"en": "Figure 3 — three structural zeros of the payload rule (running instance)",
                   "zh": "图 3 —— 负载规则的三个结构性零（贯穿实例）"},
    "fig3.q": {"en": "q[t]", "zh": "q[t]"},
    "fig3.k": {"en": "k[t]", "zh": "k[t]"},
    "fig3.v": {"en": "v_bits[t]", "zh": "v_bits[t]"},
    "fig3.v0": {"en": "V[0] is never read:  route[t] >= 0  =>  payload index = route[t]+1 >= 1",
                "zh": "V[0] 从不被读取：route[t] >= 0  =>  负载下标 = route[t]+1 >= 1"},
    "fig3.kT": {"en": "K[T-1] is never an endpoint:  an endpoint r must satisfy r < t <= T-1",
                "zh": "K[T-1] 从不充当端点：端点 r 必须满足 r < t <= T-1"},
    "fig3.q0": {"en": "q[0] never matters:  it could only serve a full-prefix match ending at r = t, which is not a legal endpoint",
                "zh": "q[0] 从不起作用：它只能服务于一条结束于 r = t 的全前缀匹配，而 r = t 不是合法端点"},

    # fig06 — naive vs fast
    "fig6.title": {"en": "Figure 6 — same 24 bits, same answers: naive recomputation vs compiled surfaces (running instance)",
                   "zh": "图 6 —— 同样的 24 个比特、同样的答案：朴素重算 vs 编译曲面（贯穿实例）"},
    "fig6.naive_title": {"en": "NAIVE (backward_bruteforce)", "zh": "朴素法（backward_bruteforce）"},
    "fig6.naive_1": {"en": "24 bit flips x full forward O(T^3)", "zh": "24 次比特翻转 x 完整前向 O(T^3)"},
    "fig6.naive_2": {"en": "each forward re-derives ALL of:", "zh": "每次前向都重新推导全部："},
    "fig6.naive_3": {"en": "  every suffix comparison", "zh": "  每一次后缀比较"},
    "fig6.naive_4": {"en": "  every tie-break", "zh": "  每一次平局决胜"},
    "fig6.naive_5": {"en": "  every payload read", "zh": "  每一次负载读取"},
    "fig6.naive_total": {"en": "total: 24 x 8^3 = {n} suffix steps", "zh": "合计：24 x 8^3 = {n} 次后缀步"},
    "fig6.fast_title": {"en": "THIS ALGORITHM", "zh": "本算法"},
    "fig6.fast_1": {"en": "one baseline forward (fast, Part 2)", "zh": "一次基线前向（快速，Part 2）"},
    "fig6.fast_2": {"en": "Q-side deletion surfaces: {runs} affine runs cover all {pairs} owner-output deletion events",
                    "zh": "Q 侧删除曲面：{runs} 条仿射 run 覆盖全部 {pairs} 个属主-输出删除事件"},
    "fig6.fast_3": {"en": "K-side deletion surfaces: {runs} affine runs cover all {pairs} owner-output deletion events",
                    "zh": "K 侧删除曲面：{runs} 条仿射 run 覆盖全部 {pairs} 个属主-输出删除事件"},
    "fig6.fast_4": {"en": "{p} one-bit (q,k) pairs -> {terms} repair terms on t-intervals",
                    "zh": "{p} 个单比特 (q,k) 对 -> {terms} 个 t 区间上的修复项"},
    "fig6.fast_5": {"en": "closed-form V credit ({n} adds)", "zh": "闭式 V credit（{n} 次累加）"},
    "fig6.fast_total": {"en": "total: {terms} compiled terms, each carrying a grad_y-weighted range sum",
                        "zh": "合计：{terms} 个编译项，各自携带一个 grad_y 加权的区间和"},
    "fig6.vs": {"en": "same 24 answers, exactly", "zh": "同样 24 个答案，逐位精确"},

    # fig07 — pipeline
    "fig7.title": {"en": "Figure 7 — the pipeline: from symbol streams to grad_z",
                   "zh": "图 7 —— 流水线：从符号流到 grad_z"},
    "fig7.b0": {"en": "q, k  (symbol streams)", "zh": "q, k（符号流）"},
    "fig7.b1a": {"en": "Part 2 — fast forward", "zh": "Part 2 —— 快速前向"},
    "fig7.b1b": {"en": "suffix array + LCE + static certificates", "zh": "后缀数组 + LCE + 静态证书"},
    "fig7.b2": {"en": "ell[t], route[t]  (baseline)", "zh": "ell[t], route[t]（基线）"},
    "fig7.b3a": {"en": "Part 4 — rebuild workbench", "zh": "Part 4 —— 重建工作台"},
    "fig7.b3b": {"en": "bidirectional suffix indexes -> CausalCutSuffixIndex", "zh": "双向后缀索引 -> CausalCutSuffixIndex"},
    "fig7.s1a": {"en": "Part 6 — Q-side deletion surfaces", "zh": "Part 6 —— Q 侧删除曲面"},
    "fig7.s1b": {"en": "(latest-occurrence heads)", "zh": "（最新出现头部）"},
    "fig7.s2a": {"en": "Part 5 — K-side deletion oracles A/H", "zh": "Part 5 —— K 侧删除预言机 A/H"},
    "fig7.s2b": {"en": "-> max(A, H) surfaces", "zh": "-> max(A, H) 曲面"},
    "fig7.s3a": {"en": "Part 12/14 — shared one-bit repair bridges", "zh": "Part 12/14 —— 共享单比特修复桥"},
    "fig7.s3b": {"en": "(sparse / diagonal)", "zh": "（稀疏 / 对角）"},
    "fig7.s4a": {"en": "Part 10/13 — surface-conditioned repair compiler", "zh": "Part 10/13 —— 曲面条件化修复编译器"},
    "fig7.s4b": {"en": "(maximal-run square certificates)", "zh": "（极大 run 平方证书）"},
    "fig7.b5": {"en": "RepairIR:  q_terms, k_terms, q_delete_runs, k_delete_runs, zeros",
                "zh": "RepairIR:  q_terms, k_terms, q_delete_runs, k_delete_runs, zeros"},
    "fig7.b6": {"en": "Part 8 — numerical contraction (no string queries left)",
                "zh": "Part 8 —— 数值收缩（不再有任何字符串查询）"},
    "fig7.b7": {"en": "credit_q, credit_k, credit_v", "zh": "credit_q, credit_k, credit_v"},
    "fig7.b8": {"en": "Part 11 — orient = 1 - 2*bit", "zh": "Part 11 — orient = 1 - 2*bit"},
    "fig7.b9": {"en": "grad_z  (identity / bernoulli)", "zh": "grad_z（identity / bernoulli）"},
    "fig7.rail": {"en": "baseline ell/route", "zh": "基线 ell/route"},
    "fig7.grady": {"en": "grad_y", "zh": "grad_y"},

    # fig09 — Q-side latest heads
    "fig9.title": {"en": "Figure 9 — heads of t = 6 (ell[6] = 6) and their conversion to owner-axis runs",
                   "zh": "图 9 —— t = 6 的头部（ell[6] = 6）及其向属主轴 run 的转换"},
    "fig9.hypL": {"en": "hypothetical length L", "zh": "假想长度 L"},
    "fig9.latest": {"en": "latest endpoint", "zh": "最新端点"},
    "fig9.onehead": {"en": "constant endpoint: ONE head covers L_lo={lo}..L_hi={hi}, endpoint={e}",
                     "zh": "端点恒定：一个头部覆盖 L_lo={lo}..L_hi={hi}，endpoint={e}"},
    "fig9.owner": {"en": "owner s", "zh": "属主 s"},
    "fig9.Ls": {"en": "L(s) = t - s", "zh": "L(s) = t - s"},
    "fig9.rs": {"en": "r(s)", "zh": "r(s)"},
    "fig9.run1": {"en": "AffineDeleteRun(t=6, s in [{lo},{hi}], len_a={la}, len_b={lb}, end_a={ea}, end_b={eb})",
                  "zh": "AffineDeleteRun(t=6, s in [{lo},{hi}], len_a={la}, len_b={lb}, end_a={ea}, end_b={eb})"},
    "fig9.unmatched": {"en": "s = t: cut the last bit -> unmatched (r = -1)",
                       "zh": "s = t：切断最后一个比特 -> 未匹配（r = -1）"},

    # fig10 — K deletion re-index
    "fig10.title": {"en": "Figure 10 — delete K[2] and re-answer t = 3: re-index and re-match",
                    "zh": "图 10 —— 删除 K[2] 并重新回答 t = 3：重排下标再匹配"},
    "fig10.before": {"en": "BEFORE (baseline)", "zh": "之前（基线）"},
    "fig10.after": {"en": "AFTER (K[2] treated as deleted)", "zh": "之后（K[2] 视为已删除）"},
    "fig10.hole": {"en": "hole at s = 2: no window may contain it",
                   "zh": "s = 2 处的空洞：任何窗口都不得包含它"},
    "fig10.broken": {"en": "baseline window [0,2] contains s = 2 -> broken",
                     "zh": "基线窗口 [0,2] 包含 s = 2 -> 被打断"},
    "fig10.baseinfo": {"en": "Q[:4] = [0,0,0,0];  baseline route(3) = (3, 2)",
                       "zh": "Q[:4] = [0,0,0,0]；基线 route(3) = (3, 2)"},
    "fig10.origidx": {"en": "cells keep their original K indices",
                      "zh": "格子保留其原始 K 下标"},
    "fig10.win": {"en": "window [0,1]: L = 2, endpoint r = 1  <- wins",
                  "zh": "窗口 [0,1]：L = 2，端点 r = 1  <- 胜出"},
    "fig10.alt": {"en": "window [3,3]: L = 1, endpoint r = 3",
                  "zh": "窗口 [3,3]：L = 1，端点 r = 3"},
    "fig10.result": {"en": "post-delete route(3, s=2) = (2, 1)   — matches the brute-force flip (ell 3->2, route 2->1)",
                     "zh": "删除后 route(3, s=2) = (2, 1)   —— 与暴力翻转一致（ell 3->2，route 2->1）"},

    # fig11 — repair threshold polarity
    "fig11.title": {"en": "Figure 11 — repair threshold surface at t = 7 (ell = 1, route = 3, window = [3,3])",
                    "zh": "图 11 —— t = 7 处的修复阈值曲面（ell = 1，route = 3，窗口 = [3,3]）"},
    "fig11.owner": {"en": "owner s", "zh": "属主 s"},
    "fig11.window": {"en": "baseline window [3,3]", "zh": "基线窗口 [3,3]"},
    "fig11.strict": {"en": "s in [0,2] (left of window): repair must be len >= 2 (strict) — an equal-length repair ends before r0 = 3 and LOSES the tie",
                     "zh": "s in [0,2]（窗口左侧）：修复必须 len >= 2（严格）—— 等长修复结束于 r0 = 3 之前，输掉平局"},
    "fig11.incl": {"en": "s in [3,6] (window + right): len >= 1 suffices (inclusive) — an equal-length repair ends at or after r0 and WINS the tie",
                   "zh": "s in [3,6]（窗口及右侧）：len >= 1 即可（含等长）—— 等长修复结束于 r0 或之后，赢得平局"},

    # fig12 — shared bridge geometry
    "fig12.title": {"en": "Figure 12 — the bridge behind the k[2] flip: pair (q_pos=5, k_pos=2, bit=0)",
                    "zh": "图 12 —— k[2] 翻转背后的修复桥：对 (q_pos=5, k_pos=2, bit=0)"},
    "fig12.qrow": {"en": "q (positions 3..7)", "zh": "q（位置 3..7）"},
    "fig12.krow": {"en": "k (positions 0..4)", "zh": "k（位置 0..4）"},
    "fig12.left": {"en": "left = 2: equal context q[3..4] == k[0..1]",
                   "zh": "left = 2：相等上下文 q[3..4] == k[0..1]"},
    "fig12.flip": {"en": "differ in bit 0 — flip me", "zh": "仅在 bit 0 不同 —— 翻转我"},
    "fig12.right": {"en": "right = 0: q[6] != k[3]", "zh": "right = 0：q[6] != k[3]"},
    "fig12.window": {"en": "after the flip: created match window q[3..5] == k'[0..2] == [0,0,1]",
                     "zh": "翻转后：新建匹配窗口 q[3..5] == k'[0..2] == [0,0,1]"},
    "fig12.r1": {"en": "d = t - 5;  lifetime t in [5, 5+right] = [5,5];  route_at(5) = (left+1+0, k_pos+0) = (3, 2)",
                 "zh": "d = t - 5；存活期 t in [5, 5+right] = [5,5]；route_at(5) = (left+1+0, k_pos+0) = (3, 2)"},
    "fig12.r2": {"en": "shift = k_pos - q_pos + 1 = -2  ->  candidate payload V[t-2] = V[3]  (brute force: t=5 ell 5->3, route 4->2)",
                 "zh": "shift = k_pos - q_pos + 1 = -2  ->  候选负载 V[t-2] = V[3]（暴力验证：t=5 ell 5->3，route 4->2）"},

    # fig13 — bridge envelope
    "fig13.title": {"en": "Figure 13 — owner s=2, bit 0: two bridges, first-win, and the surviving terms",
                    "zh": "图 13 —— 属主 s=2、bit 0：两座修复桥、首胜判定与幸存的项"},
    "fig13.t": {"en": "t", "zh": "t"},
    "fig13.rowA": {"en": "bridge A (q_pos=5, k_pos=2, left=2, right=0), shift=-2",
                   "zh": "桥 A（q_pos=5, k_pos=2, left=2, right=0），shift=-2"},
    "fig13.rowB": {"en": "bridge B (q_pos=6, k_pos=2, left=0, right=1), shift=-3",
                   "zh": "桥 B（q_pos=6, k_pos=2, left=0, right=1），shift=-3"},
    "fig13.noteA": {"en": "t=5: bridge (3,2) vs delete route (2,4) -> 3 > 2, WINS -> term [5,5]",
                    "zh": "t=5：桥 (3,2) vs 删除 route (2,4) -> 3 > 2，胜 -> 项 [5,5]"},
    "fig13.noteB": {"en": "t=6: bridge (1,2) vs delete route (3,4) -> loses, pruned;   t=7: bridge (2,3) vs baseline (1,3) -> 2 > 1, WINS -> term [7,7]",
                    "zh": "t=6：桥 (1,2) vs 删除 route (3,4) -> 落败，剪枝；  t=7：桥 (2,3) vs 基线 (1,3) -> 2 > 1，胜 -> 项 [7,7]"},
    "fig13.lifeA": {"en": "lifetime [5,5]", "zh": "存活期 [5,5]"},
    "fig13.lifeB": {"en": "lifetime [6,7]", "zh": "存活期 [6,7]"},
    "fig13.result": {"en": "k_terms[2][0] = [RepairTrackTerm(shift=-2, lo=5, hi=5), RepairTrackTerm(shift=-3, lo=7, hi=7)]   (the [7,7] term is later skipped by the payload LCE: route change, no credit)",
                     "zh": "k_terms[2][0] = [RepairTrackTerm(shift=-2, lo=5, hi=5), RepairTrackTerm(shift=-3, lo=7, hi=7)]   （[7,7] 项随后被负载 LCE 跳过：route 变了，没有 credit）"},

    # fig14 — run certificates
    "fig14.title": {"en": "Figure 14 — T = 6 certificate instance:  k = [0,0,1,0,0,0],  ell = [0,1,2,2,3,4],  route = [-1,0,1,1,2,3]",
                    "zh": "图 14 —— T = 6 证书实例：k = [0,0,1,0,0,0]，ell = [0,1,2,2,3,4]，route = [-1,0,1,1,2,3]"},
    "fig14.krow": {"en": "k", "zh": "k"},
    "fig14.run1": {"en": "run [0,1], p = 1", "zh": "run [0,1]，p = 1"},
    "fig14.run2": {"en": "run [3,5], p = 1:  hi+1 = 6 = T -> reaches the end -> NO certificate",
                   "zh": "run [3,5]，p = 1：hi+1 = 6 = T -> 到达流末尾 -> 无证书"},
    "fig14.breaker": {"en": "breaker at hi+1 = 2: k[1]=0 vs k[2]=1, xor = 1 = 2^0 -> exactly bit 0",
                      "zh": "hi+1 = 2 处的打断位：k[1]=0 vs k[2]=1，xor = 1 = 2^0 -> 恰为 bit 0"},
    "fig14.owner": {"en": "owner s = hi-p+1 = 1", "zh": "属主 s = hi-p+1 = 1"},
    "fig14.cert": {"en": "_KRunRepairCertificate(owner=1, period=1, bit=0, m_lo=2, m_hi=2)",
                   "zh": "_KRunRepairCertificate(owner=1, period=1, bit=0, m_lo=2, m_hi=2)"},
    "fig14.stats": {"en": "index: runs={runs}, certs={certs};  compile: certificate queries={cq} (hits={ch}), boundary={bq} (hits={bh}), fallback owner queries={fb}",
                    "zh": "索引：runs={runs}，certs={certs}；编译：证书查询={cq}（命中={ch}），边界={bq}（命中={bh}），回退属主查询={fb}"},
    "fig14.hit": {"en": "t=4: baseline route (3,2);  anchor M = 2 == route[4], H-ramp owner interval hits s=1 -> certificate query HITS",
                  "zh": "t=4：基线 route (3,2)；锚点 M = 2 == route[4]，H 斜坡的属主区间命中 s=1 -> 证书查询命中"},

    # fig15 — shadow repair
    "fig15.title": {"en": "Figure 15 — the repair itself on t = 4: flip k[1] and the match casts a shadow one period left",
                    "zh": "图 15 —— t = 4 上的修复本身：翻转 k[1]，匹配向左一个周期投下影子"},
    "fig15.k": {"en": "k (before)", "zh": "k（之前）"},
    "fig15.kp": {"en": "k' (after flip)", "zh": "k'（翻转后）"},
    "fig15.fliplab": {"en": "flip k[1]: 0 -> 1   (the certificate's owner, bit 0)",
                      "zh": "翻转 k[1]：0 -> 1（证书的属主，bit 0）"},
    "fig15.before": {"en": "BEFORE: baseline match", "zh": "之前：基线匹配"},
    "fig15.beforeEq": {"en": "Q[2..4] == K[0..2] == [0,0,1]   (ell = 3, r = 2, anchor M = 2)",
                       "zh": "Q[2..4] == K[0..2] == [0,0,1]   (ell = 3, r = 2，锚点 M = 2)"},
    "fig15.after": {"en": "AFTER: shadow one period left", "zh": "之后：影子左移一个周期"},
    "fig15.afterEq": {"en": "Q[3..4] == K'[0..1] == [0,1]   new match: length 2, endpoint M-p = 1",
                      "zh": "Q[3..4] == K'[0..1] == [0,1]   新匹配：长度 2，端点 M-p = 1"},
    "fig15.arrow": {"en": "flip shifts the window", "zh": "翻转使窗口平移"},
    "fig15.cmp": {"en": "compare:  deletion baseline route(4, s=1) = (1, 2)   vs   repair route (2, 1)  ->  2 > 1, repair WINS",
                  "zh": "对比：删除基线 route(4, s=1) = (1, 2)   vs   修复 route (2, 1)  ->  2 > 1，修复胜出"},
    "fig15.payload": {"en": "payload:  candidate V[M-p+1] = V[2] = 1  ==  baseline payload V[3] = 1  ->  y[4] unchanged (the term is skipped by the payload LCE of section 9)",
                      "zh": "负载：候选 V[M-p+1] = V[2] = 1  ==  基线负载 V[3] = 1  ->  y[4] 不变（此项被第 9 节的负载 LCE 跳过）"},
    "fig15.term": {"en": "compiled term:  owner 1, bit 0:  RepairTrackTerm(shift=-2, lo=4, hi=4)   (shift -2: candidate payload is V[t-2] = V[2])",
                   "zh": "编译出的项：属主 1，bit 0：RepairTrackTerm(shift=-2, lo=4, hi=4)   （shift -2：候选负载是 V[t-2] = V[2]）"},

    # fig02 — position-by-position trace
    "fig2.title": {"en": "Figure 2 — the running instance, position by position (t = 7 is the tie-break made visible)",
                   "zh": "图 2 —— 逐位置看贯穿实例（t = 7 是可视化了的平局决胜）"},
    "fig2.col.t": {"en": "t", "zh": "t"},
    "fig2.col.q": {"en": "q[t]", "zh": "q[t]"},
    "fig2.col.ell": {"en": "ell[t]", "zh": "ell[t]"},
    "fig2.col.route": {"en": "route[t]", "zh": "route[t]"},
    "fig2.col.payload": {"en": "payload", "zh": "负载"},
    "fig2.col.y": {"en": "y[t]", "zh": "y[t]"},
    "fig2.note.0": {"en": "K[:0] is empty: no match possible", "zh": "K[:0] 为空：不可能有匹配"},
    "fig2.note.1": {"en": 'Q[:2]="00" matches K[:1]="0", r=0', "zh": 'Q[:2]="00" 匹配 K[:1]="0"，r=0'},
    "fig2.note.2": {"en": '"00" ends at r=1', "zh": '"00" 结束于 r=1'},
    "fig2.note.3": {"en": '"000" ends at r=2', "zh": '"000" 结束于 r=2'},
    "fig2.note.4": {"en": '"0000" ends at r=3', "zh": '"0000" 结束于 r=3'},
    "fig2.note.5": {"en": '"00001" ends at r=4', "zh": '"00001" 结束于 r=4'},
    "fig2.note.6": {"en": '"000011" ends at r=5', "zh": '"000011" 结束于 r=5'},
    "fig2.note.7": {"en": "tie: see below", "zh": "平局：见下"},
    "fig2.tie.title": {"en": "tie at t=7 (q[7]=0): common-suffix length of Q[:8] with K[:r+1]",
                       "zh": "t=7 处的平局（q[7]=0）：Q[:8] 与 K[:r+1] 的公共后缀长度"},
    "fig2.tie.verdict": {"en": "r=0,1,2,3 all tie at length 1  ->  latest wins  ->  route[7] = 3",
                         "zh": "r=0,1,2,3 全部以长度 1 平局  ->  最晚者胜  ->  route[7] = 3"},
    "fig2.tie.note": {"en": "(the match is truncated: it never reaches back to K[0])",
                      "zh": "（注意该匹配是截断的：它不会一路回溯到 K[0]）"},

    # fig04 — flip census
    "fig4.title": {"en": "Figure 4 — flip census: what each of the 24 bits does (positions whose (ell, route) change; credit)",
                   "zh": "图 4 —— 翻转普查：24 个比特各自的作用（(ell, route) 发生变化的位置；credit）"},
    "fig4.qpanel": {"en": "q flips", "zh": "q 翻转"},
    "fig4.kpanel": {"en": "k flips", "zh": "k 翻转"},
    "fig4.vpanel": {"en": "v flips", "zh": "v 翻转"},
    "fig4.vwork": {"en": "worked example, v[1]: flipping v[1] changes no ell/route; only y[1] flips sign",
                   "zh": "演算示例 v[1]：翻转 v[1] 不改变任何 ell/route；只有 y[1] 变号"},
    "fig4.vcalc": {"en": "credit_v[1] = grad_y[1] · ((+emb1) − (−emb1)) = {gy1} × {two_e1} = {cv1}  ✓",
                   "zh": "credit_v[1] = grad_y[1] · ((+emb1) − (−emb1)) = {gy1} × {two_e1} = {cv1}  ✓"},
    "fig4.qwork": {"en": "worked example, q[2] (credit +0.162175): the flip breaks one match and shortens four",
                   "zh": "演算示例 q[2]（credit +0.162175）：这次翻转打断 1 个匹配、缩短 4 个匹配"},
    "fig4.qhead": {"en": "  t   ell->ell'  route->route'   y -> y'",
                   "zh": "  t   ell->ell'  route->route'   y -> y'"},
    "fig4.qrow.2": {"en": "  2    2 -> 0      1 -> -1     +0.120363 -> -0.111467    output becomes emb0",
                    "zh": "  2    2 -> 0      1 -> -1     +0.120363 -> -0.111467    输出变为 emb0"},
    "fig4.qrow.3": {"en": "  3    3 -> 1      2 ->  2     -0.120363 -> -0.120363    route changed, y did NOT",
                    "zh": "  3    3 -> 1      2 ->  2     -0.120363 -> -0.120363    route 变了，y 没有"},
    "fig4.qrow.4": {"en": "  4    4 -> 2      3 ->  3      unchanged                 (payload V[3] is the same)",
                    "zh": "  4    4 -> 2      3 ->  3      不变                     （负载 V[3] 相同）"},
    "fig4.qrow.5": {"en": "  5    5 -> 3      4 ->  4      unchanged",
                    "zh": "  5    5 -> 3      4 ->  4      不变"},
    "fig4.qrow.6": {"en": "  6    6 -> 4      5 ->  5      unchanged",
                    "zh": "  6    6 -> 4      5 ->  5      不变"},

    # fig05 — credit-to-logit map
    "fig5.title": {"en": "Figure 5 — from credit to logit: the orient map, with measured numbers (D=2 demo, τ = 0.5)",
                   "zh": "图 5 —— 从 credit 到 logit：orient 映射与实测数字（D = 2 演示，τ = 0.5）"},
    "fig5.bit0": {"en": "bit = 0  (z < 0)", "zh": "bit = 0（z < 0）"},
    "fig5.bit1": {"en": "bit = 1  (z > 0)", "zh": "bit = 1（z > 0）"},
    "fig5.thr": {"en": "threshold z = 0", "zh": "阈值 z = 0"},
    "fig5.up": {"en": "flip pushes z upward across it:  orient = +1",
                "zh": "翻转把 z 向上推过阈值：orient = +1"},
    "fig5.down": {"en": "flip pushes z downward across it:  orient = −1",
                  "zh": "翻转把 z 向下推过阈值：orient = −1"},
    "fig5.work": {"en": "position 1, bit-plane 1:  credit = +0.049447, bit = 0 -> orient = +1",
                  "zh": "位置 1，比特平面 1：credit = +0.049447, bit = 0 -> orient = +1"},
    "fig5.z": {"en": "z = 0.7507, tau = 0.5:  sigma(z/tau) = 0.8178",
               "zh": "z = 0.7507, tau = 0.5：sigma(z/tau) = 0.8178"},
    "fig5.id": {"en": "identity : grad = +1 × 0.049447                    = +0.049447",
                "zh": "identity : grad = +1 × 0.049447                    = +0.049447"},
    "fig5.bern": {"en": "bernoulli: grad = 0.8178 × 0.1822 / 0.5 × 0.049447 = +0.014737",
                  "zh": "bernoulli: grad = 0.8178 × 0.1822 / 0.5 × 0.049447 = +0.014737"},

    # fig08 — one position's forward
    "fig8.title": {"en": "Figure 8 — the forward for one position (t = 6): two static queries + one LCP",
                   "zh": "图 8 —— 单个位置（t = 6）的前向：两次静态查询 + 一次 LCP"},
    "fig8.tag": {"en": "text =", "zh": "text ="},
    "fig8.revQ": {"en": "reverse(Q)", "zh": "reverse(Q)"},
    "fig8.revK": {"en": "reverse(K)", "zh": "reverse(K)"},
    "fig8.s0": {"en": "t = 6:   x = T-1-t = 1     qpos = 1    rank[qpos] = 16",
                "zh": "t = 6:   x = T-1-t = 1     qpos = 1    rank[qpos] = 16"},
    "fig8.s1": {"en": "step 1 (two-sided neighbor ranks, value constraint p > 1):",
                "zh": "第 1 步（双侧邻居 rank，取值约束 p > 1）："},
    "fig8.s1a": {"en": "   find_last (rank < 16):  rank 15 -> p = 2,  LCP(qpos, sa[15]) = 6",
                 "zh": "   find_last (rank < 16):  rank 15 -> p = 2,  LCP(qpos, sa[15]) = 6"},
    "fig8.s1b": {"en": "   find_first(rank > 16):  none",
                 "zh": "   find_first(rank > 16)：无"},
    "fig8.s1c": {"en": "   best length so far: 6",
                 "zh": "   目前最优长度：6"},
    "fig8.s2": {"en": "step 2 (winner endpoint inside the rank interval of length-6 matches):",
                "zh": "第 2 步（在长度-6 匹配的 rank 区间内找胜出的端点）："},
    "fig8.s2a": {"en": "   rank interval for L = 6: [15, 17]",
                 "zh": "   L = 6 的 rank 区间：[15, 17]"},
    "fig8.s2b": {"en": "   smallest p > 1 in that interval: p* = 2   ->   endpoint e* = T-1-p* = 5",
                 "zh": "   该区间内满足 p > 1 的最小 p：p* = 2   ->   端点 e* = T-1-p* = 5"},
    "fig8.res": {"en": "=>  ell[6] = 6, route[6] = 5      (matches Figure 2)",
                 "zh": "=>  ell[6] = 6, route[6] = 5      （与图 2 一致）"},

    # fig16 — DRS scan
    "fig16.title": {"en": "Figure 16 — the DRS scan: 14 requested points, 8 skipped by the payload LCE, 2 materialized",
                    "zh": "图 16 —— DRS 扫描：14 个被请求的点，负载 LCE 跳过 8 个，物化 2 个"},
    "fig16.g2": {"en": "shift=-2 group: terms [5,5] (owner k=2), [6,6] (owner q=6), [7,7] (owner k=4)",
                 "zh": "shift=-2 组：terms [5,5]（属主 k=2）、[6,6]（属主 q=6）、[7,7]（属主 k=4）"},
    "fig16.g2m": {"en": "merged interval: [5,7];  candidate payload V[t-2];  baseline label[t] = v[route[t]+1]",
                  "zh": "合并区间：[5,7]；候选负载 V[t-2]；基线 label[t] = v[route[t]+1]"},
    "fig16.head": {"en": "   t    candidate V[t-2]    baseline label    verdict",
                   "zh": "   t       候选 V[t-2]         基线 label     判定"},
    "fig16.r5": {"en": "   5       V[3] = 0            V[6] = 1       MISMATCH  -> materialize",
                 "zh": "   5       V[3] = 0            V[6] = 1       不匹配    -> 物化"},
    "fig16.r67": {"en": "   6,7     V[4],V[5] = 1,1     V[7],V[4] = 1,1 equal     -> one LCE jump of 2",
                  "zh": "   6,7     V[4],V[5] = 1,1     V[7],V[4] = 1,1 相等      -> 一次 LCE 跳过 2 个点"},
    "fig16.g1": {"en": "shift=-1 group: merged [5,7];  t=5: V[4] = label[5] = 1,  LCE = 3",
                 "zh": "shift=-1 组：合并 [5,7]；t=5: V[4] = label[5] = 1,  LCE = 3"},
    "fig16.g1s": {"en": "   -> one jump skips t = 5,6,7  (3 points at once)",
                  "zh": "   -> 一次跳跃跳过 t = 5,6,7（一次 3 个点）"},
    "fig16.s5": {"en": "shift=-5: [7,7]  V[2]=1 == label[7]=1  skip",
                 "zh": "shift=-5: [7,7]  V[2]=1 == label[7]=1  跳过"},
    "fig16.s4": {"en": "shift=-4: [7,7]  V[3]=0 != 1  MISMATCH",
                 "zh": "shift=-4: [7,7]  V[3]=0 != 1  不匹配"},
    "fig16.s3": {"en": "shift=-3: [7,7]  V[4]=1 == 1           skip",
                 "zh": "shift=-3: [7,7]  V[4]=1 == 1           跳过"},
    "fig16.s0": {"en": "shift= 0: [7,7]  V[7]=1 == 1  skip",
                 "zh": "shift= 0: [7,7]  V[7]=1 == 1  跳过"},
    "fig16.sum": {"en": "total: 8 skipped, 2 materialized  ->  two prefix-sum arrays answer all 12 owner queries",
                  "zh": "合计：跳过 8 个，物化 2 个  ->  两个前缀和数组回答全部 12 次属主查询"},
    "fig16.sum2": {"en": "(drs_queries=12 = q_overlay 4 + k_overlay 8)",
                   "zh": "（drs_queries=12 = q_overlay 4 + k_overlay 8）"},

    # fig17 — K-side sweep
    "fig17.title": {"en": "Figure 17 — the K-side sweep at owner s = 2 (the real runs of section 6)",
                    "zh": "图 17 —— 属主 s = 2 处的 K 侧扫描线（第 6 节的真实 run）"},
    "fig17.head": {"en": "deletion runs touching owner s=2 (from section 6):",
                   "zh": "触及属主 s=2 的删除 run（来自第 6 节）："},
    "fig17.t3": {"en": "   t=3: [s in 2..2]  route (2,1):  payload V[2]=1 vs baseline V[3]=0",
                 "zh": "   t=3: [s in 2..2]  route (2,1):  负载 V[2]=1 对基线 V[3]=0"},
    "fig17.t3b": {"en": "         -> score delta = grad_y[3] · (+emb1 − (−emb1)) = −1.302297 × 0.240726",
                  "zh": "         -> 得分差 = grad_y[3] · (+emb1 − (−emb1)) = −1.302297 × 0.240726"},
    "fig17.t4": {"en": "   t=4: [s in 2..3]  route (2,1):  payload V[2]=1 vs baseline V[4]=1 -> 0",
                 "zh": "   t=4: [s in 2..3]  route (2,1):  负载 V[2]=1 对基线 V[4]=1 -> 0"},
    "fig17.t5": {"en": "   t=5: [s in 0..3]  route (2,4):  payload V[5]=1 vs baseline V[5]=1 -> 0",
                 "zh": "   t=5: [s in 0..3]  route (2,4):  负载 V[5]=1 对基线 V[5]=1 -> 0"},
    "fig17.sweep": {"en": "sweep at owner 2: total = −0.3135  (only t=3 moves)",
                    "zh": "属主 2 处的扫描线：total = −0.3135（仅 t=3 有变化）"},
    "fig17.ov1": {"en": "overlay: + DRS term [5,5] shift=-2: +0.0649 − sweep.range_sum(5,5)=0",
                  "zh": "覆盖：+ DRS 项 [5,5] shift=-2：+0.0649 − sweep.range_sum(5,5)=0"},
    "fig17.ov2": {"en": "         + DRS term [7,7] shift=-3:  0 (skipped: equal sign)",
                  "zh": "         + DRS 项 [7,7] shift=-3：0（跳过：符号相同）"},
    "fig17.res": {"en": "credit_k[2] = −0.3135 + 0.0649 = −0.2486   ✓ (matches the brute-force table)",
                  "zh": "credit_k[2] = −0.3135 + 0.0649 = −0.2486   ✓（与暴力表一致）"},
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

    # ---- fig2: per-position outputs and the t=7 tie-break ----------------
    yv = [round(float(x), 6) for x in y.flatten()]
    assert yv == [-0.111467, -0.120363, 0.120363, -0.120363,
                  0.120363, 0.120363, 0.120363, 0.120363]
    tieL = [index.lcs_end(7, r) for r in range(7)]
    assert tieL == [1, 1, 1, 1, 0, 0, 0]

    # ---- fig4: flip census (flip every bit, rerun the hard forward) ------
    def _census(side):
        out = []
        for u in range(T):
            if side == "q":
                q2 = list(q); q2[u] ^= 1
                _, e2, r2 = H.forward_naive(q2, k, v_bits, emb0, emb1, D)
            elif side == "k":
                k2 = list(k); k2[u] ^= 1
                _, e2, r2 = H.forward_naive(q, k2, v_bits, emb0, emb1, D)
            else:
                v2 = v_bits.clone(); v2[u, 0] ^= 1
                _, e2, r2 = H.forward_naive(q, k, v2, emb0, emb1, D)
            out.append([t for t in range(T)
                        if e2[t] != ell[t] or r2[t] != route[t]])
        return out
    cen = {s: _census(s) for s in ("q", "k", "v")}
    assert cen["q"] == [[], [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6],
                        [3, 4, 5, 6], [4, 5, 6], [5, 6], [6, 7], [7]]
    assert cen["k"] == [[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7],
                        [3, 4, 5, 6, 7], [4, 5, 6, 7], [5, 6, 7], [6, 7],
                        [7], []]
    assert cen["v"] == [[] for _ in range(T)]
    credit = {"q": [float(x) for x in bq.flatten()],
              "k": [float(x) for x in bk.flatten()],
              "v": [float(x) for x in bv.flatten()]}
    assert round(credit["q"][2], 6) == 0.162175
    assert round(credit["k"][2], 6) == -0.248592
    assert round(credit["v"][1], 6) == 0.468436
    gy = [float(x) for x in grad_y.flatten()]
    e1f = float(emb1[0])
    assert round(gy[1], 6) == 1.945929 and round(2 * e1f, 6) == 0.240726

    # ---- fig5: the D=2 packing + credit->logit demo ----------------------
    torch.manual_seed(21)
    qb2 = torch.randint(0, 2, (8, 2), dtype=torch.uint8)
    kb2 = torch.randint(0, 2, (8, 2), dtype=torch.uint8)
    vb2 = torch.randint(0, 2, (8, 2), dtype=torch.uint8)
    torch.manual_seed(22)
    gy2 = torch.randn(8, 2)
    e02 = torch.randn(2)
    e12 = torch.randn(2)
    qsym = H._pack_group_bits_to_python_ints(qb2)
    ksym = H._pack_group_bits_to_python_ints(kb2)
    assert qsym == [3, 0, 0, 2, 0, 3, 0, 2] and ksym == [2, 3, 1, 0, 0, 0, 1, 1]
    fq2, _, _, _, _, _, _ = H.exact_stream_bit_credits(
        qsym, ksym, vb2, gy2, e02, e12, 2)
    c5 = float(fq2[1, 1])
    assert round(c5, 6) == 0.049447 and int(qb2[1, 1]) == 0
    torch.manual_seed(23)
    z5 = torch.randn(8, 2)
    z11 = float(z5[1, 1])
    assert round(z11, 4) == 0.7507
    sig5 = 1.0 / (1.0 + math.exp(-z11 / 0.5))
    assert round(sig5, 4) == 0.8178
    assert round(sig5 * (1 - sig5) / 0.5 * c5, 6) == 0.014737

    # ---- fig8: the static-certificate forward for t=6, step by step ------
    qi, ki = H._remap_symbols(q, k)
    text8 = list(reversed(qi)) + [0] + list(reversed(ki)) + [1]
    assert text8 == [2, 3, 3, 2, 2, 2, 2, 2, 0, 2, 3, 3, 3, 2, 2, 2, 2, 1]
    idx8 = H._SuffixArrayLCE(text8)
    rank_to_p8 = [-1] * len(text8)
    for p in range(T):
        rank_to_p8[idx8.rank[(T + 1) + p]] = p
    ext8 = H._StaticMaxPByRank(rank_to_p8)
    suc8 = H._StaticRangeSuccessorP(rank_to_p8)
    x8, qpos8 = 1, 1                       # x = T-1-t, qpos = q_off + x
    rq8 = idx8.rank[qpos8]
    rp8 = ext8.find_last(0, rq8, x8)
    rs8 = ext8.find_first(rq8 + 1, len(text8), x8)
    lcp8 = idx8.lcp_pos(qpos8, idx8.sa[rp8])
    lo8, hi8 = idx8.prefix_rank_interval(qpos8, lcp8)
    ps8 = suc8.range_successor(lo8, hi8, x8)
    es8 = T - 1 - ps8
    assert (rq8, rp8, rs8, lcp8, lo8, hi8, ps8, es8) == (16, 15, None, 6, 15, 17, 2, 5)
    assert rank_to_p8[15] == 2

    # ---- fig16: DRS counters ---------------------------------------------
    assert (stats.drs_requested_points, stats.drs_semantic_equal_skips,
            stats.drs_materialized_mismatches,
            stats.drs_queries) == (14, 8, 2, 12)

    # ---- fig17: K deletion runs at owner 2 and the score arithmetic ------
    assert kdel.route(3, 2) == (2, 1)      # (also asserted above)
    assert kdel.route(4, 2) == (2, 1)
    assert kdel.route(5, 2) == (2, 4)
    assert round(gy[3], 6) == -1.302297
    delta17 = gy[3] * (2 * e1f)
    overlay17 = credit["k"][2] - delta17
    assert round(delta17, 4) == -0.3135 and round(overlay17, 4) == 0.0649

    data.update(y=yv, tieL=tieL, cen=cen, credit=credit, gy=gy, e1=e1f,
                text8=text8)
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


def _stack(sc, name, lines, left_edge, below_obj, gap=6, top_gap=14,
           font_size=12, colors=None, bolds=None):
    """Vertical stack of left-aligned labels; every line is bound to the
    previous one (or to `below_obj` for the first), never to raw pixels."""
    out = []
    for i, text in enumerate(lines):
        lab = sc.label(f"{name}.{i}", text, font_size=font_size,
                       color=(colors[i] if colors else TEXT),
                       bold=(bolds[i] if bolds else False), anchor="start")
        lab.align_left(left_edge)
        lab.below(out[-1] if out else below_obj, gap if out else top_gap)
        out.append(lab)
    return out


def fig02_trace(data, lang):
    sc = Scene("fig02-position-trace", title=S("fig2.title", lang),
               desc="Position-by-position forward trace of the running "
                    "instance, including the tie-break at t = 7.")
    cap = _caption(sc, "fig2.title", lang)
    T = data["T"]
    payload = ["emb0"] + [f"V[{data['route'][t] + 1}]={data['v'][data['route'][t] + 1]}"
                          for t in range(1, T)]
    ystr = [f"{v:+.6f}" for v in data["y"]]
    cols = [("t", "fig2.col.t", [str(t) for t in range(T)], 30, "plain"),
            ("q", "fig2.col.q", [str(x) for x in data["q"]], 40, "blue"),
            ("ell", "fig2.col.ell", [str(x) for x in data["ell"]], 44, "blue"),
            ("route", "fig2.col.route", [str(x) for x in data["route"]], 58, "blue"),
            ("payload", "fig2.col.payload", payload, 84, "cell"),
            ("y", "fig2.col.y", ystr, 92, "cell")]
    grids = []
    for name, key, values, cw, base_style in cols:
        g = sc.grid(f"col.{name}", T, 1, cw=cw, ch=26, font_size=12)
        if not grids:
            g.at(0, 0).below(cap, 36)
        else:
            g.right_of(grids[-1], 4).align_top(grids[0])
        for r, txt in enumerate(values):
            style = base_style
            if name == "payload" and r == 0:
                style = "gray"
            if name == "route" and r == 7:
                style = "green"
            g.cell(r, 0, txt, style)
        hd = sc.label(f"hd.{name}", S(key, lang), font_size=12, bold=True)
        hd.center_x_on(g).above(g, 6)
        grids.append(g)
    for t in range(T):
        lab = sc.label(f"note.{t}", S(f"fig2.note.{t}", lang), font_size=12,
                       color="#6b7280", anchor="start")
        lab.right_of(grids[-1], 14).center_y_on(grids[0].cell(t, 0))
    tt = sc.label("tie.title", S("fig2.tie.title", lang), font_size=12,
                  bold=True, anchor="start")
    tt.align_left(grids[0]).below(grids[0], 20)
    strip = sc.grid("tie", 1, 7, cw=40, ch=26, font_size=12)
    strip.align_left(grids[0]).below(tt, 22)
    for c in range(7):
        strip.cell(0, c, f"L={data['tieL'][c]}", "green" if c <= 3 else "cell")
    for c in range(7):
        lab = sc.label(f"tie.idx.{c}", f"r={c}", font_size=11, color="#9ca3af")
        lab.center_x_on(strip.cell(0, c)).above(strip.cell(0, c), 4)
    v = sc.label("tie.verdict", S("fig2.tie.verdict", lang), font_size=12,
                 color=GREEN, anchor="start")
    v.align_left(strip).below(strip, 10)
    n = sc.label("tie.note", S("fig2.tie.note", lang), font_size=12,
                 color="#6b7280", anchor="start")
    n.align_left(strip).below(v, 6)
    return sc


def _fmt_set(changed):
    if not changed:
        return "{}"
    if len(changed) == 1:
        return f"{{{changed[0]}}}"
    if len(changed) == 2:
        return f"{{{changed[0]},{changed[1]}}}"
    if len(changed) == changed[-1] - changed[0] + 1:
        return f"{{{changed[0]}..{changed[-1]}}}"
    return "{" + ",".join(map(str, changed)) + "}"


def fig04_census(data, lang):
    sc = Scene("fig04-flip-census", title=S("fig4.title", lang),
               desc="Census of all 24 bit flips of the running instance, "
                    "plus one worked example per side.")
    cap = _caption(sc, "fig4.title", lang)
    panels = []
    for side, key in (("q", "fig4.qpanel"), ("k", "fig4.kpanel"),
                      ("v", "fig4.vpanel")):
        lines = []
        colors = []
        for u in range(data["T"]):
            c = data["credit"][side][u]
            cs = "0.0" if abs(c) < 5e-7 else f"{c:+.6f}"
            lines.append(f"{side}[{u}]:  {_fmt_set(data['cen'][side][u]):<10} {cs}")
            colors.append(TEXT if abs(c) >= 5e-7 else "#9ca3af")
        title = sc.label(f"p.{side}.t", S(key, lang), font_size=12, bold=True,
                         anchor="start")
        if not panels:
            title.at(0, 0).below(cap, 32)
        else:
            title.right_of(panels[-1], 40).align_top(panels[-1][0])
        stack = _stack(sc, f"p.{side}", lines, title, title, gap=5, top_gap=8,
                       colors=colors)
        panels.append([title] + stack)
    vlines = [S("fig4.vwork", lang),
              S("fig4.vcalc", lang, gy1=f"{data['gy'][1]:.6f}",
                two_e1=f"{2 * data['e1']:.6f}",
                cv1=f"{data['credit']['v'][1]:.6f}")]
    vstack = _stack(sc, "work.v", vlines, panels[0][0], panels[0],
                    gap=6, top_gap=22, colors=[TEXT, GREEN])
    qlines = [S("fig4.qwork", lang), S("fig4.qhead", lang),
              S("fig4.qrow.2", lang), S("fig4.qrow.3", lang),
              S("fig4.qrow.4", lang), S("fig4.qrow.5", lang),
              S("fig4.qrow.6", lang)]
    _stack(sc, "work.q", qlines, panels[0][0], vstack, gap=5, top_gap=16,
           colors=[TEXT, "#6b7280", RED, "#6b7280", "#6b7280", "#6b7280",
                   "#6b7280"])
    return sc


def fig05_logit_map(data, lang):
    sc = Scene("fig05-logit-map", title=S("fig5.title", lang),
               desc="The orient map from bit-flip credit to logit gradient, "
                    "with one measured example.")
    cap = _caption(sc, "fig5.title", lang)
    lb = sc.box("bit0", S("fig5.bit0", lang), style="blue", font_size=12)
    lb.at(0, 0).below(cap, 70)
    thr = sc.box("thr", S("fig5.thr", lang), style="ink", font_size=12)
    thr.right_of(lb, 380).center_y_on(lb)
    rb = sc.box("bit1", S("fig5.bit1", lang), style="blue", font_size=12)
    rb.right_of(thr, 380).center_y_on(lb)
    sc.connect(lb, "right", thr, "left", color=RED,
               label=S("fig5.up", lang), label_side="above")
    sc.connect(rb, "left", thr, "right", color=RED,
               label=S("fig5.down", lang), label_side="below")
    lines = [S("fig5.work", lang), S("fig5.z", lang),
             S("fig5.id", lang), S("fig5.bern", lang)]
    _stack(sc, "work", lines, lb, [lb, thr, rb], gap=8, top_gap=56,
           colors=[TEXT, BLUE, TEXT, GREEN])
    return sc


def fig08_forward_query(data, lang):
    sc = Scene("fig08-forward-query", title=S("fig8.title", lang),
               desc="The static-certificate forward for t = 6 of the running "
                    "instance, as two 2D range queries plus one LCP.")
    cap = _caption(sc, "fig8.title", lang)
    t8 = data["text8"]
    g = sc.grid("text", 1, len(t8), cw=34, ch=28, font_size=12)
    g.at(0, 0).below(cap, 48)
    for c, val in enumerate(t8):
        g.cell(0, c, str(val), "gray" if c in (8, 17) else "blue")
    _index_labels(sc, g, list(range(len(t8))))
    tag = sc.label("tag", S("fig8.tag", lang), font_size=12, anchor="end")
    tag.left_of(g.cell(0, 0), 8).center_y_on(g.cell(0, 0))
    contQ = sc.enclose("revQ", g.row_cells(0, 0, 7), padding=6, style="blue")
    lq = sc.label("revQ.lab", S("fig8.revQ", lang), font_size=11, color=BLUE)
    lq.center_x_on(contQ).below(contQ, 6)
    contK = sc.enclose("revK", g.row_cells(0, 9, 16), padding=6, style="blue")
    lk = sc.label("revK.lab", S("fig8.revK", lang), font_size=11, color=BLUE)
    lk.center_x_on(contK).below(contK, 6)
    keys = ["fig8.s0", "fig8.s1", "fig8.s1a", "fig8.s1b", "fig8.s1c",
            "fig8.s2", "fig8.s2a", "fig8.s2b", "fig8.res"]
    colors = [TEXT, BLUE, BLUE, BLUE, BLUE, BLUE, BLUE, BLUE, GREEN]
    _stack(sc, "steps", [S(kk, lang) for kk in keys], g, [lq, lk],
           gap=7, top_gap=22, colors=colors)
    return sc


def fig16_drs(data, lang):
    sc = Scene("fig16-drs-scan", title=S("fig16.title", lang),
               desc="The difference-range scan: payload LCE skips the points "
                    "where candidate and baseline payloads agree.")
    cap = _caption(sc, "fig16.title", lang)
    g2 = _stack(sc, "g2", [S("fig16.g2", lang), S("fig16.g2m", lang),
                           S("fig16.head", lang), S("fig16.r5", lang),
                           S("fig16.r67", lang)], cap, cap, gap=7, top_gap=16,
                colors=[BLUE, BLUE, TEXT, RED, GREEN],
                bolds=[False, False, True, False, False])
    g1 = _stack(sc, "g1", [S("fig16.g1", lang), S("fig16.g1s", lang)],
                cap, g2[-1], gap=7, top_gap=16, colors=[BLUE, GREEN])
    s5 = sc.label("s5", S("fig16.s5", lang), font_size=12, color=GREEN,
                  anchor="start")
    s5.align_left(cap).below(g1[-1], 16)
    s4 = sc.label("s4", S("fig16.s4", lang), font_size=12, color=RED,
                  anchor="start")
    s4.right_of(s5, 24).center_y_on(s5)
    s3 = sc.label("s3", S("fig16.s3", lang), font_size=12, color=GREEN,
                  anchor="start")
    s3.align_left(s5).below(s5, 7)
    s0 = sc.label("s0", S("fig16.s0", lang), font_size=12, color=GREEN,
                  anchor="start")
    s0.right_of(s3, 24).center_y_on(s3)
    summ = sc.box("sum", [S("fig16.sum", lang), S("fig16.sum2", lang)],
                  style="ink", font_size=12)
    summ.align_left(cap).below([s3, s0], 16)
    return sc


def fig17_sweep(data, lang):
    sc = Scene("fig17-k-sweep", title=S("fig17.title", lang),
               desc="The K-side deletion sweep at owner s = 2, checked "
                    "against the brute-force credit table.")
    cap = _caption(sc, "fig17.title", lang)
    keys = ["fig17.head", "fig17.t3", "fig17.t3b", "fig17.t4", "fig17.t5",
            "fig17.sweep", "fig17.ov1", "fig17.ov2"]
    colors = [TEXT, BLUE, BLUE, "#9ca3af", "#9ca3af", TEXT, TEXT, "#9ca3af"]
    bolds = [True, False, False, False, False, False, False, False]
    stack = _stack(sc, "sw", [S(kk, lang) for kk in keys], cap, cap,
                   gap=7, top_gap=16, colors=colors, bolds=bolds)
    res = sc.box("res", S("fig17.res", lang), style="green", font_size=12)
    res.align_left(cap).below(stack[-1], 16)
    return sc


FIGURES = {
    "fig00-legend": fig00_legend,
    "fig01-running-instance": fig01_running_instance,
    "fig02-position-trace": fig02_trace,
    "fig03-structural-zeros": fig03_structural_zeros,
    "fig04-flip-census": fig04_census,
    "fig05-logit-map": fig05_logit_map,
    "fig06-naive-vs-fast": fig06_naive_vs_fast,
    "fig07-pipeline": fig07_pipeline,
    "fig08-forward-query": fig08_forward_query,
    "fig09-q-latest-heads": fig09_q_heads,
    "fig10-k-delete-reindex": fig10_k_delete,
    "fig11-repair-thresholds": fig11_thresholds,
    "fig12-shared-bridge": fig12_bridge,
    "fig13-bridge-envelope": fig13_envelope,
    "fig14-run-certificates": fig14_certificates,
    "fig15-shadow-repair": fig15_shadow,
    "fig16-drs-scan": fig16_drs,
    "fig17-k-sweep": fig17_sweep,
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
