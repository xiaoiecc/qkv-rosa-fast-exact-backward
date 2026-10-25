"""Rebuild the bilingual tutorial figures from independently checked data.

Run from any directory: python images/generate_figures.py
Only the standard library is required. No optimized matching code is imported.
"""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent
INK = "#172033"
MUTED = "#56657a"
BLUE = "#2563eb"
TEAL = "#0f766e"
PURPLE = "#7c3aed"
AMBER = "#b45309"
RED = "#b91c1c"
LINE = "#d8e1ec"


def suffix(q, k, t, r):
    n = 0
    while t - n >= 0 and r - n >= 0 and q[t - n] == k[r - n]:
        n += 1
    return n


def best(q, k, t):
    return max(((n, r) if n else (0, -1)
                for r in range(t) for n in [suffix(q, k, t, r)]),
               default=(0, -1))


def data():
    q = k = [0, 1, 0, 1, 0]
    f = [suffix(q, k, 4, r) for r in range(4)]
    qflip = q.copy()
    qflip[4] ^= 1
    deletion = []
    for s in range(4):
        a = max([(f[r], r) if f[r] else (0, -1) for r in range(s)],
                default=(0, -1))
        h = max([(z, r) if z else (0, -1)
                 for r in range(s, 4) for z in [min(f[r], r - s)]],
                default=(0, -1))
        kk = k.copy()
        kk[s] = None
        assert max(a, h) == best(q, kk, 4)
        deletion.append({"owner": s, "A": a, "H": h, "winner": max(a, h)})
    qdeletion = []
    for u in range(5):
        qq = q.copy()
        qq[u] = None
        qdeletion.append(best(qq, k, 4))
    v = [0, 0, 0, 0, 1]
    old, new = best(q, k, 4), best(qflip, k, 4)
    credit = 3 * ((2 * v[new[1] + 1] - 1) * 2 - (2 * v[old[1] + 1] - 1) * 2)
    assert old == (3, 2) and new == (1, 3) and credit == 12
    runq = list(map(int, "111101000"))
    runk = list(map(int, "010000001"))
    runf = [suffix(runq, runk, 8, r) for r in range(2, 8)]
    assert runf == [1, 2, 5, 3, 3, 3]
    cq, ck = [0] * 5, [1, 0, 0, 0, 0]
    corridors = []
    for u in (1, 2):
        qq = cq.copy()
        qq[u] = 1
        routes = [best(qq, ck, t) for t in range(u, 5)]
        alpha, beta = 1 - u, -u
        assert routes == [(t + alpha, t + beta) for t in range(u, 5)]
        corridors.append({"q_owner": u, "start": u, "stop": 4,
                          "alpha": alpha, "beta": beta, "k_owner": u + beta,
                          "routes": routes})
    events = [20000000000000000, 2, 2, -20000000000000000]
    rounded, exact, ordinary, prefixes = 0.0, 0, [], []
    for x in events:
        rounded += float(x)
        exact += x
        ordinary.append(rounded)
        prefixes.append(exact)
    assert rounded == 0 and exact == 4
    return {"primary": {"Q": q, "K": k, "t": 4, "F": f,
                        "baseline": old, "flip_Q4": new, "credit_Q4": credit,
                        "Q_deletion": qdeletion, "K_deletion": deletion},
            "run": {"Q": runq, "K": runk, "t": 8, "h": 3, "l": 2,
                    "endpoints": list(range(2, 8)), "F": runf},
            "causality": {"K": k, "t": 4, "terminal_order": [0, 2, 4, 1, 3],
                          "zero_endpos": [0, 2, 4], "causal_zero_endpos": [0, 2]},
            "continuations": {"Q": cq, "K": ck, "intervals": corridors},
            "summation": {"events": events, "ordinary_prefixes": ordinary,
                          "exact_prefixes": prefixes}}


class Svg:
    def __init__(self, title, subtitle, height=440, zh=False):
        self.zh = zh
        self.h = height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}" role="img">',
                      f'<title>{escape(title)}</title>',
                      f'<desc>{escape(subtitle)}</desc>',
                      '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#56657a"/></marker></defs>',
                      '<rect width="900" height="100%" fill="#f8fafc"/>',
                      '<g font-family="DejaVu Sans, Noto Sans CJK SC, Microsoft YaHei, sans-serif">']
        self.text(28, 36, title, 23, INK, weight=700)
        self.text(28, 61, subtitle, 13, MUTED)

    def text(self, x, y, s, size=14, color=INK, anchor="start", weight=400):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{escape(str(s))}</text>')

    def lines(self, x, y, items, size=14, color=MUTED, gap=22):
        for i, s in enumerate(items):
            self.text(x, y + i * gap, s, size, color)

    def rect(self, x, y, w, h, fill="white", stroke=LINE, radius=10):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')

    def line(self, x1, y1, x2, y2, color=LINE, width=1.5, arrow=False, dash=False):
        extra = (' marker-end="url(#arrow)"' if arrow else '') + (' stroke-dasharray="5 4"' if dash else '')
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{extra}/>')

    def path(self, points, color, width=3):
        path = " ".join(f'{"M" if i == 0 else "L"}{x},{y}' for i, (x, y) in enumerate(points))
        self.parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>')
        for x, y in points:
            self.parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')

    def word(self, x, y, label, bits, highlights=(), color=TEAL, cross=None, cell=44):
        self.text(x, y + 27, label, 15, INK, weight=700)
        for i, bit in enumerate(bits):
            xx = x + 56 + i * cell
            selected = i in highlights
            self.rect(xx, y, cell - 6, 39, "#dff5ef" if selected else "white", color if selected else LINE, 6)
            self.text(xx + (cell - 6) / 2, y + 27, bit, 20, color if selected else INK, "middle", 700)
            if cross == i:
                self.rect(xx - 2, y - 2, cell - 2, 43, "none", AMBER, 8)

    def save(self, stem):
        self.parts.extend(['</g>', '</svg>'])
        suffix = ".zh-CN" if self.zh else ""
        (OUT / f"{stem}{suffix}.svg").write_text("\n".join(self.parts) + "\n", encoding="utf-8")


def repeated(zh, d):
    s = Svg("从重跑中找共享的工作" if zh else "Find shared work inside the brute force",
            "一个 Q bit 家族：每次翻转重算全部行，每行又扫描所有终点与字符。" if zh else "One Q-bit family: each flip recomputes every row, every endpoint, and every matching character.", 462, zh)
    s.rect(28, 86, 505, 305)
    s.text(46, 111, "输出行 t" if zh else "Output row t", 14, MUTED)
    x0, y0, cw, ch = 136, 135, 74, 45
    for t in range(5):
        s.text(x0 + t * cw + 32, 129, t, 14, INK, "middle", 700)
    for u in range(5):
        s.text(44, y0 + u * ch + 25, f"flip Q[{u}]", 13, INK)
        for t in range(5):
            active = t >= u
            s.rect(x0 + t * cw, y0 + u * ch, 64, 36,
                   "#e3efff" if active else "#f1f5f9", "#bfdbfe" if active else LINE, 6)
            s.text(x0 + t * cw + 32, y0 + u * ch + 23,
                   "r < t" if active else ("不变" if zh else "same"), 13,
                   BLUE if active else MUTED, "middle")
    s.rect(556, 86, 316, 305)
    s.text(575, 115, "每个蓝格内部" if zh else "Inside each blue cell", 17, INK, weight=700)
    s.lines(575, 147, ["扫描因果终点 r < t" if zh else "Scan causal endpoints r < t",
                       "逐字符向左匹配" if zh else "Walk left through matching symbols",
                       "按 (长度, 终点) 取最大" if zh else "Maximize (length, endpoint)"], 13)
    s.line(575, 213, 850, 213)
    s.text(575, 241, "3TD × T × T × T", 21, BLUE, weight=700)
    s.text(575, 273, "O(DT⁴)" if not zh else "主导项 O(DT⁴)", 25, BLUE, weight=700)
    s.lines(575, 311, ["灰格完全相同；蓝格也共享" if zh else "Gray cells are identical; blue cells",
                       "不跨过 owner 的匹配。" if zh else "also share owner-avoiding matches."], 13)
    s.lines(28, 419, ["优化顺序：不变答案 → 共同删除 → 一处失配修复 → 压缩实际记录与生命周期。" if zh else
                       "Compression order: unchanged answers → common deletion → one repaired mismatch → records and lifetimes."], 13)
    s.save("01-repeated-work")


def partition(zh, d):
    s = Svg("先选硬路由，再把翻转分成删除与修复" if zh else "A hard route, then a complete delete / repair partition",
            "固定例子 Q = K = 01010，t = 4。所有路由都比较完整的 (长度, 终点)。" if zh else
            "Fixed example Q = K = 01010, t = 4. Compare the full pair (length, endpoint) at every maximum.", 500, zh)
    s.rect(28, 86, 409, 228)
    s.rect(463, 86, 409, 228)
    s.text(48, 116, "原始前向" if zh else "Baseline forward", 18, INK, weight=700)
    s.word(48, 137, "Q", d["primary"]["Q"], (2, 3, 4))
    s.word(48, 188, "K", d["primary"]["K"], (0, 1, 2))
    s.lines(48, 262, ["route = (3, 2) → payload P₃", "r = 4 不可见，尽管能匹配 5 个字符" if zh else "r = 4 is excluded, despite a length-5 match"], 13)
    s.text(483, 116, "仅翻转 Q[4]" if zh else "Flip only Q[4]", 18, INK, weight=700)
    s.word(483, 137, "Q", [0, 1, 0, 1, 1], (4,), cross=4)
    s.word(483, 188, "K", d["primary"]["K"], (3,))
    s.lines(483, 262, ["route = (1, 3) → payload P₄", "长度同为 1 时，终点 3 胜过终点 1" if zh else "Length ties: endpoint 3 beats endpoint 1"], 13)
    s.rect(28, 338, 247, 101, "#eef2ff", "#c7d2fe")
    s.text(46, 367, "共同删除" if zh else "Common deletion", 16, PURPLE, weight=700)
    s.text(46, 396, "Q owner 4 → (0, −1)", 14, INK)
    s.text(46, 422, "禁止跨 owner；不移动序列" if zh else "Forbid crossing; do not shift symbols", 11, MUTED)
    s.rect(328, 338, 247, 101, "#e3f5ee", "#b5ddce")
    s.text(346, 367, "修复最近一处失配" if zh else "Repair the nearest mismatch", 15, TEAL, weight=700)
    s.text(346, 396, "a = b = 0 → (1, 3)", 14, INK)
    s.text(346, 422, "Q[4] xor K[3] = 1", 12, MUTED)
    s.line(284, 385, 316, 385, MUTED, arrow=True)
    s.text(300, 368, "max", 12, MUTED, "middle")
    s.line(589, 385, 625, 385, MUTED, arrow=True)
    s.rect(638, 338, 234, 101, "#e3efff", "#bfdbfe")
    s.text(656, 367, "翻转信用" if zh else "Projected flip credit", 16, BLUE, weight=700)
    s.text(656, 398, "g₄(P₄ − P₃) = 12", 18, BLUE, weight=700)
    s.text(656, 422, "V₃=0, V₄=1, e₁=2, g₄=3", 12, MUTED)
    s.text(28, 472, "两类匹配的并集完备。路由长度变化本身不等于 payload 变化。" if zh else
           "The two match sets cover every flipped match. A changed match length alone need not change the payload.", 13, MUTED)
    s.save("02-route-partition")


def envelopes(zh, d):
    s = Svg("A 与 H 的几何：一边增，一边减" if zh else "The deletion geometry: A rises, H falls",
            "Q = K = 01010，t = 4，F(r) = [1, 0, 3, 0]。纵轴只画长度；表中保留终点。" if zh else
            "Q = K = 01010, t = 4, F(r) = [1, 0, 3, 0]. The plot shows length; the table retains endpoints.", 462, zh)
    s.rect(28, 87, 485, 299)
    xs = [88, 205, 322, 439]
    for h in range(4):
        yy = 332 - 63 * h
        s.line(77, yy, 472, yy)
        s.text(60, yy + 5, h, 13, MUTED, "middle")
    for x, owner in zip(xs, range(4)):
        s.text(x, 357, owner, 14, INK, "middle")
    s.text(78, 115, "长度" if zh else "Length", 13, MUTED)
    s.text(470, 376, "K owner s", 12, MUTED, "end")
    rows = d["primary"]["K_deletion"]
    s.path([(x, 332 - 63 * row["A"][0]) for x, row in zip(xs, rows)], TEAL)
    s.path([(x, 332 - 63 * row["H"][0]) for x, row in zip(xs, rows)], PURPLE)
    s.text(375, 135, "A", 18, TEAL, weight=700)
    s.text(82, 188, "H", 18, PURPLE, weight=700)
    s.rect(536, 87, 336, 299)
    s.text(553, 119, "s", 14, MUTED)
    s.text(604, 119, "A(s)", 14, TEAL, "middle", 700)
    s.text(684, 119, "H(s)", 14, PURPLE, "middle", 700)
    s.text(798, 119, "max(A,H)", 14, BLUE, "middle", 700)
    for i, row in enumerate(rows):
        yy = 158 + i * 42
        if i == 2:
            s.rect(545, yy - 24, 317, 35, "#fff4db", "#f5cf89", 6)
        s.text(553, yy, i)
        for x, key in [(604, "A"), (684, "H"), (798, "winner")]:
            val = row[key]
            s.text(x, yy, f"({val[0]}, {val[1]})", 14, BLUE if key == "winner" else INK, "middle")
    s.lines(553, 328, ["s=1：同长，H 的终点更新。" if zh else "At s=1, H wins the length tie.",
                       "s=2：完整字典序的最低点。" if zh else "At s=2, the full route is minimal."], 12)
    s.lines(28, 414, ["任意 owner 区间的最小删除路由：把全局谷底 owner 夹到该区间，再查询一次。" if zh else
                       "Minimum deletion over an owner interval: clamp a global valley owner into that interval, then query once.",
                       "长度相等也不能丢弃 endpoint：s=1 的 (1,2) 大于 s=2 的 (1,0)。" if zh else
                       "Equal lengths do not erase endpoint order: (1,2) at s=1 is greater than (1,0) at s=2."], 13)
    s.save("03-deletion-envelope")


def runs(zh, d):
    s = Svg("整段公式只有一个必须精确查询的边界" if zh else "A whole run formula, with one necessary exact boundary",
            "Q = 111101000，K = 010000001，t = 8。Q 尾 run 长 h=3，K 的零 run 从 l=2 开始。" if zh else
            "Q = 111101000, K = 010000001, t = 8. Q has tail-run length h=3; the K zero-run starts at l=2.", 432, zh)
    s.rect(28, 86, 844, 263)
    xs = [215 + 117 * i for i in range(6)]
    s.text(49, 126, "K endpoint r", 14, MUTED)
    s.text(49, 189, "min(h, r−l+1)", 14, MUTED)
    s.text(49, 252, "F(r)", 17, INK, weight=700)
    for i, (x, r, f) in enumerate(zip(xs, d["run"]["endpoints"], d["run"]["F"])):
        special = r == 4
        s.text(x, 126, r, 18, INK, "middle", 700)
        s.rect(x - 33, 158, 66, 45, "#f1f5f9", LINE, 7)
        s.text(x, 189, min(3, r - 1), 20, MUTED, "middle")
        s.rect(x - 33, 221, 66, 45, "#fff2d8" if special else "#def3ec", "#e8b963" if special else "#b5ddce", 7)
        s.text(x, 252, f, 22, AMBER if special else TEAL, "middle", 700)
    s.line(xs[2], 209, xs[2], 219, AMBER, arrow=True)
    s.lines(49, 299, ["r = l+h−1 = 4：两边同时离开零 run，左侧的 01 继续匹配，所以 F(4)=5。" if zh else
                       "At r = l+h−1 = 4, both zero-runs end together; the earlier 01 also matches, giving F(4)=5.",
                       "其他终点：某一侧先遇到 run 边界，匹配长度已被唯一确定。" if zh else
                       "At every other endpoint, one side leaves its run first, fixing the exact suffix length."], 13)
    s.lines(28, 379, ["表示：连续改写 A 的记录压成 (min(c,s−o), s−1)，A/H 交点解两个线性不等式。" if zh else
                       "Representation: consecutive A records become (min(c,s−o), s−1); solve two linear inequalities at A/H crossings.",
                       "计账：按真实记录付费；数值收缩也按 owner 区间更新，不再展开所有 s。" if zh else
                       "Charge actual records; numerical contraction updates owner intervals without expanding every s."], 13)
    s.save("04-run-formula")


def causal(zh, d):
    s = Svg("静态 endpos 区间，动态因果链" if zh else "Static endpos intervals, a dynamic causal list",
            "K = 01010，t = 4。suffix-link 树的 terminal 排列把同一状态的出现放在连续区间。" if zh else
            "K = 01010, t = 4. A terminal traversal of the suffix-link tree makes each state's occurrences contiguous.", 454, zh)
    s.rect(28, 87, 268, 281)
    s.text(46, 116, "suffix-link tree", 16, INK, weight=700)
    nodes = [(158, 150, "root"), (103, 206, "0"), (227, 206, "01"),
             (103, 262, "010"), (227, 262, "0101"), (103, 318, "01010")]
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5)]:
        s.line(nodes[a][0], nodes[a][1] + 8, nodes[b][0], nodes[b][1] - 12, MUTED)
    for i, (x, y, label) in enumerate(nodes):
        s.rect(x - 35, y - 15, 70, 29, "#e3f5ee" if i in (1, 3, 5) else "white", "#b5ddce" if i in (1, 3, 5) else LINE, 7)
        s.text(x, y + 5, label, 12, TEAL if i in (1, 3, 5) else INK, "middle")
    s.rect(318, 87, 554, 281)
    s.text(338, 116, "terminal order: endpoint values", 14, MUTED)
    xs = [379 + 99 * i for i in range(5)]
    ends = d["causality"]["terminal_order"]
    s.rect(340, 134, 292, 59, "#f0fdf9", "#8dd4bd", 7)
    for x, r in zip(xs, ends):
        s.rect(x - 27, 143, 54, 41, "white", LINE, 6)
        s.text(x, 171, r, 21, INK, "middle", 700)
    s.text(338, 217, "endpos(0) = {0, 2, 4}", 14, TEAL, weight=700)
    s.text(338, 243, "激活链：只含 r < 4" if zh else "Active list: only r < 4", 14, MUTED)
    active = [x for x, r in zip(xs, ends) if r < 4]
    for x1, x2 in zip(active, active[1:]):
        if x2 - x1 > 120:
            s.parts.append(f'<path d="M{x1+25},279 C{x1+45},324 {x2-49},324 {x2-29},279" fill="none" stroke="{BLUE}" stroke-width="2" marker-end="url(#arrow)"/>')
        else:
            s.line(x1 + 25, 279, x2 - 29, 279, BLUE, 2, arrow=True)
    for x, r in zip(xs, ends):
        if r == 4:
            s.rect(x - 27, 258, 54, 41, "#f1f5f9", LINE, 6)
            s.text(x, 286, "4", 19, "#94a3b8", "middle")
        else:
            s.rect(x - 27, 258, 54, 41, "#e3efff", "#bfdbfe", 6)
            s.text(x, 286, r, 20, BLUE, "middle", 700)
    s.text(338, 337, "从最早出现 0 出发，报告 0、2；下个活跃项 1 已出区间。" if zh else
           "From anchor 0, report 0 and 2; active endpoint 1 lies outside the interval.", 11, MUTED)
    s.lines(28, 399, ["逆序删除时保留节点自己的邻居指针 → 正序激活时 O(1) 接回。" if zh else
                       "Preserve each node's own neighbors during reverse deletion → reinsert in O(1) during forward activation.",
                       "每组报告 O(1+C)：C 是实际因果出现项数，不再付费扫描未来项。" if zh else
                       "Group reporting costs O(1+C), where C counts actual causal occurrences; future entries are never scanned."], 13)
    s.save("05-causal-occurrences")


def continuation(zh, d):
    s = Svg("唯一出现把未来固定成仿射轨迹" if zh else "A unique occurrence fixes an affine future",
            "Q = 00000，K = 10000。短编辑模式 1 在完整 K 中只出现一次，终点为 0。" if zh else
            "Q = 00000, K = 10000. The short edited pattern 1 occurs exactly once in full K, at endpoint 0.", 461, zh)
    s.rect(28, 87, 573, 277)
    xs = [232, 333, 434, 535]
    s.text(49, 122, "时间 t" if zh else "Time t", 14, MUTED)
    for x, t in zip(xs, range(1, 5)):
        s.text(x, 122, t, 16, INK, "middle", 700)
    for row, info in enumerate(d["continuations"]["intervals"]):
        yy = 155 + row * 88
        s.text(48, yy + 28, f'flip Q[{info["q_owner"]}]', 15, INK, weight=700)
        start = info["start"]
        s.rect(xs[start - 1] - 39, yy, xs[-1] - xs[start - 1] + 78, 47,
               "#e3f5ee" if row == 0 else "#eef2ff", "#b5ddce" if row == 0 else "#c7d2fe", 8)
        for t, route in zip(range(start, 5), info["routes"]):
            s.text(xs[t - 1], yy + 30, f"({route[0]},{route[1]})", 17, TEAL if row == 0 else PURPLE, "middle", 700)
        s.text(48, yy + 61, f'alpha={info["alpha"]}, beta={info["beta"]}', 12, MUTED)
    s.rect(624, 87, 248, 277)
    s.text(643, 118, "固定的 K owner" if zh else "One fixed K owner", 17, BLUE, weight=700)
    s.lines(643, 153, ["s = u + beta = 0", "key = (s, bit) = (0, 0)"], 15, INK, 30)
    s.line(643, 208, 851, 208)
    s.text(643, 236, "堆优先级" if zh else "Heap priority", 15, MUTED)
    s.text(643, 266, "(alpha, beta)", 19, BLUE, weight=700)
    s.lines(643, 303, ["每行只取有效堆顶" if zh else "One live winner per row",
                       "与普通候选、删除比较" if zh else "Merge with repair and deletion"], 12)
    s.lines(28, 397, ["轨迹：(L(t), r(t)) = (t+alpha, t+beta)。失配后不会复活，寿命由一次向前 LCE 决定。" if zh else
                       "Trajectory: (L(t), r(t)) = (t+alpha, t+beta). One forward LCE fixes the lifetime; a failed branch cannot revive.",
                       "这两个区间仍含 4+3 个数值项。控制事件变少，不代表全部相关求和已经消失。" if zh else
                       "These two intervals still contain 4+3 numerical terms. Fewer control events do not remove all correlation sums."], 13)
    s.save("06-continuation-intervals")


def aggregation(zh, d):
    s = Svg("小残差必须活到最终相消之后" if zh else "The small residual must survive the final cancellation",
            "每个事件本身可精确表示为 binary64；错误来自中间累计或过早投影。" if zh else
            "Each event is representable in binary64; the failure comes from intermediate summation or early projection.", 446, zh)
    s.rect(28, 87, 844, 208)
    xs = [307, 461, 615, 769]
    labels = ["+2e16", "+2", "+2", "−2e16"]
    s.text(46, 123, "事件" if zh else "Event", 14, MUTED)
    s.text(46, 182, "普通 double 前缀" if zh else "Ordinary double prefix", 13, MUTED)
    s.text(46, 245, "精确事件和" if zh else "Exact event sum", 14, TEAL, weight=700)
    for x, label in zip(xs, labels):
        s.text(x, 123, label, 19, INK, "middle", 700)
    for x, label in zip(xs, ["2e16", "2e16", "2e16", "0"]):
        s.rect(x - 62, 150, 124, 43, "#fff1f0", "#efc0bc", 6)
        s.text(x, 179, label, 19, RED, "middle")
    for x, label in zip(xs, ["2e16", "2e16 + 2", "2e16 + 4", "4"]):
        s.rect(x - 68, 213, 136, 43, "#e3f5ee", "#b5ddce", 6)
        s.text(x, 242, label, 17, TEAL, "middle", 700)
    s.rect(28, 318, 844, 89, "#e3efff", "#bfdbfe")
    s.text(46, 347, "同样需要延迟投影的最终合并" if zh else "The final merge also needs deferred rounding", 16, BLUE, weight=700)
    s.text(46, 378, "20000000000000002  −  19999999999999998  =  4", 19, INK, weight=700)
    s.text(28, 431, "Expansion 保存已经生成的浮点事件；它无法恢复行内点积此前已舍去的信息。" if zh else
           "Expansion preserves the generated floating-point events; it cannot recover information already rounded out of a dot product.", 12, MUTED)
    s.save("07-exact-aggregation")


def main():
    checked = data()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figure_data.json").write_text(json.dumps(checked, indent=2) + "\n", encoding="utf-8")
    for zh in (False, True):
        for figure in (repeated, partition, envelopes, runs, causal, continuation, aggregation):
            figure(zh, checked)
    print("Verified 3 string examples, deletion tables, affine routes, and cancellation; wrote 14 SVGs.")


if __name__ == "__main__":
    main()
