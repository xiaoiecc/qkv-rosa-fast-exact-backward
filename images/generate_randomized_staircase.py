"""Regenerate the exact staircase example used by both randomized tutorials."""
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parent
Q = [4] * 7 + [0, 1, 2]
K = [0, 2, 3, 0, 1, 2, 3, 1, 2, 4]
T = 9


def suffix(q, k, t, r):
    length = 0
    while r - length >= 0 and q[t - length] == k[r - length]:
        length += 1
    return length


def route(q, k):
    candidates = [(suffix(q, k, T, r), r) for r in range(T)]
    return max(((length, r) if length else (0, -1)) for length, r in candidates)


matches = [(r, suffix(Q, K, T, r)) for r in range(T) if suffix(Q, K, T, r)]
assert matches == [(1, 1), (5, 3), (8, 2)]
assert route(Q, K) == (3, 5)
deleted_k = K.copy()
deleted_k[4] = 8
assert route(Q, deleted_k) == (2, 8)
deleted_q = Q.copy()
deleted_q[7] = 8
assert route(deleted_q, K) == (2, 8)

LABELS = {
    "en": {
        "title": "A deletion query from three sparse endpoints",
        "left": "Equal-word endpoints at t = 9",
        "right": "Latest endpoint for each required length",
        "f": "Fₜ(r): suffix length",
        "r": "R(h): latest endpoint",
        "required": "r = h + 4",
        "foot1": "Delete K owner s = 4: keep a suffix only when R(h) − h ≥ 4.",
        "foot2": "A(4) = (1, 1); the largest feasible h is 2, so the answer is (2, 8).",
        "desc": "Nonzero suffix matches are (endpoint,length)=(1,1),(5,3),(8,2). "
                "The latest-endpoint staircase is R(1)=R(2)=8 and R(3)=5. "
                "For deletion owner 4, lengths 1 and 2 are feasible and length 3 is not. "
                "The deletion route is (length,endpoint)=(2,8).",
    },
    "zh-CN": {
        "title": "三个稀疏端点，就能回答一次删除查询",
        "left": "t = 9：相同字类的端点",
        "right": "要求长度为 h 时，最晚能用哪个端点",
        "f": "Fₜ(r)：相等后缀长",
        "r": "R(h)：最晚端点",
        "required": "r = h + 4",
        "foot1": "删除 K owner s = 4：后缀必须满足 R(h) − h ≥ 4。",
        "foot2": "A(4) = (1, 1)；最大可行长度是 2，答案为 (2, 8)。",
        "desc": "非零后缀匹配为（端点，长度）=(1,1),(5,3),(8,2)。"
                "最晚端点阶梯为 R(1)=R(2)=8，R(3)=5。删除位置 4 后，"
                "长度 1、2 可行，长度 3 不可行；删除路由为（长度，端点）=(2,8)。",
    },
}


def draw(lang):
    label = LABELS[lang]
    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="500" viewBox="0 0 1100 500" role="img">',
        f'<title>{escape(label["title"])}</title><desc>{escape(label["desc"])}</desc>',
        '<defs><style>text{font-family:Inter,Arial,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#243448}'
        '.small{font-size:16px}.tick{font-size:15px;fill:#607085}.head{font-size:19px;font-weight:600}'
        '.math{font-family:"DejaVu Sans",Arial,"Noto Sans CJK SC",sans-serif;font-size:18px}'
        '.grid{stroke:#dce5ee;stroke-width:1;stroke-dasharray:4 5}'
        '.axis{stroke:#8b9bad;stroke-width:1.5}</style></defs>',
        '<rect width="1100" height="500" rx="16" fill="#f7f9fc"/>',
        f'<text x="34" y="42" font-size="25" font-weight="700">{escape(label["title"])}</text>',
        '<rect x="32" y="68" width="480" height="330" rx="12" fill="white" stroke="#dce5ee"/>',
        '<rect x="536" y="68" width="532" height="330" rx="12" fill="white" stroke="#dce5ee"/>',
        f'<text class="head" x="56" y="100">{escape(label["left"])}</text>',
        f'<text class="head" x="560" y="100">{escape(label["right"])}</text>',
        f'<text class="small" x="56" y="131">{escape(label["f"])}</text>',
        '<path class="axis" fill="none" d="M95 150V345H470"/>',
    ]
    for height in range(1, 4):
        y = 345 - 60 * height
        pieces += [f'<line class="grid" x1="95" y1="{y}" x2="470" y2="{y}"/>',
                   f'<text class="tick" x="75" y="{y+5}">{height}</text>']
    for endpoint, length in matches:
        x, y = 115 + 40 * endpoint, 345 - 60 * length
        color = "#7255b5" if length == 3 else "#337cc1"
        pieces += [
            f'<rect x="{x-18}" y="{y}" width="36" height="{345-y}" rx="5" fill="{color}"/>',
            f'<text class="math" text-anchor="middle" x="{x}" y="{y-9}">{length}</text>',
            f'<text class="tick" text-anchor="middle" x="{x}" y="368">{endpoint}</text>',
        ]
    pieces += [
        '<text class="math" x="477" y="369">r</text>',
        '<text class="tick" x="75" y="350">0</text>',
        f'<text class="small" x="560" y="131">{escape(label["r"])}</text>',
        '<path class="axis" fill="none" d="M605 155V345H1000"/>',
    ]
    for endpoint in [0, 4, 5, 6, 7, 8]:
        y = 345 - 20 * endpoint
        pieces += [
            f'<line class="grid" x1="605" y1="{y}" x2="1000" y2="{y}"/>',
            f'<text class="tick" x="580" y="{y+5}">{endpoint}</text>',
        ]
    pieces += [
        '<path d="M620 185H860V245H980" fill="none" stroke="#337cc1" stroke-width="4"/>',
        '<path d="M650 250L950 200" fill="none" stroke="#bc7b23" stroke-width="2.5" stroke-dasharray="7 5"/>',
        f'<text class="small" x="917" y="183" fill="#986119">{escape(label["required"])}</text>',
        '<text class="math" x="730" y="169" text-anchor="middle">R(1) = R(2) = 8</text>',
        '<text class="math" x="891" y="277">R(3) = 5</text>',
    ]
    for h, endpoint in [(1, 8), (2, 8), (3, 5)]:
        x, y = 560 + 120 * h, 345 - 20 * endpoint
        pieces += [f'<text class="tick" text-anchor="middle" x="{x}" y="368">{h}</text>']
        if endpoint - h >= 4:
            pieces += [f'<circle cx="{x}" cy="{y}" r="7" fill="#18866a" stroke="white" stroke-width="2"/>']
        else:
            pieces += [f'<path d="M{x-6} {y-6}L{x+6} {y+6}M{x-6} {y+6}L{x+6} {y-6}" stroke="#c74a49" stroke-width="3"/>']
    pieces += [
        '<text class="math" x="1007" y="369">h</text>',
        f'<text class="math" x="36" y="437">{escape(label["foot1"])}</text>',
        f'<text class="math" x="36" y="471">{escape(label["foot2"])}</text>',
        '</svg>',
    ]
    name = "randomized-row-staircase.svg" if lang == "en" else "randomized-row-staircase.zh-CN.svg"
    (ROOT / name).write_text("\n".join(pieces) + "\n")
    print(name)


if __name__ == "__main__":
    for language in LABELS:
        draw(language)
