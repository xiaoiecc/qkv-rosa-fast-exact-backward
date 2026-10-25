"""Tiny independent oracle: direct hard forwards after each individual flip.

This test-only code shares no deletion, repair, SAM, or score-cache algebra
with the production core. Its deliberately expensive loops keep the hard
definition inspectable. Integer inputs have exact integer expected credits.
"""
from math import fsum


def hard_routes(q, k):
    if len(q) != len(k):
        raise ValueError("Q and K lengths differ")
    routes = []
    for t in range(len(q)):
        winner = (0, -1)
        for r in range(t):
            length = 0
            while length <= r and q[t - length] == k[r - length]:
                length += 1
            if length:
                winner = max(winner, (length, r))
        routes.append(winner)
    return routes


def hard_outputs(q, k, v, emb0, emb1):
    routes = hard_routes(q, k)
    values = [list(emb0) if r < 0 else
              [(2 * int(v[r + 1][j]) - 1) * emb1[j]
               for j in range(len(emb1))] for _, r in routes]
    return routes, values


def brute_credits(q, k, v, gradient, emb0, emb1, integer=False):
    q, k = list(map(int, q)), list(map(int, k))
    v = [list(map(int, row)) for row in v]
    n, d = len(q), len(emb1)
    baseline, base = hard_outputs(q, k, v, emb0, emb1)
    reduction = sum if integer else fsum

    def credit(qq, kk, vv):
        _, changed = hard_outputs(qq, kk, vv, emb0, emb1)
        return reduction((changed[t][j] - base[t][j]) * gradient[t][j]
                         for t in range(n) for j in range(d))

    cq, ck, cv = ([[0] * d for _ in range(n)] for _ in range(3))
    for owner in range(n):
        for bit in range(d):
            qq = q.copy()
            qq[owner] ^= 1 << bit
            cq[owner][bit] = credit(qq, k, v)
            kk = k.copy()
            kk[owner] ^= 1 << bit
            ck[owner][bit] = credit(q, kk, v)
            vv = [row.copy() for row in v]
            vv[owner][bit] ^= 1
            cv[owner][bit] = credit(q, k, vv)
    return cq, ck, cv, [x[0] for x in baseline], [x[1] for x in baseline]
