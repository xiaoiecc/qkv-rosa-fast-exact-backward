"""Budgeted causal-band proposals, using one native implementation.

Full uint64 words certify every candidate. Randomness can omit candidates but
cannot certify a false equality. The retained Python implementation lives only
in the test oracle; there is no runtime backend dispatch.
"""
from __future__ import annotations

import operator
import time

import numpy as np

if __package__:
    from . import rosa_core
    from .joint_credit import Bit, STREAMS
else:
    import rosa_core
    from joint_credit import Bit, STREAMS


def propose_pairs(q, k, D, *, max_pairs=16, seed=0, rows=None,
                  allowed_streams=("Q", "K"), allowed_owners=None,
                  bucket_cap=8, occurrence_cap=2, scan=8, candidate_cap=None,
                  candidate_trace=False):
    """Propose disjoint two-bit repairs from sampled static causal bands.

    Three randomly partitioned bands guarantee that words within Hamming
    distance two share a band. Only distinct words whose first occurrence is
    earlier than the query enter each causal bucket. Uniform word and endpoint
    samples have explicit capacities, and each word's latest causal endpoint
    is always retained. Full XOR verification and bounded backward scans create
    repair pairs, followed by seeded priorities and greedy disjoint selection.

    Native mt19937_64 replaces the Python oracle's NumPy generator. Finite-
    budget samples therefore need not match old seeds. At full bucket and
    occurrence capacity the candidate pool and priority-selected pairs agree.
    Every selected pair is later evaluated by exact conditional replay.

    Preprocessing is O(T log T), with O(T) space. R rows examine at most
    3*R*bucket_cap*(occurrence_cap+1) alignments, each for at most scan symbols.
    A separate candidate_cap bounds the retained candidate heap. Neither a
    payload, a target, nor an upstream gradient enters this sampler.
    """
    started = time.perf_counter()
    d = operator.index(D)
    if not 1 <= d <= 64:
        raise ValueError("D must lie in [1,64]")
    q, k = rosa_core._words(q, "Q"), rosa_core._words(k, "K")
    if q.ndim != 1 or k.shape != q.shape:
        raise ValueError("Q and K must have equal shape [T]")
    n = len(q)
    if n >= rosa_core._MAX_T or n*d > rosa_core._MAX_ELEMENTS:
        raise ValueError("Requires T < 2**28 and T*D <= 2**31-1")
    if d < 64 and (np.any(q >> np.uint64(d)) or np.any(k >> np.uint64(d))):
        raise ValueError("Q/K words must fit the declared D-bit alphabet")
    max_pairs, bucket_cap, occurrence_cap, scan = map(operator.index,
        (max_pairs, bucket_cap, occurrence_cap, scan))
    if max_pairs < 0 or min(bucket_cap, occurrence_cap, scan) <= 0:
        raise ValueError("pair budget must be nonnegative; capacities/scan positive")
    candidate_cap = max(1, 8*max_pairs) if candidate_cap is None else operator.index(candidate_cap)
    if candidate_cap < max_pairs:
        raise ValueError("candidate_cap must be at least max_pairs")
    if max(max_pairs, bucket_cap, occurrence_cap, scan, candidate_cap) > rosa_core._MAX_ELEMENTS:
        raise ValueError("Candidate capacities must fit int32")
    seed = operator.index(seed)
    if not 0 <= seed <= rosa_core._MAX_WORD:
        raise ValueError("seed must fit uint64")
    streams = frozenset(allowed_streams)
    if not streams <= {"Q", "K"}:
        raise ValueError("structural proposals support Q and K owners")
    if rows is None:
        selected_rows = np.arange(n, dtype=np.int32)
    else:
        requested = sorted({operator.index(t) for t in rows})
        if any(t < 0 or t >= n for t in requested):
            raise ValueError("row outside the stream")
        selected_rows = np.asarray(requested, dtype=np.int32)
    owners = None if allowed_owners is None else {
        s: frozenset(operator.index(t) for t in ts) for s, ts in allowed_owners.items()}
    if owners is not None and (not set(owners) <= set(STREAMS) or
            any(t < 0 or t >= n for ts in owners.values() for t in ts)):
        raise ValueError("invalid allowed_owners")
    stats = dict(seed=seed, bucket_probes=0, examined_alignments=0,
                 symbol_comparisons=0, candidate_events=0, retained_candidates=0,
                 selected_pairs=0, causal_symbols=0,
                 proposal_algorithm="static_causal_bands_native",
                 query_rows=len(selected_rows), distinct_words=0,
                 sampled_word_events=0, word_rng_calls=0, occurrence_rng_calls=0)
    if max_pairs == 0 or not n or not streams:
        stats["proposal_seconds"] = time.perf_counter()-started
        return [], stats
    allowed = []
    for stream in ("Q", "K"):
        mask = np.zeros(n, dtype=np.uint8)
        if stream in streams:
            if owners is None:
                mask.fill(1)
            else:
                positions = list(owners.get(stream, ()))
                mask[positions] = 1
        allowed.append(mask)
    packed, native_stats = rosa_core._pair_candidates_prepared(
        q, k, d, selected_rows, allowed[0], allowed[1], max_pairs, seed,
        bucket_cap, occurrence_cap, scan, candidate_cap, candidate_trace)
    stats.update(native_stats)
    nd = n*d
    def fields(code):
        code = int(code)
        stream = "K" if code < nd else "Q"
        if code >= nd:
            code -= nd
        return stream, code//d, code%d
    if candidate_trace:
        stats["candidate_pairs"] = [[fields(code) for code in pair]
                                    for pair in stats.get("candidate_pairs", ())]
    pairs = [tuple(Bit(*fields(code)) for code in pair) for pair in packed]
    stats["seed"] = seed
    stats["proposal_seconds"] = time.perf_counter()-started
    return pairs, stats
