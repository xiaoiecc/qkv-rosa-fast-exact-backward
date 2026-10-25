"""Test-only Python oracle for the production native causal proposal algorithm.

Only the proposal distribution is approximate. Selected pairs keep the existing
joint_bits semantics and are evaluated by its exact conditional-credit path.
This module never reads a payload, target, gradient, or loss.
"""
from __future__ import annotations

import heapq
import operator
import time

import numpy as np

# joint_credit may import this module lazily inside its public proposal function.
# Import only its definitions; never call that public function from here.
from conv_rosa_transformer.joint_credit import Bit, STREAMS, _mix64, _words


def _band_prefixes(words, first, query_words, query_rows, mask):
    """Return word IDs and each query's exact causal interval in that order."""
    keys = words & np.uint64(mask)
    order = np.lexsort((first, keys))
    ordered_keys = keys[order]
    query_keys = query_words & np.uint64(mask)
    starts = np.searchsorted(ordered_keys, query_keys, side="left")

    # Search (full band key, first occurrence) lexicographically in NumPy.
    # A query (key,t) ends precisely before first==t. Vectorizing this second
    # search avoids a Python loop over all tokens when rows=None.
    dtype = np.dtype([("band", np.uint64), ("first", np.intp)])
    entries = np.empty(len(words), dtype=dtype)
    entries["band"], entries["first"] = ordered_keys, first[order]
    queries = np.empty(len(query_rows), dtype=dtype)
    queries["band"], queries["first"] = query_keys, query_rows
    stops = np.searchsorted(entries, queries, side="left")
    return order, starts, stops


def propose_pairs(q, k, D, *, max_pairs=16, seed=0, rows=None,
                  allowed_streams=("Q", "K"), allowed_owners=None,
                  bucket_cap=8, occurrence_cap=2, scan=8, candidate_cap=None,
                  candidate_trace=False):
    """Propose the same kind of disjoint repairs using static causal buckets.

    Three bands partition a random permutation of the D bits. Distinct full K
    words are sorted by (band key, first occurrence). Each queried row samples
    at most bucket_cap distinct words uniformly without replacement from each
    matching bucket's causal prefix. A particular eligible word's inclusion
    probability in one bucket is min(1, bucket_cap / prefix_size). Overlapping
    bands and subsequent filtering do not make the final pairs uniform.

    Occurrences are grouped by full word and kept in increasing time order.
    Sample at most occurrence_cap causal endpoints, always adding the latest.
    Full-word XOR, the backward scan, four Q/K repair assignments, fixed seeded
    priority heap, and greedy disjoint selection retain joint_bits' definitions.

    This differs from the online reservoir's correlated samples across rows.
    At full bucket and occurrence capacity their candidate pools agree; finite
    capacities need not give equal pools for the same seed. Randomness can omit
    a useful candidate, but cannot make a selected pair's later credit inexact.

    Sorting/preprocessing costs O(T log T), with O(T) index storage. For R rows,
    at most 3R bucket samples inspect 3R*bucket_cap words, at most
    A=3R*bucket_cap*(occurrence_cap+1) alignments, and A*scan symbols. Occurrence
    prefix searches cost O(log T) per inspected word; at most 4A candidate
    events use a heap of candidate_cap entries. Total storage includes that
    separately budgeted heap. Validation is O(T); no online insertion loop runs.
    """
    started = time.perf_counter()
    d = operator.index(D)
    if not 1 <= d <= 64:
        raise ValueError("D must lie in [1,64]")
    q, k = _words(q, "Q", d), _words(k, "K", d)
    n = len(q)
    if len(k) != n:
        raise ValueError("Q and K must have equal length")
    max_pairs, bucket_cap, occurrence_cap, scan = map(operator.index,
        (max_pairs, bucket_cap, occurrence_cap, scan))
    if max_pairs < 0 or min(bucket_cap, occurrence_cap, scan) <= 0:
        raise ValueError("pair budget must be nonnegative; capacities/scan positive")
    candidate_cap = max(1, 8 * max_pairs) if candidate_cap is None else operator.index(candidate_cap)
    if candidate_cap < max_pairs:
        raise ValueError("candidate_cap must be at least max_pairs")
    streams = frozenset(allowed_streams)
    if not streams <= {"Q", "K"}:
        raise ValueError("structural proposals support Q and K owners")
    if rows is None:
        selected_rows = np.arange(n, dtype=np.intp)
    else:
        requested = sorted({operator.index(t) for t in rows})
        if any(t < 0 or t >= n for t in requested):
            raise ValueError("row outside the stream")
        selected_rows = np.asarray(requested, dtype=np.intp)
    owners = None if allowed_owners is None else {
        s: frozenset(operator.index(t) for t in ts) for s, ts in allowed_owners.items()}
    if owners is not None and (not set(owners) <= set(STREAMS) or
            any(t < 0 or t >= n for ts in owners.values() for t in ts)):
        raise ValueError("invalid allowed_owners")

    stats = dict(seed=int(seed), bucket_probes=0, examined_alignments=0,
                 symbol_comparisons=0, candidate_events=0, retained_candidates=0,
                 selected_pairs=0, causal_symbols=0,
                 proposal_algorithm="static_causal_bands",
                 query_rows=len(selected_rows), distinct_words=0,
                 sampled_word_events=0, word_rng_calls=0, occurrence_rng_calls=0)
    if max_pairs == 0 or not n or not streams:
        stats["proposal_seconds"] = time.perf_counter() - started
        return [], stats

    rng = np.random.default_rng(seed)
    masks = [0, 0, 0]
    for i, b in enumerate(rng.permutation(d)):
        masks[i % 3] |= 1 << int(b)
    packed_k = np.asarray(k, dtype=np.uint64)
    words, first, inverse, counts = np.unique(packed_k, return_index=True,
        return_inverse=True, return_counts=True)
    word_values = words.tolist()
    occurrences = np.argsort(inverse, kind="stable")
    offsets = np.empty(len(words) + 1, dtype=np.intp)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    query_words = np.asarray(q, dtype=np.uint64)[selected_rows]
    bands = [_band_prefixes(words, first, query_words, selected_rows, mask)
             for mask in masks]
    stats["distinct_words"] = len(words)
    # Preserve the original counter's meaning: endpoints visible by the final
    # stream row, independently of the requested query-row subset.
    stats["causal_symbols"] = max(0, n - 1)
    stats["preprocessing_seconds"] = time.perf_counter() - started
    candidates, heap = {}, []
    nd = n * d
    rank_base = 3 * nd
    seed64 = int(seed) & ((1 << 64) - 1)
    q_enabled, k_enabled = "Q" in streams, "K" in streams
    q_owners = None if owners is None else owners.get("Q", ())
    k_owners = None if owners is None else owners.get("K", ())
    bit_hashes, first_hashes = {}, {}

    def hashed_bit(code):
        hashed = bit_hashes.get(code)
        if hashed is None:
            # Bit sorts lexicographically: K before Q. The original priority
            # instead numbers streams Q=0, K=1; preserve both conventions.
            flat = code + nd if code < nd else code - nd
            hashed = _mix64(flat)
            bit_hashes[code] = hashed
        return hashed

    def offer(a, b):
        pair = (a, b) if a < b else (b, a)
        stats["candidate_events"] += 1
        if pair in candidates:
            return
        a, b = pair
        first_hash = first_hashes.get(a)
        if first_hash is None:
            first_hash = _mix64(seed64 ^ hashed_bit(a))
            first_hashes[a] = first_hash
        priority = _mix64(first_hash ^ hashed_bit(b))
        flat_a = a + nd if a < nd else a - nd
        flat_b = b + nd if b < nd else b - nd
        rank = flat_a * rank_base + flat_b
        if len(heap) == candidate_cap:
            worst_priority, worst_rank = -heap[0][0], -heap[0][1]
            if (priority, rank) >= (worst_priority, worst_rank):
                return
            _, _, removed = heapq.heappop(heap)
            del candidates[removed]
        candidates[pair] = (priority, rank)
        heapq.heappush(heap, (-priority, -rank, pair))

    for row_index, t in enumerate(selected_rows.tolist()):
        # Reuse hash work across the four assignments and repeated alignments
        # in this row. Clearing keeps scratch bounded by the per-row budget,
        # rather than accumulating a potentially large table of all T*D bits.
        bit_hashes.clear()
        first_hashes.clear()
        inspected = set()
        for order, starts, stops in bands:
            stats["bucket_probes"] += 1
            begin, end = int(starts[row_index]), int(stops[row_index])
            population = end - begin
            if population <= bucket_cap:
                selected = order[begin:end]
            else:
                selected = order[begin + rng.choice(population, bucket_cap,
                                                   replace=False, shuffle=False)]
                stats["word_rng_calls"] += 1
            stats["sampled_word_events"] += len(selected)
            inspected.update(selected.tolist())

        # np.unique assigns word IDs in full unsigned-word order, matching the
        # original scan's sorted(words) order whenever the candidate sets agree.
        for word_id in sorted(inspected):
            word = word_values[word_id]
            if (q[t] ^ word).bit_count() > 2:
                continue
            positions = occurrences[offsets[word_id]:offsets[word_id + 1]]
            causal_count = int(np.searchsorted(positions, t, side="left"))
            if causal_count <= occurrence_cap:
                chosen = positions[:causal_count]
            else:
                chosen = positions[rng.choice(causal_count, occurrence_cap,
                                              replace=False, shuffle=False)]
                stats["occurrence_rng_calls"] += 1
            # first<t was enforced by the band prefix, hence causal_count>0.
            endpoints = set(chosen.tolist())
            endpoints.add(int(positions[causal_count - 1]))
            for r in sorted(endpoints, reverse=True):
                stats["examined_alignments"] += 1
                mismatches = []
                for offset in range(min(scan, r + 1)):
                    u, s = t - offset, r - offset
                    xor = q[u] ^ k[s]
                    stats["symbol_comparisons"] += 1
                    if len(mismatches) + xor.bit_count() > 2:
                        break
                    q_allowed = q_enabled and (q_owners is None or u in q_owners)
                    k_allowed = k_enabled and (k_owners is None or s in k_owners)
                    while xor:
                        low = xor & -xor
                        j = low.bit_length() - 1
                        options = []
                        if q_allowed:
                            options.append(nd + u * d + j)
                        if k_allowed:
                            options.append(s * d + j)
                        mismatches.append(options)
                        xor ^= low
                    if len(mismatches) == 2:
                        for a in mismatches[0]:
                            for b in mismatches[1]:
                                offer(a, b)
                        break

    pairs, used = [], set()
    for pair in sorted(candidates, key=lambda p: (candidates[p], p)):
        if not any(b in used for b in pair):
            pairs.append(pair)
            used.update(pair)
            if len(pairs) == max_pairs:
                break
    stats["retained_candidates"] = len(candidates)
    stats["selected_pairs"] = len(pairs)

    def fields(code):
        stream = "K" if code < nd else "Q"
        if code >= nd:
            code -= nd
        return stream, code // d, code % d

    if candidate_trace:
        stats["candidate_pairs"] = [
            [fields(code) for code in pair]
            for pair in sorted(candidates)]
    pairs = [tuple(Bit(*fields(code)) for code in pair) for pair in pairs]
    stats["proposal_seconds"] = time.perf_counter() - started
    return pairs, stats
