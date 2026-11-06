# The native subquadratic engine

[中文](subquadratic-engine.zh-CN.md) · [Main tutorial](../ALGORITHM.md) · [Full proof](subquadratic-proof.md)

The current algorithm is taught in [ALGORITHM.md](../ALGORITHM.md), from the literal fourth-power all-flip implementation through shared deletion, periodic occurrence profiles, Q heavy/light repair, short K dictionaries, long K anchors, and streamed numerical contraction. It is the repository's only backward engine.

The [reference map](algorithm-reference.md) connects its invariants to the C++ files and independent tests. The [training guide](native-training.md) covers first-order credit, diagonal trust and joint bits, exact randomized caching, and explicit approximation. Current complete-backward timings and error measurements are in [the results](subquadratic-results.md).

For fixed packed width, numerical dimension, and joint-pair budget, the complete construction takes $\widetilde O(T^{7/4})$ time and $\widetilde O(T^{4/3})$ space. Route decisions are exact; binary64 aggregation and FFT have their stated floating-point contract. The bound does not imply a universal finite-size speedup.
