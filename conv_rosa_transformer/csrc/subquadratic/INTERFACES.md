# Internal interface for the native subquadratic engine

All APIs are C++17 in `namespace rosa_subq`. All time intervals are inclusive.
Suffix-rank intervals are half-open. All numeric arrays are row-major doubles.
`Scores::score(t,r)` reads the empty vector for r=-1 and payload[r+1] otherwise.
Public caller validates shape/finite inputs; internal indexes assume valid bounds.
Empty input is handled before constructing an index.

`index.h` supplies `SuffixArray(const std::vector<int>&)`, public `sa,rank`,
`int lcp_rank(int a,int b) const`, and `std::pair<int,int>
prefix_interval(int rank,int length) const` (half-open).

`Index(const uint64_t* q,const uint64_t* k,int n)` stores public `n`, `q`, `k`
(vectors of uint64_t), and provides const methods:

```
int lce(int qstart,int kstart) const;
int lcs(int qend,int kend) const;
int same_lcs(bool query_string,int aend,int bend) const;
int R(int t,int length,int cutoff=-1) const; // greatest K endpoint < cutoff; -1 cutoff means t
Route best(int t,int cutoff=-1) const;
Route deleted(int t,int owner) const; // full K deletion; returns baseline outside affected/noncausal owners
```

The deletion module may add further public indexed operations needed internally.

`delete.h` supplies:
```
std::vector<double> q_deletion(const Index&,const Scores&,int A,Stats&,const std::vector<int>* rows=nullptr);
std::vector<double> k_deletion(const Index&,const Scores&,int A,Stats&,const std::vector<int>* rows=nullptr);
std::vector<double> q_delete_ranges(const Index&,const Scores&,const std::vector<OwnerRange>&,int H,Stats&);
std::vector<double> k_delete_ranges(const Index&,const Scores&,const std::vector<OwnerRange>&,int H,Stats&);
```
The first two return deletion-minus-baseline per owner; ranges return FULL
deleted-payload projections per request, not deltas. Original row indexes stay
unchanged in restricted-row calls. A/H > 0; caller supplies balanced parameters.

`short_k.h` supplies:
```
void short_k(const Index&,const Scores&,int bits,int cutoff,std::vector<double>& credits,Stats&);
```
Adds short repair-minus-deletion to a preallocated n*bits vector. Builds no
oracle table; uses O(n) descriptor/job batches and indexed causal exceptions.

`repair.h` supplies:
```
void q_repair(const Index&,const Scores&,int bits,int threshold,std::vector<double>& heavy_positive,std::vector<Segment>& light,std::vector<BitRange>& exists,Stats&);
std::vector<Segment> long_k(const Index&,int anchor,Stats&);
```
`heavy_positive` is preallocated n*bits and receives positive scores only.
Existence intervals cover both heavy and light times for deletion subtraction.
long_k uses short cutoff=4*anchor and returns nonoverlapping winning intervals.

The derivation is in `docs/subquadratic-engine.md` and its linked proof.
Independent literal oracles are in `tests/cpp/subquadratic_*.cpp`; the complete
training oracle is `tests/test_subquadratic_engine.py`. The native engine has
no Python dependency. Use checked vector sizes, owned buffers and bounded indexing;
no undefined arithmetic, invalid aliasing or nonfinite fast-math assumptions.
