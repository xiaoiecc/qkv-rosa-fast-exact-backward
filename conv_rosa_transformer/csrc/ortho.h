#pragma once
// Layer 2: per-symbol orthogonal range oracle used by direct_q_best_center.
// Step-identical port of _MinPRange / _MergeRangePredecessor / _YBucket / _SymbolOrthogonalOracle.
#include <vector>
#include <optional>
#include <tuple>
#include <cstdint>

namespace rosa {

struct MinPRange {
    int n = 0, size = 1;
    std::vector<int64_t> tr;                     // inf = 1<<60, matches Python
    explicit MinPRange(const std::vector<int>& values);
    std::optional<int> find_last(int right_exclusive, int64_t threshold) const;
    std::optional<int> find_first(int left, int64_t threshold) const;
};

struct MergeRangePredecessor {
    int size = 1;
    std::vector<std::vector<int>> tree;          // sorted values per segment node
    explicit MergeRangePredecessor(const std::vector<std::optional<int>>& values_by_index);
    std::optional<int> predecessor(int left, int right, int x) const;  // max value <= x in [left,right)
};

struct YBucket {
    std::vector<int> ys, ps;                     // points sorted by y
    MinPRange minp;
    MergeRangePredecessor predp;
    explicit YBucket(const std::vector<std::pair<int,int>>& points);   // (y,p)
    std::optional<std::pair<int,int>> pred_y(int yq, int owner_u) const;
    std::optional<std::pair<int,int>> succ_y(int yq, int owner_u) const;
    std::optional<int> max_p(int ylo, int yhi, int owner_u) const;
};

struct SymbolOrthogonalOracle {
    std::vector<int> xs;                         // sorted outer (forward-SA rank) coords
    int n = 0, size = 1;
    std::vector<std::optional<YBucket>> buckets; // segment tree over x-order
    explicit SymbolOrthogonalOracle(const std::vector<std::tuple<int,int,int>>& points); // (x,y,p)
    std::vector<int> nodes(int xlo, int xhi) const;
    std::pair<std::optional<std::pair<int,int>>, std::optional<std::pair<int,int>>>
        nearest(int xlo, int xhi, int yq, int owner_u) const;
    std::optional<int> max_p(int xlo, int xhi, int ylo, int yhi, int owner_u) const;
};

}  // namespace rosa
