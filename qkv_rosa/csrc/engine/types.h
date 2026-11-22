#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rosa_subq {
struct Route {
    int length=0, end=-1;
    bool operator<(const Route& b) const { return length<b.length || (length==b.length && end<b.end); }
    bool operator>(const Route& b) const { return b<*this; }
    bool operator==(const Route& b) const { return length==b.length && end==b.end; }
    bool operator!=(const Route& b) const { return !(*this==b); }
};
struct Segment { int owner,bit,lo,hi,alpha,beta; };
struct OwnerRange { int owner,lo,hi; };
struct BitRange { int owner,bit,lo,hi; };
struct Scores {
    int n=0,d=0;
    const double* weights=nullptr;
    const double* payload=nullptr;
    const double* empty=nullptr;
    const double* row(int t) const { return weights+std::size_t(t)*d; }
    const double* value(int end) const { return end<0?empty:payload+std::size_t(end+1)*d; }
    double score(int t,int end) const {
        const double* a=row(t);const double* b=value(end);double x=0;
        for(int c=0;c<d;++c)x+=a[c]*b[c];
        return x;
    }
};
struct Stats {
    std::uint64_t suffix_queries=0,delete_pieces=0,delete_events=0;
    std::uint64_t q_heavy_rows=0,q_light_sources=0,q_segments=0;
    std::uint64_t k_anchor_probes=0,k_centers=0,k_segments=0;
    std::uint64_t short_words=0,short_pieces=0,short_queries=0,short_exceptions=0;
    std::uint64_t correlation_queries=0,correlation_direct=0,fft_transforms=0;
    std::uint64_t range_blocks=0,range_fringes=0,peak_workspace_bytes=0;
    std::uint64_t hash_probes=0,hash_verifications=0,sampled_channels=0;
};
} // namespace rosa_subq
