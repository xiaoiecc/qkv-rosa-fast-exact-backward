#pragma once
#include <cstdint>
namespace rosa_random {
using RowRoute=uint64_t;
inline RowRoute row_route(int length,int end) {
    return length>0 ? (uint64_t(uint32_t(length))<<32)|uint32_t(end+1) : 0;
}
inline int row_end(RowRoute value){return int(uint32_t(value))-1;}
inline int row_bit(uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(value);
#else
    int bit=0;while(!(value&1)){value>>=1;++bit;}return bit;
#endif
}

} // namespace rosa_random
