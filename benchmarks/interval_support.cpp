// Standalone count of coordinates retained by the complete-credit kernel.
// c++ -O3 -std=c++17 benchmarks/interval_support.cpp -o /tmp/interval-support
#include "../qkv_rosa/csrc/engine/edit_alphabet.h"
#include <iostream>
#include <random>

int main() {
    std::cout<<"length,bits,seed,retained_rows\n";
    for(int n:{512,2048,8192,32768})for(int b:{8,16,24,32,64})for(int seed:{101,223,337}) {
        std::mt19937_64 rng(seed);
        const uint64_t mask=UINT64_MAX>>(64-b);
        std::vector<uint64_t> q(n),k(n);
        std::vector<rosa_subq::Route> base(n);
        for(int t=0;t<n;++t){q[t]=rng()&mask;k[t]=rng()&mask;}
        // The support pass itself detects all causal equality, including the
        // winner. Empty route hints here exercise that independent guarantee.
        rosa_subq::Stats stats;
        auto rows=rosa_subq::interval_credit::live_coordinates(q.data(),k.data(),n,b,base,stats);
        int m=int(rows.size());
        if(rows.empty() || m>=n-n/4)m=n;
        if(m==1)m=0;
        std::cout<<n<<','<<b<<','<<seed<<','<<m<<'\n';
    }
}
