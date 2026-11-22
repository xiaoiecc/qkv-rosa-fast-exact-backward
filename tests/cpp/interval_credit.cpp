#include "../../qkv_rosa/csrc/engine/engine.h"
#include "../../qkv_rosa/csrc/causal_windows.h"
#include <iostream>
#include <random>

using rosa_subq::Route;
using Words=std::vector<uint64_t>;

// Literal causal DP. It shares neither occurrence tables nor compacted time.
static std::vector<Route> literal(const Words& q,const Words& k) {
    int n=int(q.size());std::vector<int> row(n);
    std::vector<Route> result(n);
    for(int t=1;t<n;++t)for(int r=t-1;r>=0;--r) {
        row[r]=q[t]==k[r]?1+(r?row[r-1]:0):0;
        if(row[r])result[t]=std::max(result[t],Route{row[r],r});
    }
    return result;
}

int main() {
    std::mt19937_64 rng(908671);
    uint64_t credits=0,forward_rows=0;
    for(int family=0;family<6;++family)for(int bits:{1,3,8,16,32,64}) {
        int n=family<4?37:77;
        Words q(n),k(n);uint64_t mask=UINT64_MAX>>(64-bits);
        for(int i=0;i<n;++i) {
            q[i]=rng()&mask;k[i]=rng()&mask;
            if(family==1){q[i]=0;k[i]=i%7==2?1:0;}
            if(family==2){q[i]=(i%4)&mask;k[i]=q[i]^(i==n/2);}
            if(family==3){q[i]|=UINT64_C(1)<<62;k[i]|=UINT64_C(1)<<62;}
            if(family==4){q[i]=i%5?0:1;k[i]=i%17?0:1;}
        }
        if(family==0 || family==5) {
            // Active islands, isolated gaps and unmatchable end padding.
            k[1]=q[17];k[2]=q[18];k[6]=q[26]^(UINT64_C(1)<<(bits-1));
            k[9]=q[31];k[12]=q[32];
        }
        auto base=literal(q,k);
        std::vector<int32_t> lengths(n),routes(n);
        rosa_exact::training_routes(q.data(),k.data(),n,bits,lengths.data(),routes.data());
        for(int t=0;t<n;++t)if(Route{lengths[t],routes[t]}!=base[t])return 1;
        forward_rows+=n;
        size_t cells=size_t(n)*bits;
        std::vector<uint8_t> v(cells);
        std::vector<double> g(cells),h(cells),e0(bits),e1(bits),cq,ck;
        for(size_t i=0;i<cells;++i){v[i]=rng()&1;g[i]=int(rng()%11)-5;h[i]=double(rng()%5);}
        for(int j=0;j<bits;++j){e0[j]=int(rng()%7)-3;e1[j]=int(rng()%5)-2;}
        auto value=[&](int end,int j){return end<0?e0[j]:(2*int(v[size_t(end+1)*bits+j])-1)*e1[j];};
        rosa_subq::Stats stats;
        if(rosa_subq::run(n,bits,q.data(),k.data(),v.data(),g.data(),h.data(),e0.data(),e1.data(),
                           false,0,nullptr,cq,ck,lengths.data(),routes.data(),stats,nullptr))return 2;
        for(int side=0;side<2;++side)for(int s=0;s<n;++s)for(int b=0;b<bits;++b) {
            uint64_t& w=side?k[s]:q[s];const uint64_t flip=UINT64_C(1)<<b;
            w^=flip;auto edited=literal(q,k);w^=flip;
            double expected=0.;
            for(int t=0;t<n;++t)if(edited[t].end!=base[t].end)for(int j=0;j<bits;++j) {
                size_t i=size_t(t)*bits+j;double dy=value(edited[t].end,j)-value(base[t].end,j);
                expected+=g[i]*dy+.5*h[i]*dy*dy;
            }
            if(expected!=(side?ck:cq)[size_t(s)*bits+b]) {
                std::cerr<<"credit mismatch "<<family<<" "<<bits<<" "<<side<<" "<<s<<" "<<b<<"\n";return 3;
            }
            ++credits;
        }
    }
    std::cout<<"{\"status\":\"PASS\",\"physical_flips\":"<<credits
             <<",\"forward_rows\":"<<forward_rows<<"}\n";
}
