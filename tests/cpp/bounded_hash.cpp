#include "full_word_map.h"
#include "approx_row_certificate.h"
#include "forward.h"
#include <iostream>
#include <stdexcept>

static uint64_t mix(uint64_t x) {
    x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
    return x^(x>>31);
}
static void require(bool ok) {if(!ok)throw std::runtime_error("collision oracle disagrees");}
int main() {
    uint64_t checks=0;
    for(uint64_t seed:{uint64_t(0),uint64_t(7),~uint64_t(0)}) {
        // All keys share a bucket even after the final growth. More than
        // sixteen distinct keys force the ordered overflow representation.
        rosa_exact::FullWordMap words(64,seed);std::vector<uint64_t> keys;
        for(uint64_t x=0;keys.size()<64;++x)if(!(mix(x^seed)&127))keys.push_back(x);
        for(int i=0;i<64;++i)require(words.insert(keys[i],i)==std::make_pair(i,true));
        for(int i=63;i>=0;--i) {
            require(words.find(keys[i])==i);
            require(words.insert(keys[i],-1)==std::make_pair(i,false));checks+=2;
        }

        // Colliding adjacent pairs also exceed the direct membership scan's
        // bit budget, exercising overflow lookup as well as insertion.
        auto hash_pair=[&](uint64_t a,uint64_t b) {
            const uint64_t y=mix(b^(seed+UINT64_C(0x9e3779b97f4a7c15)));
            return mix(a^seed)^((y<<1)|(y>>63));
        };
        std::vector<uint64_t> right;
        for(uint64_t x=1;right.size()<128;++x)if(!(hash_pair(0,x)&1023))right.push_back(x);
        std::vector<uint64_t> k;
        for(uint64_t x:right){k.push_back(0);k.push_back(x);}
        k.push_back(0);k.push_back(0);
        rosa_random::AdjacentPairCertificate certificate(k.data(),int(k.size()),seed);
        for(int i=0;i<128;++i)for(int delta:{0,1,63}) {
            const uint64_t a=0,b=right[i]^(delta?uint64_t(1)<<delta:0);
            const int t=int(k.size())-1;bool expected=false;
            for(int r=1;r<t;++r) {
                const uint64_t x=a^k[r-1],y=b^k[r];
                expected|=(!x&&(!y||!(y&(y-1))))||(!y&&(!x||!(x&(x-1))));
            }
            uint64_t stats[8]={};require(certificate.needs_residual(a,b,t,64,stats)==expected);++checks;
        }

        // State zero's transition dictionary has 1,024 slots at T=128.
        // Full (state,word) comparisons must survive the same collision set.
        const uint64_t salt=mix(seed+UINT64_C(0x9e3779b97f4a7c15));
        k.clear();for(uint64_t x=0;k.size()<128;++x)if(!((mix(x^seed)^salt)&1023))k.push_back(x);
        std::vector<uint64_t> q(k.rbegin(),k.rend());
        for(int pattern=0;pattern<2;++pattern) {
            if(pattern)for(int t=0;t<128;++t)q[t]=k[(t+127)%128];
            rosa_exact::SamForward forward(q.data(),k.data(),128,seed);
            std::vector<int32_t> lengths(128),routes(128);forward.run(lengths.data(),routes.data());
            for(int t=0;t<128;++t) {
                std::pair<int,int> best{0,-1};
                for(int r=0;r<t;++r) {
                    int len=0;while(len<=r&&q[t-len]==k[r-len])++len;
                    if(len)best=std::max(best,std::make_pair(len,r));
                }
                require(best==std::make_pair(int(lengths[t]),int(routes[t])));++checks;
            }
        }
    }
    std::cout<<"{\"status\":\"PASS\",\"adversarial_collision_checks\":"<<checks<<"}\n";
}
