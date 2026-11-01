#include "sam_q.h"
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
using rosa_exact::SamQRepairs;
uint64_t cases=0,ancestors=0,length_queries=0,endpoint_queries=0,root_pairs=0;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void check(const std::vector<uint64_t>&q,const std::vector<uint64_t>&k,int d,uint64_t allowed,uint64_t seed) {
    const int n=int(q.size());++cases;
    SamQRepairs sam(q.data(),k.data(),n,d,allowed,-1,0,0,seed,true);
    require(sam.materialized_auxiliary_indexes()==0,"eager auxiliary materialization");
    // Independent full-word one-bit join with the causal latest-Q cutoff.
    std::set<std::pair<uint64_t,int>> expected;uint64_t supported=0;
    for(int t=0;t<n;++t)for(int bit=0;bit<d;++bit)if((allowed>>bit)&1)
        for(int r=0;r<t;++r)if((q[t]^(uint64_t(1)<<bit))==k[r]) {
            expected.insert({q[t],bit});supported|=uint64_t(1)<<bit;break;
        }
    root_pairs+=expected.size();
    require(sam.stored_root_entries()==expected.size(),"constant-coordinate join lost or added a root");
    require(sam.supported_repair_mask()==supported,"constant-coordinate support mismatch");
    for(int state=1;state<sam.state_count();++state) {
        for(int limit=0;limit<=n;++limit) {
            int want=state;while(want&&sam.first_endpoint(want)>=limit)want=sam.suffix_link(want);
            require(sam.ancestor_before(state,limit)==want,"depth-bounded causal ancestor mismatch");++ancestors;
        }
        for(int length=1;length<=sam.state_length(state);++length) {
            int want=state;while(sam.state_length(sam.suffix_link(want))>=length)want=sam.suffix_link(want);
            require(sam.state_for_length(state,length)==want,"depth-bounded length ancestor mismatch");++length_queries;
        }
    }
    for(int t=0;t<n;++t) {
        sam.advance(t);
        int want_length=0,want_end=-1;
        for(int r=0;r<t;++r) {
            int length=0;while(length<=r&&q[t-length]==k[r-length])++length;
            if(length>0&&(length>want_length||(length==want_length&&r>want_end))) {
                want_length=length;want_end=r;
            }
        }
        require(sam.baseline().length==want_length&&sam.baseline().end==want_end,"baseline match mismatch");
        for(int state=1;state<sam.state_count();++state) {
            int want=-1;std::vector<int> expected_ends,got;
            const auto range=sam.occurrence_range(state);
            for(int p=range.first;p<range.second;++p) {
                int r=sam.occurrence_endpoints()[p];if(r<t){want=std::max(want,r);expected_ends.push_back(r);}
            }
            sam.for_each_causal_occurrence(state,[&](int r){got.push_back(r);});
            std::sort(got.begin(),got.end());std::sort(expected_ends.begin(),expected_ends.end());
            require(got==expected_ends,"lazy occurrence chain mismatch");
            require(sam.latest_endpoint(state)==want,"lazy endpoint tree mismatch");
            require(sam.latest_endpoint(state)==want,"endpoint stamp mismatch");endpoint_queries+=2;
        }
    }
}
int main() {
    std::mt19937_64 rng(0x6843da8719);
    for(int n=0;n<=5;++n)for(uint64_t bits=0;bits<(uint64_t(1)<<(2*n));++bits) {
        std::vector<uint64_t>q(n),k(n);
        for(int i=0;i<n;++i){q[i]=(bits>>i)&1;k[i]=(bits>>(n+i))&1;}
        check(q,k,1,~uint64_t(0),bits);
    }
    for(int d:{1,2,8,16,32,64})for(int trial=0;trial<24;++trial) {
        int n=8+int(rng()%41);std::vector<uint64_t>q(n),k(n);
        uint64_t mask=d==64?~uint64_t(0):(uint64_t(1)<<d)-1;
        for(int t=0;t<n;++t){q[t]=rng()&mask;k[t]=rng()&mask;}
        if(trial%3==0)for(int t=0;t<n;++t)k[t]&=7;
        if(trial%6==0)for(int t=0;t<n;++t)q[t]=(rng()&7)^(uint64_t(1)<<63);
        if(trial%6==1)for(int t=0;t<n;++t)q[t]=(rng()&7)^(uint64_t(3)<<62);
        if(trial%6==2)for(int t=0;t<n;++t)k[t]|=uint64_t(1)<<63;
        if(trial%6==3)q=k;
        check(q,k,d,trial%2?~uint64_t(0):(uint64_t(1)<<63)|0x55,rng());
    }
    for(int n:{31,64,129}) {
        std::vector<uint64_t>q(n,0),k(n,0);check(q,k,64,~uint64_t(0),0);
        k[0]=1;check(q,k,64,~uint64_t(0),1);
        for(int t=0;t<n;++t)q[t]=k[t]=uint64_t(t%4);
        k[n/2]^=uint64_t(1)<<63;check(q,k,64,~uint64_t(0),~uint64_t(0));
    }
    std::cout<<"{\"status\":\"PASS\",\"cases\":"<<cases<<",\"root_pairs\":"<<root_pairs<<",\"ancestor_queries\":"<<ancestors<<",\"length_queries\":"<<length_queries<<",\"endpoint_queries\":"<<endpoint_queries<<"}\n";
}
