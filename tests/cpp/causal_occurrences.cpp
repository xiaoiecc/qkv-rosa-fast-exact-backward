#include "sam_q.h"
#include <iostream>
#include <random>
#include <stdexcept>
using rosa_exact::SamQRepairs;
uint64_t cases=0,rows=0,queries=0,memberships=0,latest_queries=0;
void check(const std::vector<uint64_t>&q,const std::vector<uint64_t>&k,int d) {
    ++cases; const int n=int(q.size()); SamQRepairs sam(q.data(),k.data(),n,d);
    for(int t=0;t<n;++t) {
        sam.advance(t);++rows;
        for(int state=1;state<sam.state_count();++state) {
            ++queries;
            std::vector<int> got;
            sam.for_each_causal_occurrence(state,[&](int r) {
                if(r<0 || r>=t || int(got.size())>=n) throw std::runtime_error("noncausal, cycle, or duplicate");
                got.push_back(r);++memberships;
            });
            std::sort(got.begin(),got.end());
            std::vector<int>want;const auto range=sam.occurrence_range(state);
            for(int p=range.first;p<range.second;++p) {
                int r=sam.occurrence_endpoints()[p]; if(r<t)want.push_back(r);
            }
            std::sort(want.begin(),want.end());
            if(got!=want) throw std::runtime_error("causal occurrence set mismatch");
            const int expected=want.empty()?-1:want.back();
            if(sam.latest_endpoint(state)!=expected) throw std::runtime_error("latest endpoint mismatch");
            if(sam.latest_endpoint(state)!=expected) throw std::runtime_error("latest endpoint memoization mismatch");
            latest_queries+=2;
        }
    }
}
int main() {
    for(int n=0;n<=6;++n)for(uint64_t mask=0;mask<(uint64_t(1)<<(2*n));++mask) {
        std::vector<uint64_t>q(n),k(n);
        for(int i=0;i<n;++i){q[i]=(mask>>i)&1;k[i]=(mask>>(n+i))&1;}
        check(q,k,1);
    }
    std::mt19937_64 rng(99125835);
    for(int d:{1,2,3,8,16,32,64})for(int trial=0;trial<80;++trial) {
        int n=1+int(rng()%64);std::vector<uint64_t>q(n),k(n);
        const uint64_t mask=d==64?~uint64_t(0):(uint64_t(1)<<d)-1;
        for(int i=0;i<n;++i){q[i]=rng()&mask;k[i]=rng()&mask;}
        if(trial%2==0)k=q;
        if(trial%5==0)for(int i=0;i<n;++i)k[i]=q[i]=i%std::min(1<<std::min(d,8),4);
        check(q,k,d);
    }
    for(int n:{31,64,257}) {
        std::vector<uint64_t>q(n,0),k(n,0); check(q,k,1);
        k[0]=1;check(q,k,1);
        for(int i=0;i<n;++i){q[i]=i%4;k[i]=q[i];} k[n/2]^=1;check(q,k,2);
    }
    std::cout<<"{\"status\":\"PASS\",\"input_cases\":"<<cases<<",\"rows\":"<<rows<<",\"state_queries\":"<<queries<<",\"causal_memberships\":"<<memberships<<",\"latest_queries\":"<<latest_queries<<"}\n";
}
