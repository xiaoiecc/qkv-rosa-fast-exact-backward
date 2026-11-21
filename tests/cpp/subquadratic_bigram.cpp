// Physical bit flips followed by an independent suffix DP for every row.
#include "../../conv_rosa_transformer/csrc/subquadratic/bigram_credit.h"
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace rosa_subq;

std::vector<Route> match(const std::vector<uint64_t>& q,const std::vector<uint64_t>& k) {
    const int n=q.size();
    std::vector<Route> base(n);
    std::vector<int> length(n);
    for(int t=1;t<n;++t)for(int r=t-1;r>=0;--r) {
        length[r]=q[t]==k[r]?(r?length[r-1]:0)+1:0;
        if(length[r])base[t]=std::max(base[t],Route{length[r],r});
    }
    return base;
}
int main() {
    std::mt19937_64 rng(865972);
    int checked=0;
    for(int b:{4,5,8,12,13,15,16})for(int rep=0;rep<100;++rep) {
        const int n=3+rep%21,d=1+rep%5;
        std::vector<uint64_t> q(n),k(n);
        for(int t=0;t<n;++t) {
            const int space=(rep%3?4:1<<b);
            q[t]=rng()%space;k[t]=rng()%space;
        }
        if(rep%4==0)for(int t=1;t<n;++t)k[t-1]=q[t];
        if(rep%5==0)k[0]^=uint64_t(1)<<(b-1);
        const auto base=match(q,k);
        std::vector<double> w(n*d),v(n*d),empty(d),expected, a(n*b),c(n*b);
        for(auto p:{&w,&v,&empty})for(double& x:*p)x=(int(rng()%23)-11)*.25;
        Scores score{n,d,w.data(),v.data(),empty.data()};Stats st;
        if(!bigram_credit(q.data(),k.data(),base,b,score,a,c,st))continue;
        for(int side=0;side<2;++side)for(int s=0;s<n;++s)for(int j=0;j<b;++j) {
            auto& word=side?k[s]:q[s];word^=uint64_t(1)<<j;
            const auto changed=match(q,k);word^=uint64_t(1)<<j;
            double sum=0;
            for(int t=0;t<n;++t)sum+=score.score(t,changed[t].end)-score.score(t,base[t].end);
            const double got=(side?c:a)[s*b+j];
            if(got!=sum){std::cerr<<"mismatch n "<<n<<" rep "<<rep<<" b "<<b<<" side "<<side
                <<" s "<<s<<" j "<<j<<" got "<<got<<" expected "<<sum<<'\n';return 1;}
            ++checked;
        }
    }
    // max exact length zero, with quadratic two-word one-bit fanout after
    // the first exact symbols are introduced. Rejection leaves callers intact.
    std::vector<uint64_t> q(1024),k(1024);for(int i=0;i<1024;++i)k[i]=i%2;
    const auto base=match(q,k);std::vector<double>w(1024,1),v(1024,2),empty(1),cq(4096,3),ck(4096,5);
    Stats st;Scores score{1024,1,w.data(),v.data(),empty.data()};
    if(bigram_credit(q.data(),k.data(),base,4,score,cq,ck,st) || cq!=std::vector<double>(4096,3)
       || ck!=std::vector<double>(4096,5))return 1;
    std::cout<<"{\"status\":\"PASS\",\"checked\":"<<checked<<"}\n";
}
