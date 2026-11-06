// Standalone FFT contraction oracle. Routes play no part in this test.
#include "../../conv_rosa_transformer/csrc/subquadratic/correlation.h"
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

namespace {
using rosa_subq::BlockCorrelations;
using rosa_subq::CorrelationQuery;
using rosa_subq::CorrelationStats;
std::uint64_t checked=0,zero_leakage=0;
double maximum_absolute=0,maximum_normalized=0,cancellation_error=0;

void check(int n,int d,const std::vector<double>&a,const std::vector<double>&b,
           const std::vector<CorrelationQuery>&queries,int block,bool adversarial=false) {
    CorrelationStats stats;
    const auto actual=BlockCorrelations::solve(n,d,a.data(),b.data(),queries,block,stats);
    for(std::size_t i=0;i<queries.size();++i) {
        const auto&q=queries[i];long double expected=0,magnitude=0;
        for(int t=q.lo;t<=q.hi;++t)for(int c=0;c<d;++c) {
            const long double term=static_cast<long double>(a[std::size_t(t)*d+c])*b[std::size_t(t+q.shift)*d+c];
            expected+=term;magnitude+=std::abs(term);
        }
        const double error=double(std::abs(static_cast<long double>(actual[i])-expected));
        const double normalized=error/double(1+magnitude);
        if(!std::isfinite(actual[i]) || normalized>3e-12) {
            std::cerr<<"correlation mismatch n="<<n<<" d="<<d<<" block="<<block
                     <<" interval="<<q.lo<<','<<q.hi<<" shift="<<q.shift
                     <<" expected="<<double(expected)<<" actual="<<actual[i]
                     <<" normalized="<<normalized<<'\n';
            throw std::runtime_error("correlation oracle failed");
        }
        maximum_absolute=std::max(maximum_absolute,error);
        maximum_normalized=std::max(maximum_normalized,normalized);
        if(expected==0 && actual[i]!=0)++zero_leakage;
        if(adversarial)cancellation_error=std::max(cancellation_error,error);
        ++checked;
    }
}
}

int main() {
    std::mt19937_64 rng(91276);
    for(int n:{1,2,3,7,8,15,17,31,33,65,129,513})for(int d:{1,3,65}) {
        std::vector<double>a(std::size_t(n)*d),b(a.size());
        for(auto&x:a)x=double(int(rng()%4097)-2048)/1024;
        for(auto&x:b)x=double(int(rng()%4097)-2048)/1024;
        std::vector<CorrelationQuery>queries;
        for(int shift=1-n;shift<n;++shift) {
            const int lo=std::max(0,-shift),hi=std::min(n-1,n-1-shift);
            queries.push_back({lo,hi,shift});
            const int mid=lo+(hi-lo)/2;
            queries.push_back({lo,mid,shift});queries.push_back({mid,hi,shift});
            queries.push_back({mid,mid,shift});
        }
        for(int i=0;i<200;++i) {
            const int shift=int(rng()%std::uint64_t(2*n-1))-n+1;
            const int lo=std::max(0,-shift),hi=std::min(n-1,n-1-shift);
            const int x=lo+int(rng()%std::uint64_t(hi-lo+1));
            const int y=x+int(rng()%std::uint64_t(hi-x+1));
            queries.push_back({x,y,shift});
        }
        for(int block=1;block<=std::min(128,2*n);block*=2)check(n,d,a,b,queries,block);
        check(n,d,a,b,queries,rosa_subq::correlation_block(n,d,queries));
    }
    // Cancellation leaves a unit residual amid 2^50-scale products. The
    // expected absolute error is reported, never hidden by relative tolerance.
    for(int n:{64,128,257}) {
        std::vector<double>a(n),b(n,1.);
        for(int i=0;i<n;++i)a[i]=(i&1)?-std::ldexp(1.,50):std::ldexp(1.,50);
        a[0]+=1.;
        std::vector<CorrelationQuery>queries={{0,n-1,0},{0,n-2,0},{1,n-2,0}};
        for(int block:{2,4,8,16,32})check(n,1,a,b,queries,block,true);
    }
    // Equal large channels cancel; the small third channel must be compared
    // against the independent long-double accumulation, not a double dot.
    {
        const int n=128,d=3;std::vector<double>a(n*d),b(n*d);
        for(int t=0;t<n;++t) {
            a[t*d]=std::ldexp(1.,45);a[t*d+1]=-a[t*d];a[t*d+2]=1.;
            b[t*d]=b[t*d+1]=1.;b[t*d+2]=double(t%7-3)/8;
        }
        std::vector<CorrelationQuery>queries={{0,n-1,0},{3,n-4,0},{0,n-4,3},{3,n-1,-3}};
        for(int block:{2,4,8,16,32})check(n,d,a,b,queries,block,true);
    }
    std::cout<<std::setprecision(17)<<"{\"status\":\"PASS\",\"queries\":"<<checked
             <<",\"max_absolute_error\":"<<maximum_absolute
             <<",\"max_error_over_l1\":"<<maximum_normalized
             <<",\"zero_outputs_with_roundoff\":"<<zero_leakage
             <<",\"adversarial_absolute_error\":"<<cancellation_error<<"}\n";
}
