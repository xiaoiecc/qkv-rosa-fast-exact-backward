#include "../../conv_rosa_transformer/csrc/subquadratic/delete.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>

using namespace rosa_subq;
namespace {
int cases=0;
[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

// Literal route enumeration: no suffix index, occurrence AP, or envelope.
Route literal(const std::vector<std::uint64_t>& q,const std::vector<std::uint64_t>& k,
              int t,int qo=-1,int ko=-1,int cutoff=-1) {
    Route result;const int stop=cutoff<0?t:std::min(t,cutoff);
    for(int r=0;r<stop;++r) {
        int length=0;
        while(length<=t && length<=r && t-length!=qo && r-length!=ko && q[t-length]==k[r-length])++length;
        if(length)result=std::max(result,Route{length,r});
    }
    return result;
}

struct Oracle {
    int n;std::vector<Route> base,qdeleted,kdeleted;
    Oracle(const std::vector<std::uint64_t>& q,const std::vector<std::uint64_t>& k):n(int(q.size())),base(n),qdeleted(n*n),kdeleted(n*n) {
        // Independent O(n^3) scan of every possible endpoint. The recurrence
        // just caches literal suffix comparisons; deletion clips a match at
        // the forbidden physical symbol, without using A/H crossing formulas.
        std::vector<int> lengths(n*n,0);
        for(int t=0;t<n;++t)for(int r=0;r<n;++r)
            if(q[t]==k[r])lengths[t*n+r]=1+(t && r?lengths[(t-1)*n+r-1]:0);
        for(int t=0;t<n;++t) {
            for(int r=0;r<t;++r)if(lengths[t*n+r])base[t]=std::max(base[t],Route{lengths[t*n+r],r});
            for(int owner=0;owner<n;++owner)for(int r=0;r<t;++r) {
                int qlength=lengths[t*n+r],klength=qlength;
                if(owner<=t)qlength=std::min(qlength,t-owner);
                if(owner<=r)klength=std::min(klength,r-owner);
                if(qlength)qdeleted[t*n+owner]=std::max(qdeleted[t*n+owner],Route{qlength,r});
                if(klength)kdeleted[t*n+owner]=std::max(kdeleted[t*n+owner],Route{klength,r});
            }
        }
        if(n<=7)for(int t=0;t<n;++t)for(int owner=0;owner<n;++owner) {
            if(qdeleted[t*n+owner]!=literal(q,k,t,owner,-1) || kdeleted[t*n+owner]!=literal(q,k,t,-1,owner))
                fail("internal literal oracle mismatch");
        }
    }
};

void close(double actual,double expected,const char* operation,int n,int width,int item,bool integral) {
    const double tolerance=integral?0.0:2e-10*(1.0+std::abs(expected));
    if(!std::isfinite(actual) || std::abs(actual-expected)>tolerance) {
        std::ostringstream out;out.precision(17);out<<operation<<" n="<<n<<" width="<<width<<" item="<<item<<" actual="<<actual<<" expected="<<expected;
        fail(out.str());
    }
}

void check(const std::vector<std::uint64_t>& q,const std::vector<std::uint64_t>& k,
           int dimensions,std::mt19937_64& random,bool thorough,bool integral=true) {
    ++cases;const int n=int(q.size());const Oracle oracle(q,k);
    Index index(q.data(),k.data(),n),cached(q.data(),k.data(),n,true,random());
    std::vector<double> weights(n*dimensions),values(n*dimensions),empty(dimensions);
    auto number=[&](){return integral?double(int(random()%7)-3):double(int(random()%4001)-2000)/113.0;};
    for(double& x:weights)x=number();
    for(double& x:values)x=number();
    for(double& x:empty)x=number();
    Scores scores{n,dimensions,weights.data(),values.data(),empty.data()};
    for(int t=0;t<n;++t) {
        if(index.best(t)!=oracle.base[t] || cached.best(t)!=oracle.base[t])fail("baseline route mismatch");
        for(int owner=0;owner<n;++owner)if(index.deleted(t,owner)!=oracle.kdeleted[t*n+owner])fail("indexed deletion route mismatch");
        if(thorough)for(int cutoff=0;cutoff<=t;++cutoff) {
            if(index.best(t,cutoff)!=literal(q,k,t,-1,-1,cutoff))fail("indexed prefix route mismatch");
            for(int h=1;h<=t+1;++h) {
                int latest=-1;
                for(int r=0;r<cutoff;++r) {
                    int length=0;while(length<=t && length<=r && q[t-length]==k[r-length])++length;
                    if(length>=h)latest=r;
                }
                if(index.R(t,h,cutoff)!=latest || cached.R(t,h,cutoff)!=latest)fail("indexed R mismatch");
            }
        }
    }
    std::vector<int> widths;
    if(thorough)for(int width=1;width<=n;++width)widths.push_back(width);
    else {widths={1,2,std::max(1,delete_detail::square_root(n)),std::max(1,n/3),n};std::sort(widths.begin(),widths.end());widths.erase(std::unique(widths.begin(),widths.end()),widths.end());}
    if(thorough || cases%5==0)for(int width:widths)for(int t=width;t<n;++t) {
        Stats stats;delete_detail::APScanner scanner(index,t,width,0,t,stats);
        delete_detail::APPiece piece;std::vector<int> seen(t,0);
        while(scanner.next(piece))for(int r=piece.first;r<=piece.last;r+=piece.period?piece.period:1) {
            if(r<0 || r>=t || ++seen[r]!=1)fail("AP occurrence coverage overlap");
            int actual=0;while(actual<=t && actual<=r && q[t-actual]==k[r-actual])++actual;
            const int represented=piece.period?std::min(piece.hq,r-piece.left+1):piece.fixed;
            if(actual!=represented || actual<width)fail("AP full suffix profile mismatch");
        }
        for(int r=0;r<t;++r) {
            int actual=0;while(actual<=t && actual<=r && q[t-actual]==k[r-actual])++actual;
            if(bool(seen[r])!=(actual>=width))fail("AP occurrence coverage gap");
        }
    }
    std::vector<std::vector<int>> selections(3);
    for(int t=0;t<n;++t)if(t%2==0)selections[0].push_back(t);
    for(int t=0;t<n;++t)selections[1].push_back(t);
    std::shuffle(selections[1].begin(),selections[1].end(),random);
    for(int option=-1;option<int(selections.size());++option) {
        const std::vector<int>* rows=option<0?nullptr:&selections[option];
        std::vector<double> expected_q(n,0.0),expected_k(n,0.0);
        for(int t=0;t<n;++t)if(!rows || std::find(rows->begin(),rows->end(),t)!=rows->end()) {
            const double baseline=scores.score(t,oracle.base[t].end);
            for(int owner=0;owner<n;++owner) {
                expected_q[owner]+=scores.score(t,oracle.qdeleted[t*n+owner].end)-baseline;
                expected_k[owner]+=scores.score(t,oracle.kdeleted[t*n+owner].end)-baseline;
            }
        }
        for(int width:widths) {
            Stats stats;const auto aq=q_deletion(index,scores,width,stats,rows),ak=k_deletion(index,scores,width,stats,rows);
            const auto bq=q_deletion(cached,scores,width,stats,rows),bk=k_deletion(cached,scores,width,stats,rows);
            if(std::memcmp(aq.data(),bq.data(),aq.size()*sizeof(double)) || std::memcmp(ak.data(),bk.data(),ak.size()*sizeof(double)))fail("randomized cache changed arithmetic output");
            for(int owner=0;owner<n;++owner) {
                close(aq[owner],expected_q[owner],"Q deletion",n,width,owner,integral);
                close(ak[owner],expected_k[owner],"K deletion",n,width,owner,integral);
            }
        }
    }
    std::vector<OwnerRange> queries;
    if(n<=5)for(int owner=0;owner<n;++owner)for(int lo=0;lo<n;++lo)for(int hi=lo;hi<n;++hi)queries.push_back({owner,lo,hi});
    else {
        for(int owner=0;owner<n;++owner){queries.push_back({owner,0,n-1});queries.push_back({owner,owner,owner});}
        for(int i=0;i<100;++i){int a=int(random()%n),b=int(random()%n);if(a>b)std::swap(a,b);queries.push_back({int(random()%n),a,b});}
    }
    std::vector<double> expected_q(queries.size(),0.0),expected_k(queries.size(),0.0);
    for(std::size_t i=0;i<queries.size();++i)for(int t=queries[i].lo;t<=queries[i].hi;++t) {
        expected_q[i]+=scores.score(t,oracle.qdeleted[t*n+queries[i].owner].end);
        expected_k[i]+=scores.score(t,oracle.kdeleted[t*n+queries[i].owner].end);
    }
    for(int width:widths) {
        Stats stats;const auto aq=q_delete_ranges(index,scores,queries,width,stats),ak=k_delete_ranges(index,scores,queries,width,stats);
        for(std::size_t j=0;j<queries.size();++j) {
            close(aq[j],expected_q[j],"Q range",n,width,int(j),integral);
            close(ak[j],expected_k[j],"K range",n,width,int(j),integral);
        }
    }
}
}

int main(int argc,char** argv) {
    try {
        const bool quick=argc>1 && std::string(argv[1])=="--quick";
        std::mt19937_64 random(73534493);
        for(int n=1;n<=(quick?4:6);++n)for(std::uint64_t mask=0;mask<(UINT64_C(1)<<(2*n));++mask) {
            std::vector<std::uint64_t> q(n),k(n);
            for(int i=0;i<n;++i){q[i]=(mask>>i)&1;k[i]=(mask>>(n+i))&1;}
            check(q,k,3,random,true);
        }
        for(int iteration=0;iteration<(quick?30:200);++iteration) {
            const int n=1+int(random()%100),period=1+int(random()%11),alphabet=1+int(random()%6);
            std::vector<std::uint64_t> q(n),k(n),pattern(period);
            for(auto& x:pattern)x=random()%alphabet;
            for(int i=0;i<n;++i) {
                if(iteration%4==0){q[i]=random()%alphabet;k[i]=random()%alphabet;}
                else {q[i]=pattern[i%period];k[i]=pattern[(i+iteration)%period];}
            }
            if(iteration%4==2){for(int z=0;z<3;++z)q[random()%n]=UINT64_MAX;}
            if(iteration%4==3){for(int z=0;z<3;++z)k[random()%n]^=UINT64_C(1)<<63;}
            if(iteration%5==0)for(int i=0;i<n;++i){q[i]^=UINT64_C(1)<<63;k[i]^=UINT64_C(1)<<63;}
            const int dimensions=iteration%17==0?65:1+int(random()%8);
            check(q,k,dimensions,random,false,iteration%7!=0);
        }
        std::cout<<"{\"status\":\"PASS\",\"cases\":"<<cases<<"}\n";
    } catch(const std::exception& error) { std::cerr<<"FAIL case "<<cases<<": "<<error.what()<<'\n';return 1; }
}
