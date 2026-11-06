// Independent literal edited/deleted-Q routes for the bounded live frontier.
#include "../../conv_rosa_transformer/csrc/subquadratic/q_frontier.h"
#include <cstring>
#include <iostream>
#include <random>
#include <string>

namespace {
using namespace rosa_subq;
std::uint64_t fixtures=0,accepted=0,rejected=0,forward_rows=0;
[[noreturn]] void fail(const char* message){throw std::runtime_error(message);}

std::vector<Route> literal(const std::vector<std::uint64_t>&q,
                           const std::vector<std::uint64_t>&k,int deleted=-1) {
    const int n=int(q.size());std::vector<Route> routes(n);
    std::vector<int> previous(n),current(n);
    for(int t=0;t<n;++t) {
        for(int r=0;r<n;++r) {
            current[r]=t!=deleted && q[t]==k[r]?1+(t&&r?previous[r-1]:0):0;
            if(r<t && current[r])routes[t]=std::max(routes[t],Route{current[r],r});
        }
        previous.swap(current);
    }
    return routes;
}
bool same_bits(const std::vector<double>&a,const std::vector<double>&b) {
    return a.size()==b.size() && (a.empty() || !std::memcmp(a.data(),b.data(),a.size()*sizeof(double)));
}
void forward_check(const std::vector<std::uint64_t>&q,const std::vector<std::uint64_t>&k,
                   const std::vector<Route>&expected,std::uint64_t seed) {
    const int n=int(q.size());std::vector<int32_t>length(n),route(n);
    rosa_exact::SamForward forward(q.data(),k.data(),n,seed);forward.run(length.data(),route.data());
    rosa_exact::SamForward skipped(q.data(),k.data(),n,seed);
    rosa_exact::SamForward::Cursor cursor;
    for(int t=0;t<n;++t) {
        if(length[t]!=expected[t].length || route[t]!=expected[t].end)fail("forward route differs from literal");
        const int minimum=t%3?n+1:1;
        skipped.step(t,q[t],cursor,minimum);
        if(cursor.length!=expected[t].length || cursor.end!=(cursor.length>=minimum?expected[t].end:-1))
            fail("deferred endpoint cursor differs from literal");
        ++forward_rows;
    }
}
void check(const std::vector<std::uint64_t>&q,const std::vector<std::uint64_t>&k,
           int bits,bool integral,std::mt19937_64&random) {
    ++fixtures;const int n=int(q.size()),d=5;
    const auto base=literal(q,k);
    for(std::uint64_t seed:{UINT64_C(0),UINT64_C(817),UINT64_MAX})forward_check(q,k,base,seed);
    if(!n)return;
    std::vector<double>w(std::size_t(n)*d),v(w.size()),empty(d),initial(std::size_t(n)*bits);
    auto number=[&](){return integral?double(int(random()%7)-3):double(int(random()%4097)-2048)/683.;};
    for(auto&x:w)x=number();for(auto&x:v)x=number();for(auto&x:empty)x=number();for(auto&x:initial)x=number();
    if(n>3)for(int j=0;j<d;++j)w[std::size_t(n/2)*d+j]=0;
    const Scores scores{n,d,w.data(),v.data(),empty.data()};
    Index index(q.data(),k.data(),n,true,711,&base);
    for(bool include_deletion:{false,true}) {
    std::vector<double>first_result;
    bool first_status=false;
    for(std::uint64_t seed:{UINT64_C(0),UINT64_C(817),UINT64_MAX}) {
        std::vector<double>actual=initial;Stats stats;stats.q_heavy_rows=17;stats.suffix_queries=29;
        const Stats before=stats;const bool success=q_frontier(index,scores,bits,actual,stats,seed,include_deletion);
        if(seed==0){first_status=success;first_result=actual;}
        else if(success!=first_status || !same_bits(actual,first_result))fail("frontier seed changed output");
        if(!success) {
            ++rejected;
            if(!same_bits(actual,initial) || std::memcmp(&before,&stats,sizeof(stats)))fail("rejected frontier published partial output");
            continue;
        }
        ++accepted;
        for(int owner=0;owner<n;++owner) {
            const auto removed=literal(q,k,owner);
            for(int bit=0;bit<bits;++bit) {
                auto edited=q;edited[owner]^=std::uint64_t(1)<<bit;const auto routes=literal(edited,k);
                double expected=0;
                for(int t=owner;t<n;++t)expected+=scores.score(t,routes[t].end)
                    -scores.score(t,include_deletion?base[t].end:removed[t].end);
                expected+=initial[std::size_t(owner)*bits+bit];
                const double got=actual[std::size_t(owner)*bits+bit];
                if(!std::isfinite(got) || (integral?got!=expected:std::abs(got-expected)>2e-10*(1+std::abs(expected))))
                    fail(include_deletion?"full frontier credit differs from literal edit minus original"
                                         :"frontier repair credit differs from literal edit minus deletion");
            }
        }
    }
    }
}
void rejection(int n,int bits,bool work_cap) {
    std::vector<std::uint64_t>q(n,0),k(n,0);
    if(work_cap) {
        // L<=48 and at most three live edited cursors: the live certificate
        // fits cap=64. Each isolated one-bit word nevertheless consumes up
        // to 48 suffix-link fallbacks after copying the original cursor.
        const std::uint64_t tail[]={7,1,7,2,7,4,7};
        for(int t=0;t<n;++t)if(t%55>=48)k[t]=tail[t%55-48];
    } else k[0]=1;
    Index index(q.data(),k.data(),n);
    std::vector<double>w(n,1),v(n,2),empty(1,3),credit(std::size_t(n)*bits,-19.);
    for(bool include_deletion:{false,true}) {
        const auto before=credit;Stats stats;stats.q_heavy_rows=131;stats.suffix_queries=313;const Stats old=stats;
        if(q_frontier(index,Scores{n,1,w.data(),v.data(),empty.data()},bits,credit,stats,29,include_deletion))
            fail(work_cap?"copied-cursor fallback budget did not reject":"live frontier budget did not reject");
        if(!same_bits(credit,before) || std::memcmp(&old,&stats,sizeof(stats)))fail("rejection changed output or counters");
        ++rejected;
    }
}
}
int main(int argc,char**argv) {
    try {
        const bool quick=argc>1 && std::string(argv[1])=="--quick";
        std::mt19937_64 random(781029);
        check({}, {},1,true,random);
        for(int n=1;n<=(quick?4:6);++n)for(std::uint64_t mask=0;mask<(UINT64_C(1)<<(2*n));++mask) {
            std::vector<std::uint64_t>q(n),k(n);
            for(int t=0;t<n;++t){q[t]=(mask>>t)&1;k[t]=(mask>>(n+t))&1;}
            check(q,k,1,true,random);
        }
        for(int trial=0;trial<(quick?25:180);++trial) {
            const int n=1+int(random()%40),bits=trial%17==0?64:1+int(random()%8);
            const std::uint64_t mask=bits==64?UINT64_MAX:(UINT64_C(1)<<bits)-1;
            std::vector<std::uint64_t>q(n),k(n);
            for(int t=0;t<n;++t) {
                q[t]=random()&mask;k[t]=random()&mask;
                if(trial%4==1){q[t]=k[t]=(t%3)&mask;if(t==n/3)k[t]^=1;}
                if(trial%4==2){q[t]=k[t]=0;if(t==0)k[t]=1;}
                if(trial%4==3){q[t]&=1;k[t]&=1;}
            }
            check(q,k,bits,false,random);
        }
        rejection(60,1,false); // Exceeds combined live cap, without the early L certificate.
        rejection(128,1,false); // Rejects before matcher construction using saved L.
        rejection(1024,3,true);
        std::cout<<"{\"status\":\"PASS\",\"fixtures\":"<<fixtures<<",\"accepted\":"<<accepted
            <<",\"rejected\":"<<rejected<<",\"forward_rows\":"<<forward_rows<<"}\n";
    } catch(const std::exception&error){std::cerr<<"FAIL fixture "<<fixtures<<": "<<error.what()<<'\n';return 1;}
}
