#include "../../conv_rosa_transformer/csrc/subquadratic/fingerprint_k.h"
#include <cstring>
#include <iostream>
#include <random>
#include <string>
using namespace rosa_subq;
namespace {
std::uint64_t fixtures=0,accepted=0,rejected=0,values=0;
[[noreturn]] void fail(const char* message){throw std::runtime_error(message);}
bool same(const std::vector<double>&a,const std::vector<double>&b) {
    return a.size()==b.size() && (a.empty() || !std::memcmp(a.data(),b.data(),a.size()*sizeof(double)));
}
std::vector<Route> literal(const std::vector<std::uint64_t>&q,const std::vector<std::uint64_t>&k,int deleted=-1) {
    const int n=int(q.size());std::vector<Route>routes(n);std::vector<int>previous(n),current(n);
    for(int t=0;t<n;++t){for(int end=0;end<n;++end){
        current[end]=end!=deleted && q[t]==k[end]?1+(t&&end?previous[end-1]:0):0;
        if(end<t && current[end])routes[t]=std::max(routes[t],Route{current[end],end});
    }previous.swap(current);}return routes;
}
void check(const std::vector<std::uint64_t>&q,const std::vector<std::uint64_t>&k,int bits,bool integral,std::mt19937_64&random,bool sampled=false) {
    ++fixtures;const int n=int(q.size()),d=3;const auto base=literal(q,k);
    std::vector<double>w(std::size_t(n)*d),v(w.size()),empty(d),initial(std::size_t(n)*bits);
    auto number=[&](){return integral?double(int(random()%7)-3):double(int(random()%4001)-2000)/683.;};
    for(auto&x:w)x=number();for(auto&x:v)x=number();for(auto&x:empty)x=number();for(auto&x:initial)x=number();
    if(n>3)for(int c=0;c<d;++c)w[std::size_t(n/2)*d+c]=0.;
    const Scores scores{n,d,w.data(),v.data(),empty.data()};Index index(q.data(),k.data(),n,true,711,&base);
    std::vector<int>owners;
    if(sampled)owners={0,1,n/7,n/3,n/2,n-2,n-1};else for(int s=0;s<n;++s)owners.push_back(s);
    std::vector<double>expected_full=initial,expected_repair=initial;
    for(const int owner:owners) {
        const auto removed=literal(q,k,owner);
        for(int bit=0;bit<bits;++bit){auto edited=k;edited[owner]^=UINT64_C(1)<<bit;const auto routes=literal(q,edited);
            double full=0,repair=0;for(int t=owner+1;t<n;++t)for(int c=0;c<d;++c) {
                full+=scores.row(t)[c]*(scores.value(routes[t].end)[c]-scores.value(base[t].end)[c]);
                repair+=scores.row(t)[c]*(scores.value(routes[t].end)[c]-scores.value(removed[t].end)[c]);
            }
            expected_full[std::size_t(owner)*bits+bit]+=full;expected_repair[std::size_t(owner)*bits+bit]+=repair;
        }
    }
    for(bool full:{false,true}) {
        std::vector<double>first;bool first_status=false;
        for(std::uint64_t seed:{UINT64_C(0),UINT64_C(817)})for(std::uint64_t mask:{UINT64_MAX,UINT64_C(255),UINT64_C(0)}) {
            if(sampled && mask!=UINT64_MAX)continue;
            auto got=initial;Stats stats;stats.k_centers=123;stats.suffix_queries=789;const Stats before=stats;
            const auto budget=sampled?std::uint64_t(n)*std::max(1,int(std::pow(double(n),.75))):UINT64_MAX;
            const bool ok=fingerprint_detail::run(index,scores,bits,got,stats,budget,full,seed,mask);
            if(seed==0 && mask==UINT64_MAX){first=got;first_status=ok;}
            else if(ok!=first_status || !same(got,first))fail("fingerprint seed or collision changed output");
            if(!ok){++rejected;if(!same(got,initial)||std::memcmp(&before,&stats,sizeof(stats)))fail("failed fingerprint changed credit or Stats");continue;}
            ++accepted;const auto&expected=full?expected_full:expected_repair;
            for(int owner:owners)for(int bit=0;bit<bits;++bit){++values;const auto at=std::size_t(owner)*bits+bit;
                if(!std::isfinite(got[at]) || (integral?got[at]!=expected[at]:std::abs(got[at]-expected[at])>2e-10*(1+std::abs(expected[at]))))
                    fail(full?"full K fingerprint credit differs from literal":"K fingerprint repair differs from literal");
            }
        }
    }
}
void rejection(std::mt19937_64&random) {
    for(bool plateau:{false,true}) {
        const int n=128;std::vector<std::uint64_t>q(n),k(n);
        if(!plateau)for(int t=0;t<n;++t){q[t]=random()&1;k[t]=random()&1;}
        Index index(q.data(),k.data(),n);std::vector<double>w(n,1),v(n,2),empty(1,3),credit(n,17);
        for(bool full:{false,true})for(std::uint64_t budget:{UINT64_C(0),UINT64_C(128)}) {
            const auto before=credit;Stats stats;stats.k_centers=111;const Stats old=stats;
            if(fingerprint_detail::run(index,Scores{n,1,w.data(),v.data(),empty.data()},1,credit,stats,budget,full,817,0))fail("expected fingerprint budget/cap rejection");
            if(!same(credit,before)||std::memcmp(&old,&stats,sizeof(stats)))fail("fingerprint rejection published partial data");++rejected;
        }
    }
}
}
int main(int argc,char**argv){try {
    const bool quick=argc>1 && std::string(argv[1])=="--quick";std::mt19937_64 random(938211);
    for(int n=1;n<=(quick?4:6);++n)for(std::uint64_t mask=0;mask<(UINT64_C(1)<<(2*n));++mask) {
        std::vector<std::uint64_t>q(n),k(n);for(int t=0;t<n;++t){q[t]=(mask>>t)&1;k[t]=(mask>>(n+t))&1;}check(q,k,1,true,random);
    }
    for(int trial=0;trial<(quick?30:220);++trial) {
        const int n=1+int(random()%16),bits=trial%31==0?64:1+int(random()%5);
        const auto word_mask=bits==64?UINT64_MAX:(UINT64_C(1)<<bits)-1;
        std::vector<std::uint64_t>q(n),k(n);for(int t=0;t<n;++t){q[t]=random()&word_mask;k[t]=random()&word_mask;
            if(trial%4==1){q[t]=k[t]=(t%3)&word_mask;if(t==n/3)k[t]^=1;}
            if(trial%4==2){q[t]=k[t]=0;if(t==0)k[t]=1;}
            if(trial%4==3){q[t]&=1;k[t]&=1;}
        }check(q,k,bits,trial%5!=0,random);
    }
    rejection(random);
    if(!quick)for(int n:{1024,2048}){std::vector<std::uint64_t>q(n),k(n);for(int t=0;t<n;++t){q[t]=random()&1;k[t]=random()&1;}check(q,k,1,true,random,true);}
    std::cout<<"{\"status\":\"PASS\",\"fixtures\":"<<fixtures<<",\"accepted\":"<<accepted<<",\"rejected\":"<<rejected<<",\"values\":"<<values<<"}\n";
}catch(const std::exception&e){std::cerr<<"FAIL fixture "<<fixtures<<": "<<e.what()<<'\n';return 1;}}
