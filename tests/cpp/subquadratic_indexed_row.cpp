// Independent literal oracle for full selected-row repairs and their exact
// one-symbol residual, including quadratic scores around the full baseline.
// Build: c++ -O2 -std=c++17 -ffp-contract=off tests/cpp/subquadratic_indexed_row.cpp -o /tmp/rosa-indexed-row
#include "../../conv_rosa_transformer/csrc/approx_indexed_row.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

namespace {
struct Route {int length=0,end=-1;};
bool higher(Route a,Route b){return a.length>b.length || (a.length==b.length&&a.end>b.end);}
std::uint64_t literal_routes=0,cases=0,components=0,selected_rows=0;
double worst_error=0.;

Route literal(const std::vector<std::uint64_t>& q,const std::vector<std::uint64_t>& k,
              int t,int forbidden_q=-1,int forbidden_k=-1,int cap=2147483647) {
    ++literal_routes;Route best;
    for(int r=0;r<t;++r) {
        int length=0;
        while(length<cap && length<=r && t-length!=forbidden_q && r-length!=forbidden_k
              && q[t-length]==k[r-length])++length;
        const Route route{length,length?r:-1};if(higher(route,best))best=route;
    }
    return best;
}
struct Strings {const std::vector<std::uint64_t>& q;const std::vector<std::uint64_t>& k;};
int literal_lcs(const void* context,int t,int r) {
    const auto& text=*static_cast<const Strings*>(context);int length=0;
    while(t>=0 && r>=0 && text.q[t]==text.k[r]){++length;--t;--r;}
    return length;
}

template<bool Quadratic>
void check(std::vector<std::uint64_t> q,std::vector<std::uint64_t> k,int d,
           std::mt19937_64& rng,int layout,bool fractional) {
    const int n=int(q.size());const std::size_t nd=std::size_t(n)*d;
    auto value=[&](){return fractional?std::uniform_real_distribution<double>(-2.,2.)(rng):double(int(rng()%7)-3);};
    std::vector<std::uint8_t> v(nd);for(auto& x:v)x=rng()%2;
    std::vector<double> g(nd),h(nd),empty(d),scale(d),initial(nd);
    for(double& x:g)x=value();for(double& x:h)x=std::abs(value());
    for(double& x:empty)x=value();for(double& x:scale)x=value();for(double& x:initial)x=value();
    // Curvature-only rows must not disappear from the selected-row kernel.
    if constexpr(Quadratic)for(int t=0;t<n;t+=3)std::fill_n(g.data()+std::size_t(t)*d,d,0.);
    if(n>4)for(int j=0;j<d;++j){g[std::size_t(1)*d+j]=0.;h[std::size_t(1)*d+j]=0.;}
    std::vector<std::int32_t> lengths(n),ends(n),selected;
    std::vector<double> weights;
    for(int t=0;t<n;++t) {
        const Route route=literal(q,k,t);lengths[t]=route.length;ends[t]=route.end;
        if(!layout || t%3!=1) {
            selected.push_back(t);
            weights.push_back(layout?(t%7==0?0.:.5+double(t%5)/3.):1.);
        }
    }
    std::vector<double> expected[2][2];
    for(auto& mode:expected)for(auto& side:mode)side=initial;
    auto payload=[&](int end,int channel){return end<0?empty[channel]:
        (2*int(v[std::size_t(end+1)*d+channel])-1)*scale[channel];};
    for(std::size_t row=0;row<selected.size();++row) {
        const int t=selected[row];const double weight=weights[row];
        std::vector<double> costs(std::size_t(n)+1,0.);
        for(int end=-1;end<n-1;++end)for(int j=0;j<d;++j) {
            const double delta=payload(end,j)-payload(ends[t],j);const std::size_t at=std::size_t(t)*d+j;
            costs[end+1]+=g[at]*delta+(Quadratic?.5*h[at]*delta*delta:0.);
        }
        for(int owner=0;owner<n;++owner)for(int bit=0;bit<d;++bit)for(int side=0;side<2;++side) {
            const int qs=side?-1:owner,ks=side?owner:-1;
            const Route deleted=literal(q,k,t,qs,ks),gram_deleted=literal(q,k,t,qs,ks,1);
            auto& word=side?k[owner]:q[owner];word^=std::uint64_t(1)<<bit;
            const Route actual=literal(q,k,t),gram_actual=literal(q,k,t,-1,-1,1);
            word^=std::uint64_t(1)<<bit;
            const double full=higher(actual,deleted)?costs[actual.end+1]-costs[deleted.end+1]:0.;
            const double gram=higher(gram_actual,gram_deleted)?costs[gram_actual.end+1]-costs[gram_deleted.end+1]:0.;
            const std::size_t key=std::size_t(owner)*d+bit;
            expected[0][side][key]+=weight*full;
            expected[1][side][key]+=weight*(full-gram);
        }
    }
    const rosa_random::OnegramPlan plan(q.data(),k.data(),n,d,717);const Strings text{q,k};
    for(int subtract=0;subtract<2;++subtract) {
        auto cq=initial,ck=initial;std::uint64_t stats[16]={};std::size_t workspace=0;
        const int status=rosa_random::indexed_row_repairs<Quadratic>(plan,v.data(),g.data(),
            Quadratic?h.data():nullptr,empty.data(),scale.data(),selected.data(),layout?weights.data():nullptr,
            int(selected.size()),lengths.data(),ends.data(),cq.data(),ck.data(),stats,bool(subtract),
            literal_lcs,&text,&workspace);
        if(status){std::cerr<<"indexed row returned "<<status<<'\n';std::exit(1);}
        for(int side=0;side<2;++side)for(std::size_t at=0;at<nd;++at) {
            const double got=side?ck[at]:cq[at],want=expected[subtract][side][at];
            const double error=std::abs(got-want);worst_error=std::max(worst_error,error);++components;
            if(error>2e-10*(1+std::abs(want))) {
                std::cerr<<"indexed row mismatch case="<<cases<<" n="<<n<<" d="<<d<<" quadratic="<<Quadratic
                    <<" subtract="<<subtract<<" side="<<side<<" owner="<<at/d<<" bit="<<at%d
                    <<" got="<<got<<" want="<<want<<'\n';std::exit(1);
            }
        }
        ++cases;selected_rows+=selected.size();
    }
}

void both(const std::vector<std::uint64_t>&q,const std::vector<std::uint64_t>&k,
          int d,std::mt19937_64& rng,int layout=0,bool fractional=false) {
    check<false>(q,k,d,rng,layout,fractional);check<true>(q,k,d,rng,layout,fractional);
}
} // namespace

int main(int argc,char** argv) {
    const bool quick=argc>1 && std::string(argv[1])=="--quick";
    const auto start=std::chrono::steady_clock::now();std::mt19937_64 rng(2026110911);
    // Exhaustive binary strings cover competing positive-a and zero-a
    // repairs for the same owner, endpoint ties, and both subtraction modes.
    for(int n=1;n<=(quick?3:5);++n)for(std::uint64_t pattern=0;pattern<(std::uint64_t(1)<<(2*n));++pattern) {
        std::vector<std::uint64_t>q(n),k(n);
        for(int t=0;t<n;++t){q[t]=(pattern>>t)&1;k[t]=(pattern>>(n+t))&1;}
        both(q,k,1,rng);
    }
    for(int kind=0;kind<3;++kind)for(int layout=0;layout<2;++layout) {
        std::vector<std::uint64_t>q(12,7),k(12,3);q[9]=2;q[10]=0;
        // L=0: older left-extended neighbor beats latest one-symbol neighbor.
        k[0]=2;k[1]=1;k[5]=1;
        if(kind==1){k[3]=0;k[4]=2;k[5]=1;k[7]=0;k[8]=2;k[9]=1;}
        // L=2: original end1 differs from onegram end5. The b=0 K tail
        // after end5 still requires negative onegram repair subtraction.
        if(kind==2){k[1]=0;k[5]=0;k[8]=1;}
        if(literal(q,k,10).length!=kind){std::cerr<<"invalid targeted fixture\n";return 1;}
        both(q,k,3,rng,layout,true);
    }
    for(int rep=0;rep<(quick?12:96);++rep) {
        const int n=2+int(rng()%31),d=1+int(rng()%4);std::vector<std::uint64_t>q(n),k(n);
        for(int t=0;t<n;++t){q[t]=rng()%(1<<d);k[t]=rng()%(1<<d);}
        if(rep%7==0){std::fill(q.begin(),q.end(),0);std::fill(k.begin(),k.end(),0);k[n/2]=1;}
        both(q,k,d,rng,rep%2,true);
    }
    for(int rep=0;rep<(quick?2:8);++rep) {
        const int n=5+rep%5;std::vector<std::uint64_t>q(n),k(n);
        for(int t=0;t<n;++t){q[t]=UINT64_MAX^((rng()%2)<<63);k[t]=UINT64_MAX^((rng()%2)<<63);}
        both(q,k,64,rng,rep%2,true);
    }
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout<<"{\"status\":\"PASS\",\"cases\":"<<cases<<",\"selected_rows\":"<<selected_rows
             <<",\"credit_components\":"<<components<<",\"literal_routes\":"<<literal_routes
             <<",\"max_absolute_error\":"<<worst_error<<",\"seconds\":"<<seconds<<"}\n";
}
