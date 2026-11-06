// Literal edited-string verification of the indexed repair engine.
// The O(n^2) tables below are reference outputs only, never native engine state.
#include "../../conv_rosa_transformer/csrc/subquadratic/repair.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using rosa_subq::Index;
using rosa_subq::Route;
using rosa_subq::Scores;
using rosa_subq::Stats;

static std::uint64_t checked_cases=0,checked_rows=0,checked_light_boundaries=0;

static void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}

static Route literal(const std::vector<std::uint64_t>& q,const std::vector<std::uint64_t>& k,
    int t,int query_edit=-1,int key_edit=-1,int bit=0,int deleted_key=-1) {
    Route best;
    for(int r=0;r<t;++r) {
        int length=0;
        while(length<=r && length<=t) {
            const int a=t-length,b=r-length;
            if(b==deleted_key)break;
            const auto x=q[a]^(a==query_edit?(std::uint64_t(1)<<bit):0);
            const auto y=k[b]^(b==key_edit?(std::uint64_t(1)<<bit):0);
            if(x!=y)break;
            ++length;
        }
        if(length)best=std::max(best,Route{length,r});
    }
    return best;
}

static int static_count(const std::vector<std::uint64_t>& q,const std::vector<std::uint64_t>& k,
    int owner,int bit,int age) {
    const int n=int(q.size());int count=0;
    for(int source=0;source<n;++source)if(k[source]==(q[owner]^(std::uint64_t(1)<<bit))) {
        bool equal=source+age<n;
        for(int j=1;equal && j<=age;++j)equal=q[owner+j]==k[source+j];
        count+=equal;
    }
    return count;
}

static void check_q(const Index& index,int bits,int threshold,const Scores& scores,bool boundary_check) {
    const int n=index.n;const std::size_t outputs=std::size_t(n)*bits;
    std::vector<double> heavy(outputs,0),actual(outputs,0),expected(outputs,0),heavy_expected(outputs,0);
    std::vector<rosa_subq::Segment> segments;
    std::vector<rosa_subq::BitRange> unions;
    Stats stats;
    rosa_subq::q_repair(index,scores,bits,threshold,heavy,segments,unions,stats);
    actual=heavy;
    std::vector<unsigned char> light_seen(outputs*n,0),union_seen(outputs,0);
    for(const auto& segment:segments) {
        require(segment.lo>=segment.owner && segment.hi<n && segment.lo<=segment.hi,"Q light interval bounds");
        for(int t=segment.lo;t<=segment.hi;++t) {
            const std::size_t output=std::size_t(segment.owner)*bits+segment.bit;
            require(!light_seen[output*n+t],"overlapping Q light intervals");
            light_seen[output*n+t]=1;
            const Route route=literal(index.q,index.k,t,segment.owner,-1,segment.bit);
            require(route==Route{t+segment.alpha,t+segment.beta},"Q light route mismatch");
            require(route.length>t-segment.owner,"Q light route does not cross owner");
            actual[output]+=scores.score(t,route.end);++checked_rows;
        }
    }
    rosa_subq::repair_detail::FrontierIndex tree(index);
    std::uint64_t heavy_rows=0;
    for(int owner=0;owner<n;++owner)for(int bit=0;bit<bits;++bit) {
        const std::size_t output=std::size_t(owner)*bits+bit;
        const int group=tree.group(index.q[owner]^(std::uint64_t(1)<<bit));
        const int light_age=group<0?0:rosa_subq::repair_detail::first_light_age(index.forward_suffixes(),
            index.forward_suffixes().rank[owner+1],tree.points,tree.whole(group),threshold);
        for(int t=owner;t<n;++t) {
            const int age=t-owner;
            if(boundary_check) {
                require((static_count(index.q,index.k,owner,bit,age)>threshold)==(age<light_age),
                    "Q first-light order statistic mismatch");
                ++checked_light_boundaries;
            }
            const Route route=literal(index.q,index.k,t,owner,-1,bit);
            if(route.length>age) {
                expected[output]+=scores.score(t,route.end);
                if(age<light_age){heavy_expected[output]+=scores.score(t,route.end);++heavy_rows;}
                else require(light_seen[output*n+t],"missing Q light row");
            } else require(!light_seen[output*n+t],"spurious Q light row");
            ++checked_rows;
        }
    }
    for(const auto& range:unions) {
        const std::size_t output=std::size_t(range.owner)*bits+range.bit;
        require(!union_seen[output],"duplicate Q existence union");union_seen[output]=1;
        require(range.lo==range.owner && range.hi<n && range.hi>=range.lo,"Q existence bounds");
        for(int t=range.owner;t<n;++t) {
            const bool exists=literal(index.q,index.k,t,range.owner,-1,range.bit).length>t-range.owner;
            require(exists==(t<=range.hi),"Q existence union mismatch");
        }
    }
    for(int owner=0;owner<n;++owner)for(int bit=0;bit<bits;++bit) {
        const std::size_t output=std::size_t(owner)*bits+bit;
        if(!union_seen[output])for(int t=owner;t<n;++t)
            require(literal(index.q,index.k,t,owner,-1,bit).length<=t-owner,"missing Q existence union");
        require(actual[output]==expected[output],"Q complete positive score mismatch");
        require(heavy[output]==heavy_expected[output],"Q heavy positive score mismatch");
    }
    require(stats.q_heavy_rows==heavy_rows,"Q heavy row counter mismatch");
    require(stats.q_heavy_rows<=std::uint64_t(n)*n/std::uint64_t(threshold+1),"Q heavy count bound");
    require(stats.q_light_sources<=std::uint64_t(n)*bits*threshold,"Q light source bound");
    require(segments.size()<=std::size_t(n)*bits*threshold,"Q light interval bound");
    require(stats.q_segments==segments.size(),"Q segment counter mismatch");
}

static void check_k(const Index& index,int bits,int anchor,const Scores& scores) {
    const int n=index.n;const int cutoff=4*anchor;
    Stats stats;const auto segments=rosa_subq::long_k(index,anchor,stats);
    const std::size_t outputs=std::size_t(n)*bits;
    std::vector<unsigned char> seen(outputs*n,0);
    std::vector<double> actual(outputs,0),expected(outputs,0);
    for(const auto& segment:segments) {
        // The low-level API enumerates every packed one-bit label. Callers
        // with a narrower requested bit domain discard labels outside it.
        if(segment.bit>=bits)continue;
        require(segment.owner>=0 && segment.owner<n && segment.lo>=0 && segment.hi<n,"K interval bounds");
        for(int t=segment.lo;t<=segment.hi;++t) {
            const std::size_t output=std::size_t(segment.owner)*bits+segment.bit;
            require(!seen[output*n+t],"overlapping K long intervals");seen[output*n+t]=1;
            const Route flipped=literal(index.q,index.k,t,-1,segment.owner,segment.bit);
            const Route deleted=literal(index.q,index.k,t,-1,-1,0,segment.owner);
            require(flipped==Route{t+segment.alpha,t+segment.beta},"K long route mismatch");
            require(flipped.length>cutoff && flipped>deleted,"spurious K long route");
            actual[output]+=scores.score(t,flipped.end)-scores.score(t,deleted.end);++checked_rows;
        }
    }
    for(int owner=0;owner<n;++owner)for(int bit=0;bit<bits;++bit) {
        const std::size_t output=std::size_t(owner)*bits+bit;
        for(int t=0;t<n;++t) {
            const Route flipped=literal(index.q,index.k,t,-1,owner,bit);
            const Route deleted=literal(index.q,index.k,t,-1,-1,0,owner);
            const bool wanted=flipped.length>cutoff && flipped>deleted;
            require(wanted==bool(seen[output*n+t]),"missing K long interval row");
            if(wanted)expected[output]+=scores.score(t,flipped.end)-scores.score(t,deleted.end);
            ++checked_rows;
        }
        require(actual[output]==expected[output],"K complete long score mismatch");
    }
    require(stats.k_segments==segments.size(),"K segment counter mismatch");
}

static void one_case(const std::vector<std::uint64_t>& q,const std::vector<std::uint64_t>& k,
    int bits,std::mt19937_64& random,bool all_thresholds=false,bool boundary_check=false,int d=3) {
    const int n=int(q.size());Index index(q.data(),k.data(),n);
    std::vector<double> g(std::size_t(n)*d),p(g.size()),e(d);
    for(auto& x:g)x=int(random()%9)-4;
    for(auto& x:p)x=int(random()%11)-5;
    for(auto& x:e)x=int(random()%7)-3;
    const Scores scores{n,d,g.data(),p.data(),e.data()};
    std::vector<int> thresholds=all_thresholds?std::vector<int>{1,2,n}:std::vector<int>{1+int(random()%n)};
    std::sort(thresholds.begin(),thresholds.end());thresholds.erase(std::unique(thresholds.begin(),thresholds.end()),thresholds.end());
    for(int threshold:thresholds)check_q(index,bits,threshold,scores,boundary_check);
    std::vector<int> anchors=all_thresholds?std::vector<int>{1,2,std::max(1,n/4)}:std::vector<int>{1+int(random()%std::max(1,n/3))};
    std::sort(anchors.begin(),anchors.end());anchors.erase(std::unique(anchors.begin(),anchors.end()),anchors.end());
    for(int anchor:anchors)check_k(index,bits,anchor,scores);
    ++checked_cases;
}

int main(int argc,char** argv) {
    try {
        const bool quick=argc>1 && std::string(argv[1])=="--quick";
        std::mt19937_64 random(2026110607);
        const auto start=std::chrono::steady_clock::now();
        const int exhaustive=quick?4:6;
        for(int n=1;n<=exhaustive;++n) {
            const unsigned total=1u<<(2*n),mask=(1u<<n)-1;
            for(unsigned code=0;code<total;++code) {
                std::vector<std::uint64_t> q(n),k(n);
                for(int t=0;t<n;++t){q[t]=(code>>t)&1;k[t]=((code>>n)&mask)>>t&1;}
                one_case(q,k,1,random,true,true);
            }
        }
        for(int it=0;it<(quick?35:220);++it) {
            const int n=1+int(random()%64),bits=1+int(random()%5);
            std::vector<std::uint64_t> q(n),k(n);const std::uint64_t mask=(std::uint64_t(1)<<bits)-1;
            for(int t=0;t<n;++t){q[t]=random()&mask;k[t]=random()&mask;}
            if(it%4==0)for(int t=0;t<n;++t){q[t]=t%2;k[t]=0;}
            if(it%4==1) { const int period=1+int(random()%5);for(int t=period;t<n;++t){q[t]=q[t-period];k[t]=q[t];} }
            if(it%4==2) { std::fill(q.begin(),q.end(),0);std::fill(k.begin(),k.end(),0);k[random()%n]=1; }
            one_case(q,k,bits,random,true,true,1+int(random()%6));
        }
        for(int it=0;it<(quick?5:25);++it) {
            const int n=1+int(random()%18);std::vector<std::uint64_t> q(n),k(n);
            for(int t=0;t<n;++t){q[t]=(random()%2)<<63;k[t]=(random()%2)<<63;}
            if(it%3==0)for(int t=0;t<n;++t){q[t]=(std::uint64_t(t%2)<<63)|UINT64_C(0x7fffffffffffffff);k[t]=UINT64_C(0x7fffffffffffffff);}
            one_case(q,k,64,random,true,true,it%2?1:65);
        }
        for(int n:{96,128}) {
            std::vector<std::uint64_t> q(n,0),k(n,0);k[n/4]=1;k[3*n/4]=1;
            one_case(q,k,1,random,false,false);
        }
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::cout<<"{\"status\":\"PASS\",\"cases\":"<<checked_cases<<",\"literal_rows\":"<<checked_rows
            <<",\"light_boundary_checks\":"<<checked_light_boundaries<<",\"seconds\":"<<seconds<<"}\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"FAIL after "<<checked_cases<<" cases: "<<error.what()<<"\n";return 1;
    }
}
