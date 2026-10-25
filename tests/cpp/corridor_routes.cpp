#include "sam_q.h"
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <tuple>
using rosa_exact::SamQRepairs;
using Route=std::pair<int,int>;
struct C{int owner,bit,start,stop,lb,eb;};
uint64_t cases=0,rows=0,qchecks=0,kchecks=0,corridors=0,heapchecks=0,expired=0,runchecks=0;
Route hard(const std::vector<uint64_t>&q,const std::vector<uint64_t>&k,int t,int qu=-1,int ks=-1) {
    Route best{0,-1};
    for(int r=0;r<t;++r){int a=0;while(a<=r && t-a!=qu && r-a!=ks && q[t-a]==k[r-a])++a;if(a)best=std::max(best,Route{a,r});}
    return best;
}
void check(std::vector<uint64_t>q,std::vector<uint64_t>k,int d){
    ++cases;const int n=int(q.size());SamQRepairs sam(q.data(),k.data(),n,d);
    std::vector<C> cs;std::vector<SamQRepairs::Repair> detached;std::set<size_t> once;
    using Entry=std::tuple<int,int,int>;
    std::map<size_t,std::vector<Entry>> heaps;
    std::vector<int> qe(n),ke(n);
    for(int i=n-1;i>=0;--i){qe[i]=(i+1<n&&q[i]==q[i+1])?qe[i+1]:i+1;ke[i]=(i+1<n&&k[i]==k[i+1])?ke[i+1]:i+1;}
    for(int t=0;t<n;++t){
        ++rows;sam.advance(t);sam.extract_singleton_continuations(k.data(),detached);
        if(Route{sam.baseline().length,sam.baseline().end}!=hard(q,k,t))throw std::runtime_error("baseline mismatch");
        for(const auto&z:detached){
            ++corridors;const size_t key=size_t(z.owner)*d+z.bit;
            if(!once.insert(key).second)throw std::runtime_error("Q key detached twice");
            const auto range=sam.occurrence_range(z.short_state);
            if(range.second-range.first!=1 || sam.occurrence_endpoints()[range.first]!=z.end)throw std::runtime_error("not a static singleton");
            if(z.end>=t || z.length<=t-z.owner)throw std::runtime_error("invalid singleton causality or length");
            if(z.short_state!=sam.state_for_length(z.state,t-z.owner+1))throw std::runtime_error("detached short-state mismatch");
            int stop=t;while(stop+1<n&&q[stop+1]==k[z.end+stop-t+1])++stop;
            int x=t+1,y=z.end+1;while(x<n&&q[x]==k[y]){int a=std::min(qe[x]-x,ke[y]-y);if(a<=0)throw std::runtime_error("zero run step");x+=a;y+=a;}
            if(x-1!=stop || stop<=t)throw std::runtime_error("run LCE lifetime mismatch");++runchecks;
            const C c{z.owner,z.bit,t,stop,z.length-t,z.end-t};cs.push_back(c);
            const int sk=c.owner+c.eb;if(sk<0||sk>=t)throw std::runtime_error("K owner out of range");
            auto&h=heaps[size_t(sk)*d+c.bit];h.emplace_back(c.lb,c.eb,c.stop);std::push_heap(h.begin(),h.end());
        }
        std::vector<Route> qr(size_t(t+1)*d,{0,-1}),kr(size_t(t+1)*d,{0,-1});
        int last_owner=-1;
        for(const auto&z:sam.repairs()){
            if(z.owner<last_owner)throw std::runtime_error("active owner order changed");last_owner=z.owner;
            const size_t key=size_t(z.owner)*d+z.bit;
            if(qr[key].first)throw std::runtime_error("duplicate live Q key");
            qr[key]={z.length,z.end};
            sam.for_each_causal_occurrence(z.short_state,[&](int r){
                const int a=t-z.owner,sk=r-a;if(sk<0)throw std::runtime_error("negative K owner");
                int b=0;while(b<sk && q[z.owner-1-b]==k[sk-1-b])++b;
                const Route candidate{a+1+b,r};auto&v=kr[size_t(sk)*d+z.bit];v=std::max(v,candidate);
            });
        }
        std::map<size_t,Route> expected_top;
        for(const auto&c:cs)if(c.stop>=t){
            if(c.start>t)throw std::runtime_error("future corridor inserted early");
            const size_t key=size_t(c.owner)*d+c.bit;
            if(qr[key].first)throw std::runtime_error("live/corridor duplicate Q key");
            qr[key]={t+c.lb,t+c.eb};
            const size_t kk=size_t(c.owner+c.eb)*d+c.bit;
            auto it=expected_top.find(kk);if(it==expected_top.end())expected_top[kk]=qr[key];else it->second=std::max(it->second,qr[key]);
        }
        for(auto&kv:heaps){
            auto&h=kv.second;while(!h.empty()&&std::get<2>(h.front())<t){std::pop_heap(h.begin(),h.end());h.pop_back();++expired;}
            Route top{0,-1};if(!h.empty())top={t+std::get<0>(h.front()),t+std::get<1>(h.front())};
            auto it=expected_top.find(kv.first);Route want=it==expected_top.end()?Route{0,-1}:it->second;
            if(top!=want)throw std::runtime_error("heap winner/expiry mismatch");++heapchecks;
            if(top.first)kr[kv.first]=std::max(kr[kv.first],top);
        }
        for(int u=0;u<=t;++u)for(int bit=0;bit<d;++bit){
            auto qf=q;qf[u]^=uint64_t(1)<<bit;
            const auto truth=hard(qf,k,t),deletion=hard(q,k,t,u);
            const auto repair=qr[size_t(u)*d+bit];++qchecks;
            if(std::max(repair,deletion)!=truth || (repair.first>0)!=(truth.first>t-u))throw std::runtime_error("Q lifetime or route mismatch");
        }
        for(int s=0;s<t;++s)for(int bit=0;bit<d;++bit){
            auto kf=k;kf[s]^=uint64_t(1)<<bit;
            const auto truth=hard(q,kf,t),deletion=hard(q,k,t,-1,s);++kchecks;
            if(std::max(kr[size_t(s)*d+bit],deletion)!=truth)throw std::runtime_error("K union/heap route mismatch");
        }
    }
}
int main(){
    for(int n=0;n<=6;++n)for(uint64_t m=0;m<(uint64_t(1)<<(2*n));++m){std::vector<uint64_t>q(n),k(n);for(int i=0;i<n;++i){q[i]=(m>>i)&1;k[i]=(m>>(n+i))&1;}check(q,k,1);}
    std::mt19937_64 gen(183572911);
    for(int d:{1,2,3,8,32,64})for(int z=0;z<80;++z){int n=5+int(gen()%20);std::vector<uint64_t>q(n),k(n);uint64_t mask=d==64?~uint64_t(0):(uint64_t(1)<<d)-1;for(int i=0;i<n;++i){q[i]=gen()&mask;k[i]=gen()&mask;}if(z%2==0)k=q;if(z%3==0){for(int i=0;i<n;++i){q[i]=0;k[i]=0;}k[0]=uint64_t(1)<<(d-1);if(z%6==0)q[n/2]=uint64_t(1)<<(d-1);}check(q,k,d);}
    for(int n:{31,64,127}){std::vector<uint64_t>q(n,0),k(n,0);k[0]=1;check(q,k,1);q[n/2]=1;check(q,k,1);for(int i=0;i<n;++i){q[i]=i%4;k[i]=q[i];}k[n/2]^=1;check(q,k,2);}
    std::cout<<"{\"status\":\"PASS\",\"inputs\":"<<cases<<",\"baseline_rows\":"<<rows<<",\"Q_flip_routes\":"<<qchecks<<",\"K_flip_routes\":"<<kchecks<<",\"static_singleton_corridors\":"<<corridors<<",\"run_LCE_lifetimes\":"<<runchecks<<",\"heap_winner_checks\":"<<heapchecks<<",\"expired_heap_items\":"<<expired<<"}\n";
}
