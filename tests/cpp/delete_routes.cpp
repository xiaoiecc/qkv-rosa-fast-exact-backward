#include "delete_indexed.h"
#include <iostream>
#include <random>
#include <sstream>
#include <utility>
using rosa_exact::DeleteRoute;
using rosa_exact::IndexedDeleteRow;
static bool eq(DeleteRoute a,DeleteRoute b) {return a.length==b.length && a.end==b.end;}
static DeleteRoute best(DeleteRoute a,DeleteRoute b) {if (!b.length)return a;return b.length>a.length || (b.length==a.length && b.end>a.end) ? b : a;}
struct MockIndex {
    int n,t;std::vector<int> f;
    int size()const{return n;}
    int lcs(int,int r)const{return f[r];}
    int latest(int,int len)const{for(int r=t-1;r>=0;--r)if(f[r]>=len)return r;return -1;}
    std::pair<int,int> interval(int,int len)const{return {len,len+1};}
    int succ(int len,int,int start)const{for(int r=start;r<t;++r)if(f[r]>=len)return r;return -1;}
    DeleteRoute prefix_best(int,int lim)const{DeleteRoute z;for(int r=0;r<lim;++r)z=best(z,{f[r],r});return z;}
};
static uint64_t inputs=0,rows=0,route_checks=0,range_checks=0,numeric_checks=0,ar=0,kr=0,oranges=0;
static void check(const std::vector<uint64_t>& q,const std::vector<uint64_t>& k,std::mt19937& rng,bool all_intervals=true) {
  const int n=q.size(),d=4;
  IndexedDeleteRow del(q.data(),k.data(),n);
  std::vector<long long> payload(n*d),gradient(n*d),base(n),want(n),diff(n+1),vdiff((n+1)*d*2),qdiff(n+1),qwant(n);
  for(auto&x:payload)x=int(rng()%21)-10;
  long long emb[d];for(int j=0;j<d;++j)emb[j]=n?payload[j]:j+1;
  std::vector<unsigned char> bits(n*d);
  for(int i=0;i<n*d;++i){bits[i]=(payload[i]%2)!=0;payload[i]=(bits[i]?1:-1)*emb[i%d];}
  for(auto&x:gradient)x=int(rng()%21)-10;
  long long empty[d];for(auto&x:empty)x=int(rng()%21)-10;
  for(int t=0;t<n;++t) {
    MockIndex ix{n,t,std::vector<int>(t)};
    for(int r=0;r<t;++r)while(ix.f[r]<=r && q[t-ix.f[r]]==k[r-ix.f[r]])++ix.f[r];
    auto b=ix.prefix_best(t,t);del.reset(ix,t,b.length,b.end);
    if(!eq(b,del.baseline()))throw std::runtime_error("baseline");
    auto score=[&](int r){long long z=0;for(int j=0;j<d;++j)z+=gradient[t*d+j]*(r<0?empty[j]:payload[(r+1)*d+j]);return z;};
    const auto bs=score(b.end);
    std::vector<DeleteRoute> expected(n+2,b);
    for(int s=0;s<t;++s) {
      DeleteRoute z;
      for(int r=0;r<t;++r)z=best(z,{r<s?ix.f[r]:std::min(ix.f[r],r-s),r});
      if(!eq(z,del.kroute(s))) {
        std::cerr<<"input "<<inputs<<" row "<<t<<" owner "<<s<<" expected "<<z.length<<","<<z.end<<" actual "<<del.kroute(s).length<<","<<del.kroute(s).end<<"\n";
        std::cerr<<"q ";for(auto x:q)std::cerr<<x<<",";std::cerr<<"\nk ";for(auto x:k)std::cerr<<x<<",";std::cerr<<"\n";
        throw std::runtime_error("K route");
      }
      expected[s+1]=z;want[s]+=score(z.end)-bs;++route_checks;
    }
    auto qcursor=del.qcursor();
    for(int u=0;u<=t;++u) {
      DeleteRoute z;for(int r=0;r<t;++r)z=best(z,{std::min(ix.f[r],t-u),r});
      if(!eq(z,del.qroute(u)))throw std::runtime_error("Q route");
      if(!eq(z,del.qroute_monotone(u,qcursor)))throw std::runtime_error("Q monotone route");
      if(!eq(z,del.qroute_monotone(u,qcursor)))throw std::runtime_error("Q repeated monotone route");
      qwant[u]+=score(z.end)-bs;++route_checks;
    }
    for(int lo=-1;lo<=t+1;++lo) {
      if(!all_intervals && lo!=-1 && lo!=t/2 && lo!=t)continue;
      DeleteRoute z=b;
      for(int hi=lo;hi<=t+1;++hi) {
        const auto x=expected[hi+1];if(x.length<z.length || (x.length==z.length && x.end<z.end))z=x;
        if(!eq(z,del.min_k_delete_range(lo,hi)))throw std::runtime_error("K valley minimum");
        ++range_checks;
      }
    }
    for(auto z:del.qranges()){const auto v=score(z.end)-bs;qdiff[z.lo]+=v;qdiff[z.hi+1]-=v;}
    for(auto z:del.kranges()){const auto v=score(z.end)-bs;diff[z.lo]+=v;diff[z.hi+1]-=v;}
    for(auto z:del.kowner_ranges()){
      for(int j=0;j<d;++j){const int bucket=2*j+bits[(b.end+1)*d+j];
        vdiff[z.lo*d*2+bucket]+=gradient[t*d+j];vdiff[(z.hi+1)*d*2+bucket]-=gradient[t*d+j];}
    }
    ar+=del.a_record_count();kr+=del.k_route_run_count();oranges+=del.kowner_ranges().size();++rows;
  }
  long long scalar=0,qs=0,vec[2*d]={};
  for(int s=0;s<n;++s){scalar+=diff[s];qs+=qdiff[s];long long value=scalar;
    for(int j=0;j<d;++j){vec[2*j]+=vdiff[s*d*2+2*j];vec[2*j+1]+=vdiff[s*d*2+2*j+1];
      const bool bit=bits[s*d+j];value+=(bit?2:-2)*emb[j]*vec[2*j+int(!bit)];}
    if(value!=want[s] || qs!=qwant[s])throw std::runtime_error("numeric contraction");numeric_checks+=2;}
  ++inputs;
}
int main(){try{
  std::mt19937 rng(20261018);
  for(int n=0;n<=7;++n)for(uint64_t mask=0;mask<(uint64_t(1)<<(2*n));++mask){std::vector<uint64_t>q(n),k(n);for(int i=0;i<n;++i){q[i]=(mask>>i)&1;k[i]=(mask>>(n+i))&1;}check(q,k,rng);}
  for(int trial=0;trial<1200;++trial){int n=1+rng()%96;std::vector<uint64_t>q(n),k(n);const int sigma=1+(rng()%8);
    for(int i=0;i<n;++i){q[i]=rng()%sigma;k[i]=rng()%sigma;if(trial%3==0){if(i && rng()%8)q[i]=q[i-1];if(i&&rng()%8)k[i]=k[i-1];}}
    if(trial%6==0)k=q;if(trial%11==0){q.assign(n,0);k=q;k[0]=1;}if(trial%13==0){for(int i=0;i<n;++i)q[i]=k[i]=i%4;if(n>2)k[n/2]^=1;}
    check(q,k,rng,false);
  }
  for(int n:{128,256,512}){std::vector<uint64_t>q(n,0),k(n,0);check(q,k,rng,false);k[0]=1;check(q,k,rng,false);for(int i=0;i<n;++i){q[i]=k[i]=(i/17)%5;}k[n/2]^=1;check(q,k,rng,false);}
  std::cout<<"{\"status\":\"PASS\",\"numeric_mode\":\"baseline_bit_two_buckets\",\"header\":\"conv_rosa_transformer/csrc/delete_indexed.h\",\"inputs\":"<<inputs<<",\"rows\":"<<rows<<",\"route_checks\":"<<route_checks<<",\"range_checks\":"<<range_checks<<",\"numeric_checks\":"<<numeric_checks<<",\"a_runs\":"<<ar<<",\"k_runs\":"<<kr<<",\"owner_ranges\":"<<oranges<<"}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
