#pragma once
// Batched arbitrary interval correlations with bounded, streamed prefix storage.
// Exact algebra over reals. FFT introduces roundoff; no sampling and no truncation.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <vector>
#include "../exact_expansion.h"
namespace rosa_subq {
struct CorrelationQuery { int lo,hi,shift; };
inline int correlation_block(int n,int d,const std::vector<CorrelationQuery>& queries) {
 // Count the work actually requested at each power-of-two block size. A
 // shift touches one or two payload-block diagonals. Sorting once by time
 // lets their interval unions be counted in O(n+M) per candidate block.
 if(n<0 || d<1 || n>(std::numeric_limits<int>::max()-2)/4)
  throw std::invalid_argument("invalid correlation planner dimensions");
 if(queries.size()>size_t(std::numeric_limits<int>::max()))
  throw std::length_error("too many correlation queries");
 uint64_t rows=0;int longest=0;
 for(const auto&q:queries) {
  if(q.lo<0 || q.hi<q.lo || q.hi>=n || int64_t(q.lo)+q.shift<0 || int64_t(q.hi)+q.shift>=n)
   throw std::invalid_argument("invalid correlation planner query");
  const int length=q.hi-q.lo+1;rows+=uint64_t(length);longest=std::max(longest,length);
 }
 if(queries.empty())return 1;
 if(rows<=4*uint64_t(queries.size())) {
  int block=1;while(block<=longest)block*=2;return block;
 }
 std::vector<int> order(queries.size());
 for(int i=0;i<int(order.size());++i)order[i]=i;
 std::sort(order.begin(),order.end(),[&](int a,int b){return queries[a].lo<queries[b].lo;});
 std::vector<int> diagonal_end(std::max(2*n-1,0)),shift_end(diagonal_end.size()),payload_cover(std::size_t(n)+1);
 int chosen=1;long double best=std::numeric_limits<long double>::infinity();
 for(int block=1;;block*=2) {
  const int groups=(n+block-1)/block,width=2*block;
  std::fill(diagonal_end.begin(),diagonal_end.begin()+2*groups-1,0);
  std::fill(shift_end.begin(),shift_end.end(),0);
  std::fill(payload_cover.begin(),payload_cover.begin()+groups+1,0);
  uint64_t tails=0,pairs=0,prefixes=0,gradient_blocks=0;int gradient_end=0;
  for(int id:order) {
   const auto&q=queries[id];const int first=(q.lo+block-1)/block,after=(q.hi+1)/block;
   if(first>=after){tails+=uint64_t(q.hi-q.lo+1);continue;}
   tails+=uint64_t(first*block-q.lo)+uint64_t(q.hi+1-after*block);
   gradient_blocks+=std::max(0,after-std::max(first,gradient_end));gradient_end=std::max(gradient_end,after);
   int lower=q.shift/block;if(q.shift<0 && q.shift%block)--lower;
   const bool split=q.shift!=lower*block;
   ++payload_cover[std::max(0,first+lower)];--payload_cover[std::min(groups,after+lower+int(split))];
   auto diagonal=[&](int delta) {
    const int lo=std::max(first,-delta),hi=std::min(after,groups-delta);
    if(lo>=hi)return;
    int& end=diagonal_end[delta+groups-1];pairs+=std::max(0,hi-std::max(lo,end));end=std::max(end,hi);
   };
   diagonal(lower);if(split)diagonal(lower+1);
   int& end=shift_end[q.shift+n-1];
   prefixes+=uint64_t(std::max(0,after-std::max(first,end)))*(split?2:1);end=std::max(end,after);
  }
  // A row also performs an exact expansion update; treating it as only d
  // products systematically overprices FFTs relative to long scalar fringes.
  long double cost=static_cast<long double>(d+12)*tails;
  if(pairs) {
   int coverage=0,payload_blocks=0;
   for(int j=0;j<groups;++j){coverage+=payload_cover[j];payload_blocks+=coverage>0;}
   const uint64_t transforms=(uint64_t(payload_blocks)+gradient_blocks)*uint64_t(d)+pairs;
   cost+=6.L*transforms*width*std::log2(width)+5.L*pairs*d*width+12.L*prefixes;
  }
  if(cost<best){best=cost;chosen=block;}
  if(block>longest)break;
 }
 return chosen;
}
struct CorrelationStats {
 uint64_t transforms=0,block_pairs=0,pointwise_products=0,tail_products=0,prefix_adds=0,workspace_bytes=0;
};
class BlockCorrelations {
    struct Complex {
        double re=0,im=0;
        Complex()=default;Complex(double r,double i=0):re(r),im(i){}
        double real()const{return re;}double imag()const{return im;}
        Complex operator+(Complex b)const{return {re+b.re,im+b.im};}
        Complex operator-(Complex b)const{return {re-b.re,im-b.im};}
        Complex operator*(Complex b)const{return {re*b.re-im*b.im,re*b.im+im*b.re};}
        Complex operator*(double x)const{return {re*x,im*x};}
        Complex operator/(double x)const{return {re/x,im/x};}
        Complex& operator+=(Complex b){re+=b.re;im+=b.im;return *this;}
    };
    static Complex conjugate(Complex x){return {x.re,-x.im};}
    struct Plan {
        int size=0,levels=0;
        std::vector<int> order;
        std::vector<Complex> roots;
        int position(int frequency)const{return order[frequency];}
        void reset(int n) {
            if(size==n)return;size=n;order.resize(n);levels=0;
            for(int z=n;z>1;z>>=1)++levels;
            for(int k=0;k<n;++k) {
                int frequency=k,span=n,index=0;
                if(levels&1){span/=2;index+=(frequency&1)*span;frequency>>=1;}
                while(span>1){span/=4;index+=(frequency&3)*span;frequency>>=2;}
                order[k]=index;
            }
            roots.resize(n);const double pi=std::acos(-1.0);
            for(int k=0;k<n;++k){const double angle=-2*pi*k/n;roots[k]={std::cos(angle),std::sin(angle)};}
        }
        // Decimation in frequency leaves digit-reversed outputs. Callers
        // use position(k), so no separate permutation pass is necessary.
        void forward(std::vector<Complex>& values)const {
            int width=size;
            if(levels&1) {
                const int half=size/2;
                for(int j=0;j<half;++j) {
                    const Complex a=values[j],b=values[j+half];
                    values[j]=a+b;values[j+half]=(a-b)*roots[j];
                }
                width/=2;
            }
            for(;width>=4;width/=4) {
                const int quarter=width/4,step=size/width;
                for(int begin=0;begin<size;begin+=width)
                    for(int j=0;j<quarter;++j) {
                        const Complex x0=values[begin+j],x1=values[begin+j+quarter];
                        const Complex x2=values[begin+j+2*quarter],x3=values[begin+j+3*quarter];
                        const Complex a=x0+x2,b=x0-x2,c=x1+x3,raw=x1-x3,d{raw.im,-raw.re};
                        values[begin+j]=a+c;
                        values[begin+j+quarter]=(b+d)*roots[j*step];
                        values[begin+j+2*quarter]=(a-c)*roots[2*j*step];
                        values[begin+j+3*quarter]=(b-d)*roots[3*j*step];
                    }
            }
        }
        void inverse(std::vector<Complex>& values)const {
            for(auto& value:values)value=conjugate(value);
            forward(values);for(auto& value:values)value=conjugate(value)/double(size);
        }
    };

public:
 static std::vector<double> solve(int n,int d,const double* a,const double* b,
  const std::vector<CorrelationQuery>& queries,int block,CorrelationStats& stats) {
  if(n<0||d<1||block<1||(block&(block-1)))throw std::invalid_argument("invalid dimensions or non-power-of-two block");
  if(queries.size()>size_t(std::numeric_limits<int>::max()))
   throw std::length_error("too many correlation queries");
  struct Event {int id,end,sign;};
  const int groups=(n+block-1)/block, width=2*block, offset=n-1;
  std::vector<std::vector<Event>> events(groups+1);
  std::vector<rosa_exact::ExactExpansion> answer(queries.size()),prefix(std::max(2*n-1,0));
  std::vector<int> active(std::max(2*n-1,0),0);
  int last_group=0,active_queries=0;
  auto direct_range=[&](int id,int lo,int hi) {
   const int shift=queries[id].shift;
   if(lo>hi)return;
   stats.tail_products+=uint64_t(hi-lo+1)*d;
   // Four independent exact accumulators remove the loop-carried dependency
   // of a long fringe. Every rounded row dot is still retained exactly;
   // merging the lanes performs no intermediate scalar projection.
   int t=lo;
   if(hi-lo>=3) {
    rosa_exact::ExactExpansion lane[4];
    for(;t+3<=hi;t+=4) {
     double r0=0,r1=0,r2=0,r3=0;
     const double* x=a+size_t(t)*d;const double* y=b+size_t(t+shift)*d;
     for(int c=0;c<d;++c) {
      r0+=x[c]*y[c];r1+=x[d+c]*y[d+c];
      r2+=x[2*d+c]*y[2*d+c];r3+=x[3*d+c]*y[3*d+c];
     }
     lane[0].add(r0);lane[1].add(r1);lane[2].add(r2);lane[3].add(r3);
    }
    for(int z=0;z<4;++z)answer[id].add(lane[z]);
   }
   for(;t<=hi;++t) {
    double row=0;
    for(int c=0;c<d;++c)row+=a[size_t(t)*d+c]*b[size_t(t+shift)*d+c];
    answer[id].add(row);
   }
  };
  for(int id=0;id<int(queries.size());++id) {
   const auto&q=queries[id];
   if(q.lo<0||q.hi<q.lo||q.hi>=n||q.lo+q.shift<0||q.hi+q.shift>=n)throw std::invalid_argument("invalid query");
   const int first=(q.lo+block-1)/block,after=(q.hi+1)/block;
   if(after<=first){direct_range(id,q.lo,q.hi);continue;}
   direct_range(id,q.lo,first*block-1);
   direct_range(id,after*block,q.hi);
   events[after].push_back({id,0,1});events[first].push_back({id,0,-1});
   last_group=std::max(last_group,after);
  }
  if(!last_group) {
   std::vector<double> out(queries.size());for(size_t i=0;i<out.size();++i)out[i]=answer[i].to_double();
   stats.workspace_bytes=sizeof(rosa_exact::ExactExpansion)*(prefix.size()+answer.size())+sizeof(double)*out.size()+sizeof(int)*active.size();
   return out;
  }
  Plan plan;plan.reset(width);
  std::vector<Complex> payload(size_t(groups)*d*width),gradient(size_t(d)*width),scratch(width),spectrum(width);
  std::vector<unsigned char> payload_ready(groups,0);
  auto prepare_payload=[&](int j) {
   if(payload_ready[j])return;
   payload_ready[j]=1;
   for(int c=0;c<d;++c) {
    std::fill(scratch.begin(),scratch.end(),Complex{});
    for(int z=0;z<block&&j*block+z<n;++z)scratch[z].re=b[size_t(j*block+z)*d+c];
    plan.forward(scratch);++stats.transforms;
    for(int k=0;k<width;++k)payload[(size_t(j)*d+c)*width+k]=scratch[plan.position(k)];
   }
  };
  for(int i=0;i<=last_group;++i) {
   const int start=i*block;
   for(const auto&e:events[i]) {
    const int shift=queries[e.id].shift;
    if(e.sign>0)answer[e.id].add(prefix[shift+offset]);else answer[e.id].subtract(prefix[shift+offset]);
    active[shift+offset]-=e.sign;active_queries-=e.sign;
   }
   if(i==last_group)break;
   // A prefix contribution outside every open interval cancels from every
   // later pair of snapshots. Contract only the currently requested shifts;
   // exact expansions retain the same rounded coefficients on those ranges.
   if(!active_queries)continue;
   for(int c=0;c<d;++c) {
    std::fill(scratch.begin(),scratch.end(),Complex{});
    for(int z=0;z<block;++z)if(start+block-1-z<n)scratch[z].re=a[size_t(start+block-1-z)*d+c];
    plan.forward(scratch);++stats.transforms;
    for(int k=0;k<width;++k)gradient[size_t(c)*width+k]=scratch[plan.position(k)];
   }
   for(int j=0;j<groups;++j) {
    const int first_shift=(j-i)*block-block+1;
    bool useful=false;
    for(int u=0;u<2*block-1;++u) {
     const int ix=first_shift+u+offset;
     if(ix>=0&&ix<2*n-1&&active[ix]>0){useful=true;break;}
    }
    if(!useful)continue;
    prepare_payload(j);
    std::fill(spectrum.begin(),spectrum.end(),Complex{});
    for(int c=0;c<d;++c) {
     const Complex*ga=gradient.data()+size_t(c)*width;
     const Complex*pa=payload.data()+(size_t(j)*d+c)*width;
     for(int k=0;k<width;++k)spectrum[k]+=ga[k]*pa[k];
    }
    stats.pointwise_products+=uint64_t(d)*width;++stats.block_pairs;
    plan.inverse(spectrum);++stats.transforms;
    for(int u=0;u<2*block-1;++u) {
     const int ix=first_shift+u+offset;
     if(ix>=0&&ix<2*n-1&&active[ix]>0){prefix[ix].add(spectrum[plan.position(u)].re);++stats.prefix_adds;}
    }
   }
  }
  std::vector<double> result(queries.size());
  for(size_t i=0;i<queries.size();++i)result[i]=answer[i].to_double();
  stats.workspace_bytes=sizeof(Complex)*(payload.size()+gradient.size()+scratch.size()+spectrum.size()+plan.roots.size())
    +sizeof(int)*(active.size()+plan.order.size())+sizeof(rosa_exact::ExactExpansion)*(prefix.size()+answer.size())
    +2*queries.size()*sizeof(Event)+sizeof(double)*result.size()+payload_ready.size();
  return result;
 }
};
}
