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
 // Compare the universal fringe bound at every power of two. This chooses
 // a block size for the same algorithm, without recognizing input families.
 std::vector<uint64_t> count(size_t(n)+1),sum(size_t(n)+1);int longest=0;
 for(const auto&q:queries){const int len=q.hi-q.lo+1;++count[len];longest=std::max(longest,len);}
 for(int len=1;len<=n;++len){sum[len]=sum[len-1]+count[len]*uint64_t(len);count[len]+=count[len-1];}
 int chosen=1;long double best=std::numeric_limits<long double>::infinity();
 for(int block=1;;block*=2) {
  const int cap=std::min(n,2*block),groups=(n+block-1)/block;
  const uint64_t tails=sum[cap]+(count[n]-count[cap])*uint64_t(2*block);
  long double cost=static_cast<long double>(d)*tails;
  if(block<=longest)cost+=static_cast<long double>(2*block)*(2.L*d*groups*std::log2(2*block)
       +static_cast<long double>(groups)*groups*(d+std::log2(2*block)));
  if(cost<best){best=cost;chosen=block;}
  if(block>n)break;
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
  std::vector<int> last_needed(std::max(2*n-1,0),-1);
  int last_group=0;
  auto direct_range=[&](int id,int lo,int hi) {
   const int shift=queries[id].shift;
   for(int t=lo;t<=hi;++t) {
    double row=0;
    for(int c=0;c<d;++c){row+=a[size_t(t)*d+c]*b[size_t(t+shift)*d+c];++stats.tail_products;}
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
   last_needed[q.shift+offset]=std::max(last_needed[q.shift+offset],after);
  }
  if(!last_group) {
   std::vector<double> out(queries.size());for(size_t i=0;i<out.size();++i)out[i]=answer[i].to_double();
   stats.workspace_bytes=sizeof(rosa_exact::ExactExpansion)*(prefix.size()+answer.size())+sizeof(double)*out.size()+sizeof(int)*last_needed.size();
   return out;
  }
  Plan plan;plan.reset(width);
  std::vector<Complex> payload(size_t(groups)*d*width),gradient(size_t(d)*width),scratch(width),spectrum(width);
  for(int j=0;j<groups;++j)for(int c=0;c<d;++c) {
   std::fill(scratch.begin(),scratch.end(),Complex{});
   for(int z=0;z<block&&j*block+z<n;++z)scratch[z].re=b[size_t(j*block+z)*d+c];
   plan.forward(scratch);++stats.transforms;
   for(int k=0;k<width;++k)payload[(size_t(j)*d+c)*width+k]=scratch[plan.position(k)];
  }
  for(int i=0;i<=last_group;++i) {
   const int start=i*block;
   for(const auto&e:events[i]) {
    const int shift=queries[e.id].shift;
    if(e.sign>0)answer[e.id].add(prefix[shift+offset]);else answer[e.id].subtract(prefix[shift+offset]);
   }
   if(i==last_group)break;
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
     if(ix>=0&&ix<2*n-1&&last_needed[ix]>i){useful=true;break;}
    }
    if(!useful)continue;
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
     if(ix>=0&&ix<2*n-1&&last_needed[ix]>i){prefix[ix].add(spectrum[plan.position(u)].re);++stats.prefix_adds;}
    }
   }
  }
  std::vector<double> result(queries.size());
  for(size_t i=0;i<queries.size();++i)result[i]=answer[i].to_double();
  stats.workspace_bytes=sizeof(Complex)*(payload.size()+gradient.size()+scratch.size()+spectrum.size()+plan.roots.size())
    +sizeof(int)*(last_needed.size()+plan.order.size())+sizeof(rosa_exact::ExactExpansion)*(prefix.size()+answer.size())
    +2*queries.size()*sizeof(Event)+sizeof(double)*result.size();
  return result;
 }
};
}
