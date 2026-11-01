#pragma once
// Experimental numerical contraction. Exact route discovery is unchanged;
// FFT summation is approximate and must remain outside the exact default.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <random>
#include "approx_channel_pivotal.h"
#include <vector>

namespace rosa_random {
class BoundaryCorridorFFT {
public:
    struct Query {int id,start,stop,payload_start;};
    struct Counters {uint64_t groups=0,queries=0,rows=0,transforms=0,columns=0,channels=0,exact_fallbacks=0,fallback_rows=0;};
    Counters counters;
private:
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
    } plan_;
    uint64_t fingerprint_seed_=0,sampling_seed_=0;
    int sample_budget_=0;
    std::vector<double> values_,indicators_;
    std::vector<int> stamps_;
    std::vector<uint8_t> sampled_;
    std::vector<long double> base_prefix_;
    int prefix_rows_=0;
    static uint64_t mix(uint64_t x) {
        x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
        x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);return x^(x>>31);
    }
public:
    explicit BoundaryCorridorFFT(uint64_t seed=0,int sample_budget=0,uint64_t sample_seed=0)
      :fingerprint_seed_(seed),sampling_seed_(sample_seed),sample_budget_(sample_budget){}
    bool get(int id,int stop,double& value) const {
        if(id>=int(stamps_.size())||stamps_[id]!=stop)return false;
        value=values_[id];return true;
    }
    bool sampled(int id,int stop) const {
        return id>=0&&id<int(stamps_.size())&&stamps_[id]==stop&&sampled_[id];
    }
    // Conservative cancellation scale, not a promise of relative FFT error.
    // The caller can recover the old exact expansion when its final credit
    // lies this close to zero; ordinary large credits retain the FFT result.
    double error_indicator(int id,int stop)const {
        return id<int(stamps_.size())&&stamps_[id]==stop?indicators_[id]:0.;
    }
    template<bool Quadratic,class Corridor>
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((noinline))
#elif defined(_MSC_VER)
    __declspec(noinline)
#endif
    void finish(int first,const std::vector<Corridor>& corridors,int n,int d,
                const uint8_t* payload,const double* grad,const double* costs,
                const double* emb1,const double* base_scores,
                const int* score_row_end,const int* heavy_at_position,int heavy_count) {
        std::vector<Query> queries;
        for(int id=first;id>=0;id=corridors[id].next_finish) {
            const auto& c=corridors[id];queries.push_back({id,c.start,c.stop,c.start+c.end_bias+1});
        }
        prepare<Quadratic>(std::move(queries),n,d,payload,grad,costs,emb1,base_scores,score_row_end,heavy_at_position,heavy_count);
    }
    template<bool Quadratic>
    void prepare(std::vector<Query> queries,int n,int d,
                 const uint8_t* payload,const double* grad,const double* costs,
                 const double* emb1,const double* base_scores,
                 const int* score_row_end,const int* heavy_at_position,int heavy_count) {
        if(sample_budget_>0)queries.erase(std::remove_if(queries.begin(),queries.end(),
            [](const Query&q){return q.stop-q.start+1<32;}),queries.end());
        if(queries.size()<8)return;
        std::sort(queries.begin(),queries.end(),[](const Query&a,const Query&b){
            return a.payload_start!=b.payload_start?a.payload_start<b.payload_start:a.id<b.id;
        });
        const int stop=queries.front().stop;
        if constexpr(!Quadratic) {
            if(base_prefix_.empty())base_prefix_.push_back(0);
            while(prefix_rows_<=stop) {
                base_prefix_.push_back(base_prefix_.back()+static_cast<long double>(base_scores[prefix_rows_]));++prefix_rows_;
            }
        }
        for(size_t begin=0;begin<queries.size();) {
            size_t end=begin+1;while(end<queries.size()&&queries[end].payload_start==queries[begin].payload_start)++end;
            if(end-begin<8){begin=end;continue;}
            const int a=queries[begin].payload_start,r=queries[begin].stop;
            int m=0;uint64_t work=0;
            for(size_t j=begin;j<end;++j) {
                m=std::max(m,r-queries[j].start+1);
                work+=uint64_t(score_row_end[r+1]-score_row_end[queries[j].start]);
            }
            if(m<16 || a<0 || a+m>n || work<8*uint64_t(m)){begin=end;continue;}
            int size=1,levels=0;while(size<2*m-1){size*=2;++levels;}
            // Full binary column equality (up to complement) is verified.
            // Fingerprints only avoid redundant comparison work.
            std::vector<uint64_t> hashes(d,0);uint64_t all=0;
            for(int i=0;i<m;++i) {
                const uint64_t random=mix(uint64_t(i)+fingerprint_seed_+UINT64_C(0x6242b7d84e918ac3));all^=random;
                for(int j=0;j<d;++j)hashes[j]^=(uint64_t(0)-uint64_t(payload[size_t(a+i)*d+j]))&random;
            }
            std::vector<int> representative,column_group(d),flip(d);
            for(int j=0;j<d;++j) {
                flip[j]=payload[size_t(a)*d+j];if(flip[j])hashes[j]^=all;
                int group=-1;
                for(int g=0;g<int(representative.size());++g) {
                    const int other=representative[g];if(hashes[j]!=hashes[other])continue;
                    bool equal=true;
                    for(int i=0;i<m;++i)
                        if((payload[size_t(a+i)*d+j]^flip[j])!=(payload[size_t(a+i)*d+other]^flip[other])){equal=false;break;}
                    if(equal){group=g;break;}
                }
                if(group<0){group=int(representative.size());representative.push_back(j);}
                column_group[j]=group;
            }
            const int columns=int(representative.size()),raw_channels=Quadratic?2*columns:columns;
            std::vector<int> ones(columns,0),channel_column,channel_type;
            for(int group=0;group<columns;++group) {
                const int rep=representative[group];
                for(int i=0;i<m;++i)ones[group]+=payload[size_t(a+i)*d+rep]^flip[rep];
                if(!ones[group])continue;
                // On balanced binary columns, use the signed channel and
                // a compensated constant prefix. Sparse columns retain two
                // one-hot channels to avoid subtracting a dominant constant.
                if(!Quadratic || 8*std::min(ones[group],m-ones[group])>=m) {
                    channel_column.push_back(group);channel_type.push_back(2);
                } else {
                    channel_column.push_back(group);channel_type.push_back(0);
                    channel_column.push_back(group);channel_type.push_back(1);
                }
            }
            const int channels=int(channel_column.size());
            std::vector<int> signed_channels;
            for(int channel=0;channel<channels;++channel)if(channel_type[channel]==2)signed_channels.push_back(channel);
            const int population=int(signed_channels.size());
            const int sample=sample_budget_>0?std::min(population,sample_budget_):population;
            const int active_channels=channels-population+sample;
            std::vector<int> rare(m+1,0);std::vector<uint8_t> present(std::max(heavy_count,0),0);int present_count=0;
            for(int i=0;i<m;++i) {
                const int slot=heavy_at_position[a+i];rare[i+1]=rare[i]+(slot<0);
                if(slot>=0&&!present[slot]){present[slot]=1;++present_count;}
            }
            uint64_t rare_work=0;
            for(size_t j=begin;j<end;++j)rare_work+=rare[r-queries[j].start+1];
            rare_work=std::min(rare_work,work);
            const double direct=4.0*work+double(d)*(double(rare_work)+std::min(double(work-rare_work),double(m)*present_count));
            const double fft=512.0+8.0*active_channels*size*levels;
            if(direct<=fft){begin=end;continue;}
            std::vector<double> features(size_t(raw_channels)*m,0.0);
            bool finite=true;
            for(int i=0;i<m;++i) {
                const int x=r-i;
                for(int j=0;j<d;++j) {
                    const int group=column_group[j];
                    if constexpr(Quadratic) {
                        const double c0=costs[2*(size_t(x)*d+j)],c1=costs[2*(size_t(x)*d+j)+1];
                        features[size_t(2*group)*m+i]+=flip[j]?c1:c0;
                        features[size_t(2*group+1)*m+i]+=flip[j]?c0:c1;
                    } else features[size_t(group)*m+i]+=(flip[j]?-1.0:1.0)*(grad[size_t(x)*d+j]*emb1[j]);
                }
            }
            for(double value:features)if(!std::isfinite(value)){finite=false;break;}
            if(!finite){begin=end;continue;}
            long double scale_sum=0;
            for(int channel=0;channel<raw_channels;++channel) {
                double maximum=0;
                for(int i=0;i<m;++i)maximum=std::max(maximum,std::abs(features[size_t(channel)*m+i]));
                scale_sum+=maximum;
            }
            if constexpr(!Quadratic) {
                double maximum=0;for(int i=0;i<m;++i)maximum=std::max(maximum,std::abs(base_scores[r-i]));
                scale_sum+=maximum;
            }
            const double indicator=double(1024.L*std::numeric_limits<double>::epsilon()*(levels+d+8)*m*scale_sum);
            std::vector<long double> constant(m+1,0);
            std::vector<int> constant_nonzero(channels?0:m+1,0);
            int output_channel=0;
            for(int group=0;group<columns;++group) {
                if constexpr(Quadratic) {
                    const double* c0=features.data()+size_t(2*group)*m;
                    const double* c1=c0+m;
                    if(!ones[group]) {
                        for(int i=0;i<m;++i)constant[i+1]+=static_cast<long double>(c0[i]);
                    } else if(8*std::min(ones[group],m-ones[group])>=m) {
                        double* dst=features.data()+size_t(output_channel++)*m;
                        for(int i=0;i<m;++i) {
                            constant[i+1]+=.5L*(static_cast<long double>(c0[i])+static_cast<long double>(c1[i]));
                            dst[i]=.5*c1[i]-.5*c0[i];
                        }
                    } else {
                        std::copy(c0,c0+m,features.data()+size_t(output_channel++)*m);
                        std::copy(c1,c1+m,features.data()+size_t(output_channel++)*m);
                    }
                } else {
                    const double* value=features.data()+size_t(group)*m;
                    if(!ones[group])for(int i=0;i<m;++i)constant[i+1]-=static_cast<long double>(value[i]);
                    else std::copy(value,value+m,features.data()+size_t(output_channel++)*m);
                }
            }
            if constexpr(!Quadratic)if(!channels) {
                // A constant payload has the same row dot for every query.
                // Subtract the baseline before summation, exactly as the
                // direct engine does; separate large prefixes lose tiny
                // emb0/emb1 differences here.
                for(int i=0;i<m;++i)constant[i+1]=static_cast<long double>(-features[i]-base_scores[r-i]);
            }
            for(int i=0;i<m;++i) {
                if(!channels)constant_nonzero[i+1]=constant_nonzero[i]+(constant[i+1]!=0);
                constant[i+1]+=constant[i];
            }
            std::vector<double> maxima(channels,0.),channel_weight(channels,1.);
            std::vector<int> remainder;
            for(int channel=0;channel<channels;++channel) {
                const double* feature=features.data()+size_t(channel)*m;
                for(int i=0;i<m;++i)maxima[channel]=std::max(maxima[channel],std::abs(feature[i]));
                if(!maxima[channel]){channel_weight[channel]=0.;continue;}
                if(channel_type[channel]==2)remainder.push_back(channel);
            }
            int remaining_budget=sample_budget_>0?std::min(int(remainder.size()),sample_budget_):int(remainder.size());
            bool sampling_active=remaining_budget<int(remainder.size());
            if(sampling_active) {
                std::vector<double> magnitudes; magnitudes.reserve(remainder.size());
                for(int channel:remainder)magnitudes.push_back(maxima[channel]);
                const uint64_t seed=mix(sampling_seed_^mix((uint64_t(uint32_t(a))<<32)|uint32_t(r)));
                const auto sampled_weights=pivotal_channel_weights(magnitudes,remaining_budget,seed);
                for(size_t at=0;at<remainder.size();++at)channel_weight[remainder[at]]=sampled_weights[at];
            }
            double maximum_weight=1.;for(double weight:channel_weight)maximum_weight=std::max(maximum_weight,weight);
            if(channels)plan_.reset(size);
            std::vector<Complex> packed(channels?size:0),spectrum(channels?size:0,Complex{});
            for(int channel=0;channel<channels;++channel) {
                const double weight=channel_weight[channel];if(!weight)continue;
                const double* feature=features.data()+size_t(channel)*m;
                const double maximum=maxima[channel];
                if(!maximum)continue;
                int exponent=0;std::frexp(maximum,&exponent);
                // Power-of-two balancing prevents the binary payload channel
                // from drowning a small adjoint when two real FFTs are packed.
                const int scaling=std::max(-900,std::min(900,-exponent));
                const int group=channel_column[channel],which=channel_type[channel];
                const int rep=representative[group];
                for(int i=0;i<m;++i) {
                    const int bit=payload[size_t(a+i)*d+rep]^flip[rep];
                    const double p=which==2?double(2*bit-1):double(which?bit:1-bit);
                    packed[i]={std::ldexp(feature[i],scaling),p};
                }
                std::fill(packed.begin()+m,packed.end(),Complex{});plan_.forward(packed);++counters.transforms;
                for(int k=0;k<=size/2;++k) {
                    const Complex left=packed[plan_.position(k)],right=conjugate(packed[plan_.position((size-k)&(size-1))]);
                    const Complex fa=(left+right)*0.5,fb=(left-right)*Complex(0,-0.5),product=fa*fb;
                    const Complex contribution(std::ldexp(product.real(),-scaling),std::ldexp(product.imag(),-scaling));
                    spectrum[k]+=weight==1.?contribution:contribution*weight;
                }
            }
            if(channels) {
                for(int k=1;k<size/2;++k)spectrum[size-k]=conjugate(spectrum[k]);
                plan_.inverse(spectrum);++counters.transforms;
            }
            std::vector<double> answer(end-begin);
            for(size_t j=begin;j<end;++j) {
                const auto&q=queries[j];const int length=r-q.start+1;
                long double combined=constant[length];
                if(channels)combined+=static_cast<long double>(spectrum[plan_.position(length-1)].real());
                if constexpr(!Quadratic)if(channels)combined-=base_prefix_[r+1]-base_prefix_[q.start];
                const double value=double(combined);answer[j-begin]=value;if(!std::isfinite(value))finite=false;
            }
            if(finite) {
                for(size_t j=begin;j<end;++j) {
                    const int id=queries[j].id;
                    if(id>=int(values_.size())){values_.resize(id+1);indicators_.resize(id+1);stamps_.resize(id+1,-1);sampled_.resize(id+1,0);}
                    values_[id]=answer[j-begin];stamps_[id]=r;sampled_[id]=uint8_t(sampling_active);
                    indicators_[id]=!channels&&constant_nonzero[r-queries[j].start+1]<=1?0.:indicator*maximum_weight;
                }
                ++counters.groups;counters.queries+=end-begin;counters.rows+=work;
                counters.columns+=columns;counters.channels+=channels;
            }
            begin=end;
        }
    }
};
} // namespace rosa_random
