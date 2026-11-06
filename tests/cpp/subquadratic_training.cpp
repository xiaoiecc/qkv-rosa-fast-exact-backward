// Full C ABI integration, including the production translation unit.
// Ordinary: c++ -O2 -std=c++17 -ffp-contract=off tests/cpp/subquadratic_training.cpp -o /tmp/rosa-training
// Sanitized: add -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer.
// Some ptrace-based containers require ASAN_OPTIONS=detect_leaks=0.
#include "../../conv_rosa_transformer/csrc/unified_core.cpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

namespace integration_test {

static std::uint64_t calls=0,checked_values=0,checked_pairs=0,sampled_rows=0,sampled_channels=0;
static std::uint64_t literal_rows=0,literal_fixtures=0;
static double worst_error=0;

[[noreturn]] static void fail(const std::string& message) {
    std::cerr<<"subquadratic training integration: "<<message<<'\n';std::exit(1);
}

template<class T> class Buffer {
    std::vector<T> values_;
    static T guard(){return static_cast<T>(113);}
public:
    std::size_t size;
    explicit Buffer(std::size_t count,T initial=T{}) :values_(count+2,initial),size(count) {
        values_.front()=values_.back()=guard();
    }
    T* data(){return values_.data()+1;}
    const T* data()const{return values_.data()+1;}
    T& operator[](std::size_t i){return data()[i];}
    const T& operator[](std::size_t i)const{return data()[i];}
    void intact()const {
        if(values_.front()!=guard() || values_.back()!=guard())fail("output boundary canary changed");
    }
};

struct Input {
    int b,groups,n,d;
    std::size_t streams,total;
    Buffer<double> qz,kz,vz,g,e0,e1,y;
    Buffer<std::uint64_t> q,k,forward_counts;
    Buffer<std::uint8_t> v;
    Buffer<std::int32_t> length,route;
    Input(int batches,int group_count,int rows,int bits,int family,std::uint64_t seed)
      :b(batches),groups(group_count),n(rows),d(bits),streams(std::size_t(b)*groups),total(streams*n*d),
       qz(total),kz(total),vz(total),g(total),e0(std::size_t(groups)*d),e1(e0.size),y(total),
       q(streams*n),k(q.size),forward_counts(streams*ROSA_FORWARD_COUNTERS),v(total),length(q.size),route(q.size) {
        std::mt19937_64 random(seed);std::uniform_real_distribution<double> real(-1.2,1.2);
        for(std::size_t i=0;i<e0.size;++i){e0[i]=real(random);e1[i]=real(random);}
        for(std::size_t s=0;s<streams;++s)for(int t=0;t<n;++t) {
            std::uint64_t a=0,c=0;
            const int high=d-1,low=std::max(0,d-2);
            if(family==0) {a=(random()&1)<<high;c=(random()&1)<<high;}
            if(family==1) {
                const unsigned symbol=unsigned(t%3);
                a=(std::uint64_t(symbol&1)<<high)|(std::uint64_t((symbol>>1)&1)<<low);c=a;
                if(t==n/3)c^=std::uint64_t(1)<<high;
            }
            if(family==2){a=random();c=random();}
            if(family==3 && t==n/3)c=std::uint64_t(1)<<high;
            for(int j=0;j<d;++j) {
                const std::size_t at=(s*n+t)*d+j;
                const double magnitude=.2+std::abs(real(random));
                qz[at]=((a>>j)&1)?magnitude:-magnitude;
                kz[at]=((c>>j)&1)?magnitude:-magnitude;
                vz[at]=real(random);g[at]=(t%5==1 || (s==1 && streams>1))?0.0:real(random);
            }
        }
        RosaTrainingOptions options;rosa_training_defaults(&options);
        const int code=rosa_training_forward(&options,b,groups,n,d,qz.data(),kz.data(),vz.data(),e0.data(),e1.data(),
            q.data(),k.data(),v.data(),length.data(),route.data(),y.data(),forward_counts.data());
        ++calls;if(code)fail("forward returned "+std::to_string(code));
        intact();
    }
    void intact()const {
        qz.intact();kz.intact();vz.intact();g.intact();e0.intact();e1.intact();y.intact();
        q.intact();k.intact();v.intact();length.intact();route.intact();forward_counts.intact();
    }
};

struct Output {
    Buffer<double> q,k,v,e0,e1,cost;
    Buffer<std::uint64_t> counts,pairs,approximation,work;
    Output(const Input& in,int pair_count)
      :q(in.total,std::numeric_limits<double>::quiet_NaN()),k(in.total,std::numeric_limits<double>::quiet_NaN()),
       v(in.total,std::numeric_limits<double>::quiet_NaN()),e0(in.e0.size,std::numeric_limits<double>::quiet_NaN()),
       e1(in.e0.size,std::numeric_limits<double>::quiet_NaN()),cost(in.streams*pair_count*4,std::numeric_limits<double>::quiet_NaN()),
       counts(in.streams*ROSA_TRAINING_COUNTERS),pairs(in.streams*pair_count*2,UINT64_MAX),
       approximation(in.streams*ROSA_APPROXIMATION_COUNTERS,UINT64_MAX),work(in.streams*ROSA_SUBQUADRATIC_COUNTERS,UINT64_MAX) {}
    void intact()const {
        q.intact();k.intact();v.intact();e0.intact();e1.intact();cost.intact();
        counts.intact();pairs.intact();approximation.intact();work.intact();
    }
};

static void finite(const Buffer<double>& values) {
    for(std::size_t i=0;i<values.size;++i)if(!std::isfinite(values[i]))fail("nonfinite gradient");
}

static Output run(const Input& in,const RosaTrainingOptions& options,bool detailed,
                  const RosaApproximationOptions* approximation=nullptr,bool optional_counters=true,
                  bool trace=true,int expected_status=0) {
    Output out(in,options.joint_pairs);
    int code;
    if(detailed)code=rosa_training_backward_subquadratic(&options,approximation,in.b,in.groups,in.n,in.d,
        in.q.data(),in.k.data(),in.v.data(),in.qz.data(),in.kz.data(),in.vz.data(),in.g.data(),in.e0.data(),in.e1.data(),
        out.q.data(),out.k.data(),out.v.data(),out.e0.data(),out.e1.data(),out.counts.data(),
        trace?out.pairs.data():nullptr,trace?out.cost.data():nullptr,
        approximation?in.length.data():nullptr,approximation?in.route.data():nullptr,
        (approximation || optional_counters)?out.approximation.data():nullptr,out.work.data());
    else if(approximation)code=rosa_training_backward_approx(&options,approximation,in.b,in.groups,in.n,in.d,
        in.q.data(),in.k.data(),in.v.data(),in.qz.data(),in.kz.data(),in.vz.data(),in.g.data(),in.e0.data(),in.e1.data(),
        out.q.data(),out.k.data(),out.v.data(),out.e0.data(),out.e1.data(),out.counts.data(),
        trace?out.pairs.data():nullptr,trace?out.cost.data():nullptr,in.length.data(),in.route.data(),out.approximation.data());
    else code=rosa_training_backward(&options,in.b,in.groups,in.n,in.d,
        in.q.data(),in.k.data(),in.v.data(),in.qz.data(),in.kz.data(),in.vz.data(),in.g.data(),in.e0.data(),in.e1.data(),
        out.q.data(),out.k.data(),out.v.data(),out.e0.data(),out.e1.data(),out.counts.data(),
        trace?out.pairs.data():nullptr,trace?out.cost.data():nullptr);
    ++calls;in.intact();out.intact();
    if(code!=expected_status)fail("backward status="+std::to_string(code)+" expected="+std::to_string(expected_status)
        +" n="+std::to_string(in.n)+" d="+std::to_string(in.d)+" order="+std::to_string(options.credit_order));
    if(code)return out;
    finite(out.q);finite(out.k);finite(out.v);finite(out.e0);finite(out.e1);
    for(std::size_t s=0;s<in.streams;++s) {
        if(in.n)for(int j=0;j<in.d;++j) {
            const std::size_t first=s*in.n*in.d+j,last=(s*in.n+in.n-1)*in.d+j;
            if(out.q[first]!=0 || out.k[last]!=0 || out.v[first]!=0)fail("impossible causal coordinate is nonzero");
        }
        const auto count=out.counts[s*ROSA_TRAINING_COUNTERS+23];
        if(count>std::uint64_t(options.joint_pairs))fail("invalid joint-pair count");
        if(trace)for(std::uint64_t pair=0;pair<count;++pair) {
            const std::size_t p=(s*options.joint_pairs+pair)*2;
            if(out.pairs[p]>=2*std::uint64_t(in.n)*in.d || out.pairs[p+1]>=2*std::uint64_t(in.n)*in.d)
                fail("joint-pair code outside stream");
            for(int j=0;j<4;++j)if(!std::isfinite(out.cost[(s*options.joint_pairs+pair)*4+j]))fail("nonfinite joint cost");
        }
        if(approximation) {
            sampled_rows+=!approximation->sample_channels && !out.approximation[s*ROSA_APPROXIMATION_COUNTERS+27];
            sampled_channels+=out.approximation[s*ROSA_APPROXIMATION_COUNTERS+33]!=0;
        }
    }
    return out;
}

static void compare(const Buffer<double>& a,const Buffer<double>& b,bool bitwise=false) {
    if(a.size!=b.size)fail("gradient shape mismatch");
    if(bitwise && std::memcmp(a.data(),b.data(),a.size*sizeof(double)))fail("cache or optional-buffer changed bits");
    for(std::size_t i=0;i<a.size;++i) {
        const double error=std::abs(a[i]-b[i]);worst_error=std::max(worst_error,error);++checked_values;
        if(error>4e-10*(1+std::max(std::abs(a[i]),std::abs(b[i]))))fail("literal/full-coverage numeric mismatch");
    }
}

static void compare_outputs(const Input& in,const RosaTrainingOptions& options,const Output& a,const Output& b,
                            bool all=true,bool bitwise=false,bool traces=true) {
    if(all){compare(a.q,b.q,bitwise);compare(a.k,b.k,bitwise);}
    compare(a.v,b.v,bitwise);compare(a.e0,b.e0,bitwise);compare(a.e1,b.e1,bitwise);
    if(!traces)return;
    for(std::size_t s=0;s<in.streams;++s) {
        const auto count=a.counts[s*ROSA_TRAINING_COUNTERS+23];
        if(count!=b.counts[s*ROSA_TRAINING_COUNTERS+23])fail("joint proposal count changed");
        for(std::uint64_t pair=0;pair<count;++pair) {
            const std::size_t offset=s*options.joint_pairs+pair;
            for(int j=0;j<2;++j)if(a.pairs[offset*2+j]!=b.pairs[offset*2+j])fail("joint proposal changed");
            for(int j=0;j<4;++j) {
                const double x=a.cost[offset*4+j],y=b.cost[offset*4+j],error=std::abs(x-y);
                worst_error=std::max(worst_error,error);++checked_pairs;
                if(bitwise && std::memcmp(&x,&y,sizeof(double)))fail("cache changed joint-cost bits");
                if(error>4e-10*(1+std::max(std::abs(x),std::abs(y))))fail("joint cost mismatch");
            }
        }
    }
}

// This intentionally uses no production route, derivative or credit helper.
// Exhaust all causal endpoints and compare the edited packed strings literally.
static std::pair<int,int> literal_route(const std::vector<std::uint64_t>& q,
                                      const std::vector<std::uint64_t>& k,int t) {
    std::pair<int,int> best{0,-1};++literal_rows;
    for(int r=0;r<t;++r) {
        int length=0;while(length<=r && q[t-length]==k[r-length])++length;
        if(length)best=std::max(best,std::make_pair(length,r));
    }
    return best;
}

static Output literal(const Input& in,const RosaTrainingOptions& options,const Output& proposals) {
    ++literal_fixtures;Output out(in,options.joint_pairs);
    for(Buffer<double>* x:{&out.q,&out.k,&out.v,&out.e0,&out.e1})std::fill_n(x->data(),x->size,0.0);
    const int n=in.n,d=in.d;const std::size_t nd=std::size_t(n)*d;
    const auto sigmoid=[&](double x){return 1.0/(1.0+std::exp(-x/options.tau));};
    const auto chain=[&](double z,int bit,double credit) {
        const double oriented=(bit?-1.0:1.0)*credit;
        if(options.logit_mode)return oriented;
        const double p=sigmoid(z);return p*(1-p)/options.tau*oriented;
    };
    for(std::size_t stream=0;stream<in.streams;++stream) {
        const std::size_t base=stream*nd,word=stream*n,embedding=(stream%in.groups)*d;
        std::vector<std::uint64_t> q(in.q.data()+word,in.q.data()+word+n),k(in.k.data()+word,in.k.data()+word+n);
        std::vector<double> old(nd),diagonal(n),cq(nd),ck(nd),cv(nd);
        double squared_signed=0,squared_empty_span=0;
        for(int j=0;j<d;++j) {
            const double a=in.e1[embedding+j],b=std::abs(in.e0[embedding+j])+std::abs(a);
            squared_signed+=a*a;squared_empty_span+=b*b;
        }
        const double radius=std::max({std::sqrt(squared_signed),.5*std::sqrt(squared_empty_span),1e-12});
        for(int t=0;t<n;++t) {
            const auto route=literal_route(q,k,t);
            if(route.first!=in.length[word+t] || route.second!=in.route[word+t])fail("literal forward route mismatch");
            double norm=0;
            for(int j=0;j<d;++j) {
                const std::size_t at=std::size_t(t)*d+j;const double g=in.g[base+at];norm+=g*g;
                old[at]=route.second<0?in.e0[embedding+j]:
                    (2*int(in.v[base+std::size_t(route.second+1)*d+j])-1)*in.e1[embedding+j];
                if(old[at]!=in.y[base+at])fail("literal forward payload mismatch");
                if(route.second<0)out.e0[embedding+j]+=g;
                else {
                    const std::size_t source=std::size_t(route.second+1)*d+j;
                    const int sign=2*int(in.v[base+source])-1;
                    out.e1[embedding+j]+=g*sign;cv[source]+=g*(-2*sign*in.e1[embedding+j]);
                }
            }
            diagonal[t]=options.trust_diagonal*std::sqrt(norm)/radius;
        }
        const auto cost=[&]() {
            double result=0;
            for(int t=0;t<n;++t) {
                const int r=literal_route(q,k,t).second;
                for(int j=0;j<d;++j) {
                    const std::size_t at=std::size_t(t)*d+j;
                    const double now=r<0?in.e0[embedding+j]:(2*int(in.v[base+std::size_t(r+1)*d+j])-1)*in.e1[embedding+j];
                    const double delta=now-old[at];result+=in.g[base+at]*delta+.5*diagonal[t]*delta*delta;
                }
            }
            return result;
        };
        for(int t=0;t<n;++t)for(int j=0;j<d;++j) {
            const std::size_t at=std::size_t(t)*d+j;const auto mask=std::uint64_t(1)<<j;
            q[t]^=mask;cq[at]=cost();q[t]^=mask;
            k[t]^=mask;ck[at]=cost();k[t]^=mask;
            out.q[base+at]=chain(in.qz[base+at],int((q[t]>>j)&1),cq[at]);
            out.k[base+at]=chain(in.kz[base+at],int((k[t]>>j)&1),ck[at]);
            out.v[base+at]=chain(in.vz[base+at],in.v[base+at],cv[at]);
        }
        const auto count=proposals.counts[stream*ROSA_TRAINING_COUNTERS+23];
        out.counts[stream*ROSA_TRAINING_COUNTERS+23]=count;
        const auto toggle=[&](std::uint64_t code){auto& target=code>=nd?q:k;const auto at=code%nd;target[at/d]^=std::uint64_t(1)<<(at%d);};
        for(std::uint64_t pair=0;pair<count;++pair) {
            const std::size_t offset=stream*options.joint_pairs+pair;
            const auto a=proposals.pairs[offset*2],b=proposals.pairs[offset*2+1];
            out.pairs[offset*2]=a;out.pairs[offset*2+1]=b;
            const std::size_t ia=a%nd,ib=b%nd;
            const double ca=a>=nd?cq[ia]:ck[ia],cb=b>=nd?cq[ib]:ck[ib];
            toggle(a);toggle(b);const double cij=cost();toggle(a);toggle(b);
            const double interaction=cij-ca-cb;
            const double za=a>=nd?in.qz[base+ia]:in.kz[base+ia],zb=b>=nd?in.qz[base+ib]:in.kz[base+ib];
            const int ba=int(((a>=nd?q[ia/d]:k[ia/d])>>(ia%d))&1),bb=int(((b>=nd?q[ib/d]:k[ib/d])>>(ib%d))&1);
            const double pa=sigmoid(za),pb=sigmoid(zb),fa=ba?1-pa:pa,fb=bb?1-pb:pb;
            (a>=nd?out.q:out.k)[base+ia]=chain(za,ba,ca+fb*interaction);
            (b>=nd?out.q:out.k)[base+ib]=chain(zb,bb,cb+fa*interaction);
            out.cost[offset*4]=ca;out.cost[offset*4+1]=cb;out.cost[offset*4+2]=cij;out.cost[offset*4+3]=interaction;
        }
    }
    return out;
}

static void low_level(const Input& in) {
    if(in.n>7)return;
    const int n=in.n,d=in.d;const std::size_t nd=std::size_t(n)*d;
    for(int order:{1,2}) {
        RosaTrainingOptions options;rosa_training_defaults(&options);
        options.credit_order=order;options.joint_pairs=0;options.trust_diagonal=order==2?.5:0.;options.logit_mode=1;
        const auto direct=run(in,options,false),expected=literal(in,options,direct);
        compare_outputs(in,options,expected,direct);
        for(std::size_t stream=0;stream<in.streams;++stream) {
            const std::size_t base=stream*nd,word=stream*n,embedding=(stream%in.groups)*d;
            Buffer<double> q(nd),k(nd),v(nd),h(nd);Buffer<std::int32_t> length(n),route(n);
            Buffer<std::uint64_t> counts(ROSA_SUBQUADRATIC_COUNTERS);
            double e1_square=0,span_square=0;
            for(int j=0;j<d;++j) {
                const double a=in.e1[embedding+j],b=std::abs(in.e0[embedding+j])+std::abs(a);
                e1_square+=a*a;span_square+=b*b;
            }
            const double radius=std::max({std::sqrt(e1_square),.5*std::sqrt(span_square),1e-12});
            for(int t=0;t<n;++t) {
                double norm=0;for(int j=0;j<d;++j){const double g=in.g[base+std::size_t(t)*d+j];norm+=g*g;}
                for(int j=0;j<d;++j)h[std::size_t(t)*d+j]=options.trust_diagonal*std::sqrt(norm)/radius;
            }
            // Null data/output arrays at n=0 are expressly allowed by the
            // C ABI and must not be dereferenced or advanced as C++ pointers.
            const auto* pq=n?in.q.data()+word:nullptr;const auto* pk=n?in.k.data()+word:nullptr;
            const auto* pv=n?in.v.data()+base:nullptr;const auto* pg=n?in.g.data()+base:nullptr;
            int code;
            if(order==1)code=rosa_uniform(n,d,pq,pk,pv,pg,in.e0.data()+embedding,in.e1.data()+embedding,
                n?q.data():nullptr,n?k.data():nullptr,n?v.data():nullptr,n?length.data():nullptr,n?route.data():nullptr,counts.data());
            else code=rosa_quadratic(n,d,pq,pk,pv,pg,n?h.data():nullptr,in.e0.data()+embedding,in.e1.data()+embedding,
                n?q.data():nullptr,n?k.data():nullptr,n?v.data():nullptr,n?length.data():nullptr,n?route.data():nullptr,counts.data());
            ++calls;if(code)fail("low-level credit adapter returned "+std::to_string(code));
            q.intact();k.intact();v.intact();h.intact();length.intact();route.intact();counts.intact();
            finite(q);finite(k);finite(v);
            for(int t=0;t<n;++t) {
                if(length[t]!=in.length[word+t] || route[t]!=in.route[word+t])fail("low-level route mismatch");
                for(int j=0;j<d;++j) {
                    const std::size_t at=std::size_t(t)*d+j;
                    const double got[3]={((in.q[word+t]>>j)&1)?-q[at]:q[at],
                        ((in.k[word+t]>>j)&1)?-k[at]:k[at],in.v[base+at]?-v[at]:v[at]};
                    const double want[3]={expected.q[base+at],expected.k[base+at],expected.v[base+at]};
                    for(int which=0;which<3;++which) {
                        const double error=std::abs(got[which]-want[which]);worst_error=std::max(worst_error,error);++checked_values;
                        if(error>4e-10*(1+std::max(std::abs(got[which]),std::abs(want[which]))))fail("low-level literal credit mismatch");
                    }
                }
            }
            if(order==1) {
                code=rosa_matching_stats(n,pq,pk,n?length.data():nullptr,n?route.data():nullptr);++calls;
                if(code)fail("low-level matching adapter returned "+std::to_string(code));
                for(int t=0;t<n;++t)if(length[t]!=in.length[word+t] || route[t]!=in.route[word+t])fail("matching adapter route mismatch");
                length.intact();route.intact();
            }
        }
    }
}

static void exercise(const Input& in) {
    low_level(in);
    for(int order:{1,2}) {
        RosaTrainingOptions options;rosa_training_defaults(&options);
        options.credit_order=order;options.joint_pairs=order==2?2:0;options.trust_diagonal=order==2?.5:0.;
        options.joint_rows=8;options.tau=.7;options.proposal_seed=8191;options.repair_seed=131071;
        options.logit_mode=in.n%2;options.randomized=0;
        const auto ordinary=run(in,options,false),exact=run(in,options,true);
        compare_outputs(in,options,ordinary,exact,true,true);
        if(in.n<=7){const auto expected=literal(in,options,exact);compare_outputs(in,options,expected,exact);}
        options.randomized=1;const auto cached=run(in,options,true);
        compare_outputs(in,options,exact,cached,true,true);
        const auto no_optional=run(in,options,true,nullptr,false,false);
        compare_outputs(in,options,cached,no_optional,true,true,false);

        RosaApproximationOptions approximation;rosa_approximation_defaults(&approximation);
        approximation.seed=65537;approximation.sample_rows=std::max(1,in.n);
        const auto full_rows=run(in,options,true,&approximation);
        compare_outputs(in,options,exact,full_rows);
        approximation.sample_channels=in.d;
        const auto full_channels=run(in,options,true,&approximation);
        compare_outputs(in,options,exact,full_channels);

        approximation.sample_rows=2;
        for(int channels:{0,std::max(1,in.d/4)}) {
            approximation.sample_channels=channels;options.randomized=0;
            const auto partial=run(in,options,true,&approximation);
            compare_outputs(in,options,ordinary,partial,false);
            const auto partial_ordinary=run(in,options,false,&approximation);
            compare_outputs(in,options,partial,partial_ordinary,true,true);
            options.randomized=1;const auto partial_cached=run(in,options,true,&approximation);
            compare_outputs(in,options,partial,partial_cached,true,true);
        }
        options.repair_budget=1;run(in,options,true,nullptr,true,true,1);
    }
}

} // namespace integration_test

int main() {
    using namespace integration_test;
    const auto start=std::chrono::steady_clock::now();std::uint64_t cases=0;
    for(int n:{0,1,7,40})for(int d:{1,64})for(int family:{0,1}) {
        const Input input(n>1?2:1,2,n,d,family,2026110600ULL+cases);
        exercise(input);++cases;
    }
    for(int family:{2,3}){const Input input(1,2,40,64,family,2026110700ULL+cases);exercise(input);++cases;}
    const Input empty_batch(0,2,7,64,0,7);exercise(empty_batch);++cases;
    if(!sampled_rows || !sampled_channels)fail("partial-estimator paths were not exercised");
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout<<"{\"status\":\"PASS\",\"cases\":"<<cases<<",\"abi_calls\":"<<calls
             <<",\"compared_gradient_values\":"<<checked_values<<",\"compared_joint_costs\":"<<checked_pairs
             <<",\"literal_fixtures\":"<<literal_fixtures<<",\"literal_route_rows\":"<<literal_rows
             <<",\"row_sampled_streams\":"<<sampled_rows<<",\"channel_sampled_streams\":"<<sampled_channels
             <<",\"max_absolute_error\":"<<worst_error<<",\"seconds\":"<<seconds<<"}\n";
}
