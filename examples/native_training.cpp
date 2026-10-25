// Build from the repository root:
// c++ -O3 -std=c++17 -ffp-contract=off examples/native_training.cpp \
//     conv_rosa_transformer/csrc/unified_core.cpp -o /tmp/rosa-native
// No Python, NumPy, Torch, or framework C++ headers are required.
#include "../conv_rosa_transformer/csrc/training.h"
#include <array>
#include <cmath>
#include <iostream>
#include <string>

int main(int argc,char** argv) {
    constexpr int B=1,G=1,T=3,D=2,N=B*G*T*D;
    std::array<double,N> qz{-.2,.2,-.2,.2,-.2,-.2};
    std::array<double,N> kz{-.2,-.2,.2,.2,-.2,-.2};
    std::array<double,N> vz{-.2,-.2,-.2,-.2,.2,.2};
    std::array<double,D> e0{-1,-1},e1{1,1},ge0{},ge1{};
    std::array<double,N> y{},g{},gq{},gk{},gv{};
    std::array<uint64_t,T> q{},k{};
    std::array<uint8_t,N> v{};
    std::array<int32_t,T> lengths{},routes{};
    std::array<uint64_t,ROSA_FORWARD_COUNTERS> forward_counts{};
    std::array<uint64_t,ROSA_TRAINING_COUNTERS> backward_counts{};
    RosaTrainingOptions options;rosa_training_defaults(&options);
    // C options are explicit. The default is order 2, one pair, trust 0.5;
    // selecting order 1 requires clearing both second-order terms.
    if(argc>1&&std::string(argv[1])=="--first") {
        options.credit_order=1;
        options.joint_pairs=0;
        options.trust_diagonal=0.;
    }
    options.randomized=1;options.proposal_seed=123;options.repair_seed=456;
    int code=rosa_training_forward(&options,B,G,T,D,qz.data(),kz.data(),vz.data(),
        e0.data(),e1.data(),q.data(),k.data(),v.data(),lengths.data(),routes.data(),
        y.data(),forward_counts.data());
    if(code){std::cerr<<"forward error "<<code<<'\n';return code;}
    for(int j=0;j<D;++j)g[(T-1)*D+j]=y[(T-1)*D+j]-1.;
    code=rosa_training_backward(&options,B,G,T,D,q.data(),k.data(),v.data(),
        qz.data(),kz.data(),vz.data(),g.data(),e0.data(),e1.data(),
        gq.data(),gk.data(),gv.data(),ge0.data(),ge1.data(),backward_counts.data(),nullptr,nullptr);
    if(code){std::cerr<<"backward error "<<code<<'\n';return code;}
    for(double value:gq)if(!std::isfinite(value))return 10;
    std::cout<<"hard output: "<<y[N-2]<<", "<<y[N-1]<<'\n'
             <<"joint blocks: "<<backward_counts[23]<<'\n'
             <<"last Q logit gradient: "<<gq[N-2]<<", "<<gq[N-1]<<'\n';
    return 0;
}
