#pragma once
// Fixed-size PPS selection for numerical channels. Full constants are external.
// Probability construction depends only on complete, fixed feature magnitudes;
// retain these original probabilities for Horvitz--Thompson weights.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace rosa_random {
inline std::vector<long double> pivotal_channel_probabilities(
    const std::vector<double>& maxima,int budget) {
    const int n=int(maxima.size()),keep=std::max(0,std::min(n,budget));
    std::vector<long double> probability(n,0.L),weight(n,0.L);
    if(!keep)return probability;
    if(keep==n){std::fill(probability.begin(),probability.end(),1.L);return probability;}
    double maximum=0.;
    for(double value:maxima) {
        if(!std::isfinite(value)||value<0.)throw std::invalid_argument("channel magnitude must be finite and nonnegative");
        maximum=std::max(maximum,value);
    }
    if(!maximum) {
        std::fill(probability.begin(),probability.end(),static_cast<long double>(keep)/n);
    } else {
        for(int i=0;i<n;++i)weight[i]=static_cast<long double>(maxima[i])/static_cast<long double>(maximum);
        const long double floor=.1L*keep/n;
        std::vector<int> active(n);for(int i=0;i<n;++i)active[i]=i;
        int fixed=0;
        while(!active.empty()) {
            long double mass=0.L;for(int i:active)mass+=weight[i];
            const long double lambda=mass?(keep-fixed-floor*active.size())/mass:0.L;
            bool changed=false;
            for(size_t at=0;at<active.size();) {
                const int i=active[at];
                if(floor+lambda*weight[i]>=1.L) {
                    probability[i]=1.L;++fixed;
                    active.erase(active.begin()+at);changed=true;
                } else ++at;
            }
            if(changed)continue;
            for(int i:active)probability[i]=mass?floor+lambda*weight[i]:
                static_cast<long double>(keep-fixed)/active.size();
            break;
        }
    }
    // Remove only the last few rounding ulps of the waterfill sum. The
    // adjusted probabilities are the very probabilities used below and in HT.
    long double total=0.L;for(long double p:probability)total+=p;
    const long double adjustment=keep-total;
    if(adjustment)for(long double&p:probability)
        if(p>0.L&&p<1.L&&p+adjustment>0.L&&p+adjustment<1.L){p+=adjustment;break;}
    return probability;
}

inline std::vector<double> pivotal_channel_weights(
    const std::vector<double>& positive_maxima,int budget,uint64_t seed) {
    const int n=int(positive_maxima.size()),keep=std::max(0,std::min(n,budget));
    const auto probability=pivotal_channel_probabilities(positive_maxima,budget);
    std::vector<long double> work=probability;
    std::vector<uint8_t> selected(n,0);
    std::vector<double> result(n,0.);
    if(!keep)return result;
    if(keep==n){std::fill(result.begin(),result.end(),1.);return result;}
    std::mt19937_64 random(seed);
    // Random order removes arbitrary payload-column ordering effects. For
    // equal probabilities, permutation symmetry gives uniform fixed-size
    // subsets; heterogeneous probabilities retain their own marginals.
    std::vector<int> order(n);for(int i=0;i<n;++i)order[i]=i;
    for(int at=n-1;at>0;--at) {
        std::uniform_int_distribution<int> choose(0,at);
        std::swap(order[at],order[choose(random)]);
    }
    int carry=-1,chosen=0;
    for(int i:order) {
        if(work[i]>=1.L){selected[i]=1;++chosen;continue;}
        if(work[i]<=0.L)continue;
        if(carry<0){carry=i;continue;}
        const long double a=work[carry],b=work[i],sum=a+b;
        const long double u=std::generate_canonical<long double,64>(random);
        if(sum<=1.L) {
            if(u<a/sum){work[carry]=sum;work[i]=0.L;}
            else {work[carry]=0.L;work[i]=sum;}
        } else {
            if(u<(1.L-b)/(2.L-sum)){work[carry]=1.L;work[i]=sum-1.L;}
            else {work[carry]=sum-1.L;work[i]=1.L;}
        }
        if(work[carry]>=1.L){selected[carry]=1;++chosen;}
        if(work[i]>=1.L){selected[i]=1;++chosen;}
        if(work[carry]>0.L&&work[carry]<1.L)continue;
        carry=work[i]>0.L&&work[i]<1.L?i:-1;
    }
    // Integral total mass leaves at most one fractional carry, differing
    // from 0 or 1 only by accumulation roundoff. Enforce the requested count.
    if(carry>=0) {
        const int needed=keep-chosen;
        if(needed<0||needed>1)throw std::runtime_error("pivotal mass invariant");
        selected[carry]=uint8_t(needed);chosen+=needed;
    }
    if(chosen!=keep)throw std::runtime_error("pivotal sample count");
    for(int i=0;i<n;++i)if(selected[i])result[i]=double(1.L/probability[i]);
    return result;
}
} // namespace rosa_random

