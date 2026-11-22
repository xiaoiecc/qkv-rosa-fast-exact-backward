#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace rosa_random {
// Every temporal stratum contributes one complete row with inverse selection
// probability. Selection changes neither row argmaxes nor the full control.
// Distinct strata are independent; all rows have positive inclusion chance.
inline void stratified_rows(const std::vector<int32_t>&population,int budget,
    const double*g,int d,int importance,uint64_t seed,
    std::vector<int32_t>&rows,std::vector<double>&weights) {
    const int count=int(population.size()),keep=std::min(count,budget);
    rows.clear();weights.clear();rows.reserve(keep);weights.reserve(keep);
    if(keep==count){rows=population;weights.assign(count,1.);return;}
    std::mt19937_64 random(seed);
    std::vector<double> mass;
    for(int block=0;block<keep;++block) {
        const int lo=int(int64_t(block)*count/keep),hi=int(int64_t(block+1)*count/keep),size=hi-lo;
        if(!importance) {
            std::uniform_int_distribution<int>pick(lo,hi-1);
            rows.push_back(population[pick(random)]);weights.push_back(double(size));
            continue;
        }
        mass.assign(size,0.);double scale=0;
        for(int at=lo;at<hi;++at)for(int j=0;j<d;++j)
            scale=std::max(scale,std::abs(g[size_t(population[at])*d+j]));
        double sum=0;
        for(int at=lo;at<hi;++at) {
            const double*gy=g+size_t(population[at])*d;
            double squared=0;
            if(scale)for(int j=0;j<d;++j){const double x=gy[j]/scale;squared+=x*x;}
            mass[at-lo]=std::sqrt(squared);sum+=mass[at-lo];
        }
        // A 10% uniform floor bounds every weight by ten stratum sizes and
        // keeps support even for zero/underflowed norm estimates.
        for(auto&x:mass)x=sum ? .1/size+.9*x/sum : 1./size;
        const double u=std::generate_canonical<double,53>(random);double cumulative=0;
        int choice=size-1;
        for(int i=0;i<size;++i){cumulative+=mass[i];if(u<cumulative){choice=i;break;}}
        rows.push_back(population[lo+choice]);weights.push_back(1./mass[choice]);
    }
}
} // namespace rosa_random
