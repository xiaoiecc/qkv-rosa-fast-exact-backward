#include "exact_expansion.h"
#include <random>
#include <iostream>
int main(){std::mt19937_64 rng(23);std::uniform_real_distribution<double>unif(-1,1);uint64_t spills=0,updates=0;int maxn=0;for(int trial=0;trial<100;++trial){rosa_exact::ExactExpansion z;for(int i=0;i<10000;++i){bool old=z.spilled();z.add(unif(rng));spills+=!old&&z.spilled();++updates;maxn=std::max(maxn,int(z.component_count()));}}std::cout<<spills<<" "<<updates<<" "<<maxn<<"\n";}
