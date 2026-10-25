#include "exact_expansion.h"
#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using rosa_exact::ExactExpansion;
int main(){
 std::string line;
 while(std::getline(std::cin,line)){
   std::istringstream in(line);int mode,n;in>>mode>>n;ExactExpansion sum,snapshot;
   for(int i=0;i<n;++i){std::string token;in>>token;double x=std::strtod(token.c_str(),nullptr);sum.add(x);if(mode>=0 && i==mode)snapshot=sum;}
   if(mode>=0)sum.subtract(snapshot);
   std::cout<<std::hexfloat<<sum.to_double()<<" "<<sum.component_count();
   sum.for_each_component([&](double x){std::cout<<" "<<std::hexfloat<<x;});std::cout<<"\n";
 }
}
