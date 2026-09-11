#include "fusionx_d2_numeric.hpp"
#include <cassert>
#include <iostream>
using namespace fusionx;
int main(){
 assert(decode_q16_16(DType::FP4_E2M1,7).value()==6*65536);
 assert(decode_q16_16(DType::FP8_E4M3FN,0x7e).value()==448*65536);
 assert(!decode_q16_16(DType::FP8_E4M3FN,0x7f));
 assert(rne_shift(10,2)==2&&rne_shift(14,2)==4&&rne_shift(-10,2)==-2&&rne_shift(-14,2)==-4);
 std::vector<std::vector<int16_t>>a(64,std::vector<int16_t>(3)),b(3,std::vector<int16_t>(64));
 for(int i=0;i<64;++i)for(int k=0;k<3;++k)a[i][k]=int16_t((i+k)%7-3);
 for(int k=0;k<3;++k)for(int j=0;j<64;++j)b[k][j]=int16_t((j-k)%9-4);
 auto y=gemm(a,b);assert(y.size()==64&&y[0].size()==64);
 auto seq=output_sequence(64,64);assert(seq.size()==128);for(size_t i=0;i<seq.size();++i){assert(seq[i].first==i/2);assert(seq[i].second==i%2);} 
 std::cout<<"PASS FusionX D2 C++ numeric\n";
}
