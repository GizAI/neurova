#include "fusionx_d2_numeric.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace fusionx {
int64_t rne_shift(int64_t x,unsigned shift){if(!shift)return x;bool neg=x<0;uint64_t mag=neg?uint64_t(-x):uint64_t(x);uint64_t q=mag>>shift,rem=mag&((uint64_t(1)<<shift)-1),half=uint64_t(1)<<(shift-1);if(rem>half||(rem==half&&(q&1)))++q;return neg?-int64_t(q):int64_t(q);}
std::optional<int32_t> decode_q16_16(DType dt,uint16_t raw){switch(dt){
 case DType::INT4:{int v=raw&15;if(v&8)v-=16;return v<<16;}
 case DType::FP4_E2M1:{static const int m[8]={0,32768,65536,98304,131072,196608,262144,393216};return (raw&8)?-m[raw&7]:m[raw&7];}
 case DType::INT8:{int v=raw&255;if(v&128)v-=256;return v<<16;}
 case DType::FP8_E4M3FN:{int v=raw&255,s=(v&128)?-1:1,e=(v>>3)&15,m=v&7;if(e==15&&m==7)return std::nullopt;int64_t mag=e==0?(m<<7):((8+m)<<(e+6));return int32_t(s*mag);}
 case DType::INT16:return int32_t(int16_t(raw))<<16;
 default:return std::nullopt;}}
int16_t decode_q8_8(DType dt,uint16_t raw,uint16_t scale){auto d=decode_q16_16(dt,raw);if(!d)return 0;int64_t q=rne_shift(int64_t(*d)*scale,14);q=rne_shift(q,8);q=std::clamp<int64_t>(q,-32768,32767);return int16_t(q);}
int16_t requantize(int64_t acc,uint16_t scale,unsigned shift){auto q=rne_shift(acc*scale,14+shift);return int16_t(std::clamp<int64_t>(q,-32768,32767));}
std::vector<std::vector<int16_t>> gemm(const std::vector<std::vector<int16_t>>&a,const std::vector<std::vector<int16_t>>&b,uint16_t scale,unsigned shift){if(a.empty()||b.empty()||a[0].size()!=b.size())throw std::invalid_argument("shape");std::vector<std::vector<int16_t>>y(a.size(),std::vector<int16_t>(b[0].size()));for(size_t i=0;i<a.size();++i)for(size_t j=0;j<b[0].size();++j){int64_t acc=0;for(size_t k=0;k<b.size();++k)acc+=int64_t(a[i][k])*b[k][j];y[i][j]=requantize(acc,scale,shift);}return y;}
std::vector<std::pair<unsigned,unsigned>> output_sequence(unsigned rows,unsigned cols,unsigned beat_elems){std::vector<std::pair<unsigned,unsigned>>o;unsigned beats=(cols+beat_elems-1)/beat_elems;for(unsigned r=0;r<rows;++r)for(unsigned b=0;b<beats;++b)o.emplace_back(r,b);return o;}
}
