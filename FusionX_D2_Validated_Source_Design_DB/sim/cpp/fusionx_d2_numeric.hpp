#pragma once
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>
namespace fusionx {
enum class DType : uint8_t { INT4=0, FP4_E2M1=1, INT8=2, FP8_E4M3FN=3, INT16=4, BF16=5 };
int64_t rne_shift(int64_t x, unsigned shift);
std::optional<int32_t> decode_q16_16(DType dtype, uint16_t raw);
int16_t decode_q8_8(DType dtype, uint16_t raw, uint16_t scale_q2_14=0x4000);
int16_t requantize(int64_t acc,uint16_t scale_q2_14,unsigned shift);
std::vector<std::vector<int16_t>> gemm(const std::vector<std::vector<int16_t>>&a,const std::vector<std::vector<int16_t>>&b,uint16_t scale=0x4000,unsigned shift=0);
std::vector<std::pair<unsigned,unsigned>> output_sequence(unsigned rows,unsigned cols,unsigned beat_elems=32);
}
