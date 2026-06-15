cmake_minimum_required(VERSION 3.24)
project(qwenburst LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(qwenburst_kernels STATIC
  csrc/lowbit_gemv.cu
  csrc/rmsnorm.cu
  csrc/gdn_recurrent.cu
  csrc/attention_decode.cu
  csrc/sampling.cu
)

target_include_directories(qwenburst_kernels PUBLIC csrc)
target_compile_options(qwenburst_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr>
)
