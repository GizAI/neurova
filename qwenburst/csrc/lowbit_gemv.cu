#include "kernels.cuh"
#include <ATen/cuda/CUDAContext.h>

template<int BITS>
__device__ __forceinline__ int qb_lowbit_value(const uint8_t* row, int c) {
  constexpr int QMAX = (1 << BITS) - 1;
  constexpr int ZERO = (1 << (BITS - 1));
  const int bit_pos = c * BITS;
  const int byte_i = bit_pos >> 3;
  const int shift = bit_pos & 7;
  uint16_t word = row[byte_i];
  if (shift + BITS > 8) word |= static_cast<uint16_t>(row[byte_i + 1]) << 8;
  return static_cast<int>((word >> shift) & QMAX) - ZERO;
}

template<int BITS, int BLOCK, int ROWS_PER_CTA>
__global__ void lowbit_gemv_kernel(
    const uint8_t* __restrict__ qweight,
    const half* __restrict__ scales,
    const half* __restrict__ x,
    half* __restrict__ y,
    int rows,
    int cols,
    int packed_cols,
    int n_groups,
    int group_size) {
  const int row0 = blockIdx.x * ROWS_PER_CTA;
  const int tid = threadIdx.x;
  float acc[ROWS_PER_CTA];
  #pragma unroll
  for (int r = 0; r < ROWS_PER_CTA; ++r) acc[r] = 0.0f;

  for (int c = tid; c < cols; c += BLOCK) {
    const float xv = __half2float(x[c]);
    const int group = c / group_size;
    #pragma unroll
    for (int rr = 0; rr < ROWS_PER_CTA; ++rr) {
      const int row = row0 + rr;
      if (row < rows) {
        const uint8_t* qw = qweight + static_cast<int64_t>(row) * packed_cols;
        const int q = qb_lowbit_value<BITS>(qw, c);
        const float s = __half2float(scales[static_cast<int64_t>(row) * n_groups + group]);
        acc[rr] += static_cast<float>(q) * s * xv;
      }
    }
  }

  __shared__ float smem[ROWS_PER_CTA][BLOCK];
  #pragma unroll
  for (int rr = 0; rr < ROWS_PER_CTA; ++rr) smem[rr][tid] = acc[rr];
  __syncthreads();

  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      #pragma unroll
      for (int rr = 0; rr < ROWS_PER_CTA; ++rr) {
        smem[rr][tid] += smem[rr][tid + stride];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    #pragma unroll
    for (int rr = 0; rr < ROWS_PER_CTA; ++rr) {
      const int row = row0 + rr;
      if (row < rows) y[row] = __float2half_rn(smem[rr][0]);
    }
  }
}

template<int BITS, int BLOCK, int ROWS_PER_CTA>
__global__ void lowbit_gemv_pair_kernel(
    const uint8_t* __restrict__ qweight_a,
    const half* __restrict__ scales_a,
    const uint8_t* __restrict__ qweight_b,
    const half* __restrict__ scales_b,
    const half* __restrict__ x,
    half* __restrict__ y_a,
    half* __restrict__ y_b,
    int rows,
    int cols,
    int packed_cols,
    int n_groups,
    int group_size) {
  const int row0 = blockIdx.x * ROWS_PER_CTA;
  const int tid = threadIdx.x;
  float acc_a[ROWS_PER_CTA];
  float acc_b[ROWS_PER_CTA];
  #pragma unroll
  for (int r = 0; r < ROWS_PER_CTA; ++r) {
    acc_a[r] = 0.0f;
    acc_b[r] = 0.0f;
  }

  for (int c = tid; c < cols; c += BLOCK) {
    const float xv = __half2float(x[c]);
    const int group = c / group_size;
    #pragma unroll
    for (int rr = 0; rr < ROWS_PER_CTA; ++rr) {
      const int row = row0 + rr;
      if (row < rows) {
        const int64_t qoff = static_cast<int64_t>(row) * packed_cols;
        const int64_t soff = static_cast<int64_t>(row) * n_groups + group;
        const int qa = qb_lowbit_value<BITS>(qweight_a + qoff, c);
        const int qb = qb_lowbit_value<BITS>(qweight_b + qoff, c);
        acc_a[rr] += static_cast<float>(qa) * __half2float(scales_a[soff]) * xv;
        acc_b[rr] += static_cast<float>(qb) * __half2float(scales_b[soff]) * xv;
      }
    }
  }

  __shared__ float smem_a[ROWS_PER_CTA][BLOCK];
  __shared__ float smem_b[ROWS_PER_CTA][BLOCK];
  #pragma unroll
  for (int rr = 0; rr < ROWS_PER_CTA; ++rr) {
    smem_a[rr][tid] = acc_a[rr];
    smem_b[rr][tid] = acc_b[rr];
  }
  __syncthreads();

  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      #pragma unroll
      for (int rr = 0; rr < ROWS_PER_CTA; ++rr) {
        smem_a[rr][tid] += smem_a[rr][tid + stride];
        smem_b[rr][tid] += smem_b[rr][tid + stride];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    #pragma unroll
    for (int rr = 0; rr < ROWS_PER_CTA; ++rr) {
      const int row = row0 + rr;
      if (row < rows) {
        y_a[row] = __float2half_rn(smem_a[rr][0]);
        y_b[row] = __float2half_rn(smem_b[rr][0]);
      }
    }
  }
}

template<int BITS>
__global__ void lowbit_row_dequant_kernel(
    const uint8_t* __restrict__ qweight,
    const half* __restrict__ scales,
    const int64_t* __restrict__ row_ptr,
    half* __restrict__ out,
    int rows,
    int cols,
    int packed_cols,
    int n_groups,
    int group_size) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cols) return;
  int row = static_cast<int>(row_ptr[0]);
  if (row < 0 || row >= rows) {
    out[c] = __float2half_rn(0.0f);
    return;
  }
  const uint8_t* qw = qweight + static_cast<int64_t>(row) * packed_cols;
  int q = qb_lowbit_value<BITS>(qw, c);
  float s = __half2float(scales[static_cast<int64_t>(row) * n_groups + c / group_size]);
  out[c] = __float2half_rn(static_cast<float>(q) * s);
}

static void check_lowbit_args(
    torch::Tensor qweight,
    torch::Tensor scales,
    int64_t cols,
    int64_t group_size,
    int64_t bits) {
  QB_CHECK_CUDA(qweight); QB_CHECK_CUDA(scales);
  QB_CHECK_CONTIGUOUS(qweight); QB_CHECK_CONTIGUOUS(scales);
  QB_CHECK_UINT8(qweight); QB_CHECK_HALF(scales);
  TORCH_CHECK(qweight.dim() == 2, "qweight must be [rows, packed_cols]");
  TORCH_CHECK(scales.dim() == 2, "scales must be [rows, n_groups]");
  TORCH_CHECK(cols > 0 && group_size > 0, "cols and group_size must be positive");
  TORCH_CHECK(bits >= 2 && bits <= 8, "bits must be in [2, 8]");
  int rows = static_cast<int>(qweight.size(0));
  int n_groups = static_cast<int>((cols + group_size - 1) / group_size);
  int packed_need = static_cast<int>((cols * bits + 7) / 8);
  TORCH_CHECK(scales.size(0) == rows, "scales rows mismatch");
  TORCH_CHECK(scales.size(1) >= n_groups, "scales group count too small");
  TORCH_CHECK(qweight.size(1) >= packed_need, "qweight packed_cols too small");
}

template<int BITS, int ROWS_PER_CTA>
torch::Tensor launch_lowbit_gemv(torch::Tensor qweight, torch::Tensor scales, torch::Tensor x, int64_t cols, int64_t group_size) {
  QB_CHECK_CUDA(x); QB_CHECK_CONTIGUOUS(x); QB_CHECK_HALF(x);
  TORCH_CHECK(x.dim() == 1, "x must be [cols]");
  TORCH_CHECK(x.size(0) >= cols, "x cols mismatch");
  const int rows = static_cast<int>(qweight.size(0));
  const int packed_cols = static_cast<int>(qweight.size(1));
  const int n_groups = static_cast<int>((cols + group_size - 1) / group_size);
  auto y = torch::empty({rows}, x.options());

  constexpr int BLOCK = 256;
  dim3 grid((rows + ROWS_PER_CTA - 1) / ROWS_PER_CTA);
  auto stream = at::cuda::getCurrentCUDAStream();
  lowbit_gemv_kernel<BITS, BLOCK, ROWS_PER_CTA><<<grid, BLOCK, 0, stream>>>(
      qweight.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<half*>(y.data_ptr<at::Half>()),
      rows, static_cast<int>(cols), packed_cols, n_groups, static_cast<int>(group_size));
  QB_CUDA_CHECK(cudaGetLastError());
  return y;
}

template<int BITS>
torch::Tensor dispatch_lowbit_gemv_rows(torch::Tensor qweight, torch::Tensor scales, torch::Tensor x, int64_t cols, int64_t group_size, int64_t rows_per_cta) {
  switch (rows_per_cta) {
    case 4: return launch_lowbit_gemv<BITS, 4>(qweight, scales, x, cols, group_size);
    case 8: return launch_lowbit_gemv<BITS, 8>(qweight, scales, x, cols, group_size);
    case 16: return launch_lowbit_gemv<BITS, 16>(qweight, scales, x, cols, group_size);
    default: TORCH_CHECK(false, "rows_per_cta must be one of 4, 8, 16");
  }
}

torch::Tensor lowbit_gemv(torch::Tensor qweight, torch::Tensor scales, torch::Tensor x, int64_t cols, int64_t group_size, int64_t bits, int64_t rows_per_cta) {
  check_lowbit_args(qweight, scales, cols, group_size, bits);
  switch (bits) {
    case 2: return dispatch_lowbit_gemv_rows<2>(qweight, scales, x, cols, group_size, rows_per_cta);
    case 3: return dispatch_lowbit_gemv_rows<3>(qweight, scales, x, cols, group_size, rows_per_cta);
    case 4: return dispatch_lowbit_gemv_rows<4>(qweight, scales, x, cols, group_size, rows_per_cta);
    case 5: return dispatch_lowbit_gemv_rows<5>(qweight, scales, x, cols, group_size, rows_per_cta);
    case 6: return dispatch_lowbit_gemv_rows<6>(qweight, scales, x, cols, group_size, rows_per_cta);
    case 7: return dispatch_lowbit_gemv_rows<7>(qweight, scales, x, cols, group_size, rows_per_cta);
    case 8: return dispatch_lowbit_gemv_rows<8>(qweight, scales, x, cols, group_size, rows_per_cta);
    default: TORCH_CHECK(false, "bits must be in [2, 8]");
  }
}

template<int BITS, int ROWS_PER_CTA>
std::vector<torch::Tensor> launch_lowbit_gemv_pair(
    torch::Tensor qweight_a,
    torch::Tensor scales_a,
    torch::Tensor qweight_b,
    torch::Tensor scales_b,
    torch::Tensor x,
    int64_t cols,
    int64_t group_size) {
  QB_CHECK_CUDA(x); QB_CHECK_CONTIGUOUS(x); QB_CHECK_HALF(x);
  QB_CHECK_CUDA(qweight_b); QB_CHECK_CUDA(scales_b);
  QB_CHECK_CONTIGUOUS(qweight_b); QB_CHECK_CONTIGUOUS(scales_b);
  QB_CHECK_UINT8(qweight_b); QB_CHECK_HALF(scales_b);
  TORCH_CHECK(qweight_a.sizes() == qweight_b.sizes(), "pair qweight shapes must match");
  TORCH_CHECK(scales_a.sizes() == scales_b.sizes(), "pair scale shapes must match");
  TORCH_CHECK(x.dim() == 1, "x must be [cols]");
  TORCH_CHECK(x.size(0) >= cols, "x cols mismatch");
  const int rows = static_cast<int>(qweight_a.size(0));
  const int packed_cols = static_cast<int>(qweight_a.size(1));
  const int n_groups = static_cast<int>((cols + group_size - 1) / group_size);
  auto y_a = torch::empty({rows}, x.options());
  auto y_b = torch::empty({rows}, x.options());

  constexpr int BLOCK = 256;
  dim3 grid((rows + ROWS_PER_CTA - 1) / ROWS_PER_CTA);
  auto stream = at::cuda::getCurrentCUDAStream();
  lowbit_gemv_pair_kernel<BITS, BLOCK, ROWS_PER_CTA><<<grid, BLOCK, 0, stream>>>(
      qweight_a.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(scales_a.data_ptr<at::Half>()),
      qweight_b.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(scales_b.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<half*>(y_a.data_ptr<at::Half>()),
      reinterpret_cast<half*>(y_b.data_ptr<at::Half>()),
      rows, static_cast<int>(cols), packed_cols, n_groups, static_cast<int>(group_size));
  QB_CUDA_CHECK(cudaGetLastError());
  return {y_a, y_b};
}

template<int BITS>
std::vector<torch::Tensor> dispatch_lowbit_gemv_pair_rows(
    torch::Tensor qweight_a,
    torch::Tensor scales_a,
    torch::Tensor qweight_b,
    torch::Tensor scales_b,
    torch::Tensor x,
    int64_t cols,
    int64_t group_size,
    int64_t rows_per_cta) {
  switch (rows_per_cta) {
    case 4: return launch_lowbit_gemv_pair<BITS, 4>(qweight_a, scales_a, qweight_b, scales_b, x, cols, group_size);
    case 8: return launch_lowbit_gemv_pair<BITS, 8>(qweight_a, scales_a, qweight_b, scales_b, x, cols, group_size);
    case 16: return launch_lowbit_gemv_pair<BITS, 16>(qweight_a, scales_a, qweight_b, scales_b, x, cols, group_size);
    default: TORCH_CHECK(false, "rows_per_cta must be one of 4, 8, 16");
  }
}

std::vector<torch::Tensor> lowbit_gemv_pair(
    torch::Tensor qweight_a,
    torch::Tensor scales_a,
    torch::Tensor qweight_b,
    torch::Tensor scales_b,
    torch::Tensor x,
    int64_t cols,
    int64_t group_size,
    int64_t bits,
    int64_t rows_per_cta) {
  check_lowbit_args(qweight_a, scales_a, cols, group_size, bits);
  check_lowbit_args(qweight_b, scales_b, cols, group_size, bits);
  switch (bits) {
    case 2: return dispatch_lowbit_gemv_pair_rows<2>(qweight_a, scales_a, qweight_b, scales_b, x, cols, group_size, rows_per_cta);
    case 3: return dispatch_lowbit_gemv_pair_rows<3>(qweight_a, scales_a, qweight_b, scales_b, x, cols, group_size, rows_per_cta);
    case 4: return dispatch_lowbit_gemv_pair_rows<4>(qweight_a, scales_a, qweight_b, scales_b, x, cols, group_size, rows_per_cta);
    case 5: return dispatch_lowbit_gemv_pair_rows<5>(qweight_a, scales_a, qweight_b, scales_b, x, cols, group_size, rows_per_cta);
    case 6: return dispatch_lowbit_gemv_pair_rows<6>(qweight_a, scales_a, qweight_b, scales_b, x, cols, group_size, rows_per_cta);
    case 7: return dispatch_lowbit_gemv_pair_rows<7>(qweight_a, scales_a, qweight_b, scales_b, x, cols, group_size, rows_per_cta);
    case 8: return dispatch_lowbit_gemv_pair_rows<8>(qweight_a, scales_a, qweight_b, scales_b, x, cols, group_size, rows_per_cta);
    default: TORCH_CHECK(false, "bits must be in [2, 8]");
  }
}

template<int BITS>
torch::Tensor launch_lowbit_row_dequant(torch::Tensor qweight, torch::Tensor scales, torch::Tensor row, int64_t cols, int64_t group_size) {
  QB_CHECK_CUDA(row); QB_CHECK_CONTIGUOUS(row); QB_CHECK_INT64(row);
  TORCH_CHECK(row.numel() == 1, "row must be scalar int64 tensor");
  int rows = static_cast<int>(qweight.size(0));
  int packed_cols = static_cast<int>(qweight.size(1));
  int n_groups = static_cast<int>((cols + group_size - 1) / group_size);
  auto out = torch::empty({cols}, scales.options());
  constexpr int BLOCK = 256;
  dim3 grid((static_cast<int>(cols) + BLOCK - 1) / BLOCK);
  auto stream = at::cuda::getCurrentCUDAStream();
  lowbit_row_dequant_kernel<BITS><<<grid, BLOCK, 0, stream>>>(
      qweight.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      row.data_ptr<int64_t>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      rows, static_cast<int>(cols), packed_cols, n_groups, static_cast<int>(group_size));
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}

torch::Tensor lowbit_row_dequant(torch::Tensor qweight, torch::Tensor scales, torch::Tensor row, int64_t cols, int64_t group_size, int64_t bits) {
  check_lowbit_args(qweight, scales, cols, group_size, bits);
  switch (bits) {
    case 2: return launch_lowbit_row_dequant<2>(qweight, scales, row, cols, group_size);
    case 3: return launch_lowbit_row_dequant<3>(qweight, scales, row, cols, group_size);
    case 4: return launch_lowbit_row_dequant<4>(qweight, scales, row, cols, group_size);
    case 5: return launch_lowbit_row_dequant<5>(qweight, scales, row, cols, group_size);
    case 6: return launch_lowbit_row_dequant<6>(qweight, scales, row, cols, group_size);
    case 7: return launch_lowbit_row_dequant<7>(qweight, scales, row, cols, group_size);
    case 8: return launch_lowbit_row_dequant<8>(qweight, scales, row, cols, group_size);
    default: TORCH_CHECK(false, "bits must be in [2, 8]");
  }
}
