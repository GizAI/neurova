#include "kernels.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>

// Graph-friendly GPU-side greedy sampling helpers. These avoid copying logits to
// CPU inside the hot decode/MTP verification loop.

template<int BLOCK, typename scalar_t>
__global__ void argmax_rows_kernel(const scalar_t* __restrict__ logits, int64_t rows, int64_t cols, int64_t* __restrict__ out_idx) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) return;
  const scalar_t* row_ptr = logits + static_cast<int64_t>(row) * cols;

  float best = -INFINITY;
  int64_t best_i = 0;
  for (int64_t i = tid; i < cols; i += BLOCK) {
    float v = static_cast<float>(row_ptr[i]);
    if (v > best || (v == best && i < best_i)) {
      best = v;
      best_i = i;
    }
  }

  __shared__ float vals[BLOCK];
  __shared__ int64_t idxs[BLOCK];
  vals[tid] = best;
  idxs[tid] = best_i;
  __syncthreads();
  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float ov = vals[tid + stride];
      const int64_t oi = idxs[tid + stride];
      if (ov > vals[tid] || (ov == vals[tid] && oi < idxs[tid])) {
        vals[tid] = ov;
        idxs[tid] = oi;
      }
    }
    __syncthreads();
  }
  if (tid == 0) out_idx[row] = idxs[0];
}

template<int BLOCK>
__global__ void argmax_rows_kernel_half(const half* __restrict__ logits, int64_t rows, int64_t cols, int64_t* __restrict__ out_idx) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) return;
  const half* row_ptr = logits + static_cast<int64_t>(row) * cols;

  float best = -INFINITY;
  int64_t best_i = 0;
  for (int64_t i = tid; i < cols; i += BLOCK) {
    float v = __half2float(row_ptr[i]);
    if (v > best || (v == best && i < best_i)) {
      best = v;
      best_i = i;
    }
  }

  __shared__ float vals[BLOCK];
  __shared__ int64_t idxs[BLOCK];
  vals[tid] = best;
  idxs[tid] = best_i;
  __syncthreads();
  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float ov = vals[tid + stride];
      const int64_t oi = idxs[tid + stride];
      if (ov > vals[tid] || (ov == vals[tid] && oi < idxs[tid])) {
        vals[tid] = ov;
        idxs[tid] = oi;
      }
    }
    __syncthreads();
  }
  if (tid == 0) out_idx[row] = idxs[0];
}

template<int BLOCK>
__global__ void prefix_match_kernel(const int64_t* __restrict__ proposed, const int64_t* __restrict__ verified, int64_t n, int64_t* __restrict__ out) {
  // One block is enough: MTP depth is tiny, usually <= 4.
  if (threadIdx.x == 0) {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
      if (proposed[i] != verified[i]) break;
      ++count;
    }
    *out = count;
  }
}

void argmax_many_out(torch::Tensor logits, torch::Tensor out) {
  QB_CHECK_CUDA(logits); QB_CHECK_CONTIGUOUS(logits);
  QB_CHECK_CUDA(out); QB_CHECK_CONTIGUOUS(out); QB_CHECK_INT64(out);
  TORCH_CHECK(logits.dim() == 1 || logits.dim() == 2, "logits must be 1D or 2D");
  const int64_t rows = logits.dim() == 1 ? 1 : logits.size(0);
  const int64_t cols = logits.dim() == 1 ? logits.size(0) : logits.size(1);
  TORCH_CHECK(out.numel() >= rows, "out must have at least one int64 per logits row");
  constexpr int BLOCK = 1024;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (logits.scalar_type() == at::kHalf) {
    dim3 grid(static_cast<unsigned int>(rows));
    argmax_rows_kernel_half<BLOCK><<<grid, BLOCK, 0, stream>>>(
      reinterpret_cast<const half*>(logits.data_ptr<at::Half>()), rows, cols, out.data_ptr<int64_t>());
  } else if (logits.scalar_type() == at::kFloat) {
    dim3 grid(static_cast<unsigned int>(rows));
    argmax_rows_kernel<BLOCK, float><<<grid, BLOCK, 0, stream>>>(
      logits.data_ptr<float>(), rows, cols, out.data_ptr<int64_t>());
  } else {
    TORCH_CHECK(false, "argmax_many_out supports fp16/fp32 logits only");
  }
  QB_CUDA_CHECK(cudaGetLastError());
}

torch::Tensor argmax_many(torch::Tensor logits) {
  QB_CHECK_CUDA(logits); QB_CHECK_CONTIGUOUS(logits);
  TORCH_CHECK(logits.dim() == 1 || logits.dim() == 2, "logits must be 1D or 2D");
  const int64_t rows = logits.dim() == 1 ? 1 : logits.size(0);
  auto out = torch::empty({rows}, torch::TensorOptions().dtype(torch::kInt64).device(logits.device()));
  argmax_many_out(logits, out);
  return out;
}

torch::Tensor argmax(torch::Tensor logits) {
  TORCH_CHECK(logits.dim() == 1, "logits must be 1D");
  return argmax_many(logits).slice(0, 0, 1);
}

torch::Tensor count_prefix_matches(torch::Tensor proposed, torch::Tensor verified) {
  QB_CHECK_CUDA(proposed); QB_CHECK_CUDA(verified);
  QB_CHECK_CONTIGUOUS(proposed); QB_CHECK_CONTIGUOUS(verified);
  QB_CHECK_INT64(proposed); QB_CHECK_INT64(verified);
  TORCH_CHECK(proposed.dim() == 1 && verified.dim() == 1, "proposed/verified must be 1D int64");
  const int64_t n = std::min<int64_t>(proposed.numel(), verified.numel());
  auto out = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64).device(proposed.device()));
  auto stream = at::cuda::getCurrentCUDAStream();
  prefix_match_kernel<32><<<1, 32, 0, stream>>>(proposed.data_ptr<int64_t>(), verified.data_ptr<int64_t>(), n, out.data_ptr<int64_t>());
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}
