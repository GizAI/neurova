#include "kernels.cuh"
#include <ATen/cuda/CUDAContext.h>

// Runtime dynamic RMSNorm kernels for hidden=5120 and head_dim=256.
// Qwen3.5/Qwen3.6 HF RMSNorm stores offset weights and computes x * (1+weight).

template<int BLOCK, bool GATED, bool QWEN_PLUS_ONE>
__global__ void rmsnorm_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    const half* __restrict__ gate,
    half* __restrict__ y,
    int rows,
    int hidden,
    float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) return;

  const half* xr = x + static_cast<int64_t>(row) * hidden;
  const half* gr = gate ? gate + static_cast<int64_t>(row) * hidden : nullptr;
  half* yr = y + static_cast<int64_t>(row) * hidden;

  float sumsq = 0.0f;
  for (int i = tid; i < hidden; i += BLOCK) {
    float v = __half2float(xr[i]);
    sumsq += v * v;
  }

  __shared__ float smem[BLOCK];
  smem[tid] = sumsq;
  __syncthreads();
  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (tid < stride) smem[tid] += smem[tid + stride];
    __syncthreads();
  }
  float inv_rms = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);

  for (int i = tid; i < hidden; i += BLOCK) {
    float w = __half2float(weight[i]);
    if constexpr (QWEN_PLUS_ONE) w += 1.0f;
    float out = __half2float(xr[i]) * inv_rms * w;
    if constexpr (GATED) {
      float gv = __half2float(gr[i]);
      out *= qb_silu(gv);
    }
    yr[i] = __float2half_rn(out);
  }
}

static torch::Tensor rmsnorm_impl(torch::Tensor x, torch::Tensor weight, c10::optional<torch::Tensor> gate, double eps, bool qwen_plus_one) {
  QB_CHECK_CUDA(x); QB_CHECK_CUDA(weight);
  QB_CHECK_CONTIGUOUS(x); QB_CHECK_CONTIGUOUS(weight);
  QB_CHECK_HALF(x); QB_CHECK_HALF(weight);
  TORCH_CHECK(x.dim() == 1 || x.dim() == 2, "x must be [hidden] or [rows, hidden]");
  int hidden = static_cast<int>(x.dim() == 1 ? x.size(0) : x.size(1));
  int rows = static_cast<int>(x.dim() == 1 ? 1 : x.size(0));
  TORCH_CHECK(weight.size(0) == hidden, "weight hidden mismatch");

  torch::Tensor gate_t;
  const half* gate_ptr = nullptr;
  bool gated = gate.has_value();
  if (gated) {
    gate_t = gate.value();
    QB_CHECK_CUDA(gate_t); QB_CHECK_CONTIGUOUS(gate_t); QB_CHECK_HALF(gate_t);
    TORCH_CHECK(gate_t.numel() == x.numel(), "gate shape mismatch");
    gate_ptr = reinterpret_cast<const half*>(gate_t.data_ptr<at::Half>());
  }

  auto y = torch::empty_like(x);
  constexpr int BLOCK = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (qwen_plus_one && gated) {
    rmsnorm_kernel<BLOCK, true, true><<<rows, BLOCK, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()), gate_ptr,
      reinterpret_cast<half*>(y.data_ptr<at::Half>()), rows, hidden, static_cast<float>(eps));
  } else if (qwen_plus_one) {
    rmsnorm_kernel<BLOCK, false, true><<<rows, BLOCK, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()), nullptr,
      reinterpret_cast<half*>(y.data_ptr<at::Half>()), rows, hidden, static_cast<float>(eps));
  } else if (gated) {
    rmsnorm_kernel<BLOCK, true, false><<<rows, BLOCK, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()), gate_ptr,
      reinterpret_cast<half*>(y.data_ptr<at::Half>()), rows, hidden, static_cast<float>(eps));
  } else {
    rmsnorm_kernel<BLOCK, false, false><<<rows, BLOCK, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()), nullptr,
      reinterpret_cast<half*>(y.data_ptr<at::Half>()), rows, hidden, static_cast<float>(eps));
  }
  QB_CUDA_CHECK(cudaGetLastError());
  return y;
}

torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor weight, double eps) {
  return rmsnorm_impl(x, weight, c10::nullopt, eps, false);
}

torch::Tensor rmsnorm_qwen(torch::Tensor x, torch::Tensor weight, double eps) {
  return rmsnorm_impl(x, weight, c10::nullopt, eps, true);
}

torch::Tensor rmsnorm_silu_gate(torch::Tensor x, torch::Tensor weight, torch::Tensor gate, double eps) {
  return rmsnorm_impl(x, weight, gate, eps, false);
}

torch::Tensor rmsnorm_qwen_silu_gate(torch::Tensor x, torch::Tensor weight, torch::Tensor gate, double eps) {
  return rmsnorm_impl(x, weight, gate, eps, true);
}

__global__ void silu_mul_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ out,
    int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = __half2float(gate[i]);
  float u = __half2float(up[i]);
  out[i] = __float2half_rn(qb_silu(g) * u);
}

torch::Tensor silu_mul(torch::Tensor gate, torch::Tensor up) {
  QB_CHECK_CUDA(gate); QB_CHECK_CUDA(up);
  QB_CHECK_CONTIGUOUS(gate); QB_CHECK_CONTIGUOUS(up);
  QB_CHECK_HALF(gate); QB_CHECK_HALF(up);
  TORCH_CHECK(gate.sizes() == up.sizes(), "silu_mul shape mismatch");
  auto out = torch::empty_like(gate);
  const int threads = 256;
  const int64_t n = gate.numel();
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  silu_mul_kernel<<<blocks, threads, 0, stream>>>(
    reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
    reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
    reinterpret_cast<half*>(out.data_ptr<at::Half>()),
    n);
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}
