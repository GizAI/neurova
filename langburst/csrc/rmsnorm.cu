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

__global__ void silu_mul_packed_kernel(
    const half* __restrict__ mixed,
    half* __restrict__ out,
    int64_t rows,
    int64_t hidden) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t n = rows * hidden;
  if (i >= n) return;
  int64_t row = i / hidden;
  int64_t col = i - row * hidden;
  int64_t base = row * hidden * 2 + col;
  float g = __half2float(mixed[base]);
  float u = __half2float(mixed[base + hidden]);
  out[i] = __float2half_rn(qb_silu(g) * u);
}

torch::Tensor silu_mul_packed(torch::Tensor mixed, int64_t hidden) {
  QB_CHECK_CUDA(mixed);
  QB_CHECK_CONTIGUOUS(mixed);
  QB_CHECK_HALF(mixed);
  TORCH_CHECK(hidden > 0, "hidden must be > 0");
  TORCH_CHECK(mixed.dim() == 1 || mixed.dim() == 2, "mixed must be [2*hidden] or [rows, 2*hidden]");
  int64_t rows = mixed.dim() == 1 ? 1 : mixed.size(0);
  int64_t cols = mixed.dim() == 1 ? mixed.size(0) : mixed.size(1);
  TORCH_CHECK(cols == hidden * 2, "mixed last dim must be 2*hidden");
  auto out = torch::empty({rows, hidden}, mixed.options());
  const int threads = 256;
  const int64_t n = rows * hidden;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  silu_mul_packed_kernel<<<blocks, threads, 0, stream>>>(
    reinterpret_cast<const half*>(mixed.data_ptr<at::Half>()),
    reinterpret_cast<half*>(out.data_ptr<at::Half>()),
    rows,
    hidden);
  QB_CUDA_CHECK(cudaGetLastError());
  if (mixed.dim() == 1) return out.reshape({hidden}).contiguous();
  return out;
}

__global__ void sigmoid_mul_kernel(
    const half* __restrict__ x,
    const half* __restrict__ gate,
    half* __restrict__ out,
    int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float gv = __half2float(gate[i]);
  float xv = __half2float(x[i]);
  out[i] = __float2half_rn(xv / (1.0f + expf(-gv)));
}

torch::Tensor sigmoid_mul(torch::Tensor x, torch::Tensor gate) {
  QB_CHECK_CUDA(x); QB_CHECK_CUDA(gate);
  QB_CHECK_CONTIGUOUS(x); QB_CHECK_CONTIGUOUS(gate);
  QB_CHECK_HALF(x); QB_CHECK_HALF(gate);
  TORCH_CHECK(x.sizes() == gate.sizes(), "sigmoid_mul shape mismatch");
  auto out = torch::empty_like(x);
  const int threads = 256;
  const int64_t n = x.numel();
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  sigmoid_mul_kernel<<<blocks, threads, 0, stream>>>(
    reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
    reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
    reinterpret_cast<half*>(out.data_ptr<at::Half>()),
    n);
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}

__global__ void sigmoid_mul_repeat_kv_kernel(
    const half* __restrict__ v,
    const half* __restrict__ gate,
    half* __restrict__ out,
    int batch,
    int kv_heads,
    int q_heads,
    int head_dim,
    int ratio) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t n = static_cast<int64_t>(batch) * q_heads * head_dim;
  if (i >= n) return;
  int d = static_cast<int>(i % head_dim);
  int qh = static_cast<int>((i / head_dim) % q_heads);
  int b = static_cast<int>(i / (static_cast<int64_t>(q_heads) * head_dim));
  int kvh = qh / ratio;
  int64_t v_idx = (static_cast<int64_t>(b) * kv_heads + kvh) * head_dim + d;
  float gv = __half2float(gate[i]);
  float vv = __half2float(v[v_idx]);
  out[i] = __float2half_rn(vv / (1.0f + expf(-gv)));
}

torch::Tensor sigmoid_mul_repeat_kv(torch::Tensor v, torch::Tensor gate, int64_t ratio) {
  QB_CHECK_CUDA(v); QB_CHECK_CUDA(gate);
  QB_CHECK_CONTIGUOUS(v); QB_CHECK_CONTIGUOUS(gate);
  QB_CHECK_HALF(v); QB_CHECK_HALF(gate);
  TORCH_CHECK(v.dim() == 3 && gate.dim() == 3, "v and gate must be [batch, heads, head_dim]");
  TORCH_CHECK(ratio > 0, "ratio must be > 0");
  int batch = static_cast<int>(v.size(0));
  int kv_heads = static_cast<int>(v.size(1));
  int head_dim = static_cast<int>(v.size(2));
  int q_heads = static_cast<int>(gate.size(1));
  TORCH_CHECK(gate.size(0) == batch && gate.size(2) == head_dim, "gate shape mismatch");
  TORCH_CHECK(q_heads == kv_heads * ratio, "gate heads must equal kv_heads * ratio");
  auto out = torch::empty({batch, q_heads * head_dim}, v.options());
  const int threads = 256;
  const int64_t n = static_cast<int64_t>(batch) * q_heads * head_dim;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  sigmoid_mul_repeat_kv_kernel<<<blocks, threads, 0, stream>>>(
    reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
    reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
    reinterpret_cast<half*>(out.data_ptr<at::Half>()),
    batch, kv_heads, q_heads, head_dim, static_cast<int>(ratio));
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}

template<int BLOCK>
__global__ void rmsnorm_qwen_pair_cat_kernel(
    const half* __restrict__ x0,
    const half* __restrict__ w0,
    const half* __restrict__ x1,
    const half* __restrict__ w1,
    half* __restrict__ out,
    int rows,
    int hidden,
    float eps) {
  int row = blockIdx.x;
  int which = blockIdx.y;
  int tid = threadIdx.x;
  const half* x = which == 0 ? x0 : x1;
  const half* w = which == 0 ? w0 : w1;
  int64_t in_base = static_cast<int64_t>(row) * hidden;
  int64_t out_base = static_cast<int64_t>(row) * hidden * 2 + static_cast<int64_t>(which) * hidden;
  float sumsq = 0.0f;
  for (int i = tid; i < hidden; i += BLOCK) {
    float v = __half2float(x[in_base + i]);
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
    float v = __half2float(x[in_base + i]);
    float ww = __half2float(w[i]) + 1.0f;
    out[out_base + i] = __float2half_rn(v * inv_rms * ww);
  }
}

torch::Tensor rmsnorm_qwen_pair_cat(torch::Tensor x0, torch::Tensor w0, torch::Tensor x1, torch::Tensor w1, double eps) {
  QB_CHECK_CUDA(x0); QB_CHECK_CUDA(x1); QB_CHECK_CUDA(w0); QB_CHECK_CUDA(w1);
  QB_CHECK_CONTIGUOUS(x0); QB_CHECK_CONTIGUOUS(x1); QB_CHECK_CONTIGUOUS(w0); QB_CHECK_CONTIGUOUS(w1);
  QB_CHECK_HALF(x0); QB_CHECK_HALF(x1); QB_CHECK_HALF(w0); QB_CHECK_HALF(w1);
  TORCH_CHECK(x0.sizes() == x1.sizes(), "x0/x1 shape mismatch");
  TORCH_CHECK(x0.dim() == 1 || x0.dim() == 2, "x tensors must be [hidden] or [rows, hidden]");
  int hidden = static_cast<int>(x0.dim() == 1 ? x0.size(0) : x0.size(1));
  int rows = static_cast<int>(x0.dim() == 1 ? 1 : x0.size(0));
  TORCH_CHECK(w0.size(0) == hidden && w1.size(0) == hidden, "weight hidden mismatch");
  auto out = torch::empty({rows, hidden * 2}, x0.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(rows, 2);
  rmsnorm_qwen_pair_cat_kernel<256><<<grid, 256, 0, stream>>>(
    reinterpret_cast<const half*>(x0.data_ptr<at::Half>()),
    reinterpret_cast<const half*>(w0.data_ptr<at::Half>()),
    reinterpret_cast<const half*>(x1.data_ptr<at::Half>()),
    reinterpret_cast<const half*>(w1.data_ptr<at::Half>()),
    reinterpret_cast<half*>(out.data_ptr<at::Half>()),
    rows, hidden, static_cast<float>(eps));
  QB_CUDA_CHECK(cudaGetLastError());
  if (x0.dim() == 1) return out.reshape({hidden * 2}).contiguous();
  return out;
}

template<int BLOCK>
__global__ void rmsnorm_qwen_rope_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int rows,
    int hidden,
    int pos,
    int rope_dim,
    float rope_theta,
    float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) return;
  const int64_t base = static_cast<int64_t>(row) * hidden;
  float sumsq = 0.0f;
  for (int i = tid; i < hidden; i += BLOCK) {
    float v = __half2float(x[base + i]);
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
  int half_rope = rope_dim / 2;
  for (int i = tid; i < hidden; i += BLOCK) {
    float xi = __half2float(x[base + i]);
    float wi = __half2float(weight[i]) + 1.0f;
    float yi = xi * inv_rms * wi;
    if (i < rope_dim) {
      int mate = (i < half_rope) ? (i + half_rope) : (i - half_rope);
      float xm = __half2float(x[base + mate]);
      float wm = __half2float(weight[mate]) + 1.0f;
      float ym = xm * inv_rms * wm;
      float rotated = (i < half_rope) ? -ym : ym;
      int freq_i = i < half_rope ? i : (i - half_rope);
      float inv_freq = powf(rope_theta, -static_cast<float>(freq_i * 2) / static_cast<float>(rope_dim));
      float angle = static_cast<float>(pos) * inv_freq;
      float cs = cosf(angle);
      float sn = sinf(angle);
      yi = yi * cs + rotated * sn;
    }
    out[base + i] = __float2half_rn(yi);
  }
}

torch::Tensor rmsnorm_qwen_rope(torch::Tensor x, torch::Tensor weight, int64_t pos, int64_t rope_dim, double rope_theta, double eps) {
  QB_CHECK_CUDA(x); QB_CHECK_CUDA(weight);
  QB_CHECK_CONTIGUOUS(x); QB_CHECK_CONTIGUOUS(weight);
  QB_CHECK_HALF(x); QB_CHECK_HALF(weight);
  TORCH_CHECK(x.dim() == 1 || x.dim() == 2, "x must be [hidden] or [rows, hidden]");
  int hidden = static_cast<int>(x.dim() == 1 ? x.size(0) : x.size(1));
  int rows = static_cast<int>(x.dim() == 1 ? 1 : x.size(0));
  TORCH_CHECK(weight.size(0) == hidden, "weight hidden mismatch");
  TORCH_CHECK(rope_dim >= 0 && rope_dim <= hidden && rope_dim % 2 == 0, "invalid rope_dim");
  auto out = torch::empty_like(x);
  auto stream = at::cuda::getCurrentCUDAStream();
  rmsnorm_qwen_rope_kernel<256><<<rows, 256, 0, stream>>>(
    reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
    reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
    reinterpret_cast<half*>(out.data_ptr<at::Half>()),
    rows, hidden, static_cast<int>(pos), static_cast<int>(rope_dim), static_cast<float>(rope_theta), static_cast<float>(eps));
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}
