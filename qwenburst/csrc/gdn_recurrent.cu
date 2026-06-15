#include "kernels.cuh"
#include <ATen/cuda/CUDAContext.h>

// Qwen3.6/Qwen3-Next single-token recurrent Gated DeltaNet kernel.
// Specialized assumptions for the fast path:
//   q:     [kv_heads, 128] fp16, already projected/conv'd
//   k:     [kv_heads, 128] fp16, already projected/conv'd
//   v:     [v_heads, 128] fp16
//   g:     [v_heads] fp32, usually -exp(A_log) * softplus(a + dt_bias)
//   beta:  [v_heads] fp16/fp32, sigmoid(b)
//   state: [v_heads, 128, 128] fp16, updated in-place
//   out:   [v_heads, 128] fp16
// Formula implemented:
//   k,q are L2-normalized inside the kernel.
//   old_j = sum_d k[d] * S[d,j]
//   delta_j = v[j] - old_j
//   S[d,j] = exp(g) * S[d,j] + beta * k[d] * delta_j
//   out_j = sum_d q[d] * S[d,j]
// This matches the delta-rule decode shape used by Qwen's fast path well enough
// for kernel development. Exact numerical parity should be validated against HF
// FLA fused_recurrent_gated_delta_rule before production use.

constexpr int QB_GDN_D = 128;
constexpr int QB_GDN_BLOCK = 128;

__global__ void gdn_recurrent_128_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const float* __restrict__ g,
    const half* __restrict__ beta_h,
    const float* __restrict__ beta_f,
    half* __restrict__ state,
    half* __restrict__ out,
    int kv_heads,
    int v_heads) {
  int vh = blockIdx.x;
  int tid = threadIdx.x;  // one thread per value dimension
  if (vh >= v_heads || tid >= QB_GDN_D) return;

  int ratio = v_heads / kv_heads;
  int kh = vh / ratio;

  __shared__ float q_s[QB_GDN_D];
  __shared__ float k_s[QB_GDN_D];
  __shared__ float red[QB_GDN_BLOCK];
  __shared__ float delta_s[QB_GDN_D];

  float qv = __half2float(q[kh * QB_GDN_D + tid]);
  float kv = __half2float(k[kh * QB_GDN_D + tid]);
  q_s[tid] = qv;
  k_s[tid] = kv;

  float ss = qv * qv + kv * kv;  // temporary; split below by two reductions
  red[tid] = qv * qv;
  __syncthreads();
  for (int stride = QB_GDN_BLOCK / 2; stride > 0; stride >>= 1) {
    if (tid < stride) red[tid] += red[tid + stride];
    __syncthreads();
  }
  float q_inv = rsqrtf(red[0] + 1e-6f);

  red[tid] = kv * kv;
  __syncthreads();
  for (int stride = QB_GDN_BLOCK / 2; stride > 0; stride >>= 1) {
    if (tid < stride) red[tid] += red[tid + stride];
    __syncthreads();
  }
  float k_inv = rsqrtf(red[0] + 1e-6f);

  q_s[tid] *= q_inv * rsqrtf(static_cast<float>(QB_GDN_D));
  k_s[tid] *= k_inv;
  __syncthreads();

  // Each thread owns one output value dimension j=tid.
  int j = tid;
  half* S = state + static_cast<int64_t>(vh) * QB_GDN_D * QB_GDN_D;

  float old_j = 0.0f;
  #pragma unroll
  for (int d = 0; d < QB_GDN_D; ++d) {
    old_j += k_s[d] * __half2float(S[d * QB_GDN_D + j]);
  }

  float vj = __half2float(v[vh * QB_GDN_D + j]);
  float delta = vj - old_j;
  delta_s[j] = delta;
  __syncthreads();

  float decay = __expf(g[vh]);
  float beta = beta_h ? __half2float(beta_h[vh]) : beta_f[vh];

  #pragma unroll
  for (int d = 0; d < QB_GDN_D; ++d) {
    float s_old = __half2float(S[d * QB_GDN_D + j]);
    float s_new = decay * s_old + beta * k_s[d] * delta;
    S[d * QB_GDN_D + j] = __float2half_rn(s_new);
  }
  __syncthreads();

  float yj = 0.0f;
  #pragma unroll
  for (int d = 0; d < QB_GDN_D; ++d) {
    yj += q_s[d] * __half2float(S[d * QB_GDN_D + j]);
  }
  out[vh * QB_GDN_D + j] = __float2half_rn(yj);
}

torch::Tensor gdn_recurrent(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, torch::Tensor state) {
  QB_CHECK_CUDA(q); QB_CHECK_CUDA(k); QB_CHECK_CUDA(v); QB_CHECK_CUDA(g); QB_CHECK_CUDA(beta); QB_CHECK_CUDA(state);
  QB_CHECK_CONTIGUOUS(q); QB_CHECK_CONTIGUOUS(k); QB_CHECK_CONTIGUOUS(v); QB_CHECK_CONTIGUOUS(g); QB_CHECK_CONTIGUOUS(beta); QB_CHECK_CONTIGUOUS(state);
  QB_CHECK_HALF(q); QB_CHECK_HALF(k); QB_CHECK_HALF(v); QB_CHECK_HALF(state);
  TORCH_CHECK(g.scalar_type() == at::kFloat, "g must be fp32");
  TORCH_CHECK(beta.scalar_type() == at::kHalf || beta.scalar_type() == at::kFloat, "beta must be fp16 or fp32");
  TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && v.dim() == 2, "q/k/v shapes must be [heads, 128]");
  TORCH_CHECK(q.size(1) == QB_GDN_D && k.size(1) == QB_GDN_D && v.size(1) == QB_GDN_D, "head dim must be 128");
  int kv_heads = static_cast<int>(q.size(0));
  int v_heads = static_cast<int>(v.size(0));
  TORCH_CHECK(k.size(0) == kv_heads, "k heads mismatch");
  TORCH_CHECK(v_heads % kv_heads == 0, "v_heads must be divisible by kv_heads");
  TORCH_CHECK(g.size(0) == v_heads && beta.size(0) == v_heads, "g/beta v_heads mismatch");
  TORCH_CHECK(state.dim() == 3 && state.size(0) == v_heads && state.size(1) == QB_GDN_D && state.size(2) == QB_GDN_D,
              "state must be [v_heads, 128, 128]");

  auto out = torch::empty({v_heads, QB_GDN_D}, q.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const half* beta_h = beta.scalar_type() == at::kHalf ? reinterpret_cast<const half*>(beta.data_ptr<at::Half>()) : nullptr;
  const float* beta_f = beta.scalar_type() == at::kFloat ? beta.data_ptr<float>() : nullptr;
  gdn_recurrent_128_kernel<<<v_heads, QB_GDN_BLOCK, 0, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
      g.data_ptr<float>(),
      beta_h,
      beta_f,
      reinterpret_cast<half*>(state.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      kv_heads, v_heads);
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}

__global__ void gdn_recurrent_scan_128_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const float* __restrict__ g,
    const half* __restrict__ beta_h,
    const float* __restrict__ beta_f,
    half* __restrict__ state,
    half* __restrict__ out,
    int tokens,
    int kv_heads,
    int v_heads) {
  int vh = blockIdx.x;
  int tid = threadIdx.x;
  if (vh >= v_heads || tid >= QB_GDN_D) return;

  int ratio = v_heads / kv_heads;
  int kh = vh / ratio;
  half* S = state + static_cast<int64_t>(vh) * QB_GDN_D * QB_GDN_D;

  __shared__ float q_s[QB_GDN_D];
  __shared__ float k_s[QB_GDN_D];
  __shared__ float red[QB_GDN_BLOCK];
  __shared__ float delta_s[QB_GDN_D];

  for (int t = 0; t < tokens; ++t) {
    const int64_t q_off = (static_cast<int64_t>(t) * kv_heads + kh) * QB_GDN_D;
    const int64_t v_off = (static_cast<int64_t>(t) * v_heads + vh) * QB_GDN_D;
    float qv = __half2float(q[q_off + tid]);
    float kv = __half2float(k[q_off + tid]);
    q_s[tid] = qv;
    k_s[tid] = kv;

    red[tid] = qv * qv;
    __syncthreads();
    for (int stride = QB_GDN_BLOCK / 2; stride > 0; stride >>= 1) {
      if (tid < stride) red[tid] += red[tid + stride];
      __syncthreads();
    }
    float q_inv = rsqrtf(red[0] + 1e-6f);

    red[tid] = kv * kv;
    __syncthreads();
    for (int stride = QB_GDN_BLOCK / 2; stride > 0; stride >>= 1) {
      if (tid < stride) red[tid] += red[tid + stride];
      __syncthreads();
    }
    float k_inv = rsqrtf(red[0] + 1e-6f);

    q_s[tid] *= q_inv * rsqrtf(static_cast<float>(QB_GDN_D));
    k_s[tid] *= k_inv;
    __syncthreads();

    int j = tid;
    float old_j = 0.0f;
    #pragma unroll
    for (int d = 0; d < QB_GDN_D; ++d) {
      old_j += k_s[d] * __half2float(S[d * QB_GDN_D + j]);
    }

    float vj = __half2float(v[v_off + j]);
    float delta = vj - old_j;
    delta_s[j] = delta;
    __syncthreads();

    float decay = __expf(g[static_cast<int64_t>(t) * v_heads + vh]);
    float beta = beta_h ? __half2float(beta_h[static_cast<int64_t>(t) * v_heads + vh]) : beta_f[static_cast<int64_t>(t) * v_heads + vh];

    #pragma unroll
    for (int d = 0; d < QB_GDN_D; ++d) {
      float s_old = __half2float(S[d * QB_GDN_D + j]);
      float s_new = decay * s_old + beta * k_s[d] * delta;
      S[d * QB_GDN_D + j] = __float2half_rn(s_new);
    }
    __syncthreads();

    float yj = 0.0f;
    #pragma unroll
    for (int d = 0; d < QB_GDN_D; ++d) {
      yj += q_s[d] * __half2float(S[d * QB_GDN_D + j]);
    }
    out[v_off + j] = __float2half_rn(yj);
    __syncthreads();
  }
}

torch::Tensor gdn_recurrent_scan(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, torch::Tensor state) {
  QB_CHECK_CUDA(q); QB_CHECK_CUDA(k); QB_CHECK_CUDA(v); QB_CHECK_CUDA(g); QB_CHECK_CUDA(beta); QB_CHECK_CUDA(state);
  QB_CHECK_CONTIGUOUS(q); QB_CHECK_CONTIGUOUS(k); QB_CHECK_CONTIGUOUS(v); QB_CHECK_CONTIGUOUS(g); QB_CHECK_CONTIGUOUS(beta); QB_CHECK_CONTIGUOUS(state);
  QB_CHECK_HALF(q); QB_CHECK_HALF(k); QB_CHECK_HALF(v); QB_CHECK_HALF(state);
  TORCH_CHECK(g.scalar_type() == at::kFloat, "g must be fp32");
  TORCH_CHECK(beta.scalar_type() == at::kHalf || beta.scalar_type() == at::kFloat, "beta must be fp16 or fp32");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q/k/v shapes must be [tokens, heads, 128]");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "q/k/v token count mismatch");
  TORCH_CHECK(q.size(2) == QB_GDN_D && k.size(2) == QB_GDN_D && v.size(2) == QB_GDN_D, "head dim must be 128");
  int tokens = static_cast<int>(q.size(0));
  int kv_heads = static_cast<int>(q.size(1));
  int v_heads = static_cast<int>(v.size(1));
  TORCH_CHECK(k.size(1) == kv_heads, "k heads mismatch");
  TORCH_CHECK(v_heads % kv_heads == 0, "v_heads must be divisible by kv_heads");
  TORCH_CHECK(g.dim() == 2 && beta.dim() == 2 && g.size(0) == tokens && beta.size(0) == tokens, "g/beta must be [tokens, v_heads]");
  TORCH_CHECK(g.size(1) == v_heads && beta.size(1) == v_heads, "g/beta v_heads mismatch");
  TORCH_CHECK(state.dim() == 3 && state.size(0) == v_heads && state.size(1) == QB_GDN_D && state.size(2) == QB_GDN_D,
              "state must be [v_heads, 128, 128]");

  auto out = torch::empty({tokens, v_heads, QB_GDN_D}, q.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const half* beta_h = beta.scalar_type() == at::kHalf ? reinterpret_cast<const half*>(beta.data_ptr<at::Half>()) : nullptr;
  const float* beta_f = beta.scalar_type() == at::kFloat ? beta.data_ptr<float>() : nullptr;
  gdn_recurrent_scan_128_kernel<<<v_heads, QB_GDN_BLOCK, 0, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
      g.data_ptr<float>(),
      beta_h,
      beta_f,
      reinterpret_cast<half*>(state.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      tokens, kv_heads, v_heads);
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}
