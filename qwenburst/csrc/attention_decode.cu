#include "kernels.cuh"
#include <ATen/cuda/CUDAContext.h>

// Baseline decode attention for Qwen3.6 full-attention layers.
// q:       [q_heads, head_dim] fp16, q_heads=24, head_dim=256
// k_cache: [kv_heads, max_seq, head_dim] fp16, kv_heads=4
// v_cache: [kv_heads, max_seq, head_dim] fp16
// out:     [q_heads, head_dim] fp16
//
// This is intentionally simple and graph-capturable. For 64K+ context replace
// this with paged low-bit/KVQuant/BitDecoding-style fused attention.

constexpr int QB_ATT_MAX_D = 256;
constexpr int QB_ATT_BLOCK = 256;

template<int D>
__global__ void attention_decode_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    half* __restrict__ out,
    int q_heads,
    int kv_heads,
    int max_seq,
    int seq_len,
    float scale) {
  int qh = blockIdx.x;
  int dim = threadIdx.x;
  if (qh >= q_heads || dim >= D) return;
  int ratio = q_heads / kv_heads;
  int kvh = qh / ratio;

  // Numerically stable online softmax over seq_len.
  float m = -INFINITY;
  float l = 0.0f;
  float acc = 0.0f;

  for (int t = 0; t < seq_len; ++t) {
    // One thread computes only its output dim, but all threads redundantly compute score.
    // This is okay for the baseline. The production kernel should split score reduction
    // and value accumulation across warps/pages.
    float score_part = 0.0f;
    for (int d = 0; d < D; ++d) {
      float qv = __half2float(q[qh * D + d]);
      float kv = __half2float(k_cache[(static_cast<int64_t>(kvh) * max_seq + t) * D + d]);
      score_part += qv * kv;
    }
    float s = score_part * scale;
    float new_m = fmaxf(m, s);
    float alpha = __expf(m - new_m);
    float p = __expf(s - new_m);
    float vv = __half2float(v_cache[(static_cast<int64_t>(kvh) * max_seq + t) * D + dim]);
    acc = acc * alpha + p * vv;
    l = l * alpha + p;
    m = new_m;
  }
  out[qh * D + dim] = __float2half_rn(acc / fmaxf(l, 1e-20f));
}

torch::Tensor attention_decode_fp16(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, int64_t seq_len, double softmax_scale) {
  QB_CHECK_CUDA(q); QB_CHECK_CUDA(k_cache); QB_CHECK_CUDA(v_cache);
  QB_CHECK_CONTIGUOUS(q); QB_CHECK_CONTIGUOUS(k_cache); QB_CHECK_CONTIGUOUS(v_cache);
  QB_CHECK_HALF(q); QB_CHECK_HALF(k_cache); QB_CHECK_HALF(v_cache);
  TORCH_CHECK(q.dim() == 2, "q must be [q_heads, head_dim]");
  TORCH_CHECK(k_cache.dim() == 3 && v_cache.dim() == 3, "cache must be [kv_heads, max_seq, head_dim]");
  int q_heads = static_cast<int>(q.size(0));
  int kv_heads = static_cast<int>(k_cache.size(0));
  int max_seq = static_cast<int>(k_cache.size(1));
  int head_dim = static_cast<int>(q.size(1));
  TORCH_CHECK(v_cache.size(0) == kv_heads && v_cache.size(1) == max_seq && v_cache.size(2) == head_dim, "v_cache shape mismatch");
  TORCH_CHECK(k_cache.size(2) == head_dim, "k_cache dim mismatch");
  TORCH_CHECK(q_heads % kv_heads == 0, "q_heads must be divisible by kv_heads");
  TORCH_CHECK(seq_len >= 1 && seq_len <= max_seq, "seq_len out of range");
  TORCH_CHECK(head_dim == 256, "baseline kernel currently specializes head_dim=256");

  auto out = torch::empty_like(q);
  auto stream = at::cuda::getCurrentCUDAStream();
  attention_decode_kernel<256><<<q_heads, QB_ATT_BLOCK, 0, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v_cache.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      q_heads, kv_heads, max_seq, static_cast<int>(seq_len), static_cast<float>(softmax_scale));
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}
