#include "kernels.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

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

__device__ __forceinline__ float qb_fp8_e4m3_decode(uint8_t code) {
  int sign = (code >> 7) & 1;
  int exp = (code >> 3) & 0xF;
  int mant = code & 0x7;
  float val;
  if (exp == 0) {
    val = static_cast<float>(mant) * 0x1p-9f;
  } else {
    val = (1.0f + static_cast<float>(mant) * 0.125f) * exp2f(static_cast<float>(exp - 7));
  }
  return sign ? -val : val;
}

__device__ __forceinline__ uint8_t qb_fp8_e4m3_quant(float x) {
  if (!isfinite(x) || x == 0.0f) return 0;
  int sign = x < 0.0f ? 0x80 : 0;
  float ax = fabsf(x);
  ax = fminf(ax, 448.0f);
  if (ax < 0x1p-6f) {
    int mant = static_cast<int>(floorf(ax * 512.0f + 0.5f));
    mant = max(0, min(7, mant));
    return static_cast<uint8_t>(sign | mant);
  }
  int e = static_cast<int>(floorf(log2f(ax)));
  e = max(-6, min(8, e));
  float base = exp2f(static_cast<float>(e));
  int mant = static_cast<int>(floorf((ax / base - 1.0f) * 8.0f + 0.5f));
  if (mant >= 8) {
    mant = 0;
    e += 1;
  }
  if (e > 8) {
    e = 8;
    mant = 6;
  }
  int exp = e + 7;
  if (exp >= 15 && mant > 6) mant = 6;
  return static_cast<uint8_t>(sign | ((exp & 0xF) << 3) | (mant & 0x7));
}

__device__ __forceinline__ int qb_hadamard_sign(int row, int col) {
  return (__popc(static_cast<unsigned>(row & col)) & 1) ? -1 : 1;
}

template<int D>
__device__ __forceinline__ float qb_bdr_rotate_value(const half* __restrict__ x, int base, int d, int order) {
  if (order <= 1) return __half2float(x[base + d]);
  int group = d / order;
  int local = d - group * order;
  int start = group * order;
  float sum = 0.0f;
  for (int i = 0; i < order; ++i) {
    sum += static_cast<float>(qb_hadamard_sign(local, i)) * __half2float(x[base + start + i]);
  }
  return sum * rsqrtf(static_cast<float>(order));
}

__device__ __forceinline__ uint8_t qb_pack_uint4_pair(int low, int high) {
  low = max(0, min(15, low));
  high = max(0, min(15, high));
  return static_cast<uint8_t>((low & 0xF) | ((high & 0xF) << 4));
}

__device__ __forceinline__ float qb_unpack_uint4(uint8_t packed, bool high_half, float scale, float zero) {
  int nibble = high_half ? ((packed >> 4) & 0xF) : (packed & 0xF);
  return (static_cast<float>(nibble) - zero) * scale;
}

__device__ __forceinline__ int64_t qb_int4_paged_offset(
    int block,
    int kvh,
    int kv_heads,
    int block_size,
    int packed_dim_count,
    int offset,
    int packed_dim,
    bool tiled_layout) {
  if (tiled_layout) {
    return (((static_cast<int64_t>(block) * kv_heads + kvh) * packed_dim_count + packed_dim) * block_size) + offset;
  }
  return (((static_cast<int64_t>(block) * kv_heads + kvh) * block_size + offset) * packed_dim_count) + packed_dim;
}

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

  __shared__ float qk[QB_ATT_BLOCK];

  // Numerically stable online softmax over seq_len. One block owns one query
  // head. Threads reduce the QK dot product once per timestep, then each thread
  // updates one output dimension. The old baseline recomputed the same dot
  // product independently for every output dimension.
  float m = -INFINITY;
  float l = 0.0f;
  float acc = 0.0f;

  for (int t = 0; t < seq_len; ++t) {
    float prod = 0.0f;
    if (dim < D) {
      prod = __half2float(q[qh * D + dim])
        * __half2float(k_cache[(static_cast<int64_t>(kvh) * max_seq + t) * D + dim]);
    }
    qk[dim] = prod;
    __syncthreads();
    for (int stride = D / 2; stride > 0; stride >>= 1) {
      if (dim < stride) qk[dim] += qk[dim + stride];
      __syncthreads();
    }
    float s = qk[0] * scale;
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

template<int D>
__global__ void attention_kv_append_batch_kernel(
    const half* __restrict__ k_new,
    const half* __restrict__ v_new,
    half* __restrict__ k_arena,
    half* __restrict__ v_arena,
    const int64_t* __restrict__ state_indices,
    const int64_t* __restrict__ write_indices,
    int batch,
    int slots,
    int kv_heads,
    int max_seq) {
  int linear = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * kv_heads * D;
  if (linear >= total) return;
  int d = linear % D;
  int tmp = linear / D;
  int kvh = tmp % kv_heads;
  int row = tmp / kv_heads;
  int64_t slot64 = state_indices[row];
  int64_t idx64 = write_indices[row];
  if (slot64 < 0 || slot64 >= slots || idx64 < 0 || idx64 >= max_seq) return;
  int64_t src = (static_cast<int64_t>(row) * kv_heads + kvh) * D + d;
  int64_t dst = ((slot64 * kv_heads + kvh) * max_seq + idx64) * D + d;
  k_arena[dst] = k_new[src];
  v_arena[dst] = v_new[src];
}

template<int D>
__global__ void attention_decode_batch_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k_arena,
    const half* __restrict__ v_arena,
    const int64_t* __restrict__ state_indices,
    const int64_t* __restrict__ live_lengths,
    const int64_t* __restrict__ positions,
    half* __restrict__ out,
    int batch,
    int slots,
    int q_heads,
    int kv_heads,
    int max_seq,
    bool use_ring,
    float scale) {
  int qh = blockIdx.x;
  int row = blockIdx.y;
  int dim = threadIdx.x;
  if (row >= batch || qh >= q_heads || dim >= D) return;
  int64_t slot64 = state_indices[row];
  if (slot64 < 0 || slot64 >= slots) return;
  int slot = static_cast<int>(slot64);
  int seq_len = static_cast<int>(live_lengths[row]);
  if (seq_len < 1) return;
  if (seq_len > max_seq) seq_len = max_seq;
  int start = 0;
  if (use_ring && seq_len == max_seq) {
    start = static_cast<int>((positions[row] + 1) % max_seq);
  }

  int ratio = q_heads / kv_heads;
  int kvh = qh / ratio;
  __shared__ float qk[QB_ATT_BLOCK];

  float m = -INFINITY;
  float l = 0.0f;
  float acc = 0.0f;
  const int64_t q_base = (static_cast<int64_t>(row) * q_heads + qh) * D;
  const int64_t cache_base = (static_cast<int64_t>(slot) * kv_heads + kvh) * max_seq * D;

  for (int t = 0; t < seq_len; ++t) {
    int phys = start == 0 ? t : ((start + t) % max_seq);
    float prod = __half2float(q[q_base + dim]) * __half2float(k_arena[cache_base + static_cast<int64_t>(phys) * D + dim]);
    qk[dim] = prod;
    __syncthreads();
    for (int stride = D / 2; stride > 0; stride >>= 1) {
      if (dim < stride) qk[dim] += qk[dim + stride];
      __syncthreads();
    }
    float s = qk[0] * scale;
    float new_m = fmaxf(m, s);
    float alpha = __expf(m - new_m);
    float p = __expf(s - new_m);
    float vv = __half2float(v_arena[cache_base + static_cast<int64_t>(phys) * D + dim]);
    acc = acc * alpha + p * vv;
    l = l * alpha + p;
    m = new_m;
  }
  out[q_base + dim] = __float2half_rn(acc / fmaxf(l, 1e-20f));
}

torch::Tensor attention_decode_batch_fp16(
    torch::Tensor q,
    torch::Tensor k_new,
    torch::Tensor v_new,
    torch::Tensor k_arena,
    torch::Tensor v_arena,
    torch::Tensor state_indices,
    torch::Tensor write_indices,
    torch::Tensor live_lengths,
    torch::Tensor positions,
    bool use_ring,
    double softmax_scale) {
  QB_CHECK_CUDA(q); QB_CHECK_CUDA(k_new); QB_CHECK_CUDA(v_new); QB_CHECK_CUDA(k_arena); QB_CHECK_CUDA(v_arena);
  QB_CHECK_CUDA(state_indices); QB_CHECK_CUDA(write_indices); QB_CHECK_CUDA(live_lengths); QB_CHECK_CUDA(positions);
  QB_CHECK_CONTIGUOUS(q); QB_CHECK_CONTIGUOUS(k_new); QB_CHECK_CONTIGUOUS(v_new); QB_CHECK_CONTIGUOUS(k_arena); QB_CHECK_CONTIGUOUS(v_arena);
  QB_CHECK_CONTIGUOUS(state_indices); QB_CHECK_CONTIGUOUS(write_indices); QB_CHECK_CONTIGUOUS(live_lengths); QB_CHECK_CONTIGUOUS(positions);
  QB_CHECK_HALF(q); QB_CHECK_HALF(k_new); QB_CHECK_HALF(v_new); QB_CHECK_HALF(k_arena); QB_CHECK_HALF(v_arena);
  QB_CHECK_INT64(state_indices); QB_CHECK_INT64(write_indices); QB_CHECK_INT64(live_lengths); QB_CHECK_INT64(positions);
  TORCH_CHECK(q.dim() == 3, "q must be [batch, q_heads, head_dim]");
  TORCH_CHECK(k_new.dim() == 3 && v_new.dim() == 3, "k_new/v_new must be [batch, kv_heads, head_dim]");
  TORCH_CHECK(k_arena.dim() == 4 && v_arena.dim() == 4, "arena cache must be [slots, kv_heads, max_seq, head_dim]");
  int batch = static_cast<int>(q.size(0));
  int q_heads = static_cast<int>(q.size(1));
  int head_dim = static_cast<int>(q.size(2));
  int kv_heads = static_cast<int>(k_new.size(1));
  int slots = static_cast<int>(k_arena.size(0));
  int max_seq = static_cast<int>(k_arena.size(2));
  TORCH_CHECK(head_dim == 256, "batch attention kernel currently specializes head_dim=256");
  TORCH_CHECK(k_new.size(0) == batch && v_new.size(0) == batch, "k_new/v_new batch mismatch");
  TORCH_CHECK(k_new.size(2) == head_dim && v_new.size(2) == head_dim, "k_new/v_new dim mismatch");
  TORCH_CHECK(k_arena.size(1) == kv_heads && v_arena.size(1) == kv_heads, "arena kv_heads mismatch");
  TORCH_CHECK(k_arena.size(3) == head_dim && v_arena.size(3) == head_dim, "arena head_dim mismatch");
  TORCH_CHECK(v_arena.size(0) == slots && v_arena.size(2) == max_seq, "v_arena shape mismatch");
  TORCH_CHECK(q_heads % kv_heads == 0, "q_heads must be divisible by kv_heads");
  TORCH_CHECK(state_indices.size(0) == batch && write_indices.size(0) == batch && live_lengths.size(0) == batch && positions.size(0) == batch,
              "state_indices/write_indices/live_lengths/positions must be [batch]");

  auto out = torch::empty_like(q);
  auto stream = at::cuda::getCurrentCUDAStream();
  int threads = 256;
  int append_total = batch * kv_heads * head_dim;
  attention_kv_append_batch_kernel<256><<<(append_total + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<const half*>(k_new.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v_new.data_ptr<at::Half>()),
      reinterpret_cast<half*>(k_arena.data_ptr<at::Half>()),
      reinterpret_cast<half*>(v_arena.data_ptr<at::Half>()),
      state_indices.data_ptr<int64_t>(),
      write_indices.data_ptr<int64_t>(),
      batch, slots, kv_heads, max_seq);
  QB_CUDA_CHECK(cudaGetLastError());
  dim3 grid(q_heads, batch);
  attention_decode_batch_kernel<256><<<grid, QB_ATT_BLOCK, 0, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k_arena.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v_arena.data_ptr<at::Half>()),
      state_indices.data_ptr<int64_t>(),
      live_lengths.data_ptr<int64_t>(),
      positions.data_ptr<int64_t>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      batch, slots, q_heads, kv_heads, max_seq, use_ring, static_cast<float>(softmax_scale));
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}

template<int D>
__global__ void attention_kv_append_paged_kernel(
    const half* __restrict__ k_new,
    const half* __restrict__ v_new,
    half* __restrict__ k_pages,
    half* __restrict__ v_pages,
    const int64_t* __restrict__ slot_mapping,
    int batch,
    int num_blocks,
    int kv_heads,
    int block_size) {
  int linear = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * kv_heads * D;
  if (linear >= total) return;
  int d = linear % D;
  int tmp = linear / D;
  int kvh = tmp % kv_heads;
  int row = tmp / kv_heads;
  int64_t slot = slot_mapping[row];
  int64_t block = slot / block_size;
  int offset = static_cast<int>(slot % block_size);
  if (block < 0 || block >= num_blocks || offset < 0 || offset >= block_size) return;
  int64_t src = (static_cast<int64_t>(row) * kv_heads + kvh) * D + d;
  int64_t dst = ((block * kv_heads + kvh) * block_size + offset) * D + d;
  k_pages[dst] = k_new[src];
  v_pages[dst] = v_new[src];
}

template<int D>
__global__ void attention_decode_paged_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k_pages,
    const half* __restrict__ v_pages,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    half* __restrict__ out,
    int batch,
    int q_heads,
    int kv_heads,
    int max_blocks_per_row,
    int block_size,
    float scale) {
  int qh = blockIdx.x;
  int row = blockIdx.y;
  int dim = threadIdx.x;
  if (row >= batch || qh >= q_heads || dim >= D) return;
  int seq_len = seq_lens[row];
  if (seq_len < 1) return;
  int ratio = q_heads / kv_heads;
  int kvh = qh / ratio;
  __shared__ float qk[QB_ATT_BLOCK];

  float m = -INFINITY;
  float l = 0.0f;
  float acc = 0.0f;
  const int64_t q_base = (static_cast<int64_t>(row) * q_heads + qh) * D;

  for (int t = 0; t < seq_len; ++t) {
    int block_idx = t / block_size;
    int offset = t - block_idx * block_size;
    int block_id = block_tables[static_cast<int64_t>(row) * max_blocks_per_row + block_idx];
    int64_t cache_base = (static_cast<int64_t>(block_id) * kv_heads + kvh) * block_size * D;
    float prod = __half2float(q[q_base + dim]) * __half2float(k_pages[cache_base + static_cast<int64_t>(offset) * D + dim]);
    qk[dim] = prod;
    __syncthreads();
    for (int stride = D / 2; stride > 0; stride >>= 1) {
      if (dim < stride) qk[dim] += qk[dim + stride];
      __syncthreads();
    }
    float s = qk[0] * scale;
    float new_m = fmaxf(m, s);
    float alpha = __expf(m - new_m);
    float p = __expf(s - new_m);
    float vv = __half2float(v_pages[cache_base + static_cast<int64_t>(offset) * D + dim]);
    acc = acc * alpha + p * vv;
    l = l * alpha + p;
    m = new_m;
  }
  out[q_base + dim] = __float2half_rn(acc / fmaxf(l, 1e-20f));
}

torch::Tensor attention_decode_paged_fp16(
    torch::Tensor q,
    torch::Tensor k_new,
    torch::Tensor v_new,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor slot_mapping,
    torch::Tensor block_tables,
    torch::Tensor seq_lens,
    int64_t block_size_i,
    double softmax_scale) {
  QB_CHECK_CUDA(q); QB_CHECK_CUDA(k_new); QB_CHECK_CUDA(v_new); QB_CHECK_CUDA(k_pages); QB_CHECK_CUDA(v_pages);
  QB_CHECK_CUDA(slot_mapping); QB_CHECK_CUDA(block_tables); QB_CHECK_CUDA(seq_lens);
  QB_CHECK_CONTIGUOUS(q); QB_CHECK_CONTIGUOUS(k_new); QB_CHECK_CONTIGUOUS(v_new); QB_CHECK_CONTIGUOUS(k_pages); QB_CHECK_CONTIGUOUS(v_pages);
  QB_CHECK_CONTIGUOUS(slot_mapping); QB_CHECK_CONTIGUOUS(block_tables); QB_CHECK_CONTIGUOUS(seq_lens);
  QB_CHECK_HALF(q); QB_CHECK_HALF(k_new); QB_CHECK_HALF(v_new); QB_CHECK_HALF(k_pages); QB_CHECK_HALF(v_pages);
  QB_CHECK_INT64(slot_mapping);
  TORCH_CHECK(block_tables.scalar_type() == at::kInt, "block_tables must be int32");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt, "seq_lens must be int32");
  TORCH_CHECK(q.dim() == 3, "q must be [batch, q_heads, head_dim]");
  TORCH_CHECK(k_new.dim() == 3 && v_new.dim() == 3, "k_new/v_new must be [batch, kv_heads, head_dim]");
  TORCH_CHECK(k_pages.dim() == 4 && v_pages.dim() == 4, "paged cache must be [num_blocks, kv_heads, block_size, head_dim]");
  TORCH_CHECK(block_tables.dim() == 2, "block_tables must be [batch, max_blocks]");
  int batch = static_cast<int>(q.size(0));
  int q_heads = static_cast<int>(q.size(1));
  int head_dim = static_cast<int>(q.size(2));
  int kv_heads = static_cast<int>(k_new.size(1));
  int num_blocks = static_cast<int>(k_pages.size(0));
  int block_size = static_cast<int>(block_size_i);
  int max_blocks_per_row = static_cast<int>(block_tables.size(1));
  TORCH_CHECK(head_dim == 256, "paged attention kernel currently specializes head_dim=256");
  TORCH_CHECK(block_size > 0 && k_pages.size(2) == block_size, "block_size mismatch");
  TORCH_CHECK(k_new.size(0) == batch && v_new.size(0) == batch, "k_new/v_new batch mismatch");
  TORCH_CHECK(k_new.size(2) == head_dim && v_new.size(2) == head_dim, "k_new/v_new dim mismatch");
  TORCH_CHECK(k_pages.size(1) == kv_heads && v_pages.size(1) == kv_heads, "paged kv_heads mismatch");
  TORCH_CHECK(k_pages.size(3) == head_dim && v_pages.size(3) == head_dim, "paged head_dim mismatch");
  TORCH_CHECK(v_pages.size(0) == num_blocks && v_pages.size(2) == block_size, "v_pages shape mismatch");
  TORCH_CHECK(q_heads % kv_heads == 0, "q_heads must be divisible by kv_heads");
  TORCH_CHECK(slot_mapping.size(0) == batch && block_tables.size(0) == batch && seq_lens.size(0) == batch,
              "slot_mapping/block_tables/seq_lens batch mismatch");

  auto out = torch::empty_like(q);
  auto stream = at::cuda::getCurrentCUDAStream();
  int threads = 256;
  int append_total = batch * kv_heads * head_dim;
  attention_kv_append_paged_kernel<256><<<(append_total + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<const half*>(k_new.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v_new.data_ptr<at::Half>()),
      reinterpret_cast<half*>(k_pages.data_ptr<at::Half>()),
      reinterpret_cast<half*>(v_pages.data_ptr<at::Half>()),
      slot_mapping.data_ptr<int64_t>(),
      batch, num_blocks, kv_heads, block_size);
  QB_CUDA_CHECK(cudaGetLastError());
  dim3 grid(q_heads, batch);
  attention_decode_paged_kernel<256><<<grid, QB_ATT_BLOCK, 0, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k_pages.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v_pages.data_ptr<at::Half>()),
      block_tables.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      batch, q_heads, kv_heads, max_blocks_per_row, block_size, static_cast<float>(softmax_scale));
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}

template<int D>
__global__ void attention_kv_append_paged_fp8_kernel(
    const half* __restrict__ k_new,
    const half* __restrict__ v_new,
    uint8_t* __restrict__ k_pages,
    uint8_t* __restrict__ v_pages,
    const int64_t* __restrict__ slot_mapping,
    int batch,
    int num_blocks,
    int kv_heads,
    int block_size,
    float k_scale,
    float v_scale) {
  int linear = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * kv_heads * D;
  if (linear >= total) return;
  int d = linear % D;
  int tmp = linear / D;
  int kvh = tmp % kv_heads;
  int row = tmp / kv_heads;
  int64_t slot = slot_mapping[row];
  int64_t block = slot / block_size;
  int offset = static_cast<int>(slot % block_size);
  if (block < 0 || block >= num_blocks || offset < 0 || offset >= block_size) return;
  int64_t src = (static_cast<int64_t>(row) * kv_heads + kvh) * D + d;
  int64_t dst = ((block * kv_heads + kvh) * block_size + offset) * D + d;
  k_pages[dst] = qb_fp8_e4m3_quant(__half2float(k_new[src]) / k_scale);
  v_pages[dst] = qb_fp8_e4m3_quant(__half2float(v_new[src]) / v_scale);
}

template<int D>
__global__ void attention_decode_paged_fp8_kernel(
    const half* __restrict__ q,
    const uint8_t* __restrict__ k_pages,
    const uint8_t* __restrict__ v_pages,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    half* __restrict__ out,
    int batch,
    int q_heads,
    int kv_heads,
    int max_blocks_per_row,
    int block_size,
    float softmax_scale,
    float k_scale,
    float v_scale) {
  int qh = blockIdx.x;
  int row = blockIdx.y;
  int dim = threadIdx.x;
  if (row >= batch || qh >= q_heads || dim >= D) return;
  int seq_len = seq_lens[row];
  if (seq_len < 1) return;
  int ratio = q_heads / kv_heads;
  int kvh = qh / ratio;
  __shared__ float qk[QB_ATT_BLOCK];

  float m = -INFINITY;
  float l = 0.0f;
  float acc = 0.0f;
  const int64_t q_base = (static_cast<int64_t>(row) * q_heads + qh) * D;

  for (int t = 0; t < seq_len; ++t) {
    int block_idx = t / block_size;
    int offset = t - block_idx * block_size;
    int block_id = block_tables[static_cast<int64_t>(row) * max_blocks_per_row + block_idx];
    int64_t cache_base = (static_cast<int64_t>(block_id) * kv_heads + kvh) * block_size * D;
    float kval = qb_fp8_e4m3_decode(k_pages[cache_base + static_cast<int64_t>(offset) * D + dim]) * k_scale;
    float prod = __half2float(q[q_base + dim]) * kval;
    qk[dim] = prod;
    __syncthreads();
    for (int stride = D / 2; stride > 0; stride >>= 1) {
      if (dim < stride) qk[dim] += qk[dim + stride];
      __syncthreads();
    }
    float s = qk[0] * softmax_scale;
    float new_m = fmaxf(m, s);
    float alpha = __expf(m - new_m);
    float p = __expf(s - new_m);
    float vv = qb_fp8_e4m3_decode(v_pages[cache_base + static_cast<int64_t>(offset) * D + dim]) * v_scale;
    acc = acc * alpha + p * vv;
    l = l * alpha + p;
    m = new_m;
  }
  out[q_base + dim] = __float2half_rn(acc / fmaxf(l, 1e-20f));
}

torch::Tensor attention_decode_paged_fp8_e4m3(
    torch::Tensor q,
    torch::Tensor k_new,
    torch::Tensor v_new,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor slot_mapping,
    torch::Tensor block_tables,
    torch::Tensor seq_lens,
    int64_t block_size_i,
    double softmax_scale,
    double k_scale_d,
    double v_scale_d) {
  QB_CHECK_CUDA(q); QB_CHECK_CUDA(k_new); QB_CHECK_CUDA(v_new); QB_CHECK_CUDA(k_pages); QB_CHECK_CUDA(v_pages);
  QB_CHECK_CUDA(slot_mapping); QB_CHECK_CUDA(block_tables); QB_CHECK_CUDA(seq_lens);
  QB_CHECK_CONTIGUOUS(q); QB_CHECK_CONTIGUOUS(k_new); QB_CHECK_CONTIGUOUS(v_new); QB_CHECK_CONTIGUOUS(k_pages); QB_CHECK_CONTIGUOUS(v_pages);
  QB_CHECK_CONTIGUOUS(slot_mapping); QB_CHECK_CONTIGUOUS(block_tables); QB_CHECK_CONTIGUOUS(seq_lens);
  QB_CHECK_HALF(q); QB_CHECK_HALF(k_new); QB_CHECK_HALF(v_new); QB_CHECK_FP8_E4M3(k_pages); QB_CHECK_FP8_E4M3(v_pages);
  QB_CHECK_INT64(slot_mapping);
  TORCH_CHECK(block_tables.scalar_type() == at::kInt, "block_tables must be int32");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt, "seq_lens must be int32");
  TORCH_CHECK(q.dim() == 3, "q must be [batch, q_heads, head_dim]");
  TORCH_CHECK(k_new.dim() == 3 && v_new.dim() == 3, "k_new/v_new must be [batch, kv_heads, head_dim]");
  TORCH_CHECK(k_pages.dim() == 4 && v_pages.dim() == 4, "paged cache must be [num_blocks, kv_heads, block_size, head_dim]");
  TORCH_CHECK(block_tables.dim() == 2, "block_tables must be [batch, max_blocks]");
  int batch = static_cast<int>(q.size(0));
  int q_heads = static_cast<int>(q.size(1));
  int head_dim = static_cast<int>(q.size(2));
  int kv_heads = static_cast<int>(k_new.size(1));
  int num_blocks = static_cast<int>(k_pages.size(0));
  int block_size = static_cast<int>(block_size_i);
  int max_blocks_per_row = static_cast<int>(block_tables.size(1));
  TORCH_CHECK(head_dim == 256, "paged attention kernel currently specializes head_dim=256");
  TORCH_CHECK(block_size > 0 && k_pages.size(2) == block_size, "block_size mismatch");
  TORCH_CHECK(k_new.size(0) == batch && v_new.size(0) == batch, "k_new/v_new batch mismatch");
  TORCH_CHECK(k_new.size(2) == head_dim && v_new.size(2) == head_dim, "k_new/v_new dim mismatch");
  TORCH_CHECK(k_pages.size(1) == kv_heads && v_pages.size(1) == kv_heads, "paged kv_heads mismatch");
  TORCH_CHECK(k_pages.size(3) == head_dim && v_pages.size(3) == head_dim, "paged head_dim mismatch");
  TORCH_CHECK(v_pages.size(0) == num_blocks && v_pages.size(2) == block_size, "v_pages shape mismatch");
  TORCH_CHECK(q_heads % kv_heads == 0, "q_heads must be divisible by kv_heads");
  TORCH_CHECK(slot_mapping.size(0) == batch && block_tables.size(0) == batch && seq_lens.size(0) == batch,
              "slot_mapping/block_tables/seq_lens batch mismatch");
  TORCH_CHECK(k_scale_d > 0.0 && v_scale_d > 0.0, "k_scale/v_scale must be positive");

  auto out = torch::empty_like(q);
  auto stream = at::cuda::getCurrentCUDAStream();
  int threads = 256;
  int append_total = batch * kv_heads * head_dim;
  attention_kv_append_paged_fp8_kernel<256><<<(append_total + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<const half*>(k_new.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v_new.data_ptr<at::Half>()),
      reinterpret_cast<uint8_t*>(k_pages.data_ptr()),
      reinterpret_cast<uint8_t*>(v_pages.data_ptr()),
      slot_mapping.data_ptr<int64_t>(),
      batch, num_blocks, kv_heads, block_size, static_cast<float>(k_scale_d), static_cast<float>(v_scale_d));
  QB_CUDA_CHECK(cudaGetLastError());
  dim3 grid(q_heads, batch);
  attention_decode_paged_fp8_kernel<256><<<grid, QB_ATT_BLOCK, 0, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const uint8_t*>(k_pages.data_ptr()),
      reinterpret_cast<const uint8_t*>(v_pages.data_ptr()),
      block_tables.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      batch, q_heads, kv_heads, max_blocks_per_row, block_size, static_cast<float>(softmax_scale), static_cast<float>(k_scale_d), static_cast<float>(v_scale_d));
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}

template<int D>
__global__ void attention_kv_append_paged_int4_kernel(
    const half* __restrict__ k_new,
    const half* __restrict__ v_new,
    uint8_t* __restrict__ k_pages,
    uint8_t* __restrict__ v_pages,
    half* __restrict__ k_scales,
    half* __restrict__ v_scales,
    half* __restrict__ k_zeros,
    half* __restrict__ v_zeros,
    const int64_t* __restrict__ slot_mapping,
    int batch,
    int num_blocks,
    int kv_heads,
    int block_size,
    int hadamard_order,
    bool bdr_k,
    bool tiled_layout) {
  int row = blockIdx.x;
  int kvh = blockIdx.y;
  int d = threadIdx.x;
  if (row >= batch || kvh >= kv_heads || d >= D) return;
  int64_t slot = slot_mapping[row];
  int64_t block = slot / block_size;
  int offset = static_cast<int>(slot % block_size);
  if (block < 0 || block >= num_blocks || offset < 0 || offset >= block_size) return;

  __shared__ float k_vals[QB_ATT_BLOCK];
  __shared__ float v_vals[QB_ATT_BLOCK];
  __shared__ float k_minmax[QB_ATT_BLOCK];
  __shared__ float k_maxval[QB_ATT_BLOCK];
  __shared__ float v_minmax[QB_ATT_BLOCK];
  __shared__ float v_maxval[QB_ATT_BLOCK];

  const int64_t src_base = (static_cast<int64_t>(row) * kv_heads + kvh) * D;
  float kval = bdr_k ? qb_bdr_rotate_value<D>(k_new, src_base, d, hadamard_order) : __half2float(k_new[src_base + d]);
  float vval = __half2float(v_new[src_base + d]);
  k_vals[d] = kval;
  v_vals[d] = vval;
  k_minmax[d] = kval;
  k_maxval[d] = kval;
  v_minmax[d] = vval;
  v_maxval[d] = vval;
  __syncthreads();
  for (int stride = D / 2; stride > 0; stride >>= 1) {
    if (d < stride) {
      k_minmax[d] = fminf(k_minmax[d], k_minmax[d + stride]);
      k_maxval[d] = fmaxf(k_maxval[d], k_maxval[d + stride]);
      v_minmax[d] = fminf(v_minmax[d], v_minmax[d + stride]);
      v_maxval[d] = fmaxf(v_maxval[d], v_maxval[d + stride]);
    }
    __syncthreads();
  }
  float k_scale = fmaxf((k_maxval[0] - k_minmax[0]) / 15.0f, 1e-6f);
  float v_scale = fmaxf((v_maxval[0] - v_minmax[0]) / 15.0f, 1e-6f);
  float k_zero = -k_minmax[0] / k_scale;
  float v_zero = -v_minmax[0] / v_scale;
  const int64_t scale_idx = (static_cast<int64_t>(block) * kv_heads + kvh) * block_size + offset;
  if (d == 0) {
    k_scales[scale_idx] = __float2half_rn(k_scale);
    v_scales[scale_idx] = __float2half_rn(v_scale);
    k_zeros[scale_idx] = __float2half_rn(k_zero);
    v_zeros[scale_idx] = __float2half_rn(v_zero);
  }
  if (d < D / 2) {
    int q0k = static_cast<int>(nearbyintf(k_vals[d] / k_scale + k_zero));
    int q1k = static_cast<int>(nearbyintf(k_vals[d + D / 2] / k_scale + k_zero));
    int q0v = static_cast<int>(nearbyintf(v_vals[d] / v_scale + v_zero));
    int q1v = static_cast<int>(nearbyintf(v_vals[d + D / 2] / v_scale + v_zero));
    int64_t dst = qb_int4_paged_offset(
        static_cast<int>(block), kvh, kv_heads, block_size, D / 2, offset, d, tiled_layout);
    k_pages[dst] = qb_pack_uint4_pair(q0k, q1k);
    v_pages[dst] = qb_pack_uint4_pair(q0v, q1v);
  }
}

template<int D>
__global__ void attention_decode_paged_int4_kernel(
    const half* __restrict__ q,
    const uint8_t* __restrict__ k_pages,
    const uint8_t* __restrict__ v_pages,
    const half* __restrict__ k_scales,
    const half* __restrict__ v_scales,
    const half* __restrict__ k_zeros,
    const half* __restrict__ v_zeros,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    half* __restrict__ out,
    int batch,
    int q_heads,
    int kv_heads,
    int max_blocks_per_row,
    int block_size,
    float softmax_scale,
    int hadamard_order,
    bool bdr_k,
    bool tiled_layout) {
  int qh = blockIdx.x;
  int row = blockIdx.y;
  int dim = threadIdx.x;
  if (row >= batch || qh >= q_heads || dim >= D) return;
  int seq_len = seq_lens[row];
  if (seq_len < 1) return;
  int ratio = q_heads / kv_heads;
  int kvh = qh / ratio;
  __shared__ float qk[QB_ATT_BLOCK];

  float m = -INFINITY;
  float l = 0.0f;
  float acc = 0.0f;
  const int64_t q_base = (static_cast<int64_t>(row) * q_heads + qh) * D;
  float qval = bdr_k ? qb_bdr_rotate_value<D>(q, q_base, dim, hadamard_order) : __half2float(q[q_base + dim]);

  for (int t = 0; t < seq_len; ++t) {
    int block_idx = t / block_size;
    int offset = t - block_idx * block_size;
    int block_id = block_tables[static_cast<int64_t>(row) * max_blocks_per_row + block_idx];
    int64_t scale_idx = (static_cast<int64_t>(block_id) * kv_heads + kvh) * block_size + offset;
    bool high_half = dim >= (D / 2);
    int packed_dim = high_half ? dim - D / 2 : dim;
    int64_t packed_idx = qb_int4_paged_offset(
        block_id, kvh, kv_heads, block_size, D / 2, offset, packed_dim, tiled_layout);
    uint8_t kpack = k_pages[packed_idx];
    uint8_t vpack = v_pages[packed_idx];
    float kval = qb_unpack_uint4(kpack, high_half, __half2float(k_scales[scale_idx]), __half2float(k_zeros[scale_idx]));
    float prod = qval * kval;
    qk[dim] = prod;
    __syncthreads();
    for (int stride = D / 2; stride > 0; stride >>= 1) {
      if (dim < stride) qk[dim] += qk[dim + stride];
      __syncthreads();
    }
    float s = qk[0] * softmax_scale;
    float new_m = fmaxf(m, s);
    float alpha = __expf(m - new_m);
    float p = __expf(s - new_m);
    float vv = qb_unpack_uint4(vpack, high_half, __half2float(v_scales[scale_idx]), __half2float(v_zeros[scale_idx]));
    acc = acc * alpha + p * vv;
    l = l * alpha + p;
    m = new_m;
  }
  out[q_base + dim] = __float2half_rn(acc / fmaxf(l, 1e-20f));
}

torch::Tensor attention_decode_paged_int4(
    torch::Tensor q,
    torch::Tensor k_new,
    torch::Tensor v_new,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor k_scales,
    torch::Tensor v_scales,
    torch::Tensor k_zeros,
    torch::Tensor v_zeros,
    torch::Tensor slot_mapping,
    torch::Tensor block_tables,
    torch::Tensor seq_lens,
    int64_t block_size_i,
    double softmax_scale,
    int64_t hadamard_order_i,
    bool bdr_k,
    bool rotate_v,
    bool tiled_layout) {
  QB_CHECK_CUDA(q); QB_CHECK_CUDA(k_new); QB_CHECK_CUDA(v_new); QB_CHECK_CUDA(k_pages); QB_CHECK_CUDA(v_pages);
  QB_CHECK_CUDA(k_scales); QB_CHECK_CUDA(v_scales); QB_CHECK_CUDA(k_zeros); QB_CHECK_CUDA(v_zeros);
  QB_CHECK_CUDA(slot_mapping); QB_CHECK_CUDA(block_tables); QB_CHECK_CUDA(seq_lens);
  QB_CHECK_CONTIGUOUS(q); QB_CHECK_CONTIGUOUS(k_new); QB_CHECK_CONTIGUOUS(v_new); QB_CHECK_CONTIGUOUS(k_pages); QB_CHECK_CONTIGUOUS(v_pages);
  QB_CHECK_CONTIGUOUS(k_scales); QB_CHECK_CONTIGUOUS(v_scales); QB_CHECK_CONTIGUOUS(k_zeros); QB_CHECK_CONTIGUOUS(v_zeros);
  QB_CHECK_CONTIGUOUS(slot_mapping); QB_CHECK_CONTIGUOUS(block_tables); QB_CHECK_CONTIGUOUS(seq_lens);
  QB_CHECK_HALF(q); QB_CHECK_HALF(k_new); QB_CHECK_HALF(v_new); QB_CHECK_UINT8(k_pages); QB_CHECK_UINT8(v_pages);
  QB_CHECK_HALF(k_scales); QB_CHECK_HALF(v_scales); QB_CHECK_HALF(k_zeros); QB_CHECK_HALF(v_zeros); QB_CHECK_INT64(slot_mapping);
  TORCH_CHECK(!rotate_v, "INT4 BDR rotate_v is not implemented in LangBurst; use K-only BDR");
  TORCH_CHECK(block_tables.scalar_type() == at::kInt, "block_tables must be int32");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt, "seq_lens must be int32");
  TORCH_CHECK(q.dim() == 3, "q must be [batch, q_heads, head_dim]");
  TORCH_CHECK(k_new.dim() == 3 && v_new.dim() == 3, "k_new/v_new must be [batch, kv_heads, head_dim]");
  TORCH_CHECK(k_pages.dim() == 4 && v_pages.dim() == 4,
              "paged cache must be [num_blocks, kv_heads, block_size, packed_head_dim] or [num_blocks, kv_heads, packed_head_dim, block_size]");
  TORCH_CHECK(k_scales.dim() == 3 && v_scales.dim() == 3, "int4 scales must be [num_blocks, kv_heads, block_size]");
  TORCH_CHECK(k_zeros.dim() == 3 && v_zeros.dim() == 3, "int4 zero points must be [num_blocks, kv_heads, block_size]");
  int batch = static_cast<int>(q.size(0));
  int q_heads = static_cast<int>(q.size(1));
  int head_dim = static_cast<int>(q.size(2));
  int kv_heads = static_cast<int>(k_new.size(1));
  int num_blocks = static_cast<int>(k_pages.size(0));
  int block_size = static_cast<int>(block_size_i);
  int max_blocks_per_row = static_cast<int>(block_tables.size(1));
  int hadamard_order = static_cast<int>(hadamard_order_i);
  TORCH_CHECK(head_dim == 256, "paged attention kernel currently specializes head_dim=256");
  TORCH_CHECK((head_dim % 2) == 0, "INT4 KV requires even head_dim");
  TORCH_CHECK(!bdr_k || (hadamard_order > 0 && (hadamard_order & (hadamard_order - 1)) == 0 && head_dim % hadamard_order == 0),
              "BDR hadamard_order must be a power-of-two divisor of head_dim");
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  TORCH_CHECK(k_new.size(0) == batch && v_new.size(0) == batch, "k_new/v_new batch mismatch");
  TORCH_CHECK(k_new.size(2) == head_dim && v_new.size(2) == head_dim, "k_new/v_new dim mismatch");
  TORCH_CHECK(k_pages.size(1) == kv_heads && v_pages.size(1) == kv_heads, "arena kv_heads mismatch");
  if (tiled_layout) {
    TORCH_CHECK(k_pages.size(2) == head_dim / 2 && v_pages.size(2) == head_dim / 2, "tiled int4 packed head_dim mismatch");
    TORCH_CHECK(k_pages.size(3) == block_size && v_pages.size(3) == block_size, "tiled block_size mismatch");
  } else {
    TORCH_CHECK(k_pages.size(2) == block_size && v_pages.size(2) == block_size, "block_size mismatch");
    TORCH_CHECK(k_pages.size(3) == head_dim / 2 && v_pages.size(3) == head_dim / 2, "int4 packed head_dim mismatch");
  }
  TORCH_CHECK(k_scales.size(0) == num_blocks && k_scales.size(1) == kv_heads && k_scales.size(2) == block_size, "k_scales shape mismatch");
  TORCH_CHECK(v_scales.sizes() == k_scales.sizes(), "v_scales shape mismatch");
  TORCH_CHECK(k_zeros.sizes() == k_scales.sizes(), "k_zeros shape mismatch");
  TORCH_CHECK(v_zeros.sizes() == k_scales.sizes(), "v_zeros shape mismatch");
  TORCH_CHECK(q_heads % kv_heads == 0, "q_heads must be divisible by kv_heads");
  TORCH_CHECK(slot_mapping.size(0) == batch && block_tables.size(0) == batch && seq_lens.size(0) == batch,
              "slot_mapping/block_tables/seq_lens batch mismatch");

  auto out = torch::empty_like(q);
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 append_grid(batch, kv_heads);
  attention_kv_append_paged_int4_kernel<256><<<append_grid, QB_ATT_BLOCK, 0, stream>>>(
      reinterpret_cast<const half*>(k_new.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v_new.data_ptr<at::Half>()),
      reinterpret_cast<uint8_t*>(k_pages.data_ptr()),
      reinterpret_cast<uint8_t*>(v_pages.data_ptr()),
      reinterpret_cast<half*>(k_scales.data_ptr<at::Half>()),
      reinterpret_cast<half*>(v_scales.data_ptr<at::Half>()),
      reinterpret_cast<half*>(k_zeros.data_ptr<at::Half>()),
      reinterpret_cast<half*>(v_zeros.data_ptr<at::Half>()),
      slot_mapping.data_ptr<int64_t>(),
      batch, num_blocks, kv_heads, block_size, hadamard_order, bdr_k, tiled_layout);
  QB_CUDA_CHECK(cudaGetLastError());
  dim3 grid(q_heads, batch);
  attention_decode_paged_int4_kernel<256><<<grid, QB_ATT_BLOCK, 0, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const uint8_t*>(k_pages.data_ptr()),
      reinterpret_cast<const uint8_t*>(v_pages.data_ptr()),
      reinterpret_cast<const half*>(k_scales.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v_scales.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k_zeros.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v_zeros.data_ptr<at::Half>()),
      block_tables.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      batch, q_heads, kv_heads, max_blocks_per_row, block_size, static_cast<float>(softmax_scale), hadamard_order, bdr_k, tiled_layout);
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}

torch::Tensor attention_paged_int4_flash(
    torch::Tensor q,
    torch::Tensor k_new,
    torch::Tensor v_new,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor k_scales,
    torch::Tensor v_scales,
    torch::Tensor k_zeros,
    torch::Tensor v_zeros,
    torch::Tensor slot_mapping,
    torch::Tensor block_tables,
    torch::Tensor seq_lens,
    int64_t block_size_i,
    double softmax_scale,
    int64_t hadamard_order_i,
    bool bdr_k,
    bool rotate_v,
    bool tiled_layout) {
  // FlashAttention-style contract:
  //   - direct paged INT4/BDR K/V read,
  //   - no full fp16 KV staging,
  //   - online softmax over the global prefix.
  //
  // This kernel owns the INT4 paged-flash contract. Faster implementations
  // must keep this state/shape contract and replace only the internals after
  // parity passes.
  return attention_decode_paged_int4(
      q,
      k_new,
      v_new,
      k_pages,
      v_pages,
      k_scales,
      v_scales,
      k_zeros,
      v_zeros,
      slot_mapping,
      block_tables,
      seq_lens,
      block_size_i,
      softmax_scale,
      hadamard_order_i,
      bdr_k,
      rotate_v,
      tiled_layout);
}

void attention_append_paged_int4(
    torch::Tensor k_new,
    torch::Tensor v_new,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor k_scales,
    torch::Tensor v_scales,
    torch::Tensor k_zeros,
    torch::Tensor v_zeros,
    torch::Tensor slot_mapping,
    int64_t block_size_i,
    int64_t hadamard_order_i,
    bool bdr_k,
    bool rotate_v,
    bool tiled_layout) {
  QB_CHECK_CUDA(k_new); QB_CHECK_CUDA(v_new); QB_CHECK_CUDA(k_pages); QB_CHECK_CUDA(v_pages);
  QB_CHECK_CUDA(k_scales); QB_CHECK_CUDA(v_scales); QB_CHECK_CUDA(k_zeros); QB_CHECK_CUDA(v_zeros);
  QB_CHECK_CUDA(slot_mapping);
  QB_CHECK_CONTIGUOUS(k_new); QB_CHECK_CONTIGUOUS(v_new); QB_CHECK_CONTIGUOUS(k_pages); QB_CHECK_CONTIGUOUS(v_pages);
  QB_CHECK_CONTIGUOUS(k_scales); QB_CHECK_CONTIGUOUS(v_scales); QB_CHECK_CONTIGUOUS(k_zeros); QB_CHECK_CONTIGUOUS(v_zeros);
  QB_CHECK_CONTIGUOUS(slot_mapping);
  QB_CHECK_HALF(k_new); QB_CHECK_HALF(v_new); QB_CHECK_UINT8(k_pages); QB_CHECK_UINT8(v_pages);
  QB_CHECK_HALF(k_scales); QB_CHECK_HALF(v_scales); QB_CHECK_HALF(k_zeros); QB_CHECK_HALF(v_zeros); QB_CHECK_INT64(slot_mapping);
  TORCH_CHECK(!rotate_v, "INT4 BDR rotate_v is not implemented in LangBurst; use K-only BDR");
  TORCH_CHECK(k_new.dim() == 3 && v_new.dim() == 3, "k_new/v_new must be [batch, kv_heads, head_dim]");
  TORCH_CHECK(k_pages.dim() == 4 && v_pages.dim() == 4,
              "paged cache must be [num_blocks, kv_heads, block_size, packed_head_dim] or [num_blocks, kv_heads, packed_head_dim, block_size]");
  TORCH_CHECK(k_scales.dim() == 3 && v_scales.dim() == 3, "int4 scales must be [num_blocks, kv_heads, block_size]");
  TORCH_CHECK(k_zeros.dim() == 3 && v_zeros.dim() == 3, "int4 zero points must be [num_blocks, kv_heads, block_size]");
  int batch = static_cast<int>(k_new.size(0));
  int head_dim = static_cast<int>(k_new.size(2));
  int kv_heads = static_cast<int>(k_new.size(1));
  int num_blocks = static_cast<int>(k_pages.size(0));
  int block_size = static_cast<int>(block_size_i);
  int hadamard_order = static_cast<int>(hadamard_order_i);
  TORCH_CHECK(head_dim == 256, "paged append kernel currently specializes head_dim=256");
  TORCH_CHECK((head_dim % 2) == 0, "INT4 KV requires even head_dim");
  TORCH_CHECK(!bdr_k || (hadamard_order > 0 && (hadamard_order & (hadamard_order - 1)) == 0 && head_dim % hadamard_order == 0),
              "BDR hadamard_order must be a power-of-two divisor of head_dim");
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  TORCH_CHECK(v_new.size(0) == batch && v_new.size(1) == kv_heads && v_new.size(2) == head_dim, "v_new shape mismatch");
  TORCH_CHECK(k_pages.size(1) == kv_heads && v_pages.size(1) == kv_heads, "arena kv_heads mismatch");
  if (tiled_layout) {
    TORCH_CHECK(k_pages.size(2) == head_dim / 2 && v_pages.size(2) == head_dim / 2, "tiled int4 packed head_dim mismatch");
    TORCH_CHECK(k_pages.size(3) == block_size && v_pages.size(3) == block_size, "tiled block_size mismatch");
  } else {
    TORCH_CHECK(k_pages.size(2) == block_size && v_pages.size(2) == block_size, "block_size mismatch");
    TORCH_CHECK(k_pages.size(3) == head_dim / 2 && v_pages.size(3) == head_dim / 2, "int4 packed head_dim mismatch");
  }
  TORCH_CHECK(k_scales.size(0) == num_blocks && k_scales.size(1) == kv_heads && k_scales.size(2) == block_size, "k_scales shape mismatch");
  TORCH_CHECK(v_scales.sizes() == k_scales.sizes(), "v_scales shape mismatch");
  TORCH_CHECK(k_zeros.sizes() == k_scales.sizes(), "k_zeros shape mismatch");
  TORCH_CHECK(v_zeros.sizes() == k_scales.sizes(), "v_zeros shape mismatch");
  TORCH_CHECK(slot_mapping.size(0) == batch, "slot_mapping batch mismatch");

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 append_grid(batch, kv_heads);
  attention_kv_append_paged_int4_kernel<256><<<append_grid, QB_ATT_BLOCK, 0, stream>>>(
      reinterpret_cast<const half*>(k_new.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v_new.data_ptr<at::Half>()),
      reinterpret_cast<uint8_t*>(k_pages.data_ptr()),
      reinterpret_cast<uint8_t*>(v_pages.data_ptr()),
      reinterpret_cast<half*>(k_scales.data_ptr<at::Half>()),
      reinterpret_cast<half*>(v_scales.data_ptr<at::Half>()),
      reinterpret_cast<half*>(k_zeros.data_ptr<at::Half>()),
      reinterpret_cast<half*>(v_zeros.data_ptr<at::Half>()),
      slot_mapping.data_ptr<int64_t>(),
      batch, num_blocks, kv_heads, block_size, hadamard_order, bdr_k, tiled_layout);
  QB_CUDA_CHECK(cudaGetLastError());
}
