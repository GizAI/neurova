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
