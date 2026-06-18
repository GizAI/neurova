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

__global__ void resolve_greedy_speculative_kernel(
    const int64_t* __restrict__ drafts,
    const int64_t* __restrict__ targets,
    const int64_t* __restrict__ bonuses,
    const int32_t* __restrict__ cu_drafts,
    const int32_t* __restrict__ scheduled_counts,
    int64_t* __restrict__ token_matrix,
    int32_t* __restrict__ sampled_counts,
    int32_t* __restrict__ rejected_counts,
    int32_t* __restrict__ accepted_counts,
    int batch,
    int max_sampled) {
  int row = blockIdx.x;
  if (row >= batch || threadIdx.x != 0) return;
  int draft_start = row == 0 ? 0 : cu_drafts[row - 1];
  int draft_end = cu_drafts[row];
  int draft_n = draft_end - draft_start;
  int accepted = 0;
  for (; accepted < draft_n; ++accepted) {
    if (drafts[draft_start + accepted] != targets[draft_start + accepted]) {
      break;
    }
  }

  int sampled = accepted + 1;
  int64_t* row_tokens = token_matrix + static_cast<int64_t>(row) * max_sampled;
  for (int i = 0; i < max_sampled; ++i) {
    row_tokens[i] = -1;
  }
  for (int i = 0; i < accepted; ++i) {
    row_tokens[i] = drafts[draft_start + i];
  }
  if (accepted < draft_n) {
    row_tokens[accepted] = targets[draft_start + accepted];
  } else {
    row_tokens[accepted] = bonuses[row];
  }

  int scheduled = scheduled_counts[row];
  int rejected = scheduled - sampled;
  sampled_counts[row] = sampled;
  rejected_counts[row] = rejected < 0 ? 0 : rejected;
  accepted_counts[row] = accepted;
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

__global__ void argmax_unpack_state_kernel(const unsigned long long* __restrict__ state, int64_t* __restrict__ out, int64_t rows) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  unsigned long long packed = state[row];
  unsigned int tie = static_cast<unsigned int>(packed & 0xffffffffULL);
  out[row] = static_cast<int64_t>(0xffffffffu - tie);
}

void argmax_unpack_state_out(torch::Tensor state, torch::Tensor out) {
  QB_CHECK_CUDA(state); QB_CHECK_CUDA(out);
  QB_CHECK_CONTIGUOUS(state); QB_CHECK_CONTIGUOUS(out);
  QB_CHECK_INT64(state); QB_CHECK_INT64(out);
  TORCH_CHECK(state.dim() == 1 && out.dim() == 1, "state/out must be 1D");
  TORCH_CHECK(state.numel() == out.numel(), "state/out length mismatch");
  int64_t rows = state.numel();
  if (rows == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  int threads = 128;
  int blocks = static_cast<int>((rows + threads - 1) / threads);
  argmax_unpack_state_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const unsigned long long*>(state.data_ptr<int64_t>()), out.data_ptr<int64_t>(), rows);
  QB_CUDA_CHECK(cudaGetLastError());
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

std::vector<torch::Tensor> resolve_greedy_speculative(
    torch::Tensor draft_token_ids,
    torch::Tensor target_token_ids,
    torch::Tensor bonus_token_ids,
    torch::Tensor cu_num_draft_tokens,
    torch::Tensor scheduled_token_counts) {
  QB_CHECK_CUDA(draft_token_ids); QB_CHECK_CUDA(target_token_ids); QB_CHECK_CUDA(bonus_token_ids);
  QB_CHECK_CUDA(cu_num_draft_tokens); QB_CHECK_CUDA(scheduled_token_counts);
  QB_CHECK_CONTIGUOUS(draft_token_ids); QB_CHECK_CONTIGUOUS(target_token_ids); QB_CHECK_CONTIGUOUS(bonus_token_ids);
  QB_CHECK_CONTIGUOUS(cu_num_draft_tokens); QB_CHECK_CONTIGUOUS(scheduled_token_counts);
  QB_CHECK_INT64(draft_token_ids); QB_CHECK_INT64(target_token_ids); QB_CHECK_INT64(bonus_token_ids);
  TORCH_CHECK(cu_num_draft_tokens.scalar_type() == at::kInt, "cu_num_draft_tokens must be int32");
  TORCH_CHECK(scheduled_token_counts.scalar_type() == at::kInt, "scheduled_token_counts must be int32");
  TORCH_CHECK(cu_num_draft_tokens.dim() == 1 && scheduled_token_counts.dim() == 1, "counts must be 1D");
  const int64_t batch64 = cu_num_draft_tokens.numel();
  TORCH_CHECK(scheduled_token_counts.numel() == batch64, "scheduled counts must match batch");
  TORCH_CHECK(bonus_token_ids.numel() == batch64, "bonus ids must match batch");
  if (batch64 == 0) {
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(draft_token_ids.device());
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(draft_token_ids.device());
    return {
      torch::empty({0, 0}, opts_i64),
      torch::empty({0}, opts_i32),
      torch::empty({0}, opts_i32),
      torch::empty({0}, opts_i32),
    };
  }
  const int batch = static_cast<int>(batch64);
  TORCH_CHECK(target_token_ids.numel() >= draft_token_ids.numel(), "target_token_ids shorter than drafts");
  // Keep the launcher graph-friendly: do not copy cu_num_draft_tokens to CPU to
  // compute max draft length.  The matrix is only a temporary handoff buffer,
  // and speculative K is intentionally tiny, so a conservative width avoids a
  // device sync without materially changing memory use.
  const int max_sampled = static_cast<int>(std::max<int64_t>(1, draft_token_ids.numel() + 1));
  auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(draft_token_ids.device());
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(draft_token_ids.device());
  auto token_matrix = torch::empty({batch, max_sampled}, opts_i64);
  auto sampled_counts = torch::empty({batch}, opts_i32);
  auto rejected_counts = torch::empty({batch}, opts_i32);
  auto accepted_counts = torch::empty({batch}, opts_i32);
  auto stream = at::cuda::getCurrentCUDAStream();
  resolve_greedy_speculative_kernel<<<batch, 1, 0, stream>>>(
      draft_token_ids.data_ptr<int64_t>(),
      target_token_ids.data_ptr<int64_t>(),
      bonus_token_ids.data_ptr<int64_t>(),
      cu_num_draft_tokens.data_ptr<int32_t>(),
      scheduled_token_counts.data_ptr<int32_t>(),
      token_matrix.data_ptr<int64_t>(),
      sampled_counts.data_ptr<int32_t>(),
      rejected_counts.data_ptr<int32_t>(),
      accepted_counts.data_ptr<int32_t>(),
      batch,
      max_sampled);
  QB_CUDA_CHECK(cudaGetLastError());
  return {token_matrix, sampled_counts, rejected_counts, accepted_counts};
}
