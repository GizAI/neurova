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

__global__ void depthwise_conv_update_kernel(
    half* __restrict__ state,
    const half* __restrict__ x,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ out,
    int channels,
    int history,
    int kernel,
    int weight_stride,
    bool has_bias) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  float acc = 0.0f;
  const int state_base = c * history;
  const int weight_base = c * weight_stride;
  for (int i = 0; i < history; ++i) {
    acc += __half2float(state[state_base + i]) * __half2float(weight[weight_base + i]);
  }
  acc += __half2float(x[c]) * __half2float(weight[weight_base + history]);
  if (has_bias) {
    acc += __half2float(bias[c]);
  }
  for (int i = 0; i < history - 1; ++i) {
    state[state_base + i] = state[state_base + i + 1];
  }
  if (history > 0) {
    state[state_base + history - 1] = x[c];
  }
  out[c] = __float2half_rn(qb_silu(acc));
}

torch::Tensor depthwise_conv_update(torch::Tensor state, torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
  QB_CHECK_CUDA(state); QB_CHECK_CUDA(x); QB_CHECK_CUDA(weight); QB_CHECK_CUDA(bias);
  QB_CHECK_CONTIGUOUS(state); QB_CHECK_CONTIGUOUS(x); QB_CHECK_CONTIGUOUS(weight); QB_CHECK_CONTIGUOUS(bias);
  QB_CHECK_HALF(state); QB_CHECK_HALF(x); QB_CHECK_HALF(weight);
  TORCH_CHECK(bias.numel() == 0 || bias.scalar_type() == at::kHalf, "bias must be empty or fp16");
  TORCH_CHECK(state.dim() == 2, "state must be [channels, kernel-1]");
  TORCH_CHECK(x.dim() == 1 && x.size(0) == state.size(0), "x must be [channels]");
  int channels = static_cast<int>(state.size(0));
  int history = static_cast<int>(state.size(1));
  int kernel = history + 1;
  int weight_stride = 0;
  if (weight.dim() == 3) {
    TORCH_CHECK(weight.size(0) == channels && weight.size(2) == kernel, "weight must be [channels, 1, kernel]");
    weight_stride = static_cast<int>(weight.size(2));
  } else {
    TORCH_CHECK(weight.dim() == 2 && weight.size(0) == channels && weight.size(1) == kernel, "weight must be [channels, kernel]");
    weight_stride = static_cast<int>(weight.size(1));
  }
  bool has_bias = bias.numel() > 0;
  TORCH_CHECK(!has_bias || bias.numel() == channels, "bias must be [channels]");
  auto out = torch::empty_like(x);
  int threads = 256;
  int blocks = (channels + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  depthwise_conv_update_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<half*>(state.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      has_bias ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr,
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      channels, history, kernel, weight_stride, has_bias);
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}

__global__ void depthwise_conv_update_scan_kernel(
    half* __restrict__ state,
    const half* __restrict__ x,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ out,
    int tokens,
    int channels,
    int history,
    int kernel,
    int weight_stride,
    bool has_bias) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  const int state_base = c * history;
  const int weight_base = c * weight_stride;
  for (int t = 0; t < tokens; ++t) {
    float acc = 0.0f;
    for (int i = 0; i < history; ++i) {
      acc += __half2float(state[state_base + i]) * __half2float(weight[weight_base + i]);
    }
    half xv = x[t * channels + c];
    acc += __half2float(xv) * __half2float(weight[weight_base + history]);
    if (has_bias) {
      acc += __half2float(bias[c]);
    }
    for (int i = 0; i < history - 1; ++i) {
      state[state_base + i] = state[state_base + i + 1];
    }
    if (history > 0) {
      state[state_base + history - 1] = xv;
    }
    out[t * channels + c] = __float2half_rn(qb_silu(acc));
  }
}

torch::Tensor depthwise_conv_update_scan(torch::Tensor state, torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
  QB_CHECK_CUDA(state); QB_CHECK_CUDA(x); QB_CHECK_CUDA(weight); QB_CHECK_CUDA(bias);
  QB_CHECK_CONTIGUOUS(state); QB_CHECK_CONTIGUOUS(x); QB_CHECK_CONTIGUOUS(weight); QB_CHECK_CONTIGUOUS(bias);
  QB_CHECK_HALF(state); QB_CHECK_HALF(x); QB_CHECK_HALF(weight);
  TORCH_CHECK(bias.numel() == 0 || bias.scalar_type() == at::kHalf, "bias must be empty or fp16");
  TORCH_CHECK(state.dim() == 2, "state must be [channels, kernel-1]");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == state.size(0), "x must be [tokens, channels]");
  int tokens = static_cast<int>(x.size(0));
  int channels = static_cast<int>(state.size(0));
  int history = static_cast<int>(state.size(1));
  int kernel = history + 1;
  int weight_stride = 0;
  if (weight.dim() == 3) {
    TORCH_CHECK(weight.size(0) == channels && weight.size(2) == kernel, "weight must be [channels, 1, kernel]");
    weight_stride = static_cast<int>(weight.size(2));
  } else {
    TORCH_CHECK(weight.dim() == 2 && weight.size(0) == channels && weight.size(1) == kernel, "weight must be [channels, kernel]");
    weight_stride = static_cast<int>(weight.size(1));
  }
  bool has_bias = bias.numel() > 0;
  TORCH_CHECK(!has_bias || bias.numel() == channels, "bias must be [channels]");
  auto out = torch::empty_like(x);
  int threads = 256;
  int blocks = (channels + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  depthwise_conv_update_scan_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<half*>(state.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      has_bias ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr,
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      tokens, channels, history, kernel, weight_stride, has_bias);
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}

__global__ void depthwise_conv_update_batch_kernel(
    half* __restrict__ state_arena,
    const int64_t* __restrict__ state_indices,
    const half* __restrict__ x,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ out,
    int batch,
    int slots,
    int channels,
    int history,
    int kernel,
    int weight_stride,
    bool has_bias) {
  int linear = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * channels;
  if (linear >= total) return;
  int row = linear / channels;
  int c = linear - row * channels;
  int64_t slot64 = state_indices[row];
  if (slot64 < 0 || slot64 >= slots) return;
  int slot = static_cast<int>(slot64);
  int64_t state_base = (static_cast<int64_t>(slot) * channels + c) * history;
  int weight_base = c * weight_stride;

  float acc = 0.0f;
  for (int i = 0; i < history; ++i) {
    acc += __half2float(state_arena[state_base + i]) * __half2float(weight[weight_base + i]);
  }
  half xv = x[static_cast<int64_t>(row) * channels + c];
  acc += __half2float(xv) * __half2float(weight[weight_base + history]);
  if (has_bias) {
    acc += __half2float(bias[c]);
  }
  for (int i = 0; i < history - 1; ++i) {
    state_arena[state_base + i] = state_arena[state_base + i + 1];
  }
  if (history > 0) {
    state_arena[state_base + history - 1] = xv;
  }
  out[static_cast<int64_t>(row) * channels + c] = __float2half_rn(qb_silu(acc));
}

torch::Tensor depthwise_conv_update_batch(torch::Tensor state_arena, torch::Tensor state_indices, torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
  QB_CHECK_CUDA(state_arena); QB_CHECK_CUDA(state_indices); QB_CHECK_CUDA(x); QB_CHECK_CUDA(weight); QB_CHECK_CUDA(bias);
  QB_CHECK_CONTIGUOUS(state_arena); QB_CHECK_CONTIGUOUS(state_indices); QB_CHECK_CONTIGUOUS(x); QB_CHECK_CONTIGUOUS(weight); QB_CHECK_CONTIGUOUS(bias);
  QB_CHECK_HALF(state_arena); QB_CHECK_HALF(x); QB_CHECK_HALF(weight);
  QB_CHECK_INT64(state_indices);
  TORCH_CHECK(bias.numel() == 0 || bias.scalar_type() == at::kHalf, "bias must be empty or fp16");
  TORCH_CHECK(state_arena.dim() == 3, "state_arena must be [slots, channels, kernel-1]");
  TORCH_CHECK(x.dim() == 2, "x must be [batch, channels]");
  int slots = static_cast<int>(state_arena.size(0));
  int channels = static_cast<int>(state_arena.size(1));
  int history = static_cast<int>(state_arena.size(2));
  int batch = static_cast<int>(x.size(0));
  int kernel = history + 1;
  TORCH_CHECK(x.size(1) == channels, "x channels mismatch");
  TORCH_CHECK(state_indices.dim() == 1 && state_indices.size(0) == batch, "state_indices must be [batch]");
  int weight_stride = 0;
  if (weight.dim() == 3) {
    TORCH_CHECK(weight.size(0) == channels && weight.size(2) == kernel, "weight must be [channels, 1, kernel]");
    weight_stride = static_cast<int>(weight.size(2));
  } else {
    TORCH_CHECK(weight.dim() == 2 && weight.size(0) == channels && weight.size(1) == kernel, "weight must be [channels, kernel]");
    weight_stride = static_cast<int>(weight.size(1));
  }
  bool has_bias = bias.numel() > 0;
  TORCH_CHECK(!has_bias || bias.numel() == channels, "bias must be [channels]");
  auto out = torch::empty_like(x);
  int threads = 256;
  int blocks = (batch * channels + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  depthwise_conv_update_batch_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<half*>(state_arena.data_ptr<at::Half>()),
      state_indices.data_ptr<int64_t>(),
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      has_bias ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr,
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      batch, slots, channels, history, kernel, weight_stride, has_bias);
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}

__global__ void depthwise_conv_update_spec_kernel(
    half* __restrict__ state_arena,
    const int64_t* __restrict__ state_indices,
    const int32_t* __restrict__ commit_tokens,
    const half* __restrict__ x,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ out,
    int batch,
    int tokens,
    int slots,
    int channels,
    int history,
    int kernel,
    int weight_stride,
    bool has_bias) {
  constexpr int MAX_HISTORY = 16;
  int linear = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * channels;
  if (linear >= total) return;
  int row = linear / channels;
  int c = linear - row * channels;
  int64_t slot64 = state_indices[row];
  if (slot64 < 0 || slot64 >= slots) return;
  int slot = static_cast<int>(slot64);
  int commit_n = commit_tokens[row];
  if (commit_n < 0) commit_n = 0;
  if (commit_n > tokens) commit_n = tokens;
  int64_t state_base = (static_cast<int64_t>(slot) * channels + c) * history;
  int weight_base = c * weight_stride;

  float hist[MAX_HISTORY];
  for (int i = 0; i < MAX_HISTORY; ++i) hist[i] = 0.0f;
  for (int i = 0; i < history; ++i) {
    hist[i] = __half2float(state_arena[state_base + i]);
  }

  for (int t = 0; t < tokens; ++t) {
    float acc = 0.0f;
    for (int i = 0; i < history; ++i) {
      acc += hist[i] * __half2float(weight[weight_base + i]);
    }
    half xv_h = x[(static_cast<int64_t>(row) * tokens + t) * channels + c];
    float xv = __half2float(xv_h);
    acc += xv * __half2float(weight[weight_base + history]);
    if (has_bias) {
      acc += __half2float(bias[c]);
    }
    for (int i = 0; i < history - 1; ++i) {
      hist[i] = hist[i + 1];
    }
    if (history > 0) {
      hist[history - 1] = xv;
    }
    out[(static_cast<int64_t>(row) * tokens + t) * channels + c] = __float2half_rn(qb_silu(acc));
    if (t + 1 == commit_n) {
      for (int i = 0; i < history; ++i) {
        state_arena[state_base + i] = __float2half_rn(hist[i]);
      }
    }
  }
}

torch::Tensor depthwise_conv_update_spec(
    torch::Tensor state_arena,
    torch::Tensor state_indices,
    torch::Tensor commit_tokens,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias) {
  QB_CHECK_CUDA(state_arena); QB_CHECK_CUDA(state_indices); QB_CHECK_CUDA(commit_tokens); QB_CHECK_CUDA(x); QB_CHECK_CUDA(weight); QB_CHECK_CUDA(bias);
  QB_CHECK_CONTIGUOUS(state_arena); QB_CHECK_CONTIGUOUS(state_indices); QB_CHECK_CONTIGUOUS(commit_tokens); QB_CHECK_CONTIGUOUS(x); QB_CHECK_CONTIGUOUS(weight); QB_CHECK_CONTIGUOUS(bias);
  QB_CHECK_HALF(state_arena); QB_CHECK_HALF(x); QB_CHECK_HALF(weight);
  QB_CHECK_INT64(state_indices);
  TORCH_CHECK(commit_tokens.scalar_type() == at::kInt, "commit_tokens must be int32");
  TORCH_CHECK(bias.numel() == 0 || bias.scalar_type() == at::kHalf, "bias must be empty or fp16");
  TORCH_CHECK(state_arena.dim() == 3, "state_arena must be [slots, channels, kernel-1]");
  TORCH_CHECK(x.dim() == 3, "x must be [batch, tokens, channels]");
  int slots = static_cast<int>(state_arena.size(0));
  int channels = static_cast<int>(state_arena.size(1));
  int history = static_cast<int>(state_arena.size(2));
  TORCH_CHECK(history <= 16, "depthwise_conv_update_spec supports kernel history <= 16");
  int batch = static_cast<int>(x.size(0));
  int tokens = static_cast<int>(x.size(1));
  int kernel = history + 1;
  TORCH_CHECK(x.size(2) == channels, "x channels mismatch");
  TORCH_CHECK(state_indices.dim() == 1 && state_indices.size(0) == batch, "state_indices must be [batch]");
  TORCH_CHECK(commit_tokens.dim() == 1 && commit_tokens.size(0) == batch, "commit_tokens must be [batch]");
  int weight_stride = 0;
  if (weight.dim() == 3) {
    TORCH_CHECK(weight.size(0) == channels && weight.size(2) == kernel, "weight must be [channels, 1, kernel]");
    weight_stride = static_cast<int>(weight.size(2));
  } else {
    TORCH_CHECK(weight.dim() == 2 && weight.size(0) == channels && weight.size(1) == kernel, "weight must be [channels, kernel]");
    weight_stride = static_cast<int>(weight.size(1));
  }
  bool has_bias = bias.numel() > 0;
  TORCH_CHECK(!has_bias || bias.numel() == channels, "bias must be [channels]");
  auto out = torch::empty_like(x);
  int threads = 256;
  int blocks = (batch * channels + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  depthwise_conv_update_spec_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<half*>(state_arena.data_ptr<at::Half>()),
      state_indices.data_ptr<int64_t>(),
      commit_tokens.data_ptr<int32_t>(),
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      has_bias ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr,
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      batch, tokens, slots, channels, history, kernel, weight_stride, has_bias);
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}

__global__ void depthwise_conv_update_spec_trajectory_kernel(
    const half* __restrict__ state_arena,
    const int64_t* __restrict__ state_indices,
    const half* __restrict__ x,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ out,
    half* __restrict__ trajectory,
    int batch,
    int tokens,
    int slots,
    int channels,
    int history,
    int kernel,
    int weight_stride,
    bool has_bias) {
  constexpr int MAX_HISTORY = 16;
  int linear = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * channels;
  if (linear >= total) return;
  int row = linear / channels;
  int c = linear - row * channels;
  int64_t slot64 = state_indices[row];
  if (slot64 < 0 || slot64 >= slots) return;
  int slot = static_cast<int>(slot64);
  int64_t state_base = (static_cast<int64_t>(slot) * channels + c) * history;
  int weight_base = c * weight_stride;

  float hist[MAX_HISTORY];
  for (int i = 0; i < MAX_HISTORY; ++i) hist[i] = 0.0f;
  for (int i = 0; i < history; ++i) {
    hist[i] = __half2float(state_arena[state_base + i]);
  }

  for (int t = 0; t < tokens; ++t) {
    float acc = 0.0f;
    for (int i = 0; i < history; ++i) {
      acc += hist[i] * __half2float(weight[weight_base + i]);
    }
    half xv_h = x[(static_cast<int64_t>(row) * tokens + t) * channels + c];
    float xv = __half2float(xv_h);
    acc += xv * __half2float(weight[weight_base + history]);
    if (has_bias) {
      acc += __half2float(bias[c]);
    }
    for (int i = 0; i < history - 1; ++i) {
      hist[i] = hist[i + 1];
    }
    if (history > 0) {
      hist[history - 1] = xv;
    }
    out[(static_cast<int64_t>(row) * tokens + t) * channels + c] = __float2half_rn(qb_silu(acc));
    int64_t traj_base = ((static_cast<int64_t>(row) * tokens + t) * channels + c) * history;
    for (int i = 0; i < history; ++i) {
      trajectory[traj_base + i] = __float2half_rn(hist[i]);
    }
  }
}

void depthwise_conv_update_spec_trajectory_out(
    torch::Tensor state_arena,
    torch::Tensor state_indices,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out,
    torch::Tensor trajectory) {
  QB_CHECK_CUDA(state_arena); QB_CHECK_CUDA(state_indices); QB_CHECK_CUDA(x); QB_CHECK_CUDA(weight); QB_CHECK_CUDA(bias);
  QB_CHECK_CUDA(out); QB_CHECK_CUDA(trajectory);
  QB_CHECK_CONTIGUOUS(state_arena); QB_CHECK_CONTIGUOUS(state_indices); QB_CHECK_CONTIGUOUS(x); QB_CHECK_CONTIGUOUS(weight); QB_CHECK_CONTIGUOUS(bias);
  QB_CHECK_CONTIGUOUS(out); QB_CHECK_CONTIGUOUS(trajectory);
  QB_CHECK_HALF(state_arena); QB_CHECK_HALF(x); QB_CHECK_HALF(weight);
  QB_CHECK_HALF(out); QB_CHECK_HALF(trajectory);
  QB_CHECK_INT64(state_indices);
  TORCH_CHECK(bias.numel() == 0 || bias.scalar_type() == at::kHalf, "bias must be empty or fp16");
  TORCH_CHECK(state_arena.dim() == 3, "state_arena must be [slots, channels, kernel-1]");
  TORCH_CHECK(x.dim() == 3, "x must be [batch, tokens, channels]");
  int slots = static_cast<int>(state_arena.size(0));
  int channels = static_cast<int>(state_arena.size(1));
  int history = static_cast<int>(state_arena.size(2));
  TORCH_CHECK(history <= 16, "depthwise_conv_update_spec_trajectory supports kernel history <= 16");
  int batch = static_cast<int>(x.size(0));
  int tokens = static_cast<int>(x.size(1));
  int kernel = history + 1;
  TORCH_CHECK(x.size(2) == channels, "x channels mismatch");
  TORCH_CHECK(out.dim() == 3 && out.size(0) == batch && out.size(1) == tokens && out.size(2) == channels,
              "out must be [batch, tokens, channels]");
  TORCH_CHECK(trajectory.dim() == 4 && trajectory.size(0) == batch && trajectory.size(1) == tokens &&
              trajectory.size(2) == channels && trajectory.size(3) == history,
              "trajectory must be [batch, tokens, channels, history]");
  TORCH_CHECK(state_indices.dim() == 1 && state_indices.size(0) == batch, "state_indices must be [batch]");
  int weight_stride = 0;
  if (weight.dim() == 3) {
    TORCH_CHECK(weight.size(0) == channels && weight.size(2) == kernel, "weight must be [channels, 1, kernel]");
    weight_stride = static_cast<int>(weight.size(2));
  } else {
    TORCH_CHECK(weight.dim() == 2 && weight.size(0) == channels && weight.size(1) == kernel, "weight must be [channels, kernel]");
    weight_stride = static_cast<int>(weight.size(1));
  }
  bool has_bias = bias.numel() > 0;
  TORCH_CHECK(!has_bias || bias.numel() == channels, "bias must be [channels]");
  int threads = 256;
  int blocks = (batch * channels + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  depthwise_conv_update_spec_trajectory_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(state_arena.data_ptr<at::Half>()),
      state_indices.data_ptr<int64_t>(),
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      has_bias ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr,
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<half*>(trajectory.data_ptr<at::Half>()),
      batch, tokens, slots, channels, history, kernel, weight_stride, has_bias);
  QB_CUDA_CHECK(cudaGetLastError());
}

std::vector<torch::Tensor> depthwise_conv_update_spec_trajectory(
    torch::Tensor state_arena,
    torch::Tensor state_indices,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias) {
  auto out = torch::empty_like(x);
  int batch = static_cast<int>(x.size(0));
  int tokens = static_cast<int>(x.size(1));
  int channels = static_cast<int>(x.size(2));
  int history = static_cast<int>(state_arena.size(2));
  auto trajectory = torch::empty({batch, tokens, channels, history}, x.options());
  depthwise_conv_update_spec_trajectory_out(state_arena, state_indices, x, weight, bias, out, trajectory);
  return {out, trajectory};
}

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

__global__ void gdn_recurrent_ab_128_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const half* __restrict__ a,
    const half* __restrict__ b,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    half* __restrict__ state,
    half* __restrict__ out,
    int kv_heads,
    int v_heads) {
  int vh = blockIdx.x;
  int tid = threadIdx.x;
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

  float ax = __half2float(a[vh]) + dt_bias[vh];
  float softplus = ax > 20.0f ? ax : log1pf(__expf(ax));
  float g = -__expf(A_log[vh]) * softplus;
  float decay = __expf(g);
  float beta = qb_sigmoid(__half2float(b[vh]));

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

torch::Tensor gdn_recurrent_ab(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b, torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor state) {
  QB_CHECK_CUDA(q); QB_CHECK_CUDA(k); QB_CHECK_CUDA(v); QB_CHECK_CUDA(a); QB_CHECK_CUDA(b); QB_CHECK_CUDA(A_log); QB_CHECK_CUDA(dt_bias); QB_CHECK_CUDA(state);
  QB_CHECK_CONTIGUOUS(q); QB_CHECK_CONTIGUOUS(k); QB_CHECK_CONTIGUOUS(v); QB_CHECK_CONTIGUOUS(a); QB_CHECK_CONTIGUOUS(b); QB_CHECK_CONTIGUOUS(A_log); QB_CHECK_CONTIGUOUS(dt_bias); QB_CHECK_CONTIGUOUS(state);
  QB_CHECK_HALF(q); QB_CHECK_HALF(k); QB_CHECK_HALF(v); QB_CHECK_HALF(a); QB_CHECK_HALF(b); QB_CHECK_HALF(state);
  TORCH_CHECK(A_log.scalar_type() == at::kFloat && dt_bias.scalar_type() == at::kFloat, "A_log/dt_bias must be fp32");
  TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && v.dim() == 2, "q/k/v shapes must be [heads, 128]");
  TORCH_CHECK(q.size(1) == QB_GDN_D && k.size(1) == QB_GDN_D && v.size(1) == QB_GDN_D, "head dim must be 128");
  int kv_heads = static_cast<int>(q.size(0));
  int v_heads = static_cast<int>(v.size(0));
  TORCH_CHECK(k.size(0) == kv_heads, "k heads mismatch");
  TORCH_CHECK(v_heads % kv_heads == 0, "v_heads must be divisible by kv_heads");
  TORCH_CHECK(a.size(0) == v_heads && b.size(0) == v_heads && A_log.size(0) == v_heads && dt_bias.size(0) == v_heads, "a/b/A_log/dt_bias v_heads mismatch");
  TORCH_CHECK(state.dim() == 3 && state.size(0) == v_heads && state.size(1) == QB_GDN_D && state.size(2) == QB_GDN_D,
              "state must be [v_heads, 128, 128]");

  auto out = torch::empty({v_heads, QB_GDN_D}, q.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  gdn_recurrent_ab_128_kernel<<<v_heads, QB_GDN_BLOCK, 0, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
      A_log.data_ptr<float>(),
      dt_bias.data_ptr<float>(),
      reinterpret_cast<half*>(state.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      kv_heads, v_heads);
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}

__global__ void gdn_recurrent_ab_batch_128_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const half* __restrict__ a,
    const half* __restrict__ b,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    half* __restrict__ state_arena,
    const int64_t* __restrict__ state_indices,
    half* __restrict__ out,
    int batch,
    int slots,
    int kv_heads,
    int v_heads) {
  int vh = blockIdx.x;
  int row = blockIdx.y;
  int tid = threadIdx.x;
  if (row >= batch || vh >= v_heads || tid >= QB_GDN_D) return;
  int64_t slot64 = state_indices[row];
  if (slot64 < 0 || slot64 >= slots) return;
  int slot = static_cast<int>(slot64);

  int ratio = v_heads / kv_heads;
  int kh = vh / ratio;

  __shared__ float q_s[QB_GDN_D];
  __shared__ float k_s[QB_GDN_D];
  __shared__ float red[QB_GDN_BLOCK];
  __shared__ float delta_s[QB_GDN_D];

  const int64_t q_off = (static_cast<int64_t>(row) * kv_heads + kh) * QB_GDN_D;
  const int64_t v_off = (static_cast<int64_t>(row) * v_heads + vh) * QB_GDN_D;
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
  half* S = state_arena + ((static_cast<int64_t>(slot) * v_heads + vh) * QB_GDN_D * QB_GDN_D);

  float old_j = 0.0f;
  #pragma unroll
  for (int d = 0; d < QB_GDN_D; ++d) {
    old_j += k_s[d] * __half2float(S[d * QB_GDN_D + j]);
  }

  float vj = __half2float(v[v_off + j]);
  float delta = vj - old_j;
  delta_s[j] = delta;
  __syncthreads();

  const int64_t gate_off = static_cast<int64_t>(row) * v_heads + vh;
  float ax = __half2float(a[gate_off]) + dt_bias[vh];
  float softplus = ax > 20.0f ? ax : log1pf(__expf(ax));
  float g = -__expf(A_log[vh]) * softplus;
  float decay = __expf(g);
  float beta = qb_sigmoid(__half2float(b[gate_off]));

  #pragma unroll
  for (int d = 0; d < QB_GDN_D; ++d) {
    float s_old = __half2float(S[d * QB_GDN_D + j]);
    float s_new = decay * s_old + beta * k_s[d] * delta;
    S[d * QB_GDN_D + j] = __float2half_rn(s_new);
  }
  __syncthreads();

  // Compute the output from the updated state using the original q row.  On
  // Ada/SM89 the shared q buffer path showed head-dependent drift for the real
  // Qwen shape (kv_heads=16, v_heads=48) even though state updates matched the
  // scalar per-row kernel.  Keeping the state update shared-memory path and
  // deriving q directly here preserves the batch-state contract without stale
  // or corrupted q fragments.
  float q_norm_sum = 0.0f;
  #pragma unroll
  for (int d = 0; d < QB_GDN_D; ++d) {
    float qd = __half2float(q[q_off + d]);
    q_norm_sum += qd * qd;
  }
  float q_scale = rsqrtf(q_norm_sum + 1e-6f) * rsqrtf(static_cast<float>(QB_GDN_D));
  float yj = 0.0f;
  #pragma unroll
  for (int d = 0; d < QB_GDN_D; ++d) {
    yj += (__half2float(q[q_off + d]) * q_scale) * __half2float(S[d * QB_GDN_D + j]);
  }
  out[v_off + j] = __float2half_rn(yj);
}

torch::Tensor gdn_recurrent_ab_batch(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    torch::Tensor state_arena,
    torch::Tensor state_indices) {
  QB_CHECK_CUDA(q); QB_CHECK_CUDA(k); QB_CHECK_CUDA(v); QB_CHECK_CUDA(a); QB_CHECK_CUDA(b); QB_CHECK_CUDA(A_log); QB_CHECK_CUDA(dt_bias); QB_CHECK_CUDA(state_arena); QB_CHECK_CUDA(state_indices);
  QB_CHECK_CONTIGUOUS(q); QB_CHECK_CONTIGUOUS(k); QB_CHECK_CONTIGUOUS(v); QB_CHECK_CONTIGUOUS(a); QB_CHECK_CONTIGUOUS(b); QB_CHECK_CONTIGUOUS(A_log); QB_CHECK_CONTIGUOUS(dt_bias); QB_CHECK_CONTIGUOUS(state_arena); QB_CHECK_CONTIGUOUS(state_indices);
  QB_CHECK_HALF(q); QB_CHECK_HALF(k); QB_CHECK_HALF(v); QB_CHECK_HALF(a); QB_CHECK_HALF(b); QB_CHECK_HALF(state_arena);
  QB_CHECK_INT64(state_indices);
  TORCH_CHECK(A_log.scalar_type() == at::kFloat && dt_bias.scalar_type() == at::kFloat, "A_log/dt_bias must be fp32");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q/k/v shapes must be [batch, heads, 128]");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "q/k/v batch mismatch");
  TORCH_CHECK(q.size(2) == QB_GDN_D && k.size(2) == QB_GDN_D && v.size(2) == QB_GDN_D, "head dim must be 128");
  int batch = static_cast<int>(q.size(0));
  int kv_heads = static_cast<int>(q.size(1));
  int v_heads = static_cast<int>(v.size(1));
  TORCH_CHECK(k.size(1) == kv_heads, "k heads mismatch");
  TORCH_CHECK(v_heads % kv_heads == 0, "v_heads must be divisible by kv_heads");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(0) == batch && b.size(0) == batch, "a/b must be [batch, v_heads]");
  TORCH_CHECK(a.size(1) == v_heads && b.size(1) == v_heads, "a/b v_heads mismatch");
  TORCH_CHECK(A_log.size(0) == v_heads && dt_bias.size(0) == v_heads, "A_log/dt_bias v_heads mismatch");
  TORCH_CHECK(state_indices.dim() == 1 && state_indices.size(0) == batch, "state_indices must be [batch]");
  TORCH_CHECK(state_arena.dim() == 4 && state_arena.size(1) == v_heads && state_arena.size(2) == QB_GDN_D && state_arena.size(3) == QB_GDN_D,
              "state_arena must be [slots, v_heads, 128, 128]");
  int slots = static_cast<int>(state_arena.size(0));

  auto out = torch::empty({batch, v_heads, QB_GDN_D}, q.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(v_heads, batch);
  gdn_recurrent_ab_batch_128_kernel<<<grid, QB_GDN_BLOCK, 0, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
      A_log.data_ptr<float>(),
      dt_bias.data_ptr<float>(),
      reinterpret_cast<half*>(state_arena.data_ptr<at::Half>()),
      state_indices.data_ptr<int64_t>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      batch, slots, kv_heads, v_heads);
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}

__global__ void gdn_recurrent_ab_batch_norm_gate_128_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const half* __restrict__ a,
    const half* __restrict__ b,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    half* __restrict__ state_arena,
    const int64_t* __restrict__ state_indices,
    const half* __restrict__ norm_w,
    const half* __restrict__ z,
    half* __restrict__ out,
    int batch,
    int slots,
    int kv_heads,
    int v_heads,
    float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= batch || tid >= QB_GDN_D) return;

  int64_t slot64 = state_indices[row];
  if (slot64 < 0 || slot64 >= slots) return;
  int slot = static_cast<int>(slot64);
  int ratio = v_heads / kv_heads;

  __shared__ float q_s[QB_GDN_D];
  __shared__ float k_s[QB_GDN_D];
  __shared__ float red[QB_GDN_BLOCK];
  float sumsq = 0.0f;

  for (int vh = 0; vh < v_heads; ++vh) {
    int kh = vh / ratio;
    const int64_t q_off = (static_cast<int64_t>(row) * kv_heads + kh) * QB_GDN_D;
    const int64_t v_off = (static_cast<int64_t>(row) * v_heads + vh) * QB_GDN_D;

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
    half* S = state_arena + ((static_cast<int64_t>(slot) * v_heads + vh) * QB_GDN_D * QB_GDN_D);
    float old_j = 0.0f;
    #pragma unroll
    for (int d = 0; d < QB_GDN_D; ++d) {
      old_j += k_s[d] * __half2float(S[d * QB_GDN_D + j]);
    }

    float vj = __half2float(v[v_off + j]);
    float delta = vj - old_j;
    float ax = __half2float(a[static_cast<int64_t>(row) * v_heads + vh]) + dt_bias[vh];
    float softplus = ax > 20.0f ? ax : log1pf(__expf(ax));
    float g = -__expf(A_log[vh]) * softplus;
    float decay = __expf(g);
    float beta = qb_sigmoid(__half2float(b[static_cast<int64_t>(row) * v_heads + vh]));

    #pragma unroll
    for (int d = 0; d < QB_GDN_D; ++d) {
      float s_old = __half2float(S[d * QB_GDN_D + j]);
      float s_new = decay * s_old + beta * k_s[d] * delta;
      S[d * QB_GDN_D + j] = __float2half_rn(s_new);
    }
    __syncthreads();

    // Keep this output path identical to gdn_recurrent_ab_batch_128_kernel.
    // Reusing the shared q_s fragment here caused head-dependent drift on the
    // real Qwen3.6 shape after the state update path was fused with norm/gate.
    float q_norm_sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < QB_GDN_D; ++d) {
      float qd = __half2float(q[q_off + d]);
      q_norm_sum += qd * qd;
    }
    float q_scale = rsqrtf(q_norm_sum + 1e-6f) * rsqrtf(static_cast<float>(QB_GDN_D));
    float yj = 0.0f;
    #pragma unroll
    for (int d = 0; d < QB_GDN_D; ++d) {
      yj += (__half2float(q[q_off + d]) * q_scale) * __half2float(S[d * QB_GDN_D + j]);
    }
    out[v_off + j] = __float2half_rn(yj);
    sumsq += yj * yj;
    __syncthreads();
  }

  red[tid] = sumsq;
  __syncthreads();
  for (int stride = QB_GDN_BLOCK / 2; stride > 0; stride >>= 1) {
    if (tid < stride) red[tid] += red[tid + stride];
    __syncthreads();
  }
  float inv_rms = rsqrtf(red[0] / static_cast<float>(v_heads * QB_GDN_D) + eps);

  int hidden = v_heads * QB_GDN_D;
  for (int idx = tid; idx < hidden; idx += QB_GDN_BLOCK) {
    float yv = __half2float(out[static_cast<int64_t>(row) * hidden + idx]);
    float ww = __half2float(norm_w[idx]);
    float zv = __half2float(z[static_cast<int64_t>(row) * hidden + idx]);
    out[static_cast<int64_t>(row) * hidden + idx] = __float2half_rn(yv * inv_rms * ww * qb_silu(zv));
  }
}

torch::Tensor gdn_recurrent_ab_batch_norm_gate(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    torch::Tensor state_arena,
    torch::Tensor state_indices,
    torch::Tensor norm_w,
    torch::Tensor z,
    double eps) {
  QB_CHECK_CUDA(q); QB_CHECK_CUDA(k); QB_CHECK_CUDA(v); QB_CHECK_CUDA(a); QB_CHECK_CUDA(b); QB_CHECK_CUDA(A_log); QB_CHECK_CUDA(dt_bias); QB_CHECK_CUDA(state_arena); QB_CHECK_CUDA(state_indices); QB_CHECK_CUDA(norm_w); QB_CHECK_CUDA(z);
  QB_CHECK_CONTIGUOUS(q); QB_CHECK_CONTIGUOUS(k); QB_CHECK_CONTIGUOUS(v); QB_CHECK_CONTIGUOUS(a); QB_CHECK_CONTIGUOUS(b); QB_CHECK_CONTIGUOUS(A_log); QB_CHECK_CONTIGUOUS(dt_bias); QB_CHECK_CONTIGUOUS(state_arena); QB_CHECK_CONTIGUOUS(state_indices); QB_CHECK_CONTIGUOUS(norm_w); QB_CHECK_CONTIGUOUS(z);
  QB_CHECK_HALF(q); QB_CHECK_HALF(k); QB_CHECK_HALF(v); QB_CHECK_HALF(a); QB_CHECK_HALF(b); QB_CHECK_HALF(state_arena); QB_CHECK_HALF(norm_w); QB_CHECK_HALF(z);
  QB_CHECK_INT64(state_indices);
  TORCH_CHECK(A_log.scalar_type() == at::kFloat && dt_bias.scalar_type() == at::kFloat, "A_log/dt_bias must be fp32");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q/k/v must be [batch, heads, 128]");
  int batch = static_cast<int>(q.size(0));
  int kv_heads = static_cast<int>(q.size(1));
  int v_heads = static_cast<int>(v.size(1));
  TORCH_CHECK(q.size(2) == QB_GDN_D && k.size(2) == QB_GDN_D && v.size(2) == QB_GDN_D, "head dim must be 128");
  TORCH_CHECK(k.size(0) == batch && k.size(1) == kv_heads, "k shape mismatch");
  TORCH_CHECK(v.size(0) == batch && v_heads % kv_heads == 0, "v shape mismatch");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(0) == batch && b.size(0) == batch && a.size(1) == v_heads && b.size(1) == v_heads, "a/b must be [batch, v_heads]");
  TORCH_CHECK(z.dim() == 3 && z.size(0) == batch && z.size(1) == v_heads && z.size(2) == QB_GDN_D, "z must be [batch, v_heads, 128]");
  TORCH_CHECK(norm_w.numel() == v_heads * QB_GDN_D, "norm_w hidden mismatch");
  TORCH_CHECK(state_arena.dim() == 4 && state_arena.size(1) == v_heads && state_arena.size(2) == QB_GDN_D && state_arena.size(3) == QB_GDN_D, "state_arena must be [slots, v_heads, 128, 128]");
  TORCH_CHECK(state_indices.dim() == 1 && state_indices.size(0) == batch, "state_indices must be [batch]");
  int slots = static_cast<int>(state_arena.size(0));

  auto out = torch::empty({batch, v_heads, QB_GDN_D}, q.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(batch);
  gdn_recurrent_ab_batch_norm_gate_128_kernel<<<grid, QB_GDN_BLOCK, 0, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
      A_log.data_ptr<float>(),
      dt_bias.data_ptr<float>(),
      reinterpret_cast<half*>(state_arena.data_ptr<at::Half>()),
      state_indices.data_ptr<int64_t>(),
      reinterpret_cast<const half*>(norm_w.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(z.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      batch, slots, kv_heads, v_heads, static_cast<float>(eps));
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

__global__ void gdn_recurrent_ab_scan_128_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const half* __restrict__ a,
    const half* __restrict__ b,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
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

    float ax = __half2float(a[static_cast<int64_t>(t) * v_heads + vh]) + dt_bias[vh];
    float softplus = ax > 20.0f ? ax : log1pf(__expf(ax));
    float g = -__expf(A_log[vh]) * softplus;
    float decay = __expf(g);
    float beta = qb_sigmoid(__half2float(b[static_cast<int64_t>(t) * v_heads + vh]));

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

torch::Tensor gdn_recurrent_ab_scan(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    torch::Tensor state) {
  QB_CHECK_CUDA(q); QB_CHECK_CUDA(k); QB_CHECK_CUDA(v); QB_CHECK_CUDA(a); QB_CHECK_CUDA(b); QB_CHECK_CUDA(A_log); QB_CHECK_CUDA(dt_bias); QB_CHECK_CUDA(state);
  QB_CHECK_CONTIGUOUS(q); QB_CHECK_CONTIGUOUS(k); QB_CHECK_CONTIGUOUS(v); QB_CHECK_CONTIGUOUS(a); QB_CHECK_CONTIGUOUS(b); QB_CHECK_CONTIGUOUS(A_log); QB_CHECK_CONTIGUOUS(dt_bias); QB_CHECK_CONTIGUOUS(state);
  QB_CHECK_HALF(q); QB_CHECK_HALF(k); QB_CHECK_HALF(v); QB_CHECK_HALF(a); QB_CHECK_HALF(b); QB_CHECK_HALF(state);
  TORCH_CHECK(A_log.scalar_type() == at::kFloat && dt_bias.scalar_type() == at::kFloat, "A_log/dt_bias must be fp32");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q/k/v shapes must be [tokens, heads, 128]");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "q/k/v token count mismatch");
  TORCH_CHECK(q.size(2) == QB_GDN_D && k.size(2) == QB_GDN_D && v.size(2) == QB_GDN_D, "head dim must be 128");
  int tokens = static_cast<int>(q.size(0));
  int kv_heads = static_cast<int>(q.size(1));
  int v_heads = static_cast<int>(v.size(1));
  TORCH_CHECK(k.size(1) == kv_heads, "k heads mismatch");
  TORCH_CHECK(v_heads % kv_heads == 0, "v_heads must be divisible by kv_heads");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(0) == tokens && b.size(0) == tokens, "a/b must be [tokens, v_heads]");
  TORCH_CHECK(a.size(1) == v_heads && b.size(1) == v_heads, "a/b v_heads mismatch");
  TORCH_CHECK(A_log.size(0) == v_heads && dt_bias.size(0) == v_heads, "A_log/dt_bias v_heads mismatch");
  TORCH_CHECK(state.dim() == 3 && state.size(0) == v_heads && state.size(1) == QB_GDN_D && state.size(2) == QB_GDN_D,
              "state must be [v_heads, 128, 128]");

  auto out = torch::empty({tokens, v_heads, QB_GDN_D}, q.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  gdn_recurrent_ab_scan_128_kernel<<<v_heads, QB_GDN_BLOCK, 0, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
      A_log.data_ptr<float>(),
      dt_bias.data_ptr<float>(),
      reinterpret_cast<half*>(state.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      tokens, kv_heads, v_heads);
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}

__global__ void gdn_recurrent_ab_spec_128_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const half* __restrict__ a,
    const half* __restrict__ b,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    half* __restrict__ state_arena,
    const int64_t* __restrict__ state_indices,
    const int32_t* __restrict__ commit_tokens,
    half* __restrict__ out,
    int batch,
    int tokens,
    int slots,
    int kv_heads,
    int v_heads) {
  int row = blockIdx.x;
  int vh = blockIdx.y;
  int tid = threadIdx.x;
  if (row >= batch || vh >= v_heads || tid >= QB_GDN_D) return;
  int64_t slot64 = state_indices[row];
  if (slot64 < 0 || slot64 >= slots) return;
  int slot = static_cast<int>(slot64);
  int commit_n = commit_tokens[row];
  if (commit_n < 0) commit_n = 0;
  if (commit_n > tokens) commit_n = tokens;

  int ratio = v_heads / kv_heads;
  int kh = vh / ratio;
  half* S_base = state_arena + ((static_cast<int64_t>(slot) * v_heads + vh) * QB_GDN_D * QB_GDN_D);

  __shared__ float q_s[QB_GDN_D];
  __shared__ float k_s[QB_GDN_D];
  __shared__ float red[QB_GDN_BLOCK];
  __shared__ float delta_s[QB_GDN_D];

  float S_col[QB_GDN_D];
  int j = tid;
  #pragma unroll
  for (int d = 0; d < QB_GDN_D; ++d) {
    S_col[d] = __half2float(S_base[d * QB_GDN_D + j]);
  }

  for (int t = 0; t < tokens; ++t) {
    const int64_t q_off = ((static_cast<int64_t>(row) * tokens + t) * kv_heads + kh) * QB_GDN_D;
    const int64_t v_off = ((static_cast<int64_t>(row) * tokens + t) * v_heads + vh) * QB_GDN_D;
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

    float old_j = 0.0f;
    #pragma unroll
    for (int d = 0; d < QB_GDN_D; ++d) {
      old_j += k_s[d] * S_col[d];
    }

    float vj = __half2float(v[v_off + j]);
    float delta = vj - old_j;
    delta_s[j] = delta;
    __syncthreads();

    float ax = __half2float(a[(static_cast<int64_t>(row) * tokens + t) * v_heads + vh]) + dt_bias[vh];
    float softplus = ax > 20.0f ? ax : log1pf(__expf(ax));
    float g = -__expf(A_log[vh]) * softplus;
    float decay = __expf(g);
    float beta = qb_sigmoid(__half2float(b[(static_cast<int64_t>(row) * tokens + t) * v_heads + vh]));

    #pragma unroll
    for (int d = 0; d < QB_GDN_D; ++d) {
      S_col[d] = decay * S_col[d] + beta * k_s[d] * delta;
    }
    __syncthreads();

    float yj = 0.0f;
    #pragma unroll
    for (int d = 0; d < QB_GDN_D; ++d) {
      yj += q_s[d] * S_col[d];
    }
    out[v_off + j] = __float2half_rn(yj);

    if (t + 1 == commit_n) {
      #pragma unroll
      for (int d = 0; d < QB_GDN_D; ++d) {
        S_base[d * QB_GDN_D + j] = __float2half_rn(S_col[d]);
      }
    }
    __syncthreads();
  }
}

torch::Tensor gdn_recurrent_ab_spec(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    torch::Tensor state_arena,
    torch::Tensor state_indices,
    torch::Tensor commit_tokens) {
  QB_CHECK_CUDA(q); QB_CHECK_CUDA(k); QB_CHECK_CUDA(v); QB_CHECK_CUDA(a); QB_CHECK_CUDA(b); QB_CHECK_CUDA(A_log); QB_CHECK_CUDA(dt_bias); QB_CHECK_CUDA(state_arena); QB_CHECK_CUDA(state_indices); QB_CHECK_CUDA(commit_tokens);
  QB_CHECK_CONTIGUOUS(q); QB_CHECK_CONTIGUOUS(k); QB_CHECK_CONTIGUOUS(v); QB_CHECK_CONTIGUOUS(a); QB_CHECK_CONTIGUOUS(b); QB_CHECK_CONTIGUOUS(A_log); QB_CHECK_CONTIGUOUS(dt_bias); QB_CHECK_CONTIGUOUS(state_arena); QB_CHECK_CONTIGUOUS(state_indices); QB_CHECK_CONTIGUOUS(commit_tokens);
  QB_CHECK_HALF(q); QB_CHECK_HALF(k); QB_CHECK_HALF(v); QB_CHECK_HALF(a); QB_CHECK_HALF(b); QB_CHECK_HALF(state_arena);
  QB_CHECK_INT64(state_indices);
  TORCH_CHECK(commit_tokens.scalar_type() == at::kInt, "commit_tokens must be int32");
  TORCH_CHECK(A_log.scalar_type() == at::kFloat && dt_bias.scalar_type() == at::kFloat, "A_log/dt_bias must be fp32");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v shapes must be [batch, tokens, heads, 128]");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "q/k/v batch mismatch");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1), "q/k/v token count mismatch");
  TORCH_CHECK(q.size(3) == QB_GDN_D && k.size(3) == QB_GDN_D && v.size(3) == QB_GDN_D, "head dim must be 128");
  int batch = static_cast<int>(q.size(0));
  int tokens = static_cast<int>(q.size(1));
  int kv_heads = static_cast<int>(q.size(2));
  int v_heads = static_cast<int>(v.size(2));
  int slots = static_cast<int>(state_arena.size(0));
  TORCH_CHECK(k.size(2) == kv_heads, "k heads mismatch");
  TORCH_CHECK(v_heads % kv_heads == 0, "v_heads must be divisible by kv_heads");
  TORCH_CHECK(a.dim() == 3 && b.dim() == 3 && a.size(0) == batch && b.size(0) == batch, "a/b must be [batch, tokens, v_heads]");
  TORCH_CHECK(a.size(1) == tokens && b.size(1) == tokens && a.size(2) == v_heads && b.size(2) == v_heads, "a/b shape mismatch");
  TORCH_CHECK(A_log.size(0) == v_heads && dt_bias.size(0) == v_heads, "A_log/dt_bias v_heads mismatch");
  TORCH_CHECK(state_arena.dim() == 4 && state_arena.size(1) == v_heads && state_arena.size(2) == QB_GDN_D && state_arena.size(3) == QB_GDN_D,
              "state_arena must be [slots, v_heads, 128, 128]");
  TORCH_CHECK(state_indices.dim() == 1 && state_indices.size(0) == batch, "state_indices must be [batch]");
  TORCH_CHECK(commit_tokens.dim() == 1 && commit_tokens.size(0) == batch, "commit_tokens must be [batch]");

  auto out = torch::empty({batch, tokens, v_heads, QB_GDN_D}, q.options());
  dim3 grid(batch, v_heads);
  auto stream = at::cuda::getCurrentCUDAStream();
  gdn_recurrent_ab_spec_128_kernel<<<grid, QB_GDN_BLOCK, 0, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
      A_log.data_ptr<float>(),
      dt_bias.data_ptr<float>(),
      reinterpret_cast<half*>(state_arena.data_ptr<at::Half>()),
      state_indices.data_ptr<int64_t>(),
      commit_tokens.data_ptr<int32_t>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      batch, tokens, slots, kv_heads, v_heads);
  QB_CUDA_CHECK(cudaGetLastError());
  return out;
}

__global__ void gdn_recurrent_ab_spec_trajectory_128_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const half* __restrict__ a,
    const half* __restrict__ b,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const half* __restrict__ state_arena,
    const int64_t* __restrict__ state_indices,
    half* __restrict__ out,
    half* __restrict__ trajectory,
    int batch,
    int tokens,
    int slots,
    int kv_heads,
    int v_heads) {
  int row = blockIdx.x;
  int vh = blockIdx.y;
  int tid = threadIdx.x;
  if (row >= batch || vh >= v_heads || tid >= QB_GDN_D) return;
  int64_t slot64 = state_indices[row];
  if (slot64 < 0 || slot64 >= slots) return;
  int slot = static_cast<int>(slot64);

  int ratio = v_heads / kv_heads;
  int kh = vh / ratio;
  const half* S_base = state_arena + ((static_cast<int64_t>(slot) * v_heads + vh) * QB_GDN_D * QB_GDN_D);

  __shared__ float q_s[QB_GDN_D];
  __shared__ float k_s[QB_GDN_D];
  __shared__ float red[QB_GDN_BLOCK];
  __shared__ float delta_s[QB_GDN_D];

  float S_col[QB_GDN_D];
  int j = tid;
  #pragma unroll
  for (int d = 0; d < QB_GDN_D; ++d) {
    S_col[d] = __half2float(S_base[d * QB_GDN_D + j]);
  }

  for (int t = 0; t < tokens; ++t) {
    const int64_t q_off = ((static_cast<int64_t>(row) * tokens + t) * kv_heads + kh) * QB_GDN_D;
    const int64_t v_off = ((static_cast<int64_t>(row) * tokens + t) * v_heads + vh) * QB_GDN_D;
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

    float old_j = 0.0f;
    #pragma unroll
    for (int d = 0; d < QB_GDN_D; ++d) {
      old_j += k_s[d] * S_col[d];
    }

    float vj = __half2float(v[v_off + j]);
    float delta = vj - old_j;
    delta_s[j] = delta;
    __syncthreads();

    float ax = __half2float(a[(static_cast<int64_t>(row) * tokens + t) * v_heads + vh]) + dt_bias[vh];
    float softplus = ax > 20.0f ? ax : log1pf(__expf(ax));
    float g = -__expf(A_log[vh]) * softplus;
    float decay = __expf(g);
    float beta = qb_sigmoid(__half2float(b[(static_cast<int64_t>(row) * tokens + t) * v_heads + vh]));

    #pragma unroll
    for (int d = 0; d < QB_GDN_D; ++d) {
      S_col[d] = decay * S_col[d] + beta * k_s[d] * delta;
    }
    __syncthreads();

    float yj = 0.0f;
    #pragma unroll
    for (int d = 0; d < QB_GDN_D; ++d) {
      yj += q_s[d] * S_col[d];
    }
    out[v_off + j] = __float2half_rn(yj);

    half* traj_base = trajectory + ((((static_cast<int64_t>(row) * tokens + t) * v_heads + vh) * QB_GDN_D) * QB_GDN_D);
    #pragma unroll
    for (int d = 0; d < QB_GDN_D; ++d) {
      traj_base[d * QB_GDN_D + j] = __float2half_rn(S_col[d]);
    }
    __syncthreads();
  }
}

void gdn_recurrent_ab_spec_trajectory_out(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    torch::Tensor state_arena,
    torch::Tensor state_indices,
    torch::Tensor out,
    torch::Tensor trajectory) {
  QB_CHECK_CUDA(q); QB_CHECK_CUDA(k); QB_CHECK_CUDA(v); QB_CHECK_CUDA(a); QB_CHECK_CUDA(b); QB_CHECK_CUDA(A_log); QB_CHECK_CUDA(dt_bias); QB_CHECK_CUDA(state_arena); QB_CHECK_CUDA(state_indices);
  QB_CHECK_CUDA(out); QB_CHECK_CUDA(trajectory);
  QB_CHECK_CONTIGUOUS(q); QB_CHECK_CONTIGUOUS(k); QB_CHECK_CONTIGUOUS(v); QB_CHECK_CONTIGUOUS(a); QB_CHECK_CONTIGUOUS(b); QB_CHECK_CONTIGUOUS(A_log); QB_CHECK_CONTIGUOUS(dt_bias); QB_CHECK_CONTIGUOUS(state_arena); QB_CHECK_CONTIGUOUS(state_indices);
  QB_CHECK_CONTIGUOUS(out); QB_CHECK_CONTIGUOUS(trajectory);
  QB_CHECK_HALF(q); QB_CHECK_HALF(k); QB_CHECK_HALF(v); QB_CHECK_HALF(a); QB_CHECK_HALF(b); QB_CHECK_HALF(state_arena);
  QB_CHECK_HALF(out); QB_CHECK_HALF(trajectory);
  QB_CHECK_INT64(state_indices);
  TORCH_CHECK(A_log.scalar_type() == at::kFloat && dt_bias.scalar_type() == at::kFloat, "A_log/dt_bias must be fp32");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v shapes must be [batch, tokens, heads, 128]");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "q/k/v batch mismatch");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1), "q/k/v token count mismatch");
  TORCH_CHECK(q.size(3) == QB_GDN_D && k.size(3) == QB_GDN_D && v.size(3) == QB_GDN_D, "head dim must be 128");
  int batch = static_cast<int>(q.size(0));
  int tokens = static_cast<int>(q.size(1));
  int kv_heads = static_cast<int>(q.size(2));
  int v_heads = static_cast<int>(v.size(2));
  int slots = static_cast<int>(state_arena.size(0));
  TORCH_CHECK(k.size(2) == kv_heads, "k heads mismatch");
  TORCH_CHECK(v_heads % kv_heads == 0, "v_heads must be divisible by kv_heads");
  TORCH_CHECK(a.dim() == 3 && b.dim() == 3 && a.size(0) == batch && b.size(0) == batch, "a/b must be [batch, tokens, v_heads]");
  TORCH_CHECK(a.size(1) == tokens && b.size(1) == tokens && a.size(2) == v_heads && b.size(2) == v_heads, "a/b shape mismatch");
  TORCH_CHECK(A_log.size(0) == v_heads && dt_bias.size(0) == v_heads, "A_log/dt_bias v_heads mismatch");
  TORCH_CHECK(state_arena.dim() == 4 && state_arena.size(1) == v_heads && state_arena.size(2) == QB_GDN_D && state_arena.size(3) == QB_GDN_D,
              "state_arena must be [slots, v_heads, 128, 128]");
  TORCH_CHECK(state_indices.dim() == 1 && state_indices.size(0) == batch, "state_indices must be [batch]");
  TORCH_CHECK(out.dim() == 4 && out.size(0) == batch && out.size(1) == tokens &&
              out.size(2) == v_heads && out.size(3) == QB_GDN_D,
              "out must be [batch, tokens, v_heads, 128]");
  TORCH_CHECK(trajectory.dim() == 5 && trajectory.size(0) == batch && trajectory.size(1) == tokens &&
              trajectory.size(2) == v_heads && trajectory.size(3) == QB_GDN_D && trajectory.size(4) == QB_GDN_D,
              "trajectory must be [batch, tokens, v_heads, 128, 128]");

  dim3 grid(batch, v_heads);
  auto stream = at::cuda::getCurrentCUDAStream();
  gdn_recurrent_ab_spec_trajectory_128_kernel<<<grid, QB_GDN_BLOCK, 0, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
      A_log.data_ptr<float>(),
      dt_bias.data_ptr<float>(),
      reinterpret_cast<const half*>(state_arena.data_ptr<at::Half>()),
      state_indices.data_ptr<int64_t>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<half*>(trajectory.data_ptr<at::Half>()),
      batch, tokens, slots, kv_heads, v_heads);
  QB_CUDA_CHECK(cudaGetLastError());
}

std::vector<torch::Tensor> gdn_recurrent_ab_spec_trajectory(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    torch::Tensor state_arena,
    torch::Tensor state_indices) {
  int batch = static_cast<int>(q.size(0));
  int tokens = static_cast<int>(q.size(1));
  int v_heads = static_cast<int>(v.size(2));
  auto out = torch::empty({batch, tokens, v_heads, QB_GDN_D}, q.options());
  auto trajectory = torch::empty({batch, tokens, v_heads, QB_GDN_D, QB_GDN_D}, q.options());
  gdn_recurrent_ab_spec_trajectory_out(q, k, v, a, b, A_log, dt_bias, state_arena, state_indices, out, trajectory);
  return {out, trajectory};
}
