#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

#define QB_CUDA_CHECK(err) do {                                      \
  cudaError_t err__ = (err);                                         \
  if (err__ != cudaSuccess) {                                        \
    throw std::runtime_error(std::string("CUDA error: ") +          \
      cudaGetErrorString(err__) + " at " + __FILE__ + ":" +        \
      std::to_string(__LINE__));                                     \
  }                                                                  \
} while (0)

#define QB_CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define QB_CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define QB_CHECK_HALF(x) TORCH_CHECK((x).scalar_type() == at::kHalf, #x " must be fp16")
#define QB_CHECK_FP8_E4M3(x) TORCH_CHECK((x).scalar_type() == at::kFloat8_e4m3fn, #x " must be fp8_e4m3fn")
#define QB_CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be fp32")
#define QB_CHECK_UINT8(x) TORCH_CHECK((x).scalar_type() == at::kByte, #x " must be uint8")
#define QB_CHECK_INT64(x) TORCH_CHECK((x).scalar_type() == at::kLong, #x " must be int64")

// Launchers exposed through pybind.
torch::Tensor lowbit_gemv(torch::Tensor qweight, torch::Tensor scales, torch::Tensor x, int64_t cols, int64_t group_size, int64_t bits, int64_t rows_per_cta);
torch::Tensor lowbit_gemm(torch::Tensor qweight, torch::Tensor scales, torch::Tensor x, int64_t cols, int64_t group_size, int64_t bits, int64_t rows_per_cta);
torch::Tensor lowbit_marlin_gemm(torch::Tensor qweight, torch::Tensor scales, torch::Tensor x, int64_t cols, int64_t group_size);
void lowbit_marlin_gemm_out(torch::Tensor qweight, torch::Tensor scales, torch::Tensor x, torch::Tensor out, torch::Tensor workspace, int64_t cols, int64_t group_size);
void lowbit_marlin_gemm_silu_packed_out(torch::Tensor qweight, torch::Tensor scales, torch::Tensor mixed, torch::Tensor out, torch::Tensor workspace, int64_t cols, int64_t group_size);
void lowbit_marlin_mlp_streaming_out(torch::Tensor gate_up_qweight, torch::Tensor gate_up_scales, torch::Tensor down_qweight, torch::Tensor down_scales, torch::Tensor x, torch::Tensor out, torch::Tensor accum, torch::Tensor sync, int64_t epoch, int64_t hidden, int64_t intermediate, int64_t gate_group_size, int64_t down_group_size);
std::vector<torch::Tensor> lowbit_gemv_pair(
    torch::Tensor qweight_a,
    torch::Tensor scales_a,
    torch::Tensor qweight_b,
    torch::Tensor scales_b,
    torch::Tensor x,
    int64_t cols,
    int64_t group_size,
    int64_t bits,
    int64_t rows_per_cta);
torch::Tensor lowbit_row_dequant(torch::Tensor qweight, torch::Tensor scales, torch::Tensor row, int64_t cols, int64_t group_size, int64_t bits);
torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor weight, double eps);
torch::Tensor rmsnorm_qwen(torch::Tensor x, torch::Tensor weight, double eps);
torch::Tensor rmsnorm_silu_gate(torch::Tensor x, torch::Tensor weight, torch::Tensor gate, double eps);
torch::Tensor rmsnorm_qwen_silu_gate(torch::Tensor x, torch::Tensor weight, torch::Tensor gate, double eps);
torch::Tensor silu_mul(torch::Tensor gate, torch::Tensor up);
torch::Tensor silu_mul_packed(torch::Tensor mixed, int64_t hidden);
torch::Tensor sigmoid_mul(torch::Tensor x, torch::Tensor gate);
torch::Tensor sigmoid_mul_repeat_kv(torch::Tensor v, torch::Tensor gate, int64_t ratio);
torch::Tensor rmsnorm_qwen_pair_cat(torch::Tensor x0, torch::Tensor w0, torch::Tensor x1, torch::Tensor w1, double eps);
torch::Tensor rmsnorm_qwen_rope(torch::Tensor x, torch::Tensor weight, int64_t pos, int64_t rope_dim, double rope_theta, double eps);
torch::Tensor gdn_recurrent(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, torch::Tensor state);
torch::Tensor gdn_recurrent_ab(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b, torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor state);
torch::Tensor gdn_recurrent_scan(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, torch::Tensor state);
torch::Tensor gdn_recurrent_ab_scan(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b, torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor state);
torch::Tensor gdn_recurrent_ab_batch(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b, torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor state_arena, torch::Tensor state_indices);
torch::Tensor gdn_recurrent_ab_batch_norm_gate(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b, torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor state_arena, torch::Tensor state_indices, torch::Tensor norm_w, torch::Tensor z, double eps);
torch::Tensor gdn_recurrent_ab_spec(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b, torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor state_arena, torch::Tensor state_indices, torch::Tensor commit_tokens);
std::vector<torch::Tensor> gdn_recurrent_ab_spec_trajectory(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b, torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor state_arena, torch::Tensor state_indices);
void gdn_recurrent_ab_spec_trajectory_out(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b, torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor state_arena, torch::Tensor state_indices, torch::Tensor out, torch::Tensor trajectory);
torch::Tensor depthwise_conv_update(torch::Tensor state, torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
torch::Tensor depthwise_conv_update_scan(torch::Tensor state, torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
torch::Tensor depthwise_conv_update_batch(torch::Tensor state_arena, torch::Tensor state_indices, torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
torch::Tensor depthwise_conv_update_spec(torch::Tensor state_arena, torch::Tensor state_indices, torch::Tensor commit_tokens, torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
std::vector<torch::Tensor> depthwise_conv_update_spec_trajectory(torch::Tensor state_arena, torch::Tensor state_indices, torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
void depthwise_conv_update_spec_trajectory_out(torch::Tensor state_arena, torch::Tensor state_indices, torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out, torch::Tensor trajectory);
torch::Tensor attention_decode_fp16(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, int64_t seq_len, double softmax_scale);
torch::Tensor attention_decode_fp16_gated(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor gate, int64_t seq_len, double softmax_scale);
torch::Tensor attention_decode_fp16_gated_tkh(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor gate, int64_t seq_len, double softmax_scale);
torch::Tensor attention_decode_batch_fp16(torch::Tensor q, torch::Tensor k_new, torch::Tensor v_new, torch::Tensor k_arena, torch::Tensor v_arena, torch::Tensor state_indices, torch::Tensor write_indices, torch::Tensor live_lengths, torch::Tensor positions, bool use_ring, double softmax_scale);
torch::Tensor attention_decode_paged_fp16(torch::Tensor q, torch::Tensor k_new, torch::Tensor v_new, torch::Tensor k_pages, torch::Tensor v_pages, torch::Tensor slot_mapping, torch::Tensor block_tables, torch::Tensor seq_lens, int64_t block_size, double softmax_scale);
torch::Tensor attention_decode_paged_fp8_e4m3(torch::Tensor q, torch::Tensor k_new, torch::Tensor v_new, torch::Tensor k_pages, torch::Tensor v_pages, torch::Tensor slot_mapping, torch::Tensor block_tables, torch::Tensor seq_lens, int64_t block_size, double softmax_scale, double k_scale, double v_scale);
void attention_append_paged_int4(torch::Tensor k_new, torch::Tensor v_new, torch::Tensor k_pages, torch::Tensor v_pages, torch::Tensor k_scales, torch::Tensor v_scales, torch::Tensor k_zeros, torch::Tensor v_zeros, torch::Tensor slot_mapping, int64_t block_size, int64_t hadamard_order, bool bdr_k, bool rotate_v, bool tiled_layout);
void attention_append_paged_int4_spec(torch::Tensor k_new, torch::Tensor v_new, torch::Tensor k_pages, torch::Tensor v_pages, torch::Tensor k_scales, torch::Tensor v_scales, torch::Tensor k_zeros, torch::Tensor v_zeros, torch::Tensor slot_mapping, torch::Tensor commit_tokens, int64_t block_size, int64_t hadamard_order, bool bdr_k, bool rotate_v, bool tiled_layout);
torch::Tensor attention_spec_decode_paged_int4(torch::Tensor q, torch::Tensor k_new, torch::Tensor v_new, torch::Tensor k_pages, torch::Tensor v_pages, torch::Tensor k_scales, torch::Tensor v_scales, torch::Tensor k_zeros, torch::Tensor v_zeros, torch::Tensor block_tables, torch::Tensor base_seq_lens, int64_t block_size, double softmax_scale, int64_t hadamard_order, bool bdr_k, bool rotate_v, bool tiled_layout);
torch::Tensor attention_decode_paged_int4(torch::Tensor q, torch::Tensor k_new, torch::Tensor v_new, torch::Tensor k_pages, torch::Tensor v_pages, torch::Tensor k_scales, torch::Tensor v_scales, torch::Tensor k_zeros, torch::Tensor v_zeros, torch::Tensor slot_mapping, torch::Tensor block_tables, torch::Tensor seq_lens, int64_t block_size, double softmax_scale, int64_t hadamard_order, bool bdr_k, bool rotate_v, bool tiled_layout);
torch::Tensor attention_paged_int4_flash(torch::Tensor q, torch::Tensor k_new, torch::Tensor v_new, torch::Tensor k_pages, torch::Tensor v_pages, torch::Tensor k_scales, torch::Tensor v_scales, torch::Tensor k_zeros, torch::Tensor v_zeros, torch::Tensor slot_mapping, torch::Tensor block_tables, torch::Tensor seq_lens, int64_t block_size, double softmax_scale, int64_t hadamard_order, bool bdr_k, bool rotate_v, bool tiled_layout);
torch::Tensor argmax(torch::Tensor logits);
torch::Tensor argmax_many(torch::Tensor logits);
void argmax_many_out(torch::Tensor logits, torch::Tensor out);
void argmax_unpack_state_out(torch::Tensor state, torch::Tensor out);
torch::Tensor count_prefix_matches(torch::Tensor proposed, torch::Tensor verified);
std::vector<torch::Tensor> resolve_greedy_speculative(
    torch::Tensor draft_token_ids,
    torch::Tensor target_token_ids,
    torch::Tensor bonus_token_ids,
    torch::Tensor cu_num_draft_tokens,
    torch::Tensor scheduled_token_counts);

// Utility: fast sigmoid/silu in fp32.
__device__ __forceinline__ float qb_sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float qb_silu(float x) {
  return x * qb_sigmoid(x);
}
