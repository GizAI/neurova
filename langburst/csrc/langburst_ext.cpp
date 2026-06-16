#include <torch/extension.h>
#include "kernels.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdlib>

namespace py = pybind11;

int marlin_cuda(
  const void* A,
  const void* B,
        void* C,
        void* s,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize = -1,
  int dev = 0,
  cudaStream_t stream = 0,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 16
);

static void run_lowbit_marlin_gemm(torch::Tensor qweight, torch::Tensor scales, torch::Tensor x, torch::Tensor out, torch::Tensor workspace, int64_t cols, int64_t group_size) {
  QB_CHECK_CUDA(qweight); QB_CHECK_CUDA(scales); QB_CHECK_CUDA(x);
  QB_CHECK_CUDA(out); QB_CHECK_CUDA(workspace);
  QB_CHECK_CONTIGUOUS(qweight); QB_CHECK_CONTIGUOUS(scales); QB_CHECK_CONTIGUOUS(x); QB_CHECK_CONTIGUOUS(out); QB_CHECK_CONTIGUOUS(workspace);
  TORCH_CHECK(qweight.scalar_type() == at::kInt, "marlin qweight must be int32");
  QB_CHECK_HALF(scales); QB_CHECK_HALF(x);
  QB_CHECK_HALF(out);
  TORCH_CHECK(workspace.scalar_type() == at::kInt, "marlin workspace must be int32");
  TORCH_CHECK(x.dim() == 2, "x must be [batch, cols]");
  TORCH_CHECK(x.size(1) == cols, "x cols mismatch");
  TORCH_CHECK(cols % 128 == 0, "Marlin requires K divisible by 128");
  const int prob_m = static_cast<int>(x.size(0));
  const int prob_k = static_cast<int>(cols);
  TORCH_CHECK(qweight.dim() == 2 && qweight.size(0) == prob_k / 16, "marlin qweight must be [K/16, N*2]");
  TORCH_CHECK(qweight.size(1) % 2 == 0, "marlin qweight second dim must be N*2");
  const int prob_n = static_cast<int>(qweight.size(1) / 2);
  TORCH_CHECK(prob_n % 256 == 0, "Marlin requires N divisible by 256");
  TORCH_CHECK(scales.dim() == 2 && scales.size(1) == prob_n, "marlin scales must be [K/group, N]");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == prob_m && out.size(1) == prob_n, "marlin out shape mismatch");
  int groupsize = static_cast<int>(group_size);
  if (groupsize <= 0 || groupsize == prob_k) groupsize = -1;
  TORCH_CHECK(groupsize == -1 || (groupsize == 128 && scales.size(0) == prob_k / groupsize), "Marlin supports group_size -1 or 128");
  int max_par = 16;
  if (const char* env = std::getenv("LANGBURST_MARLIN_MAX_PAR")) {
    max_par = std::atoi(env);
  }
  TORCH_CHECK(max_par >= 1 && max_par <= 16, "LANGBURST_MARLIN_MAX_PAR must be in [1, 16]");
  TORCH_CHECK(workspace.numel() >= prob_n / 128 * max_par, "marlin workspace too small");
  const int dev = x.get_device();
  int err = marlin_cuda(
    x.data_ptr(),
    qweight.data_ptr(),
    out.data_ptr(),
    scales.data_ptr(),
    prob_m,
    prob_n,
    prob_k,
    workspace.data_ptr(),
    groupsize,
    dev,
    at::cuda::getCurrentCUDAStream(dev),
    -1,
    -1,
    -1,
    max_par
  );
  TORCH_CHECK(err == 0, "Marlin kernel failed with code ", err);
}

torch::Tensor lowbit_marlin_gemm(torch::Tensor qweight, torch::Tensor scales, torch::Tensor x, int64_t cols, int64_t group_size) {
  TORCH_CHECK(x.dim() == 2, "x must be [batch, cols]");
  TORCH_CHECK(qweight.dim() == 2 && qweight.size(1) % 2 == 0, "marlin qweight must be [K/16, N*2]");
  const int64_t prob_n = qweight.size(1) / 2;
  auto y = torch::empty({x.size(0), prob_n}, x.options());
  auto workspace = torch::zeros({prob_n / 128 * 16}, torch::TensorOptions().dtype(torch::kInt32).device(x.device()));
  run_lowbit_marlin_gemm(qweight, scales, x, y, workspace, cols, group_size);
  return y;
}

void lowbit_marlin_gemm_out(torch::Tensor qweight, torch::Tensor scales, torch::Tensor x, torch::Tensor out, torch::Tensor workspace, int64_t cols, int64_t group_size) {
  run_lowbit_marlin_gemm(qweight, scales, x, out, workspace, cols, group_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "LangBurst CUDA kernels";
  m.def("lowbit_gemv", &lowbit_gemv,
        "Groupwise symmetric low-bit GEMV: y = dequant(qweight, scales, bits) @ x",
        py::arg("qweight"),
        py::arg("scales"),
        py::arg("x"),
        py::arg("cols"),
        py::arg("group_size"),
        py::arg("bits"),
        py::arg("rows_per_cta") = 8);
  m.def("lowbit_row_dequant", &lowbit_row_dequant,
        "Dequantize one low-bit row, used for embeddings");
  m.def("lowbit_gemv_pair", &lowbit_gemv_pair,
        "Two groupwise symmetric low-bit GEMVs sharing the same input vector",
        py::arg("qweight_a"),
        py::arg("scales_a"),
        py::arg("qweight_b"),
        py::arg("scales_b"),
        py::arg("x"),
        py::arg("cols"),
        py::arg("group_size"),
        py::arg("bits"),
        py::arg("rows_per_cta") = 8);
  m.def("lowbit_gemm", &lowbit_gemm,
        "Groupwise symmetric low-bit GEMM: y = x @ dequant(qweight, scales, bits).T",
        py::arg("qweight"),
        py::arg("scales"),
        py::arg("x"),
        py::arg("cols"),
        py::arg("group_size"),
        py::arg("bits"),
        py::arg("rows_per_cta") = 8);
  m.def("lowbit_marlin_gemm", &lowbit_marlin_gemm,
        "Marlin W4A16 GEMM: y = x @ dequant(qweight, scales).T",
        py::arg("qweight"),
        py::arg("scales"),
        py::arg("x"),
        py::arg("cols"),
        py::arg("group_size"));
  m.def("lowbit_marlin_gemm_out", &lowbit_marlin_gemm_out,
        "Marlin W4A16 GEMM into preallocated output/workspace",
        py::arg("qweight"),
        py::arg("scales"),
        py::arg("x"),
        py::arg("out"),
        py::arg("workspace"),
        py::arg("cols"),
        py::arg("group_size"));
  m.def("rmsnorm", &rmsnorm, "RMSNorm fp16");
  m.def("rmsnorm_qwen", &rmsnorm_qwen, "Qwen RMSNorm fp16 using 1+weight");
  m.def("rmsnorm_silu_gate", &rmsnorm_silu_gate, "RMSNorm followed by SiLU gate fp16");
  m.def("rmsnorm_qwen_silu_gate", &rmsnorm_qwen_silu_gate, "Qwen RMSNorm followed by SiLU gate fp16");
  m.def("silu_mul", &silu_mul, "Fused SiLU(gate) * up fp16");
  m.def("gdn_recurrent", &gdn_recurrent,
        "Single-token Qwen-style recurrent gated delta rule, state updated in-place");
  m.def("gdn_recurrent_ab", &gdn_recurrent_ab,
        "Single-token Qwen-style recurrent gated delta rule with fused a/b gate computation");
  m.def("gdn_recurrent_scan", &gdn_recurrent_scan,
        "Block Qwen-style recurrent gated delta scan, state updated in-place");
  m.def("gdn_recurrent_ab_scan", &gdn_recurrent_ab_scan,
        "Block Qwen-style recurrent gated delta scan with fused a/b gate computation");
  m.def("gdn_recurrent_ab_batch", &gdn_recurrent_ab_batch,
        "Batched Qwen-style recurrent gated delta update over slot-indexed state arena",
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("a"),
        py::arg("b"),
        py::arg("A_log"),
        py::arg("dt_bias"),
        py::arg("state_arena"),
        py::arg("state_indices"));
  m.def("depthwise_conv_update", &depthwise_conv_update,
        "Single-token causal depthwise conv update with SiLU");
  m.def("depthwise_conv_update_scan", &depthwise_conv_update_scan,
        "Block causal depthwise conv scan with SiLU, state updated in-place");
  m.def("depthwise_conv_update_batch", &depthwise_conv_update_batch,
        "Batched causal depthwise conv update over slot-indexed state arena",
        py::arg("state_arena"),
        py::arg("state_indices"),
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias"));
  m.def("attention_decode_fp16", &attention_decode_fp16,
        "Small/medium context fp16 decode attention baseline");
  m.def("attention_decode_batch_fp16", &attention_decode_batch_fp16,
        "Batched fp16 decode attention over slot-indexed KV arena",
        py::arg("q"),
        py::arg("k_new"),
        py::arg("v_new"),
        py::arg("k_arena"),
        py::arg("v_arena"),
        py::arg("state_indices"),
        py::arg("write_indices"),
        py::arg("live_lengths"),
        py::arg("positions"),
        py::arg("use_ring"),
        py::arg("softmax_scale"));
  m.def("attention_decode_paged_fp16", &attention_decode_paged_fp16,
        "Batched fp16 decode attention over paged KV block table",
        py::arg("q"),
        py::arg("k_new"),
        py::arg("v_new"),
        py::arg("k_pages"),
        py::arg("v_pages"),
        py::arg("slot_mapping"),
        py::arg("block_tables"),
        py::arg("seq_lens"),
        py::arg("block_size"),
        py::arg("softmax_scale"));
  m.def("argmax", &argmax, "GPU argmax over a 1D fp16/fp32 logits tensor");
  m.def("argmax_many", &argmax_many, "GPU argmax over [rows, vocab] fp16/fp32 logits");
  m.def("argmax_many_out", &argmax_many_out, "Graph-capturable GPU argmax_many into preallocated int64 output");
  m.def("count_prefix_matches", &count_prefix_matches, "Count equal prefix length between two int64 vectors");
}
