#include <torch/extension.h>
#include "kernels.cuh"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "QwenBurst CUDA kernels";
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
  m.def("rmsnorm", &rmsnorm, "RMSNorm fp16");
  m.def("rmsnorm_qwen", &rmsnorm_qwen, "Qwen RMSNorm fp16 using 1+weight");
  m.def("rmsnorm_silu_gate", &rmsnorm_silu_gate, "RMSNorm followed by SiLU gate fp16");
  m.def("rmsnorm_qwen_silu_gate", &rmsnorm_qwen_silu_gate, "Qwen RMSNorm followed by SiLU gate fp16");
  m.def("gdn_recurrent", &gdn_recurrent,
        "Single-token Qwen-style recurrent gated delta rule, state updated in-place");
  m.def("attention_decode_fp16", &attention_decode_fp16,
        "Small/medium context fp16 decode attention baseline");
  m.def("argmax", &argmax, "GPU argmax over a 1D fp16/fp32 logits tensor");
  m.def("argmax_many", &argmax_many, "GPU argmax over [rows, vocab] fp16/fp32 logits");
  m.def("argmax_many_out", &argmax_many_out, "Graph-capturable GPU argmax_many into preallocated int64 output");
  m.def("count_prefix_matches", &count_prefix_matches, "Count equal prefix length between two int64 vectors");
}
