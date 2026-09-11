`timescale 1ns/1ps
package fx_d2_pkg;
  typedef enum logic [7:0] {
    FX_OP_NOP=8'h00, FX_OP_DMA_H2D=8'h01, FX_OP_DMA_D2H=8'h02,
    FX_OP_GEMM=8'h10, FX_OP_GEMV=8'h11,
    FX_OP_VECTOR=8'h20, FX_OP_RMSNORM=8'h21, FX_OP_SOFTMAX=8'h22,
    FX_OP_ROPE=8'h23, FX_OP_STATE_UPDATE=8'h24, FX_OP_MOE_TOPK=8'h25,
    FX_OP_MTP_VERIFY=8'h26, FX_OP_SPARSE_GATHER=8'h27,
    FX_OP_PATCHIFY3D=8'h30, FX_OP_ADALN=8'h31, FX_OP_FENCE=8'h7e
  } fx_opcode_e;

  typedef enum logic [7:0] {
    FX_DT_INT4=8'h00, FX_DT_FP4_E2M1=8'h01, FX_DT_INT8=8'h02,
    FX_DT_FP8_E4M3FN=8'h03, FX_DT_INT16=8'h04, FX_DT_BF16=8'h05
  } fx_dtype_e;

  typedef struct packed {
    logic [7:0] opcode, dtype, flags, reserved0;
    logic [15:0] tag, queue_id;
    logic [31:0] m, n, k, bytes;
    logic [63:0] src0, src1, src2, dst;
    logic [15:0] scale_q2_14, out_scale_q2_14;
    logic [7:0] out_shift, reserved1;
    logic [15:0] aux;
  } fx_desc_t;

  typedef struct packed {
    logic [15:0] tag, queue_id;
    logic [7:0] status, engine;
    logic [15:0] reserved0;
    logic [31:0] cycles, bytes_moved;
    logic [63:0] user_cookie, reserved1;
  } fx_cpl_t;

  typedef struct packed {
    logic write;
    logic [15:0] id;
    logic [63:0] addr;
    logic [511:0] data;
    logic [63:0] strb;
    logic last;
  } fx_mem_req_t;

  typedef struct packed {
    logic [15:0] id;
    logic [511:0] data;
    logic last, poison;
  } fx_mem_rsp_t;

  localparam int FX_DESC_W=$bits(fx_desc_t);
  localparam int FX_CPL_W=$bits(fx_cpl_t);

  function automatic fx_desc_t fx_desc_from_le512(input logic [511:0] d);
    fx_desc_t x;
    begin
      x='0;
      x.opcode=d[0 +:8];x.dtype=d[8 +:8];x.flags=d[16 +:8];x.reserved0=d[24 +:8];
      x.tag=d[32 +:16];x.queue_id=d[48 +:16];
      x.m=d[64 +:32];x.n=d[96 +:32];x.k=d[128 +:32];x.bytes=d[160 +:32];
      x.src0=d[192 +:64];x.src1=d[256 +:64];x.src2=d[320 +:64];x.dst=d[384 +:64];
      x.scale_q2_14=d[448 +:16];x.out_scale_q2_14=d[464 +:16];x.out_shift=d[480 +:8];x.reserved1=d[488 +:8];x.aux=d[496 +:16];
      fx_desc_from_le512=x;
    end
  endfunction

  function automatic logic [255:0] fx_cpl_to_le256(input fx_cpl_t x);
    logic [255:0] d;
    begin
      d='0;d[0 +:16]=x.tag;d[16 +:16]=x.queue_id;d[32 +:8]=x.status;d[40 +:8]=x.engine;
      d[48 +:16]=x.reserved0;d[64 +:32]=x.cycles;d[96 +:32]=x.bytes_moved;
      d[128 +:64]=x.user_cookie;d[192 +:64]=x.reserved1;fx_cpl_to_le256=d;
    end
  endfunction

  function automatic int unsigned fx_element_bits(input logic [7:0] dt);
    case(dt)
      FX_DT_INT4,FX_DT_FP4_E2M1: fx_element_bits=4;
      FX_DT_INT8,FX_DT_FP8_E4M3FN: fx_element_bits=8;
      default: fx_element_bits=16;
    endcase
  endfunction

  function automatic logic fx_tensor_dtype(input logic [7:0] dt);
    fx_tensor_dtype=(dt==FX_DT_INT4)||(dt==FX_DT_FP4_E2M1)||
                    (dt==FX_DT_INT8)||(dt==FX_DT_FP8_E4M3FN)||
                    (dt==FX_DT_INT16)||(dt==FX_DT_BF16);
  endfunction
endpackage
