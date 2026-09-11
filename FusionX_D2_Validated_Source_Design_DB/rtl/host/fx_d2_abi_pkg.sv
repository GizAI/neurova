`timescale 1ns/1ps
package fx_d2_abi_pkg;
  localparam int FX_D2_ABI_VERSION=2;
  localparam int FX_D2_DESCRIPTOR_BYTES=64;
  localparam int FX_D2_COMPLETION_BYTES=32;
  localparam logic [15:0] FX_D2_REG_ID=16'h0000;
  localparam logic [15:0] FX_D2_REG_VERSION=16'h0004;
  localparam logic [15:0] FX_D2_REG_CONTROL=16'h0008;
  localparam logic [15:0] FX_D2_REG_STATUS=16'h000c;
  localparam logic [15:0] FX_D2_REG_IRQ_STATUS=16'h0010;
  localparam logic [15:0] FX_D2_REG_IRQ_MASK=16'h0014;
  localparam logic [15:0] FX_D2_REG_SQ_BASE_LO=16'h0020;
  localparam logic [15:0] FX_D2_REG_SQ_BASE_HI=16'h0024;
  localparam logic [15:0] FX_D2_REG_SQ_SIZE=16'h0028;
  localparam logic [15:0] FX_D2_REG_SQ_HEAD=16'h002c;
  localparam logic [15:0] FX_D2_REG_SQ_TAIL=16'h0030;
  localparam logic [15:0] FX_D2_REG_SQ_DOORBELL=16'h0034;
  localparam logic [15:0] FX_D2_REG_CQ_BASE_LO=16'h0040;
  localparam logic [15:0] FX_D2_REG_CQ_BASE_HI=16'h0044;
  localparam logic [15:0] FX_D2_REG_CQ_SIZE=16'h0048;
  localparam logic [15:0] FX_D2_REG_CQ_HEAD=16'h004c;
  localparam logic [15:0] FX_D2_REG_CQ_TAIL=16'h0050;
  localparam logic [15:0] FX_D2_REG_CQ_DOORBELL=16'h0054;
  localparam logic [15:0] FX_D2_REG_RAS_STATUS=16'h0060;
  localparam logic [15:0] FX_D2_REG_THERMAL_STATUS=16'h0064;
endpackage
