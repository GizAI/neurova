`timescale 1ns/1ps
module tb_lowbit;
 import fx_d2_pkg::*;
 logic[7:0]dtype;logic[15:0]raw,scale;logic signed[15:0]value;logic finite,supported;
 fx_lowbit_decode dut(.dtype,.raw,.scale_q2_14(scale),.value_q8_8(value),.finite,.supported);
 task automatic check(input logic[7:0]dt,input logic[15:0]rv,input integer expected,input logic exp_finite);
  begin dtype=dt;raw=rv;scale=16'h4000;#1;if(value!==expected||finite!==exp_finite||!supported)begin $display("FAIL lowbit dt=%0d raw=%h got=%0d finite=%b",dt,rv,$signed(value),finite);$fatal(1);end end
 endtask
 initial begin
  check(FX_DT_FP4_E2M1,16'h0007,1536,1);check(FX_DT_FP4_E2M1,16'h000f,-1536,1);
  check(FX_DT_FP8_E4M3FN,16'h0038,256,1);check(FX_DT_FP8_E4M3FN,16'h007f,0,0);
  check(FX_DT_INT4,16'h000f,-256,1);check(FX_DT_INT8,16'h0080,-32768,1);
  $display("PASS tb_lowbit");$finish;
 end
endmodule
