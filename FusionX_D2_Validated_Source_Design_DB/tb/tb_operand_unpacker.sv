`timescale 1ns/1ps
module tb_operand_unpacker;
 import fx_d2_pkg::*;
 logic clk=0,rst_n=0,start,beat_valid,beat_ready,done,busy,error;logic[7:0]dtype;logic[15:0]scale;logic[511:0]beat;logic signed[128*16-1:0]vec;integer i;
 always #5 clk=~clk;
 fx_operand_unpacker #(.ELEMS(128))dut(.clk,.rst_n,.start,.dtype,.scale_q2_14(scale),.beat_valid,.beat_ready,.beat_data(beat),.vector_q8_8(vec),.done,.busy,.error);
 initial begin start=0;beat_valid=0;dtype=FX_DT_FP4_E2M1;scale=16'h4000;beat='0;repeat(3)@(posedge clk);rst_n=1;
  for(i=0;i<128;i=i+1)beat[i*4 +:4]=i[3:0];
  @(posedge clk);start<=1;@(posedge clk);start<=0;wait(beat_ready);beat_valid<=1;@(posedge clk);beat_valid<=0;wait(done);#1;
  if($signed(vec[7*16 +:16])!=1536||$signed(vec[15*16 +:16])!=-1536||error)$fatal(1,"unpack mismatch");
  $display("PASS tb_operand_unpacker");$finish;
 end
endmodule
