`timescale 1ns/1ps
module tb_mxu_64x64;
 localparam R=64;localparam C=64;
 logic clk=0,rst_n=0,clear_start,clear_done,step_valid,step_ready,step_done,result_start,result_valid,result_ready,result_last,busy;
 logic signed[R*16-1:0]a;logic signed[C*16-1:0]b;logic[5:0]row;logic beat;logic[511:0]data;logic[63:0]strb;integer i,steps,count,expected_row,expected_beat;
 always #1 clk=~clk;
 fx_mxu_array #(.ROWS(R),.COLS(C))dut(.clk,.rst_n,.clear_start,.clear_done,.step_valid,.step_ready,.a_vector(a),.b_vector(b),.step_done,.result_start,.result_valid,.result_ready,.result_row(row),.result_beat(beat),.result_data(data),.result_strb(strb),.result_last,.out_scale_q2_14(16'h4000),.out_right_shift(0),.busy);
 initial begin clear_start=0;step_valid=0;result_start=0;result_ready=0;a='0;b='0;for(i=0;i<R;i=i+1)a[i*16 +:16]=16'sd1;for(i=0;i<C;i=i+1)b[i*16 +:16]=16'sd1;
  repeat(4)@(posedge clk);rst_n=1;@(posedge clk);clear_start<=1;@(posedge clk);clear_start<=0;wait(clear_done);
  for(steps=0;steps<3;steps=steps+1)begin wait(step_ready);@(posedge clk);step_valid<=1;@(posedge clk);step_valid<=0;wait(step_done);end
  @(posedge clk);result_start<=1;@(posedge clk);result_start<=0;result_ready<=1;count=0;
  while(count<128)begin @(posedge clk);if(result_valid&&result_ready)begin expected_row=count/2;expected_beat=count%2;if(row!==expected_row||beat!==expected_beat)$fatal(1,"sequence row=%0d beat=%0d expected=%0d/%0d",row,beat,expected_row,expected_beat);for(i=0;i<32;i=i+1)if($signed(data[i*16 +:16])!=3)$fatal(1,"value mismatch count=%0d lane=%0d value=%0d",count,i,$signed(data[i*16 +:16]));if(count==127&&!result_last)$fatal(1,"last missing");count=count+1;end end
  $display("PASS tb_mxu_64x64 beats=%0d",count);$finish;
 end
endmodule
