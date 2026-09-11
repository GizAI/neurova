`timescale 1ns/1ps
module fx_mxu_protocol_sva #(parameter ROWS=64,COLS=64,localparam RW=(ROWS<=1)?1:$clog2(ROWS),localparam BW=((COLS+31)/32<=1)?1:$clog2((COLS+31)/32),localparam BPR=(COLS+31)/32)(
 input logic clk,rst_n,result_valid,result_ready,result_last,input logic[RW-1:0]result_row,input logic[BW-1:0]result_beat,input logic step_valid,step_ready,step_done
);
 default clocking cb @(posedge clk);endclocking default disable iff(!rst_n);
 p_hold_when_stalled:assert property(result_valid&&!result_ready |=> result_valid&&$stable(result_row)&&$stable(result_beat));
 p_row_holds_between_beats:assert property(result_valid&&result_ready&&(result_beat<BPR-1) |=> result_row==$past(result_row)&&result_beat==$past(result_beat)+1'b1);
 p_row_advances_after_last_beat:assert property(result_valid&&result_ready&&(result_beat==BPR-1)&&!result_last |=> result_row==$past(result_row)+1'b1&&result_beat==0);
 p_last_location:assert property(result_last |-> result_valid&&(result_row==ROWS-1)&&(result_beat==BPR-1));
 p_step_completion:assert property(step_valid&&step_ready |-> ##[1:3] step_done);
endmodule
