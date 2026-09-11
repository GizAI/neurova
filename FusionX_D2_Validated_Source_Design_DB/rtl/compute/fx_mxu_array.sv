`timescale 1ns/1ps
// Pipelined outer-product array. Output is requantized Q8.8, 32 values/512-bit beat.
// Row advances only after the final beat of that row is accepted.
module fx_mxu_array #(
  parameter int ROWS=64,
  parameter int COLS=64,
  parameter int IN_W=16,
  parameter int ACC_W=48,
  localparam int RW=(ROWS<=1)?1:$clog2(ROWS),
  localparam int BW=((COLS+31)/32<=1)?1:$clog2((COLS+31)/32),
  localparam int BEATS_PER_ROW=(COLS+31)/32
)(
  input  logic clk,
  input  logic rst_n,
  input  logic clear_start,
  output logic clear_done,
  input  logic step_valid,
  output logic step_ready,
  input  logic signed [ROWS*IN_W-1:0] a_vector,
  input  logic signed [COLS*IN_W-1:0] b_vector,
  output logic step_done,
  input  logic result_start,
  output logic result_valid,
  input  logic result_ready,
  output logic [RW-1:0] result_row,
  output logic [BW-1:0] result_beat,
  output logic [511:0] result_data,
  output logic [63:0] result_strb,
  output logic result_last,
  input  logic [15:0] out_scale_q2_14,
  input  logic [5:0] out_right_shift,
  output logic busy
);
  logic signed [ACC_W-1:0] acc [0:ROWS-1][0:COLS-1];
  logic signed [IN_W-1:0] a_q [0:ROWS-1];
  logic signed [IN_W-1:0] b_q [0:COLS-1];
  logic mul_pending;
  logic clear_busy, read_busy;
  logic [RW-1:0] clear_row, read_row;
  logic [BW-1:0] read_beat;
  integer r,c,i,col;

  function automatic logic signed [15:0] quantize(
    input logic signed [ACC_W-1:0] av,
    input logic [15:0] scale,
    input logic [5:0] rshift
  );
    longint signed prod, signed_q;
    longint unsigned mag, q, rem, half, mask;
    integer total_shift;
    logic neg;
    begin
      prod=$signed(av)*$signed({1'b0,scale});
      total_shift=14+rshift;
      neg=(prod<0);
      mag=neg ? $unsigned(-prod) : $unsigned(prod);
      if(total_shift==0) q=mag;
      else begin
        q=mag>>total_shift;
        if(total_shift>=63) begin rem=mag; half=64'h8000_0000_0000_0000; end
        else begin
          mask=(64'h1<<total_shift)-1;
          rem=mag&mask;
          half=64'h1<<(total_shift-1);
        end
        if((rem>half)||((rem==half)&&(q[0]))) q=q+1;
      end
      signed_q=neg ? -$signed(q) : $signed(q);
      if(signed_q>32767) quantize=16'sh7fff;
      else if(signed_q< -32768) quantize=16'sh8000;
      else quantize=signed_q[15:0];
    end
  endfunction

  assign step_ready=!clear_busy&&!read_busy&&!mul_pending;
  assign result_valid=read_busy;
  assign result_row=read_row;
  assign result_beat=read_beat;
  assign result_last=read_busy&&(read_row==ROWS-1)&&(read_beat==BEATS_PER_ROW-1);
  assign busy=clear_busy||read_busy||mul_pending;

  always_comb begin
    result_data='0;
    result_strb='0;
    for(i=0;i<32;i=i+1) begin
      col=read_beat*32+i;
      if(read_busy&&(col<COLS)) begin
        result_data[i*16 +:16]=quantize(acc[read_row][col],out_scale_q2_14,out_right_shift);
        result_strb[i*2 +:2]=2'b11;
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      clear_done<=1'b0;
      step_done<=1'b0;
      clear_busy<=1'b0;
      read_busy<=1'b0;
      mul_pending<=1'b0;
      clear_row<='0;
      read_row<='0;
      read_beat<='0;
    end else begin
      clear_done<=1'b0;
      step_done<=1'b0;

      if(mul_pending) begin
        for(r=0;r<ROWS;r=r+1)
          for(c=0;c<COLS;c=c+1)
            acc[r][c]<=acc[r][c]+$signed(a_q[r])*$signed(b_q[c]);
        mul_pending<=1'b0;
        step_done<=1'b1;
      end

      if(step_valid&&step_ready) begin
        for(r=0;r<ROWS;r=r+1) a_q[r]<=a_vector[r*IN_W +:IN_W];
        for(c=0;c<COLS;c=c+1) b_q[c]<=b_vector[c*IN_W +:IN_W];
        mul_pending<=1'b1;
      end

      if(clear_start&&!clear_busy&&!read_busy&&!mul_pending) begin
        clear_busy<=1'b1;
        clear_row<='0;
      end else if(clear_busy) begin
        for(c=0;c<COLS;c=c+1) acc[clear_row][c]<='0;
        if(clear_row==ROWS-1) begin
          clear_busy<=1'b0;
          clear_done<=1'b1;
        end else clear_row<=clear_row+1'b1;
      end

      if(result_start&&!read_busy&&!clear_busy&&!mul_pending) begin
        read_busy<=1'b1;
        read_row<='0;
        read_beat<='0;
      end else if(result_valid&&result_ready) begin
        if(read_beat==BEATS_PER_ROW-1) begin
          read_beat<='0;
          if(read_row==ROWS-1) read_busy<=1'b0;
          else read_row<=read_row+1'b1;
        end else read_beat<=read_beat+1'b1;
      end
    end
  end
endmodule
