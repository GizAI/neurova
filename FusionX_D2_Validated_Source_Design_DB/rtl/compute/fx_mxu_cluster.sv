`timescale 1ns/1ps
// Shared-A / independently partitioned-B cluster.
module fx_mxu_cluster #(
  parameter int ARRAYS=4,
  parameter int ROWS=64,
  parameter int COLS=64,
  localparam int AI=(ARRAYS<=1)?1:$clog2(ARRAYS),
  localparam int RW=(ROWS<=1)?1:$clog2(ROWS),
  localparam int BW=((COLS+31)/32<=1)?1:$clog2((COLS+31)/32)
)(
  input logic clk,input logic rst_n,
  input logic clear_start,output logic clear_done,
  input logic step_valid,output logic step_ready,
  input logic signed [ROWS*16-1:0] a_vector,
  input logic signed [ARRAYS*COLS*16-1:0] b_vectors,
  output logic step_done,
  input logic result_start,
  output logic result_valid,input logic result_ready,
  output logic [AI-1:0] result_array,
  output logic [RW-1:0] result_row,
  output logic [BW-1:0] result_beat,
  output logic [511:0] result_data,
  output logic [63:0] result_strb,
  output logic result_last,
  input logic [15:0] out_scale_q2_14,input logic [5:0] out_right_shift,
  output logic busy
);
  logic [ARRAYS-1:0] cd,sd,rv,rr,rl,bi;
  logic [ARRAYS*RW-1:0] row_bus;
  logic [ARRAYS*BW-1:0] beat_bus;
  logic [ARRAYS*512-1:0] data_bus;
  logic [ARRAYS*64-1:0] strb_bus;
  logic [AI-1:0] selected;
  integer i;
  genvar g;
  generate for(g=0;g<ARRAYS;g=g+1) begin: G_ARRAY
    fx_mxu_array #(.ROWS(ROWS),.COLS(COLS)) u_array(
      .clk,.rst_n,.clear_start,.clear_done(cd[g]),.step_valid,.step_ready(),
      .a_vector,.b_vector(b_vectors[g*COLS*16 +:COLS*16]),.step_done(sd[g]),
      .result_start,.result_valid(rv[g]),.result_ready(rr[g]),
      .result_row(row_bus[g*RW +:RW]),.result_beat(beat_bus[g*BW +:BW]),
      .result_data(data_bus[g*512 +:512]),.result_strb(strb_bus[g*64 +:64]),
      .result_last(rl[g]),.out_scale_q2_14,.out_right_shift,.busy(bi[g]));
  end endgenerate
  assign clear_done=&cd;
  assign step_done=&sd;
  assign step_ready=~(|bi);
  assign busy=|bi;
  always_comb begin
    selected='0;
    result_valid=1'b0;result_array='0;result_row='0;result_beat='0;
    result_data='0;result_strb='0;result_last=1'b0;rr='0;
    for(i=0;i<ARRAYS;i=i+1) begin
      if(!result_valid&&rv[i]) begin
        selected=i[AI-1:0];result_valid=1'b1;result_array=i[AI-1:0];
        result_row=row_bus[i*RW +:RW];result_beat=beat_bus[i*BW +:BW];
        result_data=data_bus[i*512 +:512];result_strb=strb_bus[i*64 +:64];
        result_last=rl[i]&&(i==ARRAYS-1);rr[i]=result_ready;
      end
    end
  end
endmodule
