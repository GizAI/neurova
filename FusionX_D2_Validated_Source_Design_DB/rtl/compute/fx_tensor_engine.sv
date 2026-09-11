`timescale 1ns/1ps
// One logical 64x64 tensor tile engine with packed low-bit operands and full M/N/K traversal.
// Multiple engines can be replicated behind an HBM arbiter for throughput scaling.
module fx_tensor_engine #(
  parameter int ROWS=64,
  parameter int COLS=64,
  parameter int ENGINE_ID=8'h10,
  localparam int RW=(ROWS<=1)?1:$clog2(ROWS),
  localparam int BW=((COLS+31)/32<=1)?1:$clog2((COLS+31)/32)
)(
 input logic clk,input logic rst_n,input logic enable,
 input logic desc_valid,output logic desc_ready,input fx_d2_pkg::fx_desc_t desc,
 output logic mem_req_valid,input logic mem_req_ready,output fx_d2_pkg::fx_mem_req_t mem_req,
 input logic mem_rsp_valid,output logic mem_rsp_ready,input fx_d2_pkg::fx_mem_rsp_t mem_rsp,
 output logic cpl_valid,input logic cpl_ready,output fx_d2_pkg::fx_cpl_t cpl,
 output logic busy,output logic fault
);
 import fx_d2_pkg::*;
 typedef enum logic[4:0]{IDLE,CLEAR,CLEAR_WAIT,A_START,A_REQ,A_WAIT,B_START,B_REQ,B_WAIT,
   STEP,STEP_WAIT,RESULT_START,WRITE_REQ,WRITE_WAIT,NEXT_TILE,CPL,ERR}state_e;
 state_e state;
 fx_desc_t q;
 logic[31:0]mt,nt,kk,read_beat,cycles,bytes_done;
 logic a_start,b_start,a_valid,b_valid,a_ready,b_ready,a_done,b_done,a_busy,b_busy,a_error,b_error;
 logic signed[ROWS*16-1:0]a_raw,a_masked;
 logic signed[COLS*16-1:0]b_raw,b_masked;
 logic clear_start,clear_done,step_valid,step_ready,step_done,result_start,result_valid,result_ready,result_last,mxu_busy;
 logic[RW-1:0]result_row;logic[BW-1:0]result_beat;logic[511:0]result_data;logic[63:0]result_strb,edge_strb;
 logic last_issued;
 integer i;integer a_beats,b_beats;integer valid_bytes;integer unsigned elem_bits;

 function automatic integer beats_for(input logic[7:0]dt,input integer elems);
  integer bits;begin case(dt)FX_DT_INT4,FX_DT_FP4_E2M1:bits=4;FX_DT_INT8,FX_DT_FP8_E4M3FN:bits=8;default:bits=16;endcase beats_for=(elems*bits+511)/512;end
 endfunction

 fx_operand_unpacker #(.ELEMS(ROWS))u_a(.clk,.rst_n,.start(a_start),.dtype(q.dtype),.scale_q2_14(q.scale_q2_14),.beat_valid(a_valid),.beat_ready(a_ready),.beat_data(mem_rsp.data),.vector_q8_8(a_raw),.done(a_done),.busy(a_busy),.error(a_error));
 fx_operand_unpacker #(.ELEMS(COLS))u_b(.clk,.rst_n,.start(b_start),.dtype(q.dtype),.scale_q2_14(q.scale_q2_14),.beat_valid(b_valid),.beat_ready(b_ready),.beat_data(mem_rsp.data),.vector_q8_8(b_raw),.done(b_done),.busy(b_busy),.error(b_error));
 fx_mxu_array #(.ROWS(ROWS),.COLS(COLS))u_mxu(.clk,.rst_n,.clear_start,.clear_done,.step_valid,.step_ready,.a_vector(a_masked),.b_vector(b_masked),.step_done,.result_start,.result_valid,.result_ready,.result_row,.result_beat,.result_data,.result_strb,.result_last,.out_scale_q2_14(q.out_scale_q2_14),.out_right_shift(q.out_shift[5:0]),.busy(mxu_busy));

 always_comb begin
  a_beats=beats_for(q.dtype,ROWS);b_beats=beats_for(q.dtype,COLS);
  for(i=0;i<ROWS;i=i+1)a_masked[i*16+:16]=(((mt*ROWS)+i)<q.m)?a_raw[i*16+:16]:'0;
  for(i=0;i<COLS;i=i+1)b_masked[i*16+:16]=(((nt*COLS)+i)<q.n)?b_raw[i*16+:16]:'0;
  a_start=(state==A_START);b_start=(state==B_START);
  a_valid=(state==A_WAIT)&&mem_rsp_valid&&!mem_rsp.poison;
  b_valid=(state==B_WAIT)&&mem_rsp_valid&&!mem_rsp.poison;
  clear_start=(state==CLEAR);step_valid=(state==STEP);result_start=(state==RESULT_START);
  desc_ready=(state==IDLE)&&enable;busy=(state!=IDLE);
  mem_rsp_ready=((state==A_WAIT)&&a_ready)||((state==B_WAIT)&&b_ready)||(state==WRITE_WAIT);
  edge_strb=result_strb;
  for(i=0;i<32;i=i+1)if(((nt*COLS)+(result_beat*32)+i)>=q.n)edge_strb[i*2+:2]=2'b00;
  mem_req_valid=1'b0;mem_req='0;
  if(state==A_REQ)begin mem_req_valid=1;mem_req.id=16'h1000;mem_req.addr=q.src0+((((mt*q.k)+kk)*a_beats+read_beat)*64);end
  else if(state==B_REQ)begin mem_req_valid=1;mem_req.id=16'h1001;mem_req.addr=q.src1+((((nt*q.k)+kk)*b_beats+read_beat)*64);end
  else if(state==WRITE_REQ&&result_valid)begin
   mem_req_valid=(edge_strb!='0);mem_req.write=1;mem_req.id=16'h1080;
   mem_req.addr=q.dst+(((((mt*ROWS)+result_row)*q.n)+(nt*COLS)+(result_beat*32))*2);
   mem_req.data=result_data;mem_req.strb=edge_strb;mem_req.last=result_last;
  end
  result_ready=(state==WRITE_REQ)&&result_valid&&((edge_strb=='0)||mem_req_ready);
  cpl_valid=(state==CPL)||(state==ERR);cpl='0;cpl.tag=q.tag;cpl.queue_id=q.queue_id;cpl.engine=ENGINE_ID;cpl.status=(state==ERR)?8'hff:8'h00;cpl.cycles=cycles;cpl.bytes_moved=bytes_done;fault=(state==ERR);
 end

 always_ff@(posedge clk or negedge rst_n)begin
  if(!rst_n)begin state<=IDLE;q<='0;mt<=0;nt<=0;kk<=0;read_beat<=0;cycles<=0;bytes_done<=0;last_issued<=0;end
  else begin
   if(state!=IDLE)cycles<=cycles+1;
   case(state)
    IDLE:if(desc_valid&&desc_ready)begin q<=desc;mt<=0;nt<=0;kk<=0;read_beat<=0;cycles<=0;bytes_done<=0;state<=(desc.m!=0&&desc.n!=0&&desc.k!=0&&fx_tensor_dtype(desc.dtype))?CLEAR:ERR;end
    CLEAR:state<=CLEAR_WAIT;
    CLEAR_WAIT:if(clear_done)state<=A_START;
    A_START:begin read_beat<=0;state<=A_REQ;end
    A_REQ:if(mem_req_valid&&mem_req_ready)state<=A_WAIT;
    A_WAIT:if(mem_rsp_valid&&mem_rsp_ready)begin if(mem_rsp.poison||a_error)state<=ERR;else begin bytes_done<=bytes_done+64;if(read_beat+1>=a_beats)state<=B_START;else begin read_beat<=read_beat+1;state<=A_REQ;end end end
    B_START:begin read_beat<=0;state<=B_REQ;end
    B_REQ:if(mem_req_valid&&mem_req_ready)state<=B_WAIT;
    B_WAIT:if(mem_rsp_valid&&mem_rsp_ready)begin if(mem_rsp.poison||b_error)state<=ERR;else begin bytes_done<=bytes_done+64;if(read_beat+1>=b_beats)state<=STEP;else begin read_beat<=read_beat+1;state<=B_REQ;end end end
    STEP:if(step_ready)state<=STEP_WAIT;
    STEP_WAIT:if(step_done)begin if(kk+1<q.k)begin kk<=kk+1;state<=A_START;end else state<=RESULT_START;end
    RESULT_START:state<=WRITE_REQ;
    WRITE_REQ:if(result_valid&&result_ready)begin last_issued<=result_last;if(edge_strb=='0)begin if(result_last)state<=NEXT_TILE;end else state<=WRITE_WAIT;end
    WRITE_WAIT:if(mem_rsp_valid&&mem_rsp_ready)begin if(mem_rsp.poison)state<=ERR;else begin bytes_done<=bytes_done+64;if(last_issued)state<=NEXT_TILE;else state<=WRITE_REQ;end end
    NEXT_TILE:begin kk<=0;if((nt+1)*COLS<q.n)begin nt<=nt+1;state<=CLEAR;end else if((mt+1)*ROWS<q.m)begin nt<=0;mt<=mt+1;state<=CLEAR;end else state<=CPL;end
    CPL:if(cpl_ready)state<=IDLE;
    ERR:if(cpl_ready)state<=IDLE;
    default:state<=ERR;
   endcase
  end
 end
endmodule
