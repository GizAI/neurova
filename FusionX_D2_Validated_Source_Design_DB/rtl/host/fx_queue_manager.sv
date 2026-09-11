`timescale 1ns/1ps
module fx_queue_manager(
 input logic clk,input logic rst_n,input logic enable,
 input logic[63:0]sq_base,input logic[31:0]sq_size,input logic[31:0]sq_tail,output logic[31:0]sq_head,
 input logic[63:0]cq_base,input logic[31:0]cq_size,input logic[31:0]cq_head,output logic[31:0]cq_tail,
 output logic host_req_valid,input logic host_req_ready,output fx_d2_pkg::fx_mem_req_t host_req,
 input logic host_rsp_valid,output logic host_rsp_ready,input fx_d2_pkg::fx_mem_rsp_t host_rsp,
 output logic cmd_valid,input logic cmd_ready,output logic cmd_tensor,output fx_d2_pkg::fx_desc_t cmd_desc,
 input logic cpl_valid,output logic cpl_ready,input fx_d2_pkg::fx_cpl_t cpl,
 output logic irq,output logic fault
);
 import fx_d2_pkg::*;
 typedef enum logic[3:0]{IDLE,FETCH_Q,FETCH_W,DISPATCH,WAIT_CPL,WRITE_Q,WRITE_W,ERR}state_e;
 state_e state;fx_desc_t dq;fx_cpl_t cq;logic[255:0]cpl_bits;logic cq_upper;logic[31:0]sq_mask,cq_mask,next_cq;
 assign sq_mask=sq_size-1;assign cq_mask=cq_size-1;assign next_cq=(cq_tail+1)&cq_mask;
 assign cmd_desc=dq;assign cmd_tensor=(dq.opcode==FX_OP_GEMM)||(dq.opcode==FX_OP_GEMV);assign cmd_valid=(state==DISPATCH);
 assign cpl_ready=(state==WAIT_CPL);assign host_rsp_ready=(state==FETCH_W)||(state==WRITE_W);
 always_comb begin
  cpl_bits=fx_cpl_to_le256(cq);cq_upper=(cq_tail&1)!=0;host_req_valid=0;host_req='0;
  if(state==FETCH_Q)begin host_req_valid=1;host_req.id=16'h0100;host_req.addr=sq_base+((sq_head&sq_mask)*64);end
  else if(state==WRITE_Q)begin host_req_valid=1;host_req.write=1;host_req.id=16'h0180;host_req.addr=cq_base+(((cq_tail&cq_mask)>>1)*64);if(cq_upper)begin host_req.data[511:256]=cpl_bits;host_req.strb[63:32]='1;end else begin host_req.data[255:0]=cpl_bits;host_req.strb[31:0]='1;end host_req.last=1;end
  irq=(state==WRITE_W)&&host_rsp_valid&&!host_rsp.poison;fault=(state==ERR);
 end
 always_ff@(posedge clk or negedge rst_n)begin
  if(!rst_n)begin state<=IDLE;sq_head<=0;cq_tail<=0;dq<='0;cq<='0;end else case(state)
   IDLE:if(enable&&(sq_size>=2)&&(cq_size>=2)&&(sq_head!=(sq_tail&sq_mask))&&(next_cq!=(cq_head&cq_mask)))state<=FETCH_Q;
   FETCH_Q:if(host_req_valid&&host_req_ready)state<=FETCH_W;
   FETCH_W:if(host_rsp_valid&&host_rsp_ready)begin if(host_rsp.poison)state<=ERR;else begin dq<=fx_desc_from_le512(host_rsp.data);state<=DISPATCH;end end
   DISPATCH:if(cmd_valid&&cmd_ready)state<=WAIT_CPL;
   WAIT_CPL:if(cpl_valid&&cpl_ready)begin cq<=cpl;state<=WRITE_Q;end
   WRITE_Q:if(host_req_valid&&host_req_ready)state<=WRITE_W;
   WRITE_W:if(host_rsp_valid&&host_rsp_ready)begin if(host_rsp.poison)state<=ERR;else begin sq_head<=(sq_head+1)&sq_mask;cq_tail<=next_cq;state<=IDLE;end end
   ERR:if(!enable)state<=IDLE;default:state<=ERR;
  endcase
 end
endmodule
