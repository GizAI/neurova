`timescale 1ns/1ps
// Routes core requests either to local 64MiB-class banked scratchpad (addr[63]=1)
// or to the external HBM/controller boundary (addr[63]=0).
module fx_device_mem_router #(
 parameter int SPAD_BANKS=32,parameter int SPAD_WORDS_PER_BANK=32768,
 localparam int SPAD_AW=$clog2(SPAD_BANKS*SPAD_WORDS_PER_BANK)
)(
 input logic clk,input logic rst_n,
 input logic t_req_valid,output logic t_req_ready,input fx_d2_pkg::fx_mem_req_t t_req,
 output logic t_rsp_valid,input logic t_rsp_ready,output fx_d2_pkg::fx_mem_rsp_t t_rsp,
 input logic s_req_valid,output logic s_req_ready,input fx_d2_pkg::fx_mem_req_t s_req,
 output logic s_rsp_valid,input logic s_rsp_ready,output fx_d2_pkg::fx_mem_rsp_t s_rsp,
 output logic ext_req_valid,input logic ext_req_ready,output fx_d2_pkg::fx_mem_req_t ext_req,
 input logic ext_rsp_valid,output logic ext_rsp_ready,input fx_d2_pkg::fx_mem_rsp_t ext_rsp
);
 import fx_d2_pkg::*;
 logic[1:0]sp_v,sp_r,sp_w,sp_rv,sp_rr;
 logic[2*SPAD_AW-1:0]sp_a;logic[1023:0]sp_wd,sp_rd;logic[127:0]sp_st;
 logic t_local_pending,s_local_pending,rr_sel,grant_t,grant_s;
 logic[15:0]t_local_id,s_local_id;
 fx_scratchpad_banked #(.CLIENTS(2),.BANKS(SPAD_BANKS),.WORDS_PER_BANK(SPAD_WORDS_PER_BANK))u_sp(
   .clk,.rst_n,.req_valid(sp_v),.req_ready(sp_r),.req_write(sp_w),.req_addr(sp_a),
   .req_wdata(sp_wd),.req_strb(sp_st),.rsp_valid(sp_rv),.rsp_ready(sp_rr),.rsp_rdata(sp_rd));

 always_comb begin
  sp_v='0;sp_w='0;sp_a='0;sp_wd='0;sp_st='0;sp_rr='0;
  if(t_req_valid&&t_req.addr[63]&&!t_local_pending)begin sp_v[0]=1;sp_w[0]=t_req.write;sp_a[0+:SPAD_AW]=t_req.addr[6+:SPAD_AW];sp_wd[0+:512]=t_req.data;sp_st[0+:64]=t_req.strb;end
  if(s_req_valid&&s_req.addr[63]&&!s_local_pending)begin sp_v[1]=1;sp_w[1]=s_req.write;sp_a[SPAD_AW+:SPAD_AW]=s_req.addr[6+:SPAD_AW];sp_wd[512+:512]=s_req.data;sp_st[64+:64]=s_req.strb;end

  grant_t=0;grant_s=0;ext_req_valid=0;ext_req='0;
  if(!rr_sel)begin
    if(t_req_valid&&!t_req.addr[63])grant_t=1;else if(s_req_valid&&!s_req.addr[63])grant_s=1;
  end else begin
    if(s_req_valid&&!s_req.addr[63])grant_s=1;else if(t_req_valid&&!t_req.addr[63])grant_t=1;
  end
  if(grant_t)begin ext_req_valid=1;ext_req=t_req;ext_req.id[15]=0;end
  else if(grant_s)begin ext_req_valid=1;ext_req=s_req;ext_req.id[15]=1;end

  t_req_ready=t_req_valid?(t_req.addr[63]?(!t_local_pending&&sp_r[0]):(grant_t&&ext_req_ready)):1'b0;
  s_req_ready=s_req_valid?(s_req.addr[63]?(!s_local_pending&&sp_r[1]):(grant_s&&ext_req_ready)):1'b0;

  t_rsp_valid=t_local_pending?sp_rv[0]:(ext_rsp_valid&&!ext_rsp.id[15]);
  s_rsp_valid=s_local_pending?sp_rv[1]:(ext_rsp_valid&&ext_rsp.id[15]);
  t_rsp='0;s_rsp='0;
  if(t_local_pending)begin t_rsp.id=t_local_id;t_rsp.data=sp_rd[0+:512];t_rsp.last=1;t_rsp.poison=0;end
  else begin t_rsp=ext_rsp;t_rsp.id[15]=0;end
  if(s_local_pending)begin s_rsp.id=s_local_id;s_rsp.data=sp_rd[512+:512];s_rsp.last=1;s_rsp.poison=0;end
  else begin s_rsp=ext_rsp;s_rsp.id[15]=0;end

  sp_rr[0]=t_local_pending&&t_rsp_ready;sp_rr[1]=s_local_pending&&s_rsp_ready;
  ext_rsp_ready=ext_rsp_valid?(ext_rsp.id[15]?s_rsp_ready:t_rsp_ready):1'b0;
 end

 always_ff@(posedge clk or negedge rst_n)begin
  if(!rst_n)begin t_local_pending<=0;s_local_pending<=0;t_local_id<=0;s_local_id<=0;rr_sel<=0;end
  else begin
   if(t_req_valid&&t_req_ready)begin if(t_req.addr[63])begin t_local_pending<=1;t_local_id<=t_req.id;end else rr_sel<=1;end
   if(s_req_valid&&s_req_ready)begin if(s_req.addr[63])begin s_local_pending<=1;s_local_id<=s_req.id;end else rr_sel<=0;end
   if(t_rsp_valid&&t_rsp_ready&&t_local_pending)t_local_pending<=0;
   if(s_rsp_valid&&s_rsp_ready&&s_local_pending)s_local_pending<=0;
  end
 end
endmodule
