`timescale 1ns/1ps
// Product target: BANKS=32, WORDS_PER_BANK=32768 => 64 MiB at 512 bits/word.
// One request per bank per cycle; different clients can access different banks concurrently.
module fx_scratchpad_banked #(
 parameter int CLIENTS=4,parameter int BANKS=32,parameter int WORDS_PER_BANK=32768,
 localparam int BW=(BANKS<=1)?1:$clog2(BANKS),localparam int WW=(WORDS_PER_BANK<=1)?1:$clog2(WORDS_PER_BANK),
 localparam int AW=BW+WW
)(
 input logic clk,input logic rst_n,
 input logic[CLIENTS-1:0]req_valid,output logic[CLIENTS-1:0]req_ready,
 input logic[CLIENTS-1:0]req_write,input logic[CLIENTS*AW-1:0]req_addr,
 input logic[CLIENTS*512-1:0]req_wdata,input logic[CLIENTS*64-1:0]req_strb,
 output logic[CLIENTS-1:0]rsp_valid,input logic[CLIENTS-1:0]rsp_ready,
 output logic[CLIENTS*512-1:0]rsp_rdata
);
 logic[511:0]mem[0:BANKS-1][0:WORDS_PER_BANK-1];
 logic[CLIENTS-1:0]rsp_q,grant;
 logic[CLIENTS*512-1:0]rdata_q;
 logic[BW-1:0]bank_sel[0:CLIENTS-1];logic[WW-1:0]word_sel[0:CLIENTS-1];
 logic[BANKS-1:0]bank_taken;
 integer c,b,j;
 assign rsp_valid=rsp_q;assign rsp_rdata=rdata_q;
 always_comb begin
   req_ready='0;grant='0;bank_taken='0;
   for(c=0;c<CLIENTS;c=c+1)begin
     bank_sel[c]=req_addr[c*AW +:BW];
     word_sel[c]=req_addr[c*AW+BW +:WW];
     if(req_valid[c]&&!rsp_q[c]&&!bank_taken[bank_sel[c]])begin grant[c]=1;req_ready[c]=1;bank_taken[bank_sel[c]]=1;end
   end
 end
 always_ff@(posedge clk or negedge rst_n)begin
  if(!rst_n)begin rsp_q<='0;rdata_q<='0;end
  else begin
   for(c=0;c<CLIENTS;c=c+1)begin
    if(rsp_q[c]&&rsp_ready[c])rsp_q[c]<=0;
    if(req_valid[c]&&grant[c])begin
     rdata_q[c*512 +:512]<=mem[bank_sel[c]][word_sel[c]];
     if(req_write[c])for(j=0;j<64;j=j+1)if(req_strb[c*64+j])mem[bank_sel[c]][word_sel[c]][j*8 +:8]<=req_wdata[c*512+j*8 +:8];
     rsp_q[c]<=1;
    end
   end
  end
 end
 initial begin
   if((BANKS&(BANKS-1))!=0)$error("BANKS must be power of two");
   if((WORDS_PER_BANK&(WORDS_PER_BANK-1))!=0)$error("WORDS_PER_BANK must be power of two");
 end
endmodule
