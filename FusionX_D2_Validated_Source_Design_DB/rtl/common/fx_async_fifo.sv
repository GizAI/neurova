`timescale 1ns/1ps
module fx_async_fifo #(parameter int W=32,parameter int DEPTH=8,localparam int AW=$clog2(DEPTH),localparam int PW=AW+1)(
 input logic wclk,input logic wrst_n,input logic wvalid,output logic wready,input logic[W-1:0]wdata,
 input logic rclk,input logic rrst_n,output logic rvalid,input logic rready,output logic[W-1:0]rdata
);
 logic[W-1:0]mem[0:DEPTH-1];logic[PW-1:0]wbin,wgray,rbin,rgray;
 (* ASYNC_REG="TRUE" *) logic[PW-1:0]rgray_w1,rgray_w2,wgray_r1,wgray_r2;
 logic[PW-1:0]wbin_next,wgray_next,rbin_next,rgray_next;logic full,empty;
 assign wbin_next=wbin+(wvalid&&wready);assign wgray_next=(wbin_next>>1)^wbin_next;
 assign rbin_next=rbin+(rvalid&&rready);assign rgray_next=(rbin_next>>1)^rbin_next;
 assign full=(wgray_next=={~rgray_w2[PW-1:PW-2],rgray_w2[PW-3:0]});assign empty=(rgray==wgray_r2);
 assign wready=!full;assign rvalid=!empty;assign rdata=mem[rbin[AW-1:0]];
 always_ff@(posedge wclk or negedge wrst_n)begin if(!wrst_n)begin wbin<=0;wgray<=0;rgray_w1<=0;rgray_w2<=0;end else begin rgray_w1<=rgray;rgray_w2<=rgray_w1;if(wvalid&&wready)begin mem[wbin[AW-1:0]]<=wdata;wbin<=wbin_next;wgray<=wgray_next;end end end
 always_ff@(posedge rclk or negedge rrst_n)begin if(!rrst_n)begin rbin<=0;rgray<=0;wgray_r1<=0;wgray_r2<=0;end else begin wgray_r1<=wgray;wgray_r2<=wgray_r1;if(rvalid&&rready)begin rbin<=rbin_next;rgray<=rgray_next;end end end
endmodule
