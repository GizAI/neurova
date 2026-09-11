`timescale 1ns/1ps
module fx_sync_bits #(parameter int W=1)(input logic clk,input logic rst_n,input logic[W-1:0]async_i,output logic[W-1:0]sync_o);
 (* ASYNC_REG="TRUE" *) logic[W-1:0]q1,q2;always_ff@(posedge clk or negedge rst_n)begin if(!rst_n)begin q1<='0;q2<='0;end else begin q1<=async_i;q2<=q1;end end assign sync_o=q2;
endmodule
