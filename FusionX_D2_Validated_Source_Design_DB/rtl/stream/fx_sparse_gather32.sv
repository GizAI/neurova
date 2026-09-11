`timescale 1ns/1ps
module fx_sparse_gather32(input logic[511:0]table,input logic[32*5-1:0]indices,output logic[511:0]y);
 integer i;logic[4:0]idx;always_comb begin y='0;for(i=0;i<32;i=i+1)begin idx=indices[i*5+:5];y[i*16+:16]=table[idx*16+:16];end end
endmodule
