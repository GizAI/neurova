`timescale 1ns/1ps
module fx_patchify3d(input logic[511:0]x,output logic[511:0]y);
 integer i;always_comb begin y='0;for(i=0;i<64;i=i+1)y[i*8+:8]=x[(((i&7)<<3)|(i>>3))*8+:8];end
endmodule
