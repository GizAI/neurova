`timescale 1ns/1ps
module fx_adaln32(input logic[511:0]x,scale,bias,output logic[511:0]y);
 integer i;logic signed[15:0]xi,si,bi;logic signed[31:0]t;always_comb begin y='0;for(i=0;i<32;i=i+1)begin xi=x[i*16+:16];si=scale[i*16+:16];bi=bias[i*16+:16];t=((xi*si)>>>8)+bi;if(t>32767)t=32767;else if(t< -32768)t=-32768;y[i*16+:16]=t[15:0];end end
endmodule
