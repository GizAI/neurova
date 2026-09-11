`timescale 1ns/1ps
module fx_rope32(input logic[511:0]x,input logic[511:0]cos_sin,output logic[511:0]y);
 integer i;logic signed[15:0]a,b,c,s;logic signed[31:0]yr,yi;
 always_comb begin y='0;for(i=0;i<16;i=i+1)begin a=x[(2*i)*16+:16];b=x[(2*i+1)*16+:16];c=cos_sin[(2*i)*16+:16];s=cos_sin[(2*i+1)*16+:16];yr=(a*c-b*s)>>>8;yi=(a*s+b*c)>>>8;y[(2*i)*16+:16]=yr[15:0];y[(2*i+1)*16+:16]=yi[15:0];end end
endmodule
