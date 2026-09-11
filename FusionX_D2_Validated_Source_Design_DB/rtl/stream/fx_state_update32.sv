`timescale 1ns/1ps
module fx_state_update32(input logic[511:0]x,state,gate,output logic[511:0]next_state);
 integer i;logic signed[15:0]xi,si,gi;logic signed[31:0]t;
 always_comb begin next_state='0;for(i=0;i<32;i=i+1)begin xi=x[i*16+:16];si=state[i*16+:16];gi=gate[i*16+:16];t=si+(((xi-si)*gi)>>>15);if(t>32767)t=32767;else if(t< -32768)t=-32768;next_state[i*16+:16]=t[15:0];end end
endmodule
