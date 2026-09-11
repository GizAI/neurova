`timescale 1ns/1ps
module fx_rmsnorm32(
 input logic [511:0] x,
 input logic [511:0] gamma,
 input logic [31:0] epsilon_q16_16,
 output logic [511:0] y
);
 integer i;longint signed xi,gi,t;longint unsigned sumsq,mean,root;
 function automatic longint unsigned isqrt(input longint unsigned v);
  longint unsigned r,bit,trial;begin r=0;bit=64'h4000_0000_0000_0000;while(bit>v)bit=bit>>2;while(bit!=0)begin trial=r+bit;if(v>=trial)begin v=v-trial;r=(r>>1)+bit;end else r=r>>1;bit=bit>>2;end isqrt=r;end
 endfunction
 always_comb begin
  sumsq=0;y='0;
  for(i=0;i<32;i=i+1)begin xi=$signed(x[i*16+:16]);sumsq=sumsq+xi*xi;end
  mean=(sumsq/32)+epsilon_q16_16;root=isqrt(mean);
  if(root==0)root=1;
  for(i=0;i<32;i=i+1)begin
   xi=$signed(x[i*16+:16]);gi=$signed(gamma[i*16+:16]);t=((xi*gi)<<8)/$signed(root);
   if(t>32767)y[i*16+:16]=16'h7fff;else if(t< -32768)y[i*16+:16]=16'h8000;else y[i*16+:16]=t[15:0];
  end
 end
endmodule
