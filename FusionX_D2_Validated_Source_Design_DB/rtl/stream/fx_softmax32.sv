`timescale 1ns/1ps
module fx_softmax32(input logic [511:0] x,output logic [511:0] y);
 integer i;integer signed mx,xi,d;longint unsigned ex[0:31];longint unsigned total,q;
 function automatic longint unsigned exp_q16(input integer signed z);
  integer n;longint unsigned r;begin
   if(z>=0)r=65536;else if(z<=-2048)r=22;else begin
    n=(-z)>>5;
    r=65536;
    repeat(64)begin if(n>0)begin r=(r*57826)>>16;n=n-1;end end
   end
   exp_q16=r;
  end
 endfunction
 always_comb begin
  mx=-32768;total=0;y='0;
  for(i=0;i<32;i=i+1)begin xi=$signed(x[i*16+:16]);if(xi>mx)mx=xi;end
  for(i=0;i<32;i=i+1)begin xi=$signed(x[i*16+:16]);d=xi-mx;ex[i]=exp_q16(d);total=total+ex[i];end
  if(total==0)total=1;
  for(i=0;i<32;i=i+1)begin q=(ex[i]<<16)/total;if(q>65535)q=65535;y[i*16+:16]=q[15:0];end
 end
endmodule
