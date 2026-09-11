`timescale 1ns/1ps
module fx_vector_alu32(
 input logic [1:0] op,
 input logic [511:0] a,b,
 input logic [15:0] scale_q2_14,
 output logic [511:0] y
);
 integer i;logic signed[31:0] t;logic signed[15:0] av,bv;
 always_comb begin
  y='0;
  for(i=0;i<32;i=i+1)begin
   av=a[i*16+:16];bv=b[i*16+:16];
   case(op)
    2'd0:t=av+bv;
    2'd1:t=av-bv;
    2'd2:t=(av*bv)>>>8;
    default:t=(av*$signed({1'b0,scale_q2_14}))>>>14;
   endcase
   if(t>32767)y[i*16+:16]=16'h7fff;else if(t< -32768)y[i*16+:16]=16'h8000;else y[i*16+:16]=t[15:0];
  end
 end
endmodule
