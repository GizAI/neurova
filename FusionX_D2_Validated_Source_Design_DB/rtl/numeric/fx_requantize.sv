`timescale 1ns/1ps
module fx_requantize #(
  parameter int ACC_W=48
)(
  input  logic signed [ACC_W-1:0] acc,
  input  logic [15:0] scale_q2_14,
  input  logic [5:0] right_shift,
  output logic signed [15:0] value_q8_8
);
  longint signed product, signed_q;
  longint unsigned magnitude, quotient, remainder, half, mask;
  integer total_shift;
  logic negative;
  always_comb begin
    product=$signed(acc)*$signed({1'b0,scale_q2_14});
    total_shift=14+right_shift;
    negative=(product<0);
    magnitude=negative ? $unsigned(-product) : $unsigned(product);
    if(total_shift==0) quotient=magnitude;
    else if(total_shift>=63) begin quotient=0; remainder=magnitude; half=64'h8000_0000_0000_0000; end
    else begin
      quotient=magnitude>>total_shift;
      mask=(64'h1<<total_shift)-1;
      remainder=magnitude&mask;
      half=64'h1<<(total_shift-1);
      if((remainder>half)||((remainder==half)&&quotient[0])) quotient=quotient+1;
    end
    signed_q=negative ? -$signed(quotient) : $signed(quotient);
    if(signed_q>32767) value_q8_8=16'sh7fff;
    else if(signed_q< -32768) value_q8_8=16'sh8000;
    else value_q8_8=signed_q[15:0];
  end
endmodule
