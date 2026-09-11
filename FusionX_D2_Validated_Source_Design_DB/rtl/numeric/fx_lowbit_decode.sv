`timescale 1ns/1ps
module fx_lowbit_decode(
  input  logic [7:0] dtype,
  input  logic [15:0] raw,
  input  logic [15:0] scale_q2_14,
  output logic signed [15:0] value_q8_8,
  output logic finite,
  output logic supported
);
  import fx_d2_pkg::*;
  longint signed decoded_q16_16, scaled_q30, rounded_q16, rounded_q8;
  integer signed shift, mag;

  function automatic longint signed rne_shift_signed(input longint signed x,input integer sh);
    longint unsigned magnitude,quotient,remainder,half,mask;
    logic negative;
    begin
      if(sh<=0) rne_shift_signed=x;
      else begin
        negative=(x<0);magnitude=negative?$unsigned(-x):$unsigned(x);
        if(sh>=63)begin quotient=0;remainder=magnitude;half=64'h8000_0000_0000_0000;end
        else begin quotient=magnitude>>sh;mask=(64'h1<<sh)-1;remainder=magnitude&mask;half=64'h1<<(sh-1);if((remainder>half)||((remainder==half)&&quotient[0]))quotient=quotient+1;end
        rne_shift_signed=negative?-$signed(quotient):$signed(quotient);
      end
    end
  endfunction

  always_comb begin
    decoded_q16_16=0;finite=1'b1;supported=1'b1;shift=0;mag=0;
    unique case(dtype)
      FX_DT_INT4: decoded_q16_16=$signed({{60{raw[3]}},raw[3:0]})<<<16;
      FX_DT_FP4_E2M1: begin
        unique case(raw[2:0])
          3'b000:mag=0;3'b001:mag=32768;3'b010:mag=65536;3'b011:mag=98304;
          3'b100:mag=131072;3'b101:mag=196608;3'b110:mag=262144;default:mag=393216;
        endcase
        decoded_q16_16=raw[3]?-mag:mag;
      end
      FX_DT_INT8: decoded_q16_16=$signed({{56{raw[7]}},raw[7:0]})<<<16;
      FX_DT_FP8_E4M3FN: begin
        if((raw[6:3]==4'hf)&&(raw[2:0]==3'h7))begin finite=1'b0;decoded_q16_16=0;end
        else begin
          if(raw[6:3]==0)mag=raw[2:0]<<<7;
          else mag=(8+raw[2:0])<<<(raw[6:3]+6);
          decoded_q16_16=raw[7]?-mag:mag;
        end
      end
      FX_DT_INT16:decoded_q16_16=$signed({{48{raw[15]}},raw})<<<16;
      FX_DT_BF16:begin
        if(raw[14:7]==8'hff)begin finite=1'b0;decoded_q16_16=0;end
        else if(raw[14:7]==0)decoded_q16_16=0;
        else begin shift=$signed({1'b0,raw[14:7]})-127+9;mag=128+raw[6:0];if(shift>31)decoded_q16_16=raw[15]?-64'sh7fff_ffff:64'sh7fff_ffff;else begin if(shift>=0)decoded_q16_16=mag<<<shift;else decoded_q16_16=mag>>>(-shift);if(raw[15])decoded_q16_16=-decoded_q16_16;end end
      end
      default:begin supported=1'b0;finite=1'b0;decoded_q16_16=0;end
    endcase
    scaled_q30=decoded_q16_16*$signed({1'b0,scale_q2_14});
    rounded_q16=rne_shift_signed(scaled_q30,14);
    rounded_q8=rne_shift_signed(rounded_q16,8);
    if(rounded_q8>32767)value_q8_8=16'sh7fff;else if(rounded_q8< -32768)value_q8_8=16'sh8000;else value_q8_8=rounded_q8[15:0];
  end
endmodule
