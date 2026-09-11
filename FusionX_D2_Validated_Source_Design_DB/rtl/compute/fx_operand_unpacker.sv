`timescale 1ns/1ps
module fx_operand_unpacker #(
  parameter int ELEMS=64,
  localparam int IW=(ELEMS<=1)?1:$clog2(ELEMS+1)
)(
  input  logic clk,
  input  logic rst_n,
  input  logic start,
  input  logic [7:0] dtype,
  input  logic [15:0] scale_q2_14,
  input  logic beat_valid,
  output logic beat_ready,
  input  logic [511:0] beat_data,
  output logic signed [ELEMS*16-1:0] vector_q8_8,
  output logic done,
  output logic busy,
  output logic error
);
  import fx_d2_pkg::*;
  logic [IW-1:0] loaded;
  integer i, epb, pos;
  logic [15:0] raw;
  logic signed [15:0] decoded;
  logic finite, supported;

  function automatic longint signed rne_shift_signed(input longint signed x,input integer sh);
    longint unsigned magnitude,quotient,remainder,half,mask;logic negative;
    begin
      if(sh<=0)rne_shift_signed=x;
      else begin negative=(x<0);magnitude=negative?$unsigned(-x):$unsigned(x);if(sh>=63)begin quotient=0;remainder=magnitude;half=64'h8000_0000_0000_0000;end else begin quotient=magnitude>>sh;mask=(64'h1<<sh)-1;remainder=magnitude&mask;half=64'h1<<(sh-1);if((remainder>half)||((remainder==half)&&quotient[0]))quotient=quotient+1;end rne_shift_signed=negative?-$signed(quotient):$signed(quotient);end
    end
  endfunction

  function automatic logic signed [15:0] decode_elem(
    input logic [7:0] dt,input logic [15:0] rv,input logic [15:0] sc
  );
    integer signed mag,sh;longint signed q16,prod,q8;
    begin
      mag=0;sh=0;q16=0;
      case(dt)
        FX_DT_INT4:q16=$signed({{60{rv[3]}},rv[3:0]})<<<16;
        FX_DT_FP4_E2M1:begin case(rv[2:0])0:mag=0;1:mag=32768;2:mag=65536;3:mag=98304;4:mag=131072;5:mag=196608;6:mag=262144;default:mag=393216;endcase q16=rv[3]?-mag:mag;end
        FX_DT_INT8:q16=$signed({{56{rv[7]}},rv[7:0]})<<<16;
        FX_DT_FP8_E4M3FN:begin if((rv[6:3]==4'hf)&&(rv[2:0]==3'h7))q16=0;else begin if(rv[6:3]==0)mag=rv[2:0]<<<7;else mag=(8+rv[2:0])<<<(rv[6:3]+6);q16=rv[7]?-mag:mag;end end
        FX_DT_INT16:q16=$signed({{48{rv[15]}},rv})<<<16;
        FX_DT_BF16:begin if((rv[14:7]==0)||(rv[14:7]==8'hff))q16=0;else begin sh=$signed({1'b0,rv[14:7]})-127+9;mag=128+rv[6:0];if(sh>31)q16=rv[15]?-64'sh7fff_ffff:64'sh7fff_ffff;else begin if(sh>=0)q16=mag<<<sh;else q16=mag>>>(-sh);if(rv[15])q16=-q16;end end end
        default:q16=0;
      endcase
      prod=q16*$signed({1'b0,sc});q8=rne_shift_signed(rne_shift_signed(prod,14),8);
      if(q8>32767)decode_elem=16'sh7fff;else if(q8< -32768)decode_elem=16'sh8000;else decode_elem=q8[15:0];
    end
  endfunction

  always_comb begin
    case(dtype)
      FX_DT_INT4,FX_DT_FP4_E2M1: epb=128;
      FX_DT_INT8,FX_DT_FP8_E4M3FN: epb=64;
      default: epb=32;
    endcase
    beat_ready=busy;
    error=busy&&!fx_tensor_dtype(dtype);
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      busy<=1'b0;
      loaded<='0;
      vector_q8_8<='0;
      done<=1'b0;
    end else begin
      done<=1'b0;
      if(start&&!busy) begin
        busy<=1'b1;
        loaded<='0;
        vector_q8_8<='0;
      end else if(busy&&error) begin
        busy<=1'b0;
      end else if(beat_valid&&beat_ready) begin
        for(i=0;i<128;i=i+1) begin
          pos=loaded+i;
          if((i<epb)&&(pos<ELEMS)) begin
            case(dtype)
              FX_DT_INT4,FX_DT_FP4_E2M1: raw={12'b0,beat_data[i*4 +:4]};
              FX_DT_INT8,FX_DT_FP8_E4M3FN: raw={8'b0,beat_data[i*8 +:8]};
              default: raw=beat_data[i*16 +:16];
            endcase
            decoded=decode_elem(dtype,raw,scale_q2_14);
            vector_q8_8[pos*16 +:16]<=decoded;
          end
        end
        if(loaded+epb>=ELEMS) begin
          loaded<=ELEMS;
          busy<=1'b0;
          done<=1'b1;
        end else loaded<=loaded+epb;
      end
    end
  end
endmodule
