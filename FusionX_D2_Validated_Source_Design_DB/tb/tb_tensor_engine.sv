`timescale 1ns/1ps
module tb_tensor_engine;
 import fx_d2_pkg::*;
 logic clk=0,rst_n=0,enable=1,dv,dr,rv,rr,sv,sr,cv,cr=1,busy,fault;fx_desc_t d;fx_mem_req_t rq;fx_mem_rsp_t rs;fx_cpl_t cpl;integer i,writes;logic pending;fx_mem_req_t pending_req;
 always #2 clk=~clk;
 fx_tensor_engine #(.ROWS(8),.COLS(8))dut(.clk,.rst_n,.enable,.desc_valid(dv),.desc_ready(dr),.desc(d),.mem_req_valid(rv),.mem_req_ready(rr),.mem_req(rq),.mem_rsp_valid(sv),.mem_rsp_ready(sr),.mem_rsp(rs),.cpl_valid(cv),.cpl_ready(cr),.cpl,.busy,.fault);
 assign rr=!pending;
 always_ff@(posedge clk)begin
  if(!rst_n)begin pending<=0;pending_req<='0;sv<=0;rs<='0;writes<=0;end else begin
   sv<=0;
   if(rv&&rr)begin pending<=1;pending_req<=rq;end
   if(pending)begin
    rs<='0;rs.id<=pending_req.id;rs.last<=1;rs.poison<=0;
    if(!pending_req.write)begin
      if(pending_req.addr>=64'h2000)begin for(i=0;i<8;i=i+1)rs.data[i*16+:16]<=((pending_req.addr-64'h2000)==0)?16'd3:16'd4;end
      else begin for(i=0;i<8;i=i+1)rs.data[i*16+:16]<=((pending_req.addr-64'h1000)==0)?16'd1:16'd2;end
    end else begin
      for(i=0;i<8;i=i+1)if($signed(pending_req.data[i*16+:16])!=16'sd2816)$fatal(1,"tensor result row=%0d lane=%0d got=%0d",writes,i,$signed(pending_req.data[i*16+:16]));
      if(pending_req.strb[15:0]!==16'hffff||pending_req.strb[63:16]!=='0)$fatal(1,"write strobe");writes<=writes+1;
    end
    sv<=1;pending<=0;
   end
  end
 end
 initial begin dv=0;d='0;repeat(4)@(posedge clk);rst_n=1;d.opcode=FX_OP_GEMM;d.dtype=FX_DT_INT16;d.tag=16'h55;d.m=8;d.n=8;d.k=2;d.src0=64'h1000;d.src1=64'h2000;d.dst=64'h3000;d.scale_q2_14=16'h4000;d.out_scale_q2_14=16'h4000;d.out_shift=8;wait(dr);@(posedge clk);dv<=1;@(posedge clk);dv<=0;wait(cv);if(fault||cpl.status!=0||writes!=8)$fatal(1,"tensor completion writes=%0d fault=%b",writes,fault);$display("PASS tb_tensor_engine writes=%0d",writes);$finish;end
endmodule
