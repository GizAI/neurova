`timescale 1ns/1ps
module tb_csr;import fx_d2_abi_pkg::*;logic clk=0,rst_n=0,wv,rv;logic[15:0]wa,ra;logic[31:0]wd,rd;logic[63:0]sqb,cqb;logic[31:0]sqs,sqt,cqs,cqh;logic sdb,cdb,ie,de;always #5 clk=~clk;
 fx_queue_csr dut(.clk,.rst_n,.wr_valid(wv),.wr_addr(wa),.wr_data(wd),.rd_valid(rv),.rd_addr(ra),.rd_data(rd),.sq_head_i(32'd3),.cq_tail_i(32'd5),.sq_base(sqb),.sq_size(sqs),.sq_tail(sqt),.cq_base(cqb),.cq_size(cqs),.cq_head(cqh),.sq_doorbell(sdb),.cq_doorbell(cdb),.irq_enable(ie),.device_enable(de));
 task wr(input[15:0]a,input[31:0]d);begin @(posedge clk);wv<=1;wa<=a;wd<=d;@(posedge clk);wv<=0;end endtask
 initial begin wv=0;rv=0;wa=0;ra=0;wd=0;repeat(2)@(posedge clk);rst_n=1;wr(FX_D2_REG_CONTROL,1);wr(FX_D2_REG_SQ_SIZE,8);wr(FX_D2_REG_CQ_SIZE,8);ra=FX_D2_REG_SQ_HEAD;#1;if(rd!=3||!de)$fatal(1);ra=FX_D2_REG_CQ_TAIL;#1;if(rd!=5)$fatal(1);$display("PASS tb_csr");$finish;end
endmodule
