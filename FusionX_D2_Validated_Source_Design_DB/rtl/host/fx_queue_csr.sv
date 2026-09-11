`timescale 1ns/1ps
module fx_queue_csr(
 input logic clk,input logic rst_n,
 input logic wr_valid,input logic[15:0]wr_addr,input logic[31:0]wr_data,
 input logic rd_valid,input logic[15:0]rd_addr,output logic[31:0]rd_data,
 input logic[31:0]sq_head_i,input logic[31:0]cq_tail_i,
 output logic[63:0]sq_base,output logic[31:0]sq_size,output logic[31:0]sq_tail,
 output logic[63:0]cq_base,output logic[31:0]cq_size,output logic[31:0]cq_head,
 output logic sq_doorbell,output logic cq_doorbell,output logic irq_enable,output logic device_enable
);
 import fx_d2_abi_pkg::*;
 logic[31:0]control,irq_mask;
 always_ff@(posedge clk or negedge rst_n)begin
  if(!rst_n)begin sq_base<=0;sq_size<=0;sq_tail<=0;cq_base<=0;cq_size<=0;cq_head<=0;control<=0;irq_mask<=0;sq_doorbell<=0;cq_doorbell<=0;end
  else begin sq_doorbell<=0;cq_doorbell<=0;if(wr_valid)case(wr_addr)
   FX_D2_REG_CONTROL:control<=wr_data;
   FX_D2_REG_IRQ_MASK:irq_mask<=wr_data;
   FX_D2_REG_SQ_BASE_LO:sq_base[31:0]<=wr_data;FX_D2_REG_SQ_BASE_HI:sq_base[63:32]<=wr_data;
   FX_D2_REG_SQ_SIZE:sq_size<=wr_data;FX_D2_REG_SQ_TAIL:sq_tail<=wr_data;FX_D2_REG_SQ_DOORBELL:sq_doorbell<=1;
   FX_D2_REG_CQ_BASE_LO:cq_base[31:0]<=wr_data;FX_D2_REG_CQ_BASE_HI:cq_base[63:32]<=wr_data;
   FX_D2_REG_CQ_SIZE:cq_size<=wr_data;FX_D2_REG_CQ_HEAD:cq_head<=wr_data;FX_D2_REG_CQ_DOORBELL:cq_doorbell<=1;
   default:begin end endcase end
 end
 assign irq_enable=irq_mask[0];assign device_enable=control[0];
 always_comb begin rd_data=0;case(rd_addr)
  FX_D2_REG_ID:rd_data=32'h46584432;FX_D2_REG_VERSION:rd_data=FX_D2_ABI_VERSION;
  FX_D2_REG_CONTROL:rd_data=control;FX_D2_REG_IRQ_MASK:rd_data=irq_mask;
  FX_D2_REG_SQ_BASE_LO:rd_data=sq_base[31:0];FX_D2_REG_SQ_BASE_HI:rd_data=sq_base[63:32];
  FX_D2_REG_SQ_SIZE:rd_data=sq_size;FX_D2_REG_SQ_HEAD:rd_data=sq_head_i;FX_D2_REG_SQ_TAIL:rd_data=sq_tail;
  FX_D2_REG_CQ_BASE_LO:rd_data=cq_base[31:0];FX_D2_REG_CQ_BASE_HI:rd_data=cq_base[63:32];
  FX_D2_REG_CQ_SIZE:rd_data=cq_size;FX_D2_REG_CQ_HEAD:rd_data=cq_head;FX_D2_REG_CQ_TAIL:rd_data=cq_tail_i;
  default:rd_data=0;endcase end
endmodule
