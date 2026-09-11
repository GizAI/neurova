`timescale 1ns/1ps
// Integrated D2 front-end compute subsystem.
// Host queue traffic and core execution are separated by asynchronous FIFOs.
// Proprietary PCIe/CXL and HBM PHY/controllers remain outside this module.
module fusionx_d2_core #(
  parameter int TENSOR_ROWS=64,
  parameter int TENSOR_COLS=64,
  parameter int SPAD_BANKS=32,
  parameter int SPAD_WORDS_PER_BANK=32768,
  parameter int FIFO_DEPTH=8
)(
  input  logic host_clk,input logic host_rst_n,
  input  logic core_clk,input logic core_rst_n,
  input  logic csr_wr_valid,input logic[15:0]csr_wr_addr,input logic[31:0]csr_wr_data,
  input  logic csr_rd_valid,input logic[15:0]csr_rd_addr,output logic[31:0]csr_rd_data,
  input  logic release_compute_host,input logic thermal_ok_host,
  input  logic hbm_init_host,input logic hbm_fatal_host,
  output logic irq,output logic fault,

  output logic host_mem_req_valid,input logic host_mem_req_ready,
  output fx_d2_pkg::fx_mem_req_t host_mem_req,
  input  logic host_mem_rsp_valid,output logic host_mem_rsp_ready,
  input  fx_d2_pkg::fx_mem_rsp_t host_mem_rsp,

  output logic device_mem_req_valid,input logic device_mem_req_ready,
  output fx_d2_pkg::fx_mem_req_t device_mem_req,
  input  logic device_mem_rsp_valid,output logic device_mem_rsp_ready,
  input  fx_d2_pkg::fx_mem_rsp_t device_mem_rsp
);
  import fx_d2_pkg::*;

  logic[63:0]sq_base,cq_base;logic[31:0]sq_size,sq_tail,sq_head,cq_size,cq_head,cq_tail;
  logic sq_db,cq_db,irq_enable,device_enable;
  logic qm_cmd_valid,qm_cmd_ready,qm_cmd_tensor;fx_desc_t qm_cmd_desc;
  logic qm_cpl_valid,qm_cpl_ready;fx_cpl_t qm_cpl;logic qm_irq,qm_fault;
  logic host_operational;

  fx_queue_csr u_csr(.clk(host_clk),.rst_n(host_rst_n),.wr_valid(csr_wr_valid),.wr_addr(csr_wr_addr),.wr_data(csr_wr_data),
    .rd_valid(csr_rd_valid),.rd_addr(csr_rd_addr),.rd_data(csr_rd_data),.sq_head_i(sq_head),.cq_tail_i(cq_tail),
    .sq_base,.sq_size,.sq_tail,.cq_base,.cq_size,.cq_head,.sq_doorbell(sq_db),.cq_doorbell(cq_db),.irq_enable,.device_enable);
  assign host_operational=device_enable&&release_compute_host&&thermal_ok_host&&hbm_init_host&&!hbm_fatal_host;

  fx_queue_manager u_qm(.clk(host_clk),.rst_n(host_rst_n),.enable(host_operational),
    .sq_base,.sq_size,.sq_tail,.sq_head,.cq_base,.cq_size,.cq_head,.cq_tail,
    .host_req_valid(host_mem_req_valid),.host_req_ready(host_mem_req_ready),.host_req(host_mem_req),
    .host_rsp_valid(host_mem_rsp_valid),.host_rsp_ready(host_mem_rsp_ready),.host_rsp(host_mem_rsp),
    .cmd_valid(qm_cmd_valid),.cmd_ready(qm_cmd_ready),.cmd_tensor(qm_cmd_tensor),.cmd_desc(qm_cmd_desc),
    .cpl_valid(qm_cpl_valid),.cpl_ready(qm_cpl_ready),.cpl(qm_cpl),.irq(qm_irq),.fault(qm_fault));

  localparam int CMD_W=FX_DESC_W+1;
  logic[CMD_W-1:0]cmd_wdata,cmd_rdata;logic cmd_wready,cmd_rvalid,cmd_rready;
  logic[FX_CPL_W-1:0]cpl_wdata,cpl_rdata;logic cpl_wvalid,cpl_wready,cpl_rvalid;
  assign cmd_wdata={qm_cmd_tensor,qm_cmd_desc};assign qm_cmd_ready=cmd_wready;
  fx_async_fifo #(.W(CMD_W),.DEPTH(FIFO_DEPTH))u_cmd_fifo(.wclk(host_clk),.wrst_n(host_rst_n),.wvalid(qm_cmd_valid),.wready(cmd_wready),.wdata(cmd_wdata),
    .rclk(core_clk),.rrst_n(core_rst_n),.rvalid(cmd_rvalid),.rready(cmd_rready),.rdata(cmd_rdata));
  fx_async_fifo #(.W(FX_CPL_W),.DEPTH(FIFO_DEPTH))u_cpl_fifo(.wclk(core_clk),.wrst_n(core_rst_n),.wvalid(cpl_wvalid),.wready(cpl_wready),.wdata(cpl_wdata),
    .rclk(host_clk),.rrst_n(host_rst_n),.rvalid(cpl_rvalid),.rready(qm_cpl_ready),.rdata(cpl_rdata));
  assign qm_cpl_valid=cpl_rvalid;assign qm_cpl=fx_cpl_t'(cpl_rdata);

  logic[3:0]status_core;logic[3:0]status_host;
  assign status_host={hbm_fatal_host,hbm_init_host,thermal_ok_host,release_compute_host};
  fx_sync_bits #(.W(4))u_status_sync(.clk(core_clk),.rst_n(core_rst_n),.async_i(status_host),.sync_o(status_core));
  logic core_operational;assign core_operational=status_core[0]&&status_core[1]&&status_core[2]&&!status_core[3];

  fx_desc_t core_desc;logic core_tensor;
  assign core_tensor=cmd_rdata[CMD_W-1];assign core_desc=fx_desc_t'(cmd_rdata[FX_DESC_W-1:0]);
  logic t_desc_ready,s_desc_ready,t_cpl_valid,s_cpl_valid,t_cpl_ready,s_cpl_ready;
  fx_cpl_t t_cpl,s_cpl;logic t_busy,s_busy,t_fault,s_fault;
  logic t_req_valid,t_req_ready,t_rsp_valid,t_rsp_ready;fx_mem_req_t t_req;fx_mem_rsp_t t_rsp;
  logic s_req_valid,s_req_ready,s_rsp_valid,s_rsp_ready;fx_mem_req_t s_req;fx_mem_rsp_t s_rsp;
  assign cmd_rready=core_operational&&(core_tensor?t_desc_ready:s_desc_ready);

  fx_tensor_engine #(.ROWS(TENSOR_ROWS),.COLS(TENSOR_COLS))u_tensor(.clk(core_clk),.rst_n(core_rst_n),.enable(core_operational),
    .desc_valid(cmd_rvalid&&core_tensor),.desc_ready(t_desc_ready),.desc(core_desc),
    .mem_req_valid(t_req_valid),.mem_req_ready(t_req_ready),.mem_req(t_req),.mem_rsp_valid(t_rsp_valid),.mem_rsp_ready(t_rsp_ready),.mem_rsp(t_rsp),
    .cpl_valid(t_cpl_valid),.cpl_ready(t_cpl_ready),.cpl(t_cpl),.busy(t_busy),.fault(t_fault));
  fx_stream_engine u_stream(.clk(core_clk),.rst_n(core_rst_n),.enable(core_operational),
    .desc_valid(cmd_rvalid&&!core_tensor),.desc_ready(s_desc_ready),.desc(core_desc),
    .mem_req_valid(s_req_valid),.mem_req_ready(s_req_ready),.mem_req(s_req),.mem_rsp_valid(s_rsp_valid),.mem_rsp_ready(s_rsp_ready),.mem_rsp(s_rsp),
    .cpl_valid(s_cpl_valid),.cpl_ready(s_cpl_ready),.cpl(s_cpl),.busy(s_busy),.fault(s_fault));

  fx_device_mem_router #(.SPAD_BANKS(SPAD_BANKS),.SPAD_WORDS_PER_BANK(SPAD_WORDS_PER_BANK))u_mem(.clk(core_clk),.rst_n(core_rst_n),
    .t_req_valid,.t_req_ready,.t_req,.t_rsp_valid,.t_rsp_ready,.t_rsp,
    .s_req_valid,.s_req_ready,.s_req,.s_rsp_valid,.s_rsp_ready,.s_rsp,
    .ext_req_valid(device_mem_req_valid),.ext_req_ready(device_mem_req_ready),.ext_req(device_mem_req),
    .ext_rsp_valid(device_mem_rsp_valid),.ext_rsp_ready(device_mem_rsp_ready),.ext_rsp(device_mem_rsp));

  always_comb begin
    cpl_wvalid=1'b0;cpl_wdata='0;t_cpl_ready=1'b0;s_cpl_ready=1'b0;
    if(t_cpl_valid)begin cpl_wvalid=1'b1;cpl_wdata=t_cpl;t_cpl_ready=cpl_wready;end
    else if(s_cpl_valid)begin cpl_wvalid=1'b1;cpl_wdata=s_cpl;s_cpl_ready=cpl_wready;end
  end
  assign irq=irq_enable&&qm_irq;
  assign fault=qm_fault||t_fault||s_fault||hbm_fatal_host;
endmodule
