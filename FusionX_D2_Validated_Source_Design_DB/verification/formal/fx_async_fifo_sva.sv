`timescale 1ns/1ps
module fx_async_fifo_sva(input logic wclk,wrst_n,wvalid,wready,input logic rclk,rrst_n,rvalid,rready);
 property p_no_write_when_full;@(posedge wclk)disable iff(!wrst_n)!wready|->!(wvalid&&wready);endproperty
 property p_no_read_when_empty;@(posedge rclk)disable iff(!rrst_n)!rvalid|->!(rvalid&&rready);endproperty
 a_no_write_when_full:assert property(p_no_write_when_full);a_no_read_when_empty:assert property(p_no_read_when_empty);
endmodule
