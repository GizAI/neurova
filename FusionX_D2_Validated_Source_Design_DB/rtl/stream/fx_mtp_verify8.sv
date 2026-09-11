`timescale 1ns/1ps
module fx_mtp_verify8(input logic[8*16-1:0]candidate,reference,output logic[3:0]accepted);
 integer i;logic stop;always_comb begin accepted=0;stop=0;for(i=0;i<8;i=i+1)begin if(!stop&&(candidate[i*16+:16]==reference[i*16+:16]))accepted=accepted+1;else stop=1;end end
endmodule
