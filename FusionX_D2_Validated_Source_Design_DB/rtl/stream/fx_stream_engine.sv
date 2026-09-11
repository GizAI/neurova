`timescale 1ns/1ps
module fx_stream_engine #(
  parameter int ENGINE_ID=8'h20
)(
  input logic clk,input logic rst_n,input logic enable,
  input logic desc_valid,output logic desc_ready,input fx_d2_pkg::fx_desc_t desc,
  output logic mem_req_valid,input logic mem_req_ready,output fx_d2_pkg::fx_mem_req_t mem_req,
  input logic mem_rsp_valid,output logic mem_rsp_ready,input fx_d2_pkg::fx_mem_rsp_t mem_rsp,
  output logic cpl_valid,input logic cpl_ready,output fx_d2_pkg::fx_cpl_t cpl,
  output logic busy,output logic fault
);
 import fx_d2_pkg::*;
 typedef enum logic[3:0]{IDLE,R0Q,R0W,R1Q,R1W,R2Q,R2W,EXEC,WQ,WW,CPL,ERR}state_e;
 state_e state;
 fx_desc_t q;
 logic[31:0]offset,cycles,bytes_done;
 logic[511:0]x0,x1,x2,y,vout,rmsout,smout,ropeout,stateout,gatherout,patchout,adalnout;
 logic[4*16-1:0]top_values;logic[4*5-1:0]top_indices;logic[3:0]mtp_accepted;
 logic needs1,needs2,supported;
 integer i;
 fx_vector_alu32 u_vec(.op(q.aux[1:0]),.a(x0),.b(x1),.scale_q2_14(q.scale_q2_14),.y(vout));
 fx_rmsnorm32 u_rms(.x(x0),.gamma(x1),.epsilon_q16_16(32'd1),.y(rmsout));
 fx_softmax32 u_sm(.x(x0),.y(smout));
 fx_rope32 u_rope(.x(x0),.cos_sin(x1),.y(ropeout));
 fx_state_update32 u_state(.x(x0),.state(x1),.gate(x2),.next_state(stateout));
 fx_sparse_gather32 u_gather(.table(x0),.indices(x1[159:0]),.y(gatherout));
 fx_patchify3d u_patch(.x(x0),.y(patchout));
 fx_adaln32 u_adaln(.x(x0),.scale(x1),.bias(x2),.y(adalnout));
 fx_topk32 #(.K(4)) u_topk(.x(x0),.values(top_values),.indices(top_indices));
 fx_mtp_verify8 u_mtp(.candidate(x0[127:0]),.reference(x1[127:0]),.accepted(mtp_accepted));

 always_comb begin
  needs1=(q.opcode==FX_OP_VECTOR)||(q.opcode==FX_OP_RMSNORM)||(q.opcode==FX_OP_ROPE)||
         (q.opcode==FX_OP_STATE_UPDATE)||(q.opcode==FX_OP_MTP_VERIFY)||
         (q.opcode==FX_OP_SPARSE_GATHER)||(q.opcode==FX_OP_ADALN);
  needs2=(q.opcode==FX_OP_STATE_UPDATE)||(q.opcode==FX_OP_ADALN);
  supported=(q.opcode==FX_OP_VECTOR)||(q.opcode==FX_OP_RMSNORM)||(q.opcode==FX_OP_SOFTMAX)||
            (q.opcode==FX_OP_ROPE)||(q.opcode==FX_OP_STATE_UPDATE)||(q.opcode==FX_OP_MOE_TOPK)||
            (q.opcode==FX_OP_MTP_VERIFY)||(q.opcode==FX_OP_SPARSE_GATHER)||
            (q.opcode==FX_OP_PATCHIFY3D)||(q.opcode==FX_OP_ADALN);
  y='0;
  case(q.opcode)
   FX_OP_VECTOR:y=vout;FX_OP_RMSNORM:y=rmsout;FX_OP_SOFTMAX:y=smout;
   FX_OP_ROPE:y=ropeout;FX_OP_STATE_UPDATE:y=stateout;FX_OP_SPARSE_GATHER:y=gatherout;
   FX_OP_PATCHIFY3D:y=patchout;FX_OP_ADALN:y=adalnout;
   FX_OP_MOE_TOPK:begin y[63:0]=top_values;y[83:64]=top_indices;end
   FX_OP_MTP_VERIFY:y[3:0]=mtp_accepted;
   default:y='0;
  endcase
 end
 assign desc_ready=(state==IDLE)&&enable;
 assign busy=(state!=IDLE);
 assign mem_rsp_ready=(state==R0W)||(state==R1W)||(state==R2W)||(state==WW);
 always_comb begin
  mem_req_valid=1'b0;mem_req='0;
  case(state)
   R0Q:begin mem_req_valid=1'b1;mem_req.id=16'h2000;mem_req.addr=q.src0+offset;end
   R1Q:begin mem_req_valid=1'b1;mem_req.id=16'h2001;mem_req.addr=q.src1+offset;end
   R2Q:begin mem_req_valid=1'b1;mem_req.id=16'h2002;mem_req.addr=q.src2+offset;end
   WQ:begin mem_req_valid=1'b1;mem_req.write=1'b1;mem_req.id=16'h2080;mem_req.addr=q.dst+offset;mem_req.data=y;mem_req.strb=(q.bytes-offset>=64)?64'hffff_ffff_ffff_ffff:((64'h1<<(q.bytes-offset))-1'b1);mem_req.last=(offset+64>=q.bytes);end
   default:begin end
  endcase
  cpl_valid=(state==CPL)||(state==ERR);cpl='0;cpl.tag=q.tag;cpl.queue_id=q.queue_id;
  cpl.status=(state==ERR)?8'hff:8'h00;cpl.engine=ENGINE_ID;cpl.cycles=cycles;cpl.bytes_moved=bytes_done;
  fault=(state==ERR);
 end
 always_ff@(posedge clk or negedge rst_n)begin
  if(!rst_n)begin state<=IDLE;q<='0;offset<=0;cycles<=0;bytes_done<=0;x0<=0;x1<=0;x2<=0;end
  else begin
   if(state!=IDLE)cycles<=cycles+1;
   case(state)
    IDLE:if(desc_valid&&desc_ready)begin q<=desc;offset<=0;cycles<=0;bytes_done<=0;x0<=0;x1<=0;x2<=0;state<=(desc.bytes!=0)?R0Q:ERR;end
    R0Q:if(mem_req_valid&&mem_req_ready)state<=R0W;
    R0W:if(mem_rsp_valid&&mem_rsp_ready)begin if(mem_rsp.poison)state<=ERR;else begin x0<=mem_rsp.data;if(needs1)state<=R1Q;else state<=EXEC;end end
    R1Q:if(mem_req_valid&&mem_req_ready)state<=R1W;
    R1W:if(mem_rsp_valid&&mem_rsp_ready)begin if(mem_rsp.poison)state<=ERR;else begin x1<=mem_rsp.data;if(needs2)state<=R2Q;else state<=EXEC;end end
    R2Q:if(mem_req_valid&&mem_req_ready)state<=R2W;
    R2W:if(mem_rsp_valid&&mem_rsp_ready)begin if(mem_rsp.poison)state<=ERR;else begin x2<=mem_rsp.data;state<=EXEC;end end
    EXEC:state<=supported?WQ:ERR;
    WQ:if(mem_req_valid&&mem_req_ready)state<=WW;
    WW:if(mem_rsp_valid&&mem_rsp_ready)begin if(mem_rsp.poison)state<=ERR;else begin bytes_done<=bytes_done+((q.bytes-offset>=64)?64:(q.bytes-offset));if(offset+64>=q.bytes)state<=CPL;else begin offset<=offset+64;state<=R0Q;end end end
    CPL:if(cpl_ready)state<=IDLE;
    ERR:if(cpl_ready)state<=IDLE;
    default:state<=ERR;
   endcase
  end
 end
endmodule
