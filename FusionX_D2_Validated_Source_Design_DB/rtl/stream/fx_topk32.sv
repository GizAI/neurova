`timescale 1ns/1ps
module fx_topk32 #(parameter K=4)(input logic[511:0]x,output logic[K*16-1:0]values,output logic[K*5-1:0]indices);
 integer i,j,k;logic signed[15:0]v;logic signed[15:0]best[0:K-1];logic[4:0]idx[0:K-1];logic inserted;
 always_comb begin
  values='0;indices='0;
  for(j=0;j<K;j=j+1)begin best[j]=-32768;idx[j]=5'h1f;end
  for(i=0;i<32;i=i+1)begin
   v=$signed(x[i*16+:16]);inserted=0;
   for(j=0;j<K;j=j+1)begin
    if(!inserted&&((v>best[j])||((v==best[j])&&(i<idx[j]))))begin
     for(k=K-1;k>j;k=k-1)begin best[k]=best[k-1];idx[k]=idx[k-1];end
     best[j]=v;idx[j]=i[4:0];inserted=1;
    end
   end
  end
  for(j=0;j<K;j=j+1)begin values[j*16+:16]=best[j];indices[j*5+:5]=idx[j];end
 end
endmodule
