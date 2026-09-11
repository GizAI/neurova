#include "fusionx_d2_abi.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef struct {volatile uint8_t *bar0;fx_d2_desc_t *sq;fx_d2_cpl_t *cq;uint32_t sq_mask,cq_mask,sq_tail,cq_head;} fx_d2_queue_t;
int fx_d2_queue_init(fx_d2_queue_t*,volatile void*,fx_d2_desc_t*,uint32_t,fx_d2_cpl_t*,uint32_t);
int fx_d2_submit(fx_d2_queue_t*,const fx_d2_desc_t*);int fx_d2_poll(fx_d2_queue_t*,fx_d2_cpl_t*);
int main(void){uint8_t bar[256]={0};fx_d2_desc_t sq[8]={0};fx_d2_cpl_t cq[8]={0},out={0};fx_d2_queue_t q;assert(fx_d2_queue_init(&q,bar,sq,8,cq,8)==0);fx_d2_desc_t d={.opcode=0x10,.tag=7};assert(fx_d2_submit(&q,&d)==0);assert(sq[0].tag==7);cq[0].tag=7;*(uint32_t*)(bar+FX_D2_REG_CQ_TAIL)=1;assert(fx_d2_poll(&q,&out)==1&&out.tag==7);puts("PASS runtime ABI");}
