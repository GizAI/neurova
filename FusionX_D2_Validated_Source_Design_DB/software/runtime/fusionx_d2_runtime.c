#include "fusionx_d2_abi.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
  volatile uint8_t *bar0;
  fx_d2_desc_t *sq;
  fx_d2_cpl_t *cq;
  uint32_t sq_mask, cq_mask;
  uint32_t sq_tail, cq_head;
} fx_d2_queue_t;

static inline uint32_t rd32(volatile uint8_t *b, uint32_t o){return *(volatile uint32_t *)(b+o);}
static inline void wr32(volatile uint8_t *b, uint32_t o,uint32_t v){*(volatile uint32_t *)(b+o)=v;}

int fx_d2_queue_init(fx_d2_queue_t *q, volatile void *bar0, fx_d2_desc_t *sq, uint32_t sq_entries, fx_d2_cpl_t *cq, uint32_t cq_entries){
 if(!q||!bar0||!sq||!cq||sq_entries<2||cq_entries<2||(sq_entries&(sq_entries-1))||(cq_entries&(cq_entries-1)))return -1;
 memset(q,0,sizeof(*q));q->bar0=bar0;q->sq=sq;q->cq=cq;q->sq_mask=sq_entries-1;q->cq_mask=cq_entries-1;
 wr32(q->bar0,FX_D2_REG_SQ_SIZE,sq_entries);wr32(q->bar0,FX_D2_REG_CQ_SIZE,cq_entries);return 0;
}
int fx_d2_submit(fx_d2_queue_t *q,const fx_d2_desc_t *d){
 uint32_t head=rd32(q->bar0,FX_D2_REG_SQ_HEAD)&q->sq_mask;uint32_t next=(q->sq_tail+1)&q->sq_mask;if(next==head)return -2;
 q->sq[q->sq_tail]=*d;__sync_synchronize();q->sq_tail=next;wr32(q->bar0,FX_D2_REG_SQ_TAIL,q->sq_tail);wr32(q->bar0,FX_D2_REG_SQ_DOORBELL,1);return 0;
}
int fx_d2_poll(fx_d2_queue_t *q,fx_d2_cpl_t *out){
 uint32_t tail=rd32(q->bar0,FX_D2_REG_CQ_TAIL)&q->cq_mask;if(q->cq_head==tail)return 0;*out=q->cq[q->cq_head];q->cq_head=(q->cq_head+1)&q->cq_mask;wr32(q->bar0,FX_D2_REG_CQ_HEAD,q->cq_head);wr32(q->bar0,FX_D2_REG_CQ_DOORBELL,1);return 1;
}
