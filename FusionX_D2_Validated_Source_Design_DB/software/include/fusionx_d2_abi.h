#ifndef FUSIONX_D2_ABI_H
#define FUSIONX_D2_ABI_H
#include <stdint.h>
#define FX_D2_ABI_VERSION 2u
#define FX_D2_DESCRIPTOR_BYTES 64u
#define FX_D2_COMPLETION_BYTES 32u
#define FX_D2_REG_ID 0x0000u
#define FX_D2_REG_VERSION 0x0004u
#define FX_D2_REG_CONTROL 0x0008u
#define FX_D2_REG_STATUS 0x000cu
#define FX_D2_REG_IRQ_STATUS 0x0010u
#define FX_D2_REG_IRQ_MASK 0x0014u
#define FX_D2_REG_SQ_BASE_LO 0x0020u
#define FX_D2_REG_SQ_BASE_HI 0x0024u
#define FX_D2_REG_SQ_SIZE 0x0028u
#define FX_D2_REG_SQ_HEAD 0x002cu
#define FX_D2_REG_SQ_TAIL 0x0030u
#define FX_D2_REG_SQ_DOORBELL 0x0034u
#define FX_D2_REG_CQ_BASE_LO 0x0040u
#define FX_D2_REG_CQ_BASE_HI 0x0044u
#define FX_D2_REG_CQ_SIZE 0x0048u
#define FX_D2_REG_CQ_HEAD 0x004cu
#define FX_D2_REG_CQ_TAIL 0x0050u
#define FX_D2_REG_CQ_DOORBELL 0x0054u
#define FX_D2_REG_RAS_STATUS 0x0060u
#define FX_D2_REG_THERMAL_STATUS 0x0064u
typedef struct __attribute__((packed)) {
  uint8_t opcode, dtype, flags, reserved0;
  uint16_t tag, queue_id;
  uint32_t m, n, k, bytes;
  uint64_t src0, src1, src2, dst;
  uint16_t scale_q2_14, out_scale_q2_14;
  uint8_t out_shift, reserved1;
  uint16_t aux;
} fx_d2_desc_t;
typedef struct __attribute__((packed)) {
  uint16_t tag, queue_id;
  uint8_t status, engine;
  uint16_t reserved0;
  uint32_t cycles, bytes_moved;
  uint64_t user_cookie;
  uint64_t reserved1;
} fx_d2_cpl_t;
_Static_assert(sizeof(fx_d2_desc_t)==64, "descriptor ABI");
_Static_assert(sizeof(fx_d2_cpl_t)==32, "completion ABI");
#endif
