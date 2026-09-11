#!/usr/bin/env python3
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]
a=json.loads((R/'abi/fusionx_d2_abi.json').read_text())
regs=a['registers']
h=['#ifndef FUSIONX_D2_ABI_H','#define FUSIONX_D2_ABI_H','#include <stdint.h>',f'#define FX_D2_ABI_VERSION {a["abi_version"]}u',f'#define FX_D2_DESCRIPTOR_BYTES {a["descriptor_bytes"]}u',f'#define FX_D2_COMPLETION_BYTES {a["completion_bytes"]}u']
for n,o in regs.items(): h.append(f'#define FX_D2_REG_{n} 0x{o:04x}u')
h += ['typedef struct __attribute__((packed)) {','  uint8_t opcode, dtype, flags, reserved0;','  uint16_t tag, queue_id;','  uint32_t m, n, k, bytes;','  uint64_t src0, src1, src2, dst;','  uint16_t scale_q2_14, out_scale_q2_14;','  uint8_t out_shift, reserved1;','  uint16_t aux;','} fx_d2_desc_t;','typedef struct __attribute__((packed)) {','  uint16_t tag, queue_id;','  uint8_t status, engine;','  uint16_t reserved0;','  uint32_t cycles, bytes_moved;','  uint64_t user_cookie;','  uint64_t reserved1;','} fx_d2_cpl_t;','_Static_assert(sizeof(fx_d2_desc_t)==64, "descriptor ABI");','_Static_assert(sizeof(fx_d2_cpl_t)==32, "completion ABI");','#endif']
(R/'software/include/fusionx_d2_abi.h').write_text('\n'.join(h)+'\n')
sv=['`timescale 1ns/1ps','package fx_d2_abi_pkg;',f'  localparam int FX_D2_ABI_VERSION={a["abi_version"]};',f'  localparam int FX_D2_DESCRIPTOR_BYTES={a["descriptor_bytes"]};',f'  localparam int FX_D2_COMPLETION_BYTES={a["completion_bytes"]};']
for n,o in regs.items(): sv.append(f"  localparam logic [15:0] FX_D2_REG_{n}=16'h{o:04x};")
sv += ['endpackage']
(R/'rtl/host/fx_d2_abi_pkg.sv').write_text('\n'.join(sv)+'\n')
