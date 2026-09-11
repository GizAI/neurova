#!/usr/bin/env python3
from pathlib import Path
import json,re,sys
R=Path(__file__).resolve().parents[1]
a=json.loads((R/'abi/fusionx_d2_abi.json').read_text());h=(R/'software/include/fusionx_d2_abi.h').read_text();sv=(R/'rtl/host/fx_d2_abi_pkg.sv').read_text()
for n,o in a['registers'].items():
    if f'#define FX_D2_REG_{n} 0x{o:04x}u' not in h:raise SystemExit(f'C ABI mismatch {n}')
    if f"FX_D2_REG_{n}=16'h{o:04x}" not in sv:raise SystemExit(f'SV ABI mismatch {n}')
if '_Static_assert(sizeof(fx_d2_desc_t)==64' not in h:raise SystemExit('descriptor size assert missing')
print(f"PASS ABI consistency registers={len(a['registers'])}")
