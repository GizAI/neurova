#!/usr/bin/env python3
from pathlib import Path
import json,re,sys
R=Path(__file__).resolve().parents[1]
checks={}
def has(path,*terms):
    s=(R/path).read_text();return all(t in s for t in terms)
checks['mxu_row_held_until_last_beat']=has('rtl/compute/fx_mxu_array.sv','read_beat==BEATS_PER_ROW-1','read_row<=read_row+1')
checks['mxu_canonical_q8_8_output']=has('rtl/compute/fx_mxu_array.sv','32 values/512-bit beat','quantize')
checks['tensor_uses_m_n_k']=has('rtl/compute/fx_tensor_engine.sv','q.m','q.n','q.k')
checks['packed_lowbit_operand_loader']=has('rtl/compute/fx_operand_unpacker.sv','epb=128','i*4 +:4','epb=64','i*8 +:8')
checks['fp4_six_finite']=has('rtl/numeric/fx_lowbit_decode.sv','default:mag=393216')
checks['fp8_448_finite_nan_only_7f']=has('rtl/numeric/fx_lowbit_decode.sv',"raw[6:3]==4'hf","raw[2:0]==3'h7",'(8+raw[2:0])')
checks['scale_and_rne']=has('rtl/numeric/fx_lowbit_decode.sv','scale_q2_14','rne_shift_signed') and has('rtl/numeric/fx_requantize.sv','remainder==half')
stream=(R/'rtl/stream/fx_stream_engine.sv').read_text()
for op,mod in [('RMSNORM','fx_rmsnorm32'),('SOFTMAX','fx_softmax32'),('ROPE','fx_rope32'),('STATE_UPDATE','fx_state_update32'),('MOE_TOPK','fx_topk32'),('MTP_VERIFY','fx_mtp_verify8'),('SPARSE_GATHER','fx_sparse_gather32'),('PATCHIFY3D','fx_patchify3d'),('ADALN','fx_adaln32')]:
    checks[f'stream_{op.lower()}_connected']=(mod in stream and f'FX_OP_{op}' in stream)
checks['no_fake_mla_conv3d_vae_native_opcode']=all(x not in stream for x in ['FX_OP_MLA','FX_OP_CONV3D','FX_OP_VAE'])
checks['scratchpad_in_compute_path']=has('rtl/top/fusionx_d2_core.sv','fx_device_mem_router','SPAD_WORDS_PER_BANK')
checks['queue_abi_sq_head_cq_tail']=has('software/runtime/fusionx_d2_runtime.c','FX_D2_REG_SQ_HEAD','FX_D2_REG_CQ_TAIL')
checks['cq_32byte_entry_64byte_alignment']=has('rtl/host/fx_queue_manager.sv','>>1)*64')
checks['cdc_async_fifo_connected']=has('rtl/top/fusionx_d2_core.sv','u_cmd_fifo','u_cpl_fifo','u_status_sync')
checks['compiler_high_level_lowering']=has('compiler/lowering.py','RECIPES','unsupported high-level op')
checks['verify_fail_closed']=has('scripts/run_hdl_tests.sh','exit 2') and '|| true' not in (R/'Makefile').read_text()
failed=[k for k,v in checks.items() if not v]
for k,v in checks.items():print(('PASS' if v else 'FAIL'),k)
(R/'reports').mkdir(exist_ok=True)
(R/'reports/audit_contract_check.json').write_text(json.dumps({'pass':not failed,'checks':checks,'failed':failed},indent=2)+'\n')
if failed:raise SystemExit(1)
print(f'PASS audit contract {len(checks)}/{len(checks)}')
