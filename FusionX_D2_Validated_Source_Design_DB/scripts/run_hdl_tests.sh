#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)";cd "$ROOT";mkdir -p build reports
missing=0
for tool in iverilog vvp verilator yosys;do if ! command -v "$tool" >/dev/null 2>&1;then echo "BLOCKED_TOOL_MISSING=$tool";missing=1;fi;done
if [[ $missing -ne 0 ]];then printf 'BLOCKED_TOOL_MISSING\n' > reports/hdl_tests.status;exit 2;fi
COMMON=(rtl/host/fx_d2_abi_pkg.sv rtl/fx_d2_pkg.sv rtl/common/fx_sync_bits.sv rtl/common/fx_async_fifo.sv rtl/numeric/fx_lowbit_decode.sv rtl/numeric/fx_requantize.sv rtl/compute/fx_operand_unpacker.sv rtl/compute/fx_mxu_array.sv rtl/compute/fx_mxu_cluster.sv rtl/compute/fx_tensor_engine.sv rtl/stream/fx_vector_alu32.sv rtl/stream/fx_rmsnorm32.sv rtl/stream/fx_softmax32.sv rtl/stream/fx_rope32.sv rtl/stream/fx_state_update32.sv rtl/stream/fx_topk32.sv rtl/stream/fx_mtp_verify8.sv rtl/stream/fx_sparse_gather32.sv rtl/stream/fx_patchify3d.sv rtl/stream/fx_adaln32.sv rtl/stream/fx_stream_engine.sv rtl/memory/fx_scratchpad_banked.sv rtl/memory/fx_device_mem_router.sv rtl/host/fx_queue_csr.sv rtl/host/fx_queue_manager.sv rtl/top/fusionx_d2_core.sv)
run_tb(){ name="$1";shift;iverilog -g2012 -Wall -s "$name" -o "build/$name" "${COMMON[@]}" "tb/$name.sv" >"reports/${name}_compile.log" 2>&1;vvp "build/$name" >"reports/${name}_run.log" 2>&1;grep -q 'PASS' "reports/${name}_run.log"; }
run_tb tb_lowbit
run_tb tb_operand_unpacker
run_tb tb_mxu_64x64
run_tb tb_tensor_engine
run_tb tb_stream_primitives
run_tb tb_csr
run_tb tb_async_fifo
verilator --lint-only --sv --Wall -Wno-fatal --top-module fusionx_d2_core "${COMMON[@]}" >reports/verilator_lint.log 2>&1
yosys -p 'read_verilog -sv rtl/compute/fx_mxu_array.sv; hierarchy -top fx_mxu_array -chparam ROWS 8 -chparam COLS 8; proc; opt; memory; opt; stat; write_verilog build/fx_mxu_array_8x8_netlist.v' >reports/yosys_mxu.log 2>&1
grep -q 'Number of cells' reports/yosys_mxu.log
printf 'PASS\n' > reports/hdl_tests.status
echo 'FUSIONX_D2_HDL_TESTS_PASS'
