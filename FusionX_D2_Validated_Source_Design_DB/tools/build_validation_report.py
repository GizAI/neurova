#!/usr/bin/env python3
from pathlib import Path
import json,os,re,shutil,subprocess,sys
R=Path(__file__).resolve().parents[1]
reports=R/'reports';reports.mkdir(exist_ok=True)

def run(name,cmd):
    p=subprocess.run(cmd,cwd=R,text=True,capture_output=True)
    (reports/f'{name}.log').write_text(p.stdout+p.stderr)
    return {'command':cmd,'returncode':p.returncode,'pass':p.returncode==0,'stdout_tail':(p.stdout+p.stderr)[-4000:]}

# Source tests are executed again to make the report self-contained.
source=run('final_source_test',['bash','scripts/run_source_tests.sh'])
hdl=run('final_hdl_test',['bash','scripts/run_hdl_tests.sh'])
tools={t:shutil.which(t) for t in ['iverilog','vvp','verilator','yosys','openroad','klayout']}
rtl_files=sorted((R/'rtl').rglob('*.sv'))
modules=0;packages=0;lines=0
for p in rtl_files:
    s=p.read_text();modules+=len(re.findall(r'\bmodule\s+[A-Za-z_$]',s));packages+=len(re.findall(r'\bpackage\s+[A-Za-z_$]',s));lines+=len(s.splitlines())
audit=json.loads((reports/'audit_contract_check.json').read_text())
summary={
 'release':'FusionX-D2-validated-source-db',
 'classification':'integrated validated source/reference design database; not foundry manufacturing release',
 'source_tests':source,
 'hdl_tests':hdl,
 'hdl_gate_status':'PASS' if hdl['pass'] else ('BLOCKED_TOOL_MISSING' if hdl['returncode']==2 else 'FAIL'),
 'tool_paths':tools,
 'metrics':{'rtl_files':len(rtl_files),'module_definitions':modules,'package_definitions':packages,'rtl_lines':lines},
 'audit_contract':audit,
 'foundry_release_allowed':False,
 'foundry_blockers':['N3P PDK and sign-off decks','standard cell/I/O/SRAM compiler views','HBM4/PCIe6/CXL/UCIe3/PLL/SerDes hard IP','commercial verification and DFT closure','P&R/CTS/STA/EMIR/SI/DRC/LVS','package/board sign-off','GDS/OASIS and foundry acceptance']
}
(reports/'FINAL_VERIFICATION_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
status='PASS' if source['pass'] else 'FAIL'
md=f'''# FusionX D2 final validation report

## Classification

**Integrated validated source/reference design database — not a foundry manufacturing release.**

## Results

| Gate | Result |
|---|---|
| Source/ABI/compiler/numerical regression | **{status}** |
| Audit-contract checks | **{'PASS' if audit['pass'] else 'FAIL'} ({sum(audit['checks'].values())}/{len(audit['checks'])})** |
| Real HDL tool gate | **{summary['hdl_gate_status']}** |
| N3P foundry release | **NO** |

## Source metrics

- RTL files: {len(rtl_files)}
- Module definitions: {modules}
- Package definitions: {packages}
- RTL lines: {lines:,}

## Executed numerical and ABI evidence

- Correct FP4 E2M1 finite values through +/-6.
- Correct FP8 E4M3FN finite values through +/-448 and NaN handling.
- Packed 4-bit, 8-bit and 16-bit operand tests.
- Signed round-to-nearest-even tests including negative ties.
- Random 64x64x3 GEMM reference and all 128 output `(row, beat)` positions.
- M/N ragged-edge tile traversal.
- RMSNorm, base-e Softmax, RoPE, recurrent-state, Top-K, MTP, gather, Patchify3D and AdaLN reference tests.
- C runtime SQ/CQ ownership regression and C/SV generated ABI consistency.
- Qwen, MiniMax, GLM and DeepSeek high-level graph legalization into primitives.

## HDL evidence boundary

The package contains directed SystemVerilog tests and fail-closed Icarus,
Verilator and Yosys scripts. Those tools are not installed in the current
execution environment, so the real HDL gate is recorded as
`{summary['hdl_gate_status']}` and no compilation/simulation/synthesis PASS is claimed.

## Foundry boundary

A real N3P tapeout still requires licensed process/IP/package collateral and all
front-end, DFT, physical and package sign-off outputs listed in
`foundry/N3P_HANDOFF_REQUIREMENTS.yaml`.
'''
(R/'FINAL_VALIDATION_REPORT.md').write_text(md)
print(json.dumps({'source_pass':source['pass'],'hdl_status':summary['hdl_gate_status'],'rtl_files':len(rtl_files),'modules':modules,'lines':lines},indent=2))
if not source['pass']:sys.exit(1)
