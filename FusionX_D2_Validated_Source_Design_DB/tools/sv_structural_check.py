#!/usr/bin/env python3
"""Token-based structural sanity checker.

This is not an HDL compiler and is never reported as one. It checks balanced
SystemVerilog block structure, unique definitions, and unresolved module
instantiations before invoking external HDL tools.
"""
from __future__ import annotations
import re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
files=[ROOT/p.strip() for p in (ROOT/'filelists/rtl.f').read_text().splitlines() if p.strip()]

def tokens(text:str):
    text=re.sub(r'/\*.*?\*/',' ',text,flags=re.S)
    text=re.sub(r'//.*',' ',text)
    text=re.sub(r'"(?:\\.|[^"\\])*"','""',text)
    return re.findall(r"[A-Za-z_$][A-Za-z0-9_$]*|::|<=|>=|==|!=|&&|\|\||\+\:|\-\:|\S",text)

definitions={};all_tokens=[]
for f in files:
    if not f.exists():raise SystemExit(f'MISSING {f.relative_to(ROOT)}')
    t=tokens(f.read_text());all_tokens.append((f,t))
    for i,x in enumerate(t[:-1]):
        if x in {'module','package','interface'}:
            name=t[i+1]
            if name in definitions:raise SystemExit(f'DUPLICATE {name}: {f} and {definitions[name]}')
            definitions[name]=f
pairs={'module':'endmodule','package':'endpackage','interface':'endinterface','begin':'end','case':'endcase','casex':'endcase','casez':'endcase','function':'endfunction','task':'endtask','generate':'endgenerate','class':'endclass','fork':'join'}
for f,t in all_tokens:
    stack=[]
    for tok in t:
        if tok in pairs:stack.append((tok,pairs[tok]))
        elif tok.startswith('end') or tok=='join':
            if not stack or stack[-1][1]!=tok:
                # named blocks and end labels are tokenized after the end; mismatch is real here.
                raise SystemExit(f'UNBALANCED {f.relative_to(ROOT)} token={tok} stack={stack[-4:]}')
            stack.pop()
    if stack:raise SystemExit(f'UNCLOSED {f.relative_to(ROOT)} {stack[-8:]}')
# Approximate instantiation discovery: known module name followed by optional # and instance identifier.
joined='\n'.join(f.read_text() for f in files)
unresolved=[]
for name in sorted(set(re.findall(r'\b(fx_[A-Za-z0-9_]+|fusionx_[A-Za-z0-9_]+)\b',joined))):
    if name not in definitions and name not in {'fx_desc_t','fx_cpl_t','fx_mem_req_t','fx_mem_rsp_t','fx_opcode_e','fx_dtype_e'}:
        # Package functions/constants are allowed.
        if re.search(r'\b'+re.escape(name)+r'\s*(?:#\s*\(|[A-Za-z_$][A-Za-z0-9_$]*\s*\()',joined):unresolved.append(name)
if unresolved:raise SystemExit('UNRESOLVED_MODULES '+','.join(unresolved))
print(f'PASS structural tokens files={len(files)} definitions={len(definitions)}')
print('NOTE: structural check is not HDL compile/elaboration')
