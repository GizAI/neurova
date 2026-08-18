#!/usr/bin/env python3
"""Convenience client: tokenize/decode with a local HF tokenizer, inference stays in AURORA asm."""
import argparse, json, urllib.request
from pathlib import Path
from transformers import AutoTokenizer

ap=argparse.ArgumentParser()
ap.add_argument('model_dir')
ap.add_argument('prompt')
ap.add_argument('--url',default='http://127.0.0.1:8080/v1/token-completions')
ap.add_argument('--max-tokens',type=int,default=32)
ap.add_argument('--chat',action='store_true',help='render the checkpoint chat template before inference')
ap.add_argument('--show-special',action='store_true',help='include tokenizer special tokens in decoded output')
a=ap.parse_args()
tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
if a.chat:
    template_path=Path(a.model_dir)/'chat_template.jinja'
    template=template_path.read_text() if template_path.exists() else None
    ids=tok.apply_chat_template(
        [{'role':'user','content':a.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        chat_template=template,
    )
else:
    ids=tok.encode(a.prompt,add_special_tokens=True)
if hasattr(ids,'input_ids'):
    ids=ids.input_ids
if hasattr(ids,'tolist'):
    ids=ids.tolist()
if ids and isinstance(ids[0],list):
    if len(ids)!=1: raise ValueError('expected one tokenized prompt')
    ids=ids[0]
body=json.dumps({'tokens':ids,'max_tokens':a.max_tokens}).encode()
req=urllib.request.Request(a.url,data=body,headers={'content-type':'application/json'})
with urllib.request.urlopen(req) as r:
    data=json.load(r)
out=data['tokens']
print(tok.decode(out,skip_special_tokens=not a.show_special))
print('token_ids=',out)
