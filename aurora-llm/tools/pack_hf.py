#!/usr/bin/env python3
"""Pack a local Hugging Face Llama/Mistral checkpoint into AURORA ALI v1.

Serving runtime stays assembly-only; this is offline tooling.
Dependencies: Python 3, numpy, ml_dtypes, safetensors.
"""
import argparse, json, math, struct
from pathlib import Path
import numpy as np
import ml_dtypes
from safetensors import safe_open

np.sctypeDict['bfloat16'] = ml_dtypes.bfloat16

MAGIC=b'AURALI01'

def align64(f):
    pad=(-f.tell()) & 63
    if pad: f.write(b'\0'*pad)

def row_stride(n): return (4+n+63)&~63

class Source:
    def __init__(self, model_dir: Path):
        self.root=model_dir
        idx=model_dir/'model.safetensors.index.json'
        self.map={}
        if idx.exists():
            data=json.loads(idx.read_text())
            self.map={k:model_dir/v for k,v in data['weight_map'].items()}
        else:
            files=sorted(model_dir.glob('*.safetensors'))
            if not files: raise SystemExit('no .safetensors weights found')
            for fp in files:
                with safe_open(fp,framework='np',device='cpu') as sf:
                    for k in sf.keys(): self.map[k]=fp
    def has(self,name): return name in self.map
    def shape(self,name):
        fp=self.map[name]
        with safe_open(fp,framework='np',device='cpu') as sf:
            return tuple(sf.get_slice(name).get_shape())
    def rows(self,name,chunk=128):
        fp=self.map[name]
        with safe_open(fp,framework='np',device='cpu') as sf:
            sl=sf.get_slice(name); shape=tuple(sl.get_shape())
            if len(shape)!=2: raise ValueError(f'{name}: expected matrix, got {shape}')
            for a in range(0,shape[0],chunk):
                b=min(a+chunk,shape[0])
                yield np.asarray(sl[a:b], dtype=np.float32)
    def vec(self,name):
        fp=self.map[name]
        with safe_open(fp,framework='np',device='cpu') as sf:
            return np.asarray(sf.get_tensor(name), dtype=np.float32)

def write_q8_from_name(f, src:Source, name, expected=None, progress=True):
    sh=src.shape(name)
    if expected and sh!=tuple(expected): raise ValueError(f'{name}: shape {sh}, expected {expected}')
    stride=row_stride(sh[1])
    if progress: print(f'  q8 {name} {sh} stride={stride}')
    for a in src.rows(name):
        m=np.max(np.abs(a),axis=1).astype(np.float32)
        scale=np.where(m>0,m/127.0,1.0).astype(np.float32)
        q=np.clip(np.rint(a/scale[:,None]),-127,127).astype(np.int8)
        out=np.zeros((a.shape[0],stride),dtype=np.uint8)
        out[:,:4]=scale.astype('<f4',copy=False).view(np.uint8).reshape(-1,4)
        out[:,4:4+sh[1]]=q.view(np.uint8)
        f.write(out.tobytes())

def write_q8_from_array(f, a):
    a=np.asarray(a,dtype=np.float32)
    stride=row_stride(a.shape[1])
    for off in range(0,a.shape[0],128):
        x=a[off:off+128]
        m=np.max(np.abs(x),axis=1).astype(np.float32)
        scale=np.where(m>0,m/127.0,1.0).astype(np.float32)
        q=np.clip(np.rint(x/scale[:,None]),-127,127).astype(np.int8)
        out=np.zeros((x.shape[0],stride),dtype=np.uint8)
        out[:,:4]=scale.astype('<f4',copy=False).view(np.uint8).reshape(-1,4)
        out[:,4:4+a.shape[1]]=q.view(np.uint8)
        f.write(out.tobytes())

def write_vec(f,a,expected=None):
    a=np.asarray(a,dtype='<f4').reshape(-1)
    if expected is not None and len(a)!=expected: raise ValueError(f'vector len {len(a)}, expected {expected}')
    f.write(a.tobytes()); align64(f)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('model_dir',type=Path)
    ap.add_argument('output',type=Path)
    ap.add_argument('--max-seq',type=int,default=4096,help='KV/RoPE context compiled into the image')
    args=ap.parse_args()
    cfg=json.loads((args.model_dir/'config.json').read_text())
    typ=cfg.get('model_type','')
    if typ not in {'llama','mistral'}:
        raise SystemExit(f'ALI v1 packer currently supports model_type llama/mistral, got {typ!r}')
    dim=int(cfg['hidden_size']); hidden=int(cfg['intermediate_size']); layers=int(cfg['num_hidden_layers'])
    heads=int(cfg['num_attention_heads']); kv_heads=int(cfg.get('num_key_value_heads',heads)); vocab=int(cfg['vocab_size'])
    default_hd=dim//heads
    if dim%heads and 'head_dim' not in cfg:
        raise SystemExit('hidden_size must divide num_attention_heads when head_dim is absent')
    hd=int(cfg.get('head_dim',default_hd)); q_dim=hd*heads; kv_dim=hd*kv_heads
    if hd<=0 or hd%2: raise SystemExit('head_dim must be a positive even integer')
    eos_cfg=cfg.get('eos_token_id',[])
    if isinstance(eos_cfg,int): eos_cfg=[eos_cfg]
    eos_ids=[int(x) for x in eos_cfg[:2]]
    eos_ids += [0]*(2-len(eos_ids))
    max_seq=min(int(args.max_seq),int(cfg.get('max_position_embeddings',args.max_seq)))
    theta=float(cfg.get('rope_theta',10000.0))
    eps=float(cfg.get('rms_norm_eps',1e-5))
    src=Source(args.model_dir)
    emb='model.embed_tokens.weight'
    if not src.has(emb): raise SystemExit(f'missing {emb}')
    with args.output.open('wb') as f:
        hdr=bytearray(64); hdr[:8]=MAGIC
        struct.pack_into('<IIIIIIII',hdr,8,1,dim,hidden,layers,heads,kv_heads,vocab,max_seq)
        struct.pack_into('<f',hdr,40,theta)
        struct.pack_into('<f',hdr,44,eps)
        struct.pack_into('<I',hdr,48,hd)
        struct.pack_into('<II',hdr,52,*eos_ids)
        f.write(hdr); align64(f)
        print('embedding')
        write_q8_from_name(f,src,emb,(vocab,dim))
        for i in range(layers):
            p=f'model.layers.{i}'
            print(f'layer {i+1}/{layers}')
            write_vec(f,src.vec(p+'.input_layernorm.weight'),dim)
            write_q8_from_name(f,src,p+'.self_attn.q_proj.weight',(q_dim,dim),False)
            write_q8_from_name(f,src,p+'.self_attn.k_proj.weight',(kv_dim,dim),False)
            write_q8_from_name(f,src,p+'.self_attn.v_proj.weight',(kv_dim,dim),False)
            write_q8_from_name(f,src,p+'.self_attn.o_proj.weight',(dim,q_dim),False)
            write_vec(f,src.vec(p+'.post_attention_layernorm.weight'),dim)
            write_q8_from_name(f,src,p+'.mlp.gate_proj.weight',(hidden,dim),False)
            write_q8_from_name(f,src,p+'.mlp.up_proj.weight',(hidden,dim),False)
            write_q8_from_name(f,src,p+'.mlp.down_proj.weight',(dim,hidden),False)
        print('final norm')
        write_vec(f,src.vec('model.norm.weight'),dim)
        print('lm head')
        if src.has('lm_head.weight'):
            write_q8_from_name(f,src,'lm_head.weight',(vocab,dim))
        else:
            print('  tied to embedding; packing embedding again')
            write_q8_from_name(f,src,emb,(vocab,dim),False)
        align64(f)
        print('RoPE tables')
        pos=np.arange(max_seq,dtype=np.float32)[:,None]
        idx=np.arange(hd//2,dtype=np.float32)[None,:]
        inv=(1.0/(theta**(2*idx/hd))).astype(np.float32)
        ang=pos*inv
        f.write(np.cos(ang).astype('<f4').tobytes()); align64(f)
        f.write(np.sin(ang).astype('<f4').tobytes())
    print(f'wrote {args.output} ({args.output.stat().st_size/1024/1024:.2f} MiB)')
    print(f'ALI: dim={dim} hidden={hidden} layers={layers} heads={heads} head_dim={hd} kv_heads={kv_heads} vocab={vocab} max_seq={max_seq}')

if __name__=='__main__': main()
