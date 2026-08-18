#!/usr/bin/env python3
import math, struct, sys
from pathlib import Path
import numpy as np

MAGIC=b'AURALI01'

def align64(f):
    p=f.tell(); pad=(-p)&63
    if pad: f.write(b'\0'*pad)

def q8_matrix(f, a):
    a=np.asarray(a,dtype=np.float32)
    outd, ind=a.shape
    stride=(ind+4+63)&~63
    for row in a:
        m=float(np.max(np.abs(row)))
        s=m/127.0 if m>0 else 1.0
        q=np.clip(np.rint(row/s),-127,127).astype(np.int8)
        f.write(struct.pack('<f',s)); f.write(q.tobytes())
        f.write(b'\0'*(stride-4-ind))

def f32_vec(f,a):
    f.write(np.asarray(a,dtype='<f4').tobytes()); align64(f)

def main(path):
    rng=np.random.default_rng(7)
    dim=32; hidden=64; layers=2; heads=4; kv_heads=4; vocab=64; max_seq=64
    hd=dim//heads
    def mat(o,i,scale=0.08): return rng.normal(0,scale,(o,i)).astype(np.float32)
    emb=mat(vocab,dim,0.15)
    ls=[]
    for _ in range(layers):
        ls.append(dict(
            an=np.ones(dim,np.float32),
            wq=mat(dim,dim), wk=mat(kv_heads*hd,dim), wv=mat(kv_heads*hd,dim), wo=mat(dim,dim),
            fn=np.ones(dim,np.float32),
            w1=mat(hidden,dim), w3=mat(hidden,dim), w2=mat(dim,hidden)
        ))
    fn=np.ones(dim,np.float32)
    out=mat(vocab,dim,0.12)
    theta=10000.0
    pos=np.arange(max_seq,dtype=np.float32)[:,None]
    idx=np.arange(hd//2,dtype=np.float32)[None,:]
    inv=1.0/(theta**(2*idx/hd))
    ang=pos*inv
    cos=np.cos(ang).astype(np.float32)
    sin=np.sin(ang).astype(np.float32)
    p=Path(path)
    with p.open('wb') as f:
        hdr=bytearray(64)
        hdr[:8]=MAGIC
        struct.pack_into('<IIIIIIII',hdr,8,1,dim,hidden,layers,heads,kv_heads,vocab,max_seq)
        struct.pack_into('<f',hdr,40,theta)
        f.write(hdr); align64(f)
        q8_matrix(f,emb)
        for L in ls:
            f32_vec(f,L['an']); q8_matrix(f,L['wq']); q8_matrix(f,L['wk']); q8_matrix(f,L['wv']); q8_matrix(f,L['wo'])
            f32_vec(f,L['fn']); q8_matrix(f,L['w1']); q8_matrix(f,L['w3']); q8_matrix(f,L['w2'])
        f32_vec(f,fn); q8_matrix(f,out); align64(f)
        f.write(cos.astype('<f4').tobytes()); align64(f)
        f.write(sin.astype('<f4').tobytes())
    print(f'wrote {p} ({p.stat().st_size} bytes)')

if __name__=='__main__':
    main(sys.argv[1] if len(sys.argv)>1 else 'test.ali')
