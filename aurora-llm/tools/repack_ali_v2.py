#!/usr/bin/env python3
"""Losslessly repack AURORA ALI v1 Q8 matrices into v2 8x16 interleaved layout.

Embedding remains v1 row-major for token lookup. Transformer/LM-head matrices
preserve every original float32 scale and int8 weight bit, changing layout only.
"""
import argparse, mmap, os, struct
from pathlib import Path
import numpy as np

MAGIC=b'AURALI01'; HEADER=64

def align64_pos(x): return (x+63)&~63
def row_stride(n): return (4+n+63)&~63
def v2_block_stride(n):
    if n%16: raise ValueError(f'v2 requires input columns multiple of 16, got {n}')
    return 64+8*n

def copy_range(mm, src_off, n, out):
    out.write(memoryview(mm)[src_off:src_off+n]); return src_off+n

def repack_matrix(mm, src_off, rows, cols, out, label):
    if rows%8: raise ValueError(f'{label}: v2 requires output rows multiple of 8, got {rows}')
    if cols%16: raise ValueError(f'{label}: v2 requires input cols multiple of 16, got {cols}')
    stride=row_stride(cols); groups=rows//8; block=v2_block_stride(cols)
    src=np.frombuffer(mm,dtype=np.uint8,count=rows*stride,offset=src_off).reshape(rows,stride)
    dst=np.zeros((groups,block),dtype=np.uint8)
    dst[:,:32]=src[:,:4].reshape(groups,8,4).reshape(groups,32)
    w=src[:,4:4+cols].reshape(groups,8,cols//16,16)
    dst[:,64:]=w.transpose(0,2,1,3).reshape(groups,cols*8)
    dst.tofile(out)
    print(f'  {label:<12} {rows:6d}x{cols:<5d} {rows*stride/1048576:7.2f} -> {groups*block/1048576:7.2f} MiB')
    return src_off+rows*stride

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input',type=Path); ap.add_argument('output',type=Path); args=ap.parse_args()
    with args.input.open('rb') as fi, args.output.open('wb') as out:
        mm=mmap.mmap(fi.fileno(),0,access=mmap.ACCESS_READ)
        try:
            hdr=bytearray(mm[:HEADER])
            if hdr[:8]!=MAGIC: raise SystemExit('bad ALI magic')
            ver,dim,hidden,layers,heads,kv_heads,vocab,max_seq=struct.unpack_from('<IIIIIIII',hdr,8)
            if ver!=1: raise SystemExit(f'expected ALI v1, got v{ver}')
            hd=struct.unpack_from('<I',hdr,48)[0] or dim//heads; qdim=hd*heads; kvdim=hd*kv_heads
            struct.pack_into('<I',hdr,8,2); out.write(hdr)
            src=align64_pos(HEADER)
            # Embedding intentionally remains row-major for random token lookup.
            n=vocab*row_stride(dim); src=copy_range(mm,src,n,out); print(f'  embedding    row-major {n/1048576:.2f} MiB')
            for i in range(layers):
                print(f'layer {i+1}/{layers}')
                n=dim*4; src=copy_range(mm,src,n,out); src=align64_pos(src)
                while out.tell()%64: out.write(b'\0')
                src=repack_matrix(mm,src,qdim,dim,out,'q_proj')
                src=repack_matrix(mm,src,kvdim,dim,out,'k_proj')
                src=repack_matrix(mm,src,kvdim,dim,out,'v_proj')
                src=repack_matrix(mm,src,dim,qdim,out,'o_proj')
                n=dim*4; src=copy_range(mm,src,n,out); src=align64_pos(src)
                while out.tell()%64: out.write(b'\0')
                src=repack_matrix(mm,src,hidden,dim,out,'gate_proj')
                src=repack_matrix(mm,src,hidden,dim,out,'up_proj')
                src=repack_matrix(mm,src,dim,hidden,out,'down_proj')
            n=dim*4; src=copy_range(mm,src,n,out); src=align64_pos(src)
            while out.tell()%64: out.write(b'\0')
            src=repack_matrix(mm,src,vocab,dim,out,'lm_head')
            src=align64_pos(src)
            while out.tell()%64: out.write(b'\0')
            # RoPE tables and any trailing bytes are already FP32/raw and stay identical.
            copy_range(mm,src,len(mm)-src,out)
            print(f'wrote {args.output}: {len(mm)/1048576:.2f} -> {out.tell()/1048576:.2f} MiB')
            print(f'ALI v2 dim={dim} hidden={hidden} layers={layers} heads={heads} head_dim={hd} kv_heads={kv_heads} vocab={vocab} max_seq={max_seq}')
        finally: mm.close()
if __name__=='__main__': main()
