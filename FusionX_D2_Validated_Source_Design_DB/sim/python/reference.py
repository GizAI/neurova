"""Executable numerical reference for the FusionX D2 validated primitive set."""
from __future__ import annotations
import math, struct
from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np

INT4=0; FP4_E2M1=1; INT8=2; FP8_E4M3FN=3; INT16=4; BF16=5


def sign_extend(v:int,bits:int)->int:
    v&=(1<<bits)-1
    return v-(1<<bits) if v&(1<<(bits-1)) else v


def rne_shift_signed(x:int,shift:int)->int:
    if shift<=0:return x
    neg=x<0;mag=-x if neg else x
    q,rem=divmod(mag,1<<shift);half=1<<(shift-1)
    if rem>half or (rem==half and (q&1)):q+=1
    return -q if neg else q


def decode_q16_16(dtype:int,raw:int)->tuple[int,bool]:
    raw&=0xFFFF
    if dtype==INT4:return sign_extend(raw,4)<<16,True
    if dtype==FP4_E2M1:
        lut=[0,32768,65536,98304,131072,196608,262144,393216]
        mag=lut[raw&7]
        return (-mag if raw&8 else mag),True
    if dtype==INT8:return sign_extend(raw,8)<<16,True
    if dtype==FP8_E4M3FN:
        v=raw&0xFF;sign=-1 if v&0x80 else 1;exp=(v>>3)&0xF;mant=v&7
        if exp==15 and mant==7:return 0,False
        mag=(mant<<7) if exp==0 else ((8+mant)<<(exp+6))
        return sign*mag,True
    if dtype==INT16:return sign_extend(raw,16)<<16,True
    if dtype==BF16:
        sign=-1 if raw&0x8000 else 1;exp=(raw>>7)&0xFF;mant=raw&0x7F
        if exp==0:return 0,True
        if exp==0xFF:return 0,False
        value=(1.0+mant/128.0)*2.0**(exp-127)
        return int(round(sign*value*65536)),True
    return 0,False


def decode_q8_8(dtype:int,raw:int,scale_q2_14:int=0x4000)->tuple[int,bool]:
    q16,finite=decode_q16_16(dtype,raw)
    scaled=rne_shift_signed(q16*scale_q2_14,14)
    q8=rne_shift_signed(scaled,8)
    return max(-32768,min(32767,q8)),finite


def pack_values(dtype:int,raw_values:Sequence[int])->bytes:
    out=bytearray()
    if dtype in (INT4,FP4_E2M1):
        for i in range(0,len(raw_values),2):
            lo=raw_values[i]&0xF;hi=(raw_values[i+1]&0xF) if i+1<len(raw_values) else 0
            out.append(lo|(hi<<4))
    elif dtype in (INT8,FP8_E4M3FN):out.extend(v&0xFF for v in raw_values)
    else:
        for v in raw_values:out.extend(struct.pack('<H',v&0xFFFF))
    return bytes(out)


def unpack_values(dtype:int,data:bytes,count:int,scale_q2_14:int=0x4000)->list[int]:
    raw=[]
    if dtype in (INT4,FP4_E2M1):
        for b in data:
            raw.extend([b&0xF,(b>>4)&0xF])
    elif dtype in (INT8,FP8_E4M3FN):raw=list(data)
    else:
        raw=[struct.unpack_from('<H',data,i)[0] for i in range(0,len(data)-1,2)]
    return [decode_q8_8(dtype,v,scale_q2_14)[0] for v in raw[:count]]


def requantize(acc:int,scale_q2_14:int=0x4000,right_shift:int=0)->int:
    q=rne_shift_signed(acc*scale_q2_14,14+right_shift)
    return max(-32768,min(32767,q))


def gemm_q8_8(a:np.ndarray,b:np.ndarray,out_scale_q2_14:int=0x4000,out_shift:int=8)->np.ndarray:
    acc=a.astype(np.int64)@b.astype(np.int64)
    f=np.vectorize(lambda x:requantize(int(x),out_scale_q2_14,out_shift),otypes=[np.int16])
    return f(acc)


def result_beats(matrix:np.ndarray,beat_elems:int=32):
    rows,cols=matrix.shape
    for r in range(rows):
        for beat in range((cols+beat_elems-1)//beat_elems):
            chunk=matrix[r,beat*beat_elems:(beat+1)*beat_elems].tolist()
            yield r,beat,chunk,(r==rows-1 and (beat+1)*beat_elems>=cols)


def tiled_shapes(m:int,n:int,k:int,tm:int=64,tn:int=64):
    return [(mi,ni,k,min(tm,m-mi),min(tn,n-ni)) for mi in range(0,m,tm) for ni in range(0,n,tn)]


def rmsnorm_q8_8(x:Sequence[int],gamma:Sequence[int],epsilon:float=1e-5)->list[int]:
    xf=np.asarray(x,dtype=np.float64)/256.0;gf=np.asarray(gamma,dtype=np.float64)/256.0
    y=xf/np.sqrt(np.mean(xf*xf)+epsilon)*gf
    return np.clip(np.rint(y*256),-32768,32767).astype(np.int16).astype(int).tolist()


def softmax_q0_16(x:Sequence[int])->list[int]:
    xf=np.asarray(x,dtype=np.float64)/256.0;z=np.exp(xf-xf.max());z/=z.sum()
    return np.rint(z*65535).astype(np.int64).tolist()


def rope_q8_8(x:Sequence[int],cos_sin:Sequence[int])->list[int]:
    out=[]
    for i in range(0,len(x),2):
        c=cos_sin[i]/256.0;s=cos_sin[i+1]/256.0;a=x[i]/256.0;b=x[i+1]/256.0
        out += [round((a*c-b*s)*256),round((a*s+b*c)*256)]
    return [max(-32768,min(32767,int(v))) for v in out]


def state_update_q8_8(x:Sequence[int],state:Sequence[int],gate_q1_15:Sequence[int])->list[int]:
    return [max(-32768,min(32767,s+((xv-s)*g>>15))) for xv,s,g in zip(x,state,gate_q1_15)]


def topk(values:Sequence[int],k:int=4):
    return sorted(enumerate(values),key=lambda p:(-p[1],p[0]))[:k]


def mtp_accept(candidate:Sequence[int],reference:Sequence[int])->int:
    n=0
    for a,b in zip(candidate,reference):
        if a!=b:break
        n+=1
    return n


def sparse_gather(table:Sequence[int],indices:Sequence[int])->list[int]:
    return [table[i%len(table)] for i in indices]


def patchify3d_beat(data:bytes)->bytes:
    if len(data)!=64:raise ValueError
    out=bytearray(64)
    for i in range(64):out[i]=data[((i&7)<<3)|(i>>3)]
    return bytes(out)


def adaln_q8_8(x:Sequence[int],scale:Sequence[int],bias:Sequence[int])->list[int]:
    return [max(-32768,min(32767,rne_shift_signed(xv*sv,8)+bv)) for xv,sv,bv in zip(x,scale,bias)]
