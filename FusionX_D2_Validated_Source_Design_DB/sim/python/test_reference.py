#!/usr/bin/env python3
import json,sys
from pathlib import Path
import numpy as np
from reference import *

ROOT=Path(__file__).resolve().parents[2]

def test_fp4():
    expected=[0,.5,1,1.5,2,3,4,6]
    for sign in (0,1):
        for i,v in enumerate(expected):
            q,finite=decode_q16_16(FP4_E2M1,i|(sign<<3));assert finite
            assert q==round((-v if sign else v)*65536),(sign,i,q)

def test_fp8():
    q,finite=decode_q16_16(FP8_E4M3FN,0x7E);assert finite and q==448*65536
    q,finite=decode_q16_16(FP8_E4M3FN,0xFE);assert finite and q==-448*65536
    q,finite=decode_q16_16(FP8_E4M3FN,0x7F);assert not finite
    q,finite=decode_q16_16(FP8_E4M3FN,0x01);assert finite and q==128

def test_packing():
    raw=list(range(16))*8;blob=pack_values(FP4_E2M1,raw);assert len(blob)==64
    decoded=unpack_values(FP4_E2M1,blob,128)
    assert decoded[7]==1536 and decoded[15]==-1536
    raw8=list(range(64));assert len(pack_values(INT8,raw8))==64
    raw16=list(range(32));assert len(pack_values(INT16,raw16))==64

def test_rne():
    assert rne_shift_signed(10,2)==2  # 2.5 -> even 2
    assert rne_shift_signed(14,2)==4  # 3.5 -> even 4
    assert rne_shift_signed(-10,2)==-2
    assert rne_shift_signed(-14,2)==-4

def test_gemm_64():
    rng=np.random.default_rng(7)
    a=rng.integers(-32,33,size=(64,3),dtype=np.int16)
    b=rng.integers(-32,33,size=(3,64),dtype=np.int16)
    y=gemm_q8_8(a,b,out_shift=0)
    beats=list(result_beats(y))
    assert len(beats)==128
    for idx,(row,beat,chunk,last) in enumerate(beats):
        assert row==idx//2 and beat==idx%2
        assert chunk==y[row,beat*32:(beat+1)*32].tolist()
    assert beats[-1][-1]

def test_edge_tiles():
    tiles=tiled_shapes(70,73,5)
    assert tiles==[(0,0,5,64,64),(0,64,5,64,9),(64,0,5,6,64),(64,64,5,6,9)]

def test_stream():
    x=list(range(-16,16));g=[256]*32
    assert rmsnorm_q8_8(x,g)!=x
    sm=softmax_q0_16(x);assert 65500<=sum(sm)<=65570 and sm[-1]>sm[0]
    cs=[]
    for _ in range(16):cs += [256,0]
    assert rope_q8_8(x,cs)==x
    st=state_update_q8_8([256]*32,[0]*32,[16384]*32);assert st==[128]*32
    tk=topk([1,7,7,3],2);assert tk==[(1,7),(2,7)]
    assert mtp_accept([1,2,3,4],[1,2,9,4])==2
    assert sparse_gather([10,20,30],[2,0,1])==[30,10,20]
    data=bytes(range(64));assert patchify3d_beat(patchify3d_beat(data))==data
    assert adaln_q8_8([256],[256],[4])==[260]

def test_abi():
    abi=json.loads((ROOT/'abi/fusionx_d2_abi.json').read_text())
    assert abi['descriptor_bytes']==64 and abi['completion_bytes']==32
    csr=abi['registers'];assert csr['SQ_HEAD']==0x2c and csr['CQ_TAIL']==0x50

def main():
    tests=[test_fp4,test_fp8,test_packing,test_rne,test_gemm_64,test_edge_tiles,test_stream,test_abi]
    for t in tests:t();print('PASS',t.__name__)
    print(f'PASS {len(tests)}/{len(tests)} FusionX D2 reference tests')
if __name__=='__main__':main()
