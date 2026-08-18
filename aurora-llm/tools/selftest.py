#!/usr/bin/env python3
import json, os, signal, socket, subprocess, sys, tempfile, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def request(port):
    body=b'{"tokens":[1,2,3],"max_tokens":4}'
    req=b'POST /v1/token-completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: '+str(len(body)).encode()+b'\r\nConnection: close\r\n\r\n'+body
    s=socket.create_connection(('127.0.0.1',port),timeout=4); s.sendall(req); data=b''
    while True:
        c=s.recv(65536)
        if not c: break
        data+=c
    payload=data.split(b'\r\n\r\n',1)[1]
    return json.loads(payload)

def run(workers,port,model):
    p=subprocess.Popen([str(ROOT/'aurora-llm'),str(model),str(workers),str(port)],stderr=subprocess.PIPE,preexec_fn=os.setsid)
    try:
        deadline=time.time()+4
        while time.time()<deadline:
            line=p.stderr.readline().decode(errors='replace')
            if 'ready:' in line: break
            if p.poll() is not None: raise RuntimeError('server exited early')
        return request(port)
    finally:
        try: os.killpg(os.getpgid(p.pid),signal.SIGTERM)
        except ProcessLookupError: pass
        try: p.wait(timeout=1)
        except subprocess.TimeoutExpired: p.kill()

def main():
    subprocess.check_call(['make'],cwd=ROOT)
    with tempfile.TemporaryDirectory() as td:
        model=Path(td)/'test.ali'
        subprocess.check_call([sys.executable,str(ROOT/'tools/make_test_model.py'),str(model)])
        a=run(1,19081,model)
        b=run(2,19082,model)
        c=run(4,19084,model)
        assert a==b==c,(a,b,c)
        print('PASS deterministic 1/2/4-worker output:',a)
        out=subprocess.check_output(['file',str(ROOT/'aurora-llm')],text=True)
        assert 'statically linked' in out,out
        print('PASS static ELF:',out.strip())

if __name__=='__main__': main()
