#!/usr/bin/env python3
import argparse, json, socket, statistics, time
ap=argparse.ArgumentParser()
ap.add_argument('--host',default='127.0.0.1'); ap.add_argument('--port',type=int,default=8080)
ap.add_argument('--runs',type=int,default=20); ap.add_argument('--prompt-len',type=int,default=16); ap.add_argument('--max-tokens',type=int,default=16)
a=ap.parse_args()
body=json.dumps({'tokens':[1]*a.prompt_len,'max_tokens':a.max_tokens},separators=(',',':')).encode()
req=b'POST /v1/token-completions HTTP/1.1\r\nHost:x\r\nContent-Length:'+str(len(body)).encode()+b'\r\nConnection: close\r\n\r\n'+body
lat=[]; rates=[]; counts=[]
for _ in range(a.runs):
 t=time.perf_counter(); s=socket.create_connection((a.host,a.port)); s.sendall(req); data=b''
 while True:
  chunk=s.recv(65536)
  if not chunk: break
  data+=chunk
 s.close(); dt=time.perf_counter()-t
 payload=data.split(b'\r\n\r\n',1)[1]
 n=len(json.loads(payload)['tokens']); lat.append(dt); counts.append(n); rates.append(n/dt if dt else 0.0)
print(f'runs={len(lat)} median_ms={statistics.median(lat)*1000:.3f} p95_ms={sorted(lat)[int(.95*(len(lat)-1))]*1000:.3f}')
print(f'generated_tokens_per_request={statistics.median(counts):g} median_tok_s={statistics.median(rates):.2f}')
