# 2026-06-16 LangBurst int4_bdr / GPU Embed / Prefill 세션 기록

이 문서는 이번 세션에서 진행한 LangBurst 추론 런타임 작업을 처음부터 끝까지 한곳에 정리한 기록이다. 기준 목표는 다음이었다.

```text
1. 운영 기본은 반드시 int4_bdr KV
2. embedding은 반드시 GPU 경로
3. 16K context를 기본 지원
4. OpenWebUI에서 연속 요청이 OOM 없이 동작
5. 응답 품질을 깨지 않고 prefill / TTFT를 최대한 단축
```

## 초기 상태

- 프로젝트 명칭과 소스 경로가 `qwenburst`에서 `langburst`로 이동되어 있었다.
- 실행 환경은 `ml-dmc8`의 `~/miniconda3` 아래 `langburst` conda env로 고정했다.
- OpenWebUI는 `http://192.168.0.47:3000/`, LangBurst API는 `http://192.168.0.47:8008/v1` 기준으로 점검했다.
- 운영 모델명은 `langburst-qwen3.6-27b-q3`이고, Q3 runtime checkpoint는 `/home/user/models/Qwen3.6-27B-langburst-q3`를 사용했다.
- 사용자가 명시한 최종 제약은 `int4_bdr`과 GPU embed를 절대 기본 경로에서 빼지 않는 것이었다.

## 확인한 문제

### 1. CPU embed가 prefill을 크게 망가뜨림

이전 측정에서 `CPU_EMBED=1`은 fp16/no-paged baseline에서도 약 `41 tok/s` 수준으로 떨어졌다. 반대로 GPU embed는 같은 계열에서 약 `900~1060 tok/s`까지 나왔다.

결론:

```text
CPU embed는 운영 기본으로 부적합.
GPU embed를 기본값으로 고정해야 함.
```

### 2. direct runtime path와 server batch path가 갈라져 있음

직접 프로파일러를 돌렸을 때 direct path는 여전히 token-by-token attention decode를 많이 탔다.

측정:

```text
prompt_tokens=1629
elapsed_s=180.809
attention_decode calls=16368
attention_decode measured share=75.98%
```

반면 OpenWebUI 서버 경로는 batch worker / paged prefill 경로를 타서 같은 규모에서 수 초 단위로 끝났다.

결론:

```text
사용자 체감 경로는 server batch path가 맞다.
direct RuntimeEngine.generate path는 아직 성능/계약 통합이 남은 별도 과제다.
```

### 3. int4_bdr prefill이 fp16 staging SDPA에 의존하던 구간이 있었음

`int4_bdr` paged KV는 persistent cache에는 맞지만, prefill block attention이 별도 fp16 staging KV와 SDPA mask를 만들면 긴 prompt에서 staging buffer가 prompt 길이에 맞춰 커진다. 이 구조는 16GB 환경에서 30K자 이상 붙여넣기 같은 요청에 취약했다.

처리 방향:

```text
persistent KV는 int4_bdr paged KV에 기록
prefill block attention도 같은 paged/batch attention contract 사용
fp16 staging KV와 prefill mask 생성 제거
```

결론:

```text
raw block / decode / batch / speculative verifier는 같은 paged KV commit contract를 타야 한다.
int4_bdr 운영 경로에서 별도 fp16 staging cache는 최종 구조가 아니다.
```

### 4. prefill hot loop에서 empty_cache를 실행하면 TTFT가 느려짐

`_trim_prefill_cuda_cache()`가 prefill chunk 직후 실행되면 `torch.cuda.synchronize() + gc.collect() + torch.cuda.empty_cache()`가 TTFT 안에 들어간다.

결론:

```text
hot prefill loop에서 cache trim 금지.
request release 이후에만 CUDA cache trim.
```

### 5. prefix cache가 16GB int4_bdr 운영에서 OOM을 유발

stateful prefix cache는 현재 GPU state snapshot/clone 성격이 남아 있다. Qwen3.6 27B Q3 + 16K window + int4_bdr에서 모델 로드 후 여유 VRAM이 매우 작기 때문에, prefix cache가 켜져 있으면 다음 요청에서 20MiB 작업 버퍼도 못 잡는 OOM이 반복됐다.

관찰된 에러:

```text
CUDA out of memory. Tried to allocate 20.00 MiB
free VRAM about 5~17 MiB
```

결론:

```text
prefix cache 기능 자체는 유지.
현재 구현은 16GB 운영 기본에서 OFF.
나중에 KV page refcount / COW 기반으로 바뀐 뒤 다시 기본 ON 후보.
```

## 적용한 코드 변경

### start script 기본값

파일:

```text
langburst/scripts/start_langburst.sh
```

변경:

```text
CPU_EMBED 기본값: 0
KV_CACHE_DTYPE 기본값: int4_bdr
CONTEXT_WINDOW 기본값: 16384
PREFIX_CACHE 기본값: off
LANGBURST_TRIM_CACHE_AFTER_REQUEST 도입
```

운영 기본 실행은 다음 의미를 갖는다.

```text
int4_bdr + GPU embed + paged KV + 16K window + prefix cache off
```

### int4_bdr paged prefill direct attention

파일:

```text
langburst/langburst/adapters/qwen36_impl/model.py
```

변경:

```text
prefill 전용 fp16 staging helper 제거
int4_bdr도 attention_decode_paged_batch(...) 경로로 통일
```

핵심:

```text
1. K/V는 persistent int4_bdr paged KV에 append
2. attention 계산은 int4_bdr paged KV를 직접 읽는 CUDA op 사용
3. fp16 staging KV와 causal mask allocation 제거
4. prefill/decode/batch/spec verifier의 KV commit contract 통일
```

### request 종료 후 cache trim

파일:

```text
langburst/langburst/engines/native/model_runner.py
```

변경:

```text
prefill 중 cache trim 제거
request finish/release 시점에 _trim_cuda_cache_after_request() 실행
```

이름도 `_trim_prefill_cuda_cache`에서 `_trim_cuda_cache_after_request`로 바꿔 실제 동작 의미와 맞췄다.

### CUDA op 추가

파일:

```text
langburst/csrc/kernels.cuh
langburst/csrc/attention_decode.cu
langburst/csrc/langburst_ext.cpp
langburst/langburst/ops.py
```

추가된 op:

```text
attention_append_paged_int4
```

목적:

```text
prefill block에서 생성된 K/V를 int4_bdr paged KV page에 직접 기록
```

## 빌드 / 환경 문제와 해결

처음 CUDA extension rebuild가 실패했다.

원인:

```text
PyTorch: 2.11.0+cu130
nvcc/nvvm: 13.2 계열
cccl: 13.3 계열
```

해결:

```text
nvidia-cuda-nvcc==13.0.88
nvidia-nvvm==13.0.88
nvidia-cuda-cccl==13.0.85
nvidia-cuda-runtime==13.0.96
nvidia-cuda-crt==13.0.88
```

검증:

```text
Cuda compilation tools, release 13.0, V13.0.88
attention_append_paged_int4 op load 확인
```

## 측정 결과

최종 운영 조건:

```text
KV_CACHE_DTYPE=int4_bdr
CPU_EMBED=0
LANGBURST_PAGED_KV=1
CONTEXT_WINDOW=16384
PREFILL_CHUNK_SIZE=64
PREFIX_CACHE=off
```

OpenWebUI/API 서버 경로에서 fp16 staging 제거 전 측정:

| 케이스 | Prompt tokens | TTFT | TTFT 기준 prompt tok/s | Decode tok/s | 결과 |
| --- | ---: | ---: | ---: | ---: | --- |
| warmup | 115 | 0.318s | 361.7 | 29.4 | 정상 |
| repeat50 | 835 | 0.915s | 913.0 | 22.4 | 정상 |
| repeat100 | 1635 | 1.774s | 921.6 | 16.8 | 정상 |
| repeat200 | 3235 | 3.712s | 871.4 | 13.2 | 정상 |
| repeat50_again | 835 | 0.912s | 915.5 | 19.7 | 정상 |

품질 확인 문장:

```text
비밀코드는 ALPHA42이고 좋아하는 색은 초록입니다.
```

chunk size sweep:

```text
chunk 64: 최선
chunk 96: 1635 tokens 기준 약 840.8 tok/s로 느림
chunk 128: OOM
```

fp16 staging 제거 후 direct int4_bdr paged attention 경로에서 측정:

| 케이스 | Prompt tokens | TTFT | TTFT 기준 prompt tok/s | Decode tok/s | 결과 |
| --- | ---: | ---: | ---: | ---: | --- |
| short repeat | 25 | 0.061s | 406.8 | 27.4 | 정상 |
| long paste 8K chars | 3557 | 10.743s | 331.1 | 13.8 | 정상, ALPHA42 회수 |
| long paste 30K chars | 13251 | 113.665s | 116.6 | 5.8 | 정상, ALPHA42 회수 |

해석:

```text
fp16 staging 제거로 긴 붙여넣기 OOM은 해소됐다.
하지만 현재 direct int4 paged attention kernel은 correctness-first 구현이라 긴 prefill 속도는 느리다.
다음 병목은 low-bit KV를 Tensor Core 친화 block attention으로 바꾸는 것이다.
```

bounded dense SDPA + direct int4_bdr fallback 라우팅 적용 후 측정:

| 케이스 | Prompt tokens | TTFT | TTFT 기준 prompt tok/s | Decode tok/s | 결과 |
| --- | ---: | ---: | ---: | ---: | --- |
| bounded SDPA 8K chars | 3557 | 4.177s | 851.6 | 13.3 | 정상, ALPHA42 회수 |
| direct int4 fallback 30K chars | 13251 | 104.867s | 126.4 | 5.9 | 정상, ALPHA42 회수 |

해석:

```text
8K token 이하 구간은 short fp16 SDPA bridge로 빠른 프리필을 우선한다.
8K token을 넘으면 fp16 staging을 버리고 int4_bdr paged KV flash/direct path로 떨어져 OOM을 피한다.
긴 입력 속도는 아직 BitDecoding식 Tensor Core low-bit attention kernel이 필요하다.
```

## 현재 결론

운영 기본은 다음이 맞다.

```text
int4_bdr + GPU embed + paged KV + 16K context + prefix cache off + prefill chunk 64
```

현재 16GB RTX 4080에서 OOM 안정성과 품질을 같이 만족하는 경로는 위 조합이다. 다만 fp16 staging 제거 후 긴 prefill 속도는 다시 낮아졌으므로, 이제 성능 목표는 `int4_bdr paged KV direct attention`의 Tensor Core / BitDecoding식 최적화다. `prefix cache`는 기능적으로 유망하지만 현재 구현은 GPU state clone 비용 때문에 기본 ON으로 두면 안 된다.

## 남은 병목

### 1. direct int4_bdr paged attention 고속화

fp16 staging 시절에는 1K~3K prompt에서 TTFT 기준 약 `870~920 tok/s`가 나왔지만, 긴 prompt에서 staging buffer OOM이 발생했다. direct int4_bdr paged attention으로 바꾼 뒤에는 30K자 prompt도 OOM 없이 처리하지만, 현재 correctness-first kernel은 긴 prefill에서 약 `116 tok/s`까지 떨어졌다.

다음 최적화 후보:

```text
1. BitDecoding식 Tensor Core 친화 low-bit KV layout
2. warp-level dequantization parallelism
3. software-pipelined dequant + attention accumulation
3. Direct RuntimeEngine path도 server batch/paged prefill contract로 통합
4. prefix cache를 GPU clone 없이 KV page refcount/COW 기반으로 재작성
```

현재 기본 라우팅:

```text
end_pos <= LANGBURST_SHORT_PREFILL_SDPA_TOKENS:
  persistent KV는 int4_bdr paged KV에 기록
  prefill attention은 short fp16 staging + SDPA bridge 사용

end_pos > LANGBURST_SHORT_PREFILL_SDPA_TOKENS:
  fp16 staging cache 제거
  int4_bdr paged KV flash/direct CUDA attention 사용
```

기본값:

```text
LANGBURST_SHORT_PREFILL_SDPA_TOKENS=8192
LANGBURST_SHORT_PREFILL_SDPA_MIN_FREE_MIB=384
LANGBURST_PAGED_ATTENTION_BACKEND=auto
```

`LANGBURST_PAGED_ATTENTION_BACKEND` 값:

| Mode | 의미 | 현재 상태 |
| --- | --- | --- |
| `auto` | dtype별 가능한 최선의 paged attention backend 자동 선택 | 기본값. tensorcore 커널이 있으면 우선, 없으면 flash |
| `direct` | 기존 direct int4 paged online-softmax kernel | 안정 baseline |
| `flash` | FlashAttention-style paged int4 backend. full fp16 KV staging 없이 paged int4 KV 직접 read | 구현됨. 현재 내부는 검증된 direct kernel을 재사용 |
| `tensorcore` | Tensor Core / BitDecoding식 low-bit KV kernel | 구현됨. 정확도 통과, 현재 microbench상 direct보다 느려 명시 실험 전용 |

### 1-1. BitDecoding 적용 계획

확인한 논문:

```text
BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache
arXiv:2503.18773
```

논문 요지:

```text
기존 low-bit KV decode는 CUDA core scalar dequant 중심이라 Tensor Core를 못 쓴다.
BitDecoding은 Tensor-Core-friendly layout, warp-level dequantization parallelism,
query transformation, software-pipelined dequantization kernel로 low-bit KV decode를 가속한다.
```

LangBurst 적용점:

```text
현재 attention_decode_paged_int4:
  correctness-first scalar dequant + online softmax

현재 attention_paged_int4_flash:
  같은 signature의 block-prefill 전용 entry
  full fp16 KV staging 없이 paged int4 KV를 직접 읽음
  state/출력 parity를 위해 현재 내부는 direct kernel 재사용

현재 attention_decode_paged_int4_tensorcore:
  QK score 계산을 WMMA/Tensor Core 경로로 수행
  int4 pages를 tile 단위로 dequant해 WMMA B fragment로 공급
  online softmax와 V accumulation contract는 direct 커널과 동일하게 유지
  direct 대비 정확도 테스트 통과

측정 결과:
  seq=512:   direct 0.302ms, tensorcore 0.381ms, 0.79x
  seq=2048:  direct 1.166ms, tensorcore 1.478ms, 0.79x
  seq=6000:  direct 3.392ms, tensorcore 4.113ms, 0.82x
  seq=12000: direct 6.084ms, tensorcore 7.711ms, 0.79x

판단:
  현재 tensorcore 커널은 진짜 WMMA를 사용하지만, V accumulation과 padded single-query head 구조의 오버헤드가 커서 기본값으로 켜지 않는다.
  다음 단계는 QK만 Tensor Core에 태우는 방식이 아니라, BitDecoding 논문식 Tensor-Core-friendly KV layout, warp-level dequant, V path까지 포함한 fused layout으로 재작성해야 한다.
```

### 2. direct runtime path 파편화

직접 profile path는 아직 `attention_decode` token-loop 성격이 남아 있다. 서버 경로는 빠르지만, runtime API와 server path가 다른 성능 특성을 보이는 것은 장기적으로 정리해야 한다.

정답 구조:

```text
RuntimeEngine.forward_batch(plan) / batch worker path를 prefill/decode/spec verify의 단일 계약으로 승격
direct generate도 같은 batch/paged path 사용
```

### 3. prefix cache 재도입 조건

현재 prefix cache는 기본 OFF다. 다시 켜려면 다음이 필요하다.

```text
1. KV page refcount
2. copy-on-write page fork
3. GDN/conv state snapshot의 경량화
4. repeated OpenWebUI system prompt TTFT benchmark
5. output identity gate
```

## 최종 운영 명령

```bash
cd /home/user/workspace/neurova/langburst
KV_CACHE_DTYPE=int4_bdr \
LANGBURST_PAGED_KV=1 \
CPU_EMBED=0 \
LANGBURST_CPU_EMBED=0 \
./scripts/start_langburst.sh
```

접속:

```text
OpenWebUI: http://192.168.0.47:3000/
Backend:   http://192.168.0.47:8008/v1
```
