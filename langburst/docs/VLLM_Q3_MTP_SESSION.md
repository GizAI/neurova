# LangBurst vLLM Q3/MTP 전환 세션 기록

작성일: 2026-06-16  
대상 프로젝트: LangBurst  
주요 원격 호스트: `ml-dmc9`  
주요 모델: `Qwen3.6-27B-langburst-q3`

이 문서는 이번 세션에서 LangBurst의 vLLM optional provider 경로를 확장하면서
실제로 결정하고 구현하고 검증한 내용을 빠짐없이 남기기 위한 기록이다. 기존
소스는 git에 남아 있으므로, 여기서는 최종 방향과 도중의 실패/수정/성공 조건을
명시한다.

## 1. 세션의 최종 방향

현재 정정된 기준은 native가 메인/기본 엔진이고 vLLM은 optional provider라는
구조다. vLLM 경로는 native 구현을 감싸지 않고, vLLM이 가진 기능을 직접 쓰는
선택 실행 경로로 둔다. 최종 구조는 다음과 같다.

```text
LangBurst host
  engine registry / feature policy / request facade

Engine providers
  native   기본 엔진, LangBurst 자체 구현
  vllm     선택 엔진
  sglang   선택 엔진
  exl3     선택 엔진
```

핵심 원칙은 다음이다.

- vLLM이 이미 잘하는 범용 서빙 기능은 vLLM이 맡는다.
- LangBurst 고유 기능만 vLLM 경로에 얇게 연결한다.
- native 구현은 삭제하지 않고 별도 provider/plugin 성격으로 유지한다.
- native는 vLLM 없이 기존처럼 독립 실행 가능해야 한다.
- vLLM 경로에서 굳이 native 런타임을 경유할 이유가 없는 기능은 경유하지 않는다.
- 기존 import 호환은 최종 구조를 위해 끊고, 테스트와 연구 도구도 새 구조로 맞춘다.

## 2. vLLM으로 넘기는 기능과 LangBurst가 유지하는 기능

이번 세션에서 다시 정리한 책임 경계는 아래와 같다.

### 2.1 vLLM으로 대체하는 영역

```text
기존 LangBurst 영역                         vLLM 경로의 판단
server.py의 OpenAI-compatible server         vLLM server 또는 AsyncLLMEngine wrapper로 위임
BatchGenerationWorker / BatchedModelRunner   vLLM scheduler/engine로 위임
ContinuousBatchScheduler                     vLLM continuous batching으로 위임
KVBlockTable / RadixPrefixCache              vLLM PagedAttention/prefix cache로 위임
GenerationConfig / sample_next               vLLM SamplingParams로 위임
HFCausalAdapter                              vLLM 표준 HF model loading으로 위임
attention_decode.cu / sampling.cu            vLLM attention/sampling hot path로 위임
표준 OpenAI API 옵션                         vLLM request/schema로 위임
```

### 2.2 LangBurst가 유지하는 영역

```text
LangBurst 영역                               유지 이유
qwen36_impl/model.py 일부                    Qwen3.6 hybrid/GDN/recurrent 고유 의미
gdn_recurrent.cu                             vLLM 기본 kernel로 대체 불가
qwen36_tools/quantize.py, audit.py           현재 q3/q4 자체 checkpoint format 유지
loader.py 일부                               LangBurst low-bit checkpoint loader 필요
RuntimeFeatures의 LangBurst-only 항목        stateful/session/episodic/TTT/ring 정책 표현
research 일부                                실험 가치는 있으나 serving hot path와 분리
native 전체                             vLLM 없는 자체 구현 실행/검증 경로
```

즉, vLLM 경로는 vLLM을 최대한 직접 사용한다. 다만 q3/q4 자체 포맷, low-bit
연산, GDN/recurrent 의미, MTP 연결처럼 vLLM 기본 기능만으로 표현되지 않는 부분은
LangBurst bridge가 담당한다.

## 3. 기존 문서와의 관계

이번 작업 이전에 이미 다음 문서들이 있었다.

- `docs/ADAPTER_ARCHITECTURE.md`: provider-host 구조, native 기본 엔진, vLLM/SGLang/EXL3 optional provider 구조.
- `docs/SERVING_ENGINE_TODO.md`: native serving backlog, paged KV, native MTP, CUDA graph, native scheduler 작업 목록.
- `docs/SPECULATIVE_RESEARCH.md`: native MTP/NEXTN 연구와 dmc8 측정값.

이번 문서는 위 문서들을 대체하지 않는다. 이 문서는 2026-06-16 세션에서 vLLM
기반 Q3 low-bit + MTP 경로가 실제로 어디까지 왔는지를 시간 순서와 판단 근거까지
남기는 세션 로그다.

## 4. 코드 구조 변경 요약

이번 세션에서 다룬 주요 파일은 다음이다.

```text
langburst/langburst/engines/
  base.py
  registry.py
  native/
    __init__.py
    provider.py
    runtime.py
    manager.py
    scheduler.py
    ...
  vllm/
    __init__.py
    provider.py
    bridge.py
    lowbit.py
    plugins.py
    qwen36.py

langburst/langburst/loader.py
langburst/scripts/bench_vllm_q3_metrics.py
langburst/tests/test_quant_lowbit_cpu.py
langburst/tests/test_engine_registry_cpu.py
```

역할은 다음과 같이 정리했다.

- `engines/vllm/provider.py`: vLLM provider 본체. 최종 vLLM kwargs를 구성하고, feature
  bridge가 만든 설정을 실제 vLLM engine에 전달한다.
- `engines/vllm/bridge.py`: LangBurst feature request를 vLLM kwargs와 metadata로
  변환한다.
- `engines/vllm/lowbit.py`: LangBurst q3/q4 low-bit checkpoint를 vLLM loader,
  quantization method, GPU cache 형태로 연결한다.
- `engines/vllm/plugins.py`: vLLM plugin registration. `langburst_lowbit`
  quantization/load format과 `Qwen3_5MTP` override를 등록한다.
- `engines/vllm/qwen36.py`: Qwen3.5/Qwen3.6 vLLM 경로에서 low-bit MTP를 올리기
  위한 custom MTP predictor/model 연결.
- `loader.py`: LangBurst low-bit CUDA kernel/Marlin wrapper. aux GPU weight
  실행과 output cache 정책을 수정했다.
- `scripts/bench_vllm_q3_metrics.py`: vLLM q3 smoke/속도 측정 스크립트.
- `tests/test_quant_lowbit_cpu.py`, `tests/test_engine_registry_cpu.py`: low-bit
  loader와 engine registry의 CPU 검증.

## 5. vLLM bridge 변경 내용

`vllm/bridge.py`에서 qwen36 low-bit 모델은 다음 기본값을 갖도록 정리했다.

```text
load_format = "langburst_lowbit"
quantization = "langburst_lowbit"
language_model_only = true
dtype = float16
enforce_eager = true
kv_cache_dtype = "fp8"    명시 override가 없으면 기본 적용
```

추가로 다음 override를 지원하게 했다.

- `LANGBURST_VLLM_MAX_NUM_BATCHED_TOKENS`: vLLM `max_num_batched_tokens` 조정.
- `kv_cache_memory_bytes`: vLLM extra kwargs 허용 목록에 추가.
- Qwen/GDN Mamba align 요구 때문에 `max_num_batched_tokens`는 최소 1568로 clamp.
- MTP를 켜면 vLLM block size 요구가 올라가므로 최소 1600으로 clamp.
- `enable_mtp`가 켜지면 `speculative_config={"method": "mtp", ...}`를 생성.

`vllm.py`에서는 `spec.extra` 병합 이후에도 한 번 더 clamp하도록 했다. 이 처리가
필요한 이유는 env나 사용자 인자가 bridge의 floor보다 낮은 값을 다시 넣을 수 있기
때문이다.

## 6. low-bit vLLM loader/cache 구현

`vllm/lowbit.py`는 이번 세션에서 가장 많이 손본 부분이다.

### 6.1 q3 index와 vLLM layer prefix mapping

`LangBurstLowBitConfig`는 vLLM이 요구하는 layer prefix를 LangBurst q3 tensor
이름으로 해석한다. strict audit 모드는 다음 환경변수로 활성화한다.

```bash
LANGBURST_VLLM_LOWBIT_AUDIT=1
LANGBURST_VLLM_LOWBIT_STRICT=1
```

이 모드에서는 다음 fallback이 남으면 실패해야 한다.

- `UnquantizedLinearMethod`
- `UnquantizedEmbeddingMethod`

목표는 다음 prefix들이 q3 index에 100% 매핑되는지 확인하는 것이다.

```text
language_model.*
model.language_model.*
lm_head.*
visual.*
mtp.*
```

### 6.2 fp16_raw와 low-bit tensor 분리

`LangBurstLowBitModelLoader`는 vLLM model object에는 `fp16_raw` weight만 직접
load하고, low-bit tensor는 quant method가 필요 시 materialize하도록 했다. 이렇게
해야 vLLM의 표준 model construction/loading 경로를 최대한 유지하면서 LangBurst
자체 checkpoint format만 얇게 붙일 수 있다.

### 6.3 GPU-only cache

CPU mmap lazy load가 decode 중에 발생하면 속도가 크게 무너진다. 그래서 다음 모드를
추가했다.

```bash
LANGBURST_VLLM_LOWBIT_GPU_ONLY=1
LANGBURST_VLLM_LOWBIT_PRELOAD=1
LANGBURST_VLLM_LOWBIT_PRELOAD_MTP=1
LANGBURST_VLLM_LOWBIT_STATS_INTERVAL=<N>
```

의미는 다음이다.

- `GPU_ONLY=1`: 허용된 low-bit tensor를 engine 시작 시 GPU에 preload하고,
  inference 중 CPU lazy load를 막는다.
- `PRELOAD=1`: preload 명시.
- `PRELOAD_MTP=1`: 기본 제외 대상인 MTP tensor까지 preload.
- `STATS_INTERVAL`: hit/miss/eviction 통계 출력.

초기 LRU cache 실험에서 `miss=계속 발생`, `hit=0` 패턴이 나왔기 때문에,
실제 q3 working set보다 작은 cache는 순환 evict/reload를 일으켜 의미가 없다는
것을 확인했다. 그래서 최종 smoke는 GPU-only resident preload를 기준으로 잡았다.

### 6.4 aux GPU routing

단일 16GB GPU에 q3 weight와 vLLM graph/profile/KV/workspace를 모두 올리기에는
scratch 여유가 부족했다. q3 포맷은 바꾸지 않고 16GB급 환경에서 돌리기 위해 일부
weight를 두 번째 GPU로 옮기는 aux routing을 추가했다.

```bash
LANGBURST_VLLM_LOWBIT_AUX_DEVICE=cuda:1
LANGBURST_VLLM_LOWBIT_AUX_REGEX=<regex>
```

기본 aux 대상은 다음이다.

MTP off:

```text
lm_head
embed_tokens
layers.0.mlp.gate_up_proj
layers.1.mlp.gate_up_proj
```

MTP on + `PRELOAD_MTP=1`:

```text
lm_head
embed_tokens
layers.0..11.mlp.gate_up_proj
mtp.layers.0.self_attn.qkv_proj
mtp.layers.0.self_attn.o_proj
mtp.layers.0.mlp.gate_up_proj
mtp.layers.0.mlp.down_proj
```

linear/embedding apply 시 aux GPU weight는 aux device에서 계산한 뒤 원래 device로
결과를 되돌린다. embedding row lookup도 aux GPU를 지원하도록 했다.

## 7. `loader.py` 수정

aux GPU weight를 사용하면서 처음에는 sampler/lm_head 경로에서 illegal memory
access가 발생했다. 원인은 custom CUDA low-bit kernel이 현재 device인 GPU0 stream
상태에서 GPU1에 있는 qweight를 사용했기 때문이다.

해결은 low-bit kernel 실행을 다음처럼 qweight device context 안으로 넣는 것이다.

```python
with torch.cuda.device(self.qweight.device):
    ...
```

또한 vLLM low-bit 경로에서는 Marlin output cache가 과도하게 persistent buffer를
잡지 않도록 다음 정책을 도입했다.

```bash
LANGBURST_MARLIN_OUT_CACHE_POLICY=decode_only
```

vLLM low-bit 경로는 이 값을 기본으로 사용한다. 목표는 decode hot path에는 cache를
남기되, vLLM profile/startup 중 큰 임시 buffer가 VRAM을 계속 점유하는 일을 줄이는
것이다.

## 8. MTP 연결 변경

MTP를 vLLM 경로에 연결하면서 처음에는 upstream `Qwen3_5MultiTokenPredictor`가
dense embedding을 먼저 만들어 약 2.37GiB를 할당했다. low-bit로 나중에 바꾸는 방식은
16GB 환경에서 맞지 않았다.

최종 방향은 다음이다.

- `Qwen3_5MTP`는 plugin override로 `LangBurstQwen36MTP`를 사용한다.
- `LangBurstQwen36MultiTokenPredictor.__init__`는 `super()`를 호출하지 않고,
  upstream 필드를 필요한 만큼 재현한다.
- MTP embedding은 처음부터 `VocabParallelEmbedding(..., quant_config=...)`으로
  만든다.
- `compilation_config`와 `do_not_compile=True`를 설정한다.

이 변경 후 target embedding/lm_head sharing 로그가 확인되었고, MTP model이 dense
embedding을 먼저 잡는 문제를 제거했다.

## 9. benchmark script 변경

`scripts/bench_vllm_q3_metrics.py`는 이번 세션에서 실제 측정을 위해 다음 인자를
받도록 정리했다.

```text
--model
--tokenizer
--qb-model
--max-model-len
--gpu-memory-utilization
--max-num-batched-tokens
--max-num-seqs
--kv-cache-dtype
--kv-cache-memory-bytes
--max-tokens
--enable-mtp
--mtp-speculative-tokens
```

기본 `gpu_memory_utilization`은 `0.965`로 조정했다.

MTP를 켜고 `--kv-cache-memory-bytes`를 주지 않으면 기본값은 다음이다.

```text
kv_cache_memory_bytes = 760000000
```

환경변수로는 다음 값을 읽는다.

```bash
LANGBURST_VLLM_MTP_KV_CACHE_MEMORY_BYTES
```

스크립트 출력은 다음 항목을 포함한다.

```text
TEXT
TOKENS
TIMING
TOK_S
METRICS_DICT
```

초기에는 CLI 인자가 실제 vLLM kwargs에 반영되지 않는 문제가 있었고, 이 때문에
`max_num_batched_tokens=128` 실험이 실제로는 다른 값으로 실행되었다. 이후 argparse와
bridge/env 반영을 수정했다.

## 10. 원격 환경

실험 호스트는 `ml-dmc9`였다.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate langburst
cd /home/user/workspace/neurova/langburst
```

GPU 상태는 다음이었다.

```text
GPU0: RTX 4070 Ti SUPER 16GB
GPU1: RTX 4070 Ti SUPER 16GB
```

초기 GPU0에는 `wan2gp` 프로세스가 약 318MiB를 사용하고 있었다. 해당 PID를 종료한
뒤 GPU0은 약 4MiB 사용 상태가 되었다. GPU1에는 다른 프로세스가 약 5.1GiB를 쓰고
있었지만 aux weight를 올릴 정도의 여유는 있었다.

## 11. 원격 sync 중 발생한 정리 이슈

작업 도중 rsync basename을 잘못 지정해 몇 차례 파일이 repo root 쪽에 잘못 생겼다.
예를 들면 다음 같은 파일들이 의도치 않게 `/home/user/workspace/neurova/langburst/`
바로 아래에 생성되었다.

```text
vllm/lowbit.py
bench_vllm_q3_metrics.py
loader.py
...
```

이 잔여 파일들은 `rm -f`로 제거하고, 올바른 위치로 다시 sync했다. 이 내용은 hidden
residue가 있었다는 의미가 아니라, 실험 중 파일 배치 실수를 발견하고 정리했다는
운영 기록이다.

## 12. 측정 과정과 실패/해결 로그

### 12.1 초기 vLLM q3 smoke

초기 상태에서도 vLLM 기본 `Qwen3_5ForConditionalGeneration` 경로로 q3 low-bit를
연결하면 짧은 출력은 가능했다.

```text
TEXT: Hello!
```

하지만 속도와 메모리 문제가 있었다.

### 12.2 작은 GPU cache의 실패

`LANGBURST_VLLM_LOWBIT_CACHE_MAX_GB=4`에서는 CPU mmap reload가 계속 발생했다.

```text
cache hits: 0
prefill: 약 8.5s
decode: 약 8.47s
```

cache를 키운 결과는 다음과 같았다.

```text
5GB cache: prefill 5.20s, decode 5.18s
6GB cache: prefill 4.91s, decode 4.88s
7GB cache: prefill 3.98s, decode 3.95s, KV 약 1920 tokens 제한
```

근본 원인은 q3 language path working set이 약 12.77GiB이고, full q3 index는 약
13.9GiB라는 점이었다. working set보다 작은 LRU cache는 순환 eviction을 만들고
실질적으로 CPU reload를 제거하지 못한다.

### 12.3 단일 GPU full preload의 한계

GPU0에 외부 프로세스가 있을 때는 OOM이 났다. GPU0을 비운 뒤 full preload 자체는
성공했다.

```text
loaded=659 skipped=345 cached_gb=12.77 gpu_only=True
model loading: 15.27 GiB
```

그러나 vLLM profile 단계에서 약 2MiB scratch도 확보하지 못해 OOM이 났다. 이때
확인한 결론은 다음이다.

- native는 16GB에 올라가지만 vLLM은 추가 overhead가 있다.
- vLLM은 scheduler/profile/graph/KV/FlashInfer workspace 여유가 필요하다.
- q3 포맷을 바꾸지 않고 single 16GB GPU에 full resident + vLLM을 넣는 것은 현재
  구조에서는 현실적이지 않다.

### 12.4 `max_num_batched_tokens=128` 실험 실패

처음에는 bench script가 CLI 인자를 제대로 전달하지 않아 실험 자체가 잘못되었다.
수정 후에는 Qwen/GDN Mamba align 요구 때문에 너무 작은 값은 불가능하다는 것을
확인했다.

```text
non-MTP minimum: 1568
MTP minimum: 1600
```

### 12.5 `lm_head` 제외 preload 실패

`lm_head`를 GPU resident preload에서 제외하면 main cache는 줄었다.

```text
main cached_gb=12.16
model loading=14.66 GiB
```

하지만 sampler에서 `lm_head.weight`를 lazy load하면서 OOM이 났다. 따라서 `lm_head`
역시 CPU lazy load가 아니라 resident 또는 aux GPU에 있어야 한다.

### 12.6 aux `lm_head` illegal memory access

`lm_head`를 GPU1로 옮겼을 때 main/aux cache는 다음과 같았다.

```text
main cached_gb=12.16
aux cached_gb=0.61
model loading=14.66 GiB
```

첫 실패는 sampler에서 illegal memory access였다. 원인은 GPU1 weight를 GPU0 current
device에서 custom CUDA kernel로 실행한 것이었다. `loader.py`의 device guard 수정으로
해결했다.

### 12.7 KV/profile tuning

`gpu_memory_utilization` 조정 결과는 다음이었다.

```text
0.70: available KV -4.09GiB로 실패
0.98: KV 0.27GiB, 1024 context에 필요한 0.33GiB 부족
0.99: startup free-memory reservation 실패
0.985: KV는 가능했으나 max_num_batched_tokens=128 assert 실패
```

`max_num_batched_tokens=1568`로 clamp한 뒤에는 block-size assert는 해결됐지만,
MLP `gate_up_proj` output 약 106MiB를 만들 free memory가 부족해 OOM이 났다.

### 12.8 aux `embed_tokens`와 일부 `gate_up_proj`

`embed_tokens`와 `lm_head`를 GPU1로 옮긴 결과는 다음이었다.

```text
main loaded=657 skipped=347 cached_gb=11.70
aux loaded=2 skipped=1002 cached_gb=1.07
model loading=14.44 GiB
```

engine startup은 성공했지만 첫 request에서 FlashInfer workspace 약 394MiB를 요구했고,
free memory가 약 308MiB라 OOM이 났다.

다음으로 layer 0/1의 `mlp.gate_up_proj`도 aux로 옮겼다.

```text
main cached_gb=11.52
aux cached_gb=1.24
model loading=14.26 GiB
```

vLLM은 여유 메모리를 더 KV로 가져가 약 2194 token KV를 만들었고, 여전히 workspace
OOM이 났다. 그래서 `gpu_memory_utilization=0.965`로 낮추어 KV를 약 1170 tokens로
줄였다.

### 12.9 non-MTP 성공

최종 non-MTP smoke는 성공했다.

실행 형태:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
LANGBURST_VLLM_LOWBIT_GPU_ONLY=1 \
LANGBURST_VLLM_LOWBIT_AUX_DEVICE=cuda:1 \
python scripts/bench_vllm_q3_metrics.py --kv-cache-dtype fp8
```

결과:

```text
TEXT: Hello!
prompt tokens: 30
completion tokens: 3
decode_after_first tokens: 2

prefill_s: 0.1870
decode_s: 0.1565

prefill tok/s: 160.39
decode-after-first tok/s: 12.78
completion-all tok/s: 19.17

lowbit stats:
  hits: 3000
  misses: 0
  evictions: 0
```

중요한 점은 inference 중 CPU weight reload가 사라졌다는 것이다. `misses=0`이므로
GPU-only resident path가 실제로 작동했다.

## 13. MTP 활성화 과정

### 13.1 MTP tensor 크기

q3 index 기준 MTP tensor는 12개였고 총 약 0.276GiB였다. 주요 tensor 크기는 다음과
같다.

```text
mtp.fc.weight: 약 100MiB, fp16_raw
mtp.layers.0.mlp.gate_up_proj: 약 87.7MiB
mtp.layers.0.mlp.down_proj: 약 43.8MiB
mtp.layers.0.self_attn.qkv_proj: 약 36.1MiB
mtp.layers.0.self_attn.o_proj: 약 15.5MiB
```

### 13.2 dense MTP embedding OOM

첫 MTP 시도는 upstream MTP predictor가 dense embedding을 먼저 만들면서 약 2.37GiB를
할당해 실패했다. 이 문제는 `LangBurstQwen36MultiTokenPredictor.__init__`에서
처음부터 quantized embedding을 만들도록 바꾸면서 해결했다.

### 13.3 MTP KV와 workspace 부족

MTP overhead 때문에 non-MTP보다 더 공격적인 aux routing이 필요했다.

초기 MTP aux 확장:

```text
layers.0..7.mlp.gate_up_proj
MTP qkv/o/gate_up/down

main cached_gb=11.11
aux cached_gb=1.94
model loading=13.94 GiB
KV: 약 1102 tokens
```

이후 MTP block-size 요구 때문에 `max_num_batched_tokens` floor를 1600으로 올렸다.
그래도 FlashInfer workspace free memory가 약 148MiB라 부족했다.

다음 확장:

```text
layers.0..11.mlp.gate_up_proj
MTP qkv/o/gate_up/down

main cached_gb=10.77
aux cached_gb=2.28
model loading=13.59 GiB
```

vLLM이 여유 memory를 다시 KV로 더 가져가 약 1575 token KV를 만들면서 workspace
OOM이 반복됐다.

### 13.4 MTP 성공 조건

최종 해결은 MTP에서 KV cache memory를 명시적으로 제한하는 것이었다.

```text
kv_cache_memory_bytes = 760000000
```

이 설정은 vLLM memory profiling을 우회하고 약 0.71GiB KV를 명시적으로 예약한다.
실제 GPU KV cache size는 1024 tokens로 잡혔다.

실행 형태:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
LANGBURST_VLLM_LOWBIT_GPU_ONLY=1 \
LANGBURST_VLLM_LOWBIT_PRELOAD_MTP=1 \
LANGBURST_VLLM_LOWBIT_AUX_DEVICE=cuda:1 \
python scripts/bench_vllm_q3_metrics.py \
  --kv-cache-dtype fp8 \
  --enable-mtp \
  --mtp-speculative-tokens 2
```

`bench_vllm_q3_metrics.py`는 MTP가 켜졌고 별도 override가 없으면 위
`kv_cache_memory_bytes=760000000`을 자동으로 넣는다.

결과:

```text
TEXT: Hello!
prompt tokens: 30
completion tokens: 3
decode_after_first tokens: 2

prefill_s: 0.217377
decode_s: 0.097598

prefill tok/s: 138.01
decode-after-first tok/s: 20.49
completion-all tok/s: 30.74

lowbit stats:
  hits: 2000
  misses: 0
  evictions: 0
```

## 14. MTP 적용 전후 비교

이번 smoke 기준 비교는 다음이다.

```text
항목                    non-MTP        MTP             변화
prefill tok/s           160.39         138.01          약 14.0% 느림
decode-after-first      12.78          20.49           약 60.4% 빠름
completion-all tok/s    19.17          30.74           약 60.3% 빠름
```

해석:

- 이 smoke는 completion이 3 token뿐이라 MTP 수치가 안정적인 최종 성능이라고 보기는
  어렵다.
- 그래도 같은 q3 low-bit resident 조건에서 MTP가 decode 구간을 실제로 개선한 것은
  확인했다.
- prefill은 MTP 모델/초기화 overhead 때문에 약간 느려졌다.
- 긴 출력, 예를 들어 `max_tokens=128` 또는 `256` 기준 재측정이 필요하다.
- vLLM metric에서 MTP acceptance rate를 직접 뽑는 작업은 아직 남아 있다.

## 15. native와 vLLM의 16GB 차이

사용자 질문의 핵심은 native는 16GB에 올라가는데 vLLM은 왜 어려운가였다.

이번 실험에서 확인한 차이는 다음이다.

- native는 LangBurst가 필요한 kernel/state/KV를 직접 좁게 잡는다.
- vLLM은 generic serving engine이라 scheduler, profiling, graph, block manager,
  FlashInfer workspace, paged KV 예약, sampling stack 등 추가 overhead가 있다.
- q3 full resident preload만으로도 model loading이 15GiB 전후까지 올라간다.
- 여기에 vLLM profile scratch와 FlashInfer workspace가 필요하므로 single 16GB는
  극도로 빠듯하다.

따라서 q3 포맷을 바꾸지 않는 조건에서 현실적인 접근은 다음이었다.

```text
GPU0: vLLM engine + 대부분의 q3 language weights + KV/workspace
GPU1: lm_head, embed_tokens, 일부 gate_up_proj, MTP projection weights
KV: fp8
MTP: kv_cache_memory_bytes로 KV 예약량 명시 제한
CPU: inference hot path weight reload 금지
```

이 접근으로 q3 포맷은 유지했고, CPU hot path도 제거했다.

## 16. q3 포맷 유지에 대한 결론

이번 세션에서는 q3 checkpoint format을 바꾸지 않았다.

현재 구현은 다음 방식이다.

- q3 index는 그대로 읽는다.
- vLLM model construction은 upstream Qwen3.5/Qwen3.6 계열 경로를 최대한 사용한다.
- LangBurst quantization method가 layer prefix를 q3 tensor로 매핑한다.
- low-bit tensor는 GPU resident cache에 올린다.
- MTP tensor도 필요 시 `PRELOAD_MTP=1`로 같은 cache 정책에 포함한다.
- 부족한 VRAM은 aux GPU routing과 KV bytes 제한으로 해결한다.

즉, q3 포맷을 vLLM native quant format으로 변환하지 않고도 vLLM 경로에서 실행하는 데
성공했다. 다만 장기적으로는 vLLM의 더 깊은 quant backend와 통합하거나 tensor
parallel/sharded low-bit 구조로 옮기면 aux transfer overhead를 줄일 수 있다.

## 17. 검증한 테스트

로컬과 원격에서 다음 검증을 실행했다.

```bash
python -m compileall -q ...
pytest -q langburst/tests/test_quant_lowbit_cpu.py langburst/tests/test_engine_registry_cpu.py
```

결과:

```text
12 passed
```

중간에 더 작은 범위로도 다음 테스트가 통과했다.

```bash
pytest -q langburst/tests/test_engine_registry_cpu.py
```

결과:

```text
8 passed
```

## 18. 현재 작동하는 대표 명령

### 18.1 non-MTP q3 vLLM smoke

```bash
CUDA_VISIBLE_DEVICES=0,1 \
LANGBURST_VLLM_LOWBIT_GPU_ONLY=1 \
LANGBURST_VLLM_LOWBIT_AUX_DEVICE=cuda:1 \
python scripts/bench_vllm_q3_metrics.py --kv-cache-dtype fp8
```

### 18.2 MTP q3 vLLM smoke

```bash
CUDA_VISIBLE_DEVICES=0,1 \
LANGBURST_VLLM_LOWBIT_GPU_ONLY=1 \
LANGBURST_VLLM_LOWBIT_PRELOAD_MTP=1 \
LANGBURST_VLLM_LOWBIT_AUX_DEVICE=cuda:1 \
python scripts/bench_vllm_q3_metrics.py \
  --kv-cache-dtype fp8 \
  --enable-mtp \
  --mtp-speculative-tokens 2
```

## 19. 현재 한계

현재 상태는 production complete가 아니라 vLLM q3 low-bit + MTP path의 중요한
smoke 성공 지점이다.

남은 한계는 다음이다.

- smoke 출력이 3 token이라 MTP 성능 수치가 충분히 안정적이지 않다.
- 긴 출력 128/256 token 기준 prefill/decode/MTP acceptance 재측정이 필요하다.
- FlashInfer/Triton JIT warmup 영향을 분리해야 한다.
- aux GPU routing은 동작하지만, aux transfer cost가 긴 출력에서 어떤 영향을 주는지
  별도 측정해야 한다.
- `kv_cache_memory_bytes=760000000`은 1024 token 수준으로 context/concurrency를
  제한한다.
- hardware profile별 aux regex/KV bytes default가 아직 config profile로 정리되지
  않았다.
- vLLM 경로에서 recurrent/stateful/ring/infinite-context 기능은 metadata와 방향은
  정리했지만, 실제 multi-turn/session correctness 검증은 더 필요하다.
- MTP acceptance rate를 vLLM metrics에서 명시적으로 추출하는 작업이 남아 있다.

## 20. 다음 작업

우선순위는 다음이다.

1. `max_tokens=128`, `256` 기준으로 non-MTP/MTP를 같은 prompt set에서 재측정한다.
2. MTP acceptance, accepted tokens, rejected tokens를 metrics에 노출한다.
3. `ml-dmc9` 2x16GB profile을 config로 고정한다.
4. aux regex와 `kv_cache_memory_bytes`를 bench script 기본값이 아니라 engine profile로
   승격한다.
5. stateful/session/ring KV/recurrent state가 vLLM 경로에서 실제 request lifecycle과
   맞는지 correctness test를 추가한다.
6. q3 format을 유지하면서 vLLM quant backend와 더 깊게 붙일 수 있는지 검토한다.
7. native provider는 별도 유지하되, vLLM path와 native path의 책임 경계가 다시
   섞이지 않도록 import/test 구조를 계속 정리한다.

## 21. 현재 결론

이번 세션의 결론은 다음이다.

- LangBurst의 기본 엔진은 native로 두는 구조가 맞다.
- vLLM 구현은 삭제하지 않고 독립 optional provider로 유지하는 것이 맞다.
- vLLM이 대체 가능한 generic serving 기능은 vLLM에 맡기는 것이 맞다.
- LangBurst q3 low-bit checkpoint는 format 변경 없이 vLLM 경로에서 실행 가능해졌다.
- CPU weight reload 없이 GPU-only hot path로 `Qwen3.6-27B-langburst-q3` smoke가
  성공했다.
- single 16GB full resident는 vLLM overhead 때문에 아직 무리이고, 현재는 2x16GB에서
  aux GPU routing + fp8 KV + KV bytes 제한이 필요하다.
- MTP도 vLLM 경로에서 올라갔고, 짧은 smoke 기준 decode 구간은 약 1.6배 빨라졌다.
- 다만 MTP 성능 주장은 긴 출력 benchmark와 acceptance metric이 붙은 뒤에 확정해야
  한다.
