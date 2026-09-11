심층적으로 뜯어본 결과, **이 패키지를 “실제 B0급 반도체 제품 RTL”이라고 평가하기는 어렵습니다.** 다만 완전히 허구인 자료도 아닙니다. SystemVerilog 구조, ECC/FIFO/DMA/연산 블록, 소프트웨어 ABI, 테스트 스캐폴딩까지 실제 코드가 있고, hard-IP를 black box로 분리한 방식도 방향 자체는 맞습니다.

제 판정은 **“잘 포장된 architecture/reference RTL + PoC”**에 가깝습니다.
**제품용 RTL 완성도는 낮고, 특히 18.45 POPS / 24 TB/s라는 성능 숫자와 현재 RTL 데이터패스 사이에 구조적인 단절이 있습니다.**

### 종합 판정

| 항목평가                       |          |
| -------------------------- | -------- |
| 아키텍처/PoC 참고용               | **6/10** |
| RTL 구현의 실질성                | **5/10** |
| 기능 정확성 신뢰도                 | **3/10** |
| 성능 주장과 RTL 일치도             | **1/10** |
| 검증 성숙도                     | **2/10** |
| CDC/Reset/RAS/Security/DFT | **2/10** |
| 물리 구현 가능 상태                | **1/10** |
| Tape-out readiness         | **0/10** |

## 1. 가장 치명적: 18.45 POPS 계산 블록이 실제 B0 top에 연결되어 있지 않습니다

문서상 B0는

`2 dies × 64 clusters/die × 16 arrays/cluster × 64 × 64`

이고 총 **8,388,608 MAC**으로 계산해서 1.1 GHz × 2 op/MAC = **18.4549376 POPS**를 냅니다. 이 산수 자체는 맞습니다.

그런데 실제 제품 top인 `rtl/top/fusionx_b0_top.sv`를 따라가면:

`fusionx_b0_top`
→ `fx_dual_die_package_top`
→ die당 **`fx_accel_core`** **하나**

입니다. (`fusionx_b0_top.sv:71–75`, `fx_dual_die_package_top.sv:34–48`)

그리고 `fx_accel_core` 기본 파라미터는 **8×8 MXU 하나**입니다. 즉 이 top hierarchy에는 문서에서 말하는 **64 clusters × 16 × 64×64 tensor plane이 들어있지 않습니다.**

그 대규모 구조는 별도 synthesis root인 `fx_b0_compute_plane_top`에만 존재합니다. 다시 말하면 현재 패키지에는 **“관리/HBM 제품 top”과 “대규모 tensor plane”이 따로 놀고 있고, 양쪽을 실제 제품 데이터패스로 묶은 RTL이 없습니다.**

이건 단순히 PHY hard-IP가 없다는 차원의 문제가 아닙니다. **핵심 compute fabric과 제품 top 자체가 통합되지 않은 상태**입니다.

---

## 2. 심지어 별도 compute plane도 128개 cluster를 동시에 먹일 수 없습니다

`fx_b0_compute_plane_top`의 입력을 보면 단 하나의

`step_valid`, `step_die`, `step_cluster`, `step_a`, `step_b`

인터페이스만 있습니다.

그리고:

`fx_b0_compute_plane_top.sv:29–31`

에서 `step_die` 하나를 one-hot으로 만들고,

`fx_b0_compute_die.sv:40`

에서 다시 `step_cluster` 하나만 one-hot으로 만듭니다.

따라서 **한 clock에 전체 2-die/128-cluster 중 한 cluster만 새 outer-product step을 받을 수 있습니다.**

한 cluster의 peak는:

16 arrays × 64 × 64 MAC × 2 op × 1.1 GHz
\= **144.18 TOPS = 0.144 POPS**

입니다.

즉 인터페이스 자체가 허용하는 cluster 공급률 기준으로는 주장한 18.45 POPS보다 정확히 **128배 작습니다.**

별도의 `fx_tensor_fabric_prod`는 모든 cluster에 병렬 입력을 받을 수 있게 만들어져 있지만, 제가 hierarchy를 추적해 보니 **어느 제품 root에도 instantiate되지 않습니다.**

이 부분은 매우 큰 설계 불일치입니다.

---

## 3. 24 TB/s HBM 주장은 현재 RTL로 구조적으로 불가능합니다

더 심각한 곳이 메모리입니다.

`fx_dual_die_package_top`은 die당 4개의 HBM 포트를 만들지만, 앞단의 `fx_hbm_fabric`은 요청 입력이 **한 개**뿐이고:

`fx_hbm_fabric.sv:27`

```
logic inflight_q;
```

한 요청을 받으면 응답이 올 때까지:

`fx_hbm_fabric.sv:34–42`

새 요청을 받지 않습니다.

즉 **4개의 HBM stack 중 여러 stack을 병렬로 구동할 수 없습니다. 한 die 전체에 outstanding transaction 하나뿐입니다.**

또 응답 구조체에는 `last`가 있는데, fabric은:

`fx_hbm_fabric.sv:48`

```
if(c_rsp_valid&&c_rsp_ready) inflight_q<=1'b0;
```

처럼 **`rsp.last`****를 보지도 않고 첫 response에서 transaction을 종료합니다.** 따라서 제대로 된 multi-beat burst infrastructure도 아닙니다.

데이터폭도 512 bit = 64 bytes입니다.

1.1 GHz라고 극도로 낙관적으로 가정해도 단순 bus capacity는:

64 B × 1.1 GHz = **70.4 GB/s**

이고, 현재 one-request → response → next-request 구조는 최소한 요청/응답 상태가 분리되므로 실제 sustained는 이보다 더 낮습니다.

양 die를 다 합쳐도 현재 protocol 구조상 **24 TB/s와는 수백 배 차이**입니다.

따라서 `results/performance_model.json`의 24 TB/s는 RTL에서 산출된 수치가 아니라 말 그대로 문서에 적힌 **analytical architecture target**입니다.

---

## 4. PCIe 6/CXL은 사실상 관리 레지스터 포트만 있습니다

`fx_pcie6_cxl_vendor_macro` 인터페이스를 확인해 보면 PCIe 쪽에서 RTL로 나오는 것은 **AXI-Lite management interface**뿐입니다.

즉 PCIe/CXL → HBM으로 모델 weight/activation을 싣는 대용량 DMA 또는 CXL.mem 데이터 경로가 이 RTL에는 없습니다.

`fusionx_b0_top.sv:33–40` 역시 PCIe macro를 AXI-Lite register block에만 연결합니다.

그 결과 현재 제품 top 기준으로는:

**Host → command registers는 있지만, Host → 192GB HBM으로 데이터를 고속 전송하는 실제 product datapath가 없습니다.**

AI accelerator에서는 이게 주변 기능이 아니라 필수 핵심 기능입니다.

---

## 5. scratchpad 용량도 문서와 RTL이 세 가지로 다릅니다

이 부분은 configuration 관리 상태를 잘 보여줍니다.

`fx_config_pkg.sv`:

- **64 MiB / die**

`configs/hardware.json`:

- **128 MB / die**

실제 `fusionx_b0_top`에서 instantiate되는 `fx_accel_core` 기본값:

- `SPAD_BANKS = 8`
- `SPAD_WORDS_PER_BANK = 1024`
- word = 512 bit = 64 B

따라서 실제 connected RTL은:

8 × 1024 × 64 B
\= **512 KiB / die**

입니다.

즉 **512 KiB ↔ 64 MiB ↔ 128 MB**라는 세 가지 서로 다른 값이 존재합니다.

제품 RTL freeze 전이라 하더라도 architecture configuration source-of-truth가 정리되지 않은 상태입니다.

---

## 6. “native BF16” 주장도 tensor RTL과 맞지 않습니다

`configs/hardware.json`에는 BF16을 native dtype이라고 씁니다.

그런데 tensor path의:

`fx_number_decode.sv`

입력은 아예:

```
input logic [7:0] raw
```

8 bit뿐입니다.

BF16/INT16 case에서는:

```
value_q8_8 = $signed({raw,8'h00});
```

로 처리합니다.

즉 **16-bit BF16 bit pattern 자체를 tensor array가 입력받을 방법이 없습니다.**

코드 주석도 사실상 “BF16 변환은 vector front-end에서 미리 해라”고 인정하고 있습니다.

따라서 이는 **BF16 native tensor execution이라기보다 8-bit 입력을 Q8.8처럼 취급하는 reference datapath**입니다.

---

## 7. Python/C++ “bit-accurate model”과 RTL이 실제로 다릅니다

이건 기능 검증 신뢰도를 꽤 떨어뜨립니다.

예를 들어 C++ attention model (`sim/cpp/fusionx_model.cpp`)은 score 계산에:

**1 / sqrt(head\_dim)**

을 적용하고 이후 자연지수 `exp()`를 씁니다.

그런데 `fx_attention_head.sv:97–116`의 RTL은 raw dot-product를 만들고 고정 shift 후 자체 \*\*`exp2_approx_q16()`\*\*를 사용합니다.

즉

- C++: `exp(x / sqrt(d))`
- RTL: 사실상 `2^(scaled x)`

계열입니다.

동일한 numerical semantics가 아닙니다.

연결된 vector softmax도 `fx_vector_memory_engine.sv:71–82`에서 `exp2_approx_q16()`을 쓰지만 C++ `softmax_q15()`은 `std::exp()`를 씁니다.

예를 들어 Q8.8 입력 차이가 -1.0이면:

- RTL 근사 기반: 약 `2^-1 = 0.5`
- reference model: `e^-1 ≈ 0.368`

이므로 작은 rounding 차이가 아니라 **알고리즘 자체가 다릅니다.**

따라서 현재 PASS한 Python/C++ regression을 **RTL의 bit-accurate golden verification**이라고 볼 수 없습니다.

---

## 8. 제공된 MXU RTL 테스트조차 실제 실행했다면 결과가 의심됩니다

`verification/tests/tb_mxu.sv`는 INT4 1, 2, 3...을 넣은 뒤 accumulator가 그냥:

`1×3 + 5×7`

같은 값을 갖기를 기대합니다.

하지만 `fx_number_decode.sv`는 INT4를 명시적으로 **Q8.8 형태로 변환한 뒤** `fx_outer_product_array`가 그 값을 그대로 곱해서 48-bit accumulator에 더합니다.

중간에 Q-format을 원래 integer MAC 단위로 되돌리는 `>>16` 같은 처리가 없습니다.

즉 TB의 expected unit과 RTL accumulator unit이 맞지 않습니다.

그리고 중요한 사실은 이 TB가 **실행된 적이 없습니다.**

패키지의 `reports/rtl_sim.log`:

```
RTL_SIM_STATUS=DEFERRED_TOOL_UNAVAILABLE
```

이고 formal도:

```
FORMAL_STATUS=DEFERRED_TOOL_UNAVAILABLE
```

입니다.

그래서 이런 문제가 걸러지지 않은 겁니다.

---

## 9. “1.1 GHz production datapath”로 보기 어려운 회로 구조입니다

이 RTL을 논리적으로 synthesis할 수 있느냐와 **1.1 GHz에서 물리적으로 닫히느냐**는 완전히 다른 문제입니다.

대표적으로 `fx_outer_product_array.sv` 한 cell은 한 clock에:

**16×16 multiplication + 48-bit accumulation**

을 끝냅니다.

중간 pipeline register가 없습니다.

그걸 production 파라미터로 펼치면 총:

**8,388,608개의 multiplier/accumulator**

가 됩니다.

Accumulator만:

8,388,608 × 48 bit
\= **402,653,184 bit ≈ 48 MiB**

의 register state입니다.

그리고 모든 accumulator가 asynchronous reset의 영향을 받습니다.

더 심각하게 `fx_b0_tensor_tile.sv` 한 tile에서 accumulator는:

16 × 64 × 64 × 48
\= **3,145,728 bit**

인데, 결과를:

`fx_b0_tensor_tile.sv:51`

```
acc_padded[result_word_q*512 +:512]
```

처럼 variable part-select 합니다.

기본 설정에서 tile당 **6144개의 512-bit word 중 하나를 선택하는 초대형 mux**가 됩니다.

실제 product design이라면 accumulator SRAM/bank 구조와 local readout fabric을 설계해야 할 부분을 거대한 flattened vector로 표현한 것입니다.

**기능 reference에는 편하지만 PPA-friendly production microarchitecture는 아닙니다.**

---

## 10. SDC 자체도 1.1 GHz를 검증하지 않습니다

B0 constraint:

`constraints/fusionx_b0.sdc.template:3–5`

```
create_clock ... -period 6.400
create_generated_clock ... -divide_by 1
```

입니다.

6.4 ns는 약 **156.25 MHz**입니다.

문서의 target인 1.1 GHz라면 약:

**0.909 ns**

clock period가 필요합니다.

물론 파일 이름부터 `.template`이고 “characterized value로 교체하라”고 써 있으므로 사기성은 없습니다. 하지만 반대로 말하면 **1.1 GHz timing closure에 대한 아무런 증거도 없는 상태**라는 뜻입니다.

특히 RTL 내부에 variable division, 큰 combinational integer sqrt, 대형 mux가 있어 0.909 ns closure를 단순히 기대하기도 어렵습니다.

---

## 11. HBM 4개를 달아 놓고 사실상 동시에 하나만 사용합니다

`fx_hbm_fabric`의 `inflight_q`가 **PORTS마다 있는 것이 아니라 fabric 전체에 하나**입니다.

그래서 4 HBM stack/die를 instantiate했지만:

- HBM0 read outstanding
- HBM1 read outstanding
- HBM2 read outstanding
- HBM3 read outstanding

같은 구조가 안 됩니다.

AI 칩이 HBM bandwidth를 제대로 활용하려면 수십\~수백 개 이상의 outstanding transaction, 채널별 scheduler, bank/rank awareness, QoS, read/write reordering 등이 필요한데, 현재 것은 **address stripe + single outstanding demux** 수준입니다.

`fx_mem_scheduler`도 마찬가지로 전역 `inflight_q` 하나입니다.

그래서 “24 TB/s HBM subsystem RTL”이라기보다 **memory protocol functional stub/reference**라고 보는 게 맞습니다.

---

## 12. Scratchpad도 8 banks인데 병렬 bank 접근을 못 합니다

`fx_scratchpad.sv`에도:

```
logic inflight_q;
```

하나만 존재합니다.

어느 bank든 요청 하나가 들어가면 응답이 끝날 때까지 전체 scratchpad가 다음 요청을 못 받습니다.

즉 8 bank를 만들어 놓고도 **8-way bandwidth를 활용하는 구조가 아닙니다.**

더구나 `fx_accel_core`의 command processor는 한 번에 DMA/GEMM/vector/sequence **한 엔진만 실행**합니다.

따라서 DMA와 GEMM을 overlap해서 double buffering하는 일반적인 accelerator 구조도 아닙니다.

이건 throughput accelerator라기보다 명백하게 **scalar command reference machine**에 가깝습니다.

---

## 13. Secure Boot는 존재하지만 실제 B0 보안을 보호하지 않습니다

`fx_secure_boot_ctrl`과 SHA-256, signature verifier wrapper는 코드로 존재합니다.

그런데 제가 RTL instantiation graph를 확인해 보니 **`fx_secure_boot_ctrl`****은 B0 top에서 instantiate되지 않습니다.**

실제 compute reset은:

`fusionx_b0_top.sv:72`

```
mgmt_rst_n && pll_locked && !emergency_reset
```

뿐입니다.

Secure boot의 `release_compute_reset` 신호가 제품 reset chain에 들어가지 않습니다.

따라서 현재 integrated top은 **signature 검증 없이 compute가 올라옵니다.**

“secure boot block이 있다”와 “칩이 secure boot된다”는 전혀 다른 이야기입니다.

현재는 전자입니다.

---

## 14. JTAG/MBIST/UCIe/NoC도 대부분 standalone block입니다

제가 module instantiation을 추적했을 때 다음 주요 블록들이 B0 제품 hierarchy에 실질적으로 연결되지 않습니다.

- `fx_jtag_tap`
- `fx_mbist_ctrl`
- `fx_secure_boot_ctrl`
- `fx_ucie3_vendor_macro`
- `fx_noc_mesh_2x2`
- `fx_collective_reduce`
- `fx_tensor_fabric_prod`
- `fx_watchdog`
- `fx_reset_sync`

특히 UCIe macro가 instantiate되지 않으므로 `fx_dual_die_package_top`의 “두 die”는 실제 chiplet interface를 사이에 둔 구조가 아니라 **동일 clock 아래** **`fx_accel_core`** **두 개를 병렬 배치한 논리적 모델**입니다.

따라서 package/interface 문서의 UCIe 3.0/CRC/retry 구조가 아직 product RTL에 구현되었다고 볼 수 없습니다.

---

## 15. CDC/RDC도 sign-off 수준이 아닙니다

흥미롭게도 제대로 된 `fx_reset_sync` 모듈은 존재합니다.

그런데 정작 top에서는 사용하지 않고:

`fusionx_b0_top.sv:63,67,72`

에

```
mgmt_rst_n && pll_locked && !emergency_reset
```

이라는 combinational expression을 그대로 core-domain asynchronous reset으로 넣습니다.

`emergency_reset`은 mgmt clock domain에서 만들어지고 `pll_locked`도 asynchronous status 성격이므로 **reset deassertion/RDC 검증이 반드시 필요한 부분**입니다.

Production RTL이라면 일반적으로 asynchronous assert / synchronous deassert reset tree와 domain별 reset controller가 필요합니다.

현재는 “reset sync block을 만들어 놓았지만 핵심 top에서 쓰지 않는” 상태입니다.

---

## 16. Thermal/power 제어도 신호만 있고 실동작이 없습니다

`fusionx_b0_top`에는:

```
gate_noncritical
thermal_irq
```

가 생성됩니다.

그런데 검색해 보면 `gate_noncritical`은 **어떤 clock gate에도 연결되지 않습니다.**

`thermal_irq`도 top의 최종 `irq`에 OR되지 않습니다.

즉 thermal controller가 throttle condition을 검출해도 실제로는:

- DVFS level만 PLL wrapper에 전달
- emergency reset은 전달
- “noncritical clock gating”은 **미구현**
- thermal interrupt propagation도 **미구현**

상태입니다.

HBM의 `init_done`, `fatal_error`도 package top에서 받아 놓고 동작/`ras_status`에 사용하지 않습니다.

RAS가 상당히 skeleton 수준입니다.

---

## 17. Verification 양도 제품 기준으로는 턱없이 부족합니다

전체 SV RTL은 약 **5천 줄**입니다. 76개 안팎의 module을 합친 수치입니다.

LOC만으로 품질을 판단하면 안 되지만, 18 POPS급 dual-die/HBM accelerator가:

- compute
- memory
- NoC
- sequence
- MoE
- security
- DFT
- power
- host
- video

모두 합해서 이 정도라는 것은 이미 **reference implementation 밀도**에 가깝습니다.

SV directed TB는 6개인데 각각 대체로 **7\~12줄** 수준입니다.

Formal은:

- FIFO
- MTP

두 개뿐이며, 실제 formal tool로 실행되지도 않았습니다.

UVM도 소스는 있지만 smoke sequence에서 실제 randomization을:

```
opcode inside {FX_OP_NOP, FX_OP_FENCE}
```

로 제한하고 있습니다.

즉 실행한다고 해도 GEMM, DMA, softmax, attention, GDN 등을 검증하는 UVM environment가 아닙니다.

---

# 그래도 잘한 부분

전부 부정적인 건 아닙니다.

**첫째**, HBM4/PCIe/UCIe/PLL/signature 같은 proprietary IP를 가짜 behavioral block으로 제품 구현인 것처럼 숨기지 않고 명시적 blackbox로 둔 것은 좋은 설계 습관입니다.

**둘째**, ECC72/64, async FIFO, handshake FSM, DMA, scratchpad arbitration 등은 최소한 RTL engineer가 알아볼 수 있는 실제 구현 형태입니다. 단순 PPT architecture 패키지는 아닙니다.

**셋째**, README와 STATUS에서 스스로 **“NOT FOR FABRICATION”, “not foundry-ready”**라고 적어 둔 것은 상당히 중요합니다. 제작 가능한 칩이라고 허위 주장하는 자료는 아닙니다.

그래서 이것을 **“쓰레기/가짜 RTL”**이라고 부르지는 않겠습니다.

다만 **“N3P에서 18.45 POPS를 내는 FusionX B0 제품 RTL”로 받아들이면 안 됩니다.**

# 최종 결론

제가 반도체 design review에서 이 패키지를 받았다면 판정은 다음과 같습니다.

> **Architecture/algorithm exploration 및 F0급 proof-of-concept를 계속 개발하는 것은 승인.**
>
> **B0 product RTL 또는 synthesis/tapeout baseline으로의 승격은 거부.**

가장 큰 이유는 PDK나 GDS가 없어서가 아닙니다. 그 단계보다 훨씬 앞에서 이미:

**① product top과 대규모 compute plane이 미통합,**
**② compute feed bandwidth가 128배 부족,**
**③ HBM concurrency가 사실상 1 outstanding/die,**
**④ host data plane 부재,**
**⑤ reference model과 RTL numerical semantics 불일치,**
**⑥ 1.1 GHz에 맞지 않는 scalar/combinational microarchitecture,**
**⑦ security/DFT/UCIe/NoC가 제품 hierarchy에 미통합,**
**⑧ 실제 HDL simulation/formal이 한 번도 실행되지 않음**

이 확인됩니다.

따라서 **현재 상태를 한 문장으로 표현하면 “실제 반도체 설계 요소를 많이 포함한 상당히 잘 꾸며진 reference RTL이지만, 성능 숫자를 실현하는 제품 반도체 설계는 아니다”**가 가장 정확합니다.

특히 **18.45 POPS와 24 TB/s는 현재 RTL 성능이 아니라 architecture parameter를 곱해서 만든 목표치**로 봐야 합니다. 실제 connected RTL에서 그 성능으로 이어지는 데이터 공급·메모리·NoC·pipeline 구조가 없습니다.