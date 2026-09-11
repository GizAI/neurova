# FusionX-D2 반도체 설계 DB 심층 기술 평가 보고서
**Gemini Architectural & Silicon Feasibility Evaluation**

---

- **평가 대상**: `FusionX_D2_Validated_Source_Design_DB` (neurova 하위)
- **평가 일자**: 2026-09-11
- **평가자**: Gemini (AI Architecture & Systems Evaluator)
- **문서 버전**: 1.0 (Final)
- **공식 분류**: 종합 아키텍처 및 실리콘 양산성 기술 평가 (Technical Feasibility Audit)

---

## 1. 종합 판정 (Executive Verdict)

| 평가 영역 | 평가 등급 | 핵심 요약 |
| :--- | :---: | :--- |
| **알고리즘 및 수치 정합성** | **A+ (최상)** | FP4/FP8/Q8 수치 규격 및 트랜스포머 연산 파이프라인 완벽 검증 |
| **소프트웨어-하드웨어 협력 설계** | **A (우수)** | 디스크립터 링버퍼, C 런타임, Python 컴파일러 로우링 일관성 완비 |
| **RTL 아키텍처 (기능 수준)** | **B+ (양호)** | FSM 상태 머신, 메모리 핸드셰이크, 타일링 순회 로직 정교함 |
| **하드웨어 합성성 (Synthesizability)** | **D (미흡)** | 조합 논리 내 나눗셈/제곱근 While 루프로 인한 타이밍 클로저 불가 |
| **물리 설계 및 실리콘 양산성** | **F (불가)** | 파운드리 PDK, GDSII, P&R, STA, DFT 등 백엔드 전무 |

> **최종 결론**:  
> 본 DB는 **"트랜스포머 연산 가속을 위한 논리적 파이프라인과 소프트웨어 인터페이스를 검증한 우수한 '기능 레벨 참조 모델(Behavioral Reference IP)'"**이다.  
> 그러나 **"현재 상태 그대로 파운드리(TSMC/삼성 등)에서 실물 칩으로 생산(Tape-out)하는 것은 불가능"**하며, 실제 실리콘을 제조하려면 연산 유닛의 다단 파이프라이닝 및 물리 설계(P&R)가 필수적이다.

---

## 2. 세부 기술 분석 (Detailed Analysis)

### 2.1 강점: 알고리즘 구현과 풀스택 인터페이스 완성도
1. **정교한 타일링 FSM (`fx_tensor_engine.sv`)**:
   - `IDLE → CLEAR → A_START → B_START → STEP → WRITE_REQ → NEXT_TILE`로 이어지는 64×64 타일 순회 및 AXI 핸드셰이크(`valid`/`ready`)가 논리적으로 빈틈없이 설계됨.
   - 불규칙한 $M/N$ 경계에 대한 스트로브(`edge_strb`) 패딩 마스킹 처리가 깔끔함.
2. **저비트 정밀도 지원 (`fx_operand_unpacker.sv`)**:
   - FP4(E2M1), FP8(E4M3FN), INT4, INT8 피연산자 언패킹 및 스케일링이 모듈화되어 하드웨어 자원을 효율적으로 쓰도록 구조화됨.
3. **소프트웨어-RTL 동기화**:
   - C 런타임(`fusionx_d2_runtime.c`), 부트 ROM, PCIe 드라이버 스켈레톤, JSON ABI가 단일 규격으로 일치하며, Python 컴파일러 프로파일(Qwen, DeepSeek, GLM 등)을 통해 실제 모델 그래프를 이 명령어로 로우링할 수 있음.

---

### 2.2 결정적 결함: 실제 하드웨어 합성 시의 치명적 문제점

#### ① 순수 조합 논리(`always_comb`) 내의 나눗셈 및 제곱근 While 루프
* **대상 모듈**: `rtl/stream/fx_rmsnorm32.sv`, `rtl/stream/fx_softmax32.sv`
* **문제 코드**:
  ```systemverilog
  always_comb begin
    for(i=0;i<32;i=i+1) begin ... sumsq=sumsq+xi*xi; end
    mean=(sumsq/32)+epsilon_q16_16; 
    root=isqrt(mean); // <-- 64비트 정수 제곱근 While 루프
    ...
    t=((xi*gi)<<8)/$signed(root); // <-- 정수 나눗셈
    y[...] = t[...];
  end
  ```
* **영향**:
  - 클럭(`clk`) 지연 없이 **단 1사이클 내에 32개 곱셈 + 누적 + 제곱근(While 반복) + 나눗셈**을 끝내도록 설계됨.
  - 실제 실리콘에서 이는 수백~수천 개의 게이트 딜레이(Logic Depth)를 발생시켜, **최대 동작 주파수($F_{max}$)가 수 MHz 수준으로 폭락**하거나 셋업 타임(Setup Time) 위반으로 타이밍 클로저가 100% 실패함.
  - **해결 방안**: CORDIC 알고리즘 또는 파이프라인 뉴턴-랩슨(Newton-Raphson) 역제곱근 유닛을 도입하여 최소 6~10클럭 파이프라인으로 쪼개야 함.

#### ② 20만 개 플립플롭의 무분별한 선언 (`fx_mxu_array.sv`)
* **문제 코드**:
  ```systemverilog
  logic signed [ACC_W-1:0] acc [0:ROWS-1][0:COLS-1]; // 48비트 x 4,096개 = 196,608 플립플롭
  ```
* **영향**:
  - 전용 시스톨릭 어레이(Systolic Array) 구조나 전용 레지스터 파일(RF)/SRAM 매크로를 쓰지 않고 단일 모듈에 20만 개의 레지스터를 직접 배치.
  - ASIC P&R(배치 및 배선) 단계에서 극심한 라우팅 혼잡(Routing Congestion)과 막대한 면적 오버헤드, 심각한 클럭 트리 스큐(Clock Tree Skew)를 초래함.

#### ③ 합성 툴 비호환 SystemVerilog 문법
* `longint`, `while` 루프, `function automatic` 등 시뮬레이터(Icarus, Verilator)에서는 구동되나 상용 논리 합성기(Synopsys Design Compiler 등)에서는 비합성(Non-synthesizable) 에러를 유발하거나 비효율적인 게이트 폭발을 일으키는 문법이 다수 존재.

---

### 2.3 백엔드 물리 설계(Physical Design) 및 파운드리 관점

* **PDK(공정 설계 키트) 부재**: TSMC N3P(3nm)를 타깃으로 명시했으나, 실제 TSMC의 표준 셀 라이브러리(.lib, .lef), I/O 패드 매크로가 바인딩되지 않음.
* **상용 필수 IP 부재**: 칩이 동작하기 위해 필수적인 HBM4 메모리 컨트롤러/PHY, PCIe 6.0/CXL 컨트롤러, PLL/클럭 발생기 IP가 목업(Mockup) 상태임.
* **마스크 산출물 부재**: 파운드리에 납품할 최종 레이아웃 파일인 **GDSII 또는 OASIS 파일이 전혀 존재하지 않음**.
* **DFT(테스트 용이화) 부재**: 제조 후 웨이퍼 불량을 걸러낼 스캔 체인(Scan Chain), ATPG, MBIST 회로가 삽입되지 않음.

---

## 3. 실제 생산(Silicon Tape-out)을 위한 4단계 로드맵

현재의 PoC 수준에서 실제 작동하는 실리콘 칩을 생산하기 위해 필요한 단계와 예상 비용:

```
[현재: D2 Source DB] 
        │
        ▼
[Phase 1] RTL 재설계 (Pipelining) : RMSNorm/Softmax 10단 파이프라인화, 시스톨릭 어레이화 (2~3개월)
        │
        ▼
[Phase 2] FPGA 에뮬레이션 : AMD Versal / Xilinx HBM 보드에서 100MHz 실시간 검증 (1~2개월)
        │
        ▼
[Phase 3] 파운드리 IP 라이선스 & 논리합성 : TSMC PDK 바인딩, Synopsys DC 합성, STA (약 30~50억 원)
        │
        ▼
[Phase 4] 물리 설계(P&R) & 마스크 제작 : 디자인하우스 외주, DRC/LVS 클린, MPW 테이프아웃 (약 50~100억 원)
```

---

## 4. 리소스 및 비용 평가 요약

* **현재까지 투입된 비용 (추정)**:
  * 실질 지출: AI 페어 프로그래밍 API 비용 및 로컬 컴퓨팅 비용 **약 50만 ~ 100만 원 수준**.
  * 산출물의 시장 가치: 시니어 칩 아키텍트 1.5~2인월(M/M) 분량의 **초기 아키텍처 IP 가치 (약 3,000만 ~ 5,000만 원 상당)**.
* **평가 총평**:
  * 적은 비용으로 매우 정교한 트랜스포머 가속기 논리 청사진을 뽑아낸 뛰어난 소프트웨어-하드웨어 Co-design 성과물이다.
  * 다만 "하드웨어 생산" 관점에서는 이제 막 개념 설계를 마친 단계이므로, **"실리콘 생산 가능"으로 오인하지 않고 "고도화된 RTL 기능 레퍼런스"로 다루는 것이 현실적이고 정확한 접근**이다.
