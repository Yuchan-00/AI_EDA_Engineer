# AI EDA ENGINEER

자연어 회로 설계 요구사항 → 요구사항 분석 → 부품/규제 조사 → **Universal Circuit IR** → 계산/SPICE → KiCad 회로도/PCB → ERC/DRC → 제조 산출물 → 독립 검토 → 자동 수정, 까지를 하나의 검증 우선(verification‑first) 파이프라인으로 수행하는 AI 기반 전자회로 설계 엔지니어링 시스템의 **아키텍처 스캐폴드**입니다.

핵심 원칙: **LLM은 설계의 진실을 결정하지 않는다.** LLM은 제안·조율(orchestration)만 하고, 사실은 IR + provenance + 결정론적 도구(계산기, SPICE, KiCad ERC/DRC, 제조파일 검사) + 독립 리뷰어가 결정합니다.

## 현재 상태 (2026‑09‑21)

| 계층 | 상태 |
|---|---|
| Universal Circuit IR (Pydantic v2) — Project/Requirements/Components/Pins/Nets/Topology/Constraints/PCB/Regulatory/Validation/Artifacts | **구현** (해시·JSON 왕복·조회) |
| Provenance (`user_requirement / authoritative / assumption / derived / llm_generated`) + `Traced[T]` | **구현** |
| Validation 상태 (`PASS / FAIL / NOT_VERIFIED / NOT_APPLICABLE / USER_INPUT_REQUIRED / UNRESOLVED`) + 보수적 집계 | **구현** |
| Validator 레지스트리 (도메인별 선택) + 구조 검증기 3종 | **구현** / 도메인 분석기 5종은 **stub (NOT_VERIFIED)** |
| 결정론적 계산기 (옴의 법칙, 분압, RC, LED 저항) | **구현** |
| KiCad CLI 래퍼 (ERC/DRC JSON → ValidationResult — **warning 포함 위반이 하나라도 있으면 FAIL**, 리포트 컨테이너 키(`sheets`/`violations`/`unconnected_items`)가 없으면 오류(조용한 PASS 금지), `--schematic-parity` 실제 평가 여부 기록, DRC `--refill-zones`·gerber `--check-zones`로 존을 채운 상태를 검사/출력, IR 층 수에 따른 gerber 층 목록, `--no-protel-ext`·`.gbrjob` manifest 기반 export, 버전 디렉터리 숫자 정렬(10.0 > 9.0)) | **구현·실제 KiCad 10.0.6로 검증** |
| KiCad s-expression 파서/직렬화기 (`tools/kicad/sexpr.py`) — 실제 라이브러리·데모 13개 파일 바이트 동일 왕복 | **구현** |
| KiCad 라이브러리 — 심볼(`extends` 해석)·풋프린트 파싱, 핀/패드/courtyard 추출 | **구현** |
| 컴파일러 IR→`.kicad_sch` (전역 라벨 연결, 라이브러리 심볼 verbatim 임베드, 결정론적 UUID, 심볼 실측 크기(본체·핀·스텁·라벨)로 격자 간격 산출 + 연결점 일치/겹침 거부, IR 핀 번호·전기 타입을 라이브러리와 대조) | **구현·실제 ERC 0 error/0 warning** (2x20 헤더 5개 40넷도 netlist == IR) |
| 컴파일러 IR→`.kicad_pcb` (KiCad 10 20260206 형식, 풋프린트 임베드, 트랙/비아/존, 하면 배치 미러링 — 패드 자식별 flip 규칙(chamfer 모서리 교환, rect_delta y 반전)을 실제 DRC로 검증, 규칙 없는 패드 자식은 거부; 패드 `pintype`은 검증된 라이브러리 심볼에서) | **구현·실제 DRC 0/0 + schematic parity 0** |
| 라우팅 | **naive placeholder** (`tools/routing/naive.py`: 패드 중심 직선 체인, DRC 무관 — 검증은 DRC가 함; 생성 트랙에 `derived`/`routing.naive` provenance 스탬프) |
| Gerber/Drill export → `ArtifactKind.GERBER`/`DRILL` 등록 (파일 집합 해시) + 형식 검사(`%TF.FileFunction` 기준, IR 층 수만큼 `Copper,L1..Ln` 요구, manifest 대조, IR에 존이 있는 층의 동박 plot에 채움 영역(G36) 존재 확인; 결과의 `artifact_hash`는 실제 읽은 파일의 해시) | **구현·실제 export로 검증** (2층·4층) |
| ngspice 러너 | 바이너리 탐지·실행 골격, rawfile 파서 stub |
| 컴파일러 IR→BOM/CPL | **구현** (미검증·**부재** MPN/제조사/패키지/공급사는 `NOT_VERIFIED`로 출력, 빈 칸 없음) |
| 컴파일러 IR→SPICE netlist | **stub** (`NotImplementedError`) |
| 제조: JLCPCB capability 모델(미검증 기본) | **구현** (authoritative 데이터 없음 → `NOT_VERIFIED`) |
| Agent 9종 (Requirement/Component/Circuit/Simulation/PCB/Regulatory/Manufacturing/Review/Repair) | 인터페이스 + 결정론적 동작만; LLM 호출 없음 |
| LLM 계층 (클라이언트 프로토콜, task 기반 라우터, 사용량 추적, OpenRouter) | 인터페이스 / OpenRouter 요청 매핑 stub |
| 독립 리뷰어 — 스펙의 14개 검토 영역 | **구현** (artifact 신선도·디스크 해시, BOM/CPL을 **컴파일된 보드 파일의 풋프린트**(reference·value·footprint / 위치·회전·면)와 대조 + CSV 신선도·디스크 해시 + Evidence, 보드 없으면 `NOT_VERIFIED`; 레이아웃 항목(배치·트랙·비아·존)의 provenance 미기록/가정/LLM이면 `ir_vs_pcb` `NOT_VERIFIED`; DRC parity 증거(어떤 회로도·보드에 대해 평가됐는지); gerber/drill 신선도+검사 결과 대조; 부품 identity(`has_authoritative_identity`)) |
| 자동 수리 — `RegenerateArtifact`(복수 artifact), `RerunTool`(ERC/DRC/출력 검사; stale·디스크 불일치 artifact는 거부), 반복 상한, oscillation 감지, 반복 내 중복 행위 제거, 수리 불가 카테고리, **IR 해시 변경 시 행위 실패 + 루프 중단**, 모든 시도를 `repair.loop` ValidationResult로 IR에 기록 | **구현·실제 도구로 수렴 검증** |
| 워크플로우 오케스트레이터 — 17 단계, 질문에서 블로킹, 제안(proposal) 적용(제안만 있고 검증 결과가 없으면 `NOT_VERIFIED`), 컴파일 거부 → `NOT_VERIFIED`(입력 없음)/`FAIL`(IR 불일치) | **구현** |
| 외부 행위 승인 게이트 (git push, 주문, 구매, 규제 제출 …) | **구현** |
| CLI (`doctor / new / run / review`) | **구현** |
| GUI | 미착수 |

테스트 176개 통과 (KiCad 10.0.6 설치 시 실제 ERC/DRC/parity/gerber/drill 실행 포함; 미설치 시 해당 테스트는 skip). `tests/test_findings_regressions.py`는 검증자가 확인한 결함 17건(BOM/CPL 미대조, 회로도 심볼 겹침, 핀 타입 불일치, 하면 chamfer, 버전 정렬, 제안=PASS, mfg 해시, ERC `sheets` 누락, 존 미채움, 4층 gerber, 레이아웃 provenance, MPN 부재, warning 은폐, 수리 기록/변형 등)에 대한 회귀 테스트다.

**검증된 vertical slice** (`tests/test_vertical_slice.py`): 저항 분압 R1/R2 + 3핀 헤더 J1 IR → `.kicad_sch` → ERC 0/0 → `.kicad_pcb`(naive 트랙) → DRC 0/0 + schematic parity 0 → gerber 9층 + drill → 형식 검사 PASS → 독립 리뷰 8개 영역 PASS → RELEASE는 `NOT_VERIFIED`(SPICE/규제/fab capability 증거 없음 — 의도된 결과). 설계 변경(R3 추가) 후 리뷰어가 모든 파생물을 stale로 판정하고, 수리 루프가 IR을 건드리지 않고 재생성·재실행만으로 2회 반복 내에 다시 PASS로 수렴한다.

**알려진 한계 (코드에 `NOT_VERIFIED`/거부로 표시됨):** ERC/DRC warning은 FAIL이며 KiCad 예외(exclusion)도 예외로 인정하지 않으므로 warning을 "수용"하는 길은 설계 변경뿐이다(전원 핀이 생기면 `power_pin_not_driven`이 즉시 FAIL — PWR_FLAG 지원 전까지). CPL의 `Mid X/Y`는 풋프린트 anchor이지 JLCPCB가 요구하는 중심점이 아니다(리뷰어도 anchor 기준으로 대조). 하면 패드의 `thermal_bridge_angle`은 KiCad 소스대로 그대로 두지만 DRC `lib_footprint_mismatch`로는 구별되지 않아 독립 검증되지 않았다. 회로도 라벨 폭은 글자당 1.27 mm로 보수적으로 잡은 추정치다(전기적 연결은 좌표 일치 검사로 별도 보장).

## 설치

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev,llm]"
```

외부 도구: KiCad 10 (`kicad-cli`), 선택적으로 ngspice, `OPENROUTER_API_KEY`.

## 사용

```bash
ai-eda doctor
```

```bash
ai-eda new demo --request "12V 입력을 5V로 변환하는 회로"
```

```bash
ai-eda run projects/demo/ir.json
```

필수 정보가 없으면 파이프라인이 `USER_INPUT_REQUIRED`에서 멈추고 질문을 출력합니다. 답변은 `--answer key=value`로 전달합니다.

```bash
ai-eda run projects/demo/ir.json --answer application="bench supply" --answer jurisdiction=EU
```

```bash
ai-eda review projects/demo/ir.json
```

## 구조

```
ai_eda/
  ir/            Universal Circuit IR + provenance + validation 상태
  validation/    Validator 프로토콜, 레지스트리, 구조/도메인 검증기
  tools/
    calc/        결정론적 계산기
    spice/       SPICE 러너 인터페이스 (ngspice)
    kicad/       s-expression 파서/직렬화, 라이브러리(심볼/풋프린트), 보드 기하, kicad-cli 래퍼(ERC/DRC/export)
    routing/     naive placeholder 라우터 (실제 라우터 아님)
    manufacturing/ fab capability 모델, Gerber/Drill 검사
  compilers/     IR → kicad_sch / kicad_pcb / BOM / CPL / gerber·drill(export) / SPICE netlist(stub)
  agents/        전문 Agent (제안만, 진실 결정 안 함)
  llm/           모델 추상화 (클라이언트, 라우터, 사용량, OpenRouter)
  review/        독립 리뷰어 (14개 영역)
  repair/        결정론적 수리 전략 + 폐쇄 루프
  workflow/      단계 정의 + 오케스트레이터
  security/      외부 행위 승인 게이트
  cli.py
docs/ARCHITECTURE.md   스펙 섹션 ↔ 모듈 매핑, 데이터 흐름, 다음 단계
```

자세한 설계 근거와 다음 단계는 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)를 보세요.
