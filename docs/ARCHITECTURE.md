# Architecture

이 문서는 프로젝트 사양의 각 섹션이 코드의 어디에 대응되는지, 그리고 스캐폴드가 강제하는 불변식(invariant)이 무엇인지 정리합니다.

## 1. 불변식

1. **IR이 유일한 원본.** `.kicad_sch`, `.kicad_pcb`, BOM, CPL, netlist, Gerber는 전부 `ArtifactRef`로 등록되는 파생물이며 `generated_from_ir_hash`를 갖는다. `CircuitIR.content_hash()`는 `validation`/`artifacts`(설계에 *대한* 상태), `project.workdir`/`project.created_at`(어디서·언제 만들었는가), 그리고 모든 중첩 객체의 `created_at`(`Provenance.created_at`은 기본값이 현재 시각)을 제외한 **설계 내용만** 해시한다 — 같은 설계를 다른 폴더에서 다른 시각에 다시 만들거나 JSON에서 다시 읽어도 해시가 같다(`tests/test_spice_findings_regressions.py`). `SourceRef.retrieved_at`은 어떤 문서 버전을 썼는지의 provenance이므로 해시에 남는다.
2. **모든 중요한 값은 `Traced[T]`.** 값 + 단위 + `Provenance(kind, source, derived_from, inputs, tool, tool_version, note)`. `llm_generated`/`assumption`은 `needs_verification=True`. `derived` 값은 계산기가 직접 `inputs`(역할 → id)를 기록하므로 재계산은 위치가 아니라 역할로 호출을 재구성한다.
3. **검증 상태는 boolean이 아니다.** `worst_status([])`는 `NOT_VERIFIED`이다. 증거가 없으면 PASS가 아니다.
4. **도구가 붙지 않은 ValidationResult는 의견이다.** `ValidationResult.tool`이 있어야 tool-backed. ERC/DRC 결과는 `tools/kicad/cli.py`만 생성하며 KiCad 버전·리포트 경로·검사 대상 파일 해시를 함께 기록한다.
5. **Agent는 IR을 직접 수정하지 않는다.** `AgentResult.proposals: list[IRProposal]`를 돌려주고, `Orchestrator.apply_proposals()`가 유일한 적용 지점이다(향후 GUI 확인/diff 훅). 제안은 증거가 아니다: 검증 결과 없이 제안만 낸 단계는 `NOT_VERIFIED`("N proposal(s) applied, nothing verified")이지 PASS가 아니다.
6. **수리는 재생성 또는 재실행뿐.** `repair/strategies.py`의 `RegenerateArtifact`, `RerunTool`. `NON_REPAIRABLE` 카테고리(`human`, `regulatory_conflict`, `unverified_component_data`, `fab_capability_shortfall`, `circuit_structure_change`)는 `NotRepairableError`로 거부되고 `RepairOutcome.unresolved`에 사유와 함께 보고된다. `RepairLoop`는 모든 행위 전후로 IR 해시를 직접 재고, 해시가 바뀐 행위는 실패("changed the design")로 기록하고 루프를 중단한다(`stopped_reason = "repair strategy mutated the IR"`). 모든 시도(전략, 설명, 전후 해시, 오류, 시각)는 `RepairAgent`가 `repair.loop` ValidationResult의 `details.actions`로 IR에 남긴다(설계 해시 밖).
7. **외부 행위는 승인 필요.** `security.require_approval(ExternalAction, detail)` — 승인은 1회용·행위+세부내용 단위이며 일반화되지 않는다.
8. **KiCad 위반은 심각도와 무관하게 FAIL.** `tools/kicad/cli.py`는 error든 warning이든(사람이 exclusion 처리한 것 포함) 위반이 하나라도 있으면 `FAIL`을 낸다(`details.warning_policy`). warning을 PASS 뒤에 숨기면 `lib_symbol_mismatch`/`lib_footprint_mismatch` 같은 컴파일러 결함도 숨겨진다. 리포트의 컨테이너 키(`sheets[].violations`, `violations`, `unconnected_items`)가 없으면 "0 violations"가 아니라 `ToolExecutionError`다.
9. **레이아웃도 추적된다.** `Placement`/`Track`/`Via`/`Zone`은 `provenance`를 가진다(기본값은 `assumption` "origin not recorded"). `route_naive`는 `derived`/`routing.naive`를 스탬프한다. `review.ir_vs_pcb`는 검증이 필요한(unrecorded/assumption/llm_generated) 레이아웃 항목이 있으면 `NOT_VERIFIED`다. `Net.serves_requirements`로 패드→넷→요구사항 사슬이 이어진다.

## 2. 사양 섹션 ↔ 모듈

| 사양 | 모듈 |
|---|---|
| 4. 요구사항 기반 설계 / 누락 정보 질문 | `ir/requirements.py` (`RequirementSet`, `MissingInformation`), `agents/requirement.py`, `workflow` `MISSING_INFORMATION` 단계 |
| 5. Universal Circuit IR | `ir/project.py` `CircuitIR` 및 하위 모듈 |
| 6. Provenance | `ir/provenance.py` |
| 7. 부품 선정·Datasheet 검증 | `ir/components.py` (`Component.has_authoritative_identity`), `validation/structural.py` `ComponentProvenanceValidator` (LLM 생성 또는 **부재** identity → `NOT_VERIFIED`), `tools/kicad/library.py`, `agents/component.py` |
| 8. 규제·안전성 | `ir/regulatory.py` (`RegulatoryProvenance` 10개 필드), `agents/regulatory.py` (관할권 없으면 `USER_INPUT_REQUIRED`) |
| 9–10. KiCad / 회로도 생성 | `tools/kicad/sexpr.py` (파서/직렬화), `tools/kicad/library.py` (심볼·풋프린트 파싱, `extends` 해석, 라이브러리 루트는 실제 실행되는 kicad-cli 설치본 우선·버전 숫자 정렬), `compilers/pins.py` (IR 핀 번호·전기 타입 ↔ 라이브러리 심볼 대조, 두 컴파일러 공용; `no_connect`는 `"<type>+no_connect"`), `compilers/schematic.py` (IR→`.kicad_sch`, 전역 라벨 연결, 라이브러리 심볼 verbatim 임베드, 연결점 일치·스텁 위 연결점·extent 겹침·stacked pin 거부), `compilers/schematic_layout.py` (`symbol_extent`: 본체 그래픽+핀+스텁+라벨 추정 폭, `layout_pitch`: 모든 extent의 최대 span+gap을 2.54 mm로 올림, 최소 25.4 mm), `compilers/ids.py` (uuid5) |
| 11. 계산 / SPICE | `tools/calc/basic.py` (계산기; 각 계산기는 `Provenance.inputs`에 역할→id를 직접 기록하고 id 개수가 틀리면 `ValueError`; `ROLES`/`ROLE_UNITS`로 역할과 기대 단위를 공개), `tools/calc/recompute.py` (`CALCULATORS` 레지스트리, `derived_values(ir)`는 `ir.parameters`뿐 아니라 부품 `electrical`/SPICE 바인딩 값·params, 기대값 nominal/at/tol, 자극·분석 params, `temperature_c`의 모든 `derived` Traced를 경로별로 순회, `recompute_parameters(ir)` → CALCULATION 단계의 `calc.recompute`: 역할로 호출을 재구성해 rel 1e-9로 대조 — 불일치 `FAIL`(human), 미등록 도구/역할 미기록·불일치/입력 부재/역할에 맞지 않는 단위 `NOT_VERIFIED`), `tools/calc/si.py` (SPICE 숫자 파싱/포맷, ngspice-46의 비정확 파서 모델 `ngspice_reads`; 측정 하니스는 `tests/ngspice_parser_harness.py`), `ir/simulation.py` (`SpiceBinding`(+`ignored_pins`)·`Stimulus`·`AnalysisSpec`·`Expectation`·`SimulationSetup`, 전부 `Traced`), `compilers/spice.py` (IR→`.cir`: title = `project.id`(`$`·따옴표·백틱 거부), `NetKind.GROUND` 넷 하나 → 노드 0, `llm_generated` 값·모델카드·**바인딩/자극/분석/기대값 컨테이너** 거부, 바인딩 없는 부품 거부(`exclude=True`+사유로 제외), R/C/L 값은 `Component.electrical`의 같은 물리량과 일치해야 함(`value_sources`), 넷에 있으나 `pin_order`에 없는 핀은 `ignored_pins`+사유 없이는 거부, `nominal == 0`에 `tol_rel`만 있으면 거부, 분석 카드·`.control`·`.include` 없음, 넷 이름은 러너의 `NODE_RE`와 최종 `validate_deck`로 검사해 컴파일된 netlist는 러너가 *내용* 때문에 거부하지 않음, `<id>.cir.report.json`에 제외 부품·무시 핀·provenance 종류·assumption·inexact 숫자 기록; `analysis_command(spec, setup)`이 `op`/`dc VVIN 0 12 1`/`tran 1u 5m`/`ac dec 10 1 1meg`를 만들고, 기대값 벡터 문법은 `v()`/`i()` + ac 전용 `vp()`/`vr()`/`vi()`(위상·실수·허수; dB는 표현 불가)), `tools/spice/ngspice_shared.py` (KiCad 번들 `ngspice.dll`의 ctypes 러너: 덱 바이트를 한 번 읽어 해시·검증 후 `ngSpice_Circ`로 적재(경로는 명령에 들어가지 않음), `bg_<command>` 워치독, 콜백으로만 실패 판정, rawfile 증거 + API 벡터 대조(workdir 경로를 명령에 못 쓰면 임시 디렉터리에 쓰고 복사, 둘 다 안 되면 `unverifiable`), ControlledExit 후 Reset+Init 복구(실제 DLL 테스트로 검증), `engine_info()`), `tools/spice/rawfile.py` (ASCII/binary rawfile 리더), `tools/spice/stage.py` (`run_spice_for(ir, tools, workdir)`: fresh한 `SPICE_NETLIST`에 모든 분석 실행 → `<workdir>/spice/results.json`(`SPICE_RESULT` artifact, format 2: `engine_info`·`conditions` 포함) + `spice.<expectation>` 결과(측정값·nominal·허용치·보간 bracket·조건·엔진 스탬프·rawfile Evidence) + 요약 `spice` + 설정에서 사라진 기대값의 `NOT_APPLICABLE` 대체 결과; `SimulationAgent`와 `RerunTool`이 공유), `agents/simulation.py` (netlist 컴파일 → `run_spice_for`; 제안 없음; 실행하지 못한 경우 이전 `spice.<id>`를 `NOT_VERIFIED`로 대체), `validation/domain.py` `AnalogBiasValidator` (현재 netlist에 대한 `spice` 결과의 `op`가 모든 비접지 넷에 유한한 전압을 주고, `spice` 요약이 PASS이며, 그 `op` 분석에 대한 기대값이 하나 이상 PASS일 때만 PASS — 수렴만으로는 `NOT_VERIFIED`; `Validator.consumes={"spice"}`로 SPICE 단계 직후 재평가) |
| 12. ERC / DRC | `tools/kicad/cli.py` `run_erc` / `run_drc(schematic_parity=..., refill_zones=True)` + `fresh_artifact` + `run_erc_for` / `run_drc_for(ir, kicad, workdir)` (오케스트레이터와 `RerunTool`이 공유; stale·디스크 불일치 artifact는 `ToolExecutionError`; parity는 `<stem>.kicad_sch`가 보드 옆에 있을 때만 요청·기록; 상태 규칙은 불변식 8) |
| 13. PCB | `ir/pcb.py` (레이아웃 provenance, `layout_items()`), `tools/kicad/geometry.py` (보드 좌표계, 하면 미러), `compilers/pcb.py` (IR→`.kicad_pcb` 20260206; 하면 패드는 자식별 flip 규칙 — 좌표·`rect_delta` y 반전, 층 교환, `chamfer` 모서리 교환, 스칼라 유지, 규칙 없는 자식은 `CompileError`; `pintype`은 `compilers/pins.py`), `tools/kicad/board.py` (리뷰어용 보드 풋프린트 읽기), `tools/routing/naive.py` (placeholder 라우터 — 오케스트레이터는 라우팅하지 않고 IR의 트랙을 컴파일만 한다) |
| 14. 제조 산출물 | `compilers/bom.py` (BOM/CPL; 부재 identity는 `NOT_VERIFIED` 셀), `compilers/gerber.py` (`GerberExporter`/`DrillExporter`: fresh한 PCB artifact → kicad-cli export(`ir.pcb.layers`의 동박층 전부 + fab 세트, `--check-zones`) → `ArtifactRef(files=[...])`), `tools/manufacturing/outputs.py` (`%TF.FileFunction` 기준 `Copper,L1..Ln`+mask+profile 검사, `.gbrjob` manifest 대조, 존 층의 동박 plot에 G36 영역 확인, `check_output_artifact(art, ir)`는 실제 읽은 파일의 `disk_hash()`를 스탬프) |
| 15. JLCPCB | `tools/manufacturing/capability.py` — 모든 한계값이 `Traced`; authoritative가 아니면 `NOT_VERIFIED` |
| 16. Independent Review (14 영역) | `review/areas.py`, `review/reviewer.py` |
| 17–19. 자동 수정 / 제한 / 루프 | `repair/strategies.py`, `repair/loop.py` (max_iterations, fingerprint 기반 oscillation 감지, IR 변형 감지·중단, 모든 시도 기록 → `RepairOutcome.as_validation_result` → `repair.loop`), `agents/repair.py` |
| 20. Optimization | 의도적으로 없음 — authoritative sourcing/cost 파이프라인 이후 |
| 21. Agent 구조 | `agents/` |
| 22. AI 모델 계층 | `llm/` (`LLMClient`, `ModelRouter.for_task`, `UsageTracker`, `OpenRouterClient`) |
| 23. GUI | 미착수. `PipelineState.outcomes`, `ReviewReport`, `RepairOutcome`, `ValidationResult.evidence`가 GUI 데이터 소스 |
| 24. 검증 우선 | `workflow/stages.py` 17단계 순서 |
| 25. 오류 상태 | `ir/validation.py` `ValidationStatus` |
| 26. 추적성 | `Traced.provenance.derived_from`, `Component.serves_requirements`, `Net.serves_requirements`, `Placement/Track/Via/Zone.provenance`, `ArtifactRef.generated_from_ir_hash`, `ValidationResult.artifact_hash/ir_hash`(+`Evidence` 경로·해시: 리뷰어의 BOM/CPL/보드 대조 결과에도 첨부), `repair.loop` 행위 기록 |
| 27. 보안 | `security/approval.py` |

## 3. 데이터 흐름

```
raw request ──RequirementAgent──▶ RequirementSet (+questions)
                                       │ USER_INPUT_REQUIRED? ─▶ stop, ask
                                       ▼
RegulatoryAgent ──▶ RegulatoryState (jurisdiction 필수)
CircuitDesignAgent / ComponentAgent ──proposals──▶ Orchestrator.apply_proposals ──▶ CircuitIR
                                       ▼
default_registry.run(ir) ──▶ ir.validation   (구조 + 도메인별 검증기)
CALCULATION: recompute_parameters(ir) ──▶ ir.validation["calc.recompute"]   (파라미터·시뮬레이션 설정·바인딩의 모든 derived 값을 역할로 재계산·대조)
SPICE: SpiceNetlistCompiler ──▶ <workdir>/<project.id>.cir (+ .cir.report.json) ──▶ run_spice_for: NgspiceShared.run(netlist, kind, <workdir>/spice/<analysis id>/, analysis_command)
       ──▶ <workdir>/spice/results.json (SPICE_RESULT: engine_info + conditions + 모든 벡터) + rawfiles ──▶ ir.validation["spice", "spice.<expectation>", 사라진 기대값의 NOT_APPLICABLE] ──▶ domain.analog.bias 재평가
compilers ──▶ ir.artifacts[kind] = ArtifactRef(generated_from_ir_hash)
   SchematicCompiler ──▶ <workdir>/<project.id>.kicad_sch ──▶ KicadCli.run_erc ──▶ ir.validation["kicad.erc"]
   PCBCompiler       ──▶ <workdir>/<project.id>.kicad_pcb ──▶ KicadCli.run_drc(schematic_parity, --refill-zones) ──▶ ir.validation["kicad.drc"]
                                                              (details: zones_refilled, schematic_parity_checked, schematic_hash, schematic_parity[])
   GerberExporter / DrillExporter (PCB artifact가 fresh할 때만; ir.pcb.layers 층, --check-zones) ──▶ <workdir>/gerber/ ──▶ check_output_artifact(art, ir) ──▶ "mfg.gerber" / "mfg.drill"
IndependentReviewer.review ──▶ ReviewReport (14 결과, 각각 ir_hash 스탬프)
RepairLoop ──▶ 재생성/재실행 → 재리뷰 → RepairOutcome ──▶ ir.validation["repair.loop"] (모든 시도)
RELEASE: ir.validation.overall() == PASS 일 때만
```

Agent 단계의 상태 규칙 (`Orchestrator._agent_stage`): 질문이 있으면 `USER_INPUT_REQUIRED`; `ValidationResult`가 있으면 그 최악값; 제안만 있거나 아무것도 없으면 `NOT_VERIFIED`(제안은 적용되지만 증거가 아니다). 단계가 만든 check id를 `consumes`로 읽는 검증기(`domain.analog.bias` ← `spice`)는 그 단계 직후 다시 실행되어 결과가 `ir.validation`에 추가된다(단계 상태는 agent의 판정 그대로, 메시지에 `re-validated: ...`로 표시).

SPICE 단계의 상태 규칙 (`SimulationAgent` + `tools/spice/stage.py`): `ir.simulation` 없음 / 엔진 없음 / 바인딩된 부품 없음(`NothingToCompileError`) → `NOT_VERIFIED`; 컴파일러가 거부(`CompileError`: 바인딩 없는 부품, `llm_generated` 값·컨테이너, 떠 있는 핀, `electrical`과 다른 SPICE 값 …) → `FAIL`(human); 분석이 실패(ngspice stderr, ControlledExit, 타임아웃 …) → `spice` `FAIL`(human, ngspice의 오류 줄 포함) — 단, *환경*이 막은 경우(`SpiceResult.unverifiable`: rawfile을 쓸 곳이 없음, 죽은 엔진, 코드 모델 미적재 + `A` 소자)는 `NOT_VERIFIED`(설계에 대한 판정이 아님); 기대값은 `|measured − nominal| ≤ max(tol_abs, tol_rel·|nominal|)`로 판정(허용치 없음, 또는 `nominal == 0`에 `tol_rel`만 → `UNRESOLVED`, 벡터 없음 → `FAIL` "vector not produced"); `at`이 샘플 사이에 있으면 보간(dc/tran 선형, ac는 log-주파수·크기 벡터는 log-log)하되 두 이웃 샘플이 `details.bracket`에 기록되고 **두 이웃 모두 허용치 안일 때만 PASS**, 둘 다 같은 쪽으로 밖이면 FAIL, 그 사이면 `UNRESOLVED`("grid too coarse") — 보간 오차가 설계 편차로 보고되지 않는다; netlist가 `assumption` 값/결정 위에 서 있으면 PASS 대신 `NOT_VERIFIED`(FAIL은 FAIL); 요약 `spice`는 기대값 결과의 최악값(기대값이 없으면 `NOT_VERIFIED`); 모든 결과에 `conditions`(부품 값 nominal, 온도 `.temp` 또는 ngspice 기본 27 °C, 코너 미시뮬레이션)와 엔진 스탬프(빌드·코드 모델·설정 해시)가 붙는다; 설정에서 제거·개명된 기대값의 옛 `spice.<id>`는 `NOT_APPLICABLE`로 대체되어 RELEASE를 막지 않는다.

컴파일 단계의 상태 규칙 (`Orchestrator._compile`): `NothingToCompileError`(부품 없음, `ir.pcb` 없음, 내보낼 PCB artifact 없음) → `NOT_VERIFIED`; 그 외 `CompileError`(핀 불일치, 미검증 심볼/풋프린트, 넷에 없는 핀 …) → `FAIL` + 메시지; 다른 예외는 결함이므로 전파. 거부된 종류의 오래된 artifact는 증거로 남지 않도록 제거한다. 빈 회로도는 ERC를 "0 violations"로 통과하므로, 부품이 없는 IR은 회로도를 아예 쓰지 않는다.

## 4. 리뷰어가 FAIL을 내는 방식과 수리 매핑

| 리뷰 결과 `details` | 전략 |
|---|---|
| `{"repair": "regenerate", "artifact": <ArtifactKind>}` 또는 `"artifacts": [...]` | `RegenerateArtifact` → 등록된 컴파일러/exporter로 재생성 (한 행위로 여러 종류; `CompileError`·도구 부재는 실패한 행위로 보고) |
| `{"repair": "rerun_tool", "tool_check": "kicad.erc"\|"kicad.drc"\|"spice"\|"mfg.gerber"\|"mfg.drill"}` 또는 `"tool_checks": [...]` | `RerunTool` → kicad-cli 재실행 / `run_spice_for` 재실행(fresh한 netlist에만) / 출력 형식 검사 재실행 |
| `{"repair": "human"}` 등 `NON_REPAIRABLE`, 또는 `repair` 키 없음 | 거부, `unresolved`에 보고 |

리뷰어가 증거를 읽는 방식: `review.ir_vs_*`는 `ArtifactRef.is_stale`/`matches_disk`(다중 파일 artifact는 `files` 집합 해시), `review.ir_vs_pcb`는 추가로 레이아웃 항목의 provenance(불변식 9); `review.pcb_vs_bom` / `review.pcb_vs_cpl`은 (1) CSV의 reference/designator 집합이 IR과 같고 (2) CSV가 stale이 아니며 (3) 디스크 해시가 기록과 같고 (4) 보드 artifact가 있고 fresh·디스크 일치일 때 `tools/kicad/board.py`로 읽은 보드 풋프린트와 행 단위로 대조한다(BOM: reference·Value·Footprint id, CPL: 위치 1e-4 mm·회전(정규화)·면) — (1)~(3) 위반은 `regenerate`(BOM/CPL), (4)의 stale 보드는 `regenerate`(PCB), 보드 없음은 `NOT_VERIFIED`, fresh한 둘이 불일치하면 컴파일러 결함이므로 `human`; 결과에는 CSV와 보드의 경로·디스크 해시가 `Evidence`로 붙는다. `review.schematic_vs_pcb`는 최신 `kicad.drc`가 (a) 현재 보드에서 실행되었고 (b) parity를 실제로 평가했으며 (c) 현재 회로도(`schematic_hash`)에 대해 평가했고 (d) parity 항목이 0개일 때만 PASS — parity 항목은 KiCad 심각도가 warning이어도 FAIL(`human`); `review.erc`/`review.drc`는 도구 결과를 그대로 통과시키되 FAIL이면 `human` + 위반 타입 목록; `review.manufacturing_outputs`는 gerber/drill artifact의 신선도 + `mfg.*` 결과가 바로 그 파일 집합에 대해 실행되었는지(`artifact_hash` == 기록 해시; `mfg.*`는 실제 읽은 파일의 해시를 찍으므로 편집된 파일의 결과는 절대 맞지 않는다) 대조. 같은 반복 안에서 동일한 행위(예: `re-run kicad.drc`)를 두 리뷰 영역이 요구하면 한 번만 실행한다(`RepairStrategy.describe`).

`review.spice_vs_requirements`: 시뮬레이션 설정/netlist/결과가 없으면 `NOT_VERIFIED`; netlist artifact가 stale·디스크 불일치면 `regenerate`(SPICE_NETLIST); `SPICE_RESULT` artifact가 stale·디스크 불일치이거나 `spice` 결과·`results.json`의 netlist 해시가 현재 netlist와 다르면 `rerun_tool`(`spice`); 기대값이 FAIL이거나 이번 실행의 결과가 없거나 존재하지 않는 요구사항을 가리키거나 **기대값 nominal이 자기가 검증한다는 요구사항의 숫자 값과 다르면**(같은 단위, 기대값 허용치 이내여야 함 — 6 V 요구사항은 4 V nominal로 검증되지 않는다) `human`; 요구사항에 비교할 숫자 값이 없거나 단위가 다르면 "traced but not compared"로 `NOT_VERIFIED`, netlist가 assumption 위에 있으면 `NOT_VERIFIED`; 모든 기대값이 현재 netlist에서 PASS이고 요구사항 값과 일치할 때만 PASS(rawfile·results.json Evidence, `requirement_id` 없는 기대값은 `details.untraced`) — 메시지는 "nominal 부품 값·단일 온도에서만 검증"임을 명시하고 `details.conditions`에 조건을 싣는다.

`review.calculations_vs_design`: provenance 형태 검사(도구·입력 존재) 뒤 리뷰어가 **직접** `recompute_parameters(ir)`를 돌려 그 판정을 통과시킨다(FAIL `human`+불일치 목록, NOT_VERIFIED+재계산 불가 목록); 현재 IR 해시에 대한 저장된 `calc.recompute` 결과와 판정이 다르면 그것도 FAIL(`human`).

검증된 시나리오(`tests/test_vertical_slice.py`): 실행 후 R3(10k, VOUT→GND)를 넷·배치·트랙·SPICE 바인딩과 함께 추가하고 R1을 5k로 바꿔(R2‖R3 = 5k) 요구사항 6 V/3 V가 그대로 성립하게 하면(기대값 nominal은 계산기로 재산출) `ir_vs_schematic / ir_vs_pcb / pcb_vs_bom / pcb_vs_cpl / manufacturing_outputs / spice_vs_requirements`가 `regenerate`로 FAIL → 1회차에 6종 재생성 → 2회차에 ERC/DRC/SPICE/출력 검사 재실행 → 3회차 리뷰 전부 PASS, IR 해시 불변. `tests/test_simulation_stage.py`: 잘못된 nominal(5 V)은 `spice.v_out` FAIL → 리뷰 `human` → 루프가 unresolved로 보고; R1을 20k로 바꾸면 `regenerate` → netlist 재생성 → `rerun_tool` → ngspice 재실행 → 기대값이 정직하게 FAIL(4 V vs 6 V, human)이고 stale한 저장 파라미터는 `calculations_vs_design`도 잡는다; nominal**과 요구사항**을 함께 갱신하면 같은 루프가 2회차에 PASS로 수렴; `results.json` 수기 편집은 `rerun_tool`로 감지·재실행; stale netlist에 대한 `RerunTool`은 거부. `tests/test_spice_findings_regressions.py`: nominal만 재산출하고 요구사항을 두면 리뷰가 `human`으로 거부, 제거된 기대값은 RELEASE를 막지 않음, assumption 값은 어디서도 PASS가 되지 않음, `O'Brien` 같은 workdir도 실행됨, ControlledExit 복구, 파서 모델 500-철자 재측정 등 24건. `tests/test_findings_regressions.py`: 배치 이동만으로 CPL이 FAIL(`regenerate`), CSV 수기 편집 FAIL, 보드와 행 불일치 FAIL(`human`), 보드 없음 NOT_VERIFIED.

## 5. 현재 위치와 다음 단계

**완료 (테스트로 증명됨, `tests/test_vertical_slice.py`, `test_simulation_stage.py`, `test_spice_netlist.py`, `test_ngspice_shared.py`, `test_si.py`, `test_schematic_compiler.py`, `test_pcb_compiler.py`, `test_findings_regressions.py`, `test_spice_findings_regressions.py`):** KiCad s-expression 파서/직렬화기, 라이브러리 심볼/풋프린트 파싱, IR→`.kicad_sch`(실제 ERC 0/0; 2x20 헤더 5개·40넷도 netlist == IR; SPICE 바인딩 없는/제외된 부품은 `exclude_from_sim yes`), IR→`.kicad_pcb`(실제 DRC 0/0 + schematic parity 0; 하면 chamfer/rect_delta flip이 `lib_footprint_mismatch` 없이 통과), naive placeholder 라우팅, gerber/drill export(2층·4층, 존 채움 포함) + 형식 검사, **SPICE** (IR→`.cir` 컴파일러, KiCad 번들 ngspice.dll 러너, rawfile 리더, `SimulationAgent`: 분압기 op/dc, RC 저역통과 tran, RC ac 크기·위상(`ac lin` 격자점)이 실제 ngspice-46에서 돌고 기대값이 계산기 출력과 대조된다 — RC 스텝 응답은 계산기와 ngspice가 1e-6 V 이내로 일치하며 테스트가 그 한계를 그대로 단언한다; `.temp 127`은 `tc1` 저항의 op를 실측대로 바꾼다; ControlledExit 뒤 복구는 실제 DLL에서 `quit`으로 유발해 검증), **CALCULATION** 단계(`calc.recompute`, 역할 기반·시뮬레이션 설정까지 순회), `domain.analog.bias`(실제 op 결과 + PASS한 op 기대값), 오케스트레이터·리뷰어·수리 루프 통합(SPICE 포함). 저항 분압 + 3핀 헤더 하나로 IR→계산 재검증→SPICE→sch→ERC→pcb→DRC→gerber→리뷰→수리가 실제 kicad-cli 10.0.6 + ngspice-46에서 끝까지 돈다.

**아직 아닌 것 (그래서 RELEASE는 항상 `NOT_VERIFIED`):** 규제 조사, authoritative fab capability, LLM 기반 요구사항/설계/부품/시뮬레이션 설정 제안, 실제 라우터(현재 라우터는 DRC를 모르며 fixture 배치에서만 clean), 다중 유닛 심볼·stacked pin 심볼(둘 다 `CompileError`)·전원 심볼(`power_pin_not_driven`은 회로에 power_in 핀이 생기면 즉시 FAIL — warning도 FAIL이므로), `.kicad_pro`(설계 규칙은 KiCad 기본값으로 DRC됨), 레이아웃 provenance의 사용자 확인 흐름(현재는 `ir_vs_pcb` `NOT_VERIFIED`로 표시만), SPICE 모델 라이브러리(반도체 모델은 IR의 authoritative `model_card`로만 들어온다 — 데이터시트/벤더 모델 파이프라인 없음), `.temp` 외의 `.options`, **부품 허용차·온도 코너/Monte Carlo**(모든 기대값은 nominal 값·단일 온도에서만 판정되며 결과와 리뷰가 그렇다고 명시한다 — `Component.electrical["tolerance"]`는 기록만 됨), ac 기대값의 dB 표현(`vp`/`vr`/`vi`만), ac/노이즈 도메인 검증기.

**알려진 한계 (문서화된 추정/미검증):** CPL `Mid X/Y`는 풋프린트 anchor(JLCPCB 중심점 아님); 하면 패드 `thermal_bridge_angle`은 KiCad 소스대로 유지하지만 DRC로 독립 검증되지 않음(양쪽 다 mismatch 없음); 회로도 라벨 폭 추정(글자당 1.27 mm + 2.54 mm)은 보수적 추정이며 전기적 안전성은 별도의 연결점 일치 검사가 보장; warning-FAIL 정책상 사람이 warning을 수용할 방법은 설계 변경뿐(KiCad exclusion도 카운트). SPICE: ngspice.dll은 프로세스 내 싱글턴이라 예기치 않은 access violation은 프로세스 재시작까지 엔진을 죽인다(`validate_deck`가 실측된 crash/poison 덱을 미리 거부); `.spiceinit`/`$SPICE_SCRIPTS/spinit`은 막지 않지만 init 직후의 `set` 목록과 그 해시가 `engine_info`로 모든 `results.json`·`spice` 결과에 남는다; `results.json`은 벡터 전체를 담으므로 긴 tran은 파일이 커진다; ngspice가 1 ULP 어긋나게 읽는 숫자는 `.cir.report.json`·`spice` details의 `inexact_numbers`로 보고할 뿐 거부하지 않는다(6자리 이하 데이터시트 값은 전부 정확); `ngspice_reads`의 13,975-철자 실험실 측정 자체는 저장소에 없고, 대신 측정 하니스와 500-철자 재측정 테스트 + 20개 고정값이 있다; `ac dec`의 격자점은 부동소수점상 정확하지 않으므로(1000.000000000002 Hz) 격자점 판정이 필요하면 `ac lin`을 쓰거나 허용치가 이웃 샘플의 간격보다 커야 한다.

권장 순서:

1. **LLM 연결** — `OpenRouterClient.complete` 구현, `RequirementAgent`의 free-text 추출(structured output → `IRProposal` with `llm_generated`), `CircuitDesignAgent`/`PCBAgent`가 IR 제안(배치·트랙·`SpiceBinding`·`SimulationSetup` 포함)을 내고 컴파일러+DRC+SPICE가 판정(`llm_generated` 값은 netlist 컴파일러가 거부하므로 제안은 검증 후에만 시뮬레이션에 도달).
2. **전원 심볼/PWR_FLAG와 `.kicad_pro` 설계 규칙** — POWER/GROUND 넷에 `power:*` 심볼 emit, `PCBCompiler.design_rules(ir)`를 프로젝트 파일로 써서 fab 한계값이 실제 DRC 규칙이 되게.
3. **Datasheet/부품 데이터 파이프라인** — 문서 아카이브 + `SourceRef.content_hash`, `ComponentAgent`; SPICE 모델 카드(authoritative `model_card`)도 여기서.
4. **규제 조사 파이프라인** — 공식 소스만, `RegulatoryProvenance` 전 필드 채움.
5. **JLCPCB capability 수집** — 공식 페이지 → authoritative `ManufacturingConstraints`.
6. **GUI** — `PipelineState`/`ReviewReport`/`RepairOutcome`를 그대로 노출.

## 6. 환경 메모

- Windows 11, Python 3.12, KiCad 10.0.6 (`%LOCALAPPDATA%\Programs\KiCad\10.0`). `find_kicad_cli()`가 이 경로를 자동 탐색한다.
- 한국어 Windows 콘솔(cp949)에서 kicad-cli 출력을 읽을 때 `UnicodeDecodeError`가 나므로 subprocess는 항상 `encoding="utf-8"`로 실행한다.
- `ngspice.exe`는 없다. SPICE 엔진은 KiCad 번들 `bin\ngspice.dll`(ngspice-46, 빌드 Apr 14 2026)을 `ai_eda.tools.spice.NgspiceShared`(ctypes)로 쓴다; `$NGSPICE_DLL`로 바꿀 수 있고 XSPICE 코드 모델은 `<KiCad>\lib\ngspice\*.cm`(또는 `$NGSPICE_CODEMODEL_DIR`). DLL이 없으면 `SimulationAgent`는 `NOT_VERIFIED`. `ai-eda doctor`가 경로·버전·코드 모델 적재 여부를 출력한다.

ngspice-46(KiCad 번들 DLL)에 대해 실측으로 확인한 사실 (코드가 의존하는 것):

- `ngGet_Vec_Info`는 정적 구조체 하나의 포인터를 돌려주므로 다음 호출 전에 벡터를 복사해야 한다(러너는 즉시 `v_realdata[:n]`으로 복사).
- `ngSpice_Circ`(줄 배열 적재)는 `source`와 같이 `Circuit: <title>`을 출력하고 `listing`이 동작하며 plot에 title이 실린다 — 러너는 파일 경로를 ngspice 명령에 넣지 않고 한 번 읽은 바이트를 그대로 적재한다(`'`, `$`, 백틱이 든 폴더에서도 실행됨; rawfile `write`만 경로가 필요하다).
- `quit`은 ControlledExit(quit_exit=True)를 호출하고 DLL은 reset을 기다린다; `ngSpice_Reset` + `ngSpice_Init` + 설정 + 코드 모델 + 자체 테스트로 복구되어 이후 실행이 정상이다(테스트로 매 실행 검증).
- `ac lin 11 100 200`의 주파수 격자점은 정확한 double(150.0)이지만 `ac dec 10 1 1meg`의 격자점은 1 ULP 어긋난다(1000.000000000002): 격자점 판정에는 `lin`을 쓴다.
- 기본 분석 온도는 27 °C("Doing analysis at TEMP = 27.000000"); `.temp 127`은 `tc1=0.01` 저항을 두 배로 만든다(분압 6 V → 4 V 실측).
- 대화형 분석 명령은 소문자 장치 이름이어야 한다(`dc VVIN …`은 실패, `dc vvin …`은 성공) — `normalise_command`가 소문자화. `bg_<command>`는 전경 명령과 동일한 데이터를 주고 `bg_halt`로 중단 가능(워치독).
- 덱에 분석 카드가 있으면 `bg_<cmd>` 아래에서 무시되므로 `validate_deck`가 거부한다; `.control`/`.include`/`.lib`/따옴표/비ASCII도 거부(따옴표 불균형은 프로세스가 끝날 때까지 DLL을 오염시키고, 비접지 노드가 없는 회로는 access violation).
- 노드 이름은 `NODE_RE`(`[A-Za-z0-9_./+-:#@[]]`)만 온전히 유지된다; 컴파일러가 같은 규칙으로 넷 이름을 거부한다.
- 반환 코드는 증거가 아니다(`ngSpice_Command`는 파싱 오류에도 0); 실패는 콜백(stderr 줄, ControlledExit, `No. of Data Rows` 부재, 새 plot 부재)으로만 판정.
- rawfile에는 `Date` 줄이 있어 바이트 동일성을 요구할 수 없다(증거: 경로+sha256). ASCII rawfile은 15자리, binary는 API 벡터와 비트 동일.
- 숫자 파서 `INPevaluate`는 정확히 반올림하지 않는다(가수 M × pow(10, E)): `10u` → 9.999999999999999e-06, `3.3` → 3.3000000000000003, 2^53 근처 홀수 가수 오독. `tools/calc/si.py`가 이를 모델링해 정확히 읽히는 철자를 고르고, 남는 것은 `inexact_numbers`로 보고한다.
- kicad-cli `sch export netlist --format spice`는 GND를 노드 이름으로 유지하고 모델 없는 심볼에 `__<ref>` 자리표시자를 쓴다(ngspice가 전체 파일을 거부); 심볼에 `(exclude_from_sim yes)`를 주면 사라진다(`dnp`는 무관) — 회로도 컴파일러가 바인딩 없는/제외된 부품에 이를 emit한다.

kicad-cli 10.0.6에 대해 실측으로 확인한 사실 (코드가 의존하는 것):

- 종료 코드: 0 = 검사 실행됨(위반 유무 무관), 5 = `--exit-code-violations`이고 위반(warning 포함) 있음, 3 = 입력 로드 실패(리포트 없음), 1 = 잘못된 플래그/값, 2 = spice netlist에서 모델 라이브러리 없음(파일은 써짐).
- ERC 위반은 `sheets[].violations[]`, DRC는 `violations` / `unconnected_items`(severity error) / `schematic_parity` 세 리스트. 항목의 `type`·`severity`·`excluded`만 안정적이며 `description`과 stdout/stderr는 모두 로컬라이즈(한국어)된다 — 텍스트 매칭 금지. `--severity-all`은 사람이 제외한 위반도 `excluded: true`로 포함하며 우리는 그것도 error로 센다.
- `--schematic-parity`는 `<stem>.kicad_sch`가 보드 옆에 있어야 한다. 없으면 exit 0, stderr에 메시지, `schematic_parity: []` — 즉 조용히 "통과". `run_drc`는 sibling 부재를 `ToolExecutionError`로, 실행 후 비어있지 않은 stderr도 오류로 취급한다(성공 실행의 stderr는 항상 비어 있었다).
- ERC JSON의 `pos`는 실제 값의 1/100 (KiCad 버그, DRC는 정상). 위치를 쓸 일이 생기면 ×100.
- Gerber: 기본은 Protel 확장자(.gtl/.gbl/.gts/.gbs/.gm1 …), `--no-protel-ext`면 전부 `.gbr`. `<stem>-job.gbrjob`는 그 실행이 쓴 파일을 정확히 나열하므로 `export_gerbers`는 manifest로 결과 목록을 만든다. 헤더 `%TF.FileFunction`은 `Copper,L1,Top` / `Soldermask,Top` / `Profile,NP`, manifest는 `SolderMask,Top` / `Profile` 로 철자가 다르다. drill export는 구멍이 없는 보드에도 `<stem>.drl`(M48…M30)을 쓴다.
- 모든 export(gerber/drill/gbrjob/pos/netlist)에는 타임스탬프가 들어가므로 바이트 동일성은 우리가 컴파일한 `.kicad_sch`/`.kicad_pcb`에만 요구한다. artifact 해시는 "어떤 파일을 검사했는가"의 증거다.
- `pcb export pos`는 기본 단위가 inch, PosY는 보드 좌표의 부호 반전, `--smd-only`는 THT(핀 헤더) 제외 — CPL 대조에 쓸 때 주의(현재 리뷰어는 `pos` 대신 `.kicad_pcb`의 `(footprint (at ..) (layer ..))`를 직접 읽는다).
- `pcb drc --refill-zones`와 `pcb export gerbers --check-zones`는 stderr를 비운 채(parity 병행 시에도) 존을 메모리에서 채운다; 없으면 DRC는 존 동박을 못 보고(존만으로 이어진 GND는 `unconnected_items`) gerber에는 G36 영역이 전혀 없다.
- 하면 풋프린트의 `PAD::Flip(TOP_BOTTOM)`: `(chamfer top_left top_right)`는 `bottom_*`로, `(rect_delta x y)`는 y 반전, custom `primitives`는 y 반전, 나머지 스칼라 유지 — 각각 `lib_footprint_mismatch` warning 유무로 실측(L_Bourns_SDR0604, Analog_QFN-28-36-2EP, AMS_LGA-10-1EP, BatteryHolder_Keystone_1057).
- KiCad 리포트 severity는 `error`/`warning`; `--severity-all`로 exclusion도 포함된다. ERC 컨테이너는 `sheets[].violations`, 이 키가 없으면 우리는 오류로 취급한다.
- `%LOCALAPPDATA%\Programs\KiCad\<ver>`의 `<ver>`는 문자열 정렬로 `9.0 > 10.0`이 되므로 숫자 튜플로 정렬한다(`cli.version_key`).
