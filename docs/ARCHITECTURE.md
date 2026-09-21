# Architecture

이 문서는 프로젝트 사양의 각 섹션이 코드의 어디에 대응되는지, 그리고 스캐폴드가 강제하는 불변식(invariant)이 무엇인지 정리합니다.

## 1. 불변식

1. **IR이 유일한 원본.** `.kicad_sch`, `.kicad_pcb`, BOM, CPL, netlist, Gerber는 전부 `ArtifactRef`로 등록되는 파생물이며 `generated_from_ir_hash`를 갖는다. `CircuitIR.content_hash()`는 `validation`/`artifacts`를 제외한 설계 내용만 해시하므로, 검증을 돌려도 설계 해시는 바뀌지 않는다.
2. **모든 중요한 값은 `Traced[T]`.** 값 + 단위 + `Provenance(kind, source, derived_from, tool, tool_version, note)`. `llm_generated`/`assumption`은 `needs_verification=True`.
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
| 11. 계산 / SPICE | `tools/calc/`, `tools/spice/runner.py` (`SpiceResult`에 engine_version + netlist_hash); `compilers/spice.py`는 stub |
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
tools.calc ──▶ ir.parameters[Traced derived]
compilers ──▶ ir.artifacts[kind] = ArtifactRef(generated_from_ir_hash)
   SchematicCompiler ──▶ <workdir>/<project.id>.kicad_sch ──▶ KicadCli.run_erc ──▶ ir.validation["kicad.erc"]
   PCBCompiler       ──▶ <workdir>/<project.id>.kicad_pcb ──▶ KicadCli.run_drc(schematic_parity, --refill-zones) ──▶ ir.validation["kicad.drc"]
                                                              (details: zones_refilled, schematic_parity_checked, schematic_hash, schematic_parity[])
   GerberExporter / DrillExporter (PCB artifact가 fresh할 때만; ir.pcb.layers 층, --check-zones) ──▶ <workdir>/gerber/ ──▶ check_output_artifact(art, ir) ──▶ "mfg.gerber" / "mfg.drill"
IndependentReviewer.review ──▶ ReviewReport (14 결과, 각각 ir_hash 스탬프)
RepairLoop ──▶ 재생성/재실행 → 재리뷰 → RepairOutcome ──▶ ir.validation["repair.loop"] (모든 시도)
RELEASE: ir.validation.overall() == PASS 일 때만
```

Agent 단계의 상태 규칙 (`Orchestrator._agent_stage`): 질문이 있으면 `USER_INPUT_REQUIRED`; `ValidationResult`가 있으면 그 최악값; 제안만 있거나 아무것도 없으면 `NOT_VERIFIED`(제안은 적용되지만 증거가 아니다).

컴파일 단계의 상태 규칙 (`Orchestrator._compile`): `NothingToCompileError`(부품 없음, `ir.pcb` 없음, 내보낼 PCB artifact 없음) → `NOT_VERIFIED`; 그 외 `CompileError`(핀 불일치, 미검증 심볼/풋프린트, 넷에 없는 핀 …) → `FAIL` + 메시지; 다른 예외는 결함이므로 전파. 거부된 종류의 오래된 artifact는 증거로 남지 않도록 제거한다. 빈 회로도는 ERC를 "0 violations"로 통과하므로, 부품이 없는 IR은 회로도를 아예 쓰지 않는다.

## 4. 리뷰어가 FAIL을 내는 방식과 수리 매핑

| 리뷰 결과 `details` | 전략 |
|---|---|
| `{"repair": "regenerate", "artifact": <ArtifactKind>}` 또는 `"artifacts": [...]` | `RegenerateArtifact` → 등록된 컴파일러/exporter로 재생성 (한 행위로 여러 종류; `CompileError`·도구 부재는 실패한 행위로 보고) |
| `{"repair": "rerun_tool", "tool_check": "kicad.erc"\|"kicad.drc"\|"mfg.gerber"\|"mfg.drill"}` 또는 `"tool_checks": [...]` | `RerunTool` → kicad-cli 재실행 / 출력 형식 검사 재실행 |
| `{"repair": "human"}` 등 `NON_REPAIRABLE`, 또는 `repair` 키 없음 | 거부, `unresolved`에 보고 |

리뷰어가 증거를 읽는 방식: `review.ir_vs_*`는 `ArtifactRef.is_stale`/`matches_disk`(다중 파일 artifact는 `files` 집합 해시), `review.ir_vs_pcb`는 추가로 레이아웃 항목의 provenance(불변식 9); `review.pcb_vs_bom` / `review.pcb_vs_cpl`은 (1) CSV의 reference/designator 집합이 IR과 같고 (2) CSV가 stale이 아니며 (3) 디스크 해시가 기록과 같고 (4) 보드 artifact가 있고 fresh·디스크 일치일 때 `tools/kicad/board.py`로 읽은 보드 풋프린트와 행 단위로 대조한다(BOM: reference·Value·Footprint id, CPL: 위치 1e-4 mm·회전(정규화)·면) — (1)~(3) 위반은 `regenerate`(BOM/CPL), (4)의 stale 보드는 `regenerate`(PCB), 보드 없음은 `NOT_VERIFIED`, fresh한 둘이 불일치하면 컴파일러 결함이므로 `human`; 결과에는 CSV와 보드의 경로·디스크 해시가 `Evidence`로 붙는다. `review.schematic_vs_pcb`는 최신 `kicad.drc`가 (a) 현재 보드에서 실행되었고 (b) parity를 실제로 평가했으며 (c) 현재 회로도(`schematic_hash`)에 대해 평가했고 (d) parity 항목이 0개일 때만 PASS — parity 항목은 KiCad 심각도가 warning이어도 FAIL(`human`); `review.erc`/`review.drc`는 도구 결과를 그대로 통과시키되 FAIL이면 `human` + 위반 타입 목록; `review.manufacturing_outputs`는 gerber/drill artifact의 신선도 + `mfg.*` 결과가 바로 그 파일 집합에 대해 실행되었는지(`artifact_hash` == 기록 해시; `mfg.*`는 실제 읽은 파일의 해시를 찍으므로 편집된 파일의 결과는 절대 맞지 않는다) 대조. 같은 반복 안에서 동일한 행위(예: `re-run kicad.drc`)를 두 리뷰 영역이 요구하면 한 번만 실행한다(`RepairStrategy.describe`).

검증된 시나리오(`tests/test_vertical_slice.py`): 실행 후 R3를 넷·배치·트랙과 함께 추가하면 `ir_vs_schematic / ir_vs_pcb / pcb_vs_bom / pcb_vs_cpl / manufacturing_outputs`가 `regenerate`로 FAIL → 1회차에 5개 재생성 → 2회차에 ERC/DRC/출력 검사 재실행 → 3회차 리뷰 전부 PASS, IR 해시 불변. `tests/test_findings_regressions.py`: 배치 이동만으로 CPL이 FAIL(`regenerate`), CSV 수기 편집 FAIL, 보드와 행 불일치 FAIL(`human`), 보드 없음 NOT_VERIFIED.

## 5. 현재 위치와 다음 단계

**완료 (테스트로 증명됨, `tests/test_vertical_slice.py`, `test_schematic_compiler.py`, `test_pcb_compiler.py`, `test_findings_regressions.py`):** KiCad s-expression 파서/직렬화기, 라이브러리 심볼/풋프린트 파싱, IR→`.kicad_sch`(실제 ERC 0/0; 2x20 헤더 5개·40넷도 netlist == IR), IR→`.kicad_pcb`(실제 DRC 0/0 + schematic parity 0; 하면 chamfer/rect_delta flip이 `lib_footprint_mismatch` 없이 통과), naive placeholder 라우팅, gerber/drill export(2층·4층, 존 채움 포함) + 형식 검사, 오케스트레이터·리뷰어·수리 루프 통합. 저항 분압 + 3핀 헤더 하나로 IR→sch→ERC→pcb→DRC→gerber→리뷰→수리가 실제 kicad-cli 10.0.6에서 끝까지 돈다.

**아직 아닌 것 (그래서 RELEASE는 항상 `NOT_VERIFIED`):** SPICE, 규제 조사, authoritative fab capability, LLM 기반 요구사항/설계/부품 제안, 실제 라우터(현재 라우터는 DRC를 모르며 fixture 배치에서만 clean), 다중 유닛 심볼·stacked pin 심볼(둘 다 `CompileError`)·전원 심볼(`power_pin_not_driven`은 회로에 power_in 핀이 생기면 즉시 FAIL — warning도 FAIL이므로), `.kicad_pro`(설계 규칙은 KiCad 기본값으로 DRC됨), CALCULATION 단계, 레이아웃 provenance의 사용자 확인 흐름(현재는 `ir_vs_pcb` `NOT_VERIFIED`로 표시만).

**알려진 한계 (문서화된 추정/미검증):** CPL `Mid X/Y`는 풋프린트 anchor(JLCPCB 중심점 아님); 하면 패드 `thermal_bridge_angle`은 KiCad 소스대로 유지하지만 DRC로 독립 검증되지 않음(양쪽 다 mismatch 없음); 회로도 라벨 폭 추정(글자당 1.27 mm + 2.54 mm)은 보수적 추정이며 전기적 안전성은 별도의 연결점 일치 검사가 보장; warning-FAIL 정책상 사람이 warning을 수용할 방법은 설계 변경뿐(KiCad exclusion도 카운트).

권장 순서:

1. **SPICE** — `compilers/spice.py` 모델 매핑(R/C/L/V/I부터; 모델은 authoritative provenance 필수), ngspice rawfile 파서, `SimulationAgent`가 요구사항(`v_out` 등)과 대조. kicad-cli의 `sch export netlist --format spice`도 후보(모델 라이브러리가 없으면 exit 2, 파일은 써짐).
2. **LLM 연결** — `OpenRouterClient.complete` 구현, `RequirementAgent`의 free-text 추출(structured output → `IRProposal` with `llm_generated`), `CircuitDesignAgent`/`PCBAgent`가 IR 제안(배치·트랙 포함)을 내고 컴파일러+DRC가 판정.
3. **전원 심볼/PWR_FLAG와 `.kicad_pro` 설계 규칙** — POWER/GROUND 넷에 `power:*` 심볼 emit, `PCBCompiler.design_rules(ir)`를 프로젝트 파일로 써서 fab 한계값이 실제 DRC 규칙이 되게.
4. **Datasheet/부품 데이터 파이프라인** — 문서 아카이브 + `SourceRef.content_hash`, `ComponentAgent`.
5. **규제 조사 파이프라인** — 공식 소스만, `RegulatoryProvenance` 전 필드 채움.
6. **JLCPCB capability 수집** — 공식 페이지 → authoritative `ManufacturingConstraints`.
7. **GUI** — `PipelineState`/`ReviewReport`/`RepairOutcome`를 그대로 노출.

## 6. 환경 메모

- Windows 11, Python 3.12, KiCad 10.0.6 (`%LOCALAPPDATA%\Programs\KiCad\10.0`). `find_kicad_cli()`가 이 경로를 자동 탐색한다.
- 한국어 Windows 콘솔(cp949)에서 kicad-cli 출력을 읽을 때 `UnicodeDecodeError`가 나므로 subprocess는 항상 `encoding="utf-8"`로 실행한다.
- ngspice 미설치 → `SimulationAgent`는 `NOT_VERIFIED` 반환.

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
