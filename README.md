# AI EDA ENGINEER

자연어 회로 설계 요구사항 → 요구사항 분석 → 부품/규제 조사 → **Universal Circuit IR** → 계산/SPICE → KiCad 회로도/PCB → ERC/DRC → 제조 산출물 → 독립 검토 → 자동 수정, 까지를 하나의 검증 우선(verification‑first) 파이프라인으로 수행하는 AI 기반 전자회로 설계 엔지니어링 시스템의 **아키텍처 스캐폴드**입니다.

핵심 원칙: **LLM은 설계의 진실을 결정하지 않는다.** LLM은 제안·조율(orchestration)만 하고, 사실은 IR + provenance + 결정론적 도구(계산기, SPICE, KiCad ERC/DRC, 제조파일 검사) + 독립 리뷰어가 결정합니다.

## 현재 상태 (2026‑09‑23)

| 계층 | 상태 |
|---|---|
| Universal Circuit IR (Pydantic v2) — Project/Requirements/Components/Pins/Nets/Topology/Constraints/PCB/Regulatory/Validation/Artifacts | **구현** (설계 해시는 시각·workdir·validation·artifacts를 제외한 설계 내용만 — 같은 설계는 언제 어디서 만들어도 같은 해시; JSON 왕복·조회) |
| Provenance (`user_requirement / authoritative / assumption / derived / llm_generated`) + `Traced[T]` (`derived`는 계산기가 역할→id `inputs`를 기록) | **구현** |
| Validation 상태 (`PASS / FAIL / NOT_VERIFIED / NOT_APPLICABLE / USER_INPUT_REQUIRED / UNRESOLVED`) + 보수적 집계 | **구현** |
| Validator 레지스트리 (도메인별 선택) + 구조 검증기 4종 (`ir.connectivity`·`ir.component_provenance`·`ir.assumptions`·`ir.llm_requirements`; `ir.assumptions`는 파라미터·요구사항뿐 아니라 SPICE 바인딩·자극·분석·기대값·온도의 assumption도 `USER_INPUT_REQUIRED`로 표면화) | **구현** / `domain.analog.bias`는 **실제 SPICE op 결과로 판정**(현재 netlist의 `spice` 결과가 모든 비접지 넷에 유한한 전압을 주고, `spice` 요약이 PASS이며, 그 op에 대한 기대값이 PASS일 때만 PASS — 수렴만으로는 `NOT_VERIFIED`; SPICE 단계 직후 재평가), 나머지 도메인 분석기 4종은 **stub (NOT_VERIFIED)** |
| 결정론적 계산기 (옴의 법칙, 분압, RC τ·스텝 응답·저역통과 크기/위상, 병렬 저항, LED 저항) + 레지스트리 + **CALCULATION 단계**(`calc.recompute`: 파라미터·기대값 nominal·바인딩 값 등 모든 `derived` 값을 등록된 계산기로 **역할 기반** 재계산해 대조 — 불일치 FAIL(human), 미등록 도구/역할 미기록/역할에 맞지 않는 단위 NOT_VERIFIED) | **구현** |
| SPICE 숫자 파싱/포맷 (`tools/calc/si.py`) — ngspice-46의 비정확 숫자 파서(`10u` → 9.999999999999999e-06)를 실측 모델링해 정확히 읽히는 철자를 선택, 남는 것은 `inexact_numbers`로 보고 | **구현·저장소 내 하니스로 DLL에서 500 철자 표본 재측정 + 20 고정값** (13,975-철자 실험실 측정은 저장소 밖) |
| KiCad CLI 래퍼 (ERC/DRC JSON → ValidationResult — **warning 포함 위반이 하나라도 있으면 FAIL**, 리포트 컨테이너 키(`sheets`/`violations`/`unconnected_items`)가 없으면 오류(조용한 PASS 금지), `--schematic-parity` 실제 평가 여부 기록, DRC `--refill-zones`·gerber `--check-zones`로 존을 채운 상태를 검사/출력, IR 층 수에 따른 gerber 층 목록, `--no-protel-ext`·`.gbrjob` manifest 기반 export, 버전 디렉터리 숫자 정렬(10.0 > 9.0)) | **구현·실제 KiCad 10.0.6로 검증** |
| KiCad s-expression 파서/직렬화기 (`tools/kicad/sexpr.py`) — 실제 라이브러리·데모 13개 파일 바이트 동일 왕복(개발 PC 실측; 저장소 안의 왕복 테스트는 인라인 샘플과 자체 컴파일 보드뿐) | **구현** |
| KiCad 라이브러리 — 심볼(`extends` 해석)·풋프린트 파싱, 핀/패드/courtyard 추출 | **구현** |
| 컴파일러 IR→`.kicad_sch` (전역 라벨 연결, 라이브러리 심볼 verbatim 임베드, 결정론적 UUID, 심볼 실측 크기(본체·핀·스텁·라벨)로 격자 간격 산출 + 연결점 일치/겹침 거부, IR 핀 번호·전기 타입을 라이브러리와 대조) | **구현·실제 ERC 0 error/0 warning** (2x20 헤더 5개 40넷도 netlist == IR) |
| 컴파일러 IR→`.kicad_pcb` (KiCad 10 20260206 형식, 풋프린트 임베드, 트랙/비아/존, 하면 배치 미러링 — 패드 자식별 flip 규칙(chamfer 모서리 교환, rect_delta y 반전)을 실제 DRC로 검증, 규칙 없는 패드 자식은 거부; 패드 `pintype`은 검증된 라이브러리 심볼에서) | **구현·실제 DRC 0/0 + schematic parity 0** |
| 라우팅 | **naive placeholder** (`tools/routing/naive.py`: 패드 중심 직선 체인, DRC 무관 — 검증은 DRC가 함; 생성 트랙에 `derived`/`routing.naive` provenance 스탬프) |
| Gerber/Drill export → `ArtifactKind.GERBER`/`DRILL` 등록 (파일 집합 해시) + 형식 검사(`%TF.FileFunction` 기준, IR 층 수만큼 `Copper,L1..Ln` 요구, manifest 대조, IR에 존이 있는 층의 동박 plot에 채움 영역(G36) 존재 확인; 결과의 `artifact_hash`는 실제 읽은 파일의 해시) | **구현·실제 export로 검증** (2층·4층) |
| ngspice 러너 (`tools/spice/ngspice_shared.py`) — KiCad 번들 `bin/ngspice.dll`(ngspice-46)을 ctypes로 구동: 덱 바이트를 한 번 읽어 해시·검증(crash/poison 덱 사전 거부) 후 `ngSpice_Circ`로 적재(파일 경로는 명령에 안 들어감 — `O'Brien` 같은 폴더도 됨), `bg_<command>` 워치독, 실패는 반환 코드가 아니라 콜백(stderr/ControlledExit/plot 부재)으로 판정, 모든 벡터 + ngspice가 쓴 rawfile(sha256)을 증거로, API 벡터와 rawfile 대조(rawfile을 쓸 곳이 없으면 `unverifiable` → NOT_VERIFIED), ControlledExit 후 자동 복구(테스트가 `quit`으로 유발해 검증), `engine_info()`(빌드·코드 모델·init 설정 해시·자체 테스트); `tools/spice/rawfile.py` ASCII/binary rawfile 리더 | **구현·실제 DLL로 검증** (배치 `ngspice` 바이너리 러너도 선택 가능 — 원본 netlist의 해시와 분석 카드를 붙인 사본의 `deck_hash`를 함께 기록하므로 SPICE 단계가 받아들인다) |
| 컴파일러 IR→BOM/CPL | **구현** (identity·공급사 셀은 `authoritative`·`user_requirement` provenance일 때만 값을 인쇄 — 모델 제안·가정·derived 값과 **부재** 값은 `NOT_VERIFIED`, 빈 칸 없음; 사용자가 직접 입력한 MPN은 값이 인쇄되되 `DatasheetHash`가 `NOT_VERIFIED`이고 `ir.component_provenance`/리뷰어도 같은 말을 한다; `DatasheetHash` 열 = MPN을 찾은 아카이브 데이터시트의 sha256, grounding 없으면 `NOT_VERIFIED`; Reference·identity·Footprint·Supplier·SupplierPN 셀과 CPL Designator는 (공백 제거 후) `=`/`+`/`-`/`@`로 시작하거나 제어문자를 담으면 `CompileError` — `Value`/`Description`만 설계의 말이라 그대로) |
| 문서 아카이브 (`tools/sources/`) — 네트워크는 외부 행위: `--online`이 `NETWORK_FETCH` "online session" 승인을 1회 grant+consume, 없으면 소켓을 열기 전에 거부; 신뢰 origin만(KiCad 라이브러리 `Datasheet` 호스트 / 규제 후보 목록의 공식 도메인 / `--trust-host` / 사용자가 준 정확한 URL — 모델이 말한 URL은 절대 아님), https만, 리다이렉트는 같은 origin·신뢰 호스트로만, URL당 실행당 1회·20 s·프로젝트 UA; 404→`missing`, 403/429/봇 차단 문구·PDF 자리에 HTML·본문 없는 HTML→`blocked`, 그 외→`error`(모두 `fetch_log.jsonl`에 기록); 저장은 `<sha256>.<ext>`+`.meta.json`(URL·최종 URL·리다이렉트·시각·content type·추출기 버전·텍스트 해시), 읽을 때마다 sha256 재검증(불일치 = `tampered`); 텍스트 추출 pypdf 페이지별/html.parser/xml.etree/text; `find_quote`는 요구사항 추출과 같은 정규화(문자 정확·공백 무시·토큰 경계) | **구현·로컬 loopback fake 서버(`tests/fake_sources.py`: Cloudflare/Akamai/unblock/JS shell/iframe shell/404/리다이렉트 체인/타임아웃)로 검증** (`tests/test_archive.py`); 실 사이트 테스트는 `AI_EDA_ONLINE=1`일 때만 |
| 부품 존재 검사 (`parts/existence.py` → `component.existence.<ref>`, tool `parts.existence`) — 심볼/풋프린트를 라이브러리 파일에서 해석(없으면 FAIL), datasheet pointer(사용자 URL → IR SourceRef → KiCad `Datasheet` 속성 순, 모델 URL 아님), 아카이브·sha256 재검증(offline/blocked/missing/refused/tampered → `NOT_VERIFIED`), **데이터시트는 PDF만** — HTML/XML/텍스트 문서(제품 페이지, 뷰어 셸, `<noscript>`의 "enable JavaScript"/"access denied"는 `blocked`)는 pointer의 증거로 아카이브되되 `NOT_VERIFIED`("not a datasheet")이고 그 위에서는 MPN도 사실도 grounding되지 않음, **MPN을 아카이브 본문에서 verbatim으로**(대소문자·공백 무시·온전한 식별자 — `LM2596S-5`는 `LM2596S-5.0/NOPB`에서 발견되지 않고 `LM2596S-5.0`은 발견된다(슬래시는 옵션 접미사 구분자); 못 찾으면 `NOT_VERIFIED`, 절대 FAIL·짧은 번호 추측 없음), 카탈로그 행(같은 MPN의 행이 여럿이면 IR의 제조사/패키지와 맞는 행이 sourcing을 뒷받침, 모든 행이 다르면 `NOT_VERIFIED`·sourcing 없음); `ComponentAgent`는 찾은 MPN만 authoritative로 재태깅(note에 추출기·버전·텍스트 해시 스탬프)하고 MPN을 지어내지 않으며 데이터시트 SourceRef의 authority는 IR의 authority → authoritative/user 제조사 → pointer 호스트 순(모델이 말한 제조사는 절대 아님); `ir.component_provenance`/`review.component_provenance`는 같은 규칙(`parts/identity.py` `mpn_grounding`)으로 **표시만 된 authoritative MPN을 `NOT_VERIFIED`로 강등**하고, 아카이브 항목(`<sha256>.meta.json`이 있는 디렉터리의 파일)만 사본으로 인정하며(해시가 맞는 임의의 파일은 `unarchived`), 검사 시점에 **MPN을 기록된 페이지에서 직접 다시 찾고**(없으면 `not in text`) 현재 추출기의 텍스트가 기록과 다르면 `re-extracted`, 변조된 사본은 `tampered`; `component.fit`은 항상 `NOT_VERIFIED`(부품 적합성 8기준은 이 버전에서 평가되지 않음) | **구현·합성 KiCad 라이브러리 + fake 서버 + 결정론적 PDF 픽스처로 검증** (`tests/test_parts_existence.py`, `test_component_agent.py`, `test_parts_regulatory_e2e.py`, `test_parts_regulatory_findings_regressions.py`) |
| Datasheet 사실 grounding (`parts/datasheet_facts.py`) — 사용자 JSON(`--answer datasheet_facts_file=`) 또는 모델(`extract_datasheet_facts=yes`, 과금)의 `DatasheetFact(key, value[, value_high], unit, page, quote)`는 (1) 그 페이지에 verbatim 인용이 있고 (2) 수량 파서가 인용을 **페이지의 수량 하나 전체**로 다시 읽으며(`-40 to 125 degC`의 `125 degC`는 범위 조각이라 거부; 범위는 `value`..`value_high`) (3) 단위가 키의 계열에 맞고(`v_*` V, `i_*` A, `power_rating` W, `tolerance` %, `operating_temperature` degC …; 계열을 모르는 키는 거부) (4) 모델의 값·단위가 그 파싱과 rel 1e-9·canonical 단위로 일치하고 (5) `package`는 인용에 **그 부품의 MPN**이 함께 있을 때만 통과(다른 행만 인용하면 차단; 여러 행을 함께 인용하면 차단되지 않으므로 사용자 파일의 사실은 그 부품의 한 행만 인용할 것); 사용자 파일의 사실은 즉시 authoritative `Traced`(section='page N', note에 인용·파싱·추출기 스탬프), **모델의 사실은 grounding이 숫자의 *의미*(0.6 V가 v_max인지 dropout인지)를 대조하지 못하므로 표로만 제시**(`confirm_facts`, 필수 질문) → **다음 실행**의 `confirm_facts=yes`로만 IR 진입(그 전에는 IR에 아무것도 없음; `no`는 폐기; 캐시 `<workdir>/parts/facts.json`), 거부는 사유와 함께 목록, identity 키 예약 | **구현·검증** (`tests/test_datasheet_facts.py`, `test_component_agent.py`) |
| 카탈로그 (`parts/catalog.py`, `--catalog CSV --catalog-date ISO`) — 사용자 CSV export(mpn/manufacturer/package 필수, JLCPCB/LCSC 헤더 alias 매핑 — `mfr`는 별도 manufacturer 열이 있을 때만 MPN, 사용자 선언 날짜, 파일 sha256이 content_hash) → `SourcingInfo` authoritative; 유통사 데이터이지 identity가 아님; `=`/`+`/`-`/`@`로 시작하거나 제어문자를 담은 셀(CSV formula injection)은 행 전체를 사유와 함께 건너뛰고 supplier part number는 허용 문자 집합만, BOM 컴파일러도 그런 identity/sourcing 셀은 `CompileError`로 거부 | **구현·검증** (`tests/test_catalog.py`) |
| LLM 부품 후보 (`agents/component.py`, `--llm`) — identity 없는 부품에 대해 후보(제조사·MPN·KiCad 심볼·풋프린트·근거)를 제안 → 라이브러리 게이트(디스크에 있는 심볼/풋프린트, 기존 것과 다르면 거부, 지시문 drop) → 표 제시 → **다음 실행**의 `confirm_parts=yes`로 사용자의 선택(`user_requirement`) — **행 단위**: 이전 실행의 표에 있던 (ref·제조사·MPN·심볼·풋프린트) 행만 확인되고, 그 뒤에 라이브러리가 고쳐져 새로 통과한 후보는 `(new)`로 다시 표에 올라 기다림; identity는 데이터시트 grounding 전까지 `llm_generated`; 모델의 datasheet URL은 기록만·절대 fetch 안 함; 근거 문장은 사람용 산문이지 적합성 평가가 아님; 캐시 `<workdir>/parts/candidates.json` | **구현·대본 클라이언트로 검증** |
| 규제 후보·적용 여부·출처 (`regulatory/`, `agents/regulatory.py`) — 큐레이션 `candidates.json`(EU LVD/EMC/RoHS/RED, KR 전기용품안전관리법·시행규칙·전파법 58조의2·적합성평가 고시(`fetchable: false`), US 47 CFR 15; 모든 URL은 항목의 `allowed_domains` 안, 규칙은 선언적, 인용은 verbatim, 최상위 provenance "unverified until fetched and grounded"); 범위 질문 `intended_use`/`mains_powered`/`highest_rated_voltage`/`evaluation_kit`/`radio`/`finished_apparatus`/`digital_device`는 non-blocking으로 묻고 `--answer`로 답(`scope_answers`에 기억; yes/no 질문에 yes/no가 아닌 답(`wifi and bluetooth`)은 미결로 남고 다시 묻는다); 적용 여부는 답변 + IR 요구사항으로 결정론적 판정하되 **`user_requirement`/`authoritative` provenance의 요구사항만 읽고**(모델 추출·가정·derived 값은 "확인하거나 직접 답하라"는 USER_INPUT_REQUIRED), AC/DC는 영문이면 값에 붙은 말(`12 V DC`, `DC 60 V`, `60 V (DC)`, `230VAC`)에서만(산문의 "AC adapter"는 아님), 한국어 교류/직류는 값·요구사항 문장 어디서든(provenance note의 모델 문장은 어느 쪽도 읽지 않음), 아니면 `mains_powered`, 둘이 모순이면 미결; **LVD/KC의 전압 규칙은 장비의 정격으로 판정** — `input_voltage`·`output_voltage` 요구사항과 사용자가 답한 `highest_rated_voltage`(제품 어디서든 최고 전압) 중 하나라도 대역 안이면 `APPLICABLE`, `NOT_APPLICABLE`은 전부 대역 밖이고 `highest_rated_voltage`가 답해졌을 때만(12 V 입력만으로는 `NOT_APPLICABLE`이 아니라 그 답을 요구; 12 V-in/400 V-out은 in scope); `evaluation_kit=yes`는 인용된 LVD Annex II / EMC Article 2(2)(e) 제외를 판정하고, 각 항목의 `not_evaluated`(RoHS Article 2(2)/2(4) 제외 등 규칙이 다루지 않는 것)가 rationale과 `regulatory.applicability` 메시지에 실린다; `regulatory.sources` PASS = 공식 문서 아카이브 + 모든 인용 발견(`expected_markers`로 문서 동일성), 인용 미발견 = FAIL(목록 오류), 오프라인 = 이전 실행 사본 재사용 아니면 `not fetched (offline)`; `regulatory.applicability` PASS = 모두 결정 + 인용된 문장 grounded(+ 평가하지 않은 제외 명시); **`regulatory.compliance` = 항상 `NOT_VERIFIED`**(엔지니어/인증기관); `review.regulatory_provenance`는 10개 provenance 필드 + 아카이브 항목인 해시 일치 파일 + **모든 발견 인용을 리뷰 시점에 그 페이지에서 직접 재탐색** + 결정된 적용 여부를 요구하고 모든 메시지에 "compliance not assessed"; LLM 제안은 `propose_regulations=yes`(과금)로만, allow-list 호스트만, `accept_regulations`로 제시-후-수락, 공식 문서가 title_quote를 담을 때만 진입 | **구현·fake 서버로 검증** (`tests/test_regulatory_candidates.py`, `test_applicability.py`, `test_regulatory_research.py`, `test_regulatory_agent.py`); 저장소 안의 테스트는 후보 목록 자신의 marker·인용으로 만든 **합성** 페이지만 쓴다 — 패키지 인용 42/42의 원본 공식 텍스트 대조는 2026-09-23 **저장소 밖** 프로브에서 한 번 한 것이고, 실 사이트 fetch는 `AI_EDA_ONLINE=1` opt-in 테스트(EUR-Lex LVD, law.go.kr XML, eCFR titles.json+XML)로만 하며 이 세션에서는 실행하지 않아 law.go.kr/eCFR 실 fetch 가능 여부는 **측정되지 않았다** |
| 시뮬레이션 IR (`ir/simulation.py`: `SpiceBinding`(+`ignored_pins`)·`Stimulus`·`AnalysisSpec`·`Expectation`, 전부 `Traced`) + 컴파일러 IR→SPICE netlist (`compilers/spice.py`: 결정론적 `.cir`, GROUND 넷 → 0, `llm_generated` 값·모델카드·바인딩/자극/분석/기대값 컨테이너 거부, 모델카드는 `.model`/`.subckt`…`.ends` 텍스트만(`.inc`·`.opt`·`.ic`·`.nodeset`·`.global`·분석 카드·서브서킷 밖의 소자 줄은 거부), 바인딩 없는 부품 거부, R/C/L 값은 양수이고 `Component.electrical`과 일치해야 함(시뮬레이션한 부품 = BOM 부품; ngspice는 R=0을 말없이 ~1 mΩ으로 돌린다), `project.id`는 ASCII(netlist 제목 줄), 넷에 있으나 소자가 안 쓰는 핀은 사유 있는 `ignored_pins`만 허용(넷에 없는 핀은 `ignored_pins`로 숨길 수 없고 `no_connect`뿐), `nominal == 0`에 `tol_rel`만 거부, 분석 카드 없음 — `analysis_command`가 러너 명령을 생성, 러너의 노드 규칙·`validate_deck`로 컴파일 시점에 검사, `.cir.report.json`; 기대값 벡터 `v()/i()` + ac 전용 `vp()/vr()/vi()`(dB 불가)) | **구현·kicad-cli SPICE export와 교차 검증** |
| SPICE 단계 (`agents/simulation.py` + `tools/spice/stage.py` `run_spice_for`) — netlist 컴파일 → 분석별 실행 → `<workdir>/spice/results.json`(`SPICE_RESULT`, `engine_info`·`conditions` 포함) + rawfile 증거 → 기대값 판정(`value/at/final/max/min`, `max(tol_abs, tol_rel·|nominal|)`; `at`이 샘플 사이면 보간 bracket을 기록하고 두 이웃 모두 허용치 안일 때만 PASS, 둘 다 같은 쪽으로 밖이면 FAIL, 그 사이(격자가 너무 성김)면 `UNRESOLVED`; assumption 값 위의 PASS는 `NOT_VERIFIED`; 환경 탓 실패는 `NOT_VERIFIED`) → `spice`·`spice.<id>` ValidationResult(+ 사라진 기대값은 `NOT_APPLICABLE`로 대체); 모든 결과는 nominal 값·단일 온도 조건임을 명시; 수리 루프의 `RerunTool`이 같은 함수를 재실행 | **구현·실제 ngspice-46으로 검증** (분압기 op/dc, RC tran이 계산기와 1e-6 V 이내 일치 — 테스트가 그 한계를 단언, RC ac 크기·위상, `.temp`) |
| 제조: JLCPCB capability 모델(미검증 기본) | **구현** (authoritative 데이터 없음 → `NOT_VERIFIED`) |
| Agent 9종 (Requirement/Component/Circuit/Simulation/PCB/Regulatory/Manufacturing/Review/Repair) | 인터페이스 + 결정론적 동작; **Requirement(추출)·Component(부품 후보·데이터시트 사실)·Regulatory(추가 규제 제안)가 LLM을 쓰되** 모두 제안 → 결정론적 대조 → 표 제시 → **다음 실행**의 사용자 확인(`confirm_requirements`/`confirm_parts`/`confirm_facts`/`accept_regulations`)의 같은 패턴, 과금 호출은 명시적 요청(`--llm` 예산, `extract_datasheet_facts`/`propose_regulations`)에만; Component는 부품의 *존재*만 검사하고 적합성(전기 스트레스·안전·규제·환경·신뢰성·제조성·소싱·비용)은 평가하지 않음을 `component.fit` `NOT_VERIFIED`로 말한다; Circuit/PCB/Simulation은 LLM 호출 없음 |
| LLM 클라이언트 (`llm/client.py`, `llm/openrouter.py`) — OpenRouter `/chat/completions` JSON+SSE, `response_format json_schema strict`, tool calls, 타입화된 `LLMError`(status/code/`Retry-After`), `usage.cost`는 제공자 보고값만(없으면 `None`, **0으로 추정 안 함**), 키는 private 속성에만·모든 출력 문자열 redact·INFO 로그에 내용 없음 | **구현·로컬 fake 서버(`tests/fake_openrouter.py`, 문서화된 계약: 401 본문, usage.cost, keepalive, 429 Retry-After, 402/500/502, 중간 오류)로 검증**; 실 서비스 검증은 `tests/test_openrouter_live.py`(키 없으면 skip) |
| LLM 서비스 (`llm/service.py` `LLMService`/`LLMBudget`) — **예산 = 승인**(생성 시 `PAID_API_CALL` 승인을 게이트에 기록·소비, 예산 없으면 거부), 예산은 **사전 조건**: 모든 요청 전에 지출 대조(알려진 비용이 `max_usd`에 도달(`>=`)했거나 토큰이 `max_tokens`에 도달하면 `BudgetExceededError`; `--llm-budget-usd 0`은 유료 클라이언트를 첫 호출부터 거부하고 무료 대본 클라이언트만 허용; 비용 미상 기록 + 토큰 예산 없음이면 거부, 둘 다 있으면 미상은 0으로 세되 WARNING·summary에 명시; 재시도·fallback도 같은 검사), 모든 요청에 `max_tokens`(task 기본값 → 남은 토큰 예산으로 축소), 429는 `Retry-After`(상한) 후 같은 모델 1회 재시도, 408/5xx/transport는 다음 후보(Sonnet 5 → Haiku 4.5)로 fallback, 401/402/400/403은 즉시 실패, `structured()`는 pydantic 검증 실패를 1회 피드백 후 재시도, 회계는 완전(서빙 응답·청구됐을 수 있는 실패(2xx 오류 본문의 usage 유지, 전송 후 transport 실패)·소비자가 중단한 스트림을 모두 task+`model_used`로 기록); 알려진 한계: 다음 요청 비용을 사전 추정하지 않아 예산 X는 X + 상한된 요청 1건까지 허용 | **구현·fake 서버로 검증** (`tests/test_llm_service.py`, `test_llm_findings_regressions.py`) |
| 요구사항 추출 + grounding + 확인 (`tools/calc/quantity.py`, `llm/extraction.py`, `llm/prompts.py`, `agents/requirement.py`) — strict 스키마 → explicit 항목은 요청의 **verbatim 인용**을 **토큰 경계**에서 찾고(`5V`는 `-5V`·`0.5A`·`12V` 안에서 발견되지 않음, `m`≠`M`) 값은 모델의 복사본이 아니라 **요청의 그 자리**에서 결정론적 수량 파서로 읽어 모델 값과 일치해야 `llm_generated`로 진입, 아니면 `assumption`으로 강등(사유 기록); `text`는 사용자의 말(`<key>: <인용>`), 모델 문장은 note에만; implicit은 rationale 필수·`llm_generated` 유지, 같은 키의 다른 종류 항목 사이로 값이 옮겨가지 않음; 지시문 구절(`ignore previous`, `prior instructions`, `api key`, `이전 안내` …) 포함 항목 drop(심층 방어), 모델이 쓴 질문은 `(model question)`으로 표시·500자 상한; 관할권은 인용이 그 코드를 온전한 토큰으로 지칭할 때만(`DE-9`는 DE가 아님), application은 인용 있을 때만(application 항목만은 값이 모델의 요약이고 인용은 note에 — 확인 표의 "in your request"가 그것을 잡으며, 확인 후 규제 단계의 `intended_use`가 된다); 같은 값의 중복은 grounding된 explicit 항목으로 병합, `12V DC`의 `12V`처럼 AC/DC 말을 뺀 인용도 grounding; 필수 질문 `confirm_requirements`(추출 표, 각 인용을 **요청 문맥과 함께**) → 표를 **이전 실행에서 본 뒤** `--answer confirm_requirements=yes`(y/ok/네/확인…, 구두점 무시)일 때만 grounding된 explicit 항목이 `user_requirement`(note에 모델+인용 유지); 이번 실행에서 (재)추출된 것의 확인은 무시되고 다시 표; 확인 단어들로만 된 답(`네 맞습니다`, `yes, correct`, `ok thanks`)도 확인, `no`/짧은 답/숫자 없는 한 단어(`sure`)는 재추출 없이 다시 질문, 같은 정정을 다시 주면 그렇다고 말함, 그 외 답은 정정으로 요청에 붙여 재추출; implicit/assumption은 `--answer accept_implicit=k1,k2` / `reject_implicit=k3`로 개별 결정(수락 → `user_requirement`, 거부 → 제거; 결정은 캐시에 남아 재구성 시 재적용), 미결정 항목은 `ir.llm_requirements`(implicit, `llm_generated`)·`ir.assumptions`(강등된 explicit·모델 assumption)가 IR_BUILD를 막고 리뷰어는 강제하지도 충족으로 세지도 않음(`NOT_VERIFIED`); 추출은 요청 텍스트 sha256으로 `extraction_cache`에 캐시(같은 요청은 호출 0회; 추출 버전·프롬프트·스키마 해시가 다르거나 스키마에 안 맞는 항목은 miss; `--answer` 값은 프롬프트에 들어가지 않고 결정론적으로만 적용; 캐시는 설계 해시에서 제외, 정정은 포함); 결과 `requirements.extraction`(tool=모델, counts/demoted/dropped/notes/cost(청구된 attempt 하나라도 미상이면 unknown)/presented/decisions) — 확인 전 `USER_INPUT_REQUIRED`, 후 `PASS`, 호출 실패 시 `NOT_VERIFIED`+체크리스트; 대조되지 않는 것: 모델이 인용에 붙인 key/category(확인 표에서 사람이 잡는다) | **구현·대본 클라이언트(`ScriptedLLMClient`)로 검증** (`tests/test_quantity.py`, `test_extraction.py`, `test_requirement_agent_llm.py`, `test_llm_findings_regressions.py`) |
| 독립 리뷰어 — 스펙의 14개 검토 영역 | **구현** (artifact 신선도·디스크 해시, BOM/CPL을 **컴파일된 보드 파일의 풋프린트**(reference·value·footprint / 위치·회전·면)와 대조 + CSV 신선도·디스크 해시 + Evidence, 보드 없으면 `NOT_VERIFIED`; 레이아웃 항목(배치·트랙·비아·존)의 provenance 미기록/가정/LLM이면 `ir_vs_pcb` `NOT_VERIFIED`; DRC parity 증거(어떤 회로도·보드에 대해 평가됐는지); gerber/drill 신선도+검사 결과 대조; `spice_vs_requirements`는 netlist·`results.json` 신선도와 해시 사슬(결과가 바로 이 netlist에서 나왔는지)을 대조하고 모든 기대값이 PASS이고 **각 기대값의 nominal이 자기가 검증한다는 요구사항의 값과 일치**할 때만 PASS(stale netlist → `regenerate`, 다른 netlist의 결과/수기 편집 → `rerun_tool`, 기대값 실패/미존재 요구사항/요구사항과 다른 nominal → `human`, 비교할 값 없음/assumption → `NOT_VERIFIED`); `calculations_vs_design`은 리뷰어가 직접 재계산해 판정(저장된 `calc.recompute`와 다르면 그것도 FAIL); `component_provenance`는 아카이브 데이터시트에 grounded된 MPN(해시 재검증, `tampered` 감지) + `component.existence.<ref>`의 최악 sub-check 통과(참조 라이브러리 항목 없음 = FAIL human); `regulatory_provenance`는 10개 필드 + 해시 일치 아카이브 파일 + 결정된 적용 여부, compliance는 절대 판정하지 않음) |
| 자동 수리 — `RegenerateArtifact`(복수 artifact), `RerunTool`(ERC/DRC/SPICE/출력 검사; stale·디스크 불일치 artifact는 거부), 반복 상한, oscillation 감지(상태 지문 = IR 해시 + 실패 검사와 그 수리 범주 + artifact 해시), 반복 내 중복 행위 제거, 실패한 행위는 루프가 진전한 뒤 재시도, 수리 불가 카테고리, **IR 해시 변경 시 행위 실패 + 루프 중단**, 모든 시도를 `repair.loop` ValidationResult로 IR에 기록(행위 하나라도 실패했으면 PASS가 아니라 NOT_VERIFIED) | **구현·실제 도구로 수렴 검증** |
| 워크플로우 오케스트레이터 — 17 단계, 질문에서 블로킹(집계 상태가 FAIL이어도 필수 질문이 있으면 멈춤), 제안(proposal) 적용(타입 검증, 전부 아니면 무; 제안만 있고 검증 결과가 없으면 `NOT_VERIFIED`), 컴파일 거부 → `NOT_VERIFIED`(입력 없음)/`FAIL`(IR 불일치) — 판정은 `compile.<kind>` ValidationResult로 기록되어 RELEASE·종료 코드·`review`에 보이고 BOM/CPL도 같은 규칙, 단계가 만든 결과를 읽는 검증기(`consumes`) 재평가, RELEASE는 tool 없는 PASS·다른 IR 버전의 PASS를 증거로 세지 않음 | **구현** |
| 외부 행위 승인 게이트 (git push, 주문, 구매, 규제 제출 …) | **구현** |
| CLI (`doctor / new / run / review`; `doctor`는 ngspice.dll 경로·버전·코드 모델 적재 여부와 LLM 키 존재 여부(값은 절대 출력 안 함) 출력, `--online`이면 키의 한도·사용량 1회 조회(플래그가 곧 `SECRET_ACCESS` 승인, `OPENROUTER_BASE_URL`로 fake 서버 테스트); `run --llm openrouter\|fake:<json> --llm-model --llm-budget-usd --llm-budget-tokens` — 예산 플래그가 곧 승인, 없으면 exit 2; `run --online [--trust-host H] [--datasheet-url REF=URL] [--source-url ID=URL] [--sources-dir DIR] [--catalog CSV --catalog-date ISO] [--regulatory-candidates PATH]` — `--online`이 곧 `NETWORK_FETCH` 승인(감사 로그에 grant+consume), 없으면 HTTP 클라이언트조차 만들지 않음, 아카이브는 `<workdir>/sources`; `--catalog`에 `--catalog-date`가 없거나 `REF=URL` 형식이 틀리면 exit 2; 단계 표 뒤에 `component.existence.*`의 미통과 sub-check와 non-blocking 질문(규제 범위 질문, 모델 제안 표)을 `--answer`로 답할 수 있게 출력; `--answer`가 `KEY=VALUE`가 아니면 exit 2; 파이프라인이 어디서 죽든 `finally`에서 IR 저장 + LLM 사용량 출력(유료 추출은 잃지 않음); 모델이 쓴 질문은 `(model question)`으로 표시; `--llm` 없으면 종전과 동일; 블로킹되거나 마지막 단계가 FAIL이면 exit 1, `NOT_VERIFIED`는 exit 0; `review`는 `<workdir>/sources`를 읽기 전용으로 열어 아카이브 사본을 재해시) | **구현·`tests/test_cli_llm.py`·`test_parts_regulatory_e2e.py`로 오프라인 데모 검증** (온라인 CLI 실행도 loopback fake로) |
| GUI | 미착수 |

테스트 실행: `python -m pytest -q -p no:cacheprovider -rs` (KiCad/ngspice가 없으면 도구 테스트는 skip; `AI_EDA_ONLINE=1`·`OPENROUTER_API_KEY`는 opt-in). 개발 PC(Windows, KiCad 10.0.6 + 번들 ngspice.dll) 기준 2026-09-23 검토 전 1040개 통과, skip 5 (skip은 `tests/test_openrouter_live.py`의 실 OpenRouter 테스트 2건 — `OPENROUTER_API_KEY`가 없어서 — 과 `tests/test_regulatory_research.py`의 `AI_EDA_ONLINE=1` 실 사이트 테스트 3건(EUR-Lex, law.go.kr, eCFR)뿐; KiCad 10.0.6 설치 시 실제 ERC/DRC/parity/gerber/drill과 번들 ngspice.dll 실행 포함, 미설치 시 해당 테스트는 skip). `tests/test_parts_regulatory_findings_regressions.py`는 부품·규제 트랙의 검증자 결함 18건에 대한 회귀 테스트다. LLM 계층은 키 없이 로컬 fake 서버(`tests/fake_openrouter.py`)와 대본 클라이언트(`ScriptedLLMClient`)로 검증된다. `tests/test_findings_regressions.py`는 검증자가 확인한 결함 17건(BOM/CPL 미대조, 회로도 심볼 겹침, 핀 타입 불일치, 하면 chamfer, 버전 정렬, 제안=PASS, mfg 해시, ERC `sheets` 누락, 존 미채움, 4층 gerber, 레이아웃 provenance, MPN 부재, warning 은폐, 수리 기록/변형 등), `tests/test_spice_findings_regressions.py`는 SPICE 단계의 결함 24건(22번 '1e-6 V 주장'만은 `test_simulation_stage.py`가 그 한계를 직접 단언; 설계 해시의 시각 의존, `run` 종료 코드, BOM과 다른 SPICE 값, 계산 리뷰의 형태 검사, log 격자 보간, 고아 `spice.<id>`, 위치 기반 재계산, 기대값 nominal 미재계산, 0 nominal 허용치, 경로의 `'`, 엔진 설정 미기록, 무시된 핀, 요구사항과 다른 nominal, assumption PASS, LLM 컨테이너, bias 과잉 PASS, nominal 전용 조건, ControlledExit·파서 측정·1e-6 V 주장, ac 위상 문법, title `$`)에 대한 회귀 테스트다. `tests/test_md_review_findings_regressions.py`는 2026-09-23 문서·코드 검토(이 README·ARCHITECTURE·CLAUDE.md의 주장을 코드와 대조한 감사 + 코드 리뷰)에서 확인된 결함 56건(1차 17건: BOM Reference/Footprint/Supplier 셀 미차단, 한 단어 확인 답변의 재추출, jurisdiction 재질문, Debian 코드 모델 경로, 제안 페이로드 미검증, FAIL에 가려진 필수 질문, `remove` 무동작, 로드 시 미지 키·스키마 버전 무시, `Traced[float]`의 문자열/bool 강제변환, OpenRouter 응답 필드 미redact, 아카이브 lookup 크래시·meta 신뢰, 수락된 제안의 allow-list 미재검사, IPv6·포트, http base URL, 아카이브 상태 매핑 테스트 부재, 의견 PASS의 RELEASE, 설계 해시의 경로·검증 상태 포함; 2차 13건: 잘린 MPN의 grounding, `DC 60 V` 표기, 답변이 추출 항목을 대체하지 않음, 카탈로그 중복 행, 기록 페이지의 대소문자·부재, 중복 `req.application`, 같은 값 중복의 순서 의존, AC/DC 말을 뺀 인용의 강등, 손상된 캐시 항목 크래시, 불완전한 fingerprint, `DE-9`의 관할권, NaN, 확인 단어 조합의 정정 취급; 3차 15건: 컴파일 거부가 RELEASE·종료 코드·`review`에 보이지 않음, BOM/CPL 거부의 크래시, 리뷰어가 tool·해시 없는 ERC/DRC/capability 결과를 통과, 추적할 것이 없는 `requirements_vs_ir` PASS, 비원자적 제안 적용, cwd 상대 workdir, 수리 루프의 상태 지문·실패 행위 재시도·실패 행위의 PASS, `--datasheet-url` 중복, CPL 회전각 `%g`, 문자·숫자 혼합 핀 번호 정렬 크래시, NaN/inf 좌표, 반복 패드 번호의 중복 uuid·반복 핀 번호의 무언 누락, 잘못된 라이브러리 객체의 조용한 대체; 4차 10건: model_card의 `.inc`/`.opt`/소자 줄 통과, `a` 꼬리를 무시한 파서 모델, 배치 러너의 사본 해시·stderr, 넷에 없는 핀의 `ignored_pins`, 비ASCII 제목의 오해 메시지, 0/음수 R·C·L, 벡터 중간의 NaN, 무한 허용치, rawfile `Command:` 줄 부재, 시스템 libngspice 미탐색; 5차 1건: 규제 실행 결과(`GroundedQuote.reason`·페이지·grounded 여부)의 설계 해시 잔류)에 대한 회귀 테스트다. Linux(Ubuntu 24.04, Python 3.12, KiCad 없음) 실측 2026-09-23: ngspice 없이 1006 passed / 119 skipped, `libngspice0`(ngspice-42, 자동 탐색)로 1045 passed / 80 skipped(남은 skip은 KiCad 라이브러리/kicad-cli 필요 73건 + `AI_EDA_ONLINE` 3건 + OpenRouter 2건 + Reset 없는 빌드의 복구 테스트 1건 + PATH의 ngspice 바이너리 1건) — KiCad 10은 Linux에서 아직 측정되지 않았다(Ubuntu 24.04의 apt `kicad`는 7.0.11).

**신뢰 모델 (부품·규제).** 모델이나 큐레이션 목록은 *제안*만 한다. 사실은 **아카이브된 문서**에 grounding될 때만 authoritative가 된다: 문서는 시스템이 신뢰하는 pointer(KiCad 라이브러리 `Datasheet` 필드, 큐레이션 공식 소스 목록의 도메인, 사용자가 준 URL/파일)에서 가져와 sha256·retrieved_at·최종 URL·content type과 함께 저장되고, 주장된 문구는 주장된 페이지의 아카이브 본문에서 verbatim으로 찾아야 한다(MPN도, 규제 인용도, 데이터시트 사실도 같은 `find_quote`). 네트워크는 외부 행위라 `--online` 없이는 소켓이 열리지 않고, 있어도 신뢰 origin·https·URL당 1회뿐이며 모델이 말한 URL은 호스트만으로는 신뢰되지 않는다 — 규제 제안의 URL은 그 호스트가 후보 목록의 공식 도메인 allow-list에 있고 사용자가 `accept_regulations`로 수락한 경우에만, fetch 시점에 allow-list를 다시 확인한 뒤 fetch된다(부품 쪽은 모델이 말한 URL을 fetch하지 않는다). 문서를 읽었다는 것은 compliance가 아니다: 규제 단계는 출처와 적용 여부만 보고하고 `regulatory.compliance`는 항상 `NOT_VERIFIED`다. 아카이브 파일은 읽을 때마다 재해시되어 변조되면 `tampered`로 그것이 뒷받침하던 모든 것이 `NOT_VERIFIED`가 되고, 검증기와 리뷰어는 태그를 믿지 않고 MPN과 규제 인용을 아카이브 본문에서 직접 다시 찾는다(아카이브 항목이 아닌 파일은 해시가 맞아도 사본이 아니다; 추출기가 바뀌어 텍스트가 달라지면 재grounding 전까지 `NOT_VERIFIED`). 데이터시트는 PDF다 — 제품 페이지 같은 HTML은 pointer의 증거일 뿐 identity를 만들지 않는다. 사용자가 `--answer`로 준 규제 범위 답변(`mains_powered` 등)은 `regulatory` 범주의 요구사항으로 기록되고(부품이 서빙해야 하는 전기 요구사항이 아님), 다른 agent의 제어 키(`confirm_parts`, `confirm_facts`, `accept_regulations` …)는 요구사항이 되지 않는다. grounding이 대조하지 *못하는* 것은 모델이 숫자에 붙인 키·의미이므로 모델의 데이터시트 사실은 사용자가 표를 보고 확인하기 전에는 IR에 없다.

**검증된 end-to-end (부품 + 규제, `tests/test_parts_regulatory_e2e.py`, loopback fake가 벤더 PDF와 패키지 후보 목록의 marker·인용으로 만든 합성 EUR-Lex 페이지를 서빙):** 오프라인 실행은 요청 0건(HTTP 클라이언트조차 생성 안 됨), `component.existence.*` `NOT_VERIFIED`("not fetched (offline)"), `regulatory.sources`/`compliance` `NOT_VERIFIED`, 표시만 된 authoritative MPN이 `ir.component_provenance`/`review.component_provenance`에서 강등, BOM `DatasheetHash` `NOT_VERIFIED`; 온라인 실행은 데이터시트 2종(R1·R2 공유 1회 fetch)·EU 공식 문서 5종을 아카이브하고 MPN 3개를 2페이지에서 verbatim으로 찾아 `component.existence.*` PASS(Evidence 경로·sha256), BOM MPN 셀 + `DatasheetHash` 채움, `regulatory.sources`/`applicability` PASS(범위 답변 12 V DC·mains 아님·radio 아님·finished apparatus·evaluation kit 아님·`highest_rated_voltage=12 V DC`로 LVD `NOT_APPLICABLE`에 Article 1 인용 "1 000 V" 증거, EMC/RoHS `APPLICABLE`, RED `NOT_APPLICABLE`, 4건 모두 10개 provenance 필드, 메시지에 평가하지 않은 제외 명시), `regulatory.compliance` `NOT_VERIFIED`, `review.component_provenance`/`regulatory_provenance` PASS(MPN·인용을 리뷰 시점에 재탐색), RELEASE는 여전히 `NOT_VERIFIED`로 `regulatory.compliance`·`mfg.capability`·`component.fit`을 지목; 저장한 IR로 오프라인 재실행은 요청 0건에 사본을 재해시해 같은 판정; 아카이브 데이터시트/LVD 텍스트를 변조하면 검증기와 리뷰어가 `tampered`(human); 카탈로그에 R1만 있으면(제조사·패키지가 IR과 같을 때) R1 PASS·R2 "not in catalog"·리뷰어가 R2를 지목; CLI `--online --trust-host`가 같은 세션을 만들고 `review`가 사본을 재검증한다.

**검증된 vertical slice** (`tests/test_vertical_slice.py`): 저항 분압 R1/R2 + 3핀 헤더 J1 IR → CALCULATION(`v_out`·`v_out_mid`와 기대값 nominal 재계산 PASS) → SPICE(ngspice-46: op + dc 스윕, `v(VOUT)` = 6 V / 3 V @ VIN = 6 V가 계산기 값과 1 % 이내, 각각 `req.v_out` 6 V / `req.v_out_half` 3 V와 일치) → `.kicad_sch` → ERC 0/0 → `.kicad_pcb`(naive 트랙) → DRC 0/0 + schematic parity 0 → gerber 9층 + drill → 형식 검사 PASS → 독립 리뷰 10개 영역 PASS → RELEASE는 `NOT_VERIFIED`(규제/fab capability 증거 없음 — 의도된 결과, 메시지가 그 항목을 명시). 설계 변경(R3 추가 + R1을 5k로 재조정해 요구사항 유지, 기대값은 계산기로 재산출) 후 리뷰어가 모든 파생물(SPICE netlist 포함)을 stale로 판정하고, 수리 루프가 IR을 건드리지 않고 재생성·재실행(ngspice 포함)만으로 3회 반복 내에 다시 PASS로 수렴한다. `tests/test_simulation_stage.py`: 잘못된 nominal은 정직하게 FAIL(human, 수리 불가), 설계 변경 후 재시뮬레이션에서 기대값이 4 V vs 6 V로 FAIL, nominal과 요구사항을 함께 갱신하면 PASS로 수렴(nominal만 재산출하면 리뷰어가 "6 V 요구사항을 4 V로 검증"을 거부), 수기 편집된 `results.json`은 감지되어 재실행.

**알려진 한계 (코드에 `NOT_VERIFIED`/거부로 표시됨):** 부품 *적합성*(전기 스트레스·안전·규제·환경·신뢰성·제조성·소싱·비용)은 어떤 코드도 평가하지 않는다 — `component.fit`이 항상 `NOT_VERIFIED`로 그렇게 말하고, 후보 표의 근거 문장은 모델의 산문이다. 데이터시트는 PDF만 인정하므로 HTML로만 제공되는 데이터시트는 grounding되지 않는다(제품 페이지와 구별할 결정론적 방법이 없어서 — pointer 증거로만 아카이브). 모델의 데이터시트 사실은 키의 *의미*를 grounding이 대조하지 못하므로 사용자 확인 전에는 IR 밖이고, 확인은 "인용·페이지·문맥 표를 보고 키가 맞다"는 사용자의 판단이다. LVD/KC 전압 규칙은 IR의 `input_voltage`/`output_voltage`와 사용자가 답한 최고 전압만 읽으며 IR의 다른 전압 값(격리 시험 전압 등)은 읽지 않고, 규칙이 다루지 않는 제외 조항은 `not_evaluated`로 명시될 뿐 판정되지 않는다. BOM의 `Value`/`Description` 열은 설계의 말이라 CSV formula 문자를 중화하지 않는다(identity/sourcing 열만 거부). ERC/DRC warning은 FAIL이며 KiCad 예외(exclusion)도 예외로 인정하지 않으므로 warning을 "수용"하는 길은 설계 변경뿐이다(전원 핀이 생기면 `power_pin_not_driven`이 즉시 FAIL — PWR_FLAG 지원 전까지). CPL의 `Mid X/Y`는 풋프린트 anchor이지 JLCPCB가 요구하는 중심점이 아니다(리뷰어도 anchor 기준으로 대조). 하면 패드의 `thermal_bridge_angle`은 KiCad 소스대로 그대로 두지만 DRC `lib_footprint_mismatch`로는 구별되지 않아 독립 검증되지 않았다. 회로도 라벨 폭은 글자당 1.27 mm로 보수적으로 잡은 추정치다(전기적 연결은 좌표 일치 검사로 별도 보장). SPICE: ngspice.dll은 프로세스 내 싱글턴이라 예기치 않은 access violation은 프로세스 재시작까지 엔진을 죽인다; 반도체 모델은 IR의 authoritative `model_card`로만 들어온다(모델 라이브러리 파이프라인 없음); ngspice가 1 ULP 어긋나게 읽는 숫자는 보고만 한다; 모든 기대값은 부품 nominal 값·단일 온도에서만 판정되며(허용차 코너/Monte Carlo 없음 — 결과와 리뷰가 그렇게 명시) ac 기대값은 크기·위상·실수·허수만(dB 불가); `.spiceinit`/`spinit`은 막지 않고 init 설정 목록·해시를 결과에 기록한다.

## 설치

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,llm,parts]"                  # AES 암호화 PDF까지 읽으려면 pip install cryptography
python -m pytest -q -p no:cacheprovider -rs        # 도구가 없으면 도구 테스트는 skip으로 표시된다
ai-eda doctor                                      # 무엇을 찾았는지(kicad-cli, 라이브러리, ngspice, LLM 키 유무) 출력
```

외부 도구: KiCad 10 (`kicad-cli`; Windows 설치본은 `bin/ngspice.dll`(ngspice-46)을 번들하므로 별도 ngspice 설치 불필요, Linux는 시스템 로더가 아는 `libngspice.so.0`(`libngspice0`)을 자동으로 찾고(`NGSPICE_DLL`로 바꿀 수 있음) 코드 모델 디렉터리 `<libdir>/ngspice`도 자동 탐색된다 — Ubuntu 24.04의 ngspice-42로 실측; 그 빌드는 `ngSpice_Reset`이 없어 ControlledExit 뒤 복구가 불가능하다는 것을 `doctor`와 `engine_info()`가 말한다; 자세한 것은 CLAUDE.md), 선택적으로 `OPENROUTER_API_KEY`. 문서 아카이브는 `httpx`(fetch)와 `pypdf`(PDF 본문)를 쓴다(`.[parts]`); AES 암호화 PDF의 본문 추출에는 `cryptography`가 추가로 필요하며 없으면 파일은 아카이브·해시 검증되되 본문 없이 `NOT_VERIFIED`("datasheet text not extractable")로 보고된다. 법제처 open API의 사용자 id는 `AI_EDA_LAW_GO_KR_OC`(기본 데모 id `test`).

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

요청문에서 요구사항을 모델로 추출하려면 `--llm`과 **예산**(= 유료 호출 승인)을 함께 줍니다. 키 없이 돌려 볼 수 있는 오프라인 대본:

```bash
ai-eda new demo --request "12V 입력을 5V 2A로 변환하는 회로, 효율 90% 이상, EU에서 판매"
ai-eda run projects/demo/ir.json --llm fake:tests/data/fake_llm_requirements.json --llm-budget-usd 0
#   → requirement_analysis USER_INPUT_REQUIRED, [confirm_requirements] 표 출력, exit 1 (추출 항목은 llm_generated)
ai-eda run projects/demo/ir.json --llm fake:tests/data/fake_llm_requirements.json --llm-budget-usd 0 --answer confirm_requirements=yes
#   → 호출 0회(캐시), 이전 실행에서 본 표의 grounding된 explicit 항목만 user_requirement, MISSING_INFORMATION PASS
#   → 모델이 추론한 implicit 항목(역전압 보호)은 아직 결정되지 않아 IR_BUILD가 [ir.llm_requirements]로 블로킹 → exit 1
ai-eda run projects/demo/ir.json --llm fake:tests/data/fake_llm_requirements.json --llm-budget-usd 0 --answer accept_implicit=input_reverse_polarity_protection
#   → 수락된 항목이 user_requirement가 되어 파이프라인 계속
#   → 회로가 없으므로 리뷰어가 "requirements not traced to any component"로 FAIL → exit 1 (정직한 결과)
```

실제 모델(`OPENROUTER_API_KEY` 필요): `--llm openrouter --llm-budget-usd 0.05` (기본 `anthropic/claude-sonnet-5`, fallback `anthropic/claude-haiku-4.5`; `--llm-model`로 변경). 모델이 무엇을 돌려주든 IR에는 요청문 인용으로 grounding된 것만, 확인 전에는 `llm_generated`로만 들어갑니다.

데이터시트와 공식 규제 문서를 실제로 가져오려면 `--online`(= 이 실행의 `NETWORK_FETCH` 승인)을 줍니다. 없으면 파이프라인은 소켓을 열지 않고 모든 출처를 `NOT_VERIFIED`("not fetched (offline)")로 보고합니다. 규제 범위 질문은 파이프라인을 멈추지 않고 출력되며 `--answer`로 답합니다.

```bash
ai-eda run projects/demo/ir.json --answer application="bench supply" --answer jurisdiction=EU \
    --answer mains_powered=no --answer radio=no --answer finished_apparatus=yes --answer evaluation_kit=no \
    --answer highest_rated_voltage="12 V DC"
#   → 부품이 있는 IR에서는 component.existence.<ref> NOT_VERIFIED (not fetched (offline)); 위의 데모 IR은 부품이 없어
#     component_selection이 "no components in the IR: nothing to check"; regulatory.sources NOT_VERIFIED, regulatory.compliance NOT_VERIFIED
#   → LVD의 전압 규칙은 user_requirement/authoritative인 input_voltage·output_voltage 요구사항과 highest_rated_voltage 답을 읽는다:
#     --llm 추출로 input_voltage가 요구사항이 된 IR에서는 highest_rated_voltage 없이는 LVD가 UNDECIDED(입력 전압만으로는 장비의
#     정격을 모른다)이고 regulatory.applicability가 그 키를 지목; LLM 없이 만든 데모 IR에는 input_voltage 요구사항이 없어
#     --answer input_voltage="12 V DC"까지 주어야 판정된다
ai-eda run projects/demo/ir.json --online --trust-host www.example-vendor.com \
    --datasheet-url U1=https://www.ti.com/lit/ds/symlink/lm2596.pdf \
    --catalog my_jlcpcb_export.csv --catalog-date 2026-09-23 --catalog-supplier JLCPCB
#   → 신뢰 호스트(부품의 KiCad Datasheet 호스트 + 후보 목록의 공식 도메인 + --trust-host)와 정확한 사용자 URL만 fetch,
#     <workdir>/sources/<sha256>.pdf|html + .meta.json + fetch_log.jsonl; MPN을 PDF 본문에서 찾은 부품만 authoritative,
#     규제는 출처·적용 여부만 (regulatory.compliance는 항상 NOT_VERIFIED)
ai-eda run projects/demo/ir.json --online --llm openrouter --llm-budget-usd 0.05 --answer extract_datasheet_facts=yes
#   → 모델이 읽은 데이터시트 사실은 [confirm_facts] 표(키·값·페이지·인용·문맥)로만 제시되고 IR에는 없음
ai-eda run projects/demo/ir.json --answer confirm_facts=yes
#   → 이전 실행에서 본 표 그대로면 authoritative(section='page N', note에 "confirmed by user")로 진입; no면 폐기
```

```bash
ai-eda review projects/demo/ir.json
#   → <workdir>/sources의 아카이브 사본을 재해시해 component/regulatory provenance를 다시 판정 (fetch 없음)
```

## 구조

```
ai_eda/
  ir/            Universal Circuit IR + provenance + validation 상태
  validation/    Validator 프로토콜, 레지스트리, 구조/도메인 검증기
  tools/
    calc/        결정론적 계산기 + 레지스트리/재계산, SPICE 숫자 파싱·포맷
    spice/       ngspice.dll 러너(ctypes), rawfile 리더, run_spice_for (SPICE 단계·RerunTool 공용)
    kicad/       s-expression 파서/직렬화, 라이브러리(심볼/풋프린트), 보드 기하, kicad-cli 래퍼(ERC/DRC/export)
    routing/     naive placeholder 라우터 (실제 라우터 아님)
    manufacturing/ fab capability 모델, Gerber/Drill 검사
    sources/     문서 아카이브: 네트워크 정책(승인·신뢰 origin·https), sha256 저장 + meta, 텍스트 추출, 인용 grounding
  parts/         부품 트랙: datasheet pointer, 존재 검사(component.existence.<ref>), identity grounding 규칙, 데이터시트 사실, 카탈로그
  regulatory/    규제 트랙: 큐레이션 후보 목록(candidates.json), 결정론적 적용 여부, 공식 문서 조사(출처·인용 grounding, compliance 판정 없음)
  compilers/     IR → kicad_sch / kicad_pcb / BOM(DatasheetHash) / CPL / gerber·drill(export) / SPICE netlist
  agents/        전문 Agent (제안만, 진실 결정 안 함; Requirement/Component/Regulatory는 LLM 제안 → 결정론적 대조 → 사용자 확인)
  llm/           모델 계층 (클라이언트 계약, OpenRouter, 대본 클라이언트, 라우터, 사용량, 예산/승인 서비스, 프롬프트, 추출 grounding)
  review/        독립 리뷰어 (14개 영역)
  repair/        결정론적 수리 전략 + 폐쇄 루프
  workflow/      단계 정의 + 오케스트레이터 + 소스 세션(정책·아카이브·카탈로그·후보 목록을 CLI 플래그로부터)
  security/      외부 행위 승인 게이트
  errors.py      공통 예외 (CompileError, ToolExecutionError, ToolUnavailableError, IRSchemaError …)
  cli.py
tests/
  conftest.py, fixtures_kicad.py, fixtures_rc.py   공용 픽스처 (분압기·RC 회로, 실제 KiCad 라이브러리 참조)
  fake_openrouter.py                                 로컬 OpenRouter fake 서버 (문서화된 계약)
  fake_sources.py                                    loopback 문서 서버 (봇 차단·리다이렉트·404·타임아웃)
  pdf_fixture.py, ngspice_parser_harness.py          결정론적 PDF 생성기, ngspice 숫자 파서 측정 하니스
  data/                                              fake_llm_requirements.json (오프라인 데모 대본), catalog_sample.csv
  test_*_findings_regressions.py                     검토 라운드별 결함 회귀 테스트 (KiCad, SPICE, LLM, 부품·규제, 문서·코드 검토)
docs/ARCHITECTURE.md   스펙 섹션 ↔ 모듈 매핑, 데이터 흐름, 다음 단계
```

자세한 설계 근거와 다음 단계는 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)를 보세요.

## 변경 이력

- 2026-09-23 문서·코드 검토: README/ARCHITECTURE/CLAUDE.md의 주장을 코드와 대조해 부정확한 문장 30여 곳을 고치고, 코드 결함 56건을 회귀 테스트와 함께 수정(`tests/test_md_review_findings_regressions.py`); Ubuntu의 ngspice-42 공유 라이브러리로 SPICE 러너를 실측(`ngSpice_Reset` 없음·solver 안내문을 stderr로 출력 → 선택적 바인딩·정보성 stderr 분류; 숫자 파서 모델은 표본 500개 일치); 설계 해시가 경로·검증 상태를 담지 않도록 모델별 design view 도입; RELEASE가 tool 없는 PASS·다른 IR 버전의 PASS를 증거로 세지 않음.
- 2026-09-23 Linux / Claude Code web 설정 노트(CLAUDE.md).
- 2026-09-23 부품 존재 검사·규제 조사(공유 문서 아카이브).
- 2026-09-23 LLM 단계(OpenRouter 클라이언트, 예산 게이트 서비스, grounding된 요구사항 추출과 사용자 확인).
- 2026-09-22 SPICE 단계(IR→netlist 컴파일러, ngspice.dll 러너, 요구사항 대비 시뮬레이션 판정).
- 2026-09-21 KiCad vertical slice 스캐폴드.
