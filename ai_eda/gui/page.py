"""The GUI page: one HTML document, its stylesheet and its script, as strings the server sends (no build step, no external resource).

Invariant: the page is a *view* of the server's JSON and a form for the
CLI's own run options. It displays what the API returns and computes no
status: every status word it shows (with its colour *and* its text, never
colour alone) is a string copied from ``ir.validation`` / ``pipeline.json``
through the project JSON, every freshness or disk label is the server's
(shown in the Korean wording the server sends with it,
:mod:`ai_eda.gui.labels`), every previewed file is shown with its own
artifact facts (generated from the current design hash? on disk?), and the
only things the script derives are presentation (a count of rows, a number
rounded for display, a path cut to its workdir-relative part for a download
link, line-break points after ``.`` / ``_`` / ``:`` in an identifier). It loads only ``/static/app.css`` and ``/static/app.js``
from its own origin and carries no inline script, no inline event handler,
no ``style`` attribute and no ``<style>`` element, so the app CSP
(:data:`ai_eda.gui.server.APP_CSP`: ``default-src 'self'``, no
``'unsafe-inline'``) allows exactly what it uses. Text reaches the document
through ``textContent`` / text nodes only (never ``innerHTML``); the preview
SVGs the server drew are parsed with ``DOMParser`` and imported with their
scripts, event attributes, style attributes and external references
removed; the documents (``report.html``, the stage-report HTML) are shown in
an ``<iframe sandbox>``. The only requests that change anything are the
POSTs the server guards: a new project and a run / review / report re-write,
whose fields are the ``RUN_OPTIONS`` dests (``data-opt``) - the server turns
them into the CLI's own flags; the page never adds a flag the user did not
set, never asks for a key or a login, and deletes nothing.

Layout: the projects column (root path, the project list with the last run,
the "새 프로젝트" form with example requests) and the project's eight tabs
(개요, 회로도, 기판, 시뮬레이션, 검증, 부품, 보고서, 파일). The location hash
``#project=<name>&tab=<tab>[&report=<stage|full>]`` selects a project, a tab
and a report, so a link (or a headless screenshot) can open any of them.
"""

from __future__ import annotations

APP_HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI EDA 엔지니어</title>
<link rel="icon" href="data:,">
<link rel="stylesheet" href="/static/app.css">
<script src="/static/app.js" defer></script>
</head>
<body>
<a class="skip" href="#main">본문으로 건너뛰기</a>
<header class="topbar">
<h1>AI EDA 엔지니어</h1>
<p class="topnote">이 컴퓨터(127.0.0.1)에서만 열리는 화면입니다. 모든 상태는 ir.json과 pipeline.json에 기록된 그대로이며, 실행은 명령줄과 같은 승인으로 <code>ai-eda</code>를 실행합니다.</p>
</header>
<div class="layout">
<aside class="sidebar" aria-label="프로젝트 목록과 새 프로젝트">
<section class="side-block">
<h2>프로젝트</h2>
<p class="small muted">프로젝트 폴더: <code id="root" class="path">불러오는 중…</code></p>
<p id="projects-error" class="notice error" role="alert" hidden></p>
<ul id="projects" class="project-list"></ul>
<p id="projects-empty" class="empty" hidden>아직 없음: 아래 '새 프로젝트'에서 만든 프로젝트가 여기에 나옵니다.</p>
</section>
<section class="side-block">
<details id="new-project-box" open>
<summary><h2>새 프로젝트</h2></summary>
<form id="new-project" novalidate>
<div class="field">
<label for="new-name">이름</label>
<input id="new-name" name="name" required maxlength="64" pattern="[A-Za-z0-9][A-Za-z0-9_\-]{0,63}" autocomplete="off" spellcheck="false" aria-describedby="new-name-hint">
<p id="new-name-hint" class="hint">영문자나 숫자로 시작하고 영문자, 숫자, '_', '-'만 쓰는 64자 이하의 이름. 프로젝트 폴더의 이름이 됩니다.</p>
</div>
<div class="field">
<label for="new-example">예시 요청문</label>
<select id="new-example" aria-describedby="new-example-hint new-example-inputs">
<option value="">예시 고르기</option>
<option value="12 V 입력을 5 V로 나누는 무부하 저항 분배기" data-template="divider" data-inputs="input_voltage=12 V|output_voltage=5 V">저항 분배기 (divider)</option>
<option value="5 V 전원, 순방향 전압 2 V·순방향 전류 10 mA인 LED 표시등" data-template="led" data-inputs="input_voltage=5 V|led_forward_voltage=2 V|led_forward_current=10 mA">LED 표시등 (led)</option>
<option value="차단 주파수 1 kHz인 1차 RC 저역통과 필터" data-template="rc_lowpass" data-inputs="cutoff_frequency=1 kHz">RC 저역통과 (rc_lowpass)</option>
<option value="5 V 입력, 1 kHz 구형파 발진기" data-template="astable" data-inputs="input_voltage=5 V|oscillation_frequency=1 kHz">구형파 발진기 (astable)</option>
<option value="ATmega128 개발 보드: DC 잭 9 V 입력, L7805 5 V 레귤레이터, 16 MHz 크리스탈, 모든 포트 1x8 헤더, ISP 2x3, UART0 1x4" data-template="atmega128_devboard" data-inputs="input_voltage=9 V|clock_frequency=16 MHz">ATmega128 개발 보드 (atmega128_devboard)</option>
</select>
<p id="new-example-hint" class="hint">고르면 다섯 가지 결정론적 회로 템플릿 중 하나의 요청문을 아래 요청문 칸에 채웁니다. LLM 없이 실행하면 요청문은 읽지 않습니다: 첫 실행의 질문(적용 분야, 시장)에 답한 뒤, 템플릿 입력을 개요의 '추가 답변'에 <code>키=값</code> 줄로 줍니다.</p>
<p id="new-example-inputs" class="hint" hidden></p>
</div>
<div class="field">
<label for="new-request">요청문</label>
<textarea id="new-request" name="request" rows="4"></textarea>
</div>
<div class="actions"><button type="submit" class="primary">만들기</button></div>
<p id="new-result" role="status"></p>
</form>
</details>
</section>
</aside>
<main id="main" tabindex="-1">
<div id="welcome" class="welcome">
<h2>프로젝트를 고르십시오</h2>
<p class="muted">왼쪽 목록에서 프로젝트를 고르거나 '새 프로젝트'로 만듭니다. 프로젝트마다 개요(단계·질문·실행), 회로도, 기판, 시뮬레이션, 검증, 부품, 보고서, 파일을 볼 수 있습니다.</p>
</div>
<div id="project-view" hidden>
<div class="project-head">
<h2 id="project-title"></h2>
<div id="project-sub"></div>
<div id="project-notices"></div>
</div>
<div id="tabs" class="tabs" role="tablist" aria-label="프로젝트 화면">
<button type="button" role="tab" id="tab-overview" data-tab="overview" aria-controls="panel-overview" aria-selected="true">개요</button>
<button type="button" role="tab" id="tab-schematic" data-tab="schematic" aria-controls="panel-schematic" aria-selected="false" tabindex="-1">회로도</button>
<button type="button" role="tab" id="tab-board" data-tab="board" aria-controls="panel-board" aria-selected="false" tabindex="-1">기판</button>
<button type="button" role="tab" id="tab-simulation" data-tab="simulation" aria-controls="panel-simulation" aria-selected="false" tabindex="-1">시뮬레이션</button>
<button type="button" role="tab" id="tab-validation" data-tab="validation" aria-controls="panel-validation" aria-selected="false" tabindex="-1">검증</button>
<button type="button" role="tab" id="tab-parts" data-tab="parts" aria-controls="panel-parts" aria-selected="false" tabindex="-1">부품</button>
<button type="button" role="tab" id="tab-reports" data-tab="reports" aria-controls="panel-reports" aria-selected="false" tabindex="-1">보고서</button>
<button type="button" role="tab" id="tab-files" data-tab="files" aria-controls="panel-files" aria-selected="false" tabindex="-1">파일</button>
</div>

<section id="panel-overview" class="panel" role="tabpanel" aria-labelledby="tab-overview">
<div id="ov-release"></div>
<h3>단계</h3>
<div id="ov-stages"></div>
<div class="run-area">
<form id="run-form" class="run-form" novalidate>
<fieldset class="box" id="questions">
<legend>질문</legend>
<div id="ov-questions"></div>
<div class="field">
<label for="extra-answers">추가 답변 (한 줄에 <code>키=값</code> 하나)</label>
<textarea id="extra-answers" rows="2" spellcheck="false" aria-describedby="extra-answers-hint"></textarea>
<p id="extra-answers-hint" class="hint">묻지 않은 답도 여기서 줍니다. LLM 없이 실행하면 요청문을 읽지 않으므로 템플릿 입력은 여기에 줍니다 - <span id="template-keys"></span> (예: <code>input_voltage=12 V</code>). 제어 답의 예: <code>pcb.routing=skip</code>, <code>confirm_design=yes</code>. 값은 그대로 한 개의 <code>--answer</code>가 됩니다.</p>
</div>
</fieldset>
<fieldset class="box" id="run-options">
<legend>실행 옵션</legend>
<div class="check">
<input type="checkbox" id="opt-online" data-opt="online" data-shape="flag" aria-describedby="opt-online-hint">
<label for="opt-online">온라인 가져오기 허용 (<code>--online</code>)</label>
</div>
<p id="opt-online-hint" class="hint">체크하면 이번 실행의 네트워크 가져오기(NETWORK_FETCH)를 승인합니다: 신뢰하는 호스트의 데이터시트와 공식 규제 문서만 가져와 sha256으로 보관합니다. 체크하지 않으면 아무것도 가져오지 않습니다.</p>
<fieldset class="sub" id="llm-box">
<legend>LLM</legend>
<div class="field">
<label for="opt-llm">제공자 (<code>--llm</code>)</label>
<select id="opt-llm" data-opt="llm" aria-describedby="llm-providers-title"></select>
</div>
<p id="llm-providers-title" class="hint">제공자 상태 (GUI는 키나 로그인을 묻지 않습니다: 키는 환경 변수, 로그인은 Claude Code CLI의 것입니다)</p>
<ul id="llm-providers" class="provider-list"></ul>
<fieldset id="llm-fields" class="plain" disabled>
<legend class="sr-only">모델과 예산</legend>
<div class="field">
<label for="opt-llm-model">모델 (<code>--llm-model</code>)</label>
<input id="opt-llm-model" data-opt="llm_model" autocomplete="off" spellcheck="false" placeholder="provider:model" aria-describedby="opt-llm-model-hint llm-pin">
<p id="opt-llm-model-hint" class="hint">provider:model 형식. 예: <code>claude:claude-sonnet-5</code>, <code>openrouter:anthropic/claude-sonnet-5</code>. 비우면 기본 제공자의 기본 모델입니다.</p>
</div>
<p id="llm-pin" class="pin"></p>
<div class="check">
<input type="checkbox" id="opt-llm-allow" data-opt="llm_allow_model_change" data-shape="flag" aria-describedby="opt-llm-allow-hint">
<label for="opt-llm-allow">모델 변경 허용 (<code>--llm-allow-model-change</code>)</label>
</div>
<p id="opt-llm-allow-hint" class="hint">입력한 모델이 이 프로젝트에 고정된 모델과 다르면 이 체크가 필요합니다. 없으면 CLI가 모델을 부르기 전에 실행을 거부하고(종료 코드 2) 그 한국어 문장이 로그에 나옵니다.</p>
<p id="llm-pin-note" class="notice warn" hidden></p>
<div class="field" data-opt="llm_fallback" data-shape="fallback">
<label for="opt-llm-fallback">백업 모델 (<code>--llm-fallback</code>)</label>
<div class="inline">
<select id="opt-llm-fallback">
<option value="">없음</option>
<option value="same">same (다른 제공자의 같은 모델)</option>
<option value="spec">직접 입력</option>
</select>
<input id="opt-llm-fallback-spec" aria-label="백업 모델 사양 (provider:model)" autocomplete="off" spellcheck="false" placeholder="provider:model" disabled>
</div>
</div>
<div class="field">
<p class="label-like" id="task-models-title">작업별 모델 (<code>--llm-task-model</code>)</p>
<div id="task-models" data-opt="llm_task_model" data-shape="tasks" role="group" aria-labelledby="task-models-title"></div>
<div class="actions"><button type="button" id="add-task-model">작업별 모델 추가</button></div>
</div>
<div class="grid2">
<div class="field">
<label for="opt-budget-usd">USD 예산 (<code>--llm-budget-usd</code>)</label>
<input id="opt-budget-usd" data-opt="llm_budget_usd" type="number" min="0" step="any" inputmode="decimal" aria-describedby="budget-note">
</div>
<div class="field">
<label for="opt-budget-tokens">토큰 예산 (<code>--llm-budget-tokens</code>)</label>
<input id="opt-budget-tokens" data-opt="llm_budget_tokens" type="number" min="1" step="1" inputmode="numeric" aria-describedby="budget-note">
</div>
</div>
<p id="budget-note" class="hint">예산은 호출당 과금(PAID_API_CALL)의 승인입니다: 호출당 과금 제공자(openrouter)를 고르면 USD 예산이나 토큰 예산 중 하나 이상이 필요합니다. <code>--llm claude</code>는 구독 사용(SUBSCRIPTION_USE)의 승인이라 예산 없이도 됩니다: 호출당 청구는 없고, 실행 로그의 <code>LLM usage:</code> 줄이 토큰과 CLI의 API 환산 추정치를 "청구 없음"으로 보여 줍니다. 토큰 예산은 구독 호출에도 적용됩니다.</p>
</fieldset>
</fieldset>
<div class="field">
<label for="opt-fab">fab capability 파일 경로 (<code>--fab-capability</code>)</label>
<input id="opt-fab" data-opt="fab_capability" autocomplete="off" spellcheck="false" aria-describedby="opt-fab-hint">
<p id="opt-fab-hint" class="hint">제조사 능력 페이지와 한계값을 적은 JSON 파일. 경로는 프로젝트 작업 폴더 기준입니다.</p>
</div>
<div class="grid2">
<div class="field">
<label for="opt-catalog">카탈로그 CSV (<code>--catalog</code>)</label>
<input id="opt-catalog" data-opt="catalog" autocomplete="off" spellcheck="false">
</div>
<div class="field">
<label for="opt-catalog-date">카탈로그 날짜 (<code>--catalog-date</code>)</label>
<input id="opt-catalog-date" data-opt="catalog_date" autocomplete="off" spellcheck="false" placeholder="YYYY-MM-DD" aria-describedby="opt-catalog-date-hint">
<p id="opt-catalog-date-hint" class="hint">카탈로그를 내보낸 날짜 (카탈로그를 주면 필요합니다)</p>
</div>
</div>
<div class="check">
<input type="checkbox" id="opt-no-pdf" data-opt="no_pdf" data-shape="flag">
<label for="opt-no-pdf">PDF 생략 (<code>--no-pdf</code>: 단계 보고서를 .md와 .html로만)</label>
</div>
<details class="more">
<summary>추가 옵션</summary>
<div class="field">
<label for="opt-trust-host">신뢰할 호스트 (<code>--trust-host</code>, 한 줄에 하나)</label>
<textarea id="opt-trust-host" data-opt="trust_host" data-shape="lines" rows="2" spellcheck="false"></textarea>
</div>
<div class="field">
<label for="opt-datasheet-url">데이터시트 URL (<code>--datasheet-url</code>, 한 줄에 REF=URL 하나)</label>
<textarea id="opt-datasheet-url" data-opt="datasheet_url" data-shape="lines" rows="2" spellcheck="false"></textarea>
</div>
<div class="field">
<label for="opt-source-url">공식 문서 URL (<code>--source-url</code>, 한 줄에 ID=URL 하나)</label>
<textarea id="opt-source-url" data-opt="source_url" data-shape="lines" rows="2" spellcheck="false"></textarea>
</div>
<div class="grid2">
<div class="field">
<label for="opt-sources-dir">문서 보관 폴더 (<code>--sources-dir</code>)</label>
<input id="opt-sources-dir" data-opt="sources_dir" autocomplete="off" spellcheck="false">
</div>
<div class="field">
<label for="opt-regulatory">규제 후보 목록 (<code>--regulatory-candidates</code>)</label>
<input id="opt-regulatory" data-opt="regulatory_candidates" autocomplete="off" spellcheck="false">
</div>
<div class="field">
<label for="opt-catalog-authority">카탈로그 출처 (<code>--catalog-authority</code>)</label>
<input id="opt-catalog-authority" data-opt="catalog_authority" autocomplete="off">
</div>
<div class="field">
<label for="opt-catalog-supplier">공급사 이름 (<code>--catalog-supplier</code>)</label>
<input id="opt-catalog-supplier" data-opt="catalog_supplier" autocomplete="off">
</div>
<div class="field">
<label for="opt-claude-cli">claude 실행 파일 (<code>--llm-claude-cli</code>)</label>
<input id="opt-claude-cli" data-opt="llm_claude_cli" autocomplete="off" spellcheck="false">
</div>
<div class="field">
<label for="opt-claude-fallback">Claude Code CLI의 자체 백업 모델 (<code>--llm-claude-fallback-model</code>)</label>
<input id="opt-claude-fallback" data-opt="llm_claude_fallback_model" autocomplete="off" spellcheck="false">
</div>
<div class="field">
<label for="opt-browser">PDF 인쇄 브라우저 (<code>--browser</code>)</label>
<input id="opt-browser" data-opt="browser" autocomplete="off" spellcheck="false">
</div>
</div>
</details>
</fieldset>
<div class="actions run-buttons">
<button type="submit" class="primary" id="btn-run" value="run">실행</button>
<button type="submit" id="btn-review" value="review" formnovalidate>검토만</button>
<button type="submit" id="btn-stage-reports" value="stage-reports" formnovalidate>보고서 다시 쓰기</button>
</div>
<p class="hint">'실행'은 <code>ai-eda run</code>(답변과 모든 옵션), '검토만'은 <code>ai-eda review</code>(옵션 없음), '보고서 다시 쓰기'는 <code>ai-eda stage-reports</code>(PDF 생략·브라우저만)를 별도 프로세스로 시작합니다.</p>
<p id="run-message" role="status"></p>
</form>
<section class="box log-box" aria-labelledby="log-title">
<h3 id="log-title">실행 로그</h3>
<p id="log-status" role="status"></p>
<pre id="log" class="log" tabindex="0" aria-label="실행 로그" hidden></pre>
<h4>이전 실행</h4>
<ul id="runs-list" class="runs-list"></ul>
</section>
</div>
</section>

<section id="panel-schematic" class="panel" role="tabpanel" aria-labelledby="tab-schematic" hidden>
<div class="toolbar">
<button type="button" id="sch-zoom-in" aria-label="확대">확대 +</button>
<button type="button" id="sch-zoom-out" aria-label="축소">축소 −</button>
<button type="button" id="sch-fit">맞춤</button>
<span id="sch-download" class="toolbar-link"></span>
</div>
<p class="hint">컴파일러가 쓴 .kicad_sch를 그 파일 자신의 lib_symbols로 그린 미리 보기입니다. KiCad 렌더가 아니며 ERC 통과를 뜻하지 않습니다. 휠로 확대/축소, 끌어서 이동, 그림을 고른 뒤 + / − / 0 / 화살표 키도 됩니다.</p>
<div id="sch-facts"></div>
<div id="sch-view"></div>
</section>

<section id="panel-board" class="panel" role="tabpanel" aria-labelledby="tab-board" hidden>
<fieldset class="layers" id="board-layers">
<legend>보이는 층</legend>
<span class="check"><input type="checkbox" id="layer-F_Cu" data-layer-class="layer-F_Cu" checked><label for="layer-F_Cu">F.Cu</label></span>
<span class="check"><input type="checkbox" id="layer-B_Cu" data-layer-class="layer-B_Cu" checked><label for="layer-B_Cu">B.Cu</label></span>
<span class="check"><input type="checkbox" id="layer-pads" data-layer-class="pads" checked><label for="layer-pads">패드</label></span>
<span class="check"><input type="checkbox" id="layer-vias" data-layer-class="vias" checked><label for="layer-vias">비아</label></span>
<span class="check"><input type="checkbox" id="layer-outline" data-layer-class="outline" checked><label for="layer-outline">외곽</label></span>
<span class="check"><input type="checkbox" id="layer-labels" data-layer-class="labels" checked><label for="layer-labels">라벨</label></span>
<span id="board-download" class="toolbar-link"></span>
</fieldset>
<div id="board-facts"></div>
<div id="board-view"></div>
</section>

<section id="panel-simulation" class="panel" role="tabpanel" aria-labelledby="tab-simulation" hidden>
<div id="sim-view"></div>
</section>

<section id="panel-validation" class="panel" role="tabpanel" aria-labelledby="tab-validation" hidden>
<div id="val-head"></div>
<div class="toolbar">
<label for="val-filter">상태로 거르기</label>
<select id="val-filter">
<option value="all">전체</option>
<option value="FAIL">FAIL</option>
<option value="NOT_VERIFIED">NOT_VERIFIED</option>
<option value="PASS">PASS</option>
<option value="other">그 밖의 상태</option>
</select>
<span id="val-count" class="muted" role="status"></span>
</div>
<div id="val-view"></div>
</section>

<section id="panel-parts" class="panel" role="tabpanel" aria-labelledby="tab-parts" hidden>
<div id="parts-view"></div>
</section>

<section id="panel-reports" class="panel" role="tabpanel" aria-labelledby="tab-reports" hidden>
<div id="report-buttons" class="segmented" role="group" aria-label="보고서 고르기"></div>
<div id="report-links" class="report-links"></div>
<div id="report-view"></div>
</section>

<section id="panel-files" class="panel" role="tabpanel" aria-labelledby="tab-files" hidden>
<div id="files-view"></div>
</section>
</div>
</main>
</div>
</body>
</html>
"""

APP_CSS = r""":root {
  color-scheme: light;
  --surface: #fcfcfb;
  --panel: #ffffff;
  --sunken: #f6f5f1;
  --text: #0b0b0b;
  --muted: #52514e;
  --border: #e6e4dc;
  --accent: #2a78d6;
  --accent-dark: #1d5fae;
  --pass: #008300;
  --fail: #e34948;
  --other: #8a8983;
  --warn: #b77700;
  /* text in the warning colour: #b77700 is 3.5-3.7:1 on the list backgrounds (below AA for small text); this is 5.6:1 or more */
  --warn-text: #8a5800;
  --mono: ui-monospace, "Noto Sans Mono CJK KR", "D2Coding", "DejaVu Sans Mono", Consolas, monospace;
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
html { background: var(--surface); }
body {
  margin: 0; background: var(--surface); color: var(--text);
  font-family: "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "Apple SD Gothic Neo", system-ui, sans-serif;
  font-size: 14px; line-height: 1.5;
  /* Hangul wraps at spaces, like the words it is (never "다 / 릅니다"); a long unbroken run still wraps where a box asks for it */
  word-break: keep-all; overflow-wrap: break-word;
}
h1 { font-size: 18px; margin: 0; white-space: nowrap; }
h2 { font-size: 16px; margin: 0 0 8px; }
h3 { font-size: 15px; margin: 18px 0 8px; }
h4 { font-size: 14px; margin: 14px 0 6px; }
p { margin: 6px 0; }
code, .path { font-family: var(--mono); font-size: 12.5px; }
.path { overflow-wrap: anywhere; }
a { color: var(--accent-dark); text-underline-offset: 3px; }
a:focus-visible, button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible,
summary:focus-visible, [tabindex]:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.muted { color: var(--muted); }
.small { font-size: 12.5px; }
.hint { color: var(--muted); font-size: 12.5px; margin: 3px 0 8px; }
.sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
.skip { position: absolute; left: -9999px; top: 0; background: var(--panel); padding: 6px 10px; z-index: 10; }
.skip:focus { left: 8px; }

.topbar { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 18px; padding: 10px 20px; border-bottom: 1px solid var(--border); background: var(--panel); }
.topnote { margin: 0; color: var(--muted); font-size: 12.5px; }
.layout { display: grid; grid-template-columns: 300px minmax(0, 1fr); min-height: calc(100vh - 48px); }
.sidebar { background: var(--panel); border-right: 1px solid var(--border); padding: 14px 16px; min-width: 0; }
.side-block + .side-block { margin-top: 16px; padding-top: 12px; border-top: 1px solid var(--border); }
main { padding: 16px 24px 40px; min-width: 0; }
main:focus { outline: none; }
@media (max-width: 1100px) {
  .layout { grid-template-columns: 240px minmax(0, 1fr); }
  .sidebar { padding: 12px; }
  main { padding: 14px 16px 32px; }
}
@media (max-width: 760px) {
  .layout { grid-template-columns: minmax(0, 1fr); }
  .sidebar { border-right: 0; border-bottom: 1px solid var(--border); }
}

details > summary { cursor: pointer; }
details > summary h2 { display: inline; }

.project-list { list-style: none; margin: 8px 0; padding: 0; display: grid; gap: 4px; }
.project-list a { display: block; padding: 7px 9px; border: 1px solid var(--border); border-radius: 6px; text-decoration: none; color: var(--text); background: var(--surface); }
.project-list a:hover { border-color: var(--accent); }
.project-list a[aria-current="page"] { border-color: var(--accent); box-shadow: inset 3px 0 0 var(--accent); background: #f3f7fd; }
.pname { font-weight: 600; display: block; overflow-wrap: anywhere; }
.prun { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; font-size: 12.5px; color: var(--muted); margin-top: 2px; }
.pwarn { display: block; font-size: 12px; color: var(--warn-text); margin-top: 2px; }

.field { display: grid; gap: 3px; margin: 8px 0; min-width: 0; align-content: start; }
.field > label, .label-like { font-weight: 600; font-size: 13px; margin: 0; }
.check { display: inline-flex; align-items: flex-start; gap: 6px; margin: 6px 12px 2px 0; }
.check input { margin-top: 3px; }
.check label { font-weight: 600; font-size: 13px; }
input, select, textarea { font: inherit; font-size: 13.5px; color: var(--text); background: var(--panel); border: 1px solid #cfccc2; border-radius: 4px; padding: 5px 7px; min-width: 0; max-width: 100%; }
textarea { resize: vertical; width: 100%; }
input:not([type="checkbox"]), select { width: 100%; }
input:disabled, select:disabled, textarea:disabled { background: var(--sunken); color: var(--muted); }
input:user-invalid, select:user-invalid, textarea:user-invalid { border-color: var(--fail); }
.inline { display: grid; grid-template-columns: minmax(0, 14em) minmax(0, 1fr); gap: 6px; }
.grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0 14px; }
@media (max-width: 900px) { .grid2, .inline { grid-template-columns: minmax(0, 1fr); } }

button { font: inherit; font-size: 13.5px; border-radius: 5px; padding: 6px 13px; border: 1px solid #cfccc2; background: var(--panel); color: var(--text); cursor: pointer; }
button:hover:not(:disabled) { border-color: var(--accent); }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
button.primary:hover:not(:disabled) { background: var(--accent-dark); }
button:disabled { cursor: not-allowed; opacity: .55; }
.actions { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 4px; }

.status { display: inline-flex; align-items: center; gap: 5px; padding: 0 7px; border: 1px solid var(--st); border-radius: 4px; background: var(--panel); color: var(--text); font-size: 12px; font-weight: 600; line-height: 19px; white-space: nowrap; font-family: var(--mono); }
.status::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--st); flex: none; }
.status-pass { --st: var(--pass); color: var(--pass); }
.status-fail { --st: var(--fail); color: #c2302f; }
.status-other { --st: var(--other); color: var(--muted); }
.status-none { --st: var(--border); color: var(--muted); font-family: inherit; font-weight: 400; }
.tag { display: inline-block; font-size: 11.5px; padding: 0 6px; border-radius: 3px; background: var(--sunken); color: var(--muted); margin-left: 6px; font-weight: 600; vertical-align: 1px; }
.tag.req { background: #fdf0e0; color: var(--warn-text); }
.tag.model { background: #eef3fb; color: var(--accent-dark); }

.notice { border-left: 3px solid var(--other); background: var(--sunken); padding: 7px 10px; margin: 8px 0; border-radius: 0 4px 4px 0; overflow-wrap: anywhere; }
.notice.warn { border-color: var(--warn); }
.notice.error { border-color: var(--fail); }
.notice.ok { border-color: var(--pass); }
.empty { color: var(--muted); border: 1px dashed #d6d3c8; background: var(--panel); padding: 12px 14px; border-radius: 6px; margin: 8px 0; overflow-wrap: anywhere; }
.welcome { max-width: 720px; padding: 24px 0; }

.project-head { margin-bottom: 8px; }
.project-head h2 { font-size: 20px; margin: 0 0 4px; overflow-wrap: anywhere; }
.project-sub { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 2px 12px; font-size: 13px; }
.project-sub dt { color: var(--muted); }
.project-sub dd { margin: 0; overflow-wrap: anywhere; }
.request-text { white-space: pre-wrap; }

.tabs { display: flex; flex-wrap: wrap; gap: 2px; border-bottom: 1px solid var(--border); margin: 12px 0 0; position: sticky; top: 0; background: var(--surface); z-index: 5; }
.tabs [role="tab"] { border: 1px solid transparent; border-bottom: 0; border-radius: 6px 6px 0 0; background: transparent; padding: 8px 14px; color: var(--muted); font-weight: 600; margin-bottom: -1px; }
.tabs [role="tab"][aria-selected="true"] { background: var(--panel); border-color: var(--border); color: var(--text); box-shadow: inset 0 3px 0 var(--accent); }
.panel { padding: 12px 0; }

.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 6px; background: var(--panel); margin: 6px 0 12px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
caption { text-align: left; padding: 6px 8px; color: var(--muted); font-size: 12.5px; }
th, td { text-align: left; vertical-align: top; padding: 5px 9px; border-bottom: 1px solid var(--border); }
thead th { background: var(--sunken); color: var(--muted); font-weight: 600; white-space: nowrap; position: sticky; top: 0; }
tbody tr:last-child td { border-bottom: 0; }
td.msg { overflow-wrap: anywhere; min-width: 16em; }
td.ev { overflow-wrap: anywhere; min-width: 10em; }
/* identifiers (idCode) carry <wbr> after '.', '_' and ':': they break there when the column must narrow, never mid-word */
td.id, td.nowrap, td.vec, td.hash, td.time, td.word { white-space: nowrap; }
td.cell { white-space: nowrap; }
td.state { min-width: 9.5em; max-width: 15em; }
td.state .fresh { display: block; color: var(--muted); font-size: 12px; margin-top: 3px; overflow-wrap: anywhere; }
td.tool { min-width: 7em; max-width: 12em; }
td.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
@media (min-width: 1101px) {
  /* wide: identifiers stay whole (Chromium breaks at a <wbr> even under nowrap; a hidden one is no break opportunity) */
  td wbr { display: none; }
}
@media (max-width: 1100px) {
  /* two columns share the width: the tables fit the column instead of hiding their last columns off-screen */
  td.msg { min-width: 8em; }
  td.ev { min-width: 6em; }
  td.state, td.tool { min-width: 0; }
  td.id, td.vec, td.hash, td.time, td.word, td.num { white-space: normal; }
}
.facts-line { font-size: 13px; margin: 4px 0 8px; overflow-wrap: anywhere; }
tr.unreached td { color: var(--muted); padding-top: 2px; padding-bottom: 2px; font-size: 12.5px; }
td details summary, .long details summary { color: var(--accent-dark); font-size: 12.5px; }
.long details { display: inline; }
.long details[open] { display: block; margin-top: 4px; }
.long details[open] > summary { margin-bottom: 2px; }
.long-full { display: block; border-left: 2px solid var(--border); padding-left: 8px; color: var(--text); }
td pre { max-height: 280px; overflow: auto; font-size: 11.5px; background: var(--sunken); padding: 6px; margin: 4px 0 0; white-space: pre-wrap; overflow-wrap: anywhere; }
.sub-checks { list-style: none; padding: 0; margin: 0; display: grid; gap: 3px; }
.sub-checks li { display: grid; grid-template-columns: 11em max-content minmax(0, 1fr); gap: 6px; align-items: start; }
.evidence { list-style: none; margin: 0; padding: 0; display: grid; gap: 3px; font-size: 12.5px; }

.release { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; padding: 8px 12px; border: 1px solid var(--border); background: var(--panel); border-radius: 6px; margin: 6px 0; }
.release .msg { flex: 1 1 100%; color: var(--muted); font-size: 12.5px; overflow-wrap: anywhere; }

.run-area { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); gap: 18px; align-items: start; margin-top: 14px; }
@media (max-width: 1300px) { .run-area { grid-template-columns: minmax(0, 1fr); } }
.box { border: 1px solid var(--border); border-radius: 6px; background: var(--panel); padding: 10px 14px 12px; margin: 0 0 14px; min-width: 0; }
.box > legend, .sub > legend { font-weight: 700; font-size: 14px; padding: 0 4px; }
fieldset.sub { border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px 8px; margin: 10px 0; min-width: 0; }
fieldset.plain { border: 0; padding: 0; margin: 0; min-width: 0; }
.question { border-bottom: 1px dashed var(--border); padding-bottom: 8px; }
.question-text { white-space: pre-wrap; font-weight: 400; }
pre.question-table { font-family: var(--mono); font-size: 11.5px; line-height: 1.45; background: #fbfbf9; border: 1px solid var(--border); border-radius: 4px; padding: 6px 8px; margin: 2px 0 4px; max-height: 300px; overflow: auto; white-space: pre; }
.provider-list { list-style: none; padding: 0; margin: 0 0 6px; display: grid; gap: 3px; font-size: 12.5px; }
.provider-list li { overflow-wrap: anywhere; }
.pin { font-size: 13px; }
.task-row { display: grid; grid-template-columns: minmax(0, 13em) minmax(0, 1fr) max-content; gap: 6px; align-items: end; margin: 4px 0; }
@media (max-width: 900px) { .task-row { grid-template-columns: minmax(0, 1fr); } }
.more { margin-top: 8px; }
.more > summary { font-weight: 600; font-size: 13px; }
.run-buttons { margin-top: 0; }
.log-box { position: sticky; top: 52px; }
@media (max-width: 1300px) { .log-box { position: static; } }
pre.log { font-family: var(--mono); font-size: 12px; line-height: 1.45; background: #fbfbf9; border: 1px solid var(--border); border-radius: 4px; padding: 8px 10px; margin: 6px 0; max-height: 460px; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; }
.runs-list { list-style: none; padding: 0; margin: 0; display: flex; flex-wrap: wrap; gap: 6px; }
.runs-list button { font-size: 12.5px; padding: 3px 9px; font-family: var(--mono); }
.runs-list button[aria-pressed="true"] { border-color: var(--accent); background: #f3f7fd; }

.toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 4px 0 8px; }
.toolbar label { font-weight: 600; font-size: 13px; }
.toolbar select { width: auto; }
.toolbar-link { margin-left: auto; font-size: 13px; }
.viewport { height: 68vh; min-height: 420px; border: 1px solid var(--border); border-radius: 6px; background: #fff; overflow: hidden; touch-action: none; cursor: grab; user-select: none; }
.viewport.dragging { cursor: grabbing; }
.viewport svg { width: 100%; height: 100%; display: block; }
.layers { display: flex; flex-wrap: wrap; align-items: center; gap: 2px 4px; border: 1px solid var(--border); border-radius: 6px; padding: 4px 12px 6px; margin: 0 0 10px; background: var(--panel); }
.layers > legend { font-weight: 600; font-size: 13px; padding: 0 4px; }
.figure { border: 1px solid var(--border); border-radius: 6px; background: var(--panel); padding: 8px; margin: 8px 0 14px; overflow-x: auto; }
.figure svg { display: block; max-width: 100%; height: auto; margin: 0 auto; }
.figure svg.fit { width: 100%; max-width: 980px; }
.board-view.hide-layer-F_Cu .layer-F_Cu,
.board-view.hide-layer-B_Cu .layer-B_Cu,
.board-view.hide-pads .pads,
.board-view.hide-vias .vias,
.board-view.hide-outline .outline,
.board-view.hide-labels .labels { display: none; }
dl.facts { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 2px 14px; margin: 6px 0 12px; font-size: 13px; }
dl.facts dt { color: var(--muted); }
dl.facts dd { margin: 0; overflow-wrap: anywhere; }

.segmented { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 8px; }
.segmented button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: #fff; }
.report-links { display: flex; flex-wrap: wrap; gap: 6px 16px; align-items: center; font-size: 13px; margin: 4px 0 8px; }
.report-frame { width: 100%; height: 76vh; min-height: 480px; border: 1px solid var(--border); border-radius: 6px; background: #fff; display: block; }
.download-zip { display: inline-block; background: var(--accent); color: #fff; border-radius: 5px; padding: 7px 14px; text-decoration: none; font-weight: 600; }
.download-zip:hover { background: var(--accent-dark); }
.file-list { margin: 6px 0 12px; padding-left: 18px; }
.file-list li { margin: 2px 0; overflow-wrap: anywhere; }
"""

APP_JS = r"""'use strict';
// AI EDA 엔지니어 GUI. A view of the server's JSON: every status word is copied from the API (ir.validation /
// pipeline.json through build_report_data); nothing here computes a status. Text goes through text nodes only.

const TAB_IDS = ['overview', 'schematic', 'board', 'simulation', 'validation', 'parts', 'reports', 'files'];
const REPORTS = [
  {key: 'architecture', label: '이론', after: 'ARCHITECTURE'},
  {key: 'component_selection', label: '부품선정', after: 'COMPONENT_SELECTION'},
  {key: 'pcb', label: '회로', after: 'PCB'},
  {key: 'release', label: '최종', after: 'RELEASE'},
];
const KIND_LABELS = {'run': '실행', 'review': '검토만', 'stage-reports': '보고서 다시 쓰기'};
const POLL_MS = 1000;

const state = {
  projects: [], name: null, data: null, tab: 'overview', report: null,
  rendered: new Set(), loadSeq: 0, log: null, pollTimer: null, taskRow: 0, schematic: null, labels: {},
};

// ------------------------------------------------------------------ small helpers

const $ = (id) => document.getElementById(id);
const enc = encodeURIComponent;

function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (/^on/i.test(key) || key === 'style') throw new Error('no inline handler or style: ' + key);
    if (key === 'class') el.className = value;
    else if (key === 'for') el.htmlFor = value;
    else el.setAttribute(key, value === true ? '' : String(value));
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

function fill(el, ...children) { el.replaceChildren(); return append(el, children); }

function statusBadge(status) {
  const s = String(status || '');
  if (!s) return h('span', {class: 'status status-none'}, '기록 없음');
  const cls = s === 'PASS' ? 'pass' : (s === 'FAIL' ? 'fail' : 'other');
  return h('span', {class: 'status status-' + cls}, s);
}

function empty(text) { return h('p', {class: 'empty'}, text); }

// the Korean wording the server sent for a report label (ai_eda.gui.labels); any other text as written
function ko(text) {
  const s = String(text || '');
  return Object.prototype.hasOwnProperty.call(state.labels, s) ? state.labels[s] : s;
}
function addLabels(map) { if (map) Object.assign(state.labels, map); }

// an identifier as code that may break only after '.', '_' or ':' (compiler.kicad_pcb, component.existence.R1), never mid-word;
// a short one (f_osc, out_high) never breaks
const ID_BREAK_MIN = 13;
function idCode(text) {
  const s = String(text || '');
  if (s.length < ID_BREAK_MIN) return code(s);
  const parts = s.split(/(?<=[._:])/);
  return h('code', {}, parts.map((part, i) => (i ? [h('wbr'), part] : part)));
}

// a recorded message as it is; a long one shows its start and the rest behind a disclosure (presentation only)
const LONG_TEXT = 280;
function longText(text) {
  const s = String(text || '');
  if (s.length <= LONG_TEXT) return s;
  const cut = s.lastIndexOf(' ', LONG_TEXT);
  const head = s.slice(0, cut > LONG_TEXT / 2 ? cut : LONG_TEXT);
  return h('span', {class: 'long'}, head + ' …', h('details', {}, h('summary', {}, '전체 보기 (' + s.length + '자)'), h('span', {class: 'long-full'}, s)));
}
function notice(kind, label, text) { return h('p', {class: 'notice ' + kind}, h('strong', {}, label + ': '), text); }
function code(text) { return h('code', {}, text); }

// a previewed file's own facts (its artifact row, copied): generated from the current design hash? on disk as registered?
function fileFacts(st, what) {
  if (!st) return [h('p', {class: 'hint'}, what + ': ir.artifacts에 등록되지 않은 파일이라 설계 해시와 비교할 기록이 없습니다.')];
  const out = [h('p', {class: 'facts-line'}, h('strong', {}, what + ' 파일 '), code(st.path),
    ' · 설계 해시와 일치: ' + (st.matches_design_hash ? '예' : '아니오') + ' (' + ko(st.freshness) + ') · 디스크: ' + ko(st.disk))];
  if (!st.matches_design_hash) {
    out.push(notice('warn', '이전 설계의 파일', st.path + '은(는) 지금의 설계 해시에서 만든 것이 아닙니다: 아래는 그 파일 그대로이며 지금의 ir.json과 다를 수 있습니다 (\'실행\'이 IR에서 다시 컴파일합니다).'));
  }
  if (!st.disk_matches) out.push(notice('warn', '디스크의 파일', st.path + ': ' + ko(st.disk) + ' - 등록된 내용 해시와 같은 파일이 아닙니다.'));
  return out;
}

function table(columns, rows, caption) {
  const head = h('tr', {}, columns.map((c) => h('th', {scope: 'col'}, c)));
  const body = h('tbody', {}, rows);
  return h('div', {class: 'table-wrap'}, h('table', {}, caption ? h('caption', {}, caption) : null, h('thead', {}, head), body));
}

function td(content, cls) { return h('td', cls ? {class: cls} : {}, content); }

function fmtNum(x) {
  if (typeof x !== 'number' || !isFinite(x)) return String(x);
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 1e-3 && a < 1e6) return String(Number(x.toPrecision(6)));
  return x.toExponential(4);
}

function fmtBytes(n) {
  if (typeof n !== 'number') return '';
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
  return (n / 1024 / 1024).toFixed(2) + ' MB';
}

function shortHash(hash) {
  const s = String(hash || '');
  const m = /^sha256:([0-9a-f]{12})/.exec(s);
  return m ? 'sha256:' + m[1] + '…' : s;
}

function parseJson(text) {
  if (!text) return null;
  try { return JSON.parse(text); } catch (e) { return null; }
}

function filesUrl(path) { return '/files/' + enc(state.name) + '/' + String(path).split('/').map(enc).join('/'); }
function previewUrl(path) { return '/preview/' + enc(state.name) + '/' + path; }
function apiUrl(path) { return '/api/projects/' + enc(state.name) + (path ? '/' + path : ''); }

function downloadLink(path, text) { return h('a', {href: filesUrl(path), download: path.split('/').pop()}, text || path); }

// the workdir-relative form of an absolute path the IR records (a download link), or null when it lies outside
function inWorkdir(path) {
  const info = state.data && state.data.info;
  if (!info || !info.workdir || !path) return null;
  const norm = (p) => String(p).replace(/\\/g, '/');
  const base = norm(info.workdir).replace(/\/+$/, '') + '/';
  const p = norm(path);
  return p.startsWith(base) && p.length > base.length ? p.slice(base.length) : null;
}

class HttpError extends Error {
  constructor(status, line) { super(line || ('HTTP ' + status)); this.status = status; }
}

async function fetchText(url, init) {
  const r = await fetch(url, Object.assign({cache: 'no-store', credentials: 'same-origin'}, init || {}));
  return {ok: r.ok, status: r.status, text: await r.text()};
}

async function getJson(url) {
  const r = await fetchText(url);
  if (!r.ok) throw new HttpError(r.status, r.text.trim());
  return JSON.parse(r.text);
}

async function postJson(url, body) {
  let r;
  try {
    r = await fetchText(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  } catch (e) {
    return {ok: false, status: 0, text: 'GUI 서버에 연결할 수 없습니다 (ai-eda gui가 끝났는지 확인하십시오): ' + String(e.message || e), json: null};
  }
  return {ok: r.ok, status: r.status, text: r.text.trim(), json: r.ok ? parseJson(r.text) : null};
}

// an SVG the server drew, parsed and imported without scripts, event attributes, style attributes or external links
function importSvg(text) {
  const doc = new DOMParser().parseFromString(text, 'image/svg+xml');
  const root = doc.documentElement;
  if (!root || root.localName !== 'svg' || doc.getElementsByTagName('parsererror').length) throw new Error('SVG를 읽을 수 없습니다');
  for (const bad of root.querySelectorAll('script, foreignObject, iframe, image, use')) bad.remove();
  for (const el of [root, ...root.querySelectorAll('*')]) {
    for (const attr of Array.from(el.attributes)) {
      const name = attr.name.toLowerCase();
      if (name.startsWith('on') || name === 'style' || ((name === 'href' || name === 'xlink:href') && !attr.value.startsWith('#'))) el.removeAttribute(attr.name);
    }
  }
  return document.importNode(root, true);
}

// fetch a preview SVG into ``box``; a 404 is the server's Korean sentence (what produces the missing thing)
async function loadSvg(box, url) {
  fill(box, h('p', {class: 'muted'}, '불러오는 중…'));
  const r = await fetchText(url);
  if (!r.ok) {
    const line = r.text.trim();
    fill(box, r.status === 404 ? empty(line) : notice('error', '그릴 수 없음', line));
    return null;
  }
  try {
    const svg = importSvg(r.text);
    fill(box, svg);
    return svg;
  } catch (e) {
    fill(box, notice('error', '그릴 수 없음', String(e.message || e)));
    return null;
  }
}

// ------------------------------------------------------------------ the hash: #project=<name>&tab=<tab>&report=<key>

function readHash() {
  const params = new URLSearchParams(location.hash.replace(/^#/, ''));
  const tab = params.get('tab');
  return {project: params.get('project'), tab: TAB_IDS.includes(tab) ? tab : 'overview', report: params.get('report')};
}

function hashFor(project, tab, report) {
  const params = new URLSearchParams();
  if (project) params.set('project', project);
  if (tab && tab !== 'overview') params.set('tab', tab);
  if (report && tab === 'reports') params.set('report', report);
  return '#' + params.toString();
}

function go(project, tab, report) {
  const next = hashFor(project, tab, report);
  if (location.hash === next) applyHash();
  else location.hash = next;
}

async function applyHash() {
  const {project, tab, report} = readHash();
  if (report && report !== state.report) {
    state.report = report;
    state.rendered.delete('reports');
  }
  if (!project) { showWelcome(); return; }
  if (project !== state.name) await openProject(project, tab);
  else selectTab(tab);
}

// ------------------------------------------------------------------ the project list and the new-project form

function runWord(p) {
  if (p.error) return {word: '열 수 없음', note: p.error};
  const r = p.last_run;
  if (!r) return {word: '실행 기록 없음'};
  if (r.error) return {word: '기록을 읽을 수 없음', note: r.error};
  if (r.aborted) return {word: '중단됨', note: r.aborted + (r.aborted_stage ? ' (' + r.aborted_stage + ')' : '')};
  if (r.blocked) return {word: '차단됨(질문 ' + (r.open_questions || []).length + '개)'};
  return {word: '완료'};
}

async function loadProjects() {
  try {
    const data = await getJson('/api/projects');
    state.projects = data.projects;
    $('root').textContent = data.root;
    $('projects-error').hidden = true;
  } catch (e) {
    fill($('projects-error'), h('strong', {}, '오류: '), String(e.message || e));
    $('projects-error').hidden = false;
    return;
  }
  renderProjectList();
}

function renderProjectList() {
  const list = $('projects');
  list.replaceChildren();
  $('projects-empty').hidden = state.projects.length > 0;
  for (const p of state.projects) {
    const run = runWord(p);
    const last = p.last_run && !p.last_run.error ? p.last_run : null;
    const link = h('a', {href: hashFor(p.name, 'overview'), 'aria-current': p.name === state.name ? 'page' : null},
      h('span', {class: 'pname'}, p.name),
      h('span', {class: 'prun'}, h('span', {}, run.word),
        last && last.last_stage ? [h('span', {}, '· ' + last.last_stage), statusBadge(last.last_status)] : null),
      run.note ? h('span', {class: 'pwarn'}, run.note) : null,
      p.workdir_mismatch ? h('span', {class: 'pwarn'}, '주의: 기록된 작업 폴더가 이 폴더와 다릅니다') : null);
    list.append(h('li', {}, link));
  }
}

async function createProject(ev) {
  ev.preventDefault();
  const form = $('new-project');
  const out = $('new-result');
  if (!form.reportValidity()) return;
  const name = $('new-name').value.trim();
  const r = await postJson('/api/projects', {name, request: $('new-request').value});
  if (!r.ok) { fill(out, notice('error', '만들 수 없음', r.text)); return; }
  const example = exampleFor($('new-request').value);
  fill(out, notice('ok', '만들었습니다', name + ' - 개요에서 \'실행\'을 누르면 첫 질문(적용 분야, 시장)까지 진행합니다. LLM 없이는 요청문을 읽지 않으므로 그 뒤 템플릿 입력을 \'추가 답변\'에 한 줄씩 주십시오'
    + (example ? ': ' : ' (키는 추가 답변 도움말에 있습니다).')), example ? inputLines(example) : null);
  form.reset();
  showExampleInputs();
  await loadProjects();
  go(name, 'overview');
}

// the example requests carry their template's inputs (data-inputs, 'key=value|key=value'): without --llm they are answers
function exampleOptions() { return Array.from($('new-example').options).filter((o) => o.value && o.dataset.template); }
function exampleFor(request) { return exampleOptions().find((o) => o.value === String(request || '').trim()) || null; }
function inputPairs(option) { return option.dataset.inputs.split('|').filter(Boolean); }
function inputLines(option) { return h('p', {class: 'hint'}, inputPairs(option).map((a, i) => [i ? ', ' : '', code(a)])); }

function showExampleInputs() {
  const option = exampleFor($('new-request').value);
  const box = $('new-example-inputs');
  box.hidden = !option;
  if (option) fill(box, '이 예시의 템플릿 입력(' + option.dataset.template + ', 추가 답변에 한 줄씩): ', inputPairs(option).map((a, i) => [i ? ', ' : '', code(a)]));
}

function fillTemplateKeys() {
  fill($('template-keys'), exampleOptions().map((o, i) => [i ? ' · ' : '', o.dataset.template + ' ',
    inputPairs(o).map((a, j) => [j ? ', ' : '', code(a.split('=')[0])])]));
}

// ------------------------------------------------------------------ one project

function showWelcome() {
  stopPolling();
  state.name = null; state.data = null;
  $('welcome').hidden = false;
  $('project-view').hidden = true;
  document.title = 'AI EDA 엔지니어';
  renderProjectList();
}

async function openProject(name, tab) {
  stopPolling();
  state.name = name; state.data = null; state.log = null; state.rendered.clear(); state.schematic = null;
  resetRunForm();
  $('welcome').hidden = true;
  $('project-view').hidden = false;
  $('project-title').textContent = name;
  document.title = name + ' - AI EDA 엔지니어';
  fill($('project-sub'));
  fill($('project-notices'), h('p', {class: 'muted'}, '불러오는 중…'));
  renderProjectList();
  selectTab(tab, true);
  await refreshProject();
  if (!state.data || state.name !== name) return;
  // the run going now (polled live), else the latest log as it ended
  const runs = state.data.runs || [];
  if (state.data.active_run) watchRun(state.data.active_run, true);
  else if (runs.length) watchRun(runs[runs.length - 1].id, false);
}

async function refreshProject() {
  const name = state.name;
  const seq = ++state.loadSeq;
  let data;
  try {
    data = await getJson(apiUrl(''));
  } catch (e) {
    if (seq !== state.loadSeq || name !== state.name) return;
    state.data = null;
    fill($('project-notices'), notice('error', '이 프로젝트를 열 수 없습니다', String(e.message || e)));
    for (const id of TAB_IDS) state.rendered.delete(id);
    return;
  }
  if (seq !== state.loadSeq || name !== state.name) return;
  state.data = data;
  state.labels = Object.assign({}, data.labels || {});
  state.rendered.clear();
  renderHead();
  fillRunForm();
  renderTab(state.tab);
}

function renderHead() {
  const d = state.data, info = d.info, meta = d.report.meta;
  const reqText = d.report.requirements.raw_input;
  const stages = d.report.stages;
  fill($('project-sub'), h('dl', {class: 'project-sub'},
    h('dt', {}, '요청문'), h('dd', {class: 'request-text'}, reqText || '(비어 있음)'),
    h('dt', {}, '작업 폴더'), h('dd', {class: 'path'}, info.workdir || ''),
    h('dt', {}, '설계 해시'), h('dd', {}, code(meta.design_hash)),
    h('dt', {}, '실행 기록'), h('dd', {}, stages ? [ko(stages.run_hash_label), ' · ', ko(meta.pipeline_note)] : '실행 기록 없음')));
  const notices = [];
  if (info.warning) notices.push(notice('warn', '주의', info.warning));
  if (info.last_run && info.last_run.error) notices.push(notice('error', 'pipeline.json을 읽을 수 없음', info.last_run.error));
  const required = d.questions.required.length;
  if (required) notices.push(h('p', {class: 'notice warn'}, h('strong', {}, '답이 필요합니다: '),
    '마지막 실행이 필수 질문 ' + required + '개에서 멈췄습니다. ', h('a', {href: '#questions', 'data-local': '1'}, '개요의 질문'), '에 답하고 \'실행\'을 누르십시오.'));
  fill($('project-notices'), notices);
}

// ------------------------------------------------------------------ tabs

function selectTab(tab, silent) {
  state.tab = tab;
  for (const id of TAB_IDS) {
    const button = $('tab-' + id);
    const on = id === tab;
    button.setAttribute('aria-selected', on ? 'true' : 'false');
    button.tabIndex = on ? 0 : -1;
    $('panel-' + id).hidden = !on;
  }
  if (!silent) renderTab(tab);
}

function renderTab(tab) {
  if (!state.data || state.rendered.has(tab)) return;
  state.rendered.add(tab);
  const renderers = {
    overview: renderOverview, schematic: renderSchematic, board: renderBoard, simulation: renderSimulation,
    validation: renderValidation, parts: renderParts, reports: renderReports, files: renderFiles,
  };
  Promise.resolve().then(() => renderers[tab]()).catch((e) => {
    fill($('panel-' + tab).querySelector('[id$="-view"]') || $('project-notices'), notice('error', '그릴 수 없음', String(e.message || e)));
  });
}

function onTabKey(ev) {
  const i = TAB_IDS.indexOf(ev.target.dataset.tab);
  if (i < 0) return;
  let next = null;
  if (ev.key === 'ArrowRight') next = TAB_IDS[(i + 1) % TAB_IDS.length];
  else if (ev.key === 'ArrowLeft') next = TAB_IDS[(i - 1 + TAB_IDS.length) % TAB_IDS.length];
  else if (ev.key === 'Home') next = TAB_IDS[0];
  else if (ev.key === 'End') next = TAB_IDS[TAB_IDS.length - 1];
  if (!next) return;
  ev.preventDefault();
  $('tab-' + next).focus();
  go(state.name, next, state.report);
}

// ------------------------------------------------------------------ 개요: stages, questions, run options, log

function renderOverview() {
  const rep = state.data.report;
  const rel = rep.release;
  fill($('ov-release'), rel.status
    ? h('div', {class: 'release'}, h('strong', {}, 'RELEASE'), statusBadge(rel.status), h('span', {class: 'muted small'}, ko(rel.freshness)),
        h('span', {class: 'msg'}, longText(rel.message)))
    : empty('RELEASE 판정 없음: 실행이 RELEASE 단계에 이르면 기록됩니다.'));
  const stages = rep.stages;
  if (!stages) {
    // a pipeline.json that cannot be read is the notice above the tabs (info.last_run.error); no English line repeats it here
    const unreadable = state.data.info.last_run && state.data.info.last_run.error;
    fill($('ov-stages'), empty(unreadable
      ? '단계 기록을 읽을 수 없음: 위 알림의 pipeline.json 오류를 보십시오. \'실행\'이 기록을 새로 씁니다.'
      : '단계 기록 없음: 아직 실행하지 않았습니다. \'실행\'이 ai-eda run을 시작하고, 그 기록(pipeline.json)의 19단계가 여기에 나옵니다.'));
  } else {
    const rows = stages.rows.map((row) => h('tr', {class: row.reached ? '' : 'unreached'},
      td(idCode(row.stage), 'id'),
      td(row.reached ? statusBadge(row.status) : h('span', {class: 'muted'}, '도달하지 않음'), 'nowrap'),
      td([row.reached ? longText(row.message) : '', row.questions ? h('span', {class: 'tag req'}, '질문 ' + row.questions + '개') : null], 'msg')));
    const extra = [];
    if (stages.blocked) extra.push(notice('warn', '차단됨', '실행이 ' + (stages.current || '') + ' 단계의 질문에서 멈췄습니다.'));
    if (stages.aborted) extra.push(notice('error', '중단됨', stages.aborted + (stages.aborted_stage ? ' (' + stages.aborted_stage + ')' : '')));
    fill($('ov-stages'), extra, table(['단계', '상태', '요약'], rows, '기록된 실행의 단계별 결과 (pipeline.json, 그대로 복사)'));
  }
  renderQuestions();
  renderRuns();
  renderLog();
}

function renderQuestions() {
  const q = state.data.questions;
  const box = $('ov-questions');
  box.replaceChildren();
  if (!q.required.length && !q.optional.length) {
    box.append(empty(state.data.info.last_run ? '열린 질문 없음: 마지막 실행이 남긴 질문이 없습니다.' : '아직 없음: \'실행\'을 누르면 파이프라인이 필요한 질문을 여기에 냅니다.'));
    return;
  }
  const confirmKey = state.data.answers_rule.confirm_key;
  const confirming = q.required.some((x) => x.key === confirmKey);
  if (q.required.length) append(box, [h('h4', {}, '필수 질문 ' + q.required.length + '개'), q.required.map((x) => questionField(x, true))]);
  if (!q.optional.length) return;
  const optional = [h('h4', {}, '선택 질문 ' + q.optional.length + '개'), q.optional.map((x) => questionField(x, false))];
  if (!confirming) { append(box, optional); return; }
  // the design table is open: its confirmation counts only when it is sent alone (the circuit agent's rule), so the optional
  // questions are folded away with why
  append(box, [h('details', {class: 'more', id: 'optional-questions'},
    h('summary', {}, '선택 질문 ' + q.optional.length + '개 (' + confirmKey + ' 답과 같은 실행에 보내지 마십시오)'),
    h('p', {class: 'notice warn'}, h('strong', {}, '주의: '), '선택 질문에 답하면 그 값이 확정된 요구사항이 됩니다. 같은 실행에 ', code(confirmKey + '=yes'),
      '가 있으면 CLI가 확인을 무시하고 표를 다시 묻고, 템플릿이 쓰지 않는 설계 요구사항(예: ', code('operating_temperature'),
      ')이면 그 템플릿은 설계를 내지 않습니다 (이유는 개요의 architecture 줄에 나옵니다).'),
    optional)]);
}

// the answers that void confirm_design=yes when they go with it (every key outside the control keys, and the requirement decisions)
function confirmVoiders(answers) {
  const rule = state.data.answers_rule;
  if (!answers[rule.confirm_key]) return [];
  return Object.keys(answers).filter((k) => k !== rule.confirm_key && (!rule.control_keys.includes(k) || rule.decision_keys.includes(k)));
}

function questionField(q, required) {
  const id = 'answer-' + q.key;
  const why = q.rationale ? id + '-why' : null;
  // a question that carries a table (confirm_design, confirm_parts ...): its first line labels the field, the whole text follows as written
  const text = String(q.question || '');
  const cut = text.indexOf('\n');
  const long = cut >= 0 ? text : null;
  const bodyId = long ? id + '-text' : null;
  const label = h('label', {for: id}, code(q.key), required ? h('span', {class: 'tag req'}, '필수') : null,
    q.source === 'llm' ? h('span', {class: 'tag model'}, '(모델 질문)') : null,
    h('span', {class: 'question-text'}, ' ' + (long ? text.slice(0, cut) : text)));
  const described = [bodyId, why].filter(Boolean).join(' ') || null;
  let input;
  if (q.options && q.options.length) {
    input = h('select', {id, 'data-answer': q.key, 'aria-describedby': described},
      h('option', {value: ''}, required ? '(고르십시오)' : '(답하지 않음)'), q.options.map((o) => h('option', {value: o}, o)));
  } else {
    input = h('input', {id, type: 'text', 'data-answer': q.key, autocomplete: 'off', 'aria-describedby': described});
  }
  const alone = state.data.answers_rule && q.key === state.data.answers_rule.confirm_key
    ? h('p', {class: 'hint'}, '이 확인은 다른 답 없이 혼자 보내십시오: 같은 실행에 제어 답(예: ', code('pcb.routing=skip'), ')이 아닌 답이 함께 가면 CLI가 이 확인을 무시하고 표를 다시 묻습니다.')
    : null;
  return h('div', {class: 'field question'}, label,
    long ? h('pre', {class: 'question-table', id: bodyId, tabindex: '0'}, long) : null,
    input, alone, why ? h('p', {class: 'hint', id: why}, '이유: ' + q.rationale) : null);
}

function collectAnswers() {
  const answers = {};
  for (const el of document.querySelectorAll('[data-answer]')) {
    const value = el.value.trim();
    if (value) answers[el.dataset.answer] = value;
  }
  const lines = $('extra-answers').value.split('\n');
  for (const [i, raw] of lines.entries()) {
    const line = raw.trim();
    if (!line) continue;
    const at = line.indexOf('=');
    if (at <= 0) throw new Error('추가 답변 ' + (i + 1) + '번째 줄은 키=값 형식이어야 합니다: ' + line);
    answers[line.slice(0, at).trim()] = line.slice(at + 1).trim();
  }
  return answers;
}

// the run form: the fields are RUN_OPTIONS dests (data-opt); the server renders them with the CLI's own flag builder

function resetRunForm() {
  $('run-form').reset();
  $('extra-answers').value = '';
  $('task-models').replaceChildren();
  fill($('run-message'));
  updateLlmFields();
}

function providerRows() { return (state.data && state.data.llm && state.data.llm.providers) || []; }

function billingLabel(billing) {
  if (billing === 'subscription') return '구독';
  if (billing === 'per_call') return '호출당 과금';
  return String(billing || '');
}

function providerLabel(row) {
  if (row.name === 'claude') return 'claude (구독, Claude Code CLI 로그인)';
  if (row.name === 'openrouter') return 'openrouter (호출당 과금)';
  return row.name + ' (' + billingLabel(row.billing) + ')';
}

function fillRunForm() {
  const rows = providerRows();
  const byName = Object.fromEntries(rows.map((r) => [r.name, r]));
  const select = $('opt-llm');
  const previous = select.value;
  const combos = rows.map((r) => [r.name]);
  for (const a of rows) for (const b of rows) if (a !== b) combos.push([a.name, b.name]);
  select.replaceChildren(h('option', {value: ''}, '사용 안 함'));
  for (const names of combos) {
    const missing = names.filter((n) => !byName[n].available);
    const label = names.map((n) => providerLabel(byName[n])).join(' + ') + (names.length > 1 ? ' - 앞의 것이 기본 제공자' : '') + (missing.length ? ' - 사용 불가' : '');
    const option = h('option', {value: names.join(','), title: missing.length ? missing.map((n) => n + ': ' + byName[n].reason).join('; ') : null}, label);
    option.disabled = missing.length > 0;
    select.append(option);
  }
  const keep = Array.from(select.options).find((o) => o.value === previous && !o.disabled);
  select.value = keep ? previous : '';
  fill($('llm-providers'), rows.map((r) => h('li', {},
    h('strong', {}, r.name), ' ', r.available ? '사용 가능' : '사용 불가', ' (' + billingLabel(r.billing) + (r.logged_in === false ? ', 로그인 안 됨' : '') + '): ',
    h('span', {class: 'muted'}, r.reason))));
  if (!rows.length) fill($('llm-providers'), h('li', {class: 'muted'}, '제공자 정보 없음'));
  const tasks = (state.data.llm && state.data.llm.tasks) || [];
  for (const sel of document.querySelectorAll('#task-models select')) fillTaskSelect(sel, tasks);
  const pinned = state.data.llm ? state.data.llm.pinned_model : null;
  fill($('llm-pin'), pinned
    ? ['이 프로젝트에 고정된 모델: ', code(pinned), ' ', h('button', {type: 'button', id: 'use-pinned'}, '고정된 모델 넣기')]
    : '고정된 모델 없음: 모델 호출을 한 첫 실행이 그 모델 사양을 이 프로젝트에 고정합니다 (ir.requirements.llm_model_spec).');
  const usePinned = $('use-pinned');
  if (usePinned) usePinned.addEventListener('click', () => { $('opt-llm-model').value = pinned; updateLlmFields(); });
  const busy = Boolean(state.data.active_run) || Boolean(state.log && state.log.running);
  setRunBusy(busy);
  updateLlmFields();
}

function fillTaskSelect(select, tasks) {
  const previous = select.value;
  fill(select, tasks.map((t) => h('option', {value: t}, t)));
  if (tasks.includes(previous)) select.value = previous;
}

function addTaskRow() {
  const n = ++state.taskRow;
  const tasks = (state.data && state.data.llm && state.data.llm.tasks) || [];
  const select = h('select', {id: 'task-kind-' + n});
  fillTaskSelect(select, tasks);
  const remove = h('button', {type: 'button'}, '삭제');
  const row = h('div', {class: 'task-row'},
    h('div', {class: 'field'}, h('label', {for: 'task-kind-' + n}, '작업'), select),
    h('div', {class: 'field'}, h('label', {for: 'task-spec-' + n}, '모델 (provider:model)'),
      h('input', {id: 'task-spec-' + n, autocomplete: 'off', spellcheck: 'false', placeholder: 'provider:model'})),
    h('div', {class: 'field'}, remove));
  remove.addEventListener('click', () => { row.remove(); $('add-task-model').focus(); });
  $('task-models').append(row);
  select.focus();
}

function selectedProviders() {
  const byName = Object.fromEntries(providerRows().map((r) => [r.name, r]));
  return $('opt-llm').value.split(',').filter(Boolean).map((n) => byName[n]).filter(Boolean);
}

function updateLlmFields() {
  const chosen = selectedProviders();
  $('llm-fields').disabled = chosen.length === 0;
  const fallback = $('opt-llm-fallback').value;
  $('opt-llm-fallback-spec').disabled = fallback !== 'spec';
  $('opt-llm-fallback-spec').required = fallback === 'spec';
  // a per-call provider needs one budget (the PAID_API_CALL approval); claude alone needs none
  const perCall = chosen.some((r) => r.billing === 'per_call');
  const usd = $('opt-budget-usd'), tokens = $('opt-budget-tokens');
  const needBudget = perCall && !usd.value.trim() && !tokens.value.trim();
  usd.required = needBudget;
  tokens.required = needBudget;
  // a hint only (the CLI compares the specs it resolves and refuses with exit 2); an empty field is the first provider's
  // default model, as the server resolved it with the CLI's own functions (llm.default_specs)
  const pinned = state.data && state.data.llm ? state.data.llm.pinned_model : null;
  const defaults = (state.data && state.data.llm && state.data.llm.default_specs) || {};
  const typed = $('opt-llm-model').value.trim();
  const effective = typed || (chosen.length ? defaults[chosen[0].name] || '' : '');
  const note = $('llm-pin-note');
  if (chosen.length && pinned && effective && effective !== pinned && !$('opt-llm-allow').checked) {
    fill(note, h('strong', {}, '주의: '), typed
      ? ['입력한 모델 ', code(typed), '은(는) 이 프로젝트에 고정된 모델 ', code(pinned), '과(와) 글자가 다릅니다. 다른 모델이면 \'모델 변경 허용\'을 체크하지 않는 한 CLI가 실행을 거부합니다(종료 코드 2).']
      : ['모델 칸을 비워 두면 이번 실행은 기본 모델 ', code(effective), '을(를) 씁니다. 이 프로젝트에 고정된 모델 ', code(pinned),
        '과(와) 다르므로 \'고정된 모델 넣기\'를 누르거나 \'모델 변경 허용\'을 체크하지 않으면 CLI가 실행을 거부합니다(종료 코드 2).']);
    note.hidden = false;
  } else {
    note.hidden = true;
  }
  $('opt-catalog-date').required = Boolean($('opt-catalog').value.trim());
}

function optionValue(el) {
  switch (el.dataset.shape) {
    case 'flag':
      return el.checked ? true : null;
    case 'lines': {
      const items = el.value.split('\n').map((s) => s.trim()).filter(Boolean);
      return items.length ? items : null;
    }
    case 'fallback': {
      const choice = $('opt-llm-fallback').value;
      if (choice === 'same') return ['same'];
      const spec = $('opt-llm-fallback-spec').value.trim();
      return choice === 'spec' && spec ? [spec] : null;
    }
    case 'tasks': {
      const pairs = {};
      for (const row of el.querySelectorAll('.task-row')) {
        const task = row.querySelector('select').value;
        const spec = row.querySelector('input').value.trim();
        if (task && spec) pairs[task] = spec;
      }
      return Object.keys(pairs).length ? pairs : null;
    }
    default: {
      const text = el.value.trim();
      return text ? text : null;
    }
  }
}

function collectOptions(kind) {
  const allowed = new Set((state.data.run_options || {})[kind] || []);
  const out = {};
  for (const el of document.querySelectorAll('#run-form [data-opt]')) {
    const dest = el.dataset.opt;
    if (!allowed.has(dest)) continue;
    if ((el.matches(':disabled')) || el.closest('fieldset:disabled')) continue;
    const value = optionValue(el);
    if (value !== null) out[dest] = value;
  }
  return out;
}

function setRunBusy(busy) {
  for (const id of ['btn-run', 'btn-review', 'btn-stage-reports']) $(id).disabled = busy;
}

async function submitRun(ev) {
  ev.preventDefault();
  if (!state.data) return;
  const kind = (ev.submitter && ev.submitter.value) || 'run';
  const out = $('run-message');
  updateLlmFields();
  const body = {options: collectOptions(kind)};
  if (kind === 'run') {
    if (!$('run-form').reportValidity()) return;
    try { body.answers = collectAnswers(); } catch (e) { fill(out, notice('error', '시작하지 않았습니다', e.message)); return; }
    const voiders = confirmVoiders(body.answers);
    if (voiders.length) {
      const key = state.data.answers_rule.confirm_key;
      fill(out, notice('error', '시작하지 않았습니다', key + ' 답은 다른 답 없이 혼자 보내야 합니다: 같은 실행에 ' + voiders.join(', ')
        + ' 답이 함께 가면 CLI(circuit 단계)가 ' + key + ' 답을 무시하고 표를 다시 묻습니다. 그 답들을 비우고 먼저 보내거나, ' + key + ' 칸을 비우십시오.'));
      return;
    }
  }
  setRunBusy(true);
  fill(out, h('span', {class: 'muted'}, KIND_LABELS[kind] + ' 시작하는 중…'));
  const r = await postJson(apiUrl(kind), body);
  if (!r.ok) {
    setRunBusy(Boolean(state.data.active_run));
    fill(out, notice('error', '시작하지 못했습니다 (HTTP ' + r.status + ')', r.text));
    return;
  }
  // the answers went with this run: never send them again by accident (a template input beside confirm_design=yes would void the confirmation)
  const sent = Object.entries(body.answers || {}).map(([k, v]) => k + '=' + v);
  $('extra-answers').value = '';
  fill(out, notice('ok', '시작했습니다', r.json.run_id + ' (' + KIND_LABELS[kind] + ') - 아래 실행 로그가 1초마다 새로 읽힙니다'),
    sent.length ? h('p', {class: 'hint'}, '보낸 답변: ', sent.map((a, i) => [i ? ', ' : '', code(a)])) : null);
  watchRun(r.json.run_id, true);
}

// the live log: polled every second while the run is going; the project is re-read when it ends

function stopPolling() {
  if (state.pollTimer) clearTimeout(state.pollTimer);
  state.pollTimer = null;
}

function watchRun(runId, live) {
  stopPolling();
  state.log = {runId, text: '', running: live, exit: null, bytes: 0, live, error: null, read: false};
  renderLog();
  renderRuns();
  pollRun();
}

async function pollRun() {
  const name = state.name, log = state.log;
  if (!log) return;
  let s;
  try {
    s = await getJson(apiUrl('runs/' + enc(log.runId)));
  } catch (e) {
    if (state.log !== log || state.name !== name) return;
    log.error = String(e.message || e);
    log.running = false;
    renderLog();
    return;
  }
  if (state.log !== log || state.name !== name) return;
  log.text = s.log_tail; log.running = s.running; log.exit = s.exit_code; log.bytes = s.log_bytes; log.read = true;
  if (s.running) log.live = true;
  renderLog();
  if (s.running) {
    setRunBusy(true);
    state.pollTimer = setTimeout(pollRun, POLL_MS);
  } else if (log.live) {
    log.live = false;
    await refreshProject();
    await loadProjects();
  }
}

function renderLog() {
  const log = state.log;
  const pre = $('log'), status = $('log-status');
  if (!log) {
    pre.hidden = true;
    pre.textContent = '';
    fill(status, empty('아직 없음: \'실행\', \'검토만\', \'보고서 다시 쓰기\'가 로그를 씁니다 (작업 폴더의 gui/runs/).'));
    return;
  }
  const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 40;
  pre.hidden = false;
  pre.textContent = log.text;
  if (log.running || atBottom) pre.scrollTop = pre.scrollHeight;
  let line;
  if (log.error) line = notice('error', '로그를 읽을 수 없음', log.error);
  else if (!log.read) line = h('span', {}, h('strong', {}, log.runId), ' 로그를 읽는 중…');
  else if (log.running) line = h('span', {}, h('strong', {}, log.runId), ' 실행 중… (1초마다 새로 읽습니다, ' + fmtBytes(log.bytes) + ')');
  else if (log.exit !== null && log.exit !== undefined) line = h('span', {}, h('strong', {}, log.runId), ' 끝남 - 종료 코드 ' + log.exit);
  else line = h('span', {}, h('strong', {}, log.runId), ' 종료 코드 기록 없음');
  fill(status, line, ' ', h('a', {href: filesUrl('gui/runs/' + log.runId + '.log'), download: log.runId + '.log'}, '로그 내려받기'));
}

function renderRuns() {
  const runs = (state.data && state.data.runs) || [];
  const list = $('runs-list');
  list.replaceChildren();
  if (!runs.length) { list.append(h('li', {class: 'muted small'}, '없음')); return; }
  for (const r of runs.slice().reverse()) {
    const pressed = state.log && state.log.runId === r.id;
    const b = h('button', {type: 'button', 'aria-pressed': pressed ? 'true' : 'false'},
      r.id + ' · ' + (KIND_LABELS[r.kind] || r.kind) + ' · ' + fmtBytes(r.size) + (r.running ? ' · 실행 중' : ''));
    b.addEventListener('click', () => watchRun(r.id, r.running));
    list.append(h('li', {}, b));
  }
}

// ------------------------------------------------------------------ 회로도: pan / zoom by the viewBox

function parseBox(text) {
  const v = String(text || '').trim().split(/[\s,]+/).map(Number);
  return v.length === 4 && v.every(isFinite) && v[2] > 0 && v[3] > 0 ? {x: v[0], y: v[1], w: v[2], h: v[3]} : null;
}

async function renderSchematic() {
  const file = state.data.previews.schematic_file;
  const st = state.data.previews.schematic_state;
  fill($('sch-download'), file ? downloadLink(file, file + ' 내려받기') : null);
  fill($('sch-facts'), st ? fileFacts(st, '회로도') : null);
  state.schematic = null;
  const svg = await loadSvg($('sch-view'), previewUrl('schematic.svg'));
  if (!svg) return;
  const sheet = parseBox(svg.getAttribute('viewBox'));
  const content = parseBox(svg.getAttribute('data-content-box')) || sheet;
  if (!sheet) { fill($('sch-view'), notice('error', '그릴 수 없음', 'viewBox가 없는 SVG입니다')); return; }
  svg.removeAttribute('width');
  svg.removeAttribute('height');
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  const box = h('div', {class: 'viewport', tabindex: '0', role: 'group', 'aria-label': '회로도 미리 보기: 휠로 확대/축소, 끌어서 이동, + / − / 0 / 화살표 키'}, svg);
  fill($('sch-view'), box);
  const margin = Math.max(content.w, content.h) * 0.06;
  const fit = {x: content.x - margin, y: content.y - margin, w: content.w + 2 * margin, h: content.h + 2 * margin};
  const view = {svg, box, sheet, fit, cur: Object.assign({}, fit)};
  state.schematic = view;
  applyView(view);
  box.addEventListener('wheel', (ev) => { ev.preventDefault(); zoomAt(view, ev.deltaY > 0 ? 1.2 : 1 / 1.2, clientPoint(view, ev.clientX, ev.clientY)); }, {passive: false});
  let drag = null;
  box.addEventListener('pointerdown', (ev) => {
    if (ev.button !== 0) return;
    ev.preventDefault();  // no text selection while dragging; focus the viewport for the keys instead
    box.focus();
    drag = {id: ev.pointerId, start: clientPoint(view, ev.clientX, ev.clientY)};
    box.setPointerCapture(ev.pointerId);
    box.classList.add('dragging');
  });
  box.addEventListener('pointermove', (ev) => {
    if (!drag || ev.pointerId !== drag.id) return;
    const p = clientPoint(view, ev.clientX, ev.clientY);
    view.cur.x -= p.x - drag.start.x;
    view.cur.y -= p.y - drag.start.y;
    applyView(view);
  });
  const end = (ev) => { if (drag && ev.pointerId === drag.id) { drag = null; box.classList.remove('dragging'); } };
  box.addEventListener('pointerup', end);
  box.addEventListener('pointercancel', end);
  box.addEventListener('keydown', (ev) => {
    const step = 0.1;
    const keys = {
      '+': () => zoomAt(view, 1 / 1.25), '=': () => zoomAt(view, 1 / 1.25), '-': () => zoomAt(view, 1.25), '0': () => fitView(view),
      'ArrowLeft': () => panBy(view, -step, 0), 'ArrowRight': () => panBy(view, step, 0),
      'ArrowUp': () => panBy(view, 0, -step), 'ArrowDown': () => panBy(view, 0, step),
    };
    const f = keys[ev.key];
    if (f) { ev.preventDefault(); f(); }
  });
}

function applyView(view) {
  const c = view.cur;
  view.svg.setAttribute('viewBox', [c.x, c.y, c.w, c.h].map((v) => Number(v.toFixed(3))).join(' '));
}

function clientPoint(view, x, y) {
  const m = view.svg.getScreenCTM();
  if (!m) return {x: view.cur.x + view.cur.w / 2, y: view.cur.y + view.cur.h / 2};
  const p = new DOMPoint(x, y).matrixTransform(m.inverse());
  return {x: p.x, y: p.y};
}

function zoomAt(view, factor, at) {
  const c = view.cur;
  const center = at || {x: c.x + c.w / 2, y: c.y + c.h / 2};
  const minW = view.fit.w / 40, maxW = Math.max(view.sheet.w, view.fit.w) * 3;
  const w = Math.min(maxW, Math.max(minW, c.w * factor));
  const k = w / c.w;
  view.cur = {x: center.x - (center.x - c.x) * k, y: center.y - (center.y - c.y) * k, w: c.w * k, h: c.h * k};
  applyView(view);
}

function panBy(view, fx, fy) {
  view.cur.x += view.cur.w * fx;
  view.cur.y += view.cur.h * fy;
  applyView(view);
}

function fitView(view) {
  view.cur = Object.assign({}, view.fit);
  applyView(view);
}

// ------------------------------------------------------------------ 기판: layer checkboxes toggle the SVG's classes

async function renderBoard() {
  const file = state.data.previews.board_file;
  const st = state.data.previews.board_state;
  fill($('board-download'), file ? downloadLink(file, file + ' 내려받기') : null);
  fill($('board-facts'), h('p', {class: 'hint'}, '그림은 지금의 ir.json(ir.pcb의 배치·배선)과 디스크의 KiCad 라이브러리로 그립니다. 내려받는 .kicad_pcb는 컴파일된 파일입니다:'),
    st ? fileFacts(st, '기판') : h('p', {class: 'hint'}, '등록된 .kicad_pcb 없음: PCB 단계가 IR에서 컴파일해 등록합니다.'));
  const box = $('board-view');
  box.className = 'figure board-view';
  const svg = await loadSvg(box, previewUrl('board.svg'));
  $('board-layers').disabled = !svg;
  if (!svg) { box.className = ''; return; }
  svg.setAttribute('aria-label', svg.querySelector('title') ? svg.querySelector('title').textContent : '기판');
  fitFigure(svg);
  applyLayers();
}

// a figure scales with the column: the viewBox keeps its proportions once the fixed width / height are dropped
function fitFigure(svg) {
  if (!parseBox(svg.getAttribute('viewBox'))) return;
  svg.removeAttribute('width');
  svg.removeAttribute('height');
  svg.classList.add('fit');
}

function applyLayers() {
  const box = $('board-view');
  for (const input of document.querySelectorAll('#board-layers input[data-layer-class]')) {
    box.classList.toggle('hide-' + input.dataset.layerClass, !input.checked);
  }
}

// ------------------------------------------------------------------ 시뮬레이션

function latestByCheck() {
  const out = {};
  for (const row of state.data.report.validation.latest) out[row.check_id] = row;
  return out;
}

function expectationRows() {
  const sim = parseJson(state.data.report.simulation.ir_json);
  const latest = latestByCheck();
  const expectations = sim && Array.isArray(sim.expectations) ? sim.expectations : [];
  return expectations.map((e) => {
    const result = latest['spice.' + e.id] || null;
    const details = result ? parseJson(result.details_json) : null;
    const unit = (e.nominal && e.nominal.unit) || (details && details.unit) || '';
    const nominal = e.nominal ? fmtNum(e.nominal.value) : '기록 없음';
    let tol;
    if (details && typeof details.tolerance === 'number') tol = fmtNum(details.tolerance) + ' ' + (details.unit || unit);
    else if (e.tol_abs) tol = fmtNum(e.tol_abs.value) + ' ' + (e.tol_abs.unit || unit);
    else if (e.tol_rel) tol = fmtNum(e.tol_rel.value * 100) + ' %';
    else tol = '기록 없음';
    const measured = details && typeof details.measured === 'number' ? fmtNum(details.measured) + ' ' + (details.unit || unit) : '기록 없음';
    return h('tr', {},
      td(idCode(e.id), 'id'), td([idCode(e.vector), ' ', h('span', {class: 'muted small'}, e.reduce + ' · ' + e.analysis_id)], 'vec'),
      td(nominal + ' ' + unit + ' ± ' + tol, 'num'), td(measured, 'num'),
      td(result ? [statusBadge(result.status), h('span', {class: 'fresh'}, ko(result.freshness))] : h('span', {class: 'muted'}, '결과 없음'), 'state'),
      td(result ? result.message : 'SPICE 단계가 이 기대값의 결과를 아직 기록하지 않았습니다', 'msg'));
  });
}

async function renderSimulation() {
  const view = $('sim-view');
  const sim = state.data.simulation;
  const parts = [];
  if (sim.error) parts.push(notice('error', 'spice/results.json을 읽을 수 없음', sim.error));
  else if (sim.missing) parts.push(empty(sim.missing));
  else {
    const cond = sim.conditions || {};
    parts.push(h('dl', {class: 'facts'},
      h('dt', {}, '엔진'), h('dd', {}, (sim.engine || '기록 없음') + ' ' + (sim.engine_version || '')),
      h('dt', {}, '결과 파일'), h('dd', {}, downloadLink(sim.file)),
      h('dt', {}, '이 IR의 결과인가'), h('dd', {}, sim.freshness),
      Object.entries(cond).map(([k, v]) => [h('dt', {}, k), h('dd', {}, typeof v === 'string' ? v : JSON.stringify(v))])));
  }
  const exp = expectationRows();
  parts.push(h('h3', {}, '기대값'));
  parts.push(exp.length
    ? table(['id', '벡터', '공칭 ± 허용차', '측정값', '상태 / 신선도', '메시지'], exp, '기대값은 ir.simulation, 측정값과 상태는 기록된 spice.<id> 결과 그대로')
    : empty('기대값 없음: 회로 템플릿이 IR에 시뮬레이션 기대값을 넣으면 SPICE 단계가 측정해 여기에 기록합니다.'));
  if (sim.analyses && sim.analyses.length) {
    parts.push(h('h3', {}, '해석'));
    parts.push(table(['id', '종류', '명령', '결과', '점 수'], sim.analyses.map((a) => h('tr', {},
      td(code(a.id), 'id'), td(a.kind), td(code(a.command)),
      td(a.unverifiable ? '이 환경에서 실행하지 못함: ' + a.unverifiable : (a.succeeded ? '성공 (기록)' : '실패 (기록)'), 'msg'),
      td(a.n_points === null || a.n_points === undefined ? '' : String(a.n_points), 'num')))));
    for (const a of sim.analyses) {
      if (!a.op) continue;
      parts.push(table(['벡터', '값'], Object.entries(a.op).map(([k, v]) => h('tr', {}, td(code(k)), td(typeof v === 'number' ? fmtNum(v) : v, 'num'))),
        '동작점 (' + a.id + '): ngspice가 기록한 값'));
    }
  }
  parts.push(h('h3', {}, '파형'));
  const figures = h('div', {});
  parts.push(figures);
  fill(view, parts);
  const waves = sim.waveforms || [];
  if (!waves.length) {
    figures.append(empty(sim.missing || sim.error
      ? '파형 없음: SPICE 단계가 spice/results.json을 쓰면 과도·DC·AC 해석마다 파형이 여기에 그려집니다.'
      : '파형 없음: spice/results.json에 그릴 과도·DC·AC 해석이 없습니다 (동작점은 위 표에 있습니다).'));
    return;
  }
  for (const w of waves) {
    const box = h('div', {class: 'figure', role: 'group', 'aria-label': w.title});
    figures.append(box);
    const svg = await loadSvg(box, previewUrl('waveform/' + enc(w.id) + '.svg'));
    if (svg) fitFigure(svg);
  }
}

// ------------------------------------------------------------------ 검증

function renderValidation() {
  const v = state.data.report.validation;
  fill($('val-head'), v.latest.length
    ? h('p', {class: 'release'}, h('strong', {}, '최신 결과 집계'), statusBadge(v.aggregate),
        h('span', {class: 'msg'}, '판정이 아닙니다 (' + ko(v.aggregate_note) + '). 출시 판정은 개요의 RELEASE 줄입니다.'))
    : null);
  const select = $('val-filter');
  const counts = {all: v.latest.length, FAIL: 0, NOT_VERIFIED: 0, PASS: 0, other: 0};
  for (const row of v.latest) counts[row.status in counts && row.status !== 'all' ? row.status : 'other'] += 1;
  for (const option of select.options) option.textContent = option.textContent.replace(/ \(\d+\)$/, '') + ' (' + counts[option.value] + ')';
  drawValidation();
}

function drawValidation() {
  const v = state.data.report.validation;
  const filter = $('val-filter').value;
  const known = ['PASS', 'FAIL', 'NOT_VERIFIED'];
  const rows = v.latest.filter((r) => filter === 'all' || (filter === 'other' ? !known.includes(r.status) : r.status === filter));
  $('val-count').textContent = rows.length + '개 표시 / 전체 ' + v.latest.length + '개';
  const view = $('val-view');
  if (!v.latest.length) { fill(view, empty('검증 결과 없음: \'실행\'하면 IR_BUILD 이후의 검증 단계들이 결과를 기록합니다.')); return; }
  if (!rows.length) { fill(view, empty('이 상태의 결과가 없습니다.')); return; }
  const body = rows.map((r) => h('tr', {},
    td(idCode(r.check_id), 'id'), td([statusBadge(r.status), h('span', {class: 'fresh'}, ko(r.freshness))], 'state'),
    td([r.tool ? idCode(r.tool) : (r.opinion ? '의견 (도구 없음)' : ''), r.tool_version ? h('span', {class: 'muted small'}, ' ' + r.tool_version) : null], 'tool'),
    td([longText(r.message), r.note ? h('p', {class: 'hint'}, ko(r.note)) : null,
      r.details_json ? h('details', {}, h('summary', {}, '세부 (details)'), h('pre', {}, r.details_json)) : null], 'msg'),
    td(r.evidence.length ? h('ul', {class: 'evidence'}, r.evidence.map(evidenceItem)) : h('span', {class: 'muted'}, '없음'), 'ev')));
  const history = v.history.length
    ? h('details', {}, h('summary', {}, '이전 결과 ' + v.history.length + '개 (history)'),
        table(['검사 id', '상태', '시각', '메시지'], v.history.map((e) => h('tr', {}, td(idCode(e.check_id), 'id'), td(statusBadge(e.status), 'nowrap'), td(e.timestamp, 'time'), td(e.message, 'msg')))))
    : null;
  fill(view, table(['검사 id', '상태 / 신선도', '도구 / 버전', '메시지', '증거'], body, '최신 검증 결과 (ir.validation, 그대로 복사)'), history);
}

function evidenceItem(ev) {
  const rel = inWorkdir(ev.path);
  const name = ev.path ? ev.path.replace(/\\/g, '/').split('/').pop() : (ev.url || '');
  return h('li', {}, ev.description ? ev.description + ': ' : '', rel ? downloadLink(rel, name) : (ev.url ? code(ev.url) : name),
    ev.state ? h('span', {class: 'muted'}, ' (' + ko(ev.state) + ')') : null);
}

// ------------------------------------------------------------------ 부품

async function renderParts() {
  const view = $('parts-view');
  const bomBox = h('div', {}, h('p', {class: 'muted'}, '불러오는 중…'));
  const cplBox = h('div', {}, h('p', {class: 'muted'}, '불러오는 중…'));
  const existBox = h('div', {});
  fill(view, h('h3', {}, 'BOM'), bomBox, h('h3', {}, 'CPL (부품 배치)'), cplBox, h('h3', {}, '부품 존재 확인'), existBox);
  drawExistence(existBox);
  await Promise.all([drawTable(bomBox, 'bom.json', true), drawTable(cplBox, 'cpl.json', false)]);
}

async function drawTable(box, file, isBom) {
  const r = await fetchText(previewUrl(file));
  if (!r.ok) { fill(box, r.status === 404 ? empty(r.text.trim()) : notice('error', '읽을 수 없음', r.text.trim())); return; }
  const t = JSON.parse(r.text);
  addLabels(t.labels);
  const facts = fileFacts(t.state, isBom ? 'BOM' : 'CPL');
  if (!t.rows.length) {
    // a header-only file: say why it has no rows (the BOM has a row per component, the CPL a row per placement)
    fill(box, facts, empty(isBom
      ? t.file + '에 행이 없습니다: BOM은 IR의 부품마다 한 줄이라, IR에 부품이 없으면 머리글만 씁니다 (MANUFACTURING_OUTPUTS 단계).'
      : t.file + '에 행이 없습니다: CPL은 기판에 배치된 부품마다 한 줄이라, 배치가 없으면(PLACEMENT를 건너뛰었거나 배치할 부품이 없으면) 머리글만 씁니다 (MANUFACTURING_OUTPUTS 단계).'),
      downloadLink(t.file, t.file + ' 내려받기'), (t.notes || []).map((n) => h('p', {class: 'hint'}, n)));
    return;
  }
  const rows = t.rows.map((row, i) => {
    const nv = isBom && t.not_verified ? new Set(t.not_verified[i] || []) : new Set();
    return h('tr', {}, t.columns.map((c) => td(nv.has(c)
      ? h('span', {title: '검증되지 않음: 이 칸의 값은 확인된 적이 없습니다'}, statusBadge(row[c]))
      : row[c], c === 'Description' || c === 'Footprint' ? 'msg' : 'cell')));
  });
  const caption = isBom
    ? t.file + ' - Value / Description 칸은 CSV의 중화 표시(앞의 \')를 풀어 보여 주고, 나머지 칸은 쓰인 그대로입니다 (넓은 표는 옆으로 스크롤됩니다)'
    : t.file + ' - 쓰인 그대로';
  fill(box, facts, table(t.columns, rows, caption), downloadLink(t.file, t.file + ' 내려받기'),
    (t.notes || []).map((n) => h('p', {class: 'hint'}, n)));
}

function drawExistence(box) {
  const results = state.data.report.components.results;
  const existence = results.filter((r) => r.check_id.startsWith('component.existence.'));
  const others = results.filter((r) => !r.check_id.startsWith('component.existence.'));
  if (!existence.length) {
    fill(box, empty('존재 확인 결과 없음: COMPONENT_SELECTION 단계가 부품마다 심볼·풋프린트·데이터시트·MPN을 확인해 기록합니다.'));
  } else {
    fill(box, table(['검사 id', '상태', '하위 검사 (기록 그대로)'], existence.map((r) => {
      const details = parseJson(r.details_json);
      const checks = details && Array.isArray(details.checks) ? details.checks : [];
      return h('tr', {}, td(idCode(r.check_id), 'id'), td(statusBadge(r.status), 'nowrap'),
        td(checks.length ? h('ul', {class: 'sub-checks'}, checks.map((c) => h('li', {}, code(c.name), statusBadge(c.status), h('span', {}, c.message))))
          : r.message, 'msg'));
    }), state.data.report.components.note));
  }
  if (others.length) {
    box.append(h('h4', {}, '다른 부품 검증'), table(['검사 id', '상태 / 신선도', '메시지'],
      others.map((r) => h('tr', {}, td(idCode(r.check_id), 'id'), td([statusBadge(r.status), h('span', {class: 'fresh'}, ko(r.freshness))], 'state'), td(longText(r.message), 'msg')))));
  }
}

// ------------------------------------------------------------------ 보고서: the HTML in a sandboxed iframe, PDF / MD links

function renderReports() {
  const found = Object.fromEntries((state.data.previews.reports || []).map((r) => [r.stage, r]));
  const keys = REPORTS.map((r) => r.key).concat(['full']);
  let current = state.report;
  if (!keys.includes(current) || (current !== 'full' && !found[current])) current = REPORTS.map((r) => r.key).find((k) => found[k]) || 'full';
  state.report = current;
  const buttons = REPORTS.map((r) => {
    const b = h('button', {type: 'button', 'aria-pressed': current === r.key ? 'true' : 'false', title: found[r.key] ? found[r.key].title : null}, r.label);
    b.disabled = !found[r.key];
    b.addEventListener('click', () => go(state.name, 'reports', r.key));
    return b;
  });
  const full = h('button', {type: 'button', 'aria-pressed': current === 'full' ? 'true' : 'false'}, '전체 (report.html)');
  full.addEventListener('click', () => go(state.name, 'reports', 'full'));
  fill($('report-buttons'), buttons, full);
  const missing = REPORTS.filter((r) => !found[r.key]);
  const links = [];
  let src;
  if (current === 'full') {
    src = previewUrl('report.html');
    links.push(h('span', {class: 'muted'}, 'report.html: ir.json과 pipeline.json으로 지금 그린 영문 보고서'));
    links.push(h('a', {href: src, target: '_blank', rel: 'noopener'}, '새 탭에서 열기'));
    if (state.data.previews.report_html_file) links.push(downloadLink(state.data.previews.report_html_file, '저장된 report.html 내려받기'));
  } else {
    const r = found[current];
    src = r.html ? previewUrl('reports/' + enc(r.html)) : null;
    links.push(h('strong', {}, r.title));
    links.push(r.pdf ? h('a', {href: previewUrl('reports/' + enc(r.pdf)), target: '_blank', rel: 'noopener'}, 'PDF 열기 (새 탭)')
      : h('span', {class: 'muted'}, state.data.browser_found ? 'PDF 없음: 이 보고서가 PDF 없이 쓰였습니다 (PDF 생략 또는 인쇄 실패)' : 'PDF 없음: PDF를 인쇄할 헤드리스 브라우저를 찾지 못했습니다'));
    if (r.md) links.push(h('a', {href: previewUrl('reports/' + enc(r.md)), target: '_blank', rel: 'noopener'}, 'Markdown 보기'));
    if (r.pdf) links.push(downloadLink('reports/' + r.pdf, 'PDF 내려받기'));
  }
  fill($('report-links'), links);
  const notes = missing.length
    ? h('p', {class: 'hint'}, '아직 없음: ' + missing.map((r) => r.label + ' 보고서는 ' + r.after + ' 단계 뒤에').join(', ') + ' 쓰입니다.')
    : null;
  if (!src) { fill($('report-view'), empty('HTML 없음: 이 보고서의 .html이 아직 쓰이지 않았습니다 (\'보고서 다시 쓰기\'가 씁니다).'), notes); return; }
  const frame = h('iframe', {class: 'report-frame', src, sandbox: '', title: current === 'full' ? 'report.html' : found[current].title, referrerpolicy: 'no-referrer'});
  fill($('report-view'), notes, frame);
}

// ------------------------------------------------------------------ 파일

async function renderFiles() {
  const view = $('files-view');
  const zip = h('p', {}, h('a', {class: 'download-zip', href: '/files/' + enc(state.name) + '.zip', download: state.name + '.zip'}, 'ZIP 내려받기'),
    h('span', {class: 'hint'}, ' ir.json, 등록된 산출물, pipeline.json, report.html, reports/, spice/ (sources/와 gui/는 넣지 않습니다)'));
  const artBox = h('div', {}, h('p', {class: 'muted'}, '불러오는 중…'));
  fill(view, zip, h('h3', {}, '산출물 (ir.artifacts)'), artBox, h('h3', {}, '다른 파일'), otherFiles());
  const r = await fetchText(previewUrl('artifacts.json'));
  if (!r.ok) { fill(artBox, notice('error', '읽을 수 없음', r.text.trim())); return; }
  const data = JSON.parse(r.text);
  addLabels(data.labels);
  if (!data.artifacts.length) {
    fill(artBox, empty('산출물 없음: SCHEMATIC, PCB, MANUFACTURING_OUTPUTS 단계가 IR에서 컴파일한 파일을 등록합니다.'));
    return;
  }
  const rows = data.artifacts.map((a) => h('tr', {},
    td(idCode(a.kind), 'id'),
    td([a.download ? downloadLink(a.download) : h('span', {class: 'path'}, a.path),
      a.files.length ? h('ul', {class: 'file-list'}, a.files.map((f) => h('li', {}, f))) : null], 'msg'),
    td(h('span', {title: a.content_hash}, idCode(shortHash(a.content_hash))), 'hash'),
    td(ko(a.disk), 'word'),
    td(a.matches_design_hash ? '예' : '아니오 (' + ko(a.freshness) + ')', 'word'),
    td([a.generator ? idCode(a.generator) : '', a.generator_version ? ' ' + a.generator_version : ''], 'tool')));
  fill(artBox, table(['종류', '파일', '내용 해시', '디스크', '설계 해시와 일치', '생성기'], rows, '설계 해시 ' + data.design_hash));
}

function otherFiles() {
  const d = state.data;
  const items = [h('li', {}, downloadLink('ir.json'), h('span', {class: 'muted'}, ' 설계 IR (원본 설계 데이터)'))];
  if (d.report.meta.pipeline_file_path) items.push(h('li', {}, downloadLink('pipeline.json'), h('span', {class: 'muted'}, ' 실행 기록')));
  if (d.previews.report_html_file) items.push(h('li', {}, downloadLink(d.previews.report_html_file), h('span', {class: 'muted'}, ' 영문 보고서')));
  for (const r of d.previews.reports || []) {
    for (const f of [r.md, r.html, r.pdf]) if (f) items.push(h('li', {}, downloadLink('reports/' + f)));
  }
  if (d.simulation.file && !d.simulation.missing) items.push(h('li', {}, downloadLink(d.simulation.file), h('span', {class: 'muted'}, ' 시뮬레이션 결과')));
  for (const r of d.runs || []) items.push(h('li', {}, downloadLink('gui/runs/' + r.id + '.log'), h('span', {class: 'muted'}, ' 실행 로그 (' + (KIND_LABELS[r.kind] || r.kind) + ')')));
  return h('ul', {class: 'file-list'}, items);
}

// ------------------------------------------------------------------ wiring

document.addEventListener('DOMContentLoaded', () => {
  if (window.matchMedia('(max-width: 760px)').matches) $('new-project-box').open = false;
  $('new-project').addEventListener('submit', createProject);
  $('new-example').addEventListener('change', (ev) => { if (ev.target.value) $('new-request').value = ev.target.value; showExampleInputs(); });
  $('new-request').addEventListener('input', showExampleInputs);
  fillTemplateKeys();
  for (const id of TAB_IDS) $('tab-' + id).addEventListener('click', () => go(state.name, id, state.report));
  $('tabs').addEventListener('keydown', onTabKey);
  $('run-form').addEventListener('submit', submitRun);
  $('run-form').addEventListener('input', updateLlmFields);
  $('run-form').addEventListener('change', updateLlmFields);
  $('add-task-model').addEventListener('click', addTaskRow);
  $('sch-zoom-in').addEventListener('click', () => { if (state.schematic) zoomAt(state.schematic, 1 / 1.25); });
  $('sch-zoom-out').addEventListener('click', () => { if (state.schematic) zoomAt(state.schematic, 1.25); });
  $('sch-fit').addEventListener('click', () => { if (state.schematic) fitView(state.schematic); });
  $('board-layers').addEventListener('change', applyLayers);
  $('val-filter').addEventListener('change', drawValidation);
  $('project-notices').addEventListener('click', (ev) => {
    const a = ev.target.closest('a[data-local]');
    if (!a) return;
    ev.preventDefault();
    go(state.name, 'overview');
    const target = $('questions');
    if (target) { target.scrollIntoView({block: 'start'}); const first = target.querySelector('input, select, textarea'); if (first) first.focus(); }
  });
  window.addEventListener('hashchange', applyHash);
  loadProjects().then(applyHash);
});
"""

__all__ = ["APP_CSS", "APP_HTML", "APP_JS"]
