"""The Korean wording of the report's fact labels, for the Korean page.

Invariant: a label is a fact about hashes, files or the run log that
:mod:`ai_eda.report.data` states in English (``fresh``, ``changed on disk``,
``recorded for an earlier IR version (IR ...)`` ...); this module only says
the same fact in Korean. It is keyed by the constants it imports from
:mod:`ai_eda.report.data` - never by a second spelling of them - so a label
that module renames is simply shown as written, never mistranslated.
:func:`korean_label` translates a text only when it *is* one of those
labels, or one of their composed forms (``<label> (IR <hash>)``, the hash
kept as written); every other text (messages, record errors, paths) is
``None`` and stays as recorded. Nothing here computes a status or reads a
file: the project JSON carries ``{label: Korean}`` for the labels it holds
(:func:`label_map`) and the page shows the Korean wording in place of the
English one - the same fact, copied.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ai_eda.report.data import (
    AGGREGATE_NOTE,
    CHANGED_ON_DISK,
    EVIDENCE_MISSING,
    EVIDENCE_NO_HASH,
    EVIDENCE_NO_PATH,
    EVIDENCE_OK,
    FRESH,
    MISSING_ON_DISK,
    MODEL_OUTPUT,
    NO_RECORDED_RUN,
    NOT_READ_OUTSIDE,
    ON_DISK,
    OPINION,
    PIPELINE_DESCRIBES_IR,
    PIPELINE_OTHER_IR,
    PIPELINE_STALE_IR,
    RUN_CURRENT_IR,
    RUN_EARLIER_IR,
    STALE,
    UNSTAMPED_CARRIED,
    UNSTAMPED_PRODUCED,
    UNSTAMPED_UNKNOWN,
)

#: every label of ai_eda.report.data, in Korean
KOREAN_LABELS: dict[str, str] = {
    FRESH: "최신 (현재 설계 해시)",
    STALE: "낡음 (이전 설계 해시)",
    UNSTAMPED_PRODUCED: "해시 없음 (기록된 실행이 만듦)",
    UNSTAMPED_CARRIED: "해시 없음 (이전 실행에서 넘어옴 - 지금은 아무것도 보증하지 않음)",
    UNSTAMPED_UNKNOWN: "해시 없음 (어느 실행인지 모름)",
    ON_DISK: "디스크에 있음",
    CHANGED_ON_DISK: "디스크에서 바뀜",
    MISSING_ON_DISK: "디스크에 없음",
    EVIDENCE_OK: "디스크에 있음 (해시 일치)",
    EVIDENCE_NO_HASH: "있음 (기록된 해시 없음)",
    EVIDENCE_NO_PATH: "경로 없음",
    EVIDENCE_MISSING: "없음",
    NOT_READ_OUTSIDE: "이 프로젝트 폴더 밖 (읽지 않음)",
    PIPELINE_DESCRIBES_IR: "pipeline.json이 이 ir.json을 기록했습니다",
    PIPELINE_STALE_IR: "pipeline.json을 쓴 뒤 ir.json이 바뀌었습니다",
    PIPELINE_OTHER_IR: "pipeline.json이 다른 ir.json의 실행을 기록했습니다",
    RUN_CURRENT_IR: "현재 IR에 대한 실행 기록",
    RUN_EARLIER_IR: "이전 IR 버전에 대한 실행 기록",
    NO_RECORDED_RUN: "기록된 실행 없음: 먼저 `ai-eda run`을 실행하십시오",
    OPINION: "의견 (도구 없음)",
    MODEL_OUTPUT: "모델 출력",
    AGGREGATE_NOTE: "최신 결과들의 집계 - 출시 판정이 아닙니다; RELEASE를 보십시오",
}
#: the composed forms ``f"{label} (IR {hash[:16]})"`` (data.py ``_Freshness.label`` / ``run_hash_label``): the Korean of the label, the rest as written
KOREAN_PREFIXES: tuple[tuple[str, str], ...] = (
    (f"{STALE} (IR ", "낡음: 이전 설계 해시 (IR "),
    (f"{RUN_EARLIER_IR} (IR ", "이전 IR 버전에 대한 실행 기록 (IR "),
)
#: the composed forms ``f"{label}: <path>"`` (data.py ``pipeline_note`` of a record naming another ir.json): the Korean of the label, the path as written
KOREAN_PATH_PREFIXES: tuple[tuple[str, str], ...] = (
    (f"{PIPELINE_OTHER_IR}: ", "pipeline.json이 다른 ir.json의 실행을 기록했습니다: "),
)


def korean_label(text: object) -> str | None:
    """The Korean of ``text`` when it is a label of :mod:`ai_eda.report.data` (or its ``(IR <hash>)`` form), else ``None``."""
    if not isinstance(text, str):
        return None
    if text in KOREAN_LABELS:
        return KOREAN_LABELS[text]
    for prefix, korean in KOREAN_PREFIXES:
        if text.startswith(prefix) and text.endswith(")"):
            return korean + text[len(prefix):]
    for prefix, korean in KOREAN_PATH_PREFIXES:
        if text.startswith(prefix) and len(text) > len(prefix):
            return korean + text[len(prefix):]
    return None


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def label_map(*values: Any) -> dict[str, str]:
    """``{label: Korean}`` for every label that occurs anywhere in ``values`` (JSON-shaped data), sorted by label."""
    found = {s: k for s in _strings(values) if (k := korean_label(s)) is not None}
    return dict(sorted(found.items()))


__all__ = ["KOREAN_LABELS", "KOREAN_PATH_PREFIXES", "KOREAN_PREFIXES", "korean_label", "label_map"]
