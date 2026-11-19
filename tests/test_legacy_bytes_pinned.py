"""The legacy 0.8 schematic keeps its bytes: every corpus design compiles to the sha256 recorded in
``tests/data/legacy_08_sha256.json`` before the textbook 1.0 layout existed, with an explicit ``layout="legacy"`` (the
default is textbook 1.0).

The recorded files embed the installed KiCad library's symbol text (``lib_symbols``), so the hashes are keyed by the
kicad-cli version they were recorded with and carry a fingerprint of that text: on another kicad-cli version or another
library the test is skipped with that reason, not failed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ai_eda.compilers import CompileContext, SchematicCompiler
from ai_eda.tools.kicad import KicadCli, KicadLibrary, sexpr
from tests.textbook_corpus import CORPUS, corpus_ir

LIB = KicadLibrary()
CLI = KicadCli()
RECORDED = json.loads((Path(__file__).parent / "data" / "legacy_08_sha256.json").read_text(encoding="utf-8"))


def _recorded() -> dict[str, dict[str, str]]:
    if LIB.symbol_file("Device") is None or not CLI.available():
        pytest.skip("KiCad CLI/libraries not installed")
    version = CLI.version()
    if version not in RECORDED["kicad_cli"]:
        pytest.skip(f"legacy hashes were recorded with kicad-cli {', '.join(RECORDED['kicad_cli'])}, not {version}")
    return RECORDED["kicad_cli"][version]


def _digests(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    lib_symbols = sexpr.dumps(sexpr.find(sexpr.parse(raw.decode("utf-8")), "lib_symbols"))
    return hashlib.sha256(raw).hexdigest(), hashlib.sha256(lib_symbols.encode()).hexdigest()


def test_every_corpus_design_is_recorded():
    assert set(next(iter(RECORDED["kicad_cli"].values()))) == set(CORPUS)


@pytest.mark.parametrize("name", CORPUS)
def test_legacy_bytes_equal_the_recorded_hash(tmp_path, name):
    recorded = _recorded()[name]
    ir = corpus_ir(name, tmp_path / "ir", LIB)
    # changed with the default switch: the default is textbook 1.0 now, so only explicit legacy must equal the 0.8 hash
    artifact = SchematicCompiler("legacy").compile(ir, CompileContext(workdir=tmp_path / "legacy", tools={"kicad_library": LIB}))
    assert artifact.generator_version == "0.8"
    digest, library = _digests(Path(artifact.path))
    if library != recorded["lib_symbols_sha256"]:
        pytest.skip(f"{name}: the installed KiCad library symbols differ from the ones the hash was recorded with")
    assert digest == recorded["sha256"], f"{name}: legacy schematic bytes changed"
