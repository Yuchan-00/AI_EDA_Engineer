"""Where KiCad, its libraries and its ngspice are found on each operating system (``ai_eda.tools.kicad.install``).

Every layout is simulated in a temporary folder with
``ai_eda.tools.kicad.install.platform`` patched, so the Windows, macOS and
Linux rules are all exercised on any machine. Nothing is executed.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from ai_eda.tools.kicad import install
from ai_eda.tools.kicad import library as kicad_library
from ai_eda.tools.spice import ngspice_shared
from ai_eda.tools.spice.ngspice_shared import find_codemodel_dir, find_ngspice_dll, find_system_ngspice


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in ("KICAD10_SYMBOL_DIR", "KICAD_SYMBOL_DIR", "NGSPICE_DLL", "NGSPICE_CODEMODEL_DIR", "LD_LIBRARY_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    return monkeypatch


def _windows_install(base: Path, version: str) -> Path:
    root = base / version
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "kicad-cli.exe").write_bytes(b"")
    (root / "bin" / "ngspice.dll").write_bytes(b"")
    (root / "lib" / "ngspice").mkdir(parents=True)
    (root / "share" / "kicad" / "symbols").mkdir(parents=True)
    return root


def _mac_bundle(app: Path) -> Path:
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "MacOS" / "kicad-cli").write_bytes(b"")
    for lib in ("symbols", "footprints", "3dmodels"):
        (app / "Contents" / "SharedSupport" / lib).mkdir(parents=True)
    (app / "Contents" / "Frameworks").mkdir(parents=True)
    (app / "Contents" / "Frameworks" / "libngspice.0.dylib").write_bytes(b"")
    cm = app / "Contents" / "PlugIns" / "sim" / "ngspice"
    cm.mkdir(parents=True)
    (cm / "analog.cm").write_bytes(b"")
    return app


# --------------------------------------------------------------------------- Windows


def test_windows_per_user_and_all_users_installs(tmp_path: Path, clean_env: pytest.MonkeyPatch):
    clean_env.setattr(install, "platform", lambda: "win32")
    local = tmp_path / "local"
    program_files = tmp_path / "pf"
    clean_env.setenv("LOCALAPPDATA", str(local))
    clean_env.setenv("ProgramFiles", str(program_files))
    _windows_install(program_files / "KiCad", "10.0")
    assert install.windows_install_bases() == [local / "Programs" / "KiCad", program_files / "KiCad"]
    cli = install.find_kicad_cli()
    assert cli == str(program_files / "KiCad" / "10.0" / "bin" / "kicad-cli.exe")
    assert install.share_dir(cli) == (program_files / "KiCad" / "10.0" / "share" / "kicad").resolve()
    assert kicad_library._default_library_roots()[0] == (program_files / "KiCad" / "10.0" / "share" / "kicad").resolve()
    dll = find_ngspice_dll()
    assert dll == (program_files / "KiCad" / "10.0" / "bin" / "ngspice.dll").resolve()
    assert find_codemodel_dir(dll) == (program_files / "KiCad" / "10.0" / "lib" / "ngspice").resolve()
    # a per-user install is searched first
    _windows_install(local / "Programs" / "KiCad", "10.0")
    assert install.find_kicad_cli() == str(local / "Programs" / "KiCad" / "10.0" / "bin" / "kicad-cli.exe")


@pytest.mark.parametrize("value", ["", "relative/dir"])
def test_windows_never_searches_a_relative_or_empty_localappdata(tmp_path: Path, clean_env: pytest.MonkeyPatch, value: str):
    clean_env.setattr(install, "platform", lambda: "win32")
    clean_env.chdir(tmp_path)
    clean_env.setenv("LOCALAPPDATA", value)
    clean_env.setenv("ProgramFiles", str(tmp_path / "pf"))
    planted = Path(value or ".") / "Programs" / "KiCad"
    _windows_install(tmp_path / planted, "10.0")  # a project folder holding a fake KiCad
    assert all(base.is_absolute() for base in install.windows_install_bases())
    assert install.find_kicad_cli() is None


# --------------------------------------------------------------------------- POSIX: the Windows layout is never used


@pytest.mark.parametrize("system", ["linux", "darwin"])
def test_posix_ignores_the_windows_locations(tmp_path: Path, clean_env: pytest.MonkeyPatch, system: str):
    """A project folder (the GUI's working directory) can not supply kicad-cli or libraries through LOCALAPPDATA."""
    clean_env.setattr(install, "platform", lambda: system)
    clean_env.setattr(install, "MACOS_APP_LOCATIONS", (str(tmp_path / "Applications" / "KiCad" / "KiCad.app"),))
    clean_env.chdir(tmp_path)
    _windows_install(tmp_path / "Programs" / "KiCad", "10.0")
    for value in ("", str(tmp_path)):
        clean_env.setenv("LOCALAPPDATA", value)
        assert install.windows_install_bases() == []
        assert install.find_kicad_cli() is None
        assert kicad_library._default_library_roots() == []


# --------------------------------------------------------------------------- macOS


def test_macos_bundle_cli_libraries_ngspice_and_code_models(tmp_path: Path, clean_env: pytest.MonkeyPatch):
    clean_env.setattr(install, "platform", lambda: "darwin")
    app = _mac_bundle(tmp_path / "Applications" / "KiCad" / "KiCad.app")
    clean_env.setattr(install, "MACOS_APP_LOCATIONS", (str(tmp_path / "missing" / "KiCad.app"), str(app)))
    clean_env.setattr(ngspice_shared, "_MACOS_HOMEBREW_LIBS", ())
    assert install.macos_bundles() == [app]
    cli = install.find_kicad_cli()
    assert cli == str(app / "Contents" / "MacOS" / "kicad-cli")
    assert install.bundle_of(cli) == app.resolve()
    share = (app / "Contents" / "SharedSupport").resolve()
    assert install.share_dir(cli) == share
    assert kicad_library._default_library_roots() == [share]
    dll = find_ngspice_dll()
    assert dll == (app / "Contents" / "Frameworks" / "libngspice.0.dylib").resolve()
    assert find_codemodel_dir(dll) == (app / "Contents" / "PlugIns" / "sim" / "ngspice").resolve()


def test_macos_home_applications_and_kicad_cli_on_path(tmp_path: Path, clean_env: pytest.MonkeyPatch):
    clean_env.setattr(install, "platform", lambda: "darwin")
    home = tmp_path / "home"
    app = _mac_bundle(home / "Applications" / "KiCad.app")
    clean_env.setenv("HOME", str(home))
    clean_env.setattr(install, "MACOS_APP_LOCATIONS", ("~/Applications/KiCad.app",))
    assert install.find_kicad_cli() == str(app / "Contents" / "MacOS" / "kicad-cli")
    # a kicad-cli on PATH (e.g. a symlink into the bundle) wins and still finds the bundle's libraries
    link_dir = tmp_path / "bin"
    link_dir.mkdir()
    link = link_dir / "kicad-cli"
    os.symlink(app / "Contents" / "MacOS" / "kicad-cli", link)
    clean_env.setattr(shutil, "which", lambda name: str(link) if name == "kicad-cli" else None)
    assert install.find_kicad_cli() == str(link)
    assert install.share_dir(str(link)) == (app / "Contents" / "SharedSupport").resolve()


def test_macos_homebrew_ngspice_is_found_by_name(tmp_path: Path, clean_env: pytest.MonkeyPatch):
    clean_env.setattr(install, "platform", lambda: "darwin")
    clean_env.setattr(install, "MACOS_APP_LOCATIONS", ())
    lib = tmp_path / "opt" / "homebrew" / "lib" / "libngspice.dylib"
    lib.parent.mkdir(parents=True)
    lib.write_bytes(b"")
    (lib.parent / "ngspice").mkdir()
    clean_env.setattr(ngspice_shared, "_MACOS_HOMEBREW_LIBS", (str(tmp_path / "usr" / "local" / "lib" / "libngspice.dylib"), str(lib)))
    assert find_system_ngspice() == lib.resolve()
    assert find_ngspice_dll() == lib.resolve()
    assert find_codemodel_dir(lib.resolve()) == (lib.parent / "ngspice").resolve()


def test_macos_bundle_without_libraries_is_not_a_library_root(tmp_path: Path, clean_env: pytest.MonkeyPatch):
    clean_env.setattr(install, "platform", lambda: "darwin")
    app = tmp_path / "KiCad.app"
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "MacOS" / "kicad-cli").write_bytes(b"")
    (app / "Contents" / "SharedSupport").mkdir()
    clean_env.setattr(install, "MACOS_APP_LOCATIONS", (str(app),))
    assert install.find_kicad_cli() == str(app / "Contents" / "MacOS" / "kicad-cli")
    assert install.share_dir(install.find_kicad_cli()) is None
    assert kicad_library._default_library_roots() == []


# --------------------------------------------------------------------------- Linux


def test_linux_distribution_layout(tmp_path: Path, clean_env: pytest.MonkeyPatch):
    clean_env.setattr(install, "platform", lambda: "linux")
    prefix = tmp_path / "usr"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "kicad-cli").write_bytes(b"")
    (prefix / "share" / "kicad" / "footprints").mkdir(parents=True)
    clean_env.setattr(shutil, "which", lambda name: str(prefix / "bin" / "kicad-cli") if name == "kicad-cli" else None)
    assert install.find_kicad_cli() == str(prefix / "bin" / "kicad-cli")
    assert install.share_dir(install.find_kicad_cli()) == (prefix / "share" / "kicad").resolve()
    assert kicad_library._default_library_roots() == [(prefix / "share" / "kicad").resolve()]
    assert install.bundled_ngspice(install.find_kicad_cli()) is None
