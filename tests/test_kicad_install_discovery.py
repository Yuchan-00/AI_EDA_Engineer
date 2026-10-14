"""Where KiCad, its libraries and its ngspice are found on each operating system (``ai_eda.tools.kicad.install``).

Every layout is simulated in a temporary folder with
``ai_eda.tools.kicad.install.platform`` patched, so the Windows, macOS and
Linux rules are all exercised on any machine. Nothing is executed.
"""

from __future__ import annotations

import ctypes.util
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

from ai_eda.tools.kicad import install
from ai_eda.tools.kicad import library as kicad_library
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.model3d.models import find_3dmodel_dir
from ai_eda.tools.spice import ngspice_shared
from ai_eda.tools.spice.ngspice_shared import find_codemodel_dir, find_ngspice_dll, find_system_ngspice
from ai_eda.tools.spice.runner import NgspiceRunner

POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX PATH semantics (an empty entry, the execute bit)")
#: the real PATH lookup (``clean_env`` replaces it with "nothing on PATH")
REAL_WHICH = install.which_absolute


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> pytest.MonkeyPatch:
    """No variable, no PATH hit and no all-users program folder of the machine running the tests reaches discovery."""
    for name in ("KICAD10_SYMBOL_DIR", "KICAD_SYMBOL_DIR", "NGSPICE_DLL", "NGSPICE_CODEMODEL_DIR", "LD_LIBRARY_PATH", "LOCALAPPDATA", "ProgramW6432", "ProgramFiles"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(install, "which_absolute", lambda name: None)
    monkeypatch.setattr(install, "DEFAULT_PROGRAM_FILES", tmp_path / "default-program-files")
    return monkeypatch


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as e:  # Windows without the symlink privilege
        pytest.skip(f"cannot create a symlink here: {e}")


def _executable(path: Path, text: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


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
    assert install.windows_install_bases() == [
        local / "Programs" / "KiCad",
        program_files / "KiCad",
        tmp_path / "default-program-files" / "KiCad",
    ]
    cli = install.find_kicad_cli()
    assert cli == str(program_files / "KiCad" / "10.0" / "bin" / "kicad-cli.exe")
    assert install.share_dir(cli) == (program_files / "KiCad" / "10.0" / "share" / "kicad").resolve()
    assert kicad_library._default_library_roots()[0] == (program_files / "KiCad" / "10.0" / "share" / "kicad").resolve()
    dll = find_ngspice_dll()
    assert dll == (program_files / "KiCad" / "10.0" / "bin" / "ngspice.dll").resolve()
    assert find_codemodel_dir(dll) == (program_files / "KiCad" / "10.0" / "lib" / "ngspice").resolve()
    # a per-user install of the same version comes first
    _windows_install(local / "Programs" / "KiCad", "10.0")
    assert install.find_kicad_cli() == str(local / "Programs" / "KiCad" / "10.0" / "bin" / "kicad-cli.exe")


def test_windows_newest_version_wins_across_per_user_and_all_users(tmp_path: Path, clean_env: pytest.MonkeyPatch):
    """An older per-user KiCad 9 never hides an all-users KiCad 10 (winget's location): one version order over all bases."""
    clean_env.setattr(install, "platform", lambda: "win32")
    local = tmp_path / "local"
    program_files = tmp_path / "pf"
    clean_env.setenv("LOCALAPPDATA", str(local))
    clean_env.setenv("ProgramFiles", str(program_files))
    old = _windows_install(local / "Programs" / "KiCad", "9.0")
    new = _windows_install(program_files / "KiCad", "10.0")
    assert install.windows_installs() == [new, old]
    assert install.find_kicad_cli() == str(new / "bin" / "kicad-cli.exe")
    assert install.library_shares() == [new / "share" / "kicad", old / "share" / "kicad"]
    assert kicad_library._default_library_roots() == [(new / "share" / "kicad").resolve(), old / "share" / "kicad"]
    assert find_ngspice_dll() == (new / "bin" / "ngspice.dll").resolve()
    # the same version in both: the per-user one first, in both lists
    tie = _windows_install(local / "Programs" / "KiCad", "10.0")
    assert install.windows_installs() == [tie, new, old]
    assert install.find_kicad_cli() == str(tie / "bin" / "kicad-cli.exe")
    assert install.library_shares() == [tie / "share" / "kicad", new / "share" / "kicad", old / "share" / "kicad"]


def test_windows_program_files_from_programw6432_then_programfiles_then_the_default(tmp_path: Path, clean_env: pytest.MonkeyPatch):
    """A 32-bit Python sees ProgramFiles = 'Program Files (x86)'; the 64-bit-only KiCad 10 is under %ProgramW6432%."""
    clean_env.setattr(install, "platform", lambda: "win32")
    pf64 = tmp_path / "Program Files"
    pf86 = tmp_path / "Program Files (x86)"
    clean_env.setenv("ProgramW6432", str(pf64))
    clean_env.setenv("ProgramFiles", str(pf86))
    kicad = _windows_install(pf64 / "KiCad", "10.0")
    assert install.windows_install_bases() == [pf64 / "KiCad", pf86 / "KiCad", tmp_path / "default-program-files" / "KiCad"]
    assert install.find_kicad_cli() == str(kicad / "bin" / "kicad-cli.exe")
    # a 64-bit Python: both variables name the same folder, the default too - searched once
    clean_env.setenv("ProgramFiles", str(pf64))
    clean_env.setattr(install, "DEFAULT_PROGRAM_FILES", pf64)
    assert install.windows_install_bases() == [pf64 / "KiCad"]
    # relative or empty variables are never searched; the default is used only when absolute
    clean_env.chdir(tmp_path)
    for value in ("", "relative"):
        clean_env.setenv("ProgramW6432", value)
        clean_env.setenv("ProgramFiles", value)
        clean_env.setattr(install, "DEFAULT_PROGRAM_FILES", Path("relative-default"))
        assert install.windows_install_bases() == []


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
    _symlink_or_skip(link, app / "Contents" / "MacOS" / "kicad-cli")
    clean_env.setattr(install, "which_absolute", lambda name: str(link) if name == "kicad-cli" else None)
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
    clean_env.setattr(install, "which_absolute", lambda name: str(prefix / "bin" / "kicad-cli") if name == "kicad-cli" else None)
    assert install.find_kicad_cli() == str(prefix / "bin" / "kicad-cli")
    assert install.share_dir(install.find_kicad_cli()) == (prefix / "share" / "kicad").resolve()
    assert kicad_library._default_library_roots() == [(prefix / "share" / "kicad").resolve()]
    assert install.bundled_ngspice(install.find_kicad_cli()) is None


# --------------------------------------------------------------------------- PATH: only its absolute entries


@POSIX_ONLY
@pytest.mark.parametrize(
    "path_of",
    [lambda abs_dir: os.pathsep + abs_dir, lambda abs_dir: abs_dir + os.pathsep, lambda abs_dir: "." + os.pathsep + "bin" + os.pathsep + abs_dir, lambda abs_dir: ""],
    ids=["leading-empty-entry", "trailing-empty-entry", "relative-entries", "empty-PATH"],
)
def test_a_kicad_cli_in_the_working_directory_is_never_found(tmp_path: Path, clean_env: pytest.MonkeyPatch, path_of):
    """An empty PATH entry (shutil.which: the current folder) or a relative one never lets a project folder supply kicad-cli."""
    clean_env.setattr(install, "which_absolute", REAL_WHICH)
    clean_env.setattr(install, "platform", lambda: "linux")
    project = tmp_path / "project"
    planted = _executable(project / "kicad-cli", "#!/bin/sh\necho PLANTED 10.0.99\n")
    _executable(project / "bin" / "kicad-cli")
    _executable(project / "ngspice")
    empty = tmp_path / "empty"
    empty.mkdir()
    clean_env.chdir(project)
    clean_env.setenv("PATH", path_of(str(empty)))
    assert install.which_absolute("kicad-cli") is None
    assert install.find_kicad_cli() is None
    assert install.share_dir("kicad-cli") is None and install.bundled_ngspice("kicad-cli") is None
    assert install.share_dir("bin/kicad-cli") is None  # a relative path is never resolved against the working directory
    assert install.share_dir("./kicad-cli") is None
    assert NgspiceRunner().binary is None
    # the same file reached through an absolute entry is found, spelled as entry + name
    clean_env.setenv("PATH", os.pathsep.join(["", ".", str(project)]))
    assert install.which_absolute("kicad-cli") == str(planted)
    assert install.find_kicad_cli() == str(planted)
    assert NgspiceRunner().binary == str(project / "ngspice")


@POSIX_ONLY
def test_which_absolute_posix_needs_an_executable_regular_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(install, "platform", lambda: "linux")
    first, second = tmp_path / "a", tmp_path / "b"
    (first / "tool").mkdir(parents=True)  # a directory is no executable
    (second).mkdir()
    (second / "tool").write_text("#!/bin/sh\n", encoding="utf-8")  # not executable
    monkeypatch.setenv("PATH", os.pathsep.join([str(first), str(second)]))
    assert install.which_absolute("tool") is None
    third = _executable(tmp_path / "c" / "tool")
    monkeypatch.setenv("PATH", os.pathsep.join([str(first), str(second), str(tmp_path / "c")]))
    assert install.which_absolute("tool") == str(third)
    with pytest.raises(ValueError, match="bare file name"):
        install.which_absolute("c/tool")


def test_which_absolute_windows_tries_pathext_and_never_the_current_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Windows rules (simulated): PATHEXT order, a name already ending in one as is, no execute bit, no implicit current folder."""
    monkeypatch.setattr(install, "platform", lambda: "win32")
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    cwd = tmp_path / "project"
    cwd.mkdir()
    (cwd / "kicad-cli.exe").write_bytes(b"MZ")  # what shutil.which on Windows would try first
    (cwd / "claude.exe").write_bytes(b"MZ")
    monkeypatch.chdir(cwd)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    monkeypatch.setenv("PATH", os.pathsep.join(["", ".", str(bindir)]))
    assert install.which_absolute("kicad-cli") is None and install.which_absolute("claude.exe") is None
    bat = bindir / "kicad-cli.bat"
    bat.write_bytes(b"@echo off\r\n")  # Windows needs no execute bit
    assert install.which_absolute("kicad-cli") == str(bat)
    exe = bindir / "kicad-cli.exe"
    exe.write_bytes(b"MZ")
    assert install.which_absolute("kicad-cli") == str(exe)  # .EXE before .BAT: PATHEXT order
    (bindir / "claude.exe").write_bytes(b"MZ")
    (bindir / "claude.exe.com").write_bytes(b"MZ")
    assert install.which_absolute("claude.exe") == str(bindir / "claude.exe")  # already a PATHEXT name: as is
    (bindir / "kicad-cli").write_bytes(b"")  # an extensionless file is not what Windows runs
    exe.unlink()
    bat.unlink()
    assert install.which_absolute("kicad-cli") is None



def test_absolute_path_only_drops_empty_and_relative_entries_for_the_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    keep = [str(tmp_path / "a"), str(tmp_path / "b")]
    original = os.pathsep.join(["", keep[0], ".", "bin", keep[1], ""])
    monkeypatch.setenv("PATH", original)
    with install.absolute_path_only():
        assert os.environ["PATH"] == os.pathsep.join(keep)
    assert os.environ["PATH"] == original
    with pytest.raises(RuntimeError), install.absolute_path_only():
        raise RuntimeError("restored on the way out too")
    assert os.environ["PATH"] == original
    monkeypatch.delenv("PATH")
    with install.absolute_path_only():  # never empty: an empty PATH is the working directory to os.get_exec_path
        assert os.environ["PATH"] == install.default_path() and os.environ["PATH"]
        assert all(Path(e).is_absolute() for e in os.environ["PATH"].split(os.pathsep))
    assert "PATH" not in os.environ
    monkeypatch.setenv("PATH", os.pathsep.join(["", ".", "bin"]))
    with install.absolute_path_only():
        assert os.environ["PATH"] == install.default_path()
        assert "" not in os.get_exec_path() and "." not in os.get_exec_path()
    assert os.environ["PATH"] == os.pathsep.join(["", ".", "bin"])


def test_the_default_path_is_the_systems_absolute_one():
    default = install.default_path()
    assert default and all(e and Path(e).is_absolute() for e in default.split(os.pathsep))
    if hasattr(os, "confstr"):
        try:
            cs_path = os.confstr("CS_PATH")
        except (ValueError, OSError):
            cs_path = None
        if cs_path and install.absolute_entries(cs_path):
            assert default == os.pathsep.join(install.absolute_entries(cs_path))


def test_absolute_path_only_cuts_the_library_search_paths_it_is_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    keep, link = str(tmp_path / "lib"), str(tmp_path / "link")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.setenv("LD_LIBRARY_PATH", os.pathsep.join(["", "lib", keep, "."]))
    monkeypatch.setenv("LIBRARY_PATH", os.pathsep.join(["", link, "lib"]))
    monkeypatch.setenv("COMPILER_PATH", ".")
    monkeypatch.delenv("DYLD_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("DYLD_FALLBACK_LIBRARY_PATH", "")
    before = {name: os.environ.get(name) for name in install.FIND_LIBRARY_PATH_VARS}
    with install.absolute_path_only(*install.FIND_LIBRARY_PATH_VARS):
        assert os.environ["LD_LIBRARY_PATH"] == keep
        assert "COMPILER_PATH" not in os.environ and "DYLD_LIBRARY_PATH" not in os.environ and "DYLD_FALLBACK_LIBRARY_PATH" not in os.environ
        assert os.environ["LIBRARY_PATH"] == link
        assert os.environ["PATH"] == str(tmp_path / "bin")
    assert {name: os.environ.get(name) for name in install.FIND_LIBRARY_PATH_VARS} == before
    with install.absolute_path_only():  # PATH alone unless named
        assert os.environ["LD_LIBRARY_PATH"] == before["LD_LIBRARY_PATH"]
    assert set(install.LOADER_PATH_VARS) < set(install.LIBRARY_PATH_VARS) < set(install.FIND_LIBRARY_PATH_VARS)


@POSIX_ONLY
def test_the_system_ngspice_lookup_runs_no_tool_from_the_working_directory(tmp_path: Path, clean_env: pytest.MonkeyPatch):
    """``ctypes.util.find_library`` may run gcc / cc / ld / objdump from PATH: an empty PATH entry would mean the project folder."""
    clean_env.setattr(install, "platform", lambda: "linux")
    project = tmp_path / "project"
    marker = tmp_path / "ran-from-the-working-directory"
    for tool in ("gcc", "cc", "ld", "objdump", "ldconfig", "x86_64-linux-gnu-gcc"):
        _executable(project / tool, f"#!/bin/sh\ntouch '{marker}'\nexit 1\n")
    clean_env.chdir(project)
    clean_env.setenv("PATH", os.pathsep + os.environ.get("PATH", ""))
    with install.absolute_path_only():
        ctypes.util.find_library("ai-eda-no-such-library")
    find_system_ngspice()
    assert not marker.exists()


@POSIX_ONLY
@pytest.mark.parametrize("path", [None, ".", os.pathsep, "bin"])
def test_an_unset_or_relative_path_runs_no_tool_from_the_working_directory(tmp_path: Path, clean_env: pytest.MonkeyPatch, path: str | None):
    """With no absolute PATH entry the guard falls back to the system default, never to an empty PATH: subprocess reads an
    empty PATH as the working directory, where ``ctypes.util`` runs ``ld`` once the loader cache does not know the name."""
    clean_env.setattr(install, "platform", lambda: "linux")
    project = tmp_path / "project"
    marker = tmp_path / "ran-from-the-working-directory"
    for tool in ("gcc", "cc", "ld", "objdump", "ldconfig", "collect2", "x86_64-linux-gnu-gcc"):
        planted = f"#!/bin/sh\necho \"$0 $*\" >> '{marker}'\nexit 1\n"  # shell builtins only: with PATH empty there is no touch
        _executable(project / tool, planted)
        _executable(project / "bin" / tool, planted)
    clean_env.chdir(project)
    if path is None:
        clean_env.delenv("PATH", raising=False)
    else:
        clean_env.setenv("PATH", path)
    for name in ("LIBRARY_PATH", "COMPILER_PATH"):
        clean_env.setenv(name, ".")
    with install.absolute_path_only(*install.FIND_LIBRARY_PATH_VARS):
        assert os.environ["PATH"] and "" not in os.get_exec_path()
        ctypes.util.find_library("ai-eda-no-such-library")
    clean_env.setattr(ctypes.util, "_findSoname_ldconfig", lambda name: None, raising=False)  # as on a machine without ngspice
    find_system_ngspice()
    assert not marker.exists()
    assert os.environ.get("PATH") == path  # restored, unset included


@pytest.mark.parametrize("soname", ["../evil/libngspice.so", "evil/libngspice.so", "/proc/self/cwd/libngspice.so", "..", "."])
def test_a_soname_that_is_not_a_bare_file_name_is_no_library(tmp_path: Path, clean_env: pytest.MonkeyPatch, soname: str):
    """``find_library`` returns whatever SONAME the file it found carries: one with a folder in it would lead the
    library-directory search into the working directory (``../../../proc/self/cwd/lib/libngspice.so``)."""
    clean_env.setattr(install, "platform", lambda: "linux")
    clean_env.setattr(ctypes.util, "find_library", lambda name: soname if name == "ngspice" else None)
    lib = tmp_path / "lib"
    lib.mkdir()
    (tmp_path / "evil").mkdir()
    (tmp_path / "evil" / "libngspice.so").write_bytes(b"")
    (lib / "evil").mkdir()
    (lib / "evil" / "libngspice.so").write_bytes(b"")
    clean_env.chdir(tmp_path)
    (tmp_path / "libngspice.so").write_bytes(b"")
    clean_env.setenv("LD_LIBRARY_PATH", str(lib))
    assert find_system_ngspice() is None
    clean_env.setattr(ctypes.util, "find_library", lambda name: "libngspice.so" if name == "ngspice" else None)
    (lib / "libngspice.so").write_bytes(b"")
    assert find_system_ngspice() == (lib / "libngspice.so").resolve()  # a bare name in an absolute entry: found


def test_on_macos_only_an_absolute_find_library_result_is_a_library(tmp_path: Path, clean_env: pytest.MonkeyPatch):
    clean_env.setattr(install, "platform", lambda: "darwin")
    clean_env.setattr(ngspice_shared, "_MACOS_HOMEBREW_LIBS", ())
    clean_env.chdir(tmp_path)
    (tmp_path / "libngspice.dylib").write_bytes(b"")
    clean_env.setenv("LD_LIBRARY_PATH", str(tmp_path))
    clean_env.setattr(ctypes.util, "find_library", lambda name: "libngspice.dylib" if name == "ngspice" else None)
    assert find_system_ngspice() is None  # a relative result is the working directory's file
    lib = tmp_path / "usr" / "lib" / "libngspice.0.dylib"
    lib.parent.mkdir(parents=True)
    lib.write_bytes(b"")
    clean_env.setattr(ctypes.util, "find_library", lambda name: str(lib) if name == "ngspice" else None)
    assert find_system_ngspice() == lib.resolve()
    clean_env.setattr(ctypes.util, "find_library", lambda name: str(tmp_path / "missing.dylib") if name == "ngspice" else None)
    assert find_system_ngspice() is None


@pytest.mark.skipif(not sys.platform.startswith("linux") or shutil.which("gcc") is None, reason="builds a shared object with gcc (Linux)")
@pytest.mark.parametrize("variable", ["LD_LIBRARY_PATH", "LIBRARY_PATH"])
def test_a_planted_library_with_a_path_soname_is_never_found(tmp_path: Path, clean_env: pytest.MonkeyPatch, variable: str):
    """The verifier's reproduction: a project-folder ``lib/libngspice.so`` whose SONAME climbs to the working directory,
    reached by ``ld -L lib`` (``LD_LIBRARY_PATH``) or ``gcc`` (``LIBRARY_PATH``) once the loader cache does not know
    ``libngspice``. Neither route is offered to ``find_library`` and its answer must be a bare file name."""
    import subprocess

    clean_env.setattr(install, "platform", lambda: "linux")
    project = tmp_path / "project"
    (project / "lib").mkdir(parents=True)
    source = tmp_path / "planted.c"
    source.write_text("int ngSpice_Init(void) { return 0; }\n", encoding="utf-8")
    soname = "../../../proc/self/cwd/lib/libngspice.so"
    built = subprocess.run(
        ["gcc", "-shared", "-fPIC", f"-Wl,-soname,{soname}", "-o", str(project / "lib" / "libngspice.so"), str(source)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    if built.returncode != 0:
        pytest.skip(f"gcc could not build the planted library: {built.stderr.strip()[:200]}")
    clean_env.chdir(project)
    clean_env.setenv(variable, "lib")
    clean_env.setattr(ctypes.util, "_findSoname_ldconfig", lambda name: None, raising=False)  # no system libngspice
    found = find_system_ngspice()
    assert found is None or project.resolve() not in found.resolve().parents


# --------------------------------------------------------------------------- relative environment overrides are unset


@pytest.mark.parametrize("value", ["", "kicad-symbols/symbols"])
def test_a_relative_or_empty_symbol_dir_is_no_library_root(tmp_path: Path, clean_env: pytest.MonkeyPatch, value: str):
    clean_env.setattr(install, "platform", lambda: "linux")
    clean_env.chdir(tmp_path)
    (tmp_path / "kicad-symbols" / "symbols").mkdir(parents=True)
    (tmp_path / "kicad-symbols" / "symbols" / "Device.kicad_sym").write_text("(kicad_symbol_lib)", encoding="utf-8")
    for name in ("KICAD10_SYMBOL_DIR", "KICAD_SYMBOL_DIR"):
        clean_env.setenv(name, value)
    assert kicad_library._default_library_roots() == []
    absolute = tmp_path / "abs" / "symbols"
    absolute.mkdir(parents=True)
    clean_env.setenv("KICAD_SYMBOL_DIR", str(absolute))
    assert kicad_library._default_library_roots() == [absolute.parent]  # the fallback variable, when absolute
    clean_env.setenv("KICAD10_SYMBOL_DIR", str(tmp_path / "kicad-symbols" / "symbols"))
    assert kicad_library._default_library_roots() == [tmp_path / "kicad-symbols"]


@pytest.mark.parametrize("value", ["", "libngspice-planted.so"])
def test_a_relative_or_empty_ngspice_override_is_unset(tmp_path: Path, clean_env: pytest.MonkeyPatch, value: str):
    clean_env.setattr(install, "platform", lambda: "linux")
    clean_env.setattr(ctypes.util, "find_library", lambda name: None)  # no system library on any machine
    clean_env.chdir(tmp_path)
    (tmp_path / "libngspice-planted.so").write_bytes(b"")
    (tmp_path / "codemodels").mkdir()
    clean_env.setenv("NGSPICE_DLL", value)
    clean_env.setenv("NGSPICE_CODEMODEL_DIR", value and "codemodels")
    assert find_ngspice_dll() is None
    assert find_codemodel_dir(None) is None
    clean_env.setenv("NGSPICE_DLL", str(tmp_path / "libngspice-planted.so"))
    clean_env.setenv("NGSPICE_CODEMODEL_DIR", str(tmp_path / "codemodels"))
    assert find_ngspice_dll() == (tmp_path / "libngspice-planted.so").resolve()
    assert find_codemodel_dir(None) == tmp_path / "codemodels"


def test_a_relative_ld_library_path_entry_is_not_searched(tmp_path: Path, clean_env: pytest.MonkeyPatch):
    clean_env.setattr(install, "platform", lambda: "linux")
    clean_env.setattr(ctypes.util, "find_library", lambda name: "libngspice-planted.so.0" if name == "ngspice" else None)
    clean_env.chdir(tmp_path)
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "libngspice-planted.so.0").write_bytes(b"")
    (tmp_path / "libngspice-planted.so.0").write_bytes(b"")
    clean_env.setenv("LD_LIBRARY_PATH", os.pathsep.join(["", ".", "lib"]))
    assert find_system_ngspice() is None
    clean_env.setenv("LD_LIBRARY_PATH", str(tmp_path / "lib"))
    assert find_system_ngspice() == (tmp_path / "lib" / "libngspice-planted.so.0").resolve()


@pytest.mark.parametrize("value", ["", "3d"])
def test_a_relative_or_empty_3dmodel_dir_is_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "3d").mkdir()
    for name in ("KICAD10_3DMODEL_DIR", "KICAD_3DMODEL_DIR"):
        monkeypatch.setenv(name, value)
    nothing = KicadLibrary(roots=[tmp_path / "nothing"])
    assert find_3dmodel_dir(nothing) is None
    monkeypatch.setenv("KICAD_3DMODEL_DIR", str(tmp_path / "3d"))
    assert find_3dmodel_dir(nothing) == tmp_path / "3d"
