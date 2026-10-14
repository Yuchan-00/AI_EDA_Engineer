"""Where KiCad is installed, per operating system.

Invariant: only the absolute locations an operating system's own KiCad
installer uses are searched, each only on that operating system, and the
``kicad-cli`` on ``PATH`` always wins. A path is never built from an empty or
relative environment variable: a location relative to the working directory
would let a project folder (GUI runs start in the project folder) supply a
``kicad-cli`` binary or library files.

* Windows - ``%LOCALAPPDATA%\\Programs\\KiCad\\<ver>`` (the per-user
  installer) and ``%ProgramFiles%\\KiCad\\<ver>`` (all users; also what
  ``winget install KiCad.KiCad`` uses), highest version first:
  ``bin\\kicad-cli.exe``, the libraries in ``share\\kicad``, ngspice as
  ``bin\\ngspice.dll`` with its code models in ``lib\\ngspice``.
* macOS - ``KiCad.app`` in ``/Applications/KiCad`` (where the DMG and the
  Homebrew cask put it), ``/Applications``, ``~/Applications/KiCad`` or
  ``~/Applications``: ``Contents/MacOS/kicad-cli``, the libraries in
  ``Contents/SharedSupport`` (``symbols``, ``footprints``, ``3dmodels``) and
  the bundle's own ngspice library, found by name inside the bundle
  (:func:`bundled_ngspice`) because its folder is not a documented interface.
* Linux and other POSIX systems - the distribution's package puts
  ``kicad-cli`` on ``PATH`` (``/usr/bin``) and the libraries in
  ``<prefix>/share/kicad``; nothing else is searched.

KiCad's global library tables (``sym-lib-table``, ``fp-lib-table`` in its
settings folder, :func:`user_config_dir`) are written by the KiCad GUI on its
first start, from its ``template`` folder; ``kicad-cli`` never writes them, and
without them its ERC / DRC warn about every part ("the current configuration
does not include the library"). :func:`install_library_tables` does what that
first start does - it only ever adds a missing table, never changes an
existing one - and is run by the installers through ``ai-eda setup-kicad``.

:func:`platform` is the one place that reads the operating system, so tests
can exercise every layout on any machine.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

#: where KiCad.app is looked for on macOS, in order (``~`` expanded at call time)
MACOS_APP_LOCATIONS: tuple[str, ...] = (
    "/Applications/KiCad/KiCad.app",
    "/Applications/KiCad.app",
    "~/Applications/KiCad/KiCad.app",
    "~/Applications/KiCad.app",
)
#: inside KiCad.app: the command-line tool and the folder holding ``symbols`` / ``footprints`` / ``3dmodels``
MACOS_CLI = Path("Contents/MacOS/kicad-cli")
MACOS_SHARE = Path("Contents/SharedSupport")
#: the bundle folders searched (in this order, sorted by name inside each) for its ngspice library and code models
MACOS_NGSPICE_DIRS: tuple[Path, ...] = (Path("Contents/Frameworks"), Path("Contents/PlugIns"), Path("Contents/SharedSupport"))
#: one code model every ngspice build ships; the folder that holds it is the code model folder
CODEMODEL_MARKER = "analog.cm"


def platform() -> str:
    """``sys.platform`` (``win32``, ``darwin``, ``linux`` ...): the only reader of the operating system here (tests patch it)."""
    return sys.platform


def _absolute_env(name: str) -> Path | None:
    """The environment variable ``name`` as an absolute path, else ``None`` (unset, empty or relative)."""
    value = os.environ.get(name, "")
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else None


def version_key(path: Path) -> tuple[int, ...]:
    """Numeric sort key for a KiCad version directory: ``10.0`` > ``9.0`` (lexicographic sorting gets this wrong)."""
    parts = path.name.split(".")
    if parts and all(p.isdigit() for p in parts):
        return tuple(int(p) for p in parts)
    return (-1,)  # non-version directories sort last


def version_dirs(base: Path) -> list[Path]:
    """Version directories under ``base`` (``...\\Programs\\KiCad``), highest version first."""
    if not base.is_dir():
        return []
    return sorted((p for p in base.iterdir() if p.is_dir()), key=version_key, reverse=True)


def windows_install_bases() -> list[Path]:
    """``<base>`` folders holding ``<ver>`` KiCad installs on Windows (per-user first); empty on any other system."""
    if platform() != "win32":
        return []
    bases: list[Path] = []
    local = _absolute_env("LOCALAPPDATA")
    if local is not None:
        bases.append(local / "Programs" / "KiCad")
    program_files = _absolute_env("ProgramFiles") or Path("C:/Program Files")
    bases.append(program_files / "KiCad")
    return bases


def macos_bundles() -> list[Path]:
    """Existing ``KiCad.app`` bundles on macOS, in :data:`MACOS_APP_LOCATIONS` order; empty on any other system."""
    if platform() != "darwin":
        return []
    home = _absolute_env("HOME")
    out: list[Path] = []
    for loc in MACOS_APP_LOCATIONS:
        if loc.startswith("~/"):
            if home is None:
                continue
            path = home / loc[2:]
        else:
            path = Path(loc)
        if path.is_dir():
            out.append(path)
    return out


def find_kicad_cli() -> str | None:
    """``kicad-cli`` on ``PATH``, else the newest Windows install, else the first macOS bundle (module docstring); ``None`` when absent."""
    found = shutil.which("kicad-cli")
    if found:
        return found
    for base in windows_install_bases():
        for ver in version_dirs(base):
            exe = ver / "bin" / "kicad-cli.exe"
            if exe.is_file():
                return str(exe)
    for bundle in macos_bundles():
        exe = bundle / MACOS_CLI
        if exe.is_file():
            return str(exe)
    return None


def _executable(binary: str | None) -> Path | None:
    if not binary:
        return None
    exe = Path(binary)
    if not exe.is_absolute():
        resolved = shutil.which(binary)
        if not resolved:
            return None
        exe = Path(resolved)
    return exe.resolve()


def _has_libraries(share: Path) -> bool:
    return (share / "symbols").is_dir() or (share / "footprints").is_dir()


def bundle_of(binary: str | None) -> Path | None:
    """The ``KiCad.app`` folder of a ``kicad-cli`` at ``KiCad.app/Contents/MacOS/kicad-cli``, else ``None``."""
    exe = _executable(binary)
    if exe is None or exe.parent.name != "MacOS" or exe.parent.parent.name != "Contents":
        return None
    bundle = exe.parent.parent.parent
    return bundle if bundle.suffix == ".app" else None


def share_dir(binary: str | None) -> Path | None:
    """The folder holding ``symbols`` / ``footprints`` / ``3dmodels`` of the KiCad whose ``kicad-cli`` is ``binary``, else ``None``.

    ``<root>/share/kicad`` for ``<root>/bin/kicad-cli(.exe)`` (Windows, Linux),
    ``KiCad.app/Contents/SharedSupport`` for a bundle's ``kicad-cli`` (macOS);
    a candidate without a ``symbols`` or ``footprints`` folder is no library.
    """
    exe = _executable(binary)
    if exe is None:
        return None
    candidates = [exe.parent.parent / "share" / "kicad"]
    bundle = bundle_of(str(exe))
    if bundle is not None:
        candidates.insert(0, bundle / MACOS_SHARE)
    for share in candidates:
        if share.is_dir() and _has_libraries(share):
            return share
    return None


def library_shares() -> list[Path]:
    """Every installed KiCad's library folder this system's installers use (Windows versions, newest first; macOS bundles)."""
    out: list[Path] = []
    for base in windows_install_bases():
        out.extend(ver / "share" / "kicad" for ver in version_dirs(base) if (ver / "share" / "kicad").is_dir())
    for bundle in macos_bundles():
        share = bundle / MACOS_SHARE
        if share.is_dir() and _has_libraries(share):
            out.append(share)
    return out


def bundled_ngspice(binary: str | None) -> Path | None:
    """The ngspice shared library that ships with the KiCad whose ``kicad-cli`` is ``binary``, else ``None``.

    Windows: ``ngspice.dll`` beside ``kicad-cli.exe``. macOS: the first
    ``libngspice*.dylib`` (sorted by name) in :data:`MACOS_NGSPICE_DIRS` of the
    bundle, searched in that order.
    """
    exe = _executable(binary)
    if exe is None:
        return None
    dll = exe.parent / "ngspice.dll"
    if dll.is_file():
        return dll
    bundle = bundle_of(str(exe))
    if bundle is None:
        return None
    for folder in MACOS_NGSPICE_DIRS:
        root = bundle / folder
        if not root.is_dir():
            continue
        found = sorted(p for p in root.rglob("libngspice*.dylib") if p.is_file())
        if found:
            return found[0]
    return None


def bundled_codemodels(dll: Path) -> Path | None:
    """The code model folder of a macOS bundle's ngspice library ``dll`` (the folder holding :data:`CODEMODEL_MARKER`), else ``None``."""
    bundle = next((p for p in dll.parents if p.suffix == ".app"), None)
    if bundle is None:
        return None
    for folder in MACOS_NGSPICE_DIRS:
        root = bundle / folder
        if not root.is_dir():
            continue
        found = sorted(p.parent for p in root.rglob(CODEMODEL_MARKER) if p.is_file())
        if found:
            return found[0]
    return None


# --- the global library tables ---------------------------------------------------------

#: KiCad's global library tables: file name -> (s-expression head, library kind) - what KiCad's first start writes
LIBRARY_TABLES: dict[str, tuple[str, str]] = {"sym-lib-table": ("sym_lib_table", "symbols"), "fp-lib-table": ("fp_lib_table", "footprints")}
#: the folder of the KiCad library folder that holds the default tables KiCad copies on its first start
TEMPLATE_DIR = "template"


def settings_version(version: str) -> str | None:
    """KiCad's settings folder name for a ``kicad-cli --version`` string: ``10.0.6`` -> ``10.0`` (major.minor), else ``None``."""
    parts = version.strip().split(".")
    if len(parts) < 2 or not parts[0].isdigit():
        return None
    minor = "".join(ch for ch in parts[1] if ch.isdigit())  # "0-rc1" -> "0"
    return f"{int(parts[0])}.{int(minor)}" if minor else None


def user_config_dir(version: str) -> Path | None:
    """The folder KiCad ``version`` reads its settings and global library tables from, else ``None``.

    ``$KICAD_CONFIG_HOME/<major.minor>`` when that variable is an absolute path
    (KiCad appends the version to it too), else ``%APPDATA%\\kicad\\<v>``
    (Windows), ``~/Library/Preferences/kicad/<v>`` (macOS),
    ``$XDG_CONFIG_HOME/kicad/<v>`` or ``~/.config/kicad/<v>`` (Linux and other
    POSIX systems); never a path built from an empty or relative variable.
    """
    ver = settings_version(version)
    if ver is None:
        return None
    root = _absolute_env("KICAD_CONFIG_HOME")
    if root is None:
        system = platform()
        if system == "win32":
            base = _absolute_env("APPDATA")
        elif system == "darwin":
            home = _absolute_env("HOME")
            base = home / "Library" / "Preferences" if home is not None else None
        else:
            home = _absolute_env("HOME")
            base = _absolute_env("XDG_CONFIG_HOME") or (home / ".config" if home is not None else None)
        if base is None:
            return None
        root = base / "kicad"
    return root / ver


@dataclass(frozen=True)
class LibraryTable:
    """One global library table: where KiCad reads it, whether it is there, and what :func:`install_library_tables` would write."""

    name: str
    path: Path | None
    present: bool
    template: Path | None


def library_tables(binary: str | None, version: str) -> list[LibraryTable]:
    """The state of each of :data:`LIBRARY_TABLES` for the KiCad whose ``kicad-cli`` is ``binary`` (``version`` as it prints it)."""
    config = user_config_dir(version)
    share = share_dir(binary)
    out: list[LibraryTable] = []
    for name in LIBRARY_TABLES:
        path = config / name if config is not None else None
        template = share / TEMPLATE_DIR / name if share is not None else None
        out.append(LibraryTable(name, path, path is not None and path.is_file(), template if template is not None and template.is_file() else None))
    return out


def _quoted(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def listed_table(name: str, share: Path) -> str | None:
    """A global table ``name`` listing every library in ``share`` by absolute path (no template on disk), else ``None`` when it holds none."""
    head, kind = LIBRARY_TABLES[name]
    folder = share / kind
    if not folder.is_dir():
        return None
    if kind == "symbols":
        libs = sorted((p.stem, p) for p in folder.glob("*.kicad_sym") if p.is_file())
    else:
        libs = sorted((p.stem, p) for p in folder.glob("*.pretty") if p.is_dir())
    if not libs:
        return None
    rows = "".join(
        f'  (lib (name {_quoted(nick)})(type "KiCad")(uri {_quoted(path.resolve().as_posix())})(options "")(descr ""))\n' for nick, path in libs
    )
    return f"({head}\n  (version 7)\n{rows})\n"


def install_library_tables(binary: str | None, version: str) -> list[str]:
    """Write each missing global library table as KiCad's first start does; one line per table saying what was done.

    A table already there is never touched (it may hold the user's own
    libraries). A missing one is copied from the KiCad library folder's
    ``template`` (the file KiCad itself copies), else written listing the
    libraries found there. Raises ``OSError`` when a table cannot be written.
    """
    share = share_dir(binary)
    lines: list[str] = []
    for table in library_tables(binary, version):
        if table.path is None:
            lines.append(f"{table.name}: no KiCad settings folder for version {version!r} (HOME / APPDATA unset)")
        elif table.present:
            lines.append(f"{table.name}: kept {table.path} (already there)")
        elif table.template is not None:
            table.path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(table.template, table.path)
            lines.append(f"{table.name}: wrote {table.path} (copied from {table.template})")
        else:
            text = listed_table(table.name, share) if share is not None else None
            if text is None:
                lines.append(f"{table.name}: not written (no KiCad library folder with {LIBRARY_TABLES[table.name][1]} found)")
                continue
            table.path.parent.mkdir(parents=True, exist_ok=True)
            table.path.write_text(text, encoding="utf-8", newline="\n")
            lines.append(f"{table.name}: wrote {table.path} (listing the libraries in {share})")
    return lines


__all__ = [
    "CODEMODEL_MARKER",
    "LIBRARY_TABLES",
    "LibraryTable",
    "TEMPLATE_DIR",
    "MACOS_APP_LOCATIONS",
    "MACOS_CLI",
    "MACOS_NGSPICE_DIRS",
    "MACOS_SHARE",
    "bundle_of",
    "bundled_codemodels",
    "bundled_ngspice",
    "find_kicad_cli",
    "install_library_tables",
    "library_shares",
    "library_tables",
    "listed_table",
    "macos_bundles",
    "platform",
    "settings_version",
    "share_dir",
    "version_dirs",
    "user_config_dir",
    "version_key",
    "windows_install_bases",
]
