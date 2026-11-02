"""What to install when ``ai-eda doctor`` finds a tool missing, per operating system.

Invariant: these are instructions printed for a person, never commands this
package runs (installing software is the installer scripts' job, with the
user's own privileges). The operating system is read through
:func:`ai_eda.tools.kicad.install.platform` and, on Linux, ``/etc/os-release``.
Every line also names the one-command installer of this repository
(``install.sh`` / ``install.cmd``), which installs all of it.
"""

from __future__ import annotations

from pathlib import Path

#: the repository's installers (README, 설치)
INSTALLER_HINT = "or run the repository installer again: ./install.sh (macOS / Linux), install.cmd (Windows)"

_REMEDIES: dict[str, dict[str, str]] = {
    "kicad": {
        "darwin": "brew install --cask kicad",
        "win32": "winget install --id KiCad.KiCad -e",
        "ubuntu": "sudo add-apt-repository ppa:kicad/kicad-10.0-releases && sudo apt install kicad",
        "debian": "Debian 12 packages KiCad 6.0 and Debian 13 KiCad 9.0; KiCad 10 is only in Debian testing / unstable - install it from https://www.kicad.org/download/",
        "arch": "sudo pacman -S kicad kicad-library",
        "fedora": "sudo dnf install kicad",
        "other": "install KiCad 10 from https://www.kicad.org/download/",
    },
    "ngspice": {
        "darwin": "brew install libngspice (or set NGSPICE_DLL to the absolute path of a libngspice.dylib)",
        "win32": "KiCad's own bin\\ngspice.dll is used: install KiCad 10 (winget install --id KiCad.KiCad -e)",
        "ubuntu": "sudo apt install libngspice0 ngspice (the second package holds the code models)",
        "debian": "sudo apt install libngspice0 ngspice (the second package holds the code models)",
        "arch": "sudo pacman -S ngspice",
        "fedora": "sudo dnf install libngspice",
        "other": "install ngspice's shared library (libngspice) or set NGSPICE_DLL to its absolute path",
    },
    "browser": {
        "darwin": "install Google Chrome, Microsoft Edge or Chromium in /Applications (only the PDF copies of the stage reports need it)",
        "win32": "Microsoft Edge is part of Windows; set AI_EDA_BROWSER to the absolute path of msedge.exe if it was removed",
        "ubuntu": "install Google Chrome's .deb (amd64: https://www.google.com/chrome/); a snap Chromium (sudo snap install chromium) prints only projects in non-hidden folders of your home folder (only the PDF copies of the stage reports need a browser)",
        "debian": "sudo apt install chromium (only the PDF copies of the stage reports need it)",
        "arch": "sudo pacman -S chromium (only the PDF copies of the stage reports need it)",
        "fedora": "sudo dnf install chromium (only the PDF copies of the stage reports need it)",
        "other": "install Chromium, Chrome or Edge, or set AI_EDA_BROWSER to its absolute path (only the PDF copies need it)",
    },
}


def _os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key.strip()] = value.strip().strip('"')
    return out


def system_family(os_release: Path = Path("/etc/os-release")) -> str:
    """``darwin``, ``win32``, ``ubuntu``, ``debian``, ``arch``, ``fedora`` or ``other`` (the key of the remedy tables)."""
    from ai_eda.tools.kicad.install import platform

    system = platform()
    if system in ("darwin", "win32"):
        return system
    info = _os_release(os_release)
    ids = [info.get("ID", "").lower(), *info.get("ID_LIKE", "").lower().split()]
    for family in ("ubuntu", "arch", "fedora", "debian"):
        if family in ids:
            return family
    return "other"


def system_label(os_release: Path = Path("/etc/os-release")) -> str:
    """A readable name of the running system (``macOS``, ``Windows``, the Linux ``PRETTY_NAME``)."""
    family = system_family(os_release)
    if family == "darwin":
        return "macOS"
    if family == "win32":
        return "Windows"
    return _os_release(os_release).get("PRETTY_NAME") or "Linux / other"


def remedy(component: str, os_release: Path = Path("/etc/os-release")) -> str:
    """The install hint for ``component`` (``kicad`` / ``ngspice`` / ``browser``) on this system, plus :data:`INSTALLER_HINT`."""
    table = _REMEDIES[component]
    hint = table.get(system_family(os_release), table["other"])
    return f"{hint}; {INSTALLER_HINT}"


__all__ = ["INSTALLER_HINT", "remedy", "system_family", "system_label"]
