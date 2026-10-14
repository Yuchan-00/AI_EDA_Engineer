#!/usr/bin/env bash
# AI EDA ENGINEER - one-command installer for macOS and Linux (Ubuntu, Debian, Arch, Fedora).
# What it installs, its options and their environment variables are in usage() just below (`--help` prints it,
# also when the script comes in on stdin: `curl ... | bash -s -- --help`).
# Invariants: it installs for the user who runs it (never under sudo: it calls sudo itself for the system
# packages), asks the package manager only for what is missing, records only absolute paths, and never stops
# because a shell profile cannot be edited. Its steps (`run`), Homebrew's installer and the ai-eda commands read
# /dev/null as stdin, so that under `curl ... | bash` none of them can read the rest of this script.

usage() {
  cat <<'EOF'
AI EDA ENGINEER - one-command installer for macOS and Linux (Ubuntu, Debian, Arch, Fedora).

  curl -fsSL https://raw.githubusercontent.com/Yuchan-00/AI_EDA_Engineer/main/install.sh | bash
  curl -fsSL https://raw.githubusercontent.com/Yuchan-00/AI_EDA_Engineer/main/install.sh | bash -s -- [options]
  ./install.sh [options]                      # from a checkout of the repository

Run it as yourself, not with sudo: it calls sudo for the system packages, and sudo asks for your password
where the system requires it (on macOS once, before Homebrew's installer, when Homebrew is missing).
On a machine with nothing set up it installs, without asking:
  1. the code   - this checkout, else the repository into ~/AI_EDA_Engineer (git clone; a tarball when git is
                  missing or cannot clone, as on a Mac without the Command Line Tools)
  2. Python     - uv (Astral) into ~/.local/bin, a uv-managed Python 3.12 and .venv with the package and its extras
  3. KiCad 10   - kicad-cli and the symbol / footprint libraries, plus ngspice's shared library (the package
                  manager; on macOS Homebrew, installed first when missing; an older KiCad is upgraded - on macOS a
                  KiCad folder Homebrew did not install is kept, renamed /Applications/KiCad-<version>)
  4. a browser  - for the PDF copies of the stage reports: Chromium where the system packages one (Arch, Fedora,
                  Debian); on Ubuntu (amd64) Google Chrome's .deb, which adds Google's apt source for its updates
                  - Ubuntu's Chromium is a snap, which cannot read a project outside your home folder (the
                  fallback on other architectures)
  5. ai-eda     - a link in ~/.local/bin and a PATH line in the shell profiles, KiCad's global library tables
                  (`ai-eda setup-kicad`, what KiCad's first start writes; also with --no-kicad when a KiCad is
                  already installed), then `ai-eda doctor` and `ai-eda selftest` (designs a board end to end;
                  run only when setup-kicad found a KiCad and its tables are in place)
Re-running it updates a git copy it downloaded (git pull; a tarball copy, and a checkout you run it from, are
yours to update) and installs only what is missing: the package manager is asked only for packages that are not
installed.
  Arch     installing anything first upgrades the whole system (pacman -Syu, once per run: Arch does not
           support partial upgrades); a run that finds every package installed syncs and upgrades nothing.
  Debian   Debian 12 (bookworm) packages KiCad 6.0 and Debian 13 (trixie) KiCad 9.0; KiCad 10 is only in
           testing / unstable. The installer installs what Debian has but cannot provide KiCad 10 there
           (`ai-eda doctor --strict` reports it): use --no-kicad and install KiCad 10 another way.

Options (environment variables in brackets, for `curl ... | bash`; a switch's variable is set to 1):
  --dir DIR        where to put the code when not run from a checkout [AI_EDA_DIR] (default ~/AI_EDA_Engineer)
  --branch NAME    branch to download [AI_EDA_BRANCH] (default main)
  --python VER     Python version for the virtual environment [AI_EDA_PYTHON] (default 3.12)
  --no-kicad       do not install KiCad / ngspice (without a KiCad the pipeline reports those checks
                   NOT_VERIFIED) [AI_EDA_NO_KICAD=1]
  --with-3d        also install KiCad's 3D model library (several GB; only the 3D preview's part heights use it)
                   [AI_EDA_WITH_3D=1]
  --no-browser     do not install a browser for the PDF reports [AI_EDA_NO_BROWSER=1]
  --with-claude    also install the Claude Code CLI (`--llm claude`; you log in with `claude` afterwards; a
                   failed download is a warning) [AI_EDA_WITH_CLAUDE=1]
  --no-path        do not link ai-eda into ~/.local/bin or edit shell profiles [AI_EDA_NO_PATH=1]
  --skip-selftest  do not run `ai-eda selftest` at the end [AI_EDA_SKIP_SELFTEST=1]
  -h, --help       this text
EOF
}

set -euo pipefail
unset CDPATH

REPO_SLUG="Yuchan-00/AI_EDA_Engineer"
REPO_URL="https://github.com/${REPO_SLUG}"
APP_DIR_DEFAULT="${HOME}/AI_EDA_Engineer"

# a switch's environment variable is on when it says 1 (true / yes are taken too)
is_on() { case "${1:-}" in 1|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]) return 0 ;; *) return 1 ;; esac; }

DIR="${AI_EDA_DIR:-}"
BRANCH="${AI_EDA_BRANCH:-main}"
PYVER="${AI_EDA_PYTHON:-3.12}"
WITH_KICAD=1;   if is_on "${AI_EDA_NO_KICAD:-}"; then WITH_KICAD=0; fi
WITH_3D=0;      if is_on "${AI_EDA_WITH_3D:-}"; then WITH_3D=1; fi
WITH_BROWSER=1; if is_on "${AI_EDA_NO_BROWSER:-}"; then WITH_BROWSER=0; fi
WITH_CLAUDE=0;  if is_on "${AI_EDA_WITH_CLAUDE:-}"; then WITH_CLAUDE=1; fi
WITH_PATH=1;    if is_on "${AI_EDA_NO_PATH:-}"; then WITH_PATH=0; fi
RUN_SELFTEST=1; if is_on "${AI_EDA_SKIP_SELFTEST:-}"; then RUN_SELFTEST=0; fi

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
run() { printf '    + %s\n' "$*"; "$@" </dev/null; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="${2:?--dir needs a folder}"; shift 2 ;;
    --branch) BRANCH="${2:?--branch needs a name}"; shift 2 ;;
    --python) PYVER="${2:?--python needs a version}"; shift 2 ;;
    --no-kicad) WITH_KICAD=0; shift ;;
    --with-3d) WITH_3D=1; shift ;;
    --no-browser) WITH_BROWSER=0; shift ;;
    --with-claude) WITH_CLAUDE=1; shift ;;
    --no-path) WITH_PATH=0; shift ;;
    --skip-selftest) RUN_SELFTEST=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done

# --------------------------------------------------------------------------- the system

OS="$(uname -s)"
FAMILY=""
case "$OS" in
  Darwin) FAMILY=macos ;;
  Linux)
    ID=""; ID_LIKE=""; PRETTY_NAME="Linux"
    # shellcheck disable=SC1091
    [ -r /etc/os-release ] && . /etc/os-release
    for f in $ID $ID_LIKE; do
      case "$f" in
        ubuntu) FAMILY=ubuntu; break ;;
        arch|archlinux) FAMILY=arch; break ;;
        fedora) FAMILY=fedora; break ;;
        debian) FAMILY=debian ;;
      esac
    done
    [ -n "$FAMILY" ] || die "unsupported Linux distribution '${PRETTY_NAME}' (supported: Ubuntu, Debian, Arch, Fedora and their derivatives); install Python 3.12+, KiCad 10 and ngspice yourself, then: uv pip install -e '.[llm,parts,dev]'"
    ;;
  *) die "unsupported system $OS (use install.ps1 / install.cmd on Windows)" ;;
esac
say "system: ${FAMILY} ($(uname -m))"

# under sudo HOME is root's: uv, the link, the profiles and KiCad's tables would all be root's, and a checkout's
# .venv root-owned. Root without SUDO_USER (a container) installs for root, as asked.
if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != root ]; then
  die "do not run this installer with sudo: it installs into your own home folder and calls sudo itself for the system packages; run it again as ${SUDO_USER}, without sudo"
fi
[ "$FAMILY" = macos ] && [ "$(id -u)" -eq 0 ] && die "do not run this installer as root on macOS (Homebrew refuses root)"

# the package manager runs as root: through sudo for a user (never on macOS, where Homebrew refuses root)
SUDO=""
if [ "$FAMILY" != macos ] && [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi
as_root() {
  if [ -n "$SUDO" ] && ! command -v sudo >/dev/null 2>&1; then
    die "'$*' needs root and sudo is not installed: as root, install sudo and add $(id -un) to the sudo (or wheel) group, then log in again and run this installer as $(id -un)"
  fi
  run $SUDO "$@"
}

APT_UPDATED=0
apt_install() {
  if [ "$APT_UPDATED" -eq 0 ]; then as_root apt-get update; APT_UPDATED=1; fi
  as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
}
# Arch: packages that are all installed already are left alone (nothing synced or upgraded for them); anything
# else is installed after a full system upgrade, once per run - partial upgrades are unsupported on Arch
PACMAN_SYNCED=0
pacman_install() {
  if pacman -Q "$@" >/dev/null 2>&1; then return 0; fi
  if [ "$PACMAN_SYNCED" -eq 0 ]; then as_root pacman -Syu --noconfirm; PACMAN_SYNCED=1; fi
  as_root pacman -S --needed --noconfirm "$@"
}
dnf_install() { as_root dnf install -y "$@"; }
# the packages of the list that are not installed (Debian / Ubuntu; Fedora), one line, space-separated
deb_missing() {
  local p out=""
  for p in "$@"; do
    dpkg-query -W -f='${Status}\n' "$p" 2>/dev/null </dev/null | grep -q ' ok installed$' || out="$out $p"
  done
  printf '%s\n' "${out# }"
}
rpm_missing() {
  local p out=""
  for p in "$@"; do
    rpm -q "$p" >/dev/null 2>&1 </dev/null || out="$out $p"
  done
  printf '%s\n' "${out# }"
}

# the first line `<kicad-cli> version` prints ("10.0.7"); nothing when $1 is not a kicad-cli that runs
kicad_version() {
  if [ -z "${1:-}" ] || [ ! -x "$1" ]; then return 0; fi
  "$1" version 2>/dev/null </dev/null | head -n 1 || true
}
# the leading number of a version text, 0 when it has none
major_of() { local m="${1:-}"; m="${m%%[!0-9]*}"; printf '%s\n' "${m:-0}"; }
# a KiCad.app's version: its own kicad-cli's, else the bundle's CFBundleShortVersionString (nothing when neither says)
kicad_app_version() {
  local v plist="$1/Contents/Info.plist"
  v="$(kicad_version "$1/Contents/MacOS/kicad-cli")"
  if [ -z "$v" ] && [ -x /usr/libexec/PlistBuddy ]; then
    v="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$plist" 2>/dev/null </dev/null || true)"
  fi
  if [ -z "$v" ] && command -v plutil >/dev/null 2>&1; then
    v="$(plutil -extract CFBundleShortVersionString raw -o - "$plist" 2>/dev/null </dev/null || true)"
  fi
  printf '%s\n' "$v"
}

# Homebrew's installer runs non-interactively (NONINTERACTIVE=1 here, and by itself under `curl | bash`), so it
# cannot ask for the administrator password it needs: sudo -v asks for it first, on the terminal, and Homebrew
# uses that cached credential
ensure_brew() {
  local b script
  if ! command -v brew >/dev/null 2>&1; then
    for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do
      if [ -x "$b" ]; then eval "$("$b" shellenv)"; break; fi
    done
  fi
  if ! command -v brew >/dev/null 2>&1; then
    say "installing Homebrew (https://brew.sh)"
    if [ "$(id -u)" -ne 0 ] && ! sudo -n true 2>/dev/null; then
      if ( : </dev/tty ) 2>/dev/null; then
        say "Homebrew's installer needs administrator rights and cannot ask for them itself: sudo asks for your password once"
        # shellcheck disable=SC2024  # the terminal is sudo's own stdin on purpose (under `curl | bash` stdin is the pipe)
        sudo -v </dev/tty || die "sudo -v failed: Homebrew's installer needs an administrator account; install Homebrew yourself (https://brew.sh), then run this installer again"
      else
        die "Homebrew is not installed, and without a terminal nobody can give its installer the administrator password: install Homebrew first (https://brew.sh), then run this installer again"
      fi
    fi
    script="$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || die "could not download Homebrew's installer"
    NONINTERACTIVE=1 /bin/bash -c "$script" </dev/null
    for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do
      if [ -x "$b" ]; then eval "$("$b" shellenv)"; break; fi
    done
  fi
  command -v brew >/dev/null 2>&1 || die "Homebrew is not available"
}

# --------------------------------------------------------------------------- 0. base tools

# only the ones this system lacks: asking the package manager for a tool that is there can fail (the curl of
# Fedora and the RHEL family conflicts with their default curl-minimal), and on Arch it upgrades the whole system
has_ca_bundle() {
  local f
  for f in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/cert.pem; do
    if [ -s "$f" ]; then return 0; fi
  done
  return 1
}
if [ "$FAMILY" = macos ]; then
  for t in curl tar; do
    command -v "$t" >/dev/null 2>&1 || die "$t is missing (it comes with macOS)"
  done
  say "base tools: curl and tar are there (git is used when it works, else a tarball)"
else
  base=""
  for t in curl tar gzip git; do
    command -v "$t" >/dev/null 2>&1 || base="$base $t"
  done
  has_ca_bundle || base="$base ca-certificates"
  if [ -z "$base" ]; then
    say "base tools: curl, tar, gzip and git are there"
  else
    say "base tools:$base"
    # shellcheck disable=SC2086  # $base is a list of package names, one word each
    case "$FAMILY" in
      ubuntu|debian) apt_install $base ;;
      arch) pacman_install $base ;;
      fedora) dnf_install $base ;;
    esac
  fi
fi

# --------------------------------------------------------------------------- 1. the code

is_checkout() { [ -f "$1/pyproject.toml" ] && grep -q '^name = "ai-eda-engineer"' "$1/pyproject.toml"; }

# a git that works: on macOS /usr/bin/git is a stub until the Command Line Tools are installed (it fails, and
# running it opens their install dialog), so there it counts only when a developer folder is set
git_works() {
  command -v git >/dev/null 2>&1 || return 1
  if [ "$FAMILY" = macos ] && [ "$(command -v git)" = /usr/bin/git ] && ! xcode-select -p >/dev/null 2>&1; then
    return 1
  fi
  git --version >/dev/null 2>&1 </dev/null
}

# the branch's tarball into $1, which does not exist or is an empty folder
download_tarball() {
  local tmp top
  tmp="$(mktemp -d)"
  run curl -fsSL "https://codeload.github.com/${REPO_SLUG}/tar.gz/refs/heads/${BRANCH}" -o "$tmp/src.tar.gz"
  run tar -xzf "$tmp/src.tar.gz" -C "$tmp"
  top="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  [ -n "$top" ] || die "the downloaded archive holds no folder"
  mkdir -p "$(dirname "$1")"
  # an empty folder is replaced, not filled: mv would put the download inside it
  if [ -d "$1" ]; then
    rmdir "$1" 2>/dev/null || die "$1 is not empty (left by a failed git clone?); remove it and run this installer again"
  fi
  run mv "$top" "$1"
  rm -rf "$tmp"
  # the folder replaced may have been the current one (`--dir .`): stand in the new one, not in the removed one
  cd "$PWD" 2>/dev/null || cd /
}

SCRIPT_DIR=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [ -n "$SCRIPT_DIR" ] && is_checkout "$SCRIPT_DIR" && [ -z "$DIR" ]; then
  APP_DIR="$SCRIPT_DIR"
  say "code: this checkout ($APP_DIR; update it yourself with git pull)"
else
  APP_DIR="${DIR:-$APP_DIR_DEFAULT}"
  # absolute from here on: the ai-eda link records this path, and a relative one would resolve from ~/.local/bin
  case "$APP_DIR" in /*) ;; *) APP_DIR="$PWD/$APP_DIR" ;; esac
  # an existing folder by its own name (`--dir .` is the current folder, not "<it>/.", which rmdir and mv refuse);
  # a new one without trailing "/." or "/"
  if [ -d "$APP_DIR" ]; then
    APP_DIR="$(cd "$APP_DIR" && pwd)"
  else
    while :; do
      case "$APP_DIR" in
        */.) APP_DIR="${APP_DIR%/.}" ;;
        ?*/) APP_DIR="${APP_DIR%/}" ;;
        *) break ;;
      esac
    done
  fi
  if is_checkout "$APP_DIR"; then
    say "code: existing copy in $APP_DIR"
    if [ -d "$APP_DIR/.git" ] && git_works; then
      run git -C "$APP_DIR" pull --ff-only || warn "could not update $APP_DIR with git pull (local changes?); using it as it is"
    fi
  elif [ -e "$APP_DIR" ] && [ -n "$(ls -A "$APP_DIR" 2>/dev/null)" ]; then
    die "$APP_DIR exists and is not a copy of this project; choose another --dir"
  else
    say "code: downloading ${REPO_URL} (${BRANCH}) into $APP_DIR"
    cloned=0
    if git_works; then
      if run git clone --depth 1 --branch "$BRANCH" "${REPO_URL}.git" "$APP_DIR"; then
        cloned=1
      else
        warn "git clone failed: downloading the tarball instead"
      fi
    fi
    [ "$cloned" -eq 1 ] || download_tarball "$APP_DIR"
  fi
fi
is_checkout "$APP_DIR" || die "$APP_DIR does not hold the project"
APP_DIR="$(cd "$APP_DIR" && pwd)"

# --------------------------------------------------------------------------- 2. Python environment

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  say "installing uv (https://docs.astral.sh/uv/) into ~/.local/bin"
  # UV_INSTALL_DIR: without it uv's installer follows XDG_BIN_HOME / XDG_DATA_HOME away from ~/.local/bin
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh
fi
command -v uv >/dev/null 2>&1 || die "uv is not available after its installer ran"
say "Python ${PYVER} environment in $APP_DIR/.venv"
run uv venv --allow-existing --python "$PYVER" "$APP_DIR/.venv"
VENV_PY="$APP_DIR/.venv/bin/python"
run uv pip install --python "$VENV_PY" -e "${APP_DIR}[llm,parts,dev]" cryptography

# --------------------------------------------------------------------------- 3. KiCad 10 + ngspice

install_kicad() {
  local pkgs ver app a missing old
  case "$FAMILY" in
    ubuntu)
      # libngspice0 is the shared library; its XSPICE code models (*.cm) come in the separate ngspice package
      pkgs=(kicad kicad-symbols kicad-footprints libngspice0 ngspice)
      [ "$WITH_3D" -eq 1 ] && pkgs+=(kicad-packages3d)
      ver="$(kicad_version "$(command -v kicad-cli || true)")"
      missing="$(deb_missing "${pkgs[@]}")"
      if [ "$(major_of "$ver")" -ge 10 ]; then
        # KiCad 10 is there (from the PPA, or however it came): no repository is added, only what is missing
        if [ -z "$missing" ]; then
          say "KiCad $ver, its libraries and ngspice are installed"
        else
          # shellcheck disable=SC2086  # a list of package names, one word each
          apt_install $missing
        fi
      else
        [ -z "$ver" ] || say "KiCad $ver is older than 10: KiCad's PPA upgrades it"
        apt_install software-properties-common gnupg
        as_root add-apt-repository -y ppa:kicad/kicad-10.0-releases
        APT_UPDATED=0
        apt_install "${pkgs[@]}"
      fi
      ;;
    debian)
      pkgs=(kicad kicad-symbols kicad-footprints libngspice0 ngspice)
      [ "$WITH_3D" -eq 1 ] && pkgs+=(kicad-packages3d)
      missing="$(deb_missing "${pkgs[@]}")"
      if [ -n "$missing" ]; then
        say "Debian's own KiCad: Debian 12 (bookworm) has 6.0, Debian 13 (trixie) 9.0; KiCad 10 is only in testing / unstable"
        # shellcheck disable=SC2086  # a list of package names, one word each
        apt_install $missing
      fi
      ver="$(kicad_version "$(command -v kicad-cli || true)")"
      if [ "$(major_of "$ver")" -lt 10 ]; then
        warn "Debian's KiCad here is ${ver:-older than 7 (it has no kicad-cli)}, not 10, and this installer cannot provide KiCad 10 on Debian: ai-eda doctor --strict reports it and the selftest cannot pass; run the installer with --no-kicad and install KiCad 10 another way"
      fi
      ;;
    arch)
      pkgs=(kicad kicad-library ngspice)
      [ "$WITH_3D" -eq 1 ] && pkgs+=(kicad-library-3d)
      # an installed KiCad older than 10 is upgraded with the system (pacman_install would leave it alone)
      ver="$(kicad_version "$(command -v kicad-cli || true)")"
      if [ -n "$ver" ] && [ "$(major_of "$ver")" -lt 10 ] && [ "$PACMAN_SYNCED" -eq 0 ]; then
        say "KiCad $ver is older than 10: upgrading the system (pacman -Syu) to Arch's current KiCad"
        as_root pacman -Syu --noconfirm
        PACMAN_SYNCED=1
      fi
      pacman_install "${pkgs[@]}"
      ;;
    fedora)
      pkgs=(kicad libngspice)
      [ "$WITH_3D" -eq 1 ] && pkgs+=(kicad-packages3d)
      missing="$(rpm_missing "${pkgs[@]}")"
      ver="$(kicad_version "$(command -v kicad-cli || true)")"
      if [ -n "$missing" ]; then
        # shellcheck disable=SC2086  # a list of package names, one word each
        dnf_install $missing
      elif [ "$(major_of "$ver")" -lt 10 ]; then
        # installed but older than 10: dnf upgrade takes Fedora's current KiCad
        as_root dnf upgrade -y "${pkgs[@]}"
      else
        say "KiCad $ver and ngspice are installed"
      fi
      ;;
    macos)
      ensure_brew
      app=""
      for a in /Applications/KiCad/KiCad.app /Applications/KiCad.app; do
        if [ -d "$a" ]; then app="$a"; break; fi
      done
      if [ -z "$app" ]; then
        run brew install --cask kicad
      else
        ver="$(kicad_app_version "$app")"
        if [ "$(major_of "$ver")" -ge 10 ]; then
          say "KiCad $ver is already in $app"
        elif brew list --cask kicad >/dev/null 2>&1 </dev/null; then
          say "KiCad ${ver:-of unknown version} in $app is older than 10: upgrading Homebrew's kicad cask"
          run brew upgrade --cask kicad
        elif [ "$app" = /Applications/KiCad/KiCad.app ]; then
          # Homebrew's cask installs the folder /Applications/KiCad: the KiCad there (not Homebrew's) is kept under
          # another name, never overwritten (`brew install --force` would delete it)
          old="/Applications/KiCad-${ver:-old}"
          while [ -e "$old" ]; do old="${old}-old"; done
          say "KiCad ${ver:-of unknown version} in /Applications/KiCad is older than 10 and not Homebrew's: keeping it as $old"
          run mv /Applications/KiCad "$old" || die "could not rename /Applications/KiCad to $old (rename or remove it yourself, then run this installer again)"
          run brew install --cask kicad
        else
          # a lone /Applications/KiCad.app does not collide with the cask's /Applications/KiCad folder, which is
          # looked at first
          say "KiCad ${ver:-of unknown version} in $app is older than 10: installing KiCad 10 into /Applications/KiCad beside it"
          run brew install --cask kicad
        fi
        ver="$(kicad_app_version /Applications/KiCad/KiCad.app)"
        if [ "$(major_of "$ver")" -lt 10 ]; then
          warn "/Applications/KiCad/KiCad.app is ${ver:-missing or of unknown version}, not KiCad 10, after Homebrew ran: install KiCad 10 from https://www.kicad.org/download/macos/"
        fi
      fi
      run brew install libngspice
      ;;
  esac
}

if [ "$WITH_KICAD" -eq 1 ]; then
  say "KiCad 10 and ngspice"
  install_kicad
else
  say "KiCad / ngspice skipped (--no-kicad)"
fi

# --------------------------------------------------------------------------- 4. browser for the PDF reports

# a snap program (Ubuntu's Chromium) can read only non-hidden folders in the home folder: not counted
is_snap() {
  local p="$1"
  case "$p" in /snap/*) return 0 ;; esac
  case "$(readlink -f "$p" 2>/dev/null || true)" in /snap/*|*/bin/snap) return 0 ;; esac
  [ -f "$p" ] && head -c 4096 "$p" 2>/dev/null | grep -q '/snap/bin/'
}
has_browser() {
  local b p
  for b in google-chrome google-chrome-stable chrome msedge chromium chromium-browser; do
    p="$(command -v "$b" 2>/dev/null || true)"
    case "$p" in /*) is_snap "$p" || return 0 ;; esac
  done
  [ -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ] && return 0
  [ -x "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge" ] && return 0
  [ -x "/Applications/Chromium.app/Contents/MacOS/Chromium" ] && return 0
  return 1
}
if [ "$WITH_BROWSER" -eq 1 ] && ! has_browser; then
  case "$FAMILY" in
    arch) say "Chromium for the PDF reports"; pacman_install chromium ;;
    fedora) say "Chromium for the PDF reports"; dnf_install chromium ;;
    debian) say "Chromium for the PDF reports"; apt_install chromium ;;
    ubuntu)
      # Ubuntu packages Chromium only as a snap, which prints only projects inside the home folder: Google Chrome's
      # .deb first (amd64 only; it adds Google's apt source for its updates)
      if [ "$(dpkg --print-architecture 2>/dev/null </dev/null || true)" = amd64 ]; then
        say "Google Chrome for the PDF reports"
        deb_dir="$(mktemp -d)"
        if run curl -fsSL https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb -o "$deb_dir/google-chrome-stable_current_amd64.deb"; then
          apt_install "$deb_dir/google-chrome-stable_current_amd64.deb" || warn "Google Chrome's package did not install"
        else
          warn "could not download Google Chrome's package"
        fi
        rm -rf "$deb_dir"
      fi
      if has_browser; then
        :
      elif command -v snap >/dev/null 2>&1 && snap version >/dev/null 2>&1 </dev/null; then
        say "Chromium (snap) for the PDF reports: it prints only projects in non-hidden folders of your home folder"
        as_root snap install chromium || warn "snap install chromium failed: the stage reports stay .md + .html"
      else
        warn "no browser for the PDF reports (Ubuntu packages Chromium only as a snap and snapd is not running here): the stage reports stay .md + .html (install Chrome or set AI_EDA_BROWSER for PDFs)"
      fi
      ;;
    macos) warn "no Chrome / Edge / Chromium in /Applications: the stage reports stay .md + .html (install one for PDFs)" ;;
  esac
fi

# --------------------------------------------------------------------------- 5. Claude Code CLI (optional)

if [ "$WITH_CLAUDE" -eq 1 ]; then
  if command -v claude >/dev/null 2>&1; then
    say "Claude Code CLI already installed"
  else
    say "Claude Code CLI (log in afterwards with: claude)"
    # optional: a failure is reported, never the end of the installation
    if ! curl -fsSL https://claude.ai/install.sh | bash; then
      warn "the Claude Code CLI did not install (see above); everything else does not need it - install it later from https://claude.com/claude-code"
    fi
  fi
fi

# --------------------------------------------------------------------------- 6. ai-eda on PATH

if [ "$WITH_PATH" -eq 1 ]; then
  mkdir -p "$HOME/.local/bin"
  run ln -sf "$APP_DIR/.venv/bin/ai-eda" "$HOME/.local/bin/ai-eda"
  # shellcheck disable=SC2016  # the profile line must expand $HOME / $PATH when the shell reads it, not now
  line='export PATH="$HOME/.local/bin:$PATH"  # added by the AI EDA ENGINEER installer'
  # ~/.profile always; a bash login shell reads ~/.bash_profile (else ~/.bash_login) instead of it when one
  # exists; zsh, macOS's login shell, reads ~/.zprofile and ~/.zshrc. A profile that cannot be written (a
  # read-only or dangling link) is reported, never a reason to stop.
  for rc in "$HOME/.profile" "$HOME/.bash_profile" "$HOME/.bash_login" "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.zprofile"; do
    case "$rc" in
      */.zshrc|*/.zprofile) [ "$FAMILY" = macos ] || [ -f "$rc" ] || continue ;;
      */.bash_profile|*/.bash_login|*/.bashrc) [ -f "$rc" ] || continue ;;
    esac
    if grep -qs 'added by the AI EDA ENGINEER installer' "$rc"; then continue; fi
    if ! { printf '\n%s\n' "$line" >> "$rc"; } 2>/dev/null; then
      warn "could not add ~/.local/bin to PATH in $rc; add this line to it yourself: $line"
    fi
  done
fi

# --------------------------------------------------------------------------- 7. check

AI_EDA="$APP_DIR/.venv/bin/ai-eda"
# what KiCad's first start does: the global library tables, without which ERC / DRC warn about every part. Also
# with --no-kicad, for a KiCad installed some other way; an existing table is never changed.
say "KiCad's global library tables (ai-eda setup-kicad)"
KICAD_READY=0
if "$AI_EDA" setup-kicad </dev/null; then
  KICAD_READY=1
elif [ "$WITH_KICAD" -eq 1 ]; then
  warn "ai-eda setup-kicad did not find a usable KiCad or could not write its library tables (ai-eda doctor below says what is missing)"
else
  say "no usable KiCad (--no-kicad): after installing KiCad 10, run ai-eda setup-kicad"
fi
say "ai-eda doctor"
"$AI_EDA" doctor </dev/null || true
status=0
# the KiCad this run was to install is not usable: the installation did not do what it was asked
if [ "$KICAD_READY" -eq 0 ] && [ "$WITH_KICAD" -eq 1 ]; then status=1; fi
if [ "$RUN_SELFTEST" -eq 1 ]; then
  if [ "$KICAD_READY" -eq 1 ]; then
    say "ai-eda selftest (designs the 12 V -> 5 V divider end to end)"
    "$AI_EDA" selftest </dev/null || status=$?
  elif [ "$WITH_KICAD" -eq 1 ]; then
    warn "ai-eda selftest not run: KiCad is not ready (ai-eda setup-kicad failed above)"
  else
    say "ai-eda selftest skipped: no KiCad with its library tables (--no-kicad); after installing KiCad 10: ai-eda setup-kicad && ai-eda selftest"
  fi
fi

cat <<EOF

AI EDA ENGINEER is installed in $APP_DIR
  ai-eda doctor                     what was found
  ai-eda gui --open                 the local web GUI (Korean)
  ai-eda new demo --request "12V 입력을 5V로 변환하는 회로"
Open a new terminal (or run: export PATH="\$HOME/.local/bin:\$PATH") so that 'ai-eda' is found.
EOF
exit "$status"
