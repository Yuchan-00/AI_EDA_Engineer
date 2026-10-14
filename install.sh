#!/usr/bin/env bash
# AI EDA ENGINEER - one-command installer for macOS and Linux (Ubuntu, Debian, Arch, Fedora).
#
#   curl -fsSL https://raw.githubusercontent.com/Yuchan-00/AI_EDA_Engineer/main/install.sh | bash
#   ./install.sh [options]                      # from a checkout of the repository
#
# On a machine with nothing set up it installs, without asking (sudo asks for your password where the
# system requires it):
#   1. the code   - this checkout, else the repository into ~/AI_EDA_Engineer (git clone, or a tarball without git)
#   2. Python     - uv (Astral) into ~/.local/bin, a uv-managed Python 3.12 and .venv with the package and its extras
#   3. KiCad 10   - kicad-cli and the symbol / footprint libraries, plus ngspice's shared library (package manager)
#   4. a browser  - for the PDF copies of the stage reports, where the system packages one (Arch, Fedora, Debian)
#   5. ai-eda     - a link in ~/.local/bin, then `ai-eda doctor` and `ai-eda selftest` (designs a board end to end)
# Re-running it updates the code (git checkouts) and repairs whatever is missing.
#
# Options (environment variables in brackets, for `curl ... | bash`):
#   --dir DIR        where to put the code when not run from a checkout [AI_EDA_DIR] (default ~/AI_EDA_Engineer)
#   --branch NAME    branch to download [AI_EDA_BRANCH] (default main)
#   --python VER     Python version for the virtual environment [AI_EDA_PYTHON] (default 3.12)
#   --no-kicad       do not install KiCad / ngspice (the pipeline then reports those checks NOT_VERIFIED)
#   --with-3d        also install KiCad's 3D model library (several GB; only the 3D preview's part heights use it)
#   --no-browser     do not install a browser for the PDF reports
#   --with-claude    also install the Claude Code CLI (`--llm claude`; you log in with `claude` afterwards)
#   --no-path        do not link ai-eda into ~/.local/bin or edit shell profiles
#   --skip-selftest  do not run `ai-eda selftest` at the end
#   -h, --help       this text
set -euo pipefail

REPO_SLUG="Yuchan-00/AI_EDA_Engineer"
REPO_URL="https://github.com/${REPO_SLUG}"
APP_DIR_DEFAULT="${HOME}/AI_EDA_Engineer"

DIR="${AI_EDA_DIR:-}"
BRANCH="${AI_EDA_BRANCH:-main}"
PYVER="${AI_EDA_PYTHON:-3.12}"
WITH_KICAD=1
WITH_3D=0
WITH_BROWSER=1
WITH_CLAUDE=0
WITH_PATH=1
RUN_SELFTEST=1

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
run() { printf '    + %s\n' "$*"; "$@"; }
usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

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

SUDO=""
if [ "$FAMILY" != macos ] && [ "$(id -u)" -ne 0 ]; then
  command -v sudo >/dev/null 2>&1 || die "sudo is needed to install system packages (or run this script as root)"
  SUDO="sudo"
fi
[ "$FAMILY" = macos ] && [ "$(id -u)" -eq 0 ] && die "do not run this installer as root on macOS (Homebrew refuses root)"

APT_UPDATED=0
apt_install() {
  if [ "$APT_UPDATED" -eq 0 ]; then run $SUDO apt-get update; APT_UPDATED=1; fi
  run $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
}
PACMAN_SYNCED=0
pacman_install() {
  if [ "$PACMAN_SYNCED" -eq 0 ]; then run $SUDO pacman -Syu --noconfirm; PACMAN_SYNCED=1; fi
  run $SUDO pacman -S --needed --noconfirm "$@"
}
dnf_install() { run $SUDO dnf install -y "$@"; }

ensure_brew() {
  if ! command -v brew >/dev/null 2>&1; then
    for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do
      [ -x "$b" ] && eval "$("$b" shellenv)" && break
    done
  fi
  if ! command -v brew >/dev/null 2>&1; then
    say "installing Homebrew (https://brew.sh)"
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do
      [ -x "$b" ] && eval "$("$b" shellenv)" && break
    done
  fi
  command -v brew >/dev/null 2>&1 || die "Homebrew is not available"
}

# --------------------------------------------------------------------------- 0. base tools

say "base tools (curl, tar, git)"
case "$FAMILY" in
  ubuntu|debian) apt_install ca-certificates curl tar gzip git ;;
  arch) pacman_install ca-certificates curl tar gzip git ;;
  fedora) dnf_install ca-certificates curl tar gzip git ;;
  macos) command -v curl >/dev/null 2>&1 || die "curl is missing" ;;
esac

# --------------------------------------------------------------------------- 1. the code

is_checkout() { [ -f "$1/pyproject.toml" ] && grep -q '^name = "ai-eda-engineer"' "$1/pyproject.toml"; }

SCRIPT_DIR=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [ -n "$SCRIPT_DIR" ] && is_checkout "$SCRIPT_DIR" && [ -z "$DIR" ]; then
  APP_DIR="$SCRIPT_DIR"
  say "code: this checkout ($APP_DIR)"
else
  APP_DIR="${DIR:-$APP_DIR_DEFAULT}"
  if is_checkout "$APP_DIR"; then
    say "code: existing copy in $APP_DIR"
    if [ -d "$APP_DIR/.git" ]; then
      run git -C "$APP_DIR" pull --ff-only || warn "could not update $APP_DIR with git pull (local changes?); using it as it is"
    fi
  elif [ -e "$APP_DIR" ] && [ -n "$(ls -A "$APP_DIR" 2>/dev/null)" ]; then
    die "$APP_DIR exists and is not a copy of this project; choose another --dir"
  else
    say "code: downloading ${REPO_URL} (${BRANCH}) into $APP_DIR"
    if command -v git >/dev/null 2>&1; then
      run git clone --depth 1 --branch "$BRANCH" "${REPO_URL}.git" "$APP_DIR"
    else
      tmp="$(mktemp -d)"
      run curl -fsSL "https://codeload.github.com/${REPO_SLUG}/tar.gz/refs/heads/${BRANCH}" -o "$tmp/src.tar.gz"
      run tar -xzf "$tmp/src.tar.gz" -C "$tmp"
      top="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
      mkdir -p "$(dirname "$APP_DIR")"
      run mv "$top" "$APP_DIR"
      rm -rf "$tmp"
    fi
  fi
fi
is_checkout "$APP_DIR" || die "$APP_DIR does not hold the project"

# --------------------------------------------------------------------------- 2. Python environment

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  say "installing uv (https://docs.astral.sh/uv/) into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
command -v uv >/dev/null 2>&1 || die "uv is not available after its installer ran"
say "Python ${PYVER} environment in $APP_DIR/.venv"
run uv venv --allow-existing --python "$PYVER" "$APP_DIR/.venv"
VENV_PY="$APP_DIR/.venv/bin/python"
run uv pip install --python "$VENV_PY" -e "${APP_DIR}[llm,parts,dev]" cryptography

# --------------------------------------------------------------------------- 3. KiCad 10 + ngspice

install_kicad() {
  case "$FAMILY" in
    ubuntu)
      apt_install software-properties-common gnupg
      run $SUDO add-apt-repository -y ppa:kicad/kicad-10.0-releases
      APT_UPDATED=0
      local pkgs=(kicad kicad-symbols kicad-footprints libngspice0)
      [ "$WITH_3D" -eq 1 ] && pkgs+=(kicad-packages3d)
      apt_install "${pkgs[@]}"
      ;;
    debian)
      local pkgs=(kicad kicad-symbols kicad-footprints libngspice0)
      [ "$WITH_3D" -eq 1 ] && pkgs+=(kicad-packages3d)
      apt_install "${pkgs[@]}"
      ;;
    arch)
      local pkgs=(kicad kicad-library ngspice)
      [ "$WITH_3D" -eq 1 ] && pkgs+=(kicad-library-3d)
      pacman_install "${pkgs[@]}"
      ;;
    fedora)
      local pkgs=(kicad libngspice)
      [ "$WITH_3D" -eq 1 ] && pkgs+=(kicad-packages3d)
      dnf_install "${pkgs[@]}"
      ;;
    macos)
      ensure_brew
      if [ -d /Applications/KiCad/KiCad.app ] || [ -d /Applications/KiCad.app ]; then
        say "KiCad.app is already in /Applications"
      else
        run brew install --cask kicad
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

has_browser() {
  for b in chromium chromium-browser google-chrome google-chrome-stable chrome msedge; do
    command -v "$b" >/dev/null 2>&1 && return 0
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
      if command -v snap >/dev/null 2>&1 && snap version >/dev/null 2>&1; then
        say "Chromium (snap) for the PDF reports"; run $SUDO snap install chromium || warn "snap install chromium failed: the stage reports stay .md + .html"
      else
        warn "Ubuntu packages Chromium only as a snap and snapd is not running here: the stage reports stay .md + .html (install Chrome or set AI_EDA_BROWSER for PDFs)"
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
    curl -fsSL https://claude.ai/install.sh | bash
  fi
fi

# --------------------------------------------------------------------------- 6. ai-eda on PATH

if [ "$WITH_PATH" -eq 1 ]; then
  mkdir -p "$HOME/.local/bin"
  run ln -sf "$APP_DIR/.venv/bin/ai-eda" "$HOME/.local/bin/ai-eda"
  # shellcheck disable=SC2016  # the profile line must expand $HOME / $PATH when the shell reads it, not now
  line='export PATH="$HOME/.local/bin:$PATH"  # added by the AI EDA ENGINEER installer'
  for rc in "$HOME/.profile" "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.zprofile"; do
    case "$rc" in
      */.zshrc|*/.zprofile) [ "$FAMILY" = macos ] || [ -f "$rc" ] || continue ;;
      */.bashrc) [ -f "$rc" ] || continue ;;
    esac
    if ! grep -qs 'added by the AI EDA ENGINEER installer' "$rc"; then
      printf '\n%s\n' "$line" >> "$rc"
    fi
  done
fi

# --------------------------------------------------------------------------- 7. check

AI_EDA="$APP_DIR/.venv/bin/ai-eda"
say "ai-eda doctor"
"$AI_EDA" doctor || true
status=0
if [ "$RUN_SELFTEST" -eq 1 ] && [ "$WITH_KICAD" -eq 0 ]; then
  say "ai-eda selftest skipped: without KiCad's libraries (--no-kicad) no circuit template can build"
elif [ "$RUN_SELFTEST" -eq 1 ]; then
  say "ai-eda selftest (designs the 12 V -> 5 V divider end to end)"
  "$AI_EDA" selftest || status=$?
fi

cat <<EOF

AI EDA ENGINEER is installed in $APP_DIR
  ai-eda doctor                     what was found
  ai-eda gui --open                 the local web GUI (Korean)
  ai-eda new demo --request "12V 입력을 5V로 변환하는 회로"
Open a new terminal (or run: export PATH="\$HOME/.local/bin:\$PATH") so that 'ai-eda' is found.
EOF
exit "$status"
