# AI EDA ENGINEER - one-command installer for Windows 10 / 11.
#
#   powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/Yuchan-00/AI_EDA_Engineer/main/install.ps1 | iex"
#   install.cmd [options]                     (from a checkout of the repository; same options as below)
#
# On a machine with nothing set up it installs, without asking (Windows shows its own UAC prompt where an
# installer needs administrator rights):
#   1. the code   - this checkout, else the repository into %USERPROFILE%\AI_EDA_Engineer (git clone, or a zip)
#   2. Python     - uv (Astral) into %USERPROFILE%\.local\bin, a uv-managed Python 3.12 and .venv with the package
#   3. KiCad 10   - with winget (KiCad.KiCad); KiCad brings kicad-cli, the libraries and ngspice (bin\ngspice.dll)
#   4. a browser  - none needed: Microsoft Edge prints the PDF copies of the stage reports
#   5. ai-eda     - %USERPROFILE%\.local\bin\ai-eda.cmd on the user PATH, then `ai-eda doctor` and `ai-eda selftest`
# Re-running it updates the code (git checkouts) and repairs whatever is missing.
#
# Options (environment variables in brackets, for `irm ... | iex`):
#   -Dir DIR        where to put the code when not run from a checkout [AI_EDA_DIR]
#   -Branch NAME    branch to download [AI_EDA_BRANCH] (default main)
#   -Python VER     Python version for the virtual environment [AI_EDA_PYTHON] (default 3.12)
#   -NoKicad        do not install KiCad (the pipeline then reports those checks NOT_VERIFIED)
#   -WithClaude     also install the Claude Code CLI (`--llm claude`; log in with `claude` afterwards)
#   -NoPath         do not add ai-eda to the user PATH
#   -SkipSelftest   do not run `ai-eda selftest` at the end
param(
    [string]$Dir = $env:AI_EDA_DIR,
    [string]$Branch = $(if ($env:AI_EDA_BRANCH) { $env:AI_EDA_BRANCH } else { "main" }),
    [string]$Python = $(if ($env:AI_EDA_PYTHON) { $env:AI_EDA_PYTHON } else { "3.12" }),
    [switch]$NoKicad,
    [switch]$WithClaude,
    [switch]$NoPath,
    [switch]$SkipSelftest
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is many times slower with the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$RepoSlug = "Yuchan-00/AI_EDA_Engineer"
$RepoUrl = "https://github.com/$RepoSlug"

function Say([string]$Text) { Write-Host "==> $Text" -ForegroundColor Cyan }
function Warn([string]$Text) { Write-Host "warning: $Text" -ForegroundColor Yellow }
# throw, never exit: under `irm ... | iex` an exit would close the user's PowerShell window
function Die([string]$Text) { throw "error: $Text" }
function Invoke-Step([string]$Exe, [string[]]$Arguments) {
    Write-Host "    + $Exe $($Arguments -join ' ')"
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe exited $LASTEXITCODE" }
}
function Test-Command([string]$Name) { return [bool](Get-Command $Name -ErrorAction SilentlyContinue) }
function Update-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$env:USERPROFILE\.local\bin;$user;$machine"
}
function Test-Checkout([string]$Path) {
    $toml = Join-Path $Path "pyproject.toml"
    return (Test-Path $toml) -and (Select-String -Path $toml -Pattern '^name = "ai-eda-engineer"' -Quiet)
}

$arch = $env:PROCESSOR_ARCHITECTURE
Say "system: Windows $([Environment]::OSVersion.Version) ($arch), PowerShell $($PSVersionTable.PSVersion)"

# --------------------------------------------------------------------------- 1. the code

$scriptDir = if ($PSCommandPath) { Split-Path -Parent $PSCommandPath } else { "" }
if ($scriptDir -and (Test-Checkout $scriptDir) -and -not $Dir) {
    $AppDir = $scriptDir
    Say "code: this checkout ($AppDir)"
} else {
    $AppDir = if ($Dir) { $Dir } else { Join-Path $env:USERPROFILE "AI_EDA_Engineer" }
    if (Test-Checkout $AppDir) {
        Say "code: existing copy in $AppDir"
        if ((Test-Path (Join-Path $AppDir ".git")) -and (Test-Command git)) {
            try { Invoke-Step git @("-C", $AppDir, "pull", "--ff-only") } catch { Warn "could not update $AppDir with git pull (local changes?); using it as it is" }
        }
    } elseif ((Test-Path $AppDir) -and (Get-ChildItem $AppDir -Force | Select-Object -First 1)) {
        Die "$AppDir exists and is not a copy of this project; choose another -Dir"
    } else {
        Say "code: downloading $RepoUrl ($Branch) into $AppDir"
        if (Test-Command git) {
            Invoke-Step git @("clone", "--depth", "1", "--branch", $Branch, "$RepoUrl.git", $AppDir)
        } else {
            $tmp = Join-Path ([IO.Path]::GetTempPath()) ("ai-eda-" + [Guid]::NewGuid().ToString("N"))
            New-Item -ItemType Directory -Path $tmp | Out-Null
            $zip = Join-Path $tmp "src.zip"
            Invoke-WebRequest -UseBasicParsing -Uri "https://codeload.github.com/$RepoSlug/zip/refs/heads/$Branch" -OutFile $zip
            Expand-Archive -Path $zip -DestinationPath $tmp
            $top = Get-ChildItem $tmp -Directory | Select-Object -First 1
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $AppDir) | Out-Null
            Move-Item $top.FullName $AppDir
            Remove-Item -Recurse -Force $tmp
        }
    }
}
if (-not (Test-Checkout $AppDir)) { Die "$AppDir does not hold the project" }

# --------------------------------------------------------------------------- 2. Python environment

Update-SessionPath
if (-not (Test-Command uv)) {
    Say "installing uv (https://docs.astral.sh/uv/) into %USERPROFILE%\.local\bin"
    $env:UV_NO_MODIFY_PATH = "1"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    Update-SessionPath
}
if (-not (Test-Command uv)) { Die "uv is not available after its installer ran" }
Say "Python $Python environment in $AppDir\.venv"
$venv = Join-Path $AppDir ".venv"
Invoke-Step uv @("venv", "--allow-existing", "--python", $Python, $venv)
$venvPython = Join-Path $venv "Scripts\python.exe"
Invoke-Step uv @("pip", "install", "--python", $venvPython, "-e", "$AppDir[llm,parts,dev]", "cryptography")

# --------------------------------------------------------------------------- 3. KiCad 10 (with its ngspice)

function Get-VersionKey([string]$Name) {
    $v = $null
    if ([version]::TryParse($Name, [ref]$v)) { return $v }
    return [version]"0.0"
}
function Find-KicadCli {
    # the same locations ai_eda.tools.kicad.install searches (per-user install first, then all users)
    $bases = @()
    if ($env:LOCALAPPDATA) { $bases += Join-Path $env:LOCALAPPDATA "Programs\KiCad" }
    if ($env:ProgramFiles) { $bases += Join-Path $env:ProgramFiles "KiCad" }
    foreach ($base in $bases) {
        if (Test-Path $base) {
            foreach ($ver in (Get-ChildItem $base -Directory | Sort-Object { Get-VersionKey $_.Name } -Descending)) {
                $exe = Join-Path $ver.FullName "bin\kicad-cli.exe"
                if (Test-Path $exe) { return $exe }
            }
        }
    }
    return $null
}
function Get-KicadMajor([string]$Cli) {
    if (-not $Cli) { return 0 }
    $text = (& $Cli version) | Select-Object -First 1
    if ($text -match '^(\d+)\.') { return [int]$Matches[1] }
    return 0
}

if ($NoKicad) {
    Say "KiCad skipped (-NoKicad)"
} else {
    $cli = Find-KicadCli
    if ((Get-KicadMajor $cli) -ge 10) {
        Say "KiCad $(& $cli version) is already installed ($cli)"
    } else {
        Say "KiCad 10 (KiCad brings kicad-cli, the libraries and ngspice)"
        if (Test-Command winget) {
            $verb = if ($cli) { "upgrade" } else { "install" }
            try {
                Invoke-Step winget @($verb, "--id", "KiCad.KiCad", "-e", "--silent", "--accept-package-agreements", "--accept-source-agreements", "--disable-interactivity")
            } catch { Warn "winget $verb KiCad.KiCad: $_" }
        } elseif (Test-Command choco) {
            try { Invoke-Step choco @("upgrade", "kicad", "-y", "--no-progress") } catch { Warn "choco upgrade kicad: $_" }
        } else {
            Die "neither winget nor Chocolatey is available to install KiCad 10: install 'App Installer' from the Microsoft Store (it provides winget) or KiCad 10 from https://www.kicad.org/download/windows/, then run this installer again"
        }
        $cli = Find-KicadCli
        if ((Get-KicadMajor $cli) -lt 10) {
            Die "KiCad 10 was not found after installing it (looked in %LOCALAPPDATA%\Programs\KiCad and %ProgramFiles%\KiCad); install it from https://www.kicad.org/download/windows/ and run this installer again"
        }
        Say "KiCad $(& $cli version) installed ($cli)"
    }
}

# --------------------------------------------------------------------------- 4. Claude Code CLI (optional)

if ($WithClaude) {
    if (Test-Command claude) {
        Say "Claude Code CLI already installed"
    } else {
        Say "Claude Code CLI (log in afterwards with: claude)"
        Invoke-RestMethod https://claude.ai/install.ps1 | Invoke-Expression
        Update-SessionPath
    }
}

# --------------------------------------------------------------------------- 5. ai-eda on PATH

$aiEda = Join-Path $venv "Scripts\ai-eda.exe"
if (-not $NoPath) {
    $bin = Join-Path $env:USERPROFILE ".local\bin"
    New-Item -ItemType Directory -Force -Path $bin | Out-Null
    # %USERPROFILE% keeps the shim ASCII when the user name is not (cmd.exe reads .cmd files in the OEM code page)
    $target = $aiEda
    if ($target.StartsWith($env:USERPROFILE, [StringComparison]::OrdinalIgnoreCase)) { $target = "%USERPROFILE%" + $target.Substring($env:USERPROFILE.Length) }
    $encoding = if ($target -match '^[\x00-\x7F]*$') { "ASCII" } else { "Default" }
    Set-Content -Path (Join-Path $bin "ai-eda.cmd") -Encoding $encoding -Value "@`"$target`" %*"
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not $userPath) { $userPath = "" }
    if (-not (($userPath -split ";") -contains $bin)) {
        [Environment]::SetEnvironmentVariable("Path", (($userPath.TrimEnd(";") + ";" + $bin).TrimStart(";")), "User")
        Say "added $bin to the user PATH"
    }
    Update-SessionPath
}

# --------------------------------------------------------------------------- 6. check

$env:PYTHONUTF8 = "1"
Say "ai-eda doctor"
& $aiEda doctor
$status = 0
if ($SkipSelftest) {
    Say "ai-eda selftest skipped (-SkipSelftest)"
} elseif ($NoKicad) {
    Say "ai-eda selftest skipped: without KiCad's libraries (-NoKicad) no circuit template can build"
} else {
    Say "ai-eda selftest (designs the 12 V -> 5 V divider end to end)"
    & $aiEda selftest
    $status = $LASTEXITCODE
}

Write-Host ""
Write-Host "AI EDA ENGINEER is installed in $AppDir"
Write-Host "  ai-eda doctor                     what was found"
Write-Host "  ai-eda gui --open                 the local web GUI (Korean)"
Write-Host "Open a new terminal so that 'ai-eda' is found on PATH."
if ($PSCommandPath) { exit $status }   # run as a file (install.cmd); under `irm | iex` the window stays open
if ($status -ne 0) { Write-Host "ai-eda selftest exited $status" -ForegroundColor Yellow }
