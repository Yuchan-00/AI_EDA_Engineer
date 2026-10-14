# AI EDA ENGINEER - one-command installer for Windows 10 / 11.
#
#   powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/Yuchan-00/AI_EDA_Engineer/main/install.ps1 | iex"
#   install.cmd [options]                     (from a checkout of the repository; same options as below)
#
# On a machine with nothing set up it installs, without asking (Windows shows its own UAC prompt where an
# installer needs administrator rights):
#   1. the code   - this checkout, else the repository into %USERPROFILE%\AI_EDA_Engineer (git clone; without git,
#                   or when the clone fails, the branch's zip, unpacked in a folder beside the target so that it is
#                   moved on one drive)
#   2. Python     - uv (Astral) into %USERPROFILE%\.local\bin, a uv-managed Python 3.12 and .venv with the package
#   3. KiCad 10   - with winget (KiCad.KiCad), and with Chocolatey when winget is missing, fails or leaves no
#                   KiCad 10; KiCad brings kicad-cli, the libraries and ngspice (bin\ngspice.dll)
#   4. a browser  - none needed: Microsoft Edge prints the PDF copies of the stage reports
#   5. ai-eda     - %USERPROFILE%\.local\bin\ai-eda.cmd on the user PATH, KiCad's global library tables
#                   (`ai-eda setup-kicad`, what KiCad's first start writes; also with -NoKicad when a KiCad is
#                   already installed), then `ai-eda doctor` and `ai-eda selftest` (run only when setup-kicad
#                   found a KiCad and its tables are in place)
# Re-running it updates the code (git checkouts) and repairs whatever is missing.
#
# Options (environment variables in brackets, for `irm ... | iex`, which passes no options; a switch's variable
# is set to 1 - true and yes count too -, and either the switch or its variable turns it on):
#   -Dir DIR        where to put the code when not run from a checkout [AI_EDA_DIR] (default
#                   %USERPROFILE%\AI_EDA_Engineer; a relative DIR is taken from the current folder)
#   -Branch NAME    branch to download [AI_EDA_BRANCH] (default main)
#   -Python VER     Python version for the virtual environment [AI_EDA_PYTHON] (default 3.12)
#   -NoKicad        do not install KiCad (without a KiCad the pipeline reports those checks NOT_VERIFIED)
#                   [AI_EDA_NO_KICAD=1]
#   -WithClaude     also install the Claude Code CLI (`--llm claude`; log in with `claude` afterwards)
#                   [AI_EDA_WITH_CLAUDE=1]
#   -NoPath         do not add ai-eda to the user PATH [AI_EDA_NO_PATH=1]
#   -SkipSelftest   do not run `ai-eda selftest` at the end [AI_EDA_SKIP_SELFTEST=1]
# For example, in Windows PowerShell:
#   $env:AI_EDA_DIR = "D:\eda"; $env:AI_EDA_NO_KICAD = "1"; irm https://raw.githubusercontent.com/Yuchan-00/AI_EDA_Engineer/main/install.ps1 | iex
#
# The same text runs as a file (install.cmd: powershell -File) and under `irm ... | iex` in Windows PowerShell
# 5.1, where no parameter is bound, $PSCommandPath is empty and the text runs in the caller's own scope. So it
# stays ASCII (5.1 reads a file without a BOM in the ANSI code page), uses nothing newer than PowerShell 5.1,
# throws instead of exiting (an exit would close the user's window), keeps its other variables, functions and
# preferences in a child scope (`& { ... }` below), and runs the uv / Claude Code installers in a child
# powershell process, so that their own `exit`, parameters and strict mode stay there. Under `irm | iex` the
# param block's variables ($Dir, $Branch, $Python, $NoKicad, $WithClaude, $NoPath, $SkipSelftest) are set in
# the caller's scope, as any variable that text assigns there: a session variable of one of these names is
# replaced.
param(
    [string]$Dir = $env:AI_EDA_DIR,
    [string]$Branch = $(if ($env:AI_EDA_BRANCH) { $env:AI_EDA_BRANCH } else { "main" }),
    [string]$Python = $(if ($env:AI_EDA_PYTHON) { $env:AI_EDA_PYTHON } else { "3.12" }),
    [switch]$NoKicad,
    [switch]$WithClaude,
    [switch]$NoPath,
    [switch]$SkipSelftest
)

& {
    Set-StrictMode -Off
    $ErrorActionPreference = "Stop"
    $ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is many times slower with the progress bar
    $PSNativeCommandUseErrorActionPreference = $false   # PowerShell 7.3+: exit codes are read below, never thrown
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    $RepoSlug = "Yuchan-00/AI_EDA_Engineer"
    $RepoUrl = "https://github.com/$RepoSlug"
    $LocalBin = Join-Path $env:USERPROFILE ".local\bin"

    function Say([string]$Text) { Write-Host "==> $Text" -ForegroundColor Cyan }
    function Warn([string]$Text) { Write-Host "warning: $Text" -ForegroundColor Yellow }
    # throw, never exit: under `irm ... | iex` an exit would close the user's PowerShell window
    function Die([string]$Text) { throw "error: $Text" }
    function Test-Flag([string]$Value) { return [bool]($Value -match '^(1|true|yes)$') }   # as install.sh's is_on
    function Test-Ascii([string]$Text) { return [bool]($Text -match '^[\x00-\x7F]*$') }
    function Invoke-Step([string]$Exe, [string[]]$Arguments) {
        Write-Host "    + $Exe $($Arguments -join ' ')"
        & $Exe @Arguments
        if ($LASTEXITCODE -ne 0) { throw "$Exe exited $LASTEXITCODE" }
    }
    function Test-Command([string]$Name) { return [bool](Get-Command $Name -ErrorAction SilentlyContinue) }
    function Update-SessionPath {
        # this session's PATH with %USERPROFILE%\.local\bin first and the registry's user and machine entries
        # (what installers just added) appended where missing; never replaced, because under `irm | iex` it is the
        # user's own window, whose activated environments must survive
        $entries = @($LocalBin) + ($env:Path -split ";") +
            ([string][Environment]::GetEnvironmentVariable("Path", "User") -split ";") +
            ([string][Environment]::GetEnvironmentVariable("Path", "Machine") -split ";")
        $seen = @{}
        $kept = @()
        foreach ($entry in $entries) {
            if (-not $entry) { continue }
            $key = $entry.TrimEnd('\')
            if ($seen.ContainsKey($key)) { continue }
            $seen[$key] = $true
            $kept += $entry
        }
        $env:Path = $kept -join ";"
    }
    function Resolve-FullPath([string]$Path) {
        # a relative path taken from the current folder, as git and uv would, so that what is written down (the
        # ai-eda.cmd shim) names an absolute path
        $full = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
        if ($full.Length -gt ([IO.Path]::GetPathRoot($full)).Length) { $full = $full.TrimEnd([char[]]@('\', '/')) }
        return $full
    }
    function Test-Checkout([string]$Path) {
        if (-not $Path) { return $false }
        $toml = Join-Path $Path "pyproject.toml"
        return [bool]((Test-Path -LiteralPath $toml -PathType Leaf) -and (Select-String -LiteralPath $toml -Pattern '^name = "ai-eda-engineer"' -Quiet))
    }
    function Copy-RepositoryZip([string]$Target) {
        # Without git: the branch's zip, unpacked in a new folder beside $Target - on the same drive, because
        # Windows PowerShell 5.1 cannot move a folder from %TEMP% to another drive - and moved into place. An
        # existing (empty) $Target gets the files; the downloaded folder is never moved into it as a subfolder.
        $parent = [IO.Path]::GetDirectoryName($Target)
        if (-not $parent) { $parent = $Target }   # a drive root, which the caller lets through only when empty
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
        $stage = Join-Path $parent ([IO.Path]::GetFileName($Target) + ".download-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
        New-Item -ItemType Directory -Path $stage | Out-Null
        try {
            $url = "https://codeload.github.com/$RepoSlug/zip/refs/heads/$Branch"
            Write-Host "    + download $url"
            $zip = Join-Path $stage "src.zip"
            Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $zip
            $tree = Join-Path $stage "x"
            Expand-Archive -LiteralPath $zip -DestinationPath $tree
            $top = Get-ChildItem -LiteralPath $tree -Directory | Select-Object -First 1
            if (-not $top) { Die "the zip from $url holds no folder" }
            if (Test-Path -LiteralPath $Target) {
                Get-ChildItem -LiteralPath $top.FullName -Force | Move-Item -Destination $Target
            } else {
                Move-Item -LiteralPath $top.FullName -Destination $Target
            }
        } finally {
            Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    function Get-WindowsPowerShell {
        # Windows PowerShell 5.1, which every Windows 10 / 11 has; from a 32-bit host through Sysnative, so that an
        # installer sees the 64-bit system (the Claude Code installer refuses a 32-bit process)
        $folder = "System32"
        if ([Environment]::Is64BitOperatingSystem -and -not [Environment]::Is64BitProcess) { $folder = "Sysnative" }
        $exe = Join-Path $env:SystemRoot "$folder\WindowsPowerShell\v1.0\powershell.exe"
        if (Test-Path -LiteralPath $exe -PathType Leaf) { return $exe }
        return "powershell.exe"
    }
    function Invoke-WebInstaller([string]$Url, [hashtable]$Environment) {
        # A third-party `irm <url> | iex` installer, run in its own powershell process with $Environment added to
        # its environment: run here, its `exit` would end this installer (or, under `irm | iex`, close the user's
        # window) and its param block and strict mode would land in this scope. Throws when it exits non-zero.
        $saved = @{}
        foreach ($name in @($Environment.Keys)) {
            $saved[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
            [Environment]::SetEnvironmentVariable($name, [string]$Environment[$name], "Process")
        }
        try {
            $command = "[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12; Invoke-RestMethod '$Url' | Invoke-Expression"
            Write-Host "    + powershell -NoProfile -ExecutionPolicy Bypass -Command `"irm $Url | iex`""
            & (Get-WindowsPowerShell) -NoProfile -ExecutionPolicy Bypass -Command $command
            if ($LASTEXITCODE -ne 0) { throw "the installer from $Url exited $LASTEXITCODE" }
        } finally {
            foreach ($name in @($saved.Keys)) { [Environment]::SetEnvironmentVariable($name, $saved[$name], "Process") }
        }
    }

    # a switch is on when it is given or when its environment variable is (the only way under `irm | iex`)
    $NoKicad = [bool]$NoKicad -or (Test-Flag $env:AI_EDA_NO_KICAD)
    $WithClaude = [bool]$WithClaude -or (Test-Flag $env:AI_EDA_WITH_CLAUDE)
    $NoPath = [bool]$NoPath -or (Test-Flag $env:AI_EDA_NO_PATH)
    $SkipSelftest = [bool]$SkipSelftest -or (Test-Flag $env:AI_EDA_SKIP_SELFTEST)

    $arch = $env:PROCESSOR_ARCHITECTURE
    if ($env:PROCESSOR_ARCHITEW6432) { $arch = "$env:PROCESSOR_ARCHITEW6432, a 32-bit PowerShell" }
    Say "system: Windows $([Environment]::OSVersion.Version) ($arch), PowerShell $($PSVersionTable.PSVersion)"
    Update-SessionPath

    # ----------------------------------------------------------------------- 1. the code

    $scriptDir = if ($PSCommandPath) { Split-Path -Parent $PSCommandPath } else { "" }
    if ($scriptDir -and -not $Dir -and (Test-Checkout $scriptDir)) {
        $AppDir = $scriptDir
        Say "code: this checkout ($AppDir)"
    } else {
        $AppDir = if ($Dir) { Resolve-FullPath $Dir } else { Join-Path $env:USERPROFILE "AI_EDA_Engineer" }
        if (Test-Checkout $AppDir) {
            Say "code: existing copy in $AppDir"
            if ((Test-Path -LiteralPath (Join-Path $AppDir ".git")) -and (Test-Command git)) {
                try { Invoke-Step git @("-C", $AppDir, "pull", "--ff-only") } catch { Warn "could not update $AppDir with git pull (local changes?); using it as it is" }
            }
        } elseif ((Test-Path -LiteralPath $AppDir) -and (Get-ChildItem -LiteralPath $AppDir -Force | Select-Object -First 1)) {
            Die "$AppDir exists and is not a copy of this project; choose another -Dir"
        } else {
            Say "code: downloading $RepoUrl ($Branch) into $AppDir"
            $cloned = $false
            if (Test-Command git) {
                try {
                    Invoke-Step git @("clone", "--depth", "1", "--branch", $Branch, "$RepoUrl.git", $AppDir)
                    $cloned = $true
                } catch { Warn "git clone failed ($_): downloading the zip instead" }
            }
            if (-not $cloned) {
                if ((Test-Path -LiteralPath $AppDir) -and (Get-ChildItem -LiteralPath $AppDir -Force | Select-Object -First 1)) {
                    Die "$AppDir is not empty (left by the failed git clone?); remove it and run this installer again"
                }
                Copy-RepositoryZip $AppDir
            }
        }
    }
    if (-not (Test-Checkout $AppDir)) { Die "$AppDir does not hold the project" }

    # ----------------------------------------------------------------------- 2. Python environment

    if (-not (Test-Command uv)) {
        Say "installing uv (https://docs.astral.sh/uv/) into $LocalBin"
        try {
            Invoke-WebInstaller "https://astral.sh/uv/install.ps1" @{ UV_INSTALL_DIR = $LocalBin; UV_NO_MODIFY_PATH = "1" }
        } catch { Warn "$_" }   # what counts is whether uv is there now
        Update-SessionPath
    }
    if (-not (Test-Command uv)) {
        Die "uv is not available after its installer ran (looked on PATH and in $LocalBin); install uv as https://docs.astral.sh/uv/getting-started/installation/ says and run this installer again"
    }
    Say "Python $Python environment in $AppDir\.venv"
    $venv = Join-Path $AppDir ".venv"
    Invoke-Step uv @("venv", "--allow-existing", "--python", $Python, $venv)
    $venvPython = Join-Path $venv "Scripts\python.exe"
    Invoke-Step uv @("pip", "install", "--python", $venvPython, "-e", "$AppDir[llm,parts,dev]", "cryptography")

    # ----------------------------------------------------------------------- 3. KiCad 10 (with its ngspice)

    function Get-KicadBases {
        # where KiCad's own installer and winget put KiCad (the folders ai_eda.tools.kicad.install searches): the
        # per-user install, then all users - %ProgramW6432% before %ProgramFiles%, which a 32-bit host sees as
        # "Program Files (x86)", then C:\Program Files when neither variable names it -, each only from a variable
        # holding an absolute path, each folder once
        $bases = @()
        $seen = @{}
        foreach ($place in @(@{ Root = $env:LOCALAPPDATA; Sub = "Programs\KiCad" }, @{ Root = $env:ProgramW6432; Sub = "KiCad" }, @{ Root = $env:ProgramFiles; Sub = "KiCad" }, @{ Root = "C:\Program Files"; Sub = "KiCad" })) {
            if (-not ([string]$place.Root -match '^([A-Za-z]:[\\/]|[\\/][\\/][^\\/])')) { continue }
            $base = Join-Path $place.Root $place.Sub
            $key = $base.TrimEnd('\')
            if ($seen.ContainsKey($key)) { continue }
            $seen[$key] = $true
            $bases += $base
        }
        return $bases
    }
    function Get-VersionKey([string]$Name) {
        # 10.0 > 9.0 (sorting the names as text gets this wrong); a folder that is no version sorts last
        $v = $null
        if ($Name -match '^\d+(\.\d+){0,3}$') {
            $text = if ($Name.Contains(".")) { $Name } else { "$Name.0" }
            if ([version]::TryParse($text, [ref]$v)) { return $v }
        }
        return [version]"0.0"
    }
    function Find-KicadCli {
        # the newest version over all the bases together (a per-user KiCad 9 must not hide an all-users KiCad 10);
        # on equal versions the earlier base, the per-user install, wins
        $found = @()
        $order = 0
        foreach ($base in @(Get-KicadBases)) {
            if (Test-Path -LiteralPath $base -PathType Container) {
                foreach ($ver in @(Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue)) {
                    $exe = Join-Path $ver.FullName "bin\kicad-cli.exe"
                    if (Test-Path -LiteralPath $exe -PathType Leaf) {
                        $found += [pscustomobject]@{ Exe = $exe; Version = (Get-VersionKey $ver.Name); Order = $order }
                    }
                }
            }
            $order++
        }
        $best = $found | Sort-Object -Property @{ Expression = { $_.Version }; Descending = $true }, @{ Expression = { $_.Order }; Descending = $false } | Select-Object -First 1
        if ($best) { return $best.Exe }
        return $null
    }
    function Get-KicadMajor([string]$Cli) {
        if (-not $Cli) { return 0 }
        try { $text = (& $Cli version) | Select-Object -First 1 } catch { return 0 }   # a kicad-cli that cannot start
        if ([string]$text -match '^(\d+)\.') { return [int]$Matches[1] }
        return 0
    }

    if ($NoKicad) {
        Say "KiCad skipped (-NoKicad / AI_EDA_NO_KICAD)"
    } else {
        $cli = Find-KicadCli
        if ((Get-KicadMajor $cli) -ge 10) {
            Say "KiCad $(& $cli version) is already installed ($cli)"
        } else {
            Say "KiCad 10 (KiCad brings kicad-cli, the libraries and ngspice)"
            # winget first; Chocolatey when winget is missing, fails (App Installer disabled by policy, no network
            # to its source) or leaves no KiCad 10
            $managers = @()
            if (Test-Command winget) {
                $managers += "winget"
                $verbs = if ($cli) { @("upgrade", "install") } else { @("install") }
                foreach ($verb in $verbs) {
                    try {
                        Invoke-Step winget @($verb, "--id", "KiCad.KiCad", "-e", "--silent", "--accept-package-agreements", "--accept-source-agreements", "--disable-interactivity")
                    } catch { Warn "winget $verb KiCad.KiCad: $_" }
                    $cli = Find-KicadCli
                    if ((Get-KicadMajor $cli) -ge 10) { break }
                }
            }
            if (((Get-KicadMajor $cli) -lt 10) -and (Test-Command choco)) {
                $managers += "Chocolatey"
                try { Invoke-Step choco @("upgrade", "kicad", "-y", "--no-progress") } catch { Warn "choco upgrade kicad: $_" }
                $cli = Find-KicadCli
            }
            if ($managers.Count -eq 0) {
                Die "neither winget nor Chocolatey is available to install KiCad 10: install 'App Installer' from the Microsoft Store (it provides winget) or KiCad 10 from https://www.kicad.org/download/windows/, then run this installer again"
            }
            if ((Get-KicadMajor $cli) -lt 10) {
                Die "KiCad 10 was not found after trying $($managers -join ' and ') (looked in $((Get-KicadBases) -join ', ')); install it from https://www.kicad.org/download/windows/ and run this installer again"
            }
            Say "KiCad $(& $cli version) installed ($cli)"
        }
        # ai-eda takes a kicad-cli on PATH before the installed ones (ai_eda.tools.kicad.install)
        $onPath = Get-Command kicad-cli -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($onPath -and ((Get-KicadMajor $onPath.Path) -lt 10)) {
            Warn "the kicad-cli on PATH ($($onPath.Path)) is older than KiCad 10 and ai-eda would use it before $cli; take its folder off PATH"
        }
    }

    # ----------------------------------------------------------------------- 4. Claude Code CLI (optional)

    if ($WithClaude) {
        if (Test-Command claude) {
            Say "Claude Code CLI already installed"
        } else {
            Say "Claude Code CLI (log in afterwards with: claude)"
            try {
                Invoke-WebInstaller "https://claude.ai/install.ps1" @{}
            } catch {
                Warn "the Claude Code CLI was not installed ($_); ai-eda works without it, only --llm claude needs it (later: irm https://claude.ai/install.ps1 | iex)"
            }
            Update-SessionPath
        }
    }

    # ----------------------------------------------------------------------- 5. ai-eda on PATH

    function ConvertTo-CmdPath([string]$Path) {
        # %USERPROFILE% for the profile folder (cmd.exe expands it in Unicode, so a non-ASCII user name stays out of
        # the file), % doubled elsewhere so that cmd.exe reads it literally
        $profileDir = $env:USERPROFILE.TrimEnd('\')
        if ($Path.StartsWith($profileDir + "\", [StringComparison]::OrdinalIgnoreCase)) {
            return "%USERPROFILE%" + $Path.Substring($profileDir.Length).Replace("%", "%%")
        }
        return $Path.Replace("%", "%%")
    }
    function Get-OemEncoding {
        # cmd.exe reads a .cmd file in its console's code page, which starts as the system's OEM code page (system
        # locale: 437 / 850 on Western systems, whose ANSI code page is 1252); the current culture's as a fallback
        $codePage = 0
        try {
            $codePage = [int](Get-ItemProperty -LiteralPath "HKLM:\SYSTEM\CurrentControlSet\Control\Nls\CodePage" -Name OEMCP).OEMCP
        } catch { $codePage = 0 }
        if ($codePage -le 0) { $codePage = [Globalization.CultureInfo]::CurrentCulture.TextInfo.OEMCodePage }
        try { return [Text.Encoding]::GetEncoding($codePage) } catch { return [Text.Encoding]::ASCII }
    }
    function Write-Shim([string]$Shim, [string]$Exe) {
        # `@"<exe>" %*` in the OEM code page, as bytes (no byte order mark, which cmd.exe would read as text);
        # a path still not ASCII after %USERPROFILE% takes its 8.3 alias where the volume has one
        $target = ConvertTo-CmdPath $Exe
        if (-not (Test-Ascii $target)) {
            $short = $null
            try { $short = (New-Object -ComObject Scripting.FileSystemObject).GetFile($Exe).ShortPath } catch { $short = $null }
            if ($short -and (Test-Ascii $short)) { $target = ConvertTo-CmdPath $short }
        }
        $text = "@`"$target`" %*`r`n"
        $encoding = Get-OemEncoding
        $bytes = $encoding.GetBytes($text)
        if ($encoding.GetString($bytes) -cne $text) {
            Warn "$Exe cannot be written in this system's OEM code page $($encoding.CodePage), so $Shim may not find it; reinstall with -Dir naming a folder whose path is ASCII"
        }
        [IO.File]::WriteAllBytes($Shim, $bytes)
    }
    function Add-UserPath([string]$Folder) {
        # HKCU\Environment\Path read as stored (its %VAR% entries, such as %USERPROFILE%\...\WindowsApps, stay
        # unexpanded), $Folder appended (as %USERPROFILE%\...) when no entry names it, raw or expanded, written
        # back as REG_EXPAND_SZ, and the change broadcast (WM_SETTINGCHANGE) to new terminals; $true when added
        $entry = $Folder
        $profileDir = $env:USERPROFILE.TrimEnd('\')
        if ($Folder.StartsWith($profileDir + "\", [StringComparison]::OrdinalIgnoreCase)) { $entry = "%USERPROFILE%" + $Folder.Substring($profileDir.Length) }
        $want = $Folder.TrimEnd('\')
        $key = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey("Environment")
        try {
            $raw = [string]$key.GetValue("Path", "", [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
            foreach ($item in ($raw -split ";")) {
                if (-not $item) { continue }
                if (($item.TrimEnd('\') -ieq $entry) -or ([Environment]::ExpandEnvironmentVariables($item).TrimEnd('\') -ieq $want)) { return $false }
            }
            $kept = $raw.TrimEnd(";")
            $value = if ($kept) { "$kept;$entry" } else { $entry }
            $key.SetValue("Path", $value, [Microsoft.Win32.RegistryValueKind]::ExpandString)
        } finally {
            $key.Close()
        }
        # deleting a variable that is not there changes nothing but still sends WM_SETTINGCHANGE
        [Environment]::SetEnvironmentVariable("AI_EDA_INSTALLER_PATH_REFRESH", $null, "User")
        return $true
    }

    $aiEda = Join-Path $venv "Scripts\ai-eda.exe"
    if ($NoPath) {
        Say "ai-eda not put on PATH (-NoPath / AI_EDA_NO_PATH): $aiEda"
    } else {
        New-Item -ItemType Directory -Force -Path $LocalBin | Out-Null
        Write-Shim (Join-Path $LocalBin "ai-eda.cmd") $aiEda
        try {
            if (Add-UserPath $LocalBin) { Say "added $LocalBin to the user PATH" }
        } catch {
            Warn "could not add $LocalBin to the user PATH ($_); add it yourself, or run $aiEda"
        }
        Update-SessionPath
    }

    # ----------------------------------------------------------------------- 6. check

    $status = 0
    $savedUtf8 = $env:PYTHONUTF8
    $env:PYTHONUTF8 = "1"
    try {
        # what KiCad's first start does: the global library tables, without which ERC / DRC warn about every part.
        # Also with -NoKicad, for a KiCad installed some other way; an existing table is never changed.
        Say "KiCad's global library tables (ai-eda setup-kicad)"
        $kicadReady = $false
        & $aiEda setup-kicad
        if ($LASTEXITCODE -eq 0) {
            $kicadReady = $true
        } elseif (-not $NoKicad) {
            Warn "ai-eda setup-kicad did not find a usable KiCad or could not write its library tables (ai-eda doctor below says what is missing)"
        } else {
            Say "no usable KiCad (-NoKicad): after installing KiCad 10, run ai-eda setup-kicad"
        }
        Say "ai-eda doctor"
        & $aiEda doctor
        # the KiCad this run was to install is not usable: the installation did not do what it was asked
        if ((-not $kicadReady) -and (-not $NoKicad)) { $status = 1 }
        if ($SkipSelftest) {
            Say "ai-eda selftest skipped (-SkipSelftest / AI_EDA_SKIP_SELFTEST)"
        } elseif ($kicadReady) {
            Say "ai-eda selftest (designs the 12 V -> 5 V divider end to end)"
            & $aiEda selftest
            $status = $LASTEXITCODE
        } elseif (-not $NoKicad) {
            Warn "ai-eda selftest not run: KiCad is not ready (ai-eda setup-kicad failed above)"
        } else {
            Say "ai-eda selftest skipped: no KiCad with its library tables (-NoKicad); after installing KiCad 10 run: ai-eda setup-kicad, then ai-eda selftest"
        }
    } finally {
        $env:PYTHONUTF8 = $savedUtf8
    }

    Write-Host ""
    Write-Host "AI EDA ENGINEER is installed in $AppDir"
    Write-Host "  ai-eda doctor                     what was found"
    Write-Host "  ai-eda gui --open                 the local web GUI (Korean)"
    if (-not $NoPath) { Write-Host "Open a new terminal so that 'ai-eda' is found on PATH." }
    if ($PSCommandPath) { exit $status }   # run as a file (install.cmd): the selftest's status is the exit code
    # under `irm | iex` the window stays open; the status is left in $LASTEXITCODE for a calling script
    if ($status -ne 0) { Write-Host "the installation check failed (exit status $status)" -ForegroundColor Yellow }
    $global:LASTEXITCODE = $status
}
