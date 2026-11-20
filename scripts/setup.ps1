# CircuitScope setup: check prerequisites, create the venv, install Python and
# Node dependencies, optionally the Playwright browser, then the model weights.
# Idempotent: safe to rerun; nothing global, no admin rights needed.
#
# Usage (from the repo root):
#   scripts\setup.bat            (or: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1)
#   scripts\setup.ps1 -WithE2E   (also installs the Playwright Chromium browser)
#   scripts\setup.ps1 -DryRun    (print every step without doing it)

param(
    [switch]$WithE2E,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Say([string]$msg) { Write-Host "[setup] $msg" }
function Die([string]$msg) {
    Write-Host "[setup] ERROR: $msg" -ForegroundColor Red
    exit 1
}

# --- Find a suitable Python 3.11 (the pinned dependency stack is built for it) ---
$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"
$PyExe = $null       # full path or launcher command prefix
$PyLabel = $null

if (Test-Path $VenvPy) {
    $ver = & $VenvPy --version 2>&1
    Say "found venv python: $ver ($VenvPy)"
    if ($ver -notmatch "Python 3\.11\.") {
        Die "The existing .venv is $ver but this project requires Python 3.11. Delete .venv and rerun setup with a Python 3.11 available."
    }
    $PyExe = $VenvPy
    $PyLabel = $VenvPy
} else {
    Say "no .venv yet; looking for a Python 3.11 interpreter..."
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        $ver = & python --version 2>&1
        Say "found 'python' on PATH: $ver"
        if ($ver -match "Python 3\.11\.") { $PyExe = $cmd.Source; $PyLabel = "$($cmd.Source) (PATH)" }
    }
    if ($null -eq $PyExe) {
        $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($null -ne $pyLauncher) {
            $ver = & py -3.11 --version 2>&1
            Say "found 'py -3.11': $ver"
            if ($ver -match "Python 3\.11\.") { $PyExe = "@py -3.11"; $PyLabel = "py -3.11 (launcher)" }
        }
    }
    if ($null -eq $PyExe) {
        $found = if ($null -ne $cmd) { & python --version 2>&1 } else { "none" }
        Die "Python 3.11 is required (found: $found). Install Python 3.11 from python.org (or 'winget install Python.Python.3.11'), then rerun scripts\setup.bat."
    }
    Say "will use Python: $PyLabel"
}

# --- Check Node 20+ ---
$NodeCmd = Get-Command node -ErrorAction SilentlyContinue
if ($null -eq $NodeCmd) {
    Die "Node.js 20 or newer is required (found: none). Install the Node LTS from nodejs.org, then rerun scripts\setup.bat."
}
$NodeVer = & node --version 2>&1
$NodeMajor = if ($NodeVer -match "^v(\d+)\.") { [int]$Matches[1] } else { 0 }
Say "found node: $NodeVer"
if ($NodeMajor -lt 20) {
    Die "Node.js 20 or newer is required (found: $NodeVer). Install the Node LTS from nodejs.org, then rerun scripts\setup.bat."
}

# --- Steps (skipped as 'done' when already satisfied; DryRun prints only) ---
function Invoke-Step {
    param([string]$What, [scriptblock]$Action, [string]$DoneCheck)
    if ($null -ne $DoneCheck -and (Invoke-Command -ScriptBlock ([scriptblock]::Create($DoneCheck)))) {
        Say "skip (already done): $What"
        return
    }
    if ($DryRun) { Say "[dryrun] would: $What"; return }
    Say $What
    & $Action
    if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) {
        Die "$What failed (exit $LASTEXITCODE). Fix the problem above and rerun scripts\setup.bat - it is safe to rerun."
    }
}

$FrontendDir = Join-Path $RepoRoot "frontend"
$NpmCmd = Get-Command npm -ErrorAction SilentlyContinue
if ($null -eq $NpmCmd) { Die "npm was not found even though node $NodeVer is installed; check your Node installation." }

Invoke-Step "create .venv with $PyLabel" `
    -Action { if ($PyExe -eq "@py -3.11") { py -3.11 -m venv $VenvDir } else { & $PyExe -m venv $VenvDir } } `
    -DoneCheck "Test-Path '$VenvPy'"

Invoke-Step "pip install -r requirements.txt (into .venv; may take a few minutes)" `
    -Action { & $VenvPy -m pip install -r (Join-Path $RepoRoot "requirements.txt") }

Invoke-Step "npm ci in frontend\ (clean install from package-lock.json; may take a few minutes)" `
    -Action { Set-Location $FrontendDir; npm ci; Set-Location $RepoRoot } `
    -DoneCheck "(Test-Path (Join-Path '$FrontendDir' 'node_modules\vite')) -and (Test-Path (Join-Path '$FrontendDir' 'package-lock.json'))"

if ($WithE2E) {
    Invoke-Step "npx playwright install chromium (browser for the e2e tests)" `
        -Action { Set-Location $FrontendDir; npx playwright install chromium; Set-Location $RepoRoot }
} else {
    Say "skipping Playwright browser (pass -WithE2E to install it; needed only for the e2e test suite)"
}

Invoke-Step "ensure GPT-2 weights are in the local Hugging Face cache (downloads on first run; skipped if cached)" `
    -Action { & $VenvPy (Join-Path $RepoRoot "scripts\download_model.py") }

Say "setup complete."
if ($DryRun) { Say "[dryrun] nothing was changed - this was a preview only." }
Write-Host ""
Write-Host "Next step: run  .\start.bat  from the repo root  (stop later with  .\stop.bat)"
Write-Host "Backend API docs will be at http://127.0.0.1:8000/docs ; the app opens in your browser automatically."
