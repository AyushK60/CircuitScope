# Start the CircuitScope stack (backend + frontend dev server) as child
# processes this script owns, wait for real readiness, then report the URL.
#
# Usage (from the repo root):
#   .\start.bat                          (wrapper for this script)
#   scripts\start.ps1 -NoBrowser         (do not open the default browser)
#   scripts\start.ps1 -DryRun            (print the plan, start nothing)
#
# Preflight fails with the exact setup step to run when something is missing.
# PIDs it started are written to outputs\run\pids.json for scripts\stop.ps1.
# It never kills anything and never starts two backends on port 8000.

param(
    [switch]$NoBrowser,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "common.ps1")

$BackendPort = 8000
$FrontendRangeStart = 5173
$FrontendRangeEnd = 5180
$RunDir = Join-Path $RepoRoot "outputs\run"
$PidsPath = Join-Path $RunDir "pids.json"

function Say([string]$msg) { Write-Host "[start] $msg" }
function Die([string]$msg) {
    Write-Host "[start] ERROR: $msg" -ForegroundColor Red
    exit 1
}

# --- Preflight (read-only; identical for DryRun and real runs) ---------------
$VenvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    Die "Missing .venv. Run  scripts\setup.bat  first (it creates .venv and installs everything)."
}
$NodeModules = Join-Path $RepoRoot "frontend\node_modules"
if (-not (Test-Path (Join-Path $NodeModules "vite"))) {
    Die "Missing frontend\node_modules (no vite). Run  scripts\setup.bat  first (it runs npm ci)."
}
$ViteJs = Join-Path $NodeModules "vite\bin\vite.js"
if (-not (Test-Path $ViteJs)) {
    Die "frontend\node_modules exists but vite\bin\vite.js is missing; run  npm ci  in frontend\ (scripts\setup.bat does it)."
}
$NodeExe = (Get-Command node -ErrorAction Stop).Source

Say "checking model cache..."
& $VenvPy (Join-Path $PSScriptRoot "download_model.py") --check
if ($LASTEXITCODE -ne 0) {
    Die "GPT-2 weights are not downloaded yet. Run  scripts\setup.bat  (or  .venv\Scripts\python scripts\download_model.py) first."
}

if (Test-PortBusy $BackendPort) {
    $health = $null
    try { $health = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/api/health" -TimeoutSec 3 } catch {}
    if ($null -ne $health -and $health.status -eq "ok") {
        Die "A backend is already answering on port $BackendPort. If CircuitScope started it, run  scripts\stop.ps1  first; if you started it yourself in another terminal, stop it there."
    }
    Die "Port $BackendPort is in use by another program that is not a CircuitScope backend. Free the port (or stop that program) and retry; this script never kills processes it did not start."
}

$FrontendPortRaw = & $VenvPy (Join-Path $PSScriptRoot "runlib.py") choose-port $FrontendRangeStart $FrontendRangeEnd
if ($LASTEXITCODE -ne 0) {
    Die "All frontend ports $FrontendRangeStart-$FrontendRangeEnd are busy (on IPv4 or IPv6). Free one of them and retry."
}
$FrontendPort = [int]$FrontendPortRaw
$BackendUrl = "http://127.0.0.1:$BackendPort"
$FrontendUrl = "http://localhost:$FrontendPort"
$PyArg = "-m backend.app.main"
$NodeArg = "$ViteJs --port $FrontendPort --strictPort"

if ($DryRun) {
    Say "[dryrun] preflight passed - this is the plan, nothing was started:"
    Write-Host "  backend : $VenvPy $PyArg"
    Write-Host "            -> $BackendUrl/api/health  (logs: outputs\run\backend.out.log, backend.err.log)"
    Write-Host "  frontend: $NodeExe $NodeArg"
    Write-Host "            -> $FrontendUrl  (logs: outputs\run\frontend.out.log, frontend.err.log)"
    Write-Host "  pids    : $PidsPath"
    Write-Host "  browser : $(if ($NoBrowser) { 'not opened (-NoBrowser)' } else { 'would open ' + $FrontendUrl })"
    Write-Host ""
    Write-Host "Next step: run  .\start.bat  to actually start the stack; stop with  .\stop.bat"
    exit 0
}

# --- Real start --------------------------------------------------------------
# Children inherit the caller's pipe handles on Windows (a piped invocation
# would otherwise stay open until the stack is stopped). Clearing the inherit
# flag on our own std handles prevents that; if Add-Type is unavailable, this
# only degrades to the harmless leak described above.
try {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class CsmHandles {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool SetHandleInformation(IntPtr h, int mask, int flags);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr GetStdHandle(int which);
}
'@ -ErrorAction Stop
    # STD_INPUT/OUTPUT/ERROR_HANDLE = -10/-11/-12 (Win32); HANDLE_FLAG_INHERIT = 1
    foreach ($which in @(-10, -11, -12)) {
        $h = [CsmHandles]::GetStdHandle($which)
        if ($h -ne [IntPtr]::Zero -and $h -ne [IntPtr](-1)) {
            [void][CsmHandles]::SetHandleInformation($h, 1, 0)
        }
    }
} catch {
    Write-Host "[start] note: could not mark std handles non-inheritable ($($_.Exception.Message)); piped output may stay open until the stack is stopped."
}

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$BackendOut = Join-Path $RunDir "backend.out.log"
$BackendErr = Join-Path $RunDir "backend.err.log"
$FrontendOut = Join-Path $RunDir "frontend.out.log"
$FrontendErr = Join-Path $RunDir "frontend.err.log"

Say "starting backend: $VenvPy $PyArg"
$BackendProc = Start-Process -FilePath $VenvPy -ArgumentList @("-m", "backend.app.main") `
    -WorkingDirectory $RepoRoot -WindowStyle Hidden `
    -RedirectStandardOutput $BackendOut -RedirectStandardError $BackendErr -PassThru

Say "starting frontend: $NodeExe $NodeArg"
$FrontendProc = Start-Process -FilePath $NodeExe -ArgumentList @($ViteJs, "--port", "$FrontendPort", "--strictPort") `
    -WorkingDirectory (Join-Path $RepoRoot "frontend") -WindowStyle Hidden `
    -RedirectStandardOutput $FrontendOut -RedirectStandardError $FrontendErr -PassThru

# Record what we started BEFORE waiting, so scripts\stop.ps1 can always stop it.
$Record = [ordered]@{
    started_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ", [System.Globalization.CultureInfo]::InvariantCulture)
    repo_root   = $RepoRoot
    backend     = [ordered]@{
        pid          = $BackendProc.Id
        command_line = "$VenvPy $PyArg"
        url          = $BackendUrl
        port         = $BackendPort
        log          = "outputs\run\backend.out.log"
    }
    frontend    = [ordered]@{
        pid          = $FrontendProc.Id
        command_line = "$NodeExe $NodeArg"
        url          = $FrontendUrl
        port         = $FrontendPort
        log          = "outputs\run\frontend.out.log"
    }
}
$Json = $Record | ConvertTo-Json -Depth 5
# BOM-less UTF-8: PowerShell 5.1's `-Encoding UTF8` writes a BOM, which trips
# some JSON readers (e.g. node's JSON.parse).
[System.IO.File]::WriteAllText($PidsPath, $Json, (New-Object System.Text.UTF8Encoding($false)))
Say "recorded PIDs in $PidsPath (backend $($BackendProc.Id), frontend $($FrontendProc.Id))"

# --- Wait for real readiness -------------------------------------------------
$BudgetSeconds = 120
$StopAt = (Get-Date).AddSeconds($BudgetSeconds)
$Healthy = $false
while ((Get-Date) -lt $StopAt) {
    $elapsed = [int]($BudgetSeconds - (($StopAt - (Get-Date)).TotalSeconds))
    Write-Host "[start] waiting for the backend ($BackendUrl/api/health)... $elapsed s / $BudgetSeconds s"
    try {
        $health = Invoke-RestMethod -Uri "$BackendUrl/api/health" -TimeoutSec 8
        if ($health.status -eq "ok" -and $health.model_loaded) { $Healthy = $true; break }
        if ($health.status -eq "ok" -and -not $health.model_loaded) {
            Die "Backend answered but the model failed to load. See $BackendErr and $BackendOut, then retry."
        }
    } catch {}
    Start-Sleep -Seconds 5
}
if (-not $Healthy) {
    Die "Backend did not become healthy within $BudgetSeconds s (the first request loads the model). See $BackendOut and $BackendErr. The frontend may still be running; use  scripts\stop.ps1  to stop everything."
}
Say "backend healthy and model loaded."

$FrontendOk = $false
$FrontendStopAt = (Get-Date).AddSeconds(60)
while ((Get-Date) -lt $FrontendStopAt) {
    try {
        $null = Invoke-WebRequest -Uri $FrontendUrl -UseBasicParsing -TimeoutSec 3
        $FrontendOk = $true; break
    } catch {}
    Start-Sleep -Seconds 2
}
if (-not $FrontendOk) {
    Die "Frontend did not answer at $FrontendUrl within 60 s. See $FrontendOut and $FrontendErr. The backend is still running; use  scripts\stop.ps1  to stop everything."
}

Write-Host ""
Write-Host "CircuitScope is running:"
Write-Host "  App:         $FrontendUrl"
Write-Host "  API docs:    $BackendUrl/docs"
Write-Host "  Stop with:   .\stop.bat"
Write-Host ""
if ($NoBrowser) {
    Say "-NoBrowser: not opening a browser window."
} else {
    Say "opening $FrontendUrl in the default browser..."
    Start-Process $FrontendUrl
}
