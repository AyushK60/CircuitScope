# Stop the CircuitScope stack started by scripts\start.ps1.
#
# Safety rules:
# - It stops ONLY the PIDs recorded in outputs\run\pids.json, and only after
#   verifying each PID's live command line still matches what was recorded.
#   It never kills by port or by process name, and never touches a process it
#   did not start.
# - After stopping, it confirms the backend and frontend ports are actually
#   free (a connect() probe, not a bind() test) and reports the result.
#
# Usage (from the repo root):  .\stop.bat   (or: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stop.ps1)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "common.ps1")

$PidsPath = Join-Path $RepoRoot "outputs\run\pids.json"

function Say([string]$msg) { Write-Host "[stop] $msg" }

if (-not (Test-Path $PidsPath)) {
    Say "no PID record at outputs\run\pids.json - nothing recorded to stop."
    Say "(Was the stack started with scripts\start.bat? This script never kills processes it did not start.)"
    if (Test-PortBusy 8000) {
        Say "note: port 8000 is busy by a process this script did not start; it was left untouched."
        exit 1
    }
    Say "port 8000 is free."
    exit 0
}

# Validate the record through runlib when a Python is available; fall back to a
# plain JSON read otherwise (runlib is the tested authority on the file shape).
$HelperPy = Get-HelperPython $RepoRoot
$Record = $null
if ($null -ne $HelperPy) {
    $out = & $HelperPy (Join-Path $PSScriptRoot "runlib.py") load-pids $PidsPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[stop] ERROR: outputs\run\pids.json is invalid (see message above); nothing was stopped." -ForegroundColor Red
        exit 1
    }
    $Record = $out | ConvertFrom-Json
} else {
    $Record = Get-Content $PidsPath -Raw | ConvertFrom-Json
}

$Stopped = 0
foreach ($name in @("backend", "frontend")) {
    $entry = $Record.$name
    $ProcId = [int]$entry.pid
    $recorded = [string]$entry.command_line
    $live = Get-LiveCommandLine -ProcessId $ProcId
    if ($null -eq $live) {
        Say "$name (pid $ProcId) is not running anymore - nothing to stop."
        continue
    }
    if (-not (Test-CommandLineMatch -Live $live -Expected $recorded)) {
        Write-Host "[stop] ERROR: refusing to stop pid $ProcId for '$name':" -ForegroundColor Red
        Write-Host "  recorded: $recorded"
        Write-Host "  live:     $live"
        Write-Host "The live command line does not match the record, so this is probably not the process we started. Nothing was killed."
        exit 1
    }
    Say "stopping $name (pid $ProcId): $recorded"
    & taskkill /PID $ProcId /T /F | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[stop] ERROR: taskkill failed for pid $ProcId (exit $LASTEXITCODE)." -ForegroundColor Red
        exit 1
    }
    $Stopped++
}
if ($Stopped -eq 0) { Say "nothing was running; PID record kept for reference." }

# Confirm the ports are actually free now.
$BackendPort = [int]$Record.backend.port
$FrontendPort = [int]$Record.frontend.port
$AllFree = $true
foreach ($p in @($BackendPort, $FrontendPort)) {
    if (Test-PortBusy $p) {
        Say "port $p is STILL busy (something is listening). If it is not a CircuitScope process, this script leaves it alone."
        $AllFree = $false
    } else {
        Say "port $p is free."
    }
}
if (-not $AllFree) { exit 1 }
Say "done: CircuitScope stopped, ports $BackendPort and $FrontendPort free."
