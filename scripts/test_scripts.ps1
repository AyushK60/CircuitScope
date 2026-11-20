# Pester-free checks for the CircuitScope PowerShell scripts (setup/start/stop)
# and the pure helpers they use. Runs only read-only checks, DryRun modes, and
# processes/port-listeners IT STARTS ITSELF (always cleaned up). Exits non-zero
# if any check fails.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\test_scripts.ps1
# (Pure unit tests for runlib.py live in backend\tests\test_runlib.py, pytest.)

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "common.ps1")

$script:Failures = 0
$script:Checks = 0
$VenvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$PidsPath = Join-Path $RepoRoot "outputs\run\pids.json"
$HelperPy = if (Test-Path $VenvPy) { $VenvPy } else { (Get-Command python -ErrorAction SilentlyContinue).Source }

function Check {
    param([string]$Name, [bool]$Ok, [string]$Detail = "")
    $script:Checks++
    if ($Ok) {
        Write-Host ("  PASS  {0}{1}" -f $Name, $(if ($Detail) { "  ($Detail)" }))
    } else {
        $script:Failures++
        Write-Host ("  FAIL  {0}{1}" -f $Name, $(if ($Detail) { "  ($Detail)" })) -ForegroundColor Red
    }
}
function Skip {
    param([string]$Name, [string]$Why)
    Write-Host ("  SKIP  {0}  ({1})" -f $Name, $Why) -ForegroundColor Yellow
}

Write-Host "== 1. runlib.py CLI (helper python: $HelperPy) =="
if ($null -eq $HelperPy) {
    Skip "runlib CLI checks" "no python found"
} else {
    # 1a. choose-port prints a bindable port inside the range
    $port = & $HelperPy (Join-Path $PSScriptRoot "runlib.py") choose-port 5173 5180
    Check "runlib choose-port returns a value" ($LASTEXITCODE -eq 0 -and $port -ge 5173 -and $port -le 5180) "port=$port"
    Check "runlib chosen port is really free (connect probe)" (-not (Test-PortBusy ([int]$port)))

    # 1b. port-status reports busy for a listener we start, free after we close it
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    $busyPort = ($listener.LocalEndpoint).Port
    $status = & $HelperPy (Join-Path $PSScriptRoot "runlib.py") port-status $busyPort
    Check "runlib port-status: our listener is 'busy'" ($LASTEXITCODE -eq 4 -and $status -match "$busyPort busy") "out=$status"
    $listener.Stop()
    $status = & $HelperPy (Join-Path $PSScriptRoot "runlib.py") port-status $busyPort
    Check "runlib port-status: closed listener is 'free'" ($LASTEXITCODE -eq 0 -and $status -match "$busyPort free") "out=$status"

    # 1b2. a port held only on IPv6 must also count as busy (regression:
    # another project held [::1]:5173 while IPv4 5173 was free, and vite
    # failed to bind; choose-port/port-status must not hand out such ports)
    $listener6 = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::IPv6Loopback, 0)
    $listener6.Start()
    $busyPort6 = ($listener6.LocalEndpoint).Port
    $status6 = & $HelperPy (Join-Path $PSScriptRoot "runlib.py") port-status $busyPort6
    Check "runlib port-status: IPv6-only listener is 'busy'" ($LASTEXITCODE -eq 4 -and $status6 -match "$busyPort6 busy") "out=$status6"
    $chosen6 = & $HelperPy (Join-Path $PSScriptRoot "runlib.py") choose-port $busyPort6 $busyPort6
    Check "runlib choose-port refuses an IPv6-held port" ($LASTEXITCODE -eq 3) "out=$chosen6"
    $listener6.Stop()

    # 1c. load-pids rejects malformed JSON, accepts a valid record
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("csm_test_" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    $bad = Join-Path $tmp "bad.json"
    Set-Content -Path $bad -Value "{ not json" -Encoding UTF8
    & $HelperPy (Join-Path $PSScriptRoot "runlib.py") load-pids $bad | Out-Null
    Check "runlib load-pids: malformed JSON rejected (exit 5)" ($LASTEXITCODE -eq 5)
    $good = Join-Path $tmp "good.json"
    Set-Content -Path $good -Value '{"started_utc":"2026-01-01T00:00:00Z","repo_root":"E:\\x","backend":{"pid":111,"command_line":"python -m backend.app.main","url":"http://127.0.0.1:8000","port":8000},"frontend":{"pid":222,"command_line":"node vite.js --port 5173 --strictPort","url":"http://localhost:5173","port":5173}}' -Encoding UTF8
    $valid = & $HelperPy (Join-Path $PSScriptRoot "runlib.py") load-pids $good
    Check "runlib load-pids: valid record accepted (exit 0)" ($LASTEXITCODE -eq 0 -and $null -ne ($valid | ConvertFrom-Json).backend.pid)
    $missing = Join-Path $tmp "nope.json"
    & $HelperPy (Join-Path $PSScriptRoot "runlib.py") load-pids $missing | Out-Null
    Check "runlib load-pids: missing file rejected (exit 5)" ($LASTEXITCODE -eq 5)
    Remove-Item -Recurse -Force $tmp
}

Write-Host "== 2. setup.ps1 -DryRun =="
$dry = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "setup.ps1") -DryRun 2>&1
Check "setup -DryRun exits 0" ($LASTEXITCODE -eq 0) "exit=$LASTEXITCODE"
Check "setup -DryRun prints dryrun markers" (($dry | Out-String) -match "\[dryrun\]")
Check "setup -DryRun reports the python it found" (($dry | Out-String) -match "Python 3\.11")
Check "setup -DryRun reports the node it found" (($dry | Out-String) -match "node: v")

Write-Host "== 3. start.ps1 -DryRun =="
$pidsHashBefore = $null
if (Test-Path $PidsPath) { $pidsHashBefore = (Get-FileHash $PidsPath).Hash }
if (Test-PortBusy 8000) {
    Skip "start -DryRun" "port 8000 is busy (a stack is running?); this check needs a free 8000"
} else {
    $dry = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "start.ps1") -DryRun 2>&1
    Check "start -DryRun exits 0" ($LASTEXITCODE -eq 0) "exit=$LASTEXITCODE"
    $dryText = $dry | Out-String
    Check "start -DryRun prints the plan" ($dryText -match "\[dryrun\] preflight passed")
    Check "start -DryRun names the backend command" ($dryText -match "backend\.app\.main")
    Check "start -DryRun passes --strictPort" ($dryText -match "strictPort")
    $pidsHashAfter = $null
    if (Test-Path $PidsPath) { $pidsHashAfter = (Get-FileHash $PidsPath).Hash }
    Check "start -DryRun leaves pids.json untouched" ($pidsHashBefore -eq $pidsHashAfter)
}

Write-Host "== 4. stop.ps1 with no usable PID record =="
$stackRunning = $false
if ((Test-Path $PidsPath) -and ($null -ne $HelperPy)) {
    $rec = (& $HelperPy (Join-Path $PSScriptRoot "runlib.py") load-pids $PidsPath 2>$null | ConvertFrom-Json)
    if ($null -ne $rec) {
        $live = Get-LiveCommandLine -ProcessId ([int]$rec.backend.pid)
        if ($null -ne $live -and (Test-CommandLineMatch -Live $live -Expected ([string]$rec.backend.command_line))) { $stackRunning = $true }
    }
}
if ($stackRunning) {
    Skip "stop with no record" "a started stack is running; not touching its PID record"
} else {
    $backup = $null
    if (Test-Path $PidsPath) { $backup = $PidsPath + ".bak"; Move-Item $PidsPath $backup -Force }
    $out = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "stop.ps1") 2>&1
    $outText = $out | Out-String
    Check "stop without record reports it" ($outText -match "nothing recorded to stop")
    Check "stop without record exits 0 when port 8000 free" ($LASTEXITCODE -eq 0 -or ($LASTEXITCODE -eq 1 -and $outText -match "left untouched")) "exit=$LASTEXITCODE"
    if ($null -ne $backup) { Move-Item $backup $PidsPath -Force }
}

Write-Host "== 5. live process command-line verification (own sleeper process) =="
$sleeper = Start-Process powershell -ArgumentList @("-NoProfile", "-Command", "Start-Sleep -Seconds 90") -WindowStyle Hidden -PassThru
try {
    $live = Get-LiveCommandLine -ProcessId $sleeper.Id
    Check "Get-LiveCommandLine returns a command line for a live process" (-not [string]::IsNullOrWhiteSpace($live)) $live
    Check "Test-CommandLineMatch accepts the live line" (Test-CommandLineMatch -Live $live -Expected $live)
    Check "Test-CommandLineMatch is quote/spacing insensitive" (Test-CommandLineMatch -Live $live -Expected ('"  ' + $live + '   "'))
    Check "Test-CommandLineMatch rejects a foreign command line" (-not (Test-CommandLineMatch -Live $live -Expected "notepad.exe unrelated"))
} finally {
    Stop-Process -Id $sleeper.Id -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Milliseconds 500
Check "Get-LiveCommandLine returns null for a stopped process" ($null -eq (Get-LiveCommandLine -ProcessId $sleeper.Id))

Write-Host "== 6. Test-PortBusy probe =="
$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
$listener.Start()
$probePort = ($listener.LocalEndpoint).Port
Check "Test-PortBusy sees our listener as busy" (Test-PortBusy $probePort)
$listener.Stop()
Start-Sleep -Milliseconds 200
Check "Test-PortBusy sees the closed port as free" (-not (Test-PortBusy $probePort))
$listener6 = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::IPv6Loopback, 0)
$listener6.Start()
$probePort6 = ($listener6.LocalEndpoint).Port
Check "Test-PortBusy sees an IPv6-only listener as busy" (Test-PortBusy $probePort6)
$listener6.Stop()

Write-Host "== 7. download_model.py --check (report only, no download/load) =="
if ($null -eq $HelperPy) {
    Skip "download_model --check" "no python found"
} else {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $out = & $HelperPy (Join-Path $PSScriptRoot "download_model.py") --check 2>&1
    $sw.Stop()
    $outText = $out | Out-String
    if ($LASTEXITCODE -eq 0) {
        Check "download_model --check exits 0 and reports cached" ($outText -match "cached") ("{0:n1}s" -f $sw.Elapsed.TotalSeconds)
    } else {
        Check "download_model --check: model not cached -> run scripts\setup.bat first" $false ("exit $LASTEXITCODE")
    }
    Check "download_model --check is fast (inspects disk only, <15 s)" ($sw.Elapsed.TotalSeconds -lt 15) ("{0:n1}s" -f $sw.Elapsed.TotalSeconds)
}

Write-Host ""
Write-Host "== summary: $($script:Checks - $script:Failures) of $script:Checks checks passed =="
if ($script:Failures -gt 0) {
    Write-Host "test_scripts: FAILED ($script:Failures failing checks)" -ForegroundColor Red
    exit 1
}
Write-Host "test_scripts: all checks passed"
exit 0
