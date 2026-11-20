# Shared functions for CircuitScope's start/stop/test scripts.
# Dot-sourced only; executes nothing itself. ASCII only (PowerShell 5.1 reads
# BOM-less files as ANSI). Local paths and plain messages throughout.

# Returns the live command line of a process, or $null if it is not running
# (or the command line cannot be read).
function Get-LiveCommandLine {
    param([int]$ProcessId)
    try {
        $proc = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction Stop
    } catch {
        return $null
    }
    if ($null -eq $proc) { return $null }
    return [string]$proc.CommandLine
}

# True when the live command line matches the recorded one after
# normalization (quotes stripped, whitespace collapsed, case-insensitive).
function Test-CommandLineMatch {
    param([string]$Live, [string]$Expected)
    if ([string]::IsNullOrWhiteSpace($Live) -or [string]::IsNullOrWhiteSpace($Expected)) { return $false }
    function Normalize([string]$s) {
        $t = $s.Replace('"', ' ')
        $t = ($t -split '\s+') -join ' '
        return $t.Trim().ToLowerInvariant()
    }
    return (Normalize $Live) -eq (Normalize $Expected)
}

# True when something is accepting TCP connections on this port on either
# loopback stack (IPv4 127.0.0.1 or IPv6 ::1). A connect() probe: unlike a
# bind() test it does not false-positive on TIME_WAIT sockets left by a
# just-stopped server. A server that only listens on ::1 counts as busy.
function Test-PortBusy {
    param([int]$Port)
    foreach ($target in @(@("127.0.0.1", [System.Net.Sockets.AddressFamily]::InterNetwork),
                         @("::1", [System.Net.Sockets.AddressFamily]::InterNetworkV6))) {
        $client = New-Object System.Net.Sockets.TcpClient($target[1])
        try {
            $task = $client.ConnectAsync($target[0], $Port)
            if ($task.Wait(1000) -and $client.Connected) { return $true }
        } catch {
            # family not supported or connect refused: not busy on this stack
        } finally {
            $client.Dispose()
        }
    }
    return $false
}

# Resolve the Python interpreter to use for helpers: prefer the project venv,
# fall back to whatever `python` is on PATH (runlib.py is standard-library
# only, so any Python 3 works for it).
function Get-HelperPython {
    param([string]$RepoRoot)
    $venvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPy) { return $venvPy }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $cmd) { return $cmd.Source }
    return $null
}
