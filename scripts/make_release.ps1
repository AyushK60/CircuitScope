# Package the committed source tree into a release zip.
#
# Usage (from the repo root): powershell -NoProfile -ExecutionPolicy Bypass -File scripts\make_release.ps1
#
# Runs `git archive HEAD` -> release\CircuitScope-source.zip (tracked files
# only, no working-tree noise, no generated data). Prints the zip's size and
# file count. release\ is git-ignored; the zip is never committed.

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Work from a clean HEAD check: refuse when the tree is dirty, so the zip
# always matches a commit (git archive HEAD ignores uncommitted edits).
$dirty = git status --porcelain
if ($dirty) {
    Write-Host "[release] ERROR: the working tree is dirty; commit (or stash) first so the zip matches a real commit:" -ForegroundColor Red
    $dirty | Select-Object -First 10 | ForEach-Object { Write-Host "  $_" }
    exit 1
}

$ReleaseDir = Join-Path $RepoRoot "release"
New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null
$ZipPath = Join-Path $ReleaseDir "CircuitScope-source.zip"
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }

git archive --format=zip --output $ZipPath HEAD
if ($LASTEXITCODE -ne 0) {
    Write-Host "[release] ERROR: git archive failed (exit $LASTEXITCODE)." -ForegroundColor Red
    exit 1
}

$sizeBytes = (Get-Item $ZipPath).Length
$sizeMb = [Math]::Round($sizeBytes / 1MB, 2)
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipView = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
try { $fileCount = $zipView.Entries.Count } finally { $zipView.Dispose() }

Write-Host "[release] wrote $ZipPath"
Write-Host "[release] commit: $(git rev-parse --short HEAD)"
Write-Host "[release] size: $sizeMb MB ($sizeBytes bytes)"
Write-Host "[release] files: $fileCount"
Write-Host "[release] note: release\ is git-ignored; do not commit the zip."
