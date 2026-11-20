@echo off
rem CircuitScope setup: checks prerequisites, installs everything local, downloads the model.
rem Safe to rerun. Use "setup.bat -WithE2E" to also install the Playwright browser,
rem "setup.bat -DryRun" to preview every step without changing anything.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" %*
