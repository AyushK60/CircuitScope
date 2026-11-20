@echo off
rem Starts the CircuitScope backend and frontend, waits for real readiness,
rem then prints the app URL (and opens the browser unless you pass -NoBrowser).
rem PIDs are recorded to outputs\run\pids.json; stop everything with stop.bat.
rem Preview without starting anything: start.bat -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1" %*
