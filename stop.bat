@echo off
rem Stops exactly the CircuitScope processes recorded by start.bat (PID record in
rem outputs\run\pids.json, command line verified before each kill), then confirms
rem the ports are free. Never kills by port or by name.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop.ps1" %*
