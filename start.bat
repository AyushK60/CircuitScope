@echo off
rem Starts the CircuitScope backend (port 8000) and frontend (port 5173) in two windows.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Create it first: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
  exit /b 1
)
start "CircuitScope backend" cmd /k ".venv\Scripts\python -m backend.app.main"
start "CircuitScope frontend" cmd /k "cd frontend && npm run dev"
echo.
echo Backend:  http://127.0.0.1:8000/docs
echo Frontend: use the URL printed in the frontend window (normally http://localhost:5173)
echo If port 5173 is busy, Vite picks the next port; the backend accepts any local port.
