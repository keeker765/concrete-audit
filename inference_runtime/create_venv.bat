@echo off
setlocal

cd /d "%~dp0"

if exist ".venv" (
  echo [INFO] .venv already exists
) else (
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create venv
    pause
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements-web.txt

if errorlevel 1 (
  echo [ERROR] Failed to install runtime dependencies
  pause
  exit /b 1
)

echo [OK] venv is ready: %cd%\.venv
endlocal
