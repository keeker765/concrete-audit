@echo off
setlocal

cd /d "%~dp0"

set "HOST=127.0.0.1"
set "PORT=7860"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] venv python not found: %PYTHON_EXE%
  echo Please run create_venv.bat first.
  pause
  exit /b 1
)

echo Starting verify web in dev mode...
echo URL: http://%HOST%:%PORT%/
echo Hot reload: ON
echo.

"%PYTHON_EXE%" app\verify_web_onnx.py --host %HOST% --port %PORT% --debug

if errorlevel 1 (
  echo.
  echo [ERROR] Web app exited with failure.
  pause
  exit /b 1
)

endlocal
