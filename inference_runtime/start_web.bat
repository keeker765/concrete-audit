@echo off
setlocal

cd /d "%~dp0"

set "HOST=127.0.0.1"
set "PORT=7860"
set "PYTHON_CMD=py -3.10"

%PYTHON_CMD% -c "import sys; print(sys.executable)" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python 3.10 not available on this machine.
  pause
  exit /b 1
)

%PYTHON_CMD% -c "import flask, torch, ultralytics, cv2, numpy" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Local Python 3.10 is missing required packages.
  echo Please install: flask torch ultralytics opencv-contrib-python numpy
  pause
  exit /b 1
)

echo Starting verify web...
echo URL: http://%HOST%:%PORT%/
for /f "delims=" %%i in ('%PYTHON_CMD% -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%i"
echo Python: %PYTHON_EXE%
echo.

%PYTHON_CMD% app\verify_web_onnx.py --host %HOST% --port %PORT%

if errorlevel 1 (
  echo.
  echo [ERROR] Web app exited with failure.
  pause
  exit /b 1
)

endlocal
