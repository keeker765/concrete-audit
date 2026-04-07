@echo off
setlocal EnableExtensions
chcp 65001 >nul

REM ===== Config =====
set "PY=C:\Users\Lenovo\AppData\Local\Programs\Python\Python310\python.exe"
set "ROOT=%~dp0"

REM ===== Args (optional) =====
REM Usage:
REM   run_compare.bat [--limit N] [--methods sp,aliked] [--batch 1] [--specimen 1]

echo ============================================================
echo Ronghe Concrete Compare (offline)
echo   Root: %ROOT%
echo   Python: %PY%
echo ============================================================

if not exist "%PY%" (
  echo [ERROR] Python not found: %PY%
  echo 请确认 Python 3.10 安装路径，或修改本 bat 里的 PY=...
  exit /b 1
)

cd /d "%ROOT%" || (
  echo [ERROR] Cannot cd to %ROOT%
  exit /b 1
)

echo [1/3] Checking core imports...
"%PY%" -c "import cv2, torch; import ultralytics; print('OK: cv2/torch/ultralytics')" 1>nul 2>nul
if errorlevel 1 (
  echo [WARN] Missing deps. Trying to install...
  echo         pip install -U opencv-python numpy torch ultralytics
  echo         (If this fails, copy the error log to me.)
  "%PY%" -m pip install -U opencv-python numpy torch ultralytics
  if errorlevel 1 (
    echo [ERROR] pip install failed.
    exit /b 2
  )
)

echo [2/3] Running one-click CNN pipeline...
set "ONECLICK=%ROOT%daima\train_one_click_cnn.py"

if not exist "%ONECLICK%" (
  echo [ERROR] Not found: %ONECLICK%
  exit /b 3
)

REM Pass through any user args to the python script
"%PY%" "%ONECLICK%" %*

if errorlevel 1 (
  echo [ERROR] train_one_click_cnn.py failed. Please copy the error output to me.
  exit /b 3
)

echo [3/3] Done.
pause
