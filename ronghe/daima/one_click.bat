@echo off
chcp 65001 >nul
echo ========================================
echo   二维码透视校正 + 双图比较工具
echo ========================================
echo.
echo 会弹出文件选择框，请一次选择两张图片。
echo.
cd /d "%~dp0"
if exist "%~dp0python\python.exe" (
    "%~dp0python\python.exe" "%~dp0one_click_cnn.py"
) else (
    C:\Users\Lenovo\miniconda3\python.exe "%~dp0one_click_cnn.py"
)
pause
