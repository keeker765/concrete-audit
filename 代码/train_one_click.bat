@echo off
chcp 65001 >nul
echo ========================================
echo   sample 批量比对工具
echo ========================================
echo.
echo 数据源: C:\Users\Lenovo\Desktop\1\sample
echo 规则: A1对B1, A2对B2, A3对B3, 跳过所有C开头文件
echo.
cd /d "%~dp0"
if exist "%~dp0python\python.exe" (
    "%~dp0python\python.exe" "%~dp0train_one_click_cnn.py"
) else (
    C:\Users\Lenovo\miniconda3\python.exe "%~dp0train_one_click_cnn.py"
)
pause
