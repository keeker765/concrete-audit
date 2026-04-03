封装目录说明
====================

启动文件：
- one_click.bat

核心文件：
- one_click_cnn.py
- best(1).pt

当前 bat 运行规则：
1. 优先使用目录内的 python\python.exe
2. 如果没有，则回退到 C:\Users\Lenovo\miniconda3\python.exe

如需独立封装到别的电脑：
- 建议把可运行的 Python 放进本目录下的 python 文件夹
- 并确保安装依赖：opencv-python、numpy、torch、ultralytics

输出目录：
- compare_outputs\时间戳目录

train_one_click_cnn为批量处理，手动在代码更改目录，目录
one_click_cnn为单独两张图比对

