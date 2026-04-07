# inference_runtime

这个目录用于单独打包“推理网站”。

目标：

1. `matcher` 使用独立目录内的原 `.pt` 模型副本推理
2. 网站代码与启动脚本单独收口到这里
3. 预处理逻辑按项目 `README.md` 主链路改编
4. 训练仍然留在原来的 `conda` 环境，网站推理可单独走 `venv`

说明：

- `models/concrete_matcher.pt` 是原单模型副本
- `models/concrete_seg_best.pt` 是分割模型副本
- 当前网站仍然沿用 `YOLO Seg .pt` 做分割，所以网站环境里仍需要 `torch`
- 二维码检测按 `src.data.build_sample_aligned_masks` 中的微信二维码逻辑改编
- 当前不再使用 ONNX 版本 matcher
- `app/concrete_match_model.py` 是打包用的本地模型定义副本，不再依赖外部 `src`

打包给别人时，直接带走整个 `inference_runtime` 目录即可，不再依赖主项目外层的模型文件。

启动方式：

- 正常模式：`start_web.bat`
- 开发热重载模式：`start_web_dev.bat`
- 这里只是先把 `matcher` 从 PyTorch 剥离成 ONNX
