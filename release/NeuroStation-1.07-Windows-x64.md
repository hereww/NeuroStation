# NeuroStation 1.07 Windows x64

- Product version: `1.07`
- Semantic version: `1.0.7`
- Build date: `2026-11-21`
- Intended release tag: `v1.0.7`

## 本版更新

- 「设备 → 头部通道定位与实时信号」新增「通道解读」，说明 CH1–CH8 当前电极位置、常见观察用途及相关伪迹。
- 点击头图或切换当前通道会高亮对应说明，重新分配电极后头图与解读同步更新。
- 解读支持简体中文和英文，面板根据窗口宽度并排或换行。

## 脑电处理模块构建（2026-11-26）

工作台左侧新增“脑电处理”。选择四频 SSVEP 原始采集场次后，可导出免校准逐窗判定，也可用本人较早场次校准并导出两套结果。结果文件夹包含判定表、参数、实验协议和字段说明。有实验标记时会附上原始记录，供外部评估。实时入口打开采集设置，采集期间可在任务页查看连续判定。运行 `NeuroStation.dist\Start-EEG-Processing.cmd` 可直达模块。

## 本机 GPU 实时模型（2026-11-27）

在左侧「模型」检查显卡、导入兼容的四频 ONNX 模型，并选择「用于 SSVEP 采集」。在「应用 → SSVEP」也可以直接选模型。勾选实时判定后，内置 Continuous CCA v3 使用本机 GPU 计算 CCA 投影和频谱。任务页和采集监视窗显示实际模型、版本和显卡。双选原始脑电和实时判定会在同一场次保存可按样本编号对齐的两份数据，模型信息写入 `session.json`。

## 使用

解压 `NeuroStation-1.07-Windows-x64.zip`，运行 `NeuroStation.dist\workstation.exe`。
