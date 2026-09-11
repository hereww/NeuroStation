# NeuroStation MVP1.0 Windows x64

## 产品描述

NeuroStation 是面向科研与教学技术验证的脑电采集工作站。本版本以 Windows 10/11 x64 为主要交付平台，提供中文优先的 PySide6 桌面界面、SSVEP 四目标采集流程、BrainFlow Synthetic 合成采集、OpenBCI Cyton 实机接入入口、会话元数据、质量报告、数据集预览，以及 OpenBCI GUI 中文集成工作区。

## 版本定位

- 产品版本：`MVP1.0`
- 语义版本：`1.0.0`
- 构建日期：`2026-09-11`
- 发布形式：Windows standalone portable build
- 目标系统：Windows 10/11 x64
- 默认模式：流程演示，不连接真实硬件

## 功能范围

- 中文优先的设备、采集应用、会话/数据集和集成中心界面；
- SSVEP 四目标 10/12/15/20 Hz 协议参数与试次流程；
- 默认流程演示，仅保存会话元数据，不伪造 EEG 样本；
- BrainFlow Synthetic 模式，生成可验收的原始数据、事件、质量报告和 manifest；
- OpenBCI Cyton 模式，支持 `AUTO` 串口发现、BrainFlow 握手诊断和 CH1-CH8 自动映射；
- OpenBCI GUI 源码锁定、中文 overlay 和 Windows runtime 集成状态展示；
- 中文路径数据集、会话摘要、原始数据预览和配置副本追踪。

## 启动

直接运行包内的 `NeuroStation.dist\workstation.exe`。也可以在 PowerShell 中执行：

```powershell
.\NeuroStation.dist\workstation.exe
.\NeuroStation.dist\workstation.exe --diagnostics
```

`--diagnostics` 只做依赖、配置、资源和串口端点预检，不会打开 Cyton 串口。真实硬件采集前，应先审核通道位置、参考电极、BIAS、显示器刷新率和光学时序。

## 交付边界

本版本是 Windows 无硬件功能 MVP，不是医疗诊断设备，也不能替代 Cyton 实机、photodiode/TTL 光学时序、显示器刷新率和安装器人工验收。SSVEP 与通道配置仍保留 `DRAFT` 标记；Synthetic、流程演示和视觉预览必须按模拟数据解释。

## 验收状态

本包已通过 Windows standalone 启动、诊断、BrainFlow Synthetic worker、中文路径会话、质量报告和 manifest smoke test。OpenBCI Cyton 的真实连接仍需现场设备和接线复核。
