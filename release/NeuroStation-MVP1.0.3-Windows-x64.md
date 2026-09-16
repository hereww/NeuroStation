# NeuroStation MVP1.0.3 Windows x64

## 版本定位

- 产品版本：`MVP1.0.3`
- 产品系列：`MVP1.0`
- 语义版本：`1.0.3`
- 发布日期：`2026-09-16`
- 发布形式：Windows standalone portable build
- 目标系统：Windows 10/11 x64
- 基于提交：`36c98f6`（诊断、离线分析、回收站和跨平台打包修复）

## 本版本新增

- 新增透明、非破坏性的 SSVEP 离线 FFT 基线分析入口 `analyze_ssvep_session.py`，生成 `analysis.json` 和 `trial_features.tsv`；
- 新增原始数据列头说明文件 `raw_columns.tsv`，明确 BrainFlow row、单位、角色和含义；
- 新增目标标记审计字段，区分软件呈现目标、人工/自报目标和 EEG 离线推断目标；
- 新增工作台“诊断”页，支持依赖、配置、资源、保存目录和串口检查，查看事件并导出 JSON 报告；
- 新增持久化诊断事件日志，记录 UI、任务、硬件预检和 OpenBCI 导入过程；
- 新增基于真实采样数据的滚动 EEG 波形显示、通道标签、显示滤波和非有限值计数；
- 新增数据集回收站页面，支持恢复、永久删除和按名称/参与者/用户编号筛选；
- 新增协议中的采集开始和人工目标标记事件定义。

## 本版本修改

- 生产桌面入口统一使用真实 `DesktopGateway`，移除 UI 静默回退到模拟采集的路径；
- 当前生产采集入口只允许 OpenBCI Cyton 实机，CI 改为执行真实硬件入口保护和配置校验，不再生成 Synthetic/Demo EEG；
- Cyton 预检按包序号判断真实丢包，将可解释的主机时间戳抖动降级为 warning，并补充严重降级、重复包、平线和饱和通道指标；
- 重构 UI、网关、数据集和采集 worker 目录边界，清理 obsolete 代码并补齐诊断、离线分析和波形模块；
- 固定 BrainFlow 版本为 `5.22.2`，更新 Windows/macOS/Linux standalone 构建对原生库的打包策略；
- 更新 macOS CI 到 Intel runner，macOS packaged smoke 跳过不稳定的 worker/GUI 启动，仅保留安全诊断校验；
- 更新中英文 UI、README、验收文档和发布工作流到 `MVP1.0.3`。

## 本版本修复

- 修复活动数据集与回收站记录同名时，活动记录可能遮蔽已删除记录的问题；
- 修复干净的 Cyton 包序列在 USB/无线突发到达时反复弹出时间戳警告的问题；
- 修复 macOS 构建混入 Linux BrainFlow `.so` 文件的问题；
- 修复 macOS standalone 包缺少 BrainFlow `.dylib` 或嵌入式 Python runtime 的问题；
- 修复打包诊断在 Windows 非 UTF-8 控制台下输出本地化 JSON 失败的问题；
- 修复 Windows 路径大小写/短路径表示变化可能泄露到桌面结果和数据集记录的问题；
- 修复发布 smoke 在无硬件环境中错误制造模拟 EEG 的问题，并改为只验证打包资源和诊断入口。

## 使用方式

解压后运行：

```powershell
.\NeuroStation.dist\workstation.exe
.\NeuroStation.dist\workstation.exe --diagnostics
```

生产采集模式固定为 OpenBCI Cyton 实机。没有连接硬件时只能查看历史记录、导入 OpenBCI 文件或运行诊断，不会生成模拟 EEG。真实采集前，请复核通道配置、参考/BIAS、电极连接、显示器刷新率和光学时序，并在设备页运行硬件预检。

离线分析只读取已保存会话并写入派生文件，不修改原始 EEG；EEG 频率推断不能证明参与者注视位置，也不能替代眼动、photodiode/TTL 和现场硬件验收。

## 验收状态

- 核心与集成测试：以本次发布构建日志为准；
- 独立 UI 测试：以本次发布构建日志为准；
- UI scale matrix smoke（100%/125%/150%）：通过；
- Windows standalone packaged smoke：通过；
- SBOM 生成与包内版本文件校验：通过；
- OpenBCI Cyton 实机、photodiode/TTL、长时间稳定性和医疗用途验收：未包含在自动化发布中。

本版本是科研与教学技术验证软件，不是医疗诊断设备。
