# NeuroStation MVP1.0.2 Windows x64

## 版本定位

- 产品版本：`MVP1.0.2`
- 产品系列：`MVP1.0`
- 语义版本：`1.0.2`
- 发布日期：`2026-09-13`
- 发布形式：Windows standalone portable build
- 目标系统：Windows 10/11 x64
- 基于提交：`8e01f8c`（Cyton 预检、质量元数据和 UI scale smoke）

## 本版本新增

- 新增 OpenBCI Cyton 硬件预检入口，执行握手、短时采样、时间戳连续性和通道平线检查；
- 新增正式 Cyton 采集前的设备预检阻断流程，预检失败时不会进入正式采集；
- SSVEP 页面新增“参与者与会话 → 协议与刺激 → 设备预检与安全确认”三步流程提示；
- 新增任务试次/目标频率提示，以及 `Esc` 取消快捷键；
- 新增用户脱敏索引导出；
- 新增 Windows 用户注册表 DPAPI 保护，非 Windows 环境明确标记 plaintext fallback；
- 新增会话验证模式、协议/通道配置 hash、质量状态、时间戳间隔、掉帧和疑似平线通道统计；
- 新增数据集名称、参与者、用户编号搜索，以及来源和状态筛选；
- 新增诊断中的显示器、DPI 和刷新率信息；
- 新增 100%、125% 和 150% UI scale smoke 测试；
- 新增 SBOM 生成和 packaged smoke 校验。

## 本版本修改

- 统一桌面 UI、CLI、采集 worker、会话文件和数据集摘要中的验证模式与质量字段；
- 将 Cyton 草稿协议/通道表与正式候选配置区分，技术验证允许显式使用草稿覆盖，正式采集默认要求完整配置；
- 更新中英文界面词条，补齐硬件预检、任务进度、数据集筛选、质量摘要和脱敏导出文案；
- 更新 Windows 文件版本、产品版本、桌面标题、SBOM 标识、README、Actions workflow 和发布归档名称为 `1.0.2`；
- 更新发布包内容，附带 `VERSION.txt`、`RELEASE_NOTES.md` 和 `SBOM.json`，便于下载后核验版本与依赖。

## 本版本修复

- 修复 Windows 构建脚本使用跨平台 PowerShell 变量导致平台判断不可靠的问题；
- 修复正式 Cyton 采集可能绕过硬件预检直接启动的问题；
- 修复协议或通道配置仍为草稿时缺少明确阻断和可操作提示的问题；
- 修复会话质量信息无法完整反映时间戳间隔、掉帧和疑似平线通道的问题；
- 修复高 DPI/不同 UI 缩放比例下缺少自动化覆盖的问题；
- 修复发布包缺少 SBOM 和打包后自检证据的问题。

## 使用方式

解压后运行：

```powershell
.\NeuroStation.dist\workstation.exe
.\NeuroStation.dist\workstation.exe --diagnostics
```

默认模式不连接硬件。真实 Cyton 采集前，请先复核通道配置、参考/BIAS、电极连接、显示器刷新率和光学时序，并在设备页运行硬件预检。

## 验收状态

- 核心与集成测试：以本次发布构建日志为准；
- 独立 UI 测试：以本次发布构建日志为准；
- UI scale matrix smoke（100%/125%/150%）：通过；
- Windows standalone packaged smoke：通过；
- SBOM 生成与包内版本文件校验：通过；
- OpenBCI Cyton 实机、photodiode/TTL、长时间稳定性和医疗用途验收：未包含在自动化发布中。

本版本是科研与教学技术验证软件，不是医疗诊断设备。
