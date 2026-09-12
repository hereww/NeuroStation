# NeuroStation MVP1.0.1 Windows x64

## 版本定位

- 产品版本：`MVP1.0.1`
- 产品系列：`MVP1.0`
- 语义版本：`1.0.1`
- 发布日期：`2026-09-12`
- 发布形式：Windows standalone portable build
- 目标系统：Windows 10/11 x64

## 本版本新增

- 采集用户管理页面，支持用户编号、姓名、年龄、性别和医疗情况本地档案；
- SSVEP 采集任务支持选择采集用户，并把用户编号和姓名写入会话结果；
- 采集页支持快速创建用户，创建后自动回填到当前任务；
- 数据集列表、结果页和会话摘要显示用户信息及关联状态；
- 用户回收站支持保留关联数据删除、恢复和永久删除；
- 演示用户受保护，真实采集模式禁止使用演示用户；
- 用户编号自动生成、重复检查、缺失用户检查和医疗字段一致性校验；
- 旧会话和未关联历史记录继续可读，未关联记录不会被猜测绑定到新用户。

## 本版本修改

- 统一用户、参与者和会话之间的数据契约，CLI、桌面网关和 BrainFlow worker 使用相同的会话字段；
- 更新中英文界面词条，增加用户管理、用户状态、关联状态和数据集用户摘要；
- 更新 OpenBCI 导入数据集摘要测试的预期行数，覆盖新增用户信息后的 UI 结构；
- 更新应用版本元数据、Windows 文件版本、桌面标题和打包归档名称为 `1.0.1`；
- README、验收入口和发布脚本同步到 MVP1.0.1。

## 本版本修复

- 修复 Windows 控制台默认编码无法输出中文 JSON 导致 worker/诊断失败的问题；
- 修复 Linux CI 缺少 `libegl1`/`libgl1` 导致 Qt packaged smoke 失败的问题；
- 修复 Linux 快速取消时 worker 已写入会话但 gateway 尚未读取终态结果的问题；
- 修复 BrainFlow x86_64 runtime 在 macOS ARM runner 上构建不兼容的问题，CI 改用 Intel `macos-13`；
- 修复桌面验收 workflow 在 main 与 tag 上重复触发造成的排队问题，tag 仅由 Release workflow 处理；
- 修复桌面任务取消后结果路径、`session.json` 和 `aborted` 状态可能不同步的问题。

## 使用方式

解压后运行：

```powershell
.\NeuroStation.dist\workstation.exe
.\NeuroStation.dist\workstation.exe --diagnostics
```

默认模式不连接硬件。真实 Cyton 采集前，请先完成串口连通性、通道位置、参考/BIAS、显示器刷新率和光学时序复核。

## 验收状态

- 核心与集成测试：`35/35` 通过；
- 独立 UI 测试：`26/26` 通过；
- Windows standalone packaged smoke：通过；
- OpenBCI Cyton 实机、photodiode/TTL、长时间稳定性和医疗用途验收：未包含在本版本自动化发布中。

本版本是科研与教学技术验证软件，不是医疗诊断设备。
