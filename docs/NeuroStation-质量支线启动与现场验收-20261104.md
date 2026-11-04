# NeuroStation 质量升级支线：启动与现场验收

日期：2026-11-04。只在独立支线运行；未合并主线。正式协议状态不表示硬件质量或个人模型已验证。

## 1. 推荐先做不连接设备的试用

最终应用目录：
`C:\Users\Administrator\.codex\worktrees\neurostation-quality-upgrade\NeuroStation\quality-output\build-20261104-180252\dist\NeuroStationQualityUpgrade`

双击 `NeuroStationQualityUpgrade.exe` 打开主工作站。仅打开窗口不启动采集或刺激。

也可用 PowerShell 打开独立实时指令窗口：

```powershell
$Package = 'C:\Users\Administrator\.codex\worktrees\neurostation-quality-upgrade\NeuroStation\quality-output\build-20261104-180252\dist\NeuroStationQualityUpgrade'
& "$Package\NeuroStationQualityUpgrade.exe" --realtime-commands
```

包内不包含 Torch、Wang2016 权重、个人模型或外部数据集。**工作站/质量界面和采集运行时已打包；实验模型推理不是无依赖的一体包功能。** 未配置外部模型环境时，不要把“独立实时窗能打开”等同于“模型可开始运行”。

无显式绑定时，包使用 `%LOCALAPPDATA%\NeuroStationQualityUpgrade` 下的 recordings/settings/diagnostics 或 realtime 子目录；不会自动导入主线的 OpenBCI 录制。关闭支线窗口即可退出试用，不需要改回 main。

压缩包应完整解压，不能只复制 EXE 或从压缩包内直接运行。移动包后如果使用了本机模型绑定，必须重新登记映射；不允许为迁移而修改原始权重/锁/训练源码的哈希。

## 2. 在本机从源码启动

```powershell
$Root = 'C:\Users\Administrator\.codex\worktrees\neurostation-quality-upgrade\NeuroStation'
& "$Root\scripts\start_quality_upgrade.ps1" -Entry check
& "$Root\scripts\start_quality_upgrade.ps1" -Entry workstation
# 二选一；独立指令入口：
& "$Root\scripts\start_quality_upgrade.ps1" -Entry realtime
```

启动器先做正式配置检查，读取 `$Root\.quality-local\runtime.json`，验证 Python 可执行文件哈希。当前采集环境为 `F:\脑机采集\NeuroStation\.venv\Scripts\python.exe`；ML 环境为同项目 `.venv-ml-gpu\Scripts\python.exe`。不要求在新 worktree 中创建 `.venv`，不向原环境安装依赖。

源码当前输出为 `$Root\quality-output\recordings` / `realtime`，设置与诊断也位于支线输出内。个人数据不进入 Git。

如需新建运行绑定，使用脚本而非覆盖旧锁：

```powershell
& 'F:\脑机采集\NeuroStation\.venv\Scripts\python.exe' -B "$Root\scripts\configure_quality_runtime.py" `
  --capture-python 'F:\脑机采集\NeuroStation\.venv\Scripts\python.exe' `
  --output-root "$Root\quality-output\my-trial" `
  --config "$Root\.quality-local\runtime-new.json"
```

脚本拒绝覆盖已有绑定文件，只读取环境元数据，不加载模型。使用模型时还需显式 `--model-python`、`--model-run` 及包含 `from`/`to` 绝对根目录的 `--root-mappings` JSON。原训练清单及其锁仍独立校验；错误映射、缺文件、内容变化或前处理不匹配必须拒绝加载。

## 3. 本机包内脚本的可选模型绑定

已有只读本机绑定：`$Root\.quality-local\runtime-packaged.json`。它只指向当前最终包的 `_internal` 源码及原有 ML 环境/模型，不复制资源。当前路径下已完成 CPU 加载后退出验证。

```powershell
$env:NEUROSTATION_RUNTIME_CONFIG = "$Root\.quality-local\runtime-packaged.json"
& "$Package\NeuroStationQualityUpgrade.exe" --realtime-commands
Remove-Item Env:NEUROSTATION_RUNTIME_CONFIG
```

这只是登记可用资源，不自动开始采集，也不把 `runtime_enabled` / `personal_validated` / `production_validated` 改成 true。仍需用户在界面上明确完成预检与开始步骤。若缺硬件质量/刺激时序证据，指令仍为 UNKNOWN；不要通过修改阈值或强制标绿绕过门控。

其他电脑不应复制本机绝对路径绑定后直接使用，应登记自己的合法环境/资源并重新验证。默认包中不附带此本机配置。

## 4. 重复软件验证

所有 --output 使用尚不存在的新目录，保留历史证据。以下均不应自动开启真实设备或刺激：

```powershell
$Python = 'F:\脑机采集\NeuroStation\.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:QT_QPA_PLATFORM = 'offscreen'
& $Python -B "$Root\scripts\verify_quality_upgrade.py" --mode focused --output "$Root\quality-output\review-focused"
& $Python -B "$Root\scripts\verify_quality_upgrade.py" --mode ui --output "$Root\quality-output\review-ui"
& $Python -B "$Root\scripts\verify_quality_upgrade.py" --mode full --output "$Root\quality-output\review-full"
& $Python -B "$Root\scripts\benchmark_quality_upgrade.py" --output "$Root\quality-output\review-benchmark" --seconds 90 --block-samples 13
Remove-Item Env:QT_QPA_PLATFORM
```

全历史测试目前因历史资源不全返回非零，这是如实失败状态。不能为了绿色结果删掉这些检查。检查 `results.json` 的 issues、skips、not_started_ids 与源身份，不只看退出码。

保存块回放与旧会话报告的命令行参数可分别通过 `scripts/replay_quality_session.py --help` 和 `scripts/report_quality_session.py --help` 查询。报告写入新 derived 目录；不要把 output 指向旧历史结果目录。

重新构建：

```powershell
& "$Root\scripts\build_quality_upgrade.ps1" -Python $Python
```

依赖支线 `.quality-local/build-tools` 的 PyInstaller。本次验证版本为 6.22.3，未安装到原环境。构建前有正式配置门禁；构建后自动执行无硬件的 5 项入口检查。真实另一台 Windows 机器仍需人工试用。

## 5. 现场验收记录（当前全部未执行）

一次现场验证一个入口。由操作员明确开始，不在自动测试中触发刺激。

| 检查 | 操作与留存证据 | 当前状态 |
|---|---|---|
| 实际接线/通道/单位 | 确认设备、实际电极位置、参考/地、通道映射、采样率；配置 formal 不是接线证明 | 未执行 |
| 串口与设备握手 | 一次只启用一个采集入口；尝试第二入口须明确拒绝占用；记录握手/错误 | 未执行 |
| 显示与风险确认 | 现场完成显示/刷新率预检、光敏风险确认与中止键检查；不修改现有保护 | 未执行 |
| 事件时间 | 记录 marker、显示帧和光电/TTL；说明时钟基准、偏移/漂移和不可确认点 | 未执行 |
| 数据质量阈值 | 建立硬件量程/噪声底/幅度/漂移依据；验证前维持 UNKNOWN | 未执行 |
| 30min 负载 / 2h 浸泡 | 留存墙钟时长、内存峰值、队列占用、包号异常、水位和封存哈希 | 未执行 |
| 故障与恢复 | 在明确可丢弃的测试录制上做慢盘/断连/进程异常，恢复只到校验通过块 | 仅软件注入已验，实机未执行 |
| 刺激影响 | 同设备开关质量监视，比较帧间隔/迟帧；真实显示 P95 与光学延迟单列 | 未执行 |
| 双屏、高 DPI、语言 | 检查通道表/停止按钮可见、双屏映射；新增质量面板主要中文，完整英文本地化尚未完成，不能算验收通过 | 未执行 |
| 无开发环境主机 | 完整解压包，检查两个窗口/worker、保存及退出；实验模型依赖单独登记 | 未执行 |

如发现丢包、非有限值、缺事件、保存失败、质量过期、模型身份不匹配，停止将该会话当合格会话使用；不改原始数据，不以补零/删点/改锁消除告警。

## 6. 合并审查与回退

```powershell
git -C $Root log --oneline 274a0cd..codex/neurostation-quality-upgrade
git -C $Root diff --stat 274a0cd..codex/neurostation-quality-upgrade
git -C $Root diff 274a0cd..codex/neurostation-quality-upgrade -- eeg_tools apps scripts tests
```

- 先审查基线快照与 main 的差异，再独立审查本轮改进。不要把用户原有未提交功能误当作本轮新增。
- 重点检查：单读取者、保存失败停止、未评估区间、原始/模型输入不变、异步质量消息、SELECT 锁定、默认禁用标志、旧统计字段和新分母区别。
- 不把 `.quality-local`、`quality-output`、本机环境、临时录制副本或个人数据加入合并提交。
- 回退试用只需关闭支线程序，继续使用原程序；不要 reset、清理或覆盖原 main 工作区。
- 原项目和旧 worktree 不需要因试用而修复/迁移。是否合并由用户明确决定，当前没有创建 PR 或推送远端。
