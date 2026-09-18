# NeuroStation 脑电采集工作站

[![Desktop acceptance](https://github.com/hereww/NeuroStation/actions/workflows/desktop.yml/badge.svg?branch=main)](https://github.com/hereww/NeuroStation/actions/workflows/desktop.yml)
[![Latest release](https://img.shields.io/github/v/release/hereww/NeuroStation?label=release)](https://github.com/hereww/NeuroStation/releases/latest)

NeuroStation 是一个面向科研与教学技术验证的跨平台脑电采集工作站。项目把 PySide6 桌面界面、BrainFlow 采集 worker、SSVEP 刺激流程、OpenBCI Cyton 接入、会话文件和数据集浏览整合在同一套工作流中。

当前交付版本为 **MVP1.0.3（语义版本 1.0.3）**，属于 MVP1.0 系列的功能维护版本。Windows 10/11 x64 提供无需 Python 环境的 standalone portable 包；Linux 和 macOS 主要用于源码测试与平台构建验收。

## 下载与版本边界

- [下载 NeuroStation MVP1.0.3 Windows x64](https://github.com/hereww/NeuroStation/releases/latest)
- [查看 MVP1.0.3 Release 说明](release/NeuroStation-MVP1.0.3-Windows-x64.md)
- [查看完整验收清单](docs/验收清单.md)
- [查看跨平台方案](docs/跨平台脑电采集工作站方案.md)

Windows 包是可直接解压运行的目录，不是 MSI 安装器，也不包含真实 Cyton 设备。当前生产入口只允许 OpenBCI Cyton 真实硬件采集；没有设备时只能查看历史记录、导入 OpenBCI 文件或运行诊断，不能生成模拟 EEG。

### MVP1.0.3 更新内容

- 新增透明的 SSVEP 离线 FFT 基线分析，生成 `analysis.json` 和 `trial_features.tsv`；
- 新增 `raw_columns.tsv` 和目标标记审计字段，区分呈现目标、人工/自报目标与 EEG 推断目标；
- 新增诊断页、持久化事件日志、JSON 报告导出和真实采样滚动波形显示；
- 新增数据集回收站，支持恢复、永久删除和筛选；
- 生产入口收敛为 OpenBCI Cyton 实机，CI 不再生成 Synthetic/Demo EEG；
- 固定 BrainFlow `5.22.2`，修复跨平台原生库混包、macOS runtime、Windows 路径和打包 smoke 问题；
- 修复活动数据集遮蔽回收站记录和干净时间戳抖动重复告警问题。

本项目不是医疗诊断设备。SSVEP 参数、通道位置、参考电极、BIAS、显示器刷新率和光学/marker 时序仍需在正式实验前人工复核。MVP1.0 的自动化测试不能替代 Cyton 实机、photodiode/TTL 和长时间稳定性验收。

## 运行来源

| 来源 | 是否连接硬件 | 数据含义 | 用途 |
| --- | --- | --- | --- |
| OpenBCI Cyton | 是 | 真实 EEG 数据 | 真实设备采集；需要完成硬件和实验时序复核 |
| OpenBCI 记录导入 | 否 | 已有真实记录 | 只读导入历史 OpenBCI recording，不修改原始文件 |

## 快速启动

### 使用 Windows 发布包

1. 下载并解压 `NeuroStation-MVP1.0.3-Windows-x64.zip`。
2. 运行 `NeuroStation.dist\workstation.exe`。
3. 首次使用先连接 Cyton USB dongle，选择采集用户并完成硬件预检。
4. 需要查看运行环境时执行：

```powershell
.\NeuroStation.dist\workstation.exe --diagnostics
```

### 从源码启动

需要 Python 3.10 或更高版本，以及当前平台可用的 PySide6、BrainFlow 和 NumPy。

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python workstation.py
```

默认界面为中文。英文界面使用 `--language en-US`；测试或隔离数据时可使用 `--dataset-root` 指定输出目录。

开发调试时可从左侧“诊断”页刷新检查、查看最近事件、清空事件或导出报告。持久化事件日志默认位于 `Documents\NeuroStation\Diagnostics\workstation-events.jsonl`；命令行 `--diagnostics` 使用同一套报告生成逻辑，适合无界面环境和打包后检查。

独立 UI 入口也可以直接运行：

```powershell
python -m apps.workstation_ui.main
```

## SSVEP 默认协议

默认协议位于 `configs/protocols/ssvep_four_target_v2.json`，当前用于 MVP 技术验证：

- 显示刷新率：60 Hz；
- 目标频率：10、12、15、20 Hz；
- 刺激时长：5 秒；
- 试次间休息：3 秒；
- 每个目标重复 3 次，共 12 个试次；
- 采集时长：93 秒，另有 5 秒准备倒计时；
- Cyton 目标采样率：250 Hz、8 通道。

每次 SSVEP 任务只采集一只眼，开始前必须选择“左眼”或“右眼”。程序在采集
worker 启动时按显示器物理横坐标重新排序：最左屏绑定左眼，最右屏绑定右眼；
选中屏幕显示 10 / 12 / 15 / 20 Hz 四目标刺激，另一块屏幕全屏保持黑色。可视
采集少于两块显示器时会被阻止，多于两块时中间显示器不参与刺激。数据集名称会
自动生成 `原名称_左眼` 或 `原名称_右眼`，已有眼别后缀会先去重。

协议文件状态为 `draft_for_workstation_mvp`。程序会校验刺激频率是否能整除配置的刷新率，但不会替代实际显示器刷新率和端到端光学时序校准。

## OpenBCI Cyton 使用流程

开始真实采集前，建议按以下顺序执行：

1. 关闭 OpenBCI GUI 或其他串口监视程序，确保 USB dongle 未被占用。
2. 准备正式通道配置：

```powershell
Copy-Item configs\channel_config_v1_template.json configs\channel_config_v1.json
```

3. 填写实际电极位置，复核参考、BIAS、AGND 和接线。`channel_config_v1_auto.json` 只负责 CH1-CH8 到板卡输入的顺序映射，不知道真实佩戴位置。
4. 运行不会打开串口的环境预检：

```powershell
python workstation.py --diagnostics
```

5. 使用短时连通性检查确认 BrainFlow 能够握手：

```powershell
python check_cyton_live.py --port AUTO --seconds 15
```

6. 通过桌面工作站或 CLI 开始正式采集：

```powershell
python run_ssvep_session.py `
  --port AUTO `
  --channel-config configs\channel_config_v1.json `
  --eye-side left
```

`AUTO` 会扫描可用串口并逐个尝试 BrainFlow `prepare_session()`。失败时会输出结构化 JSON 和可操作诊断；不会把原生 traceback 当成唯一错误信息。真实采集结束或中止后，数据和会话状态会写入输出目录。

## 会话文件

每次采集会创建一个 `session_YYYYMMDD_HHMMSS...` 目录，常见文件如下：

| 文件 | 内容 |
| --- | --- |
| `raw_brainflow.tsv` | BrainFlow 原始矩阵，包括 EEG、时间戳和 marker 通道 |
| `raw_columns.tsv` | 原始矩阵列头、BrainFlow row、单位、角色和意义 |
| `events.tsv` | 试次、目标、频率、marker 和本机时间 |
| `session.json` | 状态、设备、采样数、参与者标识和版本信息 |
| `ssvep_config.json` | 本次实际使用的协议副本 |
| `channel_config.json` | 本次实际使用的通道配置副本 |
| `quality.json` | 有效比例、RMS、采样数、时间戳和掉帧摘要 |
| `frame_timing.tsv` | 视觉预览/刺激帧的计划时间和实际时间 |
| `manifest.csv` | 输出文件大小和 SHA-256 清单 |

新 SSVEP 会话的 `session.json`、`ssvep_config.json`、状态文件、`events.tsv` 和
`frame_timing.tsv` 会记录 `eye_side`、基础数据集名称、生成后的会话名称以及实际
刺激屏幕的编号、名称和几何信息。旧会话缺少这些字段时仍可读取，并在 UI 中显示
为“未标注”。

新采集会话中的 `events.tsv` 会区分 `presented_target_id`（软件呈现目标）、
`gaze_target_id`（人工/自报标记）和 `eeg_predicted_target_id`（离线算法推断）。
其中只有第一项是软件流程事实，后两项不能冒充眼动真值。完整的列定义、同步模型、
理论依据和离线分析流程见 [`docs/离线分析与目标标记审计.md`](docs/离线分析与目标标记审计.md)。

对已有会话运行透明的 NumPy FFT 基线：

```powershell
python analyze_ssvep_session.py .\path\to\session_YYYYMMDD_HHMMSS_mmm
```

分析只生成 `analysis.json` 和 `trial_features.tsv`，不会改写原始 EEG。

### 离线去噪研究管线

MVP1.0.3 支线提供可审计的 SciPy 离线去噪流程。它不会修改
`raw_brainflow.tsv`，默认同时生成 50 Hz 和 60 Hz 两套结果，并保留原始标记轨道：

```powershell
python preprocess_eeg_session.py `
  .\path\to\session_YYYYMMDD_HHMMSS_mmm `
  --pipeline configs\denoise_pipeline_v1.json
```

默认输出到会话目录下的 `derived\denoise_v1\`，包含伪迹段、带通/陷波后的
`denoised_50hz.npz` 和 `denoised_60hz.npz`、SSVEP 特征、PSD/SNR/工频比、处理参数、
输入与派生文件 hash。短的非有限缺口只在滤波前临时插值，超过 0.2 秒的缺口保持为
无效值并进入 `artifact_segments.tsv`；原始文件始终保持不变。结果页会在派生结果存在时
显示三条处理轨道的摘要。

原始 EEG 和通道配置可能包含敏感信息。录制目录、会话目录以及本地正式通道配置已加入 `.gitignore`，不要把真实参与者数据提交到 Git 仓库。

## CLI 入口

```powershell
python workstation.py --help
python workstation.py --diagnostics
python check_cyton_live.py --help
python run_ssvep_session.py --help
```

只做协议和通道配置校验：

```powershell
python run_ssvep_session.py `
  --channel-config configs\channel_config_v1.json `
  --eye-side left `
  --validate-only
```

CI 不生成 Synthetic 或 Demo 数据。没有 Cyton 时只执行协议/通道配置校验、UI 构建和禁止模拟入口的回归测试；真实采集必须在连接设备的现场环境执行。

## 开发、测试与打包

运行核心测试、UI 测试和语法检查：

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
python -m unittest discover -s tests -v
python -m unittest discover -s apps/workstation_ui/tests -v
python -m compileall -q apps eeg_tools tests scripts workstation.py check_cyton_live.py run_ssvep_session.py neurostation_contract.py
```

Windows standalone 构建：

```powershell
.\scripts\build_desktop.ps1 -Python .\.venv\Scripts\python.exe -Mode standalone
python scripts\smoke_packaged.py
```

构建会把源码复制到 ASCII 路径的临时 stage，使用 Qt 官方 `pyside6-deploy`/Nuitka 构建，再把产物复制回 `dist/`。Windows 输出包括：

- `dist\NeuroStation.dist\`：可运行目录；
- `dist\NeuroStation-MVP1.0.3-Windows-x64.zip`：发布归档。

GitHub Actions 的职责分工如下：

- `Desktop acceptance`：`main`、Pull Request 和手动触发时运行三平台测试/构建；
- `NeuroStation release`：推送 `v*` 标签时构建 Windows 包并创建/更新 Release；
- 真实硬件检查不在云端执行，必须使用现场设备单独记录 H/T 验收证据。

## 目录结构

```text
.
├── workstation.py                 # 集成桌面工作站入口
├── neurostation_diagnostics.py    # 诊断检查、事件日志和报告导出
├── apps/workstation_ui/           # PySide6 UI、页面和中英文词条
├── eeg_tools/workstation/          # 协议、worker、网关、设备和数据集逻辑
├── configs/                        # SSVEP 协议和 Cyton 通道配置
├── check_cyton_live.py             # Cyton 连通性和信号概览
├── run_ssvep_session.py            # CLI SSVEP 入口
├── scripts/                        # 跨平台同步、构建和 smoke 工具
├── tests/                          # 核心、集成和数据集测试
├── docs/                           # 方案、验收和工程记录
└── release/                        # 发布说明
```

## 当前限制

- MVP1.0 是技术验证版本，不是临床或医疗诊断软件；
- Cyton 实机连接、连续稳定性、断线恢复和串口占用场景仍需现场验收；
- SSVEP 光学 onset、marker 延迟、实际刷新率和 photodiode/TTL 结果不能由软件配置推断；
- OpenBCI GUI 集成依赖锁定上游源码和平台 runtime，不能把源码同步成功等同于硬件采集通过；
- 当前 Windows Release 是 portable 包，尚未提供 MSI、代码签名和自动安装/卸载流程。

详细放行规则和未完成项见 [`docs/验收清单.md`](docs/验收清单.md)。
