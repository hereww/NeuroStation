# NeuroStation 脑电采集工作站

[![Desktop acceptance](https://github.com/hereww/NeuroStation/actions/workflows/desktop.yml/badge.svg?branch=main)](https://github.com/hereww/NeuroStation/actions/workflows/desktop.yml)
[![Latest release](https://img.shields.io/github/v/release/hereww/NeuroStation?label=release)](https://github.com/hereww/NeuroStation/releases/latest)

NeuroStation 是一个面向科研与教学技术验证的跨平台脑电采集工作站。项目把 PySide6 桌面界面、BrainFlow 采集 worker、SSVEP 刺激流程、OpenBCI Cyton 接入、会话文件和数据集浏览整合在同一套工作流中。

当前交付版本为 **MVP1.0.1（语义版本 1.0.1）**，属于 MVP1.0 系列的功能维护版本。Windows 10/11 x64 提供无需 Python 环境的 standalone portable 包；Linux 和 macOS 主要用于源码测试与平台构建验收。

## 下载与版本边界

- [下载 NeuroStation MVP1.0.1 Windows x64](https://github.com/hereww/NeuroStation/releases/latest)
- [查看 MVP1.0.1 Release 说明](release/NeuroStation-MVP1.0.1-Windows-x64.md)
- [查看完整验收清单](docs/验收清单.md)
- [查看跨平台方案](docs/跨平台脑电采集工作站方案.md)

Windows 包是可直接解压运行的目录，不是 MSI 安装器，也不包含真实 Cyton 设备。默认启动为不连接硬件的流程演示；Synthetic 模式产生 BrainFlow 合成数据；Cyton 模式才会打开真实串口。

### MVP1.0.1 更新内容

- 新增采集用户管理：用户编号、姓名、年龄、性别和本地医疗情况字段校验；
- 新增用户与采集会话关联，结果页和数据集列表显示用户摘要与关联状态；
- 新增用户回收站，支持保留关联数据删除、恢复和永久删除；演示用户受保护；
- 新增从 SSVEP 采集页快速创建用户并自动回填当前采集任务；
- 改进用户编号生成、重复编号校验、草稿/演示用户限制和旧会话兼容；
- 修正 OpenBCI 导入数据集摘要的行数测试，确保新增用户列后 UI 验收口径一致；
- 延续上一版本的 Windows Unicode 输出、Linux Qt 依赖、跨平台构建和取消后会话持久化修复。

本项目不是医疗诊断设备。SSVEP 参数、通道位置、参考电极、BIAS、显示器刷新率和光学/marker 时序仍需在正式实验前人工复核。MVP1.0 的自动化测试不能替代 Cyton 实机、photodiode/TTL 和长时间稳定性验收。

## 运行模式

| 模式 | 是否连接硬件 | 是否产生 EEG 样本 | 用途 |
| --- | --- | --- | --- |
| 流程演示 | 否 | 否 | 首次启动、界面和会话元数据验收 |
| 视觉预览 `--preview` | 否 | 否 | 全屏倒计时、黑白刺激和帧时序验收；需要确认光敏风险 |
| BrainFlow Synthetic | 否 | 是，合成数据 | 无硬件环境的端到端采集验收 |
| OpenBCI Cyton | 是 | 是，真实数据 | 真实设备采集；需要完成硬件和实验时序复核 |
| OpenBCI 记录导入 | 否 | 使用已有记录 | 只读导入历史 OpenBCI recording，不修改原始文件 |

## 快速启动

### 使用 Windows 发布包

1. 下载并解压 `NeuroStation-MVP1.0.1-Windows-x64.zip`。
2. 运行 `NeuroStation.dist\workstation.exe`。
3. 首次使用先选择默认流程演示，确认界面和会话目录能够正常生成。
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

独立 UI 入口也可以直接运行：

```powershell
python -m apps.workstation_ui.main
python -m apps.workstation_ui.main --preview
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
  --channel-config configs\channel_config_v1.json
```

`AUTO` 会扫描可用串口并逐个尝试 BrainFlow `prepare_session()`。失败时会输出结构化 JSON 和可操作诊断；不会把原生 traceback 当成唯一错误信息。真实采集结束或中止后，数据和会话状态会写入输出目录。

## 会话文件

每次采集会创建一个 `session_YYYYMMDD_HHMMSS...` 目录，常见文件如下：

| 文件 | 内容 |
| --- | --- |
| `raw_brainflow.tsv` | BrainFlow 原始矩阵，包括 EEG、时间戳和 marker 通道 |
| `events.tsv` | 试次、目标、频率、marker 和本机时间 |
| `session.json` | 状态、设备、采样数、参与者标识和版本信息 |
| `ssvep_config.json` | 本次实际使用的协议副本 |
| `channel_config.json` | 本次实际使用的通道配置副本 |
| `quality.json` | 有效比例、RMS、采样数、时间戳和掉帧摘要 |
| `frame_timing.tsv` | 视觉预览/刺激帧的计划时间和实际时间 |
| `manifest.csv` | 输出文件大小和 SHA-256 清单 |

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
  --validate-only
```

无硬件端到端 Synthetic 验收：

```powershell
python run_ssvep_session.py `
  --synthetic --headless `
  --participant CI `
  --session-name synthetic-acceptance `
  --output-root .acceptance-data `
  --repetitions 1 `
  --stimulus-seconds 0.05 `
  --rest-seconds 0.01 `
  --countdown-seconds 0.01
```

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
- `dist\NeuroStation-MVP1.0.1-Windows-x64.zip`：发布归档。

GitHub Actions 的职责分工如下：

- `Desktop acceptance`：`main`、Pull Request 和手动触发时运行三平台测试/构建；
- `NeuroStation release`：推送 `v*` 标签时构建 Windows 包并创建/更新 Release；
- 真实硬件检查不在云端执行，必须使用现场设备单独记录 H/T 验收证据。

## 目录结构

```text
.
├── workstation.py                 # 集成桌面工作站入口
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
