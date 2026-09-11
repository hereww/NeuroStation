# NeuroStation 跨平台脑电采集工作站

当前交付版本：**MVP1.0（语义版本 1.0.0）**，主要交付 Windows 10/11 x64 standalone portable 包。版本说明、功能边界和验收状态见 [`release/NeuroStation-MVP1.0-Windows-x64.md`](release/NeuroStation-MVP1.0-Windows-x64.md)。

Windows 下载：发布后可从 [GitHub Releases](https://github.com/hereww/NeuroStation/releases/latest) 下载 `NeuroStation-MVP1.0-Windows-x64.zip`；仓库中的 `dist/` 仅用于本地构建，不提交大体积二进制。

本项目现包含一个可运行的中文桌面工作站、OpenBCI GUI 源码集成流程，以及两个真实硬件命令行工具：

- `workstation.py`：PySide6 工作站入口，提供采集应用图标、SSVEP 参数、任务状态、结果路径和数据集导航。默认启动为**流程演示**（无需硬件、无需额外配置，点击开始即可完成并保存会话元数据）；使用 `--preview` 才进入全屏视觉预览。采集页还提供 BrainFlow Synthetic（会保存合成原始数据）和 OpenBCI Cyton（连接真实硬件）模式，所有模式都会在界面和会话元数据中明确区分。
- `apps/workstation_ui/`：由 GPT-6 独立实现的 UI 层；页面只通过网关调用采集能力。

- `check_cyton_live.py`：短时连接 Cyton，输出采样率、时间戳间隔和各通道信号统计，用于正式采集前的连通性检查。
- `run_ssvep_session.py`：显示全屏 SSVEP 刺激、同步写入 BrainFlow marker，并保存原始数据、事件、配置副本和校验清单。

当前 SSVEP 与通道配置仍标记为 `DRAFT`，用于技术验证。正式采集前必须按实际显示器、佩戴位置和接线复核配置。本工具不提供医疗诊断。

跨平台桌面应用的产品架构、设备/软件集成方式和开发路线见 [`docs/跨平台脑电采集工作站方案.md`](docs/跨平台脑电采集工作站方案.md)。

OpenBCI GUI 官方源码的拉取、上游同步和中文版重构约定见 [`integrations/openbci_gui/README.md`](integrations/openbci_gui/README.md)。

开发进度必须按 [`docs/验收清单.md`](docs/验收清单.md) 留下自动化、桌面平台和真实硬件三类证据；模拟验收不替代 Cyton 实机与 SSVEP 光学时序验收。

## 目录

```text
.
├── workstation.py              # 集成桌面工作站入口
├── apps/workstation_ui/        # 独立 PySide6 UI 与中英文词条
├── check_cyton_live.py          # 硬件连通性/信号概览
├── run_ssvep_session.py         # SSVEP 采集入口
├── configs/
│   ├── ssvep_config_v1.json     # 刺激与试次参数（草稿）
│   ├── protocols/ssvep_four_target_v2.json
│   ├── channel_config_v1_auto.json # Cyton CH1–CH8 自动板卡映射
│   └── channel_config_v1_template.json
├── eeg_tools/                   # 配置校验和会话文件写入
├── tests/                       # 不需要硬件的基础测试
└── requirements.txt
```

## 安装

建议在独立虚拟环境中安装依赖：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 启动工作站

```powershell
.\.venv\Scripts\python.exe workstation.py
```

默认显示中文。英文界面使用 `--language en-US`；验收时可用 `--dataset-root` 指定独立数据目录。导航中的“采集测试”只生成动态内存波形，结果标记为未保存，不写入正式脑电数据库或磁盘文件。SSVEP 默认参数为 10/12/15/20 Hz、12 个试次、93 秒采集时间和 5 秒准备倒计时。默认流程演示不闪烁、不连接设备，点击开始即可运行并创建 `session.json`、`protocol.json`、`events.tsv`、`manifest.csv` 等会话元数据；需要检查黑白倒计时和整屏黑白刺激时使用 `workstation.py --preview`，该模式必须先确认光敏风险，并在结果页明确标记为视觉预览。

在连接设备前，可运行不会打开串口的预检。它会报告 Windows/Linux/macOS 运行环境、Qt/BrainFlow、协议与通道表状态，以及 OpenBCI GUI 源码/中文 overlay/runtime 是否就绪：

```powershell
.\.venv\Scripts\python.exe workstation.py --diagnostics
```

其中 `configuration.status: "draft"` 表示当前使用的仍是模板或草稿配置；这与“发现到 Cyton 设备”是两回事。只有 `check_cyton_live.py` 才会实际尝试打开 USB dongle 对应的串口；若串口不可用，命令会返回结构化 JSON 和退出码 3，便于设备向导或 CI 识别。

独立 UI 入口默认启动可一键运行的流程演示（不连接硬件）：

```powershell
.\.venv\Scripts\python.exe -m apps.workstation_ui.main
```

显式启动全屏视觉预览：

```powershell
.\.venv\Scripts\python.exe -m apps.workstation_ui.main --preview
```

如只需要不创建文件的纯 UI mock，请在测试中注入 `MockGateway`；生产入口不要把预览结果解释为真实 EEG。预览支持 `--dataset-root` 指定会话目录。

BrainFlow 已作为工作台采集核心集成。Cyton 模式的串口默认是 `AUTO`：启动采集时工作站会扫描 Windows COM、Linux `/dev/tty*`、macOS `/dev/cu.*`，逐个调用 BrainFlow `prepare_session()`，第一个成功握手的设备才会进入采集。SSVEP 刺激屏幕也会自动回退到当前第一个可用 Qt 屏幕，实际端口和屏幕编号会写入 `session.json`。如果扫描失败，工作台会显示已扫描端口和 `BOARD_NOT_READY_ERROR:7` 的可操作诊断，而不是只显示原生 traceback。

Cyton 模式默认使用 `configs/channel_config_v1_auto.json`，自动匹配板卡 CH1–CH8 到 N1P–N8P 的输入顺序，不要求普通用户填写 JSON。该映射不知道实际电极佩戴位置；正式实验前仍需审核电极位置、参考、BIAS 和接线。只有勾选 SSVEP 参数页的“高级：手动指定通道配置文件”后，才会启用 JSON 路径选择。工作台中的“扫描 COM”按钮只做只读发现；真正的设备识别仍由 BrainFlow 握手完成。

默认使用 `AUTO` 扫描串口并由 BrainFlow 验证 Cyton 握手；如需固定设备，可填写 `COM5` 或对应的 `/dev/cu.*` 路径。开始前请关闭 OpenBCI GUI 或其他串口监视程序，避免设备被占用。

## 采集前准备

1. 复制通道模板，创建只保存在本机的正式通道表：

   ```powershell
   Copy-Item configs\channel_config_v1_template.json configs\channel_config_v1.json
   ```

2. 在 `configs\channel_config_v1.json` 中填写每个通道的 `electrode_position`，并复核参考电极、BIAS 和实际接线。
3. 复核 `configs\ssvep_config_v1.json` 中的显示器刷新率、频率、刺激时长和重复次数。配置的刷新率必须能被每个刺激频率整除。
4. 在不打开设备和全屏刺激的情况下预检配置：

   ```powershell
   python run_ssvep_session.py --channel-config configs\channel_config_v1.json --validate-only
   ```

草稿配置会通过结构校验，但明确输出 warning；这便于调试，不代表它已经适合正式实验。

## 使用

先做 15 秒连通性检查：

```powershell
python check_cyton_live.py --port AUTO --seconds 15
```

正常情况下输出一段 JSON。重点检查：

- `status` 为 `ok`，且 `samples` 接近期望采样率乘以采集秒数；
- `effective_rate_hz` 接近 Cyton 的 250 Hz；
- `timestamp_gap_count` 没有持续增加；
- 各通道 `finite_fraction` 接近 1，且没有长时间平线或接近负向饱和。

开始一次 SSVEP 会话：

```powershell
python run_ssvep_session.py `
  --port AUTO `
  --channel-config configs\channel_config_v1.json
```

可用 `--output-root` 指定输出根目录，用 `--repetitions` 临时覆盖配置中的重复次数。默认输出到当前用户的 `Documents\OpenBCI_GUI\Recordings`。全屏阶段可随时按 `Esc` 或 `Q` 安全中止；中止会话仍会保存已取得的数据和事件。

查看所有参数：

```powershell
python check_cyton_live.py --help
python run_ssvep_session.py --help
```

## 会话输出

每次运行创建一个 `session_YYYYMMDD_HHMMSS` 目录，包含：

| 文件 | 内容 |
| --- | --- |
| `raw_brainflow.tsv` | BrainFlow 原始板卡数据（包含 EEG、时间戳和 marker 通道） |
| `events.tsv` | 试次事件、目标、频率、marker 与本机时间 |
| `session.json` | 会话状态和基本采集信息 |
| `ssvep_config.json` | 本次实际使用的刺激配置副本 |
| `channel_config.json` | 本次实际使用的通道配置副本 |
| `quality.json` | 采样数、有效比例、RMS、时间戳间隔、掉帧摘要 |
| `manifest.csv` | 各文件大小与 SHA-256，便于归档校验 |

原始脑电与通道表可能包含敏感信息。`recordings/`、`session_*/` 和本地填写后的 `configs/channel_config_v1.json` 已默认加入 `.gitignore`，不要直接提交到代码仓库。

## 当前协议说明

- 默认 60 Hz 显示刷新率，刺激频率为 10、12、15、20 Hz。
- 四个目标位置与四个频率会组成全部配对；默认配置共运行 `4 × 4 × 10 = 160` 个试次。
- 程序按配置刷新率计时，但不会替你校准显示器的实际刷新率或端到端 marker 延迟。
- `pygame_desktop_resolution` 与 `video_controller_resolution` 是复核记录，程序以全屏窗口实际尺寸绘制。

## 无硬件测试

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m unittest discover -s apps/workstation_ui/tests -v
.\.venv\Scripts\python.exe -m compileall -q apps eeg_tools tests workstation.py check_cyton_live.py run_ssvep_session.py
```

## 跨平台构建

项目采用 Qt 官方 `pyside6-deploy`/Nuitka，每个平台在对应操作系统原生构建：

```powershell
.\scripts\build_desktop.ps1 -Python .\.venv\Scripts\python.exe
```

Linux/macOS 使用 `PYTHON=.venv/bin/python ./scripts/build_desktop.sh`。构建配置见 `pysidedeploy.spec`，产物进入 `dist/`；Windows 构建会同时生成 `dist/NeuroStation.dist/` 可运行目录和 `dist/NeuroStation-MVP1.0-Windows-x64.zip` 交付归档。`.github/workflows/desktop.yml` 同时执行三平台测试和 standalone 构建。当前构建产物是无硬件的桌面功能 MVP1.0，不能标记为 Cyton 正式采集版。

Windows 中文路径会触发部分原生 DLL 扫描器的问题，因此 PowerShell 构建脚本会在本机 ASCII 缓存目录中隔离编译，再验证 `workstation.exe` 确实存在并复制回 `dist/`。打包后的独立启动验收命令为：

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe scripts\smoke_packaged.py
```
