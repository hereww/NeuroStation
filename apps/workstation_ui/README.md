# NeuroStation 独立桌面 UI

由 GPT-6 按已确认的中文界面初稿实现。该目录是一条独立 UI 开发线，使用 PySide6，可与项目采集核心分开运行。所有采集调用集中在 `gateway.py`；页面不直接导入 BrainFlow、LSL、串口库或根目录采集程序。

生产 UI 只连接 `DesktopGateway`，采集请求只能进入 OpenBCI Cyton worker。没有 Cyton 时，用户可以执行诊断、浏览已有数据或导入 OpenBCI 记录，但不能启动 Demo、Preview、Synthetic 或任何内存测试采集。测试代码仍可注入网关以验证页面布局，但测试替身不是生产入口。

## 启动

在项目根目录执行，建议使用独立虚拟环境。需要 Python 3.10+，以及支持当前 Python/操作系统的 PySide6 wheel。

```powershell
python -m venv .venv-ui
.venv-ui\Scripts\python -m pip install -r apps/workstation_ui/requirements.txt
.venv-ui\Scripts\python -m apps.workstation_ui.main
```

Linux / macOS：

```bash
python3 -m venv .venv-ui
.venv-ui/bin/python -m pip install -r apps/workstation_ui/requirements.txt
.venv-ui/bin/python -m apps.workstation_ui.main
```

也支持 `python apps/workstation_ui/main.py`。默认中文；英文启动参数为 `--language en-US`，会话目录可用 `--dataset-root` 指定。缺少 PySide6 时会输出安装指引并以状态码 2 退出，不影响纯逻辑测试。

要运行带元数据落盘的项目集成入口，在根目录执行：

```powershell
.\.venv\Scripts\python.exe workstation.py
```

## 当前界面

- 左侧保留首页、设备、用户、采集应用、会话/数据集、OpenBCI 工作区、集成中心和诊断导航。
- 采集应用提供 SSVEP 和静息态入口；运动想象、P300 标记为尚未实现。
- SSVEP 参数页可编辑参与者、数据集名、刺激时长、休息、重复数、显示屏、绝对保存目录；自动计算预计试次与时长。
- 真实采集开始前执行 Cyton 硬件预检；worker 在倒计时和刺激阶段运行，按 Esc / Q 可中止。运行页展示正式时间轴、试次、目标、频率、阶段、采集与 Marker 状态。
- Cyton 默认显示自动通道映射（CH1–CH8 → N1P–N8P），普通用户不需要填写 JSON；只有勾选高级手动选项后才启用配置文件选择。自动映射不代表知道实际电极位置。
- 完成页展示真实样本/事件、时长、质量状态和绝对目标路径；点击“系统工作站 / 数据集”可查看并突出最近完成项。
- 数据集列表支持将完整数据集目录移入回收站；回收站可恢复，确认“从回收站移除”后才会永久删除全部文件。
- 中英文词条独立保存，中文缺失时回退英文。OpenBCI、LSL 等集成页展示规划状态。

## 时间与状态约定

默认四个目标频率为 10 / 12 / 15 / 20 Hz，刺激 5 秒、休息 3 秒、每目标重复 3 次，共 12 个试次。

```text
采集时间 = 12 × 5 + (12 − 1) × 3 = 93 秒
正式总时间 = 5 秒准备 + 93 秒采集 = 98 秒
每通道预计样本 = 93 × 250 = 23,250
8 通道采样值 = 23,250 × 8 = 186,000
预计事件 = 任务开始/结束 + 12 × 刺激开始/结束 = 26
```

最后一个试次后不追加休息。采集从 worker 启动后按 1× 实际时间运行。60 Hz 是技术验证参数，正式刺激必须单独验证刷新率与逐帧时序。

状态机为 `idle → countdown → running → completed`，倒计时或运行中可取消。活动任务阻止重复启动；取消不会生成数据集。应用只有一个 `QTimer` 推动网关；完成、取消、关闭均停止定时器。导航与语言切换不会创建第二个任务定时器。若 UI 在倒计时结束前被系统挂起，恢复后才进入运行阶段，不会在不可见时跳过整个任务。

## 文件边界

| 文件 | 职责 |
| --- | --- |
| `main.py` | 独立入口与依赖检查 |
| `app.py` | 桌面外壳、导航、统一定时器与网关命令路由 |
| `pages.py` | 页面及用户动作信号 |
| `components.py` | 通用按钮、信息区、动态测试波形、静态刺激预览 |
| `gateway.py` | `CaptureGateway` 接口和测试替身；生产实现位于 `eeg_tools/workstation/desktop_gateway.py` |
| `i18n.py` / `locales/` | 词条读取与英文回退 |
| `tests/` | 无 GUI 逻辑测试及可选 Qt 离屏测试 |

生产网关实现 `CaptureGateway`，把设备状态、任务生命周期、数据集结果通过该接口交给 UI。真实刺激与采集进程的时序由后端实现，不能依赖 UI 的 100 ms 刷新定时器。生产数据集必须提供真实落盘结果。

默认目录使用 `Path.home()/Documents/NeuroStation/datasets`；所有路径由 `pathlib` 处理，支持 Windows/Linux/macOS，并可通过原生目录选择器修改。

## 测试

从项目根目录执行：

```bash
python -m unittest discover -s apps/workstation_ui/tests -v
```

无 PySide6 仍可运行协议、数据集、配置边界和真实硬件入口拒绝模拟模式的测试。安装 PySide6 后自动运行 `QT_QPA_PLATFORM=offscreen` 的 Qt 页面构建、导航、语言切换和关闭测试。真实 Cyton 采集不在无硬件 CI 中伪造。

当前环境的执行结果请以交付说明为准；未运行的 Qt 或跨平台实机验证不代表已通过。
