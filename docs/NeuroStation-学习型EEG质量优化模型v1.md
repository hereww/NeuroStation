# NeuroStation 多通道选择性 EEG 质量优化模型 v1

## 目标

本实现提供一个非破坏性的、可回放的学习型 EEG 质量优化管线：

- 原始 `raw_brainflow.tsv` 永远不修改；
- 模型输出保存到 `derived/learned_denoise_v1/`；
- 同时输出去噪波形、伪迹概率、置信度和质量报告；
- 缺失样本、NaN/Inf、长时间异常不由模型伪造恢复；
- 当前默认 `inference.mode=shadow`，未完成真实设备和跨被试评估前不替换正式分析轨迹。

## 本地源运行部署（采集结束后异步 shadow）

当前接入是**采集结束后异步 shadow**，不是实时逐窗口部署：

1. 正常启动采集工作站，不自动启动采集或光刺激。
2. 采集开始后，网关启动一个独立监视进程，等待该次采集的终态状态文件。
3. 正式采集完成或中止并保存会话后，监视进程才读取 `raw_brainflow.tsv` 并在 CPU 上推理。
4. 结果只写入该会话的 `derived/learned_denoise_v1/`；UI 原始波形和正式质量判定保持原有链路。

只检查环境、模型和正式配置，不打开 GUI：

```powershell
& .\scripts\start_learned_workstation.ps1 -CheckOnly
```

检查通过后打开工作站（仍需操作员手动启动采集并完成安全确认）：

```powershell
& .\scripts\start_learned_workstation.ps1
```

入口固定解析项目 `.venv\Scripts\python.exe`，先运行正式 release 配置验证，再检查 NumPy、SciPy、BrainFlow、PySide6、PyTorch 和配置指定的本地模型是否可加载。脚本不安装依赖、不下载数据、不训练模型、不打开串口、不自动触发光刺激。

运行配置保持 `status: formal`、`inference.mode: shadow`。可在 `configs/learned_denoiser_v1.json` 中设置：

```json
"runtime": {
  "enabled": false,
  "model_path": "models/denoiser_v1.pt"
}
```

`enabled: false` 关闭自动后台推理。模型不存在或配置不适用时，网关跳过可选监视进程，不阻止原始采集。模型加载、输入或推理失败时，已启动的 worker 保存独立 fallback 状态，不将原始质量改为通过。启动采集失败时不启动监视进程；工作站关闭时取消等待或推理中的可选进程；UI tick 仅 poll、不 wait。

后台派生产物：

```text
<session>/derived/learned_denoise_v1/
├─ denoised_signal.npz
├─ quality_events.jsonl
├─ model_manifest.json
└─ status.json
```

此目录属于派生分析，不覆写 `raw_brainflow.tsv`、`quality.json`、`session.json` 或历史验证结论。无法取得合法会话路径、路径越出指定输出根或等待超时时，fallback 状态只写入该次采集的独立 runtime 目录，不能写入输出根的公共 `derived/`。

输入保护：

- `session.json` 的 `sampling_rate_hz` 必须已知且与模型配置一致；不自动重采样。
- 通道数必须匹配配置；NaN/Inf 保留为无效，不插值，过多无效值拒绝整会话推理。
- `sample_index` 是导出编号，不能证明无丢包。Cyton `package_num` 按 256 回卷检查连续性；发现跳跃/重复或时间戳出现大于 1 秒的明显前向缺口时，标记疑似不连续并拒绝该会话 shadow 推理。
- 批量相同时间戳不当成丢包；缺少充分包号证据时明确记录 `not_assessed`，任何情况下都不宣称采集完整性已验收。
- `confidence <= 0` 或非有限置信度计入 `inference_rejected_fraction`，质量事件标为未知/未充分评估；整窗拒绝时的零伪迹输出不解释为“无伪迹”。

**桌面 EXE 边界：**标准 EXE 不强制嵌入 PyTorch 或模型。现有旧 EXE 不等于已经接入学习模型；本次交付仅验证源运行部署，未大型重打包、未修改发布版本元数据。若要提供 ML 专用 EXE，需要后续单独验收打包依赖、模型文件、子进程入口及真实硬件行为。

现有模型只能用于 shadow 辅助观察。旧 checkpoint 划分/旧全数据评估不作为独立泛化验证，尤其不能把先前 23.45% 数字视为跨会话或跨被试效果；新的训练与独立评估由单独任务完成。

## 文件

- `eeg_tools/learned_denoise.py`：Residual TCN、归一化、滑窗推理。
- `eeg_tools/learned_dataset.py`：NeuroStation 会话读取和半合成伪迹训练数据生成。
- `scripts/build_denoise_dataset.py`：构建 paired NPZ 训练集。
- `scripts/train_denoiser.py`：PyTorch 训练入口。
- `scripts/evaluate_denoiser.py`：重构、伪迹掩码和质量指标。
- `scripts/run_learned_denoise.py`：对历史会话运行模型。
- `configs/learned_denoiser_v1.json`：正式配置；当前发布门槛为 shadow。
- `eeg_tools/workstation/learned_monitor.py`：采集终态后异步推理及派生质量事件。
- `scripts/start_learned_workstation.ps1`：项目虚拟环境源运行入口，支持 `-CheckOnly`。
- `datasets/external/`：外部数据集和逐数据集 manifest。

## 安装可选训练依赖

桌面采集的基础依赖不强制安装 PyTorch。训练或推理前安装 ML extra：

```powershell
& .\.venv\Scripts\python.exe -m pip install -e ".[ml]"
```

所有命令均使用项目 `.venv\Scripts\python.exe`；不要使用 PATH 中缺少 NumPy 等依赖的 Python 3.11。若当前解释器没有匹配的 PyTorch wheel，应准备满足项目要求且有兼容 wheel 的虚拟环境；不要为启动修改或跳过正式配置校验。

## 1. 构建本地会话训练集

```powershell
& .\.venv\Scripts\python.exe scripts/build_denoise_dataset.py `
  .tmp_preview `
  --output datasets/processed/learned_denoiser_v1.npz `
  --max-windows 20000
```

当前生成器从相对干净的会话窗口构造工程上的 clean-ish 目标，并注入眼动样、肌电样、工频、漂移和电极突变伪迹。它不是临床真值。正式训练应优先混入 `datasets/external/` 中已核验的数据集以及实际 Cyton 伪迹样本。

## 2. 训练

```powershell
& .\.venv\Scripts\python.exe scripts/train_denoiser.py `
  datasets/processed/learned_denoiser_v1.npz `
  --output models/denoiser_v1.pt `
  --epochs 12
```

训练按会话划分 train/validation/test，不按窗口随机打散同一会话。输出 checkpoint 记录数据会话列表、配置和训练历史。

## 3. 评估

```powershell
& .\.venv\Scripts\python.exe scripts/evaluate_denoiser.py `
  datasets/processed/learned_denoiser_v1.npz `
  --model models/denoiser_v1.pt `
  --output analysis_outputs/learned_denoiser_v1.json
```

评估至少应与以下三种轨迹比较：

1. 原始轨迹；
2. `eeg_tools/denoise.py` 的传统处理轨迹；
3. 学习模型轨迹。

另外必须在真实未参与训练的会话上比较 SSVEP CCA 目标频率保持情况，尤其是 10/12/15/20 Hz 和谐波，不得只看 RMSE。

## 4. 对单个会话推理

```powershell
& .\.venv\Scripts\python.exe scripts/run_learned_denoise.py `
  .tmp_preview/session_20260916_000000 `
  --model models/denoiser_v1.pt
```

输出：

```text
<session>/derived/learned_denoise_v1/
├─ denoised_signal.npz
├─ artifact_segments.tsv
├─ quality_metrics.json
└─ model_manifest.json
```

## 外部数据使用规则

- 只使用 manifest 中明确记录来源、许可、下载日期、SHA256 和缺失项的数据；
- 软件仓库的 MIT 许可证不能自动推导底层 EEG 数据可任意再分发；
- EEGdenoiseNet 的 clean/EOG/EMG 是独立 epoch 池，应按伪迹合成方案构造训练输入，不把相同行号当成同步配对；
- 外部数据只用于模型训练/验证，不覆盖本项目真实采集记录的历史结论。

## 正式版本门槛

在把 `inference.mode` 从 shadow 改为正式分析前，必须完成：

- 真实 Cyton 设备回放和现场测试；
- 按被试/会话划分的独立测试；
- 干净片段不劣化门槛；
- 工频、眼动、肌电、漂移和突变的伪迹检测评估；
- SSVEP 目标识别与误判/拒判分析；
- CPU 延迟和内存基准；
- 原始数据哈希不变、派生结果可复现；
- `& .\.venv\Scripts\python.exe scripts/validate_release_configs.py`、核心测试和界面测试通过。

模型状态不代表硬件、会话质量或临床有效性已经验收。

## 5. SSVEP 目标频率保持检查

已下载的 Dong2023 数据集可以作为下游保持性 smoke test：

```powershell
& .\.venv\Scripts\python.exe scripts/evaluate_ssvep_preservation.py `
  datasets/external/ssvep_datasets/dong2023_ssvep `
  --model models/denoiser_v1.pt `
  --max-subjects 2
```

该脚本比较去噪前后的目标频率幅度，不把幅度比直接称为识别准确率。正式发布前还需要使用独立被试、项目自己的 SSVEP CCA 流程和真实 Cyton 回放验证。
