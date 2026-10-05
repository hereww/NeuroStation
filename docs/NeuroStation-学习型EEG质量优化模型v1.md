# NeuroStation 多通道选择性 EEG 质量优化模型 v1

## 目标

本实现提供一个非破坏性的、可回放的学习型 EEG 质量优化管线：

- 原始 `raw_brainflow.tsv` 永远不修改；
- 模型输出保存到 `derived/learned_denoise_v1/`；
- 同时输出去噪波形、伪迹概率、置信度和质量报告；
- 缺失样本、NaN/Inf、长时间异常不由模型伪造恢复；
- 当前默认 `inference.mode=shadow`，未完成真实设备和跨被试评估前不替换正式分析轨迹。

## 文件

- `eeg_tools/learned_denoise.py`：Residual TCN、归一化、滑窗推理。
- `eeg_tools/learned_dataset.py`：NeuroStation 会话读取和半合成伪迹训练数据生成。
- `scripts/build_denoise_dataset.py`：构建 paired NPZ 训练集。
- `scripts/train_denoiser.py`：PyTorch 训练入口。
- `scripts/evaluate_denoiser.py`：重构、伪迹掩码和质量指标。
- `scripts/run_learned_denoise.py`：对历史会话运行模型。
- `configs/learned_denoiser_v1.json`：正式配置；当前发布门槛为 shadow。
- `datasets/external/`：外部数据集和逐数据集 manifest。

## 安装可选训练依赖

桌面采集的基础依赖不强制安装 PyTorch。训练或推理前安装 ML extra：

```powershell
python -m pip install -e ".[ml]"
```

如果当前 Python 版本没有匹配的 PyTorch wheel，应使用项目支持的 Python 3.10–3.13 环境；不要为了桌面采集入口修改正式配置校验。

## 1. 构建本地会话训练集

```powershell
python scripts/build_denoise_dataset.py `
  .tmp_preview `
  --output datasets/processed/learned_denoiser_v1.npz `
  --max-windows 20000
```

当前生成器从相对干净的会话窗口构造工程上的 clean-ish 目标，并注入眼动样、肌电样、工频、漂移和电极突变伪迹。它不是临床真值。正式训练应优先混入 `datasets/external/` 中已核验的数据集以及实际 Cyton 伪迹样本。

## 2. 训练

```powershell
python scripts/train_denoiser.py `
  datasets/processed/learned_denoiser_v1.npz `
  --output models/denoiser_v1.pt `
  --epochs 12
```

训练按会话划分 train/validation/test，不按窗口随机打散同一会话。输出 checkpoint 记录数据会话列表、配置和训练历史。

## 3. 评估

```powershell
python scripts/evaluate_denoiser.py `
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
python scripts/run_learned_denoise.py `
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
- `python scripts/validate_release_configs.py`、核心测试和界面测试通过。

模型状态不代表硬件、会话质量或临床有效性已经验收。

## 5. SSVEP 目标频率保持检查

已下载的 Dong2023 数据集可以作为下游保持性 smoke test：

```powershell
python scripts/evaluate_ssvep_preservation.py `
  datasets/external/ssvep_datasets/dong2023_ssvep `
  --model models/denoiser_v1.pt `
  --max-subjects 2
```

该脚本比较去噪前后的目标频率幅度，不把幅度比直接称为识别准确率。正式发布前还需要使用独立被试、项目自己的 SSVEP CCA 流程和真实 Cyton 回放验证。
