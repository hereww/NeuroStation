# NeuroStation 多通道选择性 EEG 质量优化模型 v1

## 目标

本实现提供一个非破坏性的、可回放的学习型 EEG 质量优化管线：

- 原始 `raw_brainflow.tsv` 永远不修改；
- 模型输出保存到 `derived/learned_denoise_v1/`；
- 同时输出去噪波形、伪迹概率、置信度和质量报告；
- 缺失样本、NaN/Inf、长时间异常不由模型伪造恢复；
- 配置仍为 `status=formal`、`inference.mode=shadow`，但当前 checkpoint 已失败：默认 `runtime.enabled=false`，只允许显式离线诊断，不自动处理正式会话或替换正式分析轨迹。

## 当前有效性结论（2026-10-05）：失败，自动推理关闭

主代理已按冻结代码完成 `seed=20261005`、12 epochs、2427/54/26 分组训练，并对固定 26 窗 test **只评估一次**。不改变 test 分组，不根据这次分数调参，也不回滚到旧泄漏模型。

| 检查 | 实际结果 | 解释 |
|---|---:|---|
| heldout noisy RMSE | **1.06184** | 半合成 clean-ish 目标下的原始基线 |
| heldout denoised RMSE | **698.374** | 明显劣于原始基线，未通过有效性门槛 |
| heldout artifact F1 | **0.024247** | 不能证明有效伪迹识别 |
| Dong2023 S1 去噪/原始目标幅度比 | **0.88484** | 目标幅度下降；不是识别准确率 |
| Dong2023 S10 去噪/原始目标幅度比 | **0.84479** | 目标幅度下降；不是跨被试验收 |

完整失败结果原样保存在 [固定 test 评估报告](../analysis_outputs/learned_denoiser_v1_heldout.json) 和 [SSVEP 目标幅度保持报告](../analysis_outputs/ssvep_preservation_v1_session_split.json)。**本地管线部署完成不等于模型有效或生产验收；当前模型明确未通过有效性门槛。** 保留当前 checkpoint 与 split 用于复现，不隐藏、覆盖失败成绩或用旧 23.45% 替代。

安全状态固定为 `runtime.enabled=false`、`runtime.validation_status=failed_holdout`，配置记录原因、报告路径及 `deployment_scope=explicit_offline_diagnostics_only`。当前 checkpoint **只能用于操作员显式发起的离线诊断，不能自动处理正式会话**。正式采集、原始质量链路及安全确认不因模型失败而取消。

## 本地源运行部署（自动 shadow 当前禁用）

已实现的可选接入机制是**采集结束后异步 shadow**，不是实时逐窗口部署。以下仅描述机制，不代表当前允许开启；默认禁用时不会启动学习型监视进程。将来启用仍须独立完成有效性与安全验收：

1. 正常启动采集工作站，不自动启动采集或光刺激。
2. 采集开始后，网关启动一个独立监视进程，等待该次采集的终态状态文件。
3. 正式采集完成或中止并保存会话后，监视进程才读取 `raw_brainflow.tsv` 并在 CPU 上推理。
4. 结果只写入该会话的 `derived/learned_denoise_v1/`；UI 原始波形和正式质量判定保持原有链路。

只检查基础管线环境、正式配置及审计状态，不打开 GUI、不打开硬件；当前禁用自动 shadow 时不加载 checkpoint：

```powershell
& .\scripts\start_learned_workstation.ps1 -CheckOnly
```

基础管线检查通过后可打开普通工作站（模型自动 shadow 保持关闭；仍需操作员手动启动采集并完成安全确认）：

```powershell
& .\scripts\start_learned_workstation.ps1
```

入口固定解析项目 `.venv\Scripts\python.exe`，先运行正式 release 配置验证，再检查 NumPy、SciPy、BrainFlow、PySide6 等基础依赖。`-CheckOnly` 明确输出 `status=pipeline_ready`、`pipeline_ready=true`、`automatic_shadow_disabled=true`、`model_validation=failed` 与 `validation_status=failed_holdout`；**pipeline_ready 只说明基础管线可运行，不是模型合格**。自动 shadow 关闭时，PyTorch 与 checkpoint 可缺失，不加载模型、不阻止普通 GUI；若误将失败模型的 enabled 单独改为 true，入口会拒绝自动 shadow，而不是绕过失败状态。脚本不安装依赖、不下载数据、不训练模型、不打开串口、不自动触发光刺激。

运行配置保持 `status: formal`、`inference.mode: shadow`。可在 `configs/learned_denoiser_v1.json` 中设置：

```json
"runtime": {
  "enabled": false,
  "model_path": "models/denoiser_v1.pt",
  "validation_status": "failed_holdout",
  "deployment_scope": "explicit_offline_diagnostics_only",
  "production_validated": false
}
```

`enabled: false` 是当前默认安全状态，关闭自动后台推理；不能仅因 checkpoint 可加载或配置 formal 就改为 true。模型不存在或配置不适用时，网关跳过可选监视进程，不阻止原始采集。模型加载、输入或推理失败时，已启动的 worker 保存独立 fallback 状态，不将原始质量改为通过。启动采集失败时不启动监视进程；工作站关闭时取消等待或推理中的可选进程；UI tick 仅 poll、不 wait。

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

当前 checkpoint 仅保留用于显式离线诊断，不能自动处理正式会话。旧 `analysis_outputs/learned_denoiser_v1.json` 的 **23.45%** 来自对全量数据（包含训练窗口）的诊断，**不得引用为独立泛化、跨会话或跨被试结果**；旧 checkpoint 还存在 validation/test 重叠问题。旧模型与报告保留在 `models/history/` 和 `analysis_outputs/history/`，不覆盖或改写其历史结论。本轮新分组训练及一次固定 test-only 评估已完成，新报告 `analysis_outputs/learned_denoiser_v1_heldout.json` 记录失败结果（见上述结论），不得用旧诊断数字填充或替代。

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

## 1. 构建本地会话与公开数据半合成训练集

**以下是后续复现命令，不要在本轮训练期间执行：当前 `datasets/processed/learned_denoiser_v1.npz` 已用于训练，不能运行构建命令覆盖该 NPZ。** 若需重新构建，另行安排并使用新的输出路径，保存现有数据及其 SHA256。

```powershell
& .\.venv\Scripts\python.exe scripts/build_denoise_dataset.py `
  .tmp_preview `
  --output datasets/processed/learned_denoiser_v1.npz `
  --max-windows 20000 `
  --eegdenoisenet-dir datasets/external/eegdenoisenet `
  --external-windows 2400 `
  --seed 20261005
```

当前生成器从相对干净的本地会话窗口构造工程上的 clean-ish 目标，并注入眼动样、肌电样、工频、漂移和电极突变伪迹。它不是临床真值，也不是实际噪声下同步配对的 clean 真值。本轮已构建的 2507 窗包含 EEGdenoiseNet 半合成窗口 2400 个及本地 clean-ish 窗口 107 个。

**EEGdenoiseNet 数据边界：**下载的 clean EEG、EOG、EMG 是独立单通道 epoch 池，没有可用于本轮划分的 subject ids、真实 session ids 或官方受试者划分。生成器把同一个单通道 clean epoch 重复成 8 通道，再对部分通道叠加不同增益的合成伪迹；这只是适配模型输入的半合成训练，**不是真实同步 8ch 采集，不提供真实空间关系或跨被试验收证据**。所有外部窗口统一使用 `EEGdenoiseNet` 这个数据源 group，不能把它解释为一个已知受试者或真实会话。

## 2. 训练

```powershell
& .\.venv\Scripts\python.exe scripts/train_denoiser.py `
  datasets/processed/learned_denoiser_v1.npz `
  --output models/denoiser_v1.pt `
  --epochs 12 `
  --seed 20261005
```

本轮固定 `seed=20261005`、`epochs=12`。训练按 `session_id` group 划分 train/validation/test，不随机拆分同一 group 的窗口；真实 group 少于 3 个直接报错，不再用随机窗口或复制 validation 伪装独立 test。旧实现导致的 validation/test 重叠已经修复，三个分组严格不交叠。

分配策略为 `session_group_window_aware_fixed_test`：保持 seed 已选出的 test group 不动，确保最大 group 进入 train，仅通过 train/validation 整组交换尽量接近目标窗口比例；同大小保持 seed 随机性。本轮分组为：

| 分组 | group 数 | 窗口数 | 实际窗口比例 | 数据范围 |
|---|---:|---:|---:|---|
| train | 3 | 2427 | **96.81%** | EEGdenoiseNet 全部 2400 窗，以及两个本地 group 的 27 窗 |
| validation | 1 | 54 | **2.15%** | `session_20260928_141515_425` |
| test | 1 | 26 | **1.04%** | `session_20260924_131134_807`，保持原选定 test 不变 |

**70/15/15 只是目标，不是本轮实际比例。** 最大 group 的 2400 窗本身已占全量约 95.73%；在整组不可拆、最大组进入 train、test 固定及保留三个非空分组的约束下，不能实现理想比例。不能把 3/1/1 group 数误写成 70/15/15 窗口比例，也不能用 test 分数重新选择分组、调参或选 checkpoint；训练仍只用 validation loss 选择 checkpoint。

输出 checkpoint 保存 dataset SHA256、完整 split metadata（策略、seed、初始分组、各 group 窗口数、实际窗口比例、逐窗口 session 与各分组窗口索引）、模型配置和训练历史。完整 split 同时写入模型旁的 `models/denoiser_v1.split.json`；控制台只显示简略 metadata，不展开数千行窗口索引。

## 3. 评估

```powershell
& .\.venv\Scripts\python.exe scripts/evaluate_denoiser.py `
  datasets/processed/learned_denoiser_v1.npz `
  --model models/denoiser_v1.pt `
  --output analysis_outputs/learned_denoiser_v1_heldout.json
```

上方命令是复现方式；本轮已完成一次评估，不因失败而重复评估或调参。默认 **test-only**：只选择 checkpoint 的 `test_sessions`，排除 train/validation。评估先拒绝任何 train/validation/test group 重叠，checkpoint 存在 dataset SHA256 时必须与输入 NPZ 匹配，并核对完整 split metadata；旧重叠 checkpoint 不允许当作 heldout 评估。旧 checkpoint 缺少 hash 时会明确标注数据身份未验证，不提升其证据级别。

报告记录 `evaluation_scope`、session groups、窗口数与数据身份。新独立报告路径为 `analysis_outputs/learned_denoiser_v1_heldout.json`，实际失败指标已在本文开头如实登记。结果仍是对半合成 clean-ish 工程目标的评估，不能称为真实噪声 clean 真值验证、真实同步多通道验收或跨被试验收。

全量评估必须显式传入 `--diagnostic-all`，其 scope 为 `all_diagnostic_not_heldout`、`heldout=false`，不得引用为泛化结果；该开关也不绕过分组重叠及 hash 检查。完整窗口索引可保存在 `--output` JSON 中，控制台不展开。

评估至少应与以下三种轨迹比较：

1. 原始轨迹；
2. `eeg_tools/denoise.py` 的传统处理轨迹；
3. 学习模型轨迹。

另外必须在真实未参与训练的会话上比较 SSVEP CCA 目标频率保持情况，尤其是 10/12/15/20 Hz 和谐波，不得只看 RMSE。

## 4. 显式离线诊断（不得用于自动处理正式会话）

下方命令只用于指定历史数据的离线复现；会生成派生诊断输出，不代表去噪有效，不恢复自动后台处理。显式离线 CLI 不因 `runtime.enabled=false` 而被禁止，但 `quality_metrics.json` 与 `model_manifest.json` 必须记录 `model_validation_status=failed_holdout`、`automatic_processing_enabled=false`、`diagnostic_only=true`、`release_gate_passed=false` 及失败 warning/报告路径。`status=completed` 仅表示诊断管线执行完成，不表示有效性通过。

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

下载状态、来源、已核验 hash、缺失项和许可限制见 [外部数据集下载审计汇总](../datasets/external/DATASET_DOWNLOAD_SUMMARY.md)。技术上可读、已下载或 hash 匹配不代表获得商用或再分发授权。

- 只使用 manifest 中明确记录来源、许可、下载日期、SHA256 和缺失项的数据；
- EEGdenoiseNet 的底层数据许可未单独声明，按**许可未知**处理；软件仓库 MIT 许可证不自动覆盖底层 EEG 数据，许可澄清前不得宣称可商用或自由再分发；
- Dong2023 为 **CC BY-NC 4.0，非商用**；Nakanishi2015 的上游许可未知。不得把未知许可或非商用数据当作已获商业授权；
- 当前下载的外部原始数据**不再分发、不随模型或桌面发布包打包**；未知许可或非商用数据不得通过本项目转发，后续任何对外使用须另行核实来源条款与授权范围；
- EEGdenoiseNet 的 clean/EOG/EMG 是独立 epoch 池，应按伪迹合成方案构造训练输入，不把相同行号当成同步配对；
- 外部数据只用于模型训练/验证，不覆盖本项目真实采集记录的历史结论。

## 正式版本门槛

当前有效性结论为失败，自动 shadow 必须保持关闭。无论将来重新启用自动 shadow，还是把 `inference.mode` 从 shadow 改为正式分析，都必须先独立完成以下验收；不能只改 enabled、formal 状态或验证标签：

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
