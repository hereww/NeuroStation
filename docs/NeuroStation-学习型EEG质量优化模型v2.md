# NeuroStation 学习型 EEG 质量优化模型 v2

## 1. 状态与证据边界（2026-10-05）

v2 是独立的、显式离线运行的**单通道共享权重残差 TCN 研究模型**。工作站、Qt、采集协议、默认正式配置和原始质量判定不接入 v2；不得自动启用。配置的 `status=formal` 是项目配置状态，不是模型有效性、真实硬件可用性或临床验收。

本轮主线程已完成固定 12 epochs 训练、**一次**独立 public-test，以及独立记录的迁移/已知会话压力回归。**结论必须同时保留：半合成 epoch-holdout 研究 gate 通过；已知 v1 会话回归失败；Dong2023 真目标识别与目标幅度保持 gate 未评估。当前仍不能用于生产或正式信号替换。** 以下实测值直接引用本轮保存的 JSON，不以训练/validation 或 fixture 测试替代 test 成绩。

- 数据：`datasets/processed/learned_denoiser_v2.npz`。
- 审计：`datasets/processed/learned_denoiser_v2.npz.manifest.json`。
- NPZ SHA256：`862923f08bd700e87f32c5b0e99b715351c1080afb5bc6e2ffc282b6496dfa95`。
- v1/raw 预检快照：`analysis_outputs/learned_denoiser_v2_preflight_snapshot.json`，由主线程保存，v2 不改写该快照。

以下仅为实际 manifest 中的**数据计数**，不属于效果指标：

| 固定 split | 总样本 | identity/clean | EOG | EMG |
|---|---:|---:|---:|---:|
| train | 8400 | 3360 | 2520 | 2520 |
| validation | 1800 | 720 | 540 | 540 |
| test | 1800 | 720 | 540 | 540 |
| 合计 | 12000 | 4800 | 3600 | 3600 |

`test` 1800 窗已经冻结，索引为 `[10200,12000)`，本轮独立指标评估**已执行且仅执行一次**；报告记录 `test_inference_executed=true`、`repeat_test_authorized=false`。`--check-only` 仍可用于排查输入和元信息，不执行推理或产生指标。不得为调阈值、选 checkpoint 或选较好模型反复查询 test，也不因后续迁移/回归失败而重新调参或重测该 test。

### 1.1 实际训练与失败 attempt 的透明记录

`models/denoiser_v2.training.json` 记录训练完成：8400 个 train 窗、1800 个 validation 窗，固定 **12 epochs，best_epoch=12**，best validation loss 为 `0.255680742793613`。选 checkpoint 仅依据 validation，不使用 test；最终 validation-only 阈值为 `0.6250606731176377`，窗口 score 严格 `>` 才替换。

训练共两次 attempt，而不是两次 public-test：第一次在保存 checkpoint 时因 **NumPy scalar / 非原生 metadata 的序列化校验**失败，日志保留 `non-primitive checkpoint metadata`，当时未执行 test。之后只修复元信息保存，将相同 `seed=20261005`、相同架构、相同 loss、固定 12 epochs 的训练重跑成功，不根据 test 结果改模型或阈值。保留失败日志 `analysis_outputs/train_denoiser_v2_failed_attempt1.log`、失败前冻结记录 `analysis_outputs/learned_denoiser_v2_failed_attempt1_lock.json`、成功后 test 前冻结记录 `analysis_outputs/learned_denoiser_v2_experiment_lock.json` 与成功训练日志 `analysis_outputs/train_denoiser_v2_epoch_split.log`；冻结记录中的 `test_inference_executed=false` 描述其保存时点，不否认随后那一次已执行的 test。

### 1.2 固定 public-test：仅半合成研究 gate 通过

来源为 `analysis_outputs/learned_denoiser_v2_test.json`，scope=`fixed_test_only_epoch_disjoint_semi_synthetic`；数据与旁 manifest SHA、全局 fingerprint、源 epoch ID 和 checkpoint split indices 均已通过报告守卫，`validation_used_as_test=false`。模型 SHA256 为 `b1f9e336a7db91672a196b95a41b10e6b20362a065cd5b77d71fa1299c03e8e7`，旁 manifest SHA256 为 `90c0d778c841629729eef0b0aac2f12e2f92ec9eff57814723a9e2f029ad876f`。

| 全部 1800 个 test 窗轨迹 | RMSE（同一 clean AC RMS=1 数据尺度） |
|---|---:|
| noop / 原始 noisy | 0.7190381716 |
| candidate（未门控） | 0.3544523950 |
| gated（选择性替换） | 0.3548664201 |
| 固定 1–45 Hz bandpass | 0.5942854764 |

| 预声明 gated 判据 | 实测 | 固定门槛 | 结果 |
|---|---:|---:|---|
| all-test RMSE 改善 | **50.6471%** | ≥10% | 通过 |
| corrupted（EOG+EMG）SNR gain | **+6.13429 dB** | ≥1 dB | 通过 |
| EOG 自身 RMSE 改善 | 54.3160% | ≥0%，不得恶化 | 通过 |
| EMG 自身 RMSE 改善 | 47.0229% | ≥0%，不得恶化 | 通过 |
| clean 替换窗口比例 | **4/720 = 0.5556%** | ≤1% | 通过 |
| clean gated change_RMSE（clean AC RMS=1） | **0.00632381** | ≤0.02 | 通过 |
| corrupted 替换样点比例 | 75.9259% | ≥10% | 通过，非全 noop |

clean gated RMSE 同为 `0.00632381153679787`，identity 目标与 noisy 相等；归一化 change_RMSE 为 `0.00632381153676953`，不混同为硬件单位的失真阈值。candidate/gated 的全部分组与分段指标均留在原 JSON，不只展示较好者。报告的 `semi_synthetic_research_gate_passed=true` **仅表示这个 epoch/fingerprint 隔离的半合成 benchmark 通过预声明门槛**；`release_gate_passed=false`、`release=false`、`production_validated=false`、`automatic_processing_enabled=false` 不变。

### 1.3 迁移与已知 v1 会话压力回归：失败与未评估均不隐藏

完整新报告为 `analysis_outputs/learned_denoiser_v2_transfer_and_known_v1_regression.json`，使用同一冻结模型 SHA 和同一 validation-only 阈值，`test_tuned=false`、`training_performed=false`；它不是重新运行 public-test，也不用于调参。

**已知 v1 回归压力测试失败。** `known_v1_regression_stress_test` 复用旧会话 `session_20260924_131134_807` 的 26 窗。报告明确 `independent=false`、`heldout=false`、`seen_by_v1_experiments=true`，target 仍是 v1 半合成 clean-ish，而不是真实噪声的配对 clean 真值。

| 旧 26 窗轨迹 | 相对其半合成 target 的 RMSE（原始信号单位） |
|---|---:|
| noop / noisy | **1.0618389182** |
| candidate | **989.8789733472** |
| gated | **989.3381983238** |

gated 明显劣于 noop，报告 `acceptance_passed=false`。不能删除或淡化这个坏成绩，不能把旧 26 窗改称新独立 test，也不能用 public-test 的改善覆盖此域外回归失败。该组数值与 public benchmark 的 clean AC RMS=1 尺度不同，不进行跨报告 RMSE 的直接横向比较。

**Dong2023 的真目标保持 gate 为 `not_assessed`。** 固定 S2/S3，各 160 trials、合计 320 trials，使用经审计的刺激区间 `[125:1125)`（250 Hz，1000 点，4 秒）。timing=`timing_verified`，但 `target_axis_mapping_confirmed=false`；不能凭候选 mapping 或频谱猜测认证真标签。报告 `cca_accuracy_status=not_assessed`、`cca_top1_accuracy=null`、`target_labelled_amplitude_status=not_assessed`，`preservation_checks.status=not_assessed` / `passed=null`，不能写成真实识别率或目标幅度 gate 通过。

在不依赖真目标标签的描述性统计中：

- 整个频率集合 fundamental 的 **gated/raw ratio-of-means=`0.8727712125`**，显示均值幅度下降。它是频率集合统计，**不是真目标 amplitude ratio**；也不同于逐点 ratio 的算术均值（报告约 `0.954129`），不得挑换统计定义宣称保真。
- gated 与 raw 的 CCA 预测一致率为 **72.8125%**；仅表示预测稳定性，**不是正确率**，raw 本身可能预测错误。
- 该迁移区间内 replacement sample/window fraction 均为 **100%**：全部分析通道窗口被替换，gated 与 candidate 在此区间一致。半合成 clean 的 ≤1% 控制没有在这里证明成立；全替换、幅度下降和预测变化暴露信号保真风险，**不能用于生产或正式信号替换**。

### 1.4 当前安全处置与显式离线运行

已核对 Boole 落盘的 `configs/learned_denoiser_v2.json` 安全审计状态（`validation_date=2026-10-05`）：

```json
{
  "enabled": false,
  "validation_status": "failed_known_session_regression",
  "research_validation_status": "passed_epoch_holdout",
  "transfer_validation_status": "not_assessed_target_mapping",
  "deployment_scope": "explicit_offline_diagnostics_only",
  "production_validated": false
}
```

安全失败、半合成研究通过、transfer mapping 未评估是三个不同证据层级，不互相覆盖。`validation_reason` 保留已知旧 26 窗回归失败与 Dong agreement 非 accuracy 的限制，`validation_reports` 分别指向本轮 public-test 和 transfer/known-v1 JSON。数值模型、训练、冻结阈值和 checkpoint 不因这次 state-only 安全审计更新而改变；`runtime.enabled=true` 仍拒绝，**不得自动启用**。原始 test/transfer/离线报告中的生成时 config SHA 和状态（包括离线报告的 `model_validation_status=not_assessed`）不回写或伪装为后来配置。

显式诊断 CLI 与 `scripts/start_denoiser_v2_diagnostics.ps1` 可用，只表示操作员发起的离线管线可运行，**NOT 正式信号替换或生产验收**。启动器不自动打开 GUI/硬件、开始采集或光刺激，不训练、不下载、不跑 test；readiness 不代表有效性。

主线程已对会话 `session_20260924_131134_807` 显式离线运行成功，派生结果位于 `analysis_outputs/offline_diagnostics/learned_denoise_v2/session_20260924_131134_807_fixed_v2/`。其 `quality_metrics.json` 记录 `status=completed`、`source_hashes_unchanged=true`、13 个源文件 SHA，且 `diagnostic_only=true`、`automatic=false`、`release=false`、`missing_data_repaired=false`；主线程最终核验原 13 files hash 一致。**这些只证明非破坏性诊断运行完成，不证明模型在该会话有效**：同一已知会话的压力回归失败仍保留，不把导出成功称为 repair 成功。


## 2. 为什么 v1 失败后需要独立 v2

v1 文档与历史报告已经记录 heldout 重建显著劣于原始基线以及目标幅度保持失败；v1 自动推理保持关闭。不得删除失败成绩、重写旧会话结论或把较早的泄漏成绩当作有效泛化证据。

v2 针对这些失败风险改变了研究设计，不把改变本身当作成功证明：

1. **数据身份和泄漏**：只以公开 clean EEG epoch 为半合成目标，EOG/EMG 是独立噪声池。先按原始 epoch ID 与全局波形 fingerprint 分组，再重采样与混合；不把相同源行号当作同步配对，不拼接独立 epoch、不复制单通道冒充真实多通道。
2. **信号尺度**：模型以 noisy 窗的 float64 均值和 AC RMS 归一化，残差回到原始输入尺度后输出。candidate、gated、noop 和固定传统 baseline 都与相同 clean、相同 valid 支持比较；不能逐方法再归一化掩盖缩放错误。
3. **干净片段保护**：强化 identity 损失；只用 validation 的 clean 窗对伪迹窗口 score 校准选择阈值。不能用 test 的错误、替换率或信噪比调阈值。
4. **防止“全不替换即安全通过”**：同时报告未门控 candidate 与 gated；corrupted 替换比例必须至少 10%，并满足 RMSE 与 SNR 门槛。全 noop 不得通过。

这些是可审计设计变更，不是对 v1 单一根因的证明，也不是 v2 在真实 Cyton 数据上已有效的结论。

## 3. 数据、分割与许可

每窗为 2 秒：源数据 256 Hz / 512 点，经 `resample_poly(125,128)` 得到 250 Hz / 500 点。clean AC RMS 统一为 1 个数据单位，伪迹按预声明 SNR 范围半合成；identity 输入严格等于 clean。

NPZ 必需字段：

```text
clean/noisy/valid_mask/artifact_mask: [N,1,500]
split: [N], train | validation | test
artifact_type: [N], clean | eog | emg
clean_epoch_id/eog_epoch_id/emg_epoch_id: [N], 原始源行号；未使用噪声 ID=-1
```

manifest 的实际结构为：

- `output.sha256`：NPZ SHA256；
- `epoch_split_manifest.fingerprint_groups[]`：`fingerprint`、`split`、`epoch_ids_by_source`；
- `epoch_split_manifest.source_statistics.<eeg/eog/emg>.raw_epoch_ids_by_split`：完整原始源 ID 池；
- `fingerprint_namespace=global_across_EEG_EOG_EMG`，跨三种源的重复波形不能进入不同 split；
- `npz_contract`：实际数组 shape/dtype；`split_statistics`、`split_sample_targets`：样本计数；
- `sources`：源路径、SHA、样本率证据和来源；许可另查外部数据审计，不从缺失字段推定授权。

这里的隔离是 **epoch-disjoint / fingerprint-disjoint**，不是 subject-disjoint。公开 epoch 池没有可用被试 ID，因此不得宣称跨被试或独立被试泛化；半合成伪迹也不等同真实同步采集噪声。

许可按现有外部数据审计和 v1 文档约束执行：EEGdenoiseNet 底层数据许可未单独澄清时按**许可未知**处理；代码仓库 MIT 不自动授权底层 EEG 数据。Dong2023 按已有审计的 **CC BY-NC 4.0（非商用）**限制，其他许可未知数据不推定可商用。不得把这些原始数据或派生数据随桌面发布包再分发；模型对外发布同样须另行核实数据授权。技术上下载、可读或 hash 匹配不等于获得商业授权。

### 本轮新增 Klados 有限样本审计（不是新的 paired 训练集）

来源为 `datasets/external/v2_metadata_audit/V2_METADATA_AUDIT.json` / `.md`，以及其 `payload/klados2016_semisimulated_eeg_eog/DATASET_MANIFEST.json`、`ARCHIVE_INTEGRITY.json`、`MEMBER_VERIFICATION.json`。官方 Klados/Bamidis archive 为 **46,757,186 bytes**，大小和 SHA 匹配上游记录；但内部 **`Pure_Data.mat` CRC 失败 / unexpected EOF**，UnRAR 检查 exit code=3。因此来源 hash 正确不代表完整配对 clean/noisy 数据可用。

仅保留 archive 与两个 CRC/size/MAT parsing 核验完整的成员：`Contaminated_Data.mat` **28,330,804 bytes**、`HEOG.mat` **1,198,492 bytes**，共 **76,286,482 bytes**。未提取或保留损坏的 clean member，不把这次下载宣称为 paired 监督训练集完成；clean target 缺失，采样率、电极名称和物理单位仍未独立核实，**本轮不用于 paired 训练**。

Klados 官方数据审计明确记录 **CC BY 4.0**，须保留归属与第三方内容 caveat；这不改变 EEGdenoiseNet 的许可未知或 Dong 的非商用限制，不推定其他数据获授权。本轮这些文件仅作研究审计/有限结构读取，仍不随桌面包发布或自动接入模型；数据与模型对外使用须分别核实授权范围。


## 4. 已确认的 core/checkpoint 合同

核心 API：

```text
eeg_tools.learned_denoise_v2.MODEL_VERSION = learned_denoise_v2
LearnedDenoiseV2Error
load_config(path=None), build_model(config)
load_model(path, config, device=cpu) -> ModelBundle
apply_model_to_array(bundle, signal[C,T], valid_mask=None) -> DenoiseResult
```

支持任意 C，各通道独立共享同一个单通道模型，不跨通道混合统计量。window=hop=500、sample_rate=250。完整 500 点窗的 score 是该通道 `artifact_probability.mean()`，只在 **score 严格大于 calibrated_threshold** 时替换；threshold=1 不替换。默认 `max_invalid_fraction=0.0`：不完整窗不推理替换，尾部不足一窗保持原样并标记 rejected；缺失、常量、坏输出不能自动宣称修复。

`DenoiseResult` 为 dataclass，以下均为 `[C,T]`：`denoised`（gated）、`model_denoised`（candidate）、`artifact_probability`、`confidence`、`replacement_mask`、`valid_mask`、`rejected_mask`，另有 `diagnostics`。`confidence=1-probability` **不是校准的成功概率**，也不是修复成功证明，不用作门禁。

`ModelBundle.metadata` / `checkpoint_metadata` 保存原生 scalar/list/dict；`calibrated_threshold` 为 canonical 字段，`validation_threshold` 为相同值的兼容别名。checkpoint 必需元信息：

| 字段 | 评估所需约束 |
|---|---|
| `checkpoint_schema_version` / `model_version` | `1` / `learned_denoise_v2` |
| `config` / `architecture` / `model_state` | core 严格核对配置、网络结构与权重；加载只用 `weights_only=True`，不降级为任意 pickle |
| `dataset_sha256` | 必须等于输入 NPZ SHA 与 manifest `output.sha256` |
| `dataset_manifest` | 原样嵌入完整 manifest，必须与旁文件内容一致 |
| `dataset_manifest_sha256` | 必须等于旁文件实际 SHA |
| `train_indices` / `validation_indices` | 分别完整且精确对应 NPZ 的 train / validation；不能重叠或包含 test |
| `selected_train_indices`（如存在） | 只允许 train 子集，不改变完整 split；不是新的 test 分割 |
| `calibrated_threshold` / `validation_threshold` | 有限 [0,1]、两值一致、test 不修改 |
| `threshold_calibration` | 仅 validation-clean 的经验阈值，详见下方 |
| `history` / `best_epoch` / `best_validation_loss` | 训练记录；checkpoint 选择只使用 validation，不使用 test |

`threshold_calibration` 的已确认字段：`scope=validation_only`、`comparison=>`、`method=clean_score_order_statistic`、`score_reduction=per_channel_window_mean`、`max_clean_false_replacement_fraction=0.01`、`clean_window_count`（完整 validation clean 数量）、`observed_clean_false_replacement_fraction<=0.01`、`empirical_only=true`、`production_calibrated=false`。不允许缺少元信息时猜测阈值、被试 ID、fingerprint 或通过记录。

评估复用训练脚本的实际 `validate_dataset_manifest` 校验源 epoch/fingerprint 分割；复用 core `validate_checkpoint_metadata` 核对结构与校准，再额外核对 checkpoint 完整索引、数据尺度、数组合同与计数。普通工作站不因这些 CLI 引入 PyTorch 必需依赖。

## 5. test 执行前预声明的研究门槛

常量写在 `scripts/evaluate_denoiser_v2.py::RESEARCH_THRESHOLDS` 并完整保存到报告；执行后不按 test 成绩修改。

| gated 判据 | 固定门槛 |
|---|---:|
| all-test RMSE 改善，`1 - gated_RMSE/noop_RMSE` | ≥10% |
| corrupted（EOG+EMG）SNR gain | ≥1 dB |
| EOG 自身 RMSE 改善 | ≥0%，不得恶化 |
| EMG 自身 RMSE 改善 | ≥0%，不得恶化 |
| clean 被替换窗口比例（任一样点替换即算该窗替换） | ≤1% |
| clean gated change_RMSE，clean AC RMS=1 单位 | ≤0.02 |
| corrupted 替换样点比例 | ≥10% |

所有判据同时通过才可写 `semi_synthetic_research_gate_passed=true`；缺失组、空 valid 或缺失数值不得通过。candidate 的成绩无论优劣都完整保留，不选择 candidate/gated 中较好的结果单独呈现。`release_gate_passed=false`、`release=false`、`production_validated=false`、`automatic_processing_enabled=false` **始终保持**。

传统比较基线为固定 1–45 Hz、4 阶 Butterworth、250 Hz、`sosfiltfilt`，不在 test 上选择参数，不要求模型胜出传统 baseline 才允许运行。全量 test、clean、EOG、EMG 和 corrupted 都保存四条轨迹的 RMSE、relative_RMSE、correlation、SNR gain、mean replacement、替换窗口比例、clean false replacement 和 clean signal distortion，以及伪迹/非伪迹分段 RMSE。信号尺度与 valid 支持一致；退化到无限 SNR 的值写 null，不把缺少可比较数值视为通过。

## 6. 命令（不自动启用工作站）

所有命令用项目 Python；下方保留复现方式，本轮训练与那一次 public-test 已由主线程执行。**不要为复现说明重新执行或覆写已完成的 test**；现阶段排错只用 `--check-only`，原报告保持不变。

```powershell
# 可复现 builder；现有冻结数据不要无理由重建或 --overwrite。
& .\.venv\Scripts\python.exe scripts/build_denoise_dataset_v2.py `
  --output datasets/processed/learned_denoiser_v2.npz --samples 12000

# 主线程负责训练；只读 train/validation 内容，test 不算指标。
& .\.venv\Scripts\python.exe scripts/train_denoiser_v2.py `
  --dataset datasets/processed/learned_denoiser_v2.npz `
  --config configs/learned_denoiser_v2.json --output models/denoiser_v2.pt `
  --epochs 12 --threads 2

# 可继续使用的元信息诊断；不推理、不写或覆盖 test 成绩。
& .\.venv\Scripts\python.exe scripts/evaluate_denoiser_v2.py --check-only --threads 2

# 本轮已由主线程执行一次，以下仅存档命令，不要再次运行或 --overwrite。
& .\.venv\Scripts\python.exe scripts/evaluate_denoiser_v2.py `
  --dataset datasets/processed/learned_denoiser_v2.npz `
  --model models/denoiser_v2.pt --config configs/learned_denoiser_v2.json `
  --output analysis_outputs/learned_denoiser_v2_test.json --threads 2
```

默认路径与 builder 一致，model 默认 `models/denoiser_v2.pt`。已有报告默认拒绝覆写，并在推理前以独占方式预留输出文件；失败/中断的预留文件也保留，以阻止静默重复 test。排错优先 `--check-only`。只有明确 `--overwrite` 才能再次测试并覆写该报告，且报告记录 `repeat_test_authorized=true`；这个开关**不是**允许反复调参查询 test 的许可。

评估默认 `--batch-size 256`，将独立 `[1,500]` 窗视作一批 `[C,500]` 输入，避免每窗单独 torch 调用。eval 与 offline 支持正整数 `--threads`，默认 2；optional torch 导入之前设置 OMP/MKL 默认环境，之后显式 `set_num_threads`，避免 32 logical CPU 对短窗模型的线程调度开销。这个参数是性能参数，不是效果门槛。

### 显式离线诊断

独立启动器支持只读 readiness 及操作员显式 session 诊断：

```powershell
& .\scripts\start_denoiser_v2_diagnostics.ps1 -CheckOnly
& .\scripts\start_denoiser_v2_diagnostics.ps1 `
  -SessionDir '<existing-session>' -OutputDir '<new-directory-outside-session>' -Threads 2
```

`-CheckOnly` 不处理会话；无明确 session 时不自动推理。路径按项目根解析，先检查正式采集配置与模型，诊断 readiness **不是**研究/生产门槛通过。也可直接使用 Python CLI：

```powershell
& .\.venv\Scripts\python.exe scripts/run_learned_denoise_v2.py `
  --session <session-directory> --output-dir analysis_outputs/v2_session_diagnostic --threads 2

& .\.venv\Scripts\python.exe scripts/run_learned_denoise_v2.py `
  --input <signal.npz> --output-dir analysis_outputs/v2_npz_diagnostic --threads 2
```

兼容 positional session 目录，但不得与 `--session` / `--input` 混用。输出必须是**新建的、源会话之外的独立目录**，不写回会话 `derived`、raw、quality、session.json。省略 `--output-dir` 时在 `analysis_outputs/offline_diagnostics/learned_denoise_v2/` 下使用唯一运行目录，不覆写历史报告。

会话输入先验证 `session.json.sampling_rate_hz` 明确为 250 Hz，再只读复用 v1 `read_session_eeg` 与 continuity audit；不使用 v1 缺省采样率，不把导出 `sample_index` 当 packet-loss 证明。缺失连续性证据、packet 跳变、异常时间戳或采样率不符时拒绝，不自动重采样/补洞。

offline NPZ 必需：恰好一种 `signal` / `eeg` / `raw` 数组 `[C,T]`、标量 `sample_rate_hz=250`、可选同形 boolean `valid_mask`；连续性证据是 `[T]` Cyton `package_num`（0..255 顺序回绕），或 `[T]` `timestamp_s` 的连续 250 Hz 网格。有 packet 证据时允许 host batch 重复时间戳，但时间倒退和明显长间隙仍拒绝。仅存数组、仅导出计数器而无连续性证据的 NPZ 不接受；此 CLI 的 NPZ 是离线采集轨迹，不是直接传入 `[N,1,500]` 的 benchmark 文件。

输出包括：`denoised_signal.npz`（raw + 七个同形结果字段）、`denoised_signal.csv`、`quality_metrics.json`、`model_manifest.json`。JSON/NPZ 明确记录 diagnostic_only、automatic=false、release=false；记录输入/模型 SHA、连续性审计及 rejection，不把未知噪声自动宣称为已 repair。`status=completed` 仅代表诊断管线完成，不代表有效性或生产通过。源文件前后 hash 校验，历史 quality 不改写。

## 7. fixture 测试与后续真实验收

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.test_learned_denoise_v2_evaluation -v
```

本节只列 fixture 复现命令与覆盖范围，**暂不登记全局测试最终数量**，待主线程汇总。测试仅用临时合成源与 mocked inference，覆盖 epoch/fingerprint/test 隔离、NPZ/manifest/checkpoint hash 守卫、full indices、calibration 固定范围、重复报告保护、分组/分段指标、尺度一致、全 noop 拒绝、独立输出及 raw/quality/session immutable。

半合成 gate 即使通过，也不满足真实多通道同步采集、未知真实噪声、跨被试、硬件现场回放、SSVEP 识别保持性、CPU/内存、光刺激安全或临床有效性验收。外部迁移/SSVEP 报告必须独立保留其数据和目标元信息，不与本 benchmark 门槛混称。v1 文件、旧报告、历史会话与正式采集协议均不改写；没有任何自动工作站部署入口。将来要启用仍需独立有效性与安全验收，不能只切配置 enabled 或 formal 标签。
