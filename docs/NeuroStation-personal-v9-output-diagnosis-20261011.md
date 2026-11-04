# 自采会话 v9 输出诊断

日期：2026-10-11（Asia/Shanghai）。本轮范围是只读数据诊断、显式离线推理和输出完整性核验。

## 结论

两份原生 Cyton 会话都可以实际输出 v9 波形。本轮已经在 RTX 4090 上运行，而非仅检查格式。不能将用户此前未定位的“无输出”现象直接归因于这两份数据。

当前工作区有三项可解释输出缺失的实现事实，尚未取得用户出问题时的具体界面和日志，所以这些不是已证实的唯一故障原因：

1. `scripts/run_ssvep_test_model.py` 只接受 `[T,8]` 的 NPY、NPZ 或无表头数值 CSV；原生 `raw_brainflow.tsv` 含表头及包序号、时间戳等列。直接使用该 TSV 已实际复现 `supported source file required`，进入模型前被拒绝。
2. `eeg_tools/workstation/acquisition_worker.py` 在采集事件中将 `eeg_predicted_target_id`、`target_confidence` 写为空。空字段本身不证明识别失败。独立离线分析能为这两份会话各生成12条预测。
3. 工作站 `process_gateway.py` 仍从 `configs/learned_denoiser_v1.json` 创建可选监测进程；该旧配置 `runtime.enabled=false`、`validation_status=failed_holdout`。v9 独立离线入口没有接入桌面自动处理。未改动上述开关、质量门槛或历史结论。

## 实际输出

| 会话 | 输入样本/通道 | v9 已处理 | 原样保留尾段 | 单独 FFT 预测数 |
|---|---:|---:|---:|---:|
| session_20261004_004101_105 | 24,371 | 24,320 | 51 | 12 |
| session_20261004_004555_413 | 24,372 | 24,320 | 52 | 12 |

两份会话所有95个完整256样本窗口都已进入模型；保留的51/52样本仅因为尾段不足整窗。全部输出为有限值，形状与输入一致，输出数值确实发生变化。处理掩码和尾段原样保留已逐样本核验。

每份会话包含：

- `original_signal.npy`：源 EEG8列的只读派生副本；保持源数值和通道次序。
- `denoised.npy`：v9 实际GPU输出 `[T,8]`。
- `processed_mask.npy`：真正进入模型的样本位置。
- `report.json`：来源、模型、设备、窗口状态、源和输出SHA256。
- `raw_fft/analysis.json`、`raw_fft/trial_features.tsv`：现有透明FFT基线独立生成的分类输出。
- `fft_comparison.json`：原始波形和v9波形通过同一FFT基线的逐试次比较。

## 运行成功与效果区别

选用既有250Hz `Wearable` checkpoint分支是显式未验证的迁移探测。源数据仍标记为 `personal_Cyton`；没有把自采会话声称为Wearable公开数据。通道电极对应、单位增益校准和个人硬件域的模型效果均未验证。没有训练或搜索参数。

按记录中呈现目标比较，第一份原始FFT/v9后FFT匹配数均为3/12，第二份为2/12与3/12。第二份有1条预测改变。没有独立注视标注，两份会话来自同一参与者，因此不能当作已验证的注视准确率或独立受试者效果提升。

全段Welch功率诊断采用固定 `fs=250, nperseg=1000, noverlap=500, detrend=constant`。第一份各通道49–51Hz频段功率下降约1.71%–2.11%；第二份约1.13%–1.31%。这是对应频带功率变化，不是全部噪声去除率，也不保证生理信号被保留。当前v9对强50Hz成分的处理幅度较小。

## 完整性与范围

- 本轮两份源会话目录中所有文件均进行了前后SHA256复核，完全一致。
- 83项冻结代码/数据绑定和6项manifest绑定的并集共87项通过模型preflight；未改v8/v9配置、权重、锁或历史报告。
- `verification.json` 独立回读已写文件，核验shape、有限值、处理mask、原样尾段、输出和来源hash。
- 正式发布配置检查通过，保持 `formal`。
- 原生适配器测试23项及现有v9运行/交付和离线分析测试38项，共61项通过。本次并未声称重新执行以前的457项测试。
- 没有clean EEG成对真值，未执行采集硬件操作、实时启用或质量验收。

实际问题应先定位用户看到的输出位置。若是桌面采集结果，需要显式离线分析与结果显示链路；若是命令行，需要原生会话适配器。下一阶段去噪效果研究应把固定传统滤波基线与v9做同一试次对照，然后再决定是否训练个人域模型。独立注视标签和实际电极映射确认仍影响效果判断。

## 可核验文件

实验目录：`C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\personal_v9_output_diagnosis_20261011`

`plan.json` 是预执行约束，`summary.json` 是两会话汇总，`verification.json` 是回读完整性验证。原始数据目录未写入任何派生结果。


## 直接对原生会话运行

新入口 `scripts/diagnose_neurostation_v9_output.py` 读取会话目录，按列名提取八通道，严格检查行列与样本序号，然后调用冻结v9。无效、平直及不足整窗的处理策略沿用冻结运行时。输入校验失败会输出JSON错误，完成时打印输出目录和处理/保留样本数。适配器不会生成分类结果。

PowerShell：

```powershell
$projectRoot = 'C:\Users\Administrator\Documents\ChatGPT\脑机采集'
$session = 'C:\Users\Administrator\Documents\NeuroStation\Datasets\session_20261004_004101_105'
$output = Join-Path $projectRoot ('analysis_outputs\personal_v9_native_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
& (Join-Path $projectRoot '.venv-ml-gpu\Scripts\python.exe') (Join-Path $projectRoot 'scripts\diagnose_neurostation_v9_output.py') --session $session --output $output --device cuda:0 --acknowledge-out-of-domain
```

`--acknowledge-out-of-domain` 明确使用未验证个人域作为迁移诊断；它不更改任何正式配置或验收结果。CPU运行时可将 `--device cuda:0` 改为 `--device cpu`。每次使用新输出目录，避免覆盖历史结果。


新适配器已在真实RTX4090上通过命令行端到端验证：第一份会话处理24,320样本、原样保留51样本，三个数组与本轮独立探测逐元素完全一致。所有原生会话文件SHA保持不变，完成前冻结绑定复核通过。证据在 `native_adapter_gpu_smoke/report.json` 与 `adapter_end_to_end_verification.json`。
