# NeuroStation v5：内部合成噪声诊断与独立安全审计

- 文档状态：`formal`；日期：2026-10-07。
- 原 v5 候选仍不通过研究验收，`runtime.enabled=false`、`production_validated=false`。
- 本轮没有新训练、权重更新、epoch 选择或参数选型；使用旧 checkpoint 做固定 GPU 推理诊断。
- 本轮未访问 D 盘，未读取 outer validation 7–8 的波形或 holdout 9–10。

## 1. 已保存验证结果的独立逐试次审计

审计工具只读取固定 JSON，不导入旧解码器，不读 NPZ，不重新加载模型。直接以保存的 trial identity、标签、预测和目标分数重算，未把原 summary 当成独立证据。

| 指标 | 基线 | v5 候选 |
|---|---:|---:|
| 正确预测 / 试次 | 443 / 840 | 442 / 840 |
| Micro accuracy | 52.7381% | 52.6190% |
| Macro accuracy | 57.4306% | 57.3264% |

- 预测分歧：19 / 840 = 2.2619%，超过预注册的 2% 上限。
- 目标频谱分数比例均值：0.9906688697853835；全部 840 项均可从保存的分数重算。
- 各 subject、dataset、electrode domain 的 micro/macro 和比例核对一致，summary mismatches 为 0。
- 新审计按预注册的“下降”执行准确率门，不把准确率提升误当失败；历史结果与阈值没有被改写。
- `audit_consistent=true` 仅表示审计一致；`research_acceptance=false`，不表示模型通过。
- 该审计没有从原波形重新计算 FFT，不能替代信号级复核。

## 2. 冻结后的内部合成诊断

| 项目 | 固定设置 |
|---|---|
| 数据 | Nakanishi2015、Wearable，各 subject 5–6 |
| Recorded-reference trial 数 | 840 |
| 条件 | identity、transient、low_frequency_drift、high_frequency_burst、common_mode |
| 总 case 数 | 4,200；每条件 840 |
| 窗口 | 256 samples × 8 channels；Nakanishi 从 sample 38 起，Wearable 从 sample 125 起 |
| 采样率 | 原生 256 / 250 Hz；不重采样 |
| 推理 | CUDA:0，RTX 4090，float32 model forward，AMP=false，batch=64 |
| 还原 | noisy_float64 + (prediction_float64 − normalized_noisy_float64) × noisy_rms |
| 模型 | 原冻结 v5 两域模型；checkpoint SHA 不变 |

**重要：这是 window 级前向，不是完整 Hann overlap-add pipeline，也没有任务解码或任务保真验收。**

误差以 recorded reference 为重构目标，每通道用 reference 方差 + 1e-8 归一化，再对通道和样本平均。误差下降比例为 `(mean_input_error - mean_output_error) / mean_input_error`，不是逐 case 百分比的平均。identity 的输入误差为 0，因此比例为 null。此处的 reference 不是 clean EEG 真值，不能将结果解释为真实 EEG 去噪精度。

| 合成条件 | 输入平均归一化 MSE | 输出平均归一化 MSE | 平均误差变化 |
|---|---:|---:|---|
| 瞬态 | 0.034081291 | 0.031055964 | 下降 8.88% |
| 低频漂移 | 0.240591873 | 0.204221284 | 下降 15.12% |
| 高频突发 | 0.013845339 | 0.015943411 | 增加 15.15% |
| 共模 | 0.089584303 | 0.076876195 | 下降 14.19% |
| 未额外加噪 | 0.000000000 | 0.002684350 | 不适用；模型仍改写输入 |

- 高频突发条件：840 项中仅 90 项误差下降，750 项增加；两个数据域的均值均变差。
- identity 条件：840 项全部发生非零改写，平均归一化误差为 0.0026843499724495934。该数值没有预注册接受阈值，只作内部描述，不能事后设置阈值宣称通过。
- 含 identity 的全局均值误差下降约 12.52%，但受各类噪声幅度影响，不能掩盖高频突发变差和 identity 改写。
- 主会话另从保存的 4,200 个 case metrics 复核唯一身份、seed、窗口和 42 个聚合 scope，零不一致。此复核是聚合重算，不是再次从波形计算模型误差。

## 3. 研究边界

1. Subject 5–6 已用于原 v5 的 epoch 选择；本诊断不是独立测试，也不是新的 holdout。
2. 合成噪声误差不等同于真实眼电、肌电、运动伪迹的去除效果。
3. 原始 SSVEP 和 wet 电极都不是天然 clean/noisy 配对真值。
4. 不因内部平均指标变好而撤销冻结的 outer-validation 失败结论。
5. 未打开 holdout；旧候选没有接入采集工作站。
6. 正式文档/协议状态不代表硬件、会话质量、临床有效性或模型部署自动验收。

## 4. 新增工程安全能力

- 固定输入路径和仅 subject 5–6 的诊断加载入口，拒绝 cohort、device、配置或 reference 重定向。
- 协议和源码在首次本轮波形加载前冻结；18 个绑定项包含原输入、协议及三个诊断源码/测试文件。
- `--check-only` 仅读 JSON/source/sidecar 和二进制 stat，不读 NPZ/PT bytes、不初始化 CUDA；未执行的检查明确列出。
- 真运行校验选定 NPZ/checkpoint hash，CPU weights-only 反序列化后严格检查 checkpoint key、配置、state key/shape/dtype/finite，再移到 GPU。
- 新输出采用 no-clobber 原子创建，拒绝越界路径、reparse/symlink 和覆盖现有文件；JSON 与 Markdown 每文件原子发布，不宣称跨文件事务。
- 原 v5 训练器、配置、模型、评估器、锁和历史报告均未改写。本轮安全增强只作用于新增审计/诊断工具，不应宣称历史训练器的 publication race 等问题已经修复。

## 5. 验证结果

- 独立审计测试：37 项，36 通过、1 跳过（Windows 真实 symlink 创建受限；模拟拒绝测试通过）。
- 内部诊断合成测试：32 项全部通过。
- 全核心测试：656 项，652 通过、4 跳过、零失败。
- 界面测试：65 项全部通过。
- 正式配置检查：`ready`、协议 `formal`，无 warning。

## 6. 下一版建议与停止线

下一版应另立实验，并仅以 train/internal 数据设计和选择：
1. 在固定合成条件中改善高频突发的重构误差，采用按噪声类型/强度分别报告的设计，避免全局平均掩盖退化。
2. 增强未额外加噪输入的保持约束，明确加入任务相关频谱与相位保真检查；具体实现、超参数和接受阈值须先冻结。
3. 新训练入口补齐完整配置/状态校验和独占输出发布，不直接复用旧训练器的有风险发布流程。
4. 继续用 GPU 训练，但在读取下一次评价波形前冻结新候选、选择规则、配置、源码、输入身份和验收门。
5. 外层 subject 7–8 已被研究筛查，后续复用必须标注 known validation，不能称全新独立验收。
6. 只有新的候选和一次性评价规则完全冻结、且具备合理验收路径后，才考虑一次性 holdout；不得通过调整旧门、挑 trial 或反复查看 holdout 使旧候选通过。

## 7. 主要产物（工作区相对路径）

- `analysis_outputs/ssvep_v5_saved_predictions_independent_audit_20261007.json` / `.md`
- `analysis_outputs/ssvep_internal_v5_benchmark_preregistration_20261007.json`
- `analysis_outputs/ssvep_internal_v5_benchmark_lock_20261007.json`
- `analysis_outputs/ssvep_internal_v5_benchmark_20261007/ssvep_internal_v5_benchmark_20261007.json` / `.md`
- `analysis_outputs/ssvep_v5_internal_benchmark_report_recount_20261007.json`
- `analysis_outputs/ssvep_v5_continuation_final_verification_20261007.json`

新报告不能覆盖旧输出。重新运行同一冻结诊断时需指定新的报告目录；这不允许修改模型或评价参数。
