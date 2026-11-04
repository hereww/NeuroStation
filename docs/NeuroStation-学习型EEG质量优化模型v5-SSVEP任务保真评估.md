# NeuroStation 学习型 EEG 质量优化模型 v5：SSVEP 任务保真评估

## 状态

本文件对应 2026-10-07 预注册的正式研究合同。它定义的是**任务保真评估**，不是降噪监督真值，也不是临床或生产验收。当前没有 v5 checkpoint，`runtime.enabled=false`，工作站不会自动替换采集信号。

## 为什么先做任务保真

`D:\SSVEP` 中的真实 SSVEP 记录有明确的刺激目标、频率和部分时序信息，但没有同一 epoch、同一电极的 clean/noisy 成对真值。因而本阶段不能用 RMSE、SNR 或“恢复到 clean”作为去噪结论。

可以先检查候选处理是否破坏 SSVEP 目标频率、二次谐波和分类结果。候选模型只有在任务性能基本不下降、频谱响应不被系统性削弱时，才有资格进入后续独立配对伪迹数据评估。

## 固定数据与分组

- 数据目录：`datasets/processed/ssvep_reference_v5_20261007`。
- Nakanishi2015 与 Wearable 分别使用 subject 1–6 作为 train、7–8 作为 validation、9–10 作为 holdout。
- 当前初始报告只允许读取 validation；holdout 只保留身份锁，不读取波形。
- 分组键使用 `dataset:subject_id`。这保证数据集内部 subject-disjoint，不证明不同公开数据集之间不存在同一参与者。
- 原始参考信号不是 clean ground truth；幅值单位保持 unknown。

## 固定解码器

不训练、不从数据调频率、不使用标签选择预测。每个试次：

1. 使用来源预先声明的刺激窗口；
2. 每通道去均值；
3. 乘 Hann 窗；
4. 计算 rFFT 功率；
5. 在声明目标频率和二次谐波的最近 FFT bin 读取功率；
6. 跨通道平均；
7. `score = fundamental + 0.5 × second_harmonic`，最大者为预测目标。

窗口固定为：

| 数据集 | 采样率 | 窗口 |
|---|---:|---|
| Nakanishi2015 | 256 Hz | `[38, 1062)`，1024 点 |
| Wearable | 250 Hz | `[125, 625)`，500 点 |

## 预注册指标与研究门

报告每个被试和数据集聚合的：

- accuracy；
- mean confidence；
- mean relative score；
- 目标频率 score；
- 候选与 reference 的预测一致率；
- 候选/reference 目标频率 score 比值。

候选信号不得使用 `clean`、`ground_truth`、`target_signal` 等命名，也不得重新标注、改变 trial 顺序或混入 holdout。

在候选指标读取之前，预先声明以下研究门：

- validation macro accuracy 下降不超过 0.02；
- validation micro accuracy 下降不超过 0.02；
- 预测不一致率不超过 0.02；
- 目标频率 score 比值不低于 0.95。

任何研究门失败都意味着“不进入部署”，但不能单独证明模型一定没有降噪价值，也不能推出临床危害。不得在读取候选结果后再修改门槛。

## 后续顺序

1. 对当前 validation reference 运行固定 FFT 基线，得到预注册基线报告；
2. 完成 v5 模型接口和跨采样率策略；
3. 只在 train 分组进行模型训练或参数学习；
4. 使用 validation 做预先声明的选择；
5. 冻结模型、配置和研究门后，进行一次性 holdout 评估；
6. 另行引入真实或明确定义的合成 paired clean/noisy 数据，才讨论监督式降噪误差；
7. 研究门通过前，保持工作站自动替换关闭。

详见机器可读合同：`analysis_outputs/learned_denoiser_v5_ssvep_task_preservation_preregistration.json`。

## 2026-10-07 validation 基线实测（无候选模型）

评估源码、原始预注册及数值细节在首次读取真实任务指标前已分别冻结。本轮仅使用 subject 7–8 的 validation：4 个 subject-recording groups、840 个试次。没有读取 holdout，没有对 train 波形运行本次基线，没有生成候选模型或 checkpoint。

| 数据集/电极域 | 正确/试次 | Micro accuracy | Macro accuracy |
|---|---:|---:|---:|
| Nakanishi2015 | 325/360 | 90.28% | 90.28% |
| Wearable（dry + wet） | 118/480 | 24.58% | 24.58% |
| Wearable dry | 54/240 | 22.50% | 22.50% |
| Wearable wet | 64/240 | 26.67% | 26.67% |
| 全部 validation | 443/840 | 52.74% | 57.43% |

Macro 是各 subject-recording group 准确率的非加权平均；micro 是正确试次总数/试次总数。全局仅汇总无量纲准确率，不跨原生幅值单位未知的数据集汇总谱功率。上述估计是描述性结果，不假设重复试次或跨数据集参与者彼此独立。

逐被试结果：

| 数据集 | Subject | 正确/试次 | Accuracy |
|---|---:|---:|---:|
| Nakanishi2015 | 7 | 145/180 | 80.56% |
| Nakanishi2015 | 8 | 180/180 | 100.00% |
| Wearable | 7 | 62/240 | 25.83% |
| Wearable | 8 | 56/240 | 23.33% |

主线程基于报告内保存的预测与标签独立重计了正确数，并复核 dataset、electrode-domain 和全局 macro/micro 聚合。该复核没有重新读取 NPZ 或波形，也没有重新选择解码参数。

### 结果的边界

- 这是固定 FFT 的原始任务参考基线，不是去噪提升，不代表某个 v5 学习型模型已通过验收。
- Wearable 的较低结果如实保留，不据此修改已锁定的频率、窗口、EPS 或研究门。单凭该准确率不能诊断噪声原因，也不能判定 wet 是 clean 真值。
- FFT 功率与分类保留不能证明相位保真、伪迹去除或临床有效性；本实现不是 CCA/FBCCA 论文结果复现。
- `EPS=1e-12` 只是数值分母保护，不是“无生理响应”的阈值。未来候选/参考目标谱分数比值在参考分数不大于 EPS 时为 undefined，不能默认接受。
- 本轮没有候选去噪输出，预注册的 4 项候选研究门均为 **not_assessed（尚未评估）**，不能写成“通过”。
- 工作站自动信号替换继续关闭，`runtime_enabled=false`、`production_validated=false`；本轮无需 GPU，未启动 GPU 训练。

### 冻结及结果文件

- 预注册：`analysis_outputs/learned_denoiser_v5_ssvep_task_preservation_preregistration.json`
- 数值合同：`analysis_outputs/ssvep_task_v5_numerical_contract_20261007.json`
- 实现锁：`analysis_outputs/ssvep_task_v5_implementation_lock_20261007.json`
- JSON 基线：`analysis_outputs/ssvep_task_v5_validation_baseline_20261007.json`
- Markdown 基线：`analysis_outputs/ssvep_task_v5_validation_baseline_20261007.md`
- 独立计数及研究门状态：`analysis_outputs/ssvep_task_v5_validation_baseline_review_20261007.json`

### 下一阶段的约束

下一阶段应先冻结 v5 的 train-only 候选实验、噪声构造、接口和评估计划，再开展 GPU 训练。若在原始记录上叠加可控伪迹构造配对训练，目标只能称为 recorded reference，不能称为真实无噪声 EEG；真实 SSVEP 上仍须独立检查任务保真。holdout 在模型、配置和一次性评估锁全部冻结之前继续不打开。研究筛查通过也不自动获得生产或临床有效性资格。

### 本轮回归验证

- 新增纯合成评估测试：68/68 通过，未读取真实数据。
- 全部核心测试：567 项，565 通过、2 跳过，0 失败。
- 工作站界面测试：65/65 通过。
- 正式发布配置检查：`ready`，协议 `formal`，warnings 为空。
- 离屏 UI smoke：1.0、1.25、1.5 三种缩放均成功。
- 最终不可变性、来源及日志哈希见 `analysis_outputs/ssvep_task_v5_final_verification_20261007.json`。

## 2026-10-07 v5 train-only 候选结果

在固定 FFT validation 基线之后，按新的 train-only 契约训练了一个离线候选：Nakanishi2015 与 Wearable 各一个多通道残差 TCN。训练输入是对 train subject 1–4 的 recorded reference 叠加确定性合成瞬态、低频漂移、高频 burst 和 common-mode 扰动；target 仍是对应 recorded reference，不是 clean EEG。subject 5–6 仅作内部验证选择，subject 7–8 只在本次冻结候选评估中读取，subject 9–10 未读取。

训练使用 RTX 4090/CUDA，AMP 关闭；checkpoint 尚未接入工作站，`runtime.enabled=false`、`production_validated=false`。

| 范围 | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| 全部 validation micro（443/840 → 442/840） | 52.7381% | 52.6190% | -0.1190 个百分点 |
| 全部 validation macro | 57.4306% | 57.3264% | -0.1042 个百分点 |
| Nakanishi2015 micro | 90.2778% | 90.2778% | 0 |
| Wearable micro | 24.5833% | 24.3750% | -0.2083 个百分点 |
| Wearable dry | 22.5000% | 22.0833% | -0.4167 个百分点 |
| Wearable wet | 26.6667% | 26.6667% | 0 |

候选目标谱分数比值的按试次加权均值为 **0.990669**。固定研究门的预测不一致率为 **19/840 = 2.2619%**，超过预注册上限 2.00%；因此本候选研究筛查**不通过，阻止部署**。其余三项门通过：macro accuracy drop、micro accuracy drop、target spectral score ratio。不能因为 accuracy 下降很小就忽略 prediction-disagreement 失败，也不能读取该 validation 结果后调阈值、改窗口或重新训练。

这次失败只表示该候选不能进入工作站自动替换，不证明合成扰动覆盖了真实噪声，也不证明它没有任何信号质量价值。候选的离线 checkpoint 和输出目录保留用于审计，但不作为正式运行时模型。

### 候选阶段审计边界

- 候选评估锁定了 Hann 残差 overlap-add、原生采样率和 `processed_v5` 接口后才读取任务指标。
- 候选完整性检查通过，trial labels、频率、电极域、shape、采样率和 reference SHA 绑定未改变。
- 失败的早期候选目录保留为历史失败尝试，不纳入验收：其中包括空 NPZ 发布错误、等权 overlap-average 实现和中途失败目录。
- 没有读取 holdout 波形；没有使用 validation 调训练、调阈值或选择候选；没有 SNR/RMSE-to-clean 结论。
- 后续若要改善，必须在 train/internal-validation 规则内另行建立新的冻结实验；不能针对这 19 个 validation 不一致试次做事后修补。

### 候选阶段文件

- 训练契约：`analysis_outputs/learned_denoiser_v5_ssvep_training_preregistration_20261007.json`
- 训练锁及修订记录：`analysis_outputs/learned_denoiser_v5_ssvep_training_lock_20261007.json`、`analysis_outputs/learned_denoiser_v5_ssvep_training_lock_patch_20261007.json`、`analysis_outputs/learned_denoiser_v5_ssvep_training_lock_patch2_20261007.json`
- GPU 训练 checkpoint：`models/denoiser_v5_ssvep.pt`
- 训练报告：`analysis_outputs/learned_denoiser_v5_ssvep_training_20261007.json`
- 候选输出：`datasets/processed/ssvep_denoiser_v5_candidate_20261007_hann3`
- 候选评估锁：`analysis_outputs/learned_denoiser_v5_ssvep_candidate_evaluation_lock_20261007.json`
- 候选 JSON：`analysis_outputs/ssvep_task_v5_candidate_validation_hann3_20261007.json`
- 候选 Markdown：`analysis_outputs/ssvep_task_v5_candidate_validation_hann3_20261007.md`
- 研究门复核：`analysis_outputs/learned_denoiser_v5_ssvep_candidate_evaluation_review_20261007.json`

### 当前结论

当前 v5 候选**不能部署**。工作站继续使用原有正式配置和既有安全门，自动信号替换保持关闭。下一步不是打开 holdout，也不是修改 validation 门槛；应先在 train/internal-validation 上重新设计更保守的候选，或者补充有明确 paired target 定义的数据，再建立新的预注册与评估锁。
