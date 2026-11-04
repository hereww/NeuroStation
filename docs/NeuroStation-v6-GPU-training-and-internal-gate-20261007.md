# NeuroStation v6：GPU 训练、内部诊断与部署阻断记录

日期：2026-10-07；配置与实验协议状态：formal。

**真实 GPU 训练与固定内部诊断已完成，但整体内部门失败，v6 禁止接入实时采集工作站。正式状态不等于质量、硬件或临床验收通过。**

## 1. 实际完成

- 在本次真实 v6 参考波形读取前冻结 40 项代码/配置/参考集绑定；两个数据域分别从新初始化模型训练，不加载旧 v5 权重。
- 使用 RTX 4090、cuda:0、PyTorch 2.14.1+cu130 / CUDA 13.0，各完成固定 18 轮；关闭 AMP 与 TF32，启用 deterministic algorithms。
- 检查 weights_only CPU 加载、完整 schema、有限权重、非零训练输出层，以及每轮门槛和最佳轮次的独立复算。
- 训练完成后另冻结 42 项诊断绑定，执行 4,200 个固定合成案例 GPU 推理；模型状态 hash 在推理前后相同。
- 新增独立标准库 JSON-only 审计，从保存案例重新汇总标量账目；mismatches=0。其代码和输入另以 5 项绑定冻结，不修改训练/诊断锁。
- 按用户指定复用 gpt-6.1-sol 编程子代理；本轮未新增下载任务，复用现有冻结参考集。

## 2. 数据范围

| 数据域 | 参考采样率 | 训练受试者 | 内部选型/诊断受试者 | 每轮训练条件窗口 | 诊断条件窗口 |
|---|---:|---|---|---:|---:|
| Nakanishi2015 | 256 Hz | 1–4 | 5–6 | 3,600 | 1,800 |
| Wearable | 250 Hz | 1–4 | 5–6 | 4,800 | 2,400 |

- 每个 trial 一个固定 256 点窗口，五种条件全部覆盖，无组合污染。840 个内部参考 trial 产生 4,200 条条件案例，其中 identity 840 条；重复条件和 epoch 不等于新增独立 EEG 样本。
- 选型与诊断使用同一批受试者 5–6。新诊断污染种子不等于新的独立受试者测试。跨域参与者身份是否重叠未知。
- 本轮未扫描、解压或访问 D:\SSVEP 原始资料，未读 7–8 的 outer/known-validation 波形，未读 9–10 的 holdout 波形或指标。此前 v5 已读过 7–8，不得重新称为未知测试集。
- 录制 SSVEP 与 wet 电极数据不是 clean ground truth。本次评估只衡量添加合成污染后重建录制参考的误差，不证明真实生理噪声已消除。

## 3. 模型与冻结策略

- 每域独立 8 通道残差 TCN，89,256 参数；hidden=32、kernel=7、dilations=[1,2,4,8,16,32]、GroupNorm=8、dropout=0。
- 输出卷积零初始化，初始模型精确 identity；训练后不能保证原样输出，必须满足 identity 改写门。
- rFFT / irFFT(n=256) 去掉残差的含边界 8–40 Hz 名义频谱 bins。0.5 tanh 限值只作用在投影前，不是最终逐点幅度界；不做投影后 clip。
- 此保护限于固定窗口的去均值 rectangular rFFT；非因果，不证明 Hann OLA、连续流、任务解码或临床保真。
- AdamW lr=0.0005、weight_decay=0.0001，batch=64、workers=0、梯度裁剪 1.0，各域 18 epoch。非 identity 误差 floor=0.01，identity loss 权重20，改写惩罚权重0.001。
- 逐域固定排序：eligible-first，再最小化最差噪声条件的 mean-error ratio、identity 误差、内部总 loss，最后最早 epoch；无合格轮仍保存 best，但明确 failed。
- 实际选择：Nakanishi2015 epoch5（唯一全部门通过轮次），Wearable epoch1（18轮均未通过高频门）。未续训、未改门、未重新选择参数。

## 4. 固定内部诊断

下表为 `(1 - mean(output normalized MSE) / mean(input normalized MSE)) × 100%`。正值是合成重建误差下降，负值是误差增加；不是 SNR 增益，也不是逐案例改善百分比的平均值。

| 污染 | Nakanishi2015 | Wearable | 最低要求 |
|---|---:|---:|---:|
| 瞬态伪迹 | +12.43% | +3.69% | ≥0% |
| 低频漂移 | +4.22% | +6.68% | ≥0% |
| 高频突发 | +7.58% | -2.27% | ≥5% |
| 共模污染 | +4.94% | +15.13% | ≥0% |

| identity 检查 | Nakanishi2015 | Wearable | 上限/要求 |
|---|---:|---:|---:|
| 归一化改写误差 | 0.0004744920055 | 0.0004730624453 | 0.0005 |
| 保护带 complex-energy ratio | 5.521893031e-16 | 1.339399101e-14 | 1e-6 |
| 保护带加权绝对相位差 | 9.854958634e-09 | 4.681665616e-08 | 0.02 rad |
| 未定义相位案例 | 0 | 0 | 必须0 |

- Nakanishi2015 全部内部门通过，但没有独立研究或上线验收。
- Wearable 高频误差增加约2.27%，没有达到至少降低5%的门槛；其余门通过不能覆盖此失败。
- 整体 all_internal_gates_passed=false，runtime_enabled=false，production_validated=false；未进入 known-validation，未替换工作站模型。
- 独立 JSON 审计仅验证保存的标量账目一致，未从 EEG 重算 phase 或重新推理；一致不能把失败变为通过。
- v5 与 v6 合成污染种子不同，不能将均值差直接当作同案例的配对性能增益。

## 5. 复验与历史保留

- 核心测试 779 项：775 通过、4 跳过。
- GPU ML 环境 v6 专项 123 项全通过（含独立审计22项）；它们是核心测试子集，不能将总数相加。
- 界面 65 项全通过；正式配置 ready / formal、warnings=[]。
- 首次合并测试的3项失败是 C 测试对全进程 CUDA 初始化状态的错误假设。真实训练和冻结前修复为逐测试状态不变，并增加冷启动无 torch/二进制读测试；保留失败日志。
- 474项历史非波形文件 hash 不变，旧v5权重、报告、锁与失败结论均保留。没有为了全目录 hash 证明读取 outer/holdout 波形。
- 训练40、诊断42、独立审计5项绑定均复核不变。诊断只读取5–6，preflight 故意 defer 1–4 的八个 NPZ；训练绑定复核另检查了它们。
- 硬件握手、通道完整性、时序预检、光敏确认、中止控制和质量评估未删减；未进行真实硬件或临床验收。

## 6. 证据与入口

- [新权重](C:/Users/Administrator/Documents/ChatGPT/脑机采集/models/denoiser_v6_ssvep.pt)
- [训练配置](C:/Users/Administrator/Documents/ChatGPT/脑机采集/configs/learned_denoiser_v6_ssvep.json)
- [训练预注册](C:/Users/Administrator/Documents/ChatGPT/脑机采集/analysis_outputs/ssvep_v6_training_preregistration_20261007.json)
- [训练锁](C:/Users/Administrator/Documents/ChatGPT/脑机采集/analysis_outputs/ssvep_v6_training_lock_20261007.json)
- [训练报告](C:/Users/Administrator/Documents/ChatGPT/脑机采集/analysis_outputs/ssvep_v6_training_20261007.json)
- [内部诊断锁](C:/Users/Administrator/Documents/ChatGPT/脑机采集/analysis_outputs/ssvep_v6_internal_evaluation_lock_20261007.json)
- [内部诊断报告](C:/Users/Administrator/Documents/ChatGPT/脑机采集/analysis_outputs/ssvep_v6_internal_diagnostic_20261007.json)
- [独立审计锁](C:/Users/Administrator/Documents/ChatGPT/脑机采集/analysis_outputs/ssvep_v6_saved_internal_audit_lock_20261007.json)
- [独立审计报告](C:/Users/Administrator/Documents/ChatGPT/脑机采集/analysis_outputs/ssvep_v6_saved_internal_audit_20261007.json)
- [训练入口](C:/Users/Administrator/Documents/ChatGPT/脑机采集/scripts/train_ssvep_denoiser_v6.py)
- [诊断入口](C:/Users/Administrator/Documents/ChatGPT/脑机采集/scripts/evaluate_ssvep_denoiser_v6_internal.py)
- [独立审计入口](C:/Users/Administrator/Documents/ChatGPT/脑机采集/scripts/audit_ssvep_saved_internal_v6.py)

权重 SHA256：`f3df5d268a0da3ac8af3852194b6f6042efe6abac180e5a9b2406be14a12ce1b`。

训练/诊断/审计输出均禁止覆盖。已有模型和报告时不要删除后重跑 v6；新训练须创建新实验标识和新的预注册/冻结绑定。

## 7. 下一步边界

针对 Wearable 高频失败创建新的独立冻结实验，定位并修复，同时约束 identity 改写；不得通过删域、放宽门或使用 holdout 选参数来宣称 v6 能上线。

只有下一实验逐域内部门都通过后，再另行预注册冻结受试者7–8的 known-validation 工程筛查。它们是已知验证集，不是最终独立留出；9–10继续封存。
