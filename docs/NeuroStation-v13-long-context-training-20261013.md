# NeuroStation v13 长上下文实验：训练完成，选择约束失败

日期：2026-10-13。正式实验配置；仅离线研究诊断，runtime_enabled=false、production_validated=false。

## 已核实结果

NVIDIA GeForce RTX 4090 已完成 16 轮训练。网络支持 512/4096 点的原生 250 Hz、8 通道输入，共 3,625 个参数。连续记录采用 4096 点上下文、hop1024、正 Hann 残差叠加与完整尾窗覆盖。

**没有任何一轮同时满足公开参考与长人工两域的 identity 约束，selected_epoch=null、selection_gate_passed=false。** 保存的是第 16 轮的诊断权重，checkpoint_role=diagnostic_last_epoch_no_eligible_checkpoint；正常 load_model 必须拒绝该失败权重。

第 16 轮公开参考 identity normalized change 为 0.005129，超过固定上限 0.002；长人工 identity 为 0.000798。不能用长人工域单独通过来替代双域通过。

## 最后一轮长人工选择集

以下指标仅为选择集结果，且 checkpoint 未通过选择约束。固定 notch 收益与学习残差收益独立计量。

| 条件 | raw MSE | notch MSE | model MSE | model 相对 notch 误差下降 |
|---|---:|---:|---:|---:|
| identity | 0.000000 | 0.000002 | 0.000408 | 输入参考一致，不适用 |
| drift | 0.564206 | 0.564208 | 0.112998 | 79.97% |
| high_burst | 0.077547 | 0.071944 | 0.002058 | 97.14% |
| gaussian | 0.137332 | 0.133182 | 0.133357 | -0.13% |
| combined | 0.811395 | 0.802734 | 0.267588 | 66.67% |

慢漂移约 80% 的选择集改善支持长上下文假设；Gaussian 相对 notch 仍略差，且公开参考未加噪输入的保真失败。上述结果不能作为有效部署证据。

## 可追溯范围

- fit public1–4；selection public5–6；连续长人工 reference fit96、selection32，独立种子。
- public7/8 和两份个人会话未被训练读取，也未用于选轮。public7/8 在项目历史中已出现，未来评估不能称为全新独立集。
- 每次 forward 只接收信号，不接收目标标签、注视信息或 clean reference。
- 训练前研究计划和冻结来源均有 SHA 绑定；没有改阈值来放行失败模型。
- 8–40 Hz 全段受保护，不能声称去除此频段内部白噪声；整段幅相仍需事后测量。
- 默认桌面模型与历史验证记录保持原样。

## 证据路径

- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\ssvep_v13_experiment_plan_20261013.json`
- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\ssvep_v13_training_lock_20261013.json`
- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\ssvep_v13_training_20261013.json`
- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\v13_training_20261013.log`
- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\models\denoiser_v13_personal.manifest.json`

## 事后诊断评估（2026-10-13）

评估程序已在 CUDA 环境实际运行；只显式读取失败选择的第 16 轮诊断权重，没有重新训练或选择 checkpoint。`completion_status=complete` 仅表示评估文件完整生成，**不表示模型通过研究门槛或项目目标完成**。选择门槛与事后研究门槛均失败，runtime、生产、真实生理 clean 验证、独立个人验证均为 false。

| 固定研究检查 | 实测值 | 预定阈值 | 结论 |
|---|---:|---:|---|
| 必须有通过选择约束的 checkpoint | false | true | 失败 |
| public7/8 identity normalized change | 0.00230812272 | ≤0.002 | 失败 |
| 长人工 identity normalized change | 0.00002083687 | ≤0.002 | 通过 |
| 长人工漂移与组合的合并整段误差相对 notch 下降 | 48.8144% | ≥50% | 失败 |
| 基频幅度最大相对误差（identity/line50） | 0.0380% | ≤2% | 通过 |
| 基频相位最大误差（identity/line50） | 0.0305° | ≤2° | 通过 |
| 第二谐波幅度最大相对误差（identity/line50） | 1.1012% | ≤5% | 通过 |
| 第二谐波相位最大误差（identity/line50） | 0.1545° | ≤3° | 通过 |
| 512/4096 点显式路径与 runtime 归一化最大差 | 0 | ≤1e-10 | 通过 |

选择集慢漂移约 80% 的改善，与事后固定人工基准漂移及组合合并整段 48.8% 是不同数据、不同汇总方式的测量，不能互相替代。幅相检查仅使用人工 identity/line50 案例，不能推广为真实 EEG 或所有噪声条件的生理保真证明。

### 同一长人工基准的传统滤波对照

以下为 20 个历史重复开发案例（10、12、15、20 Hz × 5 条件）中部 `[512:3584)` 的合并 NMSE，越小越好。上述研究门槛使用整段误差，因此不能用这张中部表格替换门槛计算。

| 条件 | notch | traditional | v13 失败诊断权重 |
|---|---:|---:|---:|
| identity | 0.0000007928 | 0.00111535 | 0.00001818 |
| line50 | 0.0000007928 | 0.00112010 | 0.00001798 |
| drift.3 | 0.88450406 | 0.00111534 | 0.31065459 |
| gaussian | 0.43193827 | 0.14244389 | 0.43143183 |
| combined | 1.31790086 | 0.14260695 | 0.80971806 |

v13 在该人工基准中减小慢漂移误差，但漂移、Gaussian、组合条件仍明显差于传统滤波。v12/v13 公共数据 augmentation 不同，公共指标不能作为公平的跨版本对照；图中的人工对照只来自逐样本相同的重复开发案例，不是全新 holdout。

### 两份个人采集记录

两份原始记录均已生成 raw、notch、traditional、v13 四轨完整数组。`session_20261004_004101_105` 为 24371×8，`session_20261004_004555_413` 为 24372×8，采样率均为 250 Hz。所有输出保持原通道顺序、采样点数和输入单位。

| 会话 | raw | notch | traditional | v13 |
|---|---:|---:|---:|---:|
| 004101_105，12 个试次 | 9/12 | 9/12 | 9/12 | 9/12 |
| 004555_413，12 个试次 | 8/12 | 8/12 | 8/12 | 8/12 |

这里是 CCA 强制 argmax 对记录中的呈现目标匹配数；没有独立注视真值，不能称为注视准确率或可靠 SSVEP 检出率。两份个人会话及 public7/8 未用于 v13 训练/选轮；public7/8 和人工案例在项目历史中已有使用，不能称为全新独立验证。

原始记录可读取、可处理并可输出，故不能把当前失败归咎为“个人数据全部有问题”。当前证据显示学习模型的参考保真与长上下文去漂移仍有冲突；实际电极位置、校准、独立注视标签以及可信 clean 参考仍未完成验证。曲线更平滑或更接近零并不自动等于恢复正确生理 EEG。

### 查看与复核入口

- 诊断评估汇总：`analysis_outputs/v13_evaluation_20261013_diagnostic/summary.json`。
- 公共、人工、归一化详表：同目录 `public.json`、`synthetic.json`、`normalization.json`。
- 个人输出：同目录 `personal_00/`、`personal_01/`，含 `.npy` 四轨数组和 `report.json`。
- 对照图：`analysis_outputs/v13_diagnostic_figures_20261013/` 下三张 PNG 和绑定图数据的 `report.json`。已逐图目视检查，个人 CH1 16 秒波形使用相同纵轴，仅显示去均值；保存数组保持原值。
- 最终验证：`analysis_outputs/ssvep_v13_final_verification_20261013.json`，以实际文件生成后的内容为准；不改写历史版本验证结论。

正常 runtime 以及默认评估入口拒绝该失败选择权重。诊断复核命令必须显式使用 `--allow-failed-selection-diagnostic`，并写入一个尚不存在的新目录；它不授权默认部署。

```powershell
.\.venv-ml-gpu\Scripts\python.exe -B scripts/evaluate_ssvep_denoiser_v13.py `
  --output-directory analysis_outputs/v13_diagnostic_review_NEW `
  --personal-session 'C:\Users\Administrator\Documents\NeuroStation\Datasets\session_20261004_004101_105' `
  --personal-session 'C:\Users\Administrator\Documents\NeuroStation\Datasets\session_20261004_004555_413' `
  --device cuda:0 --allow-failed-selection-diagnostic
```

下一阶段应先定位公共 recorded surrogate 保真约束与长人工去漂移的冲突，再冻结新的独立实验设计。继续训练时不得放宽本轮阈值、复用本轮输出挑选权重、以呈现频率模板生成“正确信号”或声称临床验收。个人效果需要经确认的布局与校准、独立注视标签和新的记录；缺失这些证据期间可以继续工程与人工数值诊断，但不能宣称恢复真实生理信号。

### 事后测试兼容性修复与来源差异

事后评估结束后，`tests/test_evaluate_ssvep_denoiser_v13.py` 仅新增 `importlib.util` 导入和 LoaderTests 的可选 PyTorch 缺失跳过条件。它不改变评估实现、任何断言、训练源码、配置、锁、checkpoint 或门槛。当前普通环境实际安装的是 CPU 版 PyTorch 2.14.1，GPU 环境使用 CUDA 版；两套环境各 11 项新增评估测试均实际运行通过，没有因这项兼容修复跳过。

原评估汇总保留评估时的测试文件 SHA，不改写历史收据。最终来源复核中 39 个评估输入有 38 个保持逐字节一致；剩余 1 个是上述测试文件，移除明确新增的两行后其字节 SHA 与原评估记录完全一致。该差异在 `analysis_outputs/v13_final_artifact_audit_20261013.json` 中单列；26 个个人源文件、13 个评估输出和 3 张图均与各自记录一致。
