# NeuroStation v12：统一归一化、个人离线输出与整段信号评估

日期：2026-10-13。配置与交付状态为 formal；模型用于显式离线研究，production_validated=false、runtime_enabled=false。

## 本轮结论

v12 已在 NVIDIA GeForce RTX 4090 上完成 24 轮训练，选择第 21 轮。训练、选择、窗口评估与整段推理使用同一 noisy-window 均值/RMS 和 residual restore，实测两条执行路径最大绝对差为 0.0。

个人会话均已生成有限、完整的 8 通道输出；原始文件 26 个 SHA256 保持不变。两份会话的固定 CCA 与呈现目标匹配仍为 9/12 和 8/12，没有提高。此计数不能解释为独立注视准确率。

公开参考上的人工污染合并 MSE 下降 51.40%，仅代表对已有 recorded surrogate 人工加噪的重建实验。长合成波形中传统滤波对慢漂移、Gaussian 和组合污染更有效，因此 v12 不替换默认工作站模型。

## 归一化与训练来源

- 对 noisy [B,8,512] 在 float64 中计算每通道均值与 RMS；reference 使用同一 noisy 统计量变换。
- 网络接收 float32 signal；输出按 noisy + (prediction-normalized_noisy) * noisy_rms 恢复到原始数值坐标。
- 固定 50 Hz Q30 notch 与学习网络分别计量；不把 notch 的收益算作网络收益。
- Wearable subject 1–4 为 fit，5–6 为 selection，7–8 为本轮 evaluation。7/8 在项目历史中已被使用，不是全新独立保留集。
- 两份个人数据均未进入训练、选轮或模型 forward；标签只用于输出之后的开发诊断。
- recorded reference 不是真实 clean EEG；实际电极位置、增益与单位尚未独立核实。

## 公开参考窗口结果

| 人工条件 | 输入 MSE | 输出 MSE | 误差下降 |
|---|---:|---:|---:|
| identity | 0.000000 | 0.001327 | 不适用（输入误差为 0） |
| drift | 0.621877 | 0.239506 | 61.49% |
| high_burst | 0.628485 | 0.032951 | 94.76% |
| gaussian | 0.634878 | 0.609683 | 3.97% |
| combined | 1.884596 | 0.949959 | 49.59% |

identity 平均变化为 0.001327，满足预先固定的 0.002 研究约束。每个条件 480 个参考窗口。保护指标只覆盖单窗离散 8–40 Hz FFT bins，不保证拼接后整段频谱或生理信号恢复。

## 长合成波形结果

每例 4096 点；10、12、15、20 Hz × 5 条件，共 20 例。下表为中间区间 [512:3584]、每轨独立去均值的 NMSE，按 4 个频率平均。误差越小越好；真实生理噪声与 clean EEG 没有该人工真值。

| 条件 | raw | notch | traditional | v12 |
|---|---:|---:|---:|---:|
| identity | 0.000000 | 0.000001 | 0.001114 | 0.000265 |
| line50 | 3.542837 | 0.000001 | 0.001119 | 0.000265 |
| drift.3 | 0.884504 | 0.884505 | 0.001114 | 0.863602 |
| gaussian | 0.445456 | 0.431935 | 0.142442 | 0.424733 |
| combined | 4.868324 | 1.317903 | 0.142608 | 1.293579 |

traditional 为固定 50 Hz Q30 notch 后四阶 1–45 Hz 零相位带通。它也会衰减 40 Hz 的二次谐波，因此需要同时看幅度/相位，而不能只按总误差排序。

在 0.3 Hz 漂移上 v12 相对 notch 的误差下降约 2.36%；Gaussian 约 1.67%；组合污染约 1.85%。这些整段指标与公开短窗口人工污染结果属于不同数据分布，不能相互替代。

## 个人采集结果

| 会话 | 样本数/通道 | raw CCA 呈现匹配 | notch | v12 |
|---|---:|---:|---:|---:|
| session_20261004_004101_105 | 24371 | 9/12 | 9/12 | 9/12 |
| session_20261004_004555_413 | 24372 | 8/12 | 8/12 | 8/12 |

两份会话的包序号连续，当前文件中未检测到非单位步进。历史时间戳 gap 不能单独当成真实丢包结论；采集质量仍有显示节奏、启动零值和布局未核验等限制。已有导入记录中的平直/疑似极值通道需单独复核，不用于充当 clean/noisy 成对参考。

## 为什么继续改进

512 点窗口仅 2.048 秒。模型固定保留 DC，又恢复每窗输入均值，无法自由修正窗均值层面的慢漂移。8–40 Hz 整段被保护，也限制该频段内噪声去除。

新增人工 oracle 诊断使用真实人工 clean 来选择每窗 0≤gain≤1 的最佳实数谱增益（部署时不可获得该真值）。漂移案例约 23.49% 的输入窗口 MSE 位于窗均值差中；即使满足相同 DC/保护限制，oracle 整段 NMSE 仍可降至约 0.128，而 v12 为约 0.864。说明训练分布和学习策略仍有改善空间，不能将失败全部归因于窗均值。该 oracle 只最小化逐窗误差，不能声称是全局 OLA 最优。

下一实验采用 4096 点上下文与连续长人工信号，保留原有版本，公开/长人工两域选择集都需约束 identity 变化。仍需单列传统滤波对照并保留个人数据的评价用途。真实 EEG 的正确性仍依赖可信采集与独立参考，不能由人工基准替代。

## 已执行验证

- v12 专项 20 项通过。
- 核心测试 1558 项运行，5 项跳过，其余通过。
- 界面测试 112 项通过。
- 正式配置检查 ready；v9 工作站 CheckOnly ready。
- v12 配置/锁/checkpoint/训练报告绑定核验通过；两份个人会话 26 个文件 SHA 未变。

## 证据路径

- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\models\denoiser_v12_personal.pt`
- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\models\denoiser_v12_personal.manifest.json`
- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\ssvep_v12_training_20261013.json`
- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\v12_evaluation_20261013\summary.json`
- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\v12_evaluation_20261013\public.json`
- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\v12_evaluation_20261013\synthetic.json`
- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\v12_architecture_diagnosis_20261013.json`
- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\personal_eeg_dataset_audit_20261013.json`

## 可直接运行的离线入口

两个原生会话已通过 `scripts/run_ssvep_v12_offline_analysis.py` 在 cuda:0 上真实处理，raw/notch/v12 数组与原评估逐样本完全一致，另行保存传统滤波对照。每次必须使用全新的输出目录；源数据只读。

```powershell
& 'C:\Users\Administrator\Documents\ChatGPT\脑机采集\.venv-ml-gpu\Scripts\python.exe' `
  'C:\Users\Administrator\Documents\ChatGPT\脑机采集\scripts\run_ssvep_v12_offline_analysis.py' `
  --session 'C:\Users\Administrator\Documents\NeuroStation\Datasets\session_20261004_004101_105' `
  --output 'C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\v12_offline_new_run' `
  --device cuda:0 --acknowledge-out-of-domain
```

已完成的输出目录：

- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\v12_offline_004101_105`
- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\v12_offline_004555_413`

每个目录含 `raw.npy`、`notch.npy`、`traditional.npy`、`v12.npy` 与最后原子写入的 `report.json`。启动零值、实际布局与校准未核实等采集限制不会被程序自动认定为通过。

四路图位于 `analysis_outputs/v12_personal_figures_20261013`。仅为显示而逐轨去均值，四路波形使用相同纵轴；PSD 为首个分析试次中 8 通道的中位数，不能据图确认真实 EEG 已恢复。

独立 CPU 长波形复现审计位于 `analysis_outputs/v12_long_waveform_audit_20261013_standalone_cpu`：20 例、80 个中间区间 MSE 指标复现通过（rtol=1e-5，atol=1e-9，最大差 2.93e-8）。此审计另保存联合最小二乘的逐通道幅相及 full/interior 指标。
