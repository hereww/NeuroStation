# NeuroStation v13 后续原因诊断（2026-10-14）

本次完成只读诊断，不训练、选轮或改变门槛。v13 仍是第 16 轮失败选择的诊断 checkpoint；正常 runtime 拒绝加载，真实生理信号恢复和独立个人验证尚未完成。原有 v8–v13 训练配置、源码、锁、权重及既有评估结果未改动。

## 梯度冲突的实测证据

固定 checkpoint 上，分别使用 fit1–4 和 selection5–6 公共参考，以及冻结种子生成的连续长人工参考。公共每被试从 240 个窗口按 `floor(i*239/31)` 取 32 个；fit 共 128 个、selection 共 64 个。长人工 fit 从 96 个均匀取 32 个，selection 使用全部 32 个。对完整数据先按原训练规则生成噪声再抽样，避免改变随机序列。公开 7/8、个人数据和事后评估案例均未用于梯度分析。

| 固定抽样角色 | public identity 对 long drift 梯度余弦 | public identity 对 public drift 梯度余弦 | identity×100 后相对 long drift 梯度范数 |
|---|---:|---:|---:|
| fit | -0.982356 | -0.998498 | 11.6491 |
| selection | -0.983728 | -0.999516 | 18.2024 |

上述负余弦表示在这个固定状态和抽样均值上，两种损失的局部一阶方向明显相反。正权重 100 不改变余弦，却放大 identity 梯度范数。不能将范数比解释为完整优化器更新的比例，也不能从局部梯度推断所有模型的不可行性。本次没有包含 gradient clipping、AdamW moments 或 weight decay，未复现完整训练轨迹。

`torch.autograd.grad` 没有写入参数 `.grad`。模型保持 eval；前后每个 state_dict tensor 及输入来源 SHA 均一致，完整梯度保存于 `gradients.npz` 并绑定数组和文件 SHA。13 项梯度专项测试通过。

## 公开参考中的低频成分

对相同冻结 trainer 的全部 fit1–4 与 selection5–6 参考计算 512 点矩形窗 rFFT Parseval 分解，0<f<8 Hz 的平均能量份额分别为 61.9326% 和 80.7167%。若使用所有窗口共用的固定低频增益 g，且其他成分不变，则 identity MSE 为 `(1-g)^2 × 低频平均能量`。为满足 0.002 门槛，该固定增益族最多允许约 5.6827% 与 4.9778% 的低频幅度衰减。

这只说明“统一强压低频”与当前完整参考保真要求不兼容。它不是自适应模型的能力上界，也不能判断这部分低频能量有多少是生理活动、有多少是已有伪迹。参考是 recorded surrogate，不能当作生理 clean 真值；512 点 FFT 的频率分辨率和泄漏也限制了频带解释。

## 很慢漂移与分窗均值恢复

新固定人工 recipe 使用 24371 点、250 Hz、8 通道，已知 10/20 Hz clean 波形；人工伪迹在全记录去均值并归一化至 RMS=3 AU。额外的每通道常量偏置为 `[0,10,...,70]` AU，用于检验 DC 是否被正确单列。这里的 clean 真值仅是人工数值真值。

比较相同固定 50 Hz notch 输入上的 v13、每窗最大 sub-8 Hz 衰减（所有 0<f<8 的离散 gain=0）与全记录 FFT 相同衰减。每窗方法保留窗均值，采用冻结 v13 的 4096 点窗口、hop1024、正 Hann 残差 OLA 和尾窗。全记录方法保留全记录 DC。两个固定衰减方法均仅接收信号，不读取 clean 选择滤波参数。

以下为中部 `[4096:20275)` 的未去均值参考 MSE 相对 notch 下降；这一统计与旧 4096 点整段研究门槛不同，不用来重判 v13。

| 人工伪迹 | v13 失败诊断权重 | 固定每窗 sub-8 衰减 + OLA | 固定全记录 FFT 对照 |
|---|---:|---:|---:|
| linear_trend | 0.00252% | 0.00725% | 99.99988% |
| sin.01 | 2.08052% | 11.63789% | 100.00000% |
| sin.03 | 32.23016% | 69.46178% | 100.00000% |
| sin.1 | 76.35882% | 99.91385% | 100.00000% |
| sin.3 | 88.13738% | 99.98723% | 99.99999% |
| step | 8.14720% | 19.33988% | 99.96154% |

线性趋势在 v13 中几乎完全残留；每窗统一低频压低也几乎没有改善，而全记录尺度的固定对照显著下降。这说明被测试的分窗 DC 恢复方式对非常慢的背景变化有明显局限，值得改动全记录尺度的基线处理。不能将固定对照称为最优 oracle、所有可能模型的上界或可直接部署方案。全记录对照对于人工 10/20 Hz clean 有效，并不保证任意真实 EEG 的低频内容被正确区分。

传统滤波对照也计算在报告中，但它会去掉设定的常量 DC。因为本次 clean 人为含有每通道 DC，传统轨未去均值 MSE 会额外含约 1750 AU²；这个 DC 项与中部的去漂移误差分别报告，不能据此声称传统算法的生理效果很差。`identity` 的 notch 误差近乎零，相对误差下降率容易被很小分母放大，也不宜用该比例作质量判断。

8 项慢漂移专项测试直接检验真实测量函数、DC 分解、联合幅相拟合、尾窗覆盖、单窗与多窗差异及无科学包 import/help。和梯度专项合并 21 项通过。真实诊断在 CPU 完成，小网络和本次固定实验不需要新增 GPU 训练。

## 下一步实现依据

1. 在全记录尺度估计、处理漂移基线，检验从分窗恢复局部均值改为一致的全记录基线处理；维持原采样点、通道顺序、单位和原信号追溯。
2. 新实验必须含线性趋势、0.01/0.03 Hz 慢漂移和阶跃，不能只用原有 0.1–0.8 Hz 加噪窗口选轮；同时加入可信低频参考，测量被误删内容。
3. 针对现有局部梯度冲突，先验证模型是否能够利用更长时间及跨通道证据区分参考低频与新增伪迹。不能通过放宽 identity 门槛或直接以呈现频率模板造波形解决。
4. 在新的独立训练计划中明确记录 reference 的局限；两份个人记录有输出且能解码，但尚缺独立注视真值、经确认的电极/增益校准和可信 clean 参考，不能据此宣称真实生理 EEG 恢复。

这些条目是新实现的依据，不代表已经实现、训练或通过。历史 v13 失败结论保持原样。

## 产物与复现

- `analysis_outputs/v13_loss_conflict_20261014/report.json` 与 `gradients.npz`。
- `analysis_outputs/v13_reference_band_diagnosis_20261013.json` 与 `analysis_outputs/v13_reference_band_figures_20261013/`；文件名沿用生成时日期，不改写历史结果。
- `analysis_outputs/v13_slow_drift_actual_model_20261014/report.json` 与 7 个人工案例 NPZ。
- `analysis_outputs/v13_followup_figures_20261014/slow_drift_recording_error.png` 与图数据收据。
- `analysis_outputs/v13_followup_diagnostic_verification_20261014.json`：最终复核绑定与测试记录。

```powershell
.\.venv-ml-gpu\Scripts\python.exe -B scripts/diagnose_ssvep_v13_loss_conflict.py `
  --output-directory analysis_outputs/v13_loss_conflict_NEW `
  --allow-failed-selection-diagnostic --device cpu --save-gradients

.\.venv-ml-gpu\Scripts\python.exe -B scripts/diagnose_ssvep_v13_slow_drift.py `
  --output-directory analysis_outputs/v13_slow_drift_NEW `
  --channel-dc-offsets --include-failed-v13 --device cpu --save-arrays
```

每次复核需使用尚不存在的独立输出目录。report.json 为最后提交的完整诊断标记；所有运行/生产/生理验收仍为 false，项目目标未完成。
