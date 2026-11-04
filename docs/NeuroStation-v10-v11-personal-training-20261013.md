# v10/v11 真训练与个人信号输出诊断

本轮在RTX4090上各完成24轮v10、v11训练。新版本使用固定50Hz/Q30零相位notch加上3,625参数的神经频谱衰减。v10因未加噪声参考的改动超阈值而失败，权重、配置、报告保留。v11提高identity损失权重至100并降低学习率到0.0003，选用第22轮。两者都没有启用实时运行或生产验收。

## 训练与数据角色

公开Wearable原生250Hz、8通道参考：subject1–4的960个试次用于fit，5–6的480个试次用于selection，另480个合成参考用于fit。每轮加入identity、漂移、高频burst、高斯、组合五种条件，五类均衡。subject7–8的480个试次在选定模型后评价；这些数据在项目中曾出现，本轮也先评价v10再评价v11，因此是复用的开发评价，不能声称首次独立holdout。

采集参考先notch，再截取真实125:637样本，不重采样、不推断个人电极映射。参考是工程条件化的记录波形，不是clean EEG生理真值。呈现目标和污染条件不进入模型forward；未用用户两份会话训练或选epoch。

## v11研究评价范围

未加人工噪声的公开参考：输出归一化MSE为0.001473，低于预设0.002。神经阶段对四类人工污染的合并重建误差下降65.88%，按类别如下。这是归一化参考重建指标，不是个人EEG噪声去除率。

| 人工污染 | 神经阶段误差下降 |
|---|---:|
| drift | 70.05% |
| high_burst | 93.23% |
| gaussian | 30.20% |
| combined | 67.09% |

保护指标仅涉及512样本FFT窗口的8–40Hz离散频点；不能直接扩展成整段波形保证。4096样本已知数值参考另外检验基频和二次谐波，四频率的最大幅度比例变化约1.27%，最大相位差约1.59e-5rad；仅证明这组人工信号上的结果。

## 整段波形的实际限制

下表是20个长波形案例、按四个频率平均的内部区域mean-centered NMSE，越小越接近已知数值参考。

| 长波形条件 | 原始 NMSE | notch | 固定传统滤波 | v11 |
|---|---:|---:|---:|---:|
| identity | 0 | 7.91752e-07 | 0.00111379 | 1.77768e-05 |
| line50 | 3.54284 | 7.91756e-07 | 0.00111854 | 1.77772e-05 |
| drift.3 | 0.884504 | 0.884505 | 0.00111378 | 0.856585 |
| gaussian | 0.441274 | 0.428068 | 0.140308 | 0.415839 |
| combined | 4.87101 | 1.31638 | 0.141667 | 1.28419 |

工频改善主要来自固定notch，不能归功于神经网络。对长波形的漂移/高斯/组合噪声，固定传统滤波基线仍明显优于v11。v11不能据此宣称已经达到最佳去噪效果。

还核验了一个实现差异：训练输入以参考RMS为单位；整段推理再次按每个含噪窗口的均值/RMS归一化。用同一公开评价窗口改变到推理约定后，Gaussian误差改善从约30.2%降到9.5%，combined从约67.1%降到40.6%。这是下一版本必须修复的训练/推理契约差异。详细证据在 `analysis_outputs/v10_personal_20261013/normalization_contract_diagnosis.json`。本轮未改写已经冻结的v10/v11结果，也未因研究gate通过就接到默认桌面或实时链路。

## 两份个人数据输出

| 会话 | 原始CCA呈现匹配 | notch后 | v11后 |
|---|---:|---:|---:|
| session_20261004_004101_105 | 9/12 | 9/12 | 9/12 |
| session_20261004_004555_413 | 8/12 | 8/12 | 8/12 |

原生24,371和24,372样本的8通道输出均保持形状和有限值，按512窗、128步长、正权重Hann归一化overlap-add完整覆盖。两份来源26个文件均未变化。49–51Hz功率下降约98.1%–99.98%，8–40Hz整段频带功率变化约−0.27%至−0.66%；频带功率变化不是生理恢复或噪声去除的证明。

个人会话仅有呈现标签，缺独立注视真值，属于开发数据。另发现20个人工案例里原始CCA就有3个基频/谐波候选混淆；v11没有改变这些候选。未把原始解码错误改写为模型损傷结论，也没有放宽去噪identity阈值。实际幅度/相位检查与候选解码分别保存。

## 可复核产物与运行

- `models/denoiser_v10_personal.pt`、对应manifest：失败训练版本，保留错误门槛。
- `models/denoiser_v11_personal.pt`、对应manifest：真实训练的研究版本，默认运行关闭。
- `analysis_outputs/ssvep_v10_training_20261013.json`、`ssvep_v11_training_20261013.json`：完整24轮记录和选择原因。
- `analysis_outputs/v11_evaluation_20261013/summary.json`：完整研究评价、局限和每份个人结果位置。
- `personal_00/raw.npy`、`notch.npy`、`v11.npy`及`personal_01`对应文件：三条独立完整波形；`report.json`有逐试次预测、SHA及处理receipt。
- `public.json`、`synthetic.json`、`waveform_preservation.json`、`personal_spectral_diagnostic.json`：公开人工污染、长波形、幅相与个人频谱证据。

复现评价需选新目录，脚本拒绝覆盖已完成输出：

```powershell
.\.venv-ml-gpu\Scripts\python.exe scripts/evaluate_ssvep_denoiser_v11.py --output-directory analysis_outputs/v11_repeat --personal-session C:\Users\Administrator\Documents\NeuroStation\Datasets\session_20261004_004101_105 --personal-session C:\Users\Administrator\Documents\NeuroStation\Datasets\session_20261004_004555_413 --device cuda:0
```

当前桌面仍使用此前v9离线结果与CCA开发诊断。新权重没有默认替换桌面模型或重建旧EXE。下一步先修复训练/推理归一化契约，再用同样整段波形、噪声、幅相和任务对照检验；正确恢复个人生理信号仍需可信参考和独立新采集。
