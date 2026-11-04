# NeuroStation v14 个人数据诊断与最终复核（2026-10-14）

## 结论先行

本轮完成了 v14 失败选择 checkpoint 的**显式诊断输出**，但没有把它接入默认桌面运行时，也没有把诊断结果当作生理 EEG 恢复或注视准确率。

两份个人采集数据都能够被完整读取、生成 `raw/notch/traditional/v14` 四条输出轨，并完成离线 CCA 计算。因此，当前证据不支持“原始文件损坏到模型完全无法输出”的判断。更准确的结论是：

- 数据可以处理，样本索引连续性此前也通过诊断；
- v14 的失败原因是训练选择门槛未通过，而不是个人文件读取失败；
- v14 在个人数据上的呈现标签匹配没有超过 raw/notch/traditional 的稳定改善证据；
- v14 checkpoint 仍然只能以失败选择诊断角色加载，默认 loader 会拒绝；
- 下一步应优先解决时序、硬件标定和独立真值问题，而不是强行发布当前 checkpoint。

## 本轮产物

### 诊断输出

- `analysis_outputs/v14_evaluation_20261014_diagnostic/summary.json`
- `analysis_outputs/v14_evaluation_20261014_diagnostic/personal_00/report.json`
- `analysis_outputs/v14_evaluation_20261014_diagnostic/personal_01/report.json`
- 每个个人会话下的 `raw.npy`、`notch.npy`、`traditional.npy`、`v14.npy`

### 可重复图表

- `analysis_outputs/v14_diagnostic_figures_20261014/figures_receipt.json`
- 两个会话各有：居中波形、全记录频谱、记录呈现匹配代理图。
- 图表是描述性诊断，不是质量分数；呈现匹配不是注视准确率。

### 最终复核

- `analysis_outputs/v14_final_verification_20261014/report.json`
- `scripts/plot_ssvep_v14_diagnostic.py`
- `scripts/verify_ssvep_v14_final.py`
- `tests/test_plot_ssvep_v14_diagnostic.py`
- `tests/test_verify_ssvep_v14_final.py`

## 个人数据结果

采样率按 250 Hz 解释，通道数为 8，原始数组保持不变：

| 会话 | 数据形状 | raw 匹配 | notch 匹配 | traditional 匹配 | v14 匹配 |
|---|---:|---:|---:|---:|---:|
| `session_20261004_004101_105` | 24371 × 8 | 9/12 | 9/12 | 9/12 | 9/12 |
| `session_20261004_004555_413` | 24372 × 8 | 8/12 | 8/12 | 8/12 | 8/12 |

这里的“匹配”只表示输出轨的强制 argmax 与记录中的呈现目标相同，**不表示用户当时确实注视该目标，也不表示准确率**。报告中的 `gaze_accuracy_claim`、`independent_personal_validation` 和 `source_domain_validated` 均为 `false`。

v14 平均 top-two score margin：

- 会话 00：`0.6035`；raw 为 `0.6025`，notch 为 `0.5979`；
- 会话 01：`0.4960`；raw 为 `0.4864`，notch 为 `0.4959`。

这些 margin 不是概率，也未经过校准；它们只说明当前透明诊断解码器在各输出轨上的相对分数间隔。

## 对“数据是否有问题”的判断

当前没有证据证明原始采集文件整体损坏：

1. 两份源目录在诊断前后 SHA-256 一致；
2. 四条输出轨的样本长度和通道数一致，数组均有限；
3. 事件窗口都能完成离线计算，没有出现全会话不可用；
4. raw、notch、traditional、v14 的匹配计数相同，说明“v14 不输出”不是因为文件无法读或数组契约不成立。

但数据仍存在会影响 SSVEP 解码可信度的条件：

- `timestamp_s` 不是严格逐样本 250 Hz 时钟，应以连续 `sample_index` 为主；
- 事件 sample-index 时长与 host `recording_time_s` 存在约 1% 左右差异，个别试次更大；
- 真实硬件增益、单位和电极布局尚未独立确认；
- 没有独立 gaze truth；
- 没有可信的 clean physiological EEG 参考。

所以目前应把问题描述为“数据可处理，但时序/硬件/标签真值尚未达到可验收条件”，而不是简单说“数据坏了”。

## v14 发布状态

v14 训练在 RTX 4090 上完成了 16 个 epoch，但没有任何 epoch 通过 selection gate：

```text
selected_epoch = null
selection_gate_passed = false
checkpoint_role = diagnostic_last_epoch_no_eligible_checkpoint
runtime_enabled = false
production_validated = false
default_deployment_authorized = false
```

因此：

- 默认 `api.load_model(...)` 继续拒绝该模型；
- 只有显式 `--allow-failed-selection-diagnostic` 才能生成本轮输出；
- 不修改默认桌面、不启动 shadow denoiser、不把 v14 输出覆盖原始数据；
- 旧 v8–v13 结果和个人源文件均未改写。

## 验证结果

- v14 runtime 合约测试：通过；
- v14 evaluator 合约测试：通过；
- v14 诊断图表和最终复核测试：通过；
- 正式配置检查：通过，`protocol_status=formal`，无 warnings；
- 正式采集默认配置测试：在 `.venv`（含 BrainFlow 5.22.2）通过 11/11；
- 采集 worker / learned monitor 回归：通过 26/26；
- 同一批正式配置测试若在 `.venv-ml-gpu` 运行，会因该环境没有 BrainFlow 而出现 2 个环境依赖错误；代码在含 BrainFlow 的 `.venv` 中已复核通过。

## 下一步建议

1. **不要发布 v14**。当前最安全的状态是保持诊断产物和 fail-closed loader。
2. 给下一次采集增加逐样本设备时钟/包序号记录，并把刺激 onset/offset 直接绑定到 sample index。
3. 记录电极布局、增益、单位、参考电极和阻抗/接触质量；在会话前后做硬件 sanity check。
4. 增加独立 gaze truth（例如摄像头或明确的用户注视协议），否则不能把呈现匹配转成准确率。
5. 优先收集可配对的 clean/noisy 参考，或把目标拆成两个模型：
   - 保守的原始波形质量监测/工频抑制；
   - 仅用于 SSVEP 分析的特征级增强。
6. 只有新的独立验证集同时通过保真、伪迹改善、时序和安全门槛后，才重新考虑训练和部署。

## 事件窗口敏感性补充诊断

为区分“数据不可用”和“事件窗口对齐不稳”，又执行了一个预注册的 sample-index 窗口裁剪网格：起始裁剪和结束裁剪分别取 `0/50/100/125/150/200` 个样本。该网格只描述敏感性，不选择新的部署参数。

产物：

- `analysis_outputs/v14_event_window_sensitivity_20261014_rerun/summary.json`
- 每个个人会话的 `report.json`
- `scripts/diagnose_ssvep_v14_event_window_sensitivity.py`
- `tests/test_diagnose_ssvep_v14_event_window_sensitivity.py`

观察结果：

- 会话 00：四条轨在固定网格中的呈现匹配最高仍为 9/12；当前 `125/0` 规则就是 9/12。
- 会话 01：raw/notch/v14 在固定网格中仍为 7–8/12；traditional 为 8/12。部分裁剪只改变 score margin，没有产生可证明的匹配提升。
- 因此，目前不能把输出不稳定简单归因为某一个固定的事件窗口偏移，也不能据此重新选择部署窗口。

这个结果进一步支持：下一轮应该先补采集时钟、刺激 onset 绑定和独立 gaze truth，再考虑重新训练/选轮，而不是在当前个人记录上继续调参直到得到较高匹配数。

## 新增 v2 个人解码候选：后部视觉通道 CCA

为了让采集信号更接近真正的 SSVEP 分析输入，新增了一个不改变原始数据的 v2 诊断解码器：

- 配置：`configs/personal_ssvep_decoder_v2.json`
- 实现：`eeg_tools/personal_ssvep_decoder_v2.py`
- 评估：`scripts/evaluate_personal_ssvep_decoder_v2.py`
- 输出：`analysis_outputs/personal_ssvep_decoder_v2_20261014/`

v2 按通道配置中预先声明的后部视觉通道 `CH5/CH6/CH7/CH8`（P7/P8/O1/O2）进行 CCA，前向计算不读取 recorded label。这个选择是基于 SSVEP 通常在后部视觉区域更有解释力的先验，不是对当前两份数据的标签调参；但物理电极位置仍需现场复核。

v2 结果：

| 会话 | v1 raw | v2 raw | v1 v14 | v2 v14 |
|---|---:|---:|---:|---:|
| `session_20261004_004101_105` | 9/12 | 9/12 | 9/12 | 9/12 |
| `session_20261004_004555_413` | 8/12 | **9/12** | 8/12 | **9/12** |

这说明当前全通道 v1 确实受前部/非视觉通道噪声影响，但也暴露出一个更重要的限制：两份记录的 20 Hz `bottom_right` 试次都被判成 10 Hz `top_left`，合计 6 个错误。频谱诊断中 10 Hz 成分在这些试次占主导，而没有足够证据证明 20 Hz 生理响应存在。

因此 v2 是向“能够从采集信号输出目标信号”的方向推进了一步，但仍不能宣称 12/12 正确，也不能绕过独立注视真值。它目前只作为显式离线诊断候选，未接入默认桌面运行时。

## 目标频率证据与刺激时序补充

新增只读证据诊断：

```text
C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\personal_ssvep_presence_20261014_r2
```

软件刺激帧序列的频谱峰与协议目标一致：10/12/15/20 Hz 均能从 `frame_timing.tsv` 中复现，说明软件协议和记录的帧标签没有明显的“20 Hz 被写成别的频率”的错误。

但 EEG 后部通道上的目标频率幅度相对上一段 rest 的比值不稳定：

- 会话 00 的 `bottom_right/20 Hz` 三次约为 `0.34/1.53/1.08`；
- 会话 01 的 `bottom_right/20 Hz` 三次约为 `0.89/0.79/0.96`。

这与 v2 中所有 20 Hz 试次容易被 10 Hz `top_left` 竞争相互印证：软件刺激时序本身看起来正确，但当前 EEG 记录没有稳定的 20 Hz 响应证据。此时继续训练一个波形模型把 10 Hz 伪迹“修成”20 Hz，会变成信号伪造而不是质量优化。

因此当前最合理的系统行为是：保留原始信号、给出 v2 的后部通道诊断结果，并在没有足够频率证据时拒绝宣称“正确目标”，而不是强制输出一个看似确定的标签。

