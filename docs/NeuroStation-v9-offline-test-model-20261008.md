# NeuroStation v9：可运行的 EEG/SSVEP 离线测试模型

日期：2026-10-08。配置状态为 **formal**，用途是**显式离线测试**，并非生产或临床验收。没有修改采集协议、桌面默认通道配置或实时工作站的安全检查。

## 1. 本次真正交付了什么

- 可加载权重：`C:\Users\Administrator\Documents\ChatGPT\脑机采集\models\denoiser_v9_ssvep.pt`。
- 权重清单：`C:\Users\Administrator\Documents\ChatGPT\脑机采集\models\denoiser_v9_ssvep.manifest.json`。
- 一条命令运行的 Windows 演示：`C:\Users\Administrator\Documents\ChatGPT\脑机采集\scripts\test_ssvep_model_demo.ps1`。
- 自己的数据入口：`C:\Users\Administrator\Documents\ChatGPT\脑机采集\scripts\start_ssvep_test_model.ps1`。
- 实际 CPU/GPU 输出、样本保留 mask、来源/输出 SHA256 和处理报告。

这不是新一轮训练，也不是把失败的 v8 换个名字。v9 **保留 v8 的已训练权重不变，新增固定的保守残差约束，然后独立重新执行内部工程评估**。v8 的训练、失败诊断和旧锁均保留。

v8 先前已使用 RTX 4090 完成 18 epochs 训练：fit subjects1–4、开发期选择 subjects5–6。v9 不改 epoch、不搜索超参数、不放宽门槛，采用同一套 4,200 个开发案例/污染种子进行配对比较。所有结果仍属于已经参与开发的 subjects5–6，不能作为独立受试者验证。

## 2. 先跑准备好的例子

在 PowerShell 中执行，**无需连接 EEG 硬件**：

```powershell
$root = 'C:\Users\Administrator\Documents\ChatGPT\脑机采集'

# 仅检查模型/代码/证据完整性，不加载权重、不初始化 CUDA、不打开硬件。
& "$root\scripts\start_ssvep_test_model.ps1" -CheckOnly

# GPU 演示，默认 Wearable / 250 Hz。
& "$root\scripts\test_ssvep_model_demo.ps1" -Device 'cuda:0'

# 无 GPU 时显式选择 CPU；另一个域必须使用原生 256 Hz。
& "$root\scripts\test_ssvep_model_demo.ps1" -Domain Nakanishi2015 -Device cpu
```

每次演示自动创建新的时间戳输出目录，不覆盖旧结果。使用已有 `.venv-ml-gpu` 环境，不会自动安装依赖、启动后台服务或修改系统设置。

例子取自此前已物化到项目的开发 subject5/trial0 数据，追加合成污染；本轮没有访问 `D:\SSVEP` 或读取7–10号受试者波形。例子包含5种条件、无效 mask 窗口、平直通道窗口和31个不足整窗的尾样本，共2,079个样本。实际验证：处理1,536个，原样保留543个。这个示例只是运行链路展示，不是干净 EEG 真值数据。

## 3. 测试自己的文件

### 输入契约

- 支持 `.npy`、`.npz` 和无表头纯数值 `.csv`。
- 波形必须为 `[时间样本数, 8]`，float32 或 float64，至少256样本。CSV 只能包含8个波形列，不能混入时间戳/标签列。
- NPZ 默认波形键 `signal`；需要其他键时使用 Python 入口的 `--signal-key`。
- 显式选择域和原生采样率：Wearable=250 Hz，Nakanishi2015=256 Hz。不做自动重采样，不凭采样率判断来源域。
- 提供8个唯一的输入通道名称，顺序与文件列顺序一致。冻结的数据清单没有已核验的解剖通道顺序，因此**填写名称不是通道/电极对应验证**。不同硬件、通道布局、电极类型和数据单位的迁移效果均未验证。
- 明确提供 `bool [时间样本数]` 的 `.npy` 有效样本 mask。若只是技术性尝试，可以显式用 `-AllFiniteValid`，但“数值有限”不意味着 EEG 质量合格。
- 必须显式确认离线用途和未核验通道映射。确认只是记录测试意图，不等于完成质量/硬件验收。

例子：替换路径和通道名称，但不要凭空填写真实电极位置。

```powershell
$root = 'C:\Users\Administrator\Documents\ChatGPT\脑机采集'
& "$root\scripts\start_ssvep_test_model.ps1" `
  -InputFile 'C:\MyEEG\signal.npy' `
  -Domain Wearable -SampleRateHz 250 `
  -ChannelOrder 'channel-1,channel-2,channel-3,channel-4,channel-5,channel-6,channel-7,channel-8' `
  -ValidMask 'C:\MyEEG\valid_mask.npy' `
  -AcknowledgeOfflineTest `
  -AcknowledgeUnverifiedChannelMapping `
  -Device 'cuda:0' `
  -OutputDirectory "$root\analysis_outputs\my_v9_test_001"
```

输入文件只读，输出必须是项目 `analysis_outputs` 下尚不存在的新目录。错误采样率、错误形状/mask、未确认测试用途、文件 SHA 不符、已有输出目录都会被拒绝。

### 输出说明

- `denoised.npy`：float64，形状不变，使用输入原本的数值单位；不假定是微伏，也不做单位转换。
- `processed_mask.npy`：bool，明确标记哪些样本实际经过模型。没有处理的窗口不是降噪成功。
- `report.json`：来源/模型/输出 SHA256、CPU/GPU信息、每窗改动与缩放值、跳过原因、边界跳变诊断。该文件最后写入，作为完整结果标记；中断留下的目录不应视为完整交付。

如需要导入 NeuroStation：当前不要用输出覆盖原始会话，也不要替换实时数据源。它是独立副本，用于离线对比。

## 4. 模型变化与当前结果

v9 输入依然只有 noisy-normalized `[B,8,256]`。先执行冻结的 v8，然后用**每个窗口一个全局非负缩放量**把归一化残差均方限制在 `0.00045`，保留舍入裕量。不把受试者、污染类型、标签或 reference 送入推理模型。

原工程门没有变化：高频 burst 重建误差下降至少0.05，其他污染条件不得为负，identity mean normalized change不高于0.0005，保护带复能量比不高于1e-6、相位偏差不高于0.02 rad。

| 内部数据域 | 高频 burst 重建误差下降 | identity normalized change | 总工程门 |
|---|---:|---:|---|
| Nakanishi2015 | 0.1418080894（14.18%） | 0.0001860423 | 通过 |
| Wearable | 0.0975027847（9.75%） | 0.0001740185 | 通过 |

**百分比是“记录参考 + 合成污染”实验中 ratio-of-mean reconstruction error reduction，不是实际原始 EEG 的噪声去除率、SNR提升或临床有效性。**

保守限制有代价：Nakanishi高频重建误差下降从v8的0.3573减到v9的0.1418；Wearable identity从v8的0.0009939减到v9的0.0001740。这个版本优先降低误改风险，不能描述成所有指标都提升。

## 5. 固定窗口与重要限制

- 非因果、固定256样本窗口、hop=256。没有做未经验证的 OLA、边界平滑或实时流式接入。
- 8–40 Hz 保护带约束和验证针对每个固定窗口。**拼接后的整段 FFT 不保证保真**，也不保证相邻窗口无边界效应；报告包含新增边界跳变诊断。
- 任一通道存在非有限值、mask内有无效样本或通道窗口近似平直时，整窗保留。尾段不足256样本也保留，不插值、不补造数据。
- 归一化采用噪声输入每通道均值/RMS和固定1e-8数值稳定项。训练清单未核验原生单位，所以不同量纲输入的域迁移仍不确定，不要把“自动恢复输入单位”当成单位校准或效果保证。
- recorded reference不是clean ground truth；还没有实际任务分类准确率、独立受试者、临床或新硬件会话验收。
- `runtime_enabled=false`、`production_validated=false`；只有明确发起的离线测试会执行。

## 6. 已完成的真实验证

- RTX4090 / CUDA完成4,200-case推理诊断；两域原门通过。
- JSON-only算术/来源审计通过，保留失败语义，不靠修改阈值。
- 两域分别执行真实CPU和CUDA推理，逐窗比较输出，原始输入SHA保持一致。
- 实际输出文件核验了无效窗口、平直窗口和尾段原样保留、处理样本mask、归一化残差预算、逐窗保护带能量泄漏和CPU/GPU一致性。
- 禁止torch import的独立子进程中，`--check-only`仍成功，确认该模式不反序列化checkpoint、不初始化CUDA。
- v8历史及v9执行源码共83个冻结绑定复核通过。
- 专项回归283项通过（旧v8的246项 + 新增37项）；正式采集核心109项通过；UI65项通过；正式配置检查通过，共457项测试。
- 首次UI测试因测试目录参数错误而未找到模块；未修改应用代码，改用正确的`apps/workstation_ui/tests`路径后65项通过。初次失败日志保留，最终日志另存。

主要证据：

- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\ssvep_v9_internal_diagnostic_20261008.json`
- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\ssvep_v9_saved_audit_20261008.json`
- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\ssvep_v9_end_to_end_verification_20261008.json`
- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\ssvep_v9_specialized_regression_20261008.log`
- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\ssvep_v9_core_regression_20261008.log`
- `C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\ssvep_v9_ui_regression_final_20261008.log`

checkpoint SHA256：`84fd1141c2b850c695520d4b5ccbda6945e9a713614f2cce9de783b6e1edc82f`。

完整权重清单、配置、冻结锁和相关源码需保留在本项目内；目前不是只拷贝一个PT就可脱离项目运行的独立安装包。

## 7. 后续提升优先级

下一阶段宜先用自己的真实会话做**只读离线对比和任务指标评估**，明确单位、通道映射和有效样本，再预注册独立验证/必要的新训练。不要在本次开发案例上继续调参后把结果称为独立通过。实时接入之前还需要连续窗/边界/时延验收，且仍保留硬件握手、通道完整性、显示时序预检、光敏风险确认和中止控制。
