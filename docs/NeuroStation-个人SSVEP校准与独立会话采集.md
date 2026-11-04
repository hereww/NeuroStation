# 个人 SSVEP 校准与独立会话采集

日期：2026-10-15。对应工具：`scripts/validate_personal_ssvep_session_contract.py`。

当前目标是获得可用于个人校准的新数据，并预留未参与调参的独立会话。现有软件按单半屏逐频呈现刺激；这能检查刺激响应和呈现目标代理标签，尚不能验证用户在同时出现的左右目标间自主选择。

## 已有数据的实际检查结果

报告：`analysis_outputs/personal_ssvep_session_contract_20261015.json`。

| 会话 | 样本数/通道 | 每个频率的试次数 | 刺激事件配对 | 人工注视标记 | 独立眼动证据 |
| --- | ---: | ---: | --- | ---: | --- |
| session_20261004_004101_105 | 24,371 | 3 | 正常 | 0 | 无 |
| session_20261004_004555_413 | 24,372 | 3 | 正常 | 0 | 无 |

两份会话读取期间文件哈希一致。它们没有达到本次工具默认的每目标 8 次门槛；`quality.json` 的显示节奏为 `review`，并记录时间戳间隙。旧协议的 `formal_candidate` 状态原样保留，工具可以读取它，但不会把它视为新的 `formal` 采集协议。历史状态本身不是信号损坏的证明。时间戳间隙仅提示复核，工具不据此判定硬件丢包。

## 下一次如何采集

1. 从当前工作区启动 NeuroStation：`.\.venv\Scripts\python.exe workstation.py`。使用正式 SSVEP 采集入口，在界面完成设备握手、8 通道配置、实际电极佩戴复核、设备/显示预检和光敏风险确认。通道预设名称不能代替实际佩戴确认。保留 Esc/Q 中止路径。
2. 为同一参与者安排三个新会话：A 用于校准，B 用于开发验证，C 用于最终评估。会话名分别写明用途，保留各自自动生成的唯一 session_id。采集前固定分组；整段录制不能复制后充当另一组，也不要将同一试次的相邻窗口分到不同组。两份旧会话已用于开发，仅作历史诊断。
3. 每个会话建议保留正式协议默认的 10 次/目标（四频共 40 试次），刺激 5 秒、试次间休息 3 秒；当前校准审计默认最低 8 次/目标。每次新会话单独达到重复次数要求。保留会话初始和试次前的默认休息，供刺激/静息对照。
4. 采集时按提示注视；记录任务要求、是否确实执行、走神/眨眼等偏离情况及记录人。`events.tsv` 的 `presented_target_id` 仅表示软件呈现。刺激窗口按 1–4 键产生的是人工/自报事件 `gaze_target_mark`，既不独立测量注视，也不等同于任务指令记录。它可提供额外信息，但仍是代理标签。
5. 在会话外保存一份采集记录，分别描述“任务指令”“软件呈现”“人工自报”“独立眼动观测”，注明每项实际来源；缺失的项写“未采集”。现有采集器尚未自动生成这份独立任务记录，不要把预测结果填回原始事件充当标签。
6. 会话结束先审计 A、B；C 按预先分组保存，等模型、窗口与门限固定后才做最终评估。采集终止、时序复核、饱和/平直、接线与参考等问题须保留记录和排查，不能靠降噪掩盖或删列绕过 8 通道契约。

没有眼动仪时仍可以开展 A/B 的呈现目标代理校准，无需伪造独立注视标签；结果应报告为呈现目标匹配、接受比例和错误指令比例。当前四频 CCA 可处理 10/12/15/20 Hz；小球三类 MLP 只支持 10/12/15 Hz，20 Hz 不能按已支持类别解释，未知输入拒绝也未获得保证。

默认协议当前使用固定随机种子，不同会话可能重复相同试次顺序。文件身份隔离不能排除顺序、时间漂移和学习效应；这需要后续预先确定并实现不同会话的顺序安排。当前工具不会自动改刺激协议或启动真实采集。

## 只读检查命令

在项目根目录执行，将示例目录替换为实际新会话。报告必须保存在所有源会话之外，使用未存在的文件名。

```powershell
.\.venv\Scripts\python.exe scripts/validate_personal_ssvep_session_contract.py `
  --session "C:\Users\Administrator\Documents\NeuroStation\Datasets\<校准A的会话目录>" `
  --session "C:\Users\Administrator\Documents\NeuroStation\Datasets\<开发B的会话目录>" `
  --output analysis_outputs/personal_ssvep_new_sessions_audit.json
```

退出码 0 和 `completion_status: complete` 仅表示审计执行成功；查看各会话的 `blocking_checks` 和 `events.structural_issues`，不要把执行成功当作模型验收。默认检查每个协议目标至少 8 次，包含完全缺失的类别。数值有限、元数据匹配、软件质量状态通过仍不能确认生理信号有效性。

工具验证原始列名、连续 sample_index、样本数与数值有限性，以及 onset/offset 配对、索引范围、目标/频率一致性、窗口长度和重复次数；读取已保存的握手、质量与显示节奏状态。它不连接硬件，不计算模型准确率，不重新训练，不验证包连续性，也不进行实际电极或物理显示延迟验收。

A/B 通过输入检查并完成人工质量复核后，可使用既有会话评估器：

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_personal_ssvep_decoder_v4_sessions.py `
  --calibration-session "C:\Users\Administrator\Documents\NeuroStation\Datasets\<校准A的会话目录>" `
  --evaluation-session "C:\Users\Administrator\Documents\NeuroStation\Datasets\<开发B的会话目录>" `
  --output analysis_outputs/personal_ssvep_new_A_to_B.json
```

该评估器会拒绝重复路径、session_id 或相同原始 EEG 哈希。审计器是单独的前置步骤，尚未嵌入评估器或界面；来源隔离也不代表 C 是未使用过的最终留出。B 的结果可用于开发，C 的结果不能再用于挑选模型或门限。

## 有独立眼动记录时的证据契约

当前没有眼动采集或同步驱动。新增工具可读取会话内可选的 `gaze_truth.json`，仅检查证据可追溯性。接入真实眼动设备后，需要保存原始眼动、校准、同步和目标区域映射文件，并用实际同步规则生成 EEG 样本坐标下的试次标注。不能根据 EEG 预测反推标签。

文件格式字段如下；这是填写说明，不是已完成的采集记录：

| 字段 | 要求 |
| --- | --- |
| schema_version | 整数 1 |
| session_id | 与 EEG session.json 完全一致 |
| source_type | eye_tracker |
| independent | 实际来源独立于 EEG 预测和呈现标签时才能声明布尔 true |
| evidence_files | 四项文件，role 为 raw_gaze、calibration、synchronization、target_mapping；每项含 path 和真实 sha256 |
| trial_labels | 每个试次恰好一项，含 trial_index、target_id、start_sample_index、end_sample_index |

证据文件路径须为会话内相对路径，四项角色使用不同的非空文件；哈希必须与当前文件一致。试次索引须对应真实 EEG 刺激试次、目标必须属于协议，区间以 EEG 样本索引表示且完全位于该刺激区间内，end_sample_index 为右开边界。来源类型、文件名或单个 independent 声明都不足以通过契约。

即使 `evidence_contract_passed: true`，也只说明文件哈希、试次覆盖和区间结构符合格式，不代表校准误差、同步精度、注视覆盖比例或标签真实性已验收。`independent`、`independent_personal_validation_candidate`、`production_validated`、`runtime_enabled` 和 `default_deployment_authorized` 继续为 false，必须另外完成证据人工复核与模型性能验证。
