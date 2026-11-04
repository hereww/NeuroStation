# NeuroStation 学习型 EEG 质量优化模型 v3

- 文档状态：`formal`。
- 适用范围：**仅显式离线诊断**，不接入采集工作站自动处理，不替换硬件验收。
- 预注册日期：2026-10-05。
- 预注册依据：`C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\learned_denoiser_v3_preregistration.json`。
- 主线程实际 CUDA micro-smoke：**`passed`**，见第 7 节已有报告；本 sidecar 未重新执行。
- v3 数据训练、validation、known benchmark 与 S4/S5 holdout 已完成；研究验收**未通过**。训练、评估和报告均为离线派生，仍不构成生产批准。

## 1. 正式状态与证据边界

`formal` 表示按项目正式配置及交付规则管理；不表示硬件、采集会话质量、去噪有效性、临床有效性或生产部署自动验收。

v3 的用途是将**冻结的 v2 单通道去噪器**与**单独学习的 applicability/domain gate**组合，判断当前窗口是否落在已支持的训练分布内。不得重新训练 v2，也不得通过新 gate 改写历史 v2 会话结论。

- 不支持域、门控拒绝、无效输入或不安全结果：**保留 raw**。
- 保留 raw 是安全行为，不是有效去噪证据；拒绝率高或全 no-op 不能单独说明模型优良。
- 实测 Dong EEG 是“不支持域”安全数据，不是已知 clean EEG，不能作为真实 noise/clean paired ground truth。
- supported-domain score、artifact score、confidence 均不视为已校准概率；尤其不能解释为“去噪成功概率”。
- 去噪候选与实际门控输出应分开报告，包含全部预定窗口、拒绝情况和无变化情况，不得只展示 selective 较好部分。
- 新门控证据不能替代真实 paired 去噪有效性、下游刺激识别验收或 production 验收。

## 2. 预注册冻结方案

以下是实验计划，不是实际完成记录。改动须由主线程显式记录新版本，不得根据 test/holdout 结果事后调参。

| 项目 | 冻结设置 |
| --- | --- |
| 实验 | `learned_denoiser_v3_domain_gate` |
| 随机种子 | `20261005` |
| 训练设备 | CUDA required；无静默 CPU fallback |
| 训练上限 | 20 epochs |
| batch size / learning rate | 128 / 0.001 |
| CPU threads | 2 |
| gate | normalized single-channel domain CNN |
| 卷积通道 | `[16,32,32]` |
| kernel / stride | `[9,7,5]` / `[2,2,2]` |
| hidden features / dropout | 32 / 0.0 |
| 输入 | float32 `[B,1,500]`，250 Hz |
| 归一化 | 每窗口 noisy mean 与 AC RMS，与 v2 一致 |
| 禁止输入 | 路径、subject ID、幅度物理单位、raw DC |
| supported / unsupported 标签 | 1 / 0；不是 clean/noisy 真值 |
| loss | balanced binary cross entropy with logits |
| checkpoint 选择 | 最小 validation balanced BCE；精确 tie 选最早 epoch |
| gate 阈值 | `max(0.5, unsupported-validation empirical order statistic)`，使严格 `>` false acceptance ≤1% |
| supported validation | 报告 acceptance；不得据此降低阈值 |
| v2 | 全程冻结，保留预注册 protected SHA-256 |

本轮数据边界继续采用冻结的 EEGdenoiseNet 12k 派生数据，不临时加入新数据。Klados 官方 RAR 的 clean 成员 CRC 出错，**不得**用于 paired 训练或 paired 有效性主张；不能以损坏内容或未核验替代 clean 作为真值。

## 3. 训练 / validation / 新 holdout / 已知 benchmark

| 用途 | 数据 | 独立性解释 |
| --- | --- | --- |
| supported train | 冻结 `learned_denoiser_v2.npz` 的 train split | 继承 v2 split，不新增 subject-disjoint support 主张 |
| supported validation | 同一冻结 NPZ 的 validation split | 仅用于预注册 checkpoint/门控选择流程 |
| unsupported train | Dong `S6.mat`、`S7.mat` | 全 targets / blocks / channels，不能当 clean 标签 |
| unsupported validation | Dong `S8.mat`、`S9.mat` | 只供 unsupported false-acceptance 阈值选择 |
| 新 unsupported holdout | Dong `S4.mat`、`S5.mat` | 与上面 subjects 不交叉；只衡量不支持域安全性，不证明 paired 去噪 |
| known v2 benchmark | 冻结 NPZ 的 test split | **已知 v2 benchmark，not new holdout**；不能重新包装成新的独立测试 |
| known session regression | `session_20260924_131134_807` | 历史已知 stress session，`independent=false`，不得用其结果调 v3 |

Dong 使用审核 metadata 指定的刺激区间 `[125:1125]`，共 1000 samples / 4 秒；每通道分成 500-sample 窗口。保留全部预定窗口，最大允许排除数为 0。

最终 Dong metadata 的 `target_axis_mapping_confirmed=false`、`target_frequencies_hz=null`。不得照抄候选 ascending 顺序或通过谱峰反推后声称作者确认。真 target-labelled accuracy 和 target-specific gate 保持 `not_assessed_target_mapping`；可报告 mapping-independent 频率集合谱保持和 prediction agreement，但 agreement 不是 accuracy。作者 visual latency `0.16s` 单列，不自动加到 `[125:1125]`。

**训练代码在 freeze 前未使用 test/new holdout；本轮已在冻结后完成一次研究评估，评估失败报告保留且不重复覆盖。**

## 4. 预定门槛与实际结果

| 门槛 | 预定值 | 本轮状态 |
| --- | --- | --- |
| 新 unsupported false acceptance | ≤0.01 | **0.0015625，PASS** |
| supported corrupted accepted fraction | ≥0.5 | **0.6277778，PASS** |
| known benchmark corrupted RMSE improvement | ≥0.25 | **0.0798104，FAIL** |
| known benchmark clean replacement fraction | ≤0.01 | **0.0055556，PASS** |
| known session RMSE ratio to noisy | ≤1.1；session 非独立 | **132.1703，FAIL** |
| all-noop | 不能通过 | **非 no-op（434 个 benchmark 窗口实际改变），PASS 此项但不抵消其它 FAIL** |
| 真 target transfer accuracy | mapping 未确认，不评估 | `not_assessed_target_mapping` |
| production validation | 不作通过主张 | **未通过/未批准，`production_validated=false`** |
| 自动处理 | 保持关闭 | **关闭，`automatic_enabled=false`** |

所有门槛是预注册约束，不得用本轮输出回调阈值、重选 subjects、改变分母或隐藏拒绝窗口。

## 5. 数据许可

- EEGdenoiseNet：数据许可尚未确认；不得声称已获商业用途、再分发或原始数据打包授权。
- Dong2023：按本轮审计记录为 **CC BY-NC 4.0**；不据此主张商业使用许可。
- “可下载”“可训练”或正式文档状态，不等于许可已审核完毕。发布时不附带未授权原始数据或受限制数据的派生再分发承诺。
- GPU readiness 与模型报告不提供许可合规、临床或生产授权结论。

## 6. GPU 依赖隔离

隔离环境：`C:\Users\Administrator\Documents\ChatGPT\脑机采集\.venv-ml-gpu`。

采集环境：`C:\Users\Administrator\Documents\ChatGPT\脑机采集\.venv`，**不升级、不移除、不改写**其中的 PyTorch、NumPy、SciPy、BrainFlow、PySide6 或采集依赖。

用户指定固定依赖写在：`C:\Users\Administrator\Documents\ChatGPT\脑机采集\requirements-ml-gpu.txt`。GPU preflight 记录实际解释器为 Python `3.14.5`，隔离环境路径与采集环境不同。

| 包 | 用户指定 pin | 已有 preflight 实测版本 |
| --- | --- | --- |
| PyTorch | `2.14.1+cu130`，官方 cu130 wheel index | `2.14.1+cu130` |
| NumPy | `2.5.3` | `2.5.3` |
| SciPy | `1.18.1` | `1.18.1` |

主线程负责在隔离环境安装。本检查脚本不安装包、不下载、不改驱动、不调用设备管理命令，也不以 `nvidia-smi` 为验证前提。driver 616.92 是用户先验提供的信息；脚本未探测 driver，报告中的 driver 仍为 `not_assessed_no_driver_probe`，不能把先验版本冒充测得值。

## 7. CUDA 环境检查脚本与实测 micro-smoke

文件：`C:\Users\Administrator\Documents\ChatGPT\脑机采集\scripts\check_gpu_training_environment.py`。

脚本在当前解释器中检查，不修改采集配置；线程环境通过 OMP/MKL `setdefault(2)` 保留已有调用者设置，PyTorch CPU threads 显式设为 2。

流程：

1. 先独占创建指定新报告；若路径已存在，拒绝，并且不加载 PyTorch、不重复执行 GPU 操作。
2. 记录 Python executable/prefix/version、包版本、PyTorch CUDA build、CUDA available、设备 count/name/capability/total VRAM。
3. 只在 CUDA 可用时执行 disposable 小 Conv1d 的 forward、backward 和一次 SGD step，输入为 synthetic `[4,1,500]`，不读取 EEG、v2/v3 checkpoint、train/validation/test/holdout。
4. 输入、预测、loss、每个 gradient、更新后的 parameters 都须有限且位于所选 CUDA device；参数须实际变化，不能以 optimizer no-op 假称通过。
5. 同步 CUDA 后记录耗时、峰值 allocated/reserved bytes。峰值是本进程 PyTorch allocator 在 smoke 区间的记录，不是整卡系统占用、可训练 batch 上限或实际 v3 training 显存结论。
6. 失败仍写 JSON，包括失败 stage/type/message。未执行项目留 null / `not_assessed`；driver 不探测。

返回码：

- `0`：synthetic CUDA forward/backward/optimizer smoke 通过，**只说明该微型操作可执行**。
- `1`：PyTorch 无法导入、CUDA 不可用、设备信息失败，或 smoke 任一步不满足有限值/设备/参数更新条件。无论是否提供 `--require-cuda`，均不自动回退 CPU。
- `2`：报告无法独占创建、报告已存在或输出写入错误；CLI 参数错误同样非零。

`--require-cuda` 记录 CUDA 是主线程的显式先决条件；不是允许缺 GPU 时改用 CPU 的开关。

### 已有主线程实测证据（未重跑）

报告：`C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\learned_denoiser_v3_gpu_preflight.json`。

报告时间：2026-10-05 08:09:11.980262 UTC（2026-10-05 16:09:11.980262 Asia/Shanghai）。

| 字段 | 已有报告实际值 |
| --- | --- |
| check / smoke status | `passed` / `passed` |
| device / name | `cuda:0` / NVIDIA GeForce RTX 4090 |
| CUDA build / capability | `13.0` / `[8,9]` |
| CUDA available / device count | true / 1 |
| total VRAM | 24,146,083,840 bytes；不是 smoke peak |
| forward / gradients / parameters finite | true / true / true |
| parameters_updated | true |
| optimizer step | disposable SGD，1 step，非真实数据训练 |
| synchronized smoke elapsed | 3.6544200999996974 秒 |
| peak allocated / reserved | 135,680 / 2,097,152 bytes |
| overall elapsed / exit code | 5.059237699999358 秒 / 0 |
| dataset_training_performed / holdout_accessed | false / false |
| v3_training_status / v3_model_quality_status | `not_assessed` / `not_assessed` |
| production / automatic processing | false / false |

该记录证明真实 CUDA 小规模 finite forward/backward/参数更新通过，**NOT 数据训练、NOT v3 模型验收**。

### 执行源码绑定

在上述报告完成后，本 sidecar 保持检查脚本不变；该脚本当前 SHA-256 为：

`a02dd3544a336680701b46e87ec84717510ff07143cd3d26acec652084a26f50`

此值是报告完成后的执行源码留存标识，不是在报告 JSON 中追写的“执行时自动测得 hash”。既有报告与脚本未被改写。未来如改源码，应保存运行时源码快照/hash 并使用新日志、新报告路径；不得让旧运行记录冒充修改后源码的结果，也不得改写主线程另行冻结的训练源码。本 GPU 检查未读取或训练任何已冻结模型。

### 主线程实际 v3 训练与研究评估（2026-10-05）

训练报告：`C:\Users\Administrator\Documents\ChatGPT\脑机采集\models\domain_gate_v3.training.json`。使用隔离 `.venv-ml-gpu` 的 `torch 2.14.1+cu130`，实际 `cuda:0 / NVIDIA GeForce RTX 4090`，固定 20 epochs，最佳 epoch 7，checkpoint 为 `models/domain_gate_v3.pt`。训练未访问 known v2 test 行或 S4/S5 holdout；压缩 NPZ 的物理解压语义已在报告中单独说明。

研究评估报告：`C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\learned_denoiser_v3_domain_gate_evaluation.json`。S4/S5 负集安全拒绝达到 0.0015625 false acceptance（S4 0.003125、S5 0）；但 known v2 corrupted improvement 仅 0.0798104（门槛 0.25），历史 known-session gated/noisy RMSE ratio 为 132.1703（门槛 1.1），因此综合研究门**失败**。这不是重新调阈值或重选被试的理由；失败结果保持冻结并保留。

该结果明确表示：v3 可作为**显式离线、保守拒绝型诊断工具**继续开发，但当前不允许自动替换、工作站接入、生产质量声明或临床有效性声明。

### 主线程后续显式检查命令（本 sidecar 不执行、不重跑）

当前已有 micro-smoke 成功报告，无须为了本交付再次执行。如果未来主线程明确要求复检，使用**从未存在的新输出路径**，不要使用上面的既有 preflight 文件：

```powershell
& 'C:\Users\Administrator\Documents\ChatGPT\脑机采集\.venv-ml-gpu\Scripts\python.exe' -B `
  'C:\Users\Administrator\Documents\ChatGPT\脑机采集\scripts\check_gpu_training_environment.py' `
  --require-cuda `
  --output 'C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\gpu_training_environment_v3_recheck1.json'
if ($LASTEXITCODE -ne 0) { throw 'CUDA readiness check failed; do not silently fall back to CPU.' }
```

若 `recheck1` 已存在，换新名字；不删除旧报告或覆盖其结论。micro-smoke 通过也不能填入“v3 GPU 训练完成”“有效去噪”“holdout 通过”。这些须等主线程提供实际训练/评估报告再做定点更新。

### 无真实 GPU 的 mock 单元测试命令

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'C:\Users\Administrator\Documents\ChatGPT\脑机采集\.venv\Scripts\python.exe' -B -m unittest discover `
  -s 'C:\Users\Administrator\Documents\ChatGPT\脑机采集\tests' `
  -p 'test_gpu_training_environment.py' -v
```

这些测试模拟 CUDA API 与 tensor 操作，不会执行真实 GPU smoke、训练或 holdout；不能用 mock 的 passed 替代真实 CUDA 运行证据。

## 8. 硬件 / raw / 历史版本保护

- 保留硬件握手、通道完整性、设备/显示时序预检、光敏风险确认、中止控制与质量评估。
- 不修改 raw TSV、MAT、NPZ、帧时序、事件、原始 metadata、历史 session 验证结论或 v1/v2 旧报告。
- 不覆盖冻结的 v2 权重、training metadata、config、数据及 manifest；protected SHA-256 以预注册文件为准。
- 新 GPU 环境、脚本、文档与门控模型不接入自动处理。采集进程不依赖隔离 GPU 环境；失败不能影响硬件采集主路径。
- 报告路径必须显式给出且为新文件；设备可枚举不代表 CUDA kernel、backward、optimizer 或真实训练已验证。

## 9. 已完成与待主线程补入的证据

| 证据 | 结果 | 报告路径 |
| --- | --- | --- |
| 隔离环境实际包版本 | 与指定 pins 一致；环境 micro-smoke `passed` | `C:\Users\Administrator\Documents\ChatGPT\脑机采集\analysis_outputs\learned_denoiser_v3_gpu_preflight.json` |
| 真实 CUDA micro-smoke | `passed`，仅 synthetic step | 同上 |
| 真实 v3 GPU 训练完成 / 最优 checkpoint | `passed`；best epoch 7/20 | `C:/Users/Administrator/Documents/ChatGPT/脑机采集/models/domain_gate_v3.pt`；训练报告 `models/domain_gate_v3.training.json` |
| validation gate threshold 与 acceptance | `0.9702852368354797`；unsupported false acceptance 0.00996094；supported acceptance 0.7772222 | `models/domain_gate_v3.training.json` |
| 新 unsupported S4/S5 holdout | **安全门通过：0.0015625 false acceptance；去噪有效性不评估** | `analysis_outputs/learned_denoiser_v3_domain_gate_evaluation.json` |
| known v2 benchmark / known session regression | **研究门失败：0.0798104 improvement；known-session ratio 132.1703** | `analysis_outputs/learned_denoiser_v3_domain_gate_evaluation.json` |
| 冻结文件与 raw 完整性复核 | **freeze verified；raw exact fallback 保持；v2 protected SHA 未改** | `analysis_outputs/learned_denoiser_v3_experiment_lock.json`、`analysis_outputs/learned_denoiser_v3_domain_gate_evaluation.json` |
| production / 临床 / 自动处理批准 | **未批准；`production_validated=false`、`automatic_processing_enabled=false`** | v3 仍仅显式离线诊断 |
