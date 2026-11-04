# NeuroStation：SSVEP 数据接入与 v5 前置审核

## 本轮交付与边界

日期：2026-10-07。已交付正式的只读完整性工具、科学来源审核、被试分组锁及原生试次参考数据。**没有训练或发布新的 v5 去噪模型，也没有开启工作站自动替换。** `formal` 仅表示交付/配置状态，不代表硬件、模型、会话质量或临床验收通过。

来源审核沿用 gpt-6-luna；新增 scanner、reference loader、CLI 与合成测试由沿用的 gpt-6.1-sol 编程代理完成。主线程完成复核、真实扫描、冻结、构建和回归测试。

本工具链没有向 D:\SSVEP 写入、解压、改名或执行其中脚本；没有停止用户下载进程。原始 EEG 不上传第三方。下载进程仍可能改变 D 盘，因此不能把一个时间点快照解释为此后全部下载完成。

## D:\SSVEP 的时间点完整性结果

扫描区间：2026-10-07 10:08:52–10:10:00（Asia/Shanghai）。快照含 9 个数据集目录、217 个非 Git 文件，总计 25,075,925,250 字节（25.08 GB，含仍在写入的文件及根目录元数据）。源码、MAT、metadata 和压缩包不能重复当作独立受试者。

| 目录 | 本地核验情况 | 科学用途与尚未通过的门 |
|---|---|---|
| Nakanishi2015 | 10 个真 MAT；`eeg` 头为 [12,8,1114,15]。1–8 已指定变量读取、有限性验证并物化；9–10 只冻结身份/头信息 | 256 Hz、12 类任务参考。不能使用文件中的额外 `train` 变量充当真值。原发布 EEG 已预处理，不能称为无噪硬件原始信号 |
| Wearable | 10 个 ZIP 的全部 member CRC 通过，102 个 subject MAT 成员；另有 S001–S010 已展开。当前物化 1–8，9–10 不读波形 | 250 Hz、8 通道、12 类、dry/wet 域差异与任务保真；102 人并未全部导入。Impedance 是元数据，不是 EEG；EEG 与 Impedance 的电极轴顺序相反 |
| Wang2016_Benchmark | 35 个文件实际为 MAT v5，`data` 头 [64,1500,40,6]；不是仅凭扩展名推断。未做全体有限性加载 | 250 Hz、40 类识别/保真候选；需专用 64 通道读取器、官方频率/相位/电极映射及独立分组 |
| BETA | 7 个 tar.gz 通过 gzip EOF CRC/ISIZE，共 70 个 MAT 成员；未解压、未解码数值 EEG | 候选；压缩完整不等于 EEG 有限或标签映射完整。频相映射/历史修正版本及许可仍有待核验 |
| eldBETA | Description 及前 6 个 subject 包通过检查；S61–S70 在扫描期间变化，S71–S80 为结束时新出现文件 | 正在下载的数据不纳入。连续 BIDS 和分段 MAT 不是可混用的同一种输入合同；映射/导入仍待完成 |
| OpenBMI_Lee2019 | 109 个文件主要为目录/下载元数据；仅 2 个 EEG MAT，顶层为 train/test struct，不是已下载全体 54 人 | 1000 Hz 候选；需要事件/结构体适配、采样与单位核验，同一被试的不同会话不能跨训练/测试组 |
| Targets160 | 5 个 RAR 识别成功但本机没有可用 7-Zip，完整性 `not_verified`；reqCodeword 是设计表，不是信号 | 暂不导入。160 个标签采用四符号顺序编码，不能套用 160 个独立单频目标；符号→Hz/onset 尚未确认 |
| MAMEM | 空目录 | 本地无可训练信号 |
| Choi2019 | 空目录 | 本地无可训练信号 |

SHA256 证明本轮文件身份，不是与官方全体文件校验和匹配的证明。ZIP CRC 或 gzip CRC/ISIZE 只证明该容器校验通过，不能证明文件就是 EEG、标签正确、波形无噪或许可已获得。

## 原始参考与监督去噪真值的区别

本轮检索的一手来源**没有建立这些本地 SSVEP 数据的真实同一 epoch、同一电极 clean/noisy 配对真值**。以下都不能自动叫作 clean target：未修改的实测 EEG、wet 电极、跨试次平均模板、滤波结果或另一时刻的 dry/wet 记录。

这些数据可以作为刺激标签参考、任务识别训练候选、降噪后任务保真检查和有限的域适配材料；并不单独证明伪迹检测或降噪准确性。所有 9 类的存储幅值单位在本轮审核中均未获得充分证据，保持 unknown，不按幅度猜成微伏或伏。

许可与科研元数据是不同的门。OpenBMI/Choi 的数据 README、MAMEM 官方页有明确数据许可记录；Nakanishi、Wearable、BETA、eldBETA、Targets160 等仍有 unknown。Wang 镜像许可声明不直接证明本地版本许可一致。当前只做用户本地技术准备，不承诺商业复用权或公开发布权。

## 实际冻结与构建

仅选 Nakanishi2015 和已展开的 Wearable S001–S010。模型指标读取前预先指定：两个数据集各 ID 1–6 = train、7–8 = validation、9–10 = holdout。分组键含 dataset namespace，只保证各数据集内部 subject-disjoint，不证明两个数据集之间不存在同一参与者，也不宣称此小样本分组具有代表性。

| 数据集 | train | validation | holdout（未物化） |
|---|---:|---:|---:|
| Nakanishi2015 | 1,080 | 360 | 360 |
| Wearable | 1,440 | 480 | 480 |
| 合计 | 2,520 | 840 | 840 |

已物化 16 位被试、3,360 个参考试次，占 190,292,032 字节（包括 sidecar/manifest）。Nakanishi 的每被试 reference 为 [180,1114,8]，Wearable 为 [240,710,8]，均 float64 [trial,time,channel]；只有 `reference`，没有 clean 数值标签。

保留原文件全部时间点、原生 256/250 Hz、通道顺序和数值。**导入阶段**没有新增滤波、重采样、归一化、插值或裁剪；这不否认源数据此前存在的处理。未知单位、Nakanishi 未完整确定的 channel order/phase 均为 null；不能把外部后枕 8 通道直接当作本机 Cyton 的同一电极配置。

freeze 只读取 audit/evidence、路径和 stat；build 仅读取 train/validation 指定变量。新 holdout 的当前文件字节、数值波形和模型指标未被 build 读取。此前独立完整性扫描曾读取它们的文件字节/头以核验身份，但没有解码 holdout 波形或计算模型指标；不要把此门扩大表述为“任何形式都没读过文件”。

物化结果已经独立核验：16 个 NPZ/sidecar SHA、reference 数组 SHA、dtype、shape、finite、标签频率、电极枚举、native 采样率，以及没有 holdout NPZ。输出拒绝覆盖；源/evidence 变化拒绝；工作区以外、D/source 内、symlink/junction 或路径越界输出拒绝。保守 256 MiB 为工作数据分配预算，不是 Python/SciPy OS 硬 RSS 上限。

## 工具和产物

- `scripts/audit_local_ssvep_v5.py`：magic-first、流式 SHA、下载变化检测、ZIP CRC、gzip EOF CRC/ISIZE；不解压，不安装 reader 或 7-Zip。
- `scripts/prepare_ssvep_reference_v5.py`：`freeze` / `build` 两阶段；不是 trainer 或 inference/deployment 入口。
- 完整性快照：`analysis_outputs/ssvep_local_v5_full_20261007_1008.json` 与同名 Markdown。
- 科学来源：`datasets/external/v5_ssvep_local_audit/SSVEP_SOURCE_AUDIT.json/.md` 及 evidence/FILE_SHA256_INVENTORY/BOUNDARY_VALIDATION。
- 分组锁：`analysis_outputs/ssvep_reference_v5_partition_lock_20261007.json`。
- 原生参考：`datasets/processed/ssvep_reference_v5_20261007/manifest.json` 及两个数据集下的 16 个 NPZ。
- 实际构建/集成验证日志：`analysis_outputs/ssvep_reference_v5_build_20261007.log`、`analysis_outputs/ssvep_reference_v5_integration_verification_20261007.log`。

## 回归验证与下一步

- scanner 合成测试 27 项：26 通过、1 跳过（宿主不允许真实 symlink；reparse/junction mock 另测通过）。
- reference 合成测试 21 项全部通过。
- 全核心套件 499 项：497 通过、2 跳过；界面测试 65 项全部通过。
- 正式配置校验通过；UI smoke exit 0。没有改发布配置、硬件握手、时序预检、光敏风险、中止控制或质量评估。
- v1–v4 及所有旧工作区保护文件的最终 SHA 核验另存 `analysis_outputs/ssvep_v5_final_verification_20261007.json`；v4 失败门、阈值和禁用状态不修改。

下一轮先注册 v5 实验合同及全试次任务保真指标，再建立有真实/合成配对定义的伪迹监督数据与独立监督测试来源。仅 train 分组可进入参数学习；validation 可用于事先声明的选择，holdout 不用于损失/阈值设计。先确定跨采样率策略和未知单位/电极限制，不能把 native 256 Hz 直接喂给旧 250 Hz 模型。

GPU 用于随后真正的训练，沿用独立 CUDA 环境与 RTX 4090；本轮主要是磁盘读取与元数据整理，不需要 GPU。v4 已失败的测试不是下一轮的新 holdout。本轮新增 SSVEP holdout 尚未读取模型指标，但不能替代独立配对去噪真值测试。完成研究门之前仍不启用工作站自动信号替换。
