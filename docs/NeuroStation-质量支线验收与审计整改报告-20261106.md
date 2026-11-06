# NeuroStation 质量支线验收与审计整改报告

日期：**2026-11-06（Asia/Shanghai）**
分支：`codex/neurostation-quality-upgrade`；未推送、未合并。
结论：**本轮质量工作区的软件交付闭环与审计修复，在已执行范围内通过；全量历史测试未通过，硬件和现场验收未完成。建议继续独立试用，不作无条件合并。**

## 1. 本次验收对象与边界

本次是对 11 月 4 日交付支线的复查和整改，不改写既有报告、原始记录或历史模型验证结论。不是人员/工期计划，也不是新的个人训练。正式配置状态不代表信号、硬件或临床有效性自动合格。

- 审计入口提交：`4ee934d9ed5d6647f5c07d43aa9bb1d24d68117f`。
- 第一项修复提交：`ef3d1f28ab9b83810ebc53e10412b07f047af6c6`，封存证据、事务式报告、异常质量拒判及 UI 状态清理。
- 最终源码提交：`6fe44b7cb41d58db5c0274333c6537c02572d32b`，补齐保存的回放结论与源文件哈希身份绑定。
- 三组最终测试及交付包使用同一源码身份：`2027eb7eb457633eb48dd31004030dab3dbdda9aad083db4c8587a207ff3c5e6`，覆盖 **392 个源码/配置文件**。
- 运行环境使用已登记的 `F:\脑机采集\NeuroStation\.venv\Scripts\python.exe`，不在支线伪造 `.venv`，不安装或改动原虚拟环境。
- 自动运行不连接真实采集硬件、不启动物理刺激、不训练或自动启用实验模型、不执行外部设备动作。
- 既有整体实现见 [11 月 4 日工程实现报告](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/docs/NeuroStation-质量支线实施与验证报告-20261104.md) 与 [11 月 4 日交付记录](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/docs/NeuroStation-质量工作区交付验收报告-20261104.md)；本报告只对本次实际复查和新增证据负责。

## 2. 审计发现与已完成整改

以下严重程度是本次工程审计的优先级，不是外部安全认证。问题均落实到代码、回归用例和最终包；没有通过取消校验、伪造训练锁或隐藏失败解决。

| 编号 / 级别 | 复查发现的风险 | 本次改动 | 验收证据 |
|---|---|---|---|
| QA-01 / P1 | 回放只比部分数值字段；未比较字段或处理清单被改动后，仍可能显示在线一致；保存的结论也缺少完整来源绑定 | 核验封存质量日志、处理清单、原始哈希链和水位；核对实际消费字节与前后来源集合；区分数值一致、封存证据核验和最终在线一致；保存 source_directory / source_hashes / source_identity | 篡改、缺 seal、非法行、回放中变化、读入前块变化、来源绑定回归通过；最终包读取旧版合成记录核验通过 |
| QA-02 / P1 | 报告或导出第二个文件写入失败、取消、源文件变化时，可能留下半成品或混入不同时间的内容 | 只读来源快照；核对消费字节及新增/移除文件；在私有同级临时目录完整写入并刷新后才发布；导出复制已验证的同一批字节；失败不发布最终目录，旧目标不覆盖 | 第二文件故障、取消、源文件改变、新文件出现、并发发布、异常退出测试通过 |
| QA-03 / P1 | 非法维度、索引、时间或重叠质量更新可能抛异常、被强制转型或留下旧预测仍可接受 | 质量门控严格验证结构和身份；拒绝小数/字符串/布尔索引、非有限/异常时间；非法更新使待返回代次失效；SELECT 锁不因拒判重新布防 | 结构破坏、重叠更新、时钟异常和晚到模型结果集成回归通过 |
| QA-04 / P2 | UI 收到损坏或缺失质量记录时可能崩溃或继续显示上一条正常状态；未来时间可能被当成新鲜数据 | 渲染前校验和深拷贝；无效输入清为 UNKNOWN；未来/非有限时间不新鲜；频谱空值安全显示 | 新增 4 项质量载荷 UI 测试；首条异常、缺失输入、时间与可变引用覆盖 |
| QA-05 / P2 | 报告重建失败、导出重试失败后仍保留旧预览或旧导出地址，容易误认本次操作成功 | 新报告失败清除当前旧视图和导出入口，但不删旧文件；导出失败清理旧导出地址并保留报告供重试；回放界面增加“封存证据核验” | 2 项报告/导出 UI 回归；可见 Windows 窗口验证 |

### 对应代码

- 回放：[replay.py](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/eeg_tools/quality/replay.py)。
- 来源快照和完整发布：[evidence.py](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/eeg_tools/quality/evidence.py)、[reporting.py](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/eeg_tools/quality/reporting.py)、[preview.py](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/eeg_tools/quality/preview.py)。
- 指令拒判：[commands.py](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/eeg_tools/quality/commands.py)。
- 可见界面：[quality_panel.py](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/apps/workstation_ui/quality_panel.py)、[quality_workspace.py](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/apps/workstation_ui/quality_workspace.py)。
- 新增测试：[核心审计回归](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/tests/test_quality_audit.py)、[质量载荷 UI 回归](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/apps/workstation_ui/tests/test_quality_payload.py)；另在现有指令 UI 和工作区 UI 测试中追加用例。

### 修复前的复现记录

- 首批 17 个审计测试方法产生 20 个失败事件、6 个错误事件，其中含子测试；至少 1 个方法修复前已通过。不是“26 个不同测试全部失败”。
- 报告重建失败后保留旧预览的问题单独复现失败。
- 首批 4 个质量载荷 UI 方法产生 5 个失败事件、6 个错误事件，含子测试。
- 并发、异常退出、最终来源绑定等测试在随后补齐；不声称这些后加测试全部做过修复前运行。
- 本轮合计新增 **29 个测试方法**：根目录测试增加 23（审计模块 22 + 指令集成 1），UI 增加 6。完整原始记录见第 9 节。

## 3. 最终自动验收结果

全部使用源码提交 `6fe44b7`。最终有效结果是 `final-v2-*`，不混用中间构建的计数。

| 检查 | 结果 | 结论 / 限制 |
|---|---:|---|
| 质量、采集、指令等核心专项 | **335 / 335**，0 失败、0 错误、0 跳过 | 通过；原基线为 312 / 312 |
| 工作站界面测试 | **129 / 129**，0 失败、0 错误、0 跳过 | 通过；离屏软件测试；原基线为 123 / 123 |
| 正式配置检查 | ready；formal；warnings 为空 | 通过；未削弱严格通道校验 |
| Python 语法及运行中源码一致性 | 三组检查均通过 | 不生成原环境字节码；源码清单一致 |
| Windows 构建后入口检查 | **6 / 6** | 包含质量工作流入口；不连接硬件 |
| 打包后的完整质量工作流 | **16 / 16** | 演示→回放→报告→导出等检查通过 |
| 最终 ZIP 独立解压再运行 | **6 / 6** 入口、**16 / 16** 工作流 | 本机清理 PATH 后验证；不是全新电脑验证 |
| 全量根目录历史测试 | **未通过** | 下列失败、错误和跳过均保留 |

### 全量历史测试：不能算通过

- 发现 **1897** 个测试入口，实际启动 **1889** 个；记录 **1735 次成功回调**。
- **1 个失败事件、135 个错误事件、25 项跳过**；另 **8 个测试方法**因类初始化失败未启动。
- unittest 错误包含子测试和类初始化事件；这些计数不是互斥的测试方法分类，**不能直接相加计算通过率**。UI 发现范围也单独列出，不与全量根目录测试混称。
- 135 个错误事件分为 129 个 `FileNotFoundError`、5 个 `ValueError`、1 个 `V6Error`，涉及未带入支线的历史预注册、训练/冻结锁、分析报告及外部数据/证据。
- 1 个断言失败为 `test_evaluate_ssvep_denoiser_v14.EvaluatorContractTests.test_default_runtime_loader_rejects_failed_v14`：期望出现 `ineligible`，实际在更早阶段拒绝为 `v14: training lock is required`。
- 25 项跳过包括本目录 `.venv` / pwsh 条件、可选 Torch/CUDA/数据条件、系统符号链接权限。未把跳过计为通过，也未启用需要额外权限的系统设置。
- 与本次入口基线逐 ID/事件比较：**没有新增失败或错误事件**；跳过 ID 与未启动 ID 不变；新增 23 个根目录成功方法。没有消除上述历史阻塞。
- “没有新增失败”不证明被缺资源阻断的路径正确。没有放宽模型锁、替换历史结论或把相关测试改成跳过以制造全绿。

## 4. 最终包可见操作及旧记录兼容

### 本次实际可见 Windows 操作

1. 在第一版整改包 `ef3d1f2` 上执行了完整 30 秒合成演示、回放、报告生成、原生 Windows 文件夹对话框导出。JSON/HTML 副本均与源报告一致。
2. 在最终 `6fe44b7` 的 **ZIP 独立解压目录**再次可见启动，选择 2026-11-04 生成的合成会话并完成回放。不是个人记录，也不是实时 EEG。
3. 最终界面显示：**150 块 / 7500 点；在线一致=是；封存证据核验=是；原始哈希未变=是；完整封存=是**。
4. 最终保存的 replay.json 绑定 **157 个 raw_blocks 层来源文件**；重新计算全部哈希和来源身份一致。会话父目录共 158 文件（额外包含 session.json），两种范围不混淆。
5. 两个可见检查所用合成会话的各 158 文件，内容和文件集合均未变。测试窗口已正常关闭。

最终包的报告和导出完整链路由 16 项打包后自动工作流再次验证；最终可见检查没有重复声称全部原生导出操作都在 `6fe44b7` 上手动重做。

合成报告的 150 条质量更新中，通过 0、异常 18、未知 132；试次/指令为零，零分母不显示虚构准确率。这是**更新条目计数**，不是独立试次准确率或时长占比。没有硬件阈值依据时保持 UNKNOWN 是预期行为，不是演示失败。

可视证据：[最终包回放截图](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261106-01/visible-final-replay.png)；[最终可见检查与来源复核](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261106-01/visible-final-verification.json)。

### 兼容及语义变化

- 未删点、未补零、未改写旧原始记录和历史结论；重新分析留在独立派生目录。
- 原有模型前处理、权重、冻结绑定源码和实验禁用标志保持不变；未把新的显示滤波替换为模型输入。
- 5 秒窗 / 1 秒步长、旧代次失效和 SELECT 一次触发锁定仍保留；拒判和超时不重新布防。
- 旧会话缺少块日志时仍可读取和生成报告，不补造块；缺 seal 的会话即便数值一样也不会成为“已核验在线一致”。
- 已中止但封存完好的已知前缀可比较；采集完整性仍独立为否，不能升级成完整采集通过。
- 失败/取消不发布半份报告；突然终止进程可能留下明显的私有 `.tmp` 目录，它不是交付报告。未声称所有文件系统断电均有同等持久性。
- 本地哈希封存不是外部数字签名；逐次读前/读后检查也不是整个文件系统的原子快照。应对静止、不被同时编辑的会话生成报告。
- HTML 最多展示 500 条来源哈希/试次，完整条目保存在 JSON，不用界面截断丢弃统计。

## 5. 性能复核：仅软件合成范围

最终测试为 8 通道、250 Hz、13 点分块，模拟 **90 秒**输入，共 **1731 块 / 22500 点**；加速墙钟运行 **43.17 秒**。使用有界异步质量队列，不是 90 秒真实设备稳态运行。

| 指标 | 实测 P95 | 对照目标 | 当前可下结论 |
|---|---:|---:|---|
| 数据块到离屏波形绘制 | **30.23 ms** | ≤250 ms | 离屏路径通过，不是物理显示延迟 |
| 记录及质量就绪 | **21.95 ms** | ≤1000 ms | 软件就绪路径通过，不等于现场屏幕质量展示延迟 |
| 质量计算本身 | **3.31 ms** | <100 ms（工程检查） | 本机合成负载通过 |

队列完成 **1731 / 1731**、丢弃 0、最终积压 0；记录回放数值一致、封存证据通过、原始哈希不变。首次积累单列：幅度 **2 秒**、PSD **8 秒**、漂移 **10 秒**、趋势 **30 秒**，不把这些积累时间省略或当作计算耗时。

**真实 30 分钟负载、2 小时长稳、物理显示与刺激时序均未运行/未测量，不计通过。**

## 6. 对原 14 项计划的本次复查落点

| 原计划 | 本次复查与改进 | 未被本次证据替代的验收 |
|---|---|---|
| NSX-001 基线/依赖 | 明确源码/包身份、环境登记与原工作区复核 | 全新主机、缺失历史研究依赖 |
| NSX-002 统计 | 统计专项回归和可见报告保持分母/标签区分 | 真实带标签会话的有效性 |
| NSX-003 / 004 共用采集/保存 | 已有采集保存回归继续通过；本次加强派生结果完整发布 | 真设备断连、真实慢盘/长稳 |
| NSX-005 事件关联 | 保留未知语义，不生成虚构刺激/样本对应 | 物理 marker、光电与设备时间源实测 |
| NSX-006—009 滤波/伪迹/质量/缓存 | 已有专项继续回归；处理实现及身份未被本次 UI 修复替换 | 真硬件阈值与任务保真验证 |
| NSX-010 界面 | 修复损坏载荷、未来时间、旧状态残留；新增封存证据核验 | 多屏/高 DPI 完整矩阵及现场刷新延迟 |
| NSX-011 报告 | 来源、内容、文件集合、取消和并发发布加强 | 大型真实会话长期数据规模测试 |
| NSX-012 指令 | 非法质量拒判、旧结果失效、SELECT 锁回归 | 真实个人模型及外部动作（本轮不启用） |
| NSX-013 回放 | 内容/清单/封存与保存结论均绑定，旧合成会话实测 | 真实场景回放集 |
| NSX-014 故障/构建 | 故障回归、最终 ZIP 解压/运行及文件核对通过 | 全量历史套件、干净 Windows 主机与现场验收 |

上述表是复查范围，不是“14 项已经全部实机验收完成”的声明。

## 7. Windows 交付物及使用

- ZIP：[NeuroStation-QualityWorkspace-20261106-6fe44b7-Windows-x64.zip](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/delivery-audit-20261106-final/NeuroStation-QualityWorkspace-20261106-6fe44b7-Windows-x64.zip)。
- 大小：**131111482 字节**；文件数：**1050**。
- ZIP SHA-256：`b73600bc70a22df49f9e2a9e984e59b1b42188ee23e949747d6dd774db70339e`。
- [交付文件哈希清单](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/delivery-audit-20261106-final/delivery-manifest.json) 与包内 BUILD-IDENTITY.json 记录源码身份；最终可见运行后再次确认解压文件集合及字节仍全匹配。
- 启动说明：[审计修复版试用说明](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/docs/NeuroStation-质量工作区试用说明-20261106.md)。

完整解压到新目录，不覆盖当前可用版本；双击包内 **Start-QualityWorkspace.cmd** 直达“信号与质量”。启动不自动运行演示或连接硬件。主工作站和实时指令窗口仍有各自启动器；实验模型默认禁用，包不带 Torch 和个人权重。

本次本机包验证将 PATH 限制为 Windows/System32 与 Windows。该结果不能替代全新 Windows 主机、杀毒环境、不同显卡/DPI 或全部运行配置检查。产品版本仍为 1.04，**用日期、源码提交及 ZIP 哈希区分构建**。

## 8. 尚需改进与合并建议

### 下一优先级：历史测试与依赖可移植性

1. 将可合成的单元/契约测试改为显式临时夹具，保持有效/无效锁和篡改路径的实际语义；不能用“只要报任意错就算通过”替代约束。
2. 把确需历史训练产物的集成检查与核心软件测试分组，但原失败记录继续公开；分类不等于修复。
3. 对必需外部证据建立显式只读路径映射与内容清单，在缺失时明确列出所需文件和哈希；不复制整批个人数据、不伪造旧锁、不重写历史身份。
4. 针对 v14 的当前失败，先恢复可验证测试前提，使 `ineligible` 分支真的被执行，再判断契约是否需要修改。未在本轮擅自放宽生产加载器。

### 现场验收清单（未执行）

- 干净 Windows 主机独立解压、启动、权限与不同 DPI / 双屏。
- 操作员明确发起的设备握手、通道完整性、设备断开/重启、保存失败和人工中止。
- 幅值等硬件阈值来源与标定；未知不得自动映射为正常。
- marker、显示/光电与采样时序关联，保留真实时间来源和误差。
- 真实采集下的波形/质量显示 P95、30 分钟负载和 2 小时长稳。
- 带合法标签的真实会话回放与统计核对；不因演示能运行宣告模型准确率或临床有效性。

**建议：先试用这份独立包。** 合并前保留本次已知历史阻塞清单；硬件相关能力未完成现场验收前，不以“全部验收通过”发布或启用。最终是否合并由用户决定，本轮没有推送或合并。

## 9. 证据索引与原工作区保护

| 证据 | 位置 |
|---|---|
| 核心专项最终结果 | [focused results](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261106-01/final-v2-focused/results.json) |
| UI 最终结果 | [UI results](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261106-01/final-v2-ui/results.json) |
| 全量错误/跳过与日志 | [full results](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261106-01/final-v2-full/results.json) / [full tests.log](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261106-01/final-v2-full/tests.log) |
| 与入口基线逐项对比 | [full-final-comparison.json](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261106-01/full-final-comparison.json) |
| 修复前复现 | [核心审计](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261106-01/regressions-before.log) / [旧报告状态](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261106-01/ui-regression-before.log) / [UI 损坏载荷](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261106-01/payload-before.log) |
| 性能与封存回放 | [performance.json](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261106-01/performance-final/performance.json) |
| 最终 ZIP 启动验证 | [build-verification.json](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261106-01/zip-final-verification/build-verification.json) |
| 最终 ZIP 16 项工作流 | [workflow-verification.json](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261106-01/zip-final-verification/quality-workflow/workflow-verification.json) |
| 最终包可见复核 | [visible-final-verification.json](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261106-01/visible-final-verification.json) |
| 包哈希再次核验 | [package-final-recheck.json](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261106-01/package-final-recheck.json) |
| 原工作区与绑定保护 | [preservation-final-verified.json](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261106-01/preservation-final-verified.json) |
| 机器可读验收摘要及证据哈希 | [quality-upgrade-audit-20261106.json](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/docs/quality-upgrade-audit-20261106.json) |

原工作区 `F:\脑机采集\NeuroStation` 的 HEAD、tracked 修改状态、暂存区摘要、登记的 **302 个文件**及 **12 项依赖**（2 个 Python 可执行文件、10 项模型产物）核对一致；4 份冻结模型绑定源码没有本轮差异。保护核验只覆盖登记范围，**不声称扫描过所有个人记录**。

最终保护复核曾因 Git 对中文路径是否加引号的文本表示差异误报状态不一致；随后用与入口相同的 `core.quotepath=false` 只读命令复核通过，未修改 Git 设置。原尝试记录保留，最终依据为上述 `preservation-final-verified.json`，不能把展示格式差异误写成真实文件变更。

测试和证据均留在支线独立输出目录；报告/摘要另附于交付 ZIP 的同级目录，**没有修改已经核验哈希的 ZIP**。此前的历史临时副本和旧报告未自行清理；未改写其结论。
