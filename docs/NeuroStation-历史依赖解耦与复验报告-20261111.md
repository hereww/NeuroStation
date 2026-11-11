# NeuroStation 历史依赖解耦与复验报告

日期：2026年11月11日（Asia/Shanghai）
分支：`codex/neurostation-quality-upgrade`
本轮起点：`e4299a41d86c511e3217ec5d5079671ecc767f91`
已验代码：`6bd56272a78718d1d9486526ac470d6a547213b2`

## 1. 本轮结论

本轮落实11月6日验收报告中“历史测试与依赖可移植性”的下一步，不重新训练模型，也不扩展实时采集功能。完成了合成测试夹具解耦、显式只读历史证据登记，以及时间确认分析的来源绑定修复。

**核心专项364项、离屏界面129项全部通过。独立契约测试560项通过、1项CUDA跳过。全量测试2077项通过，仍有8个历史依赖错误、25项跳过，不能标为全量验收通过。**

本轮是源码和验收工具交付，不是新的Windows包交付。界面没有改版，因此预览外观不会因这些修改发生明显变化。没有推送、合并、启动真实采集或刺激、训练个人模型或启用实验模型；最终合并仍由用户决定。

## 2. 实际代码改动

### 2.1 修复分析报告误绑定来源

原时间确认分析的 `analyze(report)` 接收内存数据，却取默认历史报告文件的路径和哈希。这样既可能把输入归到无关文件，也会因默认文件缺失而使合法的自定义输入失败。

现在纯内存分析返回 `source_report=null`、`source_report_sha256=null`、`source_binding_status=unbound_in_memory`；文件入口 `run()` 使用一次读取的真实字节解析和计算哈希，状态为 `bound_consumed_file_bytes`。发布前复核输入未变化，拒绝覆盖已有报告，并在解析路径前拒绝符号链接及junction。

使用临时合成JSON执行旧版与新版对比，复现了旧版错误归因及自定义输入受默认文件影响的问题；新版均按预期处理。新增6个来源回归，连同原有3项共9项通过。没有改动预测规则、原有历史输出或模型禁用标记。

- 实现：[时间确认分析脚本](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/scripts/analyze_ssvep_temporal_confirmation.py)。
- 修复前后证据：[合成来源对比](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261111-01/temporal-source-before-after.json)。

### 2.2 将可合成契约与真实历史结果分开

新增6份明确标注 `synthetic_contract_only_not_historical_training_evidence` 的测试夹具，以及共用的临时夹具构造器。v3、v4、v5、v6、v7、v8、v13、v14相关测试不再为检查接口、锁或解析规则而隐式依赖原目录下的历史文件。

关键约束如下：

- v3/v4用临时合成信号、真实文件哈希和临时锁验证契约，不复制原训练身份。
- v5的短文本仅测试解析器，不是论文、设备或数据集来源证据。拒绝验证集/留出集受试者的测试使用临时manifest；只在该测试作用域绑定临时内容的真实哈希，仍执行内容校验。
- v6的checkpoint测试将根目录及preregistration哈希临时绑定到合成文件，结束后恢复；新增篡改仍被拒绝的回归。生产常量和历史锁未修改。
- v7/v8从生产代码已经内嵌的原preregistration literal生成临时文件，要求与原冻结SHA完全相同，不生成替代历史SHA。冷进程检查使用生产预检，并限制读取范围。
- v14使用临时目录中的未训练CPU随机初始化状态和明确的合成失败history，真实执行锁、序列化和生产加载器校验。原 `ineligible` 断言保留，并已真正到达该分支；新增缺锁、权重/报告篡改、配置及checkpoint前处理不匹配回归。没有将“任意报错”当成 `ineligible` 通过。

这些结果证明软件契约在上述合成条件下工作，不证明历史训练成功、模型有效或个人数据评估通过。既有测试中的mock仍按各测试原有边界使用，不能把整套单元测试描述成无mock集成验收。

- 夹具说明及构造器：[synthetic_research_fixtures.py](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/tests/synthetic_research_fixtures.py)。
- v14测试：[test_evaluate_ssvep_denoiser_v14.py](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/tests/test_evaluate_ssvep_denoiser_v14.py)。

### 2.3 新增只读历史证据核验

新增 `EvidenceBindings` 及命令行审计入口。输入必须是显式文件清单，包含原记录绝对路径、允许的根目录映射、预期SHA-256和身份依据。工具不递归搜索、不猜文件名、不复制数据、不修锁，也不作为模型加载的自动回退。

它区分三种身份依据：`frozen_contract`（已有冻结约定）、`local_inventory`（本次本地登记）、`unknown`（身份未知）。预期哈希为null时保持阻塞，不读取内容并自动补一个哈希；文件缺失、内容不符、路径别名或读取中变化均不算验证通过。输出不得覆盖既有文件、声明输入或写入映射后的证据根。

本地清单显式映射到原工作区，只读复核了19份元数据文件：4份沿用原冻结pin，15份是本地登记pin；最终19份字节匹配、0份阻塞。清单SHA-256为 `54a99c28c6b136440c174b9897e9faabb2e53f551a23cdb38a0743c0154ba1e3`。

**这只覆盖已列出的19份元数据，不是完整递归依赖图，更不是历史结论复验。** 工具始终输出 `historical_conclusions_verified=false`。8个剩余历史集成检查尚未接入这套迁移能力；“文件在原目录存在且字节匹配”不能替代运行这些检查。

- 实现：[evidence_bindings.py](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/eeg_tools/evidence_bindings.py)。
- 命令行：[audit_historical_test_evidence.py](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/scripts/audit_historical_test_evidence.py)。
- 本地显式清单：[历史证据清单](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261111-01/historical-test-evidence-manifest-final.json)。
- 最终只读复核：[19份元数据复核结果](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261111-01/historical-test-evidence-recheck.json)。

### 2.4 验收分组和来源身份

验收入口新增 `--mode contracts`，明确选取23个模块。`full`仍发现并执行全部核心测试，没有通过删除历史检查来获得全绿。

契约组明确排除且记录了1项目录存在检查：`test_evaluate_ssvep_denoiser_v14.EvaluatorContractTests.test_personal_sessions_are_fixed_and_present`。状态为 `not_run_in_contracts`，不计通过或跳过；full中仍执行，本机通过只说明指定目录存在，不说明目录中历史内容或模型结论有效。

源码清单补入测试夹具、`.spec`、`.bat`、`.cmd`，运行后核对新增、删除和内容变化。本轮清单覆盖406份纳入规则的文件；4组结果共享源码身份 `fb89981120f26656553b7bdb220febd69a24a12ba44e30016e4732e5b08de25d`。

结果schema升至2，保留原 `successful` 的unittest语义；新增 `fully_passed_selection`，有跳过、预期失败或未启动用例时为false。契约组即使退出码0，也因1项跳过不算所选范围全通过。

## 3. 最终实测结果

以下是最后一轮 `final-v2-*` 的结果，不用中间失败尝试替代最终结果，也不删除中间日志。

| 验收范围 | 发现 / 启动 | 通过 | 断言失败事件 | 错误事件 | 跳过 | 全部通过所选范围 | 用时 |
|---|---:|---:|---:|---:|---:|---|---:|
| 独立契约 contracts | 561 / 561 | 560 | 0 | 0 | 1 | 否 | 75.85秒 |
| 核心专项 focused | 364 / 364 | 364 | 0 | 0 | 0 | 是 | 23.65秒 |
| 离屏界面 ui | 129 / 129 | 129 | 0 | 0 | 0 | 是 | 13.55秒 |
| 全量核心 full | 2110 / 2110 | 2077 | 0 | 8 | 25 | 否 | 162.01秒 |

4组正式配置检查、Python语法检查和运行期间源码不变检查均通过；最终没有未启动用例。以上组之间存在重叠，不能把通过数相加当作唯一测试总数。离屏界面测试不等于真实屏幕、双屏或设备时序验收。

运行环境使用已登记的外部Python：`F:\脑机采集\NeuroStation\.venv\Scripts\python.exe`，版本3.14.5。实际加载版本：brainflow 5.22.2、numpy 2.5.3、scipy 1.18.1、PySide6 6.11.2、torch 2.14.1。本轮没有更换环境或安装依赖；这不是干净Windows主机验证。

### 与本轮入口对比

入口为1735项通过、1个断言失败事件、135个错误事件、25项跳过，另有8项因类初始化失败未启动。最终8个错误ID均为原有错误，没有新增错误ID；128条入口异常ID已不再出现，入口1735个通过ID全部保持通过，8个原未启动用例也均通过。

发现数从1897增加到2110，是模块导入恢复和新增测试共同作用，不能直接拿不同发现数算固定分母的通过率。错误统计还包含subtest和类初始化事件，不与方法数简单互斥相加。逐ID比较保存在 [full-final-comparison.json](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261111-01/full-final-comparison.json)。

## 4. 仍未通过及未运行项目

### 4.1 八个历史依赖错误

| 检查类别 | 错误数 | 当前直接阻塞 | 后续处理边界 |
|---|---:|---|---|
| 个人校准v1、个人decoder v2/v3历史报告检查 | 3 | 支线预期位置缺各自历史summary | 显式映射、校验真实来源后只读复验；不生成替代summary |
| v14事件窗口诊断 | 1 | 支线缺历史evaluation summary | 补齐所需依赖清单，再接只读入口 |
| v14绘图输入和回执 | 2 | 支线缺原evaluation目录 | 保留历史结论，派生图与回执写新目录 |
| v14最终历史产物核验 | 1 | 支线缺历史evaluation summary | 核对完整依赖，不用单文件字节通过替代历史结果验证 |
| v7内部评估patch的check-only | 1 | 支线缺原冻结preregistration | 保留原冻结SHA，适配显式只读根路径，不改锁 |

这里只列执行遇到的首个阻塞，不保证补齐后没有更深层依赖。完整测试ID、异常及堆栈保留在 [全量结果](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261111-01/final-v2-full/results.json) 和 [全量日志](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261111-01/final-v2-full/tests.log)。

### 4.2 二十五项跳过

- 18项旧启动器检查：v2为11项、v3为7项，当前未满足它们对项目内`.venv`、pwsh及相应可选依赖的既有前置条件。外部Python可运行本轮测试，不等于这18项已通过。
- 4项符号链接检查：当前Windows测试进程不能创建所需链接。没有修改系统权限规避跳过。
- 2项CUDA检查：当前测试环境不可用，不以CPU静默替代；其中1项也出现在contracts组。
- 1项EEGdenoiseNet下载数据检查：所需数据未下载。

本轮没有重新运行Windows打包、独立包启动、干净主机、性能、30分钟负载、2小时长稳、真实设备或物理刺激/marker验收。这些项目不计入本轮通过数，也不沿用11月6日结果冒充本次执行。

## 5. 兼容影响和原工作区保护

`analyze(report)`的来源字段由错误的默认路径/哈希改为null；依赖该字段的调用方应处理“来源未绑定”，需要文件身份时走 `run()`。算法输出和实验禁用状态不变。历史报告不原地重算，旧模型加载器、锁、前处理及正式配置策略保持原样。

契约测试需要新提交的夹具文件，夹具已纳入源码身份。证据工具输出只表示文件字节核对，不授予运行或部署资格；`local_inventory`不升级为历史权威认证。

对照本轮入口清单完成最终保护核验：

- 原工作区 `F:\脑机采集\NeuroStation` 的HEAD、tracked修改状态及暂存diff摘要一致。
- 登记的302份源码、配置、测试及资源文件一致。
- 登记的12项依赖一致：2个Python可执行文件、10项模型产物。
- 4份冻结模型绑定源码一致：训练脚本、matched decoder、matched public benchmark、open commands benchmark均未修改。
- 11月6日ZIP的131111482字节和SHA保持不变：`b73600bc70a22df49f9e2a9e984e59b1b42188ee23e949747d6dd774db70339e`。

保护复核仅覆盖上述已登记范围，不声称扫描了所有个人记录。旧ZIP没有本轮提交，不能当作本轮修复后的安装包；没有清理既有历史临时副本。核验明细：[preservation-final-verified.json](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261111-01/preservation-final-verified.json)。

## 6. 分批提交与复跑方式

| 提交 | 内容 |
|---|---|
| `1dd331a` | 修复时间确认分析来源绑定并补回归 |
| `f046d05` | 合成研究契约与历史产物解耦 |
| `6bd5627` | 显式只读证据核验、验收分组及剩余可合成测试解耦 |

文档和机器摘要另作独立提交，不回写已验代码提交。最终验收源码身份与代码HEAD记在每份结果及下列机器摘要中。

在PowerShell使用已验证的解释器，每次选择新的输出目录：

```powershell
$root = 'C:\Users\Administrator\.codex\worktrees\neurostation-quality-upgrade\NeuroStation'
$python = 'F:\脑机采集\NeuroStation\.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
Set-Location -LiteralPath $root
$run = Join-Path $root ('quality-output/recheck-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
& $python -B "$root/scripts/verify_quality_upgrade.py" --mode contracts --output "$run/contracts"
& $python -B "$root/scripts/verify_quality_upgrade.py" --mode focused --output "$run/focused"
& $python -B "$root/scripts/verify_quality_upgrade.py" --mode ui --output "$run/ui"
& $python -B "$root/scripts/verify_quality_upgrade.py" --mode full --output "$run/full"
& $python -B "$root/scripts/audit_historical_test_evidence.py" `
  --manifest "$root/quality-output/audit-20261111-01/historical-test-evidence-manifest-final.json" `
  --output "$run/historical-evidence.json"
```

验收脚本自动隔离数据、设置和诊断输出，并使用离屏UI。逐组查看 `results.json` 的 `issues`、`skips`、`checks`、`fully_passed_selection`，不能只看进程退出码。全量在未处理上述历史依赖前预计仍返回非零。只读清单是本机文件，复制源码到另一台机器后须提供真实映射和独立核实的身份，不自动生成虚假的pin。

## 7. 后续开发顺序和合并门槛

1. 让剩余8个历史集成检查使用显式只读依赖入口，并枚举它们的完整依赖；不能仅把它们从full移走。历史报告、锁及绑定源码保持不变，派生结果进入新的支线目录。
2. 将18项旧启动器检查的环境入口改为显式已验证解释器，测试缺依赖、错误环境和拒绝不安全checkpoint，不在支线伪建`.venv`掩盖问题。
3. 在具备权限、CUDA和所需数据的环境中补跑相应跳过项；条件仍缺失则保持未验证。
4. 源码验收范围明确后生成新的Windows包，进行干净主机试用，再由现场操作员明确启动设备与刺激验收。真实数据回放、物理时序和长稳结果必须单独记录。

本轮可作为独立支线源码交付和下一轮开发基线；尚不满足“全部自动测试与实机验收通过”的合并依据。

## 8. 证据索引

| 内容 | 本轮文件 |
|---|---|
| 入口源码和保护基线 | [entry-state.json](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261111-01/entry-state.json) |
| 入口全量结果 | [entry-full/results.json](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261111-01/entry-full/results.json) |
| 最终独立契约 | [contracts/results.json](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261111-01/final-v2-contracts/results.json) |
| 最终核心专项 | [focused/results.json](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261111-01/final-v2-focused/results.json) |
| 最终离屏界面 | [ui/results.json](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261111-01/final-v2-ui/results.json) |
| 最终全量核心 | [full/results.json](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261111-01/final-v2-full/results.json) |
| 最终406文件源码清单 | [source-manifest.json](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/quality-output/audit-20261111-01/final-v2-full/source-manifest.json) |
| 机器可读摘要及证据SHA | [quality-upgrade-audit-20261111.json](C:/Users/Administrator/.codex/worktrees/neurostation-quality-upgrade/NeuroStation/docs/quality-upgrade-audit-20261111.json) |

详细运行证据保留在支线独立的ignored输出目录，报告和机器摘要提交到仓库。若迁移验收证据，应按机器摘要中的路径与SHA核对，不只复制摘要后宣称原日志已随源码交付。
