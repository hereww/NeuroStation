# NeuroStation 1.04 正式版更新（2026-11-17 构建）

这是正式主线的软件更新构建，产品版本仍为 1.04 / 1.0.4，构建时间与源码提交另记于 VERSION.txt 和 BUILD_MANIFEST.json，不改写此前发布历史。

- 合入质量工作区、连续采集工程改进和四条件 SSVEP 参数设置、确认及静态预览。
- 独立 schema 3 四条件参数锁定方案的真实采集仍禁用，尚未完成该协议的 gateway、硬件及物理刺激时序验收；软件正式状态不表示质量或临床有效性已通过。
- 不含个人模型权重或个人数据；不自动启用实验模型、不连接设备、不自动闪烁。
- 正常启动沿用 Documents/NeuroStation/Datasets、Settings、Diagnostics 与 NeuroStation/NeuroStation 的用户设置；显式环境变量继续优先，支线运行绑定仅在明确传入时隔离生效。
- 升级前先核验包，再备份旧程序目录，最后替换；不删除、移动或重写原始采集数据、校准配置、用户档案和历史结论。
- Windows 打包入口 scripts/build_formal_windows.py 先验证正式配置，输出到新目录，运行独立程序/worker/诊断/质量回放检查及正式包资源检查；不是新 Windows 机器或真实硬件验收。
- SBOM 是构建环境分发包清单，不代表所有包都打入程序；实验模型与 torch 不随该程序分发。

## 窗口交接与副屏全屏修复

- SSVEP 默认方案统一为刺激 5 秒、试次间休息 5 秒、每频率 10 次；40 试次的采集计划为 395 秒，含准备共 400 秒。
- 刺激窗口在已预检的主屏全屏置顶；Windows 仅向用户启动的采集子进程请求前台权限。
- 修复混合缩放双屏中副屏窗口直接全屏时错误落到主屏的情况：先在目标副屏建立普通窗口，再进入全屏。副屏不再作为主窗口的 transient 子窗口。
- 副屏默认全屏，支持 F11 / 按钮切换，Esc 仅退出副屏全屏；质量、波形、说明分为三个页签，1280×720 逻辑分辨率下可显示全部 8 通道质量行。
- 双屏静态窗口验证不连接 Cyton、不生成 EEG、不启动闪烁；不能替代真实采集与光学时序验收。

## 单向移动采集开发补充

- 在现有四频率采集入口增加左→右、右→左两项，与左固定、右固定共同构成四种可选空间条件；一次会话运行一项，不自动串联。
- 沿用半屏矩形大小、频率相位和配置时长；每试次单向移动一次，不回弹，不在刺激期复位。结束灭屏后复位；掉帧如实记录并交由质量评估。
- 方向字段贯穿 UI、子进程参数、实时状态、session/config、事件/逐帧日志、数据集仓库与详情。旧眼别后缀保持兼容，移动数据分别使用 `_左到右` / `_右到左`。
- `eye_side` 是历史兼容的起始侧，不是移动条件或实际注视标签；新增 `stimulus_condition` 和 `condition_code` 是计划呈现条件，不是眼动测量。
- 保留 Cyton 握手、完整通道、光敏风险确认、显示刷新率/GPU/VSync 检查、取消和质量记录；移动模式拒绝 headless，静态预览不自动运动或闪烁。
- 独立 schema 3 四条件方案的未锁定参数仍未确认，真实采集入口仍禁用。本补充没有替它选定统一实验频率、试次数或组间间隔。
- 本次为源码开发补充，未重新生成或替换 Windows 程序包，未进行真实硬件或光学时序验收。

### 本次软件验证

- 正式配置检查通过（独立四条件方案仍显示参数待确认，不虚构锁定）。
- 采集相关专项 49 项通过，含单向轨迹、反向、灭屏复位、掉帧、取消、历史兼容、注入测试数据的双向会话落盘；界面全量 154 项通过。
- 全量核心回归共运行 2169 项：9 个失败、9 个错误、6 个跳过。未通过项涉及旧 v6/v7/v8 模型冻结元数据校验、个人会话文件缺失及旧 GPU 文档断言；本次未修改这些冻结配置、数据或断言以掩盖问题。
- 静态界面已离屏渲染核对，未启动真实闪烁。核心全量之后新增的两项专项用例已在上述 49 项回归中单独覆盖。


## GitHub CI repair — 2026-11-17

- Failure baseline: Desktop acceptance run 44024695339. All three operating systems
  stopped in the core test step, before packaging (Linux: 6 failures/40 errors;
  Windows: 14 failures/104 errors). The warnings about deprecated action runtimes
  were not the cause of these failed tests.
- CPU PyTorch test dependencies now live in a separate CI venv, not the desktop
  packaging environment. Intel macOS uses its compatible torch/NumPy pair.
- Windows CI tests/builds run from C:/NeuroStation-ci, preserving the historical
  prohibition on D: access. Local development remains on F:. Temporary paths are
  resolved before tests (Windows short-name aliases / macOS /var symlinks), and
  subprocess output is explicitly UTF-8.
- Git source line endings are deterministic. The v6/v8 configs retain the ORIGINAL
  frozen CRLF bytes and SHA256; no hash constant or historical verdict was changed.
- Nine tests of private historical artifacts are explicitly opt-in with
  NEUROSTATION_VERIFY_HISTORICAL_EVIDENCE=1. Enabled checks still fail on missing or
  changed evidence. Default skips are NOT historical, hardware or clinical acceptance.
  Portable synthetic tests continue to exercise actual loader rejection, hash
  verification and patch report publication without private EEG or trained weights.
- The v3 smoke document test checks the recorded historical source identity, not
  the hash of a later source revision. The historical document is unchanged.
- The v4 artifact verifier now rejects Windows-style traversal/drive/ADS paths on
  POSIX too; native traversal and resolved containment checks remain active.
- Validation at repair time: clean tracked-source core suite reached 2,171 tests
  with only the release-workflow path expectation remaining; that expectation was
  updated with the new C: workspace and retested along with new CI regression tests.
  Full UI suite: 154 passed. Formal configuration check: ready, no warnings.
  GitHub build/package acceptance must be read from the new run, not inferred here.
