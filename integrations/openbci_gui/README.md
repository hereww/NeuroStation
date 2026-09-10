# OpenBCI GUI 源码集成

本目录定义工作站如何拉取、锁定和重构 OpenBCI GUI。上游源码不直接复制进主仓库，而是由脚本拉取到被忽略的 `.vendor/OpenBCI_GUI` 独立 Git 工作区；主仓库保存版本锁、重构规范和集成代码。

这样处理有三个目的：

- 保留完整上游 Git 历史和 MIT 许可证；
- 中文化和功能重构可以在独立分支提交，并持续合并 OpenBCI 上游更新；
- 不让庞大的 Processing 源码、依赖库和构建产物污染工作站主仓库。

## 拉取源码

当前锁定版本见 `upstream.lock.json`。

Windows PowerShell：

```powershell
.\scripts\sync_openbci_gui.ps1 -Action sync
.\scripts\sync_openbci_gui.ps1 -Action overlay
.\scripts\sync_openbci_gui.ps1 -Action status
```

Linux/macOS：

```bash
bash scripts/sync_openbci_gui.sh sync
bash scripts/sync_openbci_gui.sh overlay
bash scripts/sync_openbci_gui.sh status
```

`build_openbci_gui.sh` 会根据 `uname -m` 选择 Linux x64/arm64 和 macOS x64/Apple
Silicon 的 Processing 4.2 发行包；最终 runtime 仍须在对应系统原生运行器上构建，
不能用 Windows 产物替代 Linux/macOS 验收。

同步脚本执行以下检查：

1. 从官方仓库 `https://github.com/OpenBCI/OpenBCI_GUI.git` 拉取源码；
2. checkout 到锁文件中的完整 commit，而不是漂移的 `master`；
3. 校验 `origin` 或 `upstream` 中至少有一个指向锁定的官方仓库；
4. 如果源码目录有未提交修改则停止，避免覆盖重构工作。

`overlay` 会在锁定 commit 上幂等应用主仓库保存的首批中文化改造。它只允许修改声明过的 6 个路径；发现其他本地改动会停止。`status` 同时显示 `ZhCnOverlayApplied`，所以重新拉取后能立即判断中文层是否已复现。

## 重构仓库策略

实际重构不能长期提交在 detached HEAD 上。确定团队 Git 托管地址后，应先创建 OpenBCI GUI fork，然后执行：

```powershell
git -C .vendor\OpenBCI_GUI remote rename origin upstream
git -C .vendor\OpenBCI_GUI remote add origin <团队的 OpenBCI_GUI fork 地址>
git -C .vendor\OpenBCI_GUI switch -c workstation/i18n-zh-cn
```

建议长期维护：

- `upstream`：OpenBCI 官方仓库，只拉取；
- `origin`：团队 fork，保存重构提交；
- `workstation/upstream-sync`：尽量只做可回馈上游的通用重构；
- `workstation/i18n-core`：本地化框架、字体和布局适配；
- `workstation/zh-cn`：简体中文词条；
- `workstation/integration`：与本工作站的启动、会话和数据互通。

主工作站通过 `upstream.lock.json` 锁定最终验证过的 fork commit。更新上游时，先在 fork 中完成合并、自动测试和三平台人工验收，再更新锁文件。

## 中文版范围

中文化规范和分批改造范围见 [`I18N_ZH_CN.md`](I18N_ZH_CN.md)。首个中文版必须覆盖启动、数据源、设备连接、通道设置、开始/停止、记录、回放、网络、marker、警告和错误；调试日志可以后续翻译，但用户可见错误不能遗漏。

当前 overlay 已建立英文回退、`zh-CN` 词库和 Windows/macOS/Linux 中文系统字体选择，并覆盖顶部导航、开始/停止数据流、数据源、串口/BLE/Wi-Fi、会话、通道数、采样率、回放、BrainFlow streamer、SD 卡和无线电配置等首批入口。它是中文版重构的第一批源码，不代表约 3,000 个上游字符串已全部翻译；完整中文版仍需按 `I18N_ZH_CN.md` 的分批清单继续验收。

## 源码审计结论

本次检查的上游版本为 `e23869e7b5cc621e733d8fa0d81f05d477264306`：

- 主程序为 Processing/Java，`OpenBCI_GUI/` 下约有 582 个 PDE/Java/JSON/properties 文件；
- 未发现现成的 UI 国际化框架；
- 用户文字广泛硬编码在 `ControlPanel.pde`、`TopNav.pde`、`W_Networking.pde`、`SessionSettings.pde` 等文件中；
- 粗略检出约 3,000 个英文字符串字面量，其中包含 UI、日志、协议键和开发文本，必须先分类，不能机械全部翻译；
- 当前主界面主要使用 Montserrat、Open Sans 和 Raleway 字体文件，中文版本需要增加 CJK 字体并验证 ControlP5 的文字测量和截断逻辑；
- 上游已有 `release/build.py` 和 `release/package.py`，中文版构建应在其基础上增加三平台 CI，而不是另外维护完全不同的打包链。
