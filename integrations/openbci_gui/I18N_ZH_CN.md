# OpenBCI GUI 简体中文重构规范

## 1. 目标

- 支持 `en-US` 和 `zh-CN`，中文工作站默认使用 `zh-CN`；
- 用户可在全局设置中切换语言，选择写入 `GuiWideSettings.json`；
- 翻译缺失时逐项回退到英文，不能显示空白或直接崩溃；
- 数据文件、网络协议和第三方程序看到的机器标识保持兼容；
- Windows、Linux、macOS 上字体、标点、换行和高 DPI 布局一致可用。

## 2. 国际化基础层

在 OpenBCI GUI fork 中新增：

```text
OpenBCI_GUI/
├── I18n.pde
└── data/
    ├── i18n/
    │   ├── en-US.json
    │   └── zh-CN.json
    └── fonts/
        ├── NotoSansSC-Regular.otf
        ├── NotoSansSC-Medium.otf
        └── OFL.txt
```

统一调用接口：

```java
tr("common.start")
tr("device.serial_port")
trf("recording.samples", sampleCount)
```

词条键使用含义稳定的命名，禁止把英文原文直接当键：

```json
{
  "common.start": "开始",
  "common.stop": "停止",
  "device.serial_port": "串口",
  "recording.start": "开始记录",
  "recording.stop": "停止记录"
}
```

`I18n.pde` 应负责：

- 根据全局设置加载 locale；
- 合并英文默认词条与目标语言词条；
- 提供 `tr`、带参数格式化和缺失键统计；
- 开发模式显示缺失键，发布模式回退英文；
- 切换语言后通知页面重建标签和下拉选项；
- 启动时校验两种语言的占位符数量一致。

## 3. 字体

当前 Open Sans、Montserrat 和 Raleway 不能作为完整简体中文字库。中文版统一引入 Noto Sans SC Regular/Medium，并随安装包带上 OFL 许可证。Noto CJK 官方仓库采用 SIL Open Font License 1.1。[Noto CJK 字体许可](https://github.com/notofonts/noto-cjk/blob/main/Sans/LICENSE)

字体服务统一生成标题、正文、小字和等宽字体，业务页面不能再自行 `createFont("Arial", ...)`。必须测试：

- 常用汉字、希腊字母、µV、Ω、Hz 和上下标；
- 粗体回退；
- ControlP5 下拉框、按钮、Tooltip 和 TextArea；
- 100%、125%、150%、200% DPI；
- 中英文混排和文件路径中的中文字符。

## 4. 不能翻译的内容

以下内容保持稳定，显示层可以旁边增加中文解释：

- JSON key、CSV/BDF 字段名和历史设置文件 key；
- LSL stream `type`、OSC address、UDP payload 和串口命令；
- `Cyton`、`Ganglion`、`BrainFlow`、`OpenBCI` 等产品或项目名称；
- 通道标识 `CH1`、`N1P`、`SRB`、`BIAS`；
- 文件扩展名、版本号、IP 地址和端口；
- 日志错误代码及用于自动测试的稳定标识。

不要把翻译文本用作 ControlP5 controller 的内部 `name`。内部 ID 保持英文稳定，只翻译 caption；否则保存设置、回调函数和自动测试可能失效。

## 5. 首版翻译批次

### P0：框架与启动链路

- `I18n.pde`、locale 设置和英文回退；
- CJK 字体服务；
- 只读位置启动错误、系统兼容错误和更新提示；
- 语言选择器；
- 翻译完整性测试。

### P1：完成一次 Cyton 采集所需界面

- 启动页和 `ControlPanel.pde`；
- 数据源、Cyton、串口、通道数和 SD 卡设置；
- 开始/停止数据流；
- 开始/停止记录、文件名和保存位置；
- 通道开关、硬件设置和阻抗检查；
- 连接失败、串口占用、无数据和保存失败提示。

### P2：核心小组件

- Time Series、FFT、Head Plot、Band Power；
- Marker、Networking、Playback；
- Filter、布局、缩放和通道选择；
- Tooltip、弹窗和用户可见警告。

### P3：其余设备和高级功能

- Ganglion、Cyton Daisy、WiFi；
- EMG、Focus、Pulse、Analog/Digital、Spectrogram；
- 专家模式、调试界面和帮助文本。

## 6. 中文术语表

| English | 简体中文 | 说明 |
| --- | --- | --- |
| Data Stream | 数据流 | 不译为“数据小溪” |
| Start System | 启动系统 | 指连接并初始化设备 |
| Start Data Stream | 开始数据流 | 与“开始记录”区分 |
| Recording | 记录/采集记录 | 根据上下文使用 |
| Playback | 回放 | 历史数据重放 |
| Channel | 通道 | 保留 CH 编号 |
| Impedance | 阻抗 | 单位保持 kΩ |
| Gain | 增益 | 不改硬件枚举值 |
| Bias | BIAS（偏置） | 首次出现双语 |
| Reference | 参考电极 | 硬件标记可保留 SRB |
| Marker | Marker（事件标记） | 协议字段仍用 marker |
| Time Series | 时域波形 | 组件标题可用“时域波形” |
| Band Power | 频带功率 | 频段名可保留 Delta 等并加中文 |
| Notch Filter | 陷波滤波器 | 显示 50/60 Hz |
| Detrend | 去趋势 | 算法名保持稳定 |
| Packet Loss | 丢包 | 同时显示数量和比例 |
| Session | 会话 | 对应一次实验会话 |
| Widget | 小组件 | 开发文档中可保留 Widget |

术语表作为翻译审核的唯一标准，不能在不同页面交替使用“频道/通道”“录制/记录”等表达。

## 7. 设置兼容与迁移

上游 `GuiSettings.pde` 当前会严格比较 JSON key 集合。直接加入 `language` 字段会把旧设置判为不兼容并重置。重构时应同时修改为版本化、向前兼容的设置迁移：

```json
{
  "schemaVersion": 2,
  "language": "zh-CN",
  "expertMode": "OFF"
}
```

读取旧设置时补默认值，不删除用户已有配置；遇到未知新字段应保留或忽略，不能因为字段集合不完全相等而重置整个文件。

## 8. 中文验收

- P1/P2 范围内不存在用户可见英文硬编码；
- locale 文件的 key、占位符和换行通过自动检查；
- 三个平台首次启动默认中文，切换英文后重启仍保持；
- 旧版 `GuiWideSettings.json` 自动迁移且不丢专家模式等设置；
- 中文路径可以打开、记录和回放；
- 1024×640 最小窗口、高 DPI 和长中文文本不重叠；
- 字体文件和 OFL 文本进入每个平台的发布包；
- 数据文件字段、LSL/OSC/UDP 协议以及历史英文设置保持兼容；
- 由熟悉脑电采集的中文使用者做术语审核，而不是只依赖机器翻译。
