# NeuroStation 1.04 正式版更新（2026-11-17 构建）

这是正式主线的软件更新构建，产品版本仍为 1.04 / 1.0.4，构建时间与源码提交另记于 VERSION.txt 和 BUILD_MANIFEST.json，不改写此前发布历史。

- 合入质量工作区、连续采集工程改进和四条件 SSVEP 参数设置、确认及静态预览。
- 四条件真实采集仍禁用，尚未完成 gateway、硬件及物理刺激时序验收；软件正式状态不表示质量或临床有效性已通过。
- 不含个人模型权重或个人数据；不自动启用实验模型、不连接设备、不自动闪烁。
- 正常启动沿用 Documents/NeuroStation/Datasets、Settings、Diagnostics 与 NeuroStation/NeuroStation 的用户设置；显式环境变量继续优先，支线运行绑定仅在明确传入时隔离生效。
- 升级前先核验包，再备份旧程序目录，最后替换；不删除、移动或重写原始采集数据、校准配置、用户档案和历史结论。
- Windows 打包入口 scripts/build_formal_windows.py 先验证正式配置，输出到新目录，运行独立程序/worker/诊断/质量回放检查及正式包资源检查；不是新 Windows 机器或真实硬件验收。
- SBOM 是构建环境分发包清单，不代表所有包都打入程序；实验模型与 torch 不随该程序分发。
