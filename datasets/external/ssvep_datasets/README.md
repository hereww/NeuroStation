# SSVEP 外部数据集

本目录由独立数据任务搜索、下载并核验，数据文件保留在本机，详见各子目录的 `DATASET_MANIFEST.json` 和 `README.md`。

已完成：

- `ssvep_benchmark_wang2016`：35 个受试者，CC BY 4.0，约 3.71 GB。
- `dong2023_ssvep`：59 个受试者，8 通道、250 Hz、40 目标，CC BY-NC 4.0，约 701 MB。
- `nakanishi2015_ssvep`：10 个受试者，上游未声明明确许可，约 154 MB。
- `openbmi_ssvep_lee2019`：subject 1 的两个 session 子集，CC0 元数据，约 1.17 GB。

这些数据用于下游 SSVEP 目标频率/解码保持性评估，不自动作为本项目训练集，也不替代真实 Cyton 设备验证。商业再分发前必须重新核对各自数据许可。
