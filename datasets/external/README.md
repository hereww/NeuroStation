# 外部数据集目录

外部数据集由独立数据任务核验和下载。每个数据集目录都必须包含 `DATASET_MANIFEST.json` 和 `README.md`，记录来源、许可、下载日期、SHA256、文件结构以及不能确认的内容。

当前已下载并接入训练/评估流程：

- `eegdenoisenet`：clean EEG、EOG、EMG epoch 池，用于合成监督去噪样本。
- `ssvep_datasets/dong2023_ssvep`：用于目标频率保持 smoke test。

大型二进制文件只作为本地训练资料，不直接替代项目原始采集数据，也不作为正式采集记录提交。
