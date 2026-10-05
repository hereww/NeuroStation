# MNE-Python sample dataset — EEG/EOG-containing sample recording

## 结论

- 状态：`candidate_only`
- 本目录不包含信号数据或压缩包；没有把未完成传输当作数据集副本。
- 实际执行日期：`2026-10-04`。用户要求的 `2026-10-05` 相对于当前日期（2026-10-04）是未来日期，因此没有写入为实际下载日期。

## 来源与下载

- 来源页：https://mne.tools/stable/generated/mne.datasets.sample.data_path.html
- 下载地址：
  - https://mne.tools/stable/generated/mne.datasets.sample.data_path.html
- 可直接下载/限制：yes_via_mne_dataset_fetcher
- 注册与访问说明：No account requirement was identified on the MNE dataset-fetcher documentation. The official workflow downloads the project-managed sample archive; an authenticated record-level URL was not verified here.

## 许可

Follow the license and citation instructions distributed with the MNE sample dataset; do not infer redistribution rights from the Python fetcher alone.

## 数据与用途

- 模态/通道：MEG；EEG；EOG；ECG; EMG availability depends on the selected recording/run
- 对 EEG 伪迹/质量优化的价值：Useful for testing MNE-native loading, channel typing, EOG/physiological artifact detection, and preprocessing pipelines; less targeted than TUAR or multimodal PhysioNet PSG.
- 建议用途：
  - loader and preprocessing integration tests
  - EOG-referenced quality metrics
  - cross-tool validation with MNE

## 本次结果

The official MNE route is a large, multi-file managed download and did not expose a verified small record-level artifact subset during this run. It was not partially copied to avoid an incomplete dataset being mistaken for a usable corpus.

完整机器可读记录见同目录 `DATASET_MANIFEST.json`。
