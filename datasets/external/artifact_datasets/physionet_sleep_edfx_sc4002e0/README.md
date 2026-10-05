# Sleep-EDF Database 1.0.0 — SC4002E0 representative record

## 结论

- 状态：`not_downloaded`
- 本目录不包含信号数据或压缩包；没有把未完成传输当作数据集副本。
- 实际执行日期：`2026-10-04`。用户要求的 `2026-10-05` 相对于当前日期（2026-10-04）是未来日期，因此没有写入为实际下载日期。

## 来源与下载

- 来源页：https://physionet.org/content/sleep-edf/1.0.0/
- 下载地址：
  - https://physionet.org/files/sleep-edf/1.0.0/sc4002e0.rec
  - https://physionet.org/files/sleep-edf/1.0.0/sc4002e0.hyp
  - https://physionet.org/files/sleep-edf/1.0.0/SHA256SUMS.txt
- 可直接下载/限制：yes_public_url_no_registration
- 注册与访问说明：PhysioNet page states anyone can access the files subject to the license.

## 许可

Open Data Commons Attribution License v1.0 (ODC-By-1.0); comply with attribution and license terms.

## 数据与用途

- 模态/通道：EEG；horizontal EOG；submental EMG envelope；airflow；temperature；event markers
- 对 EEG 伪迹/质量优化的价值：Strong reference-channel candidate for ocular and muscle-related artifact/quality analysis; SC recordings provide EEG/EOG and a 1 Hz submental-EMG envelope.
- 建议用途：
  - artifact detection and quality scoring
  - EEG/EOG/EMG-reference experiments
  - sleep-physiology robustness checks

## 本次结果

The signal file transfer stalled in this environment before a complete .rec file was received. No partial or fabricated signal file was retained.

完整机器可读记录见同目录 `DATASET_MANIFEST.json`。
