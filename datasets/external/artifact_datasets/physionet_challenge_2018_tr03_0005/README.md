# PhysioNet/CinC Challenge 2018 — You Snooze, You Win — TR03-0005 representative record

## 结论

- 状态：`not_downloaded`
- 本目录不包含信号数据或压缩包；没有把未完成传输当作数据集副本。
- 实际执行日期：`2026-10-04`。用户要求的 `2026-10-05` 相对于当前日期（2026-10-04）是未来日期，因此没有写入为实际下载日期。

## 来源与下载

- 来源页：https://physionet.org/content/challenge-2018/1.0.0/
- 下载地址：
  - https://physionet.org/files/challenge-2018/1.0.0/training/tr03-0005/tr03-0005.mat
  - https://physionet.org/files/challenge-2018/1.0.0/training/tr03-0005/tr03-0005.hea
  - https://physionet.org/files/challenge-2018/1.0.0/training/tr03-0005/tr03-0005.arousal
  - https://physionet.org/files/challenge-2018/1.0.0/SHA256SUMS.txt
- 可直接下载/限制：yes_public_url_no_registration
- 注册与访问说明：PhysioNet page states anyone can access the files subject to the license.

## 许可

Open Data Commons Attribution License v1.0 (ODC-By-1.0); comply with attribution and license terms.

## 数据与用途

- 模态/通道：EEG；EOG；EMG；EKG；oxygen saturation；arousal annotations
- 对 EEG 伪迹/质量优化的价值：Very strong multimodal PSG candidate: EEG, EOG, and EMG were sampled at 200 Hz, with arousal annotations useful for quality/event-aware evaluation.
- 建议用途：
  - multimodal artifact detection
  - EOG/EMG reference-channel benchmarking
  - quality-aware sleep EEG preprocessing

## 本次结果

The 133,822,024-byte .mat signal transfer repeatedly stalled during range-download attempts. Metadata endpoints were reachable, but no complete signal file was retained and no metadata-only folder is presented as a dataset copy.

完整机器可读记录见同目录 `DATASET_MANIFEST.json`。
