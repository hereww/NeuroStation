# EEGLAB tutorial data — HBN ds004186 eyes-open/eyes-closed subset

## 结论

- 状态：`candidate_only`
- 本目录不包含信号数据或压缩包；没有把未完成传输当作数据集副本。
- 实际执行日期：`2026-10-04`。用户要求的 `2026-10-05` 相对于当前日期（2026-10-04）是未来日期，因此没有写入为实际下载日期。

## 来源与下载

- 来源页：https://eeglab.org/tutorials/tutorial_data.html
- 下载地址：
  - https://sccn.ucsd.edu/eeglab/download/ds004186subset.zip
- 可直接下载/限制：yes_public_url_no_registration_observed
- 注册与访问说明：The EEGLAB tutorial exposes a public ZIP URL. The page does not itself state the dataset license; consult the HBN/OpenNeuro metadata before redistribution or external release.

## 许可

Not asserted here because the EEGLAB tutorial page does not provide the subset license in the download entry; verify the upstream HBN/OpenNeuro license and citation requirements.

## 数据与用途

- 模态/通道：EEG; channel inventory and any auxiliary EOG/EMG channels must be verified from the BIDS files
- 对 EEG 伪迹/质量优化的价值：Useful as an EEG quality baseline and eyes-open/eyes-closed robustness set, but it is not treated as an EEG+EOG/EMG artifact corpus until channel inventory is verified.
- 建议用途：
  - baseline EEG quality metrics
  - state-dependent preprocessing checks
  - EEGLAB/BIDS import regression tests

## 本次结果

The official ZIP is approximately 499,003,596 bytes and the transfer did not complete in this environment. No partial ZIP was retained.

完整机器可读记录见同目录 `DATASET_MANIFEST.json`。
