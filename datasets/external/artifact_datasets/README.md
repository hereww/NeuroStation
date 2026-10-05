# EEG 伪迹与质量数据集候选

本目录由独立数据任务搜索和核验。每个子目录包含 `DATASET_MANIFEST.json` 和 `README.md`。

当前记录：

- `tuh_eeg_artifact_corpus`：高价值伪迹语料候选，但需要访问流程，未下载。
- `physionet_sleep_edfx_sc4002e0`：包含 EEG/EOG/EMG 的 Sleep-EDF 记录候选；本次信号文件传输不完整，未保留可训练文件。
- `physionet_challenge_2018_tr03_0005`：包含 EEG/EOG/EMG 的记录候选；本次信号文件传输不完整。
- `mne_sample_eeg_eog`：适合预处理回归测试，未形成独立完整下载。
- `eeglab_hbn_ds004186_subset`：适合基线质量测试，不应直接当作伪迹语料库，未下载完整 ZIP。

任何未形成完整、可校验文件的数据集均不进入训练流程。
