# EEGdenoiseNet（EEG 去噪/伪迹去除数据集）

本目录保存从 EEGdenoiseNet 官方 GitHub 仓库下载并核验的 6 个数据文件，供 EEG 去噪、EOG/EMG 伪迹去除训练使用。

## 来源

- 论文：<https://arxiv.org/abs/2009.11662>
- DOI：<https://doi.org/10.1088/2634-4386/abdbdb>
- 官方代码与数据仓库：<https://github.com/ncclabsustech/EEGdenoiseNet>
- 官方 `data/` 目录：<https://github.com/ncclabsustech/EEGdenoiseNet/tree/master/data>
- 本次下载固定到仓库 `master` 在下载时解析到的提交：`8d290661146c7189c98cc04812d37371d4b9426c`
- 仓库 README 宣传的 G-Node 存储：<https://gin.g-node.org/NCClab/EEGdenoiseNet>

## 下载结果

- GitHub 公开数据：**已下载 6/6 个文件**。
- G-Node 512-Hz 变体：**未下载**。本次核验访问返回 `HTTP 403 Forbidden`，因此没有把该变体写成“下载成功”。
- 记录日期：`2026-10-04`。用户要求的 `2026-10-05` 晚于本次执行日期，因此没有伪造未来下载日期；请以后续实际运行日期重新核对时再更新。

## 文件结构与内容核验

| 文件 | 内容 | 形状 | MAT 变量 | 采样率 | SHA256 |
|---|---|---:|---|---:|---|
| `EEG_all_epochs.mat` | clean EEG epoch 池 | `4514 × 512` | `EEG_all_epochs` | 256 Hz | `63a6cf5eec716137b73de0cb4939335c309575df8243c5a97075bc2593a0b514` |
| `EEG_all_epochs.npy` | 同上，NumPy 格式 | `4514 × 512` | — | — | `89350579007ab18daf918a26c9f730484ba3755e861bbf8a6f68d1657843a748` |
| `EOG_all_epochs.mat` | EOG artifact epoch 池 | `3400 × 512` | `EOG_all_epochs` | 256 Hz | `d13fbaffcc7f5e317a7326a2e76faa8dad071b4239dc93a54e7e8905cec2692c` |
| `EOG_all_epochs.npy` | 同上，NumPy 格式 | `3400 × 512` | — | — | `fe31f2c22efce5da7488a1047b6f05d678e80f48d22e20a70972aaf13784fa0a` |
| `EMG_all_epochs.mat` | EMG artifact epoch 池 | `5598 × 512` | `EMG_all_epochs` | 256 Hz | `a89be332116f233c0b6f7acbbdcdbe18928527240115d561cb441acc403d2c7e` |
| `EMG_all_epochs.npy` | 同上，NumPy 格式 | `5598 × 512` | — | — | `8bb16dff87582823488dd2189dba9d8fa019a8cbc9af9a8a19acb421b3fcfb16` |

核验结果：

- 所有下载数组均为 `float64`，形状如上，且数值均为有限值。
- 每个 `.mat` 中除主数组外包含 `fs`；本地实际读取到三份 `fs=256`。
- 每一类数据的 `.mat` 主数组与 `.npy` 数组形状一致，并做了逐元素完全相等核验。
- 这 6 个文件合计 `109,916,871` bytes。

## 是否存在 clean EEG / EOG / EMG 配对数据

存在适合构造监督训练样本的三类**独立 epoch 池**：

- `EEG_all_epochs`：clean EEG ground-truth 候选；
- `EOG_all_epochs`：EOG 伪迹源；
- `EMG_all_epochs`：EMG 伪迹源。

但本目录中的官方文件**不包含预先生成的 noisy EEG–clean EEG 成对样本**，也没有证据表明 EEG、EOG、EMG 三者是逐 epoch 同步采集配对的记录。训练时应依据论文/官方代码定义混合策略，生成 noisy 输入并保留对应 clean EEG 作为目标；不要把相同的行号误当作跨模态配对关系。

## 许可与缺失项

官方仓库带有 MIT `LICENSE`，但没有为底层 EEG/EOG/EMG 数据单独声明数据库/数据集许可。因此：

1. 本地可以按来源使用已下载文件进行研究核验；
2. 不应把仓库 MIT 软件许可自动解释为数据许可；
3. 重新分发或商用前应保留作者/论文归属，并向数据发布者确认适用的数据许可。

已明确缺失或未随本次下载获得的内容：

- subject/session ID、逐 epoch 记录来源；
- 电极位置/通道 montage；
- 官方 train/validation/test 划分；
- 预生成 noisy-clean paired table；
- 单独的数据集许可文本；
- README 宣传的 G-Node 512-Hz 文件。

完整机器可读记录见同目录的 [`DATASET_MANIFEST.json`](./DATASET_MANIFEST.json)。
