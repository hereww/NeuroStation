# NeuroStation v7 内部诊断兼容补丁记录（2026-10-08）

## 目的

原 v7 内部诊断已经完成真实 RTX 4090 推理，但在发布报告前被 `finite JSON required` 拒绝。定位结果是 PyTorch `torch.__version__` 在当前环境中为 `torch.TorchVersion`（`str` 子类），而冻结的 JSON 校验器要求原生 `str`。这属于报告序列化元数据兼容问题，不是模型推理、数据读取或工程门失败。

原 v7 评估器及其原有评估锁、代码锁保持不变。本补丁只转换 GPU 元数据中的 `torch_version`/`cuda_version` 字符串子类，并写入单独的偏差说明、补丁代码锁和补丁诊断锁。

## 诊断范围

- 数据域：`Nakanishi2015`、`Wearable`
- 受试者：内部已知开发受试者 5、6
- 案例数：4,200
- phase 案例：840
- 未定义 phase 桶：0
- 未访问 outer/known-validation 7–8 或 holdout 9–10
- 未访问 `D:\SSVEP`
- 记录 reference 不是 clean ground truth
- 本补丁不启用实时运行，不构成生产或临床验收

## 结果

- `Nakanishi2015`：内部工程门通过
- `Wearable`：内部工程门失败
- 失败门：`high_frequency_burst.ratio_of_mean_reduction`
- 实际值：`0.022462016659629862`
- 要求值：`0.05`
- 因此总体 `all_internal_gates_passed=false`
- `model_state_sha256_before == model_state_sha256_after`
- `runtime_enabled=false`
- `production_validated=false`
- 部署继续保持 blocked；没有接入工作站实时链路

## 审计与验证

- 补丁诊断 JSON：`analysis_outputs/ssvep_v7_internal_diagnostic_patch_20261008.json`
- 补丁诊断 Markdown：`analysis_outputs/ssvep_v7_internal_diagnostic_patch_20261008.md`
- JSON-only 审计：`analysis_outputs/ssvep_v7_saved_internal_audit_patch_20261008.json`
- 审计结论：`audit_passed=true`、`mismatch_count=0`
- v7 专项及补丁测试：253 项通过
- 正式配置检查：通过，协议状态为 `formal`
- UI smoke matrix：1.0/1.25/1.5 三个缩放档通过
- 正式采集、桌面网关和发布策略测试：35 项通过

## 下一步

不要因为序列化补丁通过就部署 v7。下一轮应作为独立 v8 实验，针对 Wearable 高频 burst 的域差异重新预注册模型/损失和验证门；不得放宽 5% 高频门槛，也不得把这次内部受试者诊断当作独立受试者验证。
