# NeuroStation v8 SSVEP 内部训练与诊断记录（2026-10-08）

## 目的

v8 是针对 v7 在 Wearable 域高频 burst 抑制不足的独立正式内部工程实验。v8 保留低频残差支路，将 `>40 Hz` 高频支路改为按通道、按 rFFT bin 预测独立的非放大衰减值，衰减范围固定为 `[0, 0.8]`；8–40 Hz 保护带和原有内部工程门保持不变。

本实验只用于离线训练和内部工程筛选，不等同于独立受试者验证、生产验收或临床有效性证明。

## 数据范围与安全边界

- 训练域：`Nakanishi2015`、`Wearable`
- fit subjects：1–4
- internal selection：5–6
- 诊断：5–6，4,200 cases，其中 840 个 identity/phase cases
- subjects 7–8、9–10：未加载
- `D:\SSVEP`：未访问
- recorded reference：未作为 clean ground truth
- 训练：RTX 4090 / `cuda:0`，18 epochs，AMP 关闭，TF32 关闭，确定性算法开启
- runtime：关闭
- production validation：关闭

## 训练结果

- `Nakanishi2015`：最佳 epoch 17，内部工程门通过
- `Wearable`：最佳 epoch 10，训练选择状态为 `gates_failed`
- 训练报告保留失败状态，不以失败结果伪装为通过

## 4,200-case 内部诊断结果

### Nakanishi2015

- 高频 burst ratio-of-mean reduction：`0.357327492194541`，要求 `>= 0.05`，通过
- identity normalized change：`0.0003726543040852864`，要求 `<= 0.0005`，通过
- 其他腐蚀条件、保护带能量/相位门：通过

### Wearable

- 高频 burst ratio-of-mean reduction：`0.08888101562484446`，要求 `>= 0.05`，通过
- transient、low-frequency drift、common-mode：通过
- identity normalized change：`0.000993923582533165`，要求 `<= 0.0005`，失败
- 因此 Wearable 域总体门失败，v8 总体 `all_internal_gates_passed=false`

## 审计与回归验证

- 诊断 JSON：`analysis_outputs/ssvep_v8_internal_diagnostic_20261008.json`
- JSON-only 审计：`analysis_outputs/ssvep_v8_saved_internal_audit_postfix_20261008.json`
- 审计结论：`audit_passed=true`，`mismatch_count=0`
- v8 模型/实验测试：174 项通过
- v8 内部诊断测试：45 项通过
- v8 JSON-only 审计测试：27 项通过
- 正式配置检查：通过，协议状态为 `formal`
- UI smoke matrix：通过（1.0 / 1.25 / 1.5）
- 正式采集与桌面回归：109 项通过；UI 回归：65 项通过

## 固定产物 SHA256

- 配置：`484950220254e0b2c256d8cea2bf4f5d8f9084fe7aa4616f406d5e240553a16a`
- 训练锁：`c229dbcfda94c9f01779238ef0823db5e0b33b938d3b1760bc1274467678fceb`
- checkpoint：`08170e1f9eb139233acdbf16dfb4f27e3405ecb81dcaf9ae4bb1e833c4ca92f8`
- 训练报告：`fc05c97efb3f9df490f29e493b2a93c0d9822980261ebb6e9b9adaf9df3b78ec`
- 诊断评估锁：`d5d4a7c5384e61d8d39440498d93b6ce55a3129bdb1fb521a79a813dcec6889b`
- 诊断代码锁：`846a12354a6bef017dc5dc9cc11d1ddfccf6ea57f8c38ebf92a1a4e64bf3131c`

## 结论

v8 已完成正式的离线 GPU 训练、内部诊断、JSON-only 算术审计和桌面回归检查。它解决了 v7 的 Wearable 高频 burst 门问题，但仍未满足 Wearable identity 保真门。因此：

- `runtime_enabled=false`
- `production_validated=false`
- `deployment=blocked`
- 不接入 NeuroStation 实时工作站
- 不放宽 `0.0005` identity 门或 `0.05` 高频门
- 后续若继续，应另行预注册针对 Wearable identity drift 的实验，或进入 locked validation 设计；不能把本次 5–6 号内部诊断称作独立验证
