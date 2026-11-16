# 实验数据、门槛和恢复契约

## 一、实验结果与gate不是同一个文件

每次真实实验在独立run目录保存config/environment/manifest、stdout、分阶段time/memory、原始NLL/预测以及result.json。gate evidence是从若干真实run计算出的汇总；gate report是按冻结规则做出的判断。不能手写所有bool=true代替实验，也不能把report PASS当原始证据。

当前runner/分析器识别的正式结果字段：

```json
{
  "schema_version": "1.0",
  "source_type": "real_model",
  "evidence_kind": "REAL_MODEL_EXPERIMENT",
  "status": "COMPLETE",
  "model_id": "exact-model-id",
  "method": "registered-method-id",
  "seed": 11,
  "protocol_sha256": "COMPUTED_FROM_REAL_PROTOCOL_FILE",
  "metrics": {}
}
```

上面是结构说明，不是可用实验结果。examples中的实际模板保持NOT_RUN/null。正式metrics必须包含足以复核的raw文件引用，不能只存最终均值。

推荐方法ID：fp_reference、hadamard_shared、spinquant_full、spinquant_r1_control、givens_ste_matched、angle_grid_matched、fixed_scale_search_matched、cellquant。受控FP32 SpinQuant需另记录implementation_variant，不掩盖精度变更。

## 二、门槛输入

```json
{
  "schema_version": "1.0",
  "gate_id": "G1",
  "source_type": "real_layer",
  "metrics": {},
  "provenance": {},
  "artifact_refs": []
}
```

- G0 canonical source为synthetic_math，G1 real_layer，G2–G4 real_model。
- 未跑not_run；文献published；测试fixture unit_test_fixture。后两者不能通过真实实验门槛。
- 缺失/None/NaN/Inf/类型错误不得当0；布尔1不是true。
- common checks全部通过、至少一个route全部通过才PASS。
- 默认CLI exit0只是报告成功写出；只有report.status=PASS才科学通过，或调用--require-pass。
- gate_rules有SHA；改规则后原报告失效。下游要验证证据和规则hash，并重新计算门槛。

G3额外把当前 `execution_fingerprint_sha256` 放到evidence.provenance，必须与实际运行硬件/代码/数据一致。G1可以从Colab迁移到AutoDL，但目标GPU的G3必须重新验证。

## 三、执行provenance

`runs/provenance/execution_provenance.json` 的实际契约：

```text
models: {
  MODEL_ID: {model_revision: resolved hex SHA, tokenizer_revision: resolved hex SHA}
}
data_manifest_path: 实际文件路径
data_manifest_sha256: 实际文件SHA256
protocol_manifest_path: 实际文件路径
protocol_sha256: 实际文件SHA256
dtype: bfloat16 / float16 / float32 / float64
```

两个manifest必须存在且hash匹配；路径相对项目根或绝对。source snapshot、包版本、GPU信息由runner自动纳入fingerprint。尚未解析revision时不能写main或假哈希进入正式作业。换GPU/driver/关键依赖/dtype后，新建计划并重新G3，不自动复用旧运行。

## 四、原始NLL格式

每行至少保存：run_id、model、method、calibration_seed、dataset_revision、document_id、sequence/block_id、start_token、nll_sum、valid_target_tokens、token_ids_sha256。

PPL=exp(Σ nll_sum / Σ valid_target_tokens)。不可平均各batch PPL。跨seed主表可报告各seed PPL和其几何汇总；比较用NLL差值。不同模型、数据集主指标按预注册等权规则聚合，不能用token总量让C4主导WikiText2。

`cellquant.metrics.ppl_from_nll`和`paired_nll_bootstrap`提供基础实现，已用人工可验证的单元数据测试。后者仅针对一个model×dataset cell，支持按calibration seed与document cluster配对重采样；它不自动构造四格主指标或任务CI，完整paper aggregator由Codex实现。

## 五、任务预测格式

每行至少：task_id、harness_commit、metric_key、example_id、calibration_seed、gold、prediction或选项分数、correct/metric contribution、prompt/tokenization hash。六任务metric keys在运行前验证并冻结，不事后选择较大的acc/acc_norm。任务平均等权，accuracy delta单位为百分点。

## 六、时间与显存

每次run记录download（单独，不混校准）、capture、solve、fusion、GPTQ、evaluation、compile、warmup的墙钟。总校准=capture+solve+fusion+GPTQ。同硬件比较；GPU阶段计时前后同步。峰值同时记录torch allocated/reserved、NVML可见/使用量、CPU RSS及单位GB/GiB。不能把多个不同阶段峰值直接相加称实测峰值。

资源预算只根据完成的代表run外推；SpinQuant旧FP64与FP32port分别记录。fake quant只能比较该后端延迟与无新增R1算子，不能变成真实INT4部署加速。

## 七、迁移与恢复

校验请求的每个文件；missing文件不能被静默跳过，空manifest不能作为迁移成功。R.bin/angles/checkpoints需要显式纳入小文件备份策略。大权重若跳过必须列清单并保留来源/分片校验；移到新机再验证完整模型。

恢复要同时核对代码、config、模型/tokenizer、校准数据、协议、dtype与环境指纹；仅run_id相同不够。中断状态保留，成功结果不可无声覆盖。每个耗时步骤可恢复不是单个shell长命令突然结束后全部重来。
