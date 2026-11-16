# 阶段5：运行完整矩阵

读取G3报告、其原始证据和hash、execution_provenance与configs/full_experiments.json。先计划并估算预算，不默认执行缺后端或缺证据的作业。需要的后端必须暴露BACKEND_CONTRACT_VERSION='1.0'、run_experiment callable和CLI。

冻结两个模型revision、tokenizer、calibration/dev/test文档与token hashes、quant groups/clip/GPTQ、harness commit、任务metric、3seeds及K。每项改变都产生新protocol/run_id，禁止覆盖已有结果。完整表每个模型seed先完成全部主方法与基线，再扩展13B/位宽。

至少包含FP参考、Hadamard、完整SpinQuant、主方法、matched STE/grid；加入一个2026强方法复现。最近的SignRot适用asym/在线down位置时应独立协议比较或明确不对齐，不能拼入主方法后继续称R1-only。

保存逐文档/块NLL sum与valid token count、逐题任务结果、每seed和层指标、timing阶段、CUDA峰值allocated/reserved、CPU RSS和日志。不平均batchPPL。正式test后不得根据结果回调超参。OOM/失败也保留状态，不静默跳过失败seed。

主结果稳定后补五核心消融；只在G3已通过且真实资源足够时执行，所有长作业可中断和恢复。按实测吞吐估算剩余GPUh，不套8×H20数字。
