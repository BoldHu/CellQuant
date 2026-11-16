# 阶段3：接入SpinQuant并验证完整模型

按docs/SPINQUANT_INTEGRATION.md固定上游commit，先核实真实文件/接口。实现adapter、Norm修正、R导出和GPTQ；不重写整套量化器。

主方法只拟合globalR1，R2固定。导出R1=Q.T并包含每一层R2；保留官方R1/R2学习的SpinQuant_full基线和R1-only归因控制。验证原始FP→合法Norm融合→旋转FP，在真实短序列上记录logits相对RMS和meanNLL差；失败先定位实现，不直接放宽tol。

canary用LLaMA-2-7B、128×2048校准、2seeds、W4A4KV4。新方法在RTN校准和最终GPTQ分别测，确认收益迁移。独立dev用于决策，不能调用默认WikiText2 test后反复调整K。记录包含采集、solve、融合和GPTQ的完整成本。

专业RTX若FP64旧路径太慢，允许受控FP32 port；但需保持算法/数据并检验数值和量化质量，清楚标our FP32 implementation。不能用最慢FP64比较夸大加速。

使用G2规范，输出实际evidence；缺少完整SpinQuant对照或GPTQ结果则INSUFFICIENT，不从论文表填数。
