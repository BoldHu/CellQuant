# 阶段6：分析真实结果并判断论文强度

读取真实完成的run结果与protocol/hash，不允许从外部论文数值拼成自己的reproduced表。Notebook04当前只有基础结果检查；请完成层级paired bootstrap、图表和G4 evidence生成。

主指标在每个seed/模型/数据集上先算sumNLL/valid tokens，再取exp；跨四格比较用log-PPL等权差值。报告原始3seed以及document/block层级paired95%CI。六任务按固定metric和等权汇总，保存逐题paired差值；不能把相关tokens当独立样本。

做两条paper路线判断：准确率（相对预先固定的最强公平基线）与效率（相对精度合格的最快基线）。严格按gate_rules，不选择更弱对手。低于阈值可写负结果或缩小结论，不能追改阈值。

交付：main表、精度—总校准时间Pareto、动态/固定尺度与STE/grid比较、Nu/N与事件复杂度、RTN→GPTQ迁移、必要消融、成本表、claim–evidence表、G4报告和审稿人式风险判断。只用fake quant时不得报真实INT4加速。更新ParoQuant/TORQ/PiSO/OBCD/CROQuant/2026后续版本查新；新颖性不足时不要保证顶会。
