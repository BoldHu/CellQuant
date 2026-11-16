# 新颖性复查提示词

当前候选仅声称：动态双侧共享尺度量化轨道上的具体胞元代数、尺度稳定组无损筛选、增量事件求解及匹配成本收益。不是新Givens族。

以当前日期检索原始论文、附录、官方代码及修订版本，优先ParoQuant、TORQ A.4、PiSO、OBCD、CROQuant，再看ThinQuant、HARP、SpectrumQuant、LIRQuant、WUSH、ConQuR、SignRot。不要把摘要未提到的内容视为不存在；阅读附录/代码。逐项检查dynamic scales on both operands、actual dequantized matmul error、safe scale-owner screening、incremental row/column coefficients、output-side panels是否已有等价算法。

写novelty_matrix：exact title/date/version/source、原方法做了什么、与本项目差异、重合风险、尚未核实部分。若发现高度等价，明确淘汰当前首创主张，不通过换术语或拼接PCA/敏感分组挽救。检索不穷尽时说明范围，不能写absolutely novel。
