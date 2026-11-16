# 阶段2：完成Colab真实层pilot

先读G1规范与configs/colab_pilot.json。现有notebook01的HF probe只能诊断，不是完整G1。请完成完整面板采集、globalR1求解与matched baselines，并在有权限和GPU时执行。

目标LLaMA-2-7b-hf层0/15/31，各q/up/down；2calibration seeds；独立fit/dev序列；完整2048上下文中可抽token，但保持每个量化组完整。q/up输入侧可以抽输出行；down必须按QB输出侧处理，不能把FFN宽度当hidden。

同一seed所有位置共享同一Q、同一R2/3/4、同一量化参数。比较Hadamard、same-family STE、grid、fixed-scale。按相同壁钟预算，同时保留评估次数；校准与held-out严格分开。记录每位置每seed NMSE、crest/scales、Nu/N、事件数、输出侧成本、峰值内存、接受率和总时间。

生成真实G1 evidence并调用cellquant.gates。没有真实模型权限/内存不足时保存RESOURCE_BLOCKED，允许较小模型做工程smoke但不能用它替代7B G1。不要为了PASS删掉失败位置、挑seed或改阈值。

Colab优先使用原有torch，先探测GPU支持的dtype；T4使用FP16就记录FP16并匹配基线。数据/模型留本地scratch，结果和小checkpoint同步Drive。HF token只从安全环境/Colab secrets读取，不打印。
