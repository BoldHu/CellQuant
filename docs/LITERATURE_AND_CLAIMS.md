# 排重索引与论文主张边界

基线审查截至2026-11-10；2026-11-15追加定向检索。检索未穷尽全部Google Scholar引用、专利与未公开稿件；不得写“绝对没有人做过”。具体代码/附录在正式实现前应重新核对修订版本。

## 最近重合对象

| 文献 | 真实来源 | 必须排除的重复主张 |
|---|---|---|
| ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference | https://arxiv.org/html/2511.10645v2 | Givens、多阶段局部旋转、双侧W4A4输出重建、GPTQ组合 |
| TORQ: Two-Level Orthogonal Rotation for MXFP4 Quantization | https://arxiv.org/abs/2605.19561 | Schur–Horn、固定scale临界角、旋转半径排除不可达阈值 |
| Optimal Post-Training Quantization Scales and Where to Find Them（PiSO） | https://arxiv.org/abs/2606.10890 | 量化cell、精确尺度搜索、增量统计量、infimum边界处理 |
| A Block Coordinate Descent Method for Nonsmooth Composite Optimization under Orthogonality Constraints | https://arxiv.org/abs/2304.03641 | 正交块坐标、分段区间、多项式驻点求解 |
| CROQuant: Complex Rank-One Quantization Algorithm, with Application to Butterfly Factorizations | https://maelchaumette.github.io/publication/croquant_journal/CROQuant_journal.pdf | 双侧乘积不变自由度与量化cell搜索；公开稿不能误写已确认接收 |

候选差异仅限动态双侧共享尺度的实际矩阵乘积损失、完整事件几何、scale稳定组的安全全组省略、配套增量求解与复杂度。任何单一构件都不足以认定顶会创新。

## 其他直接相关工作

| 方法/标题 | 时间 | 原始链接 | 不能重新命名的思路 |
|---|---|---|---|
| QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs | 2024 | https://arxiv.org/abs/2404.00456 | 随机Hadamard与Transformer计算不变性 |
| SpinQuant: LLM Quantization with Learned Rotations | 2024/ICLR2025 | https://arxiv.org/abs/2405.16406 | 可学习全局/局部正交旋转、Cayley更新 |
| DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs | 2024 | https://arxiv.org/abs/2406.01721 | 离群值分配、块旋转和重排 |
| FlatQuant: Flatness Matters for LLM Quantization | ICML2025 | https://proceedings.mlr.press/v267/sun25l.html | Kronecker可学习一般变换及融合kernel |
| OSTQuant: Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for Better Distribution Fitting | ICLR2025 | https://proceedings.iclr.cc/paper_files/paper/2025/hash/5cebc89b113920dbff7c79854ba765a3-Abstract-Conference.html | 正交+缩放与分布拟合 |
| DartQuant: Efficient Rotational Distribution Calibration for LLM Quantization | NeurIPS2025 | https://arxiv.org/abs/2511.04063 | 缓存激活上QR旋转优化，非主模型反传不等于无梯度 |
| WUSH: Near-Optimal Adaptive Transforms for LLM Quantization | ICML2026 | https://proceedings.mlr.press/v306/chen26z.html | 权重/激活联合二阶矩闭式平衡，通常非正交 |
| ReSpinQuant: Efficient Layer-Wise LLM Quantization via Subspace Residual Rotation Approximation | ICML2026 | https://proceedings.mlr.press/v306/kim26bv.html | 层间旋转和低秩残差近似 |
| ThinQuant: Scalable Rotation Learning for Weight and Activation Quantization of LLMs | 2026-09 | https://arxiv.org/abs/2609.36120 | 激活薄子空间、精确降维和正交补全 |
| LIRQuant: An Efficient Locally Isolated Rotation Quantization Method for Large Language Models | FindingsEMNLP2026 | https://aclanthology.org/2026.findings-emnlp.1589/ | 敏感通道隔离和局部学习；主实验参考 |
| HARP: Hadamard-Preconditioned Adaptive Rotation Processor for Extreme LLM Quantization | EMNLP2026 | https://aclanthology.org/2026.emnlp-main.1773/ | 块正交/butterfly/Cayley与曲率匹配 |
| SpectrumQuant: Spectrum Alignment for LLM Quantization | 2026-10 | https://arxiv.org/abs/2610.20303 | 特征向量后接Hadamard，协方差对角均衡 |
| KernelQuant: Post-Training Quantization via Representation Kernel Preservation | EMNLP2026 | https://aclanthology.org/2026.emnlp-main.856/ | Gram/kernel保持与尺度、旋转优化 |
| ConQuR: Corner Aligned Activation Quantization via Optimized Rotations for LLMs | 2026-05 | https://arxiv.org/abs/2605.10793 | Procrustes角点对齐、流式闭式校准 |
| PrismQuant: Optimal Null-Space Rotations for Grouped Quantizers | 2026-09 | https://arxiv.org/abs/2609.32429 | 量化器自由子空间与Householder/组变换 |

综述入口：**Transforms for LLM Quantization: The Great Inversion and Format Co-Design**，https://arxiv.org/abs/2608.25188 。其系统检索截止2026年6月，不能替代之后工作的排查。经典KLT能量集中与共享尺度平坦化的差异已被讨论，不是本项目原创。

## 2026-11-15补查

新增：**SignRot: LLM Quantization with Massive Outlier-Aware Sign-Adjusted Rotation**，arXiv 2611.16502，2026-11-12提交，https://arxiv.org/abs/2611.16502 ，公开代码链接 https://github.com/HYU-ARK-Lab/SignRot 。本轮核实了原始摘要和提交信息，未声称已经审完全文及所有代码。

它针对down输入巨大离群值与Hadamard后的分布碎裂，使用符号调整并配合非对称激活量化。它不等同本方案的globalR1动态胞元求解，但进一步说明“改符号/避免Hadamard碎裂”不是可直接认领的新贡献。若做近期强基线，应核对asym协议和在线位置；不能把其模块加入本方案后仍称只优化R1。

## 论文允许与禁止的措辞

本轮代码构建时（2026-11-15）追加核对了以下原始摘要/版本页；这些记录不等于全文、代码与可复现强基线审计完成，G4 新颖性字段不因此填 true。

| 查询/原始来源 | 本轮实际读取范围 | 对本项目的限制 |
|---|---|---|
| [PiSO 2606.10890v2](https://arxiv.org/abs/2606.10890v2) | 摘要、2026-10-16 修订信息 | 分段尺度优化和量化 cell 已有先例；不能泛称首次精确量化搜索 |
| [OBCD 2304.03641v4](https://arxiv.org/abs/2304.03641v4) | 摘要、2026-05-14 修订信息 | 正交约束下小块坐标更新、breakpoint 搜索已有先例 |
| [SignRot 2611.16502v1](https://arxiv.org/abs/2611.16502v1) | 摘要、2026-11-12 提交信息 | down 输入位置的符号调整/非对称量化需单独协议，不能混入仅优化 global R1 的方法 |
| [SpectrumQuant 2610.20303v2](https://arxiv.org/abs/2610.20303v2) | 摘要、2026-10-23 修订信息 | 数据感知闭式正交变换是当前效率比较的相关方向；本项目未复现其性能 |

没有据这些论文报告数字生成 reproduced 表，也未完成全部 2026-11 新工作的穷尽检索。正式写稿前仍须执行 `prompts/08_NOVELTY_RECHECK.md`。

允许（须有真实证据）：

- 对指定双侧动态量化轨道推导cell内损失的代数结构。
- 导出避免枚举scale稳定组未变化成员的无损筛选与数据相关复杂度。
- 相同角度族/数据/墙钟下优于STE或grid，收益保留到GPTQ。
- 在同后端下不新增R1在线算子；总校准耗时具有实际改善。

禁止：

- 首次Givens/首次角事件搜索/首次无反向旋转/首次双侧输出MSE。
- 只凭热力图更平坦就证明模型更准。
- FP64数值根=形式化认证，RTN局部结论=GPTQ/整模型全局最优。
- CPU合成结果=LLM提升，fake quant=INT4 kernel加速。
- 把论文报告数值和自己的不同设置混表声称严格提升。
- 未查全文/未找到公开代码就写“肯定没有实现”。

写稿前执行prompts/08_NOVELTY_RECHECK.md并记录date/query/source/version/read-scope。新颖性复查与性能gate均通过才进入完整方法主张。

## 2026-11-18 竞争基线复核

ParoQuant v2（2026-02-14，ICLR 2026）已经覆盖可优化的成对 Givens 旋转、通道缩放和配套推理内核。作者代码在本日查询的 HEAD 为 `9ee635a9c2e7d02c46b1c0ac4071d69afbbfc1f1`。若 G1 获准继续，将 ParoQuant 列为近期实测候选，先审计模型支持、激活和 KV 位宽、分组及后端是否能匹配；记录原版配置和任何适配。完整 SpinQuant 仍需单独在相同模型、token、GPTQ 和评价协议下复现。本项目尚无这两者的模型级排名。[ParoQuant 论文](https://arxiv.org/abs/2511.10645) · [作者实现](https://github.com/z-lab/paroquant/tree/9ee635a9c2e7d02c46b1c0ac4071d69afbbfc1f1) · [SpinQuant 实现](https://github.com/facebookresearch/SpinQuant)

对 11 月 1–18 日的共享旋转、动态双侧量化、角度及胞元搜索作了定向排查。理论比较需逐项对齐目标、编码、共享尺度和可行性约束。例如 TORQ 面向 MXFP4，其公开指标使用不同量化格式。本轮核查限于检索到的论文页面与作者仓库，投稿前仍应审阅相关全文和更新版本。[TORQ](https://arxiv.org/abs/2605.19561)

### 输出行对实验的比较对象

ParoQuant 已研究独立成对 Givens、通道缩放和高效推理内核。ButterflyQuant 已研究由连续 Givens 角参数化的可学习蝶形变换。Narita 与 Sato 已对局部 W4A4 误差作精确的补偿项与正交残差分解。01d 在全模型共享 R1 下推导输出行对的动态双侧乘积目标如何分离，再用输入侧组峰值增量约束同时更新。对照实验检验哪些收益来自行覆盖和激活信息，误差分解用于定位最终收益。强基线与模型级比较仍需使用相同量化协议实测。[ParoQuant](https://arxiv.org/abs/2511.10645) · [ButterflyQuant](https://arxiv.org/abs/2509.09679) · [Narita 与 Sato](https://arxiv.org/abs/2609.21450) · [本项目实验](OUTPUT_MATCHING_STUDY.md)
