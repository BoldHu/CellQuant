# 严格正交变换下的大模型分组量化：项目实施方案

版本：2026-11-15。工作名：CellQuant。研究定位：动态双侧量化胞元上的正交校准。该名称不代表新颖性已经证明，也不代表已有论文结果。

2026-11-15 构建更新：生产事件求解、匹配对照、SpinQuant/GPTQ 与评估代码已落地；数学与 tiny 结构验证已执行，真实 G1–G4 未通过。当前事实以 [实现状态](docs/IMPLEMENTATION_STATUS.md)、[验证报告](VALIDATION_REPORT.md) 和 [运行手册](docs/RUNBOOK.md) 为准，以下研究目标和门槛不变。

## 一、项目目标与最小论文问题

在固定整数码本、量化分组和 Transformer 等价约束下，研究正交旋转对权重与激活动态尺度、整数编码及矩阵乘法误差的共同影响。主线是精确利用一维旋转轨道上的量化分段结构，而不是换一种固定正交矩阵。

基础关系采用列向量约定：

\[
Y=WX=(WQ^\top)(QX),\qquad Q^\top Q=I.
\]

只构造一个共享残差流旋转 R1；以少量校准面板共同适应不同层。保留 SpinQuant 的 R2/R3/R4 与模型结构，不训练基础权重，不增加新的在线 R1 算子。方法与同一角度族、同一目标、同一计算预算的梯度/网格求解器比较，才能判断新增求解机制是否有价值。

核心待验证假设：直接优化双侧 RTN 乘积损失比单侧平坦化或 STE 更有效；尺度稳定性筛选降低事件成本；局部收益可迁移到最终 GPTQ W4A4KV4。任一假设失败都可能使项目不值得扩大。

## 二、本包是什么、还缺什么

本包包含完整研究和实现规格、Codex 项目指令、阶段提示词、5 个 notebook、CPU 数学原型、真实 HF 层探测路径、环境/迁移/结果校验与 gate 工具。

**整模型科研验证尚未完成。** global R1 多面板事件求解、CPU 增量/CUDA dense 路径、SpinQuant/GPTQ 与评估适配现已实现，并有数学和小模型测试。全 GPU 增量事件 kernel、近期强基线及真实 G1–G4 仍未完成，详见 `docs/IMPLEMENTATION_STATUS.md`。Notebook 在资源或证据不足时保持 BLOCKED；不会生成假结果。

本次可复现证据是 synthetic CPU 数学检查。没有实际运行 Colab、AutoDL、LLaMA-2 7B、LLaMA-3 8B 或 Blackwell CUDA。`VALIDATION_REPORT.md` 记录本包真实验证范围。

## 三、数学方案

初始化 \(Q_0=HD\)，用严格正交的二维旋转乘积更新：

\[
Q_K=G_{i_Kj_K}(\theta_K)\cdots G_{i_1j_1}(\theta_1)Q_0.
\]

量化编码与反量化必须分开。对称 INT4 参考采用 code [-8,7]、\(s_g=\max_{i\in g}|z_i|/7\)、nearest-even；反量化为 \(s_g C_i\)。负端 -8 在无裁剪 AbsMax 参考中通常不会被使用，仍保留实际整数范围。

输入侧目标：

\[
F_{\rm in}(Q)=\sum_\ell\omega_\ell
\frac{\|\widehat{\mathcal Q}_W(W_\ell Q^\top)\widehat{\mathcal Q}_A(QX_\ell)-W_\ell X_\ell\|_F^2}
{\|W_\ell X_\ell\|_F^2+\epsilon}.
\]

输出侧 o/down 必须另写：

\[
F_{\rm out}(Q)=\sum_\ell\nu_\ell
\frac{\|\widehat{\mathcal Q}_W(QB_\ell)\widehat{\mathcal Q}_A(U_\ell)-QB_\ell U_\ell\|_F^2}
{\|B_\ell U_\ell\|_F^2+\epsilon}.
\]

其中固定的 R2/R3/R4 与合法 gamma 融合必须反映在面板定义中；不能采集一个路径的数据、却假定它代表另一个量化计算图。

约束：在冻结面板上，不让权重和激活的固定归一化尺度风险高于 Q0。无饱和最近舍入的风险上界为 \(B(Z)=\frac14\sum_g |g|s_g^2\)。有固定裁剪时加上距可表示区间的平方距离。第一版先支持对称无裁剪，再扩展固定 clip ratio；自适应 `w_clip` 与 GPTQ 不包含在该代数定理中。

沿单个 Givens 轨道，元素是 \(a+b\cos\theta+c\sin\theta\)。固定 scale owner、code 和裁剪分支的胞元内，反量化乘积至多二阶三角，平方输出损失至多四阶。用半角代换可将驻点变成至多八次多项式。边界必须按实际 rounding 计算，不能只取两侧极限。

若量化组其余元素的最大值 M 不小于配对半径 \(r=\sqrt{u^2+v^2}\)，则 AbsMax 在整圈不变；未旋转组成员不产生新的编码事件。输入侧事件数量有数据相关界 \(O(2^b[N+gN_u])\)。输出侧整行变化需另算事件成本。完整公式、证明范围和接口见 `docs/MATHEMATICAL_SPEC.md`。

## 四、为什么还需要严格排重

不能把 Givens、Hadamard、无主模型反传、输出 MSE、固定尺度临界角枚举或分段求根当新贡献。最接近先例是 ParoQuant、TORQ、PiSO、OBCD、CROQuant；LIRQuant/ThinQuant/HARP/SpectrumQuant/WUSH/ConQuR 已覆盖许多直观改法。

候选贡献只可能来自：动态双侧量化轨道的具体代数刻画；安全筛选与增量维护的总复杂度；与强对照相比真实可复现的准确率—校准成本改善。简单范数不等式本身不足以支撑论文。

参考索引与最近补查在 `docs/LITERATURE_AND_CLAIMS.md`。进入完整实验前和写论文前分别更新查新日志；若发现等价先例，停止首创表述并重新判断研究范围。

## 五、工程架构与责任边界

| 层 | 职责 | 实现原则 |
|---|---|---|
| notebook | 配置、运行、可视化、阶段判断 | Colab/AutoDL 共用逻辑，默认不会误启动长实验 |
| runtime | 环境、哈希、路径、日志、断点 | 缺失信息明确 BLOCKED，不填默认成功 |
| 数学参考 | CPU 小矩阵真值与回归检查 | 不用小维度冒充 4096 通道问题 |
| production solver | 完整组、多面板、输入/输出侧、增量事件 | Codex 后续实现；有事件/内存预算 |
| SpinQuant adapter | 固定上游、Norm 修正、R 导出、GPTQ | 不复制或随意重写整套量化器 |
| experiment | 匹配基线、calibration、eval、汇总 | 唯一配置驱动入口，原始结果可追踪 |
| gates | correctness、pilot、canary、full、paper | 缺真实证据不放行 |

生产入口约定为 `python -m cellquant.experiment --config CONFIG`。本包若未实现该模块，runner 必须明确报告缺失，不能用占位函数产生成功状态。

## 六、阶段路线与预算

### S0：本地/Colab CPU 数学与工程 smoke

运行 notebook 00，验证实际量化算子、正交性、乘积等价、胞元代数、安全筛选与约束更新。检查 runtime 原子保存、JSON gate 不接受缺失证据。数学 G0 通过不意味着 Transformer 融合已验证。

### S1：Colab 真实层初步验证

先用 notebook 01 完成真实模型环境、权重权限、内存、层采集、Hadamard 和安全筛选统计。这个现成 probe 只是输入诊断，不是完整 G1。

Codex 完成 production solver 的小面板路径和匹配 STE/grid 后，在一个目标 7B/8B 模型的早、中、晚三层，各 q/up/down 共 9 个位置，用同一个 global Q 测 held-out NMSE。2 个校准种子；固定成对 tokens/documents，不在 test 上调参。若 Colab 只有 T4 且目标模型无法合法加载，先用 1B/3B 验证流程；小模型证据不能当 7B G1 通过。

关键：小面板缩减 token/输出行，不缩减量化组中的通道。CPU 参考对维度设 guard，禁止通过截取 64 通道来绕过并假装真实层测试。

### S2：AutoDL 硬件迁移与限额 canary

“RTX P6000 96GB”需核实真实名称，通常应是 RTX PRO 6000 Blackwell 系列；上机运行 nvidia-smi、torch CUDA GEMM、FHT 与目标模型短前向。使用 notebook 02。

只有 G1 合格，才把模型级 canary 作为方法验证；若 Colab 无资源但诊断合理，可先做小预算硬件/可执行性 probe，不能称为 full。建议完整 canary 先设 12–24 GPU 小时预算，按实测吞吐调整；原版 SpinQuant 的 FP64 路径在专业 RTX 上可能特别慢，不能承诺该预算一定覆盖完整原版训练。

canary 用单模型、固定配置、至少 2 seeds，接真正 GPTQ W4A4KV4，检查完整 FP 等价、held-out 验证 NLL/PPL、峰值显存、总校准耗时和断点恢复。达 G2 后，再批准 notebook 03 构造完整矩阵。

### S3：完整主实验

核心 2 模型 × 3 seeds，W4A4KV4；WikiText-2/C4 和 6 项零样本任务。方法至少有完整 SpinQuant、随机 Hadamard、主方法、同族 STE 和 angle-grid 对照；加入能统一复现的 2026 强基线。阶段式执行，不一次盲跑数百小时。

### S4：机制、泛化及论文

主结果方向稳定后补尺度约束、动态/固定尺度、完整/筛选、增量/朴素、RTN→GPTQ 迁移消融。13B 或额外位宽择一优先，不先扩展 MoE/70B。Notebook 04 只读取真实 run 结果，输出图表、置信区间和 claim–evidence 表。

详细 G0–G4 数字见 `docs/GATES_AND_PAPER_CRITERIA.md` 与 `configs/gate_rules.json`。这些是预注册的项目决策阈值，不是会议录用标准。

## 七、什么时候值得迁移？什么时候值得全量？

迁移分成两个不同动作：

1. **有限额可行性迁移**：Colab 正确性通过、真实数据诊断没有实现错误，但资源不足以完成真实 G1；只允许硬件兼容与少量真实层试跑，不据此启动 full sweep。
2. **研究扩展迁移**：完整 G1 上的同一个 global Q，在 held-out 面板相对 Hadamard 及同预算求解器显示实际效果，且事件成本可控。此时进入 AutoDL 单模型 canary。

最重要的是 G2：在最终 GPTQ W4A4KV4 上，局部收益没有消失，且目标 GPU 的内存/时间/精度行为明确。只有它通过且G3运行准备就绪，才值得启动全量实验。

paper 阶段有准确率和效率两条路线，但共同要求数学正确、强基线公平、多个种子、核心消融、完整原始结果与新颖性复查。只比 Hadamard 好、只有单层 NMSE、只报一个 seed 或只在 RTN 下好，均不足以写成独立顶会方法论文。

## 八、实验协议

- 模型：`meta-llama/Llama-2-7b-hf`、`meta-llama/Meta-Llama-3-8B`；不得混成 3.1、Instruct 或不同 tokenizer。
- 主量化：W4A4KV4。主协议对称 W/A/K/V，完整记录分组、clip、GPTQ damping/act-order。另开官方 asym panel 时重跑所有对照，不能混表。
- 校准：128×2048 WikiText-2 train；pilot 可先 16/32/64 条，正式结果全部匹配预算。
- 验证：从 train 独立留出 documents/blocks，用于角度预算、clip 等选择；Wiki/C4 test 仅在主配置冻结后评估。
- PPL：sum NLL / number of valid predicted tokens 后取 exp；记录 BOS、拼接、边界、不足一块丢弃策略。C4 固定 revision/子集/tokenization，不能混换采样脚本。
- 六任务：ARC-C、ARC-E、HellaSwag、LAMBADA、PIQA、WinoGrande；冻结 lm-evaluation-harness commit、task id、0-shot、metric key，保留逐题预测。
- 对照：官方完整 SpinQuant (R1+R2) 与 R1-only 归因对照分开。主方法/STE/grid/Hadamard 固定同一 R2/3/4、同一数据/预算/设备。
- 统计：至少 3 个校准 seeds；paired document/block bootstrap，保留 seed 效应。不同 token 不是独立实验重复。报告 effect size 与 CI，不只报告 p-value。
- 论文引用值单独放 `source_type=published` 表，不参与自己复现的相对增益与 gate。

## 九、资源与迁移策略

96GB 显存可以容纳 7B/8B 的本方法、GPTQ 和评估，但不是代码兼容或高速度保证。基础 BF16 权重约 12.55/14.96GiB；128×2048 的 GPTQ inps+outs 约4GiB，另有 Hessian、工作区、词表 FP64 副本等。13B 基础权重约24.25GiB。新方法避免每次全模型 FP64 旋转，不等于原版 SpinQuant 基线可按同样时间预算完成。

建议 128GB CPU RAM，7B/8B最低可先64GB；多模型正式工作128–256GB。磁盘最小200GB、论文阶段0.5–1TB。原始模型/缓存留本地盘，小型结果与旋转checkpoint同步持久目录。不要把上百GB缓存每步写Drive。

Colab 的具体GPU、运行时长与可用内存不是保证；preflight 决定 profile。AutoDL 目录是否持久必须在实际实例核实，不能把 `/root/autodl-tmp` 名称当作永久存储承诺。全量实验以已测单配置耗时估算剩余预算，不直接套用 LIRQuant 8×H20 论文时间。

GPU 架构、依赖、磁盘与阶段预算详见 `docs/HARDWARE_AND_MIGRATION.md`。

## 十、Codex 实现里程碑

| 里程碑 | 完成物 | 验收 |
|---|---|---|
| M0 | 核对本包、上游 commit、现有 tests、实现日志 | 复现 CPU evidence，无虚构已实现模块 |
| M1 | 完整数学参考、degenerate/tie/group/asym分支设计 | 数学检查与 brute-force 对照；写清适用范围 |
| M2 | 输入+输出侧 production solver；同一个 global Q | 小面板与reference一致；真实完整组可运行 |
| M3 | 匹配 STE/grid、真实层 2 seeds G1 | 真实 G1 JSON，不使用 per-layer independent Q |
| M4 | SpinQuant R 导出、Norm修正、FP等价、GPTQ | 完整模型 canary，无新增R1在线算子 |
| M5 | Blackwell兼容、恢复、计时/内存、G2 | 目标GPU实测，不能用CPU测试代替 |
| M6 | C4/任务适配、全量矩阵、raw predictions | 所有结果可追踪，paper gate可计算 |
| M7 | 关键消融、稳定性、13B/位宽扩展、论文图表 | 核心机制成立且新颖性复查完成 |

用 `prompts/00_MASTER_BUILD.md` 启动，再按编号提示词推进。每个阶段完成实际代码与验证，而不只回复计划。

## 十一、停止条件

- 漏事件、输出侧错误、模型FP等价失败：先修实现，不进入性能比较。
- 相同预算下不如 STE/grid：精确求解机制价值不足，暂停扩展。
- RTN好但GPTQ不改善：不能继续用RTN结果支持最终目标；可重定义问题但需要重新立项及排重。
- scale预算使大多数更新停滞，或真实层事件数爆炸：先证明可计算性，不能靠盲增GPU解决。
- 仅相比Hadamard有效、对SpinQuant/近期强基线无优势：可以记录技术结果，不能宣称高水平方法贡献。
- 发现等价公开方法：停止首创表述。不得加PCA、低秩、敏感分组来强行改名。

## 十二、论文组织与核心图表

候选标题应突出 quantization-aware orthogonal calibration / dynamic shared scales，而非 new Givens transform。建议正文包括：问题与反例；胞元结构定理；稳定组筛选与复杂度；完整算法；SpinQuant适配；准确率—成本实验；限制。

核心图表：

1. 两模型W4A4KV4主表（PPL与六任务）；
2. 同硬件精度—校准时间Pareto曲线；
3. 动态尺度/固定尺度/STE/grid的held-out误差比较；
4. 事件数与Nu/N、筛选加速、输出侧成本；
5. RTN与GPTQ迁移图；
6. 核心消融和跨seed误差条。

先在G2之后写方法草稿和固定实验协议；只有G4证据齐全再写“完整论文结果”和强结论。论文门槛不等于录用保证。
