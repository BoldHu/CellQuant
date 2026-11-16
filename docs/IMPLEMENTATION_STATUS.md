# 实现状态与证据边界

更新：2026-11-15。以下“已实现”描述代码；实验结论单独列出。整个研究尚未完成。

| 组件 | 代码状态 | 已执行的证据与限制 |
|---|---|---|
| 原 CPU 参考与基础设施 | 保留并修正零组物理 scale 风险 | 原 62 项测试复现；最终全套 116 项通过 |
| 共享 global Q、多面板输入侧 WQ.T/QX、输出侧 QB | IMPLEMENTED | FP64 增量胞元系数、风险约束、实际边界 rounding；3 seeds/24 cases 合成 G0 PASS |
| 完整维度 | IMPLEMENTED / REAL_PERFORMANCE_NOT_RUN | d=4096、down=4096×11008，2 tokens、1 pair、±0.0001 的工程 smoke COMPLETE；不是正式吞吐 |
| 筛选与事件增量 | IMPLEMENTED_CPU | scale-stable 输入组筛选、输出完整行 signed hull；按实际事件收费并记录阶段耗时 |
| CUDA dense GEMM/量化 | IMPLEMENTED / SMALL_CASE_VALIDATED | 本机 CUDA 小共享求解器和 4096 元素量化一致性检查；全 GPU 增量事件 kernel NOT_IMPLEMENTED |
| Hadamard / angle-grid / Givens STE / fixed-scale | IMPLEMENTED | 相同面板、坐标预算、约束、墙钟上限；小矩阵验证。grid/STE 中途 optimizer 恢复 NOT_IMPLEMENTED |
| 真实层采集与 Colab G1 | IMPLEMENTED / REAL_7B_NOT_RUN | 逐层 GPU 调度，q/up 可抽输出行、o/down 完整 hidden；小随机模型已检查采集形状 |
| SpinQuant 固定上游 / overlay / R.bin | IMPLEMENTED | commit 8f47aa3… 已核对、tracked tree clean；禁 embedding centering、保 gamma，导出全部 R2 |
| FP 旋转 / GPTQ W4A4KV4 | IMPLEMENTED / TINY_CUDA_VALIDATED | MHA/GQA 小模型 FP 相对误差约 1e-7；GPTQ 有限输出、逐层恢复与连续运行参数差 0 |
| 完整 R1+R2 CE/SGDG baseline | IMPLEMENTED / TINY_CUDA_VALIDATED | 实际两步训练与 Trainer 恢复一致；正式 100 步×8 的完整 CE 训练池尚未运行，distinct blocks 数另报 |
| SpinQuant_full_FP32_port | IMPLEMENTED / NUMERICAL_ONLY | 输出和 R1/R2 梯度误差约 1e-7；小模型训练执行。真实质量资格与效率结论 BLOCKED |
| 固定数据与实验后端 | IMPLEMENTED / REAL_PIPELINE_NOT_RUN | contract 1.0、revision/token/源码/环境指纹、唯一 run_id、完整分片 SHA、原子结果、资源采样 |
| WikiText/C4 与六任务 | IMPLEMENTED / INTERFACE_VALIDATED | 固定 harness commit+六任务 revision/metric；六任务 local fixtures 实际跑通；真实分数 NOT_RUN |
| 配对分析 / G2–G4 构造 | IMPLEMENTED | token 加权 NLL 与 paired seed/document/item bootstrap；空结果准确不足，不生成表/图 |
| 五本 notebook | DEFAULT_PATH_EXECUTED | IPython inprocess，默认 smoke/manifest-only；真实 Colab/AutoDL notebook 会话未验证 |
| 目标 RTX PRO 6000 Blackwell / 优化 FHT | NOT_RUN / BLOCKED | 本机 RTX 3050 Laptop 4GiB；FHT、FlashAttention、Triton 缺失 |
| 2026 强基线、正式消融、论文主表 | NOT_IMPLEMENTED / NOT_RUN | 不能由官方论文数字或本地 fixtures 补填 |

## 科学门槛

G0 来自实测合成数学案例，为 PASS。G1–G4 没有完整真实证据，保持 INSUFFICIENT_EVIDENCE/BLOCKED 执行状态；资源中断的比较不被写为方法失效。`configs/gate_rules.json` SHA-256 仍为 `5d36290ca487795337300f3cccbc1c4d5a18d250639498bb15455ca5ad6a554e`。

事件根为 FP64 数值根。实现枚举候选并对预测最优的 32 个可行候选做完整量化复核；接受更新必须满足实际损失/预算，不能宣称所有浮点近退化情形下的角度全局最优或区间认证。事件/时间 cap 超限明确 RESOURCE_BLOCKED，不截断通道、不缩小正式角域使任务显示成功。

本机实际可见 GPU 约 4GiB、可用 CPU RAM 约 2GiB；未加载预训练 7B/8B，未下载其权重，未测真实 PPL/下游任务收益。高 RAM Colab / 目标 AutoDL 上所需命令见 [RUNBOOK.md](RUNBOOK.md)。端到端正式协议、资源预算和真实恢复仍待目标运行审查。

完整运行记录见 [VALIDATION_REPORT.md](../VALIDATION_REPORT.md)，本轮归档索引见 `evidence/build_20261116/`；工作目录 `runs/` 还保留原始 checkpoint、失败尝试及完整日志。没有根 Git 仓库，不能编造代码 commit；交付以逐文件 source snapshot SHA-256 标识。
