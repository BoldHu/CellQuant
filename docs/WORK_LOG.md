# 工作记录

## 2026-11-15 — 初始交付

- 将上一轮研究方案转成 Codex 项目规范与 Colab→AutoDL 执行路径。
- 建立明确的 correctness / pilot / canary / full / paper 门槛。
- 本环境无 PyTorch/CUDA；只执行 CPU 数学、runtime、gate 与 notebook 默认路径测试。
- 当前不存在真实 7B/8B 结果。后续实验必须生成自己的完整 manifest 和逐样本证据。

## 2026-11-15 — 交付前冻结与验证

- 62 项 CPU 单元测试通过，覆盖数学参考、门槛、运行迁移和统计原语。
- 修复孤立最大值并列导致的 scale owner 错误，并加入回归检查。
- 重新执行最终 5 本 notebook 的 32 个默认 CPU 代码单元，源单元与执行副本逐项一致。
- 实际 notebook 引擎为 IPython inprocess；未验证真实 Colab/AutoDL GPU、外部 Jupyter kernel 或完整 LLM 后端。
- 6 个 canary、32 个核心 full 作业草案齐备；正式模型/数据/协议 revision 仍需真实解析并冻结。
- 交付验证细节见根目录 `VALIDATION_REPORT.md`；现有证据不代表完整 G0–G4 已通过。

## 后续记录模板

- 日期/里程碑：
- 代码 commit 与配置 hash：
- 实际执行命令/硬件：
- 输出目录与关键文件：
- 成功/失败/不足证据：
- 下一步及仍然未验证的假设：

## 2026-11-15 — 本轮构建与实际验证

- M0：读取规范并复现原 62 项测试。发现可用宿主 Python/Torch 和 RTX 3050 Laptop 4GiB，修正“无 PyTorch/CUDA”的旧环境假设；项目 venv 保留宿主 torch。根目录是无 .git 的交付目录，记录 source hashes，不伪造 commit。
- M1–M3：实现完整组的共享 Q 输入/输出事件求解器、signed support hull、安全筛选、解析损失/预算系数和增量更新、匹配 grid/STE/fixed-scale、逐层采集与 G1。实跑 3 seeds / 24 cases G0 PASS；完整 d=4096 / FFN=11008 的少 token、窄角域 smoke COMPLETE。正式区间 cap 超限仍阻塞。
- M4：固定 SpinQuant commit，tracked tree 保持 clean；生成带 hash 的 v2 overlay，关闭 embedding centering、保留 gamma，R1=Q.T 与全部 R2。实际 tiny MHA/GQA FP/GPTQ/RTN 恢复通过；GPTQ 中断后恢复参数与连续运行差 0。原版 FP64/明确 FP32 port 的 tiny CE/SGDG 均执行，FP32 只有数值验证。
- 修复实际测试暴露的问题：Transformers 4.44.2 API、GPTQ catcher/seq length/global RoPE、逐层恢复参数、Torch 2.6 RNG pickle 默认限制；Trainer 改用安全 tensor/primitive RNG 与文件 hash，恢复参数逐项一致。
- M5：固定 harness commit、六任务 metric 和 dataset revision；本地 fixtures 真实执行六任务。修复任务扫描开销、Windows 多进程 bootstrap 兼容与 dtype 序列化；由自己的 paired document/item bootstrap 提供 CI。逐题 choice 与 harness metric 一致性执行校验。
- M6–M7：统一 experiment contract、分片 SHA 校验、独立文档划分、全矩阵 provenance、G2/G3 实测证据构造、G1 portable bundle、日志/心跳/资源采样。完整 SpinQuant 保留 100×8 CE 预算，使用排除 dev 的完整训练池，不复用小 fit 池冒充完整训练数据。800 是 microbatch 预算；按上游 Trainer 语义可遍历整个池跨 epoch，独立 blocks 数另报。
- 最终 116 项测试通过，五本 notebook 默认路径执行成功；原始日志、环境与结果在 runs/build_final，归档在 evidence/build_20261116。门槛文件 hash 与初始包相同，G1–G4 仍无真实通过证据。
- 剩余：预训练 7B/8B、Colab G1、Blackwell/FHT、正式 GPTQ/下游/效率、近期 2026 强基线、消融与新颖性全文/代码审查；全 GPU 增量事件 kernel、grid/STE 中途 optimizer 恢复未实现。下一步命令与硬件需求见 RUNBOOK。

- 收尾核对发现 Agent.md 同步副本缺失，已运行 scripts/sync_agent_alias.py 恢复，确认与未改动的 AGENTS.md 逐字节一致；原 gate_rules.json 的初始包 SHA 复核一致。正式运行增加 clean 根代码 commit 检查，当前仅 source snapshot，不编造 commit。

- 最终中断一致性审查：修复 Q 更新与 factor/log 之间被中断时可能保存不一致 checkpoint 的窗口；新增注入中断回归，完整套件现为 116 项通过。
