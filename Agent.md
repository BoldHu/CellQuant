# CellQuant research project — Codex repository instructions

本文件是 Codex 的规范入口。`Agent.md` 是供用户阅读的同步副本；修改本文件后运行 `python scripts/sync_agent_alias.py`。本项目工作名 CellQuant 不代表已证明的新颖性或已发表的方法名称。

## 1. 任务与当前状态

目标：研究严格正交的全局残差流旋转，在固定分组 INT4 下直接降低双侧动态量化后的乘积误差；先 Colab 验证，再迁移至 AutoDL 的实际 96GB GPU。完整方法必须基于 Meta SpinQuant，而不是另建一套无法比较的量化框架。

首先阅读 `README.md`、`PROJECT_PLAN.md`、`docs/IMPLEMENTATION_STATUS.md`、`docs/MATHEMATICAL_SPEC.md`、`docs/SPINQUANT_INTEGRATION.md`、`docs/GATES_AND_PAPER_CRITERIA.md`、`configs/gate_rules.json`。每次接手先核对代码与状态记录，不根据旧聊天猜测进度。

交付包只包含已验证的 CPU 数学参考、运行/迁移底座、notebook 和完整实现规范。完整 GPU 全局旋转求解器、SpinQuant 适配、匹配基线、GPTQ/C4/零样本评估由后续 Codex 实现。不存在的功能必须保持 `NOT_IMPLEMENTED` 或 `BLOCKED`，不得用随机结果、占位数字或弱化实验冒充完成。

## 2. 科学不变量

- 列向量约定：`Y = W @ X`；`W_rot = W @ Q.T`，`X_rot = Q @ X`。官方 checkpoint 的 `R1 = Q.T`，不可写反。
- 主方法仅构造一个全模型共享 R1；R2/R3/R4 固定。不得偷偷引入各层独立残差旋转、非正交缩放、敏感通道分区或基础模型训练。
- `o_proj` / `down_proj` 是输出侧 `Q @ B`；不能把它们当成输入侧 `W @ Q.T`。FFN down 的输入维不是 hidden dimension。
- 输入侧 q/up/gate 面板允许抽输出行、token；不得抽掉归约通道再重算 scale。输出侧 Q@B 要保留完整旋转维，不能抽行后把该小矩阵当作完整 Q。
- 量化编码与反量化值必须区分。主数学参考：对称 nearest-even RTN、code [-8,7]、s=maxabs/7、无裁剪。全零组单独处理。
- 固定胞元内真实双侧乘积至多二阶三角多项式，平方损失至多四阶；此性质仅适用于冻结面板和指定量化分支，不覆盖整个 Transformer 或 GPTQ。
- 尺度变化会使未旋转成员的 code 改变。完整事件枚举必须考虑它们，除非经过正确的安全筛选。
- `M >= sqrt(u*u+v*v)` 只保证 AbsMax 稳定；非对称 min–max 需单独证明；输出侧整行变化不能套“仅两个组成员改变”的复杂度界。
- 在事件边界实际执行 rounding；同时事件、near-tie、零尺度、退化多项式须有覆盖。FP64 root finder 不是 interval-certified solver，不得使用未经证明的 certified/global-optimal 宣称。
- 学到的 Q 在实数构造中严格正交；浮点实现报告归一化正交残差和实际 FP 输出偏差。不得强称逐 bit 等价。
- 目标与预算仅保证冻结面板，不保证 PPL、held-out 或全局最优。固定 clipping 可扩展，自适应 clipping/GPTQ 必须另做迁移实验。

## 3. SpinQuant 约束

- 审计基线 commit：`8f47aa3f00e8662caf1a484153920a07e5281c3a`，仓库 `https://github.com/facebookresearch/SpinQuant`。
- `third_party/SpinQuant` 保持上游原貌；小补丁放 `patches/` 并记录 hash，适配代码放自己的模块。先检查真实函数再调用，不编造官方接口。
- 特别检查 `utils/fuse_norm_utils.py` 的 embedding 行中心化：对 LLaMA RMSNorm 严格等价实验须关闭该中心化、保留 gamma 融合，并对自己复现的所有基线一致处理。
- 导出所有 R2 键；不能只存 R1。主方法及归因对照固定同一 R2；官方 `SpinQuant_full` 另保留其 R1/R2 学习，不把弱化版本当完整官方方法。
- 官方默认旋转训练 W16A4KV4、之后 GPTQ W4A4KV4；RTN surrogate 收益不能直接宣称 GPTQ 收益。
- 若Blackwell上FP64训练过慢，可做受控 SpinQuant_full_FP32_port：保持R1/R2、CE、数据和优化器，验证数值/质量并明确标our FP32 implementation。效率结论必须对精度合格FP32版/近期方法，不只对最慢旧FP64版。
- `ptq.py` 常规路径是反量化浮点模拟；不能据此报告 packed INT4 加速、压缩权重内存或持久 KV cache 节省。
- C4 与六任务需要适配；禁止从论文抄数填进 reproduced 表。

## 4. 编码与 Notebook 规范

- 使用 `src/cellquant/` 包；notebook 是配置、调度、检查和可视化入口，不复制核心算法。
- 只维护一套逻辑，通过 JSON 配置切换 Colab/AutoDL。路径来自配置或环境变量，不硬编码用户 Drive 文件夹。
- 默认 notebook CPU smoke / manifest-only，不下载大型模型、不自动启动完整 sweep。用户明确运行相应真实阶段后继续执行已授权工作，不反复询问是否继续。
- `RUN_REAL_MODEL` / `RUN_FULL` 是执行配置，不能替代 G1/G2 科学证据。全量后端缺失、G3 gate 缺失或数据不足时必须明确阻塞，不能改门槛让任务显示成功。
- 使用宿主已有 PyTorch 做探测；不要在 Colab 第一格盲目升级/降级 torch。RTX PRO 6000 Blackwell 的架构支持、FHT、FlashAttention/Triton 分别实测。
- T4 不支持的 BF16 不能静默降级后写 BF16 结果；记录实际 FP16。跨精度结果分开，迁移后同精度重跑基线。
- 模型/tokenizer/dataset/revision/校准 token hash/量化配置/代码 commit/依赖快照均写入 manifest。没有具体 revision 不进入正式结果。
- 不打印 HF token、不将 secrets 写入 notebook 输出或归档。401/403 给准确原因；损坏权重报告完整路径、大小、校验失败，不自动无限重试。
- 校准缓存、模型权重先放运行时本地磁盘；向 Drive 同步小结果、checkpoint 与索引，不直接在挂载盘上做高频随机读写。
- 结果原子写入；每个独立作业有唯一 run_id。断点恢复必须核对配置、代码、模型与数据 hash；环境改变时分叉新 run，保留 provenance。
- 长作业通过可中断 subprocess、日志文件及心跳运行；不要 `for line in stdout` 无限阻塞 notebook。KeyboardInterrupt 要终止子进程并保存状态。
- 不吞异常、不 `except Exception: pass`、不将 SKIPPED 计为 PASS。不要用验证脚本自动写假的成功 JSON。

## 5. 实验与证据

- `source_type` 必须区分 `synthetic_math`、`real_layer`、`real_model`、`published`、`unit_test_fixture`、`not_run`。
- G0 只说明数学实现正确；G1 需要真实层上同一个 global Q 的效果，不能用每层独立挑选的最佳旋转冒充。
- 所有统计在独立 held-out documents/blocks 上评估；校准 seed 与 token/block 不是同一独立单位。PPL 在 NLL/token 加权后取 exp，不能平均各块 PPL。
- 相同 tokenization、harness commit、任务 metric、量化粒度、GPTQ、R2/R3/R4、dtype 下比较。时间用同一 GPU、相同数据和同步计时。
- 三类对照必须保留：随机 Hadamard、完整 SpinQuant、同目标/同角度参数/同预算的 STE 或 angle grid。若声称领先近期方法，还要有能复现的 2026 强基线。
- 下游六任务固定 ARC-C/ARC-E/HellaSwag/LAMBADA/PIQA/WinoGrande 的具体 task id 与 metric key，主表冻结后不得择优切换 acc/acc_norm。
- 置信区间用 paired document/block bootstrap；不要把相关 token 当独立样本。任务结果保留逐题输出，论文正式实验至少 3 个校准种子。
- `configs/gate_rules.json` 是机器门槛；文档数值必须一致。修改门槛需记录原因和时间，不能在看过测试结果后追改为通过。
- 资源不足与方法无效是不同结果；`INSUFFICIENT_EVIDENCE` 不是 FAIL，也不是 PASS。
- 论文结果只从实际日志导出，图表不能预填。论文门槛是项目投入判断，绝非顶会录用保证。

## 6. 新颖性与研究范围

最近先例：ParoQuant、TORQ、PiSO、OBCD、CROQuant；另核对 ThinQuant、LIRQuant、HARP、SpectrumQuant、WUSH、ConQuR，以及 2026-11 新工作。搜索不足时保留不确定性。

不能宣称 first Givens、first exact angle search、first gradient-free rotation、first bilateral output reconstruction、first radius pruning。贡献必须落到动态双侧量化轨道的代数刻画、无损筛选与增量复杂度、真实模型的匹配成本收益。

## 7. 执行流程与完成定义

按 `prompts/00_MASTER_BUILD.md` 分里程碑执行。每个里程碑都应交付可运行代码、实际验证、更新 `docs/IMPLEMENTATION_STATUS.md` 与 `docs/WORK_LOG.md`。先做当前阶段完整可审查的工作，不以“我可以继续”结束。

常用 CPU 验证：

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/run_math_checks.py --output runs/math_reference
python -m cellquant.gates evaluate --stage G1 --evidence examples/G1_evidence_NOT_RUN.json --output runs/gates/G1.json
```

最后一条在模板为空时应返回 INSUFFICIENT_EVIDENCE，这正是正确行为。GPU 验证未运行时，最终报告必须写清未运行原因与所需命令，不能将 CPU pass 扩展成 CUDA/7B pass。
