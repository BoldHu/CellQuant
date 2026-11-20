# CellQuant 研究实现

研究一个全模型共享的严格正交 R1，在固定分组 INT4 下直接优化冻结面板上的动态双侧量化乘积误差。基础后端是固定版本 Meta SpinQuant；项目名称不代表已证明的新颖性。

**2026-11-18 的正式 A100 G1 已完成，判定为 FAIL。** LLaMA-2-7B 两个 seed、五种方法共 10 份结果齐全。CellQuant 在独立文档上相对 Hadamard 的跨位置中位 NMSE 降幅为 2.569%，相对逐位置最强匹配搜索为 0.017%。同卡校准速度比为 0.264×。当前停在 G1，G2 和全量实验尚未启动。[正式判定](evidence/colab_a100_g1_20261118/G1.json) · [实测分析](docs/IMPLEMENTATION_STATUS.md#a100-正式-g1) · [实验协议](docs/COLAB_A100_TEST.md)

**7B 新文档 G1 复验已按上界停止，当前不进入全量实验。** 早期 pilot 比 256 对网格改善 0.0812%。另取 16 篇拟合和 32 篇验证文章后，两个种子共 64 个候选批次全部降低输出侧拟合损失，也全部增加九位置联合损失。最终两次旋转均等于网格，独立文档中位改善为 0%，校准速度比上界为 0.9858×。三个追加匹配求解器按协议跳过。[复验原始测量与结论](docs/OUTPUT_MATCHING_STUDY.md#7b-复验结果)

**7B 第 0 层 up 的退化主要集中在拟合用的 64 行。** 在同一批 32 篇验证文章上，相对 256 对角度网格，双侧选角的 W4A4 误差在原 64 行高 13.89%，在全部 11,008 行高 0.267%（文档配对 95% 区间 0.022% 至 0.521%）。单用短网格时，原 64 行已经落后 13.50%，整投影误差差距接近零。原九位置文档 pilot 的判定为 `stop_bilateral_variant`。[7B 全行复算](evidence/output_matching_20261119/bilateral_rows_7b_summary.json) · [九位置结果](evidence/output_matching_20261119/bilateral_documents_7b_summary.json)

**更换拟合行后，7B 网格的优势随之移动。** 在 32 篇新文章的第 0 层 up，B 拟合的网格相对 A 网格在 A 行误差高 32.77%，在 B 行低 28.28%。两组降幅相差 61.05 个百分点 [58.83, 63.36]。两个种子分别复现反转。两组都未拟合的 10,880 行上，变化为 −0.172% [−0.383%, +0.042%]，正数表示改善。研究证据支持固定行上测得的增益可能大幅高估整投影收益。[7B 逐篇复算](evidence/output_matching_20261119/row_crossfit_7b_summary.json) · [42M 与 7B 对照图](evidence/output_matching_20261119/row_crossfit_transfer.png)

**7B 中途换行在未见行上的误差增加了 0.230%。** 两个换行顺序相对各自固定网格的平均 NMSE 降幅为 −0.230% [−0.308%, −0.148%]，整投影为 −0.148% [−0.225%, −0.066%]。原 256 对网格在新一批 32 篇文章上再次各自赢得自己的拟合行，行组间的相对优势相差 62.54 个百分点 [60.71, 64.15]。按事先约定，单次换行策略到此结束。[7B 逐篇复算](evidence/output_matching_20261119/row_refresh_7b_summary.json) · [换行收益与退化](evidence/output_matching_20261119/row_refresh_transfer.png)

下一步回放同一条网格轨迹的短、长两个旋转，在已有的 32 篇文章上完整测量首、中、末层的 `up_proj` 和 `q_proj`。主指标为中层和末层 `up_proj` 的拟合行增益比未拟合行高多少。推送后在原 A100 运行下行，末尾给出 `row_transfer.zip`。[跨层回放协议](docs/OUTPUT_MATCHING_STUDY.md#跨层测量拟合行与未见行)

```bash
cd /content/CellQuant && git pull --ff-only && python scripts/run_output_matching.py --audit-row-transfer
```

16 份验证文档的配对抽样给出相对最强匹配对照的 95% 区间 [−0.2054%，+0.0482%]。后续 A100 拟合诊断已完成。改变配对后，CellQuant 相对网格的额外收益在一个种子上扩大，在另一个种子上缩小。依据事先记录的条件，停止这条配对修正。[文档区间](evidence/colab_a100_g1_20261118/document_intervals.json) · [A100 诊断结果](evidence/g1_pair_diagnostic_20261118/a100_summary.json) · [诊断协议](docs/G1_PAIR_DIAGNOSTIC.md)

01c 的真实 A100 短测已完成，判定为 **STOP**。按各位置起始 NMSE 归一化后，新 CellQuant 相对逐种子、逐位置较强网格的中位改善为 −0.0337%，两个种子的中位改善均为负，低于预设的 1% 投入条件。正式 G1 复验已按条件跳过，本轮到此结束。[01c 实测摘要](evidence/g1_objective_20261118/a100_summary.json) · [短测协议](docs/G1_OBJECTIVE_TRIAL.md)

2026-11-15 已从 CPU 参考扩展到生产事件求解器、匹配对照、真实模型实验入口、SpinQuant/GPTQ 适配、数据与评估流程。当时完成了数学与小模型结构验证。

2026-11-16：Colab G1 默认改为**完整模型驻留 GPU + PyTorch CUDA 数值求解**，包含事件、增量系数、GPU QR 求根与五种匹配方法。采集完所有 seed 后释放基础模型，面板继续留在 GPU。新实例可复用 `runs/prepared/g1_gpu/config.json` 的冻结配置。[GPU 操作指南](docs/COLAB_GPU_FROM_SCRATCH.md)

2026-11-17：用户真实 L4 连续 8 对诊断已完成，总耗时 354.38 秒。CellQuant 平均 20.33 秒/对，128 对仅搜索的条件性外推为 43.36 分钟，超过原 24 分钟方法预算；这不是完整 G1。已继续将动态输入组更新批量化，同卡完整维度合成案例搜索由 49.77 降至 20.10 秒，结果一致；真实 L4 新版尚待重测。在 01 保持 `RUN_PAIR_PROFILE=True`、`PROFILE_PAIR_COUNT=8`，同步新源码后重测原前缀，不修改正式对数、预算或门槛。记录见 [最新实测与工程验证](evidence/input_batch_20261117/)，操作见 [Colab 指南](docs/COLAB_GPU_FROM_SCRATCH.md)。

- 已复现原有 62 项测试；2026-11-15 构建 116 项通过，新增工作流后全套 125 项通过，包含本机可执行的 CUDA 检查。
- 合成 G0：3 seeds、24 cases，真实计算为 PASS；完整 d=4096、FFN=11008 工程 smoke 已运行，使用少 tokens/窄角域。
- 随机小 LLaMA 的 FP 融合/旋转、GPTQ、RTN 恢复、MHA/GQA、R1/R2 训练和恢复已实际执行。
- 五本阶段 notebook 和新增 GitHub/Colab 初始化 notebook 的默认路径均在本地运行成功；不会自动下载模型或启动 sweep。
- 本机 RTX 3050 Laptop 4 GiB 已完成预训练 42M 的固定行、中途换行和跨层回放实测。A100 已完成 7B G1、01d、分批拟合、两次文档 pilot、上界复验、双侧选角拟合、全行诊断、固定行与中途换行复验。7B 的六投影回放和整模型 G2–G4 待实测。

## 开始运行

新的 AutoDL / Linux GPU 服务器按 [新服务器启动说明](docs/NEW_SERVER_START.md) 配置 Remote-SSH、复用宿主 PyTorch、设置本地缓存并启动当前跨层回放。新准备命令支持显式覆盖旧 Colab 缓存路径和允许固定版本首次下载，原数据/旋转/门槛不变。

GitHub → VS Code/Colab → Drive 工作流：先打开 [初始化 notebook](notebooks/00_Colab_GitHub_Workspace.ipynb)，按 [操作说明](docs/COLAB_VSCODE.md) 连接官方 Colab 扩展、挂载 Drive 和 clone 本仓库 `master`。代码在本地编辑后 push，云端 pull 并重启 kernel；模型、数据和运行记录通过带 SHA-256 的版本快照持久化到 Drive，计算时读取 Colab 本地盘。

六本 notebook 均直接包含“挂载 Google Drive”单元。在 Colab 中从上到下运行并完成 Google 授权；本地和 AutoDL 环境默认跳过挂载。

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/run_math_checks.py --output runs/math_reference
python scripts/run_production_checks.py --output runs/production_checks
python -m cellquant.experiment --config configs/colab_pilot.json --preflight
```

Windows 当前项目解释器为 `.venv/Scripts/python.exe`；不要依赖不可用的 WindowsApps `python` 别名。中文环境建议设置 `PYTHONUTF8=1`。

完整命令、依赖、固定 checkout、数据准备、Colab→AutoDL、恢复与门槛流程见 [运行手册](docs/RUNBOOK.md)。首先运行 notebook 00，再准备真实 G1；只有真实 G1/G2/G3 合格后才进入下一阶段。

## 实现入口

| 模块 | 用途 |
|---|---|
| `quantization.py`, `events.py`, `solver.py` | 对称 RTN、完整组事件、输入/输出侧解析系数、增量更新、共享 Q、风险约束 |
| `tensor_math.py`, `tensor_solver.py`, `tensor_cells.py` | PyTorch CUDA 面板、事件/排序、批量输出前缀和、输入增量、移位 QR 数值根、匹配 STE/grid/fixed-scale；CPU 参考用于对照 |
| `baselines.py`, `capture.py`, `pilot.py` | 同面板 Hadamard/STE/grid/fixed-scale、逐层真实采集与 G1 |
| `spinquant_adapter.py`, `spinquant_train.py` | 上游 overlay、R1=Q.T 与全部 R2、CE/SGDG、GPTQ、恢复 |
| `data.py`, `experiment.py` | 固定 revisions、独立文档划分、权重完整性、统一实验后端 |
| `evaluation.py`, `analysis.py`, `evidence.py` | WikiText/C4 NLL、六任务逐题输出、配对 CI、实测门槛 |
| `runtime.py`, `runner.py`, `migration.py`, `workspace.py` | 日志/心跳/中断、指纹、作业计划、G1 迁移与 Drive 缓存备份 |

Notebook 00 数学；01 真实层 G1；02 迁移/canary/G2；03 G3 后正式矩阵；04 真实日志分析。配置草案位于 `configs/`；`prepare_experiment.py` / `prepare_matrix.py` 生成新的固定协议配置，不能手填虚构 SHA。

求解器是 FP64 数值事件搜索，不是 interval-certified 或全局最优证明。生产数值路径现有 PyTorch CUDA 实现，Python 仍负责调度、标量决策、日志与文件 I/O；没有自定义融合 CUDA 事件 kernel，也没有真实 7B 加速结论。GPTQ 是反量化浮点模拟，不能据此报告 packed INT4 加速、权重压缩内存或持久 KV 节省。

状态与实际证据：[实现状态](docs/IMPLEMENTATION_STATUS.md)、[验证报告](VALIDATION_REPORT.md)、[工作日志](docs/WORK_LOG.md)。科学约束仍由 `AGENTS.md`、数学规范、SpinQuant 规范和未修改的 `configs/gate_rules.json` 管理。近期 2026 强基线、正式消融和新颖性完整复查仍待完成。
