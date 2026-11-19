# CellQuant 研究实现

研究一个全模型共享的严格正交 R1，在固定分组 INT4 下直接优化冻结面板上的动态双侧量化乘积误差。基础后端是固定版本 Meta SpinQuant；项目名称不代表已证明的新颖性。

**2026-11-18 的正式 A100 G1 已完成，判定为 FAIL。** LLaMA-2-7B 两个 seed、五种方法共 10 份结果齐全。CellQuant 在独立文档上相对 Hadamard 的跨位置中位 NMSE 降幅为 2.569%，相对逐位置最强匹配搜索为 0.017%。同卡校准速度比为 0.264×。当前停在 G1，G2 和全量实验尚未启动。[正式判定](evidence/colab_a100_g1_20261118/G1.json) · [实测分析](docs/IMPLEMENTATION_STATUS.md#a100-正式-g1) · [实验协议](docs/COLAB_A100_TEST.md)

**2026-11-19 的 7B 输出匹配实验 01d 已完成。** 相对 128 对网格，三个输出投影的中位 NMSE 下降 1.55%，种子 11 下降 3.10%，种子 23 持平。从 256 对网格追加同一修正时，两个种子的六次非零步长试探都满足尺度预算，但输入投影增加的误差超过输出收益，最终两次都保留原网格。下一步在 A100 上复用已归档的角度做拟合短测，每 64 对联合检查一次。使用原 G1 准备好的输入，在 Colab 终端运行下行。[实测与下一轮决策](docs/OUTPUT_MATCHING_STUDY.md#7b-实测与下一轮决策) · [7B 实测图](evidence/output_matching_20261119/colab_7b_effects.png)

```bash
cd /content/CellQuant && git pull --ff-only && PYTHONPATH=src python -m cellquant.experiment --config runs/prepared/g1_gpu/config.json --screen-matching-batches evidence/output_matching_20261119/measurements.zip
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
- 本机 RTX 3050 Laptop 4 GiB 已运行 42M 预训练模型及完整输出轴形状检查。A100 已完成 7B 正式 G1 和 01d 真实层实验。分批短测已在 42M 拟合文档上验证，7B 分批结果、整模型 G2–G4 和 96 GB Blackwell 测试待运行。

## 开始运行

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
