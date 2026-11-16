# CellQuant 研究实现

研究一个全模型共享的严格正交 R1，在固定分组 INT4 下直接优化冻结面板上的动态双侧量化乘积误差。基础后端是固定版本 Meta SpinQuant；项目名称不代表已证明的新颖性。

2026-11-15 已从 CPU 参考扩展到生产事件求解器、匹配对照、真实模型实验入口、SpinQuant/GPTQ 适配、数据与评估流程。**当前通过的是数学与小模型结构验证；真实 LLaMA-2 7B / LLaMA-3 8B 的 G1–G4 尚无合格证据。**

- 已复现原有 62 项测试；2026-11-15 构建 116 项通过，新增工作流后全套 125 项通过，包含本机可执行的 CUDA 检查。
- 合成 G0：3 seeds、24 cases，真实计算为 PASS；完整 d=4096、FFN=11008 工程 smoke 已运行，使用少 tokens/窄角域。
- 随机小 LLaMA 的 FP 融合/旋转、GPTQ、RTN 恢复、MHA/GQA、R1/R2 训练和恢复已实际执行。
- 五本阶段 notebook 和新增 GitHub/Colab 初始化 notebook 的默认路径均在本地运行成功；不会自动下载模型或启动 sweep。
- 本机只有 RTX 3050 Laptop 4GiB、约 2GiB 可用 CPU RAM；FHT/FlashAttention/Triton 未安装。7B/8B 和 96GB Blackwell 项目保持 NOT_RUN。

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
| `baselines.py`, `capture.py`, `pilot.py` | 同面板 Hadamard/STE/grid/fixed-scale、逐层真实采集与 G1 |
| `spinquant_adapter.py`, `spinquant_train.py` | 上游 overlay、R1=Q.T 与全部 R2、CE/SGDG、GPTQ、恢复 |
| `data.py`, `experiment.py` | 固定 revisions、独立文档划分、权重完整性、统一实验后端 |
| `evaluation.py`, `analysis.py`, `evidence.py` | WikiText/C4 NLL、六任务逐题输出、配对 CI、实测门槛 |
| `runtime.py`, `runner.py`, `migration.py`, `workspace.py` | 日志/心跳/中断、指纹、作业计划、G1 迁移与 Drive 缓存备份 |

Notebook 00 数学；01 真实层 G1；02 迁移/canary/G2；03 G3 后正式矩阵；04 真实日志分析。配置草案位于 `configs/`；`prepare_experiment.py` / `prepare_matrix.py` 生成新的固定协议配置，不能手填虚构 SHA。

求解器是 FP64 数值事件搜索，不是 interval-certified 或全局最优证明；GPU 支持当前限于 dense GEMM/量化及 SpinQuant 路径，尚无验证过的全 GPU 增量事件 kernel。GPTQ 是反量化浮点模拟，不能据此报告 packed INT4 加速、权重压缩内存或持久 KV 节省。

状态与实际证据：[实现状态](docs/IMPLEMENTATION_STATUS.md)、[验证报告](VALIDATION_REPORT.md)、[工作日志](docs/WORK_LOG.md)。科学约束仍由 `AGENTS.md`、数学规范、SpinQuant 规范和未修改的 `configs/gate_rules.json` 管理。近期 2026 强基线、正式消融和新颖性完整复查仍待完成。
