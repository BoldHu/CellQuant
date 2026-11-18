# G1 下一轮试验：按各位置的起始误差拟合共享旋转

上轮 LLaMA-2-7B 在 A100 上的正式 G1 为 **FAIL**。CellQuant 相对 Hadamard 的位置中位 NMSE 降幅为 2.569%，相对逐种子、逐位置最强匹配搜索为 0.0172%，后者的配对文档 95% 区间为 [−0.2054%，+0.0482%]。校准用时约为角度网格的 3.79 倍。后续拟合诊断中，改配对只在一个种子上扩大 CellQuant 对网格的优势，依原协议结束这条尝试。[上轮文档区间](../evidence/colab_a100_g1_20261118/document_intervals.json) · [配对诊断](../evidence/g1_pair_diagnostic_20261118/a100_summary.json)

## 本轮唯一的方法改动

老目标对各位置的 NMSE 求和。以旧验证集的 Hadamard 为例，两个种子上最大和最小的位置 NMSE 分别相差约 35 倍、34 倍。高误差位置的同一相对降幅对旧目标贡献更大，正式 G1 却按位置的相对降幅取中位数。由此提出可检验的假设：让拟合目标对各位置的相对降幅更敏感，事件角度搜索可能在分开的文档上获得比角度网格更大的收益。

每个种子仅在其**拟合面板**上，用同一个起始 Hadamard 旋转测量各位置的初始 NMSE `h_p`。新目标将原位置项乘以 `c / max(h_p, 1e-12)`。`c` 对所有位置相同，使起点的联合损失与原值相等。权重在整个求解过程中冻结；每个方法都记录起始误差和乘数。这个目标近似平均相对改善，正式评分仍使用原始 NMSE。每个种子拟合一个全局 Q，量化组、Givens 坐标表、角度范围、权重和激活的风险上限及基线预算沿用原配置。

## 在原 01 所用的 Colab A100 上运行

使用已经准备好 `runs/prepared/g1_gpu/config.json` 和相应模型缓存的 checkout。打开原 01，在 Colab 文件栏把本轮 [更新包](../artifacts/CellQuant-G1-objective-20261118.zip) 上传到 `/content`。在一个新代码单元执行以下内容。两个开关一起开启时，短测达到下面三项条件才继续正式 G1。只想先看短测时，省略第二个环境变量即可。

```python
%cd /content/CellQuant
!python -m zipfile -e /content/CellQuant-G1-objective-20261118.zip .
%env CELLQUANT_SCREEN_OBJECTIVE=1
%env CELLQUANT_CONFIRM_OBJECTIVE=1
%run notebooks/01c_Colab_Objective_Trial.ipynb
```

若 VM 已断开，先按[原 A100 流程](COLAB_A100_TEST.md)恢复同一版本的权重、原始冻结配置和 token 清单。自定义准备目录可用 `CELLQUANT_PILOT_CONFIG` 指定。01c 默认空运行。各阶段会打印日志和小结果 ZIP 的路径，结果保存于 `runs/01_pilot/objective_*/` 和独立的实验 run 目录。断开 VM 前应保存打印的结果包，迁移到 G2 还需保存确认输入清单、协议和正式 run 中的 Q。

### 第一步：32 对短测

- 使用每个种子原有 8 份拟合文档，按冻结清单顺序前 4 份用于求解，另 4 份用于这次短测的验证。每个位置在两侧分别采样 128 和 256 个 token，均来自长度 2048 的上下文。上轮正式验证文档和最终测试集均不参与。本次验证来自以前使用过的校准材料，适合决定下一笔算力投入。
- 在同一 A100、两个种子和九个位置上测四组：旧目标 CellQuant、旧目标 129 点网格、新目标 CellQuant、新目标 129 点网格。另测共享 Hadamard 起点。每组依次执行原随机表前 32 对坐标，用最终合成的 Q 重新计算验证误差。
- 预先约定三个投入条件。每个种子和位置取两种网格中误差较小者为对照。新 CellQuant 对该对照的跨位置中位改善至少 **1%**（位置内先取两种子平均）。两个种子各自的跨位置中位改善均大于零。新 CellQuant 相对旧 CellQuant 的跨位置中位改善也大于零。
- 满足三项时输出 `TEST_ONE_G1`。完整运行而未达到条件输出 `STOP`，结束本次改法。数据缺失、预算中断或运行失败应单看状态和日志。按上轮同卡计时估算短测约 **15–25 分钟**，假设权重和输入已缓存。

### 第二步：仅在短测通过后做一次正式 G1

每个种子用原 8 份完整拟合文档和 128 对坐标重新训练，在 WikiText 训练划分中另外抽取 8 份此前没见过的文档。两种子验证集相互分开；程序按文档 ID、完整文档内容哈希和 token 块哈希排除旧两种子的拟合和验证材料。模型、tokenizer 和数据的不可变 revision 及新清单写入新的冻结协议。

正式比较包含新目标的 CellQuant、Hadamard、STE、129 点网格、固定尺度搜索，以及单独从起点重跑的旧目标 129 点网格。根目录的 `G1.json` 按原五种方法评分。`including_original_grid/G1.json` 还把旧网格计入逐种子逐位置的最强对照及最快校准时间。按上轮正式 G1 约 77 分钟加上额外网格的同卡计时，预留 **80–100 分钟**，取决于实际事件数和初始化。

若先只跑了短测并拿到 `TEST_ONE_G1`，可以直接使用刚打印的短测 run 目录，无需重复短测。从 checkout 根目录运行以下命令（用实际路径替换占位符，输出目录须为新的目录）。

```bash
python scripts/prepare_experiment.py --config runs/prepared/g1_gpu/config.json --relative-g1-screen <短测run目录> --output <新建确认输入目录>
python -m cellquant.experiment --config <新建确认输入目录>/config.json
```

两个 G1 文件都须达到原机器门槛。准确率路线要求相对 Hadamard 至少改善 10%、相对最强搜索至少改善 5%，并通过位置及种子一致性检查。效率路线要求相对 Hadamard 至少改善 5%、中位退化不超过 2%、同卡校准至少加速 2 倍，并通过位置检查。短测的 1% 只决定是否值得做这次复验。[完整门槛](GATES_AND_PAPER_CRITERIA.md) · [机器规则](../configs/gate_rules.json)

正式 run 完整时，可以从两套报告各自的目录计算 10,000 次按文档配对的区间。每次重抽样都重新选取该报告中的最强对照。

```bash
python -m cellquant.pilot_bootstrap --run-dir <正式run目录>/including_original_grid --output <正式run目录>/including_original_grid/document_intervals.json
```

### 第三步：根据正式结果投入

- 两份 G1 **PASS**，进入 96GB 目标机上的小规模 G2。向 G2 提供 `including_original_grid/G1_evidence.json`，在实验矩阵的 `solver.fit_objective` 明确设为 `relative_site_nmse`。先验证浮点等价，再比较实际 W4A4KV4 GPTQ、开发集 PPL 和完整复现的 SpinQuant。后续冻结全量矩阵时沿用同一目标。
- 本轮正式 G1 再次 **FAIL**，依项目约定停止该主线。若为 `INSUFFICIENT_EVIDENCE` 或 `RESOURCE_BLOCKED`，保留日志和已经完成的作业，按原资源处置规则诊断。
- G2 和 G3 均通过后再执行全量模型、至少三个校准种子、WikiText-2/C4 和六任务。模型级质量、同卡总成本及 G4 证据决定可以撰写的论文结论。[后续实验与论文标准](GATES_AND_PAPER_CRITERIA.md)
