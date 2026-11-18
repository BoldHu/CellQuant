# A100 拟合面板坐标诊断

2026-11-18。已完成的 LLaMA-2-7B G1 为 **FAIL**。本次测试的投入问题是，在 128 个 Givens 坐标对的预算下，改变可触及的输出行，是否能扩大 CellQuant 相对 129 点角度网格的实际优势。原始试验的两个校准种子分别覆盖了 4096 行中的 250 行和 245 行。[正式结果](../evidence/colab_a100_g1_20261118/G1.json)

## 在上次的 A100 上运行

1. 将本轮的 `CellQuant-G1-diagnostic-20261118.zip` 上传到 Colab VM 的 `/content`。在原 01 末尾的新单元执行下面四行，计算和打包都在同一台 VM 完成。更新包适用基线为上次 G1 的 checkout `7aab9e3125d1f561d9566717a6b6f0a43d92a28d`。已经通过自己的分支同步本轮代码时，可直接运行后两行。

   ```python
   %cd /content/CellQuant
   !python -m zipfile -e /content/CellQuant-G1-diagnostic-20261118.zip .
   %env CELLQUANT_RUN_PAIR_DIAGNOSTIC=1
   %run notebooks/01b_Colab_Pair_Diagnostic.ipynb
   ```

2. 若直接打开独立的 `notebooks/01b_Colab_Pair_Diagnostic.ipynb`，设置 `RUN_PAIR_DIAGNOSTIC = True` 后运行全部即可。两种入口选一种。默认读取 `runs/prepared/g1_gpu/config.json`，也可沿用 `CELLQUANT_PILOT_CONFIG`。计算发生在新子进程，输出目录和索引按本次调用新建。若换了 VM，可先用 00 恢复权重、冻结输入、G0 和 SpinQuant overlay。[输入准备](COLAB_A100_TEST.md)

3. notebook 末尾显示两个种子的结果和小型 `pair_diagnostic_<run_id>.zip` 路径。下载这个结果包即可复核每个坐标对。中途超时或显存不足时，查看 `result.json` 和两个种子各自保存的部分记录。

在终端中从项目根目录运行的等价命令如下。

```bash
python -m cellquant.experiment --config runs/prepared/g1_gpu/config.json --diagnose-pairs
```

## 实际比较的量

- 每个种子在同一 Hadamard 初值和九个完整分组面板上比较两张共享坐标表。一张是原 128 对随机表，另一张只对全表的坐标编号作一一替换。后者保持对数、重复使用关系、次序和参数个数。它按所有输出侧面板在 fit 上的初始归一化误差排序，平分时按坐标 ID 排序。
- 设输出侧面板 $p$ 的初始重构残差为 $E_p$，浮点目标为 $T_p$。每一行的分数为 $r_i=\sum_p w_p\|E_{p,i}\|_2^2/\|T_p\|_F^2$。若共享旋转只更新行集合 $S$，其他行的权重、逐行量化尺度、目标及量化输出均保持原值。因此这些面板最多能消除 $\sum_{i\in S}r_i$ 的初始目标值。在相同行数下选取得分最高的行会使该上界最大。报告中的 `coverage` 是它占初始输出侧误差的比例。输入侧的实际变化由九面板联合损失直接测量。
- 每张表在 128 对中等距取八个固定位置。网格和事件搜索各从同一个初值独立求解每一对，保留原角域、原双侧量化和风险约束。两种方法在两个种子共作 32 次配对比较，交替运行以减小顺序影响。同时记录排序、搜索和复核耗时。
- 每个入选角度先报告增量计算得到的损失，再显式形成更新后的 Q，逐面板重新做矩阵乘法和量化。原始重算损失、双侧风险以及输入侧和输出侧分别贡献的误差变化都写入记录。近舍入边界时，两种浮点计算顺序可能落入不同的编码。`mean_feasible_replay_gain_pct` 计算每一对在重算后仍满足风险且确实下降时的降幅，其余保留零角。单位为初始九面板联合损失的百分比，`cell_minus_grid_pp` 是两方法降幅相减得到的百分点。八个单对的降幅不能相加预测连续 128 对的效果。

## 如何决定是否再做一次 G1

此处的决策写在查看**新诊断**结果之前。对两个种子分别检查三个方向是否都成立。

1. `coverage.ranked > coverage.original`。
2. `mean_feasible_replay_gain_pct.ranked.cellquant` 高于同一字段中的 `original.cellquant`。
3. 重算后 ranked 的 `cell_minus_grid_pp` 为正，且超过 original 的同一差值。逐对查看原始损失和风险，判断均值是否被单个坐标对主导。

两种子均支持时，安排一次正式修正。四个搜索方法共用同一张仅由 fit 决定的坐标表，并将排序耗时计入每种方法的原墙钟预算。先冻结方法、坐标规则、代码和输入，再在与本次及此前拟合、验证均按文档 ID/哈希隔离的新验证文档上完整运行 128 对和五种方法。继续使用原 G1 门槛。如果仅有覆盖率上升，或网格同样受益而事件搜索的额外收益没有扩大，就到此停止这条配对修正。中断或资源不足应先按日志定位，保留未完成状态。

正式修正通过 G1 才进入有限的 G2 全模型 GPTQ canary。G2 和 G3 均通过后才能启动全量。论文阶段还需要两款模型、至少三个校准种子、同协议的完整 SpinQuant 与可复现近期强基线、WikiText-2/C4 的逐文档 NLL、六任务逐题输出及同卡完整校准成本。[冻结的投入门槛](GATES_AND_PAPER_CRITERIA.md)

## 重算历史文档区间

下面的命令在 CPU 上直接读取 A100 运行的字段摘录。也可将 `--run-dir` 换为包含十份完整 `pilot_result.json` 的原始目录，此时会校验 G1 登记的文件哈希。

```bash
python -m cellquant.pilot_bootstrap --run-dir evidence/colab_a100_g1_20261118 --output runs/review/document_intervals.json
```

按独立文档配对抽样 10,000 次，两次拟合和九个位置固定。每次先汇总平方误差与目标能量，再计算种子内相对收益、跨种子平均及跨位置中位数。逐位置最强对照在每次抽样内重新选。相对最强对照的点估计为 +0.0172%，95% 百分位区间为 [−0.2054%，+0.0482%]。区间描述本次两组已拟合旋转在验证文档上的不确定性。[完整区间](../evidence/colab_a100_g1_20261118/document_intervals.json)
