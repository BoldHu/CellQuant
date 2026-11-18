# Colab A100 测试与租卡决策

这轮 A100 要回答两个问题。同一预算能否完成全部坐标对，以及一个共享 Q 能否降低独立文档上九个位置的双侧 RTN 乘积误差。前一个问题用连续前缀估时，后一个问题用正式 G1 判断。模型级 GPTQ 的结论留给服务器 canary。

## 1. 在新 A100 上准备同一份实验

1. 把修订后的工作树同步到自己的 Colab checkout。沿用 GitHub 工作流时，先在本地提交所需改动并同步到自己的分支，再让 workspace notebook clone 该分支；已有 checkout 可在没有作业运行时执行 `git pull --ff-only`。重启 kernel，记录 `git rev-parse HEAD`。本地修改不会随 Colab 连接自动传过去。
2. 选择 A100 和 Python 3.12。Colab 的 2026.07 运行时提供 Python 3.12.13。保留宿主 PyTorch，HF 依赖由仓库固定。运行 `nvidia-smi -L`，核实实际分配的 GPU、可见显存与 MIG 情况，并在同一实例上完成所有正式对照。[Colab 运行时说明](https://research.google.com/colaboratory/runtime-version-faq.html)
3. 在 Colab Secrets 中设置已获 LLaMA-2 访问权限的只读 `HF_TOKEN`，授权初始化 notebook 读取。仓库旧提交里出现过明文令牌，先在自己的账号撤销旧令牌，再设置新值。[Hugging Face 令牌管理](https://huggingface.co/docs/hub/security-tokens)
4. 打开 `00_Colab_GitHub_Workspace.ipynb`，设置下列开关，按顺序执行。已有 Drive 快照时设置 `RESTORE_AREAS = ["models", "datasets", "runs"]`；首次运行可保留空列表。让模型、数据和活动结果位于 VM 本地盘。新 A100 创建新作业，复用已校验的权重和冻结 token，保留 L4 的历史 run。

```python
RUN_G0 = True
RUN_GPU_SOLVER_CHECKS = True
PREPARE_G1_INPUTS = True
```

在项目根目录核对代码与实际设备。后续命令均从这个目录运行。

```bash
git rev-parse HEAD
git -C third_party/SpinQuant rev-parse HEAD
python -c "import sys,torch; from cellquant.tensor_solver import CELL_ENGINE_VERSION; print(sys.version,torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),torch.cuda.mem_get_info(),CELL_ENGINE_VERSION)"
python -m pytest -q
python -m cellquant.experiment --config runs/prepared/g1_gpu/config.json --preflight
```

验收时核对固定上游 `8f47aa3f00e8662caf1a484153920a07e5281c3a` 和引擎 `torch_batched_bilateral_prefix_v3`。00 应有本次的 G0 `PASS`、CUDA/FHT `PASS`、tiny SpinQuant/GPTQ `PASS_TINY_STRUCTURE_ONLY`、GPU 驻留检查 `PASS_ENGINEERING_ONLY`、权重 `WEIGHTS_VERIFIED`。预检须读 JSON 中的 `readiness.status=READY`；CLI 的 `PREFLIGHT_ONLY` 及退出码仅表示执行了检查。

核对 `config.json` 和准备目录中的 protocol、token manifests、分片大小/SHA。两次运行使用相同的模型/tokenizer/dataset revisions、fit/dev token 哈希、显式模型 dtype 和 FP64 求解器。若从 L4 恢复的准备配置指向不存在的本地路径，应从冻结 revisions 在新目录重新准备并比对 token 哈希。访问失败时处理账号权限；可用显存不足时查看预检给出的总量、当前空闲量、估计需求与 85% 上限。

### 直接使用本次交付的更新包

如果修订尚未推送，可在步骤 1 将 `CellQuant-A100-update-20261118.zip` 上传到 VM 的 `/content`。下面的命令用于**新 VM 的空目录**，更新包只包含当前版本的改动，notebook 执行输出已清空。已通过 Git 同步修订时跳过本节。

```bash
git clone --no-checkout https://github.com/BoldHu/CellQuant.git /content/CellQuant
cd /content/CellQuant
git switch --detach 1b2caf08e9944b1dc83f2eb5cf96b9111f33234d
python -m zipfile -e /content/CellQuant-A100-update-20261118.zip .
git add -u
git add docs/COLAB_A100_TEST.md evidence/a100_readiness_20261118
git -c user.name=CellQuant -c user.email=cellquant@localhost commit -m "A100 pilot solver and validation"
git rev-parse HEAD
```

这会为该 VM 建立可记录的本地提交。保存提交号和更新包，打开修订后的 workspace notebook，再继续步骤 2–4。实际 GPU 作业会额外写入源码和环境指纹。

## 2. 先测真实 8 对

在同一台 VM 打开 `01_Colab_Real_Layer_Pilot.ipynb`。设置 `RUN_REAL_MODEL=True`、`RUN_PAIR_PROFILE=True`、`PROFILE_PAIR_COUNT=8`，执行到本次运行摘要。终端方式等价，选一种入口执行一次即可。shell 中的 `CFG`、`OUT` 以下沿用。

```bash
CFG=runs/prepared/g1_gpu/config.json
OUT=$(mktemp -d runs/a100-test-XXXXXXXX)
python -m cellquant.experiment --config "$CFG" --profile-pairs 8 --result-index "$OUT/profile.json"
```

如需观察 GPU 忙闲，在**同一 VM 的另一个终端**先执行下列采样命令，把路径设为本次实际的 `OUT` 目录；实验完成后按 Ctrl+C。时间戳可与作业日志对齐。利用率代表采样窗口内执行内核的时间比例，显存占用反映数据驻留。阶段墙钟使用设备同步计时。[NVIDIA 采样字段](https://docs.nvidia.com/deploy/nvidia-smi/index.html) · [PyTorch CUDA 计时](https://docs.pytorch.org/docs/2.11/notes/cuda.html#asynchronous-execution)

```bash
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 1
```

要保存采样，可在上述命令末尾加 `-f "$OUT/gpu.csv"`，在第二个终端先把 `OUT` 设为相同的实际目录。若需要区分主机提交间隙和短内核本身的成本，可在独立诊断 run 采集 CUDA 时间线；正式 G1 用正常运行的计时。

在本次 run 的 `model_placement.json` 核对所有参数位于 CUDA、`cpu_offload=false`。前缀采集 seed 11 的九个 fit 面板后释放基础模型。记录 `result.json` 的峰值显存、`timings.json` 的 load/fusion/capture，以及两个 `pair_profile/seed_11/*/profile_result.json` 中的逐对信息。

| 查看项 | 如何决定 |
|---|---|
| 两方法的状态与对数 | 固定尺度和 CellQuant 都应为 `COMPLETE`、`completed_pairs=8`，同一 seed 原顺序连续更新 Q。若有阻塞，读 `solver.reason`、`resource_diagnostics` 和最后完成的位置。 |
| 逐对的成本 | 对齐事件数、区间数、`timings` 中的事件生成、排序、系数/边界、候选搜索和最终复核。报告每对的时间、均值、最慢对以及接受数。事件数超过 200,000 或单对搜索超过 300 秒会阻塞。 |
| 原方法预算 | 128 对共 1,440 秒，平均最多 11.25 秒/对，初始化、更新和存盘还占用该预算。`projected_full_search_seconds` 只外推搜索。新版 `projected_method_seconds` 计一次固定开销，再按前缀均值外推含更新、写 checkpoint 的循环。`projected_method_headroom_seconds` 为预算余量。 |
| 是否投入正式 G1 | 两种方法均完成，含开销外推有正余量，且慢对没有显示持续超限趋势时，再投入正式试验。若超限，先用最慢对的阶段记录定位。这个工程判断没有新增科学门槛。后续坐标的事件分布仍可能变化。 |

L4 的历史数值和本地合成测量只用于定位问题。若要专门量化代码在 A100 上的加速，应在**同一 A100** 用同一冻结输入、配置和原定 8 对，分别运行旧、新代码的独立作业，记录源码与环境指纹。旧提交包含令牌，比较求解器时可仅提取相应版本的 `src/cellquant/` 到独立 checkout，使用已修订的凭据入口；两个版本都用各自的 `solver.seconds` 比较求解时间，整轮采集工作量已发生变化。

## 3. 用独立 held-out 正式判断 G1

在 01 将 `RUN_PAIR_PROFILE=False`，保留 `RUN_REAL_MODEL=True`，重跑配置和执行单元，再运行 G1 评估单元。终端等价命令如下，`--require-pass` 在科研门槛未通过时返回非零状态。

```bash
python -m cellquant.experiment --config "$CFG" --result-index "$OUT/formal.json"
RUN=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['run_dir'])" "$OUT/formal.json")
python -m cellquant.gates evaluate --stage G1 --evidence "$RUN/G1_evidence.json" --output "$OUT/G1-recomputed.json" --require-pass
python -m cellquant.workspace backup --areas runs
```

正式配置为 LLaMA-2-7B，seed 11/23，各自从独立的 8 条 fit 和 8 条 dev 文档序列采集 2048-token 上下文；每面板分别抽 128/256 个 token。位置为层 0、15、31 的 q/up/down，共九处。q/up 抽 64 个输出行，down 保留完整 hidden×FFN。每种方法每 seed 共用一份全局 Q；Hadamard 为初始对照，CellQuant、STE、grid、固定尺度四种搜索使用相同的 128 对和方法预算。保留实际 Q 的归一化正交残差、每个坐标的损失/尺度约束与逐文档记录。

验收文件是每个 seed 和方法的 `pilot_result.json`、`G1_distributions.json`、`G1_evidence.json` 和重算的 `G1.json`。两个 seed 的 pooled held-out 都要较 Hadamard 改善，任一位置对最强匹配对照的退化至多 10%。还须满足以下一条路线以及机器规则的全部共同条件。[机器门槛](../configs/gate_rules.json)

| G1 路线 | 预注册要求 |
|---|---|
| 准确率 | held-out NMSE 相对 Hadamard 的跨位置中位改善 ≥10%，相对最强匹配搜索 ≥5%，至少 6/9 位置胜出。 |
| 效率 | 相对 Hadamard 中位改善 ≥5%，相对最强搜索中位退化 ≤2%，至少 6/9 位置在 2% 以内，同卡含采集的校准加速 ≥2×。 |

资源中断时保存 `result.json`、日志、checkpoint 与当时的配置。完整但指标未达标时按 `FAIL` 记录；缺真实数据或未完成比较时按 `INSUFFICIENT_EVIDENCE` 记录。两种情形均应先审查当前阶段的证据。

## 4. 何时租服务器

**G1 PASS 才值得为方法验证租单机做 G2 canary。** 用 `python -m cellquant.migration export --run-dir "$RUN" --output "$OUT/g1_transfer"` 导出合格 G1。在目标机导入证据，执行 notebook 02，核实实际 GPU 型号、可见显存、CUDA GEMM 和目标维度 FHT。新准备的 canary 矩阵应冻结 `solver.backend=torch`、`device=cuda:0`、`compute_dtype=float64`，CellQuant 的 `solver.json` 应记录实际数值设备。原来冻结的 NumPy 配置可用作历史对照。[迁移和 canary 命令](RUNBOOK.md)

G2 在同一目标 GPU 重跑 LLaMA-2-7B 的 Hadamard、完整 R1+R2 SpinQuant 和 CellQuant，两个 seed、128×2048 校准，使用真实 GPTQ W4A4KV4 和独立 dev NLL/PPL。先检查 FP32 logits 相对 RMS 和平均 NLL 绝对差均 ≤1e-4。准确率路线要求相对 Hadamard 的 dev PPL 改善 ≥2%，各 seed 相对 SpinQuant 退化 ≤0.5%。效率路线要求各 seed 退化 ≤1%，同卡完整校准加速 ≥2×。总时间包括采集、旋转求解、融合和 GPTQ。

**G2 和 G3 都通过后再启动全量。** G3 还需目标卡实测可见显存 ≥80 GiB、校准峰值占比 ≤85%、真实中断恢复一致、哈希核验、冻结协议及基于 canary 实测的 GPU 小时计划。届时才运行 notebook 03 的两模型三种子、WikiText-2/C4 和六任务。A100 的 8 对耗时和局部 RTN 改善都不能给出这笔服务器开销的回报估计。[阶段验收](GATES_AND_PAPER_CRITERIA.md)
