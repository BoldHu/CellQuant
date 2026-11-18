# 新 Colab 服务器：GPU 驻留 G1，从环境到测试

A100 的同卡计时、正式 G1 验收和租卡条件见[本轮测试方案](COLAB_A100_TEST.md)。

本流程使用 `00_Colab_GitHub_Workspace.ipynb` 和 `01_Colab_Real_Layer_Pilot.ipynb`。旧 CPU 实验保留；新协议保存到 `runs/prepared/g1_gpu/`。不要恢复旧优化器继续跑，也不要覆盖旧准备配置。CPU 数学参考仍用于验证；真实 G1 使用 PyTorch CUDA 后端。

## 0. 先把本地修改同步到 GitHub

本次代码不会因为新建 Colab 服务器而自动出现在 GitHub。先在 VS Code 审查并提交本轮核心代码、配置、脚本、测试、说明和两本 notebook，再 push 到 `BoldHu/CellQuant` 的 `master`。

初始化 notebook 从 Colab Secrets 或进程环境读取 `HF_TOKEN`。提交前检查暂存文件与 notebook 输出。缓存、模型权重和活动 `runs/` 位于 Git 工作集之外。

```bash
git status --short
# 在 VS Code 源代码管理中审查并暂存本轮文件，提交后：
git push origin master
```

## 1. 选择服务器与 Python

使用你之前验证过的 L4 或显存更大的 GPU，选择 **Runtime version 2026.07 / Python 3.12**。Colab 官方当前仍列出该运行时的 Python 3.12.13：[运行时版本说明](https://research.google.com/colaboratory/runtime-version-faq.html)。本项目固定 Transformers 4.44.2/tokenizers 0.19.1；不要在 Python 3.13 中重复尝试安装这套依赖。

VS Code 连接新 Colab 服务器后，在 workspace notebook 顶部确认 `sys.version`。如果扩展没有显示运行时版本选择，用浏览器 Colab 的运行时设置选择版本后，使用可连接该运行时的入口；最终以 kernel 实际打印的版本、GPU 型号为准。

不要先升级或降级 torch；使用该运行时已有的 PyTorch。GPU 模式会检查剩余显存，不够就停止，不会退回 CPU。完整模型和面板的实际峰值仍需本次实测。

## 2. 打开 workspace notebook，填写一次配置

打开更新后的 `notebooks/00_Colab_GitHub_Workspace.ipynb`。顶部设置：

```python
RESTORE_AREAS = ["models", "datasets", "runs"]
RESTORE_RUNS_CONFLICT = "separate"
RUN_G0 = True
RUN_GPU_SOLVER_CHECKS = True
PREPARE_G1_INPUTS = True
BACKUP_AREAS = []
```

沿用你已有的 `DRIVE_ROOT`；如自定义路径，就填写同一个 Drive 目录。新服务器从上到下运行：

1. 配置与 Python 版本检查。
2. `drive.mount("/content/drive")`，按弹窗授权。
3. 从 GitHub clone 本项目，打印 commit；确认已包含本次 GPU 改动。
4. 初始化工作区，恢复模型、数据和 runs。模型文件复制到运行时本地缓存，计算不直接读取 Drive 挂载盘。首次没有快照时 `NOT_FOUND` 是未找到备份，不算验证通过；后续下载即可。
5. 安装基础依赖，再安装 `requirements/hf_optional.txt` 的固定 HF 依赖，保留已有 torch。
6. 运行 CPU 数学与生产参考检查，要求实际 G0 为 PASS。
7. 固定 SpinQuant commit，生成独立 overlay；安装固定 FHT commit；执行 CUDA/FHT 和 tiny SpinQuant/GPTQ 检查。
8. 新增“GPU 驻留与求解器检查”：小随机模型全部参数在 CUDA、采集层输入在 CUDA、五方法在 CUDA，以及完整 `4096×11008` 输出侧矩阵检查。
9. HF 登录并执行真实权限检查。401/403 需要解决账号/token/模型访问权限，不会自动绕过。
10. 准备新的 `g1_gpu` 配置、校准/held-out 文档与固定 revisions，检查并校验权重，再向 Drive 备份模型、数据和准备记录。

同一个服务器继续使用即可。安装脚本通过独立子进程运行，模型与求解检查也使用新进程；若 notebook 内已经导入了刚被替换的依赖，应重启 kernel 并重跑配置/工作区初始化，保留服务器与本地缓存。

这一步应该看到的状态：

- CPU 数学 G0：PASS。
- CUDA/FHT：PASS。
- SpinQuant/GPTQ：PASS_TINY_STRUCTURE_ONLY。
- GPU 驻留检查：PASS_ENGINEERING_ONLY，`all_model_parameters_cuda=True`，层输入设备为 `cuda:0`。
- 真实输入预检查：READY；权重完整性 WEIGHTS_VERIFIED。

前四项是工程检查，不是预训练 7B 的 G1 结果。没有成功报告时先看对应日志，不跳过检查。

## 3. 在同一服务器运行 01

打开更新后的 `notebooks/01_Colab_Real_Layer_Pilot.ipynb`，从路径、挂载、工作区初始化开始运行。设置 `RUN_REAL_MODEL=True`。预检查打印的路径应该是：

```text
/content/CellQuant/runs/prepared/g1_gpu/config.json
```

该文件必须包含：

```json
{
  "runtime": {"model_device": "cuda:0", "cpu_offload": false},
  "solver": {"backend": "torch", "device": "cuda:0", "compute_dtype": "float64"}
}
```

这是字段核对示意，不是完整配置；不要用这几行替换真实配置。代码会拒绝旧 CPU 配置。若设置过 `CELLQUANT_PILOT_CONFIG` 指向旧目录，需要清除该环境变量或指向新配置。

运行真实 G1 单元后，日志先给出唯一 `run_dir` 和 PID。加载完成会打印 `model_placement`，并写入 `model_placement.json`：所有模型参数必须在 CUDA。之后依次显示融合、fit/held-out、seed、层号、方法、面板和坐标对。每 30 秒心跳记录 CPU RSS、可用 RAM、CUDA allocated/reserved。

完成本次所需面板采集后，`g1.model_released` 会记录采集的 seeds 和 splits。前缀仅使用首个 seed 的 fit 面板；正式 G1 使用每个 seed 的 fit 和 held-out 面板。此时基础模型释放，共享 Q、面板和求解状态继续保留在 GPU。

结果指针文件把 notebook 绑定到本次唯一 run；后面的 G1 评估单元自动读取该 run 的真实 `G1_evidence.json`，不再默认评估空模板。运行开关、工程 PASS 和进程退出码不替代 G1 科学门槛。

### 若事件方法在第一对坐标超时

2026-11-17 的真实 L4 运行中，两个 seed 的 STE/grid 都完成了 128 对；CellQuant 与固定尺度搜索均在第一对触发 `Pair wall-clock budget exceeded: 300s`，完成对数为 0。该轮结果为 RESOURCE_BLOCKED，不能当作完整 G1、方法质量 FAIL 或 G1 PASS。顶层 `reason:null` 是汇总遗漏；新版本会将每个方法的实际原因写入顶层结果、日志和 notebook 报错。

先保存该轮结果，在项目根目录执行已有备份命令：

```bash
python -m cellquant.workspace backup --areas runs
```

同步新代码后，在同一服务器重启 kernel 并重跑 01 初始化/预检查；保留缓存、原配置与旧 run。在 01 的预检查单元设定：

```python
RUN_REAL_MODEL = True
RUN_PAIR_PROFILE = True
```

然后运行原执行单元。这个诊断入口复用正式模型加载、融合和完整真实面板采集，只对第一个 seed 的第一个原定坐标对运行固定尺度搜索与 CellQuant。角域、完整通道、FP64、事件 cap、单对 300 秒及同方法时间预算均按原配置保留；不重跑完整 STE/grid，不读 held-out 成绩，不产生 G1 证据。结果是 `PROFILE_COMPLETE` 或明确的阻塞状态，单对完成也不证明 128 对能在整个方法预算内完成。日志中的 `search_phase`、事件数、`cells_completed/cell_count` 用于定位吞吐瓶颈。

命令行等价入口为：

```bash
python -u -m cellquant.experiment --config runs/prepared/g1_gpu/config.json --profile-pair
```

诊断报告保存在本次新 run 的 `pair_profile.json`，每个方法的明细在 `pair_profile/seed_11/<method>/profile_result.json`；它们不会被 G1 聚合器接受。若仍超时，应根据阶段和事件吞吐优化实现，不能仅把 300 秒调大、缩小角域或删除对照。明确能执行原预算的完整方案后，再设置 `RUN_PAIR_PROFILE=False` 做新的正式运行。代码变化后不强行恢复旧优化器断点或把旧基线混为新协议结果。

2026-11-17 的后续修正新增 `tensor_cells.py`：输出组编码增量用 GPU 前缀和处理，稳定/固定尺度的输入使用精确秩二更新；不稳定输入完整行的边界采用缓存乘积与编码差更新。候选仍在所有面板合并的目标上选择。保持原有 FP64、角度、事件 cap、300 秒单对和方法总预算。

将本地源码提交并 push 后，在 Colab 项目终端运行 `git pull --ff-only`，重启**同一服务器的 kernel**，重跑 01 初始化与预检查即可，无需重新分配服务器或下载缓存。可在 01 执行前确认实际加载了新引擎：

```python
from cellquant.tensor_solver import CELL_ENGINE_VERSION
print(CELL_ENGINE_VERSION)
assert CELL_ENGINE_VERSION == "torch_batched_output_prefix_v1"
```

继续保持 `RUN_REAL_MODEL=True`、`RUN_PAIR_PROFILE=True`，用原执行单元启动新 run。日志将显示 `batched_output_coefficients`、`batched_constant_scale_input`、`input_coefficients`、`batched_candidates`。`local_cell_index/local_cell_count` 表示当前面板的系数进度，`cells_completed/cell_count` 表示合并后的候选搜索进度；前者运行时后者可能仍为零。最新本地检查见 `evidence/batched_cells_20261117/`。2026-11-17 用户新版 L4 首对已完成（下文），完整 128 对仍未验证；不要把合成案例或单对完成当成完整 G1 合格。

### 首对完成后：连续 8 对的预算诊断

用户 run `2026-11-17T114339.123212_0000_991980b138` 已返回 `PROFILE_COMPLETE`，总耗时 140.93 秒；固定尺度与 CellQuant 首对均完成。CellQuant 在 19.76 秒处理完 151940 个区间并进入最终复核。这是运行可行性的进展，未运行正式两种子五方法的完整拟合或 held-out 评估，01 显示 G1 `NOT_RUN` 正确，尚不能进入 02。

该次日志的有效方法总预算为 **1440 秒**，正式配置仍为 128 对。仅按首对搜索速度外推约 42.16 分钟，超过 24 分钟预算，而且尚未包含最终复核、接受更新和 checkpoint 等开销。该外推不保证后续每对同速；下一步测前 8 对的连续行为，不改正式 pair_budget 或预算。

同步本次源码与 01 notebook 后，在同一服务器重启 kernel，重跑初始化/预检查，在 01 配置单元设置：

```python
RUN_REAL_MODEL = True
RUN_PAIR_PROFILE = True
PROFILE_PAIR_COUNT = 8
```

运行原执行单元，再运行最后的“当前运行”摘要。等价 CLI：

```bash
python -u -m cellquant.experiment --config runs/prepared/g1_gpu/config.json --profile-pairs 8
```

这会对第一个 seed、两个事件方法执行原 schedule 的前 8 对，Q 连续更新，保留正式 128 对配置、每对 300 秒和每方法原 1440 秒上限。`--profile-pair` 仍兼容旧的首对诊断。报告保存 `profiled_pairs`、实际完成数、每对搜索时间和 `budget_analysis`；新执行使用独立 run，不能恢复旧首对 checkpoint 冒充连续原协议。

`PROJECTED_SEARCH_EXCEEDS_METHOD_BUDGET` 表示按这段前缀均速外推的搜索部分超预算；`SEARCH_COMPONENT_ESTIMATE_WITHIN_BUDGET` 也不证明总方法能完成，因为尚有其他开销和后续坐标差异。两者均不授予 G1。正式 G1 完整执行并通过门槛后，才进入 02 的 AutoDL canary。

原附加诊断单元曾写死 08:10 的旧 run，现在统一读取 `PILOT_RUN_DIR`。末尾摘要应打印本次目录；不要用旧目录下的 `RESOURCE_BLOCKED` 判断新运行。

### 8 对已完成：动态输入批处理后的复测

最新用户 run `2026-11-17T120623.229443_0000_083c7593b5` 返回 `PROFILE_COMPLETE`，耗时 354.38 秒。两个方法均完成 8 对；固定尺度搜索均值 8.74 秒，CellQuant 为 20.33 秒。CellQuant 的 128 对搜索部分外推为 2601.85 秒（43.36 分钟），超过 1440 秒原方法预算，不能直接进入 02 或承诺完整 128 对可完成。

已定位并修改动态输入组的逐事件调度瓶颈。新版本 `torch_batched_bilateral_prefix_v2` 保留全部通道、事件、双侧交叉项、FP64 和原预算；不稳定完整行分组使用批量编码差前缀投影，其他分组保留原增量算法。本地完整维度合成案例有实测改善，真实 L4 尚未执行新版。

下一步仍在当前服务器的 **01**：

1. 在本地提交/push 本轮源码，在 Colab 项目目录执行 `git pull --ff-only`。本轮 Codex 没有自动提交或 push。
2. 重启同一个 kernel，重新运行 01 初始化和预检查。无需释放服务器、重建环境或重新下载模型。
3. 保持 `RUN_REAL_MODEL=True`、`RUN_PAIR_PROFILE=True`、`PROFILE_PAIR_COUNT=8`，运行原执行单元和“当前运行”摘要。日志中动态组应出现 `batched_dynamic_input`；完整行分组以外的输入仍可能显示 `input_coefficients`。这次重测是验证新算法实现对已知瓶颈的作用，不是增加到 16/32 对来重复诊断。
4. 只有新记录的搜索时间与初始化/接受更新等开销留有预算余量，才将 `RUN_PAIR_PROFILE=False`，在同一本 01 发起新的正式 G1；原配置仍为两种子、五方法、128 对。前缀外推不保证后续坐标同速，正式结果决定是否完成。
5. 正式运行后运行 G1 门槛单元，再用已有 Drive 备份单元保存结果。仅真实 `G1=PASS` 后进入 02 的 AutoDL canary；`PROFILE_COMPLETE`、`COMPLETE` 和进程退出码 0 都不能替代 G1。

## 4. 长作业诊断、结束和备份

若预检查报告 `GPU-resident capture estimates ... insufficient`，先区分总容量、配置预算与当前空闲量。新版本将它们写入 `preflight["readiness"]["gpu_memory"]` 和错误文本。`currently_free` 表示当前空闲不足；`configured_budget` 表示预算上限不足；`total_capacity` 表示整卡容量不足。这个估计不是实测峰值，也不是 G1 成绩。

2026-11-17 用户报告的 L4 总容量为 22.03 GiB、85% 预算为 18.73 GiB、当前空闲仅 14.13 GiB，低于 16.55 GiB 估计；01 kernel 的 PyTorch allocated/reserved 均为零。因此不能把问题归因于当前 kernel 的 PyTorch 缓存，也不能靠提高预算比例增加空闲量。具体占用进程未确认，用户随后释放了服务器。

重新连接时先在 00 配置 `RESTORE_AREAS=["models", "datasets", "runs"]`，从已有 Drive 快照恢复本地缓存，再按上文完成环境检查。新分配的 GPU 型号可能不同；进入 01 后重新运行预检查，不复用旧 `preflight` 变量。旧代码也可在 01 的预检查单元之后新增单元检查：

```python
import json, subprocess, torch
cfg = json.loads(PILOT_CONFIG_PATH.read_text())
dev = torch.device(cfg["runtime"].get("model_device", "cuda:0"))
free, total = torch.cuda.mem_get_info(dev)
print({"GPU": torch.cuda.get_device_name(dev),
       "total_GiB": total / 2**30, "free_GiB": free / 2**30,
       "budget_GiB": total / 2**30 * cfg["runtime"].get("max_memory_fraction", .85),
       "allocated_this_kernel_GiB": torch.cuda.memory_allocated(dev) / 2**30,
       "reserved_this_kernel_GiB": torch.cuda.memory_reserved(dev) / 2**30})
r = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
    capture_output=True, text=True)
print("nvidia-smi returncode:", r.returncode)
print(r.stdout.strip() or r.stderr.strip() or "驱动未返回可见的进程记录")
```

确认占用者以后，停止属于自己的旧任务或关闭相应 kernel，再重跑预检查；不要批量杀掉未知进程。`empty_cache()` 只释放当前 PyTorch 分配器中未使用的缓存，无法释放仍被引用的张量或其他进程的分配，见 [PyTorch 说明](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.empty_cache.html)。未取得进程记录时，不把空进程列表当作显存无人使用。

同一忙碌 kernel 的下一单元会排队。在能访问同一 `/content/CellQuant` 的独立终端/kernel 执行：

```bash
cd /content/CellQuant
python -m cellquant.diagnostics --project-root /content/CellQuant
```

查看实际层号、方法、坐标对和进度序号；心跳只表示进程存活。CPU 负责 tokenizer、文件读写、校验和、日志、Python 调度及同步，CPU 使用率不可能始终为零。GPU 上大量细粒度事件计算也不保证利用率始终为 100%；必须根据本次计时评估速度。

完成或主动停止并确认子进程退出后，在工作区终端备份：

```bash
python -m cellquant.workspace backup --areas runs
```

查看实际 run 中的 `result.json`、`model_placement.json`、`state.json`、各 seed/method 的 `pilot_result.json` 和 `G1.json`。显存不足或事件/时间预算不足记录 RESOURCE_BLOCKED；缺完整对照时 G1 为 INSUFFICIENT_EVIDENCE，不能改门槛放行。G1 合格才进入 AutoDL canary。

## 验证范围与实现说明

CPU NumPy 代码保留为数学参考。新版生产数值路径使用 PyTorch 张量，包括面板变换、量化/重建、损失/预算、STE、事件几何、排序、增量系数和多项式根。为避免小矩阵 eigvals 的 CPU 内部路径，求根采用同设备的复数移位 QR；收敛失败明确阻塞，不回退 CPU，也不宣称 interval-certified 或全局最优。

科学阈值未修改。求解仍按原协议使用 FP64；模型按实际支持的 BF16/FP16 记录。GPU FP64 性能、细粒度调度开销以及实际 7B 质量需要在目标卡验证；工程 CUDA 测试通过不等于获得加速结论。新协议记录 torch CUDA 的 Hadamard/R2 随机数生成方式，所有匹配对照使用同一初始化与固定 R2，不与旧 NumPy 初始化的断点混用。

本机只有 4GiB RTX 3050 Laptop，能执行实际 CUDA 小模型和完整维度合成面板检查，无法加载预训练 7B；Colab L4/FHT 扩展上的新版整条流程由以上步骤实测。准确的本地命令、耗时、失败记录和最终结果见 `evidence/gpu_resident_20261116/`。
