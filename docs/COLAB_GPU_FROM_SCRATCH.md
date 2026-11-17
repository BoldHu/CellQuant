# 新 Colab 服务器：GPU 驻留 G1，从环境到测试

本流程使用 `00_Colab_GitHub_Workspace.ipynb` 和 `01_Colab_Real_Layer_Pilot.ipynb`。旧 CPU 实验保留；新协议保存到 `runs/prepared/g1_gpu/`。不要恢复旧优化器继续跑，也不要覆盖旧准备配置。CPU 数学参考仍用于验证；真实 G1 使用 PyTorch CUDA 后端。

## 0. 先把本地修改同步到 GitHub

本次代码不会因为新建 Colab 服务器而自动出现在 GitHub。先在 VS Code 审查并提交本轮核心代码、配置、脚本、测试、说明和两本 notebook，再 push 到 `BoldHu/CellQuant` 的 `master`。

提交前将 notebook 的 `HF_TOKEN` 清为空字符串，并清除含凭据的输出；令牌只在云端运行时填写。不要把 `.cache/`、模型权重或活动 `runs/` 加入 Git。保存的脱敏验证记录位于 `evidence/gpu_resident_20261116/`。

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
HF_TOKEN = "在云端填写有模型访问权限的 token"
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

所有 seed 面板采集完后出现 `g1.model_released`：这表示不再需要基础模型，已释放其显存，**不是搬回 RAM**。共享 Q、拟合/held-out 面板及求解状态继续保留在 GPU。显存下降到面板工作集大小属于这一阶段的预期行为。

结果指针文件把 notebook 绑定到本次唯一 run；后面的 G1 评估单元自动读取该 run 的真实 `G1_evidence.json`，不再默认评估空模板。运行开关、工程 PASS 和进程退出码不替代 G1 科学门槛。

## 4. 长作业诊断、结束和备份

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
