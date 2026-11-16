# GitHub、VS Code、Colab 与 Drive 工作流

更新：2026-11-16。代码仓库为 `https://github.com/BoldHu/CellQuant.git`，当前分支为 `master`。
本地 VS Code 编辑并提交代码；Colab clone/pull 后运行；Drive 持久保存模型缓存、数据缓存和实验记录。

## 1. 首次连接

本地打开项目，安装 Google 官方 [Colab 扩展](https://marketplace.visualstudio.com/items?itemName=Google.colab) 和 Jupyter 扩展。本仓库的 `.vscode/extensions.json` 会推荐它们。

先把本次新增工作流文件提交并 push 到 `origin/master`。VS Code Source Control 中审查更改、提交、同步即可；运行输出应保留在被忽略的 `runs/`，token 不写进代码或 notebook。

打开本地 `notebooks/00_Colab_GitHub_Workspace.ipynb`，选择右上角 **Select Kernel → Colab → New Colab Server**，按提示登录 Google 并选择资源。VS Code 编辑的是本地文件，单元在远程 kernel 执行；远程 Python 不会自动看到尚未 push/pull 的本地源码。

六本 notebook 已在配置之后、路径初始化之前直接加入 **挂载 Google Drive** 单元。选择 Colab 内核后从上到下运行，在该单元按 Google 弹窗完成授权即可。`MOUNT_DRIVE` 在 Colab 中默认启用，本地和 AutoDL 默认跳过；浏览器与 VS Code 使用同一个挂载命令。

挂载代码与已核对的 [Google 官方扩展源码](https://github.com/googlecolab/colab-vscode/blob/d238503acf565e3ee5edf71df4263cbd8af01fa9/src/colab/commands/register.ts#L71) 一致：

```python
from google.colab import drive
drive.mount("/content/drive")
```

首次授权由 Google 弹窗完成；已挂载的会话会复用挂载。云盘中没有 `CellQuant` 文件夹属于正常情况，工作区初始化会自动创建；`/content/drive/MyDrive` 不存在则说明当前配置的挂载入口不可用，需要先完成挂载单元。

可以在同一个 Colab kernel 验证 `from pathlib import Path; print(Path("/content/drive/MyDrive").is_dir())`。返回 True 后重新运行 `configure_workspace` 所在单元即可。不要手动创建本地 `MyDrive` 目录来替代挂载，否则文件会落在临时运行时而不是云盘。

在初始化 notebook 的配置单元设置：

```python
DRIVE_ROOT = "/content/drive/MyDrive/CellQuant"  # 可换成自己的目录
INSTALL_BASE_DEPENDENCIES = True
RUN_G0 = True
BACKUP_AREAS = ["runs"]
```

依次运行。它会 clone 到 `/content/CellQuant`、配置缓存、安装项目基础依赖、实际计算 G0，并把本次运行记录备份到 Drive。不替换宿主 torch，不下载大型模型。G0 结果读取实际 `runs/00_math/production/G0.json`。

已有 checkout 只复用；初始化不会擅自 pull、切换分支或覆盖改动。若使用旧的远程 checkout，先按下面的更新流程 pull。阶段 notebooks 00–04 会识别 `/content/CellQuant`，并在各自 kernel 中读取 `.cache/colab_workspace.json` 配置。

## 2. 文件在哪里

| 内容 | Colab 运行时本地路径 | Drive 持久区域 |
|---|---|---|
| 源码与 Git | `/content/CellQuant` | GitHub 管理，不备份整个 checkout |
| HF Hub 缓存（模型及原始仓库文件） | `.cache/huggingface/hub` | `models` 快照 |
| 数据集加工缓存 | `.cache/huggingface/datasets` | `datasets` 快照 |
| token blocks、配置、结果、旋转与优化器 checkpoint | `runs` | `runs` 快照 |
| HF 登录凭据 | `.cache/huggingface/token` 等 | 不进入快照 |

上述相对路径均相对于 `/content/CellQuant`。HF 缓存环境变量在 import 之前设置，冻结配置中的显式 cache 路径仍优先，不追改旧协议。[HF 官方说明](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)

Drive 中 `objects/<前两位>/<SHA-256>` 保存文件内容，`snapshots/models|datasets|runs/时间_随机ID.json` 保存路径、校验值和代码版本。相同内容只传一次；snapshot/objects 必须一并保留。缓存中的内部文件链接恢复为 Linux symlink；Windows 无需 symlink 权限时用经校验的副本。Hub 区域也可能含原始 dataset 仓库，`datasets` 区域主要是加工后的 Arrow 等缓存。

计算时从本地 SSD 读，阶段结束再同步。Google 建议避免在 Drive 上做大量小文件随机 I/O，并推荐先复制到运行时本地使用。[Colab FAQ](https://research.google.com/colaboratory/faq.html#drive-timeout)

## 3. 准备真实模型，先持久化再运行

先看 notebook 01 的实际输出。`RUN_REAL_MODEL=False`、preflight `BLOCKED`、G1 `INSUFFICIENT_EVIDENCE` 表示真实 G1 尚未开始；所有单元正常结束只说明默认流程执行完成。

2026-11-16 用户当前会话仍为 Python 3.13.16 / Transformers 5.22.1。适配器明确要求 Transformers 4.44.2，项目固定 tokenizers 0.19.1；先准备 Python 3.12 环境，再安装 `requirements/hf_optional.txt`。不要在正在使用的 Colab 系统 Python 中执行降级 Python 的命令。

切换前，在已初始化的 workspace notebook 执行 `logged_command([sys.executable, "-m", "cellquant.workspace", "backup", "--areas", "runs"], "backup_before_runtime_change")`，确认输出 `BACKED_UP`。然后在 VS Code 的 Colab kernel 选择器新建服务器，选择 GPU、可用的高 RAM 配置，并在 `Select a runtime version` 选择 `2026.07`；该版本官方列出的 Python 为 3.12.13。版本选择由扩展支持，是否显示取决于服务器提供的版本列表；如果没有该项，先更新扩展，仍没有时可使用浏览器的 Runtime → Change runtime type → Runtime Version 选择并在浏览器完成这一阶段。[Colab 官方版本说明](https://research.google.com/colaboratory/runtime-version-faq.html)、[官方扩展版本选择实现](https://github.com/googlecolab/colab-vscode/pull/424)

在新服务器重跑 workspace 初始化，设置 `RESTORE_AREAS=["runs"]`、`RESTORE_RUNS_CONFLICT="separate"`、`RUN_G0=False`、`BACKUP_AREAS=[]`。执行 `import sys; print(sys.version)` 确认实际为 3.12；重启 kernel 本身不会改变 Python 版本。然后通过 `logged_command([sys.executable, "-m", "pip", "install", "-r", "requirements/hf_optional.txt"], "install_real_model", 1800)` 安装固定依赖，保留宿主 torch。安装后重启 kernel、重跑工作区初始化与数学 notebook，记录实际环境；FHT、完整模型和性能仍需后续单独实测。

以下命令在云端项目根目录执行。后面的 Python 命令放在初始化 notebook 中，使用它定义的 `logged_command([...], "标签")`；如使用扩展的 Colab terminal，先运行 `cd /content/CellQuant`。普通 VS Code 本地 terminal 仍是你的 Windows 电脑。

```bash
python -m pip install -r requirements/hf_optional.txt
git clone https://github.com/facebookresearch/SpinQuant third_party/SpinQuant
git -C third_party/SpinQuant checkout 8f47aa3f00e8662caf1a484153920a07e5281c3a
python scripts/prepare_spinquant.py
```

已有 SpinQuant checkout 时核对 commit/clean tree，不重复 clone。FHT 在目标 torch/CUDA 下安装和探测，具体要求见 [运行手册](RUNBOOK.md)。基础安装不会伪称 FHT 已支持。

LLaMA 需要 HF 账号已获模型访问权。按用户当前偏好，workspace notebook 顶部配置单元统一提供全局变量；只在该处填写真实值：

```python
HF_TOKEN = ""  # 在本地填写令牌，提交 notebook 前恢复为空字符串
```

先运行顶部配置，再运行 notebook 中的 HF 登录单元。它将同一令牌写入当前进程的 `HF_TOKEN` 环境变量，`logged_command` 子进程会继承；不打印令牌。VS Code 扩展环境不依赖浏览器 Colab Secrets API。另开的 terminal 不会继承 notebook 的进程环境，下载命令优先在 notebook 中运行。含真实令牌的 notebook 不提交到 GitHub。

若准备数据在 workspace notebook，而真实执行在 01 的另一个 kernel，可在工作区初始化后用以下方式登录：

```python
from huggingface_hub import login
if not HF_TOKEN.strip():
    raise ValueError("先填写并运行顶部 HF_TOKEN 配置")
os.environ["HF_TOKEN"] = HF_TOKEN.strip()
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
```

该方式在当前进程设置令牌，并将登录凭据保存到当前 VM 的 `.cache/huggingface/token` / `stored_tokens`，供同一服务器中完成工作区初始化的其他 kernel 使用；不会写入 Git 凭据，也不进入项目的 Drive 快照。销毁 VM 后需重新登录。HF 账号必须已获目标模型访问权，令牌登录本身不授予 gated model 权限。登录失败时先处理权限，不进入模型下载。

登录单元会显示账号名称并执行 `HfApi.auth_check`，再下载小配置文件。若返回 403 且明确提示账号不在 `authorized list` 中，需到目标模型页面申请访问或使用已获权限账号的令牌；改变 token 输入方式不能消除该权限限制。保留真实 HTTP 异常，不再用 `force_download=True` 将它包装成泛化的下载失败。

workspace notebook 现已直接包含“准备真实 G1 输入：校准数据、模型权重与 Drive 备份”单元。上方 HF 权限检查下载的小型 `config.json` 不包含模型权重。先将顶部 `PREPARE_G1_INPUTS=True` 并运行配置单元，再运行该准备单元；它依次执行数据准备、真实预检查、权重下载/校验、Drive 备份。默认 False 不下载大模型。只有全部步骤成功才输出准备完成；预检查不为 READY 或下载校验失败均会停止。

该流程默认生成 `runs/prepared/g1/config.json`。已有冻结配置会复用，权重仍重新校验，文件不会被覆盖；若准备目录有失败残留且没有完整配置，保留原目录，设置新的 `CELLQUANT_PILOT_CONFIG`（文件名为 `config.json`），并确保 01 使用相同路径。另一个 notebook kernel 不继承当前 kernel 临时设置的环境变量，非默认路径需分别配置。

也可手动分步运行相同脚本：先准备版本与真实校准数据，再独立下载并核验模型：

```python
logged_command([sys.executable, "scripts/prepare_experiment.py",
                "--config", "configs/colab_pilot.json",
                "--output", "runs/prepared/g1"], "prepare_g1")
logged_command([sys.executable, "scripts/cache_model.py",
                "--config", "runs/prepared/g1/config.json"], "cache_model")
logged_command([sys.executable, "-m", "cellquant.workspace", "backup",
                "--areas", "models", "datasets", "runs"], "backup_inputs")
```

`cache_model.py` 只下载 immutable model/tokenizer revision，按准备阶段的分片大小/SHA-256 及 safetensors header 检查，不分配完整 LLM，不产生 G1 证据。校验记录写在唯一 `runs/cache_checks/<ID>/`。准备输出目录已存在时保持不可覆盖，重新准备需另选名字。

模型文件会同时占用运行时本地盘与 Drive。先确认两处空间；Drive 空间不会增加 GPU 显存或 CPU RAM。7B G1 当前 preflight 至少要求 **24GiB 可用 CPU RAM**，这是最低检查，不是足够内存保证；普通低 RAM Colab 可能被阻塞。T4 记录实际 FP16，不能把高 RAM 或缓存下载成功当作科学门槛通过。

然后执行：

```python
logged_command([sys.executable, "-m", "cellquant.experiment",
                "--config", "runs/prepared/g1/config.json", "--preflight"], "g1_preflight")
```

查看真实 `preflight.json`；只有 READY 才在 notebook 01 设置实际配置并显式启动 G1，或运行同一命令去掉 `--preflight`。preflight 的进程正常退出并不代表 READY，缺证据/资源仍是 BLOCKED。

01 会打印实际使用的配置路径。在 `RUN_REAL_MODEL=True` 时，准备配置缺失会直接报告目标路径及 workspace 准备入口，不再回退到 `configs/colab_pilot.json` 草案；只有默认不执行模型的模式才允许用草案做诊断。若准备配置存在但预检查仍 BLOCKED，执行单元会列出具体缺失项，保留原门槛。

每阶段结束、停止下载和训练后保存：

```python
logged_command([sys.executable, "-m", "cellquant.workspace", "backup",
                "--areas", "runs"], "backup_results")
```

这是显式的阶段备份，不是持续同步服务。Colab 突然销毁前尚未备份的内容会丢失；中断作业应等待子进程退出、checkpoint 保存结束后再备份。源文件发生变化会拒绝提交完成索引，传输中断最多留下可复用对象，不能显示 BACKED_UP。新传输校验 SHA-256；已存在对象在恢复时重新校验。备份不会自动审计正在写入的多文件 checkpoint。

## 4. 断线后的新 runtime

重新选择 Colab server、挂载同一个 Drive，运行初始化 notebook，保持 `PROJECT_ROOT=/content/CellQuant`。配置：

```python
INSTALL_BASE_DEPENDENCIES = True
RESTORE_AREAS = ["models", "datasets", "runs"]
RESTORE_RUNS_CONFLICT = "separate"
RUN_G0 = False
BACKUP_AREAS = []
```

恢复发生在新实验之前。没有该区域的快照会报告 NOT_FOUND。每个区域默认恢复最新完成快照。需要旧版本时用 `restore_area(..., snapshot=明确索引路径)`；索引可由 `python -m cellquant.workspace status --areas models datasets runs` 列出。

重启 kernel 会保留当前 VM 的本地文件；已有 checkout 不一定是空目录。若 `runs/00_math/environment.json` 等文件与备份不同，直接恢复会拒绝覆盖。初始化 notebook 的 `RESTORE_RUNS_CONFLICT="separate"` 会保留现有 `runs`，将 Drive 中的**整个快照**另存为项目根目录的 `restored_snapshots/runs/<快照哈希>/`，打印 `RESTORED_SEPARATELY` 和实际 `destination`。该目录排除在 Git 外，位于活动 `runs` 之外，避免被结果分析递归扫描；Drive 中的原始快照仍保留。这份历史副本不自动成为活动运行目录；其中的配置绝对路径与旧 gate 不改写、不自动用于续跑。重复恢复同一快照会校验同一个目录，若历史副本被改动则继续拒绝覆盖。

没有冲突时仍恢复到原路径。模型/数据缓存保持严格冲突检查，独立命令行默认也拒绝所有冲突；需要保留两份 runs 时显式执行：

```bash
python -m cellquant.workspace restore --areas runs --runs-conflict separate
```

如果只是继续使用当前 VM 中已有的结果，无须从 Drive 恢复，可设 `RESTORE_AREAS=[]`。恢复前停止写入相关目录的作业。新版 `logged_command` 会将日志末尾的实际异常及退出码带入 notebook 报错，完整日志仍保存在 `.cache/workspace_logs/`。

重新安装真实阶段的可选依赖和固定上游。基础模型不会因 runtime 丢失而必须重新从 HF 下载，但缓存恢复仍需要从 Drive 复制到本地。HF 授权凭据需要重新提供。恢复保持文件原字节，**不重写配置中的绝对路径或放行旧 gate**；保持相同本地路径，跨 AutoDL 路径使用项目迁移流程。

代码 commit、依赖、GPU、dtype、模型/数据 hash 任一改变时，原来的实验 resume 资格必须重新检查。保留旧结果，以新 run 重新准备；不混接不同版本的优化器状态或继续消费旧 G3 许可。

## 5. 日常改代码

本地 VS Code 审查后提交并 push：

```bash
git status --short
git add <本次确认要提交的文件>
git commit -m "Describe the change"
git push origin master
```

云端先停止作业并完成阶段备份，然后在 Colab terminal 执行：

```bash
cd /content/CellQuant
git status --short
git pull --ff-only origin master
python -m pip install -e '.[dev]'
```

先确认工作树干净；有修改时保存并解决冲突，不自动 reset/stash。已有本地提交导致无法 fast-forward 时停止并审查分支。依赖文件发生变化时按阶段更新对应依赖，保留经验证的 torch。

没有远程 terminal 时，在云端 notebook 执行相同命令：

```python
status = subprocess.check_output(["git", "-C", "/content/CellQuant", "status", "--porcelain"], text=True)
if status.strip():
    raise RuntimeError("Cloud checkout has local changes; save and resolve them before pulling")
subprocess.run(["git", "-C", "/content/CellQuant", "pull", "--ff-only", "origin", "master"], check=True)
```

**pull 完成后重启 notebook kernel**，重跑初始化配置，再开始新作业。重启 kernel 保留当前 VM 的本地文件；删除 runtime 会丢失本地文件，需要从 Drive 恢复。已 import 的 Python 模块不会因磁盘上的代码更新而自动替换。

本地 `.ipynb` 若有更新，在 VS Code 中保存并使用对应版本；浏览器打开的 GitHub notebook 需要重新打开才能看到新单元。仅云端 `git pull` 不会替换已经打开的本地 notebook 内容。

## 6. 01 长时间无输出、GPU 空闲时

`Loading checkpoint shards: 100%` 只表示权重加载结束。当前后端先把模型放在 CPU，融合后逐层将模型送往 GPU 采集面板；默认 `solver.device=cpu`，完整输出侧矩阵变换、匹配对照和事件根搜索会消耗 CPU。GPU 空闲不能单独证明卡死。`_is_quantized_training_enabled` 是上游遍历模型属性触发的 FutureWarning，本身不是失败原因。

新进程启动时打印唯一 run_dir/PID，之后每 30 秒输出资源心跳。`state.json` 记录实际阶段、seed、fit/heldout、层号、方法、面板/坐标对、进度序号、CPU 累计时间、RSS 和剩余系统 RAM；结束时更新最终状态。`progress_sequence` 只在算法报告进展时增加，心跳不增加它，也不产生科学证据。01 每次调用使用独立日志，重跑不会覆盖旧的 pilot.log。

在能访问同一 `/content/CellQuant` 的独立终端运行只读诊断：

```bash
cd /content/CellQuant
python -m cellquant.diagnostics --project-root /content/CellQuant
```

不需要另建 Colab 服务器；忙碌 kernel 中的新单元会排队，另一服务器也看不到本次进程。没有独立终端时，可在同一服务器的独立 kernel 中调用 notebook 的只读诊断单元。间隔 30 秒比较两次 CPU 累计时间、进度序号、可用 RAM 和进程状态。CPU 时间增长说明正在消耗 CPU，不能保证算法有效前进；若资源与进度都持续不动，需要进一步定位，不能仅凭一次 0% 断言死锁。若进程已经消失，应查看日志尾部、退出码和 result.json，不能继续把陈旧 RUNNING 文件视为活跃作业。

旧版本在采集和求解阶段可能一直留下 `phase=fusion`；诊断会标记 legacy 状态，不将它当作当前阶段。新版会校验 PID 创建时间，避免 PID 被重用而误认旧作业。诊断不导入 torch、不加载模型、不修改运行目录，也不打印命令行中的凭据。

默认两 seeds、五方法时每方法预算约 1440 秒；它不是整次 G1 的完成时间承诺，准备、held-out、无法在单个运算内部提前终止的计算仍可能耗时。科学门槛、实际矩阵维度、精度和预算保持不变。不要为让 GPU 看起来繁忙而直接变更冻结协议中的 device/dtype。

正在运行的旧进程不会自动获得新日志。先读取诊断与保留日志，待旧作业结束或停止后再同步代码、重启 kernel，以新 run 运行；不混用新旧代码续跑。模型缓存仍保留在当前 VM，无需因为诊断而重复下载或创建模型副本。

## 7. 已验证与待验证

本轮本地检查覆盖缓存路径、版本化备份/恢复、内容去重、源文件变化、传输中断、文件冲突、路径逃逸、损坏对象/索引拒绝，以及模型下载入口的本地 safetensors fixture。fixtures 不产生任何真实模型指标。

Google 登录、Drive FUSE 挂载、VS Code 远程 kernel、真实大模型下载/缓存吞吐和 Colab GPU G1 尚未在当前环境执行。它们需要你的账号及云端 runtime；CPU/本地检查不能代替这些验证。科学 gate 文件未调整。
