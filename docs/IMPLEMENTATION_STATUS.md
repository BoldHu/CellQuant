# 实现状态与证据边界

更新：2026-11-16。以下“已实现”描述代码；实验结论单独列出。整个研究尚未完成。

| 组件 | 代码状态 | 已执行的证据与限制 |
|---|---|---|
| 原 CPU 参考与基础设施 | 保留并修正零组物理 scale 风险 | 原 62 项测试复现；最终全套 116 项通过 |
| 共享 global Q、多面板输入侧 WQ.T/QX、输出侧 QB | IMPLEMENTED | FP64 增量胞元系数、风险约束、实际边界 rounding；3 seeds/24 cases 合成 G0 PASS |
| 完整维度 | IMPLEMENTED / REAL_PERFORMANCE_NOT_RUN | d=4096、down=4096×11008，2 tokens、1 pair、±0.0001 的工程 smoke COMPLETE；不是正式吞吐 |
| 筛选与事件增量 | IMPLEMENTED_CPU_REFERENCE / TORCH_CUDA | 输入稳定组筛选、输出完整行 GPU support envelope；事件排序、增量系数和 QR 数值根均在张量设备计算，小例已与 CPU 对照 |
| CUDA dense GEMM/量化/求解 | IMPLEMENTED_PYTORCH / ENGINEERING_VALIDATED | 实际 CUDA 小模型/五方法和 4096×11008、2-token 输出侧计算；没有自定义融合 CUDA kernel，真实 7B 性能未验证 |
| Hadamard / angle-grid / Givens STE / fixed-scale | IMPLEMENTED | 相同面板、坐标预算、约束、墙钟上限；小矩阵验证。grid/STE 中途 optimizer 恢复 NOT_IMPLEMENTED |
| 真实层采集与 Colab G1 | GPU_RESIDENT_IMPLEMENTED / NEW_7B_RUN_NOT_RUN | 新配置模型驻留 GPU，所有 seed 采集后释放模型，面板留在 CUDA 求解；旧用户尝试结果未知，新版 7B 待用户 Colab 执行 |
| SpinQuant 固定上游 / overlay / R.bin | IMPLEMENTED | commit 8f47aa3… 已核对、tracked tree clean；禁 embedding centering、保 gamma，导出全部 R2 |
| FP 旋转 / GPTQ W4A4KV4 | IMPLEMENTED / TINY_CUDA_VALIDATED | MHA/GQA 小模型 FP 相对误差约 1e-7；GPTQ 有限输出、逐层恢复与连续运行参数差 0 |
| 完整 R1+R2 CE/SGDG baseline | IMPLEMENTED / TINY_CUDA_VALIDATED | 实际两步训练与 Trainer 恢复一致；正式 100 步×8 的完整 CE 训练池尚未运行，distinct blocks 数另报 |
| SpinQuant_full_FP32_port | IMPLEMENTED / NUMERICAL_ONLY | 输出和 R1/R2 梯度误差约 1e-7；小模型训练执行。真实质量资格与效率结论 BLOCKED |
| 固定数据与实验后端 | IMPLEMENTED / REAL_PIPELINE_NOT_RUN | contract 1.0、revision/token/源码/环境指纹、唯一 run_id、完整分片 SHA、原子结果、资源采样 |
| WikiText/C4 与六任务 | IMPLEMENTED / INTERFACE_VALIDATED | 固定 harness commit+六任务 revision/metric；六任务 local fixtures 实际跑通；真实分数 NOT_RUN |
| 配对分析 / G2–G4 构造 | IMPLEMENTED | token 加权 NLL 与 paired seed/document/item bootstrap；空结果准确不足，不生成表/图 |
| 五本 notebook | DEFAULT_PATH_EXECUTED | IPython inprocess，默认 smoke/manifest-only；真实 Colab/AutoDL notebook 会话未验证 |
| 目标 RTX PRO 6000 Blackwell / 优化 FHT | NOT_RUN / BLOCKED | 本机 RTX 3050 Laptop 4GiB；FHT、FlashAttention、Triton 缺失 |
| 2026 强基线、正式消融、论文主表 | NOT_IMPLEMENTED / NOT_RUN | 不能由官方论文数字或本地 fixtures 补填 |

## 科学门槛

G0 来自实测合成数学案例，为 PASS。G1–G4 没有完整真实证据，保持 INSUFFICIENT_EVIDENCE/BLOCKED 执行状态；资源中断的比较不被写为方法失效。`configs/gate_rules.json` SHA-256 仍为 `5d36290ca487795337300f3cccbc1c4d5a18d250639498bb15455ca5ad6a554e`。

事件根为 FP64 数值根。实现枚举候选并对预测最优的 32 个可行候选做完整量化复核；接受更新必须满足实际损失/预算，不能宣称所有浮点近退化情形下的角度全局最优或区间认证。事件/时间 cap 超限明确 RESOURCE_BLOCKED，不截断通道、不缩小正式角域使任务显示成功。

本机实际可见 GPU 约 4GiB、可用 CPU RAM 约 2GiB；未加载预训练 7B/8B，未下载其权重，未测真实 PPL/下游任务收益。高 RAM Colab / 目标 AutoDL 上所需命令见 [RUNBOOK.md](RUNBOOK.md)。端到端正式协议、资源预算和真实恢复仍待目标运行审查。

2026-11-15 构建记录见 [VALIDATION_REPORT.md](../VALIDATION_REPORT.md)，对应归档索引为 `evidence/build_20261115/`；工作目录 `runs/` 还保留原始 checkpoint、失败尝试及完整日志。当时交付目录无根 Git，历史证据继续使用逐文件 source snapshot，不追填 commit。2026-11-16 接手已核对根 checkout 为 `BoldHu/CellQuant`、`master`、初始 commit `9a53bf7`。

## 2026-11-16：GitHub / VS Code / Colab / Drive

新增 `00_Colab_GitHub_Workspace.ipynb`、`cellquant.workspace`、`scripts/cache_model.py` 和 [工作流说明](COLAB_VSCODE.md)。模型与数据先缓存到 `/content/CellQuant` 本地盘，按 SHA-256 去重、版本化备份到用户配置的 Drive 根目录；恢复校验文件并拒绝冲突，凭据不包含在缓存区域中。权重可单独按固定 revision 下载和校验，再备份、加载并进入真实 G1。五本原阶段 notebook 会识别 Git clone 和每个 kernel 的缓存配置。

本地全套 125 项测试通过，最终增补 CLI 选项后相关 25 项再次通过；六本 notebook 的 IPython inprocess 默认路径及 6 条 CLI 初始化/备份/恢复命令实际执行成功。Google 登录、Drive FUSE、VS Code 云端 kernel、真实模型下载、Colab GPU G1 为 **NOT_RUN**，需要用户账号和远程 runtime。正式运行前需要提交本次源码并重新冻结相应指纹；Git 更新不会赋予旧 checkpoint 或旧 G3 新的资格。实际日志和 notebook 执行记录见 `runs/workspace_20261116/`，可携带记录见 `evidence/workspace_20261116/`。门槛不变。

首次云端反馈：用户提供的 notebook 已执行 `/content/CellQuant` 初始化，在 `/content/drive/MyDrive` 不可用处停止；这不代表 `CellQuant` 项目文件夹必须预先存在。已补充 VS Code 先运行末尾挂载单元并授权的引导和错误提示，保留挂载检查与自动创建项目目录的逻辑。相关本地测试 9 项通过，记录在 `evidence/workspace_first_run_20261116/`；完整 Drive 挂载、持久化与 GPU 实验仍待用户会话验证。

按用户后续要求，六本源 notebook 已直接包含前置的“挂载 Google Drive”单元，统一使用 `drive.mount("/content/drive")`。Colab 默认执行，本地/AutoDL 默认跳过；旧的 `MOUNT_DRIVE_IN_BROWSER` 改为通用 `MOUNT_DRIVE`，保留其余用户运行配置。六本 notebook 的 schema、Python 语法、单元顺序、挂载调用及失败传播 fixture 检查通过，9 项相关测试通过；真实授权/Drive 挂载仍为 NOT_RUN。记录见 `evidence/notebook_mount_20261116/`。

## 用户首轮 Colab 会话反馈（2026-11-16）

已读取用户保存的三本 00–01 notebook 输出并逐字节归档到 `evidence/colab_first_session_20261116/`。记录显示 Drive 已挂载、runs 快照 BACKED_UP、G0 输出 PASS；01 仅执行预检查，状态为 BLOCKED，`RUN_REAL_MODEL=False`，真实模型 NOT_RUN，G1 为 INSUFFICIENT_EVIDENCE。这里是用户会话输出归档，未在本机独立访问远程文件重算 gate。

该会话为 NVIDIA L4（约 22.03GiB）、Python 3.13.16、Torch 2.14.1+cu130、Transformers 5.22.1，FHT 未安装。尚缺固定模型/tokenizer/dataset revisions、真实 fit/dev tokens、明确 dtype 和 SpinQuant overlay。下一步先准备与固定 Transformers 4.44.2/tokenizers 0.19.1 匹配的运行环境，再安装/实测 FHT、准备真实 G1；不能仅打开执行开关便进入 G1 或 AutoDL canary。

## 重复恢复 runs 的冲突处理（2026-11-16）

读取用户第二次保存的 workspace notebook，实际子进程异常为 `FileExistsError`：`/content/CellQuant/runs/00_math/environment.json` 与 Drive 快照不同。原恢复器拒绝覆盖的保护行为正常；外层报错仅显示日志路径，操作引导不足。该次 pip 输出仍指向 Python 3.13，不能据此记录已切换为 Python 3.12。

新增显式 `--runs-conflict separate`：整份冲突快照另存至项目根目录的 `restored_snapshots/runs/<快照哈希>/`，返回 `RESTORED_SEPARATELY` 和目标路径，保留当前结果。历史目录位于活动 `runs` 外，避免递归分析重复收集，且已排除在 Git 外。初始化 notebook 通过 `RESTORE_RUNS_CONFLICT` 调用该选项，并在异常中附日志尾部和退出码。CLI 默认拒绝覆盖；模型/数据缓存、校验和、路径检查及 gate 均未放宽。另存副本不自动赋予 resume 资格。

12 项 workspace 测试通过；实际执行 notebook 的安装/恢复单元（关闭 pip，使用本地存储 fixture），验证冲突另存、保留两套字节、失败诊断和多区域 CLI。原始用户 notebook、日志和源码 hash 位于 `evidence/workspace_restore_conflict_20261116/`。本轮云端 Drive 重跑与真实模型 GPU 为 NOT_RUN；本地检查不构成 G1 证据。

## 用户重跑前三本 notebook（2026-11-16）

最新保存的输出显示云端 clean commit `45b2a8659b33d130653030e8165537692b399212`，Drive 已挂载且 runs 恢复返回 `RESTORED_SEPARATELY`，数学 notebook 的参考检查、G0 和归档完整性均显示 PASS。01 仍为 `RUN_REAL_MODEL=False`、preflight BLOCKED（8 项缺失）、真实模型 NOT_RUN、G1 INSUFFICIENT_EVIDENCE。环境仍为 L4 / Python 3.13.16 / Torch 2.14.1+cu130 / Transformers 5.22.1，FHT 未安装。

三本原始 notebook 及观察记录归档于 `evidence/colab_after_restore_20261116/`；这是用户提供的会话输出，未在本机直接访问云端文件独立重算。下一步为备份当前 runs、切换兼容 Python 3.12 运行时并安装固定依赖，再准备固定 SpinQuant/FHT、模型访问与真实校准数据。具体步骤已补充到 `docs/COLAB_VSCODE.md`；尚不进入 AutoDL canary。

## Python 3.12 环境准备反馈（2026-11-16）

用户最新保存的三本 notebook 显示运行环境已切换为 Python 3.12.13、Torch 2.11.0+cu128、Transformers 4.44.2、datasets 2.20.0、tokenizers 0.19.1；NVIDIA L4 约 22.03GiB，BF16 可用。云端代码为 clean commit `4732c827891d012a776bc18e09ce890ebf20e88b`，Drive 挂载/原路径恢复已执行，G0 与归档完整性显示 PASS。无需再次要求切换 Python。

01 仍为 RUN_REAL_MODEL=False、preflight BLOCKED（8 项缺失）、真实 G1 NOT_RUN/INSUFFICIENT_EVIDENCE；FHT 未安装，固定 SpinQuant overlay、模型/tokenizer/dataset revisions、两 seed 真实数据与明确 dtype 尚缺。下一步先固定上游并实测 FHT/tiny 接口，再准备 HF 权限、真实输入、下载与 Drive 备份，最后执行 G1。观察归档见 `evidence/colab_python312_20261116/`。pip 同时报告预装 gradio/gcsfs 的依赖冲突，保留原始记录，不宣称整个 Colab 软件环境无冲突或已验证真实模型。

## 用户 Colab 后端兼容检查（2026-11-16）

最新 workspace 输出已显示 fast-hadamard-transform 1.1.0 安装完成、`CUDA/FHT: PASS`；固定 SpinQuant 小随机模型返回 `PASS_TINY_STRUCTURE_ONLY`，融合相对 RMS `1.068972578044603e-07`，旋转相对 RMS `1.7605411237572577e-07`，GPTQ 为 `EXECUTED_FINITE`，该子进程退出码 0。L4 / Python 3.12.13 / Torch 2.11.0+cu128 / Transformers 4.44.2。原始三本 notebook 与观察摘要位于 `evidence/colab_backend_20261116/`；这些是用户保存的输出，未独立读取远程 report.json 或重算。

01 的缺失项从 8 项减为 7 项，SpinQuant overlay 已不再缺失；剩余为模型/tokenizer/dataset 固定 revisions、模型分片元数据、seed 11/23 的真实 fit/dev tokens、明确 dtype。RUN_REAL_MODEL 仍为 False，真实 G1 NOT_RUN/INSUFFICIENT_EVIDENCE。下一步是 HF 授权、prepare_experiment、cache_model、Drive 备份，再以 READY 配置启动真实 G1；小随机模型检查不替代真实 7B 面板效果。

HF 权限反馈（2026-11-16）：用户附件明确为 Llama-2-7b-hf 的 403/GatedRepoError，服务端提示当前账号不在 authorized list。按用户要求，workspace notebook 顶部增加空值全局 `HF_TOKEN`，登录与下载统一使用该变量；显示账号名称、先执行真实 auth_check，并保留权限异常。模型访问仍未获验证，不能把修改 token 输入方式标为解除阻塞。Notebook schema/语法、空 token 阻止联网、其他单元与输出保留检查通过；归档与记录见 `evidence/hf_token_variable_20261116/`。

## 将真实 G1 输入准备接入 notebook（2026-11-16）

用户最新 workspace 输出显示模型 auth_check 和 609 字节 config.json 读取成功；保存的 notebook 中尚无 prepare_experiment/cache_model 执行或 WEIGHTS_VERIFIED 记录。01 已改为 RUN_REAL_MODEL=True，但仍缺 7 项准备信息，说明没有读到完整的准备配置。旧逻辑在准备文件缺失时会回退草案，且执行错误只显示通用 BLOCKED。

workspace 新增默认 False 的 PREPARE_G1_INPUTS 和完整调度单元，串联既有真实数据准备、预检查、权重下载校验与 models/datasets/runs Drive 备份；复用冻结配置，不覆盖失败残留。01 在真实执行时要求准备配置存在，显示实际路径和具体缺项。保留用户已有执行开关；不修改科学门槛或把模型配置文件当权重。

相关测试 **22 passed in 8.93s**，覆盖缺配置禁止启动模型、草案仅供不执行模式、准备/校验失败不备份、执行开关、冻结配置复用与原存储完整性检查。两本 notebook schema/语法及未改动单元保留检查通过。记录与脱敏前版本归档见 `evidence/g1_input_preparation_20261116/`。本机未执行真实 HF 权重下载或 Colab G1；仍须用户在云端显式运行新增准备单元后再启动 01。

## 真实 G1 已启动，补齐运行可观测性（2026-11-16）

最新用户保存的 01 显示实际配置为 `runs/prepared/g1/config.json`、预检查 READY、L4、两个 checkpoint shards 加载完成。用户反馈约 10 分钟没有后续输出、已用系统 RAM 约 28GB、GPU 几乎空闲。无法从这些记录确认仍在计算、死锁或剩余内存是否足够；未读取云端实际 PID、state/result.json，真实 G1 结果保持未知及 INSUFFICIENT_EVIDENCE，不能再把本次尝试描述为尚未启动。

代码核对确认模型先驻留 CPU，随后逐层 GPU 采集；求解默认 CPU，事件/数值根仍为 CPU FP64。旧代码在采集、初始化、求解和 held-out 计算时没有进度输出，state.json 甚至可能一直停在 fusion。新增实际阶段/seed/层/面板/方法/坐标对进度，每 30 秒输出 RSS、CPU 累计时间、剩余 RAM 心跳；心跳不增加算法进度序号，不计为科学成功。01 使用独立日志和 unbuffered 子进程，并加入只读诊断入口。诊断校验命令、工作目录和 PID 创建时间，明确标记 legacy 状态，不加载模型或改变冻结协议。

本地相关回归 **80 passed in 9.09s**，验证五方法进度回调前后 Q/损失一致、完整输出侧采集轴、心跳不伪造进展、I/O 失败传播与陈旧 PID 拒绝；包括既有可执行的小 CUDA 测试，不代表本次 Colab 7B 验证。三个 notebook schema/语法与未改动单元保留检查通过。首轮新增采集测试时发现 Windows 读/替换状态文件的瞬时共享冲突，限定重试后最终套件通过；失败原日志也保留。记录及脱敏前版本见 `evidence/g1_live_progress_20261116/`。

新版日志尚未在用户 L4/预训练 7B 上运行；当前旧进程不会自动获得改动。先在同一服务器独立终端采样，旧作业结束/停止后再同步代码并以新 run 执行 `python -u -m cellquant.experiment --config runs/prepared/g1/config.json`（或 01）。科研门槛 hash 未变，未切换 device/dtype、削减维度或填入实验结果。全 GPU 增量事件 kernel 仍 NOT_IMPLEMENTED。


## 2026-11-16 — GPU 驻留与 PyTorch 数值后端

用户要求新 Colab 上将模型与核心数值计算放在 GPU。`runtime.model_device=cuda:0`、`cpu_offload=false` 使模型直接加载到 CUDA；采集不再调用 model.cpu()/layer.cpu()，全部 seed 面板准备后释放基础模型，面板继续保留在 GPU。模型参数设备清单写入 model_placement.json，进度心跳记录 CUDA allocated/reserved。显存不足明确 RESOURCE_BLOCKED，不自动卸载到 CPU。

新增 tensor_math.py / tensor_solver.py：完整输入/输出侧张量、RTN、全目标/预算、事件生成与排序、输出侧 support envelope、稀疏系数更新、移位 QR 数值根、实际边界 rounding、前 32 候选完整复核和五种匹配方法均采用同设备 PyTorch 运算。输出侧 STE 仅构造变化的两行梯度，已对照完整目标的更新一致。CPU NumPy 保留为参考；公开 shared-Q API 与 notebook 都按 backend 分派。文件/校验和、tokenizer、日志和 Python 调度仍在主机端，不能宣称 CPU 使用率为零或 GPU 利用率始终 100%。没有自定义融合 CUDA 事件 kernel。

为避免 PyTorch 2.6 对小 eigvals 矩阵的内部 CPU 路径，最终求根采用 torch CUDA 复数移位 QR；数值不收敛明确阻塞，未降低任何科研门槛。GPU 初始化/R2 使用已记录的 torch CUDA RNG，所有匹配方法同一初始化和固定 R2；新协议与旧 NumPy 初始化的结果/断点分开。

验证：初始 **148 passed in 13.31s**；最终 **191 passed in 23.51s**。真实本机 RTX 3050 Laptop CUDA 小 LLaMA 全参数与层输入均在 cuda:0，五方法运行通过，FP 融合/旋转相对 RMS 为 1.077980899e-07。完整输出矩阵 4096×11008、2 tokens 的 GPU 初始化和一次直接评估耗时 3.5775s，采样对应峰值分配 2496416256 bytes；这只是完整维度工程检查，不是预训练 7B、完整角域吞吐或速度比较。CPU 参考与生产 G0 重跑 PASS，空 G1 仍 INSUFFICIENT_EVIDENCE。

workspace/01 已增加 GPU 检查、新 g1_gpu 配置目录、旧 CPU 配置拒绝和本次 run 结果指针。六本 notebook schema/语法检查通过，仅清除修改单元的旧输出；原用户配置和 HF 变量保持，归档脱敏。新服务器完整步骤见 docs/COLAB_GPU_FROM_SCRATCH.md。实际命令、日志、源码 hash、失败尝试与报告在 evidence/gpu_resident_20261116/。

尚未执行：新版 Colab L4 + FHT extension、预训练 7B G1 质量/速度、AutoDL 96GB。原因：本机只有 4GiB GPU，无法加载 7B，且没有直接连接用户 Colab。云端先按指南准备环境并运行 `python -u scripts/run_gpu_resident_checks.py --fht-backend extension --full-dimension --output <新的结果目录>`，再通过 01 或 `python -u -m cellquant.experiment --config runs/prepared/g1_gpu/config.json` 实测。G1 合格前不进入正式评估；代码未提交/push。


## 2026-11-17 — G1 显存预检查诊断

读取用户保存的 Colab notebook：GPU 驻留工程检查显示 PASS_ENGINEERING_ONLY，01 在加载真实模型前被显存预检查阻塞。用户补充的实际 L4 诊断为 total=22.0343 GiB、free=14.1336 GiB、85% budget=18.7292 GiB，当前 kernel 的 PyTorch allocated/reserved 均为 0；free 低于既有 16.55 GiB 估计。具体占用进程尚未确认，用户随后释放服务器准备重新连接，不能宣称已定位或已清除某个旧模型。

预检查现在记录所配置 CUDA 设备的总容量、当前空闲、预算、估计需求、当前进程 PID 和 allocator 占用；错误区分 total_capacity/configured_budget/currently_free。保留原估计、85% 预算和所有科学门槛，不自动释放显存、杀进程或回退 CPU。操作指南补充新服务器恢复缓存、显存检查与重新预检查步骤。

实际本地全套 197 passed, 1 warning in 52.61s；警告为 Windows pytest 缓存目录写入被拒，不影响测试退出码 0。新增 6 项覆盖占用中的 L4、空闲 L4、预算限制、小显存、FP32 估计及无 CUDA；设备选择也作回归验证。本轮没有编辑用户 notebook，接手时的三本副本已脱敏归档；工作期间 00 数学与 01 notebook 又被外部操作更新，未覆盖这些变化。门槛 hash 不变。记录见 evidence/preflight_vram_20261117/。尚未执行用户新服务器重试或真实 7B G1，当前实验仍为 PREFLIGHT_BLOCKED/INSUFFICIENT_EVIDENCE；代码未提交/push。
