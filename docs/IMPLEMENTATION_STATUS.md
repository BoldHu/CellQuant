# 实现状态与证据边界

更新于 2026-11-19。2026-11-18 的 LLaMA-2-7B 正式 A100 G1 为 **FAIL**。G2–G4 尚未执行。

**01d 的 7B A100 研究已完成。** 两个种子、八种方法、九个位置和 16 篇新验证文章均有原始记录。相对 128 对网格，三个输出的中位 NMSE 改善 1.55% [1.47%, 1.63%]，种子 11 改善 3.10%，种子 23 持平。在 256 对网格上追加修正，两个种子的三次非零步长都满足尺度预算，输入代价使联合拟合损失上升，两次都返回原网格。七组区间均从逐文档测量复算一致。预定的 7B 增量条件未达到。[7B 原始测量和诊断](../evidence/output_matching_20261119/colab_7b_summary.json) · [原协议与结果](OUTPUT_MATCHING_STUDY.md#7b-实测与下一轮决策)

7B 分批追加修正已在独立拟合和验证文档上完成复验，状态为 **RETEST_STOPPED_BY_BOUND**。此前 pilot 的 +0.0812% 未能复现。两个新拟合结果各接受 0/32 批，原网格、选批及同批数随机对照在 32 篇验证文章的九个位置误差相同。相对最强匹配对照的准确率收益上界为 0%，相对最快对照的校准加速上界为 0.9858×。原正式 G1 仍为 FAIL；这次停止前完成了八份方法记录。[复验汇总](../evidence/output_matching_20261119/batch_g1_retest_7b_summary.json) · [原 pilot](../evidence/output_matching_20261119/batch_documents_7b_summary.json)

加入短网格起点的输入感知选角诊断，保留一个共享旋转、原动态量化器与风险上限。预训练 42M 的 32 篇新文章上，九位置中位 NMSE 相对同起点的输出选角下降 3.6570% [2.9300%, 4.6226%]，相对完整网格下降 1.0445% [0.6099%, 1.3469%]。7B 下一步只在上述复验的拟合面板上检验该规则。真实整模型质量与校准速度待测。[42M 原始统计](../evidence/output_matching_20261119/bilateral_documents_42m_summary.json) · [A100 操作](OUTPUT_MATCHING_STUDY.md#在选角时计入输入侧误差)

拟合目标 `relative_site_nmse` 的 01c 已在真实 A100 上运行完成，状态为 `SCREEN_COMPLETE`，判定为 **STOP**。两种子、九位置、每组 32 对。新 CellQuant 相对两个网格中较强者的中位改善为 −0.0337%，种子 11 为 −0.0627%，种子 23 为 −0.0373%。相对旧 CellQuant 为 −0.0044%。三项预设条件均未满足，程序跳过正式 G1 复验。此次评估使用原校准文档内的划分。[01c 实测摘要](../evidence/g1_objective_20261118/a100_summary.json) · [本轮协议](G1_OBJECTIVE_TRIAL.md) · [本地工程验证](../evidence/g1_objective_20261118/local_validation.json)

引擎使用 `torch_batched_bilateral_prefix_v3`。本机 RTX 3050 Laptop 上，两版完整轴混合合成案例各重复三次，单对搜索中位数从 20.34 秒降至 3.48 秒，结果一致。本地回归 310 项通过。A100 上新版 CellQuant 完成了两个 seed 各 128 对的正式求解，独立文档上的收益和校准时间见下文。[同卡合成测量](../evidence/a100_readiness_20261118/benchmark.json) · [工程验证](../evidence/a100_readiness_20261118/validation.json) · [A100 协议](COLAB_A100_TEST.md)

| 组件 | 代码状态 | 已执行的证据与限制 |
|---|---|---|
| 原 CPU 参考与基础设施 | 保留并修正零组物理 scale 风险 | 全套 310 项通过，含实际 CPU/CUDA 数值对照、连续共享 Q 和预算耗尽回归 |
| 共享 global Q、多面板输入侧 WQ.T/QX、输出侧 QB | IMPLEMENTED | FP64 增量胞元系数、风险约束、实际边界 rounding；3 seeds/24 cases 合成 G0 PASS |
| 完整维度 | REAL_A100_G1_COMPLETE | 正式试验九个面板、两个 seed。CellQuant 每个 seed 完成 128 对，均在 1,440 秒方法预算以内 |
| 筛选与事件增量 | IMPLEMENTED_CPU_REFERENCE / TORCH_CUDA | 输入稳定组筛选、输出完整行 GPU support envelope；事件排序、增量系数和 QR 数值根均在张量设备计算，小例已与 CPU 对照 |
| CUDA dense GEMM/量化/求解 | BATCHED_PYTORCH / REAL_A100_VALIDATED | 正式 A100 上 seed 11/23 的 CellQuant 含拟合采集校准时间为 724.24/737.88 秒 |
| Hadamard / angle-grid / Givens STE / fixed-scale | IMPLEMENTED | 相同面板、坐标预算、约束、墙钟上限；小矩阵验证。grid/STE 中途 optimizer 恢复 NOT_IMPLEMENTED |
| 真实层采集与 Colab G1 | RUN_COMPLETE / G1_FAIL | A100 完成 10 份结果及全部 held-out 评估。准确率与效率两条路线均未达到预注册门槛 |
| SpinQuant 固定上游 / overlay / R.bin | IMPLEMENTED | commit 8f47aa3… 已核对、tracked tree clean；禁 embedding centering、保 gamma，导出全部 R2 |
| FP 旋转 / GPTQ W4A4KV4 | IMPLEMENTED / TINY_CUDA_VALIDATED | MHA/GQA 小模型 FP 相对误差约 1e-7；GPTQ 有限输出、逐层恢复与连续运行参数差 0 |
| 完整 R1+R2 CE/SGDG baseline | IMPLEMENTED / TINY_CUDA_VALIDATED | 实际两步训练与 Trainer 恢复一致；正式 100 步×8 的完整 CE 训练池尚未运行，distinct blocks 数另报 |
| SpinQuant_full_FP32_port | IMPLEMENTED / NUMERICAL_ONLY | 输出和 R1/R2 梯度误差约 1e-7；小模型训练执行。真实质量资格与效率结论 BLOCKED |
| 固定数据与实验后端 | IMPLEMENTED / G1_EXECUTED | 正式 G1 使用冻结输入与唯一 run_id，证据记录 10 份方法结果的 SHA-256。模型级协议尚待验证 |
| WikiText/C4 与六任务 | IMPLEMENTED / INTERFACE_VALIDATED | 固定 harness commit+六任务 revision/metric；六任务 local fixtures 实际跑通；真实分数 NOT_RUN |
| 配对分析 / G2–G4 构造 | IMPLEMENTED | token 加权 NLL 与 paired seed/document/item bootstrap；空结果准确不足，不生成表/图 |
| 实验 notebook | DEFAULT_PATH_EXECUTED / COLAB_FORMAL_G1_COMPLETE | 00/01 完成 A100 预检、连续 8 对诊断及正式 G1。01b 已在同一 A100 跑完两个种子的真实拟合面板诊断 |
| 目标 RTX PRO 6000 Blackwell / 优化 FHT | NOT_RUN / BLOCKED | 本机 RTX 3050 Laptop 4GiB；FHT、FlashAttention、Triton 缺失 |
| 2026 强基线、正式消融、论文主表 | NOT_IMPLEMENTED / NOT_RUN | 不能由官方论文数字或本地 fixtures 补填 |

前缀诊断现只采集第一个 seed 的 fit 面板，正式 G1 仍采集两种子的 fit/dev。耗时摘要增加含更新和 checkpoint 的完整方法外推。G1 的摘要、恢复与证据汇总会检查旧记录的实际完成对数。新准备的 G2 和正式矩阵显式冻结 CUDA 求解器，历史 NumPy 协议仍可复现。模型级 FP 审计增加单独的 Norm 融合测量和实际运行精度的输出偏差。G2 的三路 FP32 证据需要每个运行同时有融合及最终旋转测量。

## A100 正式 G1

运行 `2026-11-18T083041.405198_0000_381f882eaf` 使用 A100-SXM4-40GB、LLaMA-2-7B、层 0/15/31 的 q/up/down。每个 seed 有 8 份拟合与 8 份独立验证文档。两种子五方法全部完成，墙钟 1 小时 16 分 38 秒。数值按预注册方式先计算每个 seed、每个位置的相对改善，再按位置平均两个 seed，最后取九个位置的中位数。每个 seed、每个位置的最强对照取 STE、网格、固定尺度中最低的 NMSE。[逐位置数据](../evidence/colab_a100_g1_20261118/G1_distributions.json)

| 预注册指标 | 实测 | 门槛 |
|---|---:|---|
| 相对 Hadamard 中位改善 | 2.5689% | 准确率 ≥10%，效率 ≥5% |
| 相对最强匹配搜索中位改善 | 0.0172% | 准确率 ≥5% |
| 优于最强匹配搜索的位置 | 6/9 | 准确率 ≥6/9 |
| 在最强匹配搜索 2% 以内的位置 | 9/9 | 效率 ≥6/9 |
| 单位置最大退化 | 0.5386% | 共同门槛 ≤10% |
| 最快匹配搜索校准耗时 / CellQuant 校准耗时 | 0.2641× | 效率 ≥2× |

两个 seed 的九位置平均 held-out NMSE 相对 Hadamard 分别下降 4.710% 和 3.924%。第 0 层 up 相对 Hadamard 下降 21.163%，相对最强匹配对照下降 0.061%。三个 down 位置相对 Hadamard 分别下降 0.017%、0.023%、0.128%。每个位置在两 seed 平均后均有正收益，各 seed 单独与 Hadamard 比较均为 8/9 个位置胜出。[门槛证据](../evidence/colab_a100_g1_20261118/G1_evidence.json)

| 同预算下的动态量化拟合与验证 | seed 11 | seed 23 |
|---|---:|---:|
| CellQuant 拟合目标相对初值下降 | 7.362% | 6.232% |
| 角度网格拟合目标相对相同初值下降 | 6.940% | 5.792% |
| CellQuant 拟合终值相对网格终值下降 | 0.453% | 0.467% |
| CellQuant 九位置平均 held-out NMSE 相对网格下降 | 0.070% | 0.067% |
| CellQuant / 网格含拟合采集的校准秒数 | 724.24 / 192.70 | 737.88 / 193.47 |

CellQuant 在两个 seed 均接受 128/128 个更新，网格均接受 126/128 个。直接相对同一种网格方法，九位置中位改善为 0.0385%。此次角度搜索在拟合面板上取得的额外收益已经很小，迁移到独立文档后进一步缩小。共同门槛全部通过，两条收益路线均为 FAIL。当前以这份真实负结果决定暂停 G2 和全量投入。现有证据是冻结面板上的 RTN 结果，模型级 GPTQ、PPL 和六任务需单独实验。[按方法的观测](../evidence/colab_a100_g1_20261118/method_measurements.json) · [重算报告](../evidence/colab_a100_g1_20261118/G1.json)

若继续研究，一项可检验的假设是随机配对限制了 128 次更新可触及的误差。两种子分别调度到 4096 个坐标中的 250 和 245 个，输出侧只有入选行可以改变。先在拟合面板上比较原调度与相同数量高误差行覆盖的初始 down 重构误差能量，再检查这些行的配对对九个面板共同目标有多大可行降幅。仅在该诊断支持假设时考虑一次 G1 修正，用拟合数据决定一张共享坐标表，四种搜索共用它，排序耗时计入各方法预算。修正前冻结规则及新独立文档，维持原 128 对、九位置、一个共享 Q 和原门槛。再次 FAIL 则停止这条主线。[阶段决策规则](GATES_AND_PAPER_CRITERIA.md)

### 独立文档复核与下一轮拟合诊断

2026-11-18 从正式 01 notebook 中的十份逐块输出归档了 720 条误差与目标能量记录。两个种子各有 8 份独立验证文档，共 16 份。固定这两次拟合，按整篇文档对所有方法和位置共同重采样 10,000 次。相对 Hadamard 的 NMSE 中位降幅为 +2.5689% [1.4791%，3.0102%]，相对角度网格为 +0.0385% [−0.0052%，0.2657%]，相对逐位置最强匹配对照为 +0.0172% [−0.2054%，0.0482%]。最后一项在每次抽样内重新选择对照。这是事后、固定两种子的 95% 百分位区间。[块级摘录](../evidence/colab_a100_g1_20261118/heldout_blocks.json) · [全部对照](../evidence/colab_a100_g1_20261118/document_intervals.json)

01b 在 A100 对两个种子的真实拟合面板完成了 32 次配对比较，共 64 次单方法评估。每个角度均用显式更新后的共享 Q 重算九个面板。按初始残差选行后，两个种子的输出误差覆盖比例分别由 6.11% 升至 7.66%、由 6.02% 升至 7.52%。CellQuant 相对网格的可行单对平均降幅优势在 seed 11 从 0.00217 升至 0.00780 个百分点，在 seed 23 从 0.00787 降至 0.00609 个百分点。两种子共同扩大优势的条件未满足，按事先记录的决策停止这条配对修正，正式 G1 保持 FAIL。[A100 实测摘要](../evidence/g1_pair_diagnostic_20261118/a100_summary.json) · [诊断协议](G1_PAIR_DIAGNOSTIC.md)

最终本地回归 322 项通过，CPU 数学 PASS，RTX 3050 小面板双种子诊断完成。01b 默认入口在本地执行成功。[本地验证](../evidence/g1_pair_diagnostic_20261118/local_validation.json)

## 科学门槛

G0 的实测合成数学案例为 PASS。G1 的完整真实层证据为 FAIL，G2–G4 为 NOT_RUN。`configs/gate_rules.json` 的 SHA-256 为 `5d36290ca487795337300f3cccbc1c4d5a18d250639498bb15455ca5ad6a554e`。

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


## 2026-11-17 — 用户 Colab G1 运行中：真实 CUDA 驻留与资源阻塞

用户重新连接后的 01 输出已从预检查 BLOCKED 变为 READY，启动时 L4 空闲 21.8448 GiB。已保存的模型设备报告为全部参数在 cuda:0、cpu_offload=false，参数字节数 13476831232；约 94 秒完成全部 seed 面板采集并释放基础模型，面板继续驻留 CUDA。当前日志显示第二个 seed 的 STE 持续推进，不能再描述为仍停在显存预检查。

seed 11 的 Hadamard、STE、网格报告 COMPLETE；固定尺度搜索与 CellQuant 各约 5 分钟后报告 RESOURCE_BLOCKED，之后继续运行 seed 23。具体 solver.reason 尚未读取，不能据时长断言一定是某种超时或数值错误。按 seed 11 耗时估计用户报告 32 分钟时剩余约 15–20 分钟，这是条件性结束时间估计，不是成功或 G1 PASS 保证。必须保留原始结果并检查阻塞原因，门槛未修改。

以上仅来自用户保存的 notebook 日志，并非本机执行或独立读取云端结果。该次观察最后保存时间为 2026-11-17T08:45:01.726457+00:00，原始副本已脱敏归档到 evidence/g1_colab_progress_20261117/58e803d3ccbc/；没有改动 notebook、代码或正在运行的配置。真实 G1 尚无完整合格证据。


## 2026-11-17 — G1 首对超时确认、数值路径优化与独立诊断入口

用户本轮 L4 运行于 2919.83 秒（约 48 分 40 秒）结束。读取用户补充的逐方法摘要：seed 11/23 的 STE 与 grid 都完成 128 对；CellQuant 与 fixed-scale 都为 `Pair wall-clock budget exceeded: 300s`、完成 0 对。这是事件搜索在预算内未完成，不是显存预检查失败，也不足以判断方法的量化收益。原始 notebook 和用户结果观察保存于 `evidence/g1_resource_blocked_20261117/`；未直接访问其云端 result 文件。

修复顶层 `reason:null`：汇总逐 seed/method 的真实失败原因、完成对数和资源诊断，CLI/notebook 直接显示；保留非零退出码与 RESOURCE_BLOCKED。事件求解增加阶段、事件数、扫描区间数与失败 checkpoint 信息。GPU 事件索引一次排序取代逐边界全表扫描；三角多项式用中点导数上界排除无根小区间，近零/切根仍走原 FP64 QR。没有改变角域、维度、量化规则、单对 300 秒、事件 cap 或科研门槛。

新增 `--profile-pair` / 01 的 `RUN_PAIR_PROFILE=True`：复用正式加载、融合和完整真实面板，只诊断第一个 seed 的第一对，对固定尺度与 CellQuant 分别执行；输出 PAIR_PROFILE_ONLY/PROFILE_COMPLETE 或实际阻塞，不生成 held-out 成绩/G1 证据，G1 聚合器明确拒绝这些记录。该工程入口保留原方法预算，且明确单对完成不等于 128 对的完整方法可完成。新的真实 L4 诊断尚未执行。

实际本地验证：完整 213 passed, 1 warning in 30.43s（仅 Windows pytest 缓存写入权限警告，exit 0）；六本 notebook schema/语法通过，只清除修改单元输出，其他用户单元保持。4 个同 GPU 小合成对照前后最终目标/预算一致，耗时降低约 1.33–3.53 倍，仅代表这些小工程案例。完整 4096×11008、128 tokens、±π/4 的单输出合成面板产生 48725 区间，在 30 秒诊断预算内只处理 979 个后 RESOURCE_BLOCKED；该限制被保留，没有算作 PASS。可见完整事件遍历吞吐仍不足，尚未解决真实 G1 的执行瓶颈，不能宣称优化后正式 G1 已可运行完毕。

下一步先备份原 runs，再同步代码、重启同一服务器 kernel，使用新单对诊断入口定位真实面板的阶段/吞吐；保留原配置，不混用旧源码断点或直接重跑完整 sweep。G1 仍为 INSUFFICIENT_EVIDENCE。代码未提交/push，AutoDL/G2–G4 未运行。

## 2026-11-17 — 真实 Colab 单对诊断再次超时

用户已执行新的 `RUN_PAIR_PROFILE=True` 入口。保存的 01 日志中，run `2026-11-17T102020.290922_0000_29bbe398e6` 于 10:32:14 UTC（北京时间 18:32:14）结束，总耗时 712.44 秒，状态 `RESOURCE_BLOCKED`。seed 11 的第一对 `[547, 526]`：固定尺度搜索在 300 秒内处理 19958/148902 个区间（13.40%），CellQuant 处理 16236/151940 个区间（10.69%），两者均触发原有单对时间上限，未完成一个坐标对。这些是工程诊断，不是 G1 质量成绩；G1 保持 `INSUFFICIENT_EVIDENCE`。

该次模型放置报告确认所有参数在 `cuda:0`，共 13476831232 bytes，`cpu_offload=false`。93.74 秒采集完所有 seed 面板后，代码显式释放基础模型，面板仍在 CUDA。CellQuant 搜索超时时活动张量占 5.458 GiB、缓存分配器保留 7.395 GiB，因此后期约 8 GB 的驱动显存占用符合当前流程。显存下降不能用于判断是否改用 CPU，也不能用于推断计算吞吐。

事件已经枚举并排序；当前实现逐区间执行系数更新、根求解和实际边界舍入，Python 循环及每区间多次 CUDA 同步仍存在。日志证实区间遍历速度不足，但没有内核级 profiler 数据，不能精确分摊各算子的耗时。后续须优化批量处理和同步，再验证完整维度的数值一致性与真实单对吞吐；现有优化尚未解决真实 G1 的预算问题，直接重跑全量不能视为修复。

原始结构化日志和解析记录：`evidence/g1_pair_profile_colab_20261117/progress.jsonl`、`observation.json`。来源仅为用户保存的 notebook 输出，未独立读取云端结果文件或验证云端源码 revision。本轮只归档证据、更新文档，未修改求解代码、notebook、预算或门槛，未重新执行测试或云端实验。

## 2026-11-17 — GPU 批量区间计算与完整维度验证

针对真实首对 300 秒超时，新增 `tensor_cells.py`，引擎标识 `torch_batched_output_prefix_v1`。输出侧按 owner 区间批量累加编码投影的变化；稳定/固定解码尺度的输入保留完整原乘积，只重算两个变化通道的外积；固定解码尺度的风险仍跟踪动态 AbsMax。不稳定输入继续用完整双侧增量系数，完整行边界以已有乘积与编码差直接计算点值，避免来回重建整组多项式。所有面板的系数在共同事件并集上合并后统一选角，保持一个共享 Q、原事件集合、FP64、原角域、300 秒单对和方法总预算。

先复现原有 **213 passed, 1 warning in 40.67s**；最终全套 **241 passed, 1 warning in 34.57s**。警告仅为 Windows pytest 缓存写入权限，退出码为 0。新增对照涵盖输入/输出、动态/固定尺度、完整/多组、不同批次大小、同时事件、零尺度、实际边界与物理风险。最初四项输出边界测试失败，定位为 AbsMax 与代入的浮点运算顺序改变了 tie 舍入；已按实际量化器顺序修复，未放宽测试容差。CPU 数学检查通过，空 G1 仍返回 `INSUFFICIENT_EVIDENCE`，门槛 SHA256 不变。

本机 RTX 3050 Laptop 的最终 CUDA 合成记录（同一工作站，均非预训练模型）：

| 案例 | 全部区间数 | 单对搜索及复核 | 含初始化/接受更新的方法总耗时 | 状态 |
|---|---:|---:|---:|---|
| 完整 4096×11008 输出，128 tokens，±π/4 | 48725 | 6.75 秒 | 13.99 秒 | COMPLETE |
| 六个 64×4096 输入 + 一个完整输出，动态尺度 | 54785 | 78.63 秒 | 86.41 秒 | COMPLETE |
| 同一混合面板，固定解码尺度 | 52304 | 6.96 秒 | 14.34 秒 | COMPLETE |

峰值分配显存约 2.77/2.88/2.88 GiB。串行旧版在同一输出案例的 30 秒诊断上限仅完成 911/48725 区间，没有完成耗时，故不报告完整搜索加速比。中间版本的混合案例在额外设置的 30 秒诊断上限阻塞，保留该失败记录；正式 300 秒上限未改变。最终动态混合案例仍偏慢，不能将单对完成推断成全部 128 对符合方法预算，更不能推出真实质量或效率收益。

命令、失败和成功日志、源文件前后 hash、完整报告见 `evidence/batched_cells_20261117/validation.json`。最终三项 GPU 运行前后源码一致。真实 L4/7B 新版首对、完整 G1、AutoDL 96GB 与 G2–G4 **NOT_RUN**；本机显存不足以加载 7B，且未直接连接用户 Colab。更新源码后按 `docs/COLAB_GPU_FROM_SCRATCH.md` 重启同一 kernel，以 `RUN_PAIR_PROFILE=True` 或 `python -u -m cellquant.experiment --config runs/prepared/g1_gpu/config.json --profile-pair` 测真实首对。Notebook 配置/输出未修改，代码未提交/push。

## 2026-11-17 — 用户真实首对完成，进入连续前缀预算诊断

最新保存的 01 为首对诊断。run `2026-11-17T114339.123212_0000_991980b138` 于 11:46:01 UTC（北京时间 19:46:01）返回 `PROFILE_COMPLETE`，总耗时 140.93 秒。固定尺度完整处理 148902 区间，CellQuant 完整处理 151940 区间，两者均 COMPLETE；模型全参数 CUDA、无 CPU offload。来源为用户 notebook 输出，未独立读取云端 profile_result.json 或重算成绩。

CellQuant 在 19.7602 秒进入最终复核；阶段耗时为事件生成 2.3300 秒、排序 0.0054 秒、系数及边界 11.4937 秒、候选搜索 5.3669 秒。这不包含最后的复核/接受更新。有效方法预算由本次日志确认是 1440 秒。按仓库正式 128 对协议及首对搜索时间外推约 42.16 分钟，超过 24 分钟且尚未包含其他开销；这是条件性外推，不是确定运行时间或置信区间。最新远程 config.json 未独立读取，128 来源为仓库正式协议。PROFILE_COMPLETE 不等于 G1 PASS，尚不进入 02。

新增 `--profile-pairs N` 和 01 的 `PROFILE_PAIR_COUNT`，当前建议 8：对第一个 seed、两个事件方法连续执行原 schedule 的前 8 对，同一个 Q 连续更新；正式 pair_budget、方法/单对预算、面板、角域和门槛均不改。结果保留 `PAIR_PROFILE_ONLY`，多对完成也被 G1 聚合器拒绝。`budget_analysis` 报告实测均值、搜索部分条件性外推和遗漏开销，不产生 full-run READY 或科学 PASS。非法计数在模型/预检查工作前拒绝。

01 末尾附加诊断原来固定引用 08:10 的旧失败 run，现改为只读 `PILOT_RUN_DIR` 的本次摘要。原 notebook 已脱敏归档；仅修改相关配置、调度、摘要和说明单元，其他单元保留。实际 34 项针对性测试通过；全套 **257 passed, 1 warning in 62.50s**，警告仅 Windows pytest 缓存权限。真实 CUDA 小面板的连续 8 对已执行，验证原 schedule、配置不变及 G1 拒绝；六本 notebook schema/语法通过。

记录见 `evidence/colab_profile_complete_20261117/`。新的真实 L4 连续 8 对及完整 G1 尚未运行，本机无法独立加载 7B；同步源码后在同一服务器重启 kernel 并初始化 01，设置 `RUN_REAL_MODEL=True`、`RUN_PAIR_PROFILE=True`、`PROFILE_PAIR_COUNT=8`，运行原执行单元及最后的当前运行摘要。无需重建服务器或重新下载缓存。本轮修改未提交/push；G1–G4 仍缺正式合格证据。

## 2026-11-17：真实连续 8 对完成，优化动态输入更新

最新用户保存的 01 对应 run `2026-11-17T120623.229443_0000_083c7593b5`，于 12:12:19 UTC（北京时间 20:12:19）结束，状态 `PROFILE_COMPLETE`，共 354.38 秒。两种方法均完成 seed 11 的连续 8 对，结果保存的正式对数为 128、方法预算为 1440 秒。该观察仅来自 notebook 输出，未独立访问 Colab 的结果文件。

| 方法 | 完成对数 | 方法总耗时 | 搜索均值/对 | 128 对搜索部分的条件性外推 |
|---|---:|---:|---:|---:|
| fixed_scale_search_matched | 8 | 83.18 秒 | 8.74 秒 | 18.65 分钟 |
| cellquant | 8 | 175.85 秒 | 20.33 秒 | 43.36 分钟 |

CellQuant 的均值外推超过 24 分钟原预算，且不含初始化、接受更新和 checkpoint 等开销；不是完整方法已经实际超时的记录。该 run 没有 held-out 或完整五方法成绩，G1 保持 `INSUFFICIENT_EVIDENCE`。动态输入系数/边界阶段在慢坐标对占 11.55–23.93 秒，是本次具体优化目标。

新增 `tensor_input_cells.py`：完整行分组的少数不稳定输入行/列在 GPU 上批量解码所有组成员，使用整数编码差前缀投影，并完整保留双侧交叉项；两旋转成员以秩二项补回。胞元内损失和风险仍解析构造，边界实际 rounding；非完整行/不稳定组密集时保留原增量路径。引擎版本更新为 `torch_batched_bilateral_prefix_v2`，禁止把旧 checkpoint 当新实现恢复。正式 128 对、±π/4、FP64、300 秒单对、1440 秒方法预算及门槛均未修改。

实际验证：先复现 **257 passed, 1 warning in 57.73s**；针对性检查 **50 passed in 31.77s**；最终全套 **281 passed, 1 warning in 83.22s**。警告仅 Windows pytest 缓存权限，退出码 0。新测试在 CPU 和实际 CUDA 上覆盖单侧/双侧不稳定、完整 4096 维、不同批次、零尺度/同时边界、原增量系数与直接量化对照、连续两对共享 Q。CPU 数学检查 PASS；空 G1 正确返回 INSUFFICIENT_EVIDENCE。

本机 RTX 3050 Laptop、FP64、相同随机数据的完整维度混合案例（六个 64×4096 输入面板和**一个** 4096×11008 输出面板，128 tokens，±π/4）实测如下。两者均处理 54785 个区间并 COMPLETE，最终角度、实际损失、两项风险值相同；每个仅运行一次，不作统计置信区间或 L4 速度保证。

| 实现 | 系数/边界 | 完整单对搜索 | 方法含初始化/更新 | 峰值分配 |
|---|---:|---:|---:|---:|
| 原输入增量路径 | 31.17 秒 | 49.77 秒 | 57.29 秒 | 2.8813 GiB |
| 输入批处理 | 1.21 秒 | 20.10 秒 | 27.61 秒 | 2.8813 GiB |

原始观察、483 条进度事件、脱敏 notebook、同卡对照、测试与源码哈希见 `evidence/input_batch_20261117/`。只修改 01 的说明单元，用户代码/开关/输出保留。下一步同步本轮源码后在同一 Colab kernel 重启并初始化 01，以 `RUN_PAIR_PROFILE=True`、`PROFILE_PAIR_COUNT=8` 重测新引擎；日志应能看到 `batched_dynamic_input`。新版真实 L4、完整 G1 和 AutoDL canary 均为 NOT_RUN，不进入 02 的正式模型 canary。本轮修改未提交或 push。
