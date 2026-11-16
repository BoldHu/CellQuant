# 实现状态与证据边界

更新：2026-11-16。以下“已实现”描述代码；实验结论单独列出。整个研究尚未完成。

| 组件 | 代码状态 | 已执行的证据与限制 |
|---|---|---|
| 原 CPU 参考与基础设施 | 保留并修正零组物理 scale 风险 | 原 62 项测试复现；最终全套 116 项通过 |
| 共享 global Q、多面板输入侧 WQ.T/QX、输出侧 QB | IMPLEMENTED | FP64 增量胞元系数、风险约束、实际边界 rounding；3 seeds/24 cases 合成 G0 PASS |
| 完整维度 | IMPLEMENTED / REAL_PERFORMANCE_NOT_RUN | d=4096、down=4096×11008，2 tokens、1 pair、±0.0001 的工程 smoke COMPLETE；不是正式吞吐 |
| 筛选与事件增量 | IMPLEMENTED_CPU | scale-stable 输入组筛选、输出完整行 signed hull；按实际事件收费并记录阶段耗时 |
| CUDA dense GEMM/量化 | IMPLEMENTED / SMALL_CASE_VALIDATED | 本机 CUDA 小共享求解器和 4096 元素量化一致性检查；全 GPU 增量事件 kernel NOT_IMPLEMENTED |
| Hadamard / angle-grid / Givens STE / fixed-scale | IMPLEMENTED | 相同面板、坐标预算、约束、墙钟上限；小矩阵验证。grid/STE 中途 optimizer 恢复 NOT_IMPLEMENTED |
| 真实层采集与 Colab G1 | IMPLEMENTED / REAL_7B_NOT_RUN | 逐层 GPU 调度，q/up 可抽输出行、o/down 完整 hidden；小随机模型已检查采集形状 |
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
