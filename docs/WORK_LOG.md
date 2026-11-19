# 工作记录

## 2026-11-15 — 初始交付

- 将上一轮研究方案转成 Codex 项目规范与 Colab→AutoDL 执行路径。
- 建立明确的 correctness / pilot / canary / full / paper 门槛。
- 本环境无 PyTorch/CUDA；只执行 CPU 数学、runtime、gate 与 notebook 默认路径测试。
- 当前不存在真实 7B/8B 结果。后续实验必须生成自己的完整 manifest 和逐样本证据。

## 2026-11-15 — 交付前冻结与验证

- 62 项 CPU 单元测试通过，覆盖数学参考、门槛、运行迁移和统计原语。
- 修复孤立最大值并列导致的 scale owner 错误，并加入回归检查。
- 重新执行最终 5 本 notebook 的 32 个默认 CPU 代码单元，源单元与执行副本逐项一致。
- 实际 notebook 引擎为 IPython inprocess；未验证真实 Colab/AutoDL GPU、外部 Jupyter kernel 或完整 LLM 后端。
- 6 个 canary、32 个核心 full 作业草案齐备；正式模型/数据/协议 revision 仍需真实解析并冻结。
- 交付验证细节见根目录 `VALIDATION_REPORT.md`；现有证据不代表完整 G0–G4 已通过。

## 后续记录模板

- 日期/里程碑：
- 代码 commit 与配置 hash：
- 实际执行命令/硬件：
- 输出目录与关键文件：
- 成功/失败/不足证据：
- 下一步及仍然未验证的假设：

## 2026-11-15 — 本轮构建与实际验证

- M0：读取规范并复现原 62 项测试。发现可用宿主 Python/Torch 和 RTX 3050 Laptop 4GiB，修正“无 PyTorch/CUDA”的旧环境假设；项目 venv 保留宿主 torch。根目录是无 .git 的交付目录，记录 source hashes，不伪造 commit。
- M1–M3：实现完整组的共享 Q 输入/输出事件求解器、signed support hull、安全筛选、解析损失/预算系数和增量更新、匹配 grid/STE/fixed-scale、逐层采集与 G1。实跑 3 seeds / 24 cases G0 PASS；完整 d=4096 / FFN=11008 的少 token、窄角域 smoke COMPLETE。正式区间 cap 超限仍阻塞。
- M4：固定 SpinQuant commit，tracked tree 保持 clean；生成带 hash 的 v2 overlay，关闭 embedding centering、保留 gamma，R1=Q.T 与全部 R2。实际 tiny MHA/GQA FP/GPTQ/RTN 恢复通过；GPTQ 中断后恢复参数与连续运行差 0。原版 FP64/明确 FP32 port 的 tiny CE/SGDG 均执行，FP32 只有数值验证。
- 修复实际测试暴露的问题：Transformers 4.44.2 API、GPTQ catcher/seq length/global RoPE、逐层恢复参数、Torch 2.6 RNG pickle 默认限制；Trainer 改用安全 tensor/primitive RNG 与文件 hash，恢复参数逐项一致。
- M5：固定 harness commit、六任务 metric 和 dataset revision；本地 fixtures 真实执行六任务。修复任务扫描开销、Windows 多进程 bootstrap 兼容与 dtype 序列化；由自己的 paired document/item bootstrap 提供 CI。逐题 choice 与 harness metric 一致性执行校验。
- M6–M7：统一 experiment contract、分片 SHA 校验、独立文档划分、全矩阵 provenance、G2/G3 实测证据构造、G1 portable bundle、日志/心跳/资源采样。完整 SpinQuant 保留 100×8 CE 预算，使用排除 dev 的完整训练池，不复用小 fit 池冒充完整训练数据。800 是 microbatch 预算；按上游 Trainer 语义可遍历整个池跨 epoch，独立 blocks 数另报。
- 最终 116 项测试通过，五本 notebook 默认路径执行成功；原始日志、环境与结果在 runs/build_final，归档在 evidence/build_20261115。门槛文件 hash 与初始包相同，G1–G4 仍无真实通过证据。
- 剩余：预训练 7B/8B、Colab G1、Blackwell/FHT、正式 GPTQ/下游/效率、近期 2026 强基线、消融与新颖性全文/代码审查；全 GPU 增量事件 kernel、grid/STE 中途 optimizer 恢复未实现。下一步命令与硬件需求见 RUNBOOK。

- 收尾核对发现 Agent.md 同步副本缺失，已运行 scripts/sync_agent_alias.py 恢复，确认与未改动的 AGENTS.md 逐字节一致；原 gate_rules.json 的初始包 SHA 复核一致。正式运行增加 clean 根代码 commit 检查，当前仅 source snapshot，不编造 commit。

- 最终中断一致性审查：修复 Q 更新与 factor/log 之间被中断时可能保存不一致 checkpoint 的窗口；新增注入中断回归，完整套件现为 116 项通过。

## 2026-11-16 — GitHub / VS Code / Colab / Drive 工作流

- 接手检查：用户已建立根 Git checkout，origin=`https://github.com/BoldHu/CellQuant.git`，master 初始 commit `9a53bf7`，开始时工作树干净。按请求本地实现工作流，本轮未自动提交或 push。
- 查阅 Google 官方 Colab 扩展指南与 Marketplace：使用远程 notebook kernel 和扩展的 Drive 挂载命令；terminal/server mounting 的实验状态保留。新增初始化 notebook 与扩展推荐，原五本 notebook 支持 clone 路径及独立 kernel 的路径恢复，保留 ZIP 入口。
- 新增 `cellquant.workspace`：本地 HF hub/datasets/runs 分区、白名单环境变量、独立凭据路径；内容寻址的 Drive 对象、版本索引、原子传输、恢复 SHA-256、冲突/路径逃逸拒绝、源文件变化/中断不发布成功索引。阶段备份须停止写入，不宣称实时保存或 checkpoint 一致性证明。
- `prepare_experiment.py` 与模型后端统一解析缓存路径，不再将 HF_HOME 改成 model cache 子目录；显式冻结路径仍优先。新增 `cache_model.py`，复用已有分片校验器，只下载固定 revision、检查大小/哈希/header，不加载 LLM、不生成 gate。
- 实际验证：全套 **125 passed in 20.19s**；CLI 最终补充默认配置读取与旧快照选择后，相关 **25 passed in 7.01s**。6 本 notebook 默认路径用 IPython inprocess 运行成功；6 条 CLI 在本地目录完成初始化、两次去重备份、指定快照恢复和状态查询。数学 notebook 重新计算 G0 为 PASS，G1–G4 未获得新的真实证据。
- 开发检查曾发现一处 cache 参数括号错误，已在测试前修正。修正旧记录中归档目录日期笔误为 evidence/build_20261115；不改变历史数据。`configs/gate_rules.json` SHA-256 仍为 `5d36290ca487795337300f3cccbc1c4d5a18d250639498bb15455ca5ad6a554e`。
- 运行记录位于 `runs/workspace_20261116/`，小型验证包位于 `evidence/workspace_20261116/`。Google 登录、Drive FUSE 挂载/吞吐、VS Code 远程会话、真实模型下载和云端 GPU G1 均 NOT_RUN；本地文件 fixture 不能替代它们。

## 2026-11-16 — 首次挂载引导修正

- 用户的云端 notebook 已运行到 `configure_workspace`，错误指向缺少 `/content/drive/MyDrive` 挂载入口，用户将其理解成缺少项目文件夹。现有代码本就会在挂载后创建 `CellQuant`，没有删除或绕过挂载检查。
- 补充错误信息、初始化 notebook 说明与操作文档：VS Code 挂载命令将单元追加在末尾，必须先单独运行并完成授权；之后重跑初始化即可。保留用户修改的配置、执行次数和原始报错输出。
- 本地执行 `python -m pytest -q tests/test_workspace.py`：9 passed in 9.11s；notebook schema 校验通过。记录在 `runs/workspace_first_run_20261116/`，归档于同名 evidence 目录；未代用户完成 Google 授权或运行云端 GPU。

## 2026-11-16 — 将挂载命令直接写入 notebook

- 核对 Google 官方 `googlecolab/colab-vscode` commit `d238503acf565e3ee5edf71df4263cbd8af01fa9` 的 `src/colab/commands/register.ts`：扩展命令插入 `from google.colab import drive` 和 `drive.mount('/content/drive')`。据此六本 notebook 直接放入同一挂载单元，位于配置之后、clone/解压/路径初始化之前。
- 在 Colab 中默认启用挂载，本地和 AutoDL 默认跳过；保留首次 Google 授权交互、未挂载时的失败检查，不捕获并吞掉授权错误。浏览器专用旧开关改为通用 `MOUNT_DRIVE`。保留用户已设置的依赖安装、G0、恢复和备份控制；被改写代码单元的旧输出清除，原文件在 `runs/notebook_mount_20261116/before/` 留存。
- 六本 notebook 的结构/语法/挂载顺序检查以及调用、缺少挂载、授权失败的离线 fixture 检查通过；`tests/test_workspace.py` 为 9 passed in 4.15s。归档见 `evidence/notebook_mount_20261116/`。实际 Google 授权、Drive 挂载和 GPU 实验没有在本机执行；本轮不更新任何科学门槛或真实实验结论。

## 2026-11-16 — 读取用户首轮 Colab 执行记录

- 用户报告 00 到 01 三本 notebook 执行成功；检查实际保存的输出，确认显示 Drive 已挂载、runs 已备份、G0 PASS；01 的执行开关仍为 False，preflight BLOCKED、真实模型 NOT_RUN、G1 INSUFFICIENT_EVIDENCE。正常结束的 notebook 不等于真实 G1 通过。
- 环境记录为 L4 / Python 3.13.16 / Torch 2.14.1+cu130 / Transformers 5.22.1；FHT 缺失、模型与数据 revisions 未准备。适配器要求 Transformers 4.44.2；PyPI 上固定 tokenizers 0.19.1 提供 cp312 wheel，未列出 cp313 wheel。查阅 Colab 官方 runtime FAQ，2026.07 提供 Python 3.12.13，可作为下一步依赖安装的候选环境，GPU/完整依赖仍需实测。
- 三本用户执行 notebook 与观察摘要保存在 `evidence/colab_first_session_20261116/`，保持原始输出和哈希，不修改其 source 或执行结果；未从 stdout 生成 G1 通过证据。下一步先备份当前 runs，再准备兼容环境、固定 SpinQuant/FHT 和真实数据。

## 2026-11-16 — 恢复现有 runs 时保留两份记录

- 从用户保存的 workspace notebook stdout 确认真实原因：已有 `runs/00_math/environment.json` 与 Drive 快照不同，触发 `FileExistsError`，并非缺少云盘目录。开始时仅该 notebook 有用户改动，根 commit 为 `8eb21e72a3396c7462ac45917553f3711a21cae6`。修改前逐字节归档，保留未修改代码单元的输出及用户执行开关。
- `restore_area(..., on_conflict="separate")` 与 CLI `--runs-conflict separate` 只适用于 runs：冲突时完整快照另存到哈希命名目录，默认策略与模型/数据缓存仍拒绝覆盖；重跑校验原副本，损坏文件/路径逃逸仍报错。当前结果与历史快照不混合，不重写配置或授予续跑资格。Notebook 新增选项与说明，失败信息附真实子进程日志尾部和退出码。
- 执行 `.venv/Scripts/python.exe -m pytest -q tests/test_workspace.py --basetemp runs/restore_conflict_tests_01`：**12 passed in 6.41s**。首次测试命令的临时目录父目录不存在，12 项均在 fixture setup 报错，未执行测试主体；原日志单独保留，改用现有 runs 下的新临时目录后通过。
- 执行 `evidence/workspace_restore_conflict_20261116/validate_workflow.py`：notebook schema/语法、用户配置与未改动单元保留、默认冲突拒绝、实际 notebook 恢复调用、两份原字节、失败详情传播、多区域 CLI 均通过。本地存储 fixture，不连接 Google Drive，不运行 GPU 模型。原始反馈、执行日志与 hash 归档在同名 evidence 目录。
- 门槛 SHA 保持 `5d36290ca487795337300f3cccbc1c4d5a18d250639498bb15455ca5ad6a554e`。用户最新 pip 日志仍为 Python 3.13，兼容环境切换与真实 G1 继续待执行。本轮未提交/push，修复需要同步到云端 checkout 后重跑。
- 收尾审查发现分析 CLI 会递归扫描 `runs`，将另存目录放到项目根目录的 `restored_snapshots/runs/<快照哈希>/` 并加入 `.gitignore`，防止历史结果被收集进活动分析。调整后以 `--basetemp runs/restore_conflict_tests_02` 重跑，**12 passed in 4.26s**；notebook/CLI fixture 再次通过。调整前日志保留于 `before_runs_isolation/`，当前验证清单记录最终源码 hash。

## 2026-11-16 — 确认云端恢复与 G0，定位真实 G1 前置项

- 用户报告前三本 notebook 均通过；读取最新保存输出，确认此前修复已在云端 commit `45b2a8659b33d130653030e8165537692b399212` 生效：Drive 已挂载、RESTORED_SEPARATELY 返回历史路径，G0 与归档完整性 PASS。
- 01 仍是默认不执行模型的模式，8 项准备缺失，包括固定 revisions、权重来源元数据、两种子真实 tokens、明确 dtype、SpinQuant overlay；真实 G1 NOT_RUN/INSUFFICIENT_EVIDENCE。Python 3.13.16 / Transformers 5.22.1 还未切换，FHT 未安装。
- 原始三本 notebook 未改动，逐字节复制到 `evidence/colab_after_restore_20261116/` 并记录 SHA 和实际缺失项。查阅 Colab 官方 runtime FAQ 与 VS Code 扩展 PR #424，补充备份、选择 2026.07/Python 3.12、确认版本和固定依赖的操作顺序。
- 本轮仅归档观察和更新操作文档，没有修改核心算法、notebook、门槛或运行新 CPU/GPU 实验。下一步在用户 Colab 服务器准备兼容环境和真实 G1，G1 合格后才进入 AutoDL canary。

## 2026-11-16 — Python 3.12 与固定 HF 依赖已在用户 Colab 安装

- 读取最新三本 notebook 输出，确认 Python 3.12.13 / Torch 2.11.0+cu128 / Transformers 4.44.2 / datasets 2.20.0 / tokenizers 0.19.1；仍是 L4，G0 PASS。云端 clean commit `4732c827891d012a776bc18e09ce890ebf20e88b`，恢复返回 RESTORED，pip 安装退出 0。保留 gradio 与 huggingface_hub、gcsfs 与 fsspec 的冲突提示，不自动升级破坏固定依赖。
- 真实 G1 尚未运行，01 仍报告 8 项前置缺失及 RUN_REAL_MODEL=False。三本 notebook 未修改，原字节和 SHA 归档于 `evidence/colab_python312_20261116/`。
- 核对 Dao-AILab/fast-hadamard-transform 官方 v1.1.0 tag，实际 commit 为 `1cc807efbd6cc001df359822d60bf6052dd66859`；阅读该版本 setup.py/README，提供从固定 Git 源码安装、保留宿主 torch、CUDA/FHT 实测和固定 SpinQuant tiny smoke 的下一步命令。此 FHT 在当前 L4/Torch 组合尚未安装或实测，不能从上游支持列表推断兼容通过。
- 本轮没有修改求解器或门槛，没有代用户运行云端 GPU；接下来应先运行后端兼容检查，再授权/准备并缓存真实模型与数据，最后在 01 显式执行真实 G1。

## 2026-11-16 — 用户 Colab CUDA/FHT 与 tiny SpinQuant/GPTQ 通过

- 检查用户最新三本 notebook 输出：FHT 1.1.0 安装完成，CUDA/FHT PASS；tiny SpinQuant 返回 PASS_TINY_STRUCTURE_ONLY，融合相对 RMS 1.068972578044603e-07、旋转相对 RMS 1.7605411237572577e-07、GPTQ EXECUTED_FINITE，退出码 0。记录为用户保存的 GPU/合成结构验证，不冒充在本机重新执行或预训练 7B 结果。
- 01 仍使用未准备的草案配置且 RUN_REAL_MODEL=False；当前 7 项缺失，overlay 已就绪，真实 G1 仍 NOT_RUN/INSUFFICIENT_EVIDENCE。下一步准备 immutable revisions、真实 fit/dev tokens 和权重，备份到 Drive，再运行真实 G1。
- 原始 notebook 未修改，逐字节归档至 `evidence/colab_backend_20261116/` 并记录 SHA；保留此前尚未提交的 Python 3.12 状态记录。补充跨 notebook kernel 的 HF 登录方式，核对本地 huggingface_hub 0.36.2 的 login 与 HF_TOKEN_PATH 行为；凭据保留在 VM 本地，不进入 Drive 快照。
- 本轮仅读取和归档结果、更新状态及后续操作说明；没有运行新的 CPU/GPU 算法实验，没有改动任何科学门槛。

## 2026-11-16 — 全局 HF_TOKEN 与明确的模型权限检查

- 读取用户粘贴的错误附件，根因是 403/GatedRepoError，服务端明确提示当前账号不在 Llama-2-7b-hf 授权名单；force_download 又将它包装为泛化 ValueError。该错误不说明需要更换 Python，也不能靠改变令牌输入方式授予权限。
- 按请求在 workspace 顶部添加空字符串全局 HF_TOKEN，替换 getpass 登录单元；统一设置进程环境和 VM 登录缓存，显示当前账号，调用 huggingface_hub 0.36.2 实际存在的 HfApi.auth_check，再读取配置文件。保留真实 401/403，不进行权限绕过或自动重试；文件中未填写真实令牌。
- 修改前 notebook 归档于 `evidence/hf_token_variable_20261116/`，归档脚本检查并按需脱敏 hf_ 令牌模式。仅清除被修改代码单元的旧输出，其他用户配置、单元和输出保留。Schema/所有代码语法/空 token 联网前阻止/未改动单元保留检查通过，没有实际登录 HF 或执行新模型实验。门槛不变，真实 G1 仍未运行。

## 2026-11-16 — 修复权限检查与真实模型输入准备之间的缺口

- 读取用户最新 notebook：auth_check 成功并读取 609 字节 config.json，保存的 workspace 尚未包含真实数据准备或完整权重校验步骤；01 的 RUN_REAL_MODEL=True，但预检查仍缺固定 revisions、分片元数据、两 seed tokens 和 dtype。依据这些记录可确认输入准备未就绪，不能据用户“模型下载好了”就移除 guard。
- workspace 顶部增加 PREPARE_G1_INPUTS=False；HF 登录后新增调度单元，依次调用现有 prepare_experiment、preflight、cache_model 和 Drive backup。完整配置存在则保持冻结内容并复核权重；失败残留不覆盖，任何失败都不会打印准备完成。01 显示实际配置，真实执行时缺准备文件直接报错，不再回退草案；BLOCKED 错误列出具体原因。用户 RUN_REAL_MODEL=True、HF 配置和未改动输出均保留。
- 新增 notebook 调度回归检查，与 workspace 测试合计 **22 passed in 8.93s**。首次有 1 项失败：测试用不唯一的 if RUN_REAL_MODEL 字符串选中了预检查单元，已改用执行单元唯一标记；不是通过修改产品行为或科学门槛修复。原日志保留为 pytest_initial_cell_selector_failure.log。pytest 缓存目录权限提示通过将测试缓存放在 runs 下解决，不改变系统 ACL。
- 两本 notebook schema、全部代码语法、未改动单元及用户配置保留、gate_rules SHA 未变检查通过；记录位于 `evidence/g1_input_preparation_20261116/`，归档按需脱敏凭据。测试为本地控制流/存储 fixtures，未运行真实下载、云端模型或新科学实验。下一步：用户在 workspace 显式打开准备开关、完成准备/权重校验/Drive 备份，再在 01 重跑真实预检查与 G1。

## 2026-11-16 — 排查加载后无输出，增加 G1 实际阶段与资源心跳

- 读取最新用户 notebook，01 已 READY，L4/固定依赖，两个权重分片加载完成。用户报告已运行约 10 分钟、占用系统 RAM 约 28GB、GPU 几乎空闲；尚无实际 PID/CPU 采样及完整结果，不能断言死锁、成功或资源足够。归档三本 notebook 的当前状态，凭据脱敏，保留原文件 SHA；没有读取远程文件或重新执行预训练模型。
- 审阅真实调用路径：CPU 驻留模型→CPU Norm 融合→CUDA 离线旋转→逐层 GPU 面板采集→默认 CPU 匹配求解与 held-out。旧日志仅有第三方输出，state.json 在 G1 主体期间可能停留 fusion。新增 RunProgress、实际阶段/层/面板/坐标对回调、30 秒心跳和只读 diagnostics CLI；心跳与实际进度分别计数，PID 创建时间防止误认历史作业。01 改为独立日志和 -u，并加入同一服务器独立终端/kernel 的诊断说明。算法、维度、dtype、device、预算和门槛不变。
- 相关回归初次 79 项通过；增加实际采集回调 fixture 后，Windows 高频读取与原子替换出现瞬时 PermissionError（1 failed / 9 passed），限定仅 Windows PermissionError 重试后，最终 **80 passed in 9.09s**。五方法 Q/损失与未接回调时一致，输出侧完整轴保留；I/O 失败仍传播。两次之前日志均保留，无假成功记录。
- 三本 notebook schema/语法、未改动单元与用户配置保留、gate hash 检查通过；实际本地只读诊断运行，无活跃真实 experiment 进程。记录见 `evidence/g1_live_progress_20261116/validation.json`、pytest 日志与脱敏前版本。用户原有 00 notebook 改动保持原样。代码未提交/push；新日志须同步后新 run 才生效。
- 未执行：新版 Colab L4 预训练 7B、实际 GPU 性能诊断及 AutoDL 96GB；本机资源不足且无法直接控制用户 Colab。真实 G1 状态为 RUN_STARTED_OUTCOME_UNKNOWN / INSUFFICIENT_EVIDENCE，全 GPU 增量事件 kernel 仍 NOT_IMPLEMENTED。


## 2026-11-16 — 将 01/G1 模型、面板与数值求解迁移到 PyTorch CUDA

- 用户明确要求 GPU 驻留与新服务器从头测试。先复现当前全套 148 项测试，审阅加载、捕获、NumPy 求解、SpinQuant 及门槛；原用户 notebook 状态脱敏归档。上游 commit 8f47aa3… 的 tracked tree 保持干净。
- 完成 GPU-resident 加载/采集，采集所有 seed 后释放基础模型而不卸载到 RAM；补设备清单与显存心跳。G1 的量化、误差、风险预算、STE/grid/fixed-scale/cellquant、事件排序、输出完整行 envelope、增量系数和数值根均在 torch 设备上计算，主机保留调度和 I/O。初次测试暴露 NumPy 2 的 .device 属性判断错误（6 failed / 26 passed），修正后对照通过。
- 初版使用 torch.linalg.eigvals；核对 PyTorch 2.6 上游实现发现小矩阵隐式转 CPU，于是改用同设备复数移位 QR。根不收敛明确 ResourceLimit；未使用 CPU fallback。补 support owner 在浮点边界右侧的延续、固定尺度/零尺度/同时舍入、五方法、公开 API、恢复与完整目标 STE 更新对照。所有失败和中间日志均保留。
- 最终 191 passed in 23.51s，包含真实 CUDA 测试。最终 GPU smoke 子进程退出 0，报告 PASS_ENGINEERING_ONLY；完整 4096×11008×2 工程计算 3.5775s，峰值 2496416256 bytes。首次 PowerShell pipeline 对 stderr 警告返回 1，未将该 shell exit 标作成功；后续使用 subprocess 取得真实退出码，记录在 release_execution_records.json。CPU 数学、生产 G0 退出 0/PASS，空 G1 不足，gate_rules SHA 不变。
- 以 g1_gpu 新协议隔离旧 CPU 准备配置和断点，记录 torch CUDA RNG/QR 后端；01 拒绝旧 CPU 配置，准确绑定本次 run 的真实 G1 文件。新增从 GitHub 同步、Python 3.12/Colab 2026.07、Drive 恢复、保留 torch、固定依赖/FHT/SpinQuant、GPU 工程检查、真实准备、运行与备份的完整指南。六本 notebook schema/语法通过；未修改单元/输出及用户 HF 配置保持。
- 本机仅有 RTX 3050 Laptop 4GiB，FHT 采用明确的 torch_reference 工程后端。新版 L4 extension/7B 与 AutoDL 尚未运行，不能宣称科学门槛通过、完整求解一定更快或有 CUDA 融合 kernel。结果在 evidence/gpu_resident_20261116/validation.json 与实际日志；代码未提交/push。


## 2026-11-17 — 显存空闲不足定位与预检查可观测性

- 核对当前代码与三本用户 notebook，GPU 工程检查由用户保存的输出显示通过，G1 在模型加载前阻塞。请求实际显存读数，确认 L4 22.03 GiB/预算 18.73 GiB 足够覆盖估计，但 free 14.13 GiB 不足 16.55 GiB；当前 PyTorch kernel 占用为零。请求进程明细之前用户释放服务器，因此未确定占用者，不猜测为某个模型。
- 增加 GPU 内存结构化报告与三类限制原因，并按配置设备查询内存；保留原需求估计、预算、模型精度、面板和门槛。没有通过扩大 max_memory_fraction、卸载 CPU 或缩小实验绕过检查，也没有执行远程进程终止。
- 新增 6 项内存预检查回归，完整 pytest 实际 197 passed, 1 warning in 52.61s、exit 0。唯一警告为 Windows .pytest_cache 写入权限；完整日志保留。本轮没有编辑 notebook；最终字节比较发现用户 00 数学与 01 文件在工作期间被外部操作更新，首次归档验证据实失败，随后记录前后 hash 并保留外部变化。脱敏归档无令牌，gate_rules SHA 不变。
- 更新新 Colab 指南：恢复 models/datasets/runs 缓存，检查实际新 GPU 和 free 显存，重跑 01 预检查；明确 empty_cache 仅处理本进程未使用的缓存。云端新会话与真实 7B 未运行，保留不足状态。本轮代码未提交/push。


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

## 2026-11-17 — 读取用户真实单对诊断，澄清显存变化

读取用户最新保存的 01 执行单元，确认 `RUN_PAIR_PROFILE=True`。新 run `2026-11-17T102020.290922_0000_29bbe398e6` 已于 10:32:14 UTC 以 `RESOURCE_BLOCKED` 结束（712.44 秒）。固定尺度搜索 300 秒完成 19958/148902 个区间，CellQuant 300 秒完成 16236/151940 个区间；二者第一对均未完成。不能把持续心跳、CUDA 驻留或小规模回归通过解释为真实 G1 已可完成。

模型设备报告为全参数 CUDA、无 CPU offload，参数 13476831232 bytes。采集完成后 93.74 秒显式释放模型，只保留 CUDA 面板；CellQuant 搜索时 allocated=5.458 GiB、reserved=7.395 GiB。结合实际释放代码和 PyTorch 缓存分配器语义，后期显存下降符合设计。逐区间 Python 调度及多次同步是当前实现的具体限制；未进行内核级性能分析，不虚构各项耗时占比。

归档 235 条原始结构化事件至 `evidence/g1_pair_profile_colab_20261117/progress.jsonl`，`observation.json` 记录 notebook SHA256、门槛 SHA256、放置/释放事件及两个方法的实际超时诊断。只提取结构化事件，不归档 token 或 notebook 配置内容；来源明确为用户 notebook，未独立验证远程源码和结果文件。仅更新证据与文档，保留用户 notebook 修改；无需为此重跑算法测试。后续仍需实现并验证更高吞吐的完整区间处理，G1/G2–G4 尚无合格证据。

## 2026-11-17 — 实现 GPU 批量胞元目标，验证原预算下的合成完整单对

- 重读项目约束并复现已有测试：213 passed，1 个 Windows pytest 缓存权限警告，40.67 秒。保留工作区中用户的 00/01 notebook 修改；仅编辑核心源码、测试、运行脚本和文档。
- 新增 `tensor_cells.py`：输出组 `C@Xhat` 的编码跳变前缀和、分面板/分输出行多项式、共同事件并集上的唯一角度搜索。输入满足 radius 条件或固定解码尺度时使用精确秩二变化；固定对照的物理动态尺度风险独立更新。完整行不稳定输入边界使用尺度比、编码差与缓存乘积，先权重后激活保留交叉项，零尺度显式回退计数。其他输入仍走完整增量算法。
- 原事件枚举、角域、FP64、event cap、300 秒单对、方法总预算、共享 R1 和最终 32 候选完整量化复核保持。增加执行批次大小（默认 1024）和引擎版本指纹；旧引擎 checkpoint 不会被当作新版同一执行。阶段诊断区分局部系数与全局候选进度，超时保留已完成阶段计时。
- 新测试首次暴露四项动态输出边界错误：先除 qmax 再代入的尺度舍入与实际 AbsMax 次序不同。修复了真实边界编码、尺度点值差及同时事件处理，未降低容差。最终 241 passed, 1 warning in 34.57s；实际 CPU/CUDA 数值对照涵盖分组、批次、固定风险、零尺度、边界与 CPU 独立求解器。
- 新增 `scripts/profile_tensor_cells.py`，可记录 CUDA 完整维度合成性能并与归档旧模块比较。旧版完整输出在 30 秒处理 911/48725 区间后 RESOURCE_BLOCKED；最终新版本完整输出 6.7458 秒完成单对，含初始化/更新 13.9898 秒。六输入加一个完整输出的动态案例 78.6332 秒完成单对（总 86.4086 秒），固定尺度案例 6.9593 秒（总 14.3403 秒）。均是随机张量、identity Q、同一 RTX 3050；没有真实九面板/7B 加速结论。最终源码前后 hash 一致，实测峰值分配约 2.77–2.88 GiB。
- 保留中间混合案例 30 秒超时、初次失败测试、早期版本的 GPU 诊断与最终结果，见 `evidence/batched_cells_20261117/`。最终比较以带源码前后校验的 `*_final.json` 为准。动态输入和少量 QR 仍有明显串行成本，128 对是否符合原总预算未得到证明；不让工程单对 COMPLETE 替代 G1。
- CPU `scripts/run_math_checks.py --output runs/batched_cells_20261117/math_reference` 通过；空 G1 按预期 INSUFFICIENT_EVIDENCE；gate_rules 与 Colab 原科学/资源配置未修改。`git diff --check` 通过，仅有正常 LF/CRLF 提示。更新数学说明、README、状态和同服务器 kernel 重启/引擎核验流程。新版真实 L4/7B、AutoDL 96GB、正式 GPTQ/评估未运行，代码未提交/push。

## 2026-11-17 — 读取用户首对成功结果，增加连续 8 对诊断

用户最新 01 的 run `2026-11-17T114339.123212_0000_991980b138` 已 PROFILE_COMPLETE，整轮 140.93 秒。固定尺度和 CellQuant 首对均完成，后者处理 151940 区间、19.7602 秒进入最终复核；实际方法总预算 1440 秒。仅按首对搜索速度和仓库 128 对正式协议外推约 42.16 分钟，不能保证全量在 24 分钟方法预算内完成。这里不是远程完整结果 JSON 的独立重算；原始 notebook、210 条结构化事件和观察摘要保存在 `evidence/colab_profile_complete_20261117/`，不生成 G1 成绩。

新增 CLI `--profile-pairs N`、API 计数及 notebook `PROFILE_PAIR_COUNT`。连续执行原 schedule 的前 N 对、同一个 Q 不重置；仍只诊断第一个 seed 的两个事件方法，不读取 held-out 分数。保留正式 pair_budget、单对与方法总预算，参数必须为 1 到正式对数之间的整数。原 `--profile-pair` 保持一对兼容。统一既有方法预算计算，未改变其公式。结果记录 requested/completed/profiled pairs，预算摘要按实测搜索均值作条件性外推，明确遗漏初始化/更新/checkpoint 等开销，不颁发科学 PASS 或全量 READY。

01 设置下一步前 8 对；原附加诊断单元中的旧 run 路径替换为本次 `PILOT_RUN_DIR` 的只读摘要。改动的五个单元及旧输出有脱敏归档，其他单元保持。34 项针对性检查通过；最终全套 257 passed, 1 warning in 62.50s（Windows pytest 缓存权限，退出码 0）。其中实际 CUDA 小面板连续 8 对，验证 schedule、配置不变、完成数和 G1 拒绝；参数/CLI/旧路径修复/非法时长与不完整诊断均覆盖。六本 notebook schema 和代码语法通过。

本轮未改数学引擎、量化配置、科学门槛或正式预算，未执行真实 L4 的新 8 对、完整 G1、AutoDL/G2–G4。下一步在同一服务器同步代码并重启 kernel，01 保持 RUN_REAL_MODEL=True、RUN_PAIR_PROFILE=True，PROFILE_PAIR_COUNT=8，再运行原执行单元与末尾摘要。先看连续吞吐，再决定完整 G1；当前不进入 02。本轮修改未提交/push。
## 2026-11-17：检查真实 8 对结果并加速动态输入组

- 读取用户最新保存的 01，run `2026-11-17T120623.229443_0000_083c7593b5`：两事件方法均完成 8 对，PROFILE_COMPLETE/354.38 秒。fixed-scale 均值 8.7429 秒，CellQuant 20.3270 秒；原正式 128 对/1440 秒预算已由本次保存输出确认。CellQuant 搜索部分均值外推 2601.85 秒，仍不宜直接发起原预算全量或进入 02。来源是用户 notebook 输出，未独立读取远程文件；没有 held-out/G1 质量结论。
- 归档脱敏 notebook、483 条日志事件、预算摘要，定位慢对的输入系数/边界更新为主要额外开销。新增 GPU 完整行动态输入批处理：全部不稳定组成员的编码差前缀投影、双侧交叉项和旋转两成员均保留；解析损失/风险与实际边界舍入。分组不适用或不稳定组密集时使用原增量实现。完整矩阵、角域、FP64、事件上限、时间预算和 gate_rules 不变。
- 数值验证先复现 257 passed/57.73 秒；针对性 50 passed/31.77 秒；完整 281 passed/83.22 秒。均只有 Windows pytest 缓存权限警告，退出码 0。数学检查 PASS，空 G1 INSUFFICIENT_EVIDENCE。新增 CPU/实际 CUDA 检查比较全部边界、胞元内点、零尺度、双侧交叉项、原增量系数、完整 d=4096 与连续共享 Q 更新。
- 同一 RTX 3050 Laptop、相同 seed/完整轴随机案例：原输入增量搜索 49.7708 秒，新批处理 20.0957 秒；系数/边界 31.1689→1.2110 秒，方法含初始化/更新 57.2919→27.6085 秒；最终角度/损失/风险完全相同，54785 区间，峰值分配均 2.8813 GiB。这是六输入加一个完整输出的单次合成工程检查，不能替代真实九位置 L4 或完整 128 对实验。
- 更新 README、实现状态、数学范围和 Colab 指南；01 只更新一个说明单元，保留用户代码/开关/输出。实际记录及命令见 `evidence/input_batch_20261117/validation.json`。新引擎真实 L4 的下一步是同步源码后在原 01 重跑同样 8 对，确认瓶颈改善和预算余量；随后才尝试正式两种子五方法 128 对及真实 G1。未提交/push，未加载本机 7B，未执行新引擎云端 G1/AutoDL。

## 2026-11-18 · A100 前的性能与验收修正

- 从根提交 `1b2caf08e9944b1dc83f2eb5cf96b9111f33234d` 复核 Colab notebook 实际入口、SpinQuant 上游与用户的 L4 输出。8 对合计 354.38 秒，动态搜索平均 20.33 秒/对。没有该次运行的内核时间线或 SM 采样。
- 在 Torch 路径批量计算输入组半径筛选标记和稳定组的阈值方程。对损失导数和预算交点使用带 FP64 保护的中点 Taylor 余项筛选。接受输出侧更新时只重算两个完整行，固定解码尺度下另存物理尺度。新增 CPU/CUDA 回归覆盖跨组、同时事件、固定尺度、切根及重叠坐标连续更新。
- 修正前缀模式多采集另外三个 seed/split 组合的问题，增加采集计时和 GPU 峰值记录。方法耗时外推包括初始化、接受更新和 checkpoint，摘要保留原搜索分量。修正 STE/grid 未跑完整坐标表也可能返回 COMPLETE 的路径，并在 G1 汇总和恢复时复核旧记录。
- 原服务器模型级入口构造 SolverConfig 时忽略 backend/device。现在从冻结矩阵选择求解器和对应断点格式。G2 另测合法 Norm 融合与原模型的 FP32 偏差，补记最终旋转在实际模型 dtype 的误差。诊断和方法改动未触碰上游 checkout、量化规则、预算或 gate_rules。
- 初始化 notebook 从环境变量或 Colab Secrets 读取 HF 凭据，清除工作树里的明文值。已提示用户撤销历史提交中的令牌。00/01 默认走离线或预检路径。用户的历史运行输出保留在工作树，传输的 notebook 使用空输出。
- 本机 RTX 3050 Laptop、Python 3.11.18、Torch 2.6.0+cu118。同一完整维度混合随机案例两版各跑三次，54,786 个事件角、54,785 个区间、328,758 个候选。单对搜索中位数 20.342 → 3.484 秒，方法中位数 27.834 → 10.972 秒，峰值分配 2.881 → 2.459 GiB。该脚本使用六个输入面板和一个输出面板，每个进程一对，无 checkpoint 写入。每次的角度、最终真实损失和风险相同。[原始计时与哈希](../evidence/a100_readiness_20261118/benchmark.json)
- 首次全套回归暴露一个新增测试的断言位置错误和四个旧引擎标签断言，修正后最终 **310 passed in 70.44s**。CPU 数学和 G0 PASS，空 G1 正确返回 INSUFFICIENT_EVIDENCE。实际 CUDA tiny SpinQuant 的融合/旋转相对 RMS 为 1.03e-7 / 1.77e-7，GPTQ 完成且输出有限。GPU 驻留五方法及 4096×11008 形状检查通过，FHT 路径为 torch_reference。六本 notebook schema/语法通过，修改的 00/01 默认路径在本地执行完毕。[验证记录](../evidence/a100_readiness_20261118/validation.json)
- 在 `runs/a100_review_20261118_150928/` 保留各次运行。使用 `python -m pytest -q`、`scripts/run_math_checks.py`、`scripts/run_production_checks.py`、`scripts/run_gpu_resident_checks.py --full-dimension --fht-backend torch_reference`、`scripts/run_spinquant_smoke.py --device cuda --gptq --fht-backend torch_reference` 和 `scripts/profile_tensor_cells.py --case mixed_full --pairs 1 --seconds-per-pair 300`。A100、新版真实 L4、完整 G1 和目标机 G2–G4 未运行，原因是本机 4 GiB GPU 且没有连接到目标运行时。云端命令及投入依据见 [A100 测试方案](COLAB_A100_TEST.md)。

## 2026-11-18 · A100 正式 G1 完成

- A100-SXM4-40GB 在北京时间 16:30–17:47 完成正式 run `2026-11-18T083041.405198_0000_381f882eaf`。模型为 LLaMA-2-7B，两个 seed、五种方法、九位置的结果均已保存。每种搜索每 seed 完成 128 对，程序状态 COMPLETE，科学判定 FAIL。G2–G4 尚未执行。
- 从用户同步的 01 notebook 输出按原 JSON 文本归档了 `G1_evidence.json` 和 `G1_distributions.json`。另从用户执行的逐方法字段摘录保存拟合前后损失与风险、接受数、校准时间和逐位置验证误差，并用正式进度流核对完成状态和坐标表。十份完整 `pilot_result.json` 的云端路径及 SHA-256 见门槛证据，本地保存的是字段摘录。[证据目录](../evidence/colab_a100_g1_20261118/)
- 从十份摘录独立重建两种子九位置的 Hadamard、CellQuant、逐位置最强对照矩阵，重算两种相对改善和校准速度比。与正式输出一致。用 `python -m cellquant.gates evaluate --stage G1 --evidence evidence/colab_a100_g1_20261118/G1_evidence.json --output evidence/colab_a100_g1_20261118/G1.json --require-pass` 再次得到 FAIL，退出码为 1，无缺失字段。共同条件全部通过，准确率和效率路线分别有两项未达标。
- 相对 Hadamard 的中位改善为 2.569%，相对最强匹配搜索为 0.017%。CellQuant 在拟合和独立验证上均能小幅优于角度网格，九位置平均 held-out NMSE 的额外相对降幅仅 0.070% 和 0.067%。同卡速度比为 0.264×。项目继续保留原门槛，当前停止 G2 和全量投入。[分析与后续选择](IMPLEMENTATION_STATUS.md#a100-正式-g1)

## 2026-11-18 · G1 文档区间与拟合面板诊断

- 从已执行的 01 提取 2 种子 × 5 方法 × 9 位置 × 8 文档的原始块字段。逐块复算全部 90 个面板 NMSE，核对 G1 矩阵。10,000 次按独立文档配对的结果在第二套求和循环中逐项复核全部五个对照的区间。相对逐位置最强对照的 95% 区间覆盖零。用原证据重跑 G1 仍为 FAIL。[真实文档区间](../evidence/colab_a100_g1_20261118/document_intervals.json)
- 新增 `pilot_bootstrap`、`--diagnose-pairs` 及 01b。后者只采集 fit，对两个种子分别测共享坐标表的输出侧覆盖率和同一初值上网格与事件角度的实际联合降幅。每个入选角度均重建 Q、完整重算九面板并复核双风险。记录每对实测、独立计时、面板/坐标表指纹和中断时的已完成记录。已在查看真实 7B 诊断前记录选择一次修正的条件。[诊断协议](G1_PAIR_DIAGNOSTIC.md)
- CPU 四维双侧测试有一组角度的增量损失为 0.02203887，用物化 Q 重算为 0.02287096。直接检查发现浮点乘法顺序改变了近舍入边界的编码。新诊断同时保留两种测量。RTX 3050 Laptop 用两个种子各跑完八条小矩阵探针，来源为 `unit_test_fixture`。A100 上同一诊断仍为 NOT_RUN。[本地实测和源码哈希](../evidence/g1_pair_diagnostic_20261118/local_validation.json)
- 最终回归 **322 passed in 82.96s**，CPU 数学参考 PASS。01b 通过格式校验与默认执行，另实测从原 notebook 的 `%run` 触发、子进程参数和归档路径，后一项使用测试子进程替身。首次整套回归检出了已执行 00 的明文 HF 凭据，已恢复环境变量/Colab Secrets 入口并脱敏本轮日志。上次的 00 输入准备开关恢复为默认关闭，已有单元输出逐项比对一致。原 01 的测量数据保留在工作树。
- 北京时间 19:01:17，A100 真实拟合诊断返回 `DIAGNOSTIC_COMPLETE`，程序计时 252.45 秒。从 01b 输出核对全部 64 个唯一的种子、位置、配对方式和方法组合，并比对两份最终摘要。两种子覆盖率和 CellQuant 重算后平均降幅均提高，seed 23 对网格的优势从 0.00787 降至 0.00609 个百分点。按既定条件停止该配对修正。本地已归档 notebook 字段摘录，逐对原始文件在本次 Colab 结果包内。[A100 摘要及结果包路径](../evidence/g1_pair_diagnostic_20261118/a100_summary.json)

## 2026-11-18 · 相对位置 NMSE 的一次修正

- 旧 A100 数据中，Hadamard 在九个位置上的最大/最小验证 NMSE 分别为 35.23 和 33.54。基于原拟合目标与 G1 相对改善统计的差异，引入 `relative_site_nmse`。只用各位置拟合起点的误差计算权重，各求解器、所有角度及增量/直接复核共享该定义。权重、激活风险仍按原单位约束。单独保留旧目标网格，从同一初值重新训练。
- 冻结 01c 短测规则。两个原种子，各将八份拟合文档按原顺序分为四加四。两种目标与两种求解器各用原坐标表的连续前 32 对。相对较强网格达到 1%、两个种子方向一致且超过旧 CellQuant 才准备用于正式 G1 的新验证文档。正式 G1 一次，照原机器规则，另写包含原目标网格的保守比较。保存逐块误差、面板与坐标哈希、短测与新验证清单之间的追溯关系。[完整协议](G1_OBJECTIVE_TRIAL.md)
- 全部 **334 项测试通过**。CPU 数学参考 PASS。本机 RTX 3050 Laptop 的四维 CUDA 小矩阵实际完成两种子、每组 32 对的短测调度，十份结果齐全。01c 的默认路径已实际执行，另用子进程替身分别核验短测停止和继续正式复验的 notebook 分支。旧 G1 的五组文档配对区间逐项复算一致，旧 G1 仍为 FAIL。本机 4 GiB 无法承载真实 7B 面板采集；新 A100 短测、正式复验和 G2–G4 均待真实运行。[可核验记录](../evidence/g1_objective_20261118/local_validation.json)
- 北京时间 19:51:31，01c 的 A100 短测运行完成，程序计时 961.41 秒。从 notebook 核对 10 组起始记录、8 个搜索作业各 32 个连续坐标步和最终摘要。重算五组对照的跨位置中位数与输出一致。新 CellQuant 相对较强网格的中位改善为 −0.0337%，只有 2/9 个位置改善。三项投入条件均未达到，判定 `STOP`。自动复验开关已开启，条件分支按设计跳过。已保存摘要及原始逐方法结果包的 Colab 路径。[01c 实测记录](../evidence/g1_objective_20261118/a100_summary.json)

## 2026-11-18 · 输出投影的可分离匹配与尺度风险

- 从原 A100 的逐块数据复算了输入侧和输出侧区间。旧求解器相对 Hadamard 的中位降幅分别为 3.3172% 和 0.0228%，两种子的 128 个坐标步仅触及 250/245 个不同输出坐标。[历史实测图](../evidence/output_matching_20261118/historical_7b/site_effects.png)
- 实现全模型共享的输出行对更新、动态 AbsMax 组风险上界、完整九位置拟合复核、面向覆盖和选角信号的对照，以及逐文档误差三项分解。新增 01d 和终端入口，沿用原 G1 输入并在 train 上单独抽验证集。每个方法保存坐标和角度、Q 哈希、时间及原始文档误差。[实验和下一次 A100 读法](OUTPUT_MATCHING_STUDY.md)
- 固定预训练 42M checkpoint 和 WikiText-2 validation 后在 RTX 3050 上实际运行。最初只在最后检查风险会丢弃有输出收益的候选。加入逐对风险分配后，第一批 16 篇新文章上相对起始网格的输出改善为 1.84%，相对同覆盖仅权重选角为 1.09%。随后冻结延长网格加一次修正，用另外八篇文章检验，输出相对延长网格改善 1.16%，九位置改善 0.71%，后者区间跨零。种子 23 的联合拟合复核选择了零步长。[第一批原始测量](../evidence/output_matching_20261118/pretrained_42m/measurements.zip) · [第二批原始测量](../evidence/output_matching_20261118/hybrid_42m/measurements.zip)
- 两个归档共 30 份方法结果。从解压后的文档误差重建全部比较与区间，同原始运行摘要逐项相等。文档 ID、内容哈希和 token 哈希均验证拟合、探索及两批验证间无重叠。两批中最大归一化正交残差为 4.73 × 10⁻¹⁶。完整 4096 × 11008 合成输出面板、八个 token 的 2048 对 × 17 角程序运行 19.22 秒，峰值 2.57 GiB。全套 341 项回归及 CPU 数学检查通过，含从冻结父输入到新面板配置的回归。01d 默认运行通过。新版 7B 和整模型指标待 A100 实测。[归档及复核](../evidence/output_matching_20261118/validation.json) · [完整形状检查](../evidence/output_matching_20261118/cuda_4096.json)
- 结束并发回归任务后，在同一批八篇文档上重复完整两种子校准。16 份 Q 哈希和逐块验证字段逐一相同，两个种子从延长网格追加输出修正各耗时 0.609 和 0.578 秒。[重复实验与哈希](../evidence/output_matching_20261118/hybrid_42m/timing_repeat.json)

## 2026-11-19 · 7B 输出匹配与分批复核

- 从 01d 的 A100 原包核对 16 份方法结果的 SHA-256、拟合与验证文档的 ID、内容及 token 哈希、原 G1 与本轮验证文档的 ID。由 1152 条逐文档、逐位置记录重算 10000 次配对抽样，七组对照区间与 Colab 完全一致。测得的最大归一化正交残差为 1.54 × 10⁻¹⁵。归档了只含方法记录与摘要的可复算副本。[测量包](../evidence/output_matching_20261119/measurements.zip) · [审计记录](../evidence/output_matching_20261119/colab_7b_summary.json)
- 128 对网格后的输出中位误差下降 1.55%，收益全部来自种子 11。256 对网格上，两个种子分别提出 1548 和 1437 个非零角，三个非零步长均满足尺度约束，但输入误差抵消了输出收益。两次保留原网格，01d 到正式 G1 的预定条件未达成。[比较与分解](OUTPUT_MATCHING_STUDY.md#7b-实测与下一轮决策)
- 新增只用拟合面板的短测。它复用同一 256 对网格和同一批候选角，改为每 64 对检查原九位置损失及尺度风险，保存每次接受或拒绝的实测值、各位置 NMSE 及同批数随机对照。已在本机预训练 42M 的两个种子上重建归档 Q 并实测，均保留四批中的两批，联合拟合损失相对原整批最佳步再降 0.53% 和 0.83%。7B 分批结果和新文档收益待 Colab 运行。[42M 逐批记录](../evidence/output_matching_20261119/fit_batches_42m.json) · [诊断入口](OUTPUT_MATCHING_STUDY.md#在-colab-运行分批拟合短测)
- 另在 RTX 3050 上从 7B 归档因子重建两个种子的 4096 维 Hadamard、128 对、256 对旋转，全部六个矩阵的 SHA-256 与 A100 记录相同。拟合短测的仓库默认配置在独立发布快照中通过 344 项测试，CPU 数学检查通过。本机 4 GiB 显存只做了 7B 旋转回放，7B 真实面板仍需在 A100 运行。[Q 哈希核对](../evidence/output_matching_20261119/colab_7b_summary.json)

## 2026-11-19 分批旋转的独立文档验证

- 保存用户提供的 7B A100 终端片段。拟合用时 72 秒，两种子均达到新文档测试条件。种子 23 实际采用 1 批、46 对，输出项减少 0.000211507，输入项增加 0.000207470。种子 11 的逐批明细由终端入口从本次运行读取。[7B 拟合记录](../evidence/output_matching_20261119/fit_batches_7b_terminal.json)
- 加入终端选项 `--validate-batches`。它读取两个种子的已选批次和固定随机参照，在排除此前全部验证文章后，每个种子新取 16 篇，仅采集验证激活，同时保存三种旋转的测量和源文件。晋级条件沿用 01d 后记录的九位置配对区间下限为正、两个种子均改善。[7B 操作与决策](OUTPUT_MATCHING_STUDY.md#分批旋转的独立文档验证)
- 在 RTX 3050 上使用 42M 预训练权重完整运行两次。六份逐方法文档测量的 SHA-256 均可重复。用逐块误差重新计算得到相同的区间和判定，六个 Q 的最大归一化正交残差为 3.90 × 10⁻¹⁶。[42M 测量包](../evidence/output_matching_20261119/batch_documents_42m.zip)
- 本轮工作区 347 项测试通过。另有一项检查 Notebook 00 的默认配置，在当前改动前的 `8e9a142` 上也失败，该版本已将 `PREPARE_G1_INPUTS` 保存为 True。用此前的默认配置 False 复核该项通过。CPU 数学检查 10 项通过。7B 新文档实测待 A100 运行。

## 2026-11-19 A100 独立文档 pilot 完成

- 归档 A100 的六份逐方法测量，校验文档、token 和面板指纹。由逐块平方误差、目标能量和 10000 次配对抽样复算全部三组比较。相对 256 对网格，九位置中位改善 0.0812% [0.0378%, 0.1329%]。相对固定同批数随机对照为 −0.0072% [−0.0726%, 0.0620%]。满足 pilot 的正式复验条件。[归档与误差分解](../evidence/output_matching_20261119/batch_documents_7b_summary.json)
- 增加正式分批复验入口 `--retest-batches`，按拟合前固定的协议另取 48 篇文章，重新拟合两个种子。从父网格算出准确率和效率路线的上界，默认在两条路线都已排除时保留独立测量并结束。完整对照包含同起点、同平面序列、同角域、同校准时间上限的网格、STE 和固定尺度搜索，`--all-controls` 可要求全部实测。完整比较供原有 G1 evaluator 和逐文档区间程序读取。[A100 命令与判据](OUTPUT_MATCHING_STUDY.md#在-colab-复验)
- 本机在预训练 42M 的 48 篇独立文章上验证了提前结束和强制补齐对照两条路径。为验证逻辑，父网格使用 2+2 对，后续使用 4 个平面、2 对一批。完整路径保存 14 份逐种子测量。两次共用的四种旋转与其逐块误差完全一致，正式 G1 汇总拒收该缩短序列的工程记录。[本地工程复核](../evidence/output_matching_20261119/batch_retest_42m_verification.json)
- 本轮 353 项自动测试通过，CPU 数学检查 10 项通过。Notebook 00 在本轮修改前已将 `PREPARE_G1_INPUTS` 保存为 `True`，已有的一项默认值测试期望 `False`，仍然失败。新增终端复验直接读取已准备的 G1 输入。本机 RTX 3050 4 GiB 仅验证了 42M 路径，7B 复验须在 A100 运行。

## 2026-11-19 A100 复验停止与输入侧选角

- 归档用户的 `batch_g1_retest.zip`。48 篇新文章中有 16 篇参与拟合，32 篇用于验证。8 份方法记录用于重新计算逐块 NMSE、文档配对区间与门槛上界。两个种子分别提出 1530、1602 个非零角，64 个候选批次全部满足风险上限，全部增加联合拟合损失。最终相对 256 对网格的实测改善为 0%，校准速度比为 0.9858×。返回 `RETEST_STOPPED_BY_BOUND`，额外三个求解器按协议跳过。[7B 复算与图](OUTPUT_MATCHING_STUDY.md#7b-复验结果)
- 实现双侧角度提案。对六个输入面板用父旋转完整分组的尺度计算单平面乘积误差增量，加上三个输出位置的精确代价。分批接受继续使用真实动态量化和两个风险上限。RTX 3050 上的预训练 42M 在两个种子均接受 4/4 批，冻结后在 32 篇新文章上相对同起点仅输出选角改善 3.6570%，相对完整网格改善 1.0445%。已用 8 份逐文档文件复算区间。[42M 配对结果](../evidence/output_matching_20261119/bilateral_documents_42m_summary.json)
- 7B 短测入口为 `python scripts/run_output_matching.py --screen-bilateral`。它使用本次 A100 复验的原拟合面板，回放网格到第 64 对。从 A100 因子在本机 GPU 重建的两份完整 4096 维父旋转均与 A100 记录的数组 SHA-256 一致。7B 新提案与整模型 GPTQ 尚待实测。[A100 协议](OUTPUT_MATCHING_STUDY.md#在选角时计入输入侧误差)
- CPU 数学检查 10 项通过。358 项自动测试通过，1 项既有测试因 Notebook 00 保存的 `PREPARE_G1_INPUTS=True` 与默认值预期不符而失败。

## 2026-11-19 A100 双侧选角短测与新文档评分入口

- 归档 A100 运行 `2026-11-19T093653.358169_0000_82281fcdcf`。两个种子相对同一 64 对前缀的仅输出选角，九位置联合拟合损失分别下降 1.56% 和 2.09%。相对 256 对网格，第 0 层 up 仍退化 16.80% 和 16.54%。RTX 3050 上回放了两个种子的前缀、完整网格和两种选角结果，全部八个矩阵哈希都与拟合记录相同。[A100 拟合复算](../evidence/output_matching_20261119/bilateral_fit_7b_summary.json)
- 固定了 32 篇新的 WikiText-2 train 文章，以及九位置中位降幅、输入和输出分组、第 0 层 up 的文档配对区间。已见 7B 文章 144 篇，小模型工程与验证用过 112 篇训练集文章，去重后共排除 235 篇。终端命令 `python scripts/run_output_matching.py --validate-bilateral` 读取归档拟合配置与两份已冻结的选择追踪。准备阶段核对原 48 个 7B 块和新选 32 个块的身份，实测阶段仅采集验证面板。[文档清单](../evidence/output_matching_20261119/bilateral_document_cohort_7b.json) · [运行与决策](OUTPUT_MATCHING_STUDY.md#7b-双侧选角的拟合与独立文档验证)
- 真实预训练 42M 在本机重新执行了共用的冻结评分路径，三组配对结果、八份方法的旋转哈希和逐块测量与上次完全一致。本机用固定公共语料和已在原 48 块逐项匹配的本地 tokenizer 检查了准备入口与清单回读，A100 评分会使用固定 7B tokenizer 重验。CPU 数学 10 项通过。362 项自动测试通过，剩余一项既有默认值检查期望 Notebook 00 的 `PREPARE_G1_INPUTS=False`，当前文件保存为 `True`。本机 4 GiB 显存已验证旋转回放，7B 文档 NMSE 仍待 A100 实测。

## 2026-11-19 A100 双侧选角独立文档结论与行诊断

- 归档运行 `2026-11-19T100323.868548_0000_f25a2cf845`。从 8 份原始方法记录、32 篇文章、9 个位置复算了全部区间、逐块误差分解和预定判定。相对完整网格中位改善 +0.0652% [−0.3560%, +0.4107%]，两个种子中位数符号相反。第 0 层 up 的 32 篇文章逐篇退化，超过两条 G1 路线共有的 10% 单位置上限。停止当前双侧版本。此前正式 G1 判定为 FAIL。[A100 原始包](../evidence/output_matching_20261119/bilateral_documents_7b.zip) · [重新计算的统计量](../evidence/output_matching_20261119/bilateral_documents_7b_summary.json)
- 第 0 层 up 两个种子相对网格分别退化 14.99% 和 12.79%。增量主要来自权重误差乘以量化激活的项。采集器在所有文章和两个种子上使用同样的 64 个等距输出行。新增 `--diagnose-bilateral-rows` 在同一批文章上回放完整网格、短前缀和双侧旋转，分别测量原行、其余行及整层的 W4A4 与两个单侧量化反事实。各组按逐文章误差能量汇总，并对照原采样行的归档测量。[实验命令与判读](OUTPUT_MATCHING_STUDY.md#7b-新文档结论与第-0-层行诊断)
- 真实预训练 42M 在 RTX 3050 上完成该诊断，校验了所有原采样 64 行的逐篇 W4A4 数值。相对完整网格，原 64 行改善 1.960%，其余 1,312 行改善 0.592%，全部 1,376 行改善 0.652%。模型大小、图的浮点精度和拟合预算与 7B 均有差异。7B 全行结果待 A100 运行。[42M 逐篇行诊断](../evidence/output_matching_20261119/bilateral_rows_42m.zip)
- CPU 数学检查 10/10 通过，文档和行诊断的 7 项测试通过。全套 365 项通过，1 项既有失败来自 Notebook 00 保存的 `PREPARE_G1_INPUTS=True`，该测试要求默认值为 `False`。

## 2026-11-19 A100 全行实测与换行对照

- 归档 A100 运行 `2026-11-19T104255.838992_0000_53a7e06f15`，原包 SHA-256 为 `a8d931923fbdca4994b5b0a50daf883a72e883bb8e0a0bfa80fd2a27b89570ff`。逐篇核对文档与 token、原 64 行的误差、分组与整投影的能量加和及三个旋转。分别用原汇总器与配对行组计算复现 27 组区间。相对完整网格，双侧选角的 W4A4 在原行退化 13.892%，其余行退化 0.209%，全部 11,008 行退化 0.267%。整投影上，短前缀相对完整网格为 +0.020% [−0.130, +0.173]，追加双侧选角相对同一短前缀为 −0.288% [−0.489, −0.083]。[A100 逐文档复算](../evidence/output_matching_20261119/bilateral_rows_7b_summary.json) · [误差图](../evidence/output_matching_20261119/bilateral_rows_7b.png)
- 增加 `--crossfit-rows`。两组等预算的共享角度网格只改变第 0 层 up 的 64 个拟合行，在 A、B、未用行、整投影与其余八个位置留存逐篇测量。先固定 7B 的 32 篇新文章，核对与此前 267 篇的文档、内容和块哈希无重合。42M 单独抽样，与这 32 篇也无重合。7B 两个种子各需重拟合一套 B 网格，同卡历史 A 网格校准各约 395 秒。7B 换行结果待 A100 实测。[验证文档](../evidence/output_matching_20261119/row_crossfit_cohort_7b.json) · [终端命令与判据](OUTPUT_MATCHING_STUDY.md#交换拟合行的配对实验)
- 本地 RTX 3050 4 GiB 的预训练 42M 换行拟合与验证已完整复跑，两个种子的旋转和逐文章结果可重复。以 A 网格为参照，B 网格在 A 行落后 5.115%，在 B 行领先 4.513%，两种子的配对差均为正。相减为 +9.628 个百分点 [8.789, 10.503]。其余 1,248 行改善 0.583%，其余八个位置的中位改善区间包含零。[42M 拟合与测量](../evidence/output_matching_20261119/row_crossfit_42m.zip)
- CPU 数学 10/10 通过，本轮专项 11 项通过。全套 369 项通过，1 项既有测试仍因 Notebook 00 保存的 `PREPARE_G1_INPUTS=True` 与预期默认值 `False` 不同而失败。新终端实验使用归档配置。
- 干净提交 `9d0c85c` 上第三次运行 42M，八个旋转、全部逐文章测量、主指标区间与前两次相同。归档保存此次实测的环境、数据清单和拟合追踪。用本地已校验的 WikiText 源文件和 tokenizer 重新构造了旧、新两组 32 个验证块，全部与各自冻结清单匹配。正式准备入口在 A100 上会用指定的 7B tokenizer 重验。[42M 复现包](../evidence/output_matching_20261119/row_crossfit_42m.zip)

## 2026-11-19 A100 换行复验与中途换行实验

- 归档 LLaMA-2-7B A100-SXM4-40GB 运行 `2026-11-19T110847.684264_0000_3575dffa5d`，原包 SHA-256 为 `0492a3a9c4d86e485b88d5d2959e8819f903ba151c856b54b02878f8f0df629e`，耗时 782.17 秒。按冻结文章清单和原始逐篇平方误差独立复算三个精度、四个行组、种子效应及文档配对区间，同时复核短前缀与其余八个位置。RTX 3050 从原 A 网格、实测 B 网格日志恢复两种子的八个旋转，数组 SHA-256 全部匹配。最大归一化正交残差为 5.30 × 10⁻¹⁶。[7B 原包](../evidence/output_matching_20261119/row_crossfit_7b.zip) · [统计复算](../evidence/output_matching_20261119/row_crossfit_7b_summary.json)
- 主指标达到原定条件。W4A4 拟合行间的反转为 +61.055 个百分点 [58.828, 63.358]。A 与 B 各自在自己的拟合行上赢得 32/32 篇。B 相对 A 在未见行改善 −0.172% [−0.383%, +0.042%]，在整投影为 −0.140% [−0.350%, +0.072%]。权重单侧量化的行组反转为 138.252 个百分点，激活单侧为 −0.212 个百分点。结论涉及固定行与独立文章的评估范围。[两模型对照图](../evidence/output_matching_20261119/row_crossfit_transfer.png)
- 新增 `--refresh-rows`，从已实测的网格中点继续，比较两个换行顺序和各自固定对照，按同一篇文章配对平均两个顺序。7B 另外冻结 32 篇文章，排除先前 331 篇。已用经哈希校验的语料、本机缓存的 tokenizer 和归档的语料指纹测试准备、读取与来源约束，验证 32 个新 token 块及与本轮 42M 文章的隔离。正式执行使用锁定的 7B tokenizer 与数据集。7B 中途换行尚待 A100 运行。[7B 协议](OUTPUT_MATCHING_STUDY.md#固定预算下中途更换拟合行)
- 本机预训练 42M 已完成相同的换行顺序对照。未见行在 A→B、B→A 上分别改善 +0.287%、−0.235%。预定平均为 +0.026% [−0.049%, +0.106%]。两个种子分别为 −0.291%、+0.344%。四条换行路径的最终尺度均在目标预算内。CPU 数学 10/10 通过，专项测试 14 项通过。全套 372 项通过，1 项既有失败源于 Notebook 00 当前保存的 `PREPARE_G1_INPUTS=True` 与预期默认值 `False` 不同。[42M 逐篇测量](../evidence/output_matching_20261119/row_refresh_42m.zip)
- 在干净提交 `bc32cbf1bd88060a3b82615fc38f3dafdd942439` 复跑 42M。两个种子的角度搜索轨迹、四组旋转哈希、逐篇误差和配对区间均与首次运行完全一致。归档已更新为本次复现的原始记录，SHA-256 为 `3d704c392c64e62f2c5b6095626683a1b5bf60f4a1bd7d1ac760e21a92e95644`。
