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
