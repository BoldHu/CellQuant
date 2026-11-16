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
