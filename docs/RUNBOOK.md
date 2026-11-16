# 当前实现的运行手册

更新：2026-11-15。以下命令必须在项目根目录执行。Windows 本地使用 `.venv/Scripts/python.exe` 替换 `python`。这是可执行路径说明；真实 G1–G4 尚未运行或通过。

## 环境与固定上游

保留宿主 PyTorch，先记录 `python -c "from cellquant.runtime import environment_report; print(environment_report())"`。本机实际为 Python 3.11.18、Torch 2.6.0+cu118、RTX 3050 Laptop 4GiB；这套 CUDA 构建不是 Blackwell 环境锁。目标机器必须使用其已验证的 CUDA/Torch，不能照搬本机轮子。

```bash
python -m pip install -e '.[dev]'
python -m pip install -r requirements/hf_optional.txt
git clone https://github.com/facebookresearch/SpinQuant third_party/SpinQuant
git -C third_party/SpinQuant checkout 8f47aa3f00e8662caf1a484153920a07e5281c3a
python scripts/prepare_spinquant.py
git clone https://github.com/EleutherAI/lm-evaluation-harness third_party/lm-evaluation-harness
git -C third_party/lm-evaluation-harness checkout d6de81643928d653435c431bae19945d41d32520
python -m pip install -r requirements/evaluation.txt
python -m pip install --no-deps -e third_party/lm-evaluation-harness
```

已有 checkout 不再 clone；只核对固定 commit 和 clean tracked tree。适配器从原目录生成 `runs/spinquant_overlay_v2`，补丁及哈希在 `patches/` 和 overlay manifest。不能修改 `third_party/SpinQuant` 后继续声称原版。

FHT 必须在目标 Torch/CUDA/架构下安装并实测。本机未安装 FHT、FlashAttention、Triton。`CELLQUANT_FHT_BACKEND=torch_reference` 仅供显式结构验证，不算扩展兼容或性能通过；正式配置默认 `extension`。FlashAttention/Triton 是分别记录的可选探测，主图采用 eager attention。

## 已可复现的局部验证

```bash
python -m pytest -q
python scripts/run_math_checks.py --output runs/local_reference
python scripts/run_production_checks.py --output runs/local_production --full-dimension
python scripts/run_spinquant_smoke.py --device cuda --gptq --fht-backend torch_reference --output runs/local_tiny
python scripts/run_spinquant_smoke.py --device cuda --gptq --kv-heads 1 --fht-backend torch_reference --output runs/local_gqa
python scripts/run_spinquant_training_smoke.py --output runs/local_training
python scripts/run_spinquant_training_smoke.py --output runs/local_training_resumed --resume runs/local_training/checkpoint-1
python scripts/run_harness_smoke.py --output runs/local_harness_fixture
```

完整维度 smoke 保留 d=4096 和 FFN=11008，但仅 2 tokens、一个 pair、角域 ±0.0001；其耗时不代表正式校准吞吐。六任务 smoke 使用明确标注的本地 unit-test fixtures，不下载 benchmark 数据，不产生可用论文准确率。

## Colab G1

运行 notebook 00，产生实际 `runs/00_math/production/G0_evidence.json`。再准备固定 revision 与真实数据：

```bash
python scripts/prepare_experiment.py --config configs/colab_pilot.json --output runs/prepared/g1
python -m cellquant.experiment --config runs/prepared/g1/config.json --preflight
python -m cellquant.experiment --config runs/prepared/g1/config.json
```

准备脚本读取 HF_TOKEN 环境变量或现有 HF 授权，不输出 token；解析 immutable model/tokenizer/WikiText revisions，保存 HF 权重分片的 LFS SHA-256/大小、实际 tokens 与文档划分，不下载基础权重。真正模型阶段会先校验完整分片再加载。

默认 3 层 × q/up/down、2 种子；每个 seed 的五个方法使用相同冻结面板、Q0、pair schedule、墙钟上限、风险预算及固定 R2/R3/R4。q/up 允许抽输出行；down 保留完整 hidden×FFN。实际结果在唯一 `runs/experiments/<run_id>`，其中的 `G1_evidence.json` 来自原始 held-out 记录。设置 notebook 01 的 `CELLQUANT_PILOT_CONFIG` 和 `CELLQUANT_G1_EVIDENCE` 指向这些真实文件。

本机可用 CPU RAM 约 2GiB、GPU 4GiB，不能装载 7B。当前实现用 CPU 暂存全模型和完整面板，preflight 对 7B/8B 至少要求 24GiB 可用 CPU RAM；这是最低检查，不是足够内存保证。普通低内存 Colab 可能阻塞，高 RAM 环境仍须实测。T4 用 FP16并单独记录；换平台或 dtype 后重新跑对照。

## 迁移、canary 与完整 SpinQuant

```bash
python -m cellquant.migration export --run-dir runs/experiments/实际G1运行目录 --output runs/g1_transfer
```

将该 bundle 和代码拷到 AutoDL 本地盘；不要在 Drive 上高频读权重。目标机执行：

```bash
python -m cellquant.migration import --bundle runs/g1_transfer --output runs/gates
python scripts/prepare_matrix.py --config configs/autodl_canary.json --output runs/prepared/canary
```

迁移保留源字节与哈希，生成新路径引用并重算同一 G1；不产生新的 AutoDL 性能证据。notebook 02 选择这个 matrix，显式运行 GPU probes 和 canary。G1 的模型/方法效果不能由硬件 smoke 替代。

Full SpinQuant 使用固定上游自定义 LLaMA、CE、SGDG、R1+全部 R2、W16A4KV4 旋转训练，100 步、全局 batch 8；单 GPU 用 accumulation=8。其 CE 池是全部合格 WikiText train blocks 排除完整 dev 文档，按原 Trainer 的 max_steps/epoch 语义使用整个池；800 是 microbatch 预算，独立 blocks 数量另行记录，不把重复使用计为新增独立样本。GPTQ 使用冻结的 128 个 fit blocks。我们按文档分块以保留独立 held-out，因此不是逐 bit 的原八卡/串接语料复现，也不能与论文原表直接混比。STE/grid 是同面板目标的归因对照，不能代替完整 SpinQuant。

```bash
python -m cellquant.evidence G2 --runs runs/prepared/canary/results --g1-evidence runs/gates/G1_evidence.json --output runs/gates
```

FP32 port 另建 overlay：`python scripts/prepare_spinquant.py --fp32-port --output runs/spinquant_overlay_fp32_v2`。在实验前冻结使用该 overlay 的配置，保持训练数据/预算/CE/SGDG/R1/R2 相同。当前只有数值与小模型训练验证，没有真实质量资格；不能用它自动填 `matched_spinquant_reproduction_present`，也不能用最慢 FP64 版本单独支持效率主张。

## 正式评估与 G3

先准备不访问 test 的协议；真实 G2 后再准备冻结 test tokens 到新目录：

```bash
python scripts/prepare_matrix.py --config configs/full_experiments.json --output runs/prepared/full_protocol
python scripts/prepare_matrix.py --config runs/prepared/full_protocol/matrix.json --output runs/prepared/full --include-tests
```

前一步锁定 C4 和六任务 revision，但不打开正式 test。后一步要求实际 G2，生成同一全矩阵 provenance，所有模型/种子/方法共用协议。C4 默认预注册 validation 第一 shard、最多 10000 文档取 256 个完整 blocks；改变子集须在 test 评分前冻结。WikiText/C4 采用逐文档分块，不是官方串接 PPL 协议。

notebook 03 默认生成 `runs/03_full/plan.json`。G3 仍需要目标 Blackwell 实测、真实模型的完整/恢复运行对比、测量支持的预算计划；目前这些证据缺失，不能运行全量。

```bash
python -m cellquant.evidence resume-audit --original 实际连续运行/result.json --resumed 实际恢复运行/result.json --output runs/resume_audit.json
python -m cellquant.evidence G3 --output runs/gates --g2-evidence runs/gates/G2_evidence.json --plan runs/03_full/plan.json --hardware-probe runs/02_canary/gpu_probe.json --canary-runs runs/prepared/canary/results --resume-audit runs/resume_audit.json
```

这里带“实际”的路径必须来自真实运行，不能用 tiny fixture 替换。G3 的真实恢复审计必须绑定正式 plan 的同一 execution fingerprint；若 canary/正式 protocol 不同，需要用正式 provenance 做只评估 dev 的恢复预演。预算估计必须引用实际测量及其覆盖范围；当前没有自动为缺失测量生成通过值。G3 还绑定计划内每个配置，直接 CLI 修改配置也不能复用旧许可。

六任务已固定：ARC-C/ARC-E/HellaSwag/PIQA 用 `acc_norm,none`；LAMBADA/WinoGrande 用 `acc,none`。逐题保留 gold、choice prediction、原始 scores、prompt hash、metric contribution；LAMBADA 保留目标序列 likelihood 和 greedy exact-match 判定，不伪称保存了自由生成文本。

## 中断、分析与边界

每个 invocation 有独立 run_id，配置与模型/数据/源码/依赖/GPU 指纹核对后才能恢复。在新配置设置 `resume_run_dir` 指向父 run；完整 Q、事件 checkpoint、Trainer 旋转/优化器/调度器/安全 RNG、GPTQ 逐层 checkpoint 各自核验。事件恢复沿用累计时间预算。grid/STE 仅支持已完成方法复用；其中途 optimizer 恢复仍未实现。

```bash
python -m cellquant.analysis --runs runs/prepared/full/results --output runs/paper_analysis_v1
```

结果必须有原始文件且 hash 一致。CI 按 paired seed/document/item 重采样，PPL 用总 NLL/token；空结果不生成图或数字。已存在真实表格时要求新输出目录。G4 仍缺近期 2026 强基线、消融、完整成本分析、真实主表和新颖性复核，保持 INSUFFICIENT_EVIDENCE。

正式模型运行还要求项目自身存在 clean、已提交的 Git checkout；本次交付目录没有 .git，因此记录 source snapshot 和 code_commit=null，不伪造 commit。发布研究协议前先将当前代码纳入你自己的版本库并提交，再在目标环境重新冻结 plan/fingerprint。仅有上游 SpinQuant commit 不等于本项目代码已版本化。
