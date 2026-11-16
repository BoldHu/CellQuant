# 实际验证记录 — 2026-11-15

执行环境：Windows，Python 3.11.18，Torch 2.6.0+cu118，RTX 3050 Laptop 4GiB，driver 546.30。使用项目 `.venv` 复用宿主 torch，没有升级或降级宿主 PyTorch。依赖快照见 `runs/build_final/dependencies.json`；这是观测记录，不是 Blackwell 验证锁。

| 实际执行 | 输出 | 结果 |
|---|---|---|
| 初始 `python -m pytest -q`（UTF-8） | `runs/build_m0/command_0.log` | 原 62 项 PASS；首次 GBK 读中文失败已定位并显式 UTF-8 修复 |
| 最终 `python -m pytest -q` | `runs/build_final/release_pytest.log` | 116 passed；包含本机 CUDA 小检查，未把 skipped 算 PASS |
| `scripts/run_math_checks.py` | `runs/00_math/reference/` | 参考检查通过，保留原小矩阵 guard |
| `scripts/run_production_checks.py` | `runs/build_final/production/` | 3 seeds / 24 cases，G0 PASS；最大胞元损失误差约 1.34e-15，预算违规 0，接受后增损 0 |
| `scripts/run_production_checks.py --full-dimension` | `runs/build_m2/dimension_4096.json` | COMPLETE；d=4096、FFN=11008、2 tokens、1 pair、±0.0001。该结果仅证明完整形状路径能运行 |
| `scripts/run_spinquant_smoke.py --device cuda --gptq --fht-backend torch_reference` | `runs/build_m4_v3/` | 随机 tiny MHA，融合/旋转约 1e-7，GPTQ 有限，恢复参数差 0，RTN 后权重精确恢复 |
| 同上加 `--kv-heads 1` | `runs/build_m4_gqa/` | 随机 tiny GQA 同样通过，完整 down 采集形状保留 |
| `scripts/run_spinquant_training_smoke.py`，再 `--resume .../checkpoint-1` | `runs/build_final/training*` | 实际 CE/SGDG 两步 R1+R2；安全 RNG/优化器/调度器恢复，比较记录另存 |
| `scripts/check_fp32_port.py` + FP32 tiny training | `runs/build_m4_fp32/`, `runs/build_m4_fp32_training/` | NUMERICAL_ONLY；输出/R1/R2 梯度相对误差约 1e-7；非真实质量资格 |
| `scripts/run_harness_smoke.py` | `runs/build_final/harness/` | 六个固定 task 的本地 fixtures、实际 HFLM、逐题输出通过；不是 benchmark 成绩 |
| `scripts/run_notebook.py ... --engine inprocess` × 5 | `runs/build_final/notebooks/` | 五本默认路径均 returncode=0，原始代码与执行副本一致 |
| 目标 kernel probe | `runs/build_final/gpu_probe.json` | 本机 BF16 GEMM 误差约 0.00163；FHT/FlashAttention/Triton MISSING，总状态 INCOMPLETE |
| 空证据与空真实结果分析 | `runs/build_final/gates/`, `runs/04_analysis/` | G1–G4 INSUFFICIENT_EVIDENCE；无预填论文图表 |

`runs/build_final/final_execution.json` 记录最终检查的真实 subprocess 命令、返回码与耗时；每条长命令有日志/心跳。tiny training 恢复的 Trainer 汇总 train_loss 只覆盖本次剩余 step，不能当连续训练全程均值比较；验证依据是恢复后参数与连续运行逐项一致。

上游保持原样，patch recipe v2 与 SHA 在 overlay manifest / `patches/`。评估 harness 固定为 `d6de81643928d653435c431bae19945d41d32520`。根目录没有 `.git`，日志中 harness 自动询问根 commit 会显示 not-a-git-repository；独立校验的上游 commit 和代码 source snapshot 已记录，未虚构根 commit。

## 明确未运行

没有预训练 7B/8B G1、完整模型 GPTQ 质量、C4/六任务真实评估、Blackwell 扩展实测、正式时间/显存对比、近期强基线或论文消融。原因是当前 GPU 4GiB、CPU 可用 RAM 约 2GiB，不满足实现的模型/面板工作集；模型/data revisions 尚需在目标运行中真实解析。下一步执行 [运行手册](docs/RUNBOOK.md) 的真实数据准备与 G1 命令；不得用这里的 CPU/tiny PASS 代替真实 gate。

最终增加了在 Q 已写入、factor/log 尚未发布时注入 KeyboardInterrupt 的回归测试：恢复使用最后一个完整原子事务，最终 Q 与连续运行逐项一致。源代码/数据指纹相关定向检查另有 28 项通过，见 final_provenance_tests.log。

## 2026-11-16 工作流增补验证

新增 GitHub/VS Code/Colab 初始化、Drive 文件快照与独立权重缓存入口。历史构建记录保持原样；本轮接手已存在根 Git checkout，master 初始 commit `9a53bf7`，本次验证包含未提交的工作树变更。

| 实际检查 | 记录 | 结果与边界 |
|---|---|---|
| 完整 pytest | `evidence/workspace_20261116/pytest.log` | 125 passed in 20.19s |
| CLI 选项收尾后相关测试 | `evidence/workspace_20261116/final_targeted.log` | 25 passed in 7.01s |
| 六本 notebook 默认路径 | `evidence/workspace_20261116/execution.json` 与 `notebooks/` | 全部退出码 0，IPython inprocess；初始化 notebook 离线模式，未 clone/挂载 |
| CLI 备份/恢复 | `evidence/workspace_20261116/cli_roundtrip.json` | 6 条命令成功、二次传输 0 bytes、恢复文件逐字节一致；本地目录 fixture |
| G0 | `evidence/workspace_20261116/G0.json` | 数学 notebook 真实重算 PASS，不代表 G1 |

新测试覆盖内容去重、版本保留、哈希损坏、路径逃逸、原子传输中断、源变化、冲突、缓存路径和固定模型下载/分片校验入口。模型入口使用本地 safetensors fixture 并替换网络下载，不算真实模型下载或质量实验。记录/代码 hash 索引见 `evidence/workspace_20261116/index.json`。

真实 Google 登录、Drive FUSE、VS Code 云端 kernel、大模型下载/缓存吞吐与 Colab GPU G1 **NOT_RUN**；需在用户实际会话中运行 [初始化 notebook](notebooks/00_Colab_GitHub_Workspace.ipynb)。科学 gate 文件哈希与 2026-11-15 相同。
