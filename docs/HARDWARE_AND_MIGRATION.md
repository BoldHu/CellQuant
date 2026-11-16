# Colab → AutoDL 96GB GPU：硬件核实与迁移方案

核查日期：2026-11-15。本文件的容量计算来自模型大小与代码路径；除另行标注的项目实测外，不把估计写成 GPU benchmark。

## 1. 先纠正硬件名称

用户称“RTX P6000 96GB”，这里按**NVIDIA RTX PRO 6000 Blackwell 96GB**设计。NVIDIA 官方的 Quadro P6000 是 24GB GDDR5X，不是 96GB。RTX PRO 6000 Blackwell 的 Workstation、Max-Q Workstation、Server 版本均有 96GB，计算能力为 12.0；功耗、散热和持续性能不同，不能因为容量相同就混为同一 GPU。[1–3]

不假定 AutoDL 当前可租型号、价格或库存已经核实；实例启动后以实际设备为准。

必须记录：

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv
```

以及：

```python
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
print(torch.cuda.get_device_properties(0).total_memory / 2**30)
print(torch.cuda.get_arch_list())
```

96GB 是产品容量标签，PyTorch 报告的是可见 bytes；租赁设备可能被分区。若只看到 24/48GB，检查是否为 MIG/其他实例配置，不能按整卡96GB进行内存预算。[3]

## 2. 软件兼容性是第一道 GPU 门槛

PyTorch 2.7 发布时首次提供 Blackwell 支持和 CUDA 12.8 预编译轮子。[4] 这是支持代际信息，**不是推荐现在强制降级到 2.7，也不是声称任意较新版本都与本项目兼容**。

建议流程：

1. 保留 Colab 当前可用 PyTorch，用轻量依赖完成 CPU 数学检查；不要为一个测试无条件重装整个 CUDA 环境。
2. 在 AutoDL 创建独立环境，使用官方稳定、支持 SM120 的 PyTorch/CUDA 组合。记录镜像、driver、Python、Torch、CUDA runtime、Triton 的精确版本。
3. 不将 SpinQuant 旧 `requirement.txt` 原样覆盖现代 Blackwell 环境。分别检查 Transformers 私有接口、FHT、FlashAttention 和量化相关扩展。
4. 必须实际运行 FP32/BF16 GEMM，以及项目所用维度的 FHT；“能 import”不算通过。
5. 若扩展缺少 SM120 构建或 PTX支持，使用经过验证的源构建/兼容实现；可用纯 PyTorch FHT 做正确性回退，但其计时不能冒充优化 CUDA kernel。
6. Torch 支持 Blackwell 不意味着所有第三方扩展也支持。修改 `TORCH_CUDA_ARCH_LIST` 不能修复一个源代码本身不支持的 kernel。
7. 在改变依赖后重新检查浮点模型等价和量化算子。不同环境的数值或速度比较必须单独标注。

完整锁文件应由**实际通过 GPU smoke 的环境**导出，不能事先声称一个未运行的版本组合已经验证。

## 3. Colab 分层使用策略

Colab 不保证具体 GPU、持续可用资源或固定运行时长；即使付费，资源仍会变化。[5]

| 实际资源 | 推荐工作 | 不应承诺 |
|---|---|---|
| CPU | G0 小矩阵、量化边界、配置与恢复测试 | 真实 GPU 性能 |
| T4 约16GB | 流式单层/小面板；必要时CPU offload | 完整7B+大量缓存同时驻留；BF16性能 |
| L4/类似24GB | 7B单层或分块pilot，紧凑面板 | 原始SpinQuant全模型FP64反向传播一定可行 |
| A100/较大显存 | 更完整的真实层pilot；可尝试小规模GPTQ canary | 资源永远可用或自动达到G2 |

Notebook 应先检测 `torch.cuda.is_bf16_supported()` 再选择 BF16；不支持时使用 FP16并记录。数学小矩阵与根求解使用CPU FP64，不需要全模型FP64。

如果拿不到足够资源，应保存“资源不足”的真实状态。不要把 1B smoke 的性能填入 7B gate。

## 4. 存储与路径

建议所有路径由三个变量确定，而不是在代码中到处硬编码：

| 变量 | Colab 示例 | AutoDL 示例 |
|---|---|---|
| `PROJECT_ROOT` | `/content/cellquant_research_kit` | `/root/autodl-tmp/cellquant_research_kit` |
| `DATA_ROOT` | `/content/cellquant_data` | `/root/autodl-tmp/cellquant_data` |
| `RUNS_ROOT` | `/content/cellquant_runs` | `/root/autodl-tmp/cellquant_runs` |

Colab 的活动计算、Hugging Face缓存和随机读取面板放本地SSD；Google Drive用于阶段结束或定期同步的checkpoint和结果。不要直接从Drive反复读取模型分片，也不要向Drive频繁写入大量小文件。

AutoDL 常见数据盘为 `/root/autodl-tmp`。系统盘镜像和数据盘是不同迁移对象；官方文档提示本地数据盘无冗余保障，需要独立备份。[6–7] 不将“关机后保留”误解为永久可靠备份。

应同步的内容：

- 代码commit、必要patch、配置与锁文件；
- tokenizer/config revision、模型文件清单与校验哈希；
- 已采集面板及其来源索引，校准/验证token哈希；
- Q/R1/R2、优化状态、已接受的坐标对/角度、预算与RNG状态；
- 每阶段结果、日志、gate报告和运行manifest。

不直接迁移 Colab 的虚拟环境目录、缓存的 `.so`、编译扩展或 CUDA 二进制。新 GPU 上重新建立环境并验证。

## 5. 断点、长任务与 notebook

- 每个阶段可单独执行，重复运行检查已有 manifest/hash，避免重复下载和重训。
- checkpoint 使用临时文件写入成功后原子重命名；保留上一个可恢复版本。
- 每接受若干旋转或达到固定分钟间隔保存状态；记录事件求解器的当前坐标对及完成状态。
- 恢复时验证配置、模型、数据和代码哈希。出现不一致时创建新run，不悄悄续到旧目录。
- 长训练由独立CLI进程执行，Notebook负责启动、展示日志和读取状态。日志读取采用可中断、带超时的poll，不使用可能永久阻塞的逐行 stdout 循环。
- 全量模式必须显式选择，并读取 G2/G3 报告。默认模式为 smoke/pilot，不因“Run all”启动多模型全量矩阵。
- Notebook显示“任务正在运行”时应输出PID、日志路径、最后更新时间和完成比例；中断notebook后不得误删仍在运行的进程或覆盖输出。
- 远端长进程的启动和重连应使用环境实际支持的方式；不要在Colab采取规避运行限制的手段。

## 6. 显存与主机资源预算

下表是存储算术，不是实测峰值。GiB=2^30 bytes。

| 项目 | LLaMA-2 7B | LLaMA-3 8B | LLaMA-2 13B |
|---|---:|---:|---:|
| BF16基础权重 | 约12.55GiB | 约14.96GiB | 约24.25GiB |
| FP32全局R1 | 64MiB | 64MiB | 100MiB |
| GPTQ 128×2048 输入和输出缓存 | 4GiB | 4GiB | 5GiB |
| 最大FP32 Hessian单份 | 约0.45GiB | 约0.77GiB | 约0.71GiB |

还需考虑分解临时量、量化尺度/误差、词表投影、CUDA工作区、events和allocator。LLaMA-3 8B单个完整词表矩阵FP64副本约3.91GiB，建议按词表行分块融合。最终INT4文件大小不能代表校准显存。

主实验初始峰值规划：7B约24–40GiB；8B约28–48GiB；13B约38–60GiB。**这些范围尚未在RTX PRO 6000测量**。96GB整卡为7B/8B顺序运行提供充足规划空间，但不能保证全局事件缓存或错误的全层激活缓存一定放得下。

- CPU RAM：pilot 64GB起，7B/8B主实验推荐128GB；13B与多数据缓存推荐128–256GB。
- 磁盘：Colab pilot预留100–200GB；完整多模型实验预留0.5–1TB。
- 不同时加载两个大模型；不一次保存所有层完整激活。
- 单卡96GB目标无需FSDP来运行新R1校准；原SpinQuant对照若保留完整反向传播，单卡batch1+checkpoint/梯度累积需实测。
- 96GB显存不意味着原代码中的FP64稠密旋转很快。主方法只在小型根求解/验证上用FP64；不得从A100/H20的结果直接推算此卡时间。

## 7. 预算与停止规则

所有时间均为**投入预算**，不是承诺的运行时长。

1. CPU/G0：首先在当前环境跑通。
2. Colab真实层：先完成一个位置和一个坐标对计时，再决定面板与角度预算；不直接发起全部位置。
3. AutoDL硬件probe：最多4 GPU小时，用于兼容性、面板提取和一层GPTQ；不能将probe当方法通过。
4. AutoDL G2 canary：初始12–24 GPU小时预算；用实测修正，超预算则停止分析瓶颈。
5. 完整论文：最初可按200–600 GPU小时规划，但必须用G2的每阶段实测重新计算；Blackwell并不按A100固定倍率缩放。

必须分别记录 `capture_seconds`、`rotation_solve_seconds`、`fusion_seconds`、`gptq_seconds`、`evaluation_seconds`。比较总校准成本时包含前四项；包含评估的总项目耗时另列。

## 8. 一次完整迁移的完成定义

迁移只有在以下各项都有证据后才完成：

1. 新机器识别为预期GPU与可见显存；
2. 模型、数据、配置和代码哈希与Colab一致；
3. 新环境数值smoke通过；
4. 同一小面板在两环境得到一致到预定容差的量化输出和损失；
5. 合成G0和完整Transformer FP等价检查重新通过；
6. 一次中断恢复测试通过；
7. 一个完整W4A4KV4 canary完成并产生独立gate报告。

若只能完成1–4，应写“环境/面板迁移通过”，不能写“完整实验已验证”。

## 官方来源

1. NVIDIA, RTX PRO 6000 Blackwell Series: https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/
2. NVIDIA, Previous Generation Desktop GPUs / Quadro P6000: https://www.nvidia.com/en-sg/products/workstations/previous-quadro-desktop-gpus/
3. NVIDIA, Supported GPUs — MIG User Guide: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/supported-gpus.html
4. PyTorch, PyTorch 2.7 Release: https://pytorch.org/blog/pytorch-2-7/
5. Google Colab FAQ: https://research.google.com/colaboratory/faq.html
6. AutoDL, 实例数据与磁盘: https://www.autodl.com/docs/instance_data/
7. AutoDL, 迁移实例: https://www.autodl.com/docs/migrate_instance/

以上来源核查于2026-11-15。发布版本、供应商可用实例和驱动仍需在实际运行时复核。
