# 在新的 AutoDL / Linux GPU 服务器启动

更新：2026-11-20。本文命令在**服务器的 Bash 终端**执行。先把本轮源码提交并 push 到 GitHub，新服务器才能 clone 到新增的缓存迁移参数。本机未连接或操作用户的新服务器。

当前研究状态：正式 A100 G1 为 FAIL，01c 及后续限定修正已按预定条件停止。当前下一项是六投影的跨层行泛化回放。这里启动该诊断；不启动 G2、GPTQ 全模型 canary 或完整实验矩阵，也不修改门槛。已有科学结果保留。

## 1. 连接和检查镜像

在 VS Code 安装 Remote-SSH，使用平台提供的 SSH 主机、端口、用户连接。Python/Jupyter 扩展安装在远端，随后打开服务器上的项目目录。[AutoDL 官方说明](https://www.autodl.com/docs/vscode/)

选择 Python 3.11 或 3.12、已有可用 PyTorch/CUDA 的镜像。保持宿主已验证的 torch；Blackwell 还需要镜像实际支持其架构，不能照搬本机 cu118。以下只是探测，不修改依赖：

```bash
nvidia-smi
python - <<'PY'
import sys, torch
print('Python:', sys.version)
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
assert sys.version_info[:2] in ((3, 11), (3, 12)), 'Use a Python 3.11/3.12 image for the pinned HF packages'
assert torch.cuda.is_available(), 'The image must expose a working CUDA device'
print('GPU:', torch.cuda.get_device_name(0))
print('Visible GiB:', torch.cuda.get_device_properties(0).total_memory / 2**30)
print('Capability:', torch.cuda.get_device_capability(0))
print('BF16:', torch.cuda.is_bf16_supported())
assert torch.cuda.is_bf16_supported(), 'The archived 7B study uses BF16; do not silently change its dtype'
x = torch.randn(512, 512, device='cuda')
y = x @ x.T
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print('CUDA GEMM passed')
PY
```

上述检查失败时先解决镜像/驱动问题。GPU 型号和版本会写入新运行记录；换卡后的时间不能与旧 A100 时间拼接成同卡速度比较。

长任务建议先进入 `tmux new -s cellquant`，然后在该终端继续后面的环境设置和登录。断开时按 Ctrl+B 再按 D，重新连接后运行 `tmux attach -t cellquant`。未安装 tmux 时先在镜像中安装它；无需把 Jupyter 端口开放到公网。

## 2. clone 到本地数据盘

```bash
# AutoDL 示例；其他 Linux 服务器改成实际可写的数据盘目录。
export CELLQUANT_DATA_DISK=/root/autodl-tmp
mkdir -p "$CELLQUANT_DATA_DISK"
cd "$CELLQUANT_DATA_DISK"
git clone https://github.com/BoldHu/CellQuant.git
cd CellQuant
git rev-parse HEAD

python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pip install -r requirements/hf_optional.txt
```

`--system-site-packages` 用于复用宿主 torch；项目的 HF 依赖在新虚拟环境中安装。每次重新打开终端先 `cd` 到项目并激活 `.venv`。已有 checkout 时使用 `git pull --ff-only`，不覆盖本地未保存修改。

## 3. 设置本机缓存、登录 HF

```bash
export CELLQUANT_PROJECT_ROOT="$PWD"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export CELLQUANT_CACHE_DIR="$HF_HUB_CACHE"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
hf auth login
```

交互输入具有 LLaMA-2 模型访问权限的 token，不把 token 写进命令、输出或 Git。模型和数据在本地 SSD 上计算；AutoDL 不使用 `google.colab.drive.mount`。如果打开旧 Colab notebook，设 `MOUNT_DRIVE=False`，选择远端 `.venv/bin/python`。

可以把旧服务器**完整的 HF hub 缓存目录**复制到新 `HF_HUB_CACHE`，保留 snapshots/blobs 和符号链接，然后由实验校验固定分片。也可以首次重新下载；下面的 `--allow-model-download` 允许下载归档指定的 revision。不要把 Drive 的 `objects/snapshots` 备份结构当成 HF hub 缓存直接复制，它须经过原 workspace 恢复器。

已有拟合轨迹、文档清单与证据 zip 已纳入仓库，当前回放不要求旧 `/content/CellQuant/runs`。准备器会生成新路径与新协议，核验旧轨迹和文档哈希。旧 `.venv` 和 CUDA 扩展不跨服务器复制。

## 4. 固定 SpinQuant 和验证扩展

```bash
git clone https://github.com/facebookresearch/SpinQuant.git third_party/SpinQuant
git -C third_party/SpinQuant checkout 8f47aa3f00e8662caf1a484153920a07e5281c3a
python scripts/prepare_spinquant.py

nvcc --version
python -m pip install packaging ninja
python -m pip install fast-hadamard-transform==1.1.0 --no-build-isolation
```

已有 SpinQuant checkout 时只核对固定 commit 和 tracked tree，不再次 clone。FHT 1.1.0 来自项目先前实际环境；新 GPU/Torch 上必须通过下一步的实际 CUDA 检查，不能将安装成功当兼容性通过。失败时保留日志，核对实际 CUDA 编译器、torch 和 GPU，不静默换 CPU 或降低实验精度。[FHT 上游](https://github.com/Dao-AILab/fast-hadamard-transform)

```bash
python scripts/run_math_checks.py --output runs/00_math/reference
python scripts/run_production_checks.py --output runs/00_math/production
python scripts/run_gpu_resident_checks.py \
  --output runs/new_server_gpu_check \
  --fht-backend extension --full-dimension
python -m pip freeze > runs/new_server_requirements.txt
python -m ipykernel install --user --name cellquant --display-name 'CellQuant'
```

GPU 检查应返回 `PASS_ENGINEERING_ONLY`；它包含小模型全 CUDA 放置、采集和求解检查，不代表 7B 或 G1 通过。重复检查时给 `run_gpu_resident_checks.py` 换一个新的输出目录，保留原结果。

## 5. 启动当前跨层回放

在项目根目录、已激活环境且完成 HF 登录的终端执行：

```bash
python -u scripts/run_output_matching.py --audit-row-transfer \
  --cache-dir "$HF_HUB_CACHE" \
  --dataset-cache-dir "$HF_DATASETS_CACHE" \
  --allow-model-download
```

首次运行会获取缺失的固定版本模型/数据并验证来源，随后重建已保存的旋转轨迹，测量已有 32 篇文章上的六个输入投影。原归档使用 `local_cache_only=true` 和 Colab 绝对路径；新增参数只在**新的准备目录**中迁移缓存并允许缺失文件下载，默认行为不变。`runtime_relocation.json` 记录路径/下载策略变化，其哈希和最终 runtime 一并写入新协议。

终端会输出新的 `Study directory` 和结果 zip 路径：

```text
runs/01_pilot/row_transfer_<唯一ID>/prepare.log
runs/01_pilot/row_transfer_<唯一ID>/study.log
runs/01_pilot/row_transfer_<唯一ID>/row_transfer.zip
```

这次是新运行，不恢复旧服务器进程或复用旧环境的执行指纹。若结果为 BLOCKED/RESOURCE_BLOCKED，按日志处理实际前置项；不改阈值，不写假的成功结果。

## 6. 保存结果与后续更新

将终端给出的 `row_transfer.zip`、准备/执行日志保存到独立持久存储，保留服务器 `runs/` 中的完整记录。AutoDL 本地盘不提供冗余可靠性保证；实例释放可能清空数据。[AutoDL 数据说明](https://www.autodl.com/docs/instance_data/)

每次本地修改后提交/push；服务器在作业结束后执行 `git pull --ff-only`。CLI 新进程会载入新源码；使用 notebook 时重启对应 kernel。不要在作业运行中更新其代码。未来若复现正式 G1，应另行准备新配置并重跑完整匹配对照；已有 FAIL 保留，迁移本身不授权 G2 或全量矩阵。
