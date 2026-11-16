# SpinQuant 最小集成与兼容审计

审计基线：`https://github.com/facebookresearch/SpinQuant`，commit `8f47aa3f00e8662caf1a484153920a07e5281c3a`。如果后续切换版本，先重新审计，不能仅更新commit字符串。

## 已确认的真实文件与符号

- `optimize_rotation.py`：`RotateModule`、`train`；global R1、每层共享head坐标R2。
- `train_utils/main.py`：`prepare_model`。
- `train_utils/modeling_llama_quant.py`：训练模型与语言模型CE。
- `train_utils/quant_linear.py`：`QuantizeLinear.forward`，每次前向FP64权重旋转。
- `train_utils/optimizer.py`：`SGDG`、`Cayley_loop`、`qr_retraction`。
- `eval_utils/main.py`：`ptq_model`。
- `eval_utils/rotation_utils.py`：`rotate_model`、`rotate_embeddings`、`rotate_head`、`rotate_attention_inputs`、`rotate_attention_output`、`rotate_mlp_input`、`rotate_mlp_output`、`rotate_ov_proj`、`QKRotationWrapper`。
- `utils/fuse_norm_utils.py`：`fuse_layer_norms`、`fuse_ln_linear`。
- `utils/quant_utils.py`：`ActQuantizer`、`WeightQuantizer`、`ActQuantWrapper`。
- `eval_utils/gptq_utils.py`：`gptq_fwrd`、`rtn_fwrd`。
- `utils/data_utils.py`：`get_wikitext2`、`CustomJsonDataset`。
- `utils/eval_utils.py`：`evaluator`。

没有在已审代码中确认的函数不得当现有API使用。C4和harness适配需要自己实现。

## 取得基线

```bash
git clone https://github.com/facebookresearch/SpinQuant.git third_party/SpinQuant
git -C third_party/SpinQuant checkout 8f47aa3f00e8662caf1a484153920a07e5281c3a
git -C third_party/SpinQuant rev-parse HEAD
```

仓库许可、模型许可分别遵循原项目。不要把下载的模型权重打进本项目zip，不假设有HF gated model访问权限。

## 先修FP等价，再量化

在审计版本中 `fuse_layer_norms` 无条件给embedding行减均值。这不是LLaMA RMSNorm的等价变换。本项目的LLaMA路径应取消该行中心化，保留gamma融合。补丁写到`patches/`并对所有自行复现方法一致应用。

三阶段比较：原始FP模型→合法gamma融合→完整R1/R2/R3/R4旋转。先用小随机LLaMA结构测试，再用真实模型固定token短序列。报告FP32和实际FP16/BF16的误差，不以一条绝对阈值掩盖大于数值基线的额外差异。

RMSNorm在gamma融合后满足N(Qx)=QN(x)。所有residual使用同一个Q。q/k/v/up/gate/lm_head使用WQᵀ；o/down输出使用QB。V/O配对R2可融合；RoPE后Q/K共同Hadamard、SwiGLU后down前Hadamard仍在线。R1=Qᵀ导出，不可因单层等价就推断任意逐层Q等价。

## Checkpoint schema

```python
state = {'R1': Q.T.contiguous().cpu().float()}
for layer_id, R2 in enumerate(r2_per_layer):
    state[f'model.layers.{layer_id}.self_attn.R2'] = R2.cpu().float()
torch.save(state, output_path)
```

每一层R2键都必须在。主方法固定R2，Hadamard/STE/grid归因对照也固定相同R2。`SpinQuant_full`基线保留官方R1/R2学习，并报告它与R1-only控制的区别。

## 官方默认协议差异

`scripts/10_optimize_rotation.sh`：8进程DDP、每卡batch1、2048、100step、BF16、checkpoint、lr1.5。变1卡时，若要匹配global batch8应gradient_accumulation_steps=8；仍需匹配样本序列。其旋转训练不是默认只取128条，128通常指GPTQ样本数。

`scripts/2_eval_ptq.sh`：默认W/A groupsize=-1；K/V脚本128，另有o输入/V输出head_dim强制分组。默认含a/k/v asym。LIRQuant论文的sym策略与它不完全相同。主协议和stock协议应分开配置、重跑基线。

README建议W16A4KV4学旋转、GPTQ得到W4A4KV4。新RTN校准器与最终GPTQ不相同，迁移收益必须在G2实测。

## 最小工程修改

1. 新建本包自己的`cellquant.spinquant_adapter`，不把所有算法塞进Trainer。
2. 新建完整group面板采集，保证基线和主方法输入规范一致。
3. 实现共享Q求解与原schema导出。
4. 加入Norm补丁、词表分块融合、CPU初始加载和按层处理选项；默认FP32融合，FP64仅参考验证可选。
5. 调用已有GPTQ/RTN/evaluator；外加C4与lm-evaluation-harness适配。
6. 保留上游commit与补丁hash，自己的代码另有commit/工作树hash。

## 最终PTQ已有命令（R.bin存在后）

```bash
torchrun --standalone --nproc_per_node=1 ptq.py \
  --input_model /models/Llama-2-7b-hf \
  --do_train False --do_eval True \
  --per_device_eval_batch_size 1 --model_max_length 2048 \
  --fp16 False --bf16 True --save_safetensors False \
  --w_bits 4 --a_bits 4 --k_bits 4 --v_bits 4 \
  --w_clip --k_groupsize 128 --v_groupsize 128 \
  --rotate --optimized_rotation_path /runs/method/R.bin \
  --nsamples 128 --seed 11
```

这条采用sym标志缺省；必须再次核对所用commit的默认参数。若使用stock asym，加入`--a_asym --k_asym --v_asym`且所有对照相同。目标T4若不支持BF16，应使用实际支持的dtype并重跑所有对照，不能继续写bf16=True。

## 部署声明

常规PTQ是反量化浮点模拟。此路径可比较数值质量和校准成本，不可报告真实packed INT4存储、Tensor Core吞吐或持久4bit KV cache压缩。R1-only算法不增加已有计算图在线算子，可以测相同float模拟后端的额外开销接近0；若要绝对INT4加速，另建真实kernel部署实验且先过硬件兼容。

## 关键来源

- https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/README.md
- https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/utils/fuse_norm_utils.py
- https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/train_utils/quant_linear.py
- https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/eval_utils/rotation_utils.py

## 专业RTX上的公平精度实现

RTX PRO 6000的96GB容量不意味着其FP64吞吐等同A100。若官方逐前向FP64旋转成为瓶颈，可实现保持R1/R2、CE、样本和优化器相同的`SpinQuant_full_FP32_port`：只将权重旋转计算改为FP32，并先在小模型/真实短步骤上与FP64参考核查梯度、正交残差和损失轨迹，再比较实际量化质量。该版本必须以“our FP32 implementation”明确标注，不称为原版bitwise复现。

主方法的效率结论必须与精度合格的这个FP32基线及2026快速方法比较，不能只用最慢FP64版获得加速。原版FP64校准时间可单独报告或标未跑。只要完整R1+R2算法、校准/评估协议和数值差异验证齐全，该受控port可作为canary完整SpinQuant比较；适配日志必须说明，不能只改名字。若其质量未验证，gate中的完整SpinQuant证据仍不足。

Canary不得直接用官方只读WikiText-2 test的默认evaluator反复调参。新adapter须先增加独立dev评估入口；正式test调用留到协议冻结后。
# 2026-11-15 实现记录补充

以下为本轮代码实际落地记录；原规范与接口审计内容保留。适配代码位于 `src/cellquant/spinquant_adapter.py`，训练位于 `spinquant_train.py`；`third_party/SpinQuant` 的固定 checkout 未修改。`prepare_spinquant.py` 生成 patch recipe v2 overlay 及补丁/文件哈希，未使用虚构的上游 wrapper 接口。

补丁覆盖 embedding 行中心化关闭、gamma 融合使用 FP32、显式 FHT dispatcher、GPTQ typed catcher/实际 sequence length/逐层 callback；可选 FP32 port 只改变原显式 FP64 旋转乘法的实现精度。原 FHT 缺失时不能自动切换为“扩展通过”。所有基线采用同样融合修正。

实际 tiny MHA/GQA 的 FP/GPTQ/RTN 与恢复记录见 `runs/build_m4_v3`、`runs/build_m4_gqa`；完整 R1/R2 CE/SGDG 两步与安全 RNG 恢复见 `runs/build_final/training*`。这些不构成真实 7B 或 Blackwell 验证。正式训练预算保留 100×8，使用全部合格 WikiText train blocks 排除独立 dev 文档，保留 Trainer 的跨 epoch 语义并记录实际独立 blocks 数；FP64/FP32 port 必须使用同一冻结流。由于分块/单卡累积协议有明确变化，不宣称逐 bit 复现上游八卡训练或论文原表。

