# 阶段4：迁移到AutoDL的96GB GPU

不要把用户简称RTX P6000当作型号事实。检测nvidia-smi、torch device name、total_memory、compute capability、MIG/可见显存。目标通常RTX PRO 6000 Blackwell 96GB，旧Quadro P6000只有24GB。

先运行notebook02的manifest与环境检查，再在真实GPU测试GEMM、所用FHT/量化扩展、短模型前向。旧SpinQuant依赖不必支持sm120；先做reference对照，不因CUDA初始化成功就宣称扩展兼容。不要盲目安装固定旧torch破坏平台环境。

迁移整个代码包、配置、gate原始证据、R.bin/稀疏角度、数据索引与哈希；大模型优先平台本地缓存。明确哪些文件跳过；校验SHA256与尺寸。不要复制HF token。保留Colab原始结果与新环境结果，跨dtype/硬件重跑同基线，不能合并计时。

execution_provenance.json合同：models={MODEL_ID:{model_revision,tokenizer_revision}}，resolved revision必须真实hex；data_manifest_path及其sha256，protocol_manifest_path及protocol_sha256，dtype。路径相对项目根或绝对；两个manifest必须存在且hash匹配。当前源代码快照和真实依赖/GPU指纹由runner自动加入；变化后不得resume旧结果。

检查中断→恢复一致性；实际峰值≤visible VRAM的85%。G1通过只进入有限canary；Colab资源不足最多先做limited_hardware_probe。G2及全部运行准备完成后写G3，notebook03只有G3 PASS且真实后端存在才执行full。
