# 给 Codex 的总提示词：构建完整研究项目

你是研究工程负责人。请在当前仓库直接实现，不要只给计划。先读根目录 AGENTS.md、PROJECT_PLAN.md、docs/IMPLEMENTATION_STATUS.md、docs/MATHEMATICAL_SPEC.md、docs/SPINQUANT_INTEGRATION.md、docs/GATES_AND_PAPER_CRITERIA.md 和 configs/gate_rules.json；检查已有文件，保留已实现且验证正确的功能。

目标：基于固定版本 facebookresearch/SpinQuant，构建“尺度约束的双侧动态量化胞元正交校准”完整项目。使用一个global R1，保持严格实数正交、基础模型参数冻结、R2/R3/R4主方案固定；可以直接导出原始R1/R2 checkpoint。主要模型LLaMA-2 7B、原始LLaMA-3 8B，主设置W4A4KV4。先Colab pilot，后AutoDL真实96GB RTX PRO 6000 Blackwell实例；所有实验从notebook运行，核心逻辑在Python包。

请持续完成所有现有权限与资源允许的代码构建工作，按以下顺序交付可运行里程碑，不用每完成一小步再问我是否继续：

1. 复现已有CPU测试；核对上游commit与真实符号；建立实现记录。已存在的CPU参考只支持小矩阵输入侧单pair，不能冒充全模型后端。
2. 完成production事件求解器：完整量化组、多个面板共享同一Q、输入侧WQ.T/QX和输出侧QB、解析cell系数、实际round边界、约束候选验证；先与reference小数据逐项一致，再处理真实d=4096。不要删掉reference guard或截断通道冒充扩展。
3. 实现按真实事件数计费的安全筛选和增量维护。记录事件生成/排序/系数更新/求根/复核各自耗时。CPU/GPU混合可以，不能每角重GEMM后声称快速算法。
4. 实现同目标/同角度预算/同时间STE、grid和fixed-scale对照，以及真实层G1报告。所有面板使用同一个globalQ；正确处理down输出侧。
5. 实现SpinQuant adapter：保留上游，记录补丁，关闭LLaMA RMSNorm路径的embedding行中心化但保留gamma融合；完成FP等价验证、R1=Q.T和全部R2导出、GPTQ W4A4KV4。对所有自有基线采用同样等价修正。原版R1+R2 SpinQuant与R1-only归因控制分开。
6. 在Blackwell上若原版FP64重旋转过慢，做严格标注和验证的SpinQuant_full_FP32_port；保持R1/R2、CE、优化器和样本不变。效率结论必须对精度合格的FP32版及2026强方法，不只对最慢旧版。
7. 实现 cellquant.experiment 后端，暴露 BACKEND_CONTRACT_VERSION='1.0'、run_experiment callable，并支持 python -m cellquant.experiment --config PATH。补全实际模型/数据/revision与execution_provenance。runner已有默认阻塞和恢复逻辑不要绕过。
8. 补全C4和固定commit的六任务评估、逐序列NLL与逐题预测、配对bootstrap、参数/时间/内存统计。canary使用独立dev，正式test只在协议冻结后运行。
9. 完成五个notebook的真实路径。00数学/环境，01真实层与G1，02迁移/canary/G2，03G3后全量，04真实结果分析/G4。默认smoke/manifest-only；缺后端或证据不能生成PASS。
10. 给出实际执行过的CPU/GPU命令、测试和未完成项；更新状态、工作日志、依赖快照、README。没有GPU时继续完成能验证的代码；GPU和真实模型结果明确NOT_RUN，不能编造。

所有研究结论遵守：动态双侧RTN的理论不直接覆盖GPTQ；小矩阵降低NMSE不等于LLM提升；fake quant不等于真实INT4加速；ParoQuant/TORQ/PiSO/OBCD/CROQuant是最近先例，不能宣称first Givens或first exact angle search。完整实验按G0→G1→G2→G3推进；结果达到G4才作为完整论文候选，绝非录用保证。

实现前可以输出简短执行计划，然后立即工作。遇到依赖/权限/资源阻塞，保留准确日志，完成其他独立工作，最终说明具体缺什么以及可执行下一步，不要返回空壳框架。
