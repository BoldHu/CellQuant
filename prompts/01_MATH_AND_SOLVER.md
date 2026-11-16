# 阶段1：把数学参考升级为生产求解器

读取AGENTS.md和docs/MATHEMATICAL_SPEC.md。已有math_reference仅是CPU小矩阵单pair真值。请先重跑tests，保留M=r孤立tie的回归案例，再实现可用于d=4096的production模块。

必须完成：完整量化组；WQ.T/QX和QB两种placement；单个globalQ跨面板共享；动态双侧scale owner+全部code事件；解析三角系数而非窄cell插值；half-angle数值根的残差验证；实际round-even边界与同时事件；包含theta=0；真实loss+BW/BA复核；安全尺度筛选；输入侧和输出侧分别计复杂度；事件预算与可恢复状态。

先支持sym AbsMax无clip，明确unsupported配置并拒绝，不默默退化。fixed-clip/asym可以后续添加，不能在当前定理里假设它们已覆盖。单个pair多个面板的最优角必须从聚合目标选择，不能给每面板独立选角。

生成3seed≥20case的G0 evidence；覆盖组内/跨组、零组、极值并列、同时threshold、输出侧、预算、单调性、screen/full等价。一般根隔离未形式认证就不使用certified一词。比较生产后端和reference codes/scales/loss，不只比热力图。

交付production源码、真实测试、事件/复杂度统计、数学范围说明、G0报告。参考guard不能删来冒充规模支持；不允许把完整组截短。
