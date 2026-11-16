# 故障定位与断点恢复提示词

先读取最近失败run的config/environment/manifest/stdout/status，不要重新下载全部模型或修改所有依赖。把问题归类：文件损坏、权限、GPU架构/扩展、OOM、数据协议、数学错误、后端缺失、性能不达标。

- safetensors疑似损坏：检查index、分片尺寸、hash与权威metadata；79字节等异常不能继续加载；只重取有问题的分片，保留准确错误。
- notebook卡住：检查子进程PID/returncode、日志mtime/尾部、CPU/GPU利用率，保证轮询可中断；不要死读stdout；KeyboardInterrupt应终止进程组并写状态。
- OOM：报告分阶段allocated/reserved/CPU峰值；优先减缓存/流式/词表分块，不更改最终量化配置并把结果当可比。
- 事件爆炸：分别统计input/out、Nu/N、owner/code events，先查漏/重复事件及稳定筛选；不得截断组。
- 恢复：只有code/model/data/config/dtype/environment指纹匹配才跳过成功run，否则新建run_id并记录parent。迁移checkpoint需检查R.bin确实被包含。

修复具体原因，增加一项能复现该问题的必要回归检查，重跑受影响范围并更新日志。不要用try/except吞错或伪造PASS。
