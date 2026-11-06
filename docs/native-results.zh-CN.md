# 原生集成的历史证据

[English](native-results.md) · [当前实测](subquadratic-results.zh-CN.md) · [当前训练接口](native-training.zh-CN.md)

仓库唯一反向引擎的当前测量在 [subquadratic-results.zh-CN.md](subquadratic-results.zh-CN.md)。下面的文件属于更早的集成实验，以各自记录的源码和二进制身份为准。它们的耗时、默认配置、算法和训练轨迹都是**历史证据**，不是当前实现的测量或使用说明。

| 保留的原始记录 | 历史范围 |
|---|---|
| [native_component.json](../benchmarks/results/native_component.json) | 硬组件、完整层耗时及当时的数值比较 |
| [exact_randomized.json](../benchmarks/results/exact_randomized.json) | 早期搜索加速实验，不是当前的查询缓存机制 |
| [exact_randomized_provenance.json](../benchmarks/results/exact_randomized_provenance.json) | 该实验的源码和探针身份 |
| [native_random_training_summary.json](../research/results/native_random_training_summary.json) | 训练与留出集表现，包含退步 |
| [native_random_roundoff_trace.json](../research/results/native_random_roundoff_trace.json) | 首次数值分叉追踪 |
| [native_cpp_validation.json](../research/results/native_cpp_validation.json) | 对应历史构建的验证记录 |

旧运行时、探针快照和可执行基准脚本不再作为备选后端保留。复现旧数字需要记录中的历史源码和环境；运行当前检出是一次新实验。保留原始记录，是为了不让好的训练或速度结果掩盖负例和出处。

当前算法请读[主教程](../ALGORITHM.zh-CN.md)、[精确缓存约定](exact-randomized.zh-CN.md)和[近似教程](randomized-backward.zh-CN.md)。本次检出的运行方法见[当前基准说明](../benchmarks/README.md)与[测试说明](../tests/README.md)。
