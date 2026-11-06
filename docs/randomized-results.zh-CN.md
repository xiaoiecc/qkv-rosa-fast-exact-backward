# 近似反向的历史证据

[English](randomized-results.md) · [当前实测](subquadratic-results.zh-CN.md) · [当前估计器](randomized-backward.zh-CN.md)

当前反向延迟、工作区和近似误差在 [subquadratic-results.zh-CN.md](subquadratic-results.zh-CN.md) 中报告。下面的旧实验以各自 JSON 记录的源码、二进制哈希为准，不描述当前运行时、默认选项、缓存或数值通道实现。

| 保留的原始记录 | 历史范围 |
|---|---|
| [random_final_native.json](../benchmarks/results/random_final_native.json) | 早期完整与近似反向耗时、梯度误差样本 |
| [randomized_final_manifest.json](../research/results/randomized_final_manifest.json) | 证据清单与实现身份 |
| [training_round_final_summary.json](../research/results/training_round_final_summary.json) | 结构化 prefix-copy 训练及当时结果 |
| [training_round_final_channel_controls.json](../research/results/training_round_final_channel_controls.json) | 通道估计控制实验，包括一阶抵消失败 |
| [training_round_final_channel_controls_full.json](../research/results/training_round_final_channel_controls_full.json) | 对应的全通道控制实验 |
| [native_random_noise_ceiling.json](../research/results/native_random_noise_ceiling.json) | 早期整条 bit 通道噪声实验，不是当前估计器的下界 |

旧可执行实现和复现脚本不再作为备选算法交付。原始记录保留了成功、慢例、估计失败和特定种子的结果，不构成新的性能宣传。历史标签 `exact`、`final`、`A8` 只按当时实验的定义解释，不能直接替换成今天的 API 调用。

当前引擎提供完整键核验缓存，以及显式残差行或中心化通道采样。约定分别见[精确随机化](exact-randomized.zh-CN.md)与[近似反向](randomized-backward.zh-CN.md)。本次检出的运行方式见[当前基准说明](../benchmarks/README.md)和[训练接口](native-training.zh-CN.md)。
