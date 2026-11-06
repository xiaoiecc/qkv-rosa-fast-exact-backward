# 原生次二次反向引擎

[English](subquadratic-engine.md) · [主教程](../ALGORITHM.zh-CN.md) · [完整证明](subquadratic-proof.md)

当前算法统一放在 [ALGORITHM.zh-CN.md](../ALGORITHM.zh-CN.md) 中讲解：从四次方的逐 bit 翻转，推到共享删除、周期出现轮廓、Q 轻重分治、短 K 字典、长 K 锚点，以及流式数值收缩。这是仓库唯一保留的反向引擎。

[算法参考索引](algorithm-reference.zh-CN.md)把不变量对应到 C++ 源码和独立测试；[训练教程](native-training.zh-CN.md)解释一阶信用、对角信任项与联合 bit、精确随机缓存及显式近似。[当前实测](subquadratic-results.zh-CN.md)给出完整反向耗时和误差。

固定打包位数、数值维数和联合 bit 预算时，完整构造的时间为 $\widetilde O(T^{7/4})$，空间为 $\widetilde O(T^{4/3})$。路由决策精确，binary64 聚合和 FFT 则遵守各自的浮点约定；复杂度界不代表每个实际长度都会更快。
