# 一次前向、一次反向：把 ROSA 当普通组件使用

[English](native-training.md) · [算法推导](../ALGORITHM.zh-CN.md) · [C API](native-api.md) · [当前实测](subquadratic-results.zh-CN.md)

层里只有一个 C++17 信用引擎。C++ 负责打包、硬路由、有限信用、局部信任模型、选中的联合角点和 logit 梯度；PyTorch 负责稠密投影、设备与布局转换，以及接入正常 autograd 图。整个 batch 一次原生前向调用、一次原生反向调用。

## 同一引擎中的三个选择

| 选项 | 默认 | 含义 |
|---|---|---|
| `credit_order` | `"second"` | `"first"` 是原线性有限信用；`"second"` 加入对角信任项和选中的联合 bit 协作 |
| `randomized` | `True` | 有界完整查询键缓存；答案和数值运算顺序不变 |
| `backward_mode` | `"exact"` | 完整计算；`"approximate"` 显式允许残差行或数值通道估计 |

```python
from conv_rosa_transformer.rosa_layer import NBitQKVRosa

first = NBitQKVRosa(128, 8, credit_order="first")
second = NBitQKVRosa(128, 8)  # 默认对角信任项和一个联合 pair
approximate = NBitQKVRosa(
    128, 8, credit_order="second", backward_mode="approximate",
    approximation_rows=32, approximation_channels=0,
)
channels = NBitQKVRosa(
    128, 64, credit_order="second", backward_mode="approximate",
    approximation_channels=8,
)
```

通道预算为零时，保留全部数值通道，使用残差行估计；通道预算为正时，在中心化数值特征中做带权采样，再完整运行结构算法，该分支不使用 `approximation_rows`。硬前向始终完整，目标和误差约定见[近似教程](randomized-backward.zh-CN.md)。

训练循环照常写：

```python
optimizer.zero_grad(set_to_none=True)
prediction = model(x)        # model 中包含 ROSA
loss = criterion(prediction, target)
loss.backward()
optimizer.step()
```

联合角点是在组件内部求值，外围模型、后续层和任务损失都只运行一次。一阶模式没有联合回放；默认最多选一对 bit 时，完整模式至多多做一次内部修改后路由求值，近似模式可能再补两次单点求值，让选中的联合块使用完整局部角点代价。

## 一阶信用与 logit 映射

一次 bit 翻转造成完整硬输出变化 $\delta$，目标为 $C^{(1)}=g^\top\delta$，其中 $g=\partial L/\partial Y$。这是反事实输出有限差分，不是阈值量化的经典导数，也没有重跑任意非线性下游损失。

当前硬 bit 为 $x$ 时，bit-1 减 bit-0 的定向信用是 $(1-2x)C$。默认 `credit_to_logit="bernoulli"` 再乘 $p(1-p)/\tau$，其中 $p=\sigma(z/\tau)$；`"identity"` 直接保留定向信用。Q/K/V 都遵守同一映射约定。

V 不改变匹配路由，它的信用和空嵌入、载荷尺度的普通梯度，都沿原路由做 $O(TD)$ 的累加。Q/K 使用[主教程](../ALGORITHM.zh-CN.md)中的共享删除与修复构造。

## 为什么二阶选项必须先指定一个模型

正常反向只给组件一个 $g$。$L_a(Y)=g_0^\top(Y-Y_0)+\tfrac12a\|Y-Y_0\|^2$ 这一族损失，在 $Y_0$ 的梯度相同，曲率却不同。因此不能凭这一个上游向量恢复任意下游 Hessian。

这里明确选择局部对角模型：

```math
C^{(2)}(\delta)=g^\top\delta+\tfrac12\sum_{t,j}h_{tj}\delta_{tj}^2.
```

输出集合的直径为 $\Delta_Y=\max(2\|e_1\|_2,\||e_0|+|e_1|\|_2)$。忽略零幅值的数值下限，默认 `trust_diagonal=0.5` 给出 $h_{tj}=\|g_t\|_2/\Delta_Y$，同一行各坐标共用；一般 `trust_diagonal=\lambda` 再乘 $2\lambda$。

沿负梯度、长度 $a\le\Delta_Y$ 的位移，在默认模型中的代价为 $-a\|g_t\|_2+a^2\|g_t\|_2/(2\Delta_Y)\le-a\|g_t\|_2/2<0$。这个尺度使有益的有限跳变仍然有益，同时惩罚过大的路由变化。结论只针对所选局部模型，不保证真实任务损失下降。Q/K 路由修改使用该项；V 保持原线性信用。

结构引擎不逐个重算二次位移。非空载荷都是带符号的 $e_1$ 顶点，逐坐标平方固定；先中心化，再加一个空载荷指示通道，就能转成至多增加一维的线性投影。这样既保留次二次结构，也减少特征构造时的抵消。

## 联合 bit 能越过单点信用看不见的障碍

两个单点修改都可能没有有效信号，但合起来有用。完整局部角点代价为 $C_i,C_j,C_{ij}$ 时，令 $\kappa=C_{ij}-C_i-C_j$。若 $\alpha_i,\alpha_j$ 是局部翻转概率，则

```math
F=\alpha_iC_i+\alpha_jC_j+\alpha_i\alpha_j\kappa,
\qquad \partial_{z_i}F=\alpha_i'(C_i+\alpha_j\kappa).
```

提案只检查有限行，保留至多 `joint_pairs` 个互不相交的块；默认每流一对，至多看 32 个提案行。它不枚举所有 bit pair，也不求解不同块之间的全部相互作用。固定预算 $J$ 的索引回放额外花 $\widetilde O(JT(D+1))$；预算随长度增长时必须把这项计入。

完整模式已有单点角点；采样过数值通道或残差时，会先完整重算选中的单点角点。联合角点始终使用完整的内部索引路由求值。选中坐标的梯度在进入共享投影参数之前被替换。

## 随机性、浮点与多个损失

精确缓存比较完整查询键，冲突触发重算。输入、局部规则和其余种子固定时，开关缓存的本引擎梯度逐 bit 一致，但未必每种输入都更快。提案随机性独立，关闭缓存不会关闭二阶提案。

完整模式包含所有数学信用项，但 binary64 分组与 FFT 可以和历史直接累加产生不同舍入；近似模式还多了采样误差。代数完整或条件无偏，都不意味着旧 float32 训练轨迹逐 bit 一致。见[数值约定](algorithm-reference.zh-CN.md)与[实测](subquadratic-results.zh-CN.md)。

信任系数与活跃行提案依赖 $g$，因此两个损失分别 backward 再累加，不必等于先相加再一次 backward。需要组合规则时，先合并损失。固定损失的正比例缩放受支持；硬信用的 double autograd 明确不支持。

前向缓存冻结硬码、logits、嵌入、选项和种子。精确缓存开关不改变框架 RNG 消耗；近似与联合提案使用保存的局部种子，评估不消耗训练提案随机数。检查点使用框架正常的 RNG 保存机制。适配器会传递非有限上游梯度，使普通 GradScaler 能发现溢出并跳过更新。

## 原生接口与集成

同一实现可通过 [training.h](../conv_rosa_transformer/csrc/training.h) 或 [NumPy 适配器](../conv_rosa_transformer/native_training.py)调用。先用默认函数初始化选项。C 中切到一阶，要一致地修改三个关联字段：

```cpp
RosaTrainingOptions options;
rosa_training_defaults(&options);
options.credit_order = 1;
options.joint_pairs = 0;
options.trust_diagonal = 0.0;
```

Python 的 `credit_order="first"` 自动给出这些默认值。C 的完整与近似入口都调用同一个引擎；单独的近似结构保存工作预算，不是后端选择器。数组布局、内存所有权、状态码和计数见 [C API](native-api.md)。

```bash
c++ -O3 -std=c++17 -ffp-contract=off examples/native_training.cpp \
    conv_rosa_transformer/csrc/unified_core.cpp -o /tmp/rosa-native
/tmp/rosa-native
/tmp/rosa-native --first --deterministic
/tmp/rosa-native --approximate
```

这是 CPU C++17 代码，CUDA 张量有传输和同步成本。稠密投影及模型状态仍是普通 PyTorch 参数，保存权重时也应保存训练配置。信用阶数和近似预算应按单位时间任务损失、验证表现、数值误差和内存选择，不能只看复杂度指数。

## 迁移已有调用

从 Python 构造器和 NumPy 调用中删除 `credit_engine`、`repair_samples`，它们已不再接受。一阶／对角＋联合由 `credit_order` 选择；采样使用 `backward_mode="approximate"`，再设置行或通道预算。C 选项保留结构体布局，但 `repair_budget` 已保留不用，必须为零。
