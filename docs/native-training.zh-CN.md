# 一次前向、一次反向：C++ 训练组件

[English](native-training.md) · [随机化的完整性与成本](exact-randomized.zh-CN.md)

QKV-ROSA 的训练接口现在分成两层。C++ 接收连续数组，负责打包、硬匹配、
有限信用、联合角点和输入梯度；PyTorch 负责稠密投影、设备与布局转换，以及
把返回的梯度接回普通计算图。独立 C++ 程序也能调用同一个接口，不依赖
Python、NumPy、LibTorch 或 PyTorch 的 C++ 头文件。

## 两个选择，各管一件事

```python
from conv_rosa_transformer.rosa_layer import NBitQKVRosa

# 原来的一阶单 bit 信用。
first = NBitQKVRosa(128, 8, credit_order="first", randomized=False)

# 对角信任项与一个联合 bit 块；开启答案精确的随机化搜索。
second = NBitQKVRosa(128, 8, credit_order="second", randomized=True)
```

`credit_order` 选择训练规则。`randomized` 选择同一规则的精确搜索方式：
它不会抽掉信用坐标，不会近似候选间的最大值，也不会改变联合块的随机种子。
开关前后应当得到相同的数值结果，区别是为找到这些结果执行多少查找。

训练循环保持普通形式：

```python
optimizer.zero_grad(set_to_none=True)
prediction = model(x)       # model 内包含 ROSA
loss = criterion(prediction, target)
loss.backward()
optimizer.step()
```

两种规则都使用硬 Q/K/V 前向和相同的最长、最新、因果路由。每个
batch/group 是一条独立流。二阶模式默认最多选一个联合块，因此每条流在
backward 内最多重放一次 ROSA 匹配；它不重新运行下游网络或任务 loss。
第一阶模式没有这项联合重放。

`randomized=False` 关闭的是新增精确搜索加速。二阶规则原本就用一个有固定
预算的联合提案器；这个提案器在两个开关状态下使用相同随机数。
第一阶没有联合提案时，精确加速也不会额外消耗 PyTorch RNG，避免改变其他
层的 Dropout 或随机数据处理。`.eval()` 不消耗这些训练随机数。

## 一阶模式到底计算什么

对于一个硬 bit 编辑造成的完整输出变化 $`\delta`$，一阶信用为：

```math
C^{(1)}=g^\top\delta,
\qquad g=\partial L/\partial Y.
```

引擎对单 bit Q/K/V 编辑精确计算这个有限差分对象。它不是阈值函数的经典
导数，也不是任意非线性下游 loss 的完整重算。默认映射把信用定向为
“bit 取 1 相对取 0”，再乘 sigmoid 概率对 logit 的导数。

## 二阶模式：对角信任模型与联合配合

普通 backward 只给组件一个上游梯度。对于同一输出 $`Y_0`$，

```math
L_a(Y)=g_0^\top(Y-Y_0)+\tfrac12a\|Y-Y_0\|^2
```

可以有相同的梯度 $`g_0`$ 和不同曲率。因此，本接口不声称自动恢复了任意
下游模型的真实 Hessian。“二阶”描述的是明确选定的局部二次信用模型：

```math
C^{(2)}(\delta)=g^\top\delta+
\tfrac12\sum_{t,j}h_{tj}\delta_{tj}^2.
```

一组输出的可能值包含空输出 $`e_0`$ 和所有带符号的 payload 顶点。
取这个集合的直径：

```math
\Delta_Y=\max\left(2\|e_1\|_2,\ \||e_0|+|e_1|\|_2\right),
\qquad h_{tj}=\frac{\|g_t\|_2}{\Delta_Y}.
```

实现对零幅度加了数值保护。每行共享同一对角系数，构造成本为
$`O(TD)`$。它对 Q/K 路线改变收费，V 保留一阶信用。
如果长度为 $`a\le\Delta_Y`$ 的编辑沿负梯度方向，那么模型成本满足：

```math
-a\|g_t\|_2+\frac{a^2\|g_t\|_2}{2\Delta_Y}
\le-\tfrac12a\|g_t\|_2\lt0.
```

这个选择避免把跨越两个对径 payload 的有用跳跃惩罚成零，也包含学习后的
空输出尺度。保证针对局部模型，不是任意真实任务 loss 的下降定理。

联合块给单点零信号提供另一条路。单改 i、单改 j、一起改的成本分别为
$`C_i,C_j,C_{ij}`$，令 $`\kappa=C_{ij}-C_i-C_j`$。固定本次角点后，
两个翻转概率的独立块目标是：

```math
F=\alpha_iC_i+\alpha_jC_j+\alpha_i\alpha_j\kappa,
\qquad
\partial_{z_i}F=\alpha_i'(C_i+\alpha_j\kappa).
```

单点成本由完整信用引擎给出，联合成本只需一次内部匹配。选中坐标的梯度
在累计到共享投影参数之前替换。默认 J=1；没有声称求全跨块交互。

![单点无信号、联合有收益的实际 ROSA 输入](../images/diameter_barrier.png)

二阶模式的度量和活动行选择依赖 g，因此一般不对不同 loss 的梯度可加。
把各项 loss 加起来再调用一次 backward 是支持的用法；分别 backward 后累积，
不保证等于前者。同一 loss 的正比例缩放受支持。硬信用反向显式不支持
二次 autograd，不提供看似可用但不完整的高阶导数。

## C++ 接口与数组布局

公开头文件是 [`csrc/training.h`](../conv_rosa_transformer/csrc/training.h)。
先调用 `rosa_training_defaults(&options)` 初始化带版本号和结构大小的配置。
默认是二阶代理。C 调用者选择原一阶时，要同步设置三个相关字段：

```cpp
RosaTrainingOptions options;
rosa_training_defaults(&options);
options.credit_order = 1;
options.joint_pairs = 0;
options.trust_diagonal = 0.0;
options.randomized = 1;  // 独立的精确搜索选项
```

Python 构造函数会根据 `credit_order` 自动完成这几项设置。

| 数组 | 布局 | 类型 |
|---|---|---|
| Q/K/V logits、输出、上游与输入梯度 | `[B,G,T,D]`，连续行主序 | `double` |
| 空输出、payload 幅度及其梯度 | `[G,D]` | `double` |
| 打包后的 Q/K | `[B,G,T]` | `uint64_t` |
| V bit | `[B,G,T,D]` | `uint8_t` |
| 匹配长度、终点 | `[B,G,T]` | `int32_t` |

`rosa_training_forward` 一次处理整个 batch：打包 Q/K/V、执行已完成的精确
前向扫描、选择 payload，并返回反向需要的硬状态。
`rosa_training_backward` 一次处理全部流：构造信任项、生成联合提案、运行
信用核心、重放选中角点、映射 Q/K/V logits，并累计两个 embedding 的梯度。
内存由调用者分配，输入在调用期间必须有效。详细状态码与可选诊断输出在头文件中。

可独立编译的例子：

```bash
c++ -O3 -std=c++17 -ffp-contract=off examples/native_training.cpp \
    conv_rosa_transformer/csrc/unified_core.cpp -o /tmp/rosa-native
/tmp/rosa-native
```

所有正式计算仍走这一套核心。测试目录里的旧 Python 算术只用于迁移对照，
不参与运行时选择。

## 数值与模型兼容性

原来的 Q/K/V/O 参数名、形状和层别名保留。权重可以严格加载，但训练配置
也应随模型保存。前向是硬值；`.eval()` 下两种信用模式的输出相同。

原生信用与映射使用 binary64 中间值。相较先前 Python/Torch 中部分 float32
运算，归约顺序和舍入可能不同；迁移比较使用明确容差，不把它宣传成旧版
所有 float32 训练轨迹都逐位相同。当前原生实现的随机化开关则不改变算术
顺序，应得到逐位相同的结果。

活动行数超过 `joint_rows` 时，原生加权行采样器使用 `mt19937_64`，冻结的
Python 实现使用 NumPy `PCG64`。采样规则保留，但同一个整数 seed 在两种
实现间未必选中同一批行。原生实现内部切换 `randomized` 则保留联合提案流
和框架 RNG 状态。

框架适配器保留 GradScaler 溢出跳步的正常行为，并支持普通优化器、有限
loss 缩放和 checkpoint 使用。C++ 核心是 CPU 实现；CUDA 输入仍有 CPU
传输和同步，没有声称提供了 GPU 匹配内核。

本轮的准确性、耗时、相同随机数训练与负面结果见
[结果记录](native-results.zh-CN.md)。选择规则应看单位时间的任务结果，
不能把信用更复杂、CPU 原生化或局部搜索更快直接等同于模型训练更好。
