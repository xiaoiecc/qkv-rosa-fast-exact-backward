# 近似反向：把有用的完整控制项留下，只估计余量

[English](randomized-backward.md) · [完整算法](../ALGORITHM.zh-CN.md) · [精确缓存](exact-randomized.zh-CN.md) · [训练接口](native-training.zh-CN.md) · [当前实测](subquadratic-results.zh-CN.md)

完整次二次引擎的路由精确，包含全部数学信用项；完整键缓存只省掉重复查询。`backward_mode="approximate"` 是另一个显式选择：减少实际求值的数值贡献，同时仍以同一个局部信用为目标。它不改变硬前向，也不先抽几个匹配候选再决定路由。

## 1. 通道预算选择两种估计器

| 近似模式中的设置 | 执行内容 |
|---|---|
| `approximation_channels=0` | 完整共享删除与一符号修复控制项，再分层采样完整的剩余行修正 |
| `approximation_channels>0` | 对中心化数值通道做固定大小带权采样，完整运行结构算法，空载荷控制保持完整 |

默认每个 batch/group 流采 32 个残差行。零通道预算表示行估计器保留**全部**数值通道，不是零工作，也不是完整模式。正通道预算的分支不使用 `approximation_rows`；两者不会默认叠加。一阶、二阶信用都支持这两个分支。

## 2. 先固定目标，再决定抽什么

记原硬输出为 $Y_t$，固定当前上游梯度和局部对角：

```math
\phi_t(y)=g_t^\top(y-Y_t)+\frac12\sum_j h_{t,j}(y_j-Y_{t,j})^2.
```

一阶取 $h=0$。Q/K bit 坐标 $i$ 的目标是 $C_i=\sum_t\phi_t(Y_t^{\operatorname{flip}(i)})$。这个对角是[训练教程](native-training.zh-CN.md)指定的局部模型，不是用一次梯度恢复任意下游 Hessian。

路由先比长度、再比终点。假设两个候选长度为 8 和 2，抽一个再取最大值，已经不再是完整集合的最大值，取回的载荷也可能完全不同。乘逆概率通常救不回这种非线性选择。我们采样的是最大值确定后的**完整行贡献**，或结构路由确定后的线性数值通道。

## 3. 完整删除加一符号修复，可以先算得很扎实

禁止匹配经过 bit $i$ 的 owner，但不移动位置，得到删除路由。令

```math
D_{t,i}=\phi_t(Y_t^{\operatorname{delete}(i)}),\qquad
R_{t,i}=\phi_t(Y_t^{\operatorname{flip}(i)})-D_{t,i}.
```

于是 $C_i=\sum_t(D_{t,i}+R_{t,i})$。同一 owner 的所有 bit 共用删除，所以保留完整 AP 删除核很划算。

一符号路由只寻找当前 Q 符号的最晚合法 K 出现，忽略更早上下文。记它的修复减删除为 $R_{1,t}$，分数仍围绕**真实完整基线** $Y_t$ 计算。按符号分类的出现列表可以便宜地算出这一项。取

```math
B=D_{\mathrm{full}}+\sum_t R_{1,t},\qquad
\rho_t=R_t-R_{1,t},\qquad C=B+\sum_t\rho_t.
```

行估计器完整保留 $B$，只采样 $\rho_t$。二阶时不能换中心：若把一符号分数改成围绕它自己的输出，会多出不同的交叉项，等式就不成立。实现见 [approx_onegram.h](../conv_rosa_transformer/csrc/approx_onegram.h) 与 [engine.h](../conv_rosa_transformer/csrc/subquadratic/engine.h)。

## 4. 抽样前先证明哪些残差为零

两类便宜证书可以缩小总体，不增加估计误差。

**没有近似相等的相邻对。** 至多翻转一个 bit 后，长度至少为二的匹配必然以某个合法双符号对结尾，满足

```math
\operatorname{popcount}(Q_{t-1}\oplus K_{r-1})+
\operatorname{popcount}(Q_t\oplus K_r)\le1.
```

没有这样的对，就说明所有相关匹配至多长一，完整与一符号路由相同，$\rho_t=0$。找到近对只是没有证明为零，不等于证明残差非零。证书索引比较完整键。

**没有合法修复出生。** 若存在 $s<u$ 使 $Q_u\oplus K_s$ 只有一个 bit，则标记 $b_u=1$。贡献到行 $t$ 的修复，其右边未改的尾巴长 $a\le L_t$，所以 Q 中心位于 $t-L_t\le u\le t$。这段窗口没有出生标记，就既无完整修复，也无一符号修复，残差为零。一符号分类计划已经提供标记和前缀和。

原生信任规则下，上游全零的行也没有贡献。剩下的行才进入抽样总体，见 [approx_row_certificate.h](../conv_rosa_transformer/csrc/approx_row_certificate.h)。

完整模式也使用这些证书。它先数清剩余行需要访问的同符号及单 bit 邻居终点；总工作不超过 $T\lfloor T^{3/4}\rfloor$ 时，直接求完残差。工作更多时才展开共享结构。所有剩余行都已证实为零且删除只需长度一时，即使设置了正通道预算，也直接返回便宜的完整控制项，不额外引入噪声。

## 5. 时间分层，概率放在分母里

把剩余 $N$ 行按时间排好，分成 $m$ 个大小尽量相等的连续层，每层 $S$ 抽一行。默认重要性概率为

```math
p_t=\frac{0.1}{|S|}+0.9\frac{\|g_t\|_2}{\sum_{u\in S}\|g_u\|_2},
\qquad \widehat C=B+\sum_S\frac{\rho_{T_S}}{p_{T_S}}.
```

范数总和为零时均匀采样，`importance=0` 则始终均匀。10% 的均匀底座保留全部支持，使每层逆权重至多为 $10|S|$。

固定硬输入、嵌入、上游梯度、局部度量、证书和联合提案，在实数算术中有

```math
\mathbb E[\rho_{T_S}/p_{T_S}]=\sum_{t\in S}\rho_t,
\qquad \mathbb E[\widehat C]=C.
```

总平方误差期望为

```math
\mathbb E\|\widehat C-C\|_2^2
=\sum_S\left(\sum_{t\in S}\frac{\|\rho_t\|_2^2}{p_t}
-\left\|\sum_{t\in S}\rho_t\right\|_2^2\right).
```

这也解释了为什么控制项和证书比盲抽更有价值：花样本之前先缩小、甚至去掉残差。上游范数只是现成的残差大小代理，并不等于残差范数。预算覆盖全部剩余行，或者已超过实现的完整计算工作阈值时，会直接求完整信用。

选中的一行枚举其等符号类与单 bit 邻居类中的全部合法终点，复用全局精确后缀索引，把最大路由求完，再对完整修正乘权重。不会随机遗漏某个本来可能赢的候选。代码见 [approx_sampling.h](../conv_rosa_transformer/csrc/approx_sampling.h) 与 [approx_indexed_row.h](../conv_rosa_transformer/csrc/approx_indexed_row.h)。

## 6. 显式通道采样减少数值维度

路由区间已经很少时，较大的数值维度仍可能占主要成本。正通道预算针对这部分工作，同时保留完整结构构造。

取一个参考带符号载荷 $P_\ast$，使用中心化特征 $X_{s,j}=P_{s,j}-P_{\ast,j}$ 和行权重 $a_{t,j}=g_{t,j}-h_{t,j}Y_{t,j}$，再加空载荷指示通道表达剩余常数差。这与完整二阶计算使用同一个精确升维。系数恒零或可达载荷跨度恒零的通道可以精确删去，空指示通道始终保留。

对其余 $N$ 个通道，取便宜的幅度代理 $w_j=(\max_t|a_{t,j}|)(\max_{s\ge1}|X_{s,j}|)$。预算 $m<N$ 时，用注水得到

```math
\pi_j=\min\left(1,\frac{0.1m}{N}+\lambda w_j\right),
\qquad \sum_j\pi_j=m.
```

Pivotal rounding 恰好选 $m$ 个通道，并保持这些边际概率；随机排列先消除固定列相邻关系的影响。入选通道使用**原始**纳入概率的倒数 $1/\pi_j$，确定入选项的权重为一。若完整通道信用向量为 $Z_j$，则

```math
\widehat Z=\sum_{j\text{ 入选}}Z_j/\pi_j,
\qquad \mathbb E[\widehat Z]=\sum_jZ_j.
```

一个流的 Q/K 删除和修复都使用同一批特征。选择不影响路由；没选中某个数值通道，也不等于删掉对应的 bit 输出坐标。预算覆盖全部有效通道时，该分支就是完整计算。实现见 [engine.h 的 Features](../conv_rosa_transformer/csrc/subquadratic/engine.h) 与 [approx_channel_pivotal.h](../conv_rosa_transformer/csrc/approx_channel_pivotal.h)。

不同坐标的误差可能高度相关。条件无偏本身不给相对误差或收敛界，尤其精确信用因抵消而很小时。也不能因为样本结果小，就只替换那些结果为精确值：一个等概率取 0 或 2、均值为 1 的无偏估计，若只把 0 改成精确值 1，均值就成了 1.5。

## 7. 哪些保持完整，误差又来自哪里

硬前向保持完整，V 和嵌入梯度沿原路由计算。选中的联合块使用完整局部角点；必要时，通过内部前向回放恢复采样过的单点角点，不重跑外围网络或损失。

数值分块与 FFT 已经属于完整次二次引擎，它们的舍入误差和行、通道采样误差是两回事；FFT 并不只在近似模式中使用。数学信用项完整，不保证历史 binary64 累加顺序；固定预算条件无偏，也不保证 Adam 更新或训练轨迹不变。

Q 和 K 误差应分别报告，还要看映射后的 logit 梯度、共享参数及输入梯度、训练和验证损失、完整训练步耗时与内存。一个很大且正确的分量，可能掩盖另一个小分量的严重误差。工作预算不是精度容差；当前表现、慢例与抵消问题见[实测](subquadratic-results.zh-CN.md)。

## 8. 照常接入模型

```python
from conv_rosa_transformer.rosa_layer import NBitQKVRosa

full = NBitQKVRosa(128, 8, credit_order="second")
rows = NBitQKVRosa(
    128, 8, credit_order="second", backward_mode="approximate",
    approximation_rows=32, approximation_channels=0,
)
channels = NBitQKVRosa(
    128, 64, credit_order="first", backward_mode="approximate",
    approximation_channels=8,
)
```

三者都使用正常 `loss.backward()` 和同一个结构引擎。适配器随前向保存采样种子，与联合提案分开。`randomized=False` 只关闭精确查询缓存，已经显式选择的近似估计器仍会采样。配置和缓存状态约定见[训练教程](native-training.zh-CN.md)与 [C API](native-api.md)。
