# 训练指南：把硬路由 QKV-ROSA 当普通层来训

[English](training.md) · [返回首页](../README.zh-CN.md) · [算法原理](algorithm.zh-CN.md) · [C/C++ 接口](c-api.zh-CN.md)

`NBitQKVRosa` 底层只使用一套 C++17 引擎，整个 batch 的硬打包、因果前向、反事实信用、局部信任模型、联合双 bit 回放和 logit 梯度映射全部在原生代码中一次性完成；PyTorch 只负责线性投影（`q_proj`, `k_proj`, `v_proj`, `o_proj`）、张量布局转换和常规 `autograd`。

你的训练循环不需要任何定制优化器或特殊技巧：

```python
import torch
from qkv_rosa import NBitQKVRosa

layer = NBitQKVRosa(channels=128, n_bits=8)
optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)

optimizer.zero_grad(set_to_none=True)
loss = (layer(x) - target).square().mean()
loss.backward()
optimizer.step()
```

---

## 1. 三个正交的核心选项

| 参数 | 默认值 | 可选值与含义 |
|---|---|---|
| `credit_order` | `"second"` | `"first"`：纯一阶线性反事实信用。<br>`"second"`：在 Q/K 上加入局部对角信任惩罚 + 有限预算的联合双 bit 协作修正。 |
| `backward_mode` | `"exact"` | `"exact"`：计算全部反事实信用项，不做统计抽样。<br>`"approximate"`：显式启用残差行分层采样或数值通道采样。 |
| `randomized` | `True` | 控制后缀索引中的完整键查询缓存。只影响缓存命中开销，绝不改变路由、联合提案或数值结果。 |

```python
# 一阶精确模式
layer_1st = NBitQKVRosa(128, 8, credit_order="first")

# 近似模式：残差行分层采样（保留全部数值通道）
layer_rows = NBitQKVRosa(
    128, 8,
    credit_order="second",
    backward_mode="approximate",
    approximation_rows=32,
    approximation_channels=0,
)

# 近似模式：固定大小数值通道 PPS 采样（此时忽略 approximation_rows）
layer_ch = NBitQKVRosa(
    128, 64,
    credit_order="second",
    backward_mode="approximate",
    approximation_channels=8,
)
```

---

## 2. 一阶信用与 Logit 梯度映射

翻转单个 bit $i$ 后，令硬输出变化为 $\delta = Y^{\text{flip}(i)} - Y$，一阶信用为：

$$
C_i^{(1)} = \sum_t g_t^\top \delta_t
$$

设当前硬 bit 为 $b \in \{0, 1\}$（由 `logit > 0` 阈值化产生），则「从 0 变为 1」方向的定向信用为 $\Delta_i = (1 - 2b)\,C_i$。

引擎支持两种把定向信用映射回连续 logit $z$ 的规则（`credit_to_logit` 参数）：
- **`"bernoulli"`（默认）**：乘以平滑 Bernoulli 概率导数 $\frac{\partial \sigma(z/\tau)}{\partial z} = \frac{p(1-p)}{\tau}$，其中 $p = \sigma(z/\tau)$（默认温度 `tau=1.0`）。
- **`"identity"`**：直接返回定向信用 $\Delta_i$。

$V$ 的 bit 信用和空嵌入 $e_0$、载荷尺度 $e_1$ 的梯度沿原路由直接累加，耗时 $O(TD)$。

---

## 3. 二阶模式：局部对角信任 + 联合双 bit 协作

### 3.1 局部对角信任模型

在单次反向传播里，组件只拿到上游梯度 $g_t$，数学上不可能凭这一个向量反推出任意下游网络的真实 Hessian。因此，默认的 `"second"` 模式显式指定了一个**基于输出集合直径的局部对角二次模型**：

$$
C^{(2)}(\delta) = g^\top \delta + \frac{1}{2} \sum_{t, j} h_{t,j}\,\delta_{t,j}^2
$$

记输出集合的直径为 $\Delta_Y = \max\big(2\|e_1\|_2,\, \||e_0| + |e_1|\|_2\big)$。在默认 `trust_diagonal=0.5` 时，对角系数取：

$$
h_{t,j} = \frac{\|g_t\|_2}{\Delta_Y}
$$

直觉含义：
- 沿着负梯度方向、幅度不超过 $\Delta_Y$ 的有益路由跳变，二次惩罚项最多抵消一半收益，仍然保持清晰下降信号；
- 剧烈但收益不大的路由跳变会受到二次项抑制，有助于稳定离散路由更新。
- 该对角惩罚只作用于 $Q$ 和 $K$；$V$ 与嵌入参数始终保持一阶线性信用。

### 3.2 联合双 bit 协作（跨越单点障碍）

有时单独翻转 bit $i$ 或 bit $j$ 都没有改善，但**同时翻转两者**能命中一条高价值匹配。

设单点代价为 $C_i, C_j$，同时翻转后的代价为 $C_{ij}$，交互项 $\kappa = C_{ij} - C_i - C_j$。记两者的局部翻转概率为 $\alpha_i, \alpha_j$，则双 bit 局部期望代价及其对 $z_i$ 的偏导为：

$$
F = \alpha_i C_i + \alpha_j C_j + \alpha_i \alpha_j \kappa,
\qquad
\frac{\partial F}{\partial z_i} = \alpha_i' (C_i + \alpha_j \kappa)
$$

引擎通过 `pair_candidates.h` 在有限个活跃行（默认 `joint_rows=32`）中选出至多 `joint_pairs`（默认每条流 1 对）互不相交的候选双 bit 块，在组件内部直接重跑轻量级精确前向来计算 $C_{ij}$——**外围模型、后续层和损失函数都只运行一次**。

---

## 4. 显式近似模式（`backward_mode="approximate"`）

当序列很长或位宽很大、希望用可控方差换取更低开销时，可以显式开启近似反向。两个互斥分支由 `approximation_channels` 控制：

1. **残差行分层采样（`approximation_channels=0`）**：
   - 完整计算共享删除项与 1-gram 单符号修复控制项（$B = D + R_1$）；
   - 先用精确零残差证书筛掉确定没有长修复的行（不引入任何误差）；
   - 对剩余残差行按时间分层，每层抽 1 行，并在**确定完整路由最大值之后**乘逆概率权重（绝不在取 $\max$ 之前抽样候选，保证实数算术下的条件无偏性）。
2. **数值通道 PPS 采样（`approximation_channels > 0`）**：
   - 在中心化数值特征上按幅度代理做固定大小带权采样（Pivotal rounding），空载荷指示通道始终完整保留，随后在降维后的特征上运行完整结构引擎。

两种近似分支都**完整保留**：硬路由前向、$V$ 与嵌入梯度，以及选中联合块的精确角点重算。

> 注意：条件无偏是实数算术下的代数性质，不承诺在任意任务或强数值抵消场景下都有统一相对误差界。

---

## 5. 集成注意事项

- **多个损失合并**：因为二阶信任权重和活跃行提案依赖上游梯度 `grad_y`，对两个损失分别 `.backward()` 并累加，结果不一定等于 `((loss1 + loss2).backward())`。如需合并规则，请先相加损失再做一次反向。
- **不支持二阶自动微分**：硬反事实路径显式标记了 `@once_differentiable`，不支持 `create_graph=True`。
- **混合精度与溢出跳过**：PyTorch 适配器会原样透传非有限（`NaN`/`Inf`）上游梯度，因此标准的 `GradScaler` 溢出检测和跳过更新可正常工作。
- **保存权重请连带保存配置**：`state_dict()` 只保存投影权重与 `emb0`/`emb1`，训练超参（`n_bits`, `credit_order`, `tau` 等）需随 checkpoint 配置一并保存。
- **输入约束**：输入形状为 `[B, T, C]`，要求 $C > 0$、$1 \le \text{n\_bits} \le 64$ 且 $C \bmod \text{n\_bits} = 0$。每 `n_bits` 个连续通道组成一条独立的路由流（共 `groups = C // n_bits` 组）。
