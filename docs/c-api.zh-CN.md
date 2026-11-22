# 独立 C/C++ API 与 ABI 参考

[English](c-api.md) · [返回首页](../README.zh-CN.md) · [算法原理](algorithm.zh-CN.md) · [训练指南](training.zh-CN.md)

底层核心提供零外部依赖的 C ABI（[`qkv_rosa/csrc/training.h`](../qkv_rosa/csrc/training.h)），无需 Python、NumPy、PyTorch 头文件或 CUDA 即可直接编入任意 C/C++17 项目。

---

## 1. 编译与运行最小示例

仓库中的 [`examples/native_training.cpp`](../examples/native_training.cpp) 演示了完整的批处理前向与反向调用：

```bash
c++ -O3 -std=c++17 -ffp-contract=off \
    examples/native_training.cpp qkv_rosa/csrc/unified_core.cpp \
    -o build/rosa-native

./build/rosa-native                          # 默认：二阶精确模式
./build/rosa-native --first --deterministic  # 一阶精确模式，关闭查询缓存
./build/rosa-native --approximate            # 显式近似模式
```

> 建议始终带上 `-ffp-contract=off`（MSVC 用 `/fp:strict`），避免编译器 FMA 融合改变浮点舍入一致性。

---

## 2. 公开 C 函数一览

所有声明位于 [`qkv_rosa/csrc/training.h`](../qkv_rosa/csrc/training.h)，实现位于 [`qkv_rosa/csrc/training_impl.h`](../qkv_rosa/csrc/training_impl.h) 和 [`qkv_rosa/csrc/unified_core.cpp`](../qkv_rosa/csrc/unified_core.cpp)：

| 函数 | 用途 |
|---|---|
| `rosa_training_defaults` | 用 ABI v1 默认值初始化 `RosaTrainingOptions`（二阶精确模式） |
| `rosa_approximation_defaults` | 用 ABI v1 默认值初始化 `RosaApproximationOptions` |
| `rosa_training_forward` | 打包硬 `Q/K/V`，执行因果前向路由，返回输出 `output` 与路由轨迹 `lengths`/`routes` |
| `rosa_training_backward` | 完整精确反事实反向传播（含信任模型、联合双 bit 回放与 logit/嵌入梯度） |
| `rosa_training_backward_approx` | 同一引擎的显式近似入口（残差行采样或数值通道 PPS 采样），需传入前向保存的 `saved_lengths`/`saved_routes` |
| `rosa_training_backward_subquadratic` | 统一诊断入口：同时支持精确（`approximation == NULL`）与近似模式，可选复用前向保存路由及输出 22 维结构计数器 |

---

## 3. 配置结构体（ABI v1）

**务必先调用默认初始化函数**，不要手填 `abi_version` 和 `struct_size`。

### `RosaTrainingOptions`

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `credit_order` | `2` | `1` = 一阶线性信用；`2` = 对角信任 + 联合双 bit |
| `randomized` | `1` | `1` = 开启完整键核验查询缓存；`0` = 关闭缓存（不改变数值结果） |
| `logit_mode` | `0` | `0` = Bernoulli sigmoid 导数映射；`1` = 定向恒等映射 |
| `joint_pairs` | `1` | 每条流最多选择的互不相交双 bit 块数量 |
| `joint_rows` | `32` | 联合提案检查的最大查询行数 |
| `active_rows` | `1` | `1` = 仅在非零上游梯度行提案；`0` = 均匀采样行 |
| `tau` | `1.0` | 正的 sigmoid 温度 |
| `trust_diagonal` | `0.5` | 局部对角信任尺度（0.5 对应 $\|g_t\|_2 / \Delta_Y$） |
| `proposal_seed` | `0` | 联合提案随机种子 |
| `repair_seed` | `0` | 查询缓存布局与近似采样独立盐值 |
| `repair_budget` / `reserved` | `0` | 保留字段，**必须保持为 0** |

在 C 中切换到一阶模式时，必须同步清零两个二阶参数，否则返回错误码 `1`：

```cpp
RosaTrainingOptions options;
rosa_training_defaults(&options);
options.credit_order = 1;
options.joint_pairs = 0;
options.trust_diagonal = 0.0;
```

### `RosaApproximationOptions`

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `sample_rows` | `32` | 每条流采样的残差行预算（正整数） |
| `importance` | `1` | `1` = 梯度范数重要性采样（含 10% 均匀底座）；`0` = 均匀时间分层 |
| `sample_channels` | `0` | `0` = 残差行估计器（保留全通道）；`1..64` = 固定大小通道 PPS 采样（忽略 `sample_rows`） |
| `seed` | `0` | 独立于联合提案的近似采样种子 |

---

## 4. 数组布局与边界约束

所有数组均须**行主序连续存储**，浮点数组采用 IEEE `double`（`float64`）：

| 数组 | 形状 | 类型 |
|---|---|---|
| `q_logits`, `k_logits`, `v_logits`, `output`, `grad_y`, `grad_q`, `grad_k`, `grad_v` | `[B, G, T, D]` | `double` |
| `emb0`, `emb1`, `grad_emb0`, `grad_emb1` | `[G, D]` | `double` |
| 打包后的 `q`, `k` | `[B, G, T]` | `uint64_t` |
| `v_bits` | `[B, G, T, D]` | `uint8_t`（仅 0 或 1） |
| `lengths`, `routes` | `[B, G, T]` | `int32_t` |
| 前向 / 训练 / 近似 / 结构计数器 | `[B*G, K]` | `uint64_t`（`K` 见头文件中宏常量：4 / 35 / 40 / 22） |

**合法形状范围**：`B >= 0`，`G > 0`，`0 <= T < 2^28`，`1 <= D <= 64`，`T * D <= INT32_MAX`，`B * G <= INT32_MAX`。支持空序列（`T == 0`）与空 batch（`B == 0`），bit 63 为普通合法符号位。

**内存所有权**：调用方负责分配全部输入输出缓冲区，并在调用期间保持有效；输出缓冲区不得与输入重叠。`grad_emb0` 与 `grad_emb1` 会在每个组 `g` 内跨 batch 累加。

**可选联合追踪缓冲区**：`pair_codes`（`[B*G, joint_pairs, 2]`）和 `pair_costs`（`[B*G, joint_pairs, 4]`）可传 `NULL`。编码 `< T*D` 表示 K bit，编码 `>= T*D` 减去 `T*D` 后表示 Q bit（商为位置、余数为 bit 下标）；4 个代价依次为 `C_i, C_j, C_ij, kappa`。

---

## 5. 返回状态码

| 状态码 | 含义 |
|---:|---|
| `0` | 成功 |
| `1` | 非法形状、选项冲突、ABI 版本不匹配或空指针 |
| `2` | 内存分配失败（`std::bad_alloc`） |
| `3` | 内部数值或执行异常 |
| `4` | 必需浮点输入中存在 `NaN` 或 `Inf` |

各诊断计数器的完整字段名定义见 [`qkv_rosa/native.py`](../qkv_rosa/native.py) 中的 `TRAINING_COUNTER_NAMES`、`APPROXIMATION_COUNTER_NAMES` 和 [`qkv_rosa/core.py`](../qkv_rosa/core.py) 中的 `COUNTER_NAMES`。
