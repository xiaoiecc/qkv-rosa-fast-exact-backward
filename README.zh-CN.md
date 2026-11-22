# QKV-ROSA 快速精确反向传播引擎

[English](README.md) · [算法原理](docs/algorithm.zh-CN.md) · [训练指南](docs/training.zh-CN.md) · [C/C++ 接口](docs/c-api.zh-CN.md) · [性能实测](docs/benchmarks.zh-CN.md)

面向硬路由 $n$-bit **QKV-ROSA**（Rapid Online Suffix Automaton）的高性能精确反事实反向传播引擎。底层采用零依赖 **C++17** 实现，提供开箱即用的 **PyTorch**、**NumPy** 与独立 **C/C++** 接口。

## 这个项目解决什么问题？

在硬路由 QKV-ROSA 中，每个时间步 $t$ 都会在严格早于自己的历史键（$r < t$）中，寻找与当前 Query 结尾**匹配长度最长**的后缀（长度相同时取最晚的位置 $r$），然后取出紧跟在 $r$ 后面的值 $V_{r+1}$ 作为输出；如果找不到任何匹配，则输出可学习的空嵌入 $e_0$。

这种硬匹配机制完全离散、不可导，普通的自动微分无法给 `Q` 和 `K` 计算梯度。为了让它可训练，我们需要计算**精确的单 bit 反事实信用（counterfactual credit）**：

> *假设把 $Q$、$K$ 或 $V$ 中的某一个 bit 单独翻过来，再按原规则重跑一次精确硬路由，模型输出会变化多少？把输出变化投影到上游梯度 $g = \partial L / \partial Y$ 上，就得到了该 bit 的反事实信用。*

最朴素的做法是逐个翻转每个 bit 再重跑前向，对于长度 $T$、位宽 $b$ 的序列，时间复杂度高达 $O(b\,T^4)$。本仓库采用**单次因果区间扫描（Causal Interval Sweep）**算法解决这一瓶颈：

- **路由 100% 精确，覆盖全部单 bit 修改**：严格遵守因果边界与「最长优先、同长取最新」规则，统一支持 $1 \le b \le 64$ 任意打包位宽。
- **随机输入下近线性期望复杂度**：先压缩不参与任何匹配的无效时间间隔，再按符号类别聚合单符号贡献，长匹配通过打包后缀索引精确查询——在独立均匀随机输入下期望时间仅 **$\widetilde{O}(T)$**（固定位宽与联合预算下一般最坏时间为 **$\widetilde{O}(T^2)$**）。
- **像普通 PyTorch 层一样训练**：无缝接入标准的 `model(x)` → `loss.backward()` → `optimizer.step()` 流程，支持一阶线性信用、二阶局部对角信任模型 + 联合双 bit 协作，以及可选的显式近似模式。
- **零外部依赖的 C++17 内核**：首次调用时按源码哈希自动编译缓存，不需要 PyTorch C++ 头文件或 CUDA 工具链，也可以直接编进独立 C/C++ 程序。

## 安装

需要 Python 3.10+、C++17 编译器（`g++`、`clang++` 或 MSVC `cl`）、NumPy，以及 PyTorch（使用 `nn.Module` 层时）。

```bash
git clone https://github.com/xiaoiecc/qkv-rosa-fast-exact-backward.git
cd qkv-rosa-fast-exact-backward
python -m pip install -e .
```

首次导入并执行时会自动编译 C++17 动态库并按源码哈希缓存。

## 快速开始

### 在 PyTorch 中使用

```python
import torch
from qkv_rosa import NBitQKVRosa

layer = NBitQKVRosa(channels=128, n_bits=8)
optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)

x = torch.randn(2, 32, 128)
target = torch.randn_like(x)

optimizer.zero_grad(set_to_none=True)
loss = (layer(x) - target).square().mean()
loss.backward()
optimizer.step()
```

### 核心配置项

| 参数 | 默认值 | 作用说明 |
|---|---|---|
| `credit_order` | `"second"` | `"first"` 为纯一阶线性反事实信用；`"second"` 在 Q/K 上加入基于输出直径的局部对角信任惩罚，以及有限预算的联合双 bit 协作修正。 |
| `backward_mode` | `"exact"` | `"exact"` 完整计算所有单 bit 信用；`"approximate"` 显式启用残差行分层采样或数值通道采样（通过 `approximation_rows` / `approximation_channels` 控制预算）。 |
| `randomized` | `True` | 在后缀索引中开启带完整键核验的查询缓存；只影响重复查询开销，绝不改变路由或数值结果。 |

```python
# 纯一阶精确反事实信用
first_order = NBitQKVRosa(128, 8, credit_order="first")

# 显式开启近似反向（每条流采样 32 个残差行）
approx = NBitQKVRosa(
    128, 8,
    credit_order="second",
    backward_mode="approximate",
    approximation_rows=32,
)
```

### 独立 C++17 调用（无需 Python 或 PyTorch）

```bash
c++ -O3 -std=c++17 -ffp-contract=off \
    examples/native_training.cpp qkv_rosa/csrc/unified_core.cpp \
    -o build/rosa-native
./build/rosa-native
```

## 使用边界说明

- **CPU 实现**：核心匹配与反向引擎运行在 CPU 上。PyTorch 层接受 CUDA 张量，但内部会自动完成主机-设备拷贝，不是 GPU 内核。
- **替代梯度特性**：默认的 `"second"` 对角信任项是显式指定的局部二次惩罚模型，并非任意下游网络的真实 Hessian。由于信任权重和联合 bit 提案依赖于上游梯度 `grad_y`，若同时有多个损失项，请先求和再做一次 `.backward()`。不支持二阶自动微分（double autograd）。
- **数值精度约定**：所有符号比较、因果边界和最长/最新平局判断均严格精确。数值归约采用 IEEE `float64`，由于区间聚合改变了加法结合顺序，结果与逐项朴素求和之间存在正常浮点舍入差异（相对 L2 误差约 $10^{-14}$）。

## 项目结构

```text
qkv_rosa/               核心 Python 包与 C++17 引擎
├── layer.py            PyTorch NBitQKVRosa 层与 autograd 桥接
├── transformer.py      Conv ∥ ROSA Transformer 模块（ConvRosaBlock, RosaTransformer）
├── native.py           批处理 NumPy 原生绑定（对接 C ABI）
├── core.py             自动编译缓存与单流低级接口
├── reference.py        独立朴素参考实现（O(T^3) 前向与 O(b·T^4) 反向，用于对拍验证）
└── csrc/               零依赖 C++17 核心源码
    ├── training.h      版本化 C ABI 头文件
    ├── forward.h       精确因果后缀自动机（SAM）前向
    └── engine/         单次因果区间扫描引擎（edit_alphabet.h, interval_credit.h, index.h, delete.h）
docs/                   中英双语核心文档（算法、训练、C API、实测）
examples/               快速上手脚本、独立 C++17 示例与 10 位乘法训练示例
tests/                  Python 单元测试与独立 C++ 物理翻转对拍测试
benchmarks/             随机分布、完整训练步与坐标压缩率评测脚本
```

## 文档导航

| 主题 | 简体中文 | English |
|---|---|---|
| 核心算法与复杂度分析 | [docs/algorithm.zh-CN.md](docs/algorithm.zh-CN.md) | [docs/algorithm.md](docs/algorithm.md) |
| 训练规则与 PyTorch 集成 | [docs/training.zh-CN.md](docs/training.zh-CN.md) | [docs/training.md](docs/training.md) |
| 独立 C/C++ 接口参考 | [docs/c-api.zh-CN.md](docs/c-api.zh-CN.md) | [docs/c-api.md](docs/c-api.md) |
| 性能与精度实测 | [docs/benchmarks.zh-CN.md](docs/benchmarks.zh-CN.md) | [docs/benchmarks.md](docs/benchmarks.md) |
| 测试与基准脚本说明 | [tests/README.md](tests/README.md) · [benchmarks/README.md](benchmarks/README.md) | — |

## 运行测试

```bash
python tests/run.py               # 快速 Python 单元测试
python tests/run.py --exhaustive  # 完整验证（含独立 C++ 物理翻转对拍）
```

## 致谢与许可

QKV-ROSA 的前向检索架构源自 **Bo Peng (BlinkDL)** 的 [RWKV-LM RWKV-v8 示例](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8)（见 `251024_rosaQKV_run.py` 中的 `ROSA_QKV_B_1bit`）。本仓库为独立实现，将其扩展至 $n$-bit 打包符号与可学习空嵌入，并提供了高效的精确反事实反向传播算法。

Copyright 2026 xiaoiecc. 基于 [MIT License](LICENSE) 开源。
