# 算法参考与实现索引

[English](algorithm-reference.md) · [完整推导](../ALGORITHM.zh-CN.md) · [形式证明](subquadratic-proof.md) · [删除证明](subquadratic-deletion-proof.md)

仓库唯一的反向引擎是 [ALGORITHM.zh-CN.md](../ALGORITHM.zh-CN.md) 中的次二次构造。本页记录接口约定，把关键不变量对应到源码、证明和独立测试，不再另讲一套实现。

## 数学约定

输出行 $t$ 的合法 K 终点满足 $r<t$。路由按 `(匹配长度, 终点)` 字典序取最大，所有零长度路由统一成 `(0,-1)`。非空路由输出 $P_{r+1}=(2V_{r+1}-1)\odot e_1$，空路由输出 $e_0$。

一个 Q/K/V 单 bit 修改 $i$ 的一阶目标是 $C_i=\sum_t g_t^\top(Y_t^{\operatorname{flip}(i)}-Y_t)$。Q/K 的二阶模式使用固定局部分数 $g_t^\top\delta+\tfrac12\sum_c h_{t,c}\delta_c^2$；原生训练约定中 V 仍保持线性，嵌入梯度沿原路由累加。选中的联合 bit 修正是额外的有限预算阶段，见[训练接口](native-training.zh-CN.md)。

禁止匹配经过某个 owner 不会移动位置。翻转后的匹配要么避开它，要么修复它处的唯一单 bit 失配，所以 `Flip = max(Delete, Repair)`。删除减原输出由一个位置的所有 bit 共用；只有修复胜出时，才加入修复减删除。

## 源码与证明

| 部分 | 不变量 | 实现与证据 |
|---|---|---|
| 前向与联合回放 | 因果 SAM 轨迹；最多 16 次平坦表探测，再做完整键有序查找 | [forward.h](../conv_rosa_transformer/csrc/forward.h)、[前向证明](forward.zh-CN.md) |
| 惰性后缀索引 | 复用已保存的原路由；完整符号比较、有界同符号段证书与精确 LCP/出现查询 | [index.h](../conv_rosa_transformer/csrc/subquadratic/index.h) |
| 前缀赢家 | 合法反向后缀的字典序邻居给出最大 LCP，出现查询解决最晚终点 | [index.h](../conv_rosa_transformer/csrc/subquadratic/index.h) |
| 单符号参考项与完整残差 | `D + R₁ + Σ(R−R₁)`；先用精确证书排除零残差，再对完整候选集合计数、求值 | [approx_onegram.h](../conv_rosa_transformer/csrc/approx_onegram.h)、[approx_row_certificate.h](../conv_rosa_transformer/csrc/approx_row_certificate.h)、[approx_indexed_row.h](../conv_rosa_transformer/csrc/approx_indexed_row.h)、完整证明第 2 节 |
| 共享删除 | 短窗口出现构成 AP；完整 LCS 是带上限仿射式，至多一个边界例外 | [delete.h](../conv_rosa_transformer/csrc/subquadratic/delete.h)、[删除证明](subquadratic-deletion-proof.md) |
| Q 删除与负项请求 | 同一平台流更新完整删除差分与 Fenwick 时间扫描，不重复投影整行 | [delete.h](../conv_rosa_transformer/csrc/subquadratic/delete.h)、完整证明第 5 节 |
| K 删除数值收缩 | 同周期所有余数共用块扫描；整数 owner 桶为每行只保留一个待处理事件 | [delete.h](../conv_rosa_transformer/csrc/subquadratic/delete.h) |
| 有界完整 Q 前沿 | 修改游标不能复活；活跃删除屏障恰有 `L[t]` 个；活跃数和总探测数同时约束成功、失败尝试 | [q_frontier.h](../conv_rosa_transformer/csrc/subquadratic/q_frontier.h)、完整证明第 5 节 |
| 压缩 Q 修复 | 每行静态来源集合互不相交；分支最多转轻一次；小来源包络归并两列有序寿命 | [repair.h](../conv_rosa_transformer/csrc/subquadratic/repair.h)、完整证明第 5 节 |
| 完整 K 挑战候选 | 两个精确半段找全长候选，原匹配受损窗口找全短赢家；owner 桶复用报告求删除 | [near_k.h](../conv_rosa_transformer/csrc/subquadratic/near_k.h)、完整证明第 6 节 |
| 核验整段指纹的 K 报告 | 全局短 baseline 限定所有因果 LCS；指纹报告和出现前驱均逐一核验，共用总工作预算 | [fingerprint_k.h](../conv_rosa_transformer/csrc/subquadratic/fingerprint_k.h)、完整证明第 6 节 |
| 短 K 修复 | 虚拟单词不复制子串；压缩 trie 区间保留最长、最晚优先级 | [short_k.h](../conv_rosa_transformer/csrc/subquadratic/short_k.h)、完整证明第 6 节 |
| 长 K 修复 | 每个长单失配匹配含有网格锚点；胜过删除的生命期是后缀 | [repair.h](../conv_rosa_transformer/csrc/subquadratic/repair.h)、完整证明第 7–9 节 |
| 数值区间 | 仿射终点对应平移相关，完整块通过流式前缀快照求值 | [correlation.h](../conv_rosa_transformer/csrc/subquadratic/correlation.h) |
| 二次行分数 | 非空带符号载荷逐坐标平方固定；额外加入空载荷指示通道 | [engine.h](../conv_rosa_transformer/csrc/subquadratic/engine.h) |
| 原生训练 | 冻结局部对角和有限联合角点；V 使用原线性散射 | [training_impl.h](../conv_rosa_transformer/csrc/training_impl.h) |

数值数组均为行主序。owner/时间区间两端包含，后缀排名区间右端不包含。指定行集的删除接口接受原始时间下标，可以为空、不连续或乱序，但不能重复；因果关系不重新编号。完整 owner/时间区间投影包含不受修改影响的原输出行，不是删除减原输出的差值。

## 参数与成本

$b$ 表示打包 bit 数，$d$ 表示数值维数。算法使用末尾长度 $A$、Q 来源阈值 $\kappa$、长 K 锚点宽度 $B$（短上限 $L_{\mathrm{cut}}=4B$）、相关块宽 $W$，以及删除时间块宽 $H$。游标上限为 $C=2\max(32,2b+2\lceil\log_2(n+1)\rceil)$，候选报告预算以 $W_n=n\max(1,\lfloor n^{3/4}\rfloor)$ 为基准。

| 阶段 | 省略对数后的时间 |
|---|---:|
| 原路由与惰性精确索引 | $n+nd$ |
| 单符号参考项 | $nb(d+1)$ |
| 接受的完整终点报告 | $(d+1)M+nb$，其中 $M=O(W_n)$ |
| 整段指纹 K 尝试，$C_K=O(b+\log n)$ | $nC_K\log n+n(b+d)+W_n(d+C_K+\log n)$ |
| 共享删除，$A\asymp\sqrt n$ | $(d+1)n^{3/2}$ |
| 有界完整 Q 游标尝试 | $nC(d+\log n)+nb$ |
| 压缩 Q 轻重与小集合，$\kappa\asymp(n/b)^{1/3}$ | $(d+1)n^{5/3}b^{1/3}+bn^{3/2}$ |
| 共用 Q 删除及 $M_Q\le nb$ 个负项请求 | $(d+1)n^{3/2}+M_Q$；直接短请求另计 $M_0d$ |
| 短 K | $nb(d+1)B$ |
| 长 K 正项 | $d n^2/B^{1/2}$ |
| 长 K 删除投影 | $(d+1)n^2/B^{1/3}$ |
| 有限联合预算回放 | $Jn(d+1)$ |

表中包含同一项的不同求值方式。完整残差报告要求 $M\le W_n$；双半段 K 报告要求 $M\le\min(4W_n,8nbL_{\mathrm{cut}})$，两者都先计数完整集合，再展开报告。完整 Q 前沿只在成功后提交，此时删除和修复都已算完；内部 `include_deletion=false` 选项只返回修复减删除。失败后，平台扫描和压缩修复继续完成同一公式。共用 Q 删除与负项请求是一次扫描，不重复付费。至多四行的短请求可直接求值，$M_0$ 单独记录这部分数量，保留它的维数成本。

固定 $b,d,J$ 并取 $B\asymp n^{3/4}$，时间为 $\widetilde O(n^{7/4})$。记录空间为 $\widetilde O(n+nd+nb+nb\kappa+n^2/B)$，固定维度时为 $\widetilde O(n^{4/3})$。固定 32 行的残差历史、owner 桶和惰性有界缓存，在固定宽度下只增加 $O(n)$ 字。这里给的是完整算法的界，不是实测加速比。波浪号不隐藏 $b,d,J$ 或标量精度；特别地，$b=d=D$ 时短匹配核带有 $D^2$ 系数。

原生 workspace 计数器是被跟踪存储的 **buffer capacity estimate（缓冲区容量估算）**，不是进程 RSS，也不保证等于整个分配器或运行时的峰值。这个测量口径与渐近空间证明分开。

## 数值与随机化约定

路由决策精确。完整模式包含有限信用公式的所有项，但分组 binary64 求和和 FFT 不保证与历史标量累加顺序逐 bit 相同，强抵消可能放大相对误差。精确随机缓存比较完整查询键，保持本引擎的数值运算顺序；相同输入和其余种子下，开关缓存的结果逐 bit 相同。机制与自适应绕过见[精确随机化](exact-randomized.zh-CN.md)。

显式近似模式采样完整残差行或数值通道，绝不在路由取最大值之前采样候选。条件期望是实数算术下的代数结论，不是训练保证或统一相对误差界。[随机反向教程](randomized-backward.zh-CN.md)给出估计器及其控制项。

## 独立检查

| 测试 | 独立预期答案 |
|---|---|
| [subquadratic_delete.cpp](../tests/cpp/subquadratic_delete.cpp) | 朴素后缀扫描、AP 覆盖与完整长度、Q/K 删除、任意时间区间、指定行边界、缓存一致性 |
| [subquadratic_repair.cpp](../tests/cpp/subquadratic_repair.cpp) | 真正修改字符串后的 Q 修复与长 K 覆盖 |
| [subquadratic_q_frontier.cpp](../tests/cpp/subquadratic_q_frontier.cpp) | 朴素完整 Q 与仅修复信用、永久死亡、活跃数/探测数拒绝及失败不提交 |
| [subquadratic_short.cpp](../tests/cpp/subquadratic_short.cpp) | 虚拟 LCP、trie 与因果字典、带权偏序、短修复、完整 K 见证与共用删除 |
| [subquadratic_indexed_row.cpp](../tests/cpp/subquadratic_indexed_row.cpp) | 朴素完整与单符号的修改/删除路由、精确残差抵消、一阶与对角分数、带权行及已有输出累加 |
| [subquadratic_fingerprint_k.cpp](../tests/cpp/subquadratic_fingerprint_k.cpp) | 朴素完整 K 信用、强制指纹碰撞、精确前驱、放置种子与失败不提交 |
| [subquadratic_correlation.cpp](../tests/cpp/subquadratic_correlation.cpp) | 任意平移区间的直接求和 |
| [test_subquadratic_engine.py](../tests/test_subquadratic_engine.py) | 完整原生信用、一二阶、随机化、近似及训练集成 |

边界覆盖包括 bit 63、`UINT64_MAX`、空路由、独立数值维数、等长平局和末尾残块。小规模独立 oracle 可以枚举二次方表，生产引擎不使用这些表。运行方式见[测试说明](../tests/README.md)，当前耗时与误差见[实测](subquadratic-results.zh-CN.md)。
