# 数学规格与生产实现边界

## 1. 记号与分组

W:m×d，X:d×T，Q:d×d，Y=WX。W'=WQᵀ，X'=QX。默认权重每行、激活每个token沿d轴共享scale；group_size=-1表示完整归约维，正值必须整除d。API内部可以转成row-major存储，但必须记录列向量数学含义。

对称b位：qmax=2^(b−1)−1，qmin=−2^(b−1)，s=maxabs/qmax，C=clamp(round_even(z/s),qmin,qmax)，D=s*C。全零组用安全scale，不得NaN。反量化结果是float，不是integer code本身。

生产实现不能通过channel subsampling减小d；该操作会改变AbsMax owner与组内量化问题。输入侧可以抽weight输出行或token。输出侧Q B中完整hidden维必须保留，必要时先做完整旋转再抽样评价行，但该估计不再是未抽样全输出目标的保证。

## 2. 正交轨道与全精度等价

Q₀=HD；Q_new=G_ij(θ)Q，G在二维块为[[cosθ,−sinθ],[sinθ,cosθ]]。输入侧实际更新是A_new=A Gᵀ、B_new=G B，其中A=WQᵀ，B=QX。AB保持不变。导出SpinQuant时R1=Qᵀ。

数学正交不代表FP32保存后完全无误差。报告 `||Q.T@Q−I||F/sqrt(d)`，并单独比较整体模型FP logits。小维CPU参考用FP64，生产Q维护用FP32或稀疏角度；是否需数值重正交由实测决定，不能重正交后假称原参数/预算自动保持。

## 3. 双侧真实误差与风险预算

Ew=Dw(A)−A，Ea=Da(B)−B：

ΔY=Ew B + A Ea + Ew Ea。

其平方含3个范数平方与3个两两内积；默认不作独立、无偏或均匀噪声假设。令||Ew||F≤rw，||Ea||F≤ra，有：

||ΔY||F ≤ rw||X||₂ + ||W||₂ra + rwra。

无饱和nearest rounding的严格风险上界：B(Z)=¼Σ_g n_g s_g²。主约束是冻结面板加权BW/BA不高于Q₀。不能推导每一组、每一层或held-out都改善。

固定clip比例时定义实际可表示格点区间I_g=[s(qmin−zero),s(qmax−zero)]，round→clamp→dequant路径满足：

B_full(Z)=Σ_i dist(z_i,I_g)² + ¼Σ_g n_gs_g²。

若实际实现先在非格点边界clamp再round，应重新推导安全界，不能照搬。自适应MSE clip search与GPTQ不在主定理内；第一生产版本限制symmetric AbsMax，后续固定clip与asym作为明确扩展。

## 4. 完整事件与胞元

每个受影响浮点条目z(θ)=a+b cosθ+c sinθ。必须包括：

1. max/min/absmax owner切换（包括sign）；
2. 组内全部成员z=(k+½)s的舍入阈值；
3. zero-point/clipping分支；
4. budget边界；
5. zero-scale/epsilon等实现分支。

“不旋转的元素值不变”不代表code不变，因为共享scale可能变化。所有root应进行区间过滤与残差复核。重复根/同时事件需要稳定去重；合并容差不能大到消掉合法窄cell。

固定cell内，Ahat=A0+Ac cosθ+As sinθ，Bhat=B0+Bc cosθ+Bs sinθ。乘积系数：

- P0=A0B0+(AcBc+AsBs)/2
- Pc1=A0Bc+AcB0
- Ps1=A0Bs+AsB0
- Pc2=(AcBc−AsBs)/2
- Ps2=(AcBs+AsBc)/2

故F=c0+Σ_(k=1..4)[ak cos(kθ)+bk sin(kθ)]。预算是二阶。θ/2半角代换后F'根至多八次、预算根至多四次；θ=π等变换奇点需单独处理。

系数必须解析构造，不能在极窄cell内用多点插值拟合后假装精确。边界使用真实round-even；多重同时事件处的结果不必等于任一单侧极限。可能只有未取得的infimum，所以论文不要无条件说每个开cell存在minimizer。

本包参考使用数值roots、候选复核与near-boundary点，非interval arithmetic认证。即使CPU参考找到了比密网格更好的点，也不能声称O(d)全局最优或所有数值根均获形式认证。

## 5. 安全筛选与复杂度

每组最多两个变化成员u,v时，r=√(u²+v²)、M=max其余abs。M≥r保证整圈scale常数；其余成员的code不变，只枚举变化成员。组内一个变化成员时r仍取与组外配对元素组成的半径。

令N=受影响组数、Nu=未通过筛选组数、L=2^b，输入侧code事件数E=O(L[N+gNu])。非等长组：O(L[N+Σ_(unstable g) |g|])。最坏Nu=N没有渐近改进；必须实测Nu/N。

独立高斯、非自适应pair选择时，n=g−2，Pr(M<r)=E exp(−M²/2)≈√(π log n)/n，因此期望事件数量O(LN√log g)。这只是理想概率模型，不是真实LLM保证。

非对称min–max一个充分稳定条件为未变范围[l,h]包含[−r,r]。不可把AbsMax概率公式直接用于asym。

输出侧Q B会改变两整行，一组内g项全变化，需通用正弦包络；可由二维点集±(a_j,b_j)的凸包支持函数组织，active pieces最多O(g)。完整复杂度需加E_out，不能套常数包络界。

## 6. 增量更新

生产版维护5个P系数矩阵和9个F系数。一次weight code事件只改一输出行，更新O(T)；一次activation code事件只改一输出列，更新O(m)。scale owner变化可能改整组，需计O(gT)/O(gm)。每个事件重GEMM属于reference策略，不得用于性能宣传。

内存O(md+dT+mT+E)，外加Q/因子及输出侧状态。通过layer/panel流式处理限制峰值。真实事件/求根/同步次数也计时，不能只报算术FLOPs。

## 7. 多面板共享Q与预算

pair schedule在fit前固定，或明确记录选择依据。每次选角必须对所有纳入目标的面板采用相同θ，聚合相同固定归一化目标；不能逐面板独立选θ再把结果平均。

给定固定Q₀预算、固定面板、固定量化器，包含θ=0并只接受实际F下降且budget可行的更新，可保证校准目标不增；不存在严格改善保证。面板重采样后旧的单调性结论不再直接适用。

初始pilot角度预算K=32/128，若事件成本可控再到512或更多。K只在训练/验证集选，不根据正式Wiki/C4测试反复调。K个角度不是K×d²可训练参数；但导出denseR1仍需d²存储。

## 8. 生产API（待Codex实现）

```python
@dataclass(frozen=True)
class QuantSpec:
    bits: int
    group_size: int
    symmetric: bool
    clip_ratio: float = 1.0

@dataclass
class CalibrationPanel:
    panel_id: str
    placement: Literal['input', 'output']
    weight: Tensor
    activation: Tensor
    token_ids_hash: str
    complete_group_width: int
    objective_weight: float

def fit_shared_rotation(panels, q0, quant_specs, schedule, budgets, config):
    # Returns one global Q, sparse factors, acceptance log and timing.
    ...

def export_spinquant_rotation(Q, r2_per_layer, path):
    # R1=Q.T; save all layer R2 keys. Does not quantize weights.
    ...
```

## 9. 当前数学原型的作用

`cellquant.math_reference.run_reference_checks` 已提供真实CPU验证，限输入侧单对、对称无裁剪、完整组、小矩阵。guard限制d≤256和m+T≤256，避免4096维朴素枚举卡住。必须通过生产后端解决规模，而不是删掉guard后无预算运行或截断组。

后续Codex应先用此reference验证production GPU在相同小数据上的codes/scales/events/selected angle/true loss一致，再跑真实模型。数学实验可以使用合成数据；正式模型/资源结论不能使用合成数据。
# 2026-11-15 实现边界补充

生产实现见 `events.py` / `solver.py`。它保留完整分组和一个共享 Q；输入侧使用最多两个移动成员的 owner envelope，输出侧使用全部 signed (b,c) 点的 support hull。胞元损失/风险系数为解析构造，事件发生时做稀疏增量维护；接受一个角度后更新完整面板。

数值实现对预测最低的 32 个可行候选做完整量化复核，且边界执行真实 nearest-even rounding。此候选策略、FP64 根与舍入容差不能提供 interval-certified 或角度全局最优保证；报告该限制，实际更新必须满足损失不增加和风险预算。全零组的物理 scale/risk 为 0；除法保护用的 1 不记为物理 scale。

G0 已实测 3 seeds/24 cases；完整 d=4096 smoke 使用窄角域与少 tokens，不能当成正式全角域复杂度或模型吞吐证据。原参考 guard 保留。原有理论与科学不变量不因实现资源上限而放宽。

## 2026-11-17：小区间求根筛选与事件索引

对于三角多项式 `f(t)=a0+Σ(ak cos(kt)+bk sin(kt))`，令区间中点为 m、半宽为 h，则导数绝对值不超过 `L=Σ k(|ak|+|bk|)`，从而 `|f(t)-f(m)| <= hL`。在实数运算中，`|f(m)| > hL` 可排除该区间内的根。CUDA 数值实现额外加入 `64*eps*Σ|coeff|` 的舍入保护；接近零、切根和未能排除的区间仍使用原 FP64 QR 求根。该浮点保护不是 interval arithmetic，也没有新增 certified/global-optimal 声明。

事件关联表按边界编号在 GPU 上排序一次，并建立累积计数偏移；查找当前边界时仅读取对应片段。它保留同时事件、owner 标记与所有成员，不改变枚举角域或事件集合。以上优化有 CPU 参考/CUDA 回归与同卡合成计时，但真实 7B 是否可在原预算内完成仍需实测，不能由小案例加速推出。

## 2026-11-17：分面板系数与批量事件计算

`tensor_cells.py` 按各面板自身的事件构造损失和风险多项式，再在所有事件的并集上加权合并，最后统一求候选角度。输出两行的平方误差可以分别计算后相加；这不允许分别选择角度，所有行、层和输入/输出侧仍接受同一个 global Q 更新。

输出侧的激活不随 R1 变化。固定 owner 的区间内，一个量化组重建为 `s(theta) * C`，其中整数向量 C 在每个胞元内固定。编码事件使投影 `C @ Xhat` 改变 `delta_C[j] * Xhat[j, :]`；对这些投影增量作 GPU 前缀和即可得到一批胞元的线性输出系数。多组先叠加输出系数，再计算该行平方误差。每个 owner 区间及批次起点完整重建一次组投影以控制累积误差；没有逐角度重算完整隐藏维 GEMM。维度、组成员和事件不截断。

输入侧仅在所有受影响组满足原有 radius 稳定条件，或采用固定解码尺度时，使用精确的秩二更新：完整原乘积减去原来的两项外积，再加上当前两项外积。其余归约通道的贡献保留在缓存中。固定解码尺度时，风险仍按物理动态 AbsMax 独立计算，不能把风险也冻结。其他输入组继续使用双侧增量系数。

完整行分组的输入边界采用点值更新 `W_new = (s_new/s_old)*W_old + s_new*(C_new-C_old)`，激活侧同理；已有乘积承担尺度变化项，编码差承担稀疏外积项。先更新权重行，再更新激活列，后者使用已更新权重以保留交叉项。零尺度走明确计数的完整组回退。其他分组保留原增量边界路径。

边界按实际旋转值、AbsMax、除法、nearest-even 的顺序计算编码。不能在边界先将 owner 系数除以 qmax 再代入角度，最后一位误差可能改变 tie 的舍入分支。输出前缀路径对编码和尺度两种点值差异都作修正；最终仍完整量化复核最多 32 个预测可行候选。

候选筛选只把已有的小区间无根判据批量化；未排除、near-tie、切根和退化情况仍使用原 FP64 数值 QR。计时分为事件生成、排序、胞元系数及边界、候选搜索、最终复核；没有改变科学门槛或增加 interval-certified/global-optimal 声明。CPU/CUDA 数值对照和合成吞吐不能替代真实 G1。

## 2026-11-17：动态输入完整行的批处理

完整行分组时，令 Cw/Cx 为整数编码，去掉本次两个旋转成员后，乘积可写成 `diag(sw) K diag(sx)`，其中 `K=Cw Cx`。稳定组的未旋转成员编码不变；对 U 个不稳定权重行、V 个不稳定激活列，以基准编码 Cw0/Cx0 展开：

`K = Cw0 Cx0 + delta_Cw Cx0 + Cw0 delta_Cx + delta_Cw delta_Cx`。

前两项变化投影由编码差的 GPU 前缀和计算，每个批次重新计算起点投影；最后的双侧交叉项保留，在 U×V 交集中收缩完整 d 维。两个旋转成员的编码外积另行加回。所有行/列使用同一角度；这不是敏感通道分组或不同 Q。稳定组、组尺度和所有归约成员都保留。

`tensor_input_cells.py` 在每个胞元中点及实际事件边界批量解码不稳定组的全部成员，用实际旋转→AbsMax→除法→nearest-even 的顺序处理边界。中点只确定 owner/code 分支，损失系数仍由 scale 的线性三角系数解析构造，不插值拟合。完整行风险也由这些尺度解析计算；零尺度的物理风险仍为零。

这是减少逐事件 Python/CUDA 调度的时间/内存折中，不宣称新的事件线性复杂度：C 个点的非旋转编码计算为 O(C d(U+V))，交叉项为 O(C d U V)，另计稀疏投影增量、批次重算和 O(C mT) 的损失系数。工作区按批次限制；若不稳定组密集或不是完整行分组，保留原逐事件增量路径。执行策略不改变事件、角域、科学门槛或资源预算。

## 2026-11-18 的候选筛选与输出行更新

设胞元中点为 m、半宽为 h，对用于找驻点或预算交点的多项式 f，令 Aₖ = |aₖ|+|bₖ|。由 Taylor 余项，在该区间有

\[
|f(t)-f(m)|\le h\min\left(\sum_k k A_k,\; |f'(m)|+\tfrac12 h\sum_k k^2 A_k\right).
\]

当 |f(m)| 大于右边加 FP64 舍入保护时，可以跳过该多项式的 QR。中点斜率的不同谐波会抵消，原先逐项取绝对值在窄胞元上可能很松。保护项随系数、h 和各阶导数上界缩放。无法排除的情形保留数值求根，边界照常执行实际舍入，最终候选仍做双侧量化复核。浮点实现的适用范围仍是冻结面板上的数值搜索。

对于已通过半径筛选的输入组，可批量求解移动成员的阈值方程，然后按组恢复每个事件的角度和成员编号。组外配对仍使用完整配对半径。不稳定组继续对完整成员枚举。接受输出侧旋转后，只有 Q B 以及目标 Q B U 中的两行改变，每行须使用其完整归约维。缓存其余行与已量化激活，重算两行并从当前全体残差与物理尺度求和，可以同时保留动态风险和固定尺度对照的语义。

