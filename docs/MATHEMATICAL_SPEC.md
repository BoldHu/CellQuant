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

