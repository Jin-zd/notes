## 1 独立同分布（i.i.d.）
在概率论和统计学中，**独立同分布 (independent and identically distributed, i.i.d.)** 是描述一系列随机变量的一个重要概念。如果一组随机变量满足以下两个条件，我们就称它们是独立同分布的：
1. **独立性 (Independence):** 这意味着序列中任何一个随机变量的取值都不会影响到其他随机变量的取值。换句话说，如果你观察到其中一个变量的值，这并不会提供关于其他任何变量值的额外信息。它们之间没有任何依赖关系。
2. **相同分布 (Identically Distributed):** 这意味着序列中的每一个随机变量都遵循相同的概率分布。这意味着它们具有相同的概率质量函数（对于离散随机变量）或相同的概率密度函数（对于连续随机变量），以及相同的参数（例如，均值和方差）。

## 2 总变差距离（Total Variation Distance）
总变差距离（Total Variation Distance, TVD），也称为**统计距离**（statistical distance）、**统计差异**（statistical difference）或**变分距离**（variational distance），是一种衡量定义在相同样本空间上的两个概率分布之间距离的方法。直观地说，它量化了这两个分布对同一事件赋予概率的最大可能差异。
### 2.1 定义
设 $P$ 和 $Q$ 是在样本空间 $\Omega$ 上的两个概率分布，令 $\mathcal{A}$ 是 $\Omega$ 中所有可测事件的集合。$P$ 和 $Q$ 之间的总变差距离，记为 $d_{TV}(P, Q)$，定义为：
$$d_{TV}(P, Q) = \sup_{A \in \mathcal{A}} |P(A) - Q(A)|$$
这意味着我们考察所有可能的事件 $A$，并找到在分布 $P$ 下该事件发生的概率与在分布 $Q$ 下该事件发生的概率之间绝对差值的最大值。
### 2.2 等价定义
总变差距离有几个等价的表达形式：
1.  **离散/连续分布的概率质量/密度函数之间 $L_1$ 距离的一半：**
    * **离散分布：** 对于具有概率质量函数 $p(x)$ 和 $q(x)$：$$d_{TV}(P, Q) = \frac{1}{2} \sum_{x \in \Omega} |p(x) - q(x)|$$
    * **连续分布：** 对于具有概率密度函数 $f(x)$ 和 $g(x)$：$$d_{TV}(P, Q) = \frac{1}{2} \int_{\Omega} |f(x) - g(x)| dx$$这种解释强调了总变差距离与两个概率分布之间区域的大小有关。
2.  **有界函数的期望值之差的最大值：**$$d_{TV}(P, Q) = \sup_{|f(x)| \leq 1} |E_P[f(X)] - E_Q[f(X)]|$$其中 $E_P$ 和 $E_Q$ 分别表示在分布 $P$ 和 $Q$ 下的期望。
### 2.3 性质
* **范围：** $0 \leq d_{TV}(P, Q) \leq 1$
    * $d_{TV}(P, Q) = 0 \iff P = Q$
    * $d_{TV}(P, Q) = 1 \iff P$ 和 $Q$ 是互斥的
* **对称性：** $d_{TV}(P, Q) = d_{TV}(Q, P)$
* **三角不等式：** $d_{TV}(P, R) \leq d_{TV}(P, Q) + d_{TV}(Q, R)$


## 3 柯西序列（Cauchy sequence）
### 3.1 定义
在一个度量空间 $(X, d)$ 中，一个序列 $\{x_n\}_{n=1}^\infty$ 被称为**柯西序列**，如果对于任意给定的 $\epsilon > 0$，都存在一个正整数 $N$，使得对于所有 $n, m > N$，都有 $d(x_n, x_m) < \epsilon$。
直观地理解，柯西序列是指随着序列的进行，序列中的项越来越“靠近”彼此。无论我们选择多么小的正数 $\epsilon$，在序列的某个位置之后的所有项之间的距离都将小于 $\epsilon$。
### 3.2 性质
1.  **收敛序列一定是柯西序列：**
    如果一个序列 $\{x_n\}$ 收敛于 $L$，那么对于任意 $\epsilon > 0$，存在 $N$ 使得当 $n > N$ 时，$d(x_n, L) < \frac{\epsilon}{2}$。因此，对于 $n, m > N$，根据三角不等式有：$$d(x_n, x_m) \le d(x_n, L) + d(L, x_m) < \frac{\epsilon}{2} + \frac{\epsilon}{2} = \epsilon$$所以，收敛序列满足柯西序列的定义。
2.  **柯西序列是有界的：**
    设 $\{x_n\}$ 是一个柯西序列。取 $\epsilon = 1$，则存在一个正整数 $N$ 使得对于所有 $n, m > N$，有 $d(x_n, x_m) < 1$。特别地，对于所有 $n > N$，有 $d(x_n, x_{N+1}) < 1$，这意味着 $x_n$ 落在以 $x_{N+1}$ 为中心，半径为 1 的开球内。
    考虑集合 $\{x_1, x_2, \dots, x_N, x_{N+1}\}$，令 $M = \max\{d(x_i, x_j) \mid 1 \le i, j \le N+1\} + 1$。那么对于序列中的任意两项 $x_i$ 和 $x_j$，它们的距离 $d(x_i, x_j)$ 都小于某个有限值，因此序列是有界的。更具体地说，我们可以找到一个包含所有序列项的足够大的球。
3.  **在完备的度量空间中，柯西序列一定是收敛序列：**
    一个度量空间被称为**完备的**，如果该空间中的每一个柯西序列都收敛到该空间中的一个点。实数集 $\mathbb{R}$ 配备上标准的绝对值度量是完备的，欧几里得空间 $\mathbb{R}^n$ 也是完备的。有理数集 $\mathbb{Q}$ 配备上标准的绝对值度量不是完备的，因为存在由有理数组成的柯西序列，其极限是无理数（不在 $\mathbb{Q}$ 中）。
4.  **柯西序列的子列：**
    如果 $\{x_n\}$ 是一个柯西序列，那么它的任何子列 $\{x_{n_k}\}$ 也是一个柯西序列。这是因为子列中的项仍然是原序列中的项，所以对于任意 $\epsilon > 0$，存在 $N$ 使得当 $n, m > N$ 时，$d(x_n, x_m) < \epsilon$。对于子列，当 $n_k, n_j > N$ 时（这意味着 $k$ 和 $j$ 都足够大），同样有 $d(x_{n_k}, x_{n_j}) < \epsilon$。
5.  **柯西序列的极限的唯一性（如果存在）：**
    如果一个柯西序列 $\{x_n\}$ 收敛到 $L_1$ 并且也收敛到 $L_2$，那么 $L_1 = L_2$。这是因为根据收敛的定义，对于任意 $\epsilon > 0$，存在 $N_1$ 使得当 $n > N_1$ 时，$d(x_n, L_1) < \frac{\epsilon}{2}$，并且存在 $N_2$ 使得当 $n > N_2$ 时，$d(x_n, L_2) < \frac{\epsilon}{2}$。取 $N = \max(N_1, N_2)$，则对于 $n > N$，有：$$d(L_1, L_2) \le d(L_1, x_n) + d(x_n, L_2) < \frac{\epsilon}{2} + \frac{\epsilon}{2} = \epsilon$$由于 $\epsilon$ 是任意正数，因此必须有 $d(L_1, L_2) = 0$，即 $L_1 = L_2$。


## 4 占用率测度 (Occupancy Measure)

### 4.1 定义
在强化学习中，占用率测度 $\mu^\pi(s, a)$ 是指在策略 $\pi$ 下，智能体在状态 $s$ 采取动作 $a$ 的长期平均频率或概率分布。更正式地说，对于一个给定的策略 $\pi$，占用率测度定义为：
$$\mu^\pi(s, a) = \lim_{T \to \infty} \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{P}(S_t = s, A_t = a | \pi)$$
或者，对于折扣奖励的情况，可以定义为折扣占用率测度：
$$\mu^\pi(s, a) = (1 - \gamma) \sum_{t=0}^{\infty} \gamma^t \mathbb{P}(S_t = s, A_t = a | \pi)$$
其中：
- $\pi$ 是智能体遵循的策略。
- $S_t$ 是 $t$ 时刻的状态。
- $A_t$ 是 $t$ 时刻采取的动作。
- $\mathbb{P}(S_t = s, A_t = a | \pi)$ 是在策略 $\pi$ 下，在 $t$ 时刻处于状态 $s$ 并采取动作 $a$ 的概率。
- $T$ 是时间步的总数。
- $\gamma \in [0, 1)$ 是折扣因子。
### 4.2 重要性
许多强化学习算法，特别是基于模型的方法和一些策略优化方法，都显式或隐式地使用了占用率测度的概念。价值函数（如状态值函数 $V^\pi(s)$ 和动作值函数 $Q^\pi(s, a)$）可以表示为占用率测度的函数。如：$$V^\pi(s) = \sum_{a \in \mathcal{A}} \mu^\pi(s, a) Q^\pi(s, a) / \sum_{a' \in \mathcal{A}} \mu^\pi(s, a')$$$$Q^\pi(s, a) = \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s, a) V^\pi(s')$$其中 $\mathcal{R}(s, a)$ 是在状态 $s$ 采取动作 $a$ 的期望奖励，$P(s'|s, a)$ 是在状态 $s$ 采取动作 $a$ 后转移到状态 $s'$ 的概率。
### 4.3 性质
- 占用率测度 $\mu^\pi(s, a) \ge 0$ 对于所有 $s \in \mathcal{S}, a \in \mathcal{A}$。
- 对于无折扣的情况，$\sum_{s \in \mathcal{S}} \sum_{a \in \mathcal{A}} \mu^\pi(s, a) = 1$ （如果状态空间和动作空间是有限的）。
- 对于折扣的情况，$\sum_{s \in \mathcal{S}} \sum_{a \in \mathcal{A}} \frac{1}{1-\gamma} \mu^\pi(s, a) = 1$ （如果初始状态分布是固定的）。

## 5 平稳分布 (Stationary Distribution)
### 5.1 定义
对于一个具有状态空间 $S$ 和转移概率矩阵 $P$ 的马尔可夫链，一个概率分布 $\pi$ （在 $S$ 上的非负行向量，元素之和为 1）被称为平稳分布，如果满足：
$$\pi P = \pi$$
平稳分布是指马尔可夫链经过一步转移后，其状态的概率分布仍然保持不变。如果初始分布就是平稳分布，那么该链在所有后续时间的分布都将保持这个状态。
### 5.2 性质
* **不变性 (Invariance):** 若初始分布为 $\pi$，则 $$P(X_0 = i) = \pi_i \implies P(X_1 = j) = \sum_{i \in S} P(X_0 = i) P(j|i) = \sum_{i \in S} \pi_i P_{ij} = (\pi P)_j = \pi_j$$以此类推，$P(X_t = j) = \pi_j$ 对所有 $t \ge 0$ 成立。
* **极限分布 (Limiting Distribution):** 对于某些具有良好性质（如不可约且非周期）的马尔可夫链，无论初始分布如何，$n \to \infty$ 时，其状态分布会收敛到一个唯一的平稳分布。此时，平稳分布也称为极限分布。
* **存在性与唯一性:**
    * 并非所有马尔可夫链都存在平稳分布。
    * 即使存在，平稳分布也可能不唯一。
    * 对于有限状态的不可约、正常返马尔可夫链，存在唯一的平稳分布。

## 6 KKT (Karush-Kuhn-Tucker) 条件
### 6.1 定义
考虑优化问题：
$$\begin{aligned}
\min_{x} \quad & f(x) \\
\text{s.t.} \quad & g_i(x) \le 0, \quad i = 1, \dots, m \\
& h_j(x) = 0, \quad j = 1, \dots, p
\end{aligned}$$
假设： $x^*$ 是局部最优解，且满足约束规范（例如 Slater 条件，LICQ）。
KKT 条件：存在乘子 $\mu_i^* \ge 0$ ($i = 1, \dots, m$) 和 $\lambda_j^*$ ($j = 1, \dots, p$)，使得以下条件成立：
1.  梯度条件 (Stationarity)：$$\nabla f(x^*) + \sum_{i=1}^{m} \mu_i^* \nabla g_i(x^*) + \sum_{j=1}^{p} \lambda_j^* \nabla h_j(x^*) = 0$$
2.  互补松弛条件 (Complementary Slackness)：$$\mu_i^* g_i(x^*) = 0, \quad i = 1, \dots, m$$如果 $g_i(x^*) < 0$ （非起作用约束），则 $\mu_i^* = 0$，如果 $\mu_i^* > 0$, 则 $g_i(x^*) = 0$ （起作用约束）。
3.  原始可行性 (Primal Feasibility)：$$g_i(x^*) \le 0, \quad i = 1, \dots, m$$$$h_j(x^*) = 0, \quad j = 1, \dots, p$$
4.  对偶可行性 (Dual Feasibility)：$$\mu_i^* \ge 0, \quad i = 1, \dots, m$$$\lambda_j^*$ 的符号没有限制。
### 6.2 重要性
* 最优性的必要条件 (在满足约束规范下)。
* 凸优化问题在满足 Slater 条件下，KKT 条件也是充分条件。
* 是许多约束优化算法的基础。
* 满足 KKT 条件的点不一定是全局最优解（可能是局部最优解或鞍点）。
* KKT 条件的成立依赖于约束规范。

## 7 Sion 最小最大定理（Sion's Minimax Theorem）
设 $X$ 是一个紧致的凸集，而 $Y$ 是一个凸集（不必紧致）。令函数 $f: X \times Y \rightarrow \mathbb{R}$ 满足以下条件：
1.  对于每个 $y \in Y$，函数 $f(\cdot, y): X \rightarrow \mathbb{R}$ 在 $X$ 上是连续的。
2.  对于每个 $x \in X$，函数 $f(x, \cdot): Y \rightarrow \mathbb{R}$ 在 $Y$ 上是拟凹的 (quasi-concave)。
则有：
$$\min_{y \in Y} \max_{x \in X} f(x, y) = \max_{x \in X} \min_{y \in Y} f(x, y)$$
* 紧致凸集 (Compact Convex Set)： 一个集合既是紧致的（闭且有界），又是凸的（集合中任意两点的连线上的所有点也属于该集合）。
* 凸集 (Convex Set)：集合中任意两点的连线上的所有点也属于该集合。
* 连续函数 (Continuous Function)：直观上，函数值的微小变化对应于自变量的微小变化。
* 拟凹函数 (Quasi-concave Function)：对于任意实数 $\alpha$，集合 $\{y \in Y \mid f(x, y) \ge \alpha \}$ 是凸集。注意，这比凹函数的要求弱。

## 8 条件数（Condition Number）
### 8.1 定义
对于一个可微函数 $f: \mathbb{R}^n \rightarrow \mathbb{R}$，在点 $x^*$ 处的 Hessian 矩阵为 $\nabla^2 f(x^*)$。如果 $\nabla^2 f(x^*)$ 是可逆的，则在 $x^*$ 处的条件数 $\kappa$ 定义为：
$$\kappa = \frac{\lambda_{\max}(\nabla^2 f(x^*))}{\lambda_{\min}(\nabla^2 f(x^*))}$$
其中，$\lambda_{\max}(\nabla^2 f(x^*))$ 和 $\lambda_{\min}(\nabla^2 f(x^*))$ 分别是 Hessian 矩阵的最大和最小特征值。
### 8.2 几何意义
条件数衡量了目标函数在最速下降方向和曲率最大方向上的伸缩比例。
* 条件数较小 ($\kappa \approx 1$)：Hessian 矩阵的特征值相近，目标函数的等高线近似于圆形。在这种情况下，梯度下降等优化算法通常收敛较快且稳定。
* 条件数较大 ($\kappa \gg 1$)：Hessian 矩阵的特征值差异很大，目标函数的等高线呈现狭长的椭圆状。在这种情况下，梯度下降等算法可能会在狭长方向上震荡，收敛速度显著减慢。
对优化算法的影响：
* 收敛速度： 高条件数通常会导致优化算法收敛缓慢，尤其是一阶方法（如梯度下降）。这是因为梯度方向可能与指向最优解的方向相差很大。
* 数值稳定性：高条件数也可能导致数值不稳定，因为算法对参数的微小变化非常敏感。
* 预处理：为了改善高条件数问题，常常采用预处理技术（如尺度变换、牛顿法的近似等），旨在改变问题的几何形状，降低条件数，从而加速收敛。

## 9 KL 散度 (Kullback-Leibler Divergence)
### 9.1 定义
KL 散度，通常记作 $D_{KL}(P||Q)$，用于衡量两个概率分布 $P$ 和 $Q$ 之间的差异。对于离散概率分布 $P(x)$ 和 $Q(x)$，其定义为：
$$D_{KL}(P||Q) = \sum_{x} P(x) \log \left( \frac{P(x)}{Q(x)} \right)$$
对于连续概率分布 $p(x)$ 和 $q(x)$，其定义为：
$$D_{KL}(P||Q) = \int_{-\infty}^{\infty} p(x) \log \left( \frac{p(x)}{q(x)} \right) dx$$
### 9.2 性质
* 非负性：$D_{KL}(P||Q) \ge 0$，当且仅当 $P = Q$ 时等号成立。
* 不对称性：一般情况下，$D_{KL}(P||Q) \neq D_{KL}(Q||P)$。这意味着 $P$ 相对于 $Q$ 的差异与 $Q$ 相对于 $P$ 的差异通常是不同的。因此，KL 散度不是一个真正的距离度量。
* 期望：KL 散度可以看作是使用分布 $Q$ 近似分布 $P$ 时，所损失的平均额外信息量（以比特或纳特为单位）。
* 当 $P(x) > 0$ 且 $Q(x) = 0$ 时，$\log \left( \frac{P(x)}{Q(x)} \right) = \infty$，导致 $D_{KL}(P||Q) = \infty$。这意味着如果真实分布中可能出现的事件在近似分布中概率为零，则 KL 散度为无穷大。
* 当 $P(x) = 0$ 时，按照惯例认为 $0 \log \left( \frac{0}{Q(x)} \right) = 0$。

## 10 蒙特卡洛方法 (Monte Carlo Method)
### 10.1 概述

蒙特卡洛方法是一种通过随机抽样来估计数值结果的计算技术。它依赖于重复的随机抽样来获得概率结果，并利用这些结果来逼近真实值。这种方法特别适用于解决那些难以通过确定性算法或解析方法解决的问题，尤其是在涉及高维度、复杂模型或不确定性的情况下。
### 10.2 关键步骤
1.  构建概率模型：
    * 确定问题的输入变量和输出变量。
    * 为输入变量定义合适的概率分布。
    * 建立输入变量与输出变量之间的关系模型。
2.  生成随机数：
    * 使用伪随机数生成器产生服从指定概率分布的随机数。
    * 确保生成的随机数具有良好的统计特性（如均匀性、独立性）。
3.  执行模拟：
    * 对于每一组随机生成的输入变量，运行模型并记录输出结果。
    * 重复此过程大量的次数（模拟次数 $N$）。
4.  分析结果：
    * 计算所有模拟输出结果的统计量，例如均值 ($\bar{X}$)、方差 ($\sigma^2$)、置信区间等。
    * 均值通常作为问题解的估计值：$$\bar{X} = \frac{1}{N} \sum_{i=1}^{N} X_i$$其中，$X_i$ 是第 $i$ 次模拟的输出结果。
5.  评估精度：
    * 通过增加模拟次数来提高估计的精度。
    * 可以使用中心极限定理等理论来估计结果的误差范围。

### 10.3 特点
* 易于并行化：每次模拟之间通常相互独立，方便并行计算。
* 收敛速度慢：通常需要大量的模拟次数才能获得较高的精度，收敛速度约为 $O(1/\sqrt{N})$。
* 高维度问题挑战：在极高维度的问题中，所需的样本数量会急剧增加（维度灾难）。
### 10.4 提高效率的方法
* 重要性抽样 (Importance Sampling)：将抽样集中在对结果贡献较大的区域。
* 方差缩减技术 (Variance Reduction Techniques)：如控制变量法、分层抽样法、对偶变量法等，旨在减少估计结果的方差，提高精度。
* 准蒙特卡洛方法 (Quasi-Monte Carlo Methods)：使用低差异序列代替伪随机数，以提高收敛速度。

## 11 胡贝尔损失（Huber Loss）
### 11.1 定义
Huber Loss 是回归任务中常用的损失函数，结合了均方误差（MSE）和平均绝对误差（MAE）的优点，对离群点（outliers）鲁棒性。公式如下：
$$
L_{\delta}(y, \hat{y}) = 
\begin{cases} 
\frac{1}{2}(y - \hat{y})^2 & \text{if} \; |y - \hat{y}| \leq \delta, \\
\delta \cdot |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise.}
\end{cases}
$$
- $y$：真实值  
- $\hat{y}$：预测值  
- $\delta$：超参数，控制平方误差与绝对误差的切换阈值（通常取 1.0）
### 11.2 特点
- 在误差较小时使用平方项（类似 MSE），误差较大时使用线性项（类似 MAE），避免对离群点过度敏感。
- 通过调整 $\delta$ 平衡 MSE 和 MAE 的特性：$\delta$ → 0 时接近 MAE，$\delta$ → ∞ 时接近 MSE。
- 在误差为 $\pm \delta$ 处一阶导数连续，优化过程更稳定。

## 12 主-最小化 / 次-最大化 (MM) 算法
### 12.1 核心思想
MM 算法是一种迭代优化算法，其核心思想是通过构建一个比目标函数更容易优化的替代函数（surrogate function），并通过迭代地最小化（或最大化）这个替代函数来逼近目标函数的最小值（或最大值）。
### 12.2 基本框架
假设我们要最小化目标函数 $f(\mathbf{x})$。MM 算法的迭代过程如下：
1. 构建替代函数 (Majorization)：在当前迭代点 $\mathbf{x}^{(k)}$，构建一个替代函数 $g(\mathbf{x} | \mathbf{x}^{(k)})$，该函数需要满足以下条件（对于最小化问题）：
   - $g(\mathbf{x}^{(k)} | \mathbf{x}^{(k)}) = f(\mathbf{x}^{(k)})$  （在当前点与目标函数值相等）
   - $g(\mathbf{x} | \mathbf{x}^{(k)}) \ge f(\mathbf{x})$ 对于所有 $\mathbf{x}$ （替代函数是目标函数的上界）
2. 最小化替代函数 (Minimization)：找到下一个迭代点 $\mathbf{x}^{(k+1)}$，通过最小化替代函数得到：$$ \mathbf{x}^{(k+1)} = \arg\min_{\mathbf{x}} g(\mathbf{x} | \mathbf{x}^{(k)}) $$
3. 迭代：重复步骤 1 和 2，直到满足收敛条件。

对于最大化问题 (Minorization-Maximization)：
类似地，对于最大化目标函数 $f(\mathbf{x})$，我们需要构建一个替代函数 $m(\mathbf{x} | \mathbf{x}^{(k)})$，满足：
1. 构建替代函数 (Minorization)：
   * $m(\mathbf{x}^{(k)} | \mathbf{x}^{(k)}) = f(\mathbf{x}^{(k)})$
   * $m(\mathbf{x} | \mathbf{x}^{(k)}) \le f(\mathbf{x})$ 对于所有 $\mathbf{x}$ (替代函数是目标函数的下界)
2. 最大化替代函数 (Maximization)：$$ \mathbf{x}^{(k+1)} = \arg\max_{\mathbf{x}} m(\mathbf{x} | \mathbf{x}^{(k)}) $$
### 12.3 性质
* 单调性 (Monotonicity)意味着目标函数的值在每次迭代中都会单调下降（或上升），保证了算法的稳定性：
    * 对于最小化：$f(\mathbf{x}^{(k+1)}) \le g(\mathbf{x}^{(k+1)} | \mathbf{x}^{(k)}) \le g(\mathbf{x}^{(k)} | \mathbf{x}^{(k)}) = f(\mathbf{x}^{(k)})$
    * 对于最大化：$f(\mathbf{x}^{(k+1)}) \ge m(\mathbf{x}^{(k+1)} | \mathbf{x}^{(k)}) \ge m(\mathbf{x}^{(k)} | \mathbf{x}^{(k)}) = f(\mathbf{x}^{(k)})$
* 简化优化问题：MM 算法的关键在于选择合适的替代函数，使得最小化（或最大化）替代函数比直接优化目标函数更容易。
### 12.4 构建替代函数
构建合适的替代函数是 MM 算法的核心挑战。常用的方法包括：
* 二次上界 (Quadratic Majorization)：利用目标函数的二阶信息（例如 Hessian 矩阵）构建二次上界。
* 琴生不等式 (Jensen's Inequality)：对于凸函数或凹函数，可以利用琴生不等式构建线性或更简单的替代函数。
* 算术-几何平均不等式 (AM-GM Inequality)：在处理包含乘积项的目标函数时非常有用。
* 泰勒展开 (Taylor Expansion)：利用目标函数的一阶或二阶泰勒展开构建局部近似。

## 13 共轭梯度下降 (Conjugate Gradient Descent)
### 13.1 核心思想
共轭梯度下降法是一种用于求解对称正定线性方程组 $\mathbf{Ax} = \mathbf{b}$ 的迭代方法，也可以推广到求解无约束非线性优化问题。其核心思想是在每次迭代中选择一个与先前搜索方向共轭的方向作为新的搜索方向，从而更快地收敛到解。
### 13.2 求解线性方程组
求解：$\mathbf{Ax} = \mathbf{b}$ (A 是对称正定矩阵)

迭代过程：
1. 初始化：
   * 选择初始猜测 $\mathbf{x}_0$。
   * 计算初始残差 $\mathbf{r}_0 = \mathbf{b} - \mathbf{Ax}_0$。
   * 设置初始搜索方向 $\mathbf{p}_0 = \mathbf{r}_0$。
   * 设置迭代次数 $k = 0$。
2. 迭代步骤：对于 $k = 0, 1, 2, \dots$：
   * 计算步长 (Step size)：$$ \alpha_k = \frac{\mathbf{r}_k^T \mathbf{r}_k}{\mathbf{p}_k^T \mathbf{Ap}_k} $$
   * 更新解：$$ \mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k $$
   * 更新残差：$$ \mathbf{r}_{k+1} = \mathbf{b} - \mathbf{Ax}_{k+1} = \mathbf{r}_k - \alpha_k \mathbf{Ap}_k $$
   * 计算共轭系数 (Conjugate coefficient)：$$ \beta_k = \frac{\mathbf{r}_{k+1}^T \mathbf{r}_{k+1}}{\mathbf{r}_k^T \mathbf{r}_k} $$
   * 更新搜索方向：$$ \mathbf{p}_{k+1} = \mathbf{r}_{k+1} + \beta_k \mathbf{p}_k $$
   * 检查收敛条件：如果满足收敛条件（例如，残差足够小），则停止迭代。
   * 更新迭代次数： $k \leftarrow k + 1$。

共轭性（Conjugacy）：
向量 $\mathbf{u}$ 和 $\mathbf{v}$ 关于对称正定矩阵 $\mathbf{A}$ 是共轭的，如果满足：
$$ \mathbf{u}^T \mathbf{Av} = 0 $$
共轭性保证了在每个新的搜索方向上，不会干扰到之前迭代中已经优化过的方向上的最优性。
性质：
* 对于一个 $n$ 维问题，共轭梯度法在理想情况下最多经过 $n$ 次迭代就能找到精确解（在没有舍入误差的情况下）。
* 每一步的搜索方向 $\mathbf{p}_k$ 是当前残差 $\mathbf{r}_k$ 和前一个搜索方向 $\mathbf{p}_{k-1}$ 的线性组合。
* 残差序列 $\{\mathbf{r}_k\}$ 是相互正交的，即 $\mathbf{r}_i^T \mathbf{r}_j = 0$ 对于 $i \neq j < k$。
* 搜索方向序列 $\{\mathbf{p}_k\}$ 是相互共轭的，即 $\mathbf{p}_i^T \mathbf{Ap}_j = 0$ 对于 $i \neq j < k$。

### 13.3 求解无约束非线性优化问题 
求解：$\min f(\mathbf{x})$

将共轭梯度的思想应用于非线性优化，其目标是找到函数 $f(\mathbf{x})$ 的最小值。此时，残差 $\mathbf{r}_k$ 被负梯度 $-\nabla f(\mathbf{x}_k)$ 所替代。

迭代过程 (Fletcher-Reeves 版本)：
1. 初始化：
   * 选择初始猜测 $\mathbf{x}_0$。
   * 计算初始梯度 $\mathbf{g}_0 = \nabla f(\mathbf{x}_0)$。
   * 设置初始搜索方向 $\mathbf{p}_0 = -\mathbf{g}_0$。
   * 设置迭代次数 $k = 0$。
2. 迭代步骤： 对于 $k = 0, 1, 2, \dots$：
   * 计算步长 (Line search)：找到一个步长 $\alpha_k > 0$，使得 $f(\mathbf{x}_k + \alpha_k \mathbf{p}_k)$ 充分减小（例如，满足 Wolfe 条件或 Armijo 条件）。
   * 更新解：$$ \mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k $$
   * 计算新的梯度：$$ \mathbf{g}_{k+1} = \nabla f(\mathbf{x}_{k+1}) $$
   * 计算共轭系数 (Fletcher-Reeves)：$$ \beta_k = \frac{\mathbf{g}_{k+1}^T \mathbf{g}_{k+1}}{\mathbf{g}_k^T \mathbf{g}_k} $$
   * 更新搜索方向：$$ \mathbf{p}_{k+1} = -\mathbf{g}_{k+1} + \beta_k \mathbf{p}_k $$
   * 检查收敛条件：如果满足收敛条件（例如，梯度足够小），则停止迭代。
   * 更新迭代次数： $k \leftarrow k + 1$。

除了 Fletcher-Reeves，还有其他计算 $\beta_k$ 的方法，例如：
* Polak-Ribière： $\beta_k = \frac{\mathbf{g}_{k+1}^T (\mathbf{g}_{k+1} - \mathbf{g}_k)}{\mathbf{g}_k^T \mathbf{g}_k}$
* Hestenes-Stiefel： $\beta_k = \frac{\mathbf{g}_{k+1}^T (\mathbf{g}_{k+1} - \mathbf{g}_k)}{\mathbf{p}_k^T (\mathbf{g}_{k+1} - \mathbf{g}_k)}$
* Dai-Yuan： $\beta_k = \frac{\mathbf{g}_{k+1}^T \mathbf{g}_{k+1}}{\mathbf{p}_k^T (\mathbf{g}_{k+1} - \mathbf{g}_k)}$
不同的 $\beta_k$ 计算方法在不同的问题上可能表现出不同的收敛性能。

## 14 对偶梯度下降（Dual Gradient Descent）
### 14.1 定义
考虑如下带约束的优化问题：
$$
\begin{aligned}
\min_{w} \quad & f(w) \\
\text{s.t.} \quad & g_i(w) \le 0, \quad i = 1, \dots, m \\
& h_j(w) = 0, \quad j = 1, \dots, p
\end{aligned}
$$
其中，$f(w)$ 是目标函数，$g_i(w)$ 是不等式约束，$h_j(w)$ 是等式约束。

定义拉格朗日函数为：
$$L(w, \alpha, \beta) = f(w) + \sum_{i=1}^{m} \alpha_i g_i(w) + \sum_{j=1}^{p} \beta_j h_j(w)$$
其中，$\alpha_i \ge 0$ 是与不等式约束相关的拉格朗日乘子，$\beta_j$ 是与等式约束相关的拉格朗日乘子。

对偶函数 $q(\alpha, \beta)$ 定义为拉格朗日函数关于 $w$ 的最小值：
$$q(\alpha, \beta) = \inf_{w} L(w, \alpha, \beta) = \inf_{w} \left( f(w) + \sum_{i=1}^{m} \alpha_i g_i(w) + \sum_{j=1}^{p} \beta_j h_j(w) \right)$$
对偶问题是最大化对偶函数，并满足拉格朗日乘子的约束：
$$
\begin{aligned}
\max_{\alpha, \beta} \quad & q(\alpha, \beta) \\
\text{s.t.} \quad & \alpha_i \ge 0, \quad i = 1, \dots, m
\end{aligned}
$$
对偶梯度下降是一种解决对偶问题的迭代优化算法。它通过在对偶变量（拉格朗日乘子）上执行梯度上升来最大化对偶函数。

假设对偶函数 $q(\alpha, \beta)$ 是可微的，我们可以计算其关于 $\alpha$ 和 $\beta$ 的梯度。根据链式法则，如果最优解 $w^*$ 存在，并且拉格朗日函数在 $w^*$ 处可微，那么：
$$
\nabla_{\alpha_i} q(\alpha, \beta) = g_i(w^*) \\
\nabla_{\beta_j} q(\alpha, \beta) = h_j(w^*)
$$

对偶梯度下降的更新规则如下：
$$
\alpha_i^{(k+1)} = \max(0, \alpha_i^{(k)} + \eta_k g_i(w^{(k)})) \\
\beta_j^{(k+1)} = \beta_j^{(k)} + \eta_k h_j(w^{(k)})
$$
其中，$\eta_k > 0$ 是学习率，$w^{(k)}$ 是在当前对偶变量 $(\alpha^{(k)}, \beta^{(k)})$ 下最小化拉格朗日函数得到的 $w$ 的值。
### 14.2 注意
* 对偶梯度下降适用于对偶问题比原始问题更容易求解的情况。
* 学习率 $\eta_k$ 的选择对算法的收敛性至关重要。
* 在每一步都需要解决一个最小化拉格朗日函数的子问题。对于某些问题，这个子问题可能仍然很困难。
* 当原始问题是凸的并且满足某些约束条件（如 Slater 条件）时，强对偶性成立，原始问题的最优值等于对偶问题的最优值。在这种情况下，求解对偶问题可以得到原始问题的解。