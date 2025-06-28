在本节中，我们将简单介绍强化学习理论是什么样的，例如其关注的问题，通常的假设，以及对强化学习理论的正确期望。在此基础上我们会介绍一些基本的工具，以及基于模型的强化学习，无模型的强化学习在一系列假设下可以获得的一些理论结果。
## 1 Introduction
### 1.1 Problems we asks in RL theory
在强化学习理论中我们通常会问什么问题？

第一类问题是关于学习的，主要有以下的形式（记号的含义会在后面详细介绍）：
例如，如果使用了某个算法，使用了 $N$ 个 样本，迭代 $k$ 次，我们学习的结果能有多好？

不妨假设使用 Q 学习。对于一定的误差 $\epsilon$，能否说明如果 $N \geq f(\epsilon)$，能以概率 $1 - \delta$ 有
$$
\|\hat{Q}_k - Q^\ast \|_\infty \leq \epsilon
$$
这里 $\hat{Q}_k$ 是在第 $k$ 次迭代后（学到的） $Q$ 函数， $Q^\ast$ 是最优的 $Q$ 函数。

类似地还可以是
$$
\|Q^{\pi_k} - Q^\ast\|_\infty \leq \epsilon
$$
这里 $Q^{\pi_k}$ 是策略 $\pi_k$对应的真实 $Q$ 函数，这是实际在问期望奖励上的差异，换言之即遗憾值。

第二类问题是关于探索的，主要有以下的形式：
例如，如果使用了某个特定的探索算法，遗憾值会有多少？

一个可能得到的结果形如：
$$
\text{Reg}(T) \leq O\left(\sqrt{T \cdot N \cdot \log \frac{NT}{\delta}}\right) + \delta T
$$
我们不去深究这个公式的具体形式。

在本节中主要关注第一类也就是学习相关的问题。

### 1.2 Assumptions in RL theory
在强化学习理论中，通常需要使用很强的假设。对于很弱的假设，通常推不出什么有用的结论，但是过强的假设通常又会偏离现实太远。强化学习理论需要在这两者之间取得平衡，使用能够产生有趣结果但是又不能偏离现实太远的假设。

在探索类型的问题中，通常考虑最差情况，目标是证明需要的时间是关于 $|S|, |A|, 1/(1 - \gamma)$ 的多项式（这通常比较悲观）。

在学习类型的问题中，通常会忽略掉探索的问题，考虑需要多少样本能够有效地学习一个策略，这主要依赖于以下可能的假设：
- “生成模型” 假设：假设能够从 $p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})$ 中采样，对于任意的 $\boldsymbol{s}, \boldsymbol{a}$。
- 神谕探索：对于每一个 $(\boldsymbol{s}, \boldsymbol{a})$，从 $p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})$ 中采样 $N$ 次。

很显然在现实的强化学习中是做不到的，但是这能让我们研究在探索简单的情况下的学习。

### 1.3 What we expect from RL theory?
在强化学习理论中，我们通常期望得到什么？这一点通常容易被误解：
1. 证明强化学习算法每次都会工作很好？对于当前的深度强化学习来说通常甚至无法保证收敛，因此这是不可能的。
2. 实际上，在强化学习理论中，通常在强假设下通过精确的理论来得到不精确的定性结果，例如理解误差与折扣、状态空间、迭代、样本之间一种定性的关系。于此同时我们尽可能使得这些假设足够合理，使得它们有可能在实际中成立（尽管并没有这样的保证），从而给出一个对可能结果的粗略指引，例如当状态空间变多时，应该迭代更多还是更少？
3. 当我们听到“可证明的保证”时，通常背后涉及的很多假设都不现实。 

## 2 Analysis on Model-Based RL
在这一小节中，我们将介绍一些关于基于模型的强化学习的理论分析，具体来说就是模型误差对于学习到的 $Q$ 函数以及策略的影响。

### 2.1 Assumptions and goals
在这里考虑以下假设：
1. 神谕探索：对于每一个 $(\boldsymbol{s}, \boldsymbol{a})$，从 $P(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})$ 中采样 $N$ 次。
2. 根据估计/ 真实的环境，可以完美估计一个策略对应的 $Q$ 函数（而不考虑现实中拟合 Q 迭代不收敛的问题）。

由于有上述的 "神谕" 提供所有 $(\boldsymbol{s}, \boldsymbol{a})$ 上的数据，且能够得到精确的 $Q$ 函数，我们可以得到一个很简单的”基于模型“的算法：
1. 估计动态：$$\hat{P}(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a}) = \frac{\# (\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')}{N}$$ 
2. 给定 $\pi$，使用 $\hat{P}$ 来估计 $\hat{Q}^{\pi}$。

此时考虑的就是 $\hat{P}$ 的不完美会带给 $Q$ 函数估算的误差。具体来说，我们会考虑以下几个 $Q$ 函数之间的关系：
- $Q^{\pi}$: 某个策略 $\pi$ 在真实环境 $P$ 下对应的 $Q$ 函数；
- $\hat{Q}^\pi$: 某个策略 $\pi$ 在估计的环境 $\hat{P}$ 下的 $Q$ 函数；
- $Q^\ast$ 表示真实环境 $P$ 下的最优 $Q$ 函数；
- $\hat{Q}^\ast$ 表示估计的环境 $\hat{P}$ 下的最优 $Q$ 函数；
- $Q^{\hat{\pi}}$ 表示估计的环境 $\hat{P}$ 下的最优 $Q$ 函数对应的（$\arg \max$）策略在真实环境下的 $Q$ 函数。 

简而言之，上述概念中 $Q$ 上方的符号表示 $Q$ 函数是否来源于估计的环境 $\hat{P}$，而右上角的角标则代表这个 $Q$ 函数对应于哪个处理。

而通常会考虑以下三组 $Q$ 函数之间的关系：
1. $Q^{\pi}$ 和 $\hat{Q}^{\pi}$ 的关系：也就是说，我们会考虑同样一个策略，在真实环境与估计环境中策略评估得到的 Q 函数之间的差异；
2. $Q^{\ast}$ 和 $\hat{Q}^{\ast}$ 的关系：也就是说，我们会考虑真实环境与估计环境中学到的最优 $Q$ 函数之间的差异；
3. $Q^{\ast}$ 和 $Q^{\hat{\pi}}$ 的关系：也就是说，我们会考虑真实环境中学到的最优 $Q$ 函数与估计环境中最优策略在真实环境中的 $Q$ 函数之间的差异。这实际上才是真正意义上的，在估计环境中学到的策略与最优策略在预期奖励上的差异。

而这三组 $Q$ 函数之间的差异通过以下方式来分析，以第一组关系为例，我们希望展示的是，对于一定的差异 $\epsilon > 0$，如果 $N \geq f(\epsilon, \delta)$，则以以概率 $1 - \delta$，有
$$
\|Q^{\pi} - \hat{Q}^{\pi}\|_\infty \leq \epsilon
$$
实际上对其中第一个问题的分析就可以给出分析后续问题的很好的工具。

在监督学习的理论中，有很多相关的工具来处理，可以考虑将这些工具迁移到强化学习的分析中。
### 2.2 Concentration inequalities
在监督学习中，通常会使用集中不等式（Concentration inequalities）来分析估计的误差。一个重要的不等式是霍夫丁不等式。

Theorem 1. _Hoeffding's inequality（霍夫丁不等式）_
如果 $X_1, \ldots, X_N$ 是独立同分布的随机变量，其又均值 $\mu$，记 $\bar{X}_n = n^{-1}\sum_{i = 1}^n X_i$，且 $a \leq X_i \leq b$，则对于任意 $\epsilon > 0$，有
$$
P\left(\bar{X}_n \geq \mu + \epsilon\right) \leq \exp\left(-\frac{2n\epsilon^2}{(b - a)^2}\right)
$$
类似地，有
$$
P\left(\bar{X}_n \leq \mu - \epsilon\right) \leq \exp\left(-\frac{2n\epsilon^2}{(b - a)^2}\right)
$$

注意：这个定理有很多种理解方式：
1. 这个定理描绘了通过采样估计均值时，估计和真实值的差异。这是一个很强的结果，因为出现过大误差的概率随着 $n$ 的增加而指数下降；
2. 对于一定的样本数 $n$ 以及容许的出错概率不超过 $\delta$，则可能的误差 $\epsilon$ 不超过 $\frac{b - a}{\sqrt{2n}} \sqrt{\log\frac{2}{\delta}}$，这可以利用 $\delta \leq 2\exp(-2n\epsilon^2/(b - a)^2)$ 得到，这意味着 $\epsilon$ 的上界正比于 $1/\sqrt{n}$。
3. 对于一定的 $\epsilon$ 以及 $\delta$，需要 $(b - a)^2\log(2/\delta)/2\epsilon^2$ 个样本保证出现超过 $\epsilon$ 的误差的概率小于 $\delta$。 

在强化学习理论中，我们需要考虑的是多类变量而不是均值，可以得出离散分布的集中不等式。

Theorem 2. _Concentration for discrete distribution（离散分布的集中不等式）_
如果 $X_1, \ldots, X_N$ 是独立同分布的离散随机变量，依照分布 $q$ 取值在 $\{1,\ldots,d\}$。
记 $q$ 为一个向量 $\overrightarrow{q} = [P(z = j)]_{j = 1}^d$，记通过 $X_1, \ldots, X_N$ 的样本估计 $q$ 为 $[\hat{q}]_j = \sum_{i = 1}^{N} \mathbb{I}(X_i = j)/N$，于是对于 $\forall \epsilon > 0$：
$$
P\left(\|\overrightarrow{q} - \hat{q}\|_2 \geq \frac{1}{\sqrt{N}} + \epsilon\right) \leq \exp(-N\epsilon^2)
$$
这可以推出（直接利用[[Concepts#25 范数 (Norm)|范数 (Norm)]]的关系）：
$$
P\left(\|\overrightarrow{q} - \hat{q}\|_1 \geq \sqrt{d} \left(\frac{1}{\sqrt{N}} + \epsilon\right)\right) \leq \exp(-N\epsilon^2)
$$
后一个推论可以被用于[[Concepts#2 总变差距离（Total Variation Distance）|总变差距离（Total Variation Distance）]]的估计（两个分布的总变差距离是 $1$-范数的一半）。

对于上述结论，可以得出类似霍夫丁不等式的一系列理解方式，其中一个就是对于样本数 $N$ 和容许的出错概率 $\delta$，可以解得
$$
\epsilon \leq \frac{1}{\sqrt{N}} \sqrt{\log\frac{1}{\delta}}
$$
这在原先强化学习中的意义是，以 $1 - \delta$ 的概率，有
$$
\|\hat{P}(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a}) - P(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})\|_1 \leq \sqrt{|S|} \left(\frac{1}{\sqrt{N}} + \epsilon\right) \leq \sqrt{\frac{|S|}{N}} + \sqrt{\frac{|S| \log 1/\delta}{N}} \leq c \sqrt{\frac{|S| \log 1/\delta}{N}}
$$
注意 $N$ 是仅仅估计一个 $(\boldsymbol{s}, \boldsymbol{a})$ 下的动态所需的样本数，因此需要的总样本数是 $|S||A|N$。

### 2.3 Relating $P$ and $Q$-function
接下来需要介绍一些针对于强化学习问题的引理。这些引理的共同目的是将 $\hat{P}$ 的估计误差同最终 $\hat{Q}^\pi$ 的误差关联起来。

在关联误差之前，先考虑将 $P$ 与 $Q^\pi$ 关联起来，这里有两种关联方式：

通过转移关联：
$$
Q^\pi(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{s}' \sim P(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})} \left[V^\pi(\boldsymbol{s}')\right]
$$
写作概率的形式就是
$$
Q^\pi(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \sum_{\boldsymbol{s}'} P(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a}) V^\pi(\boldsymbol{s}')
$$
进一步使用向量形式表示，则有
$$
Q^\pi = r + \gamma P V^\pi
$$
这里 $Q^\pi, r$ 是 $|S| |A|$ 的向量， $P$ 是 $|S| |A| \times |S|$ 的矩阵， $V^\pi$ 是 $|S|$ 的向量。

利用策略下的期望关联：
$$
V^\pi = \Pi Q^\pi
$$
其中 $\Pi$ 是一个 $|S| \times |S| |A|$ 的矩阵，与策略 $\pi(\boldsymbol{a}\mid \boldsymbol{s})$ 有关。

将两种关联方式结合：
结合上述利用转移与策略的两种表示方式得到
$$
Q^\pi = r + \gamma P^\pi Q^\pi
$$
其中 $P^\pi = P \Pi$。进一步化简得到（可证明 $I - \gamma P^\pi$ 是可逆的）
$$
Q^\pi = (I - \gamma P^\pi)^{-1} r
$$

类似地可以得到
$$
\hat{Q}^\pi = (I - \gamma \hat{P}^\pi)^{-1} r
$$
于是就成功地将 $P$ 与 $Q$ 函数建立起了联系，进而可以考察它们误差之间的联系。

### 2.4 Relating errors in $P$ and Q-function
这里考虑两个引理。

Lemma 1. _Simulation Lemma（模拟引理）_
$$
Q^\pi - \hat{Q}^\pi = \gamma (I - \gamma P^\pi)^{-1} (P - \hat{P}) V^\pi
$$

注意：一个直观的理解方式是，这里 $(I - \gamma P^\pi)^{-1}$ 是一个评估算子，而 $P - \hat{P}$ 是概率上的差异。一个理解方式是，想象 $V^\pi$ 是一个伪奖励，其先被动态的差异作用，再通过评估就得到了 $Q$ 函数的差异。

_Proof._
$$
\begin{aligned} Q^\pi - \hat{Q}^\pi &= Q^\pi - (I - \gamma \hat{P}^\pi)^{-1} r\\ &= (I - \gamma \hat{P}^\pi)^{-1} (I - \gamma \hat{P}^\pi) Q^\pi - (I - \gamma \hat{P}^\pi)^{-1} r\\ &= (I - \gamma \hat{P}^\pi)^{-1} (I - \gamma \hat{P}^\pi) Q^\pi - (I - \gamma \hat{P}^\pi)^{-1} (I - \gamma P^\pi) Q^\pi\\ &= (I - \gamma \hat{P}^\pi)^{-1} ((I - \gamma \hat{P}^\pi) - (I - \gamma P^\pi)) Q^\pi\\ &= \gamma (I - \gamma P^\pi)^{-1} (P^\pi - \hat{P}^\pi) V^\pi\\ &= \gamma (I - \gamma P^\pi)^{-1} (P\Pi - \hat{P}\Pi) Q^\pi\\ &= \gamma (I - \gamma P^\pi)^{-1} (P - \hat{P}) \Pi Q^\pi\\ &= \gamma (I - \gamma P^\pi)^{-1} (P - \hat{P}) V^\pi. \end{aligned}
$$

Lemma 2. 给定 $P^\pi$ 和任何的 $v \in \mathbb{R}^{|S||A|}$，有
$$
\|(I - \gamma P^\pi)^{-1} v\|_\infty \leq \|v\|_\infty / (1 - \gamma)
$$

注意：这意味着将评估算子作用在一个向量上，结果的无穷范数不会超过原向量的无穷范数除以 $1 - \gamma$。换言之对应于 "奖励" $v$ 的 $Q$ 函数每次最多增大到 $1/(1 - \gamma)$ 倍。
注意这里的 $1/(1 - \gamma)$ 来源于无穷级数的求和，这通常可以视作是某种有效时间跨度。

_Proof._
令 $w = (I - \gamma P^\pi)^{-1} v$，则有
$$
\begin{aligned} \|v\|_\infty &= \|(I - \gamma P^\pi) w\|_\infty\\ &\geq \|w\|_\infty - \gamma \|P^\pi w\|_\infty\\ &\geq \|w\|_\infty - \gamma \|w\|_\infty\\ &= (1 - \gamma) \|w\|_\infty. \end{aligned}
$$
前一个不等号利用了范数的三角不等式，后者利用了转移矩阵的无穷范数不超过 $1$。

### 2.5 Putting it all together
将上述两个引理结合，将 $(P - \hat{P}) V^\pi$ 代入第二个引理中的 $v$，于是
$$
\begin{aligned} \|Q^\pi - \hat{Q}^\pi\|_\infty &= \gamma \|(I - \gamma P^\pi)^{-1} (P - \hat{P}) V^\pi\|_\infty\\ &\leq \frac{\gamma}{1 - \gamma} \|(P - \hat{P}) V^\pi\|_\infty\\ &\leq \frac{\gamma}{1 - \gamma} \|(P - \hat{P})\|_\infty \|V^\pi\|_\infty\\ &\leq \frac{\gamma}{1 - \gamma} \left(\max_{\boldsymbol{s}, \boldsymbol{a}}\|P(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a}) - \hat{P}(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})\|_1\right) \|V^\pi(\boldsymbol{s})\|_\infty. \end{aligned}
$$
这里最后的不等号利用了 $|S||A| \times |S|$ 这一矩阵的无穷范数的定义，也就是各行绝对值和的最大值。

而同时可以约束 $\|V^\pi\|_\infty$：
$$
\sum_{t = 0}^{\infty} \gamma^t r_t \leq \frac{1}{1 - \gamma} R_{\max} = \frac{1}{1 - \gamma} R_{\max}
$$
通常假设 $R_{\max} = 1$，于是进一步化简
$$
\begin{aligned} \|Q^\pi - \hat{Q}^\pi\|_\infty &\leq \frac{\gamma}{(1 - \gamma)^2}\left(\max_{\boldsymbol{s}, \boldsymbol{a}}\|P(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a}) - \hat{P}(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})\|_1\right)\\ &\leq \frac{\gamma}{(1 - \gamma)^2} c_2 \sqrt{\frac{|S| \log |S||A|/\delta}{N}}. \end{aligned}
$$
第二个不等式利用了之前证明的集中不等式，这里有两点需要注意：
1. 对于每一个 $(\boldsymbol{s}, \boldsymbol{a})$ 集中前都有一个 $c$ 的系数，但是由于有最大化操作，可以选择其中最大系数的那个作为 $c_2$；
2. 注意这里根号内的 $\log$ 中多出了 $|S||A|$，这是为什么？注意之前的约束是对于单个 $(\boldsymbol{s}, \boldsymbol{a})$ 犯错概率不超过 $\delta$ 得到的，由于此时有 $|S||A|$ 个这样的对，只有让每一个 $(\boldsymbol{s}, \boldsymbol{a})$ 的概率不超过 $\delta/|S||A|$，再利用并集约束，才能保证整体犯错概率不超过 $\delta$。 

最终的结果就是
$$
\|Q^\pi - \hat{Q}^\pi\|_\infty \leq \frac{c_2 \gamma}{(1 - \gamma)^2} \sqrt{\frac{|S| \log 1/\delta}{N}}
$$
回顾强化学习理论的目标不是给出一个定量的理论保证，而是给出了一个定性的结果， 从这些角度来理解这个结果：
1. 误差会随着采样的增加而减小，误差正比于 $\sqrt{1/N}$；
2. 误差会随着时间跨度 $1/(1 - \gamma)$ 而二次增加。 

在给定每个 $(\boldsymbol{s}, \boldsymbol{a})$ 有 $N$ 个样本的情况下，如果需要以 $1 - \delta$ 的概率保证 $\|Q^\pi - \hat{Q}^\pi\|_\infty \leq \epsilon$，那么
$$
\epsilon = \frac{c_2\gamma}{(1 - \gamma)^2} \sqrt{\frac{|S| \log 1/\delta}{N}}
$$

### 2.6 Extensions to other errors
我们前面分析了 $\|Q^\pi - \hat{Q}^\pi\|_\infty$ 误差的约束，同样可以考虑 $\|Q^\ast - \hat{Q}^\ast\|_\infty$ 和 $\|Q^\ast - Q^{\hat{\pi}}\|_\infty$。
1. $Q^\ast$ 和 $\hat{Q}^\ast$ 的关系：利用$$|\sup_x f(x) - \sup_x g(x)| \leq \sup_x |f(x) - g(x)|$$有$$\|Q^\ast - \hat{Q}^\ast\|_\infty = \|\sup_\pi Q^\pi - \sup_\pi \hat{Q}^\pi\|_\infty \leq \sup_\pi \|Q^\pi - \hat{Q}^\pi\|_\infty$$
2. $Q^\ast$ 和 $Q^{\hat{\pi}}$ 的关系：首先利用三角不等式得到$$\|Q^\ast - Q^{\hat{\pi}}\|_\infty = \|Q^\ast - \hat{Q}^{\hat{\pi}} + \hat{Q}^{\hat{\pi}} - Q^{\hat{\pi}}\|_\infty \leq \|Q^\ast - \hat{Q}^{\hat{\pi}}\|_\infty + \|\hat{Q}^{\hat{\pi}} - Q^{\hat{\pi}}\|_\infty$$对于第一项，由于 $\hat{Q}^{\hat{\pi}}$ 含义为估计环境下的最优策略，在估计环境中的 $Q$ 函数，实际也就是 $\hat{Q}^\ast$，因此直接应用刚刚推导的结果，就得到了 $\epsilon$ 的约束。对于第二项，由于 $\|Q^\pi - \hat{Q}^\pi\|_\infty$ 的结果对于任何策略 $\pi$ 都成立，故可以替换为 $\hat{\pi}$，于是第二项可以约束到 $\epsilon$。

综上，就可以得到 $\|Q^\ast - Q^{\hat{\pi}}\|_\infty$ 的约束为 $2\epsilon$。

## 3 Analysis on Model-free RL
### 3.1 Abstract
上一部分主要考虑了在基于模型的设置，具体来说在忽略探索与假设可以学习到环境对应的最优 $Q$ 函数的情况下，$Q$ 函数的误差与样本数以及时间跨度等因素的关系。

这一部分讨论无模型相关的问题。在价值函数方法中，已经证明了对于一般的拟合 Q 迭代不存在收敛的理论保证。但接下来我们考虑一个更加理想的模型，在这个模型中可以进行一些理论的分析，得出 $Q$ 函数的误差的约束。

在真实环境中更新：
对于准确的 Q 迭代，我们进行贝尔曼更新
$
Q_{t + 1} \gets T Q_t
$
其中 $T$ 是贝尔曼算子，也就是
$
TQ = r + \gamma P \max_a Q
$

我们实际进行的更新：
但是现实中并不知道 $T$，实际的更新是一种近似的拟合 Q 迭代：
$$
\hat{Q}_{k + 1} \gets \arg\min_{\hat{Q}} \|\hat{Q} - \hat{T} \hat{Q}_k\|
$$
这里的 $\hat{T}$ 是一个是一个近似的贝尔曼算子，也就是
$$
\hat{T}Q = \hat{r} + \gamma \hat{P} \max_a Q
$$
采样得到的数据共同决定了这里的 $\hat{T}, \hat{r}, \hat{P}$，其中 $\hat{r}$ 与 $\hat{P}$ 分别估计为
$$
\hat{r}(\boldsymbol{s}, \boldsymbol{a}) = \frac{1}{N(\boldsymbol{s}, \boldsymbol{a})} \sum_{i} \delta((\boldsymbol{s}_i, \boldsymbol{a}_i) = (\boldsymbol{s}, \boldsymbol{a})) r_i, \quad \hat{P}(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a}) = \frac{N(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')}{N(\boldsymbol{s}, \boldsymbol{a})}
$$
注意这不是模型，在 Q 迭代的时候如果从已有数据中进行更新，那么实际上就在进行 $\hat{T}$ 所对应的更新。

误差研究：
1. 采样误差：在 $\hat{r}, \hat{P}$ 估计中具有采样误差， $\hat{T}$ 与 $T$ 不同；
2. j近似误差：在 $\hat{Q}_{k + 1}$ 的更新中， $\hat{Q}$ 也只是一个近似，与真正的最小值 $\hat{T} \hat{Q}_k$ 有差异。

假设：
这里还没有决定使用什么样的范数，我们已经知道使用 $2$-范数时无法得到关于收敛的保证。因此这里假设每次更新在无穷范数的意义下进行最小化，同时假设每一个 $\hat{Q}_{k + 1}$ 在无穷范数下都是有界的。

接下来尝试分析的是，当 $k \to \infty$ 时， $\hat{Q}_k$ 是否会收敛到 $Q^\ast$，如果是，那么我们希望得到一个关于 $\hat{Q}_k$ 与 $Q^\ast$ 之间的误差与其他因素的关系。

### 3.2 Sampling error
首先我们分析上述提到的采样误差，也就是 $|\hat{T} Q(\boldsymbol{s}, \boldsymbol{a}) - TQ(\boldsymbol{s}, \boldsymbol{a})|$，注意到
$$
\begin{aligned} &|\hat{T} Q(\boldsymbol{s}, \boldsymbol{a}) - TQ(\boldsymbol{s}, \boldsymbol{a})| \\ &= |\hat{r}(\boldsymbol{s}, \boldsymbol{a}) - r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \left(\mathbb{E}_{\hat{P}(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})} \left[\max_{a'} Q(\boldsymbol{s}', \boldsymbol{a}')\right] - \mathbb{E}_{P(s' \mid s, a)} \left[\max_{a'} Q(\boldsymbol{s}', \boldsymbol{a}')\right]\right)\\ &\leq |\hat{r}(\boldsymbol{s}, \boldsymbol{a}) - r(\boldsymbol{s}, \boldsymbol{a})| + \gamma \left|\mathbb{E}_{\hat{P}(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})} \left[\max_{a'} Q(\boldsymbol{s}', \boldsymbol{a}')\right] - \mathbb{E}_{P(s' \mid s, a)} \left[\max_{a'} Q(\boldsymbol{s}', \boldsymbol{a}')\right]\right|\\ \end{aligned}
$$
第一项是奖励的估计误差，使用霍夫丁不等式可以得到
$$
|\hat{r}(\boldsymbol{s}, \boldsymbol{a}) - r(\boldsymbol{s}, \boldsymbol{a})| \leq 2R_{\max}\sqrt{\frac{\log 1/\delta}{2N}}
$$
对于第二部分，转化为
$$
\begin{aligned} &\sum_{\boldsymbol{s}'} (\hat{P}(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a}) - P(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})) \max_{\boldsymbol{a}'} Q(\boldsymbol{s}', \boldsymbol{a}')\\  &\leq \sum_{\boldsymbol{s}'} |\hat{P}(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a}) - P(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})| \max_{\boldsymbol{s}',\boldsymbol{a}'} Q(\boldsymbol{s}', \boldsymbol{a}')\\ &= \|\hat{P}(\cdot \mid \boldsymbol{s}, \boldsymbol{a}) - P(\cdot \mid \boldsymbol{s}, \boldsymbol{a})\|_1 \|Q\|_\infty\\ &\leq c \|Q\|_\infty \sqrt{\frac{ \log 1/\delta}{N}}. \end{aligned}
$$
于是
$$
|\hat{T} Q(\boldsymbol{s}, \boldsymbol{a}) - TQ(\boldsymbol{s}, \boldsymbol{a})| \leq 2R_{\max}\sqrt{\frac{\log 1/\delta}{2N}} + c \|Q\|_\infty \sqrt{\frac{ \log 1/\delta}{N}}
$$
此时类似于之前的分析，为了保证总犯错概率不超过 $\delta$，可以在前后两个约束分别分配 $\delta/2$ 的犯错概率，之后再对 $(\boldsymbol{s}, \boldsymbol{a})$ 或者 $\boldsymbol{s}$ 取并集约束，于是可以得到
$$
\|\hat{T} Q - TQ\|_\infty \leq 2R_{\max} c_1 \sqrt{\frac{\log |S||A|/\delta}{2N}} + c_2 \|Q\|_\infty \sqrt{\frac{\log |S|/\delta}{N}}
$$

### 3.3 Approximation error
这一部分我们来分析近似误差。这里假设近似误差有如下的约束：
$$
\|\hat{Q}_{k + 1} - \hat{T} \hat{Q}_k\|_\infty \leq \epsilon_k
$$
由于这里的约束是对无穷范数的，因此这是一个很强的假设，在现实中的监督学习中，通常无法得到这样的保证。

接下来考虑 $\hat{Q}_k$ 与 $Q^\ast$ 之间的误差，此时利用 $Q^\ast$ 是[[Concepts#20 贝尔曼备份 (Bellman Backup)|贝尔曼备份 (Bellman Backup)]]的不动点，有：
$$
\begin{aligned} \|\hat{Q}_k - Q^\ast\|_\infty &= \|\hat{Q}_k - T \hat{Q}_{k - 1} + T \hat{Q}_{k - 1} - Q^\ast\|_\infty\\ &\leq \|\hat{Q}_k - T \hat{Q}_{k - 1}\|_\infty + \|T \hat{Q}_{k - 1} - TQ^\ast\|_\infty\\ &\leq \|\hat{Q}_k - T \hat{Q}_{k - 1}\|_\infty + \|T \hat{Q}_{k - 1} - TQ^\ast\|_\infty\\ &\leq \|\hat{Q}_k - T \hat{Q}_{k - 1}\|_\infty + \gamma \| \hat{Q}_{k - 1} - Q^\ast\|_\infty. 
\end{aligned}
$$
其中后一项利用 $T$ 是收缩算子。

展开这个递归式：
$$
\begin{aligned} \|\hat{Q}_k - Q^\ast\|_\infty &\leq \|\hat{Q}_k - T \hat{Q}_{k - 1}\|_\infty + \gamma \|\hat{Q}_{k - 1} - Q^\ast\|_\infty\\ &\leq \|\hat{Q}_k - T \hat{Q}_{k - 1}\|_\infty + \gamma \|\hat{Q}_{k - 1} - T \hat{Q}_{k - 2}\|_\infty + \gamma^2 \|\hat{Q}_{k - 2} - Q^\ast\|_\infty\\ &\leq \sum_{i = 0}^{k - 1}\gamma^i \|\hat{Q}_{k - i} - T \hat{Q}_{k - i - 1}\|_\infty + \gamma^k \|\hat{Q}_0 - Q^\ast\|_\infty. \end{aligned}
$$
一个暗示是，迭代次数越多，关于初始化"忘记"的越多。

取 $k \to \infty$ 的极限，进一步简化得到
$$
\lim_{k \to \infty} \|\hat{Q}_k - Q^\ast\|_\infty \leq \sum_{i = 0}^\infty \gamma^i \max_k \|\hat{Q}_{k + 1} - T \hat{Q}_{k}\|_\infty
$$
同样也是时间跨度越大，可能的误差也会越大。

### 3.4 Putting it all together
根据上述推导，$\hat{Q}_k$ 与 $Q^\ast$ 之间的误差可以写作
$$
\begin{aligned} \lim_{k \to \infty} \|\hat{Q}_k - Q^\ast\|_\infty &\leq \frac{1}{1 - \gamma} \max_k \|\hat{Q}_k - T \hat{Q}_{k - 1}\|_\infty,\\ \end{aligned}$$
其中
$$
\begin{aligned} \|\hat{Q}_k - T \hat{Q}_{k - 1}\|_\infty &= \frac{1}{1 - \gamma} \|\hat{Q}_k - \hat{T} \hat{Q}_{k - 1} + \hat{T} \hat{Q}_{k - 1} - T \hat{Q}_{k - 1}\|_\infty\\ &\leq \frac{1}{1 - \gamma} (\|\hat{Q}_k - \hat{T} \hat{Q}_{k - 1}\|_\infty + \|\hat{T} \hat{Q}_{k - 1} - T \hat{Q}_{k - 1}\|_\infty) \end{aligned}
$$
其中前一项对应于近似误差，后一项对应于采样误差。

由于 $\|Q\|_\infty$ 是 $O(R_{\max}/(1 - \gamma))$ 的，于是可以得到
$$
\begin{aligned} \lim_{k \to \infty} \|\hat{Q}_k - Q^\ast\|_\infty &= O\left(\frac{\|\epsilon\|_\infty}{1 - \gamma} + \frac{c_1 R_{\max}}{1 - \gamma} \sqrt{\frac{\log |S||A|/\delta}{2N}} + \frac{c_2 R_{\max}}{(1 - \gamma)^2} \sqrt{\frac{\log |S|/\delta}{N}}\right).\\ \end{aligned}
$$

注意：上述过程是一个相对粗略的分析，其中的一个暗示是误差中依然存在一个关于时间跨度是二次增大的项。

## 4 Summary
在本节中，我们
- 简单介绍了强化学习理论关心的问题，通常的假设，以及对强化学习理论的正确期望；
- 介绍了分析强化学习理论的基本工具例如集中不等式；
- 介绍了"神谕探索"假设下基于模型的强化学习的误差与时间跨度、样本数量等因素的关系；
- 介绍了无模型的强化学习的误差的主要组成和约束。