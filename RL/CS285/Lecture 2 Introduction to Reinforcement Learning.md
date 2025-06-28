# 0 Terminology & Notation
- $\boldsymbol{s}_t$：状态；
- $\boldsymbol{a}_t$：动作；
- $\boldsymbol{o}_t$：观测；
- $\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)$：策略（全部可观测）；
- $pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{o}_t)$：策略；
- $t$：时间步；
- $r(\boldsymbol{s}_t,\boldsymbol{a}_t)$：奖励函数，用于衡量一个状态与动作的好坏，这里我们使用了同时依赖于状态与动作的奖励函数，也可以定义仅依赖于状态的奖励函数 $r(\boldsymbol{s}_t)$  。

这些量之间可以利用以下概率图来表示 (在部分可观测的情况下)：
![](1-0.png)

另外需要注意的是以下两点：
- 状态之间满足马尔可夫性质，而观测之间并不满足马尔可夫性质。观测并不能完全描述状态。
- 我们通过奖励函数 $r(\boldsymbol{s}，\boldsymbol{a})$ 来给出一个 状态与动作的好坏，我们的目标不仅仅是最大化当前的 奖励，而是同样要考虑未来的奖励。 

# 1 Markov Decision Process
在强化学习中，我们通常将问题形式化为一个马尔科夫决策过程（MDP）。MDP 为 RL 提供了数学上的基础，因此我们一步步给出 MDP 的定义。

**Definition 1**. _Markov Chain（马尔可夫链）_
- $\mathcal{M} = \{\mathcal{S},\mathcal{T}\}$
- $\mathcal{S}$：状态空间  
- $\mathcal{T}$：转移算子，之所以考虑称为算子是因为令 $\mu_{t,i} = p(s_t = i)$，$\mathcal{T}_{i,j} = p(s_{t + 1} = i \mid s_t = j)$，考虑 $\boldsymbol{\mu}_t$ 是概率向量，则我们可以得到 $\boldsymbol{\mu}_{t + 1} = \mathcal{T} \boldsymbol{\mu}_t$
![](2-1.png)
**Definition 2**. _Markov Decision Process（马尔可夫决策过程）_
- $\mathcal{M} = \{\mathcal{S},\mathcal{A},\mathcal{T},r\}$  
- $\mathcal{S}$：状态空间
- $\mathcal{A}$：动作空间  
- $\mathcal{T}$：转移算子 (一个三维的张量)，如果记 $\mu_{t,j} = p(s_t = j)$，$\xi_{t,k} = p(a_t = k)$，$\mathcal{T}_{i,j,k} = p(s_{t + 1} = i \mid s_t = j，a_t = k)$，于是就可以写作 $\mu_{t + 1，i} = \sum_{j,k} \mathcal{T}_{i,j,k} \mu_{t，j} \xi_{t，k}$
- $r$：奖励函数 $r： \mathcal{S} \times \mathcal{A} \to \mathbb{R}$
![](2-2.png)
**Definition 3**. _Partially Observed MDP（部分可观测的马尔可夫决策过程）_
- $\mathcal{M} = \{\mathcal{S},\mathcal{A},\mathcal{O},\mathcal{T},\mathcal{E},r\}$
- $\mathcal{S}$：状态空间
- $\mathcal{A}$：动作空间
- $\mathcal{O}$：观测空间  
- $\mathcal{T}$：转移算子 (一个三维的张量)  
- $\mathcal{E}$：发射概率 $p(o_t \mid s_t)$
- $r$：奖励函数 $r： \mathcal{S} \times \mathcal{A} \to \mathbb{R}$
![](2-3.png)

# 2 The Goal of Reinforcement Learning

接下来我们考虑 RL 的目标。
在完全观测的 MDP，我们的目标是找到一个策略 $\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)$，其中 $\theta$ 是神经网络的参数，策略会在给定 $\boldsymbol{s}_t$ 下给出 $\boldsymbol{a}_t$，而环境 $p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t,\boldsymbol{a}_t)$ 会在给定 $\boldsymbol{s}_t$，$\boldsymbol{a}_t$ 下给出状态转移的结果 $\boldsymbol{s}_{t + 1}$。
![](2-4.png)
策略写作 $\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)$  旨在明确在时间步 $t$ 时，策略以状态 $s_t$ 为输入，输出动作 $a_t$。这种记法并不意味着策略本身随时间步而变化。相反，我们通常考虑的是平稳策略 (stationary policy)，即在所有时间步都保持不变的策略。
平稳策略具有马尔可夫性质，这意味着其决策仅依赖于当前状态，而与之前的历史无关。理论上可以证明，在特定假设条件下（例如有限的动作空间 $|\mathcal{A}| < \infty$ 和有限的初始分布支撑集），马尔可夫性和平稳性足以表达最优策略。
在深度强化学习（Deep RL）中，我们可能缺乏如此严格的理论保证。然而，采用这种与时间无关的策略不仅是强化学习理论的传统做法，而且在实践中也取得了显著的成功。

首先考虑有限时间跨度（finite horizon）的情况，我们可以基于概率链式法则写出一整条轨迹的概率：
$$
p_\theta(\tau) = p(\boldsymbol{s}_1) \prod_{t = 1}^{T} p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t,\boldsymbol{a}_t) \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t).
$$
其中我们通常记轨迹 $\tau = (\boldsymbol{s}_1,\boldsymbol{a}_1,\ldots,\boldsymbol{s}_T,\boldsymbol{a}_T)$。我们的目标不仅仅是当前奖励，还包括未来所有的奖励，于是一个自然的目标是：
$$
\theta^\ast = \arg\max_\theta \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[\sum_{t} r(\boldsymbol{s}_t,\boldsymbol{a}_t)\right]
$$
再导出无限时间跨度（infinite horizon）的情况，此时如果采用上面的方法，那么可能会出现一些问题，如奖励是无界的，常见的方法是：
- 用平均奖励，即 $\frac{1}{T} \sum_{t = 1}^{T} r(\boldsymbol{s}_t,\boldsymbol{a}_t)$
- 使用折旧因子（Discount favor），即 $\sum_{t = 1}^{\infty} \gamma^t r(\boldsymbol{s}_t,\boldsymbol{a}_t)$，其中 $0 < \gamma < 1$

其中折旧因子我们在之后介绍，这里我们先考虑平均奖励的情况。
做如下的转化：将 $\{(\boldsymbol{s}_t,\boldsymbol{a}_t)\}$ 视作是在增强空间（augmented space）上的马尔可夫链，满足转移概率为
$$
p_\theta((\boldsymbol{s}_{t + 1},\boldsymbol{a}_{t + 1}) \mid (\boldsymbol{s}_t,\boldsymbol{a}_t)) = p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t,\boldsymbol{a}_t) \pi_\theta(\boldsymbol{a}_{t + 1} \mid \boldsymbol{s}_{t + 1})
$$
与初始状态
$$
p_\theta((\boldsymbol{s}_1,\boldsymbol{a}_1)) = p(\boldsymbol{s}_1) \pi_\theta(\boldsymbol{a}_1 \mid \boldsymbol{s}_1)
$$
于是调整形式即可得到
$$
\theta^\ast = \arg\max_\theta \sum_{t = 1}^{T} \mathbb{E}_{(\boldsymbol{s}_t,\boldsymbol{a}_t) \sim p_\theta(\boldsymbol{s}_t,\boldsymbol{a}_t)} \left[r(\boldsymbol{s}_t,\boldsymbol{a}_t)\right]
$$
现在我们已经将原始问题转化为了一个与 $\theta$ 有关的马尔可夫链，我们就可以利用马尔可夫链的一些性质进行分析。这里不妨暂时记状态-动作转移算子为 $\mathcal{T}$ (暂时忽略该记号的原始含义)，那么我们可以得到
$$
\begin{pmatrix}   \boldsymbol{s}_{t + 1}\\   \boldsymbol{a}_{t + 1} \end{pmatrix} = \mathcal{T} \begin{pmatrix}   \boldsymbol{s}_t\\   \boldsymbol{a}_t \end{pmatrix},
\quad 
\begin{pmatrix}   \boldsymbol{s}_{t + k}\\   \boldsymbol{a}_{t + k} \end{pmatrix} = \mathcal{T}^k \begin{pmatrix}   \boldsymbol{s}_t\\   \boldsymbol{a}_t \end{pmatrix}
$$
根据马尔可夫链的性质，一个有限状态的马尔可夫链，如果满足不可约性和非周期性，那么它存在唯一的静止分布，并且从任何初始分布出发，最终都会收敛到这个静止分布。
现在我们考虑 $p(\boldsymbol{s}_t,\boldsymbol{a}_t)$ 是否会收敛到某个静止分布。如果由 $s_t$ 和 $a_t$ 构成的状态空间所定义的马尔可夫链满足不可约性和非周期性这两个条件，那么 $p(\boldsymbol{s}_t,\boldsymbol{a}_t)$ 就会收敛到唯一的静止分布。

在这种情况下，从长远来看，系统的状态和动作的分布会趋于稳定。因此，最终获得的奖励也会趋近于基于这个静止分布的期望奖励。根据序列极限的基本性质，如果奖励序列收敛到一个固定的值，那么该奖励序列的平均值也会收敛到这个相同的值。
所以，如果上述定义的马尔可夫链满足不可约性和非周期性，那么从这个角度来看，对奖励进行平均是有意义的，因为平均奖励最终会收敛到基于静止分布的奖励。
于是我们转化为形式
$$
\theta^\ast = \arg\max_\theta \frac{1}{T} \sum_{t = 1}^{T} \mathbb{E}_{(\boldsymbol{s}_t,\boldsymbol{a}_t) \sim p_\theta(\boldsymbol{s}_t,\boldsymbol{a}_t)} \left[r(\boldsymbol{s}_t,\boldsymbol{a}_t)\right] \rightarrow \mathbb{E}_{(\boldsymbol{s},\boldsymbol{a}) \sim p_\theta(\boldsymbol{s},\boldsymbol{a})} \left[r(\boldsymbol{s},\boldsymbol{a})\right]
$$
总结一下，我们的目标是：
- 有限时间跨度： $$\theta^\ast = \arg\max_\theta \sum_{t = 1}^{T} \mathbb{E}_{(\boldsymbol{s}_t,\boldsymbol{a}_t) \sim p_\theta(\boldsymbol{s}_t,\boldsymbol{a}_t)} \left[r(\boldsymbol{s}_t,\boldsymbol{a}_t)\right]$$
- 无限时间跨度（平均奖励）： $$\theta^\ast = \arg\max_\theta \mathbb{E}_{(\boldsymbol{s},\boldsymbol{a}) \sim p_\theta(\boldsymbol{s},\boldsymbol{a})} \left[r(\boldsymbol{s},\boldsymbol{a})\right]$$其中 $p_\theta(\boldsymbol{s}，\boldsymbol{a})$ 是[[Concepts#5 平稳分布 (Stationary Distribution)|平稳分布(Stationary Distribution)]]。 
- 无限时间跨度 (折旧奖励)： $$\theta^\ast = \arg\max_\theta \mathbb{E}_{(\boldsymbol{s}_t,\boldsymbol{a}_t) \sim p_\theta(\boldsymbol{s}_t,\boldsymbol{a}_t)} \left[\sum_{t = 1}^{\infty} \gamma^t r(\boldsymbol{s}_t,\boldsymbol{a}_t)\right]$$
值得注意的是，这里的期望起到了非常重要的作用，这让不连续的奖励能够由于 $p_\theta$ 对 $\theta$ 的连续性而被平滑化，从而使用 SGD 等方法进行优化。
  
在 RL 相关工作中，我们经常会看到[[Concepts#4 占用率测度 (Occupancy Measure)|占用率测度(Occupancy Measure)]]相关的概念 (尽管在本课程中很少出现)，在很多时候使用这一概念可以极大简化相关的公式，主要来说是为了去掉对时间的求和。这里我们给出这一概念的定义。  
  
**Definition 4.** _Occupancy Measure（占用率测度）_
占用率测度刻画了不同的状态-动作对在轨迹中出现的概率
$$
\rho^{\pi_\theta}(\boldsymbol{s},\boldsymbol{a}) = (1 - \gamma)\sum_{t = 1}^{T} \gamma^t p_\theta(\boldsymbol{s}_t = \boldsymbol{s},\boldsymbol{a}_t = \boldsymbol{a})
$$
其中 $1 - \gamma$ 用于保证概率分布的归一化性质，这里 $T$ 是轨迹的长度，也可以是 $\infty$ 。  

类似地我们可以定义状态的占用率测度：  
$$
\rho^{\pi_\theta}(\boldsymbol{s}) = (1 - \gamma) \sum_{t = 1}^{T} \gamma^t  p(\boldsymbol{s}_t = \boldsymbol{s})
$$
可以证明 
$$
\rho^{\pi_\theta}(\boldsymbol{s}) = \int_{\mathcal{A}} \rho^{\pi_\theta}(\boldsymbol{s},\boldsymbol{a}) \mathrm{d}\boldsymbol{a}
$$
  
占用率测度可以极大简化 RL 中的公式，例如对于折旧奖励的情况，我们可以将 RL 的目标写作
$$
\theta^\ast = \arg\max_\theta \frac{1}{1 - \gamma} \mathbb{E}_{(\boldsymbol{s},\boldsymbol{a}) \sim \rho^{\pi_\theta}(\boldsymbol{s},\boldsymbol{a})} \left[r(\boldsymbol{s},\boldsymbol{a})\right]
$$
# 3 The Anatomy of an RL Algorithm

在 RL 算法中，我们通常会有以下三个部分：
- Part 1：生成样本 (也就是运行策略获取轨迹)  
- Part 2：拟合一个模型 / 估计奖励等返回值  
- Part 3：更新策略
![](2-5.png)
多数 RL 算法都是基于这一框架的，但是不同的算法在这三个部分上有不同的实现，也有不同的复杂度：
- Part 1 的复杂度取决于不同的任务类型在获取数据的成本，如在真实世界中运行来获取数据的效率就远低于在模拟器中运行的效率 。
- 而 Part 2 与 Part 3 的复杂度则取决于我们使用的算法是无模型的（model-free）还是基于模型的（model-based），相对来说基于模型的算法更加复杂。

# 4 Value Functions

接下来我们引入 RL 中的一个重要概念：**价值函数（value function）**。 
在 RL 中，我们优化的目标通常是
$$
\mathbb{E}_{\tau \sim p_\theta(\tau)} \left[\sum_{t} r(\boldsymbol{s}_t,\boldsymbol{a}_t)\right]
$$
不难发现，我们可以利用一系列嵌套的期望来处理这个式子，最外层对 $\boldsymbol{s}_1 \sim p(\boldsymbol{s}_1)$ 取期望，然后对 $\boldsymbol{a}_1 \sim \pi_\theta(\boldsymbol{a}_1 \mid \boldsymbol{s}_1)$ 取期望，以此类推，写出来就是$$
\mathbb{E}_{\boldsymbol{s}_1 \sim p(\boldsymbol{s}_1)} \left[\mathbb{E}_{\boldsymbol{a}_1 \sim \pi_\theta(\boldsymbol{a}_1 \mid \boldsymbol{s}_1)} \left[r(\boldsymbol{s}_1,\boldsymbol{a}_1)+ \mathbb{E}_{\boldsymbol{s}_2 \sim p(\boldsymbol{s}_2 \mid \boldsymbol{s}_1,\boldsymbol{a}_1)} \left[\mathbb{E}_{\boldsymbol{a}_2 \sim \pi_\theta(\boldsymbol{a}_2 \mid \boldsymbol{s}_2)}\left[r(\boldsymbol{s}_2,\boldsymbol{a}_2) + \cdots\right] \right]\right]\right]
$$我们一个直观的想法是定义价值函数来表示中间复杂的嵌套部分，这样我们可以得到如下的递归式：$$
Q(\boldsymbol{s}_1,\boldsymbol{a}_1) = r(\boldsymbol{s}_1,\boldsymbol{a}_1) + \mathbb{E}_{\boldsymbol{s}_2 \sim p(\boldsymbol{s}_2 \mid \boldsymbol{s}_1,\boldsymbol{a}_1)} \left[Q(\boldsymbol{s}_2,\pi_\theta(\boldsymbol{a}_2 \mid \boldsymbol{s}_2))\right]
$$不难发现，如果我们知道了最佳策略的 $Q$，那么我们就已经得到了最佳的策略。
我们用如下方式定义价值函数：
**Definition 5**. _Q-function（Q 函数）_
$$
Q^{\pi}(\boldsymbol{s}_t,\boldsymbol{a}_t) = \sum_{t' = t}^{T} \mathbb{E}_{\pi} \left[r(\boldsymbol{s}_{t'},\boldsymbol{a}_{t'}) \mid \boldsymbol{s}_t,\boldsymbol{a}_t\right]
$$
**Definition 6**. _Value function（价值函数）_
$$
V^{\pi}(\boldsymbol{s}_t) = \sum_{t' = t}^{T} \mathbb{E}_{\pi} \left[r(\boldsymbol{s}_{t'},\boldsymbol{a}_{t'}) \mid \boldsymbol{s}_t\right]
$$
另一种方式可以写作
$$
V^{\pi}(\boldsymbol{s}_t) = \mathbb{E}_{\boldsymbol{a}_t \sim \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \left[Q^{\pi}(\boldsymbol{s}_t,\boldsymbol{a}_t)\right]
$$

在我们定义价值函数之后，我们可以把 RL 的目标写作
$$
\theta^\ast = \arg\max_\theta \mathbb{E}_{\boldsymbol{s}_1 \sim p(\boldsymbol{s}_1)} \left[V^\pi(\boldsymbol{s}_1)\right]
$$
使用价值函数的好处是什么?
- 如果我们知道 $\pi$ 与 $Q^\pi$，我们就可以更新 $\pi$。（令 $\pi'(\boldsymbol{a} \mid \boldsymbol{s}) = 1$，如果 $\boldsymbol{a} = \arg\max_{\boldsymbol{a}} Q^\pi(\boldsymbol{s},\boldsymbol{a})$）
- 用来计算梯度，从而增加更优的动作（也就是满足 $Q^\pi(\boldsymbol{s},\boldsymbol{a}) > V^\pi(\boldsymbol{s})$ 的动作）的概率。  
这分别对应了后续介绍的 **基于价值的方法（Value-based Method）** 与 **演员-评论家算法（Actor-Critic algorithm）**。

# 5 Types of Algorithms

作为 RL 的引入章节，我们简单介绍一下 RL 中的几种算法：
- Policy gradients：直接微分目标函数，从而得到梯度。
- Value-based：估计**最优策略**的价值函数或 Q 函数，从而得到策略。  
- Actor-critic：估计**当前策略**的价值函数或 Q 函数，用来更新策略。  
- Model-based RL：估计环境的模型，用这一转移模型来进行规划或更新策略。  

以下是各个算法在 RL 各部分完成的任务：
## 5.1 Model-based RL
Part 2 中，我们需要估计 $p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t,\boldsymbol{a}_t)$，也就是用数据来拟合模型，而在 Part 3 中 有多种可能：
- 直接利用模型来进行规划，进行轨迹的优化 (类似于传统最优控制的思路)，或者棋类游戏在这个模型内进行搜索（如蒙特卡洛树搜索）等。
- 将梯度反向传播到策略中（对于一条轨迹和目标函数 $J(\theta) = \sum_{t} r(\boldsymbol{s}_t)$，直接利用链式法则求策略的梯度）。
- 利用模型学习一个价值函数，用其来更新策略。用来动态规划或者生成无模型学习器（model-free learner）使用的虚拟数据。
算法的例子有：动态规划（Dyna），引导策略搜索（Guided policy search）。

## 5.2 Value function based algorithms
在 Part 2 中，我们需要估计 $V^\pi(\boldsymbol{s}_t)$ 或 $Q^\pi(\boldsymbol{s}_t,\boldsymbol{a}_t)$。在 Part 3 中，令 $\pi(\boldsymbol{s}) = \arg\max_{\boldsymbol{a}} Q^\pi(\boldsymbol{s},\boldsymbol{a})$。

![](2-6.png)
算法的例子有：Q学习（Q-learning），深度Q网络（DQN），时序差分学习（Temporal difference learning），拟合值迭代（Fitted value iteration）。

## 5.3 Direct policy gradient algorithms
Part 2 中计算 $R_\tau = \sum_{t} r(\boldsymbol{s}_t,\boldsymbol{a}_t)$， Part 3 中依据 $\theta \gets \theta + \nabla_\theta \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[R_\tau\right]$ 更新参数。
![](2-7.png))
算法的例子有：REINFORCE，自然策略梯度（Natural policy gradient），信赖域策略优化（Trust region policy optimization，TRPO），近端策略优化（Proximal policy optimization，PPO)。

## 5.4 Actor-critic algorithms
Part 2 中估计 $V^\pi(\boldsymbol{s}_t)$ 或 $Q^\pi(\boldsymbol{s}_t,\boldsymbol{a}_t)$，Part 3 中依据 $\theta \gets \theta + \nabla_\theta \mathbb{E}\left[Q(\boldsymbol{s}_t,\boldsymbol{a}_t)\right]$ 更新策略。
![](2-8.png)
算法的例子有：异步优势演员-评论家算法（Asynchronous advantage actor-critic，A3C），软演员-评论家算法（Soft actor-critic，SAC）。

# 6 Tradeoffs Between Algorithms

## 6.1 Different tradeoffs

在 RL 中，不同的算法有不同的权衡，以下是一些可能的权衡：

**采样效率**
核心问题是，算法是否是 off-policy？
- off-policy：可以在不生成新样本（使用旧样本）的情况下更新策略。 
- on-policy：使用的样本必须与最新的策略对应，即只要策略有改变，我们就必须生成新的样本。
![](2-9.png)

值得注意的是，我们可能会选择一个采样效率低的算法，这是因为我们的任务获取数据的成本可能很低，例如在模拟器中。更本质的是，采样效率只表明了需要多少数据，但不代表实际的训练时间。

**稳定性和易于使用**
算法一定收敛吗？如果收敛，会收敛到局部最优吗？每一次运行都会收敛吗？之所以会有这些问题，是因为 RL 通常并不是真正的梯度下降。具体来说：
- 价值函数拟合：在最好情况下是最小化拟合的误差（贝尔曼误差），如对于 Q 函数是 $\left|Q(\boldsymbol{s},\boldsymbol{a}) - (r(\boldsymbol{s},\boldsymbol{a}) + Q(\boldsymbol{s'},\boldsymbol{a}'))\right|$，不等同于最大化期望的奖励，在最坏的情况下甚至不在优化任何目标。
- 基于模型：训练过程中会优化模型的精度，但模型的精确不意味着策略的优异。 
- 策略梯度：是梯度下降，但通常是最低效的。  

## 6.2 Different assumptions
在不同的 RL 算法中，我们通常会基于不同的假设，如：
- 观测是全部可观测还是部分可观测？对于价值函数方法来说，通常我们会假设全部可观测。简单来说，单个观测无法确切的表示状态。考虑我们在利用 $r_t + \gamma V(\boldsymbol{s}_{t + 1})$ 时，我们的 $V(\boldsymbol{s}_{t + 1})$ 与过去的状态无关。然而在部分可观测的情况下，$V(\boldsymbol{o}_{t + 1})$ 会受到过去的状态影响。
- 环境的动态是随机的还是确定性的？
- 状态空间 $\mathcal{S}$，动作空间 $\mathcal{A}$ 是连续的还是离散的？
- 问题是片段式（episodic）还是无限时间跨度（infinite horizon）？对于策略梯度来说，我们一般假设的是片段式（即有限时间跨度）。对于一些基于模型的方法，也会有类似假设。
- 动态，奖励函数的连续性与平滑性。这是很多基于模型的 RL 算法的假设，一些价值函数方法也会有这个假设（具有这些性质的奖励函数更容易被神经网络学习 Q 函数和价值函数）。