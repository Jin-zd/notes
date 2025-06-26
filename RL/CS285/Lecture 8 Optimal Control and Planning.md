在正式介绍基于模型的强化学习（Model-based RL）之前，我们先介绍一下如何利用一个模型来进行最优控制以及规划。
## 1 Introduction to Model-based RL
在之前的部分，我们介绍了无模型的强化学习，我们的优化目标是 
$$
\theta^\ast = \arg\max_{\theta} \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[\sum_{t = 1}^T r(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]
$$
其中 
$$
p_\theta(\tau) = p(\boldsymbol{s}_1) \prod_{t = 1}^T \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t) p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)
$$
在无模型的强化学习中假设不知道转移概率 $p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$，并且也不尝试去估计它。例如总是用采样来估计 $p_\theta(\tau)$ 下的期望，又比如我们将拟合价值迭代转化为拟合 Q 迭代，来巧妙避免了对转移概率的依赖。
![](8-1.png)

在以下情形下，通常知道转移概率：
1. 游戏：例如 Atari games，chess，Go  
2. 容易建模的系统：例如导航一辆车  
3. 仿真环境：仿真的机器人，视频游戏 

通过以下方式，通常可以学习到转移概率：
1. System identification：拟合一个已知模型（例如 Guassian）的未知参数  
2. Learning：用观测得到的转移数据拟合一个通用目的的模型

当知道转移概率时，通常情况下我们可以让整个任务更加简单。基于模型的强化学习中我们会学习转移概率的模型，然后利用这个模型来选择动作。而在这一节中，我们会介绍如何利用系统动态（dynamics）来进行决策：
- 完全知道系统动态时如何选择动作
- 优化控制（Optimization control）、轨迹优化（trajectory optimization）、规划（planning） 的相关方式  

在之后的几节 [[Lecture 9 Model-Based Reinforcement Learning]] 与 [[Lecture 9 Model-Based Reinforcement Learning]] 中，我们将介绍如何学习动态，如何在这基础上学习一个策略。

## 2 Terminology
在动态完全已知的情况下，我们通常可以尝试直接得到一个最优的动作序列，而不需要通过采样数据来近似地学习一个策略。

首先考虑确定性情况，环境告诉智能体当前的状态 $\boldsymbol{s}_1$，智能体选择一系列动作 $\boldsymbol{a}_1, \ldots, \boldsymbol{a}_T$，即计划，并执行，选择方式为：
$$
\boldsymbol{a}_1, \ldots, \boldsymbol{a}_T = \arg\max_{\boldsymbol{a}_1, \ldots, \boldsymbol{a}_T} \sum_{t = 1}^T r(\boldsymbol{s}_t, \boldsymbol{a}_t), \quad \text{ s.t. } \boldsymbol{s}_{t + 1} = f(\boldsymbol{s}_t, \boldsymbol{a}_t)
$$
对于随机情况，首先表示出采取一系列动作得到的轨迹的概率： 
$$
p_\theta(\boldsymbol{s}_1, \ldots, \boldsymbol{s}_T \mid \boldsymbol{a}_1, \ldots, \boldsymbol{a}_T) = p(\boldsymbol{s}_1) \prod_{t = 1}^T p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)
$$
通过最大化这个轨迹上期望的奖励为可以给出一个最优的动作序列：
$$
\boldsymbol{a}_1, \ldots, \boldsymbol{a}_T = \arg\max_{\boldsymbol{a}_1, \ldots, \boldsymbol{a}_T} \mathbb{E}\left[\sum_{t = 1}^T r(\boldsymbol{s}_t, \boldsymbol{a}_t) \mid \boldsymbol{a}_1, \ldots, \boldsymbol{a}_T\right]
$$

不难发现这样的方式在随机的情况可能是次优的的，因为每当完成一次状态转移后，我们都会获得一些新信息，这些信息可以帮助我们更好地选择动作。例如当进行 $\boldsymbol{a}_1$ 到达 $\boldsymbol{s}_2$ 后，我们就不再需要考虑 $p(\boldsymbol{s}_2 \mid \boldsymbol{s}_1, \boldsymbol{a}_1)$ 中的随机性，而可以进行更好地决策。上述的一次性选择一系列动作的方式是 **开环控制（open-loop control）**。接下来给出相关的定义：

**Definition 1**. _open-loop control（开环控制）_
智能体观测初始化状态，选择一系列动作并且执行。 
![](8-2.png)

**Definition 2**. _close-loop control（闭环控制）_
在每个时间步，智能体观察环境，选择动作，并且根据环境的反馈调整动作。
![](8-3.png)


基于之前的简单讨论，我们知道闭环控制可能会有更好的性能，因此在基于模型的强化学习中通常使用闭环控制。在闭环的情况下，我们的目标可以转化为无模型的强化学习的目标，只是环境 动态变为了已知：
$$
p(\tau) = p(\boldsymbol{s}_1) \prod_{t = 1}^T \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t) p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)
$$
于是
$$
\pi = \arg\max_\pi \mathbb{E}_{\tau \sim p(\tau)}\left[\sum_{t = 1}^T r(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]
$$

## 3 Open-loop Planning

本节课的后续部分将主要关注开环规划（open-loop planning），考虑一种问题的抽象表示：
$$
\boldsymbol{a}_1, \ldots, \boldsymbol{a}_T = \arg\max_{\boldsymbol{a}_1, \ldots, \boldsymbol{a}_T} J(\boldsymbol{a}_1, \ldots, \boldsymbol{a}_T)
$$
这里记 $\boldsymbol{A} = [\boldsymbol{a}_1, \ldots, \boldsymbol{a}_T]$，于是问题变为 
$$
\boldsymbol{A} = \arg\max_{\boldsymbol{A}} J(\boldsymbol{A})
$$
随机优化（Stochastic Optimization）是一种黑箱优化，换言之我们不利用目标函数的具体形式，而是通过采样来估计最优解。以下是一些常见的随机优化方法：

**Guess and Check：**
一个最简单的方式是猜测并验证：
1. 在一定分布上采样一系列的 $\boldsymbol{A}_1, \ldots, \boldsymbol{A}_N$；
2. 计算 $J(\boldsymbol{A}_1), \ldots, J(\boldsymbol{A}_N)$，选择最佳的序列。  
这一方法也被称为随机打靶法（random shooting method），尽管这看起来很简单，但是在一些低维，短时间跨度的简单任务上是可行的，而且这可以很快实现，且很容易并行化，在现代 GPU 上可以非常高效。

**Cross-Entropy Method（CEM）：**
这一方法在第一步选择采样上更加聪明，假设已经得到了一系列样本，我们希望在那些好样本所在的地方有更高的概率被采样。
重复以下过程：
1. 采样 $\boldsymbol{A}_1, \ldots, \boldsymbol{A}_N$ 从某个分布 $p(\boldsymbol{A})$；
2. 计算 $J(\boldsymbol{A}_1), \ldots, J(\boldsymbol{A}_N)$；
3. 选择最好的 $\boldsymbol{A}_{i_1}, \ldots, \boldsymbol{A}_{i_M}$，其中 $M < N$，通常可选 $M = N/10$；
4. 用这些最好的样本拟合一个新的分布 $p(\boldsymbol{A})$，例如高斯分布。 
![](8-4.png)

这一算法有很好的理论结果，如果生成足够多初始样本与足够多迭代次数，我们可以收敛到最优解。在实际中通常也有很好的效果，而且也可以利用好并行化。
可以进一步改进的方式。例如 CMA-ES（类似于 CEM + Momentum）

上述的随机打靶与 CEM 都属于黑盒方法，这些黑盒方法的好处是容易并行，非常简单。但问题是对于高维问题（例如超过维度超过 30-60）长序列问题难以解决，而且仅仅适用于开环规划。

## 4 Discrete case: Monte Carlo Tree Search (MCTS)
对于各类游戏，蒙特卡洛树搜索（MCTS）通常是一个很好的选择。要解决的核心问题是：如何在避免展开整棵树的同时尽可能准确的估计状态的价值。
通常的做法是先展开树到一定程度，之后使用一系列的推演来估计状态的价值。
但由于我们没办法覆盖所有可能性，一系列推演的返回值不等于真实的奖励，而是具有一定的波动范围，我们应该搜索什么样的地方？是搜索价值高的节点，还是搜索较少的节点？这实际上是一个探索与利用折中的问题，我们会在之后详细讨论。

这里的简单直觉是尽量选择价值高的节点进行搜索，但是同时也要探索那些探索较少的节点。基于这样的直觉，可以得到一个通用的 MCTS。  

重复以下过程:
1. 使用 $TreePolicy(s_1)$ 找到叶节点 $s_l$；
2. 利用 $DefaultPolicy(s_l)$ 估计 $s_l$ 的价值；
3. 更新 $s_1$ 到 $s_l$ 的路径上所有节点的价值。 
![](8-5.png)

这里的 DefaultPolicy 是一个随机策略。常见的 TreePolicy 是树置信上限（Upper Confidence Bound for Trees，UCT），也就是
- 如果 $s_t$ 的动作有未被访问过的，选择一个未被访问过的动作； 
- 否则选择分数最高的动作（最高的子节点）。  
这里的分数是：
$$
Score(s_t) = \frac{Q(s_t)}{N(s_t)} + 2C \sqrt{\frac{2\ln N(s_{t - 1})}{N(s_{t})}}
$$
后一个项是探索项，$Q(s_t)$ 是 $s_t$ 的平均奖励，$N(s_t)$ 是 $s_t$ 被访问的次数。

## 5 Trajectory Optimization with Derivatives
通常假设动态是确定的。在这一领域使用略有差异的表示法。我们通常使用 $\boldsymbol{x}_t$ 表示状态，$\boldsymbol{u}_t$ 表示 动作，$f(\boldsymbol{x}_t, \boldsymbol{u}_t)$ 表示动态，$c(\boldsymbol{x}_t, \boldsymbol{u}_t)$ 表示代价。目标是 
$$
\min_{\boldsymbol{u}_1, \ldots, \boldsymbol{u}_T} \sum_{t = 1}^T c(\boldsymbol{x}_t, \boldsymbol{u}_t), \quad \text{ s.t. } \boldsymbol{x}_{t + 1} = f(\boldsymbol{x}_t, \boldsymbol{u}_t)
$$
这里可以把 $f$ 代入目标函数，于是目标是 
$$
\min_{\boldsymbol{u}_1, \ldots, \boldsymbol{u}_T} c(\boldsymbol{x}_1, \boldsymbol{u}_1) + c(f(\boldsymbol{x}_1, \boldsymbol{u}_1), \boldsymbol{u}_2) + \cdots + c(f(f(\ldots)\ldots), \boldsymbol{u}_T)
$$
通常来说对于这样具有显式表达式的函数，我们只需要利用反向传播计算梯度并进行优化即可，也就是知道 $\frac{\text{d}f}{\text{d}\boldsymbol{x}_t}, \frac{\text{d}f}{\text{d}\boldsymbol{u}_t}, \frac{\text{d}c}{\text{d}\boldsymbol{x}_t}, \frac{\text{d}c}{\text{d}\boldsymbol{u}_t}$，但通常使用一阶梯度下降的方法效果不佳，我们会使用使用二阶方法，更加具体来说，我们的轨迹优化有以下两类方法：
- 打靶法（Shooting method）：仅仅对 $\boldsymbol{u}_t$ 进行优化。这样的方法会使得较早的动作影响过大，后面的动作过小，造成部分特征值过大，部分特征值过小，从而导致数值不稳定。 
![](8-6.png)
- 搭配方法（Collocation method）：同时优化 $\boldsymbol{x}_t, \boldsymbol{u}_t$（也可能不优化 $\boldsymbol{u}_t$, 而是由约束条件决定动作），带有一系列约束条件，此时相对的会有较好的数值稳定性$$\min_{\boldsymbol{u}_1, \ldots, \boldsymbol{u}_T, \boldsymbol{x}_1, \ldots, \boldsymbol{x}_T} \sum_{t = 1}^T c(\boldsymbol{x}_t, \boldsymbol{u}_t), \text{ s.t. } \boldsymbol{x}_{t + 1} = f(\boldsymbol{x}_t, \boldsymbol{u}_t)$$ 这样的方式对于一阶方法也是可行的。

这里我们考虑打靶法，使用二阶方法。
在轨迹优化和优化控制中，我们通常不使用神经网络这种相对全局的策略，而是使用一些局部的 策略，例如时变线性策略（Time-varying linear policy）$\boldsymbol{K}_t \boldsymbol{s}_t + \boldsymbol{k}_t$。
![](8-7.png)

**Linear Case：LQR**
对于环境动态是线性的情况，我们可以使用线性二次型调节器（Linear Quadratic Regulator，LQR）来解决问题。在线性情况中，通常假设动态是线性的，代价是二次的。在不同的时间步，我们会使用不同的 $\boldsymbol{F}_t, \boldsymbol{f}_t, \boldsymbol{C}_t, \boldsymbol{c}_t$ 来描述这一动态：
$$
f(\boldsymbol{x}_t, \boldsymbol{u}_t) = \boldsymbol{F}_t \begin{bmatrix}  \boldsymbol{x}_t\\  \boldsymbol{u}_t \end{bmatrix} + \boldsymbol{f}_t
$$
$$
c(\boldsymbol{x}_t, \boldsymbol{u}_t) = \frac{1}{2}\begin{bmatrix}  \boldsymbol{x}_t\\  \boldsymbol{u}_t \end{bmatrix}^T \boldsymbol{C}_t \begin{bmatrix}  \boldsymbol{x}_t\\  \boldsymbol{u}_t \end{bmatrix} + \begin{bmatrix}  \boldsymbol{x}_t\\  \boldsymbol{u}_t \end{bmatrix}^T \boldsymbol{c}_t
$$
其中我们假设 $\boldsymbol{C}_t$ 是对称的，并且 
$$
\boldsymbol{C}_t = \begin{bmatrix}  \boldsymbol{C}_{\boldsymbol{x}_t,\boldsymbol{x}_t} & \boldsymbol{C}_{\boldsymbol{x}_t,\boldsymbol{u}_t}\\  \boldsymbol{C}_{\boldsymbol{u}_t,\boldsymbol{x}_t} & \boldsymbol{C}_{\boldsymbol{u}_t,\boldsymbol{u}_t} \end{bmatrix}, \boldsymbol{c}_t = \begin{bmatrix}  \boldsymbol{c}_{\boldsymbol{x}_t}\\  \boldsymbol{c}_{\boldsymbol{u}_t}  \end{bmatrix}
$$
在 LQR 中，我们使用线性的策略
$$
\boldsymbol{u}_t = \boldsymbol{K}_t \boldsymbol{x}_t + \boldsymbol{k}_t
$$
同时我们使用 $V_t(\boldsymbol{x}_t)$ 表示在第 $t$ 时间步位于 $\boldsymbol{x}_t$ 的代价，$Q_t(\boldsymbol{x}_t, \boldsymbol{u}_t)$ 表示在第 $t$ 时间步位于 $\boldsymbol{x}_t$，采取 $\boldsymbol{u}_t$ 的代价。

考虑如下的推导过程:
**用 $\boldsymbol{x}_T$ 表示 $\boldsymbol{u}_T$：** 由于最后一个动作 $\boldsymbol{u}_T$ 对之后不产生任何影响，于是可以直接解出最后一个动作，这是基本情况，有 
$$
Q_T(\boldsymbol{x}_T, \boldsymbol{u}_T) = const + \frac{1}{2} \begin{bmatrix}  \boldsymbol{x}_T\\  \boldsymbol{u}_T \end{bmatrix}^T \boldsymbol{C}_T \begin{bmatrix}  \boldsymbol{x}_T\\  \boldsymbol{u}_T \end{bmatrix} + \begin{bmatrix}  \boldsymbol{x}_T\\  \boldsymbol{u}_T \end{bmatrix}^T \boldsymbol{c}_T
$$
$$\nabla_{\boldsymbol{u}_T} Q(\boldsymbol{x}_T, \boldsymbol{u}_T) = \boldsymbol{C}_{\boldsymbol{u}_T,\boldsymbol{x}_T} \boldsymbol{x}_T + \boldsymbol{C}_{\boldsymbol{u}_T,\boldsymbol{u}_T} \boldsymbol{u}_T + \boldsymbol{c}_{\boldsymbol{u}_T} = 0
$$
解得 
$$
\boldsymbol{u}_T = -\boldsymbol{C}_{\boldsymbol{u}_T,\boldsymbol{u}_T}^{-1} (\boldsymbol{C}_{\boldsymbol{u}_T,\boldsymbol{x}_T} \boldsymbol{x}_T + \boldsymbol{c}_{\boldsymbol{u}_T})
$$
基于 $\boldsymbol{u}_T = \boldsymbol{K}_T \boldsymbol{x}_T + \boldsymbol{k}_T$，可以得到 
$$
\boldsymbol{K}_T = -\boldsymbol{C}_{\boldsymbol{u}_T,\boldsymbol{u}_T}^{-1} \boldsymbol{C}_{\boldsymbol{u}_T,\boldsymbol{x}_T}, \quad \boldsymbol{k}_T = -\boldsymbol{C}_{\boldsymbol{u}_T,\boldsymbol{u}_T}^{-1} \boldsymbol{c}_{\boldsymbol{u}_T}
$$
注意到之前的时间步的代价依赖于后续的时间步的代价，因此需要考虑 $V(\boldsymbol{x}_T)$：
$$
V(\boldsymbol{x}_T) = const + \frac{1}{2} \begin{bmatrix}  \boldsymbol{x}_T\\  \boldsymbol{K}_T \boldsymbol{x}_T + \boldsymbol{k}_T \end{bmatrix}^T \boldsymbol{C}_T \begin{bmatrix}  \boldsymbol{x}_T\\  \boldsymbol{K}_T \boldsymbol{x}_T + \boldsymbol{k}_T \end{bmatrix} + \begin{bmatrix}  \boldsymbol{x}_T\\  \boldsymbol{K}_T \boldsymbol{x}_T + \boldsymbol{k}_T \end{bmatrix}^T \boldsymbol{c}_T
$$
记 
$$
\boldsymbol{V}_T = \boldsymbol{C}_{\boldsymbol{x}_T,\boldsymbol{x}_T} + \boldsymbol{C}_{\boldsymbol{x}_T,\boldsymbol{u}_T} \boldsymbol{K}_T + \boldsymbol{K}_T^T \boldsymbol{C}_{\boldsymbol{u}_T,\boldsymbol{x}_T} + \boldsymbol{K}_T^T \boldsymbol{C}_{\boldsymbol{u}_T,\boldsymbol{u}_T} \boldsymbol{K}_T
$$
$$
\boldsymbol{v}_T = \boldsymbol{C}_{\boldsymbol{x}_T,\boldsymbol{u}_T} \boldsymbol{k}_T + \boldsymbol{K}_T^T \boldsymbol{C}_{\boldsymbol{u}_T,\boldsymbol{u}_T} \boldsymbol{k}_T + \boldsymbol{c}_{\boldsymbol{x}_T} + \boldsymbol{K}_T^T \boldsymbol{c}_{\boldsymbol{u}_T}
$$
就可以形式地写出 
$$
V(\boldsymbol{x}_T) = const + \frac{1}{2} \boldsymbol{x}_T^T \boldsymbol{V}_{T} \boldsymbol{x}_T + \boldsymbol{x}_T^T \boldsymbol{v}_T
$$
**用 $\boldsymbol{x}_{T - 1}$ 表示 $\boldsymbol{u}_{T - 1}$：** 接下来考虑如何解出 $\boldsymbol{u}_{T - 1}$，用 $\boldsymbol{x}_{T - 1}$ 表示。但值得注意的是 $\boldsymbol{u}_{T - 1}$ 还会影响 $\boldsymbol{x}_T$，故 
$$
f(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1}) = \boldsymbol{F}_{T - 1} \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix} + \boldsymbol{f}_{T - 1}
$$
$$
Q(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1}) = const + \frac{1}{2} \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix}^T \boldsymbol{C}_{T - 1} \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix} + \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix}^T \boldsymbol{c}_{T - 1} + V(f(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1}))
$$
于是 
$$
V(\boldsymbol{x}_T) = const + \frac{1}{2} \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix}^T \boldsymbol{F}_{T - 1}^T \boldsymbol{V}_T \boldsymbol{F}_{T - 1} \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix} + \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix}^T \boldsymbol{F}_{T - 1}^T \boldsymbol{V}_T \boldsymbol{f}_{T - 1} + \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix}^T \boldsymbol{F}_{T - 1}^T \boldsymbol{v}_T
$$
代回 $Q(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1})$ 中，可以得到 
$$
Q(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1}) = const + \frac{1}{2} \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix}^T \boldsymbol{Q}_{T - 1} \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix} + \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix}^T \boldsymbol{q}_{T - 1}
$$
其中 
$$
\boldsymbol{Q}_{T - 1} = \boldsymbol{C}_{T - 1} + \boldsymbol{F}_{T - 1}^T \boldsymbol{V}_T \boldsymbol{F}_{T - 1}, \quad \boldsymbol{q}_{T - 1} = \boldsymbol{c}_{T - 1} + \boldsymbol{F}_{T - 1}^T \boldsymbol{V}_T \boldsymbol{f}_{T - 1} + \boldsymbol{F}_{T - 1}^T \boldsymbol{v}_T
$$
类似地求导，可以得到 
$$
\nabla_{\boldsymbol{u}_{T - 1}} Q(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1}) = \boldsymbol{Q}_{\boldsymbol{u}_{T - 1}, \boldsymbol{x}_{T - 1}} \boldsymbol{x}_{T - 1} + \boldsymbol{Q}_{\boldsymbol{u}_{T - 1}, \boldsymbol{u}_{T - 1}} \boldsymbol{u}_{T - 1} + \boldsymbol{q}_{\boldsymbol{u}_{T - 1}}^T = 0
$$
其中 
$$
\boldsymbol{K}_{T - 1} = -\boldsymbol{Q}_{\boldsymbol{u}_{T - 1}, \boldsymbol{u}_{T - 1}}^{-1} \boldsymbol{Q}_{\boldsymbol{u}_{T - 1}, \boldsymbol{x}_{T - 1}}, \boldsymbol{k}_{T - 1} = -\boldsymbol{Q}_{\boldsymbol{u}_{T - 1}, \boldsymbol{u}_{T - 1}}^{-1} \boldsymbol{q}_{\boldsymbol{u}_{T - 1}}
$$
从 $t = T - 1$ 到 $t = 1$ 的部分推导过程是完全类似的，于是整理可以得到算法的反向传播：
从 $T$ 开始, 递推计算 $\boldsymbol{K}_t, \boldsymbol{k}_t$。
- $\boldsymbol{Q}_t = \boldsymbol{C}_t + \boldsymbol{F}_t^T \boldsymbol{V}_{t + 1} \boldsymbol{F}_t$；
- $\boldsymbol{q}_t = \boldsymbol{c}_t + \boldsymbol{F}_t^T \boldsymbol{V}_{t + 1} \boldsymbol{f}_t + \boldsymbol{F}_t^T \boldsymbol{v}_{t + 1}$；
- $Q(\boldsymbol{x}_t, \boldsymbol{u}_t) = const + \frac{1}{2} \begin{bmatrix}  \boldsymbol{x}_t\\  \boldsymbol{u}_t  \end{bmatrix}^T \boldsymbol{Q}_t \begin{bmatrix}  \boldsymbol{x}_t\\  \boldsymbol{u}_t  \end{bmatrix} + \begin{bmatrix}  \boldsymbol{x}_t\\  \boldsymbol{u}_t  \end{bmatrix}^T \boldsymbol{q}_t$；
- $\boldsymbol{u}_t \gets \arg\min_{\boldsymbol{u}_t} Q(\boldsymbol{x}_t, \boldsymbol{u}_t)$；
- $\boldsymbol{K}_t = -\boldsymbol{Q}_{\boldsymbol{u}_t,\boldsymbol{u}_t}^{-1} \boldsymbol{Q}_{\boldsymbol{u}_t,\boldsymbol{x}_t}$；
- $\boldsymbol{k}_t = -\boldsymbol{Q}_{\boldsymbol{u}_t,\boldsymbol{u}_t}^{-1} \boldsymbol{q}_{\boldsymbol{u}_t}$；
- $\boldsymbol{V}_t = \boldsymbol{Q}_{\boldsymbol{x}_t,\boldsymbol{x}_t} + \boldsymbol{Q}_{\boldsymbol{x}_t,\boldsymbol{u}_t} \boldsymbol{K}_t + \boldsymbol{K}_t^T \boldsymbol{Q}_{\boldsymbol{u}_t,\boldsymbol{x}_t} + \boldsymbol{K}_t^T \boldsymbol{Q}_{\boldsymbol{u}_t,\boldsymbol{u}_t} \boldsymbol{K}_t$；
- $\boldsymbol{v}_t = \boldsymbol{q}_{\boldsymbol{x}_t} + \boldsymbol{Q}_{\boldsymbol{x}_t,\boldsymbol{u}_t} \boldsymbol{k}_t + \boldsymbol{K}_t^T \boldsymbol{Q}_{\boldsymbol{u}_t} + \boldsymbol{K}_t^T \boldsymbol{Q}_{\boldsymbol{u}_t,\boldsymbol{u}_t} \boldsymbol{k}_t$；
- $V(\boldsymbol{x}_t) = const + \frac{1}{2} \boldsymbol{x}_t^T \boldsymbol{V}_t \boldsymbol{x}_t + \boldsymbol{x}_t^T \boldsymbol{v}_t$。

而在前向传播中，我们从 $\boldsymbol{x}_1$ 开始，递推计算 $\boldsymbol{x}_t, \boldsymbol{u}_t$：
- $\boldsymbol{u}_t = \boldsymbol{K}_t \boldsymbol{x}_t + \boldsymbol{k}_t$；
- $\boldsymbol{x}_{t + 1} = \boldsymbol{F}_t \begin{bmatrix}  \boldsymbol{x}_t\\  \boldsymbol{u}_t  \end{bmatrix} + \boldsymbol{f}_t$。

完整的 LQR 算法包括反向传播和前向传播：
- 在反向传播中，计算 $\boldsymbol{K}_t, \boldsymbol{k}_t, \boldsymbol{V}_t, \boldsymbol{v}_t$，也就是仅仅学到了 $\boldsymbol{u}_t$ 的计算方式，没有学到 $\boldsymbol{u}_t, \boldsymbol{x}_t$。
- 在前向传播中，使用 $\boldsymbol{u}_t$ 的计算方式 来迭代地计算 $\boldsymbol{u}_t, \boldsymbol{x}_t$。

## 6 LQR for Stochastic and Nonlinear Systems
### 6.1 Stochastic dynamics
这里我们考虑随机动态，我们的动态由如下高斯分布给出：
$$
\boldsymbol{x}_{t + 1} \sim \mathcal{N}(f(\boldsymbol{x}_t, \boldsymbol{u}_t), \boldsymbol{\Sigma}_t)
$$
其中 
$$
f(\boldsymbol{x}_t, \boldsymbol{u}_t) = \boldsymbol{F}_t \begin{bmatrix}  \boldsymbol{x}_t\\  \boldsymbol{u}_t \end{bmatrix} + \boldsymbol{f}_t
$$
引入这里的随机性后，不难发现原先的推导过程中 $Q(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1})$ 的形式会略有变化为 
$$
Q(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1}) = const + \frac{1}{2} \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix}^T \boldsymbol{C}_{T - 1} \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix} + \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix}^T \boldsymbol{c}_{T - 1} + \mathbb{E}\left[V(f(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1}))\right]
$$
考虑其中的 $\mathbb{E}\left[V(f(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1}))\right]$，这一形式实际上有相同的结果，这是因为 
$$
\boldsymbol{x}_{T}^\top \boldsymbol{V}_T \boldsymbol{x}_{T} = \sum_{i,j} \boldsymbol{V}_{T,ij} x_{T,i} x_{T,j}
$$
而利用[[Concepts#15 二阶矩（Second Moment）|二阶矩（Sencond Moment）]]的定义，可知 
$$
\mathbb{E}\left[x_{T,i} x_{T,j}\right] = \boldsymbol{\Sigma}_{T, i,j} + \mu_{T,i} \mu_{T,j}
$$
其中 $\mu_{T}, \boldsymbol{\Sigma}_{T}$ 分别是 $\boldsymbol{x}_T$ 的均值与方差。从而可以得到 
$$
\mathbb{E}\left[\sum_{i,j} \boldsymbol{V}_{T,i,j} x_{T,i} x_{T,j}\right] = \sum_{i,j} \boldsymbol{V}_{T,ij} \left(\boldsymbol{\Sigma}_{T,ij} + \mu_{T,i} \mu_{T,j}\right) = \text{tr}(\boldsymbol{V}_T \boldsymbol{\Sigma}_T) + \boldsymbol{\mu}_T^\top \boldsymbol{V}_T \boldsymbol{\mu}_T
$$
由于 $\boldsymbol{\Sigma}_T$ 与 $f(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1})$ 无关，会被吸收进入常数项。于是我们可以得到 
$$
\mathbb{E}\left[V(f(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1}))\right] = const + f(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1})^T \boldsymbol{V}_T f(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1}) + f(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1})^T \boldsymbol{v}_T
$$
对于其余的 $t$ 可以同理推导。因此最终可以得到与确定的情况相同的 $\boldsymbol{K}_t, \boldsymbol{k}_t$。但需要注意的是虽然 $\boldsymbol{u}_t$ 的计算公式并不会改变，但是 $\boldsymbol{x}_t$ 会改变，自然 $\boldsymbol{u}_t$ 会发生变化。

### 6.2 Nonliner case：DDP/ iterative LQR
在非线性情况中，我们利用线性 - 二次系统来近似非线性系统。具体来说，我们使用泰勒展开来做到这一点：分别把动态与代价方程分别展开到一阶与二阶：
$$
f(\boldsymbol{x}_t, \boldsymbol{u}_t) \approx f(\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t) + \nabla_{\boldsymbol{x}_t, \boldsymbol{u}_t} f(\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t) \begin{bmatrix}  \boldsymbol{x}_t - \hat{\boldsymbol{x}}_t\\  \boldsymbol{u}_t - \hat{\boldsymbol{u}}_t \end{bmatrix} 
$$
$$
c(\boldsymbol{x}_t, \boldsymbol{u}_t) \approx c(\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t) + \nabla_{\boldsymbol{x}_t, \boldsymbol{u}_t} c(\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t) \begin{bmatrix}  \boldsymbol{x}_t - \hat{\boldsymbol{x}}_t\\  \boldsymbol{u}_t - \hat{\boldsymbol{u}}_t \end{bmatrix} + \frac{1}{2} \begin{bmatrix}  \boldsymbol{x}_t - \hat{\boldsymbol{x}}_t\\  \boldsymbol{u}_t - \hat{\boldsymbol{u}}_t \end{bmatrix}^T \nabla_{\boldsymbol{x}_t, \boldsymbol{u}_t}^2 c(\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t) \begin{bmatrix}  \boldsymbol{x}_t - \hat{\boldsymbol{x}}_t\\  \boldsymbol{u}_t - \hat{\boldsymbol{u}}_t \end{bmatrix}
$$
一个可能让人困惑的点是，这里的 $\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t$ 是上一次迭代时的结果。为了记号简便，记 $\delta \boldsymbol{x}_t = \boldsymbol{x}_t - \hat{\boldsymbol{x}}_t, \delta \boldsymbol{u}_t = \boldsymbol{u}_t - \hat{\boldsymbol{u}}_t$，记近似后的系统动态与奖励函数分别为 
$$
\bar{f}(\delta \boldsymbol{x}_t, \delta \boldsymbol{u}_t) = \boldsymbol{F}_t \begin{bmatrix}  \delta \boldsymbol{x}_t\\  \delta \boldsymbol{u}_t \end{bmatrix} + \boldsymbol{f}_t
$$
$$
\bar{c}(\delta \boldsymbol{x}_t, \delta \boldsymbol{u}_t) = \frac{1}{2} \begin{bmatrix}  \delta \boldsymbol{x}_t\\  \delta \boldsymbol{u}_t \end{bmatrix}^T \boldsymbol{C}_t \begin{bmatrix}  \delta \boldsymbol{x}_t\\  \delta \boldsymbol{u}_t \end{bmatrix} + \begin{bmatrix}  \delta \boldsymbol{x}_t\\  \delta \boldsymbol{u}_t \end{bmatrix}^T \boldsymbol{c}_t
$$
我们可以在 $\bar{f}, \bar{c}, \delta \boldsymbol{x}_t, \delta \boldsymbol{u}_t$ 上使用 LQR。

于是考虑如下的迭代 LQR 算法：
重复如下直至收敛：
- $\boldsymbol{F}_t = \nabla_{\boldsymbol{x}_t, \boldsymbol{u}_t} f(\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t)$；
- $\boldsymbol{C}_t = \nabla_{\boldsymbol{x}_t, \boldsymbol{u}_t}^2 c(\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t)$；
- $\boldsymbol{c}_t = \nabla_{\boldsymbol{x}_t, \boldsymbol{u}_t} c(\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t)$；
- 利用 $\delta \boldsymbol{x}_t, \delta \boldsymbol{u}_t$ 运行 LQR 的反向过程；  
- 利用 $\boldsymbol{u}_t = \boldsymbol{K}_t \delta \boldsymbol{x}_t + \boldsymbol{k}_t + \hat{\boldsymbol{u}}_t$ 运行 LQR 的前向过程；
- 更新 $\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t$ 基于前向过程的结果。  

值得注意的是，原先的 LQR 并不是迭代的，只需要单次的反向传播与前向传播即可解得最优解。而迭代 LQR 是迭代的，每次迭代都会更新 $\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t$，就如同在梯度下降中的更新参数一样迭代地进行。

### 6.3 Comparison with Newton's method

将这种方式与牛顿法（Newton's method）对比：
Newton's method 计算函数最小值的方式是：
重复如下直至收敛：
- $\boldsymbol{g} = \nabla_{\boldsymbol{x}} g(\hat{\boldsymbol{x}})$；
- $\boldsymbol{H} = \nabla_{\boldsymbol{x}}^2 g(\hat{\boldsymbol{x}})$；
- $\hat{\boldsymbol{x}} \gets \arg\min_{\boldsymbol{x}} \frac{1}{2} (\boldsymbol{x} - \hat{\boldsymbol{x}})^T \boldsymbol{H} (\boldsymbol{x} - \hat{\boldsymbol{x}}) + \boldsymbol{g}^T (\boldsymbol{x} - \hat{\boldsymbol{x}})$。

事实上迭代 LQR 的基本思想与牛顿法是一样的，而迭代 LQR 是牛顿法的一种近似。如果要去掉这种“近似”得到完全牛顿法，对应的算法是微分动态规划（Differential Dynamic Programming，DDP），在这个算法中，考虑二阶的动态
$$
f(\boldsymbol{x}, \boldsymbol{u}) \approx f(\hat{\boldsymbol{x}}, \hat{\boldsymbol{u}}) + \nabla_{\boldsymbol{x}, \boldsymbol{u}} f(\hat{\boldsymbol{x}}, \hat{\boldsymbol{u}}) \begin{bmatrix}  \boldsymbol{x} - \hat{\boldsymbol{x}}\\  \boldsymbol{u} - \hat{\boldsymbol{u}} \end{bmatrix} + \frac{1}{2} \begin{bmatrix}  \boldsymbol{x} - \hat{\boldsymbol{x}}\\  \boldsymbol{u} - \hat{\boldsymbol{u}} \end{bmatrix}^T \nabla_{\boldsymbol{x}, \boldsymbol{u}}^2 f(\hat{\boldsymbol{x}}, \hat{\boldsymbol{u}}) \begin{bmatrix}  \boldsymbol{x} - \hat{\boldsymbol{x}}\\  \boldsymbol{u} - \hat{\boldsymbol{u}} \end{bmatrix}
$$
值得注意的是，这里的 $\nabla_{\boldsymbol{x}, \boldsymbol{u}}^2 f(\hat{\boldsymbol{x}}, \hat{\boldsymbol{u}})$ 会是三阶张量，因此上述表达式可能有一些不够明确。

对于牛顿法中 
$$
\hat{\boldsymbol{x}} \gets \arg\min_{\boldsymbol{x}} \frac{1}{2} (\boldsymbol{x} - \hat{\boldsymbol{x}})^T \boldsymbol{H} (\boldsymbol{x} - \hat{\boldsymbol{x}}) + \boldsymbol{g}^T (\boldsymbol{x} - \hat{\boldsymbol{x}})
$$
事实上这在实际中有一些问题，即使使用了二阶近似，我们的更新依然存在着一定的信赖域，有可能会离开这部分区域。

我们可以将这种思想应用到迭代 LQR 中，修改前向传播，使得 
$$
\boldsymbol{u}_t = \boldsymbol{K}_t (\boldsymbol{x}_t - \hat{\boldsymbol{x}}_t) + \alpha \boldsymbol{k}_t + \hat{\boldsymbol{u}}_t
$$
当减小 $\alpha$ 时，动作会更加接近原来的动作。一个极端的例子是 $\alpha = 0$，此时由于初始状态改变为 $0$，计算得到的 $\boldsymbol{u}_1$ 也不变。以此类推，在实际中，可以从大到小搜索 $\alpha$，直到达到了一些提升。
![](8-8.png)