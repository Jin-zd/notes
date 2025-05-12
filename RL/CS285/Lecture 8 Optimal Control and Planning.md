在我们正式介绍 [Model-based RL](https://zhida.zhihu.com/search?content_id=254146735&content_type=Article&match_order=1&q=Model-based+RL&zhida_source=entity) 之前, 我们先介绍一下我们能够如何利用一个模型来进行 optimal control 以及 planning.

## 1 Introduction to Model-based RL

在之前的部分, 我们介绍了 [Model-free RL](https://zhida.zhihu.com/search?content_id=254146735&content_type=Article&match_order=1&q=Model-free+RL&zhida_source=entity), 我们的优化目标是 $\theta^\ast = \arg\max_{\theta} \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[\sum_{t = 1}^T r(\boldsymbol{s}_t, \boldsymbol{a}_t)\right].\\$ 其中 $p_\theta(\tau) = p(\boldsymbol{s}_1) \prod_{t = 1}^T \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t) p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t).\\$ 在 Model-free RL 中我们假设不知道转移概率 $p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$, 并且也不尝试去估计它. 例如我们总是用采样来估计 $p_\theta(\tau)$ 下的期望, 又比如我们将 [fitted value iteration](https://zhida.zhihu.com/search?content_id=254146735&content_type=Article&match_order=1&q=fitted+value+iteration&zhida_source=entity) 转化为 [fitted Q iteration](https://zhida.zhihu.com/search?content_id=254146735&content_type=Article&match_order=1&q=fitted+Q+iteration&zhida_source=entity), 来巧妙避免了对转移概率的依赖.  

![](https://pic4.zhimg.com/v2-d932ef4857ee0e4ccbc0df485e843bff_1440w.jpg)

### 1.1 What if we know the dynamics?

在以下情形下, 我们通常知道转移概率:

1.  游戏: 例如 Atari games, chess, Go  
    
2.  容易建模的系统: 例如导航一辆车  
    
3.  仿真环境: 仿真的机器人, video games  
    

通过以下方式, 我们通常可以学习到转移概率:

1.  [System identification](https://zhida.zhihu.com/search?content_id=254146735&content_type=Article&match_order=1&q=System+identification&zhida_source=entity): 拟合一个已知模型 (例如 Guassian)的未知参数  
    
2.  Learning: 用观测得到的转移数据拟合一个通用目的的 model  
    

当我们知道转移概率时, 通常情况下我们可以让整个任务更加简单. Model-based RL 中我们会学习转移概率的模型, 然后利用这个模型来选择 action. 而在这一节中, 我们会介绍如何利用系统 dynamics 来进行决策:

-   完全知道系统 dynamics 时如何选择 action  
    
-   Optimization control, [trajectory optimization](https://zhida.zhihu.com/search?content_id=254146735&content_type=Article&match_order=1&q=trajectory+optimization&zhida_source=entity), planning 的相关方式  
    

在之后的几节 **Model-based Reinforcement Learning** 与 **Policy-based Reinforcement Learning** 中, 我们将介绍如何学习 dynamics, 如何在这基础上 learn 一个 policy.

### 2 Terminology

在 dynamics 完全已知的情况下, 我们通常可以尝试直接得到一个最优的 action sequence, 而不需要通过采样数据来近似地学习一个 policy.

首先考虑 **deterministic case**: environment 告诉 agent 当前的状态 $\boldsymbol{s}_1$, agent 选择一系列 $\boldsymbol{a}_1, \ldots, \boldsymbol{a}_T$, 即 plan, 并执行, 选择方式为 $\boldsymbol{a}_1, \ldots, \boldsymbol{a}_T = \arg\max_{\boldsymbol{a}_1, \ldots, \boldsymbol{a}_T} \sum_{t = 1}^T r(\boldsymbol{s}_t, \boldsymbol{a}_t), \text{ s.t. } \boldsymbol{s}_{t + 1} = f(\boldsymbol{s}_t, \boldsymbol{a}_t).\\$

对于 **stochastic case**, 首先表示出采取一系列 actions 得到的 trajectory 的概率 $p_\theta(\boldsymbol{s}_1, \ldots, \boldsymbol{s}_T \mid \boldsymbol{a}_1, \ldots, \boldsymbol{a}_T) = p(\boldsymbol{s}_1) \prod_{t = 1}^T p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t).\\$ 通过最大化这个轨迹上期望的 reward 为可以给出一个最优的 action sequence $\boldsymbol{a}_1, \ldots, \boldsymbol{a}_T = \arg\max_{\boldsymbol{a}_1, \ldots, \boldsymbol{a}_T} \mathbb{E}\left[\sum_{t = 1}^T r(\boldsymbol{s}_t, \boldsymbol{a}_t) \mid \boldsymbol{a}_1, \ldots, \boldsymbol{a}_T\right].\\$

不难发现这样的方式在 stochastic 的情况可能是 suboptimal 的, 因为每当我们完成一次状态转移后, 我们都会获得一些新信息, 这些信息可以帮助我们更好地选择 action. 例如当我们进行 $\boldsymbol{a}_1$ 到达 $\boldsymbol{s}_2$ 后, 我们就不再需要考虑 $p(\boldsymbol{s}_2 \mid \boldsymbol{s}_1, \boldsymbol{a}_1)$ 中的随机性, 而可以进行更好地决策. 上述的一次性选择一系列 actions 的方式是 **open-loop control**. 我们接下来给出相关的定义:

**Definition 1**. _open-loop control_

_agent 观测初始化状态, 选择一系列 action, 并且执行.  
_

![](https://pic1.zhimg.com/v2-9ec683d0ac81441713114e786bebfe98_1440w.jpg)

open loop control

**Definition 2**. _close-loop control_

_在每个时间步, agent 观察环境, 选择 action, 并且根据环境的反馈调整 action.  
_

![](https://pic3.zhimg.com/v2-fd2e03ca134e5240b996aa5b01f8373e_1440w.jpg)

close loop control

基于我们之前的简单讨论, 我们知道 close-loop control 可能会有更好的性能, 因此我们在 model-based RL 中通常使用 close-loop control. 在 close-loop 的情况下, 我们的目标可以转化为 model-free RL 的 objective, 只是环境 dynamic 变为了已知: $p(\tau) = p(\boldsymbol{s}_1) \prod_{t = 1}^T \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t) p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t).\\$ 于是 $\pi = \arg\max_\pi \mathbb{E}_{\tau \sim p(\tau)}\left[\sum_{t = 1}^T r(\boldsymbol{s}_t, \boldsymbol{a}_t)\right].\\$

## 3 Open-loop Planning

本节课的后续部分我们将主要关注 open-loop planning, 我们考虑一种问题的抽象表示: $\boldsymbol{a}_1, \ldots, \boldsymbol{a}_T = \arg\max_{\boldsymbol{a}_1, \ldots, \boldsymbol{a}_T} J(\boldsymbol{a}_1, \ldots, \boldsymbol{a}_T),\\$ 这里记 $\boldsymbol{A} = [\boldsymbol{a}_1, \ldots, \boldsymbol{a}_T]$, 于是我们的问题变为 $\boldsymbol{A} = \arg\max_{\boldsymbol{A}} J(\boldsymbol{A}).\\$ **Stochastic optimization** 是一种 blockbox optimization, 换言之我们不利用目标函数的具体形式, 而是通过采样来估计最优解. 以下是一些常见的 [stochastic optimization](https://zhida.zhihu.com/search?content_id=254146735&content_type=Article&match_order=1&q=stochastic+optimization&zhida_source=entity) 方法:

### 3.1 guess and check

一个最简单的方式是 **guess and check**:

1.  在一定分布上采样一系列的 $\boldsymbol{A}_1, \ldots, \boldsymbol{A}_N$  
    
2.  计算 $J(\boldsymbol{A}_1), \ldots, J(\boldsymbol{A}_N)$, 选择最佳的序列  
    

这一方法也被称为 "random shooting method", 尽管这看起来很简单, 但是在一些低维, short horizon 的简单任务是可行的, 而且这可以很快实现, 且很容易并行化, 在现代 GPU 上可以非常高效.

### 3.2 Cross-Entropy Method (CEM)

这一方法我们在第一步选择 sample 上更加聪明, 假设已经得到了一系列 samples, 我们希望在那些好样本所在的地方有更高的概率被采样:

重复以下过程:

1.  采样 $\boldsymbol{A}_1, \ldots, \boldsymbol{A}_N$ 从某个分布 $p(\boldsymbol{A})$  
    
2.  计算 $J(\boldsymbol{A}_1), \ldots, J(\boldsymbol{A}_N)$  
    
3.  选择最好的 $\boldsymbol{A}_{i_1}, \ldots, \boldsymbol{A}_{i_M}$, 其中 $M < N$, 通常可选 $M = N/10$  
    
4.  用这些最好的样本拟合一个新的分布 $p(\boldsymbol{A})$, 例如高斯分布  
    

![](https://pic1.zhimg.com/v2-5fdfe1b71f9dfe541a39f4d276c9de48_1440w.jpg)

CEM

这一算法有很好的理论结果, 如果我们生成足够多初始样本与足够多迭代次数, 我们可以收敛到最优解. 在实际中通常也有很好的效果, 而且也可以利用好并行化.

我们可以进一步改进的方式, 例如 **CMA-ES** (类似于 CEM + momentum)

**Summary**: 上述的 random shooting 与 CEM 都属于 blockbox 方法, 这些 blackbox 方法的好处是容易并行, 非常简单. 但问题是对于高维问题 (例如超过维度超过 30-60), 长序列问题 难以解决, 而且仅仅适用于 open-loop planning.

## 4 Discrete case: [Monte Carlo Tree Search](https://zhida.zhihu.com/search?content_id=254146735&content_type=Article&match_order=1&q=Monte+Carlo+Tree+Search&zhida_source=entity) (MCTS)

-   对于各类游戏, MCTS 通常是一个很好的选择. 要解决的核心问题是: 我们如何在避免展开整棵树的同时尽可能准确的估计状态的价值.  
    
-   通常的做法是我们先展开树到一定程度, 之后使用一系列的 rollout 来估计状态的价值.  
    
-   但由于我们没办法覆盖所有可能性, 一系列 rollouts 的返回值不等于真实的 reward, 而是具有一定的波动范围, 我们应该搜索什么样的地方? 是搜索价值高的节点? 还是搜索较少的节点? 这实际上是一个 exploration 与 exploitation 的 trade-off 问题, 我们会在 **exploration** 一节中详细讨论.  
    

这里的简单 intuition 是尽量选择价值高的节点进行搜索, 但是同时也要探索那些探索较少的节点. 基于这样的 intuition, 我们可以得到一个 **generic MCTS search**:  

重复以下过程:

1.  使用 $TreePolicy(s_1)$ 找到叶节点 $s_l$.  
    
2.  利用 $DefaultPolicy(s_l)$ 估计 $s_l$ 的价值.  
    
3.  更新 $s_1$ 到 $s_l$ 的路径上所有节点的价值.  
    

![](https://pica.zhimg.com/v2-3c07428699b744e8f08e3e6978257a4e_1440w.jpg)

这里的 DefaultPolicy 是一个随机策略. 常见的 TreePolicy 是 UCT (Upper Confidence Bound for Trees), 也就是

-   如果 $s_t$ 的 actions 有未被访问过的, 选择一个未被访问过的 action.  
    
-   否则选择 score 最高的 action (最高的子节点)  
    

这里的 score 是 $Score(s_t) = \frac{Q(s_t)}{N(s_t)} + 2C \sqrt{\frac{2\ln N(s_{t - 1})}{N(s_{t})}}.\\$ 后一个 term 是 exploration term, $Q(s_t)$ 是 $s_t$ 的平均 reward, $N(s_t)$ 是 $s_t$ 被访问的次数.

## 5 Trajectory Optimization with Derivatives

我们通常假设 dynamics 是 deterministic 的. 在这一领域我们使用略有差异的 notations, 我们通常使用 $\boldsymbol{x}_t$ 表示状态, $\boldsymbol{u}_t$ 表示 action, $f(\boldsymbol{x}_t, \boldsymbol{u}_t)$ 表示 dynamics, $c(\boldsymbol{x}_t, \boldsymbol{u}_t)$ 表示 cost. 我们的目标是 $\min_{\boldsymbol{u}_1, \ldots, \boldsymbol{u}_T} \sum_{t = 1}^T c(\boldsymbol{x}_t, \boldsymbol{u}_t), \text{ s.t. } \boldsymbol{x}_{t + 1} = f(\boldsymbol{x}_t, \boldsymbol{u}_t).\\$ 这里我们可以把 $f$ 代入目标函数, 于是我们的目标是 $\min_{\boldsymbol{u}_1, \ldots, \boldsymbol{u}_T} c(\boldsymbol{x}_1, \boldsymbol{u}_1) + c(f(\boldsymbol{x}_1, \boldsymbol{u}_1), \boldsymbol{u}_2) + \cdots + c(f(f(\ldots)\ldots), \boldsymbol{u}_T).\\$ 通常来说对于这样具有显式表达式的函数, 我们只需要利用反向传播计算梯度并进行优化即可, 也就是知道 $\frac{\text{d}f}{\text{d}\boldsymbol{x}_t}, \frac{\text{d}f}{\text{d}\boldsymbol{u}_t}, \frac{\text{d}c}{\text{d}\boldsymbol{x}_t}, \frac{\text{d}c}{\text{d}\boldsymbol{u}_t}$, 但通常使用一阶梯度下降的方法效果不佳, 我们会使用使用二阶方法, 更加具体来说, 我们的 trajectory optimization 有以下两类方法:

-   shooting method: 仅仅对 $\boldsymbol{u}_t$ 进行优化, 这样的方法会使得较早的 action 影响过大, 后面的 action 过小, 造成部分特征值过大, 部分特征值过小, 从而导致数值不稳定.  
    

![](https://picx.zhimg.com/v2-956a036578b149787e29d051c1fc8ef7_1440w.jpg)

较早的 action 影响更大

-   collocation method: 同时优化 $\boldsymbol{x}_t, \boldsymbol{u}_t$ (也可能不优化 $\boldsymbol{u}_t$, 而是由约束条件决定 action), 带有一系列约束条件, 此时相对的会有较好的数值稳定性 $\min_{\boldsymbol{u}_1, \ldots, \boldsymbol{u}_T, \boldsymbol{x}_1, \ldots, \boldsymbol{x}_T} \sum_{t = 1}^T c(\boldsymbol{x}_t, \boldsymbol{u}_t), \text{ s.t. } \boldsymbol{x}_{t + 1} = f(\boldsymbol{x}_t, \boldsymbol{u}_t).\\$ 这样的方式对于一阶方法也是可行的.  
    

这里我们考虑 shooting method, 使用二阶方法.

**Side Note**: 在 trajectory optimization 和 optimal control 中, 我们通常不使用神经网络这种相对 global 的 policy, 而是使用一些 local 的 policy, 例如 time-varying linear policy $\boldsymbol{K}_t \boldsymbol{s}_t + \boldsymbol{k}_t$.

![](https://pic1.zhimg.com/v2-964ce6ff44bfb3ccb69dd86e3266df74_1440w.jpg)

local policy

### 5.1 Linear Case: [LQR](https://zhida.zhihu.com/search?content_id=254146735&content_type=Article&match_order=1&q=LQR&zhida_source=entity)

对于环境 dynamic 是线性的情况, 我们可以使用 **LQR** (Linear Quadratic Regulator) 来解决问题. 在 linear case 中, 通常假设 dynamics 是线性的, cost 是二次的. 在不同的时间步, 我们会使用不同的 $\boldsymbol{F}_t, \boldsymbol{f}_t, \boldsymbol{C}_t, \boldsymbol{c}_t$ 来描述这一 dynamic $f(\boldsymbol{x}_t, \boldsymbol{u}_t) = \boldsymbol{F}_t \begin{bmatrix}  \boldsymbol{x}_t\\  \boldsymbol{u}_t \end{bmatrix} + \boldsymbol{f}_t.\\$ $c(\boldsymbol{x}_t, \boldsymbol{u}_t) = \frac{1}{2}\begin{bmatrix}  \boldsymbol{x}_t\\  \boldsymbol{u}_t \end{bmatrix}^T \boldsymbol{C}_t \begin{bmatrix}  \boldsymbol{x}_t\\  \boldsymbol{u}_t \end{bmatrix} + \begin{bmatrix}  \boldsymbol{x}_t\\  \boldsymbol{u}_t \end{bmatrix}^T \boldsymbol{c}_t.\\$ 其中我们假设 $\boldsymbol{C}_t$ 是对称的, 并且 $\boldsymbol{C}_t = \begin{bmatrix}  \boldsymbol{C}_{\boldsymbol{x}_t,\boldsymbol{x}_t} & \boldsymbol{C}_{\boldsymbol{x}_t,\boldsymbol{u}_t}\\  \boldsymbol{C}_{\boldsymbol{u}_t,\boldsymbol{x}_t} & \boldsymbol{C}_{\boldsymbol{u}_t,\boldsymbol{u}_t} \end{bmatrix}, \boldsymbol{c}_t = \begin{bmatrix}  \boldsymbol{c}_{\boldsymbol{x}_t}\\  \boldsymbol{c}_{\boldsymbol{u}_t}  \end{bmatrix}.\\$ 在 LQR 中, 我们使用线性的 policy $\boldsymbol{u}_t = \boldsymbol{K}_t \boldsymbol{x}_t + \boldsymbol{k}_t.\\$ 同时我们使用 $V_t(\boldsymbol{x}_t)$ 表示在第 $t$ 时间步位于 $\boldsymbol{x}_t$ 的 cost, $Q_t(\boldsymbol{x}_t, \boldsymbol{u}_t)$ 表示在第 $t$ 时间步位于 $\boldsymbol{x}_t$, 采取 $\boldsymbol{u}_t$ 的 cost.

我们考虑如下的推导过程:

**用 $\boldsymbol{x}_T$ 表示 $\boldsymbol{u}_T$:** 由于最后一个 action $\boldsymbol{u}_T$ 对之后不产生任何影响, 于是我们可以直接解出最后一个 action, 这是我们的 Base case. 我们有 $Q_T(\boldsymbol{x}_T, \boldsymbol{u}_T) = const + \frac{1}{2} \begin{bmatrix}  \boldsymbol{x}_T\\  \boldsymbol{u}_T \end{bmatrix}^T \boldsymbol{C}_T \begin{bmatrix}  \boldsymbol{x}_T\\  \boldsymbol{u}_T \end{bmatrix} + \begin{bmatrix}  \boldsymbol{x}_T\\  \boldsymbol{u}_T \end{bmatrix}^T \boldsymbol{c}_T.\\$ $\nabla_{\boldsymbol{u}_T} Q(\boldsymbol{x}_T, \boldsymbol{u}_T) = \boldsymbol{C}_{\boldsymbol{u}_T,\boldsymbol{x}_T} \boldsymbol{x}_T + \boldsymbol{C}_{\boldsymbol{u}_T,\boldsymbol{u}_T} \boldsymbol{u}_T + \boldsymbol{c}_{\boldsymbol{u}_T} = 0.\\$ 解得 $\boldsymbol{u}_T = -\boldsymbol{C}_{\boldsymbol{u}_T,\boldsymbol{u}_T}^{-1} (\boldsymbol{C}_{\boldsymbol{u}_T,\boldsymbol{x}_T} \boldsymbol{x}_T + \boldsymbol{c}_{\boldsymbol{u}_T}).\\$ 基于 $\boldsymbol{u}_T = \boldsymbol{K}_T \boldsymbol{x}_T + \boldsymbol{k}_T$, 我们可以得到 $\boldsymbol{K}_T = -\boldsymbol{C}_{\boldsymbol{u}_T,\boldsymbol{u}_T}^{-1} \boldsymbol{C}_{\boldsymbol{u}_T,\boldsymbol{x}_T}, \quad \boldsymbol{k}_T = -\boldsymbol{C}_{\boldsymbol{u}_T,\boldsymbol{u}_T}^{-1} \boldsymbol{c}_{\boldsymbol{u}_T}.\\$ 注意到之前的时间步的 cost 依赖于后续的时间步的 cost, 因此我们需要考虑 $V(\boldsymbol{x}_T)$, $V(\boldsymbol{x}_T) = const + \frac{1}{2} \begin{bmatrix}  \boldsymbol{x}_T\\  \boldsymbol{K}_T \boldsymbol{x}_T + \boldsymbol{k}_T \end{bmatrix}^T \boldsymbol{C}_T \begin{bmatrix}  \boldsymbol{x}_T\\  \boldsymbol{K}_T \boldsymbol{x}_T + \boldsymbol{k}_T \end{bmatrix} + \begin{bmatrix}  \boldsymbol{x}_T\\  \boldsymbol{K}_T \boldsymbol{x}_T + \boldsymbol{k}_T \end{bmatrix}^T \boldsymbol{c}_T.\\$ 我们记 $\boldsymbol{V}_T = \boldsymbol{C}_{\boldsymbol{x}_T,\boldsymbol{x}_T} + \boldsymbol{C}_{\boldsymbol{x}_T,\boldsymbol{u}_T} \boldsymbol{K}_T + \boldsymbol{K}_T^T \boldsymbol{C}_{\boldsymbol{u}_T,\boldsymbol{x}_T} + \boldsymbol{K}_T^T \boldsymbol{C}_{\boldsymbol{u}_T,\boldsymbol{u}_T} \boldsymbol{K}_T,\\$ $\boldsymbol{v}_T = \boldsymbol{C}_{\boldsymbol{x}_T,\boldsymbol{u}_T} \boldsymbol{k}_T + \boldsymbol{K}_T^T \boldsymbol{C}_{\boldsymbol{u}_T,\boldsymbol{u}_T} \boldsymbol{k}_T + \boldsymbol{c}_{\boldsymbol{x}_T} + \boldsymbol{K}_T^T \boldsymbol{c}_{\boldsymbol{u}_T}.\\$ 就可以形式地写出 $V(\boldsymbol{x}_T) = const + \frac{1}{2} \boldsymbol{x}_T^T \boldsymbol{V}_{T} \boldsymbol{x}_T + \boldsymbol{x}_T^T \boldsymbol{v}_T\\$

**用 $\boldsymbol{x}_{T - 1}$ 表示 $\boldsymbol{u}_{T - 1}$:** 接下来我们考虑如何解出 $\boldsymbol{u}_{T - 1}$, 用 $\boldsymbol{x}_{T - 1}$ 表示. 但值得注意的是 $\boldsymbol{u}_{T - 1}$ 还会影响 $\boldsymbol{x}_T$, 故 $f(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1}) = \boldsymbol{F}_{T - 1} \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix} + \boldsymbol{f}_{T - 1}.\\$ $Q(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1}) = const + \frac{1}{2} \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix}^T \boldsymbol{C}_{T - 1} \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix} + \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix}^T \boldsymbol{c}_{T - 1} + V(f(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1})).\\$ 于是 $V(\boldsymbol{x}_T) = const + \frac{1}{2} \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix}^T \boldsymbol{F}_{T - 1}^T \boldsymbol{V}_T \boldsymbol{F}_{T - 1} \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix} + \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix}^T \boldsymbol{F}_{T - 1}^T \boldsymbol{V}_T \boldsymbol{f}_{T - 1} + \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix}^T \boldsymbol{F}_{T - 1}^T \boldsymbol{v}_T \\$ 代回 $Q(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1})$ 中, 我们可以得到 $Q(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1}) = const + \frac{1}{2} \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix}^T \boldsymbol{Q}_{T - 1} \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix} + \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix}^T \boldsymbol{q}_{T - 1}\\$ 其中 $\boldsymbol{Q}_{T - 1} = \boldsymbol{C}_{T - 1} + \boldsymbol{F}_{T - 1}^T \boldsymbol{V}_T \boldsymbol{F}_{T - 1}, \quad \boldsymbol{q}_{T - 1} = \boldsymbol{c}_{T - 1} + \boldsymbol{F}_{T - 1}^T \boldsymbol{V}_T \boldsymbol{f}_{T - 1} + \boldsymbol{F}_{T - 1}^T \boldsymbol{v}_T.\\$ 类似地求导, 我们可以得到 $\nabla_{\boldsymbol{u}_{T - 1}} Q(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1}) = \boldsymbol{Q}_{\boldsymbol{u}_{T - 1}, \boldsymbol{x}_{T - 1}} \boldsymbol{x}_{T - 1} + \boldsymbol{Q}_{\boldsymbol{u}_{T - 1}, \boldsymbol{u}_{T - 1}} \boldsymbol{u}_{T - 1} + \boldsymbol{q}_{\boldsymbol{u}_{T - 1}}^T = 0.\\$ 其中 $\boldsymbol{K}_{T - 1} = -\boldsymbol{Q}_{\boldsymbol{u}_{T - 1}, \boldsymbol{u}_{T - 1}}^{-1} \boldsymbol{Q}_{\boldsymbol{u}_{T - 1}, \boldsymbol{x}_{T - 1}}, \boldsymbol{k}_{T - 1} = -\boldsymbol{Q}_{\boldsymbol{u}_{T - 1}, \boldsymbol{u}_{T - 1}}^{-1} \boldsymbol{q}_{\boldsymbol{u}_{T - 1}}.\\$ 从 $t = T - 1$ 到 $t = 1$ 的部分推导过程是完全类似的, 于是整理我们可以得到算法的 **backward pass**:

从 $T$ 开始, 递推计算 $\boldsymbol{K}_t, \boldsymbol{k}_t$.

-   $\boldsymbol{Q}_t = \boldsymbol{C}_t + \boldsymbol{F}_t^T \boldsymbol{V}_{t + 1} \boldsymbol{F}_t$  
    
-   $\boldsymbol{q}_t = \boldsymbol{c}_t + \boldsymbol{F}_t^T \boldsymbol{V}_{t + 1} \boldsymbol{f}_t + \boldsymbol{F}_t^T \boldsymbol{v}_{t + 1}$.  
    
-   $Q(\boldsymbol{x}_t, \boldsymbol{u}_t) = const + \frac{1}{2} \begin{bmatrix}  \boldsymbol{x}_t\\  \boldsymbol{u}_t  \end{bmatrix}^T \boldsymbol{Q}_t \begin{bmatrix}  \boldsymbol{x}_t\\  \boldsymbol{u}_t  \end{bmatrix} + \begin{bmatrix}  \boldsymbol{x}_t\\  \boldsymbol{u}_t  \end{bmatrix}^T \boldsymbol{q}_t$.  
    
-   $\boldsymbol{u}_t \gets \arg\min_{\boldsymbol{u}_t} Q(\boldsymbol{x}_t, \boldsymbol{u}_t)$.  
    
-   $\boldsymbol{K}_t = -\boldsymbol{Q}_{\boldsymbol{u}_t,\boldsymbol{u}_t}^{-1} \boldsymbol{Q}_{\boldsymbol{u}_t,\boldsymbol{x}_t}$  
    
-   $\boldsymbol{k}_t = -\boldsymbol{Q}_{\boldsymbol{u}_t,\boldsymbol{u}_t}^{-1} \boldsymbol{q}_{\boldsymbol{u}_t}$.  
    
-   $\boldsymbol{V}_t = \boldsymbol{Q}_{\boldsymbol{x}_t,\boldsymbol{x}_t} + \boldsymbol{Q}_{\boldsymbol{x}_t,\boldsymbol{u}_t} \boldsymbol{K}_t + \boldsymbol{K}_t^T \boldsymbol{Q}_{\boldsymbol{u}_t,\boldsymbol{x}_t} + \boldsymbol{K}_t^T \boldsymbol{Q}_{\boldsymbol{u}_t,\boldsymbol{u}_t} \boldsymbol{K}_t$  
    
-   $\boldsymbol{v}_t = \boldsymbol{q}_{\boldsymbol{x}_t} + \boldsymbol{Q}_{\boldsymbol{x}_t,\boldsymbol{u}_t} \boldsymbol{k}_t + \boldsymbol{K}_t^T \boldsymbol{Q}_{\boldsymbol{u}_t} + \boldsymbol{K}_t^T \boldsymbol{Q}_{\boldsymbol{u}_t,\boldsymbol{u}_t} \boldsymbol{k}_t$.  
    
-   $V(\boldsymbol{x}_t) = const + \frac{1}{2} \boldsymbol{x}_t^T \boldsymbol{V}_t \boldsymbol{x}_t + \boldsymbol{x}_t^T \boldsymbol{v}_t$.  
    

而在 **forward pass** 中, 我们从 $\boldsymbol{x}_1$ 开始, 递推计算 $\boldsymbol{x}_t, \boldsymbol{u}_t$:

-   $\boldsymbol{u}_t = \boldsymbol{K}_t \boldsymbol{x}_t + \boldsymbol{k}_t$.  
    
-   $\boldsymbol{x}_{t + 1} = \boldsymbol{F}_t \begin{bmatrix}  \boldsymbol{x}_t\\  \boldsymbol{u}_t  \end{bmatrix} + \boldsymbol{f}_t$.  
    

完整的 LQR 算法包括 backward pass 与 forward pass.

-   在 backward pass, 我们计算 $\boldsymbol{K}_t, \boldsymbol{k}_t, \boldsymbol{V}_t, \boldsymbol{v}_t$, 也就是仅仅学到了 $\boldsymbol{u}_t$ 的计算方式, 没有学到 $\boldsymbol{u}_t, \boldsymbol{x}_t$.  
    
-   在 forward pass 中, 我们使用 $\boldsymbol{u}_t$ 的计算方式 来迭代地计算 $\boldsymbol{u}_t, \boldsymbol{x}_t$.  
    

## 6 LQR for Stochastic and Nonlinear Systems

### 6.1 Stochastic dynamics

这里我们考虑 stochastic dynamics, 我们的 dynamics 由如下高斯分布给出: $\boldsymbol{x}_{t + 1} \sim \mathcal{N}(f(\boldsymbol{x}_t, \boldsymbol{u}_t), \boldsymbol{\Sigma}_t).\\$ 其中 $f(\boldsymbol{x}_t, \boldsymbol{u}_t) = \boldsymbol{F}_t \begin{bmatrix}  \boldsymbol{x}_t\\  \boldsymbol{u}_t \end{bmatrix} + \boldsymbol{f}_t\\$ 引入这里的随机性后, 不难发现原先的推导过程中 $Q(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1})$ 的形式会略有变化为 $Q(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1}) = const + \frac{1}{2} \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix}^T \boldsymbol{C}_{T - 1} \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix} + \begin{bmatrix}  \boldsymbol{x}_{T - 1}\\  \boldsymbol{u}_{T - 1} \end{bmatrix}^T \boldsymbol{c}_{T - 1} + \mathbb{E}\left[V(f(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1}))\right].\\$ 我们考虑其中的 $\mathbb{E}\left[V(f(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1}))\right]$, 这一形式实际上有相同的结果, 这是因为 $\boldsymbol{x}_{T}^\top \boldsymbol{V}_T \boldsymbol{x}_{T} = \sum_{i,j} \boldsymbol{V}_{T,ij} x_{T,i} x_{T,j}. \\$ 而利用二阶矩的定义, 可知 $\mathbb{E}\left[x_{T,i} x_{T,j}\right] = \boldsymbol{\Sigma}_{T, i,j} + \mu_{T,i} \mu_{T,j}\\$ 其中 $\mu_{T}, \boldsymbol{\Sigma}_{T}$ 分别是 $\boldsymbol{x}_T$ 的均值与方差. 从而可以得到 $\mathbb{E}\left[\sum_{i,j} \boldsymbol{V}_{T,i,j} x_{T,i} x_{T,j}\right] = \sum_{i,j} \boldsymbol{V}_{T,ij} \left(\boldsymbol{\Sigma}_{T,ij} + \mu_{T,i} \mu_{T,j}\right) = \text{tr}(\boldsymbol{V}_T \boldsymbol{\Sigma}_T) + \boldsymbol{\mu}_T^\top \boldsymbol{V}_T \boldsymbol{\mu}_T.\\$ 由于 $\boldsymbol{\Sigma}_T$ 与 $f(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1})$ 无关, 会被吸收进入常数项, 于是我们可以得到 $\mathbb{E}\left[V(f(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1}))\right] = const + f(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1})^T \boldsymbol{V}_T f(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1}) + f(\boldsymbol{x}_{T - 1}, \boldsymbol{u}_{T - 1})^T \boldsymbol{v}_T.\\$ 对于其余的 $t$ 可以同理推导, 因此我们最终可以得到与确定的情况相同的 $\boldsymbol{K}_t, \boldsymbol{k}_t$. 但需要注意的是虽然 $\boldsymbol{u}_t$ 的计算公式并不会改变, 但是我们的 $\boldsymbol{x}_t$ 会改变, 自然 $\boldsymbol{u}_t$ 会发生变化.

### 6.2 Nonliner case: [DDP](https://zhida.zhihu.com/search?content_id=254146735&content_type=Article&match_order=1&q=DDP&zhida_source=entity)/ [iterative LQR](https://zhida.zhihu.com/search?content_id=254146735&content_type=Article&match_order=1&q=iterative+LQR&zhida_source=entity)

在 nonlinear case 中, 我们利用 linear-quadratic system 来近似 nonlinear system. 具体来说, 我们使用 Taylor 展开来做到这一点: 我们分别把 dynamic 与 cost 方程分别展开到一阶与二阶 $f(\boldsymbol{x}_t, \boldsymbol{u}_t) \approx f(\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t) + \nabla_{\boldsymbol{x}_t, \boldsymbol{u}_t} f(\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t) \begin{bmatrix}  \boldsymbol{x}_t - \hat{\boldsymbol{x}}_t\\  \boldsymbol{u}_t - \hat{\boldsymbol{u}}_t \end{bmatrix} \\$ $c(\boldsymbol{x}_t, \boldsymbol{u}_t) \approx c(\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t) + \nabla_{\boldsymbol{x}_t, \boldsymbol{u}_t} c(\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t) \begin{bmatrix}  \boldsymbol{x}_t - \hat{\boldsymbol{x}}_t\\  \boldsymbol{u}_t - \hat{\boldsymbol{u}}_t \end{bmatrix} + \frac{1}{2} \begin{bmatrix}  \boldsymbol{x}_t - \hat{\boldsymbol{x}}_t\\  \boldsymbol{u}_t - \hat{\boldsymbol{u}}_t \end{bmatrix}^T \nabla_{\boldsymbol{x}_t, \boldsymbol{u}_t}^2 c(\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t) \begin{bmatrix}  \boldsymbol{x}_t - \hat{\boldsymbol{x}}_t\\  \boldsymbol{u}_t - \hat{\boldsymbol{u}}_t \end{bmatrix}.\\$ 一个可能让人困惑的点是, 这里的 $\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t$ 是上一次迭代时的结果. 为了记号简便, 我们记 $\delta \boldsymbol{x}_t = \boldsymbol{x}_t - \hat{\boldsymbol{x}}_t, \delta \boldsymbol{u}_t = \boldsymbol{u}_t - \hat{\boldsymbol{u}}_t$, 我们记近似后的系统 dynamics 与 reward 函数分别为 $\bar{f}(\delta \boldsymbol{x}_t, \delta \boldsymbol{u}_t) = \boldsymbol{F}_t \begin{bmatrix}  \delta \boldsymbol{x}_t\\  \delta \boldsymbol{u}_t \end{bmatrix} + \boldsymbol{f}_t.\\$ $\bar{c}(\delta \boldsymbol{x}_t, \delta \boldsymbol{u}_t) = \frac{1}{2} \begin{bmatrix}  \delta \boldsymbol{x}_t\\  \delta \boldsymbol{u}_t \end{bmatrix}^T \boldsymbol{C}_t \begin{bmatrix}  \delta \boldsymbol{x}_t\\  \delta \boldsymbol{u}_t \end{bmatrix} + \begin{bmatrix}  \delta \boldsymbol{x}_t\\  \delta \boldsymbol{u}_t \end{bmatrix}^T \boldsymbol{c}_t.\\$ 我们可以在 $\bar{f}, \bar{c}, \delta \boldsymbol{x}_t, \delta \boldsymbol{u}_t$ 上使用 LQR.

于是我们考虑如下的 **iterative LQR** 算法:

重复如下直至收敛:

-   $\boldsymbol{F}_t = \nabla_{\boldsymbol{x}_t, \boldsymbol{u}_t} f(\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t)$  
    
-   $\boldsymbol{C}_t = \nabla_{\boldsymbol{x}_t, \boldsymbol{u}_t}^2 c(\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t)$  
    
-   $\boldsymbol{c}_t = \nabla_{\boldsymbol{x}_t, \boldsymbol{u}_t} c(\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t)$  
    
-   利用 $\delta \boldsymbol{x}_t, \delta \boldsymbol{u}_t$ 运行 LQR 的反向过程  
    
-   利用 $\boldsymbol{u}_t = \boldsymbol{K}_t \delta \boldsymbol{x}_t + \boldsymbol{k}_t + \hat{\boldsymbol{u}}_t$ 运行 LQR 的前向过程  
    
-   更新 $\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t$ 基于前向过程的结果  
    

值得注意的是, 原先的 LQR 并不是迭代的, 只需要单次的 backward pass 与 forward pass 即可解得最优解. 而 iterative LQR 是迭代的, 每次迭代都会更新 $\hat{\boldsymbol{x}}_t, \hat{\boldsymbol{u}}_t$, 就如同在梯度下降中的更新参数一样迭代地进行.

### Comparison with Newton's method

将这种方式与 Newton's method 对比:

**Newton's method** 计算函数最小值的方式是:

重复如下直至收敛:

-   $\boldsymbol{g} = \nabla_{\boldsymbol{x}} g(\hat{\boldsymbol{x}})$  
    
-   $\boldsymbol{H} = \nabla_{\boldsymbol{x}}^2 g(\hat{\boldsymbol{x}})$  
    
-   $\hat{\boldsymbol{x}} \gets \arg\min_{\boldsymbol{x}} \frac{1}{2} (\boldsymbol{x} - \hat{\boldsymbol{x}})^T \boldsymbol{H} (\boldsymbol{x} - \hat{\boldsymbol{x}}) + \boldsymbol{g}^T (\boldsymbol{x} - \hat{\boldsymbol{x}})$  
    

事实上 Iterative LQR 的基本思想与 Newton's method 是一样的, 而 iLQR 是 Newton's method 的一种近似. 如果要去掉这种 "近似" 得到 full Newton's method, 对应的算法是 DDP (differential dynamic programming), 在这个算法中, 我们考虑二阶的 dynamic $f(\boldsymbol{x}, \boldsymbol{u}) \approx f(\hat{\boldsymbol{x}}, \hat{\boldsymbol{u}}) + \nabla_{\boldsymbol{x}, \boldsymbol{u}} f(\hat{\boldsymbol{x}}, \hat{\boldsymbol{u}}) \begin{bmatrix}  \boldsymbol{x} - \hat{\boldsymbol{x}}\\  \boldsymbol{u} - \hat{\boldsymbol{u}} \end{bmatrix} + \frac{1}{2} \begin{bmatrix}  \boldsymbol{x} - \hat{\boldsymbol{x}}\\  \boldsymbol{u} - \hat{\boldsymbol{u}} \end{bmatrix}^T \nabla_{\boldsymbol{x}, \boldsymbol{u}}^2 f(\hat{\boldsymbol{x}}, \hat{\boldsymbol{u}}) \begin{bmatrix}  \boldsymbol{x} - \hat{\boldsymbol{x}}\\  \boldsymbol{u} - \hat{\boldsymbol{u}} \end{bmatrix}.\\$ 值得注意的是, 这里的 $\nabla_{\boldsymbol{x}, \boldsymbol{u}}^2 f(\hat{\boldsymbol{x}}, \hat{\boldsymbol{u}})$ 会是三阶张量, 因此上述表达式可能有一些不够明确.

**Some improvement:** 对于 Newton's method 中 $\hat{\boldsymbol{x}} \gets \arg\min_{\boldsymbol{x}} \frac{1}{2} (\boldsymbol{x} - \hat{\boldsymbol{x}})^T \boldsymbol{H} (\boldsymbol{x} - \hat{\boldsymbol{x}}) + \boldsymbol{g}^T (\boldsymbol{x} - \hat{\boldsymbol{x}})\\$ 事实上这在实际中有一些问题, 即使我们使用了二阶近似, 我们的更新依然存在着一定的 trust region, 我们有可能会离开这部分区域.

我们可以将这种思想应用到 iLQR 中, 修改 forward pass, 使得 $\boldsymbol{u}_t = \boldsymbol{K}_t (\boldsymbol{x}_t - \hat{\boldsymbol{x}}_t) + \alpha \boldsymbol{k}_t + \hat{\boldsymbol{u}}_t\\$ 当我们减小 $\alpha$ 时, 我们的 action 会更加接近原来的 action. 一个极端的例子是 $\alpha = 0$, 此时由于初始状态改变为 $0$, 我们计算得到的 $\boldsymbol{u}_1$ 也不变, 以此类推. 在实际中, 我们可以从大到小搜索 $\alpha$, 直到我们达到了一些 improvement.

![](https://pica.zhimg.com/v2-cc96eed701b70997d82762d85e87ce46_1440w.jpg)

* * *