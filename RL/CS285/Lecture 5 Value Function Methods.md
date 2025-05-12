# 1 Introduction to Value Function Methods

在 [actor-critic 算法](https://zhida.zhihu.com/search?content_id=253801844&content_type=Article&match_order=1&q=actor-critic+%E7%AE%97%E6%B3%95&zhida_source=entity)中, 我们引入了 value function, 它们其实已经告诉了我们在每个 state 下应该采取什么 action: 
$$
A^\pi(\boldsymbol{s},\boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \gamma\mathbb{E}_{\boldsymbol{s}' \sim p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})} \left[V^\pi(\boldsymbol{s}')\right] - V^\pi(\boldsymbol{s}) \approx r(\boldsymbol{s}, \boldsymbol{a}) + \gamma V^\pi(\boldsymbol{s}') - V^\pi(\boldsymbol{s})
$$ 
或是 
$$
A^\pi(\boldsymbol{s}, \boldsymbol{a}) = Q^\pi(\boldsymbol{s}, \boldsymbol{a}) - V^\pi(\boldsymbol{s}) = Q^\pi(\boldsymbol{s}, \boldsymbol{a}) - \mathbb{E}_{\boldsymbol{a} \sim \pi(\boldsymbol{a} \mid \boldsymbol{s})} \left[Q^\pi(\boldsymbol{s}, \boldsymbol{a})\right]
$$
这给了我们一个思路: 我们能否完全不使用 policy gradient 而仅仅使用 value function 呢? 这便是 **value-based methods** 的基本思想了, 在 value-based methods 中没有显式的 policy, 仅仅学习一定形式的 value function. 回顾以下记号:
-   $A^\pi(\boldsymbol{s}, \boldsymbol{a})$ 表示在 state $\boldsymbol{s}$ 下采取 action $\boldsymbol{a}$ 相较于平均 action 的优势.  
-   $\arg\max_{\boldsymbol{a}} A^\pi(\boldsymbol{s}, \boldsymbol{a})$ 表示在 state $\boldsymbol{s}$ 下依照策略 $\pi$ 可以采取的最优 action.  

很显然如果 $A^\pi$ 准确, 那么采取 $\arg\max_{\boldsymbol{a}} A^\pi(\boldsymbol{s}, \boldsymbol{a})$ 至少和按照 $\boldsymbol{a} \sim \pi$ 采取的 action 一样好. 于是我们可以使用方式更新 policy: 
$$
\pi'(\boldsymbol{a}_t \mid \boldsymbol{s}_t) = \begin{cases}  1, & \text{if } \boldsymbol{a}_t = \arg\max_{\boldsymbol{a}_t} A^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t),\\  0, & \text{otherwise}. \end{cases}
$$ 
上述过程我们展示了如何利用 value function 来更新 policy, 如果在此之后进行 **policy evaluation** (我们在 actor-critic 中讨论过这一概念), 那么我们就可以重复上述过程, 不断得到更优的 policy 的 value function, 就得到了 **Policy iteration** 算法.

在通用 RL 框架中, 在 Part 2 中我们会 fit $A^\pi$ (或者 $Q^\pi$, $V^\pi$). 在 Part 3 中我们使用 $\pi \gets \pi'$,

![](https://pic3.zhimg.com/v2-4fde59627dfb0c7046b4d7803907b4a0_1440w.jpg)

具体来说, 我们从以下具体设定中来讨论如何进行 policy evaluation:

# 2 Policy Iteration and Value Iteration (Known Dynamics)

**Assumption:** 我们知道环境的 dynamics $p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})$, 并且 $\boldsymbol{s}, \boldsymbol{a}$ 都是离散的 (例如 $4 \times 4$ 的网格, $4$ 个 action).

![](https://pica.zhimg.com/v2-e647c1add0641de9a36fa2e6d951407a_1440w.jpg)

Tabular Case

此时 $V^\pi(\boldsymbol{s})$ 可以被存储在一个表格中, 我们的转移算子 $\mathcal{T}$ 为一个 $16 \times 16 \times 4$ 的张量. 于是我们可以利用 bootstrapped update: 
$$
V^\pi(\boldsymbol{s}) \gets \mathbb{E}_{\boldsymbol{a} \sim \pi(\boldsymbol{a} \mid \boldsymbol{s})} \left[r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{s}' \sim p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})} \left[V^\pi(\boldsymbol{s}')\right]\right]
$$ 
其中里层的 $V^\pi$ 基于已有的表格进行更新, 这就是 bootstrapped update. 这个式子之所以有实际意义是因为我们知道了 $p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})$, 也就是我们知道了 $\mathcal{T}$.

而如果我们采用之前描述的 deterministic policy, 那么可以进一步简化为 
$$
V^\pi(\boldsymbol{s}) \gets r(\boldsymbol{s}, \pi(\boldsymbol{s})) + \gamma \mathbb{E}_{\boldsymbol{s}' \sim p(\boldsymbol{s}' \mid \boldsymbol{s}, \pi(\boldsymbol{s}))} \left[V^\pi(\boldsymbol{s}')\right]
$$

因此我们得到了基于 dynamic programming 的 **[policy iteration](https://zhida.zhihu.com/search?content_id=253801844&content_type=Article&match_order=1&q=policy+iteration&zhida_source=entity)** 算法:
1.  fit $V^\pi(\boldsymbol{s}) \gets r(\boldsymbol{s}, \pi(\boldsymbol{s})) + \gamma \mathbb{E}_{\boldsymbol{s}' \sim p(\boldsymbol{s}' \mid \boldsymbol{s}, \pi(\boldsymbol{s}))} \left[V^\pi(\boldsymbol{s}')\right]$,  
2.  $\pi \gets \pi'$.  

上述过程的第一步可以写成线性方程组, 通过解线性方程组的方式一次性求解出所有的 $V^\pi(\boldsymbol{s})$.

事实上, 第二步的 $A^\pi(\boldsymbol{s}, \boldsymbol{a})$ 可以被视作是 $Q^\pi(\boldsymbol{s}, \boldsymbol{a})$. 故利用 $\arg\max_{\boldsymbol{a}} Q(\boldsymbol{s}, \boldsymbol{a})$ 就可以恢复出 policy. 因此我们并不需要显式保存 $\pi$, 因此可以得到以下 **[value iteration](https://zhida.zhihu.com/search?content_id=253801844&content_type=Article&match_order=1&q=value+iteration&zhida_source=entity)** 算法:

1.  fit $Q(\boldsymbol{s}, \boldsymbol{a}) \gets r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{s}' \sim p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})} \left[V(\boldsymbol{s}')\right]$,  
2.  $V(\boldsymbol{s}) \gets \max_{\boldsymbol{a}} Q(\boldsymbol{s}, \boldsymbol{a})$.  

更进一步的, 我们可以将第二步取 max 的过程直接写在第一步中或者将第一步的 $Q$ function 直接写在第二步中, 分别得到经典的 **value iteration** 与 **Q-iteration** 算法.

# 3 Fitted Value Iteration & Q-Iteration

在之前的 tabular case 中, 我们使用一个大表格来存储 value function. 然而在实际问题中这是不可行的:
-   **curse of dimensionality**: 如果进行离散化 (当然可能本身就是离散的), 我们的 $\mathcal{S}$ 与 $\mathcal{A}$ 大小会随着它们的维度指数上升.  
-   利用函数连续性可以把握临近状态与动作之间的关系, 而使用表格则无法做到这一点.

![](https://pic1.zhimg.com/v2-9c438dc0e433537cc6265b6a95024a4a_1440w.jpg)

curse of dimensionality

于是我们考虑一个参数 $\phi$ 的神经网络 $V: \mathcal{S} \rightarrow \mathbb{R}$, 参照 value iteration 中 $V$ function 的更新方式, 这里使用 
$$
L(\phi) = \frac{1}{2} \left\|V_\phi(\boldsymbol{s}) - \max_{\boldsymbol{a}} Q^\pi(\boldsymbol{s}, \boldsymbol{a})\right\|^2
$$ 
于是就有了 **fitted value iteration (known dynamics)** 算法:
1.  令 $y_i \gets \max_{\boldsymbol{a}_i} (r(\boldsymbol{s}_i, \boldsymbol{a}_i) + \gamma \mathbb{E}_{\boldsymbol{s}_i' \sim p(\boldsymbol{s}_i' \mid \boldsymbol{s}_i, \boldsymbol{a}_i)} \left[V_\phi(\boldsymbol{s}_i')\right])$,  
2.  令 $\phi \gets \arg \min_\phi \frac{1}{2} \left\|V_\phi(\boldsymbol{s}_i) - y_i\right\|^2$.  


不要忘记这一算法的核心假设: 我们**已知 dynamics**, 我们才能在第一步中需要找出 $\boldsymbol{s}_i$ 处"当前策略"的最优 action $\boldsymbol{a}_i$.

然而在**未知 dynamics** 的情况下, 我们通常**不能从一个非初始状态多次采样**. 也就是说我们至多只能得到一个 $(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)$ 的样本, 自然无法处理 $\max_{\boldsymbol{a}_i}$ 的问题.

一个可能被误解的点是: 尽管 $r(\boldsymbol{s}_t, \boldsymbol{a}_t)$ 的写法好像我们知道了 reward 的解析形式, 然而我们其实并不知道, 我们只是知道了 $\boldsymbol{s}_t$ 状态下采取 $\boldsymbol{a}_t$ action 的 reward 的样本. 在 model-free RL 中, 我们通常不尝试学习 reward function.

这里考虑利用参数为 $\phi$ 的神经网络学习 $Q$ function (在通常的实践中, 如果动作空间是离散的, 我们学习一个从 $\mathcal{S}$ 到 $\mathcal{A}$ 上**全体动作 Q 值**的映射, 如果动作空间是连续的, 我们会使用 $\mathcal{S} \times \mathcal{A}$ 到 Q 值的映射), 也就是 
$$
Q_\phi(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{s}' \sim p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})} \left[\max_{\boldsymbol{a}'} Q_\phi(\boldsymbol{s}', \boldsymbol{a}')\right]
$$
尽管看起来只是发生了简单的转换, 但这里其实有本质不同, 我们可以应用这样的方式处理任何的 policy. 我们就得到了 **fitted Q-iteration (unknown dynamics)** 算法:

1.  令 $y_i \gets r(\boldsymbol{s}_i, \boldsymbol{a}_i) + \gamma \mathbb{E}_{\boldsymbol{s}_i' \sim p(\boldsymbol{s}_i' \mid \boldsymbol{s}_i, \boldsymbol{a}_i)}\left[\max_{\boldsymbol{a}'} Q_\phi(\boldsymbol{s}_i', \boldsymbol{a}')\right]$, 由于我们这里只有一个 $\boldsymbol{s}'$, 于是我们近似为 $y_i \gets r(\boldsymbol{s}_i, \boldsymbol{a}_i) + \gamma \max_{\boldsymbol{a}'} Q_\phi(\boldsymbol{s}_i', \boldsymbol{a}')$,  
2.  令 $\phi \gets \arg \min_\phi \frac{1}{2} \left\|Q_\phi(\boldsymbol{s}_i, \boldsymbol{a}_i) - y_i\right\|^2$.  


这一算法与 off-policy actor-critic 有许多相似之处, 例如我们都需要一个 $\max_{\boldsymbol{a}'} Q_\phi(\boldsymbol{s}_i', \boldsymbol{a}')$ 的操作, 这个操作中的 $\boldsymbol{a}'$ 通常是基于当前的 $Q$ function 生成的. 这一算法同样应用于 off-policy 的情况. 我们可以想象一系列的 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}', r)$ 覆盖了整个空间, 当我们在所有这些数据上表现很好时, 就达到了我们的目标. 与 Actor-Critic 算法不同的是, 我们只需要一个网络即可.

![](https://pic4.zhimg.com/v2-c0a7e296e2df76ac9af9d5552b097fbf_1440w.jpg)

fitted Q-iteration

**full fitted Q-iteration** 算法:
1.  使用一些 policy 收集数据集 $\{(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)\}$, 获得大小为 $N$ 的数据集.  
2.  重复以下 $K$ 次:  
3.  令 $y_i \gets r(\boldsymbol{s}_i, \boldsymbol{a}_i) + \gamma \max_{\boldsymbol{a}_i'} Q_\phi(\boldsymbol{s}_i', \boldsymbol{a}_i')$,  
4.  令 $\phi \gets \arg \min_\phi \frac{1}{2} \sum_{i = 1}^N \left\|Q_\phi(\boldsymbol{s}_i, \boldsymbol{a}_i) - y_i\right\|^2$. (实际使用 $S$ 步梯度下降)  

# 4 From Q-iteration to [Q-learning](https://zhida.zhihu.com/search?content_id=253801844&content_type=Article&match_order=1&q=Q-learning&zhida_source=entity)

我们的 fitted Q-iteration 在优化什么呢? 我们定义 Error term $\mathcal{E}$, 
$$
\mathcal{E} = \frac{1}{2} \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}) \sim \mathcal{R}} \left[\left(Q_\phi(\boldsymbol{s}, \boldsymbol{a}) - (r + \gamma \max_{\boldsymbol{a}'} Q_\phi(\boldsymbol{s}', \boldsymbol{a}'))\right)^2\right]
$$
关于 $\mathcal{E}$ 项, 我们有以下的结论:

-   如果 $\mathcal{E} = 0$, 则 $$Q_\phi(\boldsymbol{s}, \boldsymbol{a}) = r + \gamma \max_{\boldsymbol{a}'} Q_\phi(\boldsymbol{s}', \boldsymbol{a}')$$这样的 $Q$ function 就是 **optimal Q function**, 对应于 optimal policy.  
-   如果 $\mathcal{E} \neq 0$, 我们无法给出任何的理论保证 (除非是 tabular case).  


正因为很多实际问题中我们无法得到 $\mathcal{E} = 0$, 故这个算法没有任何的理论保证.

我们也可以利用上述算法的特例构造一个 online 的算法, **online Q iteration** 也即 **(Q-Learning)** 算法:
1.  采取 action $\boldsymbol{a}_t$, 得到 $(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)$,  
2.  设置 $y_i \gets r(\boldsymbol{s}_i, \boldsymbol{a}_i) + \gamma \max_{\boldsymbol{a}_i'} Q_\phi(\boldsymbol{s}_i', \boldsymbol{a}_i')$,  
3.  采取一步梯度下降 $\phi \gets \phi - \alpha \nabla_\phi \frac{1}{2} \left\|Q_\phi(\boldsymbol{s}_i, \boldsymbol{a}_i) - y_i\right\|^2$.  

上述算法是最经典的 Q-learning 算法. 值得注意的算法是 off-policy 的, 因此在第一步理论上可以采取任何的 policy.

## 4.1 Exploration with Q-learning

在 online Q-learning 中, 我们在采样时通常并不会使用 argmax policy, 因为这可能会让我们陷入一些固定的 action 中. 常见的的方式有:

 $\epsilon$\-greedy policy

$$
\pi(\boldsymbol{a} \mid \boldsymbol{s}) = \begin{cases}  1 - \epsilon, & \text{if } \boldsymbol{a} = \arg\max_{\boldsymbol{a}'} Q_\phi(\boldsymbol{s}, \boldsymbol{a}'),\\  \epsilon / (|\mathcal{A}| - 1), & \text{otherwise}. \end{cases}
$$
这意味着多数时间我们会采取最优的 action, 走到一个较优的区域, 但是偶尔会采取非最优的 action, 以便于探索. 我们也可以随着训练过程逐渐减小 $\epsilon$.

 softmax policy / Boltzmann exploration

$$
\pi(\boldsymbol{a} \mid \boldsymbol{s}) \propto \exp\left(Q_\phi(\boldsymbol{s}, \boldsymbol{a}) / \tau\right)
$$
这有一个好处是接近 optimal 的次优解同样会被很高概率采取, 另一个好处是不会采用极差的 action.

更多 exploration 技巧将会在后续章节 **Exploration** 中详细介绍. 值得注意的是, 这里改变的 **仅仅是采样策略**, 并不影响更新方式中依然使用的 greedy action, 以及我们最终得到的 policy.

## 4.2 Brief Summary

  
我们通过以下简单的表格来总结一下本节介绍的 value-based methods (以 Q-function 相关的算法为例):

![](https://pic2.zhimg.com/v2-353d29e3a6d9692e5c3c52e27d6f4f39_1440w.jpg)

brief summary

# 5 Value Functions in Theory

这一小节中, 我们尝试从理论的角度来理解 value function methods.

## 5.1 Tabular case

回顾 value iteration 算法,
1.  fit $Q^\pi(\boldsymbol{s}, \boldsymbol{a}) \gets r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{s}' \sim p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})} \left[V^\pi(\boldsymbol{s}')\right]$,  
2.  $V(\boldsymbol{s}) \gets \max_{\boldsymbol{a}} Q^\pi(\boldsymbol{s}, \boldsymbol{a})$.  
    

对于 tabular case, 第一步相当于在更新每一个 $(\boldsymbol{s}, \boldsymbol{a})$ 位置的值. 在第二步中, 我们选取每一行 (对于 $\boldsymbol{a}$) 的最大值.

为了理论分析的简洁, 我们考虑以下的记号:

**Definition 1**. **_[Bellman operator](https://zhida.zhihu.com/search?content_id=253801844&content_type=Article&match_order=1&q=Bellman+operator&zhida_source=entity)_** _$\mathcal{B}$: 
$$
\mathcal{B}V = \max_{\boldsymbol{a}} r_{\boldsymbol{a}} + \gamma \mathcal{T}_{\boldsymbol{a}} V
$$
其中 $\mathcal{T}_{\boldsymbol{a}}$ 可以理解为是矩阵, 储存在 $\boldsymbol{a}$ 下所有状态到所有状态的转移概率, 也就是 $\mathcal{T}_{\boldsymbol{a}, i, j} = p(\boldsymbol{s'} = i \mid \boldsymbol{s} = j, \boldsymbol{a})$. 而 $r_{\boldsymbol{a}}$ 为 $\boldsymbol{a}$ 下所有状态的 reward._

**Proposition 1**. _$V^\ast$ (最优的 $V$ function) 是 Bellman operator 的不动点, 也就是 
$$
V^\ast(\boldsymbol{s}) = \max_{\boldsymbol{a}} r(\boldsymbol{s},\boldsymbol{a}) + \gamma \mathbb{E}\left[V^\ast(\boldsymbol{s'})\right]
$$
化简得到 $\mathcal{B}V^\ast = V^\ast$ 而且这一不动点存在且唯一, 且始终对应于最优策略._

我们一定能够收敛到最优的 $V$ function 吗? 事实上我们有以下结果:

**Theorem 1**. _在 tabular case 中, value iteration 一定会收敛到最优的 $V$ function._

证明的核心是 Bellman operator 是一个 contraction mapping, 也就是对于任意的 $V, \bar{V}$, 有 
$$
\left\|\mathcal{B}V - \mathcal{B}\bar{V}\right\|_\infty \leq \gamma \left\|V - \bar{V}\right\|_\infty
$$
注意当我们代入 $\bar{V}$ with $V^\ast$ 时, 我们就有 $\|\mathcal{B} V - V^\ast\|_\infty \leq \gamma \|V - V^\ast\|_\infty$ 一个值得注意的点是这里的范数是 $\infty$ 范数, 也就是最大的差值.

  
_Proof_
我们证明 $\mathcal{B}$ 是 contraction mapping:
$$
\begin{aligned}         \left\|\mathcal{B}V - \mathcal{B}\bar{V}\right\|_\infty &= \left\|\max_{\boldsymbol{a}} (r_{\boldsymbol{a}} + \gamma \mathcal{T}_{\boldsymbol{a}} V) - \max_{\boldsymbol{a}} (r_{\boldsymbol{a}} + \gamma \mathcal{T}_{\boldsymbol{a}} \bar{V})\right\|_\infty\\  &\leq \left\|\max_{\boldsymbol{a}} (r_{\boldsymbol{a}} + \gamma \mathcal{T}_{\boldsymbol{a}} V - r_{\boldsymbol{a}} - \gamma \mathcal{T}_{\boldsymbol{a}} \bar{V})\right\|_\infty\\  &= \gamma \left\|\max_{\boldsymbol{a}} \mathcal{T}_{\boldsymbol{a}} (V - \bar{V})\right\|_\infty\\  &= \gamma \max_{\boldsymbol{a}} \|\mathcal{T}_{\boldsymbol{a}} (V - \bar{V})\|_\infty\\  &\leq \gamma \max_{\boldsymbol{a}} \|V - \bar{V}\|_\infty = \gamma \|V - \bar{V}\|_\infty.  \end{aligned}
$$其中利用了 $\mathcal{T}_{\boldsymbol{a}}$ 作为一个转移矩阵的性质, 也就是 $\|\mathcal{T}_{\boldsymbol{a}} V\|_\infty \leq \|V\|_\infty$  

## 5.2 Non-tabular case

接下来考虑 non-tabular case, 也就是 fitted value iteration, 不妨记 $\Omega$ 为当前架构下所有可能的网络参数集合, 于是第二步中的更新可以写成 
$$
V' \gets \arg\min_{V' \in \Omega} \frac{1}{2} \sum \left\|V'(\boldsymbol{s}) - (\mathcal{B} V(s))\right\|^2
$$
我们定义一个新的算子 $\Pi$ 来抽象这一过程: 
$$
\Pi V = \arg\min_{V' \in \Omega} \frac{1}{2} \sum \|V'(\boldsymbol{s}) - V(\boldsymbol{s})\|^2
$$
假设所有可能的 $V$ function 是一个平面, 而可表示的是一条直线, 此时网络的 $V$ 一定在直线上, 然而 $\mathcal{B}V$ 可能落在平面的任何位置, 而训练过程则是在直线上找到距离 $\mathcal{B}V$ 最近的 $V'$, 故个算子是向 $\Omega$ 的投影 (在 $l_2$ 范数下), 从而这一更新本身是一个 contraction mapping.

于是整个 fitted value iteration 就是 $V \gets \Pi \mathcal{B} V$ 但是尽管这两个算子都是 contraction mapping, 但是它们在组合后不一定是 contraction mapping, 因为它们的 contraction 是在不同范数下的.

这依然可以用平面与直线的例子来解释, 在直线上找到距离平面最近的点, 这个点却可能反而离 $V^\ast$ 更远.

![](https://pic4.zhimg.com/v2-a1a9edefeb0e5876ccf42d37ceaebcc5_1440w.jpg)

**Summary**: value iteration 有收敛性保证, 但是 fitted value iteration 却没有, 在实际中通常也不会收敛到最优解.

## 5.3 Corollaries

事实 fitted Q-iteration/ Q-learning 也没有收敛性保证, 我们定义 Bellman operator $\mathcal{B}$ 为 
$$
\mathcal{B} Q = r + \gamma \mathcal{T} \max_{\boldsymbol{a}} Q.
$$
类似地定义 $\Pi$, 这两个算子都是 contraction mapping, 但是它们的组合却不是 contraction mapping. 但是一个令人困惑的点在于, 我们似乎始终在进行 regression? 要注意的是, 这里并不是进行梯度下降, 而属于半梯度方法 (semi-gradient), 因为我们的更新 
$$
\phi \gets \phi - \alpha \nabla_\phi \frac{1}{2} \left\|Q_\phi(\boldsymbol{s}_i, \boldsymbol{a}_i) - y_i\right\|^2
$$
中的目标也包含了 $\phi$, 而我们的梯度实际上没有流过这部分. 一个可以得到一定收敛性保证的是使用 residual gradient (也就是完全求梯度), 但是其有数值问题, 实际效果反而不如 Q-learning.

另一个推论是, 我们之前的 actor-critic 算法也没有收敛性保证, bootstrap 与 fitted 两个过程类似于 $\mathcal{B}, \Pi$, 但是其组合并不是 contraction mapping.

# 6 Summary

在本节中, 我们:
-   介绍了 value function methods, 并给出了 policy iteration 与 value iteration 算法.  
-   介绍了 fitted value iteration 与 Q-iteration 算法, 并给出了 Q-learning 算法.  
-   从理论上分析了 value function methods 的性质, 得到了 tabular case 的收敛性保证, 得出 fitted value iteration,Q-iteration, actor-critic 算法没有收敛性保证.