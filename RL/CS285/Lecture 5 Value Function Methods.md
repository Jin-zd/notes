# 1 Introduction to Value Function Methods
在演员-评论家算法中，我们引入了价值函数，已经告诉了我们在每个状态下应该采取什么动作： 
$$
A^\pi(\boldsymbol{s},\boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \gamma\mathbb{E}_{\boldsymbol{s}' \sim p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})} \left[V^\pi(\boldsymbol{s}')\right] - V^\pi(\boldsymbol{s}) \approx r(\boldsymbol{s}, \boldsymbol{a}) + \gamma V^\pi(\boldsymbol{s}') - V^\pi(\boldsymbol{s})
$$
或是 
$$
A^\pi(\boldsymbol{s}, \boldsymbol{a}) = Q^\pi(\boldsymbol{s}, \boldsymbol{a}) - V^\pi(\boldsymbol{s}) = Q^\pi(\boldsymbol{s}, \boldsymbol{a}) - \mathbb{E}_{\boldsymbol{a} \sim \pi(\boldsymbol{a} \mid \boldsymbol{s})} \left[Q^\pi(\boldsymbol{s}, \boldsymbol{a})\right]
$$
这给了我们一个思路：能否完全不使用策略梯度而仅仅使用价值函数呢？这便是 **基于价值的方法（Value-based methods）** 的基本思想了，在基于价值的方法中没有显式的策略，仅仅学习一定形式的价值函数。回顾以下记号：
-  $A^\pi(\boldsymbol{s}, \boldsymbol{a})$ 表示在状态 $\boldsymbol{s}$ 下采取动作 $\boldsymbol{a}$ 相较于平均动作的优势。  
-  $\arg\max_{\boldsymbol{a}} A^\pi(\boldsymbol{s}, \boldsymbol{a})$ 表示在状态 $\boldsymbol{s}$ 下依照策略 $\pi$ 可以采取的最优动作。 

很显然如果 $A^\pi$ 准确，那么采取 $\arg\max_{\boldsymbol{a}} A^\pi(\boldsymbol{s}, \boldsymbol{a})$ 至少和按照 $\boldsymbol{a} \sim \pi$ 采取的动作一样好。于是可以使用方式更新策略： 
$$
\pi'(\boldsymbol{a}_t \mid \boldsymbol{s}_t) = \begin{cases}  1, & \text{if } \boldsymbol{a}_t = \arg\max_{\boldsymbol{a}_t} A^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t),\\  0, & \text{otherwise}. \end{cases}
$$
上述过程展示了如何利用价值函数来更新策略。如果在此之后进行策略评估，那么就可以重复上述过程，不断得到更优的策略的价值函数，就得到了 **策略迭代（Policy iteration）** 算法。

在通用 RL 框架中，在 Part 2 中我们会拟合 $A^\pi$ （或者 $Q^\pi$, $V^\pi$）。在 Part 3 中我们使用 $\pi \gets \pi'$：
![](5-1.png)

下面从具体设定中来讨论如何进行策略评估。

# 2 Policy Iteration and Value Iteration (Known Dynamics)

假设：我们知道环境的动态 $p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})$，并且 $\boldsymbol{s}, \boldsymbol{a}$ 都是离散的（例如 $4 \times 4$ 的网格， $4$ 个动作）
![](5-2.png)

此时 $V^\pi(\boldsymbol{s})$ 可以被存储在一个表格中，转移算子 $\mathcal{T}$ 为一个 $16 \times 16 \times 4$ 的张量。于是可以利用自举更新：
$$
V^\pi(\boldsymbol{s}) \gets \mathbb{E}_{\boldsymbol{a} \sim \pi(\boldsymbol{a} \mid \boldsymbol{s})} \left[r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{s}' \sim p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})} \left[V^\pi(\boldsymbol{s}')\right]\right]
$$
其中里层的 $V^\pi$ 基于已有的表格进行更新，这就是自举更新。这个式子之所以有实际意义，是因为我们知道了 $p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})$，也就是我们知道了 $\mathcal{T}$。

而如果采用之前描述的确定性策略，那么可以进一步简化为：
$$
V^\pi(\boldsymbol{s}) \gets r(\boldsymbol{s}, \pi(\boldsymbol{s})) + \gamma \mathbb{E}_{\boldsymbol{s}' \sim p(\boldsymbol{s}' \mid \boldsymbol{s}, \pi(\boldsymbol{s}))} \left[V^\pi(\boldsymbol{s}')\right]
$$
因此得到了基于动态规划的 **策略迭代（Policy Iteration）** 算法：
1. 拟合 $V^\pi(\boldsymbol{s}) \gets r(\boldsymbol{s}, \pi(\boldsymbol{s})) + \gamma \mathbb{E}_{\boldsymbol{s}' \sim p(\boldsymbol{s}' \mid \boldsymbol{s}, \pi(\boldsymbol{s}))} \left[V^\pi(\boldsymbol{s}')\right]$；
2. $\pi \gets \pi'$。
上述过程的第一步可以写成线性方程组，通过解线性方程组的方式一次性求解出所有的 $V^\pi(\boldsymbol{s})$。

事实上，第二步的 $A^\pi(\boldsymbol{s}, \boldsymbol{a})$ 可以被视作是 $Q^\pi(\boldsymbol{s}, \boldsymbol{a})$。故利用 $\arg\max_{\boldsymbol{a}} Q(\boldsymbol{s}, \boldsymbol{a})$ 就可以恢复出策略。因此我们并不需要显式保存 $\pi$，因此可以得到以下 **价值迭代（Value Iteration）** 算法：
1. 拟合 $Q(\boldsymbol{s}, \boldsymbol{a}) \gets r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{s}' \sim p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})} \left[V(\boldsymbol{s}')\right]$；
2. $V(\boldsymbol{s}) \gets \max_{\boldsymbol{a}} Q(\boldsymbol{s}, \boldsymbol{a})$。
更进一步的，我们可以将第二步取最大值的过程直接写在第一步中，或者将第一步的 $Q$ 函数直接写在第二步中，分别得到经典的 **价值迭代** 与 **Q 迭代（Q-iteration）** 算法。

# 3 Fitted Value Iteration & Q-Iteration
在之前的表格示例中, 我们使用一个大表格来存储价值函数。然而在实际问题中这是不可行的：
- 维数灾难：如果进行离散化（当然可能本身就是离散的），$\mathcal{S}$ 与 $\mathcal{A}$ 大小会随着它们的维度指数上升。
- 利用函数连续性可以把握临近状态与动作之间的关系，而使用表格则无法做到这一点。
![](5-3.png)
于是考虑一个参数 $\phi$ 的神经网络 $V: \mathcal{S} \rightarrow \mathbb{R}$，参照价值迭代中价值函数的更新方式，这里使用
$$
L(\phi) = \frac{1}{2} \left\|V_\phi(\boldsymbol{s}) - \max_{\boldsymbol{a}} Q^\pi(\boldsymbol{s}, \boldsymbol{a})\right\|^2
$$
于是就有了 **拟合值迭代，已知动态（Fitted value iteration, known dynamics）** 算法：
1. 令 $y_i \gets \max_{\boldsymbol{a}_i} (r(\boldsymbol{s}_i, \boldsymbol{a}_i) + \gamma \mathbb{E}_{\boldsymbol{s}_i' \sim p(\boldsymbol{s}_i' \mid \boldsymbol{s}_i, \boldsymbol{a}_i)} \left[V_\phi(\boldsymbol{s}_i')\right])$；
2. 令 $\phi \gets \arg \min_\phi \frac{1}{2} \left\|V_\phi(\boldsymbol{s}_i) - y_i\right\|^2$。

不要忘记这一算法的核心假设：已知动态，才能在第一步中需要找出 $\boldsymbol{s}_i$ 处"当前策略"的最优 动作 $\boldsymbol{a}_i$。然而在未知动态的情况下，我们通常不能从一个非初始状态多次采样。也就是说至多只能得到一个 $(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)$ 的样本，自然无法处理 $\max_{\boldsymbol{a}_i}$ 的问题。

一个可能被误解的点是：尽管 $r(\boldsymbol{s}_t, \boldsymbol{a}_t)$ 的写法好像知道了奖励的解析形式，然而其实并不知道，我们只是知道了 $\boldsymbol{s}_t$ 状态下采取 $\boldsymbol{a}_t$ 动作的奖励的样本。在无模型 RL 中，我们通常不尝试学习奖励函数。

这里考虑利用参数为 $\phi$ 的神经网络学习 $Q$ 函数（在通常的实践中，如果动作空间是离散的，我们学习一个从 $\mathcal{S}$ 到 $\mathcal{A}$ 上全体动作 Q 值的映射，如果动作空间是连续的，我们会使用 $\mathcal{S} \times \mathcal{A}$ 到 Q 值的映射），也就是 
$$
Q_\phi(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{s}' \sim p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})} \left[\max_{\boldsymbol{a}'} Q_\phi(\boldsymbol{s}', \boldsymbol{a}')\right]
$$
尽管看起来只是发生了简单的转换，但这里其实有本质不同，我们可以应用这样的方式处理任何的策略，我们就得到了 **拟合 Q 迭代，未知动态（Fitted Q-iteration, unknown dynamics）** 算法：
1. 令 $y_i \gets r(\boldsymbol{s}_i, \boldsymbol{a}_i) + \gamma \mathbb{E}_{\boldsymbol{s}_i' \sim p(\boldsymbol{s}_i' \mid \boldsymbol{s}_i, \boldsymbol{a}_i)}\left[\max_{\boldsymbol{a}'} Q_\phi(\boldsymbol{s}_i', \boldsymbol{a}')\right]$，由于这里只有一个 $\boldsymbol{s}'$, 于是近似为 $y_i \gets r(\boldsymbol{s}_i, \boldsymbol{a}_i) + \gamma \max_{\boldsymbol{a}'} Q_\phi(\boldsymbol{s}_i', \boldsymbol{a}')$；
2. 令 $\phi \gets \arg \min_\phi \frac{1}{2} \left\|Q_\phi(\boldsymbol{s}_i, \boldsymbol{a}_i) - y_i\right\|^2$。
![](5-4.png)
这一算法与 off-policy 演员-评论家算法有许多相似之处。如我们都需要一个 $\max_{\boldsymbol{a}'} Q_\phi(\boldsymbol{s}_i', \boldsymbol{a}')$ 的操作，这个操作中的 $\boldsymbol{a}'$ 通常是基于当前的 $Q$ 函数生成的。这一算法同样应用于 off-policy 的情况。我们可以想象一系列的 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}', r)$ 覆盖了整个空间，当我们在所有这些数据上表现很好时，就达到了我们的目标。与演员-评论家算法不同的是，我们只需要一个网络即可。

完整的 **拟合 Q 迭代（Fitted Q-iteration）** 算法：
1. 使用一些 policy 收集数据集 $\{(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)\}$，获得大小为 $N$ 的数据集； 
2. 重复以下 $K$ 次； 
3. 令 $y_i \gets r(\boldsymbol{s}_i, \boldsymbol{a}_i) + \gamma \max_{\boldsymbol{a}_i'} Q_\phi(\boldsymbol{s}_i', \boldsymbol{a}_i')$；
4. 令 $\phi \gets \arg \min_\phi \frac{1}{2} \sum_{i = 1}^N \left\|Q_\phi(\boldsymbol{s}_i, \boldsymbol{a}_i) - y_i\right\|^2$（实际使用 $S$ 步梯度下降）。 

# 4 From Q-iteration to Q-learning
拟合 Q 迭代在优化什么呢？定义误差项 $\mathcal{E}$： 
$$
\mathcal{E} = \frac{1}{2} \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}) \sim \mathcal{R}} \left[\left(Q_\phi(\boldsymbol{s}, \boldsymbol{a}) - (r + \gamma \max_{\boldsymbol{a}'} Q_\phi(\boldsymbol{s}', \boldsymbol{a}'))\right)^2\right]
$$
关于 $\mathcal{E}$ 项，有以下的结论：
- 如果 $\mathcal{E} = 0$，则 $$Q_\phi(\boldsymbol{s}, \boldsymbol{a}) = r + \gamma \max_{\boldsymbol{a}'} Q_\phi(\boldsymbol{s}', \boldsymbol{a}')$$这样的 $Q$ 函数就是 **最优 Q 函数**，对应于最优策略。 
- 如果 $\mathcal{E} \neq 0$，无法给出任何的理论保证（除非是表格案例）。

正因为很多实际问题中我们无法得到 $\mathcal{E} = 0$，故这个算法没有任何的理论保证。

我们也可以利用上述算法的特例构造一个 online 的算法，也即 **Q 学习（Q-Learning）** 算法：
1. 采取动作 $\boldsymbol{a}_t$, 得到 $(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)$；
2. 设置 $y_i \gets r(\boldsymbol{s}_i, \boldsymbol{a}_i) + \gamma \max_{\boldsymbol{a}_i'} Q_\phi(\boldsymbol{s}_i', \boldsymbol{a}_i')$；
3. 采取一步梯度下降 $\phi \gets \phi - \alpha \nabla_\phi \frac{1}{2} \left\|Q_\phi(\boldsymbol{s}_i, \boldsymbol{a}_i) - y_i\right\|^2$。
上述算法是最经典的 Q 学习算法。值得注意的算法是 off-policy 的，因此在第一步理论上可以采取任何的策略。

## 4.1 Exploration with Q-learning
在 online Q 学习中，在采样时通常并不会使用最大策略，因为这可能会让我们陷入一些固定的动作中。常见的的方式有：
- $\epsilon$ 贪心策略：
$$
\pi(\boldsymbol{a} \mid \boldsymbol{s}) = \begin{cases}  1 - \epsilon, & \text{if } \boldsymbol{a} = \arg\max_{\boldsymbol{a}'} Q_\phi(\boldsymbol{s}, \boldsymbol{a}'),\\  \epsilon / (|\mathcal{A}| - 1), & \text{otherwise}. \end{cases}
$$
这意味着多数时间我们会采取最优的动作，走到一个较优的区域，但是偶尔会采取非最优的动作，以便于探索。我们也可以随着训练过程逐渐减小 $\epsilon$。
 - softmax 策略 / 玻尔兹曼（Boltzmann）探索：
$$
\pi(\boldsymbol{a} \mid \boldsymbol{s}) \propto \exp\left(Q_\phi(\boldsymbol{s}, \boldsymbol{a}) / \tau\right)
$$
这有一个好处是接近最优的次优解同样会被很高概率采取，另一个好处是不会采用极差的动作。

更多探索技巧将会在后续章节 **Exploration** 中详细介绍。值得注意的是，这里改变的仅仅是采样策略，并不影响更新方式中依然使用的贪心动作，以及我们最终得到的策略。

## 4.2 Brief Summary
我们通过以下简单的表格来总结一下本节介绍的基于价值的方法（以 Q 函数相关的算法为例）：

|   算法    |  假设  | on-policy / off-policy |     更新方式     |
| :-----: | :--: | :--------------------: | :----------: |
|  Q 迭代   | 已知动态 |           /            | 动态规划/求解线性方程组 |
| 拟合 Q 迭代 | 未知动态 |       off-policy       |     批量更新     |
|  Q 学习   | 未知动态 |       off-policy       |   单样本在线更新    |

# 5 Value Functions in Theory
我们尝试从理论的角度来理解基于价值的方法。
## 5.1 Tabular case
回顾价值迭代算法：
1. 拟合 $Q^\pi(\boldsymbol{s}, \boldsymbol{a}) \gets r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{s}' \sim p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})} \left[V^\pi(\boldsymbol{s}')\right]$；
2. $V(\boldsymbol{s}) \gets \max_{\boldsymbol{a}} Q^\pi(\boldsymbol{s}, \boldsymbol{a})$。

对于表格案例，第一步相当于在更新每一个 $(\boldsymbol{s}, \boldsymbol{a})$ 位置的值。在第二步中，我们选取每一行（对于 $\boldsymbol{a}$）的最大值。

为了理论分析的简洁，考虑以下的记号：
**Definition 1**. _Bellman operator $\mathcal{B}$（贝尔曼算子）_ 
$$
\mathcal{B}V = \max_{\boldsymbol{a}} r_{\boldsymbol{a}} + \gamma \mathcal{T}_{\boldsymbol{a}} V
$$
其中 $\mathcal{T}_{\boldsymbol{a}}$ 可以理解为是矩阵，储存在 $\boldsymbol{a}$ 下所有状态到所有状态的转移概率，也就是 $\mathcal{T}_{\boldsymbol{a}, i, j} = p(\boldsymbol{s'} = i \mid \boldsymbol{s} = j, \boldsymbol{a})$。而 $r_{\boldsymbol{a}}$ 为 $\boldsymbol{a}$ 下所有状态的奖励。

**Proposition 1**. $V^\ast$（最优的价值函数）是贝尔曼算子的不动点。也就是：
$$
V^\ast(\boldsymbol{s}) = \max_{\boldsymbol{a}} r(\boldsymbol{s},\boldsymbol{a}) + \gamma \mathbb{E}\left[V^\ast(\boldsymbol{s'})\right]
$$
化简得到 $\mathcal{B}V^\ast = V^\ast$ 而且这一不动点存在且唯一，且始终对应于最优策略。

一定能够收敛到最优的价值函数吗？事实上有以下结果：

**Theorem 1**. 在表格案例中，价值迭代一定会收敛到最优的价值函数。
证明的核心是贝尔曼算子是一个压缩映射，也就是对于任意的 $V, \bar{V}$，有 
$$
\left\|\mathcal{B}V - \mathcal{B}\bar{V}\right\|_\infty \leq \gamma \left\|V - \bar{V}\right\|_\infty
$$
注意当我们代入 $\bar{V}$ 和 $V^\ast$ 时，我们就有 $\|\mathcal{B} V - V^\ast\|_\infty \leq \gamma \|V - V^\ast\|_\infty$。一个值得注意的点是这里的范数是 $\infty$ 范数，也就是最大的差值。

_Proof._
我们证明 $\mathcal{B}$ 是压缩映射：
$$
\begin{aligned}         
\left\|\mathcal{B}V - \mathcal{B}\bar{V}\right\|_\infty 
&= \left\|\max_{\boldsymbol{a}} (r_{\boldsymbol{a}} + \gamma \mathcal{T}_{\boldsymbol{a}} V) - \max_{\boldsymbol{a}} (r_{\boldsymbol{a}} + \gamma \mathcal{T}_{\boldsymbol{a}} \bar{V})\right\|_\infty\\  
&\leq \left\|\max_{\boldsymbol{a}} (r_{\boldsymbol{a}} + \gamma \mathcal{T}_{\boldsymbol{a}} V - r_{\boldsymbol{a}} - \gamma \mathcal{T}_{\boldsymbol{a}} \bar{V})\right\|_\infty\\  
&= \gamma \left\|\max_{\boldsymbol{a}} \mathcal{T}_{\boldsymbol{a}} (V - \bar{V})\right\|_\infty\\  
&= \gamma \max_{\boldsymbol{a}} \|\mathcal{T}_{\boldsymbol{a}} (V - \bar{V})\|_\infty\\  &\leq \gamma \max_{\boldsymbol{a}} \|V - \bar{V}\|_\infty = \gamma \|V - \bar{V}\|_\infty
\end{aligned}
$$
其中利用了 $\mathcal{T}_{\boldsymbol{a}}$ 作为一个转移矩阵的性质，也就是 $\|\mathcal{T}_{\boldsymbol{a}} V\|_\infty \leq \|V\|_\infty$。

## 5.2 Non-tabular case
接下来考虑非表格案例，也就是拟合价值迭代。不妨记 $\Omega$ 为当前架构下所有可能的网络参数集合，于是第二步中的更新可以写成：
$$
V' \gets \arg\min_{V' \in \Omega} \frac{1}{2} \sum \left\|V'(\boldsymbol{s}) - (\mathcal{B} V(s))\right\|^2
$$
我们定义一个新的算子 $\Pi$ 来抽象这一过程：
$$
\Pi V = \arg\min_{V' \in \Omega} \frac{1}{2} \sum \|V'(\boldsymbol{s}) - V(\boldsymbol{s})\|^2
$$
假设所有可能的价值函数是一个平面，而可表示的是一条直线，此时网络的 $V$ 一定在直线上。然而 $\mathcal{B}V$ 可能落在平面的任何位置，而训练过程则是在直线上找到距离 $\mathcal{B}V$ 最近的 $V'$，故这个算子是向 $\Omega$ 的投影（在 $L_2$ 范数下），从而这一更新本身是一个压缩映射。

于是整个拟合价值迭代就是 
$$
V \gets \Pi \mathcal{B} V
$$
但是尽管这两个算子都是压缩映射，但是它们在组合后不一定是压缩映射，因为它们的压缩是在不同范数下的。

这依然可以用平面与直线的例子来解释，在直线上找到距离平面最近的点, 这个点却可能反而离 $V^\ast$ 更远。
![](5-5.png)

价值迭代有收敛性保证，但是拟合价值迭代却没有，在实际中通常也不会收敛到最优解。

## 5.3 Corollaries
事实拟合 Q 迭代/ Q 学习也没有收敛性保证，定义贝尔曼算子 $\mathcal{B}$ 为
$$
\mathcal{B} Q = r + \gamma \mathcal{T} \max_{\boldsymbol{a}} Q
$$
类似地定义 $\Pi$，这两个算子都是压缩映射，但是它们的组合却不是压缩映射。但是一个令人困惑的点在于，我们似乎始终在进行回归？要注意的是，这里并不是进行梯度下降，而属于半梯度方法 （semi-gradient），因为我们更新
$$
\phi \gets \phi - \alpha \nabla_\phi \frac{1}{2} \left\|Q_\phi(\boldsymbol{s}_i, \boldsymbol{a}_i) - y_i\right\|^2
$$
的目标也包含了 $\phi$，而我们的梯度实际上没有流过这部分。一个可以得到一定收敛性保证的是使用 残差梯度（residual gradient），也就是完全求梯度，但是其有数值问题，实际效果反而不如 Q 学习。
另一个推论是，我们之前的演员-评论家算法也没有收敛性保证，自举与拟合两个过程类似于 $\mathcal{B}$ 和 $\Pi$，但是其组合并不是压缩映射。

# 6 Summary
在本节中，我们：
- 介绍了基于价值的方法，并给出了策略迭代与价值迭代算法。  
- 介绍了拟合价值迭代与 Q 迭代算法，并给出了 Q 学习算法。
- 从理论上分析了基于价值的方法的性质，得到了表格案例的收敛性保证，得出拟合价值迭代， Q 迭代，演员-评论家算法没有收敛性保证。