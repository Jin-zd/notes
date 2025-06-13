# 1 Introduction to Actor-Critic Algorithms
回顾在 REINFORCE 中，我们的策略梯度的形式是 
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i = 1}^N \sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t}) \hat{Q}_{i,t}
$$
这里的 $\hat{Q}_{i,t}$ 是我们对从 $\boldsymbol{s}_{i,t}$ 开始采用 $\boldsymbol{a}_{i,t}$ 的奖励的估计。这里的估计方式是[[Concepts#10 蒙特卡洛方法 (Monte Carlo Method)|蒙特卡洛方法 (Monte Carlo Method)]]：通过将单个轨迹后续所有奖励累加得到的。由于只有单个而且很长的轨迹，这种估计方式方差很大。而且我们不能使用多个轨迹，因为收集多个轨迹需要与多次环境交互，而我们通常无法在非起始状态下开始交互。

这里采用蒙特卡洛方法估计的未来奖励是对如下的 $Q$ 函数的估计：
**Definition 1**. _Q function（Q 函数）_
$$
Q^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t) = \sum_{t' = t}^T \mathbb{E}_{\tau \sim \pi_\theta} \left[r(\boldsymbol{s}_{t'}, \boldsymbol{a}_{t'}) \mid \boldsymbol{s}_t, \boldsymbol{a}_t\right]
$$
回顾我们在策略梯度中还可以使用基线，也就是 $b_t = \frac{1}{N} \sum_{i = 1}^N r(\tau)$，这里的基线是利用整条轨迹估计的，可以理解为是对 $V^\pi(\boldsymbol{s}_1)$ 的一个无偏估计。

我们借鉴这一思想，定义一个依赖于状态的基线：
**Definition 2**. _value function（价值函数）_
$$
V^\pi(\boldsymbol{s}_t) = \mathbb{E}_{\boldsymbol{a}_t \sim \pi_\theta} \left[Q^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]
$$

我们从未来的奖励中减去这个基线，也就可以得到如下定义的优势函数：
**Definition 3**. _advantage function（优势函数）_
$$
A^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t) = Q^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t) - V^\pi(\boldsymbol{s}_t)
$$
正如其名字暗示的那样，优势函数表示在当前状态下采取动作 $\boldsymbol{a}_t$ 的优势。类似地我们可以写出新的梯度估计：
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i = 1}^N \sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t}) A^\pi(\boldsymbol{s}_{i,t}, \boldsymbol{a}_{i,t})
$$
在刚才的过程中，我们从 Q 函数估计中减去了一个基线，也就是 $V^\pi(\boldsymbol{s}_t)$。由于这里的基线仅仅依赖于状态，依照如下过程可以证明仅依赖于状态的基线都是无偏的：
$$\begin{aligned}  \mathbb{E}_{\boldsymbol{a}_t \sim \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \left[\nabla_\theta \log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t) V(\boldsymbol{s}_t)\right] &= \mathbb{E}_{\boldsymbol{a}_t \sim \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \left[\nabla_\theta \log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)\right] V(\boldsymbol{s}_t)\\  &= \nabla_\theta\int \pi_\theta(\boldsymbol{a}_t\mid \boldsymbol{s}_t)\text{d}\boldsymbol{a}_t V(\boldsymbol{s}_t)\\   &= 0 V(\boldsymbol{s}_t) = 0. \end{aligned}
$$
然而，尽管引入 $V^\pi(\boldsymbol{s}_t)$ 作为基线本身不会引入偏差，使用一些其他的方式估计 $Q^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t)$ 可能会引入偏差，进而导致整个梯度估计存在偏差。而上一节中介绍的蒙特卡洛估计是无偏的。

# 2 Policy Evaluation
我们不难注意到 $Q$ 函数和价值函数之间有一个关系，也就是 
$$
Q^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t) = r(\boldsymbol{s}_t, \boldsymbol{a}_t) + \mathbb{E}_{\boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)}\left[V^\pi(\boldsymbol{s}_{t + 1})\right]
$$
由于我们不能观测多个 $\boldsymbol{s}_{t + 1}$，故这里只能用单样本近似 $\mathbb{E}_{\boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)}\left[V^\pi(\boldsymbol{s}_{t + 1})\right]$，近似结果为：
$$
Q^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t) \approx r(\boldsymbol{s}_t, \boldsymbol{a}_t) + V^\pi(\boldsymbol{s}_{t + 1})
$$
这个近似是无偏的，相较于策略梯度中的未来奖励，由于这里仅有一步的近似，方差会小很多。

基于这一近似，我们也可以估计优势函数： 
$$
A^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t) \approx r(\boldsymbol{s}_t, \boldsymbol{a}_t) + V^\pi(\boldsymbol{s}_{t + 1}) - V^\pi(\boldsymbol{s}_t)
$$
但我们还需要知道 $V^\pi(\boldsymbol{s})$，目前我们使用一个参数为 $\phi$ 神经网络 $\hat{V}^\pi(\boldsymbol{s})$ 来拟合 $V^\pi(\boldsymbol{s})$。利用数据拟合这些价值函数的过程称为**策略评估（policy evaluation）**。回到原先的 RL 的通用框架，我们在 Part 2 需要拟合这些价值函数。
![](4-1.png)
我们可以使用蒙特卡洛的方法进行策略评估，也就是： 
$$
V^\pi(\boldsymbol{s}_t) \approx \frac{1}{N} \sum_{i = 1}^N \sum_{t' = t}^T r(\boldsymbol{s}_{i,t'}, \boldsymbol{a}_{i,t'})
$$
然而由于通常情况下我们做不到多次从 $t$ 出发进行采样，除非我们有一个环境的模型。故我们使用的是单个样本估计：
$$
V^\pi(\boldsymbol{s}_t) \approx \sum_{t' = t}^T r(\boldsymbol{s}_{t'}, \boldsymbol{a}_{t'})
$$
也就是我们用以下的目标进行训练：
$$
\left\{\left(\boldsymbol{s}_{i,t}, \sum_{t' = t}^T r(\boldsymbol{s}_{t'}, \boldsymbol{a}_{t'})\right)\right\}
$$
为了简便起见，我们通常记我们的目标为 $y_{i,t}$，我们的损失为 
$$
L(\phi) = \frac{1}{2} \sum_{i = 1}^N \left(\hat{V}^\pi(\boldsymbol{s}_{i,t}) - \sum_{t' = t}^T r(\boldsymbol{s}_{i,t'}, \boldsymbol{a}_{i,t'})\right)^2
$$
上述的目标也被称为**蒙特卡洛目标（Monte Carlo target）**，一个问题是其方差很大，为了减小这个方差，我们通常使用的是以下的**自举目标（Bootstrap target）**：
$$
y_{i,t} = \sum_{t' = t}^T \mathbb{E}_{\pi_\theta} \left[r(\boldsymbol{s}_{t'}, \boldsymbol{a}_{t'} \mid \boldsymbol{s}_{i,t})\right] \approx r(\boldsymbol{s}_{i,t}, \boldsymbol{a}_{i,t}) + V(\boldsymbol{s}_{i,t + 1}) \approx r(\boldsymbol{s}_{i,t}, \boldsymbol{a}_{i,t}) + \hat{V}^\pi(\boldsymbol{s}_{i,t + 1})
$$
此时训练数据可以表示为
$$
\left\{\left(\boldsymbol{s}_{i,t}, r(\boldsymbol{s}_{i,t}, \boldsymbol{a}_{i,t}) + \hat{V}^\pi(\boldsymbol{s}_{i,t + 1})\right)\right\}
$$
![](4-2.png)

# 3 Actor-Critic Algorithm

在上述讨论中, 我们有一个策略，被称为**演员（actor）**，用于生成动作，一个价值函数，被称为**评论家（critic）**，用于评估状态的价值。这种结合策略梯度和价值函数的方法被称为**演员-评论家算法（Actor-Critic Algorithms）**。我们可以得到一个简单的演员-评论家算法，即批量演员-评论家算法：
1. 从策略 $\pi_\theta$ 采样一系列轨迹；
2. 拟合 $\hat{V}^\pi(\boldsymbol{s})$；
3. 计算优势函数 $\hat{A}^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t) \approx r(\boldsymbol{s}_t, \boldsymbol{a}_t) + \hat{V}^\pi(\boldsymbol{s}_{t + 1}) - \hat{V}^\pi(\boldsymbol{s}_t)$；
4. 计算梯度 $\nabla_\theta J(\theta) \approx \sum_{i} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t}) \hat{A}^\pi(\boldsymbol{s}_{i,t}, \boldsymbol{a}_{i,t})$；
5. 更新 $\theta$。 
这里有多种选择的是第二步，我们可以拟合蒙特卡洛目标，也可以拟合自举目标。

# 4 Discount Factor
在之前的讨论中，我们只考虑了有限时间跨度的情况，如果 $T = \infty$，那么 $\hat{V}_\theta^\pi$ 可能会变得任意地大。一个简单的技巧是引入折旧因子，$\gamma \in [0, 1]$，当前奖励的权重更高，未来的奖励以几何级数递减。$\gamma$ 会改变马尔可夫决策过程的性质，类似于我们添加了一个死亡状态，每一步都有 $1 - \gamma$ 的概率进入这个死亡状态（由于我们无法离开），此后所有的奖励都是 $0$。
![](4-3.png)

我们考虑引入折旧因子后 $V$ 函数的目标有什么形式上的变化，在自举目标中我们有 
$$
y_{i,t} \approx r(\boldsymbol{s}_{i,t}, \boldsymbol{a}_{i,t}) + \gamma \hat{V}^\pi(\boldsymbol{s}_{i,t + 1})
$$
值得注意的是，在蒙特卡洛策略梯度中，我们似乎有两种选择：
从 $t$ 开始的奖励为 $\sum_{t' = t}^T \gamma^{t' - t} r(\boldsymbol{s}_{i,t'}, \boldsymbol{a}_{i,t'})$，也就是 
选择1：
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i = 1}^N \sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t}) \sum_{t' = t}^T \gamma^{t' - t} r(\boldsymbol{s}_{i,t'}, \boldsymbol{a}_{i,t'})
$$
选择2：
$$
\begin{aligned}  
\nabla_\theta J(\theta) 
&\approx \frac{1}{N} \sum_{i = 1}^N \left(\sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t})\right) \left(\sum_{t = 1}^T \gamma^{t - 1} r(\boldsymbol{s}_{i,t}, \boldsymbol{a}_{i,t})\right)\\  
&= \frac{1}{N} \sum_{i = 1}^N \sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t}) \sum_{t' = t}^T \gamma^{t' - 1} r(\boldsymbol{s}_{i,t'}, \boldsymbol{a}_{i,t'})\\  
&= \frac{1}{N} \sum_{i = 1}^N \sum_{t = 1}^T \gamma^{t - 1} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t}) \sum_{t' = t}^T \gamma^{t' - t} r(\boldsymbol{s}_{i,t'}, \boldsymbol{a}_{i,t'})  
\end{aligned}
$$
这两种选择是不同的。其中前一种与演员-评论家算法中的自举目标是一致的。后一种实际上意味着在未来的决策不重要（梯度多了一个 $\gamma^{t - 1}$），而不仅仅是当前的决策更多考虑当前奖励。通常情况下这是不合理的，我们希望的是我们的策略在未来同样有效，而不是仅仅在当前有效。

在[[Lecture 2 Introduction to Reinforcement Learning]]一节介绍无限时间跨度的情况时，我们给出了利用平均奖励的方法（也就是乘上 $1 / T$），尽管我们能够得到一个漂亮的结果，但这在计算上通常不可行，因为这需要知道对应的马尔可夫链的平稳分布。一种妥协是使用折旧因子。事实上折旧也可以视作降低方差的方式，因为奖励总和降低了。

我们可以自然地得出引入折旧因子后的演员-评论家算法：
1. 从策略 $\pi_\theta$ 采样 $\{\boldsymbol{s}_{i}, \boldsymbol{a}_{i}\}$；
2. 拟合 $\hat{V}^\pi(\boldsymbol{s})$；
3. 计算优势函数 $\hat{A}^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t) \approx r(\boldsymbol{s}_t, \boldsymbol{a}_t) + \gamma\hat{V}^\pi(\boldsymbol{s}_{t + 1}) - \hat{V}^\pi(\boldsymbol{s}_t)$；
4. 计算梯度 $\nabla_\theta J(\theta) \approx \sum_{i} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t}) \hat{A}^\pi(\boldsymbol{s}_{i,t}, \boldsymbol{a}_{i,t})$；
5. 更新 $\theta$。

这里我们简短地讨论一下演员-评论家算法中的一些设计决策。我们考虑如何利用网络表示 $V^\pi(\boldsymbol{s})$ 与 $\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})$：
- 使用两个网络，好处是简单且稳定，但是没有共享信息。 
- 共用一个网络，但是可能需要有更多的超参数调节等技巧。
![](4-4.png)

# 5 Online Actor-Critic Algorithm

## 5.1 On-Policy Actor-Critic Algorithm
在策略梯度中，我们的更新频率相对较低，也就是我们总是先收集 $N$ 个轨迹，然后进行策略更新。而演员-评论家算法中，我们可以得到一个完全 **online** 的算法，也就是每进行一次交互，就进行一次策略更新。具体过程是：
1. 采取动作 $\boldsymbol{a}_t$, 得到 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}', r)$；
2. 更新 $\hat{V}^\pi(\boldsymbol{s}) = r + \gamma \hat{V}^\pi(\boldsymbol{s}')$；
3. 计算优势函数 $\hat{A}^\pi(\boldsymbol{s}, \boldsymbol{a}) = r + \gamma \hat{V}^\pi(\boldsymbol{s}') - \hat{V}^\pi(\boldsymbol{s})$；
4. 计算梯度 $\nabla_\theta J(\theta) \approx \nabla_\theta \log \pi_\theta(\boldsymbol{a} \mid \boldsymbol{s}) \hat{A}^\pi(\boldsymbol{s}, \boldsymbol{a})$；
5. 更新 $\theta$。

注意区分 offline/ online 与 off-policy/ on-policy。offline/ online 是指在学习过程中是否与环境不断交互，而 off-policy/ on-policy 是指我们的数据是否基于当前的策略。然而这对 deep RL 来说是有许多问题的，因为仅仅使用一个样本进行更新是不稳定的，有很大的方差。我们考虑以下两个方式来解决这个问题：
- **同步并行的演员-评论家算法（synchronous parallel）**：我们有多个同步的工作线程，每个工作线程每一步都会进行一个动作，这样我们的批量大小就等于工作线程的数量。一个问题在于不同 工作线程结束的时间可能不同，因此会有一定的同步开销。
- **异步的演员-评论家算法（asynchronous）**：我们同样有多个工作线程，但它们未必同步，每一次当我们收集到批量大小个样本时，我们就进行一次更新。但是这一方式的问题是，一个批量中可能混有不同参数的样本（考虑一次更新时可能某个工作线程正在采集数据），这在数学上是不等价的，但由于单次更新参数变化有限，因此通常问题不会太大。（这就是 Asynchronous Advantage Actor-Critic，A3C 算法的核心思想）。
![](4-5.png)

## 5.2 Off-Policy Actor-Critic Algorithm (Problematic)

我们能否去除掉 on-policy 假设呢？这就引出了 off-policy 演员-评论家算法。我们此时只有一个线程，我们会有一个**回放缓冲区** $\mathcal{R}$，每一步我们都会将 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}', r)$ 存入回放缓冲区，然后我们从其中中采样进行更新。此时我们的每次更新方差就会非常小。

online 演员-评论家算法（off-policy）**（存在问题的）**：
1. 采取动作 $\boldsymbol{a}_t$，得到 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}', r)$, 储存在 $\mathcal{R}$；
2. 从 $\mathcal{R}$ 采样一个批量； 
3. 更新 $\hat{V}^\pi(\boldsymbol{s}_i)$，使用 $y_i= r_i + \gamma \hat{V}^\pi(\boldsymbol{s}_i')$ 与 $L(\phi) = \frac{1}{N} \sum_{i = 1}^N \left\|\hat{V}^\pi(\boldsymbol{s}_i) - y_i\right\|^2$，其中 $N$ 是批量大小；  
4. 计算优势函数 $\hat{A}^\pi(\boldsymbol{s}_i, \boldsymbol{a}_i) = r + \gamma \hat{V}^\pi(\boldsymbol{s}_i') - \hat{V}^\pi(\boldsymbol{s}_i)$；
5. 计算梯度 $\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_i \mid \boldsymbol{s}_i) \hat{A}^\pi(\boldsymbol{s}_i, \boldsymbol{a}_i)$；
6. 更新 $\theta$。

这个算法中有两个部分都是不正确的：
- 我们从回放缓冲区中采样的 $\boldsymbol{s}_i$ 对应的 $\boldsymbol{a}_i$ 并不是按照最新的策略采样的。这会导致我们计算的 $V^\pi$ 函数不再能够反映当前的策略。
- 基于同样的理由，我们的策略梯度也是不正确的，一个可能的方式是我们可以使用重要性采样。

## 5.3 Fixing Off-Policy Actor-Critic Algorithm
**Fixing Step 3：**
对于第一个问题, 我们可以利用 $Q$ 函数而不是 $V$ 函数，因为 $Q$ 函数包含了动作。即使 $\boldsymbol{a}_i$ 是旧的，意味着基于当前策略在 $\boldsymbol{s}_i$ 我们不再会选择 $\boldsymbol{a}_i$，但对应的 $r_i$ 依然准确，估计的 $Q$ 函数仍然正确。

我们考虑更新第 $3$ 步为：更新 $\hat{Q}^\pi(\boldsymbol{s}_i, \boldsymbol{a}_i)$ 基于 $y_i = r_i + \gamma \hat{V}^\pi(\boldsymbol{s}')$，之后做回归：
$$
L(\phi) = \frac{1}{N} \sum_{i = 1}^N \left\|\hat{Q}^\pi(\boldsymbol{s}_i, \boldsymbol{a}_i) - y_i\right\|^2
$$
但是此时我们不再学习 $V$ 了，因此我们要把目标也用 $Q$ 替换。但是注意 $Q$ 函数依赖于下一个动作，而这并不存在于 $\mathcal{R}$ 中（即使存在，也不对应于当前策略）。

注意转化 $V$ 为 $Q$ 的方式是：
$$
V^\pi(\boldsymbol{s}') = \mathbb{E}_{\boldsymbol{a}' \sim \pi(\boldsymbol{s}\mid \boldsymbol{a})}[Q^\pi(\boldsymbol{s}', \boldsymbol{a}')]
$$
一个无偏的估计是使用 $y_i = r_i + \gamma \hat{Q}_\phi^\pi(\boldsymbol{s}_i', \boldsymbol{a}_i')$，其中 $\boldsymbol{a}_i'$ 并不来自 $\mathcal{R}$，而是我们要从 $\pi_\theta(\boldsymbol{a}_i' \mid \boldsymbol{s}_i')$ 中采样。

**Fixing Step 5：**
对于第 $5$ 步，我们不能再使用缓冲中的动作 $\boldsymbol{a}_i$，而是使用基于当前策略 $\pi$ 选择的动作 $\boldsymbol{a}_i^\pi \sim \pi_\theta(\boldsymbol{a}, \boldsymbol{s}_i)$。

在实际中我们使用 $Q$ 函数：
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_i^\pi \mid \boldsymbol{s}_i) \hat{Q}^\pi(\boldsymbol{s}_i, \boldsymbol{a}_i^\pi)
$$
而不是优势函数：
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_i^\pi \mid \boldsymbol{s}_i) \hat{A}^\pi(\boldsymbol{s}_i, \boldsymbol{a}_i^\pi)
$$
这样的方式看起来会有更大的方差，但是这里方差的问题可以通过其他方式解决：$\boldsymbol{a}_i^\pi$ 是基于策略获取而不是基于环境采样的，故生成多个 $\boldsymbol{a}_i^\pi$ 成本非常低，也就是说我们可以使用多个 $\boldsymbol{a}_i^\pi$ 来减小方差。


于是经过修复后我们得到了一个可以使用的 online 演员-评论家算法（off-policy）：
1. 采取动作 $\boldsymbol{a}_t$, 得到 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}', r)$，储存在 $\mathcal{R}$；
2. 从 $\mathcal{R}$ 采样一个批量；
3. 更新 $\hat{Q}^\pi(\boldsymbol{s}_i, \boldsymbol{a}_i)$，基于 $y_i = r_i + \gamma \hat{Q}_\phi^\pi(\boldsymbol{s}_i', \boldsymbol{a}_i')$； 
4. 计算梯度 $\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_i^\pi \mid \boldsymbol{s}_i) \hat{Q}^\pi(\boldsymbol{s}_i, \boldsymbol{a}_i^\pi)$；
5. 更新 $\theta$。

一个可能的问题： $\boldsymbol{s}_i$ 并不来自于 $p_\theta(\boldsymbol{s})$，而这是我们用样本估计梯度的假设。然而我们这里没有办法解决这个问题，只能接受这一点。但是基于以下直觉，这个问题可以忽略：我们的目标是找到 $p_\theta(\boldsymbol{s})$ 上的最优策略，这里我们其实尝试找到一个更加广泛的分布上的最优策略，尽管我们做了多余的工作，但是这并不会导致我们的策略不是最优的。

注意：
- 在第 $4$ 步中我们可以有更加花哨的技巧，使用重参数化技巧，来更好地估计积分。
- 这里我们使用的是随机策略，在接下来介绍的 Q 学习中我们使用的是确定性策略。

# 6 Techniques for Reducing Bias
我们不妨回顾一下我们的两种方法：
- 演员-评论家算法：有更低的方差，但是不再是无偏的，只要评论家不是完美的。
- 策略梯度：没有偏差，但是会有很大的方差，因为使用的是单样本估计。

在策略梯度算法中我们主要关注了如何减小方差，而在演员-评论家算法中我们将关注如何减小以至于消除偏差。
首先我们可以从策略梯度出发，得到两个利用评论家在保持无偏的同时实现很低的方差的方法。
## 6.1 Critics as state-dependent baselines
由于如果基线仅依赖于状态，那么这个估计值依然是无偏的，我们的梯度估计式为：
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i = 1}^{N} \sum_{t = 1}^{T} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_i \mid \boldsymbol{s}_i) \left(\left(\sum_{t' = t}^{T} \gamma^{t' - t} r(\boldsymbol{s}_{i,t'}, \boldsymbol{a}_{i,t'})\right) - \hat{V}_\phi^\pi(\boldsymbol{s}_i)\right)
$$
此时我们使用了评论家作为基线，我们在保持无偏的同时得到了更低的方差。
我们能否利用除了 $\boldsymbol{s}$ 以外的更多信息来进一步降低方差呢？答案是可以的，但是过程会更加复杂。

## 6.2 Control variates: action-dependent baselines
由于未来奖励更加接近 $\hat{Q}_\phi^\pi(\boldsymbol{s}, \boldsymbol{a})$，用其作为基线理论上会有更低的方差。然而由于评论家是不是完美的，会引入偏差，考虑梯度估计式：
$$
\frac{1}{N} \sum_{i = 1}^{N} \sum_{t = 1}^{T} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_i \mid \boldsymbol{s}_i) \left(\hat{Q}_{i,t} - Q_\phi^\pi(\boldsymbol{s}_i, \boldsymbol{a}_i)\right)+ \frac{1}{N} \sum_{i = 1}^{N} \sum_{t = 1}^{T} \nabla_\theta \mathbb{E}_{\boldsymbol{a} \sim \pi_\theta(\boldsymbol{a}_{t} \mid \boldsymbol{s}_{i,t})} \left[Q_\phi^\pi(\boldsymbol{s}_{i,t}, \boldsymbol{a}_t)\right]
$$
由于此时基线依赖于动作，第二项不再是 $0$，但可以利用策略生成很多的动作来估计，然后减去这一项的值，就得到了无偏的梯度，并且两项的方差都很低（Q-Prop algorithm）。

## 6.3 N-step returns
在原先的优势函数中，我们采用自举（bootstrap）的的方式，也就是 
$$
\hat{A}^\pi_C(\boldsymbol{s}_t, \boldsymbol{a}_t) = r(\boldsymbol{s}_t, \boldsymbol{a}_t) + \gamma \hat{V}^\pi(\boldsymbol{s}_{t + 1}) - \hat{V}^\pi(\boldsymbol{s}_t)
$$
其中 $C$ 表示评论家。在策略梯度中我们使用的是蒙特卡洛的方式，也就是 
$$
\hat{A}^\pi_{MC}(\boldsymbol{s}_t, \boldsymbol{a}_t) = \sum_{t' = t}^T \gamma^{t' - t} r(\boldsymbol{s}_{t'}, \boldsymbol{a}_{t'})
$$
我们可以在两者中间取一个折中，也就是多步回报（$n$\-step return）。 我们知道靠近当前的估计的方差较小，因此我们就在这个方差变得过大之前使用采样，在这之后的的部分使用评论家。于是优势函数为：
$$
\hat{A}^\pi_{n}(\boldsymbol{s}_t, \boldsymbol{a}_t) = \sum_{t' = t}^{t + n} \gamma^{t' - t} r(\boldsymbol{s}_{t'}, \boldsymbol{a}_{t'}) + \gamma^n \hat{V}^\pi(\boldsymbol{s}_{t + n}) - \hat{V}^\pi(\boldsymbol{s}_t)
$$
通常 $n > 1$ 效果更好。$n$ 越大则偏差越小，但是方差会增大。
![](4-6.png)

## 6.4 Generalized advantage estimation
我们为什么一定要选择一个 $n$ 呢？我们其实可以加权所有的 $n$ 的情况，也就是构造一个 **广义优势估计（Generalized Advantage Estimation，GAE）**：
$$
\hat{A}^\pi_{GAE}(\boldsymbol{s}_t, \boldsymbol{a}_t) = \sum_{n = 1}^{\infty} w_n \hat{A}^\pi_{n}(\boldsymbol{s}_t, \boldsymbol{a}_t)
$$
一个常见的方式是使用 $w_n \sim \lambda^{n - 1}$，即指数衰减。于是我们可以化简得到：
$$
\hat{A}^\pi_{GAE}(\boldsymbol{s}_t, \boldsymbol{a}_t) = \sum_{t' = t}^\infty (\gamma \lambda)^{t' - t} \left(r(\boldsymbol{s}_{t'}, \boldsymbol{a}_{t'}) + \gamma \hat{V}^\pi(\boldsymbol{s}_{t' + 1}) - \hat{V}^\pi(\boldsymbol{s}_{t'})\right)
$$
于是我们可以使用 $\lambda$ 来控制偏差-方差权衡。不难注意到如果使用广义优势估计， $\gamma$ 也扮演着控制偏差-方差权衡的角色。 
  
无论是多步回报还是广义优势估计，都只适用于 on-policy 的情况，因为 off-policy 时，计算 
$$
\sum_{t' = t}^{T} \gamma^{t' - t} r(\boldsymbol{s}_{t'}, \boldsymbol{a}_{t'})
$$显然不在对应当前策略的优势。

# 7 Summary
在本节中, 我们:
- 介绍了演员-评论家算法：如何设置目标进行策略评估，以及如何使用评论家作为基线来减小方差。
- 引出了折旧因子的概念，将这一概念应用到策略梯度与演员-评论家算法中。
- 引出了 online 算法的概念，介绍 online 演员-评论家算法。
- 介绍了如何得到 off-policy 的演员-评论家算法。
- 介绍如何利用评论家作为基线，以及多步回报与广义优势估计来进一步减小方差。