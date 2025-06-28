本部分是离线强化学习的第二部分内容，我们会介绍现代的一些离线强化学习算法，它们采用有效的方法来解决分布偏移问题。
## 1 Recap of offline RL
回顾一下对离线强化学习的范式：
- $\mathcal{D} = \{(\boldsymbol{s}_i, \boldsymbol{a}_i, r_i, \boldsymbol{s}_i')\}$；
- $\boldsymbol{s} \sim d^{\pi_\beta}(\boldsymbol{s})$；
- $\boldsymbol{a} \sim \pi_\beta(\boldsymbol{a} \mid \boldsymbol{s})$；
- $\boldsymbol{s}' \sim p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})$；
- $r \gets r(\boldsymbol{s}, \boldsymbol{a})$。

这里 $\pi_\beta$ 是行为策略，通常并不知道它具体是如何收集的。
![](14-1.png)

在前面的分析中，我们介绍了离线强化学习中核心的问题：分布偏移。在这里依旧考虑 Q 学习，我们反复进行的[[Concepts#20 贝尔曼备份 (Bellman Backup)|贝尔曼备份 (Bellman Backup)]]可以写作
$$
\pi_{new} = \arg\max_\pi \mathbb{E}_{\boldsymbol{a} \sim \pi(\boldsymbol{a}\mid \boldsymbol{s})} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right]
$$
在这个过程中，$Q$ 函数不断拟合对抗分布上的结果，从而使得 $Q$ 函数不断被高估。

在前面一节中，我们讨论的一些经典算法并没有解决这个问题，它们使用表示能力有限的线性模型使得分布偏移不会是一个严重的问题，但是在如果想要开发出能够实际应用的离线深度强化学习算法，我们必须要解决这一个问题。

## 2 Policy constraint methods
### 2.1 Explicit policy constraint methods
一个直观的想法是， $\pi_{new}$ 选取太自由了，这使得贝尔曼更新中可能会引入一些完全不合理的分布来愚弄 $Q$ 函数。一个直观的做法是将策略和 $Q$ 函数分开来，并限制策略的选择。

具体来说，循环地进行以下两个步骤：
1. 策略评估：固定策略，估计 $Q$ 函数：$$Q(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{a'} \sim \pi_{new}(\boldsymbol{a'} \mid \boldsymbol{s'})} \left[Q(\boldsymbol{s'}, \boldsymbol{a'})\right]$$
2. 策略改进：固定 $Q$ 函数，优化策略，但是这个过程中对策略有一个约束：$$\pi_{new} = \arg\max_\pi \mathbb{E}_{\boldsymbol{a} \sim \pi(\boldsymbol{a}\mid \boldsymbol{s})} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right] \text{ s.t. } D_{KL}(\pi \mid \pi_\beta) \leq \epsilon$$
![](14-2.png)
注意：这是一个很早的想法，已经得到了充分的研究。这一方法给我们提供了一个进行改进的基础，但是依然有很多值得解决的问题。

问题1：通常并不知道 $\pi_\beta$ 是什么，这意味着如果要应用这种策略约束的方法，我们需要利用行为克隆来估计 $\pi_\beta$，或者需要有一种处理约束的巧妙方法，通过 $\pi_\beta$ 的样本进行估计。通常后者效果更好，但它们都是合理的选择。

问题2：这一处理方式有些过于悲观，但同时又不够悲观。
- 过于悲观：策略会被限制在一个非常小的空间内，无法有效地改进（例如 $\pi_\beta$ 是完全随机策略，但是我们并不应该期望学到的 $\pi$ 也是非常接近随机的）；
- 不够悲观：两个 KL 散度意义下接近的分布，依然可能存在着一些错误非常大的地方，这依然给我们愚弄 $Q$ 函数的机会。

上述的方法其实属于显式策略约束方法，也就是会有一个对 $\pi_{new}$ 的显式约束。这里继续以 KL 散度约束为例，给出一些具体的方法：
1. 直接修改演员目标：对于 KL 散度有$$D_{KL}(\pi \parallel \pi_\beta) = \mathbb{E}_\pi \left[\log \pi(\boldsymbol{a} \mid \boldsymbol{s}) - \log \pi_\beta(\boldsymbol{a} \mid \boldsymbol{s})\right] = -\mathbb{E}_\pi \log \pi_\beta(\boldsymbol{a} \mid \boldsymbol{s}) - \mathcal{H}(\pi)$$于是修改目标为 $$\theta \gets \arg\max_\theta \mathbb{E}_{\boldsymbol{s} \sim D}\left[ \mathbb{E}_{\boldsymbol{a} \sim \pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})} \left[Q(\boldsymbol{s}, \boldsymbol{a}) + \lambda \log \pi_\beta(\boldsymbol{a} \mid \boldsymbol{s})\right] + \lambda \mathcal{H}(\pi(\boldsymbol{a} \mid \boldsymbol{s}))\right]$$这里的 $\lambda$ 是拉格朗日乘子（这是对约束优化问题的一种处理方式），也可以利用[[Concepts#14 对偶梯度下降（Dual Gradient Descent）|对偶梯度下降（Dual Gradient Descent）]]来进行优化，也可以选择将 $\lambda$ 作为一个正则化参数，此时受到影响的只有演员更新，而 $Q$ 函数不受影响。
2. 修改奖励函数：使用$$\tilde{r}(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) - D_{KL}(\pi(\boldsymbol{a} \mid \boldsymbol{s}) \parallel \pi_\beta(\boldsymbol{a} \mid \boldsymbol{s}))$$来替代奖励函数，这里的好处是能够同样惩罚那些未来会导致 KL 散度增大的动作，因为 Q 函数也会受到这一项的影响。

另一种可能的约束条件是支持约束： $\pi(\boldsymbol{a} \mid \boldsymbol{s}) \geq 0$ 仅当 $\pi_\beta(\boldsymbol{a} \mid \boldsymbol{s}) \geq \epsilon$。这可以利用 **Maximum Mean Discrepancy (MMD)** 来进行近似。这里的代价是非常难实现，但是通常这更接近我们想要的。

参见：Wu, Tucker, Nachum. Behavior Regularized Offline Reinforcement Learning. '19

### 2.2 Implicit policy constraint methods
实际上可以将原先的显式约束转化为对 $\pi_{new}$ 的隐式约束。利用对偶性，我们的问题
$$
\max_\pi \mathbb{E}_{\boldsymbol{a} \sim \pi(\boldsymbol{a}\mid \boldsymbol{s})} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right] \text{ s.t. } D_{KL}(\pi \mid \pi_\beta) \leq \epsilon
$$
可以转化为
$$
\pi^\ast(\boldsymbol{s} \mid \boldsymbol{a}) = \frac{1}{Z(\boldsymbol{s})} \pi_\beta(\boldsymbol{a} \mid \boldsymbol{s}) \exp\left(\frac{1}{\lambda} A^\pi(\boldsymbol{s}, \boldsymbol{a})\right)
$$
考虑对 KL 散度展开为
$$
D_{KL}(\pi \mid \pi_\beta) = \mathbb{E}_{\boldsymbol{a} \sim \pi(\boldsymbol{a}\mid \boldsymbol{s})} [\log \pi(\boldsymbol{a}\mid \boldsymbol{s}) - \log \pi_\beta(\boldsymbol{a}\mid \boldsymbol{s})]
$$
并利用概率分布的等式约束 
$$
\int \pi(\boldsymbol{a} \mid \boldsymbol{s}) \text{d}\boldsymbol{a} = 1
$$
得到拉格朗日函数
$$
\begin{aligned} L(\pi, \lambda, \mu) &= \mathbb{E}_{\boldsymbol{a} \sim \pi(\boldsymbol{a}\mid \boldsymbol{s})} [Q(\boldsymbol{s}, \boldsymbol{a}) - \lambda\log \pi(\boldsymbol{a}\mid \boldsymbol{s}) + \lambda\log \pi_\beta(\boldsymbol{a}\mid \boldsymbol{s})] + \lambda\epsilon - \mu\left(\int \pi(\boldsymbol{a}\mid \boldsymbol{s}) \text{d}\boldsymbol{a} - 1\right)\\ &= \int \pi(\boldsymbol{a} \mid \boldsymbol{s})\left[Q(\boldsymbol{s}, \boldsymbol{a}) - \lambda\log \pi(\boldsymbol{a}\mid \boldsymbol{s}) + \lambda\log \pi_\beta(\boldsymbol{a}\mid \boldsymbol{s})\right]\text{d}\boldsymbol{a} + \lambda \epsilon - \mu \int \pi(\boldsymbol{a}\mid \boldsymbol{s})\text{d}\boldsymbol{a} + \mu\\ \end{aligned}
$$
考虑将上式对 $\pi(\boldsymbol{a}\mid \boldsymbol{s})$ 求[[Concepts#23 变分导数 (Variational Derivative)|变分导数 (Variational Derivative)]]并令其为零
$$
\frac{\partial L(\pi, \lambda, \mu)}{\partial \pi(\boldsymbol{a} \mid \boldsymbol{s})} = Q(\boldsymbol{s}, \boldsymbol{a}) - (\lambda + \lambda \log \pi(\boldsymbol{a}\mid \boldsymbol{s})) + \lambda \log \pi_\beta(\boldsymbol{a}\mid \boldsymbol{s}) - \mu = 0
$$
从而
$$
\pi(\boldsymbol{a} \mid \boldsymbol{s}) = \pi_\beta(\boldsymbol{a}\mid \boldsymbol{s})\exp\left(\frac{Q(\boldsymbol{s}, \boldsymbol{a}) - \mu - \lambda}{\lambda}\right)
$$
利用归一化条件可以得到
$$
\int \pi_\beta(\boldsymbol{a}\mid \boldsymbol{s})\exp\left(\frac{Q(\boldsymbol{s}, \boldsymbol{a}) - \mu - \lambda}{\lambda}\right)\text{d}\boldsymbol{a} = 1
$$
定义归一化常量
$$
Z(\boldsymbol{s}) = \exp\left(\frac{\mu + \lambda - V(\boldsymbol{s})}{\lambda}\right)
$$
可得
$$
\pi^\ast(\boldsymbol{a} \mid \boldsymbol{s}) = \frac{1}{Z(\boldsymbol{s})} \pi_\beta(\boldsymbol{a}\mid \boldsymbol{s})\exp\left(\frac{A(\boldsymbol{s}, \boldsymbol{a})}{\lambda}\right)
$$

直觉：$\lambda$ 趋近于 $0$ 时，相当于在使用贪心策略，而如果 $\lambda$ 是较大，那么此时依然会让有优势的动作概率高，但是那些在 $\pi_\beta$ 中概率低的动作会被很大限制，从而缓解分布偏移问题。

目前将一步更新后的 $\pi^\ast$ 用 $\pi_\beta$ 表示了出来，但是只有 $\pi_\beta$ 的样本。可以通过以下转化来通过样本估计 $\pi^\ast$：我们希望新策略接近于 $\pi^\ast$，因此可以使用 KL 散度，要求
$$
\min_{\pi} D_{KL}(\pi^* | \pi) = \min_{\pi} \mathbb{E}_{\boldsymbol{a} \sim \pi^\ast(\cdot|\boldsymbol{s})} [\log \pi^\ast(\boldsymbol{a}|\boldsymbol{s}) - \log \pi(\boldsymbol{a}|\boldsymbol{s})]
$$
忽略其中与 $\pi$ 无关的项,，可以得到新的目标为
$$
\max_{\pi} \mathbb{E}_{\boldsymbol{a} \sim \pi^*(\cdot|\boldsymbol{s})} [\log \pi(\boldsymbol{a}|\boldsymbol{s})]
$$
由于样本不来自于 $\pi^\ast$，这里使用重要性采样得到
$$
\max_{\pi} \mathbb{E}_{\boldsymbol{a} \sim \pi_\beta(\cdot|\boldsymbol{s})} \left[\frac{\pi^\ast(\boldsymbol{a}|\boldsymbol{s})}{\pi_\beta(\boldsymbol{a}|\boldsymbol{s})} \log \pi(\boldsymbol{a}|\boldsymbol{s})\right]
$$
代入 $\pi^\ast$ 的表达式，可以得到优化问题
$$
\max_{\pi} \mathbb{E}_{\boldsymbol{a} \sim \pi_\beta(\cdot|\boldsymbol{s})} \left[\log \pi(\boldsymbol{a}|\boldsymbol{s})\frac{1}{Z(\boldsymbol{s})} \exp\left(\frac{1}{\lambda} A(\boldsymbol{s},\boldsymbol{a})\right)\right]
$$
整理即可得到
$$
\pi_{new}(\boldsymbol{a} \mid \boldsymbol{s}) = \arg\max_\pi \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}) \sim \pi_{\beta}} \left[\log \pi(\boldsymbol{a} \mid \boldsymbol{s})\frac{1}{Z(\boldsymbol{s})} \exp\left(\frac{1}{\lambda} A^{\pi_{old}}(\boldsymbol{s},\boldsymbol{a})\right)\right]
$$
这相当于是以权重
$$
w(\boldsymbol{s}, \boldsymbol{a}) = \frac{1}{Z(\boldsymbol{s})} \exp\left(\frac{1}{\lambda} A^{\pi_{old}}(\boldsymbol{s},\boldsymbol{a})\right)
$$
的加权行为克隆，换言在模仿数据集中的数据，但是模仿好的动作更多。通常情况下，在实际算法中会忽略掉 $Z(\boldsymbol{s})$。

稍作整理可以得到优势加权演员-评论家（Advantage-Weighted Actor-Critic，AWAC）算法，在实际中使用梯度下降交替地更新：
$$
L_C(\phi) = \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}') \sim \mathcal{D}} \left[\left( Q_{\phi}(\boldsymbol{s}, \boldsymbol{a}) - (r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{a}' \sim \pi_\theta(\boldsymbol{a'} \mid \boldsymbol{s}')} \left[Q_\phi(\boldsymbol{s}', \boldsymbol{a}')\right])\right)^2\right]
$$
$$
L_A(\theta) = \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}) \sim \mathcal{D}} \left[\log \pi_\theta(\boldsymbol{a} \mid \boldsymbol{s}) \exp\left(\frac{1}{\lambda} A^{\pi_{\phi}}(\boldsymbol{s},\boldsymbol{a})\right)\right]
$$
其中 $\lambda$ 是一个温度参数。当 $\lambda$ 趋于 $0$，就得到贪心策略，而当 $\lambda$ 趋于 $\infty$，就得到 $\pi_\beta$。

理论上我们也可以使用么蒙特卡洛方法来估计奖励，此时对应的就是优势加权回归。

一个值得注意的是，上述转化的过程在优化问题的意义上是等价的，换言之，收敛到的结果满足 
$$
\max_\pi \mathbb{E}_{\boldsymbol{a} \sim \pi(\boldsymbol{a}\mid \boldsymbol{s})} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right] \text{ s.t. } D_{KL}(\pi \mid \pi_\beta) \leq \epsilon
$$
中的 KL 散度约束，然而没有任何保证说明了在转化后的问题的优化过程中， $\pi_\theta$ 满足这一约束，因此依然会出现查询分布外动作的问题。具体来说，在更新 $\phi$ 和计算优势的过程中也使用了 $\pi_\theta$ 下的期望，这两个地方都有可能出现上述问题。

参见：
- Peng\*, Kumar\*, Levine. Advantage-Weighted Regression. '19  
- Nair, Dalal, Gupta, Levine. Accelerating Online Reinforcement Learning with Offline Datasets. '20  

### 2.3 Implicit Q-learning (IQL)
回顾刚才介绍的优势加权演员-评论算法，事实上我们依然无法完全避免分布外动作。
如果将
$$
\mathbb{E}_{\boldsymbol{a}' \sim \pi_{new}} \left[Q(\boldsymbol{s}', \boldsymbol{a}')\right]
$$
用一个独立的网络 $V(\boldsymbol{s}')$ 来表示，更新 $Q$ 函数的过程可以表示为
$$
Q(\boldsymbol{s}, \boldsymbol{a}) \gets r(\boldsymbol{s}, \boldsymbol{a}) + \gamma V(\boldsymbol{s}')
$$
此时在更新 $Q$ 函数时就完全避免了查询分布外动作。类似地，如果依然使用演员，那么优势的计算也不再需要查询分布外动作：
$$
A(\boldsymbol{s}, \boldsymbol{a}) = Q(\boldsymbol{s}, \boldsymbol{a}) - V(\boldsymbol{s})
$$

需要以某种方式来训练价值函数网络，同时又需要避免查询分布外动作，只能利用数据集中的状态，因此我们需要也只能考虑如下形式的更新方式
$$
V \gets \arg\min_V \frac{1}{N} \sum_{i = 1}^{N} l(V(\boldsymbol{s}_i), Q(\boldsymbol{s}_i, \boldsymbol{a}_i))
$$
这一做法的直觉是，尽管在较大的状态空间中，每一个状态只会出现一次，自然也只存在一个动作，但是有可能有一些相近的状态，其中选择了其他的动作。由于泛化性，可以将这些相似的状态视作同一个，此时在这个状态处，存在着多个数据集中出现的动作。

值得注意的是，上述每一个状态附近的动作有好有坏，它们都来源于 $\pi_\beta$，如果使用均方误差，则学到的会是 $\mathbb{E}_{\pi_\beta}\left[Q(\boldsymbol{s}, \boldsymbol{a})\right]$，结合价值函数的含义，价值函数应当对应于当前的 $Q$ 函数，而当前的 $Q$ 函数应当学习到了数据中 $\boldsymbol{s}$ 附近所有 $Q(\boldsymbol{s}, \boldsymbol{a})$ 的一个接近上界。此时可以使用分位数损失来进行更新：
$$
l_2^\tau(x) = \begin{cases} (1 - \tau) x^2, & x > 0\\ \tau x^2, & x \leq 0 \end{cases}
$$
适当地选择这里的 $\tau$，可以更多地惩罚负面错误。
![](14-3.png)
注意：
- 此时避免了计算 $\pi_{new}$ 下的 $Q$ 函数期望，也就避免了查询分布外样本，也就避免了之前的高估问题；
- 如果简单使用均方误差，由于没有使用任何新策略的状态，我们会低估，但是使用[[Concepts#24 分位数损失（Expectile Loss）|分位数损失（Expectile Loss）]]并调整 $\tau$ 的取值，可以使得 $Q$ 函数更加接近于最优 $Q$ 函数。
 

也可以从以下角度来直观地理解这里的做法：
考虑支撑集
$$
\Omega(\boldsymbol{s}) = \{\boldsymbol{a}: \pi_\beta(\boldsymbol{a} \mid \boldsymbol{s}) \geq \epsilon\}
$$
当使用很大的 $\tau$ 时，这里的更新类似于
$$
V(\boldsymbol{s}) \gets \max_{\boldsymbol{a} \in \Omega(\boldsymbol{s})} Q(\boldsymbol{s}, \boldsymbol{a})
$$
与均方误差不同的是，没有复制数据集中的策略 $\pi_\beta$，而是在每个状态上独立地选择最优的分布内动作。
这一思想可以被用来构建隐式 Q 学习（Implicit Q-learning，IQL）算法：
将策略 $\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})$，$Q$ 函数 $Q_\phi(\boldsymbol{s}, \boldsymbol{a})$ 和价值函数 $V_\psi(\boldsymbol{s})$ 分别用 $\theta$， $\phi$， $\psi$ 参数化，进而得到如下更新方式：
- 更新价值函数：$$\begin{aligned} L_V(\psi) &= \mathbb{E}{(\boldsymbol{s}, \boldsymbol{a}) \sim \mathcal{D}} \left[l_2^\tau(V\psi(\boldsymbol{s}), Q_\phi(\boldsymbol{s}, \boldsymbol{a}))\right]\\ L_Q(\phi) &= \mathbb{E}{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}') \sim \mathcal{D}} \left[(Q\phi(\boldsymbol{s}, \boldsymbol{a}) - (r(\boldsymbol{s}, \boldsymbol{a}) + \gamma V_\psi(\boldsymbol{s}')))^2\right]\end{aligned}$$
- 更新演员：$$L_A(\theta) = \mathbb{E}_{\boldsymbol{s} \sim \mathcal{D}} \left[\log \pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})\exp\left(\frac{1}{\lambda} A^{\pi_{\phi}}(\boldsymbol{s},\boldsymbol{a})\right)\right]$$

参见：Kostrikov, Nair, Levine. Offline Reinforcement Learning with Implicit Q-Learning. '21

## 3 Conservative Q-learning
### 3.1 Introduction
接下来介绍的一类方法是保守 Q 学习（Conservative Q-learning），相较于在策略上进行约束，这里会在 $Q$ 函数上限制那些高估的动作。

考虑一个新的目标：
$$
\begin{aligned} \hat{Q}^\pi = \arg\min_Q \max_\mu &\,\,\alpha \mathbb{E}_{\boldsymbol{s} \sim \mathcal{D}, \boldsymbol{a} \sim \mu(\boldsymbol{a} \mid \boldsymbol{s})} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right]\\  &+ \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}') \sim \mathcal{D}} \left[(Q(\boldsymbol{s}, \boldsymbol{a}) - (r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{a}' \sim \pi(\boldsymbol{a}' \mid \boldsymbol{s}')} \left[Q(\boldsymbol{s}', \boldsymbol{a}')\right]))^2\right]\\ \end{aligned}
$$
其中后一项为常规的贝尔曼误差，而前一项则是显式地将过大的 $Q$ 值降低，可以证明当 $\alpha$ 足够大时，则 $\hat{Q}^\pi$ 是 $Q^\pi$ 的一个下界。
![](14-4.png)
上面的目标有些过于悲观了，因为拉低了所有高的 $Q$ 值，即使它们是在分布内，因此最后无法得到真正的最优 $Q$ 函数，其实可以得到一个更好的目标：
$$
\begin{aligned} \hat{Q}^\pi = \arg\min_Q \max_\mu &\,\,\alpha \mathbb{E}_{\boldsymbol{s} \sim \mathcal{D}, \boldsymbol{a} \sim \mu(\boldsymbol{a} \mid \boldsymbol{s})} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right] - \alpha \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}) \sim \mathcal{D}} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right] + \mathcal{R}(\mu)\\  &+ \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}') \sim \mathcal{D}} \left[(Q(\boldsymbol{s}, \boldsymbol{a}) - (r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{a}' \sim \pi(\boldsymbol{a}' \mid \boldsymbol{s}')} \left[Q(\boldsymbol{s}', \boldsymbol{a}')\right]))^2\right]\\ \end{aligned}
$$
分析：
- 这里新添加的一项 $- \alpha \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}) \sim \mathcal{D}} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right]$ 能够拉高那些在 $\mathcal{D}$ 内的 $Q$ 值；
- 原先的 $\alpha \mathbb{E}_{\boldsymbol{s} \sim \mathcal{D}, \boldsymbol{a} \sim \mu(\boldsymbol{a} \mid \boldsymbol{s})} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right]$ 依旧可以拉低那些高估的分布外 $Q$ 值； 
- $\mathcal{R}(\mu)$ 是一个正则化项，在之后介绍。

一个很重要的观察是：
- 估计偏离合理估计越多，那么前两项的综合作用越强；
- 如果估计就是合理的，例如高 $Q$ 值都在分布内，那么前两项的作用就可以相互抵消。

此时相应的理论保证是，对于任意的策略 $\pi$ 和状态 $\boldsymbol{s} \in \mathcal{D}$，有
$$
\mathbb{E}_{\pi(\boldsymbol{a} \mid \boldsymbol{s})}\left[\hat{Q}^\pi(\boldsymbol{s}, \boldsymbol{a})\right] \leq \mathbb{E}_{\pi(\boldsymbol{a} \mid \boldsymbol{s})}\left[Q^\pi(\boldsymbol{s}, \boldsymbol{a})\right]
$$
这已经足够了。

### 3.2 CQL (Conservative Q-learning)
整理给出保守 Q 学习（Conservative Q-learning，CQL）算法，之前定义的目标记作 $\mathcal{L}_{CQL}(\hat{Q}^\pi)$，保守 Q 学习的算法流程如下：
1. 使用 $\mathcal{D}$ 利用 $\mathcal{L}_{CQL}(\hat{Q}^\pi)$ 更新 $\hat{Q}^\pi$；
2. 更新策略 $\pi$。

这里策略 $\pi$ 的更新还没有给出，具体来说：
- 如果动作是离散的，使用$$\pi(\boldsymbol{a} \mid \boldsymbol{s}) = \begin{cases} 1, & \text{if } \boldsymbol{a} = \arg\max_{\boldsymbol{a}'} \hat{Q}^\pi(\boldsymbol{s}, \boldsymbol{a}')\\ 0, & \text{otherwise}. \end{cases}$$  
- 如果动作是连续的，可以使用一个独立的模型表示策略（更像是一个演员-评论家的结构）：$$\theta \gets \theta + \alpha \nabla_\theta \sum_{i} \mathbb{E}_{\boldsymbol{a} \sim \pi_\theta(\boldsymbol{a} \mid \boldsymbol{s}_i)} \left[\hat{Q}(\boldsymbol{s}_i, \boldsymbol{a})\right]$$

### 3.3 Regularizer
注意上述算法中由于没有指定正则化项的具体形式，算法是不完整的。而定义的目标其实是一族目标，每一个具体的 $\mathcal{R}(\mu)$ 都可以实例化出一个具体的算法。

例如，一个常见的正则化项是
$$
\mathcal{R}(\mu) = \mathbb{E}_{\boldsymbol{s} \sim \mathcal{D}} \left[\mathcal{H}(\mu(\cdot \mid \boldsymbol{s}))\right]
$$
这是一个最大熵正则化。

此时把目标中和 $\mu$ 有关的部分提出来，并且将对 $\boldsymbol{s}$ 的期望移到最外面，考虑
$$
\begin{aligned} \arg\max_{\mu(\cdot\mid\boldsymbol{s})} \mathbb{E}_{\boldsymbol{a} \sim \mu(\boldsymbol{a} \mid \boldsymbol{s})}\left[Q(\boldsymbol{s}, \boldsymbol{a})\right] + \mathcal{H}(\mu(\cdot \mid \boldsymbol{s})) &= \arg\max_{\mu(\cdot\mid\boldsymbol{s})} \mathbb{E}_{\boldsymbol{a} \sim \mu(\boldsymbol{a} \mid \boldsymbol{s})}\left[Q(\boldsymbol{s}, \boldsymbol{a}) - \log \mu(\boldsymbol{a} \mid \boldsymbol{s})\right], \end{aligned}
$$
前面在优势加权演员-评论家算法推导中已经见过了如何利用拉格朗日乘子法和变分导数来解决这类带约束的优化问题（待优化的是函数 $\mu$ 满足归一化条件），不妨作为练习，可得最优的
$$
\mu(\boldsymbol{a} \mid \boldsymbol{s}) \propto \exp(Q(\boldsymbol{s}, \boldsymbol{a}))
$$
也就有
$$
\mu(\boldsymbol{a} \mid \boldsymbol{s}) = \frac{\exp(Q(\boldsymbol{s}, \boldsymbol{a}))}{\sum_{\boldsymbol{a}'} \exp(Q(\boldsymbol{s}, \boldsymbol{a}'))}
$$
因此可知
$$
\max_{\mu(\cdot\mid\boldsymbol{s})} \mathbb{E}_{\boldsymbol{a} \sim \mu(\boldsymbol{a} \mid \boldsymbol{s})}\left[Q(\boldsymbol{s}, \boldsymbol{a}) - \log \mu(\boldsymbol{a} \mid \boldsymbol{s})\right] = \log \sum_{\boldsymbol{a}} \exp(Q(\boldsymbol{s}, \boldsymbol{a}))
$$

注意：
- 在离散情况下，右侧所有 $\exp(Q(\boldsymbol{s}, \boldsymbol{a}))$ 的求和是可以计算的，于是原目标所有与 $\mu$ 有关的部分就都用 $Q$ 表示出来了，此时不再需要显式地构造 $\mu$。
- 在连续情况下，没办法直接计算这个求和。此时可以使用重要性采样，在另一个容易处理的分布 $\rho(\boldsymbol{a} \mid \boldsymbol{s})$ 下采样。从而得到$$\mathbb{E}_{\boldsymbol{a} \sim \mu(\boldsymbol{a} \mid \boldsymbol{s})}\left[Q(\boldsymbol{s}, \boldsymbol{a}) - \log \mu(\boldsymbol{a} \mid \boldsymbol{s})\right] = \mathbb{E}_{\boldsymbol{a} \sim \rho(\boldsymbol{a} \mid \boldsymbol{s})}\left[\frac{\mu(\boldsymbol{a} \mid \boldsymbol{s})}{\rho(\boldsymbol{a} \mid \boldsymbol{s})}\left(Q(\boldsymbol{s}, \boldsymbol{a}) - \log \mu(\boldsymbol{a} \mid \boldsymbol{s})\right)\right]$$再使用 $\exp(Q(\boldsymbol{s}, \boldsymbol{a}))$ 作为 $\mu(\boldsymbol{a} \mid \boldsymbol{s})$ 即可在忽略掉常数的情况下计算出上式。

## 4 Model-Based Offline RL
目前我们介绍的一些离线强化学习算法都是无模型的，也可以考虑基于模型的方法。

在无模型类型的算法中，事实上对于每一个状态动作对，只有一个转移结果，这会带来高的方差，而一个模型就可以回答“如果…… 会怎样”的问题：如果转移到了另一个状态会怎么样？

以贝尔曼备份为例，如果有一个模型，那么可以计算
$$
Q(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{s}' \sim p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})} \left[\arg\max_{\boldsymbol{a}'}Q(\boldsymbol{s}', \boldsymbol{a}')\right]
$$
而在无模型的情况下，只能使用一个样本来估计这个期望，也就是
$$
Q(\boldsymbol{s}, \boldsymbol{a}) \approx r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \arg\max_{\boldsymbol{a}'} Q(\boldsymbol{s}', \boldsymbol{a}')
$$

在之前介绍的基于模型的强化学习中，我们主要使用 Dyna 类型的算法。具体来说，我们会使用一个拼接的轨迹， 一部分是真实环境中的长轨迹，另一部分是从这些轨迹上的状态出发，基于模型产生的短轨迹。
![](14-5.png)
然而这样的方式在离线情况下有一些问题，由于无法再收集新的数据来更新模型，因此模型可能会在分布外状态给出很差的预测，例如给出一个很高的奖励，那么这就给了我们的策略可乘之机，我们的策略会学到依据模型转移到这些分布外的奖励很高的状态，最终我们同样会造成严重高估的 Q 值。

### 4.1 MOPO: Model-Based Offline Policy Optimization
这里介绍一种直接修改奖励函数的算法。考虑
$$
\tilde{r}(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) - \lambda u(\boldsymbol{s}, \boldsymbol{a})
$$
这里 $u(\boldsymbol{s}, \boldsymbol{a})$ 是一个不确定性惩罚。在这个更改后，就可以使用任何基于模型的强化学习算法。

直觉：不确定性惩罚的作用直观来说就是惩罚那些利用分布外状态的行为，使得它们不在值得。

设计抉择：通常会使用在基于模型的方法中介绍的衡量不确定性的方法，例如模型集成，使用不同模型的预测的分歧等指标来作为不确定性指标。

通常情况下，为了得到这一算法一些相对理论的保证， $u(\boldsymbol{s}, \boldsymbol{a})$ 需要满足一些要求，这会在后面的理论分析中介绍。

参见：
- Yu\*, Thomas\*, Yu, Ermon, Zou, Levine, Finn, Ma. MOPO: Model-Based Offline Policy Optimization. '20  
- Kidambi et al., MOReL : Model-Based Offline Reinforcement Learning. '20 (concurrent)  

### 4.2 Theoretical Analysis on MOPO
接下来考虑这背后的一些理论分析。

假设：
1. 价值模型足够强大，能够表达出真实的价值函数； 
2. 模型误差被 $u(\boldsymbol{s}, \boldsymbol{a})$ 约束住（在例如[[Concepts#2 总变差距离（Total Variation Distance）|总变差距离（Total Variation Distance）]]下）。

**Theorem 1**. 在上述假设下，有
$$
\eta_M(\tilde{\pi}) \geq \sup_\pi \{\eta_M(\pi) - 2\lambda \epsilon_u(\pi)\}
$$
不那么严格地说，$\eta_M(\tilde{\pi})$ 是在模型中训练出的策略的真实回报，$\epsilon_u(\pi) = \bar{\mathbb{E}}_{(\boldsymbol{s}, \boldsymbol{a}) \sim \rho_{\widehat{T}}^\pi}\left[u(\boldsymbol{s}, \boldsymbol{a})\right]$ 是不确定性惩罚的在模型的动态和处理 $\pi$ 下的期望。

含义：
1. 如果用 $\pi_\beta$ 代入，那么有$$\eta_M(\tilde{\pi}) \geq \eta_M(\pi_\beta) - 2\lambda \epsilon_u(\pi_\beta)$$这里的理解角度是，虽然我们的模型表现一般，但是由于模型在 $\pi_\beta$ 上的数据训练，故在数据集所在分布上误差很小。不难理解，在模型情况下，$\pi_\beta$ 访问的状态和动作和在真实动态下不会有很大的差异，自然在这部分数据上的不确定性惩罚也会很小。因此 $\epsilon_u(\pi_\beta)$ 很小，于是几乎可以认为：$$\eta_M(\tilde{\pi}) \geq \eta_M(\pi_\beta)$$也就是说，我们的策略至少比行为策略要好。
2. 如果代入 $\pi^\ast$，同时可以得到最优性差距：$$\eta_M(\tilde{\pi}) \geq \eta_M(\pi^\ast) - 2\lambda \epsilon_u(\pi^\ast)$$这里从另一种角度考虑，如果我们的模型非常准确，使得不确定性惩罚始终很小，那么此时也有 $\eta_M(\tilde{\pi})$ 会接近于最优的 $\eta_M(\pi^\ast)$，也就是说，我们的策略也会接近于最优。

参见：Yu\*, Thomas\*, Yu, Ermon, Zou, Levine, Finn, Ma. MOPO: Model-Based Offline Policy Optimization. '20.

### 4.3 COMBO: Conservative Off-Policy Model-Based Optimization
就像是保守 Q 学习中最小化策略产生的动作的 $Q$ 值一样，这里可以最小化模型产生的状态-动作对的 $Q$ 值。

考虑
$$
\begin{aligned} \hat{Q}^{k + 1} \gets \arg\min_Q &\,\,\beta\left(\mathbb{E}_{\boldsymbol{s}, \boldsymbol{a} \sim \rho(\boldsymbol{s}, \boldsymbol{a})} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right] - \mathbb{E}_{\boldsymbol{s}, \boldsymbol{a} \sim \mathcal{D}} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right]\right)\\ &+ \frac{1}{2} \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}') \sim d_f} \left[(Q(\boldsymbol{s}, \boldsymbol{a}) - (r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{a}' \sim \pi(\boldsymbol{a}' \mid \boldsymbol{s}')} \left[\hat{Q}^k(\boldsymbol{s}', \boldsymbol{a}')\right]))^2\right]. \end{aligned}
$$
这里 $\rho(\boldsymbol{s}, \boldsymbol{a})$ 和 $d_f$ 都是某种采样的分布。

直觉：这里的基本想法和保守 Q 学习中是一样的：
- 第一项 $\mathbb{E}_{\boldsymbol{s}, \boldsymbol{a} \sim \rho(\boldsymbol{s}, \boldsymbol{a})} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right]$ 拉低模型产生的状态-动作对的 $Q$ 值；  
- 第二项 $\mathbb{E}_{\boldsymbol{s}, \boldsymbol{a} \sim \mathcal{D}} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right]$ 拉高那些在数据集中的状态-动作对的 $Q$ 值； 
- 如果模型产生的状态-动作对分布和数据集中的分布一致，那么这两项的作用可以相互抵消。  

这一算法通常比前面的基于模型的离线策略优化（MOPO）表现更好。

参见：Yu, Kumar, Rafailov, Rajeswaran, Levine, Finn. COMBO: Conservative Offline Model-Based Policy Optimization. 2021.

### 4.4 Trajectory Transformer
之前介绍的都基于 Dyna 形式算法，我们也可能考虑轨迹优化这种没有显式策略的做法。

基本想法：
1. 训练一个轨迹模型：$p_\beta(\tau) = p_\beta(\boldsymbol{s}_1, \boldsymbol{a}_1, \ldots, \boldsymbol{s}_T, \boldsymbol{a}_T)$，会希望走出那些在这个分布下概率高的轨迹，而避免那些概率低的，也就避免了分布外状态和动作。  
2. 使用一个表示力很强的模型（如 Transformer）。  

这里考虑一个超维度的序列模型（否则输出空间太大了），第一个词元是 $\boldsymbol{s}_{1,1}$ （也即第一个状态的第一个维度），基于这个词元预测 $\boldsymbol{s}_{1,2}$, 以此类推直到 $\boldsymbol{s}_{1,d_{s}}$，然后是 $\boldsymbol{a}_{1,1}$ 到 $\boldsymbol{a}_{1,d_{a}}$，以此类推直到 $\boldsymbol{a}_{T,d_{a}}$，这里的 $d_{s}$ 和 $d_{a}$ 是状态和动作的维度。

![](14-6.png)

由于这一模型的表示能力非常强，在那些分布内的数据上，我们能够在很长的序列中做出准确的预测。

这样的模型可以被用来规划，一个可行的方法是使用束搜索，相较于语言模型中的束搜索，这里使用 $\sum_t r(\boldsymbol{s}_t, \boldsymbol{a}_t)$ 而非概率作为依据，也可以使用蒙特卡洛搜索等方法。

注意：
- 由于生成的都是分布外状态与动作很少的那些高概率轨迹，因此这个方法通常能够奏效。
- 但是这个方法的计算成本非常高，因为通常使用了非常大的模型，但是相应的也能够捕捉那些非常复杂的行为策略以及动态。 

参见：Janner, Li, Levine. Reinforcement Learning as One Big Sequence Modeling Problem. 2021.

## 5 Summary, Applications, Open Questions

### 5.1 Which off-line RL algorithm should I use?
- 如果仅仅进行离线学习：
	- 保守 Q 学习（CQL）：只有一个超参数，并且已经被广泛测验与使用过了。
	- 隐式 Q 学习（IQL）：更加灵活（同时适用于离线 + 在线），但是有更多超参数。 
- 如果在离线训练的基础上进行在线微调：
	- 不宜使用保守 Q 学习，因为通常过于悲观，不适合微调。
	- 优势加权演员-评论家（AWAC）：广泛应用且经过检验。 
	- 隐式 Q 学习（IQL）：通常比优势加权演员-评论家表现更好。  
- 如果在特定领域内有训练模型的很好方式：
	- 基于模型的保守离线策略优化（COMBO）：和保守 Q 学习有类似的形式，但是此时可以利用模型，训练一个模型在一些任务中可能并不容易，因此要结合领域的实际情况来选择。
	- 轨迹 Transformer：非常强大和有效的模型，但是需要非常高的计算成本（所以可能不太适合过高的时间跨度）。

### 5.2 The power of offline RL
如果使用在线的方式，通常会有以下的流程：
1. 做运行强化学习的相关准备：安全机制，自动数据收集，奖励设计，复位等；
2. 等待很长时间来运行强化学习算法；
3. 修改算法，回到 $2$，直到满意；
4. 丢弃所有没有用的数据，开始下一个任务。  

而在离线强化学习中：
1. 收集初始数据：人类提供的数据，利用控制器来收集数据，基线策略或是以上的混合；
2. 训练一个离线策略；  
3. 修改算法，回到 $2$，直到满意，但是与在线不同的是，在 $2$ 中我们不需要重新收集数据； 
4. 收集更多数据，添加到数据集中，回到 $2$；
5. 在未来的其他任务中，如果在同一个领域里，则可以直接使用之前的数据。

![](14-7.png)
如果没有很好的仿真器，那么相较于在线强化学习，离线强化学习的过程可以更加高效。

### 5.3 Applications
例如，Offline RL in robotic manipulation: MT-Opt, AMs
参见：
- Kalashnikov, Irpan, Pastor, Ibarz, Herzong, Jang, Quillen, Holly, Kalakrishnan, Vanhoucke, Levine. QT-Opt: Scalable Deep Reinforcement Learning of Vision-Based Robotic Manipulation Skills.
- Kalashnikov, Varley, Chebotar, Swanson, Jonschkowski, Finn, Levine, Hausman. MT-Opt: Continuous Multi-Task Robotic Reinforcement Learning at Scale. 2021.  

例如，Actionable Models: off-line RL with Goals
这里提出的做法是：
1. 通过离线强化学习算法（具体来说是一种基于保守 Q 学习的保守的算法）进行无监督预训练（没有任何奖励）来训练一个目标条件 $Q$ 函数；
2. 使用任务奖励和有限的数据进行微调。

参见：Chebotar, Hausman, Lu, Xiao, Kalashnikov, Varley, Irpan, Eysenbach, Julian, Finn, Levine. Actionable Models: Unsupervised Offline Reinforcement Learning of Robotic Skills. 2021.
![](14-8.png)

### 5.4 Takeaways, conclusions, future directions
离线强化学习真正的理想是：
1. 利用任何可能的策略或者策略的混合来收集数据；  
2. 在数据上运行离线强化学习算法；  
3. 在真实环境中部署。  

这样的理想和目前的离线强化学习算法依然有一些距离，这可以理解为几方面的问题：
1. 从离线强化学习流程的角度：在监督学习中，不需要部署的过程，因为会有训练集和测试集的划分，我们能对结果有一个很好的评估。但是在离线强化学习中依然需要在线评估，这可能依然昂贵且危险；  
2. 从统计保证的角度：最大的挑战是分布偏移 / 反事实；
3. 从可扩展的方法和大规模应用的角度：目前离线强化学习应用的场景和规模还是有限的。 

### 5.5 Summary
在本节中，
- 我们介绍了离线强化学习中的一些基本概念，以及其中存在的核心问题：分布偏移；
- 我们介绍了传统的一些离线强化学习算法，例如 基于重要性采样的方法，以及基于价值函数的 LSTD，LSPI 等方法。 
- 我们介绍了一些最新的离线强化学习算法，它们可以进行进一步细分：
	- 基于约束条件的方法：AWAC，IQL；  
	- 基于保守的方法：CQL；
	- 使用模型的方法：MOPO，COMBO，Trajectory Transformer 等。  
- 最后我们介绍了一些离线强化学习的应用，以及一些未来的方向。