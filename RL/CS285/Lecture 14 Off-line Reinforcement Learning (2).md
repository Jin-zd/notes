本部分是 offline RL 的第二部分内容, 我们会介绍现代的一些 offline RL 算法, 它们采用有效的方法来解决 [distribution shift](https://zhida.zhihu.com/search?content_id=255253693&content_type=Article&match_order=1&q=distribution+shift&zhida_source=entity) 问题.

## 1 Recap of offline RL

回顾一下我们对 offline RL 的 formalization:

-   $\mathcal{D} = \{(\boldsymbol{s}_i, \boldsymbol{a}_i, r_i, \boldsymbol{s}_i')\}$  
    
-   $\boldsymbol{s} \sim d^{\pi_\beta}(\boldsymbol{s})$  
    
-   $\boldsymbol{a} \sim \pi_\beta(\boldsymbol{a} \mid \boldsymbol{s})$  
    
-   $\boldsymbol{s}' \sim p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})$  
    
-   $r \gets r(\boldsymbol{s}, \boldsymbol{a})$  
    

这里 $\pi_\beta$ 是 behavior policy, 我们通常并不知道它具体是如何收集的.

![](https://pic4.zhimg.com/v2-f854f1036ac54c4e09e8d6943ffa882b_1440w.jpg)

offline RL 的 setting

在前面的分析中, 我们介绍了 offline RL 中核心的问题: **distribution shift**. 在这里我们依旧考虑 Q-learning, 我们反复进行的 Bellman backup 可以写作 $\pi_{new} = \arg\max_\pi \mathbb{E}_{\boldsymbol{a} \sim \pi(\boldsymbol{a}\mid \boldsymbol{s})} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right],\\$ 在这个过程中, $Q$ function 不断拟合 adversarial distribution 上的结果, 从而使得 $Q$ function 不断被高估.

在前面一节中, 我们讨论的一些经典算法并没有解决这个问题, 它们使用表示能力有限的线性模型使得 distribution shift 不会是一个严重的问题. 但是在如果我们想要开发出能够实际应用的 offline deep RL 算法, 我们必须要解决这一个问题.

## 2 Policy constraint methods

### 2.1 Explicit policy constraint methods

一个直观的想法是, 我们的 $\pi_{new}$ 选取太自由了, 这使得 Bellman 更新中可能会引入一些完全不合理的 distribution 来愚弄 Q-function. 一个直观的做法是将 policy 和 Q-function 分开来, 并限制 policy 的选择:

具体来说, 我们循环地进行以下两个步骤:

1.  **policy evaluation**: 我们固定 policy, 估计 $Q$ function: $Q(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{a'} \sim \pi_{new}(\boldsymbol{a'} \mid \boldsymbol{s'})} \left[Q(\boldsymbol{s'}, \boldsymbol{a'})\right].\\$
2.  **policy improvement**: 我们固定 $Q$ function, 优化 policy, 但是这个过程中我们对 policy 有一个约束: $\pi_{new} = \arg\max_\pi \mathbb{E}_{\boldsymbol{a} \sim \pi(\boldsymbol{a}\mid \boldsymbol{s})} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right] \text{ s.t. } D_{KL}(\pi \mid \pi_\beta) \leq \epsilon.\\$

![](https://pica.zhimg.com/v2-b7ad582c68a9a1c8635b240ef3c32638_1440w.jpg)

对 policy 进行约束, 防止其利用那些 OOD actions

**Remark:** 这是一个很早的 idea, 已经得到了充分的研究. 这一方法给我们提供了一个进行改进的基础, 但是依然有很多值得解决的问题:

**Issue 1:** 通常我们并不知道 $\pi_\beta$ 是什么, 这意味着如果我们要应用这种 policy constraint 的方法, 我们需要利用 **behavior cloning** 来估计 $\pi_\beta$, 或者我们需要有一种处理约束的巧妙方法, 通过 $\pi_\beta$ 的样本进行估计. 通常后者效果更好, 但它们都是合理的选择.

**Issue 2:** 这一处理方式有些 过于 pessimistic, 但同时又不够 pessimistic.

-   **过于 pessimistic**: 我们的 policy 会被限制在一个非常小的空间内, 让我们无法有效地改进. (例如 $\pi_\beta$ 是完全随机 policy, 但是我们并不应该期望学到的 $\pi$ 也是非常接近随机的).  
    
-   **不够 pessimistic**: 两个 KL 散度意义下接近的分布, 依然可能存在着一些错误非常大的地方, 这依然给我们愚弄 Q-function 的机会.  
    

上述的方法其实属于 **explicit policy constraint methods**, 也就是我们会有一个对 $\pi_{new}$ 的显式约束. 这里我们继续以 **KL 散度约束**为例, 给出一些具体的方法:

1.  直接修改 actor objective: 对于 KL 散度我们有 $D_{KL}(\pi \parallel \pi_\beta) = \mathbb{E}_\pi \left[\log \pi(\boldsymbol{a} \mid \boldsymbol{s}) - \log \pi_\beta(\boldsymbol{a} \mid \boldsymbol{s})\right] = -\mathbb{E}_\pi \log \pi_\beta(\boldsymbol{a} \mid \boldsymbol{s}) - \mathcal{H}(\pi)\\$

于是我们修改 objective 为 $\theta \gets \arg\max_\theta \mathbb{E}_{\boldsymbol{s} \sim D}\left[ \mathbb{E}_{\boldsymbol{a} \sim \pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})} \left[Q(\boldsymbol{s}, \boldsymbol{a}) + \lambda \log \pi_\beta(\boldsymbol{a} \mid \boldsymbol{s})\right] + \lambda \mathcal{H}(\pi(\boldsymbol{a} \mid \boldsymbol{s}))\right].\\$ 这里的 $\lambda$ 是拉格朗日乘子 (这是我们对约束优化问题的一种处理方式), 我们也可以利用 **dual gradient descent** 来进行优化, 也可以选择将 $\lambda$ 作为一个正则化参数. 此时我们受到影响的只有 actor update, 而 Q-function 不受影响.

1.  修改 reward function:

我们使用 $\tilde{r}(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) - D_{KL}(\pi(\boldsymbol{a} \mid \boldsymbol{s}) \parallel \pi_\beta(\boldsymbol{a} \mid \boldsymbol{s}))$ 来替代 reward function, 这里的好处是我们能够同样惩罚那些未来会导致 KL 散度增大的 action, 因为 Q-function 也会受到这一项的影响.

另一种可能的约束条件是 **support constraint**: $\pi(\boldsymbol{a} \mid \boldsymbol{s}) \geq 0$ 仅当 $\pi_\beta(\boldsymbol{a} \mid \boldsymbol{s}) \geq \epsilon$. 这可以利用 **Maximum Mean Discrepancy (MMD)** 来进行近似. 这里的代价是我们变得非常难 implement, 但是通常这更接近我们想要的.

参见:

-   Wu, Tucker, Nachum. Behavior Regularized Offline Reinforcement Learning. '19

### 2.2 Implicit policy constraint methods

实际上我们可以将原先的显式约束转化为对 $\pi_{new}$ 的隐式约束. 利用 duality, 我们的问题 $\max_\pi \mathbb{E}_{\boldsymbol{a} \sim \pi(\boldsymbol{a}\mid \boldsymbol{s})} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right] \text{ s.t. } D_{KL}(\pi \mid \pi_\beta) \leq \epsilon.\\$ 可以转化为 $\pi^\ast(\boldsymbol{s} \mid \boldsymbol{a}) = \frac{1}{Z(\boldsymbol{s})} \pi_\beta(\boldsymbol{a} \mid \boldsymbol{s}) \exp\left(\frac{1}{\lambda} A^\pi(\boldsymbol{s}, \boldsymbol{a})\right).\\$

考虑对 KL 散度展开为 $D_{KL}(\pi \mid \pi_\beta) = \mathbb{E}_{\boldsymbol{a} \sim \pi(\boldsymbol{a}\mid \boldsymbol{s})} [\log \pi(\boldsymbol{a}\mid \boldsymbol{s}) - \log \pi_\beta(\boldsymbol{a}\mid \boldsymbol{s})],\\$ 并利用概率分布的等式约束 $\int \pi(\boldsymbol{a} \mid \boldsymbol{s}) \text{d}\boldsymbol{a} = 1$, 得到拉格朗日函数 $\begin{aligned} L(\pi, \lambda, \mu) &= \mathbb{E}_{\boldsymbol{a} \sim \pi(\boldsymbol{a}\mid \boldsymbol{s})} [Q(\boldsymbol{s}, \boldsymbol{a}) - \lambda\log \pi(\boldsymbol{a}\mid \boldsymbol{s}) + \lambda\log \pi_\beta(\boldsymbol{a}\mid \boldsymbol{s})] + \lambda\epsilon - \mu\left(\int \pi(\boldsymbol{a}\mid \boldsymbol{s}) \text{d}\boldsymbol{a} - 1\right)\\ &= \int \pi(\boldsymbol{a} \mid \boldsymbol{s})\left[Q(\boldsymbol{s}, \boldsymbol{a}) - \lambda\log \pi(\boldsymbol{a}\mid \boldsymbol{s}) + \lambda\log \pi_\beta(\boldsymbol{a}\mid \boldsymbol{s})\right]\text{d}\boldsymbol{a} + \lambda \epsilon - \mu \int \pi(\boldsymbol{a}\mid \boldsymbol{s})\text{d}\boldsymbol{a} + \mu\\ \end{aligned} \\$ 考虑将上式对 $\pi(\boldsymbol{a}\mid \boldsymbol{s})$ 求 **变分导数** 并令其为零: $\frac{\partial L(\pi, \lambda, \mu)}{\partial \pi(\boldsymbol{a} \mid \boldsymbol{s})} = Q(\boldsymbol{s}, \boldsymbol{a}) - (\lambda + \lambda \log \pi(\boldsymbol{a}\mid \boldsymbol{s})) + \lambda \log \pi_\beta(\boldsymbol{a}\mid \boldsymbol{s}) - \mu = 0. \\$ 从而 $\pi(\boldsymbol{a} \mid \boldsymbol{s}) = \pi_\beta(\boldsymbol{a}\mid \boldsymbol{s})\exp\left(\frac{Q(\boldsymbol{s}, \boldsymbol{a}) - \mu - \lambda}{\lambda}\right).\\$ 利用归一化条件可以得到 $\int \pi_\beta(\boldsymbol{a}\mid \boldsymbol{s})\exp\left(\frac{Q(\boldsymbol{s}, \boldsymbol{a}) - \mu - \lambda}{\lambda}\right)\text{d}\boldsymbol{a} = 1,\\$ 定义归一化常量 $Z(\boldsymbol{s}) = \exp\left(\frac{\mu + \lambda - V(\boldsymbol{s})}{\lambda}\right)$, 可得 $\pi^\ast(\boldsymbol{a} \mid \boldsymbol{s}) = \frac{1}{Z(\boldsymbol{s})} \pi_\beta(\boldsymbol{a}\mid \boldsymbol{s})\exp\left(\frac{A(\boldsymbol{s}, \boldsymbol{a})}{\lambda}\right).\\$

**intuition:** $\lambda$ 趋近于 $0$ 时, 我们相当于在使用 greedy policy, 而如果 $\lambda$ 是较大, 那么此时我们依然会让有优势的 action 概率高, 但是那些在 $\pi_\beta$ 中概率低的 action 会被很大限制, 从而缓解 distribution shift 问题.

目前我们将一步更新后的 $\pi^\ast$ 用 $\pi_\beta$ 表示了出来, 但是我们只有 $\pi_\beta$ 的样本. 我们可以通过以下转化来通过样本估计 $\pi^\ast$:我们希望我们的新策略接近于 $\pi^\ast$, 因此我们可以使用 KL 散度, 要求 $\min_{\pi} D_{KL}(\pi^* | \pi) = \min_{\pi} \mathbb{E}_{\boldsymbol{a} \sim \pi^\ast(\cdot|\boldsymbol{s})} [\log \pi^\ast(\boldsymbol{a}|\boldsymbol{s}) - \log \pi(\boldsymbol{a}|\boldsymbol{s})],\\$ 忽略其中与 $\pi$ 无关的项, 我们可以得到新的目标为 $\max_{\pi} \mathbb{E}_{\boldsymbol{a} \sim \pi^*(\cdot|\boldsymbol{s})} [\log \pi(\boldsymbol{a}|\boldsymbol{s})],\\$ 由于我们的样本不来自于 $\pi^\ast$, 这里使用 importance sampling 得到 $\max_{\pi} \mathbb{E}_{\boldsymbol{a} \sim \pi_\beta(\cdot|\boldsymbol{s})} \left[\frac{\pi^\ast(\boldsymbol{a}|\boldsymbol{s})}{\pi_\beta(\boldsymbol{a}|\boldsymbol{s})} \log \pi(\boldsymbol{a}|\boldsymbol{s})\right].\\$ 代入 $\pi^\ast$ 的表达式, 我们可以得到优化问题 $\max_{\pi} \mathbb{E}_{\boldsymbol{a} \sim \pi_\beta(\cdot|\boldsymbol{s})} \left[\log \pi(\boldsymbol{a}|\boldsymbol{s})\frac{1}{Z(\boldsymbol{s})} \exp\left(\frac{1}{\lambda} A(\boldsymbol{s},\boldsymbol{a})\right)\right].\\$ 整理即可得到 $\pi_{new}(\boldsymbol{a} \mid \boldsymbol{s}) = \arg\max_\pi \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}) \sim \pi_{\beta}} \left[\log \pi(\boldsymbol{a} \mid \boldsymbol{s})\frac{1}{Z(\boldsymbol{s})} \exp\left(\frac{1}{\lambda} A^{\pi_{old}}(\boldsymbol{s},\boldsymbol{a})\right)\right].\\$ 这相当于是以 weight $w(\boldsymbol{s}, \boldsymbol{a}) = \frac{1}{Z(\boldsymbol{s})} \exp\left(\frac{1}{\lambda} A^{\pi_{old}}(\boldsymbol{s},\boldsymbol{a})\right)\\$ 的 weighted behavior cloning. 换言在 imitating dataset 中的数据, 但是我们模仿好的 action 更多. 通常情况下, 在实际算法中我们会忽略掉 $Z(\boldsymbol{s})$.

稍作整理我们可以得到 **[AWAC](https://zhida.zhihu.com/search?content_id=255253693&content_type=Article&match_order=1&q=AWAC&zhida_source=entity) (Advantage-Weighted Actor-Critic)** 算法, 在实际中我们使用梯度下降交替地更新: $L_C(\phi) = \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}') \sim \mathcal{D}} \left[\left( Q_{\phi}(\boldsymbol{s}, \boldsymbol{a}) - (r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{a}' \sim \pi_\theta(\boldsymbol{a'} \mid \boldsymbol{s}')} \left[Q_\phi(\boldsymbol{s}', \boldsymbol{a}')\right])\right)^2\right]\\$ $L_A(\theta) = \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}) \sim \mathcal{D}} \left[\log \pi_\theta(\boldsymbol{a} \mid \boldsymbol{s}) \exp\left(\frac{1}{\lambda} A^{\pi_{\phi}}(\boldsymbol{s},\boldsymbol{a})\right)\right].\\$ 其中 $\lambda$ 是一个 temperature 参数. 当 $\lambda$ 趋于 $0$, 我们就得到 greedy policy, 而当 $\lambda$ 趋于 $\infty$, 我们就得到 $\pi_\beta$.

**Side Note:** 理论上我们也可以使用 MC 方法来估计 reward, 此时对应的就是 **advantage weighted regression**.

一个值得注意的是, 我们上述转化的过程在**优化问题的意义上**是等价的, 换言之, 我们收敛到的结果满足 $\max_\pi \mathbb{E}_{\boldsymbol{a} \sim \pi(\boldsymbol{a}\mid \boldsymbol{s})} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right] \text{ s.t. } D_{KL}(\pi \mid \pi_\beta) \leq \epsilon\\$ 中的 KL 散度约束, 然而没有任何保证说明了在我们转化后的问题的**优化过程中**, 我们的 $\pi_\theta$ 满足这一约束, 因此我们依然会出现 query OOD actions 的问题. 具体来说, 在我们更新 $\phi$ 和计算 advantage 的过程中也使用了 $\pi_\theta$ 下的期望, 这两个地方都有可能出现上述问题.

参见:

-   Peng\*, Kumar\*, Levine. Advantage-Weighted Regression. '19  
    
-   Nair, Dalal, Gupta, Levine. Accelerating Online Reinforcement Learning with Offline Datasets. '20  
    

### 2.3 Implicit Q-learning ([IQL](https://zhida.zhihu.com/search?content_id=255253693&content_type=Article&match_order=1&q=IQL&zhida_source=entity))

回顾刚才我们介绍的 **AWAC** 算法, 事实上我们依然无法完全避免 OOD actions. 如果将 $\mathbb{E}_{\boldsymbol{a}' \sim \pi_{new}} \left[Q(\boldsymbol{s}', \boldsymbol{a}')\right]$ 用一个独立的网络 $V(\boldsymbol{s}')$ 来表示. 我们更新 Q-function 的过程可以表示为 $Q(\boldsymbol{s}, \boldsymbol{a}) \gets r(\boldsymbol{s}, \boldsymbol{a}) + \gamma V(\boldsymbol{s}').\\$ 此时在更新 Q-function 时就完全避免了 query OOD actions. 类似地, 如果我们依然使用 actor, 那么 advantage 的计算也不再需要 query OOD actions: $A(\boldsymbol{s}, \boldsymbol{a}) = Q(\boldsymbol{s}, \boldsymbol{a}) - V(\boldsymbol{s}).\\$

我们需要以某种方式来训练我们的 V-function 网络, 同时又需要 **避免 query OOD action**, 也 **只能利用数据集中的 states**. 因此我们需要也只能考虑如下形式的更新方式 $V \gets \arg\min_V \frac{1}{N} \sum_{i = 1}^{N} l(V(\boldsymbol{s}_i), Q(\boldsymbol{s}_i, \boldsymbol{a}_i)).\\$

这一做法的 intuition 是, 尽管在较大的 state space 中, 每一个 state 只会出现一次, 自然也只存在一个 action, 但是有可能有一些相近的 state, 其中我们选择了其他的 action. 由于泛化性, 我们可以将这些相似的 states 视作同一个, 此时在这个状态处, 存在着多个 dataset 中出现的 actions.

值得注意的是, 上述每一个 states 附近的 action 有好有坏, 它们都来源于 $\pi_\beta$, 如果我们使用 MSE loss, 则我们学到的会是 $\mathbb{E}_{\pi_\beta}\left[Q(\boldsymbol{s}, \boldsymbol{a})\right]$, 结合 V-function 的含义, 我们的 V-function 应当对应于当前的 Q-function, 而我们当前的 Q-function 应当学习到了**数据中** $\boldsymbol{s}$ 附近所有 $Q(\boldsymbol{s}, \boldsymbol{a})$ 的一个接近上界. 此时我们可以使用 **expectile loss** 来进行更新:

$l_2^\tau(x) = \begin{cases} (1 - \tau) x^2, & x > 0\\ \tau x^2, & x \leq 0 \end{cases},\\$适当地选择这里的 $\tau$, 我们可以更多地惩罚 negative error.

![](https://pic4.zhimg.com/v2-eb44d4bb1263c73df0bda73e47b368d3_1440w.jpg)

expectile loss

**Remark:**

-   此时我们避免了计算 $\pi_{new}$ 下的 Q-function 期望, 也就避免了 query OOD samples, 也就避免了之前的 overestimation 问题.  
    
-   如果简单使用 MSE loss, 由于我们没有使用任何新 policy 的 states, 我们会 underestimate, 但是使用 expectile loss 并调整 $\tau$ 的取值, 我们可以使得我们的 Q-function 更加接近于 optimal Q-function.  
    

### Ideas

我们也可以从以下角度来直观地理解这里的做法:

考虑支撑集 $\Omega(\boldsymbol{s}) = \{\boldsymbol{a}: \pi_\beta(\boldsymbol{a} \mid \boldsymbol{s}) \geq \epsilon\}$, 当我们使用很大的 $\tau$ 时, 我们这里的更新类似于 $V(\boldsymbol{s}) \gets \max_{\boldsymbol{a} \in \Omega(\boldsymbol{s})} Q(\boldsymbol{s}, \boldsymbol{a})\\$ 与 MSE 不同的是, 我们没有复制数据集中的 policy $\pi_\beta$, 而是我们会在每个 states 上独立地选择最优的 in distribution action. 这一思想可以被用来构建 **Implicit Q-learning (IQL)** 算法:

我们将 policy $\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})$, Q-function $Q_\phi(\boldsymbol{s}, \boldsymbol{a})$ 和 V-function $V_\psi(\boldsymbol{s})$ 分别用 $\theta$, $\phi$, $\psi$ 参数化. 进而得到如下更新方式:

-   更新 value function: $$$\begin{aligned} L_V(\psi) &= \mathbb{E}{(\boldsymbol{s}, \boldsymbol{a}) \sim \mathcal{D}} \left[l_2^\tau(V\psi(\boldsymbol{s}), Q_\phi(\boldsymbol{s}, \boldsymbol{a}))\right]\\ L_Q(\phi) &= \mathbb{E}{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}') \sim \mathcal{D}} \left[(Q\phi(\boldsymbol{s}, \boldsymbol{a}) - (r(\boldsymbol{s}, \boldsymbol{a}) + \gamma V_\psi(\boldsymbol{s}')))^2\right]\end{aligned}\\$$$
-   更新 actor: $L_A(\theta) = \mathbb{E}_{\boldsymbol{s} \sim \mathcal{D}} \left[\log \pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})\exp\left(\frac{1}{\lambda} A^{\pi_{\phi}}(\boldsymbol{s},\boldsymbol{a})\right)\right].\\$

参见: Kostrikov, Nair, Levine. Offline Reinforcement Learning with Implicit Q-Learning. '21

## 3 Conservative Q-learning

### 3.1 Introduction

接下来我们介绍的一类方法是 **Conservative Q-learning**, 相较于我们在 policy 上进行约束, 这里我们会在 Q-function 上限制那些 overestimate 的 action.

考虑一个新的 objective: $\begin{aligned} \hat{Q}^\pi = \arg\min_Q \max_\mu &\,\,\alpha \mathbb{E}_{\boldsymbol{s} \sim \mathcal{D}, \boldsymbol{a} \sim \mu(\boldsymbol{a} \mid \boldsymbol{s})} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right]\\  &+ \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}') \sim \mathcal{D}} \left[(Q(\boldsymbol{s}, \boldsymbol{a}) - (r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{a}' \sim \pi(\boldsymbol{a}' \mid \boldsymbol{s}')} \left[Q(\boldsymbol{s}', \boldsymbol{a}')\right]))^2\right]\\ \end{aligned} \\$ 其中后一项为我们常规的 Bellman error, 而前一项则是我们**显式地将过大的 Q-values 降低**. 可以证明当 $\alpha$ 足够大时, 则 $\hat{Q}^\pi$ 是 $Q^\pi$ 的一个下界.

![](https://pic2.zhimg.com/v2-5a7dec7eb35db54bb4e493d5a8f9e497_1440w.jpg)

人为将过高的 Q value 降低

上面的 objective 有些**过于 pessimistic** 了, 因为我们在 push down 所有高的 Q-value, 即使它们是在分布内, 因此我们最后无法得到真正的 optimal Q-function. 我们其实可以得到一个更好的 objective: $\begin{aligned} \hat{Q}^\pi = \arg\min_Q \max_\mu &\,\,\alpha \mathbb{E}_{\boldsymbol{s} \sim \mathcal{D}, \boldsymbol{a} \sim \mu(\boldsymbol{a} \mid \boldsymbol{s})} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right] - \alpha \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}) \sim \mathcal{D}} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right] + \mathcal{R}(\mu)\\  &+ \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}') \sim \mathcal{D}} \left[(Q(\boldsymbol{s}, \boldsymbol{a}) - (r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{a}' \sim \pi(\boldsymbol{a}' \mid \boldsymbol{s}')} \left[Q(\boldsymbol{s}', \boldsymbol{a}')\right]))^2\right]\\ \end{aligned} \\$ **分析:**

-   这里新添加的一项 $- \alpha \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}) \sim \mathcal{D}} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right]$ 能够 push up 那些在 $\mathcal{D}$ 内的 Q-values.  
    
-   原先的 $\alpha \mathbb{E}_{\boldsymbol{s} \sim \mathcal{D}, \boldsymbol{a} \sim \mu(\boldsymbol{a} \mid \boldsymbol{s})} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right]$ 依旧可以 push down 那些 overestimate 的 OOD Q-values.  
    
-   $\mathcal{R}(\mu)$ 是一个 regularizer, 我们在之后介绍.  
    

一个很重要的观察是:

-   估计偏离合理估计越多, 那么前两项的综合作用越强.  
    
-   如果估计就是合理的, 例如高 Q-value 都在分布内, 那么前两项的作用就可以相互抵消.  
    

此时相应的理论保证是, 对于任意的 policy $\pi$ 和 state $\boldsymbol{s} \in \mathcal{D}$, 有 $\mathbb{E}_{\pi(\boldsymbol{a} \mid \boldsymbol{s})}\left[\hat{Q}^\pi(\boldsymbol{s}, \boldsymbol{a})\right] \leq \mathbb{E}_{\pi(\boldsymbol{a} \mid \boldsymbol{s})}\left[Q^\pi(\boldsymbol{s}, \boldsymbol{a})\right].\\$ 这已经足够了.

### 3.2 CQL (Conservative Q-learning)

最后我们整理给出 **CQL (Conservative Q-learning)** 算法, 记之前定义的 objective 可以记作 $\mathcal{L}_{CQL}(\hat{Q}^\pi)$, 我们 CQL 的算法流程如下:

1.  使用 $\mathcal{D}$ 利用 $\mathcal{L}_{CQL}(\hat{Q}^\pi)$ 更新 $\hat{Q}^\pi$.  
    
2.  更新 policy $\pi$.  
    

这里 policy $\pi$ 的更新我们还没有给出, 具体来说:

-   如果 actions 是离散的, 我们使用 $\pi(\boldsymbol{a} \mid \boldsymbol{s}) = \begin{cases} 1, & \text{if } \boldsymbol{a} = \arg\max_{\boldsymbol{a}'} \hat{Q}^\pi(\boldsymbol{s}, \boldsymbol{a}')\\ 0, & \text{otherwise}. \end{cases}\\$  
    
-   如果 actions 是连续的, 我们可以使用一个独立的模型表示 policy (更像是一个 actor-critic 的结构) $\theta \gets \theta + \alpha \nabla_\theta \sum_{i} \mathbb{E}_{\boldsymbol{a} \sim \pi_\theta(\boldsymbol{a} \mid \boldsymbol{s}_i)} \left[\hat{Q}(\boldsymbol{s}_i, \boldsymbol{a})\right]\\$

### 3.3 Regularizer

注意上述算法中由于我们没有指定 regularizer 的具体形式, 我们的的算法是不完整的. 而我们定义的 objective 其实是一族 objective, 每一个具体的 $\mathcal{R}(\mu)$ 都可以实例化出一个具体的算法.

**Example 1**. _一个常见的 regularizer 是 $\mathcal{R}(\mu) = \mathbb{E}_{\boldsymbol{s} \sim \mathcal{D}} \left[\mathcal{H}(\mu(\cdot \mid \boldsymbol{s}))\right]$, 这是一个 maximum entropy regularization._

_此时我们把 objective 中和 $\mu$ 有关的部分提出来, 并且将对 $\boldsymbol{s}$ 的期望移到最外面, 考虑 $\begin{aligned} \arg\max_{\mu(\cdot\mid\boldsymbol{s})} \mathbb{E}_{\boldsymbol{a} \sim \mu(\boldsymbol{a} \mid \boldsymbol{s})}\left[Q(\boldsymbol{s}, \boldsymbol{a})\right] + \mathcal{H}(\mu(\cdot \mid \boldsymbol{s})) &= \arg\max_{\mu(\cdot\mid\boldsymbol{s})} \mathbb{E}_{\boldsymbol{a} \sim \mu(\boldsymbol{a} \mid \boldsymbol{s})}\left[Q(\boldsymbol{s}, \boldsymbol{a}) - \log \mu(\boldsymbol{a} \mid \boldsymbol{s})\right], \end{aligned}\\$ 我们前面在 AWAC 算法推导中已经见过了如何利用 **拉格朗日乘子法** 和 **变分导数** 来解决这类带约束的优化问题 (待优化的是函数 $\mu$ 满足归一化条件), 不妨作为练习, 可得最优的 $\mu(\boldsymbol{a} \mid \boldsymbol{s}) \propto \exp(Q(\boldsymbol{s}, \boldsymbol{a}))$, 也就有 $\mu(\boldsymbol{a} \mid \boldsymbol{s}) = \frac{\exp(Q(\boldsymbol{s}, \boldsymbol{a}))}{\sum_{\boldsymbol{a}'} \exp(Q(\boldsymbol{s}, \boldsymbol{a}'))}.\\$ 因此可知 $\max_{\mu(\cdot\mid\boldsymbol{s})} \mathbb{E}_{\boldsymbol{a} \sim \mu(\boldsymbol{a} \mid \boldsymbol{s})}\left[Q(\boldsymbol{s}, \boldsymbol{a}) - \log \mu(\boldsymbol{a} \mid \boldsymbol{s})\right] = \log \sum_{\boldsymbol{a}} \exp(Q(\boldsymbol{s}, \boldsymbol{a})).\\$_

-   _在离散情况下, 右侧所有 $\exp(Q(\boldsymbol{s}, \boldsymbol{a}))$ 的求和是可以计算的, 于是原 objective 所有与 $\mu$ 有关的部分就都用 $Q$ 表示出来了, 此时不再需要显式地构造 $\mu$._  
    
-   _在连续情况下, 我们没办法直接计算这个求和. 此时我们可以使用 importance sampling, 在另一个容易处理的分布 $\rho(\boldsymbol{a} \mid \boldsymbol{s})$ 下采样, 从而得到 $\mathbb{E}_{\boldsymbol{a} \sim \mu(\boldsymbol{a} \mid \boldsymbol{s})}\left[Q(\boldsymbol{s}, \boldsymbol{a}) - \log \mu(\boldsymbol{a} \mid \boldsymbol{s})\right] = \mathbb{E}_{\boldsymbol{a} \sim \rho(\boldsymbol{a} \mid \boldsymbol{s})}\left[\frac{\mu(\boldsymbol{a} \mid \boldsymbol{s})}{\rho(\boldsymbol{a} \mid \boldsymbol{s})}\left(Q(\boldsymbol{s}, \boldsymbol{a}) - \log \mu(\boldsymbol{a} \mid \boldsymbol{s})\right)\right],\\$ 再使用 $\exp(Q(\boldsymbol{s}, \boldsymbol{a}))$ 作为 $\mu(\boldsymbol{a} \mid \boldsymbol{s})$ 即可在忽略掉常数的情况下计算出上式._  
    

## 4 Model-Based Offline RL

目前我们介绍的一些 offline RL 算法都是 model-free 的, 我们也可以考虑 model-based 的方法.

在 model-free 类型的算法中, 事实上对于每一个 state-action pair, 我们只有一个 transition 结果, 这会带来高的 variance. 而一个 model 就可以回答 "what if" 的问题: 我们如果转移到了另一个 state 会怎么样?

**Example 2**. _以 Bellman backup 为例, 如果我们有一个 model, 那么我们可以计算 $Q(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{s}' \sim p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})} \left[\arg\max_{\boldsymbol{a}'}Q(\boldsymbol{s}', \boldsymbol{a}')\right].\\$ 而在 model-free 的情况下, 我们只能使用一个样本来估计这个期望, 也就是 $Q(\boldsymbol{s}, \boldsymbol{a}) \approx r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \arg\max_{\boldsymbol{a}'} Q(\boldsymbol{s}', \boldsymbol{a}').\\$_

在我们之前介绍的 model-based RL 中, 我们主要使用 Dyna-style 的算法. 具体来说, 我们会使用一个拼接的 trajectory. 一部分是 **真实环境** 中的长 trajectory, 另一部分是 从这些 trajectories 上的 states 出发, 基于 model 产生的短 trajectories.

![](https://pic1.zhimg.com/v2-f87f0b512ba97e675640f80a2edfc7aa_1440w.jpg)

Dyna-style 的 model-based 算法

然而这样的方式在 offline 情况下有一些问题, 由于我们无法再收集新的数据来更新我们的 model, 因此我们的 model 可能会在 OOD states 给出很差的预测, 例如给出一个很高的 reward, 那么这就给了我们的 policy 可乘之机, 我们的 policy 会学到依据 model 转移到这些 OOD 的 reward 很高的 states. 最终我们同样会造成严重高估的 Q-values.

### 4.1 MOPO: Model-Based Offline Policy Optimization

这里我们介绍一种直接修改 reward function 的算法. 考虑 $\tilde{r}(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) - \lambda u(\boldsymbol{s}, \boldsymbol{a})\\$ 这里 $u(\boldsymbol{s}, \boldsymbol{a})$ 是一个 **uncertainty penalty**. 然后在这个更改后, 我们就可以使用任何 model-based RL 算法.

**Intuition:** uncertainty penalty 的作用直观来说就是惩罚那些利用 OOD states 的行为, 使得它们不在值得.

**Design choice:** 通常会使用我们在 model-based RL 中介绍的衡量 uncertainty 的方法, 例如 **model ensemble**, 我们使用不同 model 的预测的 disagreement 等指标来作为 uncertainty 指标.

通常情况下, 为了得到这一算法一些相对理论的保证, $u(\boldsymbol{s}, \boldsymbol{a})$ 需要满足一些要求, 这会在后面的理论分析中介绍.

参见:

-   Yu\*, Thomas\*, Yu, Ermon, Zou, Levine, Finn, Ma. MOPO: Model-Based Offline Policy Optimization. '20  
    
-   Kidambi et al., MOReL : Model-Based Offline Reinforcement Learning. '20 (concurrent)  
    

### 4.2 Theoretical Analysis on MOPO

接下来我们考虑这背后的一些理论分析:

**Assumption:** 我们假设

1.  我们的 value model 足够强大, 能够表达出真实的 value function.  
    
2.  model error 被 $u(\boldsymbol{s}, \boldsymbol{a})$ bound 住 (在例如 total variation distance 下).  
    

**Theorem 1**. _在上述假设下, 我们有_

_$\eta_M(\tilde{\pi}) \geq \sup_\pi \{\eta_M(\pi) - 2\lambda \epsilon_u(\pi)\},\\$ 不那么严格地说, $\eta_M(\tilde{\pi})$ 是在 model 中训练出的 policy 的真实 return, $\epsilon_u(\pi) = \bar{\mathbb{E}}_{(\boldsymbol{s}, \boldsymbol{a}) \sim \rho_{\widehat{T}}^\pi}\left[u(\boldsymbol{s}, \boldsymbol{a})\right]$ 是 uncertainty penalty 的在 **model 的 dynamic 和 policy**_ _$\pi$ 下的期望._

**Implication:**

1.  如果我们用 $\pi_\beta$ 代入, 那么我们有 $\eta_M(\tilde{\pi}) \geq \eta_M(\pi_\beta) - 2\lambda \epsilon_u(\pi_\beta).\\$ 这里的理解角度是, 虽然我们的 model 表现一般, 但是由于我们的 model 在 $\pi_\beta$ 上的数据训练, 故在数据集所在分布上误差很小. 不难理解在 model 下, $\pi_\beta$ 访问的 states 和 actions 和在真实 dynamics 下不会有很大的差异, 自然在这部分数据上的 uncertainty penalty 也会很小. 因此 $\epsilon_u(\pi_\beta)$ 很小, 于是几乎可以认为: $\eta_M(\tilde{\pi}) \geq \eta_M(\pi_\beta),\\$ 也就是说, 我们的 policy 至少比 behavior policy 要好.  
    
2.  如果我们代入 $\pi^\ast$, 同时我们可以得到 **optimality gap**: $\eta_M(\tilde{\pi}) \geq \eta_M(\pi^\ast) - 2\lambda \epsilon_u(\pi^\ast).\\$ 这里从另一种角度考虑, 如果我们的 model 非常准确, 使得 uncertainty penalty 始终很小, 那么此时也有我们的 $\eta_M(\tilde{\pi})$ 会接近于 optimal 的 $\eta_M(\pi^\ast)$, 也就是说, 我们的 policy 也会接近于 optimal.  
    

参见: Yu\*, Thomas\*, Yu, Ermon, Zou, Levine, Finn, Ma. MOPO: Model-Based Offline Policy Optimization. '20.

### 4.3 COMBO: Conservative Off-Policy Model-Based Optimization

**Basic idea:** 就像是 CQL 中我们最小化 policy 产生的 actions 的 Q-values 一样, 这里我们可以最小化 model 产生的 state-action tuples 的 Q-values.

考虑 $\begin{aligned} \hat{Q}^{k + 1} \gets \arg\min_Q &\,\,\beta\left(\mathbb{E}_{\boldsymbol{s}, \boldsymbol{a} \sim \rho(\boldsymbol{s}, \boldsymbol{a})} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right] - \mathbb{E}_{\boldsymbol{s}, \boldsymbol{a} \sim \mathcal{D}} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right]\right)\\ &+ \frac{1}{2} \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}') \sim d_f} \left[(Q(\boldsymbol{s}, \boldsymbol{a}) - (r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{a}' \sim \pi(\boldsymbol{a}' \mid \boldsymbol{s}')} \left[\hat{Q}^k(\boldsymbol{s}', \boldsymbol{a}')\right]))^2\right]. \end{aligned} \\$ 这里 $\rho(\boldsymbol{s}, \boldsymbol{a})$ 和 $d_f$ 都是某种采样的分布.

**Intuition:** 这里的基本想法和 CQL 中是一样的

-   第一项 $\mathbb{E}_{\boldsymbol{s}, \boldsymbol{a} \sim \rho(\boldsymbol{s}, \boldsymbol{a})} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right]$ push down model 产生的 state-action pairs 的 Q-values.  
    
-   第二项 $\mathbb{E}_{\boldsymbol{s}, \boldsymbol{a} \sim \mathcal{D}} \left[Q(\boldsymbol{s}, \boldsymbol{a})\right]$ push up 那些在数据集中的 state-action pairs 的 Q-values.  
    
-   如果 model 产生的 state-action pairs 分布和数据集中的分布一致, 那么这两项的作用可以相互抵消.  
    

**Remark:** 这一算法通常比前面的 MOPO 表现更好.

参见: Yu, Kumar, Rafailov, Rajeswaran, Levine, Finn. COMBO: Conservative Offline Model-Based Policy Optimization. 2021.

### 4.4 Trajectory Transformer

之前我们介绍的都基于 Dyna-style 的算法, 我们也可能考虑 **trajectory optimization** 这种没有显式 policy 的做法.

**Basic Ideas:**

1.  训练一个 trajectory model: $p_\beta(\tau) = p_\beta(\boldsymbol{s}_1, \boldsymbol{a}_1, \ldots, \boldsymbol{s}_T, \boldsymbol{a}_T)$, 我们会希望走出那些在这个分布下概率高的 trajectories, 而避免那些概率低的, 也就避免了 OOD states 和 actions.  
    
2.  使用一个表示力很强的模型 (Transformer)  
    

这里考虑一个 **over dimension 的 sequence model** (否则输出空间太大了), 第一个 token 是 $\boldsymbol{s}_{1,1}$ (也即第一个 state 的第一个维度), 基于这个 token 我们预测 $\boldsymbol{s}_{1,2}$, 以此类推直到 $\boldsymbol{s}_{1,d_{s}}$, 然后是 $\boldsymbol{a}_{1,1}$ 到 $\boldsymbol{a}_{1,d_{a}}$, 以此类推直到 $\boldsymbol{a}_{T,d_{a}}$. 这里的 $d_{s}$ 和 $d_{a}$ 是 state 和 action 的维度.

![](https://pica.zhimg.com/v2-a19f3e23a3f3bb910f87612bd43dc060_1440w.jpg)

Trajectory Transformer 的使用方式

由于这一模型的表示能力非常强, 在那些分布内的数据上, 我们能够在很长的序列中做出准确的预测.

这样的 model 可以被用来 planning, 一个可行的方法是使用 **beam search**, 相较于 language model 中的 beam search, 这里我们使用 $\sum_t r(\boldsymbol{s}_t, \boldsymbol{a}_t)$ 而非 probability 作为依据. 我们也可以使用 MCTS 等方法.

**Remark:**

-   由于我们生成的都是 OOD states 与 actions 很少的那些高概率 trajectory, 因此这个方法通常能够 work.  
    
-   但是这个方法的计算成本非常高, 因为通常我们也使用了非常大的模型. 但是相应的也能够捕捉那些非常复杂的 behavior policy 以及 dynamic.  
    

参见: Janner, Li, Levine. Reinforcement Learning as One Big Sequence Modeling Problem. 2021.

## 5 Summary, Applications, Open Questions

### 5.1 Which offline RL algorithm should I use?

-   如果我们仅仅进行 offline:  
    

-   CQL: 只有一个超参数, 并且已经被广泛测验与使用过了.  
    
-   IQL: 更加灵活 (同时适用于 offline + online), 但是有更多 hyperparameters.  
    

-   如果我们 offline train 的基础上进行 online fine-tuning:  
    

-   不宜使用 CQL, 因为通常过于 pessimistic, 不适合 fine-tuning.  
    
-   AWAC: 广泛应用且经过检验.  
    
-   IQL: 通常比 AWAC 表现更好  
    

-   如果在特定领域内有训练 model 的很好方式:  
    

-   COMBO: 和 CQL 有类似的形式, 但是我们此时可以利用 model. 但是训练一个 model 在一些任务中可能并不容易, 因此要结合领域的实际情况来选择.  
    
-   Trajectory transformer: 非常强大和有效的模型, 但是需要非常高的计算成本 (所以可能不太适合过高的 horizon)  
    

**Side Note:** 虽然本课程是在 2023 年开设的, 但是这一节的内容录制于 21 年底, 因此一些最新的算法可能并没有被包含在内.

### 5.2 The power of offline RL

### Workflows in oneline RL

如果我们使用 online 的方式, 我们通常会有以下的流程:

1.  做运行 RL 的相关准备: 安全机制, 自动数据收集, rewards design, resets 等
2.  等待很长时间来运行 RL
3.  修改算法, 回到 $2$, 直到满意
4.  丢弃所有没有用的数据, 开始下一个任务.  
    

### Workflows in offline RL

而在 offline RL 中:

1.  收集初始数据: 人类提供的数据, 利用 controller 来收集数据, baseline policy 或是以上的混合  
    
2.  训练一个 offline policy  
    
3.  修改算法, 回到 $2$, 直到满意, 但是与 online 不同的是, 在 $2$ 中我们不需要重新收集数据.  
    
4.  收集更多数据, 添加到数据集中, 回到 $2$.  
    
5.  在未来的其他任务中, 如果在同一个 domain 里, 则可以直接使用之前的数据.

![](https://pic3.zhimg.com/v2-a646a694aad1f362614db25f7e3f0200_1440w.jpg)

如果我们没有很好的 simulator, 那么相较于 online RL, offline RL 的过程可以更加高效.

### 5.3 Applications

**Example 3**. _Offline RL in robotic manipulation: MT-Opt, AMs_

_参见:_

-   _Kalashnikov, Irpan, Pastor, Ibarz, Herzong, Jang, Quillen, Holly, Kalakrishnan, Vanhoucke, Levine. QT-Opt: Scalable Deep Reinforcement Learning of Vision-Based Robotic Manipulation Skills._  
    
-   _Kalashnikov, Varley, Chebotar, Swanson, Jonschkowski, Finn, Levine, Hausman. MT-Opt: Continuous Multi-Task Robotic Reinforcement Learning at Scale. 2021._  
    

**Example 4**. _Actionable Models: Offline RL with Goals_

_这里提出的做法是:_

1.  _通过 offline RL 算法 (具体来说是一种基于 CQL 的 conservative 算法) 进行无监督预训练 (没有任何 reward) 来训练一个 goal-conditioned Q function_  
    
2.  _使用 task reward 和有限的数据进行 finetune._  
    

_参见: Chebotar, Hausman, Lu, Xiao, Kalashnikov, Varley, Irpan, Eysenbach, Julian, Finn, Levine. Actionable Models: Unsupervised Offline Reinforcement Learning of Robotic Skills. 2021._

![](https://pic3.zhimg.com/v2-dec52fa4845af60204d1950324f299a8_1440w.jpg)

Actionable Models: Offline RL with Goals

### 5.4 Takeaways, conclusions, future directions

Offline RL 真正的理想是:

1.  利用任何可能的 policy 或者 policy 的混合来收集数据  
    
2.  在数据上运行 offline RL 算法  
    
3.  在真实环境中部署  
    

这样的理想和目前的 offline RL 算法依然有一些 gap, 这可以理解为几方面的问题:

1.  从 offline RL workflow 的角度: 在 supervised learning 中, 我们不需要 deploy 的过程, 因为我们会有**训练集和测试集**的划分, 我们能对我们的结果有一个很好的评估. 但是在 offline RL 中我们依然需要 online evaluation, 这可能依然 expensive 且可能危险的.  
    
2.  从 statistical guarantee 的角度: 最大的挑战是 distribution shift/ counterfactuals.  
    
3.  从 scalable 方法和 large-scale application 的角度: 目前 offline RL 应用的场景和规模还是有限的.  
    

### 5.5 Summary

在本节中,

-   我们介绍了 offline RL 中的一些基本概念, 以及其中存在的核心问题: distribution shift.  
    
-   我们介绍了传统的一些 offline RL 算法, 例如 基于 importance sampling 的方法, 以及基于 value function 的 LSTD, LSPI 等方法.  
    
-   我们介绍了一些最新的 offline RL 算法, 它们可以进行进一步细分:  
    

-   基于 constraints 的方法: AWAC, IQL  
    
-   基于 conservative 的方法: CQL  
    
-   使用 model 的方法: MOPO, COMBO, trajectory transformer 等  
    

-   最后我们介绍了一些 offline RL 的应用, 以及一些未来的方向.