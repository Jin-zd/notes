在本节中, 我们先从 policy gradient 与 [policy iteration](https://zhida.zhihu.com/search?content_id=253759430&content_type=Article&match_order=1&q=policy+iteration&zhida_source=entity) 的关系出发, 从理论角度解释 natural policy gradient, 并探讨一些更加 advanced 的 policy gradient 方法. 由于 CS285 课程本节内容中并没有展开 [TRPO](https://zhida.zhihu.com/search?content_id=253759430&content_type=Article&match_order=1&q=TRPO&zhida_source=entity), [PPO](https://zhida.zhihu.com/search?content_id=253759430&content_type=Article&match_order=1&q=PPO&zhida_source=entity) 等算法, 补充的部分内容可能存在疏漏.
## 1 Policy gradient as policy iteration

回顾 [REINFORCE](https://zhida.zhihu.com/search?content_id=253759430&content_type=Article&match_order=1&q=REINFORCE&zhida_source=entity) 算法:

1.  利用 $\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)$ 采样 $\{\tau^i\}$  
    
2.  计算 $\nabla_\theta J(\theta) = \frac{1}{N} \sum_{i} \sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t}) \left(\sum_{t' = t}^T r(\boldsymbol{s}_{i,t'}, \boldsymbol{a}_{i,t'})\right)$  
    
3.  更新 $\theta \gets \theta + \alpha \nabla_\theta J(\theta)$  
    

从更加广泛的角度, 对于这些 RL 算法来说, 我们有以下几步:

1.  估计当前策略 $\pi$ 该估计 $\hat{A}^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t)$  
    
2.  利用 $\hat{A}^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t)$ 获得改进后的 $\pi'$  
    

实际上上面的 REINFORCE 算法就是这样的一个例子. 这其实是一个很熟悉的形式, 也就是 policy iteration:

1.  fit $A^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t)$  
    
2.  $\pi \gets \pi'$. (使用 argmax)  
    

考虑二者更新上的差别, 只是在 policy iteration 中我们直接使用了 argmax, 而在 policy gradient 我们仅仅是进行了较小的更新. 因此某种程度上 policy gradient 是 policy iteration 的一个 soft version.

尽管这看似是一个 trivial 的联系, 然而, 考虑这样的一个问题: 在 policy iteration 中, 通常我们能够保证每次更新都能增大 $J(\pi)$, 但是在 policy gradient 中, 我们并不能保证这一点. 那么, 有没有什么方法来保证这一点呢?

## 2 Theoretical Analysis of Natural Policy Gradient

注意到当我们给定 $J(\theta)$ 时, 我们进行一步参数更新, 得到 $\theta'$, 我们希望能够尽可能地增大 $J(\theta')$. 注意我们通常的更新方式并不保证这一点, 我们虽然朝着梯度方向走了一步, 但是如果走的太远, 我们的 $J(\theta')$ 也可能会下降.

在之前关于 policy gradient 的一节中, 我们引入了 **natural policy gradient** 的方法, 通过 KL 散度的约束来保证我们的更新不会太远, 这在直觉上比我们使用通常的 policy gradient 更加合理. 事实上, 我们可以从理论上证明 natural policy gradient 的有效性. 在这一理论结果的基础上, 我们进一步介绍更加 advanced 的 policy gradient 方法, 例如 TRPO, PPO 等.

### 2.1 Surrogate Advantage

对于 $J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[\sum_{t = 1}^T \gamma^t r(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]$, 我们 claim 如下定理

**Theorem 1**. $J(\theta') - J(\theta) = \mathbb{E}_{\tau \sim p_{\theta'}(\tau)} \left[\sum_{t = 1}^T \gamma^t A^{\pi_\theta}(\boldsymbol{s}_t, \boldsymbol{a}_t)\right].\\$

_注意这里期望的分布是_ $p_{\theta'}(\tau)$_, 而不是_ $p_{\theta}(\tau)$_._

**Remark:** 这一个式子的意义是, 如果我们在最大化原 $\theta$ 对应的 advantage function 在新的 $\theta'$ 下的期望, 那么我们就能够 maximize $J(\theta') - J(\theta)$, 也就是 maximize $J(\theta')$, 即优化 RL 目标.

_Proof._ $\begin{aligned}  J(\theta') - J(\theta) &= J(\theta') - \mathbb{E}_{\boldsymbol{s}_0 \sim p_{\boldsymbol{s}_0}} \left[V^{\pi_\theta}(\boldsymbol{s}_0)\right]\\  &= J(\theta') - \mathbb{E}_{\tau \sim p_{\theta'}(\tau)} \left[V^{\pi_\theta}(\boldsymbol{s}_0)\right]\\  &= J(\theta') - \mathbb{E}_{\tau \sim p_{\theta'}(\tau)} \left[\sum_{t = 0}^\infty \gamma^t V^{\pi_\theta}(\boldsymbol{s}_t) - \sum_{t = 1}^\infty \gamma^t V^{\pi_\theta}(\boldsymbol{s}_t)\right]\\  &= J(\theta') + \mathbb{E}_{\tau \sim p_{\theta'}(\tau)} \left[\sum_{t = 0}^{\infty} \gamma^t \left(\gamma V^{\pi_\theta}(\boldsymbol{s}_{t + 1}) - V^{\pi_\theta}(\boldsymbol{s}_t)\right)\right]\\  &= \mathbb{E}_{\tau \sim p_{\theta'}(\tau)} \left[\sum_{t = 0}^\infty \gamma^t r(\boldsymbol{s}_t, \boldsymbol{a}_t)\right] + \mathbb{E}_{\tau \sim p_{\theta'}(\tau)} \left[\sum_{t = 0}^{\infty} \gamma^t \left(\gamma V^{\pi_\theta}(\boldsymbol{s}_{t + 1}) - V^{\pi_\theta}(\boldsymbol{s}_t)\right)\right]\\  &= \mathbb{E}_{\tau \sim p_{\theta'}(\tau)} \left[\sum_{t = 0}^\infty \gamma^t r(\boldsymbol{s}_t, \boldsymbol{a}_t) + \gamma V^{\pi_\theta} (\boldsymbol{s}_{t + 1}) - V^{\pi_\theta}(\boldsymbol{s}_t)\right]\\  &= \mathbb{E}_{\tau \sim p_{\theta'}(\tau)} \left[\sum_{t = 0}^\infty \gamma^t A^{\pi_\theta}(\boldsymbol{s}_t, \boldsymbol{a}_t)\right].  \end{aligned}\\$

第二个等号是基于对于不同 policy 采样的轨迹, 关于 $\boldsymbol{s}_0$ 的边缘分布是相同的. ◻

我们的目标是最大化 $J(\theta')$, 根据上述定理, 我们的目标可以转化为

$\begin{aligned}  \mathbb{E}_{\tau \sim p_{\theta'}(\tau)} \left[\sum_{t} \gamma^t A^{\pi_\theta}(\boldsymbol{s}_t, \boldsymbol{a}_t)\right] &= \sum_{t} \mathbb{E}_{\boldsymbol{s}_t \sim p_{\theta'}(\boldsymbol{s}_t)} \left[\mathbb{E}_{\boldsymbol{a}_t \sim \pi_{\theta'}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \left[\gamma^t A^{\pi_\theta}(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]\right]\\  &= \sum_{t} \mathbb{E}_{\boldsymbol{s}_t \sim p_{\theta'}(\boldsymbol{s}_t)} \left[\mathbb{E}_{\boldsymbol{a}_t \sim \pi_{\theta}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \left[\frac{\pi_{\theta'}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}{\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \gamma^t A^{\pi_\theta}(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]\right]\\ \end{aligned} \\$ 考虑等式的最右侧, 一个棘手的地方在于, 上式最外层的期望是关于 $p_{\theta'}(\boldsymbol{s}_t)$ 的, 我们希望被我们最大化的式子可以仅和 $p_\theta(\boldsymbol{s}_t)$ 有关, 一个理想的目标是

$\begin{aligned}  \bar{A}(\theta') &:= \sum_{t} \mathbb{E}_{\boldsymbol{s}_t \sim p_{\theta}(\boldsymbol{s}_t)} \left[\mathbb{E}_{\boldsymbol{a}_t \sim \pi_{\theta}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \left[\frac{\pi_{\theta'}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}{\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \gamma^t A^{\pi_\theta}(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]\right]\\  &= \mathbb{E}_{\tau \sim p_{\theta}(\tau)} \left[\sum_{t} \frac{\pi_{\theta'}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}{\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \gamma^t A^{\pi_\theta}(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]. \end{aligned} \\$ 这一目标 $\bar{A}(\theta')$ 有一个非常简洁的形式, 被我们称为 **surrogate advantage**, 很显然将外层的期望由 $p_{\theta'}(\boldsymbol{s}_t)$ 替换为 $p_{\theta}(\boldsymbol{s}_t)$ 会改变整个式子的值, 这样的改变在什么情况下是可以忽略的呢? 如果二者足够接近即 $J(\theta') - J(\theta) \approx \bar{A}(\theta')$, 那么我们就可以使用如下方式最大化 $J(\theta')$:

$\theta \gets \arg \max_{\theta'} \bar{A}(\theta').\\$

### 2.2 Bounding the distribution change

在上述替换中, 我们涉及到的是两个分布 $p_\theta(\boldsymbol{s}_t)$ 与 $p_{\theta'}(\boldsymbol{s}_t)$ 之间的距离. 然而通常情况下每一次进行参数更新时我们更容易获取的是两个 policy 之间分布的差异. 为了处理将两个分布的差异转化为它们选取 action 的差异, 我们首先考虑一下概念以及引理:

**Definition 1**. _coupling_

_两个分布_ $\mu, \nu$ _的 coupling 是定义在同一概率空间_ $\mathcal{X}$ _中的随机变量对_ $(X,Y)$_, 使得_ $X$ _的边缘分布为_ $\mu$_,_ $Y$ _的边缘分布为_ $\nu$_._

**Lemma 1**. _令_ $p_X$ _与_ $p_Y$ _为概率空间_ $\mathcal{X}$ _上的两个分布,_ $(X, Y)$ _是一对 coupling, 则 对于任意的 coupling_ $(X, Y)$ _都有_

$\Delta_{TV}(p_X, p_Y) \leq \Pr\left[X \neq Y\right],\\$

_且存在一个最优的 coupling_ $(X, Y)$ _使得_

$\Delta_{TV}(p_X, p_Y) = \Pr\left[X \neq Y\right].\\$

我们声明如下定理:

**Theorem 2**. _如果_ $\pi_\theta$ _接近_ $\pi_{\theta'}$_, 则存在它们的一个 coupling_ $(\pi_\theta, \pi_{\theta'})$ _使得_ $p_\theta(\boldsymbol{s}_t)$ _接近_ $p_{\theta'}(\boldsymbol{s}_t)$_._

这一定理中 "接近" 的表述尚不明确, 我们会在证明中给出更加明确的表述:

_Proof._ **Deterministic policy:** 对于简单的情形, 考虑 $\pi_\theta$ 是一个 deterministic policy, 使得 $\boldsymbol{a}_t = \pi_\theta(\boldsymbol{s}_t)$, 我们称 $\pi_{\theta'}$ 接近 $\pi_\theta$, 如果

$p_{\theta'}(\boldsymbol{a}_t \neq \pi_\theta(\boldsymbol{s}_t) \mid \boldsymbol{s}_t) \leq \epsilon, \forall \boldsymbol{s}_t \in \mathcal{S}\\$我们有

$p_{\theta'}(\boldsymbol{s}_t) = (1 - \epsilon)^t p_\theta(\boldsymbol{s}_t) + (1 - (1 - \epsilon)^t) p_{mistake}(\boldsymbol{s}_{t})\\$这其实就是我们在 imitation learning 中见过的方程, 类似地得到

$\sum_{\boldsymbol{s}_t} |p_{\theta'}(\boldsymbol{s}_t) - p_\theta(\boldsymbol{s}_t)| \leq 2 \epsilon t.\\$**Stochastic policy:** 考虑一般的情形, $\pi_\theta$ 是任意一个分布, 我们称 $\pi_{\theta'}$ 接近 $\pi_\theta$, 如果

$\Delta_{TV}(\pi_{\theta'}(\boldsymbol{a}_t \mid \boldsymbol{s}_t), \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)) \leq \epsilon, \forall \boldsymbol{s}_t.\\$由引理我们知道, 对于任意的 $\boldsymbol{s}_t$, 存在 $\pi_{\theta'}(\cdot\mid\boldsymbol{s}_t)$ 与 $\pi_{\theta}(\cdot\mid\boldsymbol{s}_t)$ 的 optimal coupling (可以理解为存在一种联合采样的方式), 使得它们最多有 $\epsilon$ 概率采取不同的 action, 于是我们可以得到和 deterministic 情况相同的结论.

综上, 我们有

$\sum_{\boldsymbol{s}_t} |p_{\theta'}(\boldsymbol{s}_t) - p_\theta(\boldsymbol{s}_t)| \leq 2 \epsilon t.\\$ ◻

**Remark:** 这里的 optimal coupling 如何构造其实并不重要, 核心在于当 total variational distance 足够小时, 我们"能够"使得两个分布 $p_\theta(\boldsymbol{s}_t)$, $p_{\theta'}(\boldsymbol{s}_t)$ 接近.  
  
但 TRPO 的这一部分真要细究可能是有一些不太严谨: 在我们通常的数据采集方式下, 我们并没有构造出一个 optimal coupling, 而通常是相互独立地采样, 不妨考虑 $\mathcal{A} = \{-1,1\}$, deterministic dynamic, $\pi_1(-1 \mid \boldsymbol{s}) = 0.5$ , $\pi_1(1 \mid \boldsymbol{s}) = 0.5$, $\pi_2(-1 \mid \boldsymbol{s}) = 0.6$ , $\pi_2(1 \mid \boldsymbol{s}) = 0.4$, 那么 $\Delta_{TV}(\pi_1, \pi_2) = 0.1$, 如果采用一般的独立采样, 此时两个 policy 采取同一 action 的概率为 $0.5$, 而不是 $0.9$, 自然也没办法得到上述的结论.  
  
在我们假设 $$$\pi_{\theta'}$$$ 与 $$$\pi_\theta$$$ 接近的情况下, 我们可以得到

$\begin{aligned}  \mathbb{E}_{p_{\theta'}(\boldsymbol{s}_t)} \left[f(\boldsymbol{s}_t)\right] &= \sum_{\boldsymbol{s}_t} p_{\theta'}(\boldsymbol{s}_t) f(\boldsymbol{s}_t)\\  &\geq \sum_{\boldsymbol{s}_t} p_\theta(\boldsymbol{s}_t) f(\boldsymbol{s}_t) - \sum_{\boldsymbol{s}_t} |p_{\theta'}(\boldsymbol{s}_t) - p_\theta(\boldsymbol{s}_t)| |f(\boldsymbol{s}_t)|\\  &\geq \sum_{\boldsymbol{s}_t} p_\theta(\boldsymbol{s}_t) f(\boldsymbol{s}_t) - \sum_{\boldsymbol{s}_t} |p_{\theta'}(\boldsymbol{s}_t) - p_\theta(\boldsymbol{s}_t)| \max_{\boldsymbol{s}_t} f(\boldsymbol{s}_t)\\  &\geq \mathbb{E}_{p_\theta(\boldsymbol{s}_t)} \left[f(\boldsymbol{s}_t)\right] - 2 \epsilon t \max_{\boldsymbol{s}_t} f(\boldsymbol{s}_t). \end{aligned}\\$ 只要 $\epsilon$ 足够小, 我们就能得到 $f(\boldsymbol{s}_t)$ 的期望差异足够小. 类似地

$\sum_{t} \mathbb{E}_{\boldsymbol{s}_t \sim p_{\theta'}(\boldsymbol{s}_t)} \left[\mathbb{E}_{\boldsymbol{a}_t \sim \pi_{\theta}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \left[\frac{\pi_{\theta'}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}{\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \gamma^t A^{\pi_\theta}(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]\right] \geq \bar{A}(\theta') - \sum_{t} 2\epsilon tC \\$ 其中 $C$ 是 $O(T r_{\max})$ 或者 $O(r_{\max} / (1 - \gamma))$, 分别对应 horizon 有限和无限的情形.

上述过程告诉我们, 只要 $\pi_{\theta'}$ 与 $\pi_\theta$ 足够接近, 我们优化 $\bar{A}(\theta')$ 就可以最大化 $J(\theta')$ 的一个很紧的下界. 然而优化下界等同于优化目标吗? 考虑 **MM** 算法, 随着我们不断更新 surrogate advantage, 我们可以保证 $J(\theta')$ 单调不减.

## 3 Relation to Natural Policy Gradient

上述过程指出, 若每次更新前后的策略 total variation distance 满足一定上界, 那么我们就可以通过优化 $\bar{A}(\theta')$ 的方法有效地最大化 $J(\theta')$. 于是可以得到了一个约束优化问题, 即在 total variation distance 有界的情况下最大化 $\bar{A}(\theta')$.

但一个问题是 total variation distance 是不可微的, 我们考虑将其放缩到一个可微的函数 KL divergence 上.

**Theorem 3**. _对于 total variation distance, 我们有上界_

$\Delta_{TV}(\pi_{\theta'}, \pi_\theta) \leq \sqrt{\frac{1}{2} D_{KL}(\pi_{\theta'} \parallel \pi_\theta)}\\$

于是我们实际需要解决的就是约束优化问题:

$\theta' \gets \max_{\theta'}\bar{A}(\theta') \text{ s.t. } D_{KL}(\pi_{\theta'} \parallel \pi_\theta ) \leq \epsilon\\$ 一个简单的方法是使用 Lagrange 乘子法, 定义拉格朗日函数

$\mathcal{L}(\theta', \lambda) := \bar{A}(\theta') + \lambda \left(\epsilon - D_{KL}(\pi_{\theta'} \parallel \pi_\theta )\right)\\$

我们接下来考虑这一优化问题与之前介绍的 natural policy gradient 的关系: 如果我们在 $\theta$ 处对 $\bar{A}(\theta')$ 进行一阶近似, 也就是 (我们稍微滥用了记号, 记 $\theta' = \theta$ 处的梯度为 $\nabla_{\theta} \bar{A}(\theta) := \nabla_{\theta'} \bar{A}(\theta')\big|_{\theta' = \theta}$)

$\begin{aligned}  \nabla_{\theta} \bar{A}(\theta) &= \nabla_{\theta'} \bar{A}(\theta')\big|_{\theta' = \theta}\\  &= \sum_t \mathbb{E}_{\boldsymbol{s}_t \sim p_{\theta}(\boldsymbol{s}_t)} \left[\mathbb{E}_{\boldsymbol{a}_t \sim \pi_{\theta}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \left[\frac{\pi_{\theta'}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}{\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \gamma^t \nabla_{\theta'} \log \pi_{\theta'}(\boldsymbol{a}_t \mid \boldsymbol{s}_t) A^{\pi_\theta}(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]\right] \bigg|_{\theta' = \theta}\\  &= \sum_t \mathbb{E}_{\boldsymbol{s}_t \sim p_{\theta}(\boldsymbol{s}_t)} \left[\mathbb{E}_{\boldsymbol{a}_t \sim \pi_{\theta}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \left[\frac{\pi_{\theta}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}{\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \gamma^t \nabla_\theta \log \pi_{\theta}(\boldsymbol{a}_t \mid \boldsymbol{s}_t) A^{\pi_\theta}(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]\right]\\  &= \sum_t \mathbb{E}_{\boldsymbol{s}_t \sim p_{\theta}(\boldsymbol{s}_t)} \left[\mathbb{E}_{\boldsymbol{a}_t \sim \pi_{\theta}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \left[\gamma^t \nabla_\theta \log \pi_{\theta}(\boldsymbol{a}_t \mid \boldsymbol{s}_t) A^{\pi_\theta}(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]\right]\\  &= \nabla_{\theta} J(\theta). \end{aligned}\\$ 就还原得到了标准的 policy gradient 目标. 注意在 natural policy gradient 中我们使用了 $D_{KL}$ 的二阶近似, 如此近似后就可以得到我们推导出的 natural policy gradient:

$\theta' \gets \arg\max (\theta' - \theta)^T \nabla_\theta J(\theta) \text{ s.t. } \|\theta' - \theta\|_{F(\theta)}^2 \leq \epsilon.\\$ 另一方面, 这间接说明了为什么 natural policy gradient 在一定程度上是 work 的, 因为我们优化的目标几乎是 $J(\theta')$ 的一个下界, 于是我们的更新几乎是在优化 $J(\theta')$. 同时, 不难发现标准的 policy gradient 其实使用了错误的约束, 我们的约束不应该在 $\theta$ 这样的参数空间, 而应该在分布空间中, 其中所有的参数对分布的影响是相同的.  

![](https://pica.zhimg.com/v2-a7c727ea12ba66a57573e67a4a75b6f6_1440w.jpg)

Natural Policy Gradient 的约束条件示意图

![](https://pica.zhimg.com/v2-094c22fcd685148bb185f8228b085120_1440w.jpg)

Natural Policy Gradient 效果

值得注意的是, 上述对 $\hat{A}(\theta')$ 的一阶近似并不一定是必要的, 对 KL 散度的二阶近似也不一定是必要的: 例如 TRPO, PPO Penalty, PPO Clip 等算法其实都可以从上述的拉格朗日函数出发, 通过不同的近似方式来进行优化.

## 4 Introduction to more advanced policy gradient

### 4.1 Ideas in TRPO

在 **TRPO** 中, 我们继承了 natural policy gradient 中的近似方式, 回顾我们的更新:

$\theta' = \theta + \alpha F^{-1}(\theta) \nabla_\theta J(\theta).\\$ 其中动态学习率为

$\alpha = \sqrt{\frac{2\epsilon}{(\nabla_\theta J(\theta))^T F^{-1}(\theta) \nabla_\theta J(\theta)}}.\\$ 为了简便起见, 我们记 $\hat{g} := \nabla_\theta J(\theta)$, 如果我们能够计算出 $\hat{x} := F^{-1} \hat{g}$, 那么我们就无需再计算逆矩阵, 事实上只需要进行更新:

$\theta' = \theta + \sqrt{\frac{2\epsilon}{\hat{x}^T F(\theta)\hat{x}}} \hat{x}.\\$

事实上, 我们可以通过 **conjugate gradient descent** 来计算 $\hat{x}$. 从而我们可以有效地避免计算 Fisher 信息矩阵的逆.

### line search

另一方面, 在 natural policy gradient 中, 由于我们对 KL 散度以及 surrogate advantage 进行了不同程度的近似, 故 KL 散度约束条件未必满足, 也未必能够保证 $\bar{A}(\theta') \geq 0$. 在 TRPO 中, 我们引入了 line search 的方法, 我们进行更新时考虑

$\theta' = \theta + \alpha^j \sqrt{\frac{2\epsilon}{\hat{x}^T F(\theta)\hat{x}}} \hat{x}\\$ 其中 $\alpha \in (0,1)$, 我们逐渐增大 $j$, 也就是逐渐减小更新幅度, 直到 $\bar{A}(\theta') \geq 0$ 且 KL 散度约束条件满足.

于是我们可以得到 **TRPO** 算法:

循环 $k = 0, 1, 2, \ldots$:

1.  利用 $\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)$ 采样 $\{\tau^i\}$  
    
2.  估计 advantage function $\hat{A}^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t)$  
    
3.  计算 $\hat{g}$  
    
4.  利用 conjugate gradient descent 计算 $\hat{x} = F^{-1} \hat{g}$  
    
5.  通过 line search 更新得到 $\theta_{k + 1}$.  
    

**Summary of TRPO:**

我们给出了 TRPO 算法的基本思想, 我们的

-   TRPO 算法继承了 natural policy gradient 的理论严谨性, 并引入了一些实际的改进.  
    
-   TRPO 算法有效通过 conjugate gradient descent 避免了计算 Fisher 信息矩阵的逆. 此时我们只需要计算 Fisher 信息矩阵即可, 然而这依然相当昂贵.  
    
-   与 natural policy gradient 相比, TRPO 中进行了改进检查, 只有当更新的确能够增大目标函数且符合约束时才进行更新.  
    
-   由于 TRPO 不使用常规梯度下降, 因此也不能使用 Adam 等优化器.  
    

### 4.2 Ideas in PPO Penalty

我们可以考虑使用如下的 **dual gradient descent**, 也就是迭代进行如下两步:

1.  对于 $\theta'$ 最大化 $\mathcal{L}(\theta', \lambda)$.  
    
2.  $\lambda \gets \lambda + \alpha \left(D_{KL}(\pi_{\theta'}(\boldsymbol{a}_t \mid \boldsymbol{s}_t) \parallel \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t) - \epsilon)\right)$.  
    

这里的 intuition 是, 如果约束条件违反过大, 那么我们就增大 $\lambda$, 使得约束条件更加严格, 否则就降低.

基于这一思路, 并配合一些实际的 design choices, 我们可以得到 **PPO Penalty** 算法的基本思路 (更加详细的算法可以仿照后续的 \\textbf{**PPO Clip 进行设计**}):

循环 $k = 0, 1, 2, \ldots$:

1.  利用 $\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)$ 采样 $\{\tau^i\}$  
    
2.  估计 advantage function $\hat{A}^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t)$  
    
3.  利用 $K$ 步梯度下降计算 $\theta_{k + 1} = \arg\max_{\theta'} \bar{A}(\theta') - \beta_k D_{KL}(\pi_{\theta'} \parallel \pi_{\theta_k})$  
    
4.  如果 $D_{KL}(\pi_{\theta_{k + 1}} \parallel \pi_{\theta_k}) > 1.5 \epsilon$, 则增大惩罚项 $\beta_{k + 1} = 2\beta_{k}$. 如果 $D_{KL}(\pi_{\theta_{k + 1}} \parallel \pi_{\theta_k}) < 2\epsilon/3$, 则减小惩罚项 $\beta_{k + 1} = \beta_{k} / 2$. 否则 $\beta_{k + 1} = \beta_k$.  
    

一个直观的感受是, PPO Penalty 与 TRPO 相比非常容易实现.

### 4.3 Ideas in PPO Clip

通常当我们提到 PPO 时, 我们指的是 **PPO Clip**. PPO Clip 的一个关键思想是, 我们不再使用 KL 散度的约束, 而是直接使用 IS objective, 以牺牲理论的严谨性为代价, 将 KL 散度对更新范围的约束转化为对 IS ratio 的约束.

在 PPO Clip 中, 我们使用如下的 surrogate objective:

$\bar{A}(\theta')^{CLIP} = \mathbb{E}_{\tau \sim p_{\theta}(\tau)} \left[\sum_{t} \min\left(\rho(\theta', \theta) \hat{A}_t, \text{clip}(\rho(\theta', \theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t\right)\right].\\$  
**Remark:**

-   这里 $\rho(\theta', \theta) = \frac{\pi_{\theta'}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}{\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}$ 是 IS ratio.  
    
-   clip 函数是一个截断函数, 这里 $$$1 - \epsilon$$$ 与 $$$1 + \epsilon$$$ 分别是上下界, 保证 IS ratio 不会过大或过小:

-   当估计的 advantage $\hat{A}_t$ 为正, 这说明我们新的 policy 要增大这一 action 的概率, $$$\rho(\theta',\theta) > 1$$$ , 然而更新幅度不能太大, 我们截断 IS ratio 为 $$$1 + \epsilon$$$ , 当被截断时该项没有梯度, 也即我们不会基于这一项进行更新.
-   当估计的 advantage $$$\hat{A}_t$$$ 为负, 这说明我们新的 policy 要减小这一 action 的概率, $$$\rho(\theta',\theta) < 1$$$ , 然而更新幅度不能太大, 我们截断 IS ratio 为 $$$1 - \epsilon$$$ , 当被截断时该项没有梯度, 也即说我们不会基于这一项进行更新.  
    

-   这里 $$$\hat{A}_t$$$ 是估计的 advantage, 我们会使用在 **Actor-Critic** 一节中介绍的 **GAE** 进行估计: $$$\hat{A}^\pi_{GAE}(\boldsymbol{s}_t, \boldsymbol{a}_t) = \sum_{t' = t}^\infty (\gamma \lambda)^{t' - t} \left(r(\boldsymbol{s}_{t'}, \boldsymbol{a}_{t'}) + \gamma \hat{V}^\pi(\boldsymbol{s}_{t' + 1}) - \hat{V}^\pi(\boldsymbol{s}_{t'})\right).\\$$$ 而这里的一个 design choice 是我们仅仅使用 $$$T_0$$$ 步 (不是 episode 长度, 而是一个远小于 episode 长度的量) 的 GAE 估计, 回顾我们 GAE 需要满足的 **on-policy** 假设, 我们会在接下来的算法描述中保证这一点.  
    
-   上述过程中我们使用的 $$$V^\pi$$$ 是一个和 actor 共享参数的网络, 我们构建如下的 objective 使得它们同步地优化: $$$L(\theta') = \bar{A}(\theta')^{CLIP} - c_1 L_{VF}(\theta') + c_2 \mathcal{H}(\pi_{\theta'}),\\$$$这里第二项是 value function 的 MSE loss, 最后一项是 entropy bonus, 用于增加探索性, 我们会在 **exploration** 一节中进一步讨论.

于是我们也可以得到 **PPO Clip** 的算法:

![](https://pic3.zhimg.com/v2-8a7d94d01046b59245f3f518f17a4f1e_1440w.jpg)

### Summary of PPO

在 PPO 算法中, 我们牺牲了部分理论的保证, 但相较于 TRPO 算法的好处在于, 我们利用多步梯度下降取代了 TRPO 中的二阶近似, 既降低了计算成本, 也让我们的算法更加容易实现, 同时也让我们能够使用 Adam 等优化器.

## 5 Summary

本节中, 我们:

-   从数学推导出发给出了优化目标函数核心的 surrogate advantage  
    
-   给出了基于这一理论结果与 natural policy gradient 的关系  
    
-   natural policy gradient 的优化算法 TRPO  
    
-   在牺牲一些数学严谨性的前提下, 给出了 PPO Penalty 与 PPO Clip 的优化算法及其背后的思想