## 1 Introduction to Offline RL

### 1.1 Why we need offline RL?

在当前的深度学习中, 我们通常已经可以得到相当不错的泛化性, 例如在一些分类任务上, 一个在大量数据上训练的模型可以在未见过的数据上取得很好的效果, 这样的效果对于光照, 视角等变化都相当 robust. 然而, 在 RL 中, 通常我们难以实现很好的泛化性, 即使是稍微改变光照, 也可能导致整个 policy 的崩溃.

![](https://pic2.zhimg.com/v2-77dfeb0ed455d30a0ba7356cf19cd249_1440w.jpg)

目前的强化学习与监督学习在泛化性有着巨大差异

如果思考背后的原因, 当前深度学习之所以能够取得巨大的成功, 核心的要素是 large data + large model. 然而, 我们之前介绍的 RL 算法与监督学习的情境有着很大的差异: 我们的 RL 算法通常依赖不断与环境交互以收集数据, 如果在真实世界收集, 则这一过程的效率相当低下, 如果在虚拟环境中生成, 则通常又缺乏 diversity.

回顾过去我们介绍的 RL 算法, 无论是 on-policy 还是 off-policy 算法 (虽然前者样本效率更低), 这些 online 算法对于每一个新任务都必须从头开始收集数据, 在完成一个任务后, 所用的数据就丢弃了. 通常情况下, 我们不可能从零开始收集一个 ImageNet 规模的数据集. 没有大规模的数据集, 我们也自然难以训练一个大规模的模型.

![](https://pica.zhimg.com/v2-ddb9d252c15cfd195704147be66d210c_1440w.jpg)

无论是 on-policy 还是 off-policy 算法, 这些 online 算法对于每一个新任务都必须从头开始收集数据

在正式介绍 offline RL 的具体设定之前, 我们先考虑其背后的基本想法: offline RL 的目的就是获得一种 data-drivin 的 RL 方法. 在 offline RL 中:

-   假设我们有一个用 **某种方式** 收集的固定数据集 $\mathcal{D}$. (**Note: 数据集未必对应于 expert data**)  
    
-   我们的目标是从这个数据集中学习一个 policy, 在这个学习过程中我们无法与环境交互.  
    
-   最后我们将学习到的 policy 部署 (deploy)到真实环境中.

![](https://pic2.zhimg.com/v2-911918b16f8cf94ee15ac2cc79b9171d_1440w.jpg)

offline RL 的基本范式

### 1.2 Formalization of offline RL

我们给出 offline RL 的 formalization:

-   **dataset**: $\mathcal{D} = \{(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_{i}', r_i)\}$  
    
-   **states**: $\boldsymbol{s} \sim d^{\pi_\beta}(\boldsymbol{s})$, 这里 $d^{\pi_\beta}(\boldsymbol{s})$ 是由 policy $\pi_\beta$ 产生的 state distribution  
    
-   **actions**: $\boldsymbol{a} \sim \pi_\beta(\boldsymbol{a}\mid \boldsymbol{s})$  
    
-   **next states**: $\boldsymbol{s}' \sim p(\boldsymbol{s}'\mid \boldsymbol{s}, \boldsymbol{a})$  
    
-   **rewards**: $r \sim r(\boldsymbol{s}, \boldsymbol{a})$  
    

这里的 $\beta$ 是我们未知的收集数据的 policy.

offline RL 的基础是 **[off-policy evaluation](https://zhida.zhihu.com/search?content_id=255084074&content_type=Article&match_order=1&q=off-policy+evaluation&zhida_source=entity) (OPE)**, 也就是**仅仅利用固定数据集** $\mathcal{D}$ 评估当前 policy 的好坏. 评估当前 policy 的好坏是我们改进 policy 的基础.

而 **offline RL** (有时候也称作 **batch RL**, **fully off-policy RL**) 的 **objective** 则是: 给定 $\mathcal{D}$, 学习 **best possible** 的 $\pi_\theta$. (**Note:** best possible 并不一定是 MDP 中的最优策略, 而只是在 $\mathcal{D}$ 能够学习到的最优策略)

### 1.3 Expectation and intuition in offline RL

在上述的 formalization 中, 我们提出 offline RL 能够学到 best possible 的 policy. 但是我们的数据集包含了质量参差不齐的行为, 这是可能的吗? 实际上这基于以下几点期望:

-   **Chaos to order**: offline RL 能够从具有好行为与差行为的 $\mathcal{D}$ 中找到其中好的部分

![](https://pic2.zhimg.com/v2-e082338c162c9fb0224e39b0e308fa8b_1440w.jpg)

上图中对应于 imitation learning 中学习 expert policy, 下图则是 offline RL 中我们要从一系列或好或坏的数据中学习

-   **Generalization**: 一个地方的好行为可能在另一个地方也是好的 (即使另一个地方没有这一好的行为的数据)  
    
-   **"Stitching"**: 不同行为中好的部分可以被拼接重组. 一个例子是, 如果我们有打开抽屉的数据, 以及从打开的抽屉中拿东西的数据, 那么通过 offline RL 我们可以学习到如何打开抽屉并拿东西.

![](https://picx.zhimg.com/v2-3dd42b40a94872bc97c459d56ed05e55_1440w.jpg)

我们期望 offline RL 能够具有将所学的&quot;技能&quot;拼接起来的能力

一个容易误解的地方是 offline RL 与 imitation learning 的关系:

-   offline RL **不是 imitation learning**: offline RL 理论上通常可以比 dataset 中最好的 policy 更好, 而 imitation learning 通常只能期望得到 dataset 中的平均 policy. 可以证明在某些 structural assumptions 下, offline RL 可以比 imitation learning 更好, 即使数据是 optimal 的.

如果我们有一个 offline RL 的算法, 那么我们在每次开始新任务时就可以重新利用我们已有的数据, 同时也可以利用其他人在其他任务上收集的数据, 此时训练成本会大大降低, 在每次训练的时候就不需要收集一个 ImageNet 大小的数据集来实现好的 generalization.

![](https://pic4.zhimg.com/v2-79891c530d784099412618caec9792e7_1440w.jpg)

offline RL 期望实现的愿景

### 1.4 Challenges in offline RL

然而, 在 offline RL 中一些挑战:

**Example 1**. _在 [QT-Opt](https://zhida.zhihu.com/search?content_id=255084074&content_type=Article&match_order=1&q=QT-Opt&zhida_source=entity): Scalable Deep Reinforcement Learning of Vision-Based Robotic Manipulation Skills 中, 作者通过 offline training + online finetuning 的方式. 也就是先进行 offline RL, 然后在真实环境中进行微调._

_文中使用的 offline 数据有 580k, 而 online 数据仅有 28k (使用最新 policy 收集). 结果是仅使用 offline 时我们仅仅获得了 $87\%$ 的成功率, 而经过 online 微调后能够达到 $96\%$ 成功率. 这在一定程度上说明了 offline RL + online finetuning 的有效性._

_然而实际上 online 额外引入的数据不到 offline 的 $5\%$, 却产生了明显的表现提升, 似乎扩大 offline 数据集规模还没有最后很少的一点 online 数据来的好. 直观来说, 在 offline RL 中一定额外存在了一些 online RL 中不存在的问题, 限制了 offline RL 的表现._

![](https://pica.zhimg.com/v2-5f7fc38b82085952b8884ecd5f309c4c_1440w.jpg)

利用 offline learning + online finetuning 的方式训练一个抓取物体的 policy

**Example 2**. _从 [Stabilizing Off-Policy Q-Learning](https://zhida.zhihu.com/search?content_id=255084074&content_type=Article&match_order=1&q=Stabilizing+Off-Policy+Q-Learning&zhida_source=entity) via Bootstrapping Error Reduction 中我们发现, 对于不同规模的数据, 实际的表现几乎没有区别, reward 在 $-750$ 附近徘徊. 但是如果我们观察 Q-values 的曲线, 会发现估计的 Q-values 会非常大._

![](https://picx.zhimg.com/v2-d388406f9b25bc4015766132a1df6353_1440w.jpg)

offline RL 极为严重的 overestimation 问题

_事实上, 这并不是一个巧合!_

这里的一个根本性问题是: **counterfactual queries**:

一个例子是, 对于人类驾驶员来说, 即使其水平再差, 也有一些永远不会做的事情, 例如在马路中间进行一个急转弯, 因此我们的数据集中不可能 cover 所有可能的行为. 但是在训练时, 当我们处在一定的 state 中, 会进行一个比较: 那些 **out of distribution (OOD)** 的行为是否比分布内的行为更好?

-   在 online RL 中, 我们不用担心这个问题: 如果我们发现依据当前 policy, 有一个 OOD 行为的 reward 很高. 真金不怕火炼, 我们可以尝试一遍, 如果这个 OOD 行为很好, 自然能够通过检验, 但如果产生了糟糕的结果, 也就能学到这个 OOD 的行为是不好的 (当然, 这也是为什么不希望将这些算法用于现实世界中一些危险的场景).  
    
-   但是在 offline RL 中, 我们没有这样真实尝试的机会. 我们需要有办法处理这些 OOD 的行为, 例如想办法避免它们. 但是同时我们不能 naively 要求行为一定 **出现** 在 $\mathcal{D}$ 中, 因为我们需要泛化到不在 $\mathcal{D}$ 中但是依然 in-distribution 的行为.

![](https://pic1.zhimg.com/v2-e2b20b33dfc9f8e011da73f9559e7b3a_1440w.jpg)

对于那些 OOD 的 action, 我们的 Q function 可能会有很大的偏差

### 1.5 Math in distribution shift

上述问题也可以理解是 $\pi$ 与 $\pi_{\beta}$ 之间的 **distribution shift**. 如果我们有 $\pi$ 对应的数据, 那么相当于我们就可以 online 检验. 前面我们采用了一种直观地理解方式, 事实上我们可以通过数学的方式来理解这一问题.

回顾在监督学习中, 我们通常解决的是一个 **empirical risk minimization (ERM)** problem: $\theta \gets \arg\min_\theta \mathbb{E}_{\boldsymbol{x} \sim p(\boldsymbol{x}), y \sim p(y\mid \boldsymbol{x})} \left[(f_\theta(\boldsymbol{x}) - y)^2\right],\\$ 这里的 $p(x), p(y\mid x)$ 是训练数据的分布.

基于我们对监督学习的认识, 如果没有 overfit, 那么我们期望在统计意义上, $\mathbb{E}_{\boldsymbol{x} \sim p(\boldsymbol{x}), y \sim p(y\mid \boldsymbol{x})} \left[(f_\theta(\boldsymbol{x}) - y)^2\right]\\$ 会很低. 但是对于另一个分布 $\bar{p}(\boldsymbol{x})$, $\mathbb{E}_{\boldsymbol{x} \sim \bar{p}(\boldsymbol{x}), y \sim p(y\mid \boldsymbol{x})} \left[(f_\theta(\boldsymbol{x}) - y)^2\right]\\$ 可能会很高. 然而值得注意的是, 上述只是统计上的结果, 即使 $\boldsymbol{x}^\ast \sim p(\boldsymbol{x})$, 我们的 error 也可能很高.

在监督学习中, 我们可以构造出 adversarial examples, 刻意选取一个 $\boldsymbol{x}^\ast \gets \arg\max_{\boldsymbol{x}} f_\theta(\boldsymbol{x})$, 此时我们可能会得到极大的误差 $(f_\theta(\boldsymbol{x}^\ast) - y)$. 类似地, 我们也可以构造出 adversarial distribution $\bar{p}(\boldsymbol{x})$, 使得这个分布上的误差都很大.

### 1.6 Math in distribution shift in offline RL

但是这和我们的 offline RL 有什么关系呢? 在 offline RL 中, 我们更新 policy 的方式实际上就在构造这样的 adversarial distribution!

回顾 off-policy 的 Q-learning 中, 我们会进行如下的 Bellman backup: $Q(\boldsymbol{s}, \boldsymbol{a}) \gets r(\boldsymbol{s}, \boldsymbol{a}) + \mathbb{E}_{\boldsymbol{a'} \sim \pi_{new}(\boldsymbol{a}' \mid \boldsymbol{s}')}[Q(\boldsymbol{s}', \boldsymbol{a'})]\\$其中 $\boldsymbol{s}, \boldsymbol{a}$ 是从 $\mathcal{D}$ 中采样的, 而 $\boldsymbol{a}'$ 来源于最新的 argmax policy $\pi_{new}$ (实际上 Q-function 更新的标准写法应该是 $\arg\max_{\boldsymbol{a}'} Q(\boldsymbol{s}', \boldsymbol{a}')$, 但是为了更好兼容像 [actor-critic](https://zhida.zhihu.com/search?content_id=255084074&content_type=Article&match_order=1&q=actor-critic&zhida_source=entity) 等算法, 我们使用上述写法). 为了记号简便, 记 $y(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \mathbb{E}_{\boldsymbol{a'} \sim \pi_{new}}[Q(\boldsymbol{s}', \boldsymbol{a'})]$.

我们的整个 Q-learning 可以理解为是在循环进行 "**training/ update**" 和 "**evaluation**" 的过程, training 是通过 objective $\mathcal{L} = \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}) \sim \pi_\beta(\boldsymbol{s}, \boldsymbol{a})} \left[(Q(\boldsymbol{s}, \boldsymbol{a}) - y(\boldsymbol{s}, \boldsymbol{a}))^2\right],\\$来更新 $Q$ function, 而 evaluation 则是利用训练得到的 $Q$ function 来计算 $y(\boldsymbol{s}, \boldsymbol{a})$. 不难理解, 其中 training/ update 发生在 $\pi_\beta$ 上, 而 evaluation 发生在 $\pi_{new}$ 上.

如果 $\pi_{new} = \pi_\beta$, 那么我们 evaluation 与 training 的分布是一致的, 我们应当期望 Q-function 能得到很好的结果.

然而事实上在 evaluation 中获取 $\mathbb{E}_{\boldsymbol{a'} \sim \pi_{new}}[Q(\boldsymbol{s}', \boldsymbol{a'})]$ 时, 我们 query 了 OOD 的数据. 具体来说, $Q$ function 在一系列不同的 action 上对应不同的值, 并不是所有的 action 都在分布内. 在那些 OOD 的 action 上, 我们的 $Q$ function 可能会被高估. 由于 Q-learning 中我们会使用 argmax policy, 可以理解为我们的 $\pi_{new}$ 会在那些 Q-value 极高的地方进行选择. 这相当于是人为构造了一个 adversarial distribution $\arg\max_{\pi} \mathbb{E}_{\boldsymbol{a}' \sim \pi(\boldsymbol{a}'\mid \boldsymbol{s}')} \left[Q(\boldsymbol{s}', \boldsymbol{a}')\right],\\$并使用在这个 distribution 上的 action 进行 evaluation, 将其结果作为 $y(\boldsymbol{s}, \boldsymbol{a})$. 这个结果自然是偏差极大的.

而我们的 training 目标依赖于这个 evaluation 的结果, 也就是我们在更新 $Q$ function 时, 会使得 $Q(\boldsymbol{s}, \boldsymbol{a})$ 尽可能地接近 $y(\boldsymbol{s}, \boldsymbol{a})$. 但是我们的 $y(\boldsymbol{s}, \boldsymbol{a})$ 来自于构造的 adversarial sample, 也就是我们在更新 $Q$ function 时, 会使得 $Q(\boldsymbol{s}, \boldsymbol{a})$ 尽可能地接近一个高估的值. 长此以往, $Q$ function 就会向着高估的方向发展.

**Side Note:** 上述分析主要是针对 Q-learning 的, 由于其使用了 argmax policy, 因此分析起来更加直观. 但是类似的问题也会出现在 actor-critic 等算法中.

### 1.7 Comparison with overestimation in online Q-learning

我们似乎在 Q-Learning 中也见到了 overestimation 问题, 它们的本质其实有很大的差异:

-   在 Q-Learning 中, 我们的问题来自于 $\max$ 操作在 noise 的影响下的高估, 也就是 $\mathbb{E}[\max(X, Y)] \geq \max(\mathbb{E}[X], \mathbb{E}[Y]),\\$ 而我们想要的是 $\max(\mathbb{E}[X], \mathbb{E}[Y])$ 但是我们实际估计出的是 $\mathbb{E}[\max(X, Y)]$.  
    
-   在 offline RL 中, 核心的问题是 distribution shift 引起的, 导致我们的 $Q$ function 不断拟合 adversarial distribution 上的结果, 从而使得 $Q$ function 不断被高估. 而在在 online RL 中, 由于会定期更新数据, 一旦某个 Q-value 被高估, 那么下次收集数据时, 会有一定概率被真实的 reward signal 纠正.  
    

上述分析实际告诉了我们: 很多在 online 情况下可以被纠正的错误在 offline 情况下是无法被纠正的. 在常规 RL 挑战中的 sampling error 与 function approximation error 在 offline RL 中会变得更加严重.

![](https://picx.zhimg.com/v2-0312e439cf485e266373f66810da27cd_1440w.jpg)

假设中间的位置缺少数据, value function 预测很高的值, 在 online RL 中我们可以尝试并发现其实际价值, 但是在 offline RL 中没有办法纠错

## 2 Batch RL via Importance Sampling

在接下来的这两节中, 我们会讨论在 Deep RL 之前的一些 offline RL 算法. 现如今我们通常不会将这些算法作为一个默认选择, 但是其背后的思想是当前 offline RL 算法的基础. (**Side Note:** 在很多早期的 literature, 这称为 batch RL).

在这一节中, 我们先考虑 **offline RL with policy gradient**:

### 2.1 Basic ideas and problems

考虑 RL objective: $J(\theta) = \max_\pi \sum_{t = 1}^{T} \mathbb{E}_{\boldsymbol{s}_t \sim d^{\pi_\theta}(\boldsymbol{s}), \boldsymbol{a}_t \sim \pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})} \left[\gamma^t r(\boldsymbol{s}_t, \boldsymbol{a}_t)\right],\\$ 这里我们将 $\pi$ 用 $\theta$ 参数化, 梯度可以表示为 $\begin{aligned} \nabla_\theta J(\theta) &= \mathbb{E}_{\tau \sim \pi_\theta(\tau)} \left[\sum_{t = 1}^{T} \nabla_\theta \gamma^t\log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t) \hat{Q}(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]\\ &\approx \frac{1}{N} \sum_{i = 1}^{N} \sum_{t = 1}^{T} \nabla_\theta \gamma^t \log \pi_\theta(\boldsymbol{a}_{i, t} \mid \boldsymbol{s}_{i, t}) \hat{Q}(\boldsymbol{s}_{i, t}, \boldsymbol{a}_{i, t}) \end{aligned}\\$ 其中 $\hat{Q}(\boldsymbol{s}_{i, t}, \boldsymbol{a}_{i, t}) \approx \sum_{t' = t}^{T} \gamma^{t' - t} r_{t', i}$ 是 reward to go, 但是很显然这里的梯度估计需要我们有 $\pi_\theta$ 的 samples, 而由于我们只有 $\pi_\beta$ 的数据, 因此我们使用 importance sampling, 也就是 $\nabla_\theta J(\theta) = \frac{1}{N} \sum_{i = 1}^{N} \frac{\pi_\theta(\tau_i)}{\pi_\beta(\tau_i)} \sum_{t = 1}^{T} \nabla_\theta \gamma^t \log \pi_\theta(\boldsymbol{a}_{i, t} \mid \boldsymbol{s}_{i, t}) \hat{Q}(\boldsymbol{s}_{i, t}, \boldsymbol{a}_{i, t})\\$ 而这个 importance ratio 可以表示为 $\frac{\pi_\theta(\tau_i)}{\pi_\beta(\tau_i)}= \prod_{t = 1}^{T} \frac{\pi_\theta(\boldsymbol{a}_{i, t} \mid \boldsymbol{s}_{i, t})}{\pi_\beta(\boldsymbol{a}_{i, t} \mid \boldsymbol{s}_{i, t})}\\$ 这是一个 exponential in $T$ 的式子, 因此虽然这一估计是 unbiased 的, 但是 variance 会非常大.

### 2.2 Unable to ignore the past

在 **advanced policy gradient** 一节的讨论, 如果 $\pi_\theta$ 与 $\pi_{\theta'}$ 足够接近, 则 $J(\theta') - J(\theta) \approx \mathbb{E}_{\tau \sim p_{\theta}(\tau)} \left[\sum_{t} \frac{\pi_{\theta'}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}{\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \gamma^t A^{\pi_\theta}(\boldsymbol{s}_t, \boldsymbol{a}_t)\right].\\$ 这个式子在这个 setting 的理解方式与之前**并不一样**, 此时 $\theta$ 对应于有数据的分布, $\theta'$ 是我们目前需要改进的 policy 的参数. 由于 $J(\theta)$ 是固定的, 我们通过梯度上升来最大化 $J(\theta')$, 对 $\nabla_{\theta'} J(\theta')$ 的估计可以通过 $\nabla_{\theta'} \mathbb{E}_{\tau \sim p_{\theta}(\tau)} \left[\sum_{t} \frac{\pi_{\theta'}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}{\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \gamma^t A^{\pi_\theta}(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]\\$ 来进行, 其中的 $A^{\pi_\theta}(\boldsymbol{s}_t, \boldsymbol{a}_t)$ 可以替换为数据集中的 reward to go.

在这个意义下, 如果 $\pi_\theta$ 与 $\pi_\beta$ 足够接近, 也就是如果 policy 的改变量在一定范围内, 则可将 $t$ 时间步之前的 ratio 全部忽略掉, 但是很显然这在 offline RL 中通常是不成立的, 因为我们的目标 policy 比 $\pi_\beta$ 要好很多.

### 2.3 Applying Causality

因此我们将 ratio 拆分为以下部分: $\frac{1}{N} \sum_{i = 1}^{N} \sum_{t = 1}^{T} \left(\prod_{t' = 1}^{t - 1} \frac{\pi_\theta(\boldsymbol{a}_{i, t'} \mid \boldsymbol{s}_{i, t'})}{\pi_\beta(\boldsymbol{a}_{i, t'} \mid \boldsymbol{s}_{i, t'})}\right) \nabla_\theta \gamma^t \log \pi_\theta(\boldsymbol{a}_{i, t} \mid \boldsymbol{s}_{i, t}) \left(\prod_{t' = t}^{T} \frac{\pi_\theta(\boldsymbol{a}_{i, t'} \mid \boldsymbol{s}_{i, t'})}{\pi_\beta(\boldsymbol{a}_{i, t'} \mid \boldsymbol{s}_{i, t'})}\right)\hat{Q}(\boldsymbol{s}_{i, t}, \boldsymbol{a}_{i, t})\\$ 前一个累乘考虑了我们 到达 $s_{i, t}$ 的概率差异, 后一个累乘考虑了 reward to go 的差异.

考虑对 return 的估计部分, 我们事实上可以进一步化简 $\begin{aligned} \left(\prod_{t' = t}^{T} \frac{\pi_\theta(\boldsymbol{a}_{i, t'} \mid \boldsymbol{s}_{i, t'})}{\pi_\beta(\boldsymbol{a}_{i, t'} \mid \boldsymbol{s}_{i, t'})}\right)\hat{Q}(\boldsymbol{s}_{i, t}, \boldsymbol{a}_{i, t}) &= \sum_{t' = t}^{T} \left(\prod_{t'' = t}^{T} \frac{\pi_\theta(\boldsymbol{a}_{i, t''} \mid \boldsymbol{s}_{i, t''})}{\pi_\beta(\boldsymbol{a}_{i, t''} \mid \boldsymbol{s}_{i, t''})}\right) \gamma^{t' - t} r_{i, t'}\\ &= \sum_{t' = t}^{T} \left(\prod_{t'' = t}^{t'} \frac{\pi_\theta(\boldsymbol{a}_{i, t''} \mid \boldsymbol{s}_{i, t''})}{\pi_\beta(\boldsymbol{a}_{i, t''} \mid \boldsymbol{s}_{i, t''})}\right) \gamma^{t' - t} r_{i, t'}\\ \end{aligned}\\$ 其中后一个等号利用了 causality. 然而这个式子仍然是 exponential in $T$ 的, 但我们稍微改进了一点. 事实上, 为了避免 exponentially exploding importance weights 的问题, 我们 **必须** 使用 value function estimation.

接下来我们介绍 causality 外其他改进过高 variance 的方法:

### 2.4 Doubly robust estimator

Doubly robust estimator 是统计学上的一个概念, 通俗来讲是利用两个 estimator 来使得估计更加稳定. 在 RL 中, 这准确来说属于一个 **off-policy evaluation** 算法而不是 RL 算法, 通过这个算法我们可以得到 value 的估计值, 从而将其放入 importance sampling gradient estimator 中.

首先依然重申我们想要改进 policy 的前提是我们能够 evaluation policy, 这等价于我们能够估计 $V^{\pi_\theta}(\boldsymbol{s})$. 在我们原先的 importance sampling estimator 中, 我们有 $\begin{aligned} V^{\pi_\theta}(\boldsymbol{s}_1) &\approx \sum_{t = 1}^{T} \left(\prod_{t' = 1}^{t} \frac{\pi_\theta(\boldsymbol{a}_{t'} \mid \boldsymbol{s}_{t'})}{\pi_\beta(\boldsymbol{a}_{t'} \mid \boldsymbol{s}_{t'})}\right) \gamma^t r_{t}\\ &= \sum_{t = 1}^{T} \left(\prod_{t' = 1}^{t} \rho_{t'}\right) \gamma^t r_{t}\\ &= \rho_1 (r_1 + \gamma (\rho_2 r_2 + \gamma (\rho_3 r_3 + \cdots)))\\ &= \bar{V}^T \end{aligned} \\$ 此时存在着递归关系 $\bar{V}^{T + 1 - t} = \rho_t (r_{t} + \gamma \bar{V}^{T - t})$. 如果令 $\bar{V}^0 = 0$, 那么我们就可以递归地计算出 $\bar{V}^T$, 也就能够 evaluation policy. 但是这样的 estimator 有很大的 variance.

我们可以考虑使用仅 value function estimator $\hat{V}(\boldsymbol{s})$ 或 $\hat{Q}(\boldsymbol{s}, \boldsymbol{a})$ 来作为 estimator, 此时可能类似于 value function baseline, 由于 value function 本身的误差, 我们的估计可能是 biased 的.

于是我们可以考虑将上述两个 estimator 结合起来, 得到一个 doubly robust estimator: $\bar{V}_{DR}^{T + 1 - t} = \hat{V}(\boldsymbol{s}_t) + \rho_t (r_{t} + \gamma \bar{V}_{DR}^{T - t} - \hat{Q}(\boldsymbol{s}_t, \boldsymbol{a}_t)),\\$ 这里 $\hat{V}(\boldsymbol{s}_t)$ 可以通过 $\hat{Q}(\boldsymbol{s}_t, \boldsymbol{a}_t)$ 得到. 一个直观的比喻是, value function estimator 是一个基本估计, 可以防止方差爆炸, 而 importance sampling estimator 是一个校正方式, 可以帮助我们纠正 bias.

参见: Jiang, N. and Li, L. (2015). Doubly robust off-policy value evaluation for reinforcement learning

### 2.5 Marginalized importance sampling

和前面的 doubly robust estimator 类似, 这一方法在 literature 中属于 off-policy evaluation 算法而不是 RL 算法.

在我们之前讨论的 importance sampling 中, 我们都是将 ratio 表示为 action 的概率的比值, 而事实上我们也可以将其表示为 state probability 的比值, 也就是使用 $w(\boldsymbol{s},\boldsymbol{a}) = \frac{d^{\pi_\theta}(\boldsymbol{s},\boldsymbol{a})}{d^{\pi_\beta}(\boldsymbol{s},\boldsymbol{a})},\\$很显然我们不知道这个比值, 但如果我们能够估计这一点, 那么就能通过 $J(\theta) \approx \frac{1}{N} \sum_{i} w(\boldsymbol{s}_i, \boldsymbol{a}_i) r_i\\$来计算我们的 objective.

通常我们会使用神经网络来表示我们的 $w(\boldsymbol{s}, \boldsymbol{a})$, 并列出一系列 consistency condition, 用处理 Bellman equation 的方式处理, 只不过我们求解的是 $w(\boldsymbol{s}, \boldsymbol{a})$.

一个例子是 (Zhang et al., GenDICE: Generalized Offline Estimation of Stationary Values), 我们考虑以下 weighted marginal $p_\pi(\boldsymbol{s}, \boldsymbol{a}) = (1 - \gamma) \sum_{t = 0}^T \gamma^t p(\boldsymbol{s}_t = \boldsymbol{s}, \boldsymbol{a}_t = \boldsymbol{a})\\$ 于是我们可以列出以下等式 (类似于我们求解 Markov Chain 的 stationary distribution): $d^{\pi_\beta}(\boldsymbol{s}', \boldsymbol{a}') w(\boldsymbol{s}', \boldsymbol{a}') = (1 - \gamma) p_0(\boldsymbol{s}') \pi_\theta(\boldsymbol{a}' \mid \boldsymbol{s}') + \gamma \sum_{\boldsymbol{s}, \boldsymbol{a}} p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a}) \pi_\theta(\boldsymbol{a}' \mid \boldsymbol{s}') d^{\pi_\beta}(\boldsymbol{s}, \boldsymbol{a}) w(\boldsymbol{s}, \boldsymbol{a})\\$ 等式的左边就是在 $\pi_\theta$ 下见到 $(\boldsymbol{s}',\boldsymbol{a}')$ 的概率, 等式右侧第一项等于我们开始于 $\boldsymbol{s}'$ 且采取 $\boldsymbol{a}'$ 的概率, 第二项表示从另一状态到达 $\boldsymbol{s}'$ 且采取 $\boldsymbol{a}'$ 的概率. 具体的求解方法可以参见原论文.

### 2.6 Additional readings: importance sampling

-   Classic work on importance sampled policy gradients and return estimation:

-   Precup, D. (2000). Eligibility traces for off-policy policy evaluation.
-   Peshkin, L. and Shelton, C. R. (2002). Learning from scarce experience.  
    

-   Doubly robust estimators and other improved importance-sampling estimators:  
    

-   Jiang, N. and Li, L. (2015). Doubly robust off-policy value evaluation for reinforcement learning.
-   Thomas, P. and Brunskill, E. (2016). Data-efficient off-policy policy evaluation for reinforcement learning.  
    

-   Analysis and theory:  
    

-   Thomas, P. S., Theocharous, G., and Ghavamzadeh, M. (2015). High-confidence off-policy evaluation.  
    

-   Marginalized importance sampling:  
    

-   Hallak, A. and Mannor, S. (2017). Consistent on-line off-policy evaluation.
-   Liu, Y., Swaminathan, A., Agarwal, A., and Brunskill, E. (2019). Off-policy policy gradient with state distribution correction

## 3 Batch RL via Linear Fitted Value Functions

我们介绍的第二种 classic offline RL 算法是基于 linear fitted value functions 的. 尽管现如今我们使用深度神经网络来表示 value function, 但是 linear case 中我们可以得到 close form 的解, 这或许可以给我们研究更加有效的 offline RL 算法提供一些 intuitions.

### 3.1 Offline value function estimation

实际上, 过去基于 value function 的 offline RL 算法与当前的 deep RL 的关注点并不相同:

-   在过去的 offline RL 中, 人们通常考虑将 dynamic programming 与 Q-learning 中的现有 idea 应用在 offline setting 中, 并且通过 linear function approximation 来获取闭合形式的解.  
    
-   在现在的 offline RL 中, 我们通常会使用更加有表示力的模型, 例如深度神经网络, 并且主要的挑战是 distributional shift.  
    

因此接下来我们介绍的方法不会解决 distribution shift 的问题 (因为其表示能力有限, distribution shift 不会是一个严重的问题), 我们主要关注如何在 offline RL 下估计 value function. 同样值得注意的是, 由于其没有解决 distribution shift 的问题, 在 deep RL 中应用同样的估计方法**不会取得好的效果**.

### 3.2 Linear models

考虑一个 feature map $\Phi$ 表示为 $|\mathcal{S}| \times K$ 的矩阵, 其中每一行表示一个 state 的特征. (不用担心 infinite state 的问题, 我们可以利用 samples 来解决这一点). 这一映射是 **人为设计** 的.

在这一类方法中, 我们实际上是在 feature space 中进行操作, 如果我们需要在 feature space 中进行 **(offline) model-based RL**, 实际上主要包含以下几个步骤 (我们目前假设我们能知道 dynamic 的信息, 之后再考虑 sample-based 的 setting):

1.  估计 reward  
    
2.  估计 transition  
    
3.  恢复 value function 并改进 policy  
    

在这类方法中, 我们考虑以下的 feature space 中的 model:

### Reward model:

我们使用一个线性变换 $\boldsymbol{w}_r$ 来恢复 reward: $\Phi \boldsymbol{w}_r \approx \overrightarrow{r}$.

基于上述的 reward model, 我们可以利用最小二乘估计表示出 $\boldsymbol{w}_{r} = (\Phi^T \Phi)^{-1} \Phi^T \overrightarrow{r}.\\$

### Transition model:

我们考虑 policy 与 dynamic 共同作用下的 state space transition $\boldsymbol{P}^\pi$, 因此其是一个 $|\mathcal{S}| \times |\mathcal{S}|$ 的矩阵, 同样我们需要考虑其在 feature space 中的表示 $\boldsymbol{P}_\Phi^\pi$, 其是一个 $K \times K$ 的矩阵. 这两个不同空间的 transition matrix 之间的关系可以表示为 $\Phi \boldsymbol{P}_\Phi^\pi = \boldsymbol{P}^\pi \Phi,\\$ 不难通过两侧左乘上一个 state, 得到转移后的 state 的 feature space 的分布, 来理解这个等式的意义. 此时使用最小二乘估计我们可以得到 $\boldsymbol{P}_\Phi^\pi = (\Phi^T \Phi)^{-1} \Phi^T \boldsymbol{P}^\pi \Phi.\\$

![](https://pic4.zhimg.com/v2-5465ffae10c03267f482c9362d8a5cc3_1440w.jpg)

### Estimate value function:

我们使用一个线性变换 $\boldsymbol{w}_V$ 来估计 value function: $V^\pi \approx V_\Phi^\pi = \Phi \boldsymbol{w}_V$. 我们可以写出向量形式的 Bellman equation: $V^\pi = \boldsymbol{r} + \gamma \boldsymbol{P}^\pi V^\pi,\\$ 可以得出当前 policy 下 state space 的 value function 为 $V^\pi = (\boldsymbol{I} - \gamma \boldsymbol{P}^\pi)^{-1} \boldsymbol{r}.\\$ 这里可以证明 $\boldsymbol{I} - \gamma \boldsymbol{P}^\pi$ 可逆, 再应用之前最小二乘法的思想, 我们可以得到 $\boldsymbol{w}_V = (\boldsymbol{I} - \gamma \boldsymbol{P}_\Phi^\pi)^{-1} \boldsymbol{w}_r.\\$ 实际上这里的 $\boldsymbol{P}_\Phi^\pi$ 和 $\boldsymbol{w}_r$ 都有闭合形式的解, 所以我们并不需要学习对应的 $\boldsymbol{w}_r$, $\boldsymbol{P}_\Phi^\pi$, 代入后进行简化可以得到 $\boldsymbol{w}_V = (\Phi^T \Phi - \gamma \Phi^T \boldsymbol{P}^\pi \Phi)^{-1} \Phi^T \boldsymbol{r}.\\$

经过上述过程, 我们就知道了如何通过 feature map 得到 $\boldsymbol{w}_V$, 进而得到 value function, 进而利用这个 value function 来改进 policy. 最终处理 feature map 之外, 我们只需要利用 samples 来估计 $\boldsymbol{r}$ 和 $\boldsymbol{P}^\pi$. 这一做法被称为 **least-squares temporal difference (LSTD)**

### 3.3 Empirical MDP

在上述的过程中, 我们需要知道 $\boldsymbol{P}^\pi$ 和 $\boldsymbol{r}$, 但是在 offline RL 中我们并没有这些信息. 我们可以通过 samples 来估计这些信息.

考虑 $\mathcal{D} = \{(\boldsymbol{s}_i, \boldsymbol{a}_i, r_i, \boldsymbol{s}_i')\}$, 由于我们也没有全体 states, 我们可以考虑用一个 $|\mathcal{D}| \times K$ 矩阵 $\Phi'$ 来表示这个 **empirical MDP** 的 feature map, 其中每一行表示一个 sample 的特征 $\phi(\boldsymbol{s}_i')$. 类似地, 我们可以用 $\boldsymbol{r}_i = r(\boldsymbol{s}_i)$ 来估计 $\boldsymbol{r}$.

这里的做法称为 empirical MDP, 换言之是由我们的 empirical samples 诱导的一个 MDP.

我们整理上述介绍的过程得到一个完整的算法, 也就是重复以下过程:

1.  $\pi'(\boldsymbol{s}) \gets Greedy(\Phi\boldsymbol{w}_V)$  
    
2.  估计 $V^{\pi'}$  
    

不难发现这一算法和 value iteration 是非常相似的, 区别主要在于我们使用了 feature map 而不是在 state space 中进行操作. 同时我们在 empirical MDP 中进行操作, 而不是在真实的 MDP 中进行操作.

### 3.4 Least-squares policy iteration (LSPI)

但是需要注意的是, 前面用 samples 估计的时候, 要求我们的 samples 来源于 $\pi$, 但是在 offline RL 的情况下, 我们没有相应的 samples.

这里的做法与我们在 off-policy Q-learning 和 off-policy actor-critic 算法中是一样的, 即不再 evaluate V-function 而是 evaluate Q-function. 这样我们就可以得到一个 off-policy 的算法: **Least-squares policy iteration (LSPI)**

**idea:** 将 LSTD 替换为 LSTDQ, 也就是我们用 Q-function 来替代 V-function, 换言之此时的 $\Phi$ 的会是一个 $|\mathcal{S}| |\mathcal{A}| \times K$ 的矩阵, 其中每一行表示一个 state-action pair 的特征.

![](https://pic4.zhimg.com/v2-0f3411b5b6d557f5583a6622d8f96d59_1440w.jpg)

我们可以同样推导出 $\boldsymbol{w}_Q = (\Phi^T \Phi - \gamma \Phi^T \boldsymbol{P}^\pi \Phi)^{-1} \Phi^T \boldsymbol{r}\\$ 这里 $\boldsymbol{r}$ 用 $\boldsymbol{r}_i = r(\boldsymbol{s}_i, \boldsymbol{a}_i)$ 来估计, 而 $\Phi$ 用 $\Phi_i' = \phi(\boldsymbol{s}_i', \pi(\boldsymbol{s}_i'))$ 来估计. 这里要注意的一点是, 我们的 $\Phi$ 不随着 policy 改变, 而 $\Phi'$ 是随着 policy 改变的.

最终我们得到了 **LSPI** 算法:

1.  计算 $\pi_k$ 对应的 $\boldsymbol{w}_Q$  
    
2.  更新 $\pi_{k + 1}(\boldsymbol{s}) = \arg\max_{\boldsymbol{a}} \phi(\boldsymbol{s}, \boldsymbol{a}) \boldsymbol{w}_Q$  
    
3.  令 $\Phi_i' = \phi(\boldsymbol{s}_i', \pi_{k + 1}(\boldsymbol{s}_i'))$  
    

值得再次强调的是, 这里的算法没有解决 distribution shift 的问题. 总的来说, 任何近似的 dynamic programming (也就是 fitted value/ Q iteration) 方法都会受到 distribution shift 的影响, 我们必须解决这一问题.

在 offline RL 的第二部分中, 我们会介绍现代的 offline RL 方法, 它们可以在很大程度上解决 distribution shift 的问题.