## 1 Introduction to Offline RL
### 1.1 Why we need offline RL?
在当前的深度学习中，我们通常已经可以得到相当不错的泛化性，例如在一些分类任务上，一个在大量数据上训练的模型可以在未见过的数据上取得很好的效果，这样的效果对于光照，视角等变化都相当鲁棒。然而，在强化学习中，通常难以实现很好的泛化性，即使是稍微改变光照，也可能导致整个策略的崩溃。
![](13-1.png)

如果思考背后的原因，当前深度学习之所以能够取得巨大的成功，核心的要素是大数据 + 大模型。然而，之前介绍的强化学习算法与监督学习的情境有着很大的差异：强化学习算法通常依赖不断与环境交互以收集数据，如果在真实世界收集，则这一过程的效率相当低下，如果在虚拟环境中生成，则通常又缺乏多样性。

回顾过去介绍的强化学习算法，无论是 on-policy 还是 off-policy 算法（虽然前者样本效率更低），这些在线的算法对于每一个新任务都必须从头开始收集数据，在完成一个任务后，所用的数据就丢弃了。通常情况下，不可能从零开始收集一个 ImageNet 规模的数据集。没有大规模的数据集，也自然难以训练一个大规模的模型。
![](13-2.png)

在正式介绍离线强化学习的具体设定之前，我们先考虑其背后的基本想法：离线强化学习的目的就是获得一种数据驱动的强化学习方法。在离线强化学习中：
- 假设有一个用某种方式收集的固定数据集 $\mathcal{D}$。（数据集未必对应于专家数据）
- 我们的目标是从这个数据集中学习一个策略，在这个学习过程中无法与环境交互。
- 最后将学习到的策略部署到真实环境中。

![](13-3.png)

### 1.2 Formalization of offline RL
我们给出离线强化学习的范式：
- 数据集：$\mathcal{D} = \{(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_{i}', r_i)\}$；
- 状态：$\boldsymbol{s} \sim d^{\pi_\beta}(\boldsymbol{s})$，这里 $d^{\pi_\beta}(\boldsymbol{s})$ 是由策略 $\pi_\beta$ 产生的状态分布；
- 动作： $\boldsymbol{a} \sim \pi_\beta(\boldsymbol{a}\mid \boldsymbol{s})$；
- 下个状态：$\boldsymbol{s}' \sim p(\boldsymbol{s}'\mid \boldsymbol{s}, \boldsymbol{a})$ ；
- 奖励：$r \sim r(\boldsymbol{s}, \boldsymbol{a})$。

这里的 $\beta$ 是未知的收集数据的策略。

离线强化学习的基础是离线评估（Off-policy evaluation，OPE），也就是仅仅利用固定数据集 $\mathcal{D}$ 评估当前策略的好坏。评估当前策略的好坏是改进策略的基础。

而离线强化学习（有时候也称作批量强化学习，完全离线策略强化学习）的目标则是：给定 $\mathcal{D}$，学习尽可能好的 $\pi_\theta$。（尽可能好并不一定是马尔可夫决策过程中的最优策略，而只是在 $\mathcal{D}$ 能够学习到的最优策略）

### 1.3 Expectation and intuition in offline RL
在上述的范式中，我们提出离线强化学习能够学到尽可能好的策略，但是数据集包含了质量参差不齐的行为，这是可能的吗？实际上这基于以下几点期望：
- 从混乱到有序：离线强化学习能够从具有好行为与差行为的 $\mathcal{D}$ 中找到其中好的部分；![](13-4.png)
- 泛化：一个地方的好行为可能在另一个地方也是好的（即使另一个地方没有这一好的行为的数据）；
- “缝合” ：不同行为中的优质部分能够被拼接重组。举例来说，要是拥有打开抽屉的数据，还有从打开的抽屉里拿取物品的数据，那么借助离线强化学习，就能学会如何完成打开抽屉并拿取物品这一系列动作。
![](13-5.png)

一个容易误解的地方是离线强化学习与模仿学习的关系：
离线强化学习不是模仿学习。离线强化学习理论上通常可以比数据集中最好的策略更好，而模仿学习通常只能期望得到数据集中的平均策略。可以证明在某些结构性假设下，离线强化学习可以比模仿学习更好，即使数据是最优的。

如果有一个离线强化学习的算法，那么在每次开始新任务时就可以重新利用已有的数据，同时也可以利用其他人在其他任务上收集的数据，此时训练成本会大大降低，在每次训练的时候就不需要收集一个 ImageNet 大小的数据集来实现好的泛化性。
![](13-6.png)

### 1.4 Challenges in offline RL
然而，在离线强化学习中存在一些挑战：

例如，在 QT-Opt: Scalable Deep Reinforcement Learning of Vision-Based Robotic Manipulation Skills 中，作者通过离线训练 + 在线微调的方式，也就是先进行离线强化学习，然后在真实环境中进行微调。

文中使用的离线数据有 580k，而在线数据仅有 28k（使用最新策略收集）。结果是仅使用离线时仅仅获得了 $87\%$ 的成功率，而经过在线微调后能够达到 $96\%$ 成功率。这在一定程度上说明了离线强化学习+ 在线微调的有效性。
然而实际上在线额外引入的数据不到离线的 $5\%$，却产生了明显的表现提升，似乎扩大离线数据集规模还没有最后很少的一点在线数据来的好。直观来说，在离线强化学习中一定额外存在了一些在线强化学习中不存在的问题，限制了离线强化学习的表现。
![](13-7.png)

再例如，从 Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction 中发现，对于不同规模的数据，实际的表现几乎没有区别，奖励在 $-750$ 附近徘徊。但是如果我们观察 Q 值的曲线，会发现估计的 Q 值会非常大。
![](13-8.png)
事实上, 这并不是一个巧合！
这里的一个根本性问题是：反事实查询。

一个例子是，对于人类驾驶员来说，即使其水平再差，也有一些永远不会做的事情，例如在马路中间进行一个急转弯，因此数据集中不可能覆盖所有可能的行为。但是在训练时，当处在一定的状态中，会进行一个比较：那些分布外（OOD）的行为是否比分布内的行为更好？
- 在在线强化学习中，不用担心这个问题：如果发现依据当前策略，有一个分布外行为的奖励很高。真金不怕火炼，可以尝试一遍，如果这个分布外行为很好，自然能够通过检验，但如果产生了糟糕的结果，也就能学到这个分布外的行为是不好的（当然，这也是为什么不希望将这些算法用于现实世界中一些危险的场景）。
- 但是在离线强化学习中，没有这样真实尝试的机会。需要有办法处理这些分布外的行为，例如想办法避免它们，但是同时不能天真的要求行为一定出现在 $\mathcal{D}$ 中，因为需要泛化到不在 $\mathcal{D}$ 中但是依然分布内的行为。

![](13-9.png)

### 1.5 Math in distribution shift
上述问题也可以理解是 $\pi$ 与 $\pi_{\beta}$ 之间的分布偏移。如果有 $\pi$ 对应的数据，那么相当于我们就可以在线检验。前面采用了一种直观地理解方式，事实上可以通过数学的方式来理解这一问题。

回顾在监督学习中，通常解决的是一个经验风险最小化（Empirical risk minimization，ERM）问题：
$$
\theta \gets \arg\min_\theta \mathbb{E}_{\boldsymbol{x} \sim p(\boldsymbol{x}), y \sim p(y\mid \boldsymbol{x})} \left[(f_\theta(\boldsymbol{x}) - y)^2\right]
$$
这里的 $p(x), p(y\mid x)$ 是训练数据的分布。

基于对监督学习的认识，如果没有过拟合，那么期望在统计意义上
$$
\mathbb{E}_{\boldsymbol{x} \sim p(\boldsymbol{x}), y \sim p(y\mid \boldsymbol{x})} \left[(f_\theta(\boldsymbol{x}) - y)^2\right]
$$
会很低。但是对于另一个分布 $\bar{p}(\boldsymbol{x})$
$$
\mathbb{E}_{\boldsymbol{x} \sim \bar{p}(\boldsymbol{x}), y \sim p(y\mid \boldsymbol{x})} \left[(f_\theta(\boldsymbol{x}) - y)^2\right]
$$
可能会很高。然而值得注意的是，上述只是统计上的结果，即使 $\boldsymbol{x}^\ast \sim p(\boldsymbol{x})$，误差也可能很高。

在监督学习中，可以构造出对抗样本，刻意选取一个 $\boldsymbol{x}^\ast \gets \arg\max_{\boldsymbol{x}} f_\theta(\boldsymbol{x})$，此时可能会得到极大的误差 $(f_\theta(\boldsymbol{x}^\ast) - y)$。类似地，也可以构造出对抗性分布 $\bar{p}(\boldsymbol{x})$，使得这个分布上的误差都很大。

### 1.6 Math in distribution shift in offline RL
但是这和离线强化学习有什么关系呢？在离线强化学习中，更新策略的方式实际上就在构造这样的对抗性分布。

回顾 off-policy 的 Q 学习中，我们会进行如下的贝尔曼备份： 
$$
Q(\boldsymbol{s}, \boldsymbol{a}) \gets r(\boldsymbol{s}, \boldsymbol{a}) + \mathbb{E}_{\boldsymbol{a'} \sim \pi_{new}(\boldsymbol{a}' \mid \boldsymbol{s}')}[Q(\boldsymbol{s}', \boldsymbol{a'})]
$$
其中 $\boldsymbol{s}, \boldsymbol{a}$ 是从 $\mathcal{D}$ 中采样的，而 $\boldsymbol{a}'$ 来源于最新的策略 $\arg\max \pi_{new}$ （实际上 Q 函数更新的标准写法应该是 $\arg\max_{\boldsymbol{a}'} Q(\boldsymbol{s}', \boldsymbol{a}')$，但是为了更好兼容像演员-评论家等算法，我们使用上述写法），为了记号简便，记 $y(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \mathbb{E}_{\boldsymbol{a'} \sim \pi_{new}}[Q(\boldsymbol{s}', \boldsymbol{a'})]$。

整个 Q 学习可以理解为是在循环进行“训练/更新”和“评估”的过程，训练是通过目标
$$
\mathcal{L} = \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}) \sim \pi_\beta(\boldsymbol{s}, \boldsymbol{a})} \left[(Q(\boldsymbol{s}, \boldsymbol{a}) - y(\boldsymbol{s}, \boldsymbol{a}))^2\right]
$$
来更新 $Q$ 函数，而评估则是利用训练得到的 $Q$ 函数来计算 $y(\boldsymbol{s}, \boldsymbol{a})$。不难理解，其中训练/更新发生在 $\pi_\beta$ 上，而评估发生在 $\pi_{new}$ 上。

如果 $\pi_{new} = \pi_\beta$，那么评估与训练的分布是一致的，应当期望 Q 函数能得到很好的结果。
然而事实上在评估中获取 $\mathbb{E}_{\boldsymbol{a'} \sim \pi_{new}}[Q(\boldsymbol{s}', \boldsymbol{a'})]$ 时，我们查询了分布外的数据。具体来说， $Q$ 函数在一系列不同的动作上对应不同的值，并不是所有的动作都在分布内。在那些分布外的动作上，$Q$ 函数可能会被高估。由于 Q 学习中会使用 $\arg\max$ 策略，可以理解为 $\pi_{new}$ 会在那些 Q 值极高的地方进行选择。这相当于是人为构造了一个对抗性分布： 
$$
\arg\max_{\pi} \mathbb{E}_{\boldsymbol{a}' \sim \pi(\boldsymbol{a}'\mid \boldsymbol{s}')} \left[Q(\boldsymbol{s}', \boldsymbol{a}')\right]
$$
并使用在这个 分布上的动作进行评估，将其结果作为 $y(\boldsymbol{s}, \boldsymbol{a})$，这个结果自然是偏差极大的。

而训练目标依赖于这个评估的结果，也就是在更新 $Q$ 函数时，会使得 $Q(\boldsymbol{s}, \boldsymbol{a})$ 尽可能地接近 $y(\boldsymbol{s}, \boldsymbol{a})$。 但是 $y(\boldsymbol{s}, \boldsymbol{a})$ 来自于构造的对抗样本，也就是在更新 $Q$ 函数时，会使得 $Q(\boldsymbol{s}, \boldsymbol{a})$ 尽可能地接近一个高估的值。长此以往，$Q$ 函数就会向着高估的方向发展。

注意：上述分析主要是针对 Q 学习的，由于其使用了 $\arg\max$ 策略，因此分析起来更加直观，但是类似的问题也会出现在演员-评论家等算法中。

### 1.7 Comparison with overestimation in online Q-learning
我们似乎在 Q 学习中也见到了高估问题，它们的本质其实有很大的差异：
- 在 Q 学习中，问题来自于 $\max$ 操作在噪声的影响下的高估，也就是 $\mathbb{E}[\max(X, Y)] \geq \max(\mathbb{E}[X], \mathbb{E}[Y])$，而想要的是 $\max(\mathbb{E}[X], \mathbb{E}[Y])$，但是实际估计出的是 $\mathbb{E}[\max(X, Y)]$。 
- 在离线强化学习中，核心的问题是分布偏移引起的，导致 $Q$ 函数不断拟合对抗性分布上的结果，从而使得 $Q$ 函数不断被高估。而在线强化学习中，由于会定期更新数据，一旦某个 Q 值被高，那么下次收集数据时，会有一定概率被真实的奖励信号纠正。

上述分析实际告诉了我们：很多在在线情况下可以被纠正的错误在离线情况下是无法被纠正的。在常规强化学习挑战中的采样误差与函数近似误差在离线强化学习中会变得更加严重。
![](13-10.png)

## 2 Batch RL via Importance Sampling
在接下来的这两节中，我们会讨论在深度强化学习之前的一些离线强化学习算法。现如今通常不会将这些算法作为一个默认选择，但是其背后的思想是当前离线强化学习算法的基础（在很多早期的文献中，这称为批量强化学习）。

在这一节中，先考虑带策略梯度的离线强化学习。
### 2.1 Basic ideas and problems
考虑强化学习目标：
$$
J(\theta) = \max_\pi \sum_{t = 1}^{T} \mathbb{E}_{\boldsymbol{s}_t \sim d^{\pi_\theta}(\boldsymbol{s}), \boldsymbol{a}_t \sim \pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})} \left[\gamma^t r(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]
$$
这里将 $\pi$ 用 $\theta$ 参数化，梯度可以表示为 
$$
\begin{aligned} \nabla_\theta J(\theta) &= \mathbb{E}_{\tau \sim \pi_\theta(\tau)} \left[\sum_{t = 1}^{T} \nabla_\theta \gamma^t\log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t) \hat{Q}(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]\\ &\approx \frac{1}{N} \sum_{i = 1}^{N} \sum_{t = 1}^{T} \nabla_\theta \gamma^t \log \pi_\theta(\boldsymbol{a}_{i, t} \mid \boldsymbol{s}_{i, t}) \hat{Q}(\boldsymbol{s}_{i, t}, \boldsymbol{a}_{i, t}) \end{aligned}
$$
其中 $\hat{Q}(\boldsymbol{s}_{i, t}, \boldsymbol{a}_{i, t}) \approx \sum_{t' = t}^{T} \gamma^{t' - t} r_{t', i}$ 是未来奖励总和，但是很显然这里的梯度估计需要有 $\pi_\theta$ 的样本，而由于只有 $\pi_\beta$ 的数据，因此使用重要性采样，也就是 
$$
\nabla_\theta J(\theta) = \frac{1}{N} \sum_{i = 1}^{N} \frac{\pi_\theta(\tau_i)}{\pi_\beta(\tau_i)} \sum_{t = 1}^{T} \nabla_\theta \gamma^t \log \pi_\theta(\boldsymbol{a}_{i, t} \mid \boldsymbol{s}_{i, t}) \hat{Q}(\boldsymbol{s}_{i, t}, \boldsymbol{a}_{i, t})
$$
而这个重要性比率可以表示为 
$$
\frac{\pi_\theta(\tau_i)}{\pi_\beta(\tau_i)}= \prod_{t = 1}^{T} \frac{\pi_\theta(\boldsymbol{a}_{i, t} \mid \boldsymbol{s}_{i, t})}{\pi_\beta(\boldsymbol{a}_{i, t} \mid \boldsymbol{s}_{i, t})}
$$
这是一个在 $T$ 上呈指数形式的式子，因此虽然这一估计是无偏的，但是方差会非常大。

### 2.2 Unable to ignore the past
在[[Lecture 7 Advanced Policy Gradients]]一节的讨论，如果 $\pi_\theta$ 与 $\pi_{\theta'}$ 足够接近，则 
$$
J(\theta') - J(\theta) \approx \mathbb{E}_{\tau \sim p_{\theta}(\tau)} \left[\sum_{t} \frac{\pi_{\theta'}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}{\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \gamma^t A^{\pi_\theta}(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]
$$
这个式子在这个设置的理解方式与之前并不一样，此时 $\theta$ 对应于有数据的分布， $\theta'$ 是目前需要改进的策略的参数。由于 $J(\theta)$ 是固定的，通过梯度上升来最大化 $J(\theta')$，对 $\nabla_{\theta'} J(\theta')$ 的估计可以通过 
$$
\nabla_{\theta'} \mathbb{E}_{\tau \sim p_{\theta}(\tau)} \left[\sum_{t} \frac{\pi_{\theta'}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}{\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \gamma^t A^{\pi_\theta}(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]
$$
来进行，其中的 $A^{\pi_\theta}(\boldsymbol{s}_t, \boldsymbol{a}_t)$ 可以替换为数据集中的未来奖励总和。

在这个意义下,，如果 $\pi_\theta$ 与 $\pi_\beta$ 足够接近，也就是如果策略的改变量在一定范围内，则可将 $t$ 时间步之前的比率全部忽略掉，但是很显然这在离线强化学习中通常是不成立的，因为目标策略比 $\pi_\beta$ 要好很多。

### 2.3 Applying Causality
因此将比率拆分为以下部分：
$$
\frac{1}{N} \sum_{i = 1}^{N} \sum_{t = 1}^{T} \left(\prod_{t' = 1}^{t - 1} \frac{\pi_\theta(\boldsymbol{a}_{i, t'} \mid \boldsymbol{s}_{i, t'})}{\pi_\beta(\boldsymbol{a}_{i, t'} \mid \boldsymbol{s}_{i, t'})}\right) \nabla_\theta \gamma^t \log \pi_\theta(\boldsymbol{a}_{i, t} \mid \boldsymbol{s}_{i, t}) \left(\prod_{t' = t}^{T} \frac{\pi_\theta(\boldsymbol{a}_{i, t'} \mid \boldsymbol{s}_{i, t'})}{\pi_\beta(\boldsymbol{a}_{i, t'} \mid \boldsymbol{s}_{i, t'})}\right)\hat{Q}(\boldsymbol{s}_{i, t}, \boldsymbol{a}_{i, t})
$$
前一个累乘考虑了到达 $s_{i, t}$ 的概率差异，后一个累乘考虑了未来奖励总和的差异。

考虑对回报的估计部分，事实上可以进一步化简 
$$
\begin{aligned} \left(\prod_{t' = t}^{T} \frac{\pi_\theta(\boldsymbol{a}_{i, t'} \mid \boldsymbol{s}_{i, t'})}{\pi_\beta(\boldsymbol{a}_{i, t'} \mid \boldsymbol{s}_{i, t'})}\right)\hat{Q}(\boldsymbol{s}_{i, t}, \boldsymbol{a}_{i, t}) &= \sum_{t' = t}^{T} \left(\prod_{t'' = t}^{T} \frac{\pi_\theta(\boldsymbol{a}_{i, t''} \mid \boldsymbol{s}_{i, t''})}{\pi_\beta(\boldsymbol{a}_{i, t''} \mid \boldsymbol{s}_{i, t''})}\right) \gamma^{t' - t} r_{i, t'}\\ &= \sum_{t' = t}^{T} \left(\prod_{t'' = t}^{t'} \frac{\pi_\theta(\boldsymbol{a}_{i, t''} \mid \boldsymbol{s}_{i, t''})}{\pi_\beta(\boldsymbol{a}_{i, t''} \mid \boldsymbol{s}_{i, t''})}\right) \gamma^{t' - t} r_{i, t'}\\ \end{aligned}
$$
其中后一个等号利用了因果性，然而这个式子仍然是在 $T$ 上呈指数形式，但稍微改进了一点。事实上，为了避免指数级爆炸的重要性权重的问题，必须使用价值函数估计。

接下来介绍因果性外其他改进过高方差的方法。

### 2.4 Doubly robust estimator
双重稳健估计量（Doubly robust estimator）是统计学上的一个概念，通俗来讲是利用两个估计量来使得估计更加稳定。在强化学习中，这准确来说属于一个离线评估算法而不是强化学习算法，通过这个算法可以得到价值的估计值，从而将其放入重要性采样梯度估计量中。

首先依然重申想要改进策略的前提是能够评估策略，这等价于能够估计 $V^{\pi_\theta}(\boldsymbol{s})$。在原先的重要性采样估计量中，有
$$
\begin{aligned} V^{\pi_\theta}(\boldsymbol{s}_1) &\approx \sum_{t = 1}^{T} \left(\prod_{t' = 1}^{t} \frac{\pi_\theta(\boldsymbol{a}_{t'} \mid \boldsymbol{s}_{t'})}{\pi_\beta(\boldsymbol{a}_{t'} \mid \boldsymbol{s}_{t'})}\right) \gamma^t r_{t}\\ &= \sum_{t = 1}^{T} \left(\prod_{t' = 1}^{t} \rho_{t'}\right) \gamma^t r_{t}\\ &= \rho_1 (r_1 + \gamma (\rho_2 r_2 + \gamma (\rho_3 r_3 + \cdots)))\\ &= \bar{V}^T \end{aligned}
$$
此时存在着递归关系 $\bar{V}^{T + 1 - t} = \rho_t (r_{t} + \gamma \bar{V}^{T - t})$。如果令 $\bar{V}^0 = 0$，那么就可以递归地计算出 $\bar{V}^T$，也就能够评估策略，但是这样的估计量有很大的方差。

可以考虑使用仅价值函数估计量 $\hat{V}(\boldsymbol{s})$ 或 $\hat{Q}(\boldsymbol{s}, \boldsymbol{a})$ 来作为估计量，此时可能类似于价值函数基线，由于价值函数本身的误差，估计可能是有偏差的。

于是可以考虑将上述两个估计量结合起来，得到一个双重稳健估计量：
$$
\bar{V}_{DR}^{T + 1 - t} = \hat{V}(\boldsymbol{s}_t) + \rho_t (r_{t} + \gamma \bar{V}_{DR}^{T - t} - \hat{Q}(\boldsymbol{s}_t, \boldsymbol{a}_t))
$$
这里 $\hat{V}(\boldsymbol{s}_t)$ 可以通过 $\hat{Q}(\boldsymbol{s}_t, \boldsymbol{a}_t)$ 得到。一个直观的比喻是，价值函数估计量是一个基本估计，可以防止方差爆炸，而重要性采样估计量是一个校正方式，可以帮助纠正偏差。

参见：Jiang, N. and Li, L. (2015). Doubly robust off-policy value evaluation for reinforcement learning

### 2.5 Marginalized importance sampling
和前面的双重稳健估计量类似，这一方法在文献中属于离线评估算法而不是强化学习算法。

在之前讨论的重要性采样中，都是将比率表示为动作的概率的比值，而事实上也可以将其表示为状态概率的比值，也就是使用
$$
w(\boldsymbol{s},\boldsymbol{a}) = \frac{d^{\pi_\theta}(\boldsymbol{s},\boldsymbol{a})}{d^{\pi_\beta}(\boldsymbol{s},\boldsymbol{a})}
$$
很显然不知道这个比值，但如果能够估计这一点，那么就能通过
$$
J(\theta) \approx \frac{1}{N} \sum_{i} w(\boldsymbol{s}_i, \boldsymbol{a}_i) r_i
$$
来计算目标。

通常会使用神经网络来表示 $w(\boldsymbol{s}, \boldsymbol{a})$，并列出一系列一致性条件，用处理贝尔曼方程的方式处理，只不过求解的是 $w(\boldsymbol{s}, \boldsymbol{a})$。

一个例子是（Zhang et al., GenDICE: Generalized Offline Estimation of Stationary Values），考虑以下加权边际
$$
p_\pi(\boldsymbol{s}, \boldsymbol{a}) = (1 - \gamma) \sum_{t = 0}^T \gamma^t p(\boldsymbol{s}_t = \boldsymbol{s}, \boldsymbol{a}_t = \boldsymbol{a})
$$
于是可以列出以下等式（类似于求解马尔可夫链的平稳分布）：
$$
d^{\pi_\beta}(\boldsymbol{s}', \boldsymbol{a}') w(\boldsymbol{s}', \boldsymbol{a}') = (1 - \gamma) p_0(\boldsymbol{s}') \pi_\theta(\boldsymbol{a}' \mid \boldsymbol{s}') + \gamma \sum_{\boldsymbol{s}, \boldsymbol{a}} p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a}) \pi_\theta(\boldsymbol{a}' \mid \boldsymbol{s}') d^{\pi_\beta}(\boldsymbol{s}, \boldsymbol{a}) w(\boldsymbol{s}, \boldsymbol{a})
$$
等式的左边就是在 $\pi_\theta$ 下见到 $(\boldsymbol{s}',\boldsymbol{a}')$ 的概率，等式右侧第一项等于开始于 $\boldsymbol{s}'$ 且采取 $\boldsymbol{a}'$ 的概率，第二项表示从另一状态到达 $\boldsymbol{s}'$ 且采取 $\boldsymbol{a}'$ 的概率，具体的求解方法可以参见原论文。

### 2.6 Additional readings: importance sampling
关于重要性采样策略梯度和回报估计的经典工作：
- Precup, D. (2000). Eligibility traces for off-policy policy evaluation.
- Peshkin, L. and Shelton, C. R. (2002). Learning from scarce experience.  
- Doubly robust estimators and other improved importance-sampling estimators:  
- Jiang, N. and Li, L. (2015). Doubly robust off-policy value evaluation for reinforcement learning.
- Thomas, P. and Brunskill, E. (2016). Data-efficient off-policy policy evaluation for reinforcement learning.  

分析与理论：
- Thomas, P. S., Theocharous, G., and Ghavamzadeh, M. (2015). High-confidence off-policy evaluation.  
- Marginalized importance sampling:  
- Hallak, A. and Mannor, S. (2017). Consistent on-line off-policy evaluation.
- Liu, Y., Swaminathan, A., Agarwal, A., and Brunskill, E. (2019). Off-policy policy gradient with state distribution correction

## 3 Batch RL via Linear Fitted Value Functions
我们介绍的第二种经典的离线强化学习算法是基于线性拟合值函数的。尽管现如今使用深度神经网络来表示价值函数，但是线性例子中可以得到闭式解，这或许可以给我们研究更加有效的离线强化学习算法提供一些直觉。
### 3.1 Offline value function estimation
实际上，过去基于价值函数的离线强化学习算法与当前的深度学习的关注点并不相同：
- 在过去的离线强化学习中，人们通常考虑将动态规划与 Q 学习中的现有想法应用在离线设置中，并且通过线性函数逼近来获取闭合形式的解。 
- 在现在的离线强化学习中，我们通常会使用更加有表示力的模型，例如深度神经网络，并且主要的挑战是分布偏移。

因此接下来我们介绍的方法不会解决分布偏移的问题（因为其表示能力有限，分布偏移不会是一个严重的问题），我们主要关注如何在离线强化学习下估计价值函数。同样值得注意的是，由于其没有解决分布偏移的问题, 在深度强化学习中应用同样的估计方法不会取得好的效果。

### 3.2 Linear models
考虑一个特征图 $\Phi$ 表示为 $|\mathcal{S}| \times K$ 的矩阵，其中每一行表示一个状态的特征（不用担心无限状态的问题，可以利用采样来解决这一点），这一映射是人为设计的。

在这一类方法中，我们实际上是在特征空间中进行操作，如果需要在特征空间中进行（离线）基于模型的强化学习，实际上主要包含以下几个步骤（目前假设能知道动态的信息，之后再考虑基于采样的设置）：
1. 估计奖励；
2. 估计转移；
3. 恢复价值函数并改进策略； 

在这类方法中，考虑以下的特征空间中的模型：

**Reward model：**
我使用一个线性变换 $\boldsymbol{w}_r$ 来恢复奖励：
$$
\Phi \boldsymbol{w}_r \approx \overrightarrow{r}
$$
基于上述的奖励模型，可以利用最小二乘估计表示出 
$$
\boldsymbol{w}_{r} = (\Phi^T \Phi)^{-1} \Phi^T \overrightarrow{r}
$$

**Transition model：**
考虑策略与动态共同作用下的状态空间转移 $\boldsymbol{P}^\pi$，因此其是一个 $|\mathcal{S}| \times |\mathcal{S}|$ 的矩阵，同样需要考虑其在特征空间中的表示 $\boldsymbol{P}_\Phi^\pi$，其是一个 $K \times K$ 的矩阵。这两个不同空间的转移矩阵之间的关系可以表示为
$$
\Phi \boldsymbol{P}_\Phi^\pi = \boldsymbol{P}^\pi \Phi
$$
不难通过两侧左乘上一个状态，得到转移后的状态的特征空间的分布，来理解这个等式的意义。此时使用最小二乘估计可以得到 
$$
\boldsymbol{P}_\Phi^\pi = (\Phi^T \Phi)^{-1} \Phi^T \boldsymbol{P}^\pi \Phi
$$

![](13-11.png)

**Estimate value function：**
使用一个线性变换 $\boldsymbol{w}_V$ 来估计价值函数：
$$
V^\pi \approx V_\Phi^\pi = \Phi \boldsymbol{w}_V
$$
可以写出向量形式的贝尔曼方程
$$
V^\pi = \boldsymbol{r} + \gamma \boldsymbol{P}^\pi V^\pi
$$
可以得出当前策略下状态空间的价值函数为 
$$
V^\pi = (\boldsymbol{I} - \gamma \boldsymbol{P}^\pi)^{-1} \boldsymbol{r}
$$
这里可以证明 $\boldsymbol{I} - \gamma \boldsymbol{P}^\pi$ 可逆，再应用之前最小二乘法的思想，可以得到 
$$
\boldsymbol{w}_V = (\boldsymbol{I} - \gamma \boldsymbol{P}_\Phi^\pi)^{-1} \boldsymbol{w}_r
$$
实际上这里的 $\boldsymbol{P}_\Phi^\pi$ 和 $\boldsymbol{w}_r$ 都有闭合形式的解，所以我并不需要学习对应的 $\boldsymbol{w}_r$,，$\boldsymbol{P}_\Phi^\pi$，代入后进行简化可以得到 
$$
\boldsymbol{w}_V = (\Phi^T \Phi - \gamma \Phi^T \boldsymbol{P}^\pi \Phi)^{-1} \Phi^T \boldsymbol{r}
$$

经过上述过程，我们就知道了如何通过特征图得到 $\boldsymbol{w}_V$，进而得到价值函数，进而利用这个价值函数来改进处理。最终处理特征图之外，只需要利用采样来估计 $\boldsymbol{r}$ 和 $\boldsymbol{P}^\pi$。这一做法被称为最小二乘时间差分（Least-squares temporal difference，LSTD）。

### 3.3 Empirical MDP
在上述的过程中，我们需要知道 $\boldsymbol{P}^\pi$ 和 $\boldsymbol{r}$，但是在离线强化学习中并没有这些信息。我们可以通过采样来估计这些信息。

考虑 $\mathcal{D} = \{(\boldsymbol{s}_i, \boldsymbol{a}_i, r_i, \boldsymbol{s}_i')\}$，由于没有全体状态，可以考虑用一个 $|\mathcal{D}| \times K$ 矩阵 $\Phi'$ 来表示这个经验性马尔可夫决策过程（Empirical MDP）的特征图，其中每一行表示一个采样的特征 $\phi(\boldsymbol{s}_i')$。类似地，可以用 $\boldsymbol{r}_i = r(\boldsymbol{s}_i)$ 来估计 $\boldsymbol{r}$。

这里的做法称为经验性马尔可夫决策过程，换言之是由经验性采样诱导的一个马尔可夫决策过程。
整理上述介绍的过程得到一个完整的算法，也就是重复以下过程：
1. $\pi'(\boldsymbol{s}) \gets Greedy(\Phi\boldsymbol{w}_V)$；
2. 估计 $V^{\pi'}$。

不难发现这一算法和价值迭代是非常相似的，区别主要在于使用了特征图而不是在状态空间中进行操作。同时在经验性马尔可夫决策过程中进行操作，而不是在真实的马尔可夫决策过程中进行操作。

### 3.4 Least-squares policy iteration (LSPI)
但是需要注意的是，前面用采样估计的时候，要求采样来源于 $\pi$，但是在离线强化学习的情况下，我们没有相应的样本。

这里的做法与我们在 off-policy Q 学习和 off-policy 演员-评论家算法中是一样的，即不再配合价值函数而是评估 $Q$ 函数，这样就可以得到一个 off-policy 的算法：最小二乘策略迭代（Least-squares policy iteration，LSPI）。

想法：将 LSTD 替换为 LSTDQ，也就是用 Q 函数来替代价值函数，换言之此时的 $\Phi$ 的会是一个 $|\mathcal{S}| |\mathcal{A}| \times K$ 的矩阵，其中每一行表示一个状态-动作对的特征。
![](13-12.png)

同样可以推导出 
$$
\boldsymbol{w}_Q = (\Phi^T \Phi - \gamma \Phi^T \boldsymbol{P}^\pi \Phi)^{-1} \Phi^T \boldsymbol{r}
$$
这里 $\boldsymbol{r}$ 用 $\boldsymbol{r}_i = r(\boldsymbol{s}_i, \boldsymbol{a}_i)$ 来估计，而 $\Phi$ 用 $\Phi_i' = \phi(\boldsymbol{s}_i', \pi(\boldsymbol{s}_i'))$ 来估计。这里要注意的一点是， $\Phi$ 不随着策略改变，而 $\Phi'$ 是随着策略改变的。

最终得到了最小二乘策略迭代算法：
1. 计算 $\pi_k$ 对应的 $\boldsymbol{w}_Q$；
2. 更新 $\pi_{k + 1}(\boldsymbol{s}) = \arg\max_{\boldsymbol{a}} \phi(\boldsymbol{s}, \boldsymbol{a}) \boldsymbol{w}_Q$  
3. 令 $\Phi_i' = \phi(\boldsymbol{s}_i', \pi_{k + 1}(\boldsymbol{s}_i'))$。

值得再次强调的是，这里的算法没有解决分布偏移的问题。
总的来说，任何近似的动态规划（也就是拟合价值迭代或拟合Q迭代）方法都会受到分布偏移的影响，我们必须解决这一问题。
在离线强化学习的第二部分中，我们会介绍现代的离线强化学习方法，它们可以在很大程度上解决分布偏移的问题。