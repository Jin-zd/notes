# 1 Basics of Model-Based RL
在上一节中，我们介绍了在已知系统动态的情况下，如何通过 MCTS、LQR 等方式来进行规划。这实际上已经回答了为什么要学习一个模型的问题：如果知道动态 $f(\boldsymbol{s}_t, \boldsymbol{a}_t)$，我们就可以利用上一节中学到的方法。
在本节中依然介绍关于开环规划的方法，对于闭环（也就是学习一个策略）的做法，我们会在下一节中介绍。

在现实中的很多问题中，我们不知道动态，这里以确定性的情况为例。
基于朴素的直觉，可以给出一个基于模型的强化学习 0.5 版本：
1. 运行基本策略 $\pi_0(\boldsymbol{a}_t, \boldsymbol{s}_t)$，收集 $\mathcal{D} = \{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')_i\}$；
2. 学习动态模型 $f(\boldsymbol{s}, \boldsymbol{a})$ 来最小化 $\sum_{i} \|f(\boldsymbol{s}_i, \boldsymbol{a}_i) - \boldsymbol{s}'_i\|^2$；
3. 依据 $f(\boldsymbol{s}, \boldsymbol{a})$ 来进行规划。

这样的一个算法奏效吗？总的来说，这样的算法是奏效的。
- 这也是经典的机器人技术中系统识别的工作原理。
- 对基本策略的设计需要有一定的考量，来覆盖尽可能多的情况。
- 这样的做法在能够基于物理知识来人为设定所需参数时非常高效，只需要很少的参数。

然而，对于一些更加复杂的问题，例如在现实环境中，可能不能使用简单的一些物理知识来设计参数，我们需要更加复杂的模型。
但对深度神经网络这些表示能力强的模型来说，上述 v0.5 的算法存在分布偏移（distribution shift）的问题，在[[Lecture 1 Imitation Learning]]中我们见过类似的问题（$p_{data}(\boldsymbol{s}_t) \neq p_\theta(\boldsymbol{s}_t)$）。 在这里，问题体现在如果记训练动态模型的数据分布是 $p_{\pi_0}$，训练的策略得到的数据分布是 $p_{\pi_f}$。当使用 $\pi_f$ 时走到了一个 $p_{\pi_0}$ 中概率极小的状态，此时动态模型很可能会给出错误的结果，从而可能进入更加极端的状态，导致进一步的错误。
![](9-1.png)

一个非常直观的改进方式是，借鉴 DAgger 的思想，可以使用 $\pi_f$ 来收集数据。于是可以给出基于模型的强化学习 1.0 版本，这也是最简单的通常能够奏效的基于模型的强化学习算法：
1. 运行基本策略 $\pi_0(\boldsymbol{a}_t, \boldsymbol{s}_t)$，收集 $\mathcal{D} = \{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')_i\}$；
2. 学习动态模型 $f(\boldsymbol{s}, \boldsymbol{a})$ 来最小化 $\sum_{i} \|f(\boldsymbol{s}_i, \boldsymbol{a}_i) - \boldsymbol{s}'_i\|^2$；
3. 依据 $f(\boldsymbol{s}, \boldsymbol{a})$ 来进行规划；
4. 使用 $\pi_f$ 来收集数据，更新 $\mathcal{D}$，重复 2-4。

实际上，有一个很简单的改进：这里的直觉是，尽管在上述规划中会给出整个动作序列，但是在采取一些动作后，有了新的观测，有可能偏离了我们的预期（这是很有可能的，因为学习的模型不等于真实动态），就可以基于观测结果重新生成规划，这就是基于模型的强化学习 1.5 版本（模型预测控制，Model Predictive Control，MPC）的算法：
1. 运行基本模型 $\pi_0(\boldsymbol{a}_t, \boldsymbol{s}_t)$，收集 $\mathcal{D} = \{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')_i\}$；
2. 学习动态模型 $f(\boldsymbol{s}, \boldsymbol{a})$ 来最小化 $\sum_{i} \|f(\boldsymbol{s}_i, \boldsymbol{a}_i) - \boldsymbol{s}'_i\|^2$；
3. 依据 $f(\boldsymbol{s}, \boldsymbol{a})$ 来进行规划； 
4. 执行第一个规划的动作，观测到新的状态 $\boldsymbol{s}'$（MPC）；
5. 添加 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')$ 到 $\mathcal{D}$，重复 3-5，每 $N$ 次回到 2。 


总的来说，重规划总是能够减小模型错误。重规划的引入让每一个动作的重要性下降了，可以使用更短的时间跨度。

一个值得思考的是，这里的 MPC 是闭环规划吗？并不是，因为在规划的时候依然生成了整个动作序列，尽管每一个动作后我们都会重规划，但是规划时不知道只采用一个动作。

# 2 Uncertainty in Model-Based RL
## 2.1 Introduction
一个问题是，在基于模型的方法与无模型的方法之间存在着一个性能差距：
- 虽然在开始时，基于模型的强化学习很容易就得到了一个正的奖励，而无模型的强化学习通常起始时是很大的负值。
- 然而在经过训练后，基于模型的强化学习的奖励可能会陷入在这样较低的正值，而无模型的强化学习的性能可能显著超过基于模型的强化学习。
![](9-2.png)

一个相对明显的问题是，训练开始阶段数据较少时，模型能力相对较强，此时过拟合数据可能导致会出现非常多极端的点（试想用 $n$ 次多项式拟合 $n + 1$ 个点），规划器就会利用这些极端的点，导致无法到达那些真正高奖励的区域。
![](9-3.png)
此时的困难在于：
- 当数据较少时，不希望过拟合数据。 
- 在训练后期数据已经非常多时，又希望模型有更强的表示能力。  

在实际的很多情况下，基于模型的强化学习在一定阶段就会无法进行有效探索，从而导致最终的奖励陷入在一个较低的正值。

不妨考虑一个靠近悬崖的机器人，其离悬崖越近，则奖励越高，但由于模型有误差，悬崖的边界可能并不准确。于是我们可以考虑利用不确定性估计来解决。如果模型对预测的 $\boldsymbol{s}'$ 非常不确信，那么就可以避免这样的动作。
![](9-4.png)

## 2.2 Uncertainty-Aware Neural Net Models
### 2.2.1 Idea 1: output entropy
一个相当直观的想法是训练一个能够反映不确定性的模型。一个朴素的的想法是使用输出熵（模型输出结果的不确定性）。

然而这是错误的：既然我们想要的是更低熵的模型，那么损失函数就会逼迫模型做出极为自信的结果。换言之，模型对它的输出很自信，但我们对模型依然不确定的。

实质上，这样的方式只能够预测动态有多嘈杂，而完全无法反映模型有多少不确定性。这其实是两种截然不同的不确定性。
- 偶然（统计）不确定性 aleatoric (statistical) uncertainty：这里的不确定性来源于数据自身，不会由于数据量的多少而改变，无论有多少数据，数据自身的不确定性是不会改变的（例如扔骰子）。 
- 认知（模型）不确定性 epistemic (model) uncertainty：这里的不确定性来源于我们对模型的不确定性，当有更多的数据时，模型不确定性会减小。  
![](9-5.png)

试想构建一个模型来对骰子的结果进行建模，如果使用输出熵的方式，也就是降低输出的熵，那么模型会变得非常自信，例如认为骰子的结果永远是 $1$，这显然是错误的。
### 2.2.2 Idea 2: model uncertainty
第二个直觉是我们估计模型不确定性。这一种不确定性可以视作是对参数 $\theta$ 的不确定性。
换言之，我们关心的是 $p(\theta \mid \mathcal{D})$。接下来考虑如何估计这一分布。

## 2.3 Direct estimation
能否直接估计 $p(\theta \mid \mathcal{D})$ 呢？理论上没有问题，但是实际上不太可行，因为在预测时需要利用积分：
$$
\int p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \theta) p(\theta \mid \mathcal{D}) \text{d}\theta
$$
正如完全的[[Concepts#16 贝叶斯推断（Bayesian Inference）|贝叶斯推断（Bayesian Inference）]]一样，我们需要对所有的参数进行积分，然而这个积分基本上没有办法处理。

## 2.4 Bayesian neural network
如果基于一些假设，我们就可以得到一些近似的方法。这里的方法是贝叶斯神经网络（BNN），简单介绍一下其思想：对于通常的网络，权重都是单个值，而在贝叶斯神经网络中，权重是一个分布。我们关心的 $p(\theta \mid \mathcal{D})$ 其实就是所有参数的联合分布。
![](9-6.png)
通常得到这个联合分布是非常困难的，可以有以下的一些近似方法：
- 独立性假设： $p(\theta \mid \mathcal{D}) \approx \prod_i p(\theta_i \mid \mathcal{D})$，这里的 $\theta_i$ 是网络中的每一个参数，这里的近似非常粗糙，但是在实际中是可行的。
- 分布选择：通常我们会选择一个简单的分布，例如高斯分布，也就是 $p(\theta_i \mid \mathcal{D}) = \mathcal{N}(\mu_i, \sigma_i^2)$，这相当于是每一个参数由原先的单个值变为了一个 $\mu_i, \sigma_i$。

## 2.5 Bootstrap ensembles:
接下来考虑自举聚合（Bootstrap ensembles）方法：同时训练多个网络，希望它们在训练数据上有相同的表现，但是在测试数据上有不同的表现。具体来说在测试数据上，采用如下的加权方式：
$$
p(\theta \mid \mathcal{D}) \approx \frac{1}{N} \sum_{i}\delta(\theta_i)
$$
这里的 $\delta$ 是[[Concepts#17 狄拉克 $ delta$ 函数 (Dirac Delta Function)|狄拉克 δ 函数 (Dirac Delta Function)]]， $\theta_i$ 是第 $i$ 个网络的参数。在预测时就利用 
$$
\int p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \theta) p(\theta \mid \mathcal{D}) \text{d}\theta \approx \frac{1}{N} \sum_{i} p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \theta_i)
$$
注意我们要平均的是分布，而不是预测。
![](9-7.png)
这里如何得到多个网络呢？这里使用的方法是自举。生成多个“独立的” 数据集，从而得到一系列“独立的”模型：
- 简单的做法是将数据集分成多份，每一份训练一个网络，于是就得到了多个网络，但是这样比较浪费。
- 一个较好的做法是利用有放回采样来生成 $\mathcal{D}_i$，实际证明这样的方式足够给出一个合理的不确定性估计。
- 在深度学习中，由于训练网络成本很高，因此使用的模型数量通常较小，故上述做法比较粗略，但通常依然是奏效的。与此同时，在深度学习中随机梯度下降与随机初始化足够让模型表现出足够的差异，因此有放回抽样是不必要的。

## 2.6 Planning with Uncertainty
在上一节中介绍了利用如 guess and check、cross-entropy method (CEM) 等方式进行规划。可以将这里的不确定性感知模型应用在规划上：考虑 $J(\boldsymbol{a}_1, \boldsymbol{a}_2, \ldots, \boldsymbol{a}_H)$，其中 $\boldsymbol{s}_{t + 1} = f(\boldsymbol{s}_t, \boldsymbol{a}_t)$，可以考虑 不确定性感知规划：
$$
J(\boldsymbol{a}_1, \boldsymbol{a}_2, \ldots, \boldsymbol{a}_H) = \frac{1}{N}\sum_{i = 1}^{N} \sum_{t = 1}^{H} r(\boldsymbol{s}_{t, i}, \boldsymbol{a}_{t, i}), \quad \boldsymbol{s}_{t + 1, i} = f(\boldsymbol{s}_{t, i}, \boldsymbol{a}_{t, i})
$$
通常来说，对于候选的动作序列 $\boldsymbol{a}_1, \ldots, \boldsymbol{a}_H$，考虑以下几步：
1. 采样 $\theta \sim p(\theta \mid \mathcal{D})$；
2. 在每一时间步, 采样 $\boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_{t} \mid \boldsymbol{s}_{t}, \boldsymbol{a}_{t}, \theta)$；
3. 计算 $R = \sum_t r(\boldsymbol{s}_t, \boldsymbol{a}_t)$；
4. 重复 1- 3 多次来计算平均奖励。

其他的一些方式是矩匹配（例如前面提到的用高斯分布的贝叶斯神经网络），或者基于贝叶斯神经网络更加复杂的后验推断。

# 3 Model-Based RL with complex observation
这一小节我们以图像为例介绍如何处理基于模型的强化学习中的复杂观测。不难得出这些复杂观测有以下特点：
- high dimensionality：图片的维度可能远远高于状态的维度.  
- Redundancy：一张图片中很多信息可能是冗余的（考虑游戏图片中完全重复的背景）
- Partial observability：涉及到物体运动时，单张图片无法描述物体的速度和加速度等复杂信息。
![](9-8.png)
可以画出如下的概率图：
![](9-9.png)

一个直观的方式是分别学习 $p(\boldsymbol{o}_t \mid \boldsymbol{s}_{t})$ 和 $p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$，前者是高维但是与动态无关，后者是低维但是与动态相关。这样主要工作就转化到了学习前者 $p(\boldsymbol{o}_t \mid \boldsymbol{s}_{t})$ 上，不过这样的设计并非一定必要。在这一类方法中，需要训练以下三个模型：
- 观测模型（observation model）： $p(\boldsymbol{o}_t \mid \boldsymbol{s}_{t})$  
- 动态模型（dynamics model）： $p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$  
- 奖励模型（reward model）： $p(r_t \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$  

用概率图模型来表示，就是：
![](9-10.png)

对于完全可观测的情况，使用最大似然估计（MLE）来训练 
$$
\max_\phi \frac{1}{N} \sum_{i = 1}^{N} \sum_{t = 1}^{T} \log p(\boldsymbol{s}_{t + 1, i} \mid \boldsymbol{s}_{t,i}, \boldsymbol{a}_{t,i})
$$
而对于部分可观测的情况，考虑 
$$
\max_\phi \sum_{t = 1}^{T} \mathbb{E}_{\boldsymbol{s}_{t}, \boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_t, \boldsymbol{s}_{t + 1})} \left[\log p(\boldsymbol{s}_{t + 1, i} \mid \boldsymbol{s}_{t,i}, \boldsymbol{a}_{t,i}) + \log p(\boldsymbol{o}_{t, i} \mid \boldsymbol{s}_{t, i})\right]
$$
然而这里的问题在于我们对于状态空间并没有一个模型，接下来我们考虑如何学习一个状态空间模型。

## 3.1 State space (latent space) models

在复杂观测的情况下，我们并没有一个明确的状态空间，但是我们可以考虑学习一个状态空间模型，其中的状态空间 $\mathcal{S}$ 是一个低维的潜在空间。这里仅讨论这一类方法背后的思想，对于变分推断等内容我们会在之后的专门一节[[Lecture 16 Variational Inference and Generative Model]]中讨论。

为方便理解，将其中一种可能的设计方式与 [[Concepts#18 变分自编码器 (Variational Autoencoder, VAE)|变分自编码器 (Variational Autoencoder, VAE)]] 中的思想类比：将观测空间 $\mathcal{O}$ 中的观测投影到一个低维的潜在空间 $\mathcal{S}$ 中，这类似于将数据空间 $\mathcal{X}$ 中的数据投影到一个低维的潜在空间 $\mathcal{Z}$。

于是就有很多相似的核心概念：例如需要从潜在空间到观测空间的映射 $p_\phi(\boldsymbol{o} \mid \boldsymbol{s})$，这同样会被称为解码。类似地，由于 $p(\boldsymbol{s} \mid \boldsymbol{o})$ 这类后验分布难以直接计算，会训练一个带参数的编码器 $q_\psi(\boldsymbol{s} \mid \boldsymbol{o})$ 来近似这个后验分布。

而在基于模型的强化学习中的潜在空间模型，还要考虑一些其他的问题：
潜在空间不能简单假设各维度独立，而且要有隐含的动态，如 
$$
p(\boldsymbol{s}) = p(\boldsymbol{s}_1) \prod_t p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)
$$
单个观测不足以决定状态，因此可能需要建模 $q_\psi(\boldsymbol{s}_t \mid \boldsymbol{o}_{1:t})$（以至于还有利用 $\boldsymbol{a}_{1:t}$）而不是 $q_\psi(\boldsymbol{s}_t \mid \boldsymbol{o}_t)$，实际上我们还可以学习其他的后验分布： 
- 完全平滑后验（full smoothing posterior）：$q_\psi(\boldsymbol{s}_t, \boldsymbol{s}_{t + 1} \mid \boldsymbol{o}_{1:T}, \boldsymbol{a}_{1:T})$  
- 单步编码器（single-step encoder）：$q_\psi(\boldsymbol{s}_t \mid \boldsymbol{o}_t)$  

## 3.2 Example with single-step encoder
考虑最简单的 $q_\psi(\boldsymbol{s}_t \mid \boldsymbol{o}_t)$，目前也只考虑简单的确定性情况（随机情况需要变分推断，在之后讨论）。此时编码器可以表示为 
$$
q_\psi(\boldsymbol{s}_t \mid \boldsymbol{o}_t) = \delta(\boldsymbol{s}_t = g_\psi(\boldsymbol{o}_t)) \Rightarrow \boldsymbol{s}_t = g_\psi(\boldsymbol{o}_t)
$$
于是优化目标就是 
$$
\max_{\phi, \psi} \frac{1}{N} \sum_{i = 1}^{N} \sum_{t = 1}^{T} \log p_\phi(g_\psi(\boldsymbol{o}_{t + 1, i}) \mid g_\psi(\boldsymbol{o}_{t, i}), \boldsymbol{a}_{t, i}) + \log p(\boldsymbol{o}_{t, i} \mid g_\psi(\boldsymbol{o}_{t, i}))
$$
这里所有都是可微的，所以可以用反向传播来训练。如果还要考虑奖励模型，那么目标是 
$$
\max_{\phi, \psi} \frac{1}{N} \sum_{i = 1}^{N} \sum_{t = 1}^{T} \log p_\phi(g_\psi(\boldsymbol{o}_{t + 1, i}) \mid g_\psi(\boldsymbol{o}_{t, i}), \boldsymbol{a}_{t, i}) + \log p_\phi(\boldsymbol{o}_{t, i} \mid g_\psi(\boldsymbol{o}_{t, i})) + \log p_\phi(r_{t, i} \mid g_\psi(\boldsymbol{o}_{t, i}), \boldsymbol{a}_{t, i})
$$
三项分别是潜在空间动态、图像重建、奖励模型。

可以得到基于模型的隐状态强化学习：
1. 运行基本策略 $\pi_0(\boldsymbol{a}_t, \boldsymbol{o}_t)$，收集 $\mathcal{D} = \{(\boldsymbol{o}, \boldsymbol{a}, \boldsymbol{o}')_i\}$；
2. 学习动态模型 $p_\phi(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$，奖励模型 $p_\phi(r_t \mid \boldsymbol{s}_t)$，观测模型 $p_\phi，编码器 $g_\psi(\boldsymbol{o}_t)$；
3. 依据上述模型来进行规划；
4. 执行第一个规划的动作，观测到新的状态 $\boldsymbol{o}'$（MPC）；
5. 添加 $(\boldsymbol{o}, \boldsymbol{a}, \boldsymbol{o}')$ 到 $\mathcal{D}$，重复 3-5，每 $N$ 次回到 2。

实际也可以直接学习一个在观测空间的动态模型，只不过这里由于部分可观测，可能需要使用序列模型。

# 4 Summary
在本节中, 我们
- 讨论了如何学习一个动态模型；
- 介绍了如何利用不确定性感知模型来避免模型的过度自信，以及如何利用不确定性感知模型来进行规划；
- 讨论了如何处理复杂观测，简要介绍了如何学习一个状态空间模型。