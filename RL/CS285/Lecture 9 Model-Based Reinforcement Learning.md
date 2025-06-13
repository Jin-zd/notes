# 1 Basics of Model-Based RL

在上一节中, 我们介绍了在已知系统 dynamic 的情况下, 如何通过 MCTS, LQR 等方式来进行 planning. 这实际上已经回答了我们为什么要学习一个 model 的问题: 如果我们知道 dynamic $f(\boldsymbol{s}_t, \boldsymbol{a}_t)$, 我们就可以利用上一节中学到的方法. 在本节中我们依然关于 open loop planning 的方法, 对于 close loop (也就是学习一个 policy) 的做法, 我们会在下一节中介绍.

而在现实中的很多问题中, 我们不知道 dynamic, 这里我们以 deterministic 的情况为例. 基于 naive intuition, 我们可以给出一个 model-based RL version 0.5:
1.  运行 base policy $\pi_0(\boldsymbol{a}_t, \boldsymbol{s}_t)$, 收集 $\mathcal{D} = \{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')_i\}$,  
2.  学习 dynamic model $f(\boldsymbol{s}, \boldsymbol{a})$ 来最小化 $\sum_{i} \|f(\boldsymbol{s}_i, \boldsymbol{a}_i) - \boldsymbol{s}'_i\|^2$  
3.  依据 $f(\boldsymbol{s}, \boldsymbol{a})$ 来进行 plan  

这样的一个算法 work 吗? 总的来说, 这样的算法是 work 的.
-   这也是经典的 robotics 中 system identification 的工作原理.  
-   我们对 base policy 的设计需要有一定的考量, 来覆盖尽可能多的情况.  
-   这样的做法在我们能够**基于物理知识**来人为设定所需参数时非常高效, 我们只需要很少的参数.  

然而, 对于一些更加复杂的问题, 例如在现实环境中, 我们可能不能使用简单的一些物理知识来设计参数, 我们需要更加复杂的模型. 但对深度神经网络这些表示能力强的模型来说, 上述 v0.5 的算法存在问题 **distribution shift (mismatch)**, 在 imitation learning 中我们见过类似的问题 ($p_{data}(\boldsymbol{s}_t) \neq p_\theta(\boldsymbol{s}_t)$). 在这里, 问题体现在 如果记训练 dynamic model 的数据分布是 $p_{\pi_0}$, 训练的策略得到的数据分布是 $p_{\pi_f}$. 当使用 $\pi_f$ 时走到了一个 $p_{\pi_0}$ 中概率极小的状态, 此时 dynamic model 很可能会给出错误的结果, 从而可能进入更加极端的状态, 导致进一步的错误.

![](https://pic3.zhimg.com/v2-8f7664c6a6959cbe3bef741f9175bb74_1440w.jpg)

distribution shift 问题

一个非常直观的改进方式是, 我们借鉴 **DAgger** 的思想, 我们可以使用 $\pi_f$ 来收集数据. 于是我们可以给出 model-based RL version 1.0, 这也是最简单的通常能够 work 的 model-based RL 算法:
1.  运行 base policy $\pi_0(\boldsymbol{a}_t, \boldsymbol{s}_t)$, 收集 $\mathcal{D} = \{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')_i\}$,  
2.  学习 dynamic model $f(\boldsymbol{s}, \boldsymbol{a})$ 来最小化 $\sum_{i} \|f(\boldsymbol{s}_i, \boldsymbol{a}_i) - \boldsymbol{s}'_i\|^2$  
3.  依据 $f(\boldsymbol{s}, \boldsymbol{a})$ 来进行 plan  
4.  使用 $\pi_f$ 来收集数据, 更新 $\mathcal{D}$, 重复 2-4  

实际上, 我们有一个很简单的改进: 这里的 intuition 是, 尽管我们在上述 planning 中会给出整个动作序列, 但是在我们采取一些动作后, 我们有了新的观测, 有可能我们偏离了我们的预期 (这是很有可能的, 因为我们学习的 model 不等于真实 dynamic), 就可以基于观测结果重新生成 plan, 这就是 model-based RL version 1.5 (**Model Predictive Control [MPC](https://zhida.zhihu.com/search?content_id=254214503&content_type=Article&match_order=1&q=MPC&zhida_source=entity)**) 的算法:
1.  运行 base policy $\pi_0(\boldsymbol{a}_t, \boldsymbol{s}_t)$, 收集 $\mathcal{D} = \{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')_i\}$,  
2.  学习 dynamic model $f(\boldsymbol{s}, \boldsymbol{a})$ 来最小化 $\sum_{i} \|f(\boldsymbol{s}_i, \boldsymbol{a}_i) - \boldsymbol{s}'_i\|^2$  
3.  依据 $f(\boldsymbol{s}, \boldsymbol{a})$ 来进行 plan  
4.  执行第一个规划的 action, 观测到新的状态 $\boldsymbol{s}'$ (MPC)  
5.  添加 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')$ 到 $\mathcal{D}$, 重复 3-5; 每 $N$ 次回到 2.  

**Remark:**
-   总的来说, replanning 总是能够减小 model errors. replan 的引入让每一个 action 的重要性下降了.  
-   可以使用更短的 horizon.  

一个值得思考的是, 这里的 MPC 是 close loop planning 吗? 并不是, 因为我们在 planning 的时候依然生成了整个动作序列, 尽管每一个动作后我们都会 replan, 但是 planning 时**不知道我们只采用一个 action**.

# 2 Uncertainty in Model-Based RL

## 2.1 Introduction

一个问题是, 在 model-based 与 model-free 之间存在着一个 performance gap:
-   虽然在开始时, model-based RL 很容易就得到了一个正的 reward, 而 model-free RL 通常起始时是很大的负值. 
-   然而在经过训练后, model-based RL 的 reward 可能会陷入在这样较低的正值, 而 model-free RL 的性能可能显著超过 model-based RL.  
![](https://pica.zhimg.com/v2-e5ea19accf532e7f6cd9b874ea7604b4_1440w.jpg)

绿色曲线与蓝色曲线分别对应于 model-based 与 model-free 方法

一个相对明显的问题是, 训练开始阶段数据较少时, 模型能力相对较强, 此时过拟合数据导致可能会出现非常多极端的点 (试想用 $n$ 次多项式拟合 $n + 1$ 个点), 我们的 planner 就会利用这些极端的点, 导致我们无法到达那些真正高 reward 的区域.

![](https://pica.zhimg.com/v2-4212c46f3ab1e1c694b79a18b36b3314_1440w.jpg)

此时的困难在于:
-   当数据较少时, 我们不希望我们 overfit data  
-   在训练后期数据已经非常多时, 我们又希望我们的 model 有更强的表示能力  

在实际的很多情况下, model-based RL 在一定阶段就会无法进行有效 exploration, 从而导致最终的 reward 陷入在一个较低的正值.

不妨考虑一个靠近悬崖的机器人, 其离悬崖越近, 则 reward 越高, 但由于我们的 model 有误差, 悬崖的边界可能并不准确. 于是我们可以考虑利用 **uncertainty estimation** 来解决. 如果 model 对预测的 $\boldsymbol{s}'$ 非常不确信, 那么我们就可以避免这样的 action.

![](https://pic2.zhimg.com/v2-f10fcd8c3f219e7d828f06944d165839_1440w.jpg)

## 2.2 Uncertainty-Aware Neural Net Models
### 2.2.1 Idea 1: output entropy:

一个相当直观的想法是我们训练一个能够反映 uncertainty 的 model. 一个 naive 的想法是使用 **output entropy** (模型输出结果的 uncertainty).

然而这是错误的: 既然我们想要的是更低 entropy 的模型, 那么我们的损失函数就会逼迫模型做出极为自信的结果. 换言之, 模型对它的输出很自信, 但我们对模型依然 uncertain.

实质上, 这样的方式只能够预测 dynamics 有多 noisy, 而完全无法反映我们的 model 有多少 uncertainty. 这其实是两种截然不同的 uncertainty.

-   **aleatoric (statistical) uncertainty**: 这里的 uncertainty 来源于数据自身, 不会由于数据量的多少而改变, 无论我们有多少数据, 数据自身的 uncertainty 是不会改变的. (例如扔骰子)  
    
-   **epistemic (model) uncertainty**: 这里的 uncertainty 来源于我们对 model 的不确定性, 当我们有更多的数据时, 我们的 model uncertainty 会减小.  
    

![](https://pic4.zhimg.com/v2-b5d3494b55b666c4024049b08c4e1b55_1440w.jpg)

试想我们构建一个 model 来对骰子的结果进行建模, 如果我们使用 output entropy 的方式, 也就是降低输出的 entropy, 那么我们的 model 会变得非常自信, 例如认为骰子的结果永远是 $1$, 这显然是错误的.

### 2.2.2 Idea 2: model uncertainty:

第二个 intuition 是我们估计 **model uncertainty**. 这一种 uncertainty 可以视作是对参数 $\theta$ 的 uncertainty.

换言之, 我们关心的是 $p(\theta \mid \mathcal{D})$. 接下来我们考虑如何估计这一分布.

## 2.3 Direct estimation:

我们能否直接估计 $p(\theta \mid \mathcal{D})$ 呢? 理论上没有问题, 但是实际上不太可行. 因为我们在预测时需要利用积分 
$$
\int p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \theta) p(\theta \mid \mathcal{D}) \text{d}\theta
$$
正如完全的 Bayesian inference 一样, 我们需要对所有的参数进行积分, 然而这个积分基本上没有办法处理.

## 2.4 Bayesian neural network
如果我们基于一些假设, 我们就可以得到一些近似的方法. 这里的方法是 **Bayesian neural network**, 我们简单介绍一下其思想: 对于通常的网络, 我们的权重都是单个 value, 而在 Bayesian neural network 中, 我们的权重是一个分布. 我们关心的 $p(\theta \mid \mathcal{D})$ 其实就是所有参数的联合分布.

![](https://pic4.zhimg.com/v2-3ee0519d52cabcd10ecfc987296c19b9_1440w.jpg)

左侧是常规神经网络, 右侧是 BNN

通常得到这个联合分布是非常困难的, 我们可以有以下的一些近似方法:

-   独立性假设: $p(\theta \mid \mathcal{D}) \approx \prod_i p(\theta_i \mid \mathcal{D})$, 这里的 $\theta_i$ 是网络中的每一个参数. 这里的近似非常粗糙, 但是在实际中是可行的.  
-   分布选择: 通常我们会选择一个简单的分布, 例如 Gaussian distribution, 也就是 $p(\theta_i \mid \mathcal{D}) = \mathcal{N}(\mu_i, \sigma_i^2)$. 这相当于是每一个参数由原先的单个 value 变为了一个 $\mu_i, \sigma_i$.  
    

具体来说, 参见: Blundell et al., Weight Uncertainty in Neural Networks 与 Gal et al., Concrete Dropout

## 2.5 Bootstrap ensembles:

接下来我们考虑 **Bootstrap ensembles** 方法: 我们同时训练多个网络, 希望它们在训练数据上有相同的表现, 但是在测试数据上有不同的表现. 具体来说在测试数据上, 我们采用如下的加权方式: 
$$
p(\theta \mid \mathcal{D}) \approx \frac{1}{N} \sum_{i}\delta(\theta_i)
$$
这里的 $\delta$ 是 Dirac delta function, $\theta_i$ 是第 $i$ 个网络的参数. 在预测时就利用 
$$
\int p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \theta) p(\theta \mid \mathcal{D}) \text{d}\theta \approx \frac{1}{N} \sum_{i} p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \theta_i)
$$
注意我们要平均的是分布, 而不是 prediction.

![](https://picx.zhimg.com/v2-3a86676b615ee119827c338eaa5b1bf3_1440w.jpg)

我们这里如何得到多个网络呢? 这里使用的方法是 bootstrap, 我们生成多个 "independent" datasets, 从而得到一系列 "independent" models.
-   简单的做法是将 dataset 分成多份, 每一份训练一个网络, 于是我们就得到了多个网络. 但是这样比较浪费.  
-   一个较好的做法是利用有放回采样来生成 $\mathcal{D}_i$. 实际证明这样的方式足够给出一个合理的 uncertainty estimation.  
-   在 DL 中, 由于训练网络成本很高, 因此使用的模型数量通常较小, 故上述做法比较粗略, 但通常依然是 work 的. 与此同时, 在 DL 中 SGD 与随机初始化足够让模型表现出足够的差异, 因此 draw with replacement 是不必要的.  

## 2.6 Planning with Uncertainty

在上一节中, 我们介绍了利用如 guess and check, cross-entropy method (CEM) 等方式进行 planning. 我们这里可以将这里的 uncertainty-aware model 应用在 planning 上: 考虑 $J(\boldsymbol{a}_1, \boldsymbol{a}_2, \ldots, \boldsymbol{a}_H)$, 其中 $\boldsymbol{s}_{t + 1} = f(\boldsymbol{s}_t, \boldsymbol{a}_t)$. 我们可以考虑 uncertainty-aware planning: 
$$
J(\boldsymbol{a}_1, \boldsymbol{a}_2, \ldots, \boldsymbol{a}_H) = \frac{1}{N}\sum_{i = 1}^{N} \sum_{t = 1}^{H} r(\boldsymbol{s}_{t, i}, \boldsymbol{a}_{t, i}), \quad \boldsymbol{s}_{t + 1, i} = f(\boldsymbol{s}_{t, i}, \boldsymbol{a}_{t, i})
$$
通常来说, 对于候选的 action 序列 $\boldsymbol{a}_1, \ldots, \boldsymbol{a}_H$, 我们考虑以下几步:
1.  采样 $\theta \sim p(\theta \mid \mathcal{D})$  
2.  在每一时间步, 采样 $\boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_{t} \mid \boldsymbol{s}_{t}, \boldsymbol{a}_{t}, \theta)$  
3.  计算 $R = \sum_t r(\boldsymbol{s}_t, \boldsymbol{a}_t)$  
4.  重复 1- 3 多次来计算平均的 reward.  

其他的一些方式是 moment matching (例如前面提到的用高斯分布的 BNN), 或者基于 BNNs 更加复杂的后验推断.

# 3 Model-Based RL with complex observation
这一小节我们以 图像 为例介绍如何处理 model-based RL 中的 complex observation. 不难得出这些 complex observations 有以下特点:
-   **high dimensionality**: 图片的维度可能远远高于状态的维度.  
-   **Redundancy**: 一张图片中很多信息可能是冗余的 (考虑游戏图片中完全重复的背景).  
-   **Partial observability**: 涉及到物体运动时, 单张图片无法描述物体的速度和加速度等复杂信息.

![](https://picx.zhimg.com/v2-58bef68f058b66649826f3b9396fdc3d_1440w.jpg)

一些复杂的 observation 示例
我们可以画出如下的概率图:

![](https://pic2.zhimg.com/v2-06ae5a8e2a6a4b2ba75e6b07c2d9e965_1440w.jpg)

一个直观的方式是分别学习 $p(\boldsymbol{o}_t \mid \boldsymbol{s}_{t})$ 和 $p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$, 前者是高维但是与动态无关, 后者是低维但是与动态相关. 这样我们的主要工作就转化到了学习前者 $p(\boldsymbol{o}_t \mid \boldsymbol{s}_{t})$ 上. 不过这样的设计并非一定必要. 在这一类方法中, 我们需要训练以下三个模型:
-   **observation model**: $p(\boldsymbol{o}_t \mid \boldsymbol{s}_{t})$  
-   **dynamics model**: $p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$  
-   **reward model**: $p(r_t \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$  

用概率图模型来表示, 就是:

![](https://pica.zhimg.com/v2-6e40c9d37eefb74e5e2fb5aac2c0e5fc_1440w.jpg)

对于 full observed 情况, 我们会使用 MLE 来训练 
$$
\max_\phi \frac{1}{N} \sum_{i = 1}^{N} \sum_{t = 1}^{T} \log p(\boldsymbol{s}_{t + 1, i} \mid \boldsymbol{s}_{t,i}, \boldsymbol{a}_{t,i})
$$
而对于 partially observed 的情况, 我们考虑 
$$
\max_\phi \sum_{t = 1}^{T} \mathbb{E}_{\boldsymbol{s}_{t}, \boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_t, \boldsymbol{s}_{t + 1})} \left[\log p(\boldsymbol{s}_{t + 1, i} \mid \boldsymbol{s}_{t,i}, \boldsymbol{a}_{t,i}) + \log p(\boldsymbol{o}_{t, i} \mid \boldsymbol{s}_{t, i})\right]
$$
然而这里的问题在于我们对于 state space 并没有一个模型. 接下来我们考虑如何学习一个 state space model.

## 3.1 State space (latent space) models

在 complex observation 的情况下, 我们并没有一个明确的 state space, 但是我们可以考虑学习一个 state space model, 其中的 state space $\mathcal{S}$ 是一个低维的 latent space. 这里我们仅讨论这一类方法背后的思想, 对于 variational inference 等内容我们会在之后的专门一节 **variational inference and generative model** 中讨论.

为方便理解, 我们将**其中一种可能的**设计方式与 [VAE](https://zhida.zhihu.com/search?content_id=254214503&content_type=Article&match_order=1&q=VAE&zhida_source=entity) 中的思想类比: 我们将 observation space $\mathcal{O}$ 中的 observation 投影到一个低维的 latent space $\mathcal{S}$ 中, 这类似于我们将数据空间 $\mathcal{X}$ 中的数据投影到一个低维的 latent space $\mathcal{Z}$.

于是就有很多相似的核心概念: 例如我们需要从 latent space 到 observation space 的映射 $p_\phi(\boldsymbol{o} \mid \boldsymbol{s})$, 这同样会被称为 **decoder**. 类似地, 由于 $p(\boldsymbol{s} \mid \boldsymbol{o})$ 这类后验分布难以直接计算, 我们会训练一个带参数的 **encoder** $q_\psi(\boldsymbol{s} \mid \boldsymbol{o})$ 来近似这个后验分布.

而在 model-based RL 中的 latent space model, 我们还要考虑一些其他的问题:

我们的 latent space 不能简单假设各维度独立, 而且要有隐含的 dynamic 如 
$$
p(\boldsymbol{s}) = p(\boldsymbol{s}_1) \prod_t p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)
$$
-   单个 observation 不足以决定 state, 因此我们可能需要建模 $q_\psi(\boldsymbol{s}_t \mid \boldsymbol{o}_{1:t})$ (以至于还有利用 $\boldsymbol{a}_{1:t}$) 而不是 $q_\psi(\boldsymbol{s}_t \mid \boldsymbol{o}_t)$. 实际上我们还可以学习其他的后验分布:  
-   full smoothing posterior: $q_\psi(\boldsymbol{s}_t, \boldsymbol{s}_{t + 1} \mid \boldsymbol{o}_{1:T}, \boldsymbol{a}_{1:T})$  
-   single-step encoder: $q_\psi(\boldsymbol{s}_t \mid \boldsymbol{o}_t)$  

## 3.2 Example with single-step encoder

我们考虑最简单的 $q_\psi(\boldsymbol{s}_t \mid \boldsymbol{o}_t)$, 目前也只考虑简单的 deterministic case (stochastic case 需要 variational inference, 在之后讨论). 此时我们的 encoder 可以表示为 
$$
q_\psi(\boldsymbol{s}_t \mid \boldsymbol{o}_t) = \delta(\boldsymbol{s}_t = g_\psi(\boldsymbol{o}_t)) \Rightarrow \boldsymbol{s}_t = g_\psi(\boldsymbol{o}_t)
$$
于是我们的优化目标就是 
$$
\max_{\phi, \psi} \frac{1}{N} \sum_{i = 1}^{N} \sum_{t = 1}^{T} \log p_\phi(g_\psi(\boldsymbol{o}_{t + 1, i}) \mid g_\psi(\boldsymbol{o}_{t, i}), \boldsymbol{a}_{t, i}) + \log p(\boldsymbol{o}_{t, i} \mid g_\psi(\boldsymbol{o}_{t, i}))
$$
这里所有都是可微的, 所以我们可以用反向传播来训练. 如果还要考虑 reward model, 那么我们的目标是 
$$
\max_{\phi, \psi} \frac{1}{N} \sum_{i = 1}^{N} \sum_{t = 1}^{T} \log p_\phi(g_\psi(\boldsymbol{o}_{t + 1, i}) \mid g_\psi(\boldsymbol{o}_{t, i}), \boldsymbol{a}_{t, i}) + \log p_\phi(\boldsymbol{o}_{t, i} \mid g_\psi(\boldsymbol{o}_{t, i})) + \log p_\phi(r_{t, i} \mid g_\psi(\boldsymbol{o}_{t, i}), \boldsymbol{a}_{t, i})
$$
三项分别是 **latent space dynamics**, **image reconstruction**, **reward model**.

我们可以得到 [model-based reinforcement learning](https://zhida.zhihu.com/search?content_id=254214503&content_type=Article&match_order=1&q=+model-based+reinforcement+learning&zhida_source=entity) with latent state:
1.  运行 base policy $\pi_0(\boldsymbol{a}_t, \boldsymbol{o}_t)$, 收集 $\mathcal{D} = \{(\boldsymbol{o}, \boldsymbol{a}, \boldsymbol{o}')_i\}$,  
    
2.  学习 dynamic model $p_\phi(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$, reward model $p_\phi(r_t \mid \boldsymbol{s}_t)$, observation model $p_\phi(\boldsymbol{o}_t \mid \boldsymbol{s}_t)$, encoder $g_\psi(\boldsymbol{o}_t)$  
3.  依据上述模型 来进行 plan  
4.  执行第一个规划的 action, 观测到新的状态 $\boldsymbol{o}'$ (MPC)  
5.  添加 $(\boldsymbol{o}, \boldsymbol{a}, \boldsymbol{o}')$ 到 $\mathcal{D}$, 重复 3-5; 每 $N$ 次回到 2.  

**Side Note:** 我们实际也可以直接学习一个在 observation space 的 dynamic, 只不过这里由于 partial observability, 我们可能需要使用 sequence model.

# 4 Summary

在本节中, 我们

-   讨论了如何学习一个 dynamic model  
    
-   介绍了如何利用 uncertainty-aware model 来避免模型的 over-confidence, 以及如何利用 uncertainty-aware model 来进行 planning
-   最后我们讨论了如何处理 complex observation, 简要介绍了如何学习一个 state space model.