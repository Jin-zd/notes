## Introduction

为什么一些看起来对人类更简单的游戏, 例如 Montezuma's Revenge, 对于 RL 算法来说却是一个很大的挑战?

这是因为 [exploration problem](https://zhida.zhihu.com/search?content_id=254636918&content_type=Article&match_order=1&q=exploration+problem&zhida_source=entity) 的困难性. 这一问题的困难性主要体现在以下两个方面:

-   任务的时间延续性  
    
-   对规则的未知程度  
    

![](https://picx.zhimg.com/v2-3ee2e0044d2cfc2f070666b44b079fd3_1440w.jpg)

右图为 Montezuma&#39;s Revenge

### How extended the task is

一方面, 当任务的时间延续性增加时, 我们采用常规的 exploration 策略很可能无法探索到有效的 reward 信号. 例如在 Montezuma's Revenge 中, 我们需要完成一系列非常长序列的任务, 而这些序列并不包含中间 reward.

**Side Note:** 我们可以试图添加一些中间 reward 来帮助 agent 更好地学习. 在上述例子中, 我们可以添加一些中间的 reward, 例如 killed by the skeleton. 但是这样的做法也会带来一些问题, 例如 killed by the skeleton 如果设置正的 reward 很可能导致 agent 最终学会最快地被 skeleton 杀死, 然而设置负的 reward 则可能导致 agent 不会尝试通过 skeleton, 导致无法到达更远的地方.

### How unknown the rules are

另一方面, 不同于人类对游戏规则有先验的理解 (例如上图中的骷髅图标, 对于人类来说我们知道要避免的, 而钥匙的位置是我们想要到达的), RL 算法并不知道这一些, 只能够通过 trail and error 来学习.

不难发现, 无论是任务的时间延续性增加, 还是规则的未知程度增加, 都会给我们的 exploration problem 带来更大的挑战, 也就需要更加有效的 exploration strategy.

## Exploration and exploitation

### Definition and Examples

两种对 exploration problem 的潜在定义方式:

-   一个 agent 如何才能发现一些高 reward 的策略, 如果这些策略需要一个很长时间的复杂行为, 且每一个行为都没有 reward?  
    
-   一个 agent 如何决定是继续尝试一些新的行为, 还是继续做已知最优的行为?  
    

事实上二者都是 **exploration** 问题:

-   **Exploitation**: 选择已知可获取最高 reward 的行为  
    
-   **Exploration**: 选择之前没有尝试过的行为, 期望能够发现更高 reward 的行为  
    

**Example 1**. _考虑选择饭店的例子: exploitation 是选择已知的好的饭店, exploration 是尝试一些新的饭店._

### Optimal Exploration

不难理解 exploration 是一个非常困难的问题, 我们最高的目标显然是推导出一个 optimal 的 exploration strategy. (当然, 我们必须先定义什么是 optimal, 通常情况下有利用 regret 与 Bayes-optimal strategy 两种方式, 我们在之后的讨论中会详细介绍)

自然地, 在不同难度的问题上, 我们可以得到不同程度的理论结果, 从获取理论结果的难度上分类, 从简单到困难依次为:

-   **[multi-armed bandits](https://zhida.zhihu.com/search?content_id=254636918&content_type=Article&match_order=1&q=multi-armed+bandits&zhida_source=entity)**: 相当于单步的 stateless MDP  
    
-   **contextual bandits**: 相当于单步 MDP  
    
-   **small, finite MDPs**: 例如 tractable planning, model-based RL  
    
-   **large, infinite MDPs, continuous space**: 例如 deep RL  
    

**Idea:** 尽管对于复杂的问题我们难以得到具有很强理论保证的算法, 但是我们接下来会从简单的问题出发介绍一些可行的 exploration strategy, 并尝试将这些方法应用到更复杂的问题中.

接下来我们用 multi-armed bandits 作为例子, 逐步引出 exploration problem 的一些基本方法:

### Multi-arm Bandits

![](https://pic2.zhimg.com/v2-855e2e47a158987735c82fb9fe0ccddb_1440w.jpg)

Multi-arm Bandits

**Definition 1**. _multi-armed bandits_

_在 multi-armed bandits 中, 我们有 $n$ 个 arm, 对应于动作空间 $\mathcal{A}$ 中的动作 $\{a_1, a_2, \ldots, a_n\}$. 每一个 arm 都有一个 reward $r(a_i)$, 其独立服从于某个分布 $p(r \mid a_i)$. 我们的目标是最大化累积 reward._

_不妨假设 $r(a_i) = p_{\theta_i}(r_i)$, 其中 $\theta_i$ 是概率分布的参数. 我们可以给 $\theta_i$ 一个先验分布 $p(\theta)$. 并记我们的 belief state (对 state 的估计)为 $\hat{p}(\theta)$_

为了简化起见, 我们考虑静态 MAB, 即 $\theta_i$ 是固定的, 上述过程可以视作一个单步的 Partially Observable Markov Decision Process (POMDP):

1.  state space: 仅有单个状态 $\theta = [\theta_1, \ldots, \theta_n]$  
    
2.  action space: $\mathcal{A} = \{a_1, a_2, \ldots, a_n\}$  
    
3.  observation space: $\mathbb{R}$ (如果我们假设 reward 是连续的)  
    
4.  transition model: $\theta_{t + 1} = \theta_t$ (静态 MAB)  
    
5.  observation model: $p(r \mid a_i, \theta) = p_{\theta_i}(r)$  
    
6.  reward: $r(a_i) \sim p_{\theta_i}(r)$  
    

当我们求解这个 POMDP 时, 我们就得到了 optimal 的 exploration strategy. 这样的做法可能有一些大材小用, 我们实际可以使用更加简单的 strategies. 它们在 big-O notation 的意义下和 optimal strategy 是一样的. 以下是我们衡量 exploration 算法的好坏的标准:

**Definition 2**. _regret_

_对于一个 exploration strategy, 我们定义 regret 为 $Reg(T) = T\cdot E\left[r(a^\ast)\right] - \sum_{t = 1}^{T} r(a_t)$, 前者是最优 action 的期望 reward, 后者是 agent 实际获得的 reward._

我们一个 exploration strategy 的好坏由其 regret 来衡量, 我们希望 regret 越小越好.

## Three Classes of Exploration Methods

接下来我们介绍几种简单的 exploration strategies, 可以理论证明它们在 big-O notation 的意义下是 optimal 的, 尽管它们的实际表现可能存在着一定的差异, 我们可以将他们作为复杂问题的 exploration strategy 的启发:

### Optimistic exploration

对每一个 action $a$ 记录 $\hat{\mu}_a$, 我们考虑 optimistic estimate $a = \arg\max_a \hat{\mu}_a + C \sigma_a,\\$ 这里的 $\sigma_a$ 是 action $a$ 的某种 variance 估计.

**intuition**: 是我们要尝试每一个 arm 知道我们能够足够确信它不够好.

**Example 2**. _一个例子是 UCB (见 Finite-time analysis of the Multiarmed Bandit Problem. Auer et al.): $a = \arg\max_a \hat{\mu}_a + \sqrt{\frac{2\ln T}{N(a)}}.\\$_

**Remark:** 可以证明该算法的 regret 是 $O(\log T)$, 对这个问题在理论上是 optimal 的了.

### Probability matching/ posterior sampling

在这一做法中, 我们会保留一个 belief state $\hat{p}(\theta_1, \ldots, \theta_n)$.

**idea:** 假设我们的 belief state 是正确的, 使用 $\theta_1,\ldots, \theta_n \sim \hat{p}(\theta_1, \ldots, \theta_n)$, 依据此选择 optimal action.

这样的方式比直接求解 POMDP 要简单很多, 这样的算法称为 posterior sampling, 也称为 **[Thompson sampling](https://zhida.zhihu.com/search?content_id=254636918&content_type=Article&match_order=1&q=Thompson+sampling&zhida_source=entity)**.

**Remark:** 分析这一算法的理论性质是很困难的, 但是实际中表现很好. 具体可见 An Empirical Evaluation of Thompson Sampling. Chapelle, Li.

### Information gain

在这一类方法中, 我们考虑如何最大化 **[information gain](https://zhida.zhihu.com/search?content_id=254636918&content_type=Article&match_order=1&q=information+gain&zhida_source=entity)**, 在 MAB 问题中, 我们希望最大化对 $\theta$ 的 information gain. 由于我们实际的 infomation gain 是由于我们的 action 而产生的, 故我们可以定义一定的 action $a$ 下的 information gain:

**Definition 3**. _information gain_

_我们定义观测到 $y$ 后 (这里理解为是对随机变量的单个观测) 的 information gain 为 $IG(z,y) = \mathcal{H}(\hat{p}(z)) - \mathcal{H}(\hat{p}(z) \mid y).\\$_

不难基于信息论的知识发现其与 互信息 (mutual information) 之间的联系:

**Proposition 1**. _$\mathbb{E}_y\left[IG(z,y)\right] = \mathcal{I}(z,y).\\$_

**Definition 4**. _information gain_

_我们定义 action $a$ 下的 information gain 为 $IG(z,y \mid a) = \mathbb{E}_{y}\left[\mathcal{H}(\hat{p}(z)) - \mathcal{H}(\hat{p}(z) \mid y) \mid a\right].\\$ 这里 $y$ 的期望是基于 $p(y \mid a)$._

**Example 3**. _应用在 MAB 问题中的算法例子是 Learning to Optimize via Information-Directed Sampling. Russo et al. 具体来说:_

-   _观测的变量是 $y = r_a$, 需要估计的是 $z = \theta_a$, 其中 $\theta_a$ 是 action $a$ 的 reward 的参数._  
    
-   _记 $g(a) = IG(\theta_a, r_a \mid a)$ 为 information gain of $a$._  
    
-   _记 $\Delta(a) = \mathbb{E}[r(a^\ast) - r(a)]$ 为 expected suboptimality of $a$._  
    
-   _我们选择 $a$ 基于 $\arg\min_a \frac{\Delta(a)^2}{g(a)}.\\$_

_不难观察发现这一项也平衡了 exploitation 与 exploration._

## Overview of Exploration in RL

在更加复杂的情形中, 我们没办法得出这些简单问题中能得到的理论保证, 但是我们可能会基于这些简单问题中的一些 insight 来设计 exploration strategy. 我们可以将其应用到 RL 中, 以下是对这一些做法的总览:

-   Optimistic exploration  
    

-   new state = good state  
    
-   需要通过对状态**访问计数**等方式来估计状态是否足够新  
    
-   通常通过 **[exploration bonus](https://zhida.zhihu.com/search?content_id=254636918&content_type=Article&match_order=1&q=exploration+bonus&zhida_source=entity)** 来实现  
    

-   Thompson sampling style algorithms  
    

-   学习一个 Q-functions 或 policies 上的 (信念)分布  
    
-   从信念分布中采样, 并依据这一采样进行决策  
    

-   Information gain style algorithms  
    

-   考虑访问新状态的 information gain  
    

## Optimistic exploration in RL

回顾 UCB: $a = \arg\max_a \hat{\mu}_a + \sqrt{\frac{2\ln T}{N(a)}},\\$ 这里后一项是 exploration bonus.

值得注意的是, MAB 的特殊性在于其只有单个 state, 对于 RL 中的 exploration, 我们不仅要对 action 有 exploration bonus, 同时对 state 也要有 exploration bonus, 也就是我们通常会考虑 $N(\boldsymbol{s}, \boldsymbol{a})$ 或 $N(\boldsymbol{s})$ 形式的 count.

### Count-based exploration

**intuition:** 一个 state $\boldsymbol{s}$ 或 state-action pair $\boldsymbol{s}, \boldsymbol{a}$ 被访问的次数越少, 我们需要添加的 exploration bonus 越大.

基于这一想法, 可以定义添加了 exploration bonus 的 reward function: $r^+(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \mathcal{B}(N(\boldsymbol{s}))\\$ 这里的 $\mathcal{B}(N(\boldsymbol{s}))$ 是一个随着 $N(\boldsymbol{s})$ 增加而减小的函数. 此时我们使用 $r^+(\boldsymbol{s}, \boldsymbol{a})$ 作为 reward function, 进行 exploration.

**Remark:**

-   优点: 很容易添加到任何的 RL 算法中  
    
-   缺点: 需要调整出一个合适的 $\mathcal{B}(N(\boldsymbol{s}))$  
    

### Counting in complex problems

在复杂的问题中, 我们重复访问一个 state 的可能性是很小的, 而在连续 state space 中, 我们不可能到达同一个 state 多次.

![](https://pic2.zhimg.com/v2-f3b45e3454b1a97e58f1b3d950deee01_1440w.jpg)

复杂问题中几乎不可能重复访问同一个状态

在这种情况下, 我们需要设计其他的方法来估计 $N(\boldsymbol{s})$. 然而值得注意的是, 状态之间的相似性各有不同, 我们可以拟合一个 [density model](https://zhida.zhihu.com/search?content_id=254636918&content_type=Article&match_order=1&q=density+model&zhida_source=entity) $p_\theta(\boldsymbol{s})$ (或 $p_\theta(\boldsymbol{s}, \boldsymbol{a})$). 如果一个状态与已见到的状态相似, 那么即使其没有被访问过, 也可以有很大的 $p_\theta(\boldsymbol{s})$.

**Example 4**. _然而 density 和 count 通常还是有一定差异的, 我们如果能够将 density 转化为一个 pseudo-count, 那么我们就可以沿用原先对 exploration bonus 的设计. 实际上, 我们能够把 $p_\theta(\boldsymbol{s})$ 转化为 psuedo-count $N(\boldsymbol{s})$, 这里的做法基于 Unifying Count-Based Exploration and Intrinsic Motivation. Bellemare et al. 2016:_

_我们知道, 在访问 $\boldsymbol{s}$ 前后, 我们可以列出以下两个方程: $P(\boldsymbol{s}) = \frac{N(\boldsymbol{s})}{n},\, P'(\boldsymbol{s}) = \frac{N(\boldsymbol{s}) + 1}{n + 1}\\$_

_我们可以让 $p_\theta(\boldsymbol{s})$ 与 $p_{\theta'}(\boldsymbol{s})$ 也遵循同样的这两个方程:_

1.  _利用当前见过的所有 states $\mathcal{D}$ 拟合一个 density model $p_\theta(\boldsymbol{s})$,_  
    
2.  _走一步 $i$ 观测到 $\boldsymbol{s}_i$._  
    
3.  _用 $\mathcal{D} \cup \boldsymbol{s}_i$ 拟合一个新的 density model $p_{\theta'}(\boldsymbol{s})$_  
    
4.  _使用 $p_{\theta}(s)$ 与 $p_{\theta'}(s)$ 来更新 $\hat{N}(\boldsymbol{s})$_  
    
5.  _设置 $r^+(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \mathcal{B}(\hat{N}(\boldsymbol{s}))$_  
    

_其中第 4 步进行的方式如下: 联立 $p_\theta(\boldsymbol{s}_i) = \frac{\hat{N}(\boldsymbol{s}_i)}{\hat{n}}, \, p_{\theta'}(\boldsymbol{s}_i) = \frac{\hat{N}(\boldsymbol{s}_i) + 1}{\hat{n} + 1}\\$ 可以解出 $\hat{N}(\boldsymbol{s}_i)$ 与 $\hat{n}$ 两个未知数: $\begin{cases}  \hat{N}(\boldsymbol{s}_i) = \hat{n} p_\theta(\boldsymbol{s}_i)\\  \hat{n} = \frac{1 - p_{\theta'}(\boldsymbol{s}_i)}{p_{\theta'}(\boldsymbol{s}_i) - p_\theta(\boldsymbol{s}_i)}  \end{cases}  \\$_

![](https://pic3.zhimg.com/v2-1c38666f0f8625b6db64250293554ba2_1440w.jpg)

Unifying Count-Based Exploration and Intrinsic Motivation 这篇文章中的效果

### Choice of bonus function

首先考虑使用什么样的 bonus function $\mathcal{B}$, 我们有以下几种有效的选择:

-   UCB: $\mathcal{B}(N(\boldsymbol{s})) = \sqrt{\frac{2\ln T}{N(\boldsymbol{s})}}$  
    
-   MBIE-EB (Strehl & Littman, 2008): $\mathcal{B}(N(\boldsymbol{s})) = \sqrt{\frac{1}{N(s)}}$  
    
-   BEB (Kolter & Ng, 2009): $\mathcal{B}(N(\boldsymbol{s})) = \frac{1}{N(\boldsymbol{s})}$  
    

### Choice of density model

接下来考虑使用什么样的 density model $p_\theta$: 由于这里我们的目标仅仅是一个输出 density 的 model, 并不需要能够从中采样或者生成数据, 因此我们可以使用一些简单的模型, 例如 CTS model: condition each pixel on its top-left neighbor.

其他的一些模型: stochastic networks, compression length, EX2

## More Novelty-Seeking Exploration

由于我们只需要能够输出 scores, 我们还可以以下的一些更加新颖的思路:

### Counting with hashes

**Idea:** 我们依然使用 counts, 但是我们利用一个 hash function $\phi(\boldsymbol{s})$ 将 $\boldsymbol{s}$ 映射到一个 $k$\-bit 的 hash, 并尽可能使得相似状态有相似的 hash.

具体来说, 我们可以利用 **auto encoder (AE)** 学习一个 encoder, 在此基础上进行降采样将 $\phi(\boldsymbol{s})$ 转化为只有 01 的 hash.

![](https://pica.zhimg.com/v2-a3dce9ec6f4eb04bb059ac7f7e133562_1440w.jpg)

#Exploration: A Study of Count-Based Exploration 学习 hash function 的模型

参见: Tang et al. "#Exploration: A Study of Count-Based Exploration"

### Implicit density modeling with [exemplar model](https://zhida.zhihu.com/search?content_id=254636918&content_type=Article&match_order=1&q=exemplar+model&zhida_source=entity)s

**Idea:** 我们可以利用 classifier 来进行 density estimation, 如果一个 state 很容易与已见过的 states 区分, 则其是 novel 的, 如果一个 state 与过去 states 难以区分, 则说明其有很高的 density.

具体来说, 对于每一个观测到的 state $\boldsymbol{s}$, 我们拟合一个 classifier 将其与所有过去的 states $\mathcal{D}$ 区分开, 我们利用 classifier error 来获得 density. 我们认为 $\{\boldsymbol{s}\}$ 是 positive, 而 $\mathcal{D}$ 是 negative.

**Definition 5**. _exemplar model_

_对于数据集 $X = \{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n\}$, exemplar model 是一系列模型, $\{D_{\boldsymbol{x}_1}, \ldots, D_{\boldsymbol{x}_n}\}$, 其中 $D_{\boldsymbol{x}_i}$ 是一个 binary classifier, 用于区分 $\boldsymbol{x}_i$ 与其他数据._

**Theorem 1**. _对于离散分布, 我们可以使用 $p_\theta(\boldsymbol{s}) = \frac{1 - D_{\boldsymbol{s}}(\boldsymbol{s})}{D_{\boldsymbol{s}}(\boldsymbol{s})}.\\$ 表示 state $\boldsymbol{s}$ 的 density. 其中 $D_{\boldsymbol{s}}(\boldsymbol{s})$ 表示的是将 $\boldsymbol{s}$ 归类为 positive 的概率._

_Proof._ 不妨假设负样本的概率为 $q(\boldsymbol{s})$, 这其实就是我们想要估计的 density. 很显然 optimal classifier 将 $\boldsymbol{s}$ 归类为 positive 的概率为 $D_{\boldsymbol{s}}(\boldsymbol{s}) = \frac{p(\boldsymbol{s})}{p(\boldsymbol{s}) + q(\boldsymbol{s})}.\\$ 可以解得 $q(\boldsymbol{s}) = \frac{1 - D_{\boldsymbol{s}}(\boldsymbol{s})}{D_{\boldsymbol{s}}(\boldsymbol{s})} p(\boldsymbol{s}).\\$ 注意由于正类的概率密度函数是一个点分布, 因此 $p(\boldsymbol{s}) = 1$, 也就是 $p_\theta(\boldsymbol{s}) = \frac{1 - D_{\boldsymbol{s}}(\boldsymbol{s})}{D_{\boldsymbol{s}}(\boldsymbol{s})}.\\$ ◻

然而很显然对于连续分布, 由于点分布的概率密度认为是 $\infty$, 我们需要进行一些正则化处理, 例如将点分布换成一个很小的高斯分布.

由于对每一个 state 都需要训练一个 classifier, 这一方法可能会很昂贵, 我们可以考虑使用 amortized model, 例如只训练一个模型, 并且将 exemplar 作为 condition.

参见: EX2: Exploration with Exemplar Models for Deep Reinforcement Learning. Ostrovski et al. 2017

### Heuristic estimation of counts via errors

**Idea:** 在我们训练神经网络时, 对于那些出现概率大或密度高的数据, 其上的误差会很小 (否则总 loss 就会很大), 而在密度小的数据上的误差会很大.

具体来说, 不妨考虑我们有一个 target function $f^\ast(\boldsymbol{s}, \boldsymbol{a})$, 给定 buffer $\mathcal{D} = \{(\boldsymbol{s}_i, \boldsymbol{a}_i)\}$, 拟合一个 $\hat{f}_\theta(\boldsymbol{s}, \boldsymbol{a})$. 我们可以使用 $\mathcal{E} = \|\hat{f}_\theta(\boldsymbol{s}, \boldsymbol{a}) - f^\ast(\boldsymbol{s}, \boldsymbol{a})\|^2$ 作为 bonus, 这一值越大, 说明这一 state-action pair 越 novel.

接下来我们可以考虑 $f^\ast(\boldsymbol{s}, \boldsymbol{a})$ 的选择问题.

-   一个通常的选择是令 $f^\ast(\boldsymbol{s}, \boldsymbol{a}) = \boldsymbol{s}'$, 也就是预测下一个 state (这与 information gain 有一定的联系).  
    
-   更简单的选择是 $f^\ast(\boldsymbol{s}, \boldsymbol{a}) = f_\phi(\boldsymbol{s}, \boldsymbol{a})$, 其中 $\phi$ 是一个 random parameter vector. 根据前面的 intuition, 这里的 $f_\phi$ 的形式完全不重要, 只要它是一个在 $\mathcal{S} \times \mathcal{A}$ 上不容易简单拟合的函数即可.

![](https://pic4.zhimg.com/v2-1185892b634a2d4d357156137e11e0b1_1440w.jpg)

参见: Burda et al. Exploration by Random Network Distillation. 2018

## Posterior Sampling in Deep RL

前面我们介绍了非常多种 **optimistic exploration/ novelty-seeking exploration** 的方法, 接下来我们考虑 MAB 中引入的第二个方法: **Thompson sampling**.

### Introduction

回顾 Thompson sampling: $\theta_1,\ldots, \theta_n \sim \hat{p}(\theta_1, \ldots, \theta_n),\\$$a = \arg\max_a \mathbb{E}_{\theta_a}\left[r(a)\right].\\$

**Idea:** 在 MAB 中 $\hat{p}(\theta_1, \ldots, \theta_n)$ 可以视作是 rewards 上的分布, 这在 MDP 中的近似是 Q-function.

具体来说, 我们考虑以下的过程:

1.  从 $p(Q)$ 中 sample 一个 Q-function $Q$  
    
2.  依据 $Q$ 做一个 episode 的 action  
    
3.  更新 $p(Q)$  
    

这里使用 Q-function 的好处在于, 由于 Q-learning 是 off-policy, 我们并不关心我们收集数据的 Q-function 是什么.

但我们尚未解决的问题是, 我们应该如何表示 $p(Q)$ 呢?

### Bootstrap

回忆我们在 model-based RL 中为了衡量模型的 uncertainty 以避免滥用 dynamic model, 我们引入了 **bootstrap ensemble model** 方法. 类似地, 我们可以使用 ensemble Q-function 来表示 $p(Q)$.

基于之前讨论过的做法:

-   给定 dataset $\mathcal{D}$, 我们利用有放回采样得到 $N$ 个 dataset $\mathcal{D}_1, \ldots, \mathcal{D}_N$, 于是我们可以训练 $N$ 个 model $f_{\theta'}$. 在 DL 中的技巧是使用同一个 dataset, 但是使用不同的 initialization.  
    
-   为了 sample $p(\theta)$, 我们只需要 sample 一个 index $i$ 并使用 $f_{\theta_i}$.  
    
-   这依然非常昂贵, 更进一步的做法是 (参见后面提到的 paper) 使用一个 shared network, 但是用多个不同的 head (最后的层).

![](https://pic1.zhimg.com/v2-850e1075b97ebf7e95b453b890f0663c_1440w.jpg)

不同 Q 函数共用大部分网络

### Comparison with $\epsilon$\-greedy

相较于仅有单个 $Q$ 函数的算法, 这一算法引入随机性通过 $p(Q)$ 实现了 exploration, 而 $\epsilon$\-greedy 则是在现有 $Q$ 的基础上通过随机性进行 exploration. 这两种方式都引入了随机性, 然而它们的表现可能会有很大的不同.

其背后的原因主要在于:

-   在 $\epsilon$\-greedy 中, 我们在一个 episode 中并不会坚持固定的某种行为, 而是会来回震荡, 这可能会导致我们无法走到一个有价值的位置.  
    
-   利用 random Q-functions 进行 exploration 时, 我们可以在整个 episode 中坚持一个一致的策略 strategy.

![](https://picx.zhimg.com/v2-e97c656ca349816fae176ef591364d33_1440w.jpg)

在这个潜艇的游戏中, 不同 Q function 可以对应不同的整局游戏一致的策略, 例如倾向于往上. 而 epsilon-greedy 则会上下震荡

**Remark:**

1.  不难发现这个算法相当简单, 我们对原始的 reward function 不需要做任何的修改, 只需要使用 ensemble Q-function.  
    
2.  然而, 遗憾的是, 这个做法的表现不如 counted-based methods/ pseudo-counts, 例如在 Montezuma's Revenge 中完全没有用.  
    

参见: Osband et al. Deep Exploration via Bootstrapped DQN. 2016

## Information Gain in Deep RL

在介绍完前两种 exploration strategies 即 optimistic exploration 与 Thompson sampling 后, 我们接下来考虑第三种 exploration strategy: **information gain**.

首先需要考虑的是, 在 MAB 中我们的未知只有一个 state $\theta$, 但在 Deep RL 中我们未知的有很多, 选取什么作为 information gain 的指标呢? 主要可以考虑的有:

-   使用 reward $r(\boldsymbol{s}, \boldsymbol{a})$: 在 reward 稀疏时并不是很有用.  
    
-   使用 $p(\boldsymbol{s})$: 那么我们又回到了某种 count-based exploration.  
    
-   使用 dynamics $p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})$ 是一个很好的选择, 但这样的做法同样是 heuristic 的, 更多是一种经验性和直觉性的做法.  
    

然而, 对于复杂的问题来说, 直接使用 information gain, 无论我们估计什么都是 intractable 的, 因此我们通常会使用一些 approximations:

### prediction gain

考虑一个 density model $p_\theta(\boldsymbol{s})$, 是当我们见到一个新状态 $\boldsymbol{s}$ 之后我们更新了参数 $\theta$ 到 $\theta'$, 我们用 $\log p_{\theta'}(\boldsymbol{s}) - \log p_\theta(\boldsymbol{s})$ 作为一种 information gain 的粗略估计.

**intuition:** 如果 $\log p_{\theta'}(\boldsymbol{s}) - \log p_\theta(\boldsymbol{s})$ 很大, 也就意味着原先我们在 $\boldsymbol{s}$ 附近的 uncertainty 很大, 而现在我们对 $\boldsymbol{s}$ 的 uncertainty 降低了很多.

**Remark:** 这一做法比较粗略, 同时也与 count-based exploration 有着较强的联系.

参见:

-   Schmidhuber. (1991). A possibility for implementing curiosity and boredom in model-building neural controllers (定义了 boredom 的概念, 当某个状态被充分探索 (预测误差趋近于零)时, 智能体对该状态失去兴趣 (内在奖励降低), 转而探索其他区域)  
    
-   Bellemare, Srinivasan, Ostroviski, Schaul, Saxton, Munos. (2016). Unifying Count Based Exploration and Intrinsic Motivation (介绍了 information gain 与 count-based method 之间的一些联系)  
    

### VIME

我们不难得出以下的结论:

**Proposition 2**. _$\mathbb{E}_y \left[IG(z, y)\right] = \mathbb{E}_y \left[D_{KL}(p(z\mid y) \parallel p(z))\right].\\$_

这意味着 $y$ 中包含的信息越多, 那么 $p(z\mid y)$ 与 $p(z)$ 的 KL 散度就越大.

在 **VIME** 中, 我们学习一个参数为 $\theta$ 的 dynamic model $p_\theta(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$, 并且考虑对参数 $z = \theta$ 的 information gain, 我们的观测则是 $y = (\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1})$. 于是我们想要最大化的就是 $D_{KL}(p(\theta \mid h, \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}) \parallel p(\theta \mid h))\\$ 这里的 $h$ 是 history transitions.

**Idea:** 不难发现如果一个 transition 让我们对 $\theta$ 的 belief 有更大的变化, 则其更加 informative.

首先明确我们需要对 model uncertainty 也就是 $\theta$ 进行一个表示, 这里使用我们在 model-based RL 中提到的 **Bayesian neural network**. 首先使用独立性假设 $p(\theta \mid h) = \prod_i p(\theta_i \mid h)\\$ 然后我们使用一个高斯分布来表示 $p(\theta_i \mid h)$, 也就是 $p(\theta_i \mid h) = \mathcal{N}(\mu_{\phi,i}, \sigma_{\phi,i})$. 这里的 $\phi$ 是参数, 换言之我们会使用一个 BNN $q_\phi(\theta)$ 来近似 $p(\theta \mid h)$.

那么我们应该如何保证近似的有效性呢? 我们可以使用 variational inference, 具体来说, 我们考虑最小化 KL 散度: $\begin{aligned}  D_{KL}(q_\phi(\theta) \parallel p(\theta \mid h)) &= \int q_\phi(\theta) \log \frac{q_\phi(\theta)}{p(\theta \mid h)} \text{d}\theta\\  &= \int q_\phi(\theta) \log \frac{q_\phi(\theta) p(h)}{p(\theta, h)} \text{d}\theta\\  &= \log p(h) - \int q_\phi(\theta) \log \frac{p(\theta, h)}{q_\phi(\theta)} \text{d}\theta \end{aligned}\\$ 如果熟悉 variational inference 的话, 我们会发现前者 $\log p(h)$ 是 evidence, 而 $\int q_\phi(\theta) \log \frac{p(\theta, h)}{q_\phi(\theta)} \text{d}\theta\\$ 是 ELBO. 如果我们假设 evidence 不变, 我们的任务就转化为最大化 ELBO, 也即最小化 $D_{KL}(q_\phi(\theta) \parallel p(\theta) p(h \mid \theta)).\\$ 优化这个 ELBO 的过程需要使用 **stochastic gradient variational Bayes (SGVB)** 算法, 我们暂不展开.

在有了一个有效的近似之后, 我们的 information gain 就可以近似表示为 $D_{KL}(q_{\phi'}(\theta) \parallel q_\phi(\theta)).\\$ 其中 $\phi'$ 是观测到一个新样本后更新后的参数.

整个训练流程大致如下:

1.  For $n = 1,\ldots,N$:  
    

2.  重复以下 $K$ 次:  
    

3.  从环境中采样一个 transition $(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}, r_t)$, 并添加到 buffer $\mathcal{R}$  
    
4.  通过 $\phi_{n + 1}$ 与新样本近似 $\phi_{n + 1}'$, 基于此计算 information gain 作为 bonus term  
    
5.  利用 $r_t$ 与 bonus term 构造获取新的 reward  
    

6.  从 buffer $\mathcal{R}$ 中采样一个 minibatch, 通过最小化 ELBO 训练 BNN $q_{\phi_n}(\theta)$ 至 $q_{\phi_{n + 1}}(\theta)$  
    
7.  利用我们构造的 reward 以及任意的 RL 算法训练 policy.  
    

**Remark:** Approximate IG 的好处是我们有数学上更强的保证. 但是其缺点是 models 通常会更加复杂, 通常更难得到有效使用.

### Exploration with model errors

这一部分我们介绍的方法并没有利用 information gain, 考虑 **VIME** 中的 $D_{KL}(q_{\phi'}(\theta \mid h) \parallel q_\phi(\theta \mid h))\\$可以理解为惩罚观测一个 state 产生的 gradient. 这里的 intuition 其实和基于 model error 的 exploration 很像, 在误差很大的地方的 state 产生的 gradient 自然也会很大:

以下是一些示例:

Stadie, Levine, Abbeel (2015). Incentivizing Exploration in Reinforcement Learning with Deep Predictive Models.

-   用 AE 来编码图片 observation  
    
-   使用隐码空间上的预测模型  
    
-   利用 model error 作为 exploration bonus  
    

更多的变种可见 Formal Theory of Creativity, Fun, and Intrinsic Motivation:

-   利用 model error 作为 exploration bonus  
    
-   利用 model gradient 作为 exploration bonus  
    

## Summary

在本节中, 我们:

-   引出了 exploration 问题  
    
-   介绍了 MAB 中常见的三种 exploration strategies: optimistic exploration, Thompson sampling, information gain  
    
-   将三种常见的 exploration strategy 推广到了 Deep RL 中, 具体来说:  
    

-   Optimistic exploration:  
    

-   使用 counts 与 pseudo-counts 作为 exploration bonus  
    
-   除了转化为计数, 我们还可以使用一些更加新颖的方法, 例如 hashing, exemplar models  
    

-   Thompson sampling style algorithms:  
    

-   通过 bootstrap ensemble 维持一个 models 的概率分布  
    
-   一个选择是选择 Q-function 上的分布  
    

-   Information gain style algorithms  
    

-   这一类方法比较复杂, 通常需要使用一些 approximations  
    
-   我们可以使用 variational inference 近似来估计 information gain: VIME