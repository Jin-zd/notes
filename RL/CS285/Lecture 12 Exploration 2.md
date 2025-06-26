## 1 Another perspective on exploration
前面的讨论中，我们知道奖励稀疏会使得探索变得困难，并介绍了一些探索策略，例如乐观探索、汤普森采样、信息增益这三个主要类别。但不妨进一步考虑，在完全没有任何奖励的情况下，我们能否实现多样的行为？

这样的想法似乎是合理的，不妨考虑一个小孩玩地面上的玩具，在这个过程中并不存在任何形式的奖励，但是这些操作的过程却是一种有效的探索，同时有助于其未来达成更复杂的任务。

如果能在无奖励的情况下实现探索，那么这意味着能
- 在无监督的情况下学习技能，再利用这些技能实现真正的任务； 
- 学习一系列子技能并使用分层强化学习（核心思想是将复杂的任务分解为多个层次的子任务）； 
- 探索所有可能的行为。  

这很显然是非常有意义的。
![](12-1.png)

## 2 Definition & concepts from information theory
在本节中，我们会给出信息论中的一些定义与概念

**Definition 1**. _entropy（熵）_
对于一个分布 $p(x)$，其熵定义为 
$$
\mathcal{H}(p) = -\sum_x p(x) \log p(x)
$$
不难发现当分布是一个均匀分布时，其熵是最大的，当其是一个点分布时。其熵是最小的。故熵在一定程度上可以表示一个分布的覆盖的广泛程度。
![](12-2.png)
**Definition 2**. _mutual information（互信息）_
对于两个分布 $p(x)$ 与 $p(y)$，其互信息定义为 
$$
\mathcal{I}(x; y) = D_{KL}(p(x, y) \parallel p(x)p(y)) = \sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}
$$
互信息可以描述两个变量之间的相关性，也可以理解为一个变量中包含的关于另一个变量的信息量。
![](12-3.png)
在这里，记 $\pi(\boldsymbol{s})$ 表示策略 $\pi$ 下状态的边缘分布，于是 $\mathcal{H}(\pi(\boldsymbol{s}))$ 可以表示策略覆盖状态空间的广泛程度。考虑如下 "定义" 的赋权：

**Definition 3**. _"empowerment（赋权）" (Polani et ai.)_
$$
\mathcal{I}(\boldsymbol{s}_{t + 1}, \boldsymbol{a}_{t}) = \mathcal{H}(\boldsymbol{s}_{t + 1}) - \mathcal{H}(\boldsymbol{s}_{t + 1} \mid \boldsymbol{a}_t)
$$
直觉：这一项衡量了策略探索的能力，这一能力越强，则表明
- $\mathcal{H}(\boldsymbol{s}_{t + 1})$ 越大，表明此时可以探索或到达的状态很多；  
- $\mathcal{H}(\boldsymbol{s}_{t + 1} \mid \boldsymbol{a}_t)$ 越小，说明此时可以通过采取某个动作很确定的到达某个状态；
- 这项赋权表示了策略的动作能否显著影响未来状态，当这一项较大时，说明在信息论意义下，智能体拥有较强的 "控制权"（Control authority）。

接下来会介绍几种在无奖励情况下进行探索的算法。

## 3 Learning by reaching imagined goals
### 3.1 Basic Workflow
在这一部分，我们讨论如何在没有奖励的情况下通过达成目标来学习。

在[[Lecture 1 Imitation Learning]]中我们简单介绍过目标条件模仿学习，其一个基本思想是通过模仿学习来学习一个策略，但是这一策略是在一个特定的目标下的，也就是 $\pi(\boldsymbol{a} \mid \boldsymbol{s}, \boldsymbol{G})$，其中 $\boldsymbol{G}$ 是一个目标状态（可以是一个图像或者一个状态)，通常可以来自于专家轨迹中的一些中间状态。

在目标条件强化学习中，我们希望目标不是来自于已经收集到的数据，而最好能够覆盖更广泛的状态空间。可以使用[[Concepts#18 变分自编码器 (Variational Autoencoder, VAE)|变分自编码器 (Variational Autoencoder, VAE)]]来作为一个状态空间模型学习状态的分布。记编码器为 $p_\theta(\boldsymbol{s} \mid \boldsymbol{z})$，解码器为 $q_\phi(\boldsymbol{z} \mid \boldsymbol{s})$。

在训练过程中，我们利用隐状态提出一系列目标。并且实现这些目标。具体来说：
1. 获取目标：$\boldsymbol{z}_g \sim p(\boldsymbol{z}), \boldsymbol{G} \sim p_\theta(\boldsymbol{s} \mid \boldsymbol{z}_g)$ （这一过程类似于利用训练好的变分自编码器生成图像）；
2. 尝试利用 $\pi(\boldsymbol{a}\mid \boldsymbol{x}, \boldsymbol{G})$ 来实现这个目标，到达最终状态 $S$；
3. 使用数据更新 $\pi$；
4. 将 $\boldsymbol{S}$ 添加到数据（为什么不用路径上的其他状态？这与后面的一个近似有关）中，利用数据更新 $p_\theta(\boldsymbol{x} \mid \boldsymbol{z})$ 和 $q_\phi(\boldsymbol{z} \mid \boldsymbol{x})$。

![](12-4.png)

### 3.2 Skew Fit
在上面的算法流程中，有一些尚不清晰的地方。在第 $4$ 步中，并不能简单地通过最大似然估计来更新变分自编码器，否则可能陷入生成那些相似的状态。以下是解决这个问题的想法：

Idea：希望目标能够尽可能均匀地覆盖所有合法的状态。
考虑 $q_\psi^G(\boldsymbol{G})$ 是生成目标的模型（将变分自编码器换了一种写法），一个朴素的想法是，使用 $\mathcal{S}$ 上的均匀分布。然而这其实并不合理，如果把图片想象为 $\mathbb{R}^n$ 上的点，那么其中绝大多数点都是不合法的。不妨考虑 $U_{\mathcal{S}}$ 是所有合法状态上的均匀分布，目标可以是最小化分布 $U_{\mathcal{S}}$ 与 $q_\psi^G(\boldsymbol{G})$ 之间的 KL 散度，这等价于目标
$$
L(\psi) = \mathbb{E}_{\boldsymbol{S} \sim U_{\mathcal{S}}} \left[\log q_\psi^G(\boldsymbol{S})\right]
$$
然而并没有办法得到一个合法状态上的均匀分布 $U_\mathcal{S}$。但注意在目标达成的过程中收集到一系列实际最终到达的状态 $\boldsymbol{S}$，它们是合法的状态，记其所属分布为 $p_\psi^S(\boldsymbol{S})$，这里的 $\psi$ 的出现不代表其有一个显式的模型，而是表明其与 $q_\psi^G(\boldsymbol{G})$ 生成的目标 $\boldsymbol{G}$ 有一定关联。
之后利用重要性采样，得到 
$$
\begin{aligned}     L(\psi) &= \mathbb{E}_{\boldsymbol{S} \sim U_{\mathcal{S}}} \left[\log q_\psi^G(\boldsymbol{S})\right]\\     &= \mathbb{E}_{\boldsymbol{S} \sim p_\psi^S(\boldsymbol{S})} \left[\frac{U_{\mathcal{S}}(\boldsymbol{S})}{p_\psi^S(\boldsymbol{S})}\log q_\psi^G(\boldsymbol{S})\right]\\     &\propto \mathbb{E}_{\boldsymbol{S} \sim p_\psi^S(\boldsymbol{S})} \left[\frac{1}{p_\psi^S(\boldsymbol{S})}\log q_\psi^G(\boldsymbol{S})\right] \end{aligned}
$$
但是期望内的 $p_\psi^S(\boldsymbol{S})$ 并没有具体的形式来计算，因此论文中使用的做法是使用近似 $p_\psi^S(\boldsymbol{S}) \approx q_\psi^G(\boldsymbol{S})$（不难发现这基于仅仅使用 $\boldsymbol{S}$ 训练 $q_\psi^G(\boldsymbol{S})$），于是目标变为 
$$
\mathbb{E}_{\boldsymbol{S} \sim p_\psi^S(\boldsymbol{S})} \left[q_\psi^G(\boldsymbol{S})^\alpha \log q_\psi^G(\boldsymbol{S})\right]
$$
其中 $\alpha = -1$，并且期望使用训练中的一系列 $\boldsymbol{S}$ 样本估计。

此时我们就能够最大化熵 $\mathcal{H}(p(\boldsymbol{G}))$。而论文中将 $\alpha$ 作为一个超参数 $\alpha \in \left.\left[-1, 0\right)\right.$，这一过程修改了上面的第 $4$ 步，相当于给了不同的数据不同的权重。

这一算法可以被称作 "Skew fit"，训练的模型给了那些新颖的状态更高的出现概率，在这一意义上和前面的基于计数的探索给新颖的状态一定的附加奖励有一定的相似之处。
![](12-5.png)

### 3.3 Connection to empowerment
考虑这一算法的目标，记目标为 $\boldsymbol{G}$，策略实际到达的状态为 $\boldsymbol{S}$，则
- 一方面由于引入了刚才的权重，我们会 $\max \mathcal{H}(p(\boldsymbol{G}))$，这对应于赋权中覆盖状态空间的广泛程度。 
- 另一方面，在训练策略使 $\boldsymbol{S}$ 更加接近 $\boldsymbol{G}$，换言之使得 $p(\boldsymbol{G} \mid \boldsymbol{S})$ 更加确定，也就有 $\mathcal{H}(p(\boldsymbol{G} \mid \boldsymbol{S}))$ 减小，这对应于赋权中给定目标 $\boldsymbol{G}$ 时，能够实现这一目标的能力。

综合上述两项，这意味着在最大化赋权：
$$
\max \mathcal{H}(p(\boldsymbol{G})) - \mathcal{H}(p(\boldsymbol{G} \mid \boldsymbol{S})) = \mathcal{I}(\boldsymbol{S}; \boldsymbol{G})
$$
本部分内容参见:
- Nair\*, Pong\*, Bahl, Dalal, Lin, L. Visual Reinforcement Learning with Imagined Goals. '18  
- Dalal\*, Pong\*, Lin\*, Nair, Bahl, Levine. Skew-Fit: State-Covering Self-Supervised Reinforcement Learning. '19  

## 4 State marginal matching
### 4.1 Basic Ideas
考虑以下的状态边际匹配问题：我们希望学习一个策略 $\pi$ 使得其对应的状态边际 $p_\pi(\boldsymbol{s})$ 接近一个目标 $p^\ast(\boldsymbol{s})$。通常情况下可以使用 KL 散度作为最小化目标。

此时可以设计一个新的奖励函数：
$$
\tilde{r}(\boldsymbol{s}) = p^\ast(\boldsymbol{s}) - p_\pi(\boldsymbol{s})
$$
这个奖励为什么有意义呢？可以注意到我们的样本来自于 $p_\pi(\boldsymbol{s})$，因此就有期望的奖励的性质：、
$$
\mathbb{E}_{p_\pi(\boldsymbol{s})}\left[\tilde{r}(\boldsymbol{s})\right] = -D_{KL}(p_\pi(\boldsymbol{s}) \parallel p^\ast(\boldsymbol{s}))
$$
一个特例是如果 $p^\ast(\boldsymbol{s})$ 是一个均匀分布，那么 
$$
D_{KL}(p_\pi(\boldsymbol{s}) \parallel p^\ast(\boldsymbol{s})) = \mathcal{H}(p_\pi(\boldsymbol{s}))
$$
注意：这里的 $p_\pi(\boldsymbol{s})$ 通常通过模型来拟合，因此 $p_\pi$ 并非一定能够很好地对应当前的策略 $\pi$。

类似的给出学习的过程：
1. 更新 $\pi(\boldsymbol{a} \mid \boldsymbol{s})$ 以最大化 $\mathbb{E}_\pi\left[\tilde{r}(\boldsymbol{s})\right]$；
2. 用 $\pi$ 收集到的数据来更新 $p_\pi(\boldsymbol{s})$。

然而这样的做法有一些问题：
- 不妨把状态空间分为 $k$  部分，假设开始时 $\pi$ 主要覆盖在第 $1$ 部分，于是第 $1$ 部分对应的密度增大；
- 由于我们设计的 $\tilde{r}$，不妨假设奖励鼓励探索第 $2$ 部分，但随着探索的增多，接下来策略又依次主要走向第 $3,4,\ldots,k$ 部分，可能此时 $1$ 部分的密度又逐渐减小了；
- 我们策略可能循环地依次聚焦于第 $1,2,\ldots,k$ 部分，在这些部分间来回震荡，尽管 $p_{\pi}$ 可能最终有很好的覆盖整个状态空间，但 $\pi$ 可能在这两部分来回震荡，并最终仅局限于其中一个部分。

![](12-6.png)

尽管状态空间模型可以对所有状态有一个相对较好的覆盖，但是最终的策略可能仅仅是几条蓝色曲线中的一条。

有一种相对简单的解决方法：
1. 学习 $\pi^k(\boldsymbol{a} \mid \boldsymbol{s})$ 来最大化 $\mathbb{E}_{\pi}\left[\tilde{r}^k(\boldsymbol{s})\right]$，这里的 $k$ 表示迭代次数。
2. 更新 $p_{\pi^k}(\boldsymbol{s})$ 来拟合过去所有的状态边际。

最终我返回 $\pi^\ast(\boldsymbol{a} \mid \boldsymbol{s}) = \sum_{k} \pi^k(\boldsymbol{a} \mid \boldsymbol{s})$。

这里的解决方案基于博弈论，事实上 $p_\pi(\boldsymbol{s}) = p^\ast(\boldsymbol{s})$ 是 $\pi^k$ 和 $p_{\pi^k}$ 之间的一个[[Concepts#22 纳什均衡（Nash Equilibrium）|纳什均衡（Nash Equilibrium）]]，尽管混合策略的做法看起来很奇怪，但事实上最后一个时间步 $k_{\max}$ 的策略不是纳什均衡，而混合的策略是。
![](12-7.png)

通常情况下在状态边际匹配后，可以通过分层强化学习等方式对具体任务进行进一步优化和微调。

参见：
- Lee\*, Eysenbach\*, Parisotto\*, Xing, Levine, Salakhutdinov. Efficient Exploration via State Marginal Matching  
- Hazan, Kakade, Singh, Van Soest. Provably Efficient Maximum Entropy Exploration  

### 4.2 Theoretical perspective of maximizing entropy
回顾之前讨论的两种做法的目标：
- Skew-Fit：$\max \mathcal{H}(p(G)) - \mathcal{H}(p(G \mid S)) = \mathcal{I}(S; G)$。
- SMM（State marginal matching）：（$p^\ast(\boldsymbol{s}) = C$ 的特殊情况）：$\max \mathcal{H}(p_\pi(S))$。

在刚刚介绍的状态边际匹配方法中，似乎仅仅最大化了状态边际的熵，这是我们能做到的最好的事情吗？
考虑以下的情境：先在一个没有奖励的环境中进行充分的探索，在测试时，一个对手会选择最差目标 $G$。
直观地，如果存在某个合法的状态使得策略无法到达，那么对手就会选择这个状态作为目标，因此应当均匀地覆盖所有可能的状态，得到一个状态空间上的均匀分布，也就是应当训练使得
$$
p(G) = \arg\max_p\mathcal{H}(p(G))
$$
换言之由于不知道任何关于目标的信息，最大化熵是能做的的最好的事情。

参见：
- Lee\*, Eysenbach\*, Parisotto\*, Xing, Levine, Salakhutdinov. Efficient Exploration via State Marginal Matching  
- Gupta, Eysenbach, Finn, Levine. Unsupervised Meta-Learning for Reinforcement Learning

## 5 Covering the space of skills
值得注意的是，目标的概念与技能并不相同：
- 目标是一个状态，也就是一个具体的目标。
- 技能通常比状态更加复杂，例如到达某个状态的同时不经过一些区域。

![](12-8.png)

在这里，我们学习一系列不同的技能 $\pi(\boldsymbol{a}\mid \boldsymbol{s}, z)$，其中 $z$ 是一个技能的索引。

直觉：不同的技能应当访问不同的状态空间区域。
![](12-9.png)
基于这样的想法，不同技能对应的状态区域应当是很容易区分的。可以考虑形式为 $r(\boldsymbol{s}, z) = \log p_D(z \mid \boldsymbol{s})$ 的奖励，这里的 $D$ 是某种判别模型，给定一个状态，预测技能，如果从一个状态中能够很好地预测技能，也就是那么这个状态就是一个"好"状态。

训练的目标就是 
$$
\pi(\boldsymbol{a} \mid \boldsymbol{s}, z) = \arg\max_\pi \sum_z \mathbb{E}_{\boldsymbol{s} \sim \pi(\boldsymbol{s} \mid z)} \left[r(\boldsymbol{s}, z)\right]
$$
训练过程中，判别模型与策略都在训练。在训练起始阶段，不同 技能可能是相近的，为了让判别模型能够更好地区分，不同技能之间会有更大的差异。二者相互协同，而非生成对抗网络（GAN）中那样相互对抗。
![](12-10.png)

事实上能证明上述过程对应于最大化目标：
$$
\mathcal{I}(z, \boldsymbol{s}) = \mathcal{H}(z) - \mathcal{H}(z \mid \boldsymbol{s})
$$
最大化需要两方面：
- 最大化 $\mathcal{H}(z)$，只需要技能的先验为均匀分布；  
- 最小化 $\mathcal{H}(z \mid \boldsymbol{s})$，这可以通过最大化 $\log p(z\mid \boldsymbol{s})$ 来实现，也就是技能能够很好地被判别器区分。  

参见：
- Eysenbach, Gupta, Ibarz, Levine. Diversity is All You Need.  
- Gregor et al. Variational Intrinsic Control. 2016