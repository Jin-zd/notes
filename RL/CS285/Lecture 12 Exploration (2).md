## Another perspective on exploration

前面的讨论中, 我们知道 reward 稀疏会使得 exploration 变得困难, 并介绍了一些 exploration strategies, 例如 **optimistic exploration, [Thompson sampling](https://zhida.zhihu.com/search?content_id=254880416&content_type=Article&match_order=1&q=Thompson+sampling&zhida_source=entity), [information gain](https://zhida.zhihu.com/search?content_id=254880416&content_type=Article&match_order=1&q=information+gain&zhida_source=entity)** 这三个主要类别. 但我们不妨进一步考虑, 在**完全没有任何 reward** 的情况下, 我们能否实现多样的行为?

这样的想法似乎是合理的, 不妨考虑一个小孩玩地面上的玩具, 在这个过程中并不存在任何形式的 reward, 但是这些操作的过程却是一种有效的探索, 同时有助于其未来达成更复杂的任务.

如果我们能在无 reward 的情况下实现 exploration, 那么这意味着我们能

-   在无监督的情况下学习技能, 再利用这些 skills 实现真正的任务  
    
-   学习一系列 sub-skills 并使用 hierarchical RL (核心思想是将复杂的任务分解为多个层次的子任务)  
    
-   explore 所有可能的行为  
    

这很显然是非常有意义的.

![](https://pic4.zhimg.com/v2-8250de7f3e9ffeb1f338ce0fe8dc4d31_1440w.jpg)

在训练时以无 reward 的状态进行 exploration, 为未来完成任务做准备

## Definition & concepts from information theory

在本节中, 我们会给出 Information Theory 中的一些定义与概念

**Definition 1**. _entropy_

_对于一个分布_ $p(x)$_, 其 entropy 定义为_ $\mathcal{H}(p) = -\sum_x p(x) \log p(x)\\$

不难发现当分布是一个均匀分布时, 其 entropy 是最大的, 当其是一个点分布时, 其 entropy 是最小的, 故 entropy 在一定程度上可以表示一个分布的覆盖的广泛程度.

![](https://pica.zhimg.com/v2-cdc0520fde68fa5e9a4ef1d3346a0834_1440w.jpg)

entropy 的直观理解

**Definition 2**. _mutual information_

_对于两个分布_ $p(x)$ _与_ $p(y)$_, 其 mutual information 定义为_ $\mathcal{I}(x; y) = D_{KL}(p(x, y) \parallel p(x)p(y)) = \sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}\\$

互信息可以描述两个变量之间的相关性, 也可以理解为一个变量中包含的关于另一个变量的信息量.

![](https://pic4.zhimg.com/v2-8e6fe5012f0935b06acbfa3b187f9fb1_1440w.jpg)

mutual information 的直观理解

在这里, 我们记 $\pi(\boldsymbol{s})$ 表示 policy $\pi$ 下 state 的边缘分布, 于是 $\mathcal{H}(\pi(\boldsymbol{s}))$ 可以表示 policy 覆盖状态空间的广泛程度. 我们考虑如下 "定义" 的 **[empowerment](https://zhida.zhihu.com/search?content_id=254880416&content_type=Article&match_order=1&q=empowerment&zhida_source=entity)**:

**Definition 3**. _"empowerment" (Polani et ai.)_

$\mathcal{I}(\boldsymbol{s}_{t + 1}, \boldsymbol{a}_{t}) = \mathcal{H}(\boldsymbol{s}_{t + 1}) - \mathcal{H}(\boldsymbol{s}_{t + 1} \mid \boldsymbol{a}_t)\\$

**intuition:** 这一项衡量了我们的 policy 探索的能力, 这一能力越强, 则表明

-   $\mathcal{H}(\boldsymbol{s}_{t + 1})$ 越大, 表明我们此时可以探索或到达的状态很多  
    
-   $\mathcal{H}(\boldsymbol{s}_{t + 1} \mid \boldsymbol{a}_t)$ 越小, 说明此时我们可以通过采取某个 action 很确定的到达某个状态.  
    
-   这项 **empowerment** 表示了我们的 policy 的 action 能否显著影响未来 state, 当这一项较大时, 说明在信息论意义下, agent 拥有较强的 "控制权" (control authority).  
    

我们接下来会介绍几种在无 reward 情况下进行 exploration 的算法.

## Learning by reaching imagined goals

### Basic Workflow

在这一部分, 我们讨论如何在没有 reward 的情况下通过 reaching goal 来学习.

在 imitation learning 中我们简单介绍过 **[goal conditioned imitation learning](https://zhida.zhihu.com/search?content_id=254880416&content_type=Article&match_order=1&q=goal+conditioned+imitation+learning&zhida_source=entity)**, 其一个基本思想是通过 imitation learning 来学习一个 policy, 但是这一 policy 是在一个特定的 goal 下的, 也就是 $\pi(\boldsymbol{a} \mid \boldsymbol{s}, \boldsymbol{G})$, 其中 $\boldsymbol{G}$ 是一个 goal state (可以是一个 image 或者一个 state), 通常可以来自于 expert trajectory 中的一些中间状态.

在 goal conditioned RL 中, 我们希望我们的 goal 不是来自于已经收集到的数据, 而最好能够覆盖更广泛的状态空间. 我们可以使用 VAE 来作为一个 **state space model** 学习 states 的分布. 我们记 decoder 为 $p_\theta(\boldsymbol{s} \mid \boldsymbol{z})$, encoder 为 $q_\phi(\boldsymbol{z} \mid \boldsymbol{s})$.

在训练过程中, 我们利用 latent space 提出一系列 goals, 并且实现这些 goals. 具体来说:

1.  获取 goals: $\boldsymbol{z}_g \sim p(\boldsymbol{z}), \boldsymbol{G} \sim p_\theta(\boldsymbol{s} \mid \boldsymbol{z}_g)$ (这一过程类似于我们利用训练好的 VAE 生成图像)  
    
2.  尝试利用 $\pi(\boldsymbol{a}\mid \boldsymbol{x}, \boldsymbol{G})$ 来实现这个 goal, 到达最终状态 $S$  
    
3.  使用数据更新 $\pi$  
    
4.  将 $\boldsymbol{S}$ 添加到数据 (为什么不用路径上的其他状态? 这与我们后面的一个近似有关)中, 利用数据更新 $p_\theta(\boldsymbol{x} \mid \boldsymbol{z})$ 和 $q_\phi(\boldsymbol{z} \mid \boldsymbol{x})$  
    

![](https://pic2.zhimg.com/v2-dc2caa69857f8840264f7a55f59774af_1440w.jpg)

利用实际到达的状态训练 VAE, 并利用 VAE 不断生成新的 goal

### Skew Fit

在上面的算法流程中, 有一些尚不清晰的地方. 在第 $4$ 步中, 我们并不能简单地通过 MLE 来更新我们的 VAE, 否则我们可能陷入生成那些相似的状态. 以下是我们解决这个问题的 idea:

**Idea:** 我们希望 goal 能够尽可能均匀地覆盖所有合法的状态.

考虑 $q_\psi^G(\boldsymbol{G})$ 是生成 goal 的模型 (我们将 VAE 换了一种写法), 一个 naive 的想法是, 我们使用 $\mathcal{S}$ 上的均匀分布. 然而这其实并不合理. 如果把图片想象为 $\mathbb{R}^n$ 上的点, 那么其中绝大多数点都是不合法的. 不妨考虑 $U_{\mathcal{S}}$ 是所有 **合法状态** 上的均匀分布. 我们的目标可以是最小化分布 $U_{\mathcal{S}}$ 与 $q_\psi^G(\boldsymbol{G})$ 之间的 KL 散度, 这等价于目标

$    L(\psi) = \mathbb{E}_{\boldsymbol{S} \sim U_{\mathcal{S}}} \left[\log q_\psi^G(\boldsymbol{S})\right]\\$然而我们并没有办法得到一个合法状态上的均匀分布 $U_\mathcal{S}$ , 但注意我们在 goal-reaching 的过程中收集到一系列实际最终到达的状态 $\boldsymbol{S}$ , 它们是合法的状态, 我们记其所属分布为 $p_\psi^S(\boldsymbol{S})$ , 这里的 $\psi$ 的出现不代表其有一个显式的模型, 而是表明其与 $q_\psi^G(\boldsymbol{G})$ 生成的 goal $\boldsymbol{G}$ 有一定关联.

之后我们利用 importance sampling, 得到 $\begin{aligned}     L(\psi) &= \mathbb{E}_{\boldsymbol{S} \sim U_{\mathcal{S}}} \left[\log q_\psi^G(\boldsymbol{S})\right]\\     &= \mathbb{E}_{\boldsymbol{S} \sim p_\psi^S(\boldsymbol{S})} \left[\frac{U_{\mathcal{S}}(\boldsymbol{S})}{p_\psi^S(\boldsymbol{S})}\log q_\psi^G(\boldsymbol{S})\right]\\     &\propto \mathbb{E}_{\boldsymbol{S} \sim p_\psi^S(\boldsymbol{S})} \left[\frac{1}{p_\psi^S(\boldsymbol{S})}\log q_\psi^G(\boldsymbol{S})\right] \end{aligned}\\$ 但是期望内的 $p_\psi^S(\boldsymbol{S})$ 我们并没有具体的形式来计算, 因此论文中使用的做法是使用近似 $p_\psi^S(\boldsymbol{S}) \approx q_\psi^G(\boldsymbol{S})$(不难发现这基于我们仅仅使用 $\boldsymbol{S}$ 训练 $q_\psi^G(\boldsymbol{S})$ ), 于是目标变为 $\mathbb{E}_{\boldsymbol{S} \sim p_\psi^S(\boldsymbol{S})} \left[q_\psi^G(\boldsymbol{S})^\alpha \log q_\psi^G(\boldsymbol{S})\right],\\$ 其中 $\alpha = -1$, 并且期望我们使用训练中的一系列 $\boldsymbol{S}$ 样本估计.

此时我们就能够最大化熵 $\mathcal{H}(p(\boldsymbol{G}))$. 而论文中将 $\alpha$ 作为一个超参数 $\alpha \in \left.\left[-1, 0\right)\right.$, 这一过程修改了上面的第 $4$ 步, 相当于给了不同的 data 不同的 **weight**:

这一算法可以被称作 "**skew fit**", 我们训练的模型给了那些 novelty state 更高的出现概率, 在这一意义上和前面的 count-based exploration 给 novel 的 states 一定的 bonus 有一定的相似之处.

![](https://pica.zhimg.com/v2-91ed78b9acaec1a06fcbbdcf5631e580_1440w.jpg)

通过 skew fit 覆盖更加广泛的状态

### Connection to empowerment

考虑这一算法的 objective, 记我们的 goal 为 $\boldsymbol{G}$, policy 实际到达的 state 为 $\boldsymbol{S}$, 则

-   一方面由于我们引入了刚才的 weight, 我们会 $\max \mathcal{H}(p(\boldsymbol{G}))$. 这对应于我们 empowerment 中覆盖状态空间的广泛程度.  
    
-   另一方面, 我们在训练 policy 使 $\boldsymbol{S}$ 更加接近 $\boldsymbol{G}$, 换言之使得 $p(\boldsymbol{G} \mid \boldsymbol{S})$ 更加确定, 也就有 $\mathcal{H}(p(\boldsymbol{G} \mid \boldsymbol{S}))$ 减小. 这对应于 empowerment 中给定目标 $\boldsymbol{G}$ 时, 我们能够实现这一目标的能力.  
    

综合上述两项, 这意味着我们在最大化 empowerment $\max \mathcal{H}(p(\boldsymbol{G})) - \mathcal{H}(p(\boldsymbol{G} \mid \boldsymbol{S})) = \mathcal{I}(\boldsymbol{S}; \boldsymbol{G})\\$

本部分内容参见:

-   Nair\*, Pong\*, Bahl, Dalal, Lin, L. Visual Reinforcement Learning with Imagined Goals. '18  
    
-   Dalal\*, Pong\*, Lin\*, Nair, Bahl, Levine. Skew-Fit: State-Covering Self-Supervised Reinforcement Learning. '19  
    

## State marginal matching

### Basic Ideas

我们考虑以下的 **[state marginal matching](https://zhida.zhihu.com/search?content_id=254880416&content_type=Article&match_order=1&q=state+marginal+matching&zhida_source=entity)** 问题: 我们希望学习一个 policy $\pi$ 使得其对应的 state marginal $p_\pi(\boldsymbol{s})$ 接近一个目标 $p^\ast(\boldsymbol{s})$. 通常情况下我们可以使用 KL 散度作为我们的最小化目标.

此时我们可以设计一个新的奖励函数 $\tilde{r}(\boldsymbol{s}) = p^\ast(\boldsymbol{s}) - p_\pi(\boldsymbol{s}),\\$ 这个 reward 为什么 make sense 呢? 可以注意到我们的样本来自于 $p_\pi(\boldsymbol{s})$, 因此我们就有期望的 reward 的性质: $\mathbb{E}_{p_\pi(\boldsymbol{s})}\left[\tilde{r}(\boldsymbol{s})\right] = -D_{KL}(p_\pi(\boldsymbol{s}) \parallel p^\ast(\boldsymbol{s})).\\$

一个特例是如果 $p^\ast(\boldsymbol{s})$ 是一个均匀分布, 那么 $D_{KL}(p_\pi(\boldsymbol{s}) \parallel p^\ast(\boldsymbol{s})) = \mathcal{H}(p_\pi(\boldsymbol{s})).\\$

**Note:** 这里的 $p_\pi(\boldsymbol{s})$ 通常通过模型来拟合, 因此 $p_\pi$ 并非一定能够很好地对应当前的 policy $\pi$.

类似的我们给出学习的过程:

1.  更新 $\pi(\boldsymbol{a} \mid \boldsymbol{s})$ 以最大化 $\mathbb{E}_\pi\left[\tilde{r}(\boldsymbol{s})\right]$  
    
2.  用 $\pi$ 收集到的数据来更新 $p_\pi(\boldsymbol{s})$.  
    

然而这样的做法有一些问题:

**Example 1**.

-   _不妨把状态空间分为_ $k$ _部分, 假设开始时我们的_ $\pi$ _主要覆盖在第_ $1$ _部分, 于是第_ $1$ _部分对应的密度增大_  
    
-   _由于我们设计的_ $\tilde{r}$_, 不妨假设 reward 鼓励我们探索第_ $2$ _部分, 但随着我们探索的增多, 接下来我们的策略又依次主要走向第_ $3,4,\ldots,k$ _部分. 可能此时_ $1$ _部分的密度又逐渐减小了._  
    
-   _我们 policy 可能循环地依次聚焦于第_ $1,2,\ldots,k$ _部分, 在这些部分间来回震荡, 尽管_ $p_{\pi}$ _可能最终有很好的覆盖整个状态空间, 但_ $\pi$ _可能在这两部分来回震荡, 并最终仅局限于其中一个部分._

![](https://pica.zhimg.com/v2-55b313e68509eff4c8038ac75c63a35a_1440w.jpg)

尽管我们的 state space model 可以对所有状态有一个相对较好的 coverage, 但是我们最终的 policy 可能仅仅是几条蓝色曲线中的一条

我们有一种相对简单的解决方法:

1.  学习 $\pi^k(\boldsymbol{a} \mid \boldsymbol{s})$ 来最大化 $\mathbb{E}_{\pi}\left[\tilde{r}^k(\boldsymbol{s})\right]$, 这里的 $k$ 表示迭代次数.  
    
2.  更新 $p_{\pi^k}(\boldsymbol{s})$ 来拟合**过去所有的** state marginals.  
    

最终我们返回 $\pi^\ast(\boldsymbol{a} \mid \boldsymbol{s}) = \sum_{k} \pi^k(\boldsymbol{a} \mid \boldsymbol{s})$.

**Note:** 这里的解决方案基于博弈论, 事实上 $p_\pi(\boldsymbol{s}) = p^\ast(\boldsymbol{s})$ 这是 $\pi^k$ 和 $p_{\pi^k}$ 之间的一个纳什均衡. 尽管混合策略的做法看起来很奇怪, 但事实上最后一个时间步 $k_{\max}$ 的 policy 不是纳什均衡, 而混合的 policy 是.

![](https://pic3.zhimg.com/v2-9825546adbcae31bb056e02c1f958472_1440w.jpg)

SMM 算法的效果 (SAC 算法我们会在后面的 control as inference 一节讲到)

通常情况下在 SMM 后, 我们可以通过 hierarchical RL 等方式对具体任务进行进一步优化和微调.

参见:

-   Lee\*, Eysenbach\*, Parisotto\*, Xing, Levine, Salakhutdinov. Efficient Exploration via State Marginal Matching  
    
-   Hazan, Kakade, Singh, Van Soest. Provably Efficient Maximum Entropy Exploration  
    

### Theoretical perspective of maximizing entropy

回顾我们之前讨论的两种做法的目标:

-   **Skew-Fit**: $\max \mathcal{H}(p(G)) - \mathcal{H}(p(G \mid S)) = \mathcal{I}(S; G)$  
    
-   **SMM (state marginal matching)** ($p^\ast(\boldsymbol{s}) = C$ 的特殊情况): $\max \mathcal{H}(p_\pi(S))$  
    

在刚刚介绍的 state marginal matching 方法中, 我们似乎仅仅最大化了 state marginal 的 entropy. 这是我们能做到的最好的事情吗?

我们考虑以下的情境: 我们先在一个没有 reward 的环境中进行充分的探索, 在测试时, 一个 adversary 会选择 worst goal $G$.

直观地, 如果存在某个 valid state 使得我们的 policy 无法到达, 那么 adversary 就会选择这个 state 作为 goal, 因此我们应当均匀地覆盖所有可能的 state, 得到一个状态空间上的均匀分布, 也就是应当训练使得 $p(G) = \arg\max_p\mathcal{H}(p(G))\\$ 换言之由于我们不知道任何关于 goal 的信息, 最大化 entropy 是我们能做的的最好的事情.

参见:

-   Lee\*, Eysenbach\*, Parisotto\*, Xing, Levine, Salakhutdinov. Efficient Exploration via State Marginal Matching  
    
-   Gupta, Eysenbach, Finn, Levine. Unsupervised Meta-Learning for Reinforcement Learning.  
    

## Covering the space of skills

值得注意的是, **goal** 的概念与 **skill** 并不相同:

-   **goal**: goal 是一个 state, 也就是一个具体的目标  
    
-   **skill**: skill 通常比 state 更加复杂, 例如到达某个 state 的同时不经过一些区域.  
    

![](https://pica.zhimg.com/v2-f6e76242729e6cf34743f1a58db40b58_1440w.jpg)

可能的 skill

在这里, 我们学习一系列不同的 skills $\pi(\boldsymbol{a}\mid \boldsymbol{s}, z)$, 其中 $z$ 是一个 skill 的 index.

**intuition**: 不同的 skills 应当访问不同的 state-space regions.

![](https://picx.zhimg.com/v2-d1fd61edda6d80e4d32d334482d13b83_1440w.jpg)

我们期望不同 skills 应当具有的特点

基于这样的想法, 不同 skills 对应的 states 区域应当是很容易区分的. 我们可以考虑形式为 $r(\boldsymbol{s}, z) = \log p_D(z \mid \boldsymbol{s})$ 的 reward, 这里的 $D$ 是某种 discriminative model, 给定一个 state, 预测 skill. 如果从一个 states 中我们能够很好地预测 skill, 也就是那么这个 states 就是一个 "good" states.

我们训练的 objective 就是 $\pi(\boldsymbol{a} \mid \boldsymbol{s}, z) = \arg\max_\pi \sum_z \mathbb{E}_{\boldsymbol{s} \sim \pi(\boldsymbol{s} \mid z)} \left[r(\boldsymbol{s}, z)\right].\\$训练过程中, discriminative model 与 policy 都在训练. 在训练起始阶段, 不同 skills 可能是相近的, 为了让 discriminative model 能够更好地区分, 不同 skills 之间会有更大的差异. 二者相互协同, 而非 GAN 中那样相互对抗.

![](https://pic2.zhimg.com/v2-373d8427d1180545d8b3dc038aaa62d9_1440w.jpg)

训练过程示意图

事实上我们能证明上述过程对应于最大化 objective $\mathcal{I}(z, \boldsymbol{s}) = \mathcal{H}(z) - \mathcal{H}(z \mid \boldsymbol{s})\\$ 最大化需要两方面:

-   最大化 $\mathcal{H}(z)$, 只需要 skills 的先验为均匀分布  
    
-   最小化 $\mathcal{H}(z \mid \boldsymbol{s})$, 这可以通过最大化 $\log p(z\mid \boldsymbol{s})$ 来实现, 也就是 skills 能够很好地被 discriminator 区分.  
    

参见:

-   Eysenbach, Gupta, Ibarz, Levine. Diversity is All You Need.  
    
-   Gregor et al. Variational Intrinsic Control. 2016