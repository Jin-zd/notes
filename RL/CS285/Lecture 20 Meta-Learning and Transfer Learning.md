在本节中我们将介绍 **[meta-learning](https://zhida.zhihu.com/search?content_id=256210233&content_type=Article&match_order=1&q=meta-learning&zhida_source=entity)** 和 **[transfer learning](https://zhida.zhihu.com/search?content_id=256210233&content_type=Article&match_order=1&q=transfer+learning&zhida_source=entity)** 的相关概念以及它们在 RL 中的应用.

## 1 Introduction

重新回顾我们介绍 exploration 中提到的问题, 在 Breakout (打砖块的那个游戏) 中非常容易, 但是在像 Montezuma's Revenge 这样的游戏中就非常困难, 即使是游戏中最简单的一层.

![](https://pic4.zhimg.com/v2-622c116c5fc16206a4ca7d0d437b9efb_1440w.jpg)

Breakout 与 Montezuma&#39;s Revenge

在这个游戏中, reward 的结构主要如下, 这样的 reward 结构并不能很好地给我们的算法提供指引

-   Getting key = reward  
    
-   Opening door = reward  
    
-   Getting killed by skull = bad  
    

与 RL 算法相反的是, 事实上 Breakout 对人类来说反而更难, Montezauma's Revenge 反而很简单, 这是因为我们有很多先验知识, 这些对于问题结构的先验知识可以让我们很容易地解决这个问题:

-   基于物理常识, 我们知道人会受重力影响往下掉  
    
-   基于 video game 的知识, 我们知道要躲避骷髅头, 楼梯可以被攀登, 钥匙可以开门, 进入新的房间意味着我们推进了游戏.  
    

在我们玩 Montezauma's Revenge 的时候, 我们实际在做的是一种 **transfer learning**. 我们之前介绍这个游戏的时候将其作为一个 exploration 的例子, 或许更好的做法是设计一个能够更好地 transfer 的算法. 直观地来说, 就是从解决过去的 tasks 中获取关于新 task 的知识.

目前我们介绍的 RL 算法通常并不能像人类一样利用先验知识, 但如果我们从 transfer learning 的角度出发时, 我们可以设计一个合理的算法.

在 transfer learning 中我们要将所学知识存储在哪里?

-   **Q-function**: 能够告诉我们哪些 actions 和 states 是好的.  
    
-   **[Policy](https://zhida.zhihu.com/search?content_id=256210233&content_type=Article&match_order=1&q=Policy&zhida_source=entity)**: 可以告诉我们哪些 actions 潜在可能是好的. 有些 actions 永远不是好的, 因此我们至少能够学会排除它们.  
    
-   **Models**: 不同任务中的物理法则也许是一致的.  
    
-   **[Features/hidden states](https://zhida.zhihu.com/search?content_id=256210233&content_type=Article&match_order=1&q=Features%2Fhidden+states&zhida_source=entity)**: 可以告诉我们什么样的 representation 是好的, 例如通过之前的任务发现骷髅头是危险的.  
    

### 1.1 Transfer learning terminology

**Definition 1**. _transfer learning_

_**transfer learning**: 使用过去一系列 tasks 上的 experience 来实现新 task 上更快的学习或者更好的表现._

**Note:** 在 RL 中, 通常一个 task 对应于一个 MDP.

**Definition 2**. _source domain, target domain_

-   _**source domain**: 我们获取 experience 的任务所属的 domain_  
    
-   _**target domain**: 我们希望解决的任务所属的 domain_  
    

通常来说我们希望能够在 target domain 上有很好地表现. 当然对于 life-long learning 的情况我们会希望在 source domain 有同样好的表现, 但目前我们仅考虑在 target domain 上的表现.

**Definition 3**. _"shot"_

_**"shot"**: 在 target domain 中我们需要的尝试次数. 以下是一些常见的 "shot" 的定义:_

-   _0-shot: 只需要运行 source domain 中训练的 policy 就能得到很好的表现._  
    
-   _1-shot: 只需要尝试这个 task 一次, 这在不同任务上可能有不同含义, 例如 Montezuma's Revenge 这种 video game 中可能是仅玩一个 episode, 在 robot manipulation 这种 task 中可能就是进行一次操作._  
    
-   _few-shot: 尝试几次._  
    

我们如何建模这些 transfer learning 的问题呢? 事实上, 这对不同 domain 的任务通常有较大的差异. 通常来说, 主要的方式有以下几类:

1.  **Forward transfer**: (通常是在一个 task 上) 学习一个能有效 transfer 的 policy  
    

2.  在 source task 上训练, 然后在 target task 上 train/ fine-tune.  
    
3.  因为训练用的 source task 只有一个, 因此依赖于 tasks 之间具有很大的相似性.  
    

4.  **Multi-task transfer**: 在多个任务上训练, 然后 transfer 到新的 task.  
    

5.  可以在 multi-task learning 中共享 representations 和 layers.  
    
6.  因为训练的 task 有多个, 它们形成了一个分布, 新的 task 只需要接近 training task 所属的分布 (接近所属分布比接近单个 task 更容易实现).  
    

7.  **Meta-learning**: learn to learn on many tasks  
    

8.  在 source domain 上训练时我们就要考虑 adapting 到新的 task 上.  
    

我们这节内容主要会将重心放在 **meta-learning** 上. 不过我们依然会先介绍一些 [forward transfer](https://zhida.zhihu.com/search?content_id=256210233&content_type=Article&match_order=1&q=forward+transfer&zhida_source=entity) 和 [multi-task transfer](https://zhida.zhihu.com/search?content_id=256210233&content_type=Article&match_order=1&q=multi-task+transfer&zhida_source=entity) 的方法 (这些方法通常比较分散, 因此我们可能只能 cover 其中的一部分重要的 principles).

## 2 Forward transfer learning

### 2.1 Pretraining + finetuning

在 CV 等领域中, [pretraining + finetuning](https://zhida.zhihu.com/search?content_id=256210233&content_type=Article&match_order=1&q=pretraining+%2B+finetuning&zhida_source=entity) 是一个非常常见的做法, 例如在 ImageNet 上进行预训练, 然后在其他任务上进行 finetuning, 这属于一种 **forward transfer learning**.

![](https://pic3.zhimg.com/v2-7b2b8ab5472fdd36b469ced572ee83b8_1440w.jpg)

Pretraining + finetuning

这样的基本流程应用在 RL 中也是可行的, 这是我们从 supervised Learning 中直接继承过来的方法, 因此我们不会详细地介绍, 简单来说, 是在预训练后替换最后的若干层, 固定部分层的参数 (也可能不固定), 然后在 target domain 上进行 finetuning.

在 RL 中, 我们可以应用类似的方法, 这样的方法可能遇到什么问题吗?

-   **Domain shift**: 在 source domain 中学到的 representation 可能在 target domain 中并不适用. 可能存在一个 gap, 例如虚拟环境中的驾驶环境过于单一 (这只是 visually 不同, 但是背后的 mechanism 是一致的).  
    
-   **difference in the MDP**: 在 source domain 中的一些事情在 target domain 中可能并不适用, 例如二者可能有不同的物理.  
    
-   **finetuning issues**: 在 finetuning 过程中可能依旧需要 exploration, 但是有可能 pretraining 得到的 optimal policy 已经是 deterministic 的了.  
    

接下来我们依次介绍如何处理这些问题.

### 2.2 Domain adaptation for [domain shift](https://zhida.zhihu.com/search?content_id=256210233&content_type=Article&match_order=1&q=domain+shift&zhida_source=entity)

在 CV 中, 一个解决 domain shift 的方法是 **invariance assumption.**

**Intuition**: 任何在 domains 间不同的都是无关的.

**Example 1**.

-   _例如现实世界中可能有雨, 而 simulator 中可能没有. 那么 invariance assumption 则告诉我们是否有雨和我们如何驾驶无关._  
    
-   _另一方面在现实世界和 simulaor 中驾驶地点在分布上是一致的, 因此驾驶地点根据 assumption 是有关的._  
    

严格来说, invariance assumption 是说:

**Definition 4**. _invariance assumption_

_假设输入分布 $p(x)$ 在两个 domain 是不同的, invariance assumption 表示存在一种 representation $z = f(x)$ 使得_

-   _条件分布对齐: $p(y\mid z) = p(y\mid x)$, 也就是 $z$ 中包含了预测 $y$ 所需的所有信息._  
    
-   _边缘分布对齐: 在 source domain 和 target domain 中, $p(z)$ 是一致的._  
    

这类 domain shift 在 CV 中已经得到了广泛的研究, 相关的方法主要 **domain confusion**, **domain adversarial neural network (DANN)**. 在这些方法中, 我们需要 target domain 中的少量图片.

这里我们以 **Adversarial Domain Adaptation (ADA)** 为例, 简单介绍一下其原理:

**Example 2**. _**Idea:** 在网络中添加一些 layer (通常在网络的卷积部分后), 分别输入 source domain 和 target domain 中的图片, 计算它们在这些 layer 中的 activation, 使用一些特定的 loss 使得它们尽可能接近. 也就是让它们的 activation 的分布 $p(z)$ 一致._

_具体实现上, 我们会训练一个 binary classifier $D_\phi(z)$ 输入这些层的 activation, 并让其判断是否属于 target domain. 之后我们会求出 $D_\phi(z)$ 的梯度, 并将其反向 (使得结果更加不 discriminative), 并反向传播回原网络._

![](https://pic4.zhimg.com/v2-7f24ce811210137d7a969a1841d860fd_1440w.jpg)

Adversarial Domain Adaptation (ADA)

### Applied to RL

在 RL 中使用这样的方法依旧需要我们有一些 target domain 中的 image 等数据, 只不过我们不再需要在 target domain 中重新利用 RL 等算法进行训练.

但是也有一些需要注意的, 例如如果 target domain 中的数据都是质量很差的 driving data, 那么要求网络中间的几层变得 invariant 可能会影响我们得到一个好的 policy.

### 2.3 Domain adaptation in RL for dynamic

如果 dynamic 并不一致, 那么仅仅输出一个一致的 representation 可能并不足够, 因为我们不能忽略这些 dynamic 的差异.

这里的一个做法是惩罚那些在 source domain 可以做到, 但是在 target domain 中做不到的行为:

例如在现实中到达目标中间可以有一堵墙, 而在 simulator 中没有, 那么我们可以做的是修改 reward 为: $\tilde{r}(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \Delta r(\boldsymbol{s}, \boldsymbol{a}).\\$ 一个可能的做法是 $\Delta r(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}) = \log p_{target}(\boldsymbol{s}_{t + 1}\mid \boldsymbol{s}_t, \boldsymbol{a}_t) - \log p_{source}(\boldsymbol{s}_{t + 1}\mid \boldsymbol{s}_t, \boldsymbol{a}_t).\\$ 我们有很多做法来避免训练一个 dynamic model, 一个可行的做法是使用 discriminator, 这里会使用 $2$ 个 discriminator 来 estimate 条件概率, $\begin{aligned} \Delta r(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}) &= \log p_{target}(\boldsymbol{s}_{t + 1}\mid \boldsymbol{s}_t, \boldsymbol{a}_t) - \log p_{source}(\boldsymbol{s}_{t + 1}\mid \boldsymbol{s}_t, \boldsymbol{a}_t)\\ &= \log \frac{p_{target}(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1})}{p_{target}(\boldsymbol{s}_t, \boldsymbol{a}_t)} - \log \frac{p_{source}(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1})}{p_{source}(\boldsymbol{s}_t, \boldsymbol{a}_t)}\\ &= \log \frac{p_{target}(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1})}{p_{source}(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1})} - \log \frac{p_{target}(\boldsymbol{s}_t, \boldsymbol{a}_t)}{p_{source}(\boldsymbol{s}_t, \boldsymbol{a}_t)}\\ &= \log p(target\mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}) - \log p(source\mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1})\\  &\quad\quad- \log p(target\mid \boldsymbol{s}_t, \boldsymbol{a}_t) + \log p(source\mid \boldsymbol{s}_t, \boldsymbol{a}_t), \end{aligned}\\$ 其中最后利用了 optimal classifier 的 $p(target\mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}) = \frac{p_{target}(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1})}{p_{target}(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}) + p_{source}(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1})}.\\$

![](https://pic3.zhimg.com/v2-c759c3407461690bcdef1fafb59fc458_1440w.jpg)

Off-Dynamics Reinforcement Learning: Training for Transfer with Domain Classifiers

上述做法等价于我们**在两个 domain 的 intersection 中进行学习**. 但相应的问题是我们没有有效处理那些在 source domain 中做不到, 但是在 target domain 中可以做到的事情.

参见: Eysenbach et al., “Off-Dynamics Reinforcement Learning: Training for Transfer with Domain Classifiers”

### 2.4 What if we can also finetune

如果我们还能在 target domain 中进行 finetune, 也存在一些 RL 中的问题使得其相对 supervised learning 更加困难:

1.  RL tasks 通常不那么 diverse, 在 CV 和 NLP 中的 pretraining 时我们通常会在一个非常广的情境进行训练, 例如很广泛的图片, 然后在较小的领域中 finetune. 但是在 RL 中通常我们 pretraining 的领域也不够 diverse.  
    
2.  在 fully observed MDPs 中训练得到的 optimal policy 通常是 deterministic 的, 但是在 finetuning 中我们可能需要 exploration, 这样的的 low-entropy policy 通常 adapt 到新 setting 的非常缓慢.  
    

因此通过 naive 的方法进行 pretraining 和 finetuning 可能并不合适, 我们需要一些特定的方式来进行 pretraining, 以保证我们 pretraining 得到的 policy 具有足够的随机性: 这里的方法可以参加 **Exploration** 一节的第二部分, 以及 **Control as inference** 一节中的 **MaxEnt RL** 方法. 这两种方式都可以作为 pretraining 的好方法, 来使得终止时的 policy 更加 stochastic.

### 2.5 Maximize forward transfer

我们如何让我们的 forward transfer 尽可能有效呢?

**Basic intuition:** 我们的 training domain 越广泛, 我们就越有可能通过 zero shot 泛化到略有不同的领域上:

一个基本的做法是 **"randomization"** (对于 dynamics/ appearance 等). 这样的做法被广泛地应用在 **sim2real** 的 transfer 中, 例如通过调整环境的参数使其更加广泛, 以覆盖真实环境的 dynamic.

**Example 3**. _一个相对近期的 paper 是 EPOpt, 在 training 中使用一系列不同的 parameter, 例如 hopper 的质量. 当我们仅使用单个质量进行训练时, 则测试质量仅在很小范围时表现较好. 但如果我们在训练中使用了多个质量 (实际上使用的是一个正态分布的质量), 则我们可以在更广泛的质量上表现较好._

![](https://pic4.zhimg.com/v2-8c4f663218e9022ff7bfa3ffca7537ab_1440w.jpg)

EPOpt

尽管在一定意义上, 这样我们可能会牺牲一定的 optimality 来换取 generalization, 但是对于深度神经网络来说在多个 setting 上保持 optimality 并不是不可能的.

事实上, 如果我们有 target domain 中的一些 experience, 我们可以逐步缩小 distribution of parameter, 使其与真实情况更加接近.

## 3 Multi-task transfer learning

通过学习 multimple tasks, 我们通常学习地更快, 也取到更好的效果, 因为我们可以得到一些共享的 representation. 简单来说, 这样的做法可以加快 learning 的过程, 也可以为 down-stream 任务提供更好的 pretraining.

实质上, 在 RL 的 setting 下进行 multi-task transfer learning 只需要修改我们的我们原先的 MDP 为 **joint MDP** 上的 single-task RL, 主要的做法有以下几种:

### 3.1 Mixing initial states

回顾单个任务时我们会从 $p(\boldsymbol{s}_1)$ 采样一个 $\boldsymbol{s}_1$, 而对多任务, 我们只需要修改 $p(\boldsymbol{s}_1)$ 的分布为多个任务的初始 states 的分布的加权平均 (相当于按照概率选择 MDP, 再根据对应 MDP 选择初始状态).

但是上述做法虽然对于 Atari games 是合理的, 因为不同游戏开始时的 states 从 image 来看是不同的. 但是对于 robot 来说可能不同任务的初始状态都是一样的.

### 3.2 Different task contexts

针对上述问题, 我们可以设置不同任务的 context, 例如一个 one-hot vector, goal image 或者 textual description.

我们称作这样的 policy 为 contextual policy: $\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s}, \omega)$. 相当于修改 state space 和 state $\tilde{\mathcal{S}} = \mathcal{S} \times \Omega, \quad \tilde{\boldsymbol{s}} = \begin{bmatrix} \boldsymbol{s}\\ \omega \end{bmatrix}.\\$ 这实际和修改初始分布的方式在某种意义上是一致的. 我们并不需要修改我们的 RL 算法, 只需要修改我们的 state space 和 state.

![](https://pic1.zhimg.com/v2-14df6400e3870f30348508d27574153e_1440w.jpg)

不同的 context

### 3.3 Goal-conditioned policies

另一个常见的做法是使用 goal-conditioned policies, 考虑 $\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s}, \boldsymbol{g})$, 我们会用 reward $r(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{g}) = \delta(\boldsymbol{s} = \boldsymbol{g})$ 或 $r(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{g}) = \delta(\|\boldsymbol{s} - \boldsymbol{g}\| \leq \epsilon)$ 来定义 reward.

这样的做法比较 convenient, 因为我们不需要认为设计各个 task 的 reward. 同时也能 zero-shot 来 transfer 到其他的 goal.

但是训练这样的 goal-conditioned policy 可能并不容易, 而且并非所有 tasks 都等同于一个 goal reaching, 例如一个 task 可能是避免经过某一区域到达目的地, 但是 goal 本身无法表示这一限制.

这需要一些比较好的技巧, 例如选择训练的 goals, 表示 value function, formulation rewards 和 loss function 等等, 具体可以参考:

-   Kaelbling. Learning to achieve goals.  
    
-   Schaul et al. Universal value function approximators.  
    
-   Andrychowicz et al. Hindsight experience replay.  
    
-   Eysenbach et al. C-learning: Learning to achieve goals via recursive classification  
    

## 4 Meta-Learning

### 4.1 Introduction

简单来说, **meta-learning** 是 **learning to learn** 的学习方式, 在某种程度上可以理解为是对 multi-task learning 在逻辑上的拓展.

meta-learning 有多种实现方式:

-   学习一个 optimizer  
    
-   学习一个包含了过去 experience 的 RNN  
    
-   学习一个 representation  
    

![](https://pic2.zhimg.com/v2-cc8295c9fad58aacb3cb8be1dffa7d25_1440w.jpg)

learn an optimizer

**Why is meta-learning a good idea?**

通常 Deep RL 尤其是 model-free 方法需要大量 samples, 如果我们能够得到一个更快的 RL learner, 那么我们就能更加高效地学习. 如果我们有了一个 meta-learner, 我们就能够

-   更加智能地 explore  
    
-   避免尝试那些已经知道无用的 actions  
    
-   更容易也更快获得正确的 feature representation  
    

### 4.2 Formulation

我们接下来以 image recognition 为例, 介绍一下什么是 meta-learning.

在常规的监督学习中, 我们会有一个 training set 和 test set, 我们在训练集上利用一个模型 $f(x) \to y$, 接受输入 (图片) $x$, 输出 (label) $y$, 其中模型的参数 $\theta$ 通过 $\theta^\ast = \arg\min_\theta \mathcal{L}(\theta, \mathcal{D}_{tr})\\$ 来获取. 这里的 $\mathcal{L}$ 是一个 generic loss function, 例如关于 $f_\theta(x)$ 和 $y$ 的 cross-entropy loss.

而在 meta-learning 中, 我们会有**一系列 task**, 这些 tasks 被进一步分为 **meta-training set** 和 **meta-test set**:

**Definition 5**. _meta-training set_

_meta-training set 中包含多个 tasks, 每个任务有 support set (training set) 和 query set (test set)._

**Definition 6**. _meta-test set_

_meta-test set 中包含多个不同于 meta-training set 中的 tasks, 每个任务有 support set (training set) 和 query set (test set), 用于评估 meta-learning 的效果._

![](https://picx.zhimg.com/v2-d3c5f0a4351bc293186e7819d6db7647_1440w.jpg)

meta-training set and meta-test set

我们在 meta-learning 中学习的是一个从 support/training set 到 **模型 (参数)** 的映射, 具体来说, 我们会学习一个 $f: \mathcal{D}_{train} \mapsto (x \mapsto y),\\$ 如果用 $\theta$ 来表示这个 meta-learning 中学习的 $f$ 的参数, 并且使用 $\phi_i$ 表示 $f_\theta$ 将训练集 $\mathcal{D}_{train}^i$ 映射到的模型的参数. 我们可以写出 general 的 meta-learning 学习的参数 $\theta$ 为: $\theta^\ast = \arg\min_\theta \sum_{i = 1}^n \mathcal{L}(\phi_i, \mathcal{D}_i^{test}),\\$ 这一过程称为 **meta-training**, 其中 $\phi_i = f_\theta(\mathcal{D}_i^{train})$, 获取 $\phi_i$ 的过程称为 **adaptation**. 直观来说也就是学习一种学习方式, 使得在 meta-training set 的任务上应用这种方式得到的参数的平均 loss 最小.

**Example 4**. _基于 RNN 的 meta-learner_

_这里我们考虑 meta-learning 的 RNN 实现, 我们最终的训练得到的 $f$ 的参数 $\theta$ 就是 RNN 的参数 (以及可能存在的 $\theta_p$, 后续提到). 而在**每个 task** 中, 我们会使用一个**新的 hidden state**, 依次读入 support set 中的每一个元素, 并更新 hidden state, 于是我们每个 task 中学到 $x \mapsto y$ 的参数 $\phi_i$ 可以表示为 $\phi_i = \begin{bmatrix} h_i & \theta_p \end{bmatrix}.\\$_

_读入 support set 的方式有多种选择, 例如 RNN, transformer 等, 对于不同的模型结构, 我们得到的 $\phi_i$ 可能会有不同的形式._

![](https://pic1.zhimg.com/v2-9fc9abf32aaaeea42b64f90ae05a271a_1440w.jpg)

基于 RNN 的 meta-learner

## 5 Meta Reinforcement Learning

### 5.1 Basic idea

在我们介绍的 regular RL 中, 我们的学习的参数是 $\theta^\ast = \arg\max_\theta \mathbb{E}_{\pi_\theta(\tau)}\left[r(\tau)\right],\\$ 而在 meta RL 则是 $\theta^\ast = \arg\max_\theta \sum_{i = 1}^n \mathbb{E}_{\pi_{\phi_i}(\tau)}\left[r(\tau)\right],\\$ 其中 $\phi_i = f_\theta(\mathcal{M}_i)$, 这里 $\mathcal{M}_i$ 是一个 MDP $\{\mathcal{S}, \mathcal{A}, \mathcal{P}, r\}$.

类似于 supervised meta-learning 的方式, 我们会有一系列 meta-training MDPs $\mathcal{M}_i$, 假设它们 $\mathcal{M}_i \sim p(\mathcal{M})$.

-   在 meta-training time, 我们会学习 $f_\theta$.  
    
-   在 meta-test time, 我们采样 $\mathcal{M}_{test} \sim p(\mathcal{M})$, 我们可以得到 $\phi_i = f_\theta(\mathcal{M}_{test})$, 作为结果, 并且利用 $\phi_i$ 评估 meta-learning 的效果.  
    

![](https://picx.zhimg.com/v2-15dedd7aa11d08c85b6e69dff9abd6d7_1440w.jpg)

some examples

### 5.2 Relation to contextual policy

meta-learning 与 contextual policy 有很紧密的联系, 可以认为 meta-learning 相当于是让 contextual policy 的 $\omega$ 等条件来源于 $\mathcal{M}_i$ 中通过 $f_\theta$ 推断得来, 而不是人为指定.

### 5.3 Basic algorithm idea

值得注意的是, 和 supervised meta-learning 不同的是, 我们的 $f_\theta$ 接收的不是一个 training set, 而是一个 MDP, 我们的数据需要自己与环境交互得到!

因此整个训练过程可以视作重复以下过程:

1.  采样 task id $i$, 收集 task $i$ 的数据 $\mathcal{D}_i$ (收集 support set)  
    
2.  通过 $\phi_i = f_\theta(\mathcal{M}_i)$ 得到 adapt 的 policy $\phi_i$  
    
3.  利用 adapted policy $\pi_{\phi_i}$ 收集数据 $\mathcal{D}_i'$ (收集 query set)  
    
4.  利用 $\mathcal{L}(D_i', \phi_i)$ 更新 $\theta$  
    

上述的 $1-3$ 步对应于 **adaptation**, 而第 $4$ 步对应于 **meta-training**.

在这基础上有很多相当 intuitive 的改进方式:

-   在第 $4$ 步前进行多轮的 adaptation steps  
    
-   在第 $4$ 步更新 $\theta$ 时使用多个 tasks 进行更新  
    

在接下来我们会介绍几种关于 $f$ 与 $\mathcal{L}$ 的具体实现.

## 6 Meta-RL with recurrent policy

在这里我们考虑基于 RNN 等 recurrent network 的 meta-learner.

### 6.1 Basic Ideas

我们考虑一个读入**所有 past experience** RNN, 这些 experience $(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)$ 可能来源于不同的 episode, 我们称它们属于同一个 **meta-episode**. 在将他们全部读入后, 我们得到了一个 hidden state $h_i$, 之后我们可以考虑一个输入 $h_i$ 与 当前 state 并输出 action 的模型作为我们的 policy. 这里类似地有 $\phi_i = \begin{bmatrix} h_i & \theta_p \end{bmatrix}.\\$

这看起来好像就在训练一个 RNN policy, 这里核心的区别在于 RNN hidden states 在不同 episode 间不会被清除, 这是我们的 recurrent policy 能够学会 explore 的关键:

**Example 5**. _由于此时的 policy 是关于 $h_i$ 的, 此时的 meta-episode 中包含了过去多个 episode 的信息, 如果在连续几个 episode 采取的行为中我们都没有得到 reward, 下一个 epoch 时, 我们的 policy 考虑的就不是 "在当前 state 下应该采取什么 action", 而是 "我知道在当前 state 我过去已经尝试过这些 actions 了, 结果并不理想, 那么我应该采取什么 action"._

![](https://pic3.zhimg.com/v2-590a291f8953787bec66d432c9ad5e60_1440w.jpg)

meta-RL with recurrent policy

_事实上, 当我们 **给 policy** 这种看到多个 episode 的机会时, exploration 问题就转化为了解决这种 high level 的 MDP._

在 regular RL 中我们在 $\theta = \arg\max_\theta \mathbb{E}_{\pi_\theta(\tau)}\left[\sum_{t = 1}^T r(\boldsymbol{s}_t, \boldsymbol{a}_t)\right],\\$ 目前有了更多关于 meta RL 的方法, 但它们 high level 的 idea 都是给与 policy 这样一种多 episode 的 experience.

### 6.2 Algorithm and design choices

我们可以得出一个基于 recurrent policy 的 basic meta-RL algorithm:

1.  对于每个 task $i$, 初始化 hidden state  
    
2.  对于每一个时间步 (RNN 的时间步) $t$:  
    

3.  利用当前由 $h_t$ 决定的 policy 采取一个 action 更新当前任务的 dataset $\mathcal{D}_i = \mathcal{D}_i \cup \{(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}, r_t)\}\\$
4.  通过 $\mathcal{D}_i$ 更新 hidden state: $h_{t + 1} = f_\theta(h_t, \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}, r_t)\\$

5.  利用 $\theta \gets \theta - \alpha \nabla_\theta \sum_i \mathcal{L}_i(\phi_i, \mathcal{D}_i^{test})$ 更新 $\theta$, 这里 $\mathcal{D}_i^{test}$ 可以通过最终的 $\phi_i$ 采样得到.  
    

这一类基于 recurrent policy 的方法中有很多种 architectures 选择:

-   **standard RNN/LSTM**: 参见 Duan, Schulman, Chen, Bartlett, Sutskever, Abbeel. RL2: Fast Reinforcement Learning via Slow Reinforcement Learning. 2016.  
    
-   **attention + temporal convolution**: 参见 Mishra, Rohaninejad, Chen, Abbeel. A Simple Neural Attentive Meta-Learner.  
    
-   **parallel permutation-invariant context encoder**: 参见 Rakelly\*, Zhou\*, Quillen, Finn, Levine. Efficient Off-Policy Meta-Reinforcement learning via Probabilistic Context Variables.

![](https://picx.zhimg.com/v2-0d9020e022adc9d868b21a195a83f767_1440w.jpg)

examples on architectures

### 6.3 Reference

还有更多关于基于 recurrent policy 的 meta-RL 实例, 参见:

-   Heess, Hunt, Lillicrap, Silver. Memory-based control with recurrent neural networks. 2015  
    
-   Wang, Kurth-Nelson, Tirumala, Soyer, Leibo, Munos, Blundell, Kumaran, Botvinick. Learning to Reinforcement Learning. 2016  
    
-   Duan, Schulman, Chen, Bartlett, Sutskever, Abbeel. RL2: Fast Reinforcement Learning via Slow Reinforcement Learning. 2016  
    

![](https://picx.zhimg.com/v2-efede100d8f51e5ec4df3dea79a609a7_1440w.jpg)

some examples

## 7 Gradient-Based Meta-Learning

### 7.1 Basic Ideas

回顾 pretraining + finetuning scheme, 这也可视作某种 meta-learning: 对于很多 CV 任务来说, 我们在 pretraining 中学会如何提取特征, 使得**模型参数在参数空间中处于一个很好的位置**, 从这个位置上出发, 只需要少量的 gradient step 就可以得到特定任务上的模型.

我们能否借鉴这种 scheme 来得到一个更好的 meta learning 算法呢?

回顾我们的 meta RL 的目标是 $\theta^\ast = \arg\max_\theta \sum_{i = 1}^n \mathbb{E}_{\pi_{\phi_i}(\tau)}\left[R(\tau)\right],\\$ 其中 $\phi_i = f_\theta(\mathcal{M}_i)$.

我们考虑把 $f_\theta$ 建模为一个 **RL algorithm** 而不是一个 RNN, 具体来说, $f_\theta$ 可以视作一个从 $\theta$ 出发**进行一个 gradient step** 的算法: $\phi_i = f_\theta(\mathcal{M}_i) = \theta + \alpha \nabla_\theta J_i(\theta),\\$ 这里 $J_i(\theta)$ 是在 $\mathcal{M}_i$ 上的某种 objective. 此时我们 meta-learning 的目标就是找到一个 $\theta$ 使其在所有的 task $\mathcal{M}_i$ 上进行一个 gradient step 后能够得到平均最大的 reward, 这样的方式对应于 **MAML (model-agnostic meta-learning)**

### 7.2 MAML (Model-Agnostic Meta-Learning)

在 MAML 中, 我们的目标是找到一个能够最大化 $\sum_i J_i\left[\theta + \alpha \nabla_\theta J_i(\theta)\right]\\$ 的 $\theta$, 这里的 $J_i\left[\theta + \alpha \nabla_\theta J_i(\theta)\right]$ 可以是在 $\mathcal{M}_i$ 上进行一步更新后的参数在 query set 上的 loss, 写出梯度更新式为 $\theta \gets \theta + \beta \sum_{i} \nabla_\theta J_i\left[\theta + \alpha \nabla_\theta J_i(\theta)\right].\\$ 这在某种意义上有一种 "second-order" 的感觉, 不过对于自动微分的深度学习框架来说实现起来并不困难. 这样的算法相当于是在参数空间中找到一个 $\theta$ 使得从这个位置出发很容易到达各个 task 的 optimal.

![](https://pic2.zhimg.com/v2-7aaa0eef7026caf75f16b31dabb6be45_1440w.jpg)

MAML (Model-Agnostic Meta-Learning)

### Brief summary

简单总结一下我们刚才介绍的 MAML 算法和前面的一些通用框架:

-   supervised learning: $f(x) \to y$  
    
-   supervised meta-learning: $f(\mathcal{D}_{train})(x) \to y$  
    
-   model-agnotic meta-learning: $f_{MAML}(\mathcal{D}_{train})(x) \to y$, 这里 $f_{MAML}(\mathcal{D}_{train}) = f_{\theta'}(x)$, 其中 $\theta' = \theta - \alpha \sum_{(x,y) \in \mathcal{D}_{train}} \nabla_\theta \mathcal{L}(f_\theta(x), y).\\$

MAML 的方式相较于基于 recurrent policy 的方式有一个明显的好处: 在 MAML 中我们可以在 meta-test time 选择**实际进行更多的 gradient steps**, 在得到的 $\phi_i$ 基础上进一步微调, 这是更加 "flexible" 的. 而 recurrent policy 的方式得到的策略已经由我们的 RNN 结构和 $\theta$ 定死了.

这样的做法是非常合理的, 对于距离 source task 分布较远的 target task, 我们自然需要更多的 gradient steps 来进行微调.

### 7.3 Reference

-   MAML meta-policy gradient estimators:  
    

-   Finn, Abbeel, Levine. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.  
    
-   Foerster, Farquhar, Al-Shedivat, Rocktaschel, Xing, Whiteson. DiCE: The Infinitely Differentiable Monte Carlo Estimator.  
    
-   Rothfuss, Lee, Clavera, Asfour, Abbeel. ProMP: Proximal Meta-Policy Search.  
    

-   Improving exploration:  
    

-   Gupta, Mendonca, Liu, Abbeel, Levine. Meta-Reinforcement Learning of Structured Exploration Strategies.  
    
-   Stadie\*, Yang\*, Houthooft, Chen, Duan, Wu, Abbeel, Sutskever. Some Considerations on Learning to Explore via Meta-Reinforcement Learning.  
    

-   Hybrid algorithms (not necessarily gradient-based):  
    

-   Houthooft, Chen, Isola, Stadie, Wolski, Ho, Abbeel. Evolved Policy Gradients.  
    
-   Fernando, Sygnowski, Osindero, Wang, Schaul, Teplyashin, Sprechmann, Pirtzel, Rusu. Meta Learning by the Baldwin Effect.  
    

## 8 Meta-RL as a POMDP

### 8.1 Formalization

我们可以将一个 **meta-learning 的任务**建立为一个 POMDP, 原先的 MDP 可以表示为 $\mathcal{M} = \{\mathcal{S}, \mathcal{A}, \mathcal{P}, r\}$, 这里考虑将其建立为 $\tilde{M} = \{\tilde{\mathcal{S}}, \mathcal{A}, \tilde{\mathcal{P}}, r, \tilde{\mathcal{O}}, \mathcal{E}\}$, 其中

-   $\tilde{\mathcal{S}} = \mathcal{S} \times \mathcal{Z}$, $\tilde{\boldsymbol{s}} = (\boldsymbol{s},\boldsymbol{z})$, $\boldsymbol{z}$ 包含了**解决当前 task 所需的全部信息**, 也就是说真正的 state space 还包含了 task 的信息.  
    
-   $\tilde{\mathcal{O}} = \mathcal{S}$, 原先的 state space 变为了当前的 observation space, 其中 $\tilde{\boldsymbol{o}} = \boldsymbol{s}$.  
    

基于我们上述的定义, 学会一个 task 意味着我们能够通过交互的数据推断出其对应的 $\boldsymbol{z}$, 与此同时, 还有知道如何利用这些信息采取 actions. 这两部分**都是我们 meta-learner 的组成部分**:

-   推断 $\boldsymbol{z}$: 我们会学习一个 **inference network** 来近似 $p(\boldsymbol{z} \mid \boldsymbol{s}_{1:i}, \boldsymbol{a}_{1:i}, r_{1:i})$, 也就是在当前的 observation 下, 我们对 $\boldsymbol{z}$ 的后验分布. 这像是一个任务识别器.  
    
-   依据 $\boldsymbol{z}$ 采取 action: 我们会学习一个 policy $\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s}, \boldsymbol{z})$, 也就是在当前的 observation 和 task 的信息下, 我们如何采取 action. 这像是一个根据任务做调整的执行器.  
    

**Remark:** 上述可能让人困惑的地方在于, 这里的 $\theta$ 与通常决定我们如何采取 action 的参数 $\theta$ 并不一样. 这里的 $\theta$ 是 meta-learner 参数的一部分, 而 $\boldsymbol{z}$ 更像是一般 RL 中的参数. 这一种做法在概念上和 RNN meta-RL 很接近: 在 RNN meta-RL 中, 我们的 $\phi$ 像是一般 RL 中 policy 的参数, 只是这一类方法中我们使用的是随机的 $\boldsymbol{z}$ 而不是 $\phi$.

### 8.2 posterior sampling

由于通过 $p(\boldsymbol{z} \mid \boldsymbol{s}_{1:i}, \boldsymbol{a}_{1:i}, r_{1:i})$ 进行完全的贝叶斯推断是 intractable 的, 这里会使用在 **exploration** 中讨论的 **posterior sampling** 方法, 也就是采样一个 $\boldsymbol{z}$, 假设这个单点分布就是真实的后验分布, 并依据这个 $\boldsymbol{z}$ 采取 action.

**Intuition:**

-   刚开始学习一系列任务时, 由于对其并不了解, 因此 $p(\boldsymbol{z} \mid \boldsymbol{s}_{1:i}, \boldsymbol{a}_{1:i}, r_{1:i})$ 会非常均匀, 也就是我们会随机选择一种"行动方式", 或者说随机完成一个"任务". 当然由于此时 policy 也很弱, 因此完成的"任务"也很怪异.  
    
-   随着训练不断进行, posterior 越来越能够从 context 中提取任务的信息, 给出更加确定的 $\boldsymbol{z}$, 而 policy 也能够在这些 $\boldsymbol{z}$ 上表现越来越好.  
    
-   当我们对任务有足够了解后, 给定 $\boldsymbol{s}_{1:i}, \boldsymbol{a}_{1:i}, r_{1:i}$, 相当于指定了一个任务, 我们会有一个更好的后验分布, 在这个后验分布采样的 $\boldsymbol{z}$ 配合上我们训练好的 $\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s},\boldsymbol{z})$ 可以能将这个任务完成的很好.  
    

### 8.3 Basic algorithm idea

具体来说, 我们的算法流程如下:

1.  从 $\hat{p}(\boldsymbol{z} \mid \boldsymbol{s}_{1:i}, \boldsymbol{a}_{1:i}, r_{1:i})$ 中采样 $\boldsymbol{z}$ (通过 variational inference 来估计这个 posterior)  
    
2.  依据 $\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t, \boldsymbol{z})$ 采取 action 来收集更多数据, 这些数据称作 context, 记作 $\boldsymbol{c}_{1:i} = \{\boldsymbol{s}_{1:i}, \boldsymbol{a}_{1:i}, \boldsymbol{s}'_{1:i} r_{1:i}\}$.  
    
3.  重复上述过程若干次后, 我们再按照上述思路收集数据, 然后通过数据和 context, 利用损失函数更新 meta-learner $\theta$ 以及估计的 posterior.  
    

这里的前两步可以视作是 adaptation, 而最后一步则是 meta-training.

**Note:** 这并不是我们 optimal 的 choice, 但是在某种意义上已经相当好了. 潜在的 suboptimality 在于我们开始时相较于按照一个随机的 task $\boldsymbol{z}$ 进行 explore, 尝试多个 task 直到找到最优区域, 似乎并不如开始时不考虑非要完成某个 task, 直接在一个 episode 中探索那些 reward 可能更高的区域.

![](https://pic2.zhimg.com/v2-7fc27d4f67a7a67765c3bd8820e00ebf_1440w.jpg)

潜在的 suboptimality

### 8.4 Variational inference for meta-RL

我们还没有介绍如何得到这样一个 posterior $p(\boldsymbol{z} \mid \boldsymbol{s}_{1:i}, \boldsymbol{a}_{1:i}, r_{1:i})$, 这里会使用 variational inference, 一个具体的例子是如下的 **PEARL: Probabilistic Embeddings for Actor-Critic Reinforcement Learning**:

首先这里训练 policy 的算法是 **SAC**, 我们会有 policy 和 Q function $\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s}, \boldsymbol{z})$, $Q_\theta(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{z})$, 同时会通过 **inference network**: $q_\phi(\boldsymbol{z} \mid \boldsymbol{s}_{1:i}, \boldsymbol{a}_{1:i}, r_{1:i})$ 来近似 posterior. 它们都是 meta-learner 的一部分!

**Note:** 这里的 $\phi$ 是 inference network 的参数, 也是 meta-learner 的一部分. 需要和我们之前讨论的 RNN meta-learner 中的 $\phi$ 以及 general form meta RL 的 $\phi$ 区分开来!

由于我们实质上构建了一个关于 $\boldsymbol{c}_{1:i}$ 和 $\boldsymbol{z}$ 的 latent variable model, 为了这个 model 能够建模好 context 数据, 我们需要最大化这些 context 的对数似然 $\log p(\boldsymbol{c}_{1:i})$.

利用 variational inference, 我们有 $\log p(\boldsymbol{c}_{1:i}) \geq \mathbb{E}_{z \sim q_\phi(\boldsymbol{z} \mid \boldsymbol{c}_{1:i})} \left[\log p_\theta(\boldsymbol{c}_{1:i} \mid \boldsymbol{z}) - D_{KL}(q_\phi(\boldsymbol{z} \mid \boldsymbol{c}_{1:i}) \parallel p(\boldsymbol{z}))\right],\\$ 这里 $\log p(\boldsymbol{c}_{1:i} \mid \boldsymbol{z})$ 可以理解为给定任务信息后, 我们生成 context 的 likelihood, 其中包含了 dynamic 以及 policy 相关的部分, 这里我们不显式建模它, 而是将其转化为某种 reward 的度量, 这里可以利用一些角度反映其联系 (笔者自己想的, 不一定非常严谨):

_Proof._ 由于在 **PEARL 中, 我们假设不同 task 的 dynamic 相同**, 因而 ELBO 可以进一步把 dynamic 提到外面去, 由于其和 $\theta, \phi$ 都没有关系, 在求梯度时会消失, 不妨就写为 $\mathbb{E}_{z \sim q_\phi(\boldsymbol{z} \mid \boldsymbol{c}_{1:i})} \left[\sum_{j = 1}^i \log \pi_\theta(\boldsymbol{a}_i \mid \boldsymbol{s}_i, \boldsymbol{z}) - D_{KL}(q_\phi(\boldsymbol{z} \mid \boldsymbol{c}_{1:i}) \parallel p(\boldsymbol{z}))\right],\\$ 这里用来训练 policy 的算法是 **SAC**, 可以参见我们在 **Reframing control as inference problem** 的讨论, 在那里我们有 $\pi(\boldsymbol{a} \mid \boldsymbol{s}) = \exp(Q(\boldsymbol{s},\boldsymbol{a}) - V(\boldsymbol{s}))$, 其中 $V(\boldsymbol{s}) = \log \int \exp(Q(\boldsymbol{s}, \boldsymbol{a}))\text{d}\boldsymbol{a}$, 类似地这里我们有 $\pi(\boldsymbol{a} \mid \boldsymbol{s}, \boldsymbol{z}) \propto Q(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{z}),\\$ 于是 $\pi_\theta(\boldsymbol{a}_i \mid \boldsymbol{s}_i, \boldsymbol{z})$ 可以进一步化为某种 return 的度量. ◻

经过 pratical 的调整后, 对于每一个 task $i$, 我们得到实际的 objective: $(\theta,\phi) = \arg\max_{\theta,\phi} \frac{1}{N} \sum_{i = 1}^N \mathbb{E}_{\boldsymbol{z} \sim q_\phi, b^i \sim \mathcal{B^i}} \left[R_i(b^i) - D_{KL}(q_\phi(\boldsymbol{z} \mid \boldsymbol{c}_{1:K}^i) \parallel p(\boldsymbol{z}))\right].\\$ 这里的 $R_i(b^i)$ 是一种关于 return 的度量, $b^i \sim \mathcal{B^i}$ 是 replay buffer 中的一个 batch. 后一项 KL divergence 让我们的 policy 保持接近于 prior (具体来说使用标准正态分布), 其中的 $\boldsymbol{c}_{1:K}^i$ 是一组从 buffer 中采样的 context (为了更好的 exploration, context 可能比更新用的 batch 要小).

另外, 文中使用一系列 sample-wise $\Psi_\phi(\boldsymbol{z} \mid \boldsymbol{c}^i_j)$ (通过预测 mean 和 variance), 将其平均来得到采样的全部 context 的 $q_\phi(\boldsymbol{z} \mid \boldsymbol{c}^i_{1:K})$.

![](https://picx.zhimg.com/v2-953207a8ef401589bf6382582d5c1619_1440w.jpg)

design choice on inference network

参见: Rakelly\*, Zhou\*, Quillen, Finn, Levine. Efficient Off-Policy Meta-Reinforcement learning via Probabilistic Context Variables. ICML 2019.

### Side Note:

我们在之前的内容中介绍过多种处理 POMDP 的方法, 其中一种对应于 policy with memory, 这实质上对应于 meta-learning with RNN. 而我们知道在处理 POMDP 的方法中存在 explicit state estimation 这类的方法, 例如构建一个 state space model, 实际上这也可以导出一类 meta-learning 的算法.

### 8.5 Reference

-   Rakelly\*, Zhou\*, Quillen, Finn, Levine. Efficient Off-Policy Meta-Reinforcement learning via Probabilistic Context Variables. ICML 2019  
    
-   Zintgraf, Igl, Shiarlis, Mahajan, Hofmann, Whiteson. Variational Task Embeddings for Fast Adaptation in Deep Reinforcement Learning.  
    
-   Humplik, Galashov, Hasenclever, Ortega, Teh, Heess. Meta reinforcement learning as task inference.  
    

## 9 Summary

### 9.1 The three perspective on meta-RL

最后我们将介绍三种不同的 meta-RL 的视角, 它们对应于我们之前介绍的三种方法:

### Perspective 1:

训练一个 RNN, 将 $f_\theta(\mathcal{M}_i)$ 当作一个 black box.

-   在概念上很简单  
    
-   相对容易应用  
    
-   容易引发 meta-overfitting: 如果 meta-test 的 task 稍微偏离了分布, 那么我们可能无法得到较好的表现, 而且由于 forward pass 已经是确定的了, 没有办法调整.  
    
-   在现实中不容易优化, 尽管目前的 transformer 改善了这一点.  
    

### Perspective 2:

将 $f_\theta(\mathcal{M}_i)$ 当作一个 RL algorithm, 通过 MAML 来学习.

-   good extrapolation, 当我们的任务略微偏离时, 我们可以通过类似于多个 gradient steps 来得到一个较好的表现.  
    
-   conceptually elegant  
    
-   通常较为复杂, 也需要很多 samples  
    
-   不容易扩展到 actor-critic 这类 temporal difference 的方法  
    

### Perspective 3:

将 $f_\theta(\mathcal{M}_i)$ 建模为一个 inference problem, 任务转化为 inference $z$.

-   可以通过 posterior sampling 来进行简单有效的 exploration  
    
-   elegantly 归约到了求解一个 POMDP  
    
-   同样容易 meta-overfitting  
    
-   在现实中也不容易优化  
    

但是这三种方式同样有很多共性:

-   inference 的方式我们 inference 的 $\boldsymbol{z}$ 就像是 RNN 中的 $\phi$, 只不过我们换成了一个随机变量.  
    
-   gradient based 方法也可以通过一些特定的网络结构转化为另外两种方法, 如果我们给 gradient 添加 noise, 那么就更像是一个 inference process.  
    

### 9.2 Meta-RL and emergent phenomena

在 RL 和 认知科学的交叉领域中, 我们会发现与 meta learning 有着很多相似的现象, 例如人类和动物学习的方式有多种方式:

1.  高效的 model-free RL  
    
2.  episodic recall  
    
3.  model-based RL  
    
4.  causal inference  
    

这些方法似乎都发生在人类学习的某些层面, 目前尚不明确为什么某些算法会在某些情况下被使用, 也许存在一个更高层次的 meta-learning 决定了在什么情况下使用什么样的算法, 进而产生了这类智能的现象. 目前的一些研究也发现了 meta-RL 引发了 episodic learning, causal inference 等方法.

![](https://pic4.zhimg.com/v2-ceb7cde9b2cebe6c0a6e3125090c4821_1440w.jpg)

Meta-RL and emergent phenomena

### 9.3 Summary of entire lecture

在本节中, 我们

-   从一些例子出发指出在很多任务中, 关于任务的知识可以有助于我们进行学习, 并引出了 transfer learning 和 meta-learning 的概念.  
    
-   在 transfer learning 部分, 我们主要介绍了两种 transfer 的方式: forward transfer 和 multi-task transfer.  
    

-   在 forward transfer 中, 我们介绍了将其在 RL 中应用的一些 issue, 从 [domain adaptation](https://zhida.zhihu.com/search?content_id=256210233&content_type=Article&match_order=1&q=domain+adaptation&zhida_source=entity), difference in MDP, finetuning 几个角度介绍了一些示例方法.  
    
-   在 multi-task transfer 中, 我们介绍了几种将 multi-task 问题转化为 joint MDP 的方法, 例如 mixing initial states, task contexts, goal-conditioned policies 等.  
    

-   在 meta-learning 部分, 我们首先介绍了其 formulation, 给出了其 general form 以及和通常的 learning problem 的区别, 接下来引出 meta-RL 的问题, 并介绍了几种方法.  
    

-   Meta-RL with recurrent policy, 这里我们介绍了一种基于 RNN 的 meta-learner.  
    
-   Meta-RL with gradient-based methods, 这里我们介绍了 MAML 的方法.  
    
-   Meta-RL as POMDP, 这里我们介绍了其 formulation, 并且主要关注了基于 posterior sampling 的方法, 例如 PEARL.  
    

-   最后, 我们对比了几种 meta-learning 的算法, 以及 meta-RL 和一些人类智能的现象之间的关系.