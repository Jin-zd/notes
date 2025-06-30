在本节中我们将介绍元学习和迁移学习的相关概念以及它们在强化学习中的应用。
## 1 Introduction
重新回顾我们介绍探索中提到的问题，在 Breakout（打砖块的那个游戏）中非常容易，但是在像 Montezuma's Revenge 这样的游戏中就非常困难，即使是游戏中最简单的一层。
![](20-1.png)

在这个游戏中，奖励的结构主要如下，这样的奖励结构并不能很好地给我们的算法提供指引。
- Getting key = reward  
- Opening door = reward  
- Getting killed by skull = bad  

与强化学习算法相反的是，事实上 Breakout 对人类来说反而更难，Montezauma's Revenge 反而很简单，这是因为我们有很多先验知识，这些对于问题结构的先验知识可以让我们很容易地解决这个问题：
- 基于物理常识，我们知道人会受重力影响往下掉；
- 基于电子游戏的知识，我们知道要躲避骷髅头，楼梯可以被攀登，钥匙可以开门，进入新的房间意味着我们推进了游戏.。

在我们玩 Montezauma's Revenge 的时候，实际在做的是一种迁移学习。我们之前介绍这个游戏的时候将其作为一个探索的例子，或许更好的做法是设计一个能够更好地迁移的算法。直观地来说，就是从解决过去的任务中获取关于新任务的知识。

目前介绍的强化学习算法通常并不能像人类一样利用先验知识，但如果从迁移学习的角度出发时, 我们可以设计一个合理的算法。

在迁移学习中要将所学知识存储在哪里？
- $Q$ 函数：能够告诉我们哪些动作和状态是好的； 
- 策略：可以告诉我们哪些动作潜在可能是好的，有些动作永远不是好的，因此至少能够学会排除它们；
- 模型：不同任务中的物理法则也许是一致的； 
- 图像/隐藏状态：可以告诉我们什么样的表征是好的,，例如通过之前的任务发现骷髅头是危险的。

迁移学习术语：
Definition 1. _transfer learning（迁移学习）_
迁移学习：使用过去一系列任务上的经验来实现新任务上更快的学习或者更好的表现。
在强化学习中，通常一个任务对应于一个马尔可夫决策过程。

Definition 2. _source domain, target domain（源领域，目标领域）_
- 源领域：获取经验的任务所属的领域；
- 目标领域：希望解决的任务所属的领域。

通常来说我们希望能够在目标领域上有很好地表现。当然对于终身学习的情况我们会希望在源领域有同样好的表现，但目前仅考虑在目标领域上的表现.

Definition 3. _"shot"_
在目标领域中需要的尝试次数，以下是一些常见的 "shot" 的定义：
- 0-shot:：只需要运行源领域中训练的策略就能得到很好的表现。
- 1-shot：只需要尝试这个任务一次，这在不同任务上可能有不同含义，例如 Montezuma's Revenge 这种电子游戏中可能是仅玩一个轮次，在机器人操作这种任务中可能就是进行一次操作。
- few-shot：尝试几次。

如何建模这些迁移学习的问题呢？事实上，这对不同领域的任务通常有较大的差异。通常来说，主要的方式有以下几类：
1. 正向迁移：（通常是在一个任务上）学习一个能有效迁移的策略：
	- 在源任务上训练，然后在目标任务上训练 / 微调。
	- 因为训练用的源任务只有一个，因此依赖于任务之间具有很大的相似性。
2. 多任务迁移：在多个任务上训练，然后迁移到新的任务：
	- 可以在多任务学习中共享表征和层。
	- 因为训练的任务有多个，它们形成了一个分布，新的任务只需要接近训练任务所属的分布（接近所属分布比接近单个任务更容易实现）。
3. 元学习：在许多任务上学习如何学习：
	- 在源领域上训练时我们就要考虑适应到新的任务上。

我们这节内容主要会将重心放在元学习上。不过我们依然会先介绍一些正向迁移和多任务迁移的方法（这些方法通常比较分散，因此可能只能涵盖其中的一部分重要的原则）。

## 2 Forward transfer learning
### 2.1 Pretraining + finetuning
在计算机视觉等领域中，预训练+微调是一个非常常见的做法，例如在 ImageNet 上进行预训练，然后在其他任务上进行微调，这属于一种正向迁移学习。
![](20-2.png)

这样的基本流程应用在强化学习中也是可行的，这是我们从监督学习中直接继承过来的方法，因此我们不会详细地介绍。简单来说，是在预训练后替换最后的若干层，固定部分层的参数（也可能不固定），然后在目标领域上进行微调。

在强化学习中，我们可以应用类似的方法，这样的方法可能遇到什么问题吗？
- 领域偏移：在源领域中学到的表征在目标领域中可能并不适用。可能存在一种差距，例如虚拟环境中的驾驶环境过于单一（这只是在视觉上不同，但背后的机制是一致的）； 
- 马尔可夫决策过程中的差异：源领域中的一些情况在目标领域中可能并不适用，例如二者可能有不同的物理特性； 
- 微调问题：在微调过程中可能依旧需要探索，但有可能预训练得到的最优策略已经是确定性的了。

接下来我们依次介绍如何处理这些问题。

### 2.2 Domain adaptation for domain shift
在计算机视觉中，一个解决领域偏移的方法是不变性假设。

直觉：任何在领域间不同的都是无关的。

例如：
- 例如现实世界中可能有雨，而模拟器中可能没有，那么不变性假设则告诉我们是否有雨和如何驾驶无关；
- 另一方面在现实世界和模拟器中驾驶地点在分布上是一致的，因此驾驶地点根据假设是有关的。

严格来说，不变性假设是说：

Definition 4. _invariance assumption（不变性假设）_
假设输入分布 $p(x)$ 在两个领域是不同的，不变性假设表示存在一种表征 $z = f(x)$ 使得
- 条件分布对齐：$p(y\mid z) = p(y\mid x)$，也就是 $z$ 中包含了预测 $y$ 所需的所有信息。
- 边缘分布对齐：在源领域和目标领域中，$p(z)$ 是一致的。 

这类领域偏移在计算机视觉中已经得到了广泛的研究，相关的方法主要领域混淆（Domain confusion）、领域对抗神经网络（Domain adversarial neural network ，DANN）。在这些方法中，我们需要目标领域中的少量图片。

这里以对抗域适应（Adversarial Domain Adaptation，ADA）为例，简单介绍一下其原理：
想法：在网络中添加一些层（通常在网络的卷积部分后），分别输入源领域和目标领域中的图片，计算它们在这些层中的激活值，使用一些特定的损失使得它们尽可能接近，也就是让它们的激活值的分布 $p(z)$ 一致。
具体实现上，我们会训练一个二元分类器 $D_\phi(z)$ 输入这些层的激活值，并让其判断是否属于目标领域。之后我们会求出 $D_\phi(z)$ 的梯度，并将其反向（使得结果更加不有区别性的），并反向传播回原网络。
![](20-3.png)


应用到强化学习：
在强化学习中使用这样的方法依旧需要有一些目标领域中的图像等数据，只不过不再需要在目标领域中重新利用强化学习等算法进行训练。

但是也有一些需要注意的，例如如果目标领域中的数据都是质量很差驾驶数据，那么要求网络中间的几层变得恒定，可能会影响我们得到一个好的策略。

### 2.3 Domain adaptation in RL for dynamic
如果动态并不一致，那么仅仅输出一个一致的表征可能并不足够，因为不能忽略这些动态的差异。

这里的一个做法是惩罚那些在源领域可以做到，但是在目标领域中做不到的行为。
例如在现实中到达目标中间可以有一堵墙，而在模拟器中没有，那么可以做的是修改奖励为：
$$
\tilde{r}(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \Delta r(\boldsymbol{s}, \boldsymbol{a})
$$
一个可能的做法是
$$
\Delta r(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}) = \log p_{target}(\boldsymbol{s}_{t + 1}\mid \boldsymbol{s}_t, \boldsymbol{a}_t) - \log p_{source}(\boldsymbol{s}_{t + 1}\mid \boldsymbol{s}_t, \boldsymbol{a}_t)
$$
我们有很多做法来避免训练一个动态模型，一个可行的做法是使用判别器，这里会使用 $2$ 个判别器来估计条件概率：
$$
\begin{aligned} \Delta r(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}) &= \log p_{target}(\boldsymbol{s}_{t + 1}\mid \boldsymbol{s}_t, \boldsymbol{a}_t) - \log p_{source}(\boldsymbol{s}_{t + 1}\mid \boldsymbol{s}_t, \boldsymbol{a}_t)\\ &= \log \frac{p_{target}(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1})}{p_{target}(\boldsymbol{s}_t, \boldsymbol{a}_t)} - \log \frac{p_{source}(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1})}{p_{source}(\boldsymbol{s}_t, \boldsymbol{a}_t)}\\ &= \log \frac{p_{target}(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1})}{p_{source}(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1})} - \log \frac{p_{target}(\boldsymbol{s}_t, \boldsymbol{a}_t)}{p_{source}(\boldsymbol{s}_t, \boldsymbol{a}_t)}\\ &= \log p(target\mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}) - \log p(source\mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1})\\  &\quad\quad- \log p(target\mid \boldsymbol{s}_t, \boldsymbol{a}_t) + \log p(source\mid \boldsymbol{s}_t, \boldsymbol{a}_t), \end{aligned}
$$
其中最后利用了最优分类器的
$$
p(target\mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}) = \frac{p_{target}(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1})}{p_{target}(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}) + p_{source}(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1})}
$$
![](20-4.png)

上述做法等价于在两个领域的交集中进行学习，但相应的问题是我们没有有效处理那些在源领域中做不到，但是在目标领域中可以做到的事情。

参见：Eysenbach et al., “Off-Dynamics Reinforcement Learning: Training for Transfer with Domain Classifiers”

### 2.4 What if we can also finetune
如果还能在目标领域中进行微调，也存在一些强化学习中的问题使得其相对监督学习更加困难：
1. 强化学习任务通常不那么多样化，在计算机视觉和自然语言处理的预训练时通常会在一个非常广的情境进行训练，例如很广泛的图片，然后在较小的领域中微调。但是在强化学习中通常我们预训练的领域也不够多样化。
2. 在完全可观测马尔可夫决策过程中训练得到的最优策略通常是确定性的，但是在微调中可能需要探索，这样的低熵策略通常适应到新场景的速度非常缓慢。

因此通过朴素的方法进行预训练和微调可能并不合适，我们需要一些特定的方式来进行预训练，以保证我们预训练得到的策略具有足够的随机性：这里的方法可以参考[[Lecture 12 Exploration 2]]，以及[[Lecture 17 Reframing Control as an Inference Problem]]一节中的最大熵强化学习方法。这两种方式都可以作为预训练的好方法，来使得终止时的策略更加随机。

### 2.5 Maximize forward transfer
我们如何让我们的前向迁移尽可能有效呢？
基本直觉：我们的训练域越广泛，我们就越有可能通过零样本泛化到略有不同的领域上。  

一个基本的做法是 “随机化”（对于动态 / 表现等）。这样的做法被广泛地应用在从模拟到现实的迁移中，例如通过调整环境的参数使其更加广泛，以覆盖真实环境的动态。  

例如，一篇相对近期的论文是 EPOpt，在训练中使用一系列不同的参数，例如跳跃者的质量。当我们仅使用单个质量进行训练时，则测试质量仅在很小范围时表现较好。但如果在训练中使用了多个质量（实际上使用的是一个正态分布的质量），则可以在更广泛的质量上表现较好。
![](20-5.png)

尽管在一定意义上，这样我们可能会牺牲一定的最优性来换取泛化性，但是对于深度神经网络来说在多个设置上保持最优性并不是不可能的。

事实上，如果我们有目标域中的一些经验，可以逐步缩小参数分布，使其与真实情况更加接近。

## 3 Multi-task transfer learning
通过学习多任务，我们通常学习得更快，也能取得更好的效果，因为可以得到一些共享的表征。简单来说，这样的做法可以加快学习的过程，也可以为下游任务提供更好的预训练。  

实质上，在强化学习的设定下进行多任务迁移学习只需要将我们原先的马尔可夫决策过程修改为联合马尔可夫决策过程上的单任务强化学习，主要的做法有以下几种。

### 3.1 Mixing initial states

回顾单个任务时我们会从 $p(\boldsymbol{s}_1)$ 采样一个 $\boldsymbol{s}_1$，而对多任务，我们只需要修改 $p(\boldsymbol{s}_1)$ 的分布为多个任务的初始状态的分布的加权平均（相当于按照概率选择马尔可夫决策过程，再根据对应马尔可夫决策过程选择初始状态）。

但是上述做法虽然对于 Atari 游戏是合理的，因为不同游戏开始时的状态从图像来看是不同的，但是对于机器人来说可能不同任务的初始状态都是一样的。

### 3.2 Different task contexts
针对上述问题，我们可以设置不同任务的上下文，例如一个独热向量、目标图像或者文本描述。

称这样的策略为上下文策略：$\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s}, \omega)$，相当于修改状态空间和状态：
$$
\tilde{\mathcal{S}} = \mathcal{S} \times \Omega, \quad \tilde{\boldsymbol{s}} = \begin{bmatrix} \boldsymbol{s}\\ \omega \end{bmatrix}
$$
这实际和修改初始分布的方式在某种意义上是一致的。我们并不需要修改强化学习算法，只需要修改状态空间和状态。
![](20-6.png)


### 3.3 Goal-conditioned policies
另一个常见的做法是使用目标条件策略，考虑 $\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s}, \boldsymbol{g})$，我们会用奖励 $r(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{g}) = \delta(\boldsymbol{s} = \boldsymbol{g})$ 或 $r(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{g}) = \delta(\|\boldsymbol{s} - \boldsymbol{g}\| \leq \epsilon)$ 来定义奖励。

这样的做法比较方便，因为我们不需要人为设计各个任务的奖励。同时也能零样本迁移到其他目标。

但是训练这样的目标条件策略可能并不容易，而且并非所有任务都等同于目标达成，例如一个任务可能是避免经过某一区域到达目的地，但是目标本身无法表示这一限制。

这需要一些比较好的技巧，例如选择训练的目标，表示价值函数，制定奖励和损失函数等等，具体可以参考：
- Kaelbling. Learning to achieve goals.  
- Schaul et al. Universal value function approximators.  
- Andrychowicz et al. Hindsight experience replay.  
- Eysenbach et al. C-learning: Learning to achieve goals via recursive classification  

## 4 Meta-Learning
### 4.1 Introduction
简单来说，元学习是学会学习的学习方式，在某种程度上可以理解为是对多任务学习在逻辑上的拓展。

元学习有多种实现方式：
- 学习一个优化器。 
- 学习一个包含了过去经验的循环神经网络。
- 学习一个表征。
![](20-7.png)


为什么元学习是个好主意？
通常深度强化学习，尤其是无模型方法需要大量样本，如果能够得到一个更快的强化学习学习器，那么我们就能更加高效地学习。如果我们有了一个元学习器，就能够
- 更加智能地探索；
- 避免尝试那些已经知道无用的动作；
- 更容易也更快获得正确的特征表示。

### 4.2 Formulation
接下来以图像识别为例，介绍一下什么是元学习。

在常规的监督学习中，我们会有一个训练集和测试集，在训练集上利用一个模型 $f(x) \to y$，接受输入（图片）$x$，输出（标签） $y$，其中模型的参数 $\theta$ 通过
$$
\theta^\ast = \arg\min_\theta \mathcal{L}(\theta, \mathcal{D}_{tr})
$$
来获取。这里的 $\mathcal{L}$ 是一个通用损失函数，例如关于 $f_\theta(x)$ 和 $y$ 的交叉熵损失。

而在元学习中，我们会有一系列任务，这些任务被进一步分为元训练集和元测试集。

Definition 5. _meta-training set（元训练集）_
元训练集中包含多个任务，每个任务有支持集（训练集）和查询集（测试集）。

Definition 6. _meta-test set（元测试集）_
元测试集包含多个与元训练集不同的任务，每个任务都有支持集（训练集）和查询集（测试集），用于评估元学习的效果。
![](20-8.png)

在元学习中学习的是一个从支持 / 训练集到模型（参数）的映射，具体来说，我们会学习一个
$$
f: \mathcal{D}_{train} \mapsto (x \mapsto y)
$$
如果用 $\theta$ 来表示这个元学习中学习的 $f$ 的参数，并且使用 $\phi_i$ 表示 $f_\theta$ 将训练集 $\mathcal{D}_{train}^i$ 映射到的模型的参数。可以写出一般的元学习学习的参数 $\theta$ 为：
$$
\theta^\ast = \arg\min_\theta \sum_{i = 1}^n \mathcal{L}(\phi_i, \mathcal{D}_i^{test})
$$
这一过程称为元学习，其中 $\phi_i = f_\theta(\mathcal{D}_i^{train})$ 获取 $\phi_i$ 的过程称为适应。直观来说也就是学习一种学习方式，使得在元训练集的任务上应用这种方式得到的参数的平均损失最小。

基于循环神经网络的元学习器：
这里考虑元学习的循环神经网络实现。最终的训练得到的 $f$ 的参数 $\theta$ 就是循环神经网络的参数（以及可能存在的 $\theta_p$, 后续提到），而在每个任务中，我们会使用一个新的隐藏状态，依次读入支持集中的每一个元素，并更新隐藏状态，于是每个任务中学到 $x \mapsto y$ 的参数 $\phi_i$ 可以表示为
$$
\phi_i = \begin{bmatrix} h_i & \theta_p \end{bmatrix}
$$
读入支持集的方式有多种选择，例如 RNN，Transformer 等,，对于不同的模型结构，得到的 $\phi_i$ 可能会有不同的形式。
![](20-9.png)
## 5 Meta Reinforcement Learning
### 5.1 Basic idea
在我们介绍的常规强化学习中，学习的参数是
$$
\theta^\ast = \arg\max_\theta \mathbb{E}_{\pi_\theta(\tau)}\left[r(\tau)\right]
$$
而在元强化学习则是 
$$
\theta^\ast = \arg\max_\theta \sum_{i = 1}^n \mathbb{E}_{\pi_{\phi_i}(\tau)}\left[r(\tau)\right]
$$
其中 $\phi_i = f_\theta(\mathcal{M}_i)$，这里 $\mathcal{M}_i$ 是一个马尔可夫决策过程 $\{\mathcal{S}, \mathcal{A}, \mathcal{P}, r\}$。

类似于监督元学习的方式，会有一系列元训练马尔可夫决策过程 $\mathcal{M}_i$，假设它们 $\mathcal{M}_i \sim p(\mathcal{M})$：
- 在元训练时间，学习 $f_\theta$；
- 在元测试时间，采样 $\mathcal{M}_{test} \sim p(\mathcal{M})$，可以得到 $\phi_i = f_\theta(\mathcal{M}_{test})$，作为结果，并且利用 $\phi_i$ 评估元学习的效果。
![](20-10.png)

### 5.2 Relation to contextual policy
元学习与上下文策略有很紧密的联系，可以认为元学习相当于是让上下文策略的 $\omega$ 等条件来源于 $\mathcal{M}_i$ 中通过 $f_\theta$ 推断得来，而不是人为指定。
### 5.3 Basic algorithm idea
值得注意的是，和监督元学习不同的是，$f_\theta$ 接收的不是一个训练集，而是一个马尔可夫决策过程，数据需要自己与环境交互得到。

因此整个训练过程可以视作重复以下过程：
1. 采样任务编号 $i$，收集任务 $i$ 的数据 $\mathcal{D}_i$（收集支持集）；
2. 通过 $\phi_i = f_\theta(\mathcal{M}_i)$ 得到适应的策略 $\phi_i$；
3. 利用适应策略 $\pi_{\phi_i}$ 收集数据 $\mathcal{D}_i'$（收集查询集）；
4. 利用 $\mathcal{L}(D_i', \phi_i)$ 更新 $\theta$。

上述的 $1-3$ 步对应于适应，而第 $4$ 步对应于元训练。

在这基础上有很多相当直观的改进方式：
- 在第 $4$ 步前进行多轮的适应步骤；
- 在第 $4$ 步更新 $\theta$ 时使用多个任务进行更新。  

在接下来我会介绍几种关于 $f$ 与 $\mathcal{L}$ 的具体实现。

## 6 Meta-RL with recurrent policy
在这里我们考虑基于循环神经网络等循环网络的元学习器。
### 6.1 Basic Ideas
考虑一个读入所有过去经验循环神经网络，这些经验 $(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)$ 可能来源于不同的轮次，我们称它们属于同一个元轮次。在将它们全部读入后，得到了一个隐藏状态 $h_i$, ，之后可以考虑一个输入 $h_i$ 与当前状态并输出动作的模型作为策略。这里类似地有
$$
phi_i = \begin{bmatrix} h_i & \theta_p \end{bmatrix}
$$
这看起来好像就在训练一个循环神经网络的策略，这里核心的区别在于循环神经网络的隐藏状态在不同轮次间不会被清除，这是循环策略能够学会探索的关键。

由于此时的策略是关于 $h_i$ 的，此时的元轮次中包含了过去多个轮次的信息，如果在连续几个轮次采取的行为中都没有得到奖励，下一个时期时，策略考虑的就不是"在当前状态下应该采取什么行动", 而是"我知道在当前状态我过去已经尝试过这些行动了，结果并不理想，那么我应该采取什么行动"。
![](20-11.png)
事实上，当给策略这种看到多个回合的机会时，探索问题就转化为了解决这种高层次的马尔可夫决策过程。

在常规强化学习中我们在
$$
\theta = \arg\max_\theta \mathbb{E}_{\pi_\theta(\tau)}\left[\sum_{t = 1}^T r(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]
$$
目前有了更多关于元强化学习的方法，但它们高层次的理念都是给予策略这样一种多回合的经验。

### 6.2 Algorithm and design choices
我们可以得出一个基于循环策略的基本元强化学习算法：
1. 对于每个任务 $i$，初始化隐藏状态；  
2. 对于每一个时间步（循环神经网络的时间步）$t$：
3.  利用当前由 $h_t$ 决定的策略采取一个动作更新当前任务的数据集：$$\mathcal{D}_i = \mathcal{D}_i \cup \{(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}, r_t)\}$$
4.  通过 $\mathcal{D}_i$ 更新隐藏状态：$$h_{t + 1} = f_\theta(h_t, \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}, r_t)$$
5.  利用 $\theta \gets \theta - \alpha \nabla_\theta \sum_i \mathcal{L}_i(\phi_i, \mathcal{D}_i^{test})$ 更新 $\theta$，这里 $\mathcal{D}_i^{test}$ 可以通过最终的 $\phi_i$ 采样得到。 

这一类基于循环策略的方法中有很多种架构选择：
- 标准循环神经网络 / 长短期记忆网络：参见 Duan, Schulman, Chen, Bartlett, Sutskever, Abbeel. RL2: Fast Reinforcement Learning via Slow Reinforcement Learning. 2016.  
- 注意力+时间卷积：参见 Mishra, Rohaninejad, Chen, Abbeel. A Simple Neural Attentive Meta-Learner.  
- 并行排列不变上下文编码器：参见 Rakelly\*, Zhou\*, Quillen, Finn, Levine. Efficient Off-Policy Meta-Reinforcement learning via Probabilistic Context Variables.
![](20-12.png)

### 6.3 Reference
还有更多关于基于循环策略的元强化学习实例，参见：
- Heess, Hunt, Lillicrap, Silver. Memory-based control with recurrent neural networks. 2015  
- Wang, Kurth-Nelson, Tirumala, Soyer, Leibo, Munos, Blundell, Kumaran, Botvinick. Learning to Reinforcement Learning. 2016  
- Duan, Schulman, Chen, Bartlett, Sutskever, Abbeel. RL2: Fast Reinforcement Learning via Slow Reinforcement Learning. 2016  
![](20-13.png)

## 7 Gradient-Based Meta-Learning
### 7.1 Basic Ideas
回顾预训练+微调方案，这也可视作某种元学习：对于很多计算机视觉任务来说，在预训练中学会如何提取特征，使得模型参数在参数空间中处于一个很好的位置，从这个位置上出发，只需要少量的梯度步就可以得到特定任务上的模型。

能否借鉴这种方案来得到一个更好的元学习算法呢？

回顾元强化学习的目标是
$$
\theta^\ast = \arg\max_\theta \sum_{i = 1}^n \mathbb{E}_{\pi_{\phi_i}(\tau)}\left[R(\tau)\right]
$$
其中 $\phi_i = f_\theta(\mathcal{M}_i)$。

考虑把 $f_\theta$ 建模为一个强化学习算法而不是一个循环神经网络。具体来说, $f_\theta$ 可以视作一个从 $\theta$ 出发进行一个梯度步长的算法：
$$
\phi_i = f_\theta(\mathcal{M}_i) = \theta + \alpha \nabla_\theta J_i(\theta)
$$
这里 $J_i(\theta)$ 是在 $\mathcal{M}_i$ 上的某种目标。此时元学习的目标就是找到一个 $\theta$ 使其在所有的任务 $\mathcal{M}_i$ 上进行一个梯度步长后能够得到平均最大的奖励，这样的方式对应于模型无关元学习（Model-agnostic meta-learning，MAML）。

### 7.2 MAML (Model-Agnostic Meta-Learning)
在模型无关元学习中，目标是找到一个能够最大化
$$
\sum_i J_i\left[\theta + \alpha \nabla_\theta J_i(\theta)\right]
$$
的 $\theta$, 这里的 $J_i\left[\theta + \alpha \nabla_\theta J_i(\theta)\right]$ 可以是在 $\mathcal{M}_i$ 上进行一步更新后的参数在查询集上的损失，写出梯度更新式为
$$
\theta \gets \theta + \beta \sum_{i} \nabla_\theta J_i\left[\theta + \alpha \nabla_\theta J_i(\theta)\right]
$$
这在某种意义上有一种“二阶的”的感觉，不过对于自动微分的深度学习框架来说实现起来并不困难。这样的算法相当于是在参数空间中找到一个 $\theta$ 使得从这个位置出发很容易到达各个任务的最优。
![](20-14.png)



简单总结一下我们刚才介绍的模型无关元学习算法和前面的一些通用框架：
- 监督学习：$f(x) \to y$。
- 监督元学习：$f(\mathcal{D}_{train})(x) \to y$。
- 模型无关元学习：$f_{MAML}(\mathcal{D}_{train})(x) \to y$，这里 $f_{MAML}(\mathcal{D}_{train}) = f_{\theta'}(x)$，其中$$\theta' = \theta - \alpha \sum_{(x,y) \in \mathcal{D}_{train}} \nabla_\theta \mathcal{L}(f_\theta(x), y)$$
模型无关元学习的方式相较于基于循环策略的方式有一个明显的好处：在模型无关元学习中，可以在元测试时间选择实际进行更多的梯度步骤，，在得到的 $\phi_i$ 基础上进一步微调，这是更加“灵活”的，而循环处理的方式得到的策略已经由循环神经网络结构和 $\theta$ 定死了。

这样的做法是非常合理的，对于距离源任务分布较远的目标任务，我们自然需要更多的梯度步骤来进行微调。

### 7.3 Reference
- MAML meta-policy gradient estimators:  
- Finn, Abbeel, Levine. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.  
- Foerster, Farquhar, Al-Shedivat, Rocktaschel, Xing, Whiteson. DiCE: The Infinitely Differentiable Monte Carlo Estimator.  
- Rothfuss, Lee, Clavera, Asfour, Abbeel. ProMP: Proximal Meta-Policy Search.  
- Improving exploration:  
- Gupta, Mendonca, Liu, Abbeel, Levine. Meta-Reinforcement Learning of Structured Exploration Strategies.  
- Stadie\*, Yang\*, Houthooft, Chen, Duan, Wu, Abbeel, Sutskever. Some Considerations on Learning to Explore via Meta-Reinforcement Learning.  
- Hybrid algorithms (not necessarily gradient-based):  
- Houthooft, Chen, Isola, Stadie, Wolski, Ho, Abbeel. Evolved Policy Gradients.  
- Fernando, Sygnowski, Osindero, Wang, Schaul, Teplyashin, Sprechmann, Pirtzel, Rusu. Meta Learning by the Baldwin Effect.  

## 8 Meta-RL as a POMDP
### 8.1 Formalization
可以将一个元学习的任务建立为一个部分可观测马尔可夫决策过程（POMDP），原先的马尔可夫决策过程（MDP）可以表示为 $\mathcal{M} = \{\mathcal{S}, \mathcal{A}, \mathcal{P}, r\}$，这里考虑将其建立为 $\tilde{M} = \{\tilde{\mathcal{S}}, \mathcal{A}, \tilde{\mathcal{P}}, r, \tilde{\mathcal{O}}, \mathcal{E}\}$，其中
- $\tilde{\mathcal{S}} = \mathcal{S} \times \mathcal{Z}$，$\tilde{\boldsymbol{s}} = (\boldsymbol{s},\boldsymbol{z})$，$\boldsymbol{z}$ 包含了解决当前任务所需的全部信息，也就是说真正的状态空间还包含了任务的信息；
- $\tilde{\mathcal{O}} = \mathcal{S}$，原先的状态空间变为了当前的观测空间，其中 $\tilde{\boldsymbol{o}} = \boldsymbol{s}$。

基于上述的定义，学会一个任务意味着能够通过交互的数据推断出其对应的 $\boldsymbol{z}$，与此同时，还有知道如何利用这些信息采取动作。这两部分都是元学习器的组成部分：
- 推断 $\boldsymbol{z}$：我们会学习一个推断网络来近似 $p(\boldsymbol{z} \mid \boldsymbol{s}_{1:i}, \boldsymbol{a}_{1:i}, r_{1:i})$，也就是在当前的观测下，对 $\boldsymbol{z}$ 的后验分布，这像是一个任务识别器；
- 依据 $\boldsymbol{z}$ 采取动作：我们会学习一个策略 $\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s}, \boldsymbol{z})$，也就是在当前的观测和任务的信息下，如何采取动作， 这像是一个根据任务做调整的执行器。 

注意：上述可能让人困惑的地方在于，这里的 $\theta$ 与通常决定如何采取动作的参数 $\theta$ 并不一样。这里的 $\theta$ 是元学习器参数的一部分，而 $\boldsymbol{z}$ 更像是一般强化学习中的参数。这一种做法在概念上和循环神经网络元强化学习很接近：在循环神经网络的元强化学习中，$\phi$ 像是一般强化学习中策略的参数，只是这一类方法中使用的是随机的 $\boldsymbol{z}$ 而不是 $\phi$。

### 8.2 posterior sampling
由于通过 $p(\boldsymbol{z} \mid \boldsymbol{s}_{1:i}, \boldsymbol{a}_{1:i}, r_{1:i})$ 进行完全的贝叶斯推断是难处理的，这里会使用在探索中讨论的后验采样方法，也就是采样一个 $\boldsymbol{z}$，假设这个单点分布就是真实的后验分布，并依据这个 $\boldsymbol{z}$ 采取动作。

直觉：
- 刚开始学习一系列任务时，由于对其并不了解，因此 $p(\boldsymbol{z} \mid \boldsymbol{s}_{1:i}, \boldsymbol{a}_{1:i}, r_{1:i})$ 会非常均匀，也就是会随机选择一种"行动方式"，或者说随机完成一个"任务"，当然由于此时策略很弱，因此完成的"任务"也很怪异；
- 随着训练不断进行，后验越来越能够从上下文中提取任务的信息，给出更加确定的 $\boldsymbol{z}$，而策略也能够在这些 $\boldsymbol{z}$ 上表现越来越好；
- 当对任务有足够了解后，给定 $\boldsymbol{s}_{1:i}, \boldsymbol{a}_{1:i}, r_{1:i}$，相当于指定了一个任务，会有一个更好的后验分布，在这个后验分布采样的 $\boldsymbol{z}$ 配合上训练好的 $\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s},\boldsymbol{z})$ 可以能将这个任务完成的很好。

### 8.3 Basic algorithm idea
具体来说，算法流程如下：
1. 从 $\hat{p}(\boldsymbol{z} \mid \boldsymbol{s}_{1:i}, \boldsymbol{a}_{1:i}, r_{1:i})$ 中采样 $\boldsymbol{z}$（通过[[Concepts#3 变分推断（Variational Inference，VI）|变分推断（Variational Inference，VI）]]来估计这个后验）；
2. 依据 $\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t, \boldsymbol{z})$ 采取动作来收集更多数据，这些数据称作上下文，记作 $\boldsymbol{c}_{1:i} = \{\boldsymbol{s}_{1:i}, \boldsymbol{a}_{1:i}, \boldsymbol{s}'_{1:i} r_{1:i}\}$；
3. 重复上述过程若干次后，再按照上述思路收集数，然后通过数据和上下文，利用损失函数更新元学习器 $\theta$ 以及估计的后验。 

这里的前两步可以视作是适应，而最后一步则是元训练。

这并不是最优的选择，但是在某种意义上已经相当好了。潜在的次优性在于开始时相较于按照一个随机的任务 $\boldsymbol{z}$ 进行探索，尝试多个任务直到找到最优区域，似乎并不如开始时不考虑非要完成某个任务，直接在一个轮次中探索那些奖励可能更高的区域。
![](20-15.png)


### 8.4 Variational inference for meta-RL
我们还没有介绍如何得到这样一个后验 $p(\boldsymbol{z} \mid \boldsymbol{s}_{1:i}, \boldsymbol{a}_{1:i}, r_{1:i})$，这里会使用[[Concepts#3 变分推断（Variational Inference，VI）|变分推断（Variational Inference，VI）]]，一个具体的例子是如下的 PEARL: Probabilistic Embeddings for Actor-Critic Reinforcement Learning：

首先这里训练策略的算法是软演员-评论家（SAC），我们会有策略和 Q 函数 $\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s}, \boldsymbol{z})$，$Q_\theta(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{z})$，同时会通过推断网络：$q_\phi(\boldsymbol{z} \mid \boldsymbol{s}_{1:i}, \boldsymbol{a}_{1:i}, r_{1:i})$ 来近似后验，它们都是元学习器的一部分。

这里的 $\phi$ 是推断网络的参数，也是元学习器的一部分，需要和之前讨论的循环神经网络元学习器中的 $\phi$ 以及一般形式元强化学习的 $\phi$ 区分开来。

由于实质上构建了一个关于 $\boldsymbol{c}_{1:i}$ 和 $\boldsymbol{z}$ 的隐变量模型，为了这个模型能够建模好上下文数，我们需要最大化这些上下文的对数似然 $\log p(\boldsymbol{c}_{1:i})$。

利用变分推断，有
$$
\log p(\boldsymbol{c}_{1:i}) \geq \mathbb{E}_{z \sim q_\phi(\boldsymbol{z} \mid \boldsymbol{c}_{1:i})} \left[\log p_\theta(\boldsymbol{c}_{1:i} \mid \boldsymbol{z}) - D_{KL}(q_\phi(\boldsymbol{z} \mid \boldsymbol{c}_{1:i}) \parallel p(\boldsymbol{z}))\right]
$$
这里 $\log p(\boldsymbol{c}_{1:i} \mid \boldsymbol{z})$ 可以理解为给定任务信息后，生成上下文的似然，其中包含了动态以及处理相关的部分，这里不显式建模它，而是将其转化为某种奖励的度量，这里可以利用一些角度反映其联系：

_Proof._ 
由于在 PEARL 中，我们假设不同任务的动态相同，因而证据下界（ELBO）可以进一步把动态提到外面去，由于其和 $\theta, \phi$ 都没有关系，在求梯度时会消失，不妨就写为
$$
\mathbb{E}_{z \sim q_\phi(\boldsymbol{z} \mid \boldsymbol{c}_{1:i})} \left[\sum_{j = 1}^i \log \pi_\theta(\boldsymbol{a}_i \mid \boldsymbol{s}_i, \boldsymbol{z}) - D_{KL}(q_\phi(\boldsymbol{z} \mid \boldsymbol{c}_{1:i}) \parallel p(\boldsymbol{z}))\right]
$$
这里用来训练策略的算法是软演员-评论家，可以参见在[[Lecture 17 Reframing Control as an Inference Problem]]的讨论，在那里有
$$
\pi(\boldsymbol{a} \mid \boldsymbol{s}) = \exp(Q(\boldsymbol{s},\boldsymbol{a}) - V(\boldsymbol{s}))
$$
其中
$$
V(\boldsymbol{s}) = \log \int \exp(Q(\boldsymbol{s}, \boldsymbol{a}))\text{d}\boldsymbol{a}
$$
类似地这里有
$$
\pi(\boldsymbol{a} \mid \boldsymbol{s}, \boldsymbol{z}) \propto Q(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{z})
$$
于是 $\pi_\theta(\boldsymbol{a}_i \mid \boldsymbol{s}_i, \boldsymbol{z})$ 可以进一步化为某种回报的度量。

经过实际的调整后，对于每一个任务 $i$，得到实际的目标：
$$
(\theta,\phi) = \arg\max_{\theta,\phi} \frac{1}{N} \sum_{i = 1}^N \mathbb{E}_{\boldsymbol{z} \sim q_\phi, b^i \sim \mathcal{B^i}} \left[R_i(b^i) - D_{KL}(q_\phi(\boldsymbol{z} \mid \boldsymbol{c}_{1:K}^i) \parallel p(\boldsymbol{z}))\right]
$$
这里的 $R_i(b^i)$ 是一种关于回报的度量，$b^i \sim \mathcal{B^i}$ 是回放缓冲区中的一个批次。后一项 KL 散度让策略保持接近于先验（具体来说使用标准正态分布），其中的 $\boldsymbol{c}_{1:K}^i$ 是一组从缓冲区中采样的上下文（为了更好地探索，上下文可能比用于更新的批量更小）。

另外，文中使用一系列样本级的 $\Psi_\phi(\boldsymbol{z} \mid \boldsymbol{c}^i_j)$（通过预测均值和方差），将其平均来得到采样的全部上下文的 $q_\phi(\boldsymbol{z} \mid \boldsymbol{c}^i_{1:K})$。
![](20-16.png)

参见：Rakelly\*, Zhou\*, Quillen, Finn, Levine. Efficient Off-Policy Meta-Reinforcement learning via Probabilistic Context Variables. ICML 2019.


我们在之前的内容中介绍过多种处理部分可观测马尔可夫决策过程（POMDP）的方法，其中一种方法对应带记忆的策略，这实质上对应基于循环神经网络的元学习。而我们知道，在处理部分可观测马尔可夫决策过程的方法中，存在显式状态估计这类方法，例如构建一个状态空间模型，实际上这也可以推导出一类元学习算法。

### 8.5 Reference
- Rakelly\*, Zhou\*, Quillen, Finn, Levine. Efficient Off-Policy Meta-Reinforcement learning via Probabilistic Context Variables. ICML 2019  
- Zintgraf, Igl, Shiarlis, Mahajan, Hofmann, Whiteson. Variational Task Embeddings for Fast Adaptation in Deep Reinforcement Learning.  
- Humplik, Galashov, Hasenclever, Ortega, Teh, Heess. Meta reinforcement learning as task inference.  

## 9 Summary
### 9.1 The three perspective on meta-RL
最后我们将介绍三种不同的元强化学习的视角，它们对应于我们之前介绍的三种方法：

视角 1：
训练一个循环神经网络，将 $f_\theta(\mathcal{M}_i)$ 当作一个黑盒：
- 在概念上很简单；
- 相对容易应用；  
- 容易引发元过拟合：如果元测试的任务稍微偏离了分布，那么我们可能无法得到较好的表现，而且由于前向传播已经是确定的了，没有办法调整。
- 在现实中不容易优化，尽管目前的 Transformer 改善了这一点。

视角 2：
将 $f_\theta(\mathcal{M}_i)$ 当作一个强化学习算法，通过模型无关元学习算法来学习：
- 良好的外推能力，当任务略有偏离时，可以通过类似多个梯度步骤的方式来获得较好的表现；
- 从概念上讲很优雅；
- 通常较为复杂，也需要很多样本；
- 不容易扩展到演员-评论家这类时间差分的方法。

视角 3：
将 $f_\theta(\mathcal{M}_i)$ 建模为一个推断问题，任务转化为推断 $z$：
- 可以通过后验采样来进行简单有效的探索；
- 优雅地归约到了求解一个部分可观测马尔可夫决策过程；
- 同样容易元过拟合；
- 在现实中也不容易优化。

但是这三种方式同样有很多共性：
- 推断的方式：推断的 $\boldsymbol{z}$ 就像是循环神经网络中的 $\phi$，只不过换成了一个随机变量；
- 基于梯度的方法也可以通过一些特定的网络结构转化为另外两种方法，如果给梯度添加噪声，那么就更像是一个推理过程。

### 9.2 Meta-RL and emergent phenomena
在强化学习和认知科学的交叉领域中，我们会发现与元学习有着很多相似的现象，例如人类和动物学习的方式有多种方式：
1. 高效的无模型强化学习
2. 情景回忆
3. 基于模型的强化学习
4. 因果推理

这些方法似乎都发生在人类学习的某些层面，目前尚不明确为什么某些算法会在某些情况下被使用，也许存在一个更高层次的元学习，决定了在什么情况下使用什么样的算法，进而产生了这类智能的现象。目前的一些研究也发现，元强化学习引发了情景学习、因果推理等方法。
![](20-17.png)

### 9.3 Summary of entire lecture
在本节中，我们
- 从一些例子出发指出在很多任务中，关于任务的知识可以有助于我们进行学习，并引出了迁移学习和元学习的概念；
- 在迁移学习部分，主要介绍了两种迁移的方式：正向迁移和多任务迁移；
- 在正向迁移中，介绍了将其在强化学习中应用的一些问题，从领域自适应、马尔可夫决策过程的差异、微调几个角度介绍了一些示例方法；
- 在多任务迁移中，介绍了几种将多任务问题转化为联合马尔可夫决策过程的方法，例如混合初始状态、任务上下文、目标条件策略等；
- 在元学习部分，首先介绍了其公式化表述，给出了其一般形式以及和通常的学习问题的区别，接下来引出元强化学习的问题，并介绍了几种方法；
- 基于循环策略的元强化学习，这里介绍了一种基于循环神经网络的元学习器；
- 基于梯度方法的元强化学习，这里介绍了模型无关元学习的方法；
- 元强化学习作为部分可观测马尔可夫决策过程，这里介绍了其公式化表述，并且主要关注了基于后验采样的方法，例如概率嵌入辅助策略学习；
- 最后，对比了几种元学习的算法，以及元强化学习和一些人类智能的现象之间的关系。