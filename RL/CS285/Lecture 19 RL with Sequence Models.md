在本节中，我们将首先介绍部分可观测马尔可夫决策过程（POMDP）的基本概念以及与普通马尔可夫决策过程（MDP）的差异，通常的强化学习算法能否处理部分可观测性的情况，以及状态空间模型和历史状态等处理部分可观测马尔可夫决策过程的方法。在此之后，我们将介绍强化学习与语言模型（LM）如何结合，如将对齐建模成一个单步的强化学习问题，以及基于人类反馈的强化学习（RLHF）、直接偏好优化（DPO）等方法。最后我们将介绍如何将语言模型中的一些任务建模为多步强化学习，并且使用基于价值的方法来处理。

## 1 Partial observed MDP
### 1.1 Introduction
回顾在马尔可夫决策过程中，状态满足的一个核心性质是马尔可夫性质，也就是给定当前的状态， 过去的状态与未来的状态是独立的，记作 $\boldsymbol{s}_{t + 1} \perp \boldsymbol{s}_{1:t - 1} \mid \boldsymbol{s}_t$。

然而在部分可观测马尔可夫决策过程中，我们失去的一个核心就是观测不再具有马尔可夫性，一个直观的理解方式是，仅仅依靠当前观测不能完全描述状态，或者说过去的观测是有用的。

为什么需要考虑部分可观测马尔可夫决策过程呢？事实上，绝大多数现实中的问题都是部分可观测马尔可夫决策过程。
- 当然在一些特定情境例如 Atari 游戏中，我们可能的确有一些信息遗漏了，但是绝大多数信息还是可以通过当前的观测得到，因此可以将其视作马尔可夫决策过程处理。 
- 但是在很多更加实际的情境中，可能有非常多的信息观测不到：
	- 在驾驶的任务中，如果仅凭视觉观察可能无法获取车辆周围的所有信息；
	- 在电子游戏中，一些过去的信息也重要，例如 Minecraft 中，我们需要记住自己做过什么事情，这些事情影响了之后的动作是否有效；
	- 在与人类交互的任务中，人类的内心状态也是无法观测到的；
	- 在对话中，每一次的收到的回答是观测。但是对话的上下文影响着观测背后的含义。
![](19-1.png)

### 1.2 Partially observed MDPs can be weird
在部分可观测马尔可夫决策过程中，可能会发生一些在马尔可夫决策过程中不会发生的事，这些可能是很奇怪的，我们会从以下两个例子来展示。

例如，信息收集动作：
在部分可观测的情境下，一些不造成任何真实状态改变的动作，也无法获得奖励的动作可能是最优的，例如那些可能告诉我们更多有用信息的行为。
对于走出一个迷宫的任务，如果知道迷宫的形式，那么我们自然可以将其当作普通马尔可夫决策过程处理，将位置信息作为状态，设计合理的奖励信号，然后运行强化学习算法，最终我们会得到一个策略，这个策略会告诉我们在每个位置应该怎么走才能尽快走出迷宫。
但是如果要得到一个能够走出所有迷宫的策略，换言之不知道迷宫具体形式时，我们需要建模一个部分可观测马尔可夫决策过程问题，其中状态还包括了是迷宫的设定。此时如果选择爬到迷宫的顶上俯瞰迷宫（假设规则允许），虽然本身不改变状态，不会带来奖励，但是可以获得更多的关于状态的信息，从而更好地走出迷宫。
![](19-2.png)

再例如，随机最优策略：
在完全观测的情况下，始终存在一个确定性的最优策略（可能也同时有最优的随机策略），但是在部分可观测马尔可夫决策过程中，完全有可能不存在确定性的最优策略。
一个很简单的情况是考虑 $\mathcal{S} = \{A,B,C\}$，表示从左到右三个网格，但它们对应于同一个观测 $\boldsymbol{o}$，动作空间 $\mathcal{A} = \{l, r\}$ 是向左向右，且目标是到达位于中间的 $B$ 状态。
由于在观测看来哪里都是一样的，因此如果是确定性策略，则要么永远向左或者永远向右，这意味着总有 $A,C$ 中的一个位置让我们永远到不了 $B$，但是一个随机策略可以让我们最终到达 $B$。
![](19-3.png)

## 2 Handling Partial Observability
### 2.1 Directly use observations as states
能否直接将观测当作状态来处理呢？这里考虑仅仅利用当前的观测也就是无记忆策略 $\pi(\boldsymbol{a} \mid \boldsymbol{o})$。可以回顾过去介绍的强化学习算法，考虑它们能否直接应用在部分可观测的情境下。

#### 2.1.1 Policy gradient
可以直接使用观测当作状态处理，这在[[Lecture 3 Policy Gradients]]一节中已经推导过了：
_Proof._ 
不妨记整条轨迹为 $\tau_{\boldsymbol{o}}$，那么利用链式法则，可以得到
$$
\begin{aligned} \log p(\tau_{\boldsymbol{o}}) &= \log p(\boldsymbol{o}_1) + \log p(\boldsymbol{a}_1 \mid \boldsymbol{o}_1) + \log p(\boldsymbol{o}_2 \mid \boldsymbol{o}_1, \boldsymbol{a}_1) + \log p(\boldsymbol{a}_2 \mid \boldsymbol{o}_{1:2}, \boldsymbol{a}_1) + \cdots\\ &= \log p(\boldsymbol{o}_1) + \sum_{t} \log p(\boldsymbol{a}_t \mid \boldsymbol{o}_{1:t}, \boldsymbol{a}_{1:t - 1}) + \sum_{t} \log p(\boldsymbol{o}_{t + 1} \mid \boldsymbol{o}_{1:t}, \boldsymbol{a}_{1:t})\\ &= \log p(\boldsymbol{o}_1) + \sum_{t} \log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{o}_t) + \sum_{t} \log p(\boldsymbol{o}_{t + 1} \mid \boldsymbol{o}_{1:t}, \boldsymbol{a}_{1:t}), \end{aligned}
$$
由于和 $\theta$ 有关的只有策略，于是对 $\theta$ 求梯度有
$$
\nabla_\theta \log p(\tau_{\boldsymbol{o}}) = \sum_{t} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{o}_t)
$$
于是就有
$$
\nabla_\theta J(\theta) = \mathbb{E}_{p(\tau_{\boldsymbol{o}})}\left[\nabla_\theta p(\tau_{\boldsymbol{o}}) r(\tau_{\boldsymbol{o}})\right]\approx \frac{1}{N} \sum_{i = 1}^N \left(\sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{o}_{i,t})\right) \left(\sum_{t = 1}^T r(\boldsymbol{o}_{i,t}, \boldsymbol{a}_{i,t})\right)
$$
简单来说，上述过程仅利用了链式法则没有基于马尔可夫性质，因此可以直接使用观测替换状态。

然而，要注意优势函数的选择。对于[[Concepts#10 蒙特卡洛方法 (Monte Carlo Method)|蒙特卡洛方法 (Monte Carlo Method)]]，根据上述推导是没有问题的。但是使用时间差分（Temporal Difference，TD）的方法就不行了，也就是不能使用 $r_t + \gamma\hat{V}(\boldsymbol{o}_{t + 1}) - \hat{V}(\boldsymbol{o}_t)$ 这类估计作为优势。

为什么不行呢？直观来说，优势函数作为关于 $\boldsymbol{s}_t$ 的函数，其不应当依赖过去的 $\boldsymbol{s}_{t - 1}$，但是当使用观测直接代替状态时则会产生问题：我们并不能训练一个 $\hat{V}(\boldsymbol{o}_t)$，因为单个 $\boldsymbol{o}_t$ 不足以描述状态，不同的过去可能导致同一个 $\boldsymbol{o}_t$ 对应于不同的价值。

不妨思考以下问题：
1. 用观测代替状态时还可以应用因果性吗？
2. 可以使用 $V(\boldsymbol{o}_t)$ 作为基线吗？

这两个做法其实都是可以的。对于因果性，同样没有利用马尔可夫性质，具体证明可以参考[[Lecture 3 Policy Gradients]]一节中的推导。对于价值函数作为基线，根据类似的推导可知只要基线不依赖于动作，使用什基线都是无偏的。

#### 2.1.2 Value-based methods
在这一类方法中，会有价值函数，通过以下的方式更新：
$$
Q^\ast(\boldsymbol{s}, \boldsymbol{a}) \gets r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \max_{\boldsymbol{a}'} Q(\boldsymbol{s}', \boldsymbol{a}')
$$
将这里的状态替换为观测是不行的，背后的原因和不能使用时间差分估计作为优势的原因是一样的，因为同一个观测并不足以描述状态，也不足以确定一个价值。

#### 2.1.3 Model-based RL methods
在这一类方法中，会学习一个动态模型：
$$
\hat{p}(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})
$$
将这里的状态替换为观测也是不行的，考虑以下环境：
我们面前有两扇关闭的门，其中有且仅有一扇门是可以打开的，考虑问题的建模：
- 状态：$\mathcal{S} = \{l,r\}$，表示哪一扇门是真的能够打开的；
- 动作：$\mathcal{A} = \{change, open\}$，表示换一扇门或者尝试打开；  
- 观测：$\mathcal{O} = \{pass, left, right\}$，表示结果或者面对的门是哪一扇。

在每一轮次开始时，会随机产生一个保持不变的状态。在经过多个轮次训练后，模型会学到接近于以下的"动态"：
$$
p(\boldsymbol{o}' = pass \mid \boldsymbol{o} = left, \boldsymbol{a} = open) = 0.5
$$
也就是当我们站在左侧的门时，尝试打开门有 $0.5$ 的概率成功。这似乎是合理的，但其实不然，这意味着无论真实状态如何，每次尝试开门都有 $0.5$ 概率门打开，因而最终学到的策略就是反复尝试开同一扇门，这显然是不对的。

事实上目前学习的动态无法捕捉如果尝试开某扇门打不开，那么无论尝试多少遍也打不开这一点。换言之，这样一个马尔可夫模型 $p(\boldsymbol{o}'\mid \boldsymbol{o}, \boldsymbol{a})$ 无法表示非马尔可夫的（对于观测来说）的环境。

上述的讨论中将使用的模型局限在了无记忆的一类，这样的限制并不一定合理，在现实中为了解决这些非马尔可夫的问题，我们可能会使用历史观测作为输入。

### 2.2 State space model
在[[Lecture 16 Variational Inference and Generative Model]]一节中，我们介绍了一个处理部分可观测性的方法，也就是使用状态空间模型。具体来说，一个实际的选择是使用一个[[Concepts#18 变分自编码器 (Variational Autoencoder, VAE)|变分自编码器 (Variational Autoencoder, VAE)]]，其中的 $\boldsymbol{x}$ 是一连串的观测，$\boldsymbol{z}$ 是状态。具体来说，可以有设计：

先验：
我们希望先验不再是常规变分自编码器中那样的各维度相互独立，而是具有隐含的动态，也就是
$$
p(\boldsymbol{z}) = p(\boldsymbol{z}_1) \prod_{t} p(\boldsymbol{z}_{t + 1} \mid \boldsymbol{z}_t, \boldsymbol{a}_t)
$$
可以使用 $p(\boldsymbol{z}_1) = \mathcal{N}(0, I)$，而 $p(\boldsymbol{z}_{t + 1} \mid \boldsymbol{z}_t, \boldsymbol{a}_t)$ 则是学习得到的。

解码器：
我们希望解码器处理各时间步的观测是独立的，也就是
$$
p(\boldsymbol{o} \mid \boldsymbol{z}) = \prod_{t} p(\boldsymbol{o}_t \mid \boldsymbol{z}_t)
$$

编码器：
我们通常不能假设各时间步的信息是独立的，因为单个观测实际上不足以表示整个隐藏状态，在基于模型的强化学习中讨论过了多种选择，其中一个是
$$
q_\phi(\boldsymbol{z} \mid \boldsymbol{o}) = \prod_{t} q_\phi(\boldsymbol{z}_t \mid \boldsymbol{o}_{1:t})
$$
通常可以利用 LSTM，Transformer 等序列式模型来表示 $q_\phi(\boldsymbol{z}_t \mid \boldsymbol{o}_{1:t})$。

这样的结构设计如何进行学习可以参考 "Dream to Control: Learning Behaviors by Latent Imagination, Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi"。

注意：这种方法属于一种基于模型的方法，因为我们在先验中学习了动态。这样的做法也存在局限性，在一些情况下进行下一个观测的预测可能很困难且并不必要，生成这些观测（例如图片），可能比得到最优策略更困难。

### 2.3 History states
一个替代方案是使用历史状态，考虑定义 $\boldsymbol{s}_t = (\boldsymbol{o}_1, \boldsymbol{o}_2, \ldots, \boldsymbol{o}_t)$，这样定义的状态满足马尔可夫性质 $\boldsymbol{s}_{t + 1} \bot \boldsymbol{s}_{t - 1} \mid \boldsymbol{s}_t$，因为 $\boldsymbol{s}_t$ 包含了 $\boldsymbol{s}_{t - 1}$ 包含的所有信息，因此 $\boldsymbol{s}_{t + 1}$ 与 $\boldsymbol{s}_{t - 1}$ 是条件独立的。

使用这样的历史观测作为状态是可行的，因此通过类似
$$
Q(\boldsymbol{o}_1,\ldots,\boldsymbol{o}_t, \boldsymbol{a}) \gets r(\boldsymbol{o}_t, \boldsymbol{a}) + \gamma \max_{\boldsymbol{a}'} Q(\boldsymbol{o}_1,\ldots,\boldsymbol{o}_{t + 1}, \boldsymbol{a}')
$$
的做法，可以在部分可观测马尔可夫决策过程中应用基于价值算法，但是注意到这里的 $Q$ 函数需要能够处理不同长度的历史，我们需要设计特定的模型架构。

选择 1：
可以考虑把历史图像堆叠起来并输入 $Q$ 函数。这对于较短的历史是可行的，对于历史很长的情况，可以以启发式的方式仅保留最近的几个观测。
能否采用启发式的方式取决于问题的具体设定，例如如果想要记住爬上迷宫时的所有观测，并基于这些观测来决定下一步怎么走，那么仅采用最近的几个观测是不够的。
![](19-4.png)

选择 2：
最通用的方式是使用一个序列模型。以 $Q$ 函数为例，模型会在最后一个观测上输出 Q 值。这可以通过 RNN，LSTM，Transformer 这类模型实现。对于其他的东西类似于策略，动态模型也是类似。
![](19-5.png)
一个实现细节：
对于标准的深度 Q 学习：
1. 收集转移 $(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}, r_t)$, 添加到回放缓冲区 $\mathcal{R}$；
2. 从 $\mathcal{R}$ 中采样小批次；
3. 利用小批次更新 $Q$ 函数。

但是对于带历史状态的深度 Q 学习：
1. 收集转移 $(\boldsymbol{o}_t, \boldsymbol{a}_t, \boldsymbol{o}_{t + 1}, r_t)$，并通过拼接 $\boldsymbol{o}_1,\ldots,\boldsymbol{o}_{t - 1}$，再将其添加到回放缓冲区 $\mathcal{R}$；
2. 采样小批次，但是这里每一个都是 $(\boldsymbol{o}_1,\ldots,\boldsymbol{o}_t, \boldsymbol{a}_t, \boldsymbol{o}_{t + 1}, r_t)$ 的形式；
3. 利用小批次更新 $Q$ 函数。 

然而此时保存的数据是 $O(T^2)$ 的，一个实际的做法是存储 RNN/ LSTM 的隐藏状态，例如 $\boldsymbol{o}_1,\ldots,\boldsymbol{o}_t$ 都被 $\boldsymbol{h}_t$ 所总结了。这样的技巧对于长序列的问题这表现的非常好，但是目前这还没有一个对 Transformer 的版本。
![](19-6.png)

参见：Kapturowski, Recurrent Experience Replay in Distributed Reinforcement Learning, ICLR 2019.

## 3 RL and language models
### 3.1 Language Models
本质上，语言模型是一个预测下一个词元的模型。

例如，对于输入序列，我们会先进行位置编码，然后进行自注意力和前馈网络，重复上述两个步骤 $N$ 次，最后输出一个 $softmax$ 层，从 $x_i$ 的嵌入输出 $x_{i + 1}$ 的概率。

在这里不用考虑这个语言模型的细节，我们只需要知道其实一个自回归模型，但不是马尔可夫的， 因为其需要所有的过去的词元来预测下一个词元。
![](19-7.png)

语言通常是使用自监督学习来训练的。通过这种训练方式，我们只学习了一个语言的分布，但如果我们希望其能够做到人类对齐，工具使用，实现对话目标，而不是仅仅生成符合语法的句子，仅仅依靠上述自监督的预训练是不够的，而这些任务却可以通过某种奖励函数来实现，此时就需要使用强化学习。

然而，为了应用强化学习来实现这些目标，必须回答一些问题：
- 对于一个语言模型，（部分可观测的）马尔可夫决策过程的定义是什么？ 
- 奖励是什么？ 
- 使用什么样的算法？  

### 3.2 A basic formulation
目前进行一个简单的简化，以单步问答中的对齐为例，以上下文作为状态 $\boldsymbol{s}$，而回答作为 $\boldsymbol{a}$。因此语言模型在表示 $p(\boldsymbol{a} \mid \boldsymbol{s})$，一个例子是
$$
p(\boldsymbol{a}\mid \boldsymbol{s}) = p(x_5\mid x_{1:4}) p(x_6\mid x_{1:5})
$$
![](19-8.png)
这里需要注意的这里有两种时间步，对于语言模型的时间步和强化学习的时间步。在上述简化的建模中，我们的强化学习只有一个时间步。

可以将 $p(\boldsymbol{a} \mid \boldsymbol{s})$ 作为策略 $\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})$，目标就是
$$
\mathbb{E}_{\pi_\theta(\boldsymbol{a}\mid\boldsymbol{s})}\left[r(\boldsymbol{s}, \boldsymbol{a})\right]
$$
于是我们就将对齐任务形式化成为了一个单步的强化学习问题。

### 3.3 Language models and policy gradients
考虑直接应用策略梯度：
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta(\boldsymbol{a}\mid\boldsymbol{s})}\left[\nabla_\theta \log \pi_\theta(\boldsymbol{a}\mid\boldsymbol{s}) r(\boldsymbol{s}, \boldsymbol{a})\right]
$$
自然可以通过采样来估计这个梯度。通常会使用近端策略优化（PPO）风格的方式而不是经典的 REINFORCE。

尽管二者严格上都是 on-policy 的，但是二者在使用上有一些差异：
- 在传统 REINFORCE 类方法中，收集样本的策略和当前的策略必须完全一致，严格来说即使我们仅仅采取了一个梯度步长，也需要重新收集数据。对于语言模型来说，进行采样需要很高的计算成本；
- 在近端策略优化类方法中，在近似地优化替代优势，只要收集样本的策略与目前的策略差异足够小，通过优化替代优势就可以有效地优化目标。因此对于收集到的样本，我们可以更新策略，相当多个梯度步长，直到其偏离收集时的策略有一定距离（更详细的推导可以参考[[Lecture 7 Advanced Policy Gradients]]一节）。

对于语言模型来说，使用近端策略优化形式的算法是更好的，这样可以大幅减小进行采样的次数，具体来说可以得到以下目标：
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i = 1}^{N} \nabla_\theta \frac{\pi_\theta(\boldsymbol{a}_i \mid \boldsymbol{s})}{\bar{\pi}(\boldsymbol{a}_i \mid \boldsymbol{s})} \log \pi_\theta(\boldsymbol{a}_{i} \mid \boldsymbol{s}_{i}) A(\boldsymbol{s}_{i}, \boldsymbol{a}_{i})
$$
其中 $\bar{\pi}$ 是收集数据的策略，$\pi_\theta$ 是当前的策略，$A(\boldsymbol{s}, \boldsymbol{a})$ 是优势函数。可运行的做法是重复以下过程：
1. 收集批次 $\mathcal{B} = \{\boldsymbol{a}_i\}, \boldsymbol{a}_i \sim \pi_\theta(\boldsymbol{a}\mid\boldsymbol{s})$；
2. 评估 $r(\boldsymbol{s}, \boldsymbol{a}_i)$；
3. 更新 $\bar{\pi} \gets \pi_\theta$；
4. 采样小批次 $\mathcal{M} \subset \mathcal{B}$；
5. 在小批次上进行近端策略优化更新，回到 4，重复 $K$ 次 ；

一个例子是 $|\mathcal{B}| = 1000, |\mathcal{M}| = 64$，也就是会定期更新 $\bar{\pi}$，并用新的策略进行采样，但是在更新前会使用原先策略收集的数据进行 $K$ 次梯度下降更新。

### 3.4 Learned rewards
上述介绍中，已经给出了使用强化学习进行语言模型训练的一个基本框架，但是我们还没有给出其中奖励函数的具体设置。

使用一个神经网络 $r_\psi(\boldsymbol{s}, \boldsymbol{a})$ 来表示奖励函数是非常自然的事情，因为为了实现例如对齐等目标，一个奖励函数不仅需要判断回答是否正确，还需要反映出接近的回答之间的差距，且这些问题可能非常广泛，需要处理开放词汇表的问题。
![](19-9.png)

### 3.5 RL from human feedback
那么如何训练这样的奖励函数呢？一个朴素的想法是收集一系列问答对，然后让人类来给这些回答打分，得到一系列问答对和对应的奖励，进而利用监督学习来训练奖励函数。

但是对于人类来说，想要打出这样定量的分数是非常困难的，尤其是那些相对主观的问题。然而对人来说，将两个表达进行比较是更加容易的。

因此可以使用偏好作为预测的目标，具体来说，对于一个问题 $\boldsymbol{s}$ 和多个可能的答案 $\boldsymbol{a}_1,\boldsymbol{a}_2$，让模型学习预测人们认为 $\boldsymbol{a}_1$ 更好的概率。但是我们想要的是能够用于强化学习的奖励函数而不是偏好，一个简单的方式是将预测用奖励函数来参数化：
$$
p(\boldsymbol{a}_1 \succ \boldsymbol{a}_2 \mid \boldsymbol{s}) = \frac{\exp(r_\psi(\boldsymbol{s}, \boldsymbol{a}_1))}{\exp(r_\psi(\boldsymbol{s}, \boldsymbol{a}_1)) + \exp(r_\psi(\boldsymbol{s}, \boldsymbol{a}_2))} = \sigma(r_\psi(\boldsymbol{s}, \boldsymbol{a}_1) - r_\psi(\boldsymbol{s}, \boldsymbol{a}_2))
$$
这里的 $\succ$ 表示偏好的关系，$\sigma$ 是 $sigmoid$ 函数。在实际训练中，依然在最大化偏好预测正确的概率，但是此时的概率是由奖励函数来表示的。

这个偏好表达式有经济学和心理学的背景（Luce's Choice Axiom），在强化学习中的应用也可也追溯到基于偏好的强化学习相关的工作。

### 3.6 Overall method: Aligning language models with RLHF
总体方法：通过强化学习从人类反馈中对齐语言模型：
1. 运行监督学习（和可能的微调）来获得初始 $\pi_\theta(\boldsymbol{a}\mid\boldsymbol{s})$；
2. 对于每一个 $\boldsymbol{s}$，生成 $K$ 个 $\boldsymbol{a}_k \sim \pi(\boldsymbol{a} \mid \boldsymbol{s})$，添加到 $\mathcal{D}$ 中；
3. 让人标注 $\boldsymbol{a}_{i,k}$ 在回答 $\boldsymbol{s}_i$ 上的偏好；
4. 利用 $\mathcal{D}$ 来训练奖励函数 $r_\psi(\boldsymbol{s}, \boldsymbol{a})$；
5. 通过强化学习使用 $r_\psi$ 来训练 $\pi_\theta$，回到 $2$（不过之后标注的数据会少很多）。

参见：
- Ziegler et al. Fine-Tuning Language Models from Human Preferences. 2019.  
- Ouyang et al. Training language models to follow instructions with human feedback. 2019.  

然而依然存在一些问题：
- 获取人类偏好可能很昂贵；
- 高估问题：如果没有反复回到第 $2$ 步，则进行的实际是离线强化学习，可能训练一段时间后表现反而下降了。这通常通过添加一项惩罚项来防止策略偏离监督学习的策略过多，也就是$$\mathbb{E}_{\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})} \left[r(\boldsymbol{s}, \boldsymbol{a})\right] - \beta D_{KL}(\pi_\theta \parallel \pi_\beta) = \mathbb{E}_{\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})} \left[r(\boldsymbol{s}, \boldsymbol{a}) + \beta \log \pi_\beta(\boldsymbol{a} \mid \boldsymbol{s}) - \beta \log \pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})\right]$$
- 奖励模型需要很好。通常我们会使用一个很大的 Transformer，本身就是一个语言模型，然后进行微调以输出奖励。

### 3.7 Aligning language models with DPO
上述做法中，我们将对齐建模为一个强化学习问题，最大化我们策略的期望奖励。但实际上通过一些数学推导，可以构建出奖励函数与其下的最优策略之间的确切关系，此时相较于先用偏好优化奖励函数再优化策略，不如直接用偏好优化策略。这就是直接偏好优化（Direct Preference Optimization，DPO）的做法。

考虑在给定奖励函数优化处理时的目标，对于每一个 $\boldsymbol{s}$，优化的是
$$
J(\pi(\cdot\mid\boldsymbol{s})) = \mathbb{E}_{\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})} \left[r(\boldsymbol{s}, \boldsymbol{a})\right] - \beta D_{KL}(\pi_\theta(\cdot \mid \boldsymbol{s}) \parallel \pi_\beta(\cdot \mid \boldsymbol{s}))
$$
从这个目标中可以构造一个拉格朗日乘子函数
$$
L(\pi(\cdot \mid \boldsymbol{s}), \lambda) = J(\pi(\cdot\mid\boldsymbol{s})) - \lambda \left(\int \pi(\boldsymbol{a} \mid \boldsymbol{s})\text{d}\boldsymbol{a} - 1\right)
$$
对 $\pi(\cdot \mid \boldsymbol{s})$ 求[[Concepts#23 变分导数 (Variational Derivative)|变分导数 (Variational Derivative)]]，就得到
$$
\frac{\partial L}{\partial \pi(\cdot\mid\boldsymbol{s})} = r(\boldsymbol{s}, \boldsymbol{a}) + \beta \log \pi_\beta(\boldsymbol{a} \mid \boldsymbol{s}) - \beta - \beta \log \pi(\boldsymbol{a} \mid \boldsymbol{s}) - \lambda = 0
$$
解得
$$
\pi^\ast(\boldsymbol{a} \mid \boldsymbol{s}) = \frac{1}{Z(\boldsymbol{s})} \exp\left(\frac{1}{\beta}r(\boldsymbol{s}, \boldsymbol{a})\right) \pi_\beta(\boldsymbol{a} \mid \boldsymbol{s})
$$
这个式子可以转化为  
$$
r(\boldsymbol{s}, \boldsymbol{a}) = \beta \log \frac{\pi(\boldsymbol{a} \mid \boldsymbol{s})}{\pi_\beta(\boldsymbol{a} \mid \boldsymbol{s})} + \beta \log Z(\boldsymbol{s})
$$
这是给定一个策略 $\pi$ 是最优策略的情况下对应的奖励函数的形式，也可理解为每一个策略隐含了一个潜在的奖励函数（可以想象为是某种行为准则）：
$$
r(\boldsymbol{s}, \boldsymbol{a}) = \beta \log \frac{\pi(\boldsymbol{a} \mid \boldsymbol{s})}{\pi_\beta(\boldsymbol{a} \mid \boldsymbol{s})} + \beta \log Z(\boldsymbol{s})
$$
我们希望这个隐含的潜在奖励函数能够最大化偏好数据的似然，也就是
$$
\mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}_1 \succ \boldsymbol{a}_2) \sim \mathcal{D}}\left[\log p(\boldsymbol{a}_1 \succ \boldsymbol{a}_2 \mid \boldsymbol{s})\right] = \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}_1 \succ \boldsymbol{a}_2) \sim \mathcal{D}}\left[\log\sigma(r(\boldsymbol{s}, \boldsymbol{a}_1) - r(\boldsymbol{s}, \boldsymbol{a}_2))\right]
$$
这用策略的形式表达，也就是在最大化
$$
\mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}_1 \succ \boldsymbol{a}_2) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi(\boldsymbol{a}_1 \mid \boldsymbol{s})}{\pi_\beta(\boldsymbol{a}_1 \mid \boldsymbol{s})} - \beta \log \frac{\pi(\boldsymbol{a}_2 \mid \boldsymbol{s})}{\pi_\beta(\boldsymbol{a}_2 \mid \boldsymbol{s})}\right)\right]
$$
整理即可得到直接偏好优化的损失函数
$$
\mathcal{L}_{DPO} = -\mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}_1 \succ \boldsymbol{a}_2) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi(\boldsymbol{a}_1 \mid \boldsymbol{s})}{\pi_\beta(\boldsymbol{a}_1 \mid \boldsymbol{s})} - \beta \log \frac{\pi(\boldsymbol{a}_2 \mid \boldsymbol{s})}{\pi_\beta(\boldsymbol{a}_2 \mid \boldsymbol{s})}\right)\right]
$$
从以上推导可以发现，直接偏好优化与基于人类反馈的强化学习在理论上想要实现的是同一个目标，但是直接偏好优化直接优化策略而不是奖励函数，因此避免了训练一个奖励模型的问题，降低了训练的成本。与此同时，这也意味着不需要进行强化学习的训练，可以避免强化学习中的训练不稳定性的问题。

## 4 Multi-step RL and language model
### 4.1 Introduction
在之前的讨论中，我们使用了一个简化的单步强化学习的情境来介绍强化学习和语言模型的结合，其中的一个就是利用强化学习进行对齐。但是实际中对话通常是多步的，如果我们希望能够进行多步的对话，达成对话的真正目标而非仅仅是单步回答符合偏好，我需要建模一个多步的强化学习问题。

除了对话之外，语言模型的许多应用都是多步的，例如我们希望训练一个能够使用命令行工具的语言模型，其需要通过多次交互来完成任务，并且根据过去的操作的结果来决定下一步的操作。

这样的多步交互问题和基于人类反馈的强化学习有着很大的不同：
- 在通常的基于人类反馈的强化学习中：
	- 状态 $\boldsymbol{s}$：问题/上下文；
	- 动作 $\boldsymbol{a}$：回答；
	- 奖励 $r(\boldsymbol{s}, \boldsymbol{a})$：来自于人类偏好数据训练的模型。
- 而在这些多步交互的强化学习中，一种可能的建模是： 
	- 动作 $\boldsymbol{a}_t$：单轮的回答；  
	- 观测 $\boldsymbol{o}_t$：当前收到的反馈（例如用户的回答，或者者是工具的输出）；
	- 状态 $\boldsymbol{s}_t$：到目前为止所有的观测和动作；  
	- 奖励：整个对话的输出，例如是否完成了任务。 

进行了多步强化学习的建模后，如何训练呢？两种可能的做法是策略梯度和基于价值的方法。

策略梯度：
在多步强化学习中，奖励通常是延迟的，且涉及到多步决策的问题，无法简单地对单步进行一个评价。通常情况下无法像基于人类反馈的强化学习那样简单地通过偏好训练一个奖励模型。
然而，对于（近端策略优化风格的）策略梯度来说，由于需要 on-policy 的数据，由于奖励模型的缺失，这意味着需要有人类时刻进行标注，尽管在某种程度上可行，但是非常昂贵。当然，如果像是工具使用这类无需人类标注的任务，则会容易许多。

基于价值的方法：
在基于价值的方法中，由于摆脱了 on-policy 的限制，可以使用一个预先准备好的数据集（例如人类相互交流的数据，或者过去部署模型时的数据）来进行训练，使用离线强化学习的方法。

这一部分我们主要讨论基于价值的方法，当然在成本较低的情况下也可以使用策略梯度类型的方法，具体方法和之前介绍的类似。

### 4.2 Design choice on time step
在基于人类反馈的强化学习中，由于奖励模型基于对一组问答的偏好，因此将上下文建模为状态，回答建模为动作，得到一个单步的强化学习问题是非常自然的。

而对于多步强化学习来说，既然奖励本身不一定和单步的问答有关，那么将单个回答建模为动作也就不再是唯一的选择了，例如我们可以将每一个词元视作一个动作。

选择 1：每个话语的时间步长
每一个提问是观测，每一个回答是动作。这和上一节讨论的设置比较类似，只是问题变成了多步，且没有单步的奖励。
![](19-10.png)
注意：
- 这是一个自然的选择，我们也会得到一个较短的时间跨度（对话通常不会太多次交互）；
- 但是问题是会有极大的动作空间，因为其中会包含模型的所有可能的回答。

选项 2：每个词元的时间步长
将每一个词元视作单一个时间步。此时回答中的每一个词元都是一个动作（这可能有点奇怪，因为在这些动作之间没有新的观测），而用户回答中的每一个词元都是一个观测。
![](19-11.png)
注意：
- 这里的一个好处是动作空间小了很多，变成只有词元的种类；
- 但是问题是我们会有一个非常长的时间跨度。

目前的研究中，尚未得出二者之间的优劣，不过我们可以分析它们的优缺点。

### 4.3 Per-utterance time step
在基于价值的方法中，我们必然要学习一个价值函数。而最一般的情况下没有一个动态模型，因而我们会学习一个 $Q$ 函数。此时其输入会是 $\boldsymbol{s}_t, \boldsymbol{a}_t$，前者是过去所有的对话信息 $\boldsymbol{o}_1,\boldsymbol{a}_1,\boldsymbol{o}_2,\boldsymbol{a}_2,\ldots,\boldsymbol{o}_{t}$，后者是候选动作 $\boldsymbol{a}_t$。

但正如前面分析的那样，在每个话语的建模方式中，候选动作是所有可能的回答，很显然不可能遍历所有的回答，因此会有两个主要的选择：
1. 单独使用一个演员网络，并且将 $Q$ 函数的输出作为一个类似于单步强化学习的奖励来训练演员。
2. 想办法“解码”出使得 Q 值最大化的动作。可能的方法是束搜索或者通过采样得到等。

关于网络的具体结构，我们可以使用两个/单个预训练的语言模型分别编码状态 $\boldsymbol{s}_t$ 和动作 $\boldsymbol{a}_t$，将得到的两个嵌入输入"真正的"预测 Q 值的网络中，这一整个可以视作是一个预测 Q 值的评论家。
![](19-12.png)

### 4.4 Per-token time step
这里为了简化起见，我们假设每一个词元都是一个单词。

这一类建模中，训练 Q 函数的方式依然是一种自举的方式：对于当前的 $Q(\boldsymbol{s}, \boldsymbol{a})$，依然通过
$$
Q(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \max_{\boldsymbol{a}'} Q(\boldsymbol{s}', \boldsymbol{a}')
$$
进行更新，但是这里区分以下两种情况：
1. 如果下一个词元是智能体选择的，那么就选择其中最大的一个，并添加上奖励，作为当前的 Q 值的目标。
2. 如果下一个词元是环境给出的，换言之这里的 $\boldsymbol{a}'$ 不由智能体决定，那么就不能用 $\arg\max$了，而是选择数据集词元对应的 Q 值，并添加上奖励，作为当前的 Q 值的目标。
![](19-13.png)
而训练结束后生成词元的方式也很类似于原先的大语言模型，只是区别在于，预测下一个词元的概率不是基于文本的真实分布，而是基于各个词元对应于 Q 值。

从上述描述中可以发现这样的训练过程变得更简单了，但是问题是时间跨度变得非常长。

### 4.5 Putting it all together
最后我们可以整理一下基于价值的方法在训练多步强化学习任务的时候的做法。
首先可以使用常见的基于价值的方法的方法，例如目标网络、经验回放缓冲区、双 Q 学习等等。

在具体算法上我们可以使用在线强化学习，也可以使用离线强化学习，但是通常在离线设置中，基于价值的方法通常会更实用.

如果使用离线强化学习的方式，也有一些值得注意的点，这些在离线强化学习已经涉及过了，我们不再赘述：
- 处理分布偏移；
- 明确的策略约束：对动作有 KL 散度的约束；
- 保守 Q 学习（CQL）风格的惩罚；
- 隐式 Q 学习（IQL）风格备份；
- 目前没有一个明确的最佳选择。

### 4.6 Some examples
例如：Human-Centric Dialog Training via Offline Reinforcement Learning
一些具体的设计选择：
- 演员-评论家+策略约束（KL 散度）；
- 奖励来自于人类的观点；
- 使用话语作为时间步。
![](19-14.png)
参见：Human-Centric Dialog Training via Offline Reinforcement Learning, Jaques et al. 2020

再例如：CHAI: A CHatbot AI for Task Oriented Dialogue with Offline Reinforcement Learning
一些具体的设计选择：
- Q 函数+CQL（且最大的 Q 值通过样本得到）；
- 奖励来自于任务（Craigslist 谈判）；
- 使用话语作为时间步。
![](19-15.png)
参见：CHAI: A CHatbot AI for Task Oriented Dialogue with Offline Reinforcement Learning, Verma et al. 2022

再例如：Offline RL for Natural Language Generation with Implicit Language Q Learning
一些具体的设计选择：
- Q 函数结合 IQL 与 CQL（IQL 备份 + CQL 惩罚）（同样最大的 Q 值通过样本得到）；
- 使用行为克隆策略网络进行策略提取（从值函数到实际策略，需避免之前提到的查询分布外动作）；
- 奖励来自于任务（视觉对话）；
- 使用词元作为时间步。
![](19-16.png)
参见：Offline RL for Natural Language Generation with Implicit Language Q Learning, Snell et al. 2022

## 5 Summary
在本节中，我们
- 介绍了 部分可观测马尔可夫决策过程 的相关概念，以及其相较于 马尔可夫决策过程 的不同之处。
	- 分析了常见的各种强化学习算法能否直接应用于 部分可观测马尔可夫决策过程 的问题，以及通过状态空间模型以及历史状态来解决 部分可观测马尔可夫决策过程 的问题。
- 介绍了如何将 RL 和语言模型结合起来，通过 RLHF 和 DPO 等算法来进行对齐。
- 介绍了如何将 RL 和语言模型结合起来，通过建模多步 RL 来进行多步的对话，以及工具使用等任务。
	- 这里主要介绍了基于价值的方法，给出了两种不同的时间步选择，以及在这两种情况下的训练方法。
- 介绍了不同时间步选择下的一些具体设计选择以及实际的例子.