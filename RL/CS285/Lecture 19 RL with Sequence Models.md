在本节中，我们将首先介绍 [POMDP](https://zhida.zhihu.com/search?content_id=256059320&content_type=Article&match_order=1&q=POMDP&zhida_source=entity) 的基本概念以及与普通 MDP 的差异, 通常的 RL 算法能否处理 partial observability 的情况, 以及 [state space model](https://zhida.zhihu.com/search?content_id=256059320&content_type=Article&match_order=1&q=state+space+model&zhida_source=entity) 和 [history states](https://zhida.zhihu.com/search?content_id=256059320&content_type=Article&match_order=1&q=history+states&zhida_source=entity) 等处理 POMDP 的方法. 在此之后, 我们将介绍 RL 与 language model 如何结合, 如将 alignment 建模成一个单步的 RL 问题, 以及 [RLHF](https://zhida.zhihu.com/search?content_id=256059320&content_type=Article&match_order=1&q=RLHF&zhida_source=entity), DPO 等方法. 最后我们将介绍如何将 LM 中的一些任务建模为 multi-step RL, 并且使用 [value-based methods](https://zhida.zhihu.com/search?content_id=256059320&content_type=Article&match_order=1&q=value-based+methods&zhida_source=entity) 来处理.

## 1 Partial observed MDP

### 1.1 Introduction

回顾在 MDP 中, states 满足的一个核心性质是 [Markov property](https://zhida.zhihu.com/search?content_id=256059320&content_type=Article&match_order=1&q=Markov+property&zhida_source=entity), 也就是给定当前的 state, 过去的状态与未来的状态是独立的, 记作 $\boldsymbol{s}_{t + 1} \perp \boldsymbol{s}_{1:t - 1} \mid \boldsymbol{s}_t$.

然而在 POMDP 中, 我们失去的一个核心就是 observation 不再具有 Markov property, 一个直观的理解方式是, 仅仅依靠当前观测不能完全描述状态, 或者说过去的观测是有用的.

为什么我们需要考虑 POMDP 呢? 事实上, 绝大多数现实中的问题都是 POMDP.

-   当然在一些特定情境例如 Atari 游戏中, 我们可能的确有一些信息遗漏了, 但是绝大多数信息还是可以通过当前的观测得到, 因此我们可以将其视作 MDP 处理.  
    
-   但是在很多更加实际的情境中, 可能有非常多的信息观测不到:  
    

-   在驾驶的任务中, 如果仅凭视觉观察可能无法获取车辆周围的所有信息.  
    
-   在 video games 中, 一些过去的信息也重要, 例如 minecraft 中, 我们需要记住自己做过什么事情, 这些事情影响了我们之后的 actions 是否有效.  
    
-   在与人类交互的任务中, 人类的内心状态也是无法观测到的.  
    
-   在对话中, 每一次的收到的回答是 observation, 但是对话的上下文影响着 observation 背后的含义.

![](https://picx.zhimg.com/v2-d56c60e0cd63fe08ab02f8fbd56612a1_1440w.jpg)

现实中的绝大多数问题都是 partially observed

### 1.2 Partially observed MDPs can be weird

在 POMDP 中, 可能会发生一些在 MDP 中不会发生的事, 这些可能是很奇怪的, 我们会从以下两个例子来展示.

**Example 1**. _information-gathering action_

_在 partially observed 的情境下, 一些不造成任何真实 state 改变的 action, 也无法获得 reward 的 action 可能是 optimal 的. 例如那些可能告诉我们更多有用信息的行为._

_对于走出一个迷宫的任务, 如果我们知道迷宫的形式, 那么我们自然可以将其当作普通 MDP 处理, 将位置信息作为 state, 设计合理的 reward 信号, 然后运行 RL 算法, 最终我们会得到一个 policy, 这个 policy 会告诉我们在每个位置应该怎么走才能尽快走出迷宫._

_但是如果我们要得到一个能够走出所有迷宫的 policy, 换言之我们不知道迷宫具体形式时, 我们需要建模一个 POMDP 问题, 其中 state 还包括了是迷宫的 configuration. 此时如果我们选择爬到迷宫的顶上俯瞰迷宫 (假设规则允许), 虽然本身不改变我们的 state, 不会带来 reward, 但是我们可以获得更多的关于 state 的信息, 从而更好地走出迷宫._

![](https://pic4.zhimg.com/v2-79503652571f98a80eef3ecca680dd49_1440w.jpg)

partially observed 的情况下, 一些不会带来任何 reward 的行为可能是 optimal 的

**Example 2**. _stochastic optimal policies_

_在完全观测的情况下, 始终存在一个 deterministic 的 optimal policy (可能也同时有 optimal 的随机 policy), 但是在 POMDP 中, 完全有可能不存在 deterministic 的 optimal policy._

_一个很简单的情况是考虑 $\mathcal{S} = \{A,B,C\}$, 表示从左到右三个网格, 但它们对应于同一个观测 $\boldsymbol{o}$. action space $\mathcal{A} = \{l, r\}$ 是向左向右. 且目标是到达位于中间的 $B$ 状态._

_由于在观测看来哪里都是一样的, 因此如果是 deterministic policy, 则要么永远向左或者永远向右, 这意味着总有 $A,C$ 中的一个位置让我们永远到不了 $B$, 但是一个随机策略可以让我们最终到达 $B$._

![](https://pic3.zhimg.com/v2-0de8dd5e655f6b818400d0b81b94db82_1440w.jpg)

在这个问题中不存在 deterministic 的 optimal policy

## 2 Handling Partial Observability

### 2.1 Directly use observations as states

我们能否直接将 observation 当作 state 来处理呢? 这里考虑仅仅利用当前的 observation 也就是 **memoryless policy** $\pi(\boldsymbol{a} \mid \boldsymbol{o})$. 我们可以回顾过去介绍的 RL 算法, 考虑它们能否直接应用在 partially observed 的情境下.

### 2.1.1 policy gradient

我们可以直接使用 observation 当作 state 处理, 这在 **policy gradient** 一节中我们已经推导过了:

_Proof._ 不妨记整条轨迹为 $\tau_{\boldsymbol{o}}$, 那么利用链式法则, 我们可以得到 $\begin{aligned} \log p(\tau_{\boldsymbol{o}}) &= \log p(\boldsymbol{o}_1) + \log p(\boldsymbol{a}_1 \mid \boldsymbol{o}_1) + \log p(\boldsymbol{o}_2 \mid \boldsymbol{o}_1, \boldsymbol{a}_1) + \log p(\boldsymbol{a}_2 \mid \boldsymbol{o}_{1:2}, \boldsymbol{a}_1) + \cdots\\ &= \log p(\boldsymbol{o}_1) + \sum_{t} \log p(\boldsymbol{a}_t \mid \boldsymbol{o}_{1:t}, \boldsymbol{a}_{1:t - 1}) + \sum_{t} \log p(\boldsymbol{o}_{t + 1} \mid \boldsymbol{o}_{1:t}, \boldsymbol{a}_{1:t})\\ &= \log p(\boldsymbol{o}_1) + \sum_{t} \log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{o}_t) + \sum_{t} \log p(\boldsymbol{o}_{t + 1} \mid \boldsymbol{o}_{1:t}, \boldsymbol{a}_{1:t}), \end{aligned} \\$ 由于和 $\theta$ 有关的只有 policy, 于是对 $\theta$ 求梯度有 $\nabla_\theta \log p(\tau_{\boldsymbol{o}}) = \sum_{t} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{o}_t).\\$ 于是就有 $\nabla_\theta J(\theta) = \mathbb{E}_{p(\tau_{\boldsymbol{o}})}\left[\nabla_\theta p(\tau_{\boldsymbol{o}}) r(\tau_{\boldsymbol{o}})\right]\approx \frac{1}{N} \sum_{i = 1}^N \left(\sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{o}_{i,t})\right) \left(\sum_{t = 1}^T r(\boldsymbol{o}_{i,t}, \boldsymbol{a}_{i,t})\right).\\$ ◻

简单来说, 上述过程仅利用了链式法则没有基于 Markov property, 因此可以直接使用 observation 替换 state.

然而, 要注意 advantage function 的选择. 对于 **Monte Carlo estimation**, 根据上述推导是没有问题的. 但是使用 **Temporal Difference (TD)** 的方法就不行了, 也就是我们不能使用 $r_t + \gamma\hat{V}(\boldsymbol{o}_{t + 1}) - \hat{V}(\boldsymbol{o}_t)$ 这类 estimation 作为 advantage.

为什么不行呢? 直观来说, 我们的 value function 作为关于 $\boldsymbol{s}_t$ 的函数, 其不应当依赖过去的 $\boldsymbol{s}_{t - 1}$. 但是当使用 observation 直接代替 state 时则会产生问题: 我们并不能训练一个 $\hat{V}(\boldsymbol{o}_t)$, 因为单个 $\boldsymbol{o}_t$ 不足以描述 state, 不同的过去可能导致同一个 $\boldsymbol{o}_t$ 对应于不同的 value.

**Side Note:** 不妨思考以下问题:

1.  用 observation 代替 state 时还可以应用 causality 吗?  
    
2.  可以使用 $V(\boldsymbol{o}_t)$ 作为 **baseline** 吗?  
    

这两个做法其实都是可以的. 对于 causality, 我们同样没有利用 Markov property, 具体证明可以参考 **policy gradient** 一节中的推导. 对于 value function 作为 baseline, 根据类似的推导可知只要 baseline 不依赖于 action, 使用什么 baseline 都是无偏的.

### 2.1.2 value-based methods

在这一类方法中, 我们会有 value function, 通过以下的方式更新: $Q^\ast(\boldsymbol{s}, \boldsymbol{a}) \gets r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \max_{\boldsymbol{a}'} Q(\boldsymbol{s}', \boldsymbol{a}'),\\$ 将这里的 state 替换为 observation 是不行的, 背后的原因和不能使用 TD estimation 作为 advantage 的原因是一样的. 因为同一个 observation 并不足以描述 state, 也不足以确定一个 value.

### 2.1.3 [model-based RL](https://zhida.zhihu.com/search?content_id=256059320&content_type=Article&match_order=1&q=model-based+RL&zhida_source=entity) methods

在这一类方法中, 我们会学习一个 dynamic model $\hat{p}(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a}),\\$ 将这里的 state 替换为 observation 也是不行的. 考虑以下环境:

**Example 3**. _我们考虑如下的情境: 我们面前有两扇关闭的门, 其中有且仅有一扇门是可以打开的. 考虑问题的建模:_

-   _state: $\mathcal{S} = \{l,r\}$, 表示哪一扇门是真的能够打开的_  
    
-   _action: $\mathcal{A} = \{change, open\}$, 表示换一扇门或者尝试打开_  
    
-   _observation: $\mathcal{O} = \{pass, left, right\}$, 表示结果或者面对的门是哪一扇._  
    

_在每一 episode 开始时, 会随机产生一个保持不变的 state. 在经过多个 episode 训练后, model 会学到接近于以下的 "dynamics": $p(\boldsymbol{o}' = pass \mid \boldsymbol{o} = left, \boldsymbol{a} = open) = 0.5,\\$ 也就是当我们站在左侧的门时, 尝试打开门有 $0.5$ 的概率成功. 这似乎是合理的. 但其实不然, 这意味着无论真实 state 如何, 每次尝试开门都有 $0.5$ 概率门打开. 因而最终学到的 policy 就是反复尝试开同一扇门, 这显然是不对的._

_事实上我们目前学习的 dynamic 无法 capture 如果尝试开某扇门打不开, 那么无论尝试多少遍也打不开这一点. 换言之, 这样一个 Markovian 的 model $p(\boldsymbol{o}'\mid \boldsymbol{o}, \boldsymbol{a})$ 无法表示 Non-Markovian (对于 observation 来说) 的 environment._

上述的讨论中我们将使用的模型局限在了 memoryless 一类, 这样的限制并不一定合理, 在现实中为了解决这些 non-Markovian 的问题, 我们可能会使用 observation history 作为输入.

### 2.2 State space model

在 variational inference 一节中, 我们介绍了一个处理 partial observability 的方法, 也就是使用 **state space model**, 具体来说, 一个 practical choice 是使用一个 VAE, 其中的 $\boldsymbol{x}$ 是一连串的 observation, $\boldsymbol{z}$ 是 state, 具体来说, 我们可以有设计:

### Prior:

我们希望 prior 不再是常规 VAE 中那样的各维度相互独立, 而是具有隐含的 dynamics, 也就是 $p(\boldsymbol{z}) = p(\boldsymbol{z}_1) \prod_{t} p(\boldsymbol{z}_{t + 1} \mid \boldsymbol{z}_t, \boldsymbol{a}_t).\\$ 我们可以使用 $p(\boldsymbol{z}_1) = \mathcal{N}(0, I)$, 而 $p(\boldsymbol{z}_{t + 1} \mid \boldsymbol{z}_t, \boldsymbol{a}_t)$ 则是学习得到的.

### Decoder:

我们希望 decoder 处理各时间步的观测是独立的, 也就是 $p(\boldsymbol{o} \mid \boldsymbol{z}) = \prod_{t} p(\boldsymbol{o}_t \mid \boldsymbol{z}_t).\\$

### Encoder:

我们通常不能假设各时间步的信息是独立的, 因为单个观测实际上不足以表示整个 hidden state. 我们在 model-based RL 中讨论过了多种选择, 其中一个是 $q_\phi(\boldsymbol{z} \mid \boldsymbol{o}) = \prod_{t} q_\phi(\boldsymbol{z}_t \mid \boldsymbol{o}_{1:t}).\\$ 我们通常可以利用 LSTM, transformer 等序列式模型来表示 $q_\phi(\boldsymbol{z}_t \mid \boldsymbol{o}_{1:t})$.

这样的结构设计如何进行 learning 可以参考 "Dream to Control: Learning Behaviors by Latent Imagination, Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi".

**Remark:** 这种方法属于一种 model-based 的方法, 因为我们在 prior 中学习了 dynamic. 这样的做法也存在局限性. 在一些情况下进行下一个 observation 的 prediction 可能很困难且并不必要, 生成这些 observation (例如图片), 可能比得到 optimal policy 更困难.

### 2.3 history states

一个替代方案是使用 history states, 考虑定义 $\boldsymbol{s}_t = (\boldsymbol{o}_1, \boldsymbol{o}_2, \ldots, \boldsymbol{o}_t)$, 这样定义的 state 满足 Markov property $\boldsymbol{s}_{t + 1} \bot \boldsymbol{s}_{t - 1} \mid \boldsymbol{s}_t$, 因为 $\boldsymbol{s}_t$ 包含了 $\boldsymbol{s}_{t - 1}$ 包含的所有信息, 因此 $\boldsymbol{s}_{t + 1}$ 与 $\boldsymbol{s}_{t - 1}$ 是条件独立的.

使用这样的 history observation 作为 state 是可行的, 因此通过类似 $Q(\boldsymbol{o}_1,\ldots,\boldsymbol{o}_t, \boldsymbol{a}) \gets r(\boldsymbol{o}_t, \boldsymbol{a}) + \gamma \max_{\boldsymbol{a}'} Q(\boldsymbol{o}_1,\ldots,\boldsymbol{o}_{t + 1}, \boldsymbol{a}'),\\$ 的做法, 可以在 POMDP 中应用 value-based 算法. 但是注意到这里的 Q function 需要能够处理不同长度的 history, 我们需要设计特定的 model architecture.

### choice 1:

我们可以考虑把 history image 堆叠起来并输入 Q function, 这对于较短的 history 是可行的, 对于 history 很长的情况, 我们可以以 heuristic 的方式仅保留最近的几个 observation.

能否采用 heuristic 的方式取决于问题的具体设定, 例如如果我们想要记住我们爬上迷宫时的所有观测, 并基于这些观测来决定下一步怎么走, 那么仅采用最近的几个 observation 是不够的.

![](https://pic2.zhimg.com/v2-079576641ff0cda7e9a8612cde8060d9_1440w.jpg)

将连续几个 history observations 作为输入

### choice 2:

最通用的方式是使用一个 sequence model. 以 Q-function 为例, 模型会在最后一个 observation 上输出 Q value. 这可以通过 RNN, LSTM, Transformer 这类模型实现. 对于其他的东西类似于 policy, dynamic model 也是类似.

![](https://pic1.zhimg.com/v2-17b208d1828a1e20854ae323b023a416_1440w.jpg)

使用序列式模型来处理多个 observations

### A practical detail

对于标准的 deep Q-learning:

1.  收集 transition $(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}, r_t)$, 添加到 replay buffer $\mathcal{R}$  
    
2.  从 $\mathcal{R}$ 中采样 minibatch.  
    
3.  利用 minibatch 更新 Q function.  
    

但是对于 deep Q-learning with history states:

1.  收集 transition $(\boldsymbol{o}_t, \boldsymbol{a}_t, \boldsymbol{o}_{t + 1}, r_t)$, 并通过拼接 $\boldsymbol{o}_1,\ldots,\boldsymbol{o}_{t - 1}$, 再将其添加到 replay buffer $\mathcal{R}$  
    
2.  sample minibatch, 但是这里每一个都是 $(\boldsymbol{o}_1,\ldots,\boldsymbol{o}_t, \boldsymbol{a}_t, \boldsymbol{o}_{t + 1}, r_t)$ 的形式  
    
3.  利用 minibatch 更新 Q function.  
    

然而此时我们保存的数据是 $O(T^2)$ 的, 一个实际的做法是存储 RNN/ LSTM 的 hidden state, 例如 $\boldsymbol{o}_1,\ldots,\boldsymbol{o}_t$ 都被 $\boldsymbol{h}_t$ 所总结了. 这样的 trick 对于长序列的问题这表现的非常好, 但是目前这还没有一个对 transformer 的版本.

![](https://pic3.zhimg.com/v2-5898085787bcf8115e96a5c44a5e676e_1440w.jpg)

保存 hidden states 来节省空间和避免重复计算

参见: Kapturowski, Recurrent Experience Replay in Distributed Reinforcement Learning, ICLR 2019.

## 3 RL and language models

### 3.1 Language Models

本质上, language model 是一个预测 next token 的模型.

**Example 4**. _对于输入序列, 我们会先进行 positional encoding, 然后进行 self-attention 和 feedforward network, 重复上述两个步骤 $N$ 次, 最后输出一个 softmax 层, 从 $x_i$ 的 embedding 输出 $x_{i + 1}$ 的概率._

在这里我们不用考虑这个 language model 的细节, 我们只需要知道其实一个自回归模型, 但不是 Markovian 的, 因为其需要所有的过去的 token 来预测下一个 token.

![](https://pic4.zhimg.com/v2-599a61b4a17beac9943711cfbc301887_1440w.jpg)

Language Model 的基本概念

language model 通常是使用自监督学习来训练的. 通过这种训练方式, 我们只学习了一个语言的分布, 但如果我们希望其能够做到 human alignment, tool use, 实现 dialog goal, 而不是仅仅生成符合语法的句子, 仅仅依靠上述自监督的预训练是不够的. 而这些任务却可以通过某种 reward function 来实现, 此时我们就需要使用 RL.

然而, 为了应用 RL 来实现这些目标, 我们必须回答一些问题:

-   对于一个 language model, (PO) MDP 的定义是什么?  
    
-   reward 是什么?  
    
-   使用什么样的算法?  
    

### 3.2 A basic formulation

目前我们进行一个简单的简化, 以单步问答中的 alignment 为例, 以上下文作为 state $\boldsymbol{s}$, 而回答作为 $\boldsymbol{a}$. 因此我们我们的 LM 在表示 $p(\boldsymbol{a} \mid \boldsymbol{s})$, 一个例子是 $p(\boldsymbol{a}\mid \boldsymbol{s}) = p(x_5\mid x_{1:4}) p(x_6\mid x_{1:5})\\$

![](https://picx.zhimg.com/v2-8bab715456a2fd96728d21fadff8fbdf_1440w.jpg)

将单步问答转化为单步 RL 的 formulation

这里需要注意的这里有两种 time step, 对于 LM 的 time step 和 RL 的 time step, 在上述简化的建模中, 我们的 RL 只有一个 time step.

我们可以将 $p(\boldsymbol{a} \mid \boldsymbol{s})$ 作为我们的 policy $\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})$, 我们的 objective 就是 $\mathbb{E}_{\pi_\theta(\boldsymbol{a}\mid\boldsymbol{s})}\left[r(\boldsymbol{s}, \boldsymbol{a})\right].\\$ 于是我们就将 alignment 任务 formalize 成为了一个单步的 RL problem.

### 3.3 Language models and policy gradients

我们考虑直接应用 policy gradient: $\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta(\boldsymbol{a}\mid\boldsymbol{s})}\left[\nabla_\theta \log \pi_\theta(\boldsymbol{a}\mid\boldsymbol{s}) r(\boldsymbol{s}, \boldsymbol{a})\right].\\$

我们自然可以通过 sample 来估计这个梯度. 通常我们会使用 PPO-style 的方式而不是经典的 REINFORCE:

尽管二者严格上都是 on-policy 的, 但是二者在使用上有一些差异:

-   在传统 REINFORCE 类方法中, 我们的收集 sample 的 policy 和当前的 policy 必须完全一致, 严格来说即使我们仅仅采取了一个 gradient step, 我们也需要重新收集数据. 对于 LM 来说, 进行 sampling 需要很高的计算成本.  
    
-   在 PPO 类方法中, 我们在近似地优化 surrogate advantage, 只要我们收集 samples 的 policy 与目前的 policy 差异足够小, 通过优化 surrogate advantage 就可以有效地优化 objective. 因此对于收集到的 samples, 我们可以更新 policy 相当多个 gradient steps 直到其偏离收集时的 policy 有一定距离. (更详细的推导可以参考 **Advanced Policy Gradient** 一节)  
    

对于 LM 来说, 使用 PPO 形式的算法是更好的, 这样可以大幅减小进行 sampling 的次数, 具体来说我们可以得到以下 objective: $\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i = 1}^{N} \nabla_\theta \frac{\pi_\theta(\boldsymbol{a}_i \mid \boldsymbol{s})}{\bar{\pi}(\boldsymbol{a}_i \mid \boldsymbol{s})} \log \pi_\theta(\boldsymbol{a}_{i} \mid \boldsymbol{s}_{i}) A(\boldsymbol{s}_{i}, \boldsymbol{a}_{i}),\\$ 其中 $\bar{\pi}$ 是收集数据的 policy, $\pi_\theta$ 是当前的 policy, $A(\boldsymbol{s}, \boldsymbol{a})$ 是 advantage function. 我们可运行的做法是重复以下过程:

1.  收集 batch $\mathcal{B} = \{\boldsymbol{a}_i\}, \boldsymbol{a}_i \sim \pi_\theta(\boldsymbol{a}\mid\boldsymbol{s})$  
    
2.  evaluate $r(\boldsymbol{s}, \boldsymbol{a}_i)$  
    
3.  更新 $\bar{\pi} \gets \pi_\theta$  
    
4.  sample minibatch $\mathcal{M} \subset \mathcal{B}$  
    
5.  再 minibatch 上进行 PPO update, 回到 4, 重复 $K$ 次  
    

一个例子是 $|\mathcal{B}| = 1000, |\mathcal{M}| = 64$, 也就是我们会定期更新 $\bar{\pi}$, 并用新的 policy 进行采样, 但是在更新前我们会使用原先 policy 收集的数据进行 $K$ 次梯度下降更新.

### 3.4 Learned rewards

上述介绍中, 我们已经给出了使用 RL 进行 LM 训练的一个基本框架, 但是我们还没有给出其中 reward function 的具体设置.

使用一个神经网络 $r_\psi(\boldsymbol{s}, \boldsymbol{a})$ 来表示 reward function 是非常自然的事情, 因为为了实现例如 alignment 等目标, 一个 reward function 不仅需要判断回答是否正确, 还需要反映出接近的回答之间的差距, 且这些问题可能非常广泛, 需要处理 open-vocabulary 的问题.

![](https://pic2.zhimg.com/v2-64c892947fc7a2a0f6aff7a314cd0855_1440w.jpg)

学习一个可以给 question-answer (state-action) 打分的 reward function, 其需要有很强的 generalization 能力

### 3.5 RL from human feedback

那么我们我们如何训练这样的 reward function 呢? 一个 naive 的想法是收集一系列 question-answer pairs, 然后让人类来给这些回答打分, 我们得到了一系列 question-answer pairs 和对应的 reward. 进而利用监督学习来训练 reward function.

但是对于人类来说, 想要打出这样定量的分数是非常困难的, 尤其是那些相对主观的问题. 然而对人来说, 将两个表达进行比较是更加容易的.

因此我们可以使用 preference 作为预测的目标, 具体来说, 对于一个 question $\boldsymbol{s}$ 和多个可能的 answer $\boldsymbol{a}_1,\boldsymbol{a}_2$, 我们让模型学习预测人们认为 $\boldsymbol{a}_1$ 更好的概率. 但是我们想要的是能够用于 RL 的 reward function 而不是 preference, 一个简单的方式是将预测用 reward function 来参数化:

$p(\boldsymbol{a}_1 \succ \boldsymbol{a}_2 \mid \boldsymbol{s}) = \frac{\exp(r_\psi(\boldsymbol{s}, \boldsymbol{a}_1))}{\exp(r_\psi(\boldsymbol{s}, \boldsymbol{a}_1)) + \exp(r_\psi(\boldsymbol{s}, \boldsymbol{a}_2))} = \sigma(r_\psi(\boldsymbol{s}, \boldsymbol{a}_1) - r_\psi(\boldsymbol{s}, \boldsymbol{a}_2)).\\$ 这里的 $\succ$ 表示 preference 的关系, $\sigma$ 是 sigmoid 函数. 在实际训练中, 我们依然在最大化偏好预测正确的概率, 但是我们此时的概率是由 reward function 来表示的.

**Side Note:** 这个偏好表达式有经济学和心理学的背景 (Luce's Choice Axiom), 在 RL 中的应用也可也追溯到 preference-based RL 相关的工作.

### 3.6 Overall method: Aligning language models with RLHF

1.  运行监督学习 (和可能的 finetuning) 来获得初始 $\pi_\theta(\boldsymbol{a}\mid\boldsymbol{s})$  
    
2.  对于每一个 $\boldsymbol{s}$, 生成 $K$ 个 $\boldsymbol{a}_k \sim \pi(\boldsymbol{a} \mid \boldsymbol{s})$, 添加到 $\mathcal{D}$ 中.  
    
3.  让人标注 $\boldsymbol{a}_{i,k}$ 在回答 $\boldsymbol{s}_i$ 上的偏好  
    
4.  利用 $\mathcal{D}$ 来训练 reward function $r_\psi(\boldsymbol{s}, \boldsymbol{a})$  
    
5.  通过 RL 使用 $r_\psi$ 来训练 $\pi_\theta$, 回到 $2$ (不过之后标注的数据会少很多).  
    

参见:

-   Ziegler et al. Fine-Tuning Language Models from Human Preferences. 2019.  
    
-   Ouyang et al. Training language models to follow instructions with human feedback. 2019.  
    

然而依然存在一些 issues:

-   获取 human preference 可能很昂贵.  
    
-   **overestimation problem**: 如果我们没有反复回到第 $2$ 步, 则进行的实际是 offline RL. 可能训练一段时间后表现反而下降了. 这通常通过添加一项惩罚项来防止我们的 policy 偏离监督学习的 policy 过多. 也就是 $\mathbb{E}_{\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})} \left[r(\boldsymbol{s}, \boldsymbol{a})\right] - \beta D_{KL}(\pi_\theta \parallel \pi_\beta) = \mathbb{E}_{\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})} \left[r(\boldsymbol{s}, \boldsymbol{a}) + \beta \log \pi_\beta(\boldsymbol{a} \mid \boldsymbol{s}) - \beta \log \pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})\right]\\$
-   reward model 需要很好, 通常我们会使用一个很大的 transformer, 本身就是一个 language model, 然后进行 finetune 以输出 reward.  
    

### 3.7 Aligning language models with DPO

上述做法中, 我们将 alignment 建模为一个 RL 问题, 最大化我们 policy 的期望 reward. 但实际上通过一些数学的 derivation, 我们可以构建出 reward function 与其下的 optimal policy 之间的确切关系, 此时相较于先用 preference 优化 reward function 再优化 policy, 不如直接用 preference 优化 policy. 这就是 **DPO (Direct Preference Optimization)** 的做法.

考虑在给定 reward function 优化 policy 时的 objective, 对于每一个 $\boldsymbol{s}$, 我们优化的是 $J(\pi(\cdot\mid\boldsymbol{s})) = \mathbb{E}_{\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})} \left[r(\boldsymbol{s}, \boldsymbol{a})\right] - \beta D_{KL}(\pi_\theta(\cdot \mid \boldsymbol{s}) \parallel \pi_\beta(\cdot \mid \boldsymbol{s})),\\$ 从这个 objective 中可以构造一个拉格朗日乘子函数, $L(\pi(\cdot \mid \boldsymbol{s}), \lambda) = J(\pi(\cdot\mid\boldsymbol{s})) - \lambda \left(\int \pi(\boldsymbol{a} \mid \boldsymbol{s})\text{d}\boldsymbol{a} - 1\right),\\$ 对 $\pi(\cdot \mid \boldsymbol{s})$ 求 **变分导数**, 就得到 $\frac{\partial L}{\partial \pi(\cdot\mid\boldsymbol{s})} = r(\boldsymbol{s}, \boldsymbol{a}) + \beta \log \pi_\beta(\boldsymbol{a} \mid \boldsymbol{s}) - \beta - \beta \log \pi(\boldsymbol{a} \mid \boldsymbol{s}) - \lambda = 0.\\$ 解得 $\pi^\ast(\boldsymbol{a} \mid \boldsymbol{s}) = \frac{1}{Z(\boldsymbol{s})} \exp\left(\frac{1}{\beta}r(\boldsymbol{s}, \boldsymbol{a})\right) \pi_\beta(\boldsymbol{a} \mid \boldsymbol{s}).\\$  
这个式子可以转化为  
$r(\boldsymbol{s}, \boldsymbol{a}) = \beta \log \frac{\pi(\boldsymbol{a} \mid \boldsymbol{s})}{\pi_\beta(\boldsymbol{a} \mid \boldsymbol{s})} + \beta \log Z(\boldsymbol{s}),\\$  
这是给定一个 policy $\pi$ 是 optimal policy 的情况下对应的 reward function 的形式, 也可理解为每一个 policy 隐含了一个 latent 的 reward function (你可以想象为是某种行为准则): $r(\boldsymbol{s}, \boldsymbol{a}) = \beta \log \frac{\pi(\boldsymbol{a} \mid \boldsymbol{s})}{\pi_\beta(\boldsymbol{a} \mid \boldsymbol{s})} + \beta \log Z(\boldsymbol{s}),\\$ 我们希望这个隐含的 latent reward function 能够最大化 preference 数据的似然, 也就是 $\mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}_1 \succ \boldsymbol{a}_2) \sim \mathcal{D}}\left[\log p(\boldsymbol{a}_1 \succ \boldsymbol{a}_2 \mid \boldsymbol{s})\right] = \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}_1 \succ \boldsymbol{a}_2) \sim \mathcal{D}}\left[\log\sigma(r(\boldsymbol{s}, \boldsymbol{a}_1) - r(\boldsymbol{s}, \boldsymbol{a}_2))\right],\\$ 这用 policy 的形式表达, 也就是在最大化 $\mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}_1 \succ \boldsymbol{a}_2) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi(\boldsymbol{a}_1 \mid \boldsymbol{s})}{\pi_\beta(\boldsymbol{a}_1 \mid \boldsymbol{s})} - \beta \log \frac{\pi(\boldsymbol{a}_2 \mid \boldsymbol{s})}{\pi_\beta(\boldsymbol{a}_2 \mid \boldsymbol{s})}\right)\right].\\$ 整理即可得到 DPO 的损失函数 $\mathcal{L}_{DPO} = -\mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}_1 \succ \boldsymbol{a}_2) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi(\boldsymbol{a}_1 \mid \boldsymbol{s})}{\pi_\beta(\boldsymbol{a}_1 \mid \boldsymbol{s})} - \beta \log \frac{\pi(\boldsymbol{a}_2 \mid \boldsymbol{s})}{\pi_\beta(\boldsymbol{a}_2 \mid \boldsymbol{s})}\right)\right].\\$从以上推导可以发现, DPO 与 RLHF 在理论上想要实现的是同一个目标, 但是 DPO 直接优化 policy 而不是 reward function, 因此避免了训练一个 reward model 的问题, 降低了训练的成本. 与此同时, 这也意味着我们不需要进行 RL 的训练, 可以避免 RL 中的训练不稳定性的问题.

## 4 Multi-step RL and language model

### 4.1 Introduction

在之前的讨论中, 我们使用了一个简化的单步 RL 的情境来介绍 RL 和 language model 的结合, 其中的一个就是利用 RL 进行 alignment. 但是实际中我们的对话通常是多步的, 如果我们希望能够进行多步的对话, 达成对话的真正目标而非仅仅是单步回答符合偏好, 我们需要需要建模一个多步的 RL 问题.

除了对话之外, LM 的许多应用都是多步的, 例如我们希望训练一个能够使用命令行工具的语言模型, 其需要通过多次交互来完成任务, 并且根据过去的操作的结果来决定下一步的操作.

这样的多步交互问题和 RLHF 有着很大的不同:

-   在通常的 RLHF 中:  
    

-   state $\boldsymbol{s}$: 问题/上下文  
    
-   action $\boldsymbol{a}$: 回答  
    
-   reward $r(\boldsymbol{s}, \boldsymbol{a})$: 来自于人类偏好数据训练的模型  
    

-   而在这些多步交互的 RL 中, 一种可能的建模是:  
    

-   action $\boldsymbol{a}_t$: 单轮的回答  
    
-   observation $\boldsymbol{o}_t$: 当前收到的反馈 (例如用户的回答, 或者者是工具的输出)  
    
-   state $\boldsymbol{s}_t$: 到目前为止所有的 observation 和 action.  
    
-   reward: 整个对话的 outcome, 例如是否完成了任务.  
    

进行了多步 RL 的建模后, 我们如何训练呢? 两种可能的做法是 policy gradient 和 value-based methods.

### Policy gradient

在多步 RL 中, 我们的 reward 通常是 delayed 的, 且涉及到多步决策的问题, 我们无法简单地对单步进行一个评价. 通常情况下我们无法像 RLHF 那样简单地通过 preference 训练一个 reward model.

然而, 对于 (PPO-style 的) policy gradient 来说, 由于我们需要 on-policy 的数据, 由于 reward model 的缺失, 这意味着需要有人类时刻进行标注, 尽管在某种程度上可行, 但是非常昂贵. 当然, 如果像是 tool use 这类无需人类标注的任务, 则会容易许多.

### Value-based methods

在 value-based methods 中, 由于拜托了 on-policy 的限制, 我们可以使用一个预先准备好的数据集 (例如人类相互交流的数据, 或者过去部署模型时的数据) 来进行训练, 使用 offline RL 的方法.

这一部分我们主要讨论 value-based methods, 当然在成本较低的情况下也可以使用 policy gradient 类型的方法, 具体方法和之前介绍的类似.

### 4.2 Design choice on time step

在 RLHF 中, 由于我们的 reward model 基于对一组问答的偏好, 因此将上下文建模为 state, 回答建模为 action, 得到一个单步的 RL 问题是非常自然的.

而对于多步 RL 来说, 既然我们的 reward 本身不一定和单步的问答有关, 那么将单个回答建模为 action 也就不再是唯一的选择了, 例如我们可以将每一个 token 视作一个 action.

### choice 1: per-utterance time step

每一个提问是 observation, 每一个回答是 action. 这和我们上一节讨论的 setting 比较类似, 只是问题变成了多步, 且没有单步的 reward.

![](https://pica.zhimg.com/v2-a309669bdd26b9dea5b516690c52575c_1440w.jpg)

per-utterance time step

**Remark:**

-   这是一个自然的选择, 我们也会得到一个较短的 horizon (对话通常不会太多次交互)  
    
-   但是问题是我们会有极大的 action space, 因为其中会包含我们 model 的所有可能的回答.  
    

### choice 2: per-token time step

将每一个 token 视作单一个 time step. 此时我们回答中的每一个 token 都是一个 action (这可能有点奇怪, 因为在这些 action 之间没有新的 observation). 而用户回答中的每一个 token 都是一个 observation.

![](https://pic4.zhimg.com/v2-13502d3dab173f6494f602c8ff99bf73_1440w.jpg)

per-token time step

**Remark:**

-   这里的一个好处是我们的 action space 小了很多, 变成只有 token 的种类.  
    
-   但是问题是我们会有一个非常长的 horizon.  
    

目前的研究中, 尚未得出二者之间的优劣. 不过我们可以分析它们的优缺点.

### 4.3 per-utterance time step

在 value-based methods 中, 我们必然要学习一个 value function. 而最 general 的情况下我们没有一个 dynamic model, 因而我们会学习一个 Q function. 此时其输入会是 $\boldsymbol{s}_t, \boldsymbol{a}_t$, 前者是过去所有的对话信息 $\boldsymbol{o}_1,\boldsymbol{a}_1,\boldsymbol{o}_2,\boldsymbol{a}_2,\ldots,\boldsymbol{o}_{t}$, 后者是 candidate action $\boldsymbol{a}_t$.

但正如前面分析的那样, 在 per-utterance 的建模方式中, candidate action 是所有可能的回答, 很显然不可能遍历所有的回答. 因此我们会有两个主要的选择:

1.  单独使用一个 actor 网络, 并且将 Q function 的输出作为一个类似于单步 RL 的 reward 来训练 actor.  
    
2.  想办法 "decode" 出使得 Q value 最大化的 action. 可能的方法是 **beam search** 或者通过 samples 得到等.  
    

关于网络的具体结构, 我们可以使用两个/单个预训练的 LM 分别编码 states $\boldsymbol{s}_t$ 和 actions $\boldsymbol{a}_t$, 将得到的两个 embedding 输入 "真正的" 预测 Q value 的网络中, 这一整个可以视作是一个预测 Q value 的 critic.

![](https://picx.zhimg.com/v2-a858f4bcaa80b49473db5d0349bd4591_1440w.jpg)

critic 的整体结构

### 4.4 per-token time step

这里为了简化起见, 我们假设每一个 token 都是一个 word.

这一类建模中, 训练 Q function 的方式依然是一种 bootstrap 的方式: 对于当前的 $Q(\boldsymbol{s}, \boldsymbol{a})$, 我们依然通过 $Q(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \max_{\boldsymbol{a}'} Q(\boldsymbol{s}', \boldsymbol{a}')\\$ 进行更新, 但是这里区分以下两种情况:

1.  如果下一个 token 是 agent 选择的, 那么就选择其中最大的一个, 并添加上 reward, 作为当前的 Q value 的 target.  
    
2.  如果下一个 token 是环境给出的, 换言之这里的 $\boldsymbol{a}'$ 不由 agent 决定, 那么我们就不能用 argmax 了, 而是选择 dataset token 对应的 Q value, 并添加上 reward, 作为当前的 Q value 的 target.

![](https://pic2.zhimg.com/v2-77e24cc5eda026235b4ccebfd8c8fb5d_1440w.jpg)

进行 bootstrap 更新的方式 (这里进行的更新是 a = &#39;facing&#39; 对应的 Q function, target 会利用下一时间步最优的 action &#39;each&#39; 等的 Q 和 r 进行计算)

  

而训练结束后生成 token 的方式也很类似于原先的 LLM, 只是区别在于, 我们预测下一个 token 的概率不是基于文本的真实分布, 而是基于各个 token 对应于 Q value.

从上述描述中可以发现这样的训练过程变得更简单了, 但是问题是 horizon 变得非常长.

### 4.5 Putting it all together

最后我们可以整理一下 value-based methods 在训练 multi-step RL 任务的时候的做法.

首先我们可以使用常见的 value-based methods 的方法, 例如 **target network**, **replay buffer**, **double Q-learning** 等等.

在具体算法上我们可以使用 online RL, 也可以使用 offline RL. 但是通常在 offline setting 中, value-based methods 通常会更实用.

如果使用 offline RL 的方式, 也有一些值得注意的点, 这些在 **offline RL** 已经涉及过了, 我们不再赘述:

-   处理 distribution shift  
    
-   explicit policy constraint: 对 actor 有 KL 散度的约束  
    
-   CQL-style penalty  
    
-   IQL-style backup  
    
-   目前没有一个明确的最佳选择  
    

### 4.6 Some examples

**Example 5**. _Human-Centric Dialog Training via Offline Reinforcement Learning_

_一些具体的 design choice:_

-   _Actor-Critic + policy constraint (KL divergence)_  
    
-   _reward 来自于 human 的观点._  
    
-   _使用 utterance 作为 time step_  
    

![](https://pic3.zhimg.com/v2-f486ef242e9d736bb571099ba2124098_1440w.jpg)

Human-Centric Dialog Training via Offline Reinforcement Learning

_参见: Human-Centric Dialog Training via Offline Reinforcement Learning, Jaques et al. 2020_

**Example 6**. _CHAI: A CHatbot AI for Task Oriented Dialogue with Offline Reinforcement Learning_

_一些具体的 design choice:_

-   _Q-function + CQL (且最大的 Q value 通过 samples 得到)_  
    
-   _reward 来自于任务 (Craigslist negotiation)_  
    
-   _使用 utterance 作为 time step_

![](https://picx.zhimg.com/v2-0fa481da7f2ee4dc77cc6a89840e2a81_1440w.jpg)

CHAI: A CHatbot AI for Task Oriented Dialogue with Offline Reinforcement Learning

_参见: CHAI: A CHatbot AI for Task Oriented Dialogue with Offline Reinforcement Learning, Verma et al. 2022_

**Example 7**. _Offline RL for Natural Language Generation with Implicit Language Q Learning_

_一些具体的 design choice:_

-   _Q-function with IQL + CQL (IQL backup + CQL penalty) (同样最大的 Q value 通过 samples 得到)_  
    
-   _policy extraction (从 value function 到实际的 policy, 需要避免之前提到的 query OOD actions) with BC actor_  
    
-   _reward 来自于任务 (visual dialogue)_  
    
-   _使用 token 作为 time step_

![](https://pica.zhimg.com/v2-f4856e7587280c2caebf38fdc01e45fe_1440w.jpg)

Offline RL for Natural Language Generation with Implicit Language Q Learning

_参见: Offline RL for Natural Language Generation with Implicit Language Q Learning, Snell et al. 2022_

## 5 Summary

在本节中, 我们

-   介绍了 POMDP 的相关概念, 以及其相较于 MDP 的不同之处.  
    
-   分析了常见的各种 RL 算法能否直接应用于 POMDP 的问题, 以及通过 state space model 以及 history states 来解决 POMDP 的问题.  
    
-   介绍了如何将 RL 和 language model 结合起来, 通过 RLHF 和 DPO 等算法来进行 alignment.  
    
-   介绍了如何将 RL 和 language model 结合起来, 通过建模 multi-step RL 来进行多步的对话, 以及 tool use 等任务.  
    

-   这里主要介绍了 value-based methods, 给出了两种不同的 time step 的选择, 以及在这两种情况下的训练方法.  
    
-   介绍了不同 time step 选择下的一些具体 design choice 以及实际的例子.