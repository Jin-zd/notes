在过去的部分中，我们主要讨论了强化学习。也用一些篇幅讨论了最优控制。实际上强化学习的研究可以追溯到对人类和动物的行为研究上，我们过去讨论的方法是否正确建模了人类的行为？是否有更好地方式进行这种行为的建模？

在本节中，我们将从人类行为的特征出发，尝试将最优控制、强化学习与规划都推导为一种概率推断，并基于这一点改进强化学习算法，得出如软演员 - 评论家算法（Soft actor-critic，SAC）等算法，在下一节中，我们将讨论如何将这些想法应用到逆强化学习中。
## 1 Probabilistic models of human behavior
### 1.1 Optimal Control as a Model of Human Behavior
回顾对最优控制的讨论，当环境是确定的时，我们选取一系列动作使得奖励最大化，也就是 
$$
\boldsymbol{a}_1, \ldots, \boldsymbol{a}_T = \arg\max_{\boldsymbol{a}_1, \ldots, \boldsymbol{a}_T} \sum_{t} r(\boldsymbol{s}_t, \boldsymbol{a}_t), \text{ s.t. } \boldsymbol{s}_{t + 1} = f(\boldsymbol{s}_t, \boldsymbol{a}_t)
$$
当环境是随机的时，如果通过闭环控制规划（等价于一个策略）来进行控制，可以得到也可以在随机的情况下进行最优控制：
$$
\pi = \arg\max_\pi \mathbb{E}_{\boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t), \boldsymbol{a}_t \sim \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} \left[\sum_{t} r(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]
$$

在某种程度上，通常基于理性等假设，人类会选择最优的行为，这很像是某种最优控制。基于这一点，使用行为克隆来达到接近人类的表现是有意义的。与此同时，如果我们能够从人类的行为中恢复出奖励函数，那么就可以用这一奖励函数进行强化学习（也可以利用其推测人类行为），这是逆强化学习的基本思想。

然而，人类和动物的行为并非无时无刻最优，例如猴子将一个物体放到指定位置，显然最优的方式应当是沿着一条直线移动，但它可能会选择一个次优的策略（如路径中略有偏移）。而人类的行为也可能受到情绪等因素的影响产生细小的偏差，导致其偏离完全的最优行为，虽然通常这样的偏差并不会影响它们完成任务。

上述的分析中，我们发现重要的一点：
- 人类和其他动物的行为是随机性的，而不是完全的最优行为；
- 但是通常情况下，人类和其他动物的行为都能有效地完成任务，换言之在每一个时刻，做出好的行为的可能性更大（通常是最有可能的行为）。

### 1.2 Probabilistic Models of Human Behavior
过去介绍的线性二次型调节器（Linear Quadratic Regulator，LQR）方法等都是确定性的，这样的建模方式无法解释为什么人类会有一些随机性的行为。这里引入一个概率模型来解释这些接近最优的行为。

首先先考虑状态、动作、下一状态之间的概率图模型，也就是
![](17-1.png)

在这个模型中，不存在进行控制所需的奖励，因此我们并不能够利用这个模型来解决控制问题。

为了包含解决问题所需的奖励信，这里引入一个二元的最优性变量 $\mathcal{O}_t$，它们表示 $t$ 时间步的行为是否最优。这些最优性变量是观测变量，通常我们我们会假设 
$$
p(\mathcal{O}_t \mid \boldsymbol{s}_t, \boldsymbol{a}_t) = \exp(r(\boldsymbol{s}_t, \boldsymbol{a}_t))
$$
其中奖励是负值（可以通过减去最大奖励实现）来保证概率属于 $[0,1]$，同时为了记号的简便，我们总是用 $\mathcal{O}_t$ 来表示 $\mathcal{O}_t = 1$。
![](17-2.png)

为了理解这种定义方式的合理性，考虑 $p(\tau \mid \mathcal{O}_{1:T})$，可以化简得到 
$$
\begin{aligned} p(\tau \mid \mathcal{O}_{1:T}) = \frac{p(\tau, \mathcal{O}_{1:T})}{p(\mathcal{O}_{1:T})} \propto p(\tau, \mathcal{O}_{1:T}) &= p(\tau) p(\mathcal{O}_1 \mid \tau) \prod_{t = 2}^{T} p(\mathcal{O}_t \mid \tau, \mathcal{O}_{1:t - 1}) \\ &= p(\tau) p(\mathcal{O}_1 \mid \boldsymbol{s}_1, \boldsymbol{a}_1) \prod_{t = 2}^{T} p(\mathcal{O}_t \mid \boldsymbol{s}_t, \boldsymbol{a}_t)\\ &= p(\tau) \prod_{t = 1}^{T} p(\mathcal{O}_t \mid \boldsymbol{s}_t, \boldsymbol{a}_t)\\ &= p(\tau) \exp\left(\sum_{t = 1}^T r(\boldsymbol{s}_t, \boldsymbol{a}_t)\right)\\&\propto p(\boldsymbol{s}_1) \prod_{t = 1}^{T} p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t) \exp\left(\sum_{t = 1}^T r(\boldsymbol{s}_t, \boldsymbol{a}_t)\right). \end{aligned} 
$$
这里最后的正比可能让人困惑，但实际上是合理的，由于
$$
p(\tau) = p(\boldsymbol{s}_1) \prod_{t = 1}^{T} p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t) p(\boldsymbol{s}_t \mid \boldsymbol{a}_t)
$$
这里的 $p(\boldsymbol{s}_t \mid \boldsymbol{a}_t)$ 并不是策略，而是像先验一样的一个分布，通常假设这是均匀分布（这是因为当不知道关于目标 $\mathcal{O}_{1:T}$ 的任何信息时，我们无从知道何种行为更有可能）。

### 1.3 Analysis of the result
对确定性的情况，上述结果有一个更加直观的理解方式：
$$
p(\tau \mid \mathcal{O}_{1:T}) \propto \mathbb{I}[p(\tau) \neq 0] \exp\left(\sum_{t = 1}^{T} r(\boldsymbol{s}_t, \boldsymbol{a}_t) \right)
$$
简单来说，最优轨迹的概率最高，而次优轨迹的概率会随着奖励的减小而减小。且由于指数的存在，这一下降速度非常快，那些奖励很低（无法到达目标）的轨迹的概率会极小。更为重要的是，进行控制的过程就转化为了最大化后验概率的过程。换言之，使得后验概率 $p(\tau \mid \mathcal{O}_{1:T})$ 最大的轨迹就是能够最大化奖励的最优轨迹。

对于随机性的情况，上述最大化后验概率的方式是有问题的，我们会在本节的[[#3 Control as Variational Inference]]部分介绍。
### 1.4 Summary
为什么要研究这个模型呢？
- 这个模型能够建模次优性，能从这个模型中恢复奖励函数，这对逆强化学习非常重要；
- 前面的分析中发现，这个模型让我们能够将控制和规划转化为推断的问题； 
- 后续会发现，这个模型也解释了为什么随机性行为被偏好（这对探索与迁移学习也非常有用，越是随机，越有可能迁移到新的任务上）。

在接下来的一小节中，会具体展开如何用刚才的模型进行推断，为了进行确切的推断，假设：
1. $p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$ 是已知的，也就是知道环境的动态；
2. $p(\mathcal{O}_t \mid \boldsymbol{s}_t, \boldsymbol{a}_t) = \exp(r(\boldsymbol{s}_t, \boldsymbol{a}_t))$ 是已知的，也就是知道奖励函数。

总的来说，会有以下三种推断问题：
1. 计算反向信息：$\beta_t(\boldsymbol{s}_t, \boldsymbol{a}_t) = p(\mathcal{O}_{t:T} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$（也就是给定当前的状态和行为，当前及之后的行为是否最优）；
2. 计算策略：$p(\boldsymbol{a}_t \mid \boldsymbol{s}_t, \mathcal{O}_{1:T})$，即使在确定性的情况下，我们通常更关心如何得到一个策略而不是一个完整的规划，因此会考虑的是 $p(\boldsymbol{a}_t \mid \boldsymbol{s}_t, \mathcal{O}_{1:T})$；
3. 计算前向信息：$\alpha_t(\boldsymbol{s}_t) = p(\boldsymbol{s}_t \mid \mathcal{O}_{1:{t - 1}})$（也就是如果前面的行为是最优的，那么当前的状态的分布是什么，这对逆强化学习非常重要）。 

这里的反向信息、前向信息是概率图模型（例如隐马尔可夫模型）中的术语，它们具有不同的用处，简单来说在本节中需要用反向信息来计算策略，而前向信息可以用来计算状态边际分布，进而用来进行逆强化学习，这些会在下一节中具体讨论。

## 2 Control as Inference
在这里我们具体介绍将控制问题转化为推断问题的方法，尽管之前简要提及了在随机情况下直接最大化后验概率是有问题的，但是目前依然以随机性动态的普遍情况进行分析。
### 2.1 Backward messages
#### 2.1.1 Calculation
首先考虑反向信息，就是 $\beta_t(\boldsymbol{s}_t, \boldsymbol{a}_t) = p(\mathcal{O}_{t:T} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$，也就是给定当前的状态和行为，当前及之后的行为是否最优，首先可以进行如下的转化：
$$
\begin{aligned} \beta_t(\boldsymbol{s}_t, \boldsymbol{a}_t) &= p(\mathcal{O}_{t:T} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)\\ &= \int p(\mathcal{O}_{t:T}, \boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t) \text{d}\boldsymbol{s}_{t + 1}\\ &= \int p(\mathcal{O}_{t + 1:T} \mid \boldsymbol{s}_{t + 1}) p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t) p(\mathcal{O}_t \mid \boldsymbol{s}_t, \boldsymbol{a}_t) \text{d}\boldsymbol{s}_{t + 1}\\ \end{aligned}
$$
这里的 $p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$ 是动态，$p(\mathcal{O}_t \mid \boldsymbol{s}_t, \boldsymbol{a}_t) \propto \exp(r(\boldsymbol{s}_t, \boldsymbol{a}_t))$ 对应于奖励，于是关心剩下的 $p(\mathcal{O}_{t + 1:T} \mid \boldsymbol{s}_{t + 1})$，不妨记其为 $\beta_{t + 1}(\boldsymbol{s}_{t + 1})$。

对于 $\beta_{t + 1}(\boldsymbol{s}_{t + 1})$，注意到也有如下表示形式：
$$
p(\mathcal{O}_{t + 1:T} \mid \boldsymbol{s}_{t + 1}) = \int p(\mathcal{O}_{t + 1:T} \mid \boldsymbol{s}_{t + 1}, \boldsymbol{a}_{t + 1}) p(\boldsymbol{a}_{t + 1} \mid \boldsymbol{s}_{t + 1}) \text{d}\boldsymbol{a}_{t + 1}
$$
其中 $p(\mathcal{O}_{t + 1:T} \mid \boldsymbol{s}_{t + 1}, \boldsymbol{a}_{t + 1})$ 是 $\beta_{t + 1}(\boldsymbol{s}_{t + 1}, \boldsymbol{a}_{t + 1})$。

但值得注意的是，这里的 $p(\boldsymbol{a}_{t + 1} \mid \boldsymbol{s}_{t + 1})$ 并不是策略，而是像先验一样的一个分布，通常假设这是均匀分布（这是因为当不知道关于目标 $\mathcal{O}_{1:T}$ 的任何信息时，无从知道何种行为更有可能）。

不难发现可以递归地计算反向信息，从 $t = T - 1$ 到 $t = 1$（之所以从 $t = T - 1$ 开始是因为知道 $\beta_T(\boldsymbol{s}_T, \boldsymbol{a}_T) = \exp(r(\boldsymbol{s}_T, \boldsymbol{a}_T))$，并可以计算出 $\beta_T(\boldsymbol{s}_T)$）：
$$
\begin{aligned} &\beta_t(\boldsymbol{s}_t, \boldsymbol{a}_t) = p(\mathcal{O}_{t} \mid \boldsymbol{s}_t, \boldsymbol{a}_t) \mathbb{E}_{\boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)}[\beta_{t + 1}(\boldsymbol{s}_{t + 1})]\\ &\beta_t(\boldsymbol{s}_t) = \mathbb{E}_{\boldsymbol{a}_t \sim p(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}[\beta_t(\boldsymbol{s}_t, \boldsymbol{a}_t)] \end{aligned}
$$

#### 2.1.2 Intuition
仔细考虑这个过程，为了提供一些直觉，考虑令
$$
V_t(\boldsymbol{s}_t) = \log \beta_t(\boldsymbol{s}_t)
$$
$$
Q_t(\boldsymbol{s}_t, \boldsymbol{a}_t) = \log \beta_t(\boldsymbol{s}_t, \boldsymbol{a}_t)
$$
这里使用 $V,Q$ 这两个记号并不是巧合，事实上它们是某种软版本的 $V,Q$ 函数。

有
$$
V(\boldsymbol{s}_t) = \log \int \exp(Q(\boldsymbol{s}_t, \boldsymbol{a}_t)) \text{d}\boldsymbol{a}_t
$$
由于指数的存在，$\exp(Q(\boldsymbol{s}_t, \boldsymbol{a}_t))$ 中最大的一项会占主导地位，故这可以想象为一种软版本的最大化（在本节后续内容中，如果出现软最大化指的都是这样的操作），也就是当 $Q_t(\boldsymbol{s}_t, \boldsymbol{a}_t)$ 增大时：
$$
V_t(\boldsymbol{s}_t) \rightarrow \max_{\boldsymbol{a}_t} Q_t(\boldsymbol{s}_t, \boldsymbol{a}_t)
$$
另一方面由于 $\beta_t(\boldsymbol{s}_t, \boldsymbol{a}_t)$ 与 $\beta_t(\boldsymbol{s}_t)$ 的关系中 $p(\mathcal{O}_t \mid \boldsymbol{s}_t, \boldsymbol{a}_t) = \exp(r(\boldsymbol{s}_t, \boldsymbol{a}_t))$，有
$$
Q_t(\boldsymbol{s}_t, \boldsymbol{a}_t) = r(\boldsymbol{s}_t, \boldsymbol{a}_t) + \log \mathbb{E}_{\boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t,\boldsymbol{a}_t)}\left[\exp(V_{t + 1}(\boldsymbol{s}_{t + 1}))\right]
$$
在确定性情况下可以写作
$$
Q_t(\boldsymbol{s}_t, \boldsymbol{a}_t) = r(\boldsymbol{s}_t, \boldsymbol{a}_t) + V_{t + 1}(\boldsymbol{s}_{t + 1})
$$
恰好就是确定性的情况下的贝尔曼方程。但值得注意的是，在随机性的情况下这里存在一个过高估计的问题（因为最大的一个 $V$ 会占主导地位，即使其概率非常小），我们之后会进一步讨论。

总的来说，$\log \beta_t(\boldsymbol{s}_t,\boldsymbol{a}_t)$ 像是一个 $Q$ 函数，而 $\log \beta_t(\boldsymbol{s}_t)$ 像是一个价值函数。

#### 2.1.3 Side Note on Action Prior
最后讨论一下为什么总是可以假设动作对是均匀的，因为当其不均匀时，相当于
$$
V(\boldsymbol{s}_t) = \log \int \exp(Q(\boldsymbol{s}_t, \boldsymbol{a}_t) + \log p(\boldsymbol{a}_t \mid \boldsymbol{s}_t)) \text{d}\boldsymbol{a}_t
$$
$$
Q(\boldsymbol{s}_t, \boldsymbol{a}_t) = r(\boldsymbol{s}_t, \boldsymbol{a}_t) + \log \mathbb{E}\left[\exp(V(\boldsymbol{s}_{t + 1}))\right]
$$
只需要修改奖励函数为
$$
\tilde{r}(\boldsymbol{s}_t, \boldsymbol{a}) = r(\boldsymbol{s}_t, \boldsymbol{a}) + \log p(\boldsymbol{a} \mid \boldsymbol{s}_t)
$$
就可以满足原先的贝尔曼更新。于是总是可以通过调整奖励的方式使得动作对是均匀的，再结合我们设置为均匀分布的基本直觉即可发现使用均匀分布是合理的。

### 2.2 Policy
利用反向信息的结果，能够计算最优策略 $\pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t) = p(\boldsymbol{a}_t \mid \boldsymbol{s}_t, \mathcal{O}_{1:T})$，有 
$$
\begin{aligned} p(\boldsymbol{a}_t \mid \boldsymbol{s}_t, \mathcal{O}_{1:T}) &= p(\boldsymbol{a}_t \mid \boldsymbol{s}_t, \mathcal{O}_{t:T})\\ &= \frac{p(\boldsymbol{a}_t, \boldsymbol{s}_t \mid \mathcal{O}_{t:T})}{p(\boldsymbol{s}_t \mid \mathcal{O}_{t:T})}\\ &= \frac{p(\mathcal{O}_{t:T}\mid \boldsymbol{a}_t, \boldsymbol{s}_t) p(\boldsymbol{a}_t, \boldsymbol{s}_t) / p(\mathcal{O}_{t:T})}{p(\mathcal{O}_{t:T} \mid \boldsymbol{s}_t) p(\boldsymbol{s}_t) / p(\mathcal{O}_{t:T})}\\ &= \frac{p(\mathcal{O}_{t:T}\mid \boldsymbol{a}_t, \boldsymbol{s}_t)}{p(\mathcal{O}_{t:T} \mid \boldsymbol{s}_t)} \frac{p(\boldsymbol{a}_t, \boldsymbol{s}_t)}{p(\boldsymbol{s}_t)}\\ &= \frac{\beta_t(\boldsymbol{s}_t, \boldsymbol{a}_t)}{\beta_t(\boldsymbol{s}_t)} \end{aligned}
$$
这里推导的最后忽略了动作对 $p(\boldsymbol{a}_t \mid \boldsymbol{s}_t)$，于是策略就是
$$
\pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t) = \frac{\beta_t(\boldsymbol{s}_t, \boldsymbol{a}_t)}{\beta_t(\boldsymbol{s}_t)}
$$
如果把 $\beta_t$ 与价值函数的关系代入进去，就得到了
$$
\pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t) = \exp(Q_t(\boldsymbol{s}_t, \boldsymbol{a}_t) - V_t(\boldsymbol{s}_t)) = \exp(A_t(\boldsymbol{s}_t, \boldsymbol{a}_t))
$$
这也很显然符合直觉，有数高的动作会有更高的概率被选择，同时如果添加温度参数 $\alpha$，有
$$
\pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t) = \exp\left(\frac{1}{\alpha}A_t(\boldsymbol{s}_t, \boldsymbol{a}_t)\right)
$$
不难发现这与玻尔兹曼探索相似，同时当温度下降时更加接近贪心策略。

### 2.3 Forward messages
最后考虑前向信息与反向信息的差别在于，前向信息的形式非常复杂：
$$
\begin{aligned} \alpha_t(\boldsymbol{s}_t) &= p(\boldsymbol{s}_t \mid \mathcal{O}_{1:t - 1})\\ &= \int p(\boldsymbol{s}_t, \boldsymbol{s}_{t - 1}, \boldsymbol{a}_{t - 1} \mid \mathcal{O}_{1:t - 1}) \text{d}\boldsymbol{s}_{t - 1} \text{d}\boldsymbol{a}_{t - 1}\\ &= \int p(\boldsymbol{s}_t \mid \boldsymbol{s}_{t - 1}, \boldsymbol{a}_{t - 1}, \mathcal{O}_{1:t - 1}) p(\boldsymbol{a}_{t - 1} \mid \boldsymbol{s}_{t - 1}, \mathcal{O}_{1:t - 1}) p(\boldsymbol{s}_{t - 1} \mid \mathcal{O}_{1:t - 1}) \text{d}\boldsymbol{s}_{t - 1} \text{d}\boldsymbol{a}_{t - 1}\\ &= \int p(\boldsymbol{s}_t \mid \boldsymbol{s}_{t - 1}, \boldsymbol{a}_{t - 1}) p(\boldsymbol{a}_{t - 1} \mid \boldsymbol{s}_{t - 1}, \mathcal{O}_{t - 1}) p(\boldsymbol{s}_{t - 1} \mid \mathcal{O}_{1:t - 1}) \text{d}\boldsymbol{s}_{t - 1} \text{d}\boldsymbol{a}_{t - 1}\\ \end{aligned}
$$
其中 $p(\boldsymbol{s}_t \mid \boldsymbol{s}_{t - 1}, \boldsymbol{a}_{t - 1})$ 是动态，考虑剩余两项 $p(\boldsymbol{a}_{t - 1} \mid \boldsymbol{s}_{t - 1}, \mathcal{O}_{t - 1}) p(\boldsymbol{s}_{t - 1} \mid \mathcal{O}_{1:t - 1})$ 的意义：
$$
\begin{aligned} &\quad\,\,\, p(\boldsymbol{a}_{t - 1} \mid \boldsymbol{s}_{t - 1}, \mathcal{O}_{t - 1}) p(\boldsymbol{s}_{t - 1} \mid \mathcal{O}_{1:t - 1})\\  &= \frac{p(\mathcal{O}_{t - 1} \mid \boldsymbol{s}_{t - 1}, \boldsymbol{a}_{t - 1}) p(\boldsymbol{a}_{t - 1} \mid \boldsymbol{s}_{t - 1})}{p(\mathcal{O}_{t - 1} \mid \boldsymbol{s}_{t - 1})} \frac{p(\mathcal{O}_{t - 1} \mid \boldsymbol{s}_{t - 1}) p(\boldsymbol{s}_{t - 1} \mid \mathcal{O}_{1:t - 2})}{p(\mathcal{O}_{t - 1}\mid \mathcal{O}_{1:t - 2})}\\ &= \frac{p(\mathcal{O}_{t - 1} \mid \boldsymbol{s}_{t - 1}, \boldsymbol{a}_{t - 1}) p(\boldsymbol{a}_{t - 1} \mid \boldsymbol{s}_{t - 1}) p(\boldsymbol{s}_{t - 1} \mid \mathcal{O}_{1:t - 2})}{p(\mathcal{O}_{t - 1}\mid \mathcal{O}_{1:t - 2})} \end{aligned}
$$
这里 $p(\mathcal{O}_{t - 1} \mid \boldsymbol{s}_{t - 1}, \boldsymbol{a}_{t - 1})$ 是动态，$p(\boldsymbol{a}_{t - 1} \mid \boldsymbol{s}_{t - 1})$ 是动作对，$p(\boldsymbol{s}_{t - 1} \mid \mathcal{O}_{1:t - 2})$ 是 $\alpha_{t - 1}(\boldsymbol{s}_{t - 1})$，$p(\mathcal{O}_{t - 1}\mid \mathcal{O}_{1:t - 2})$ 同样是归一化因子。

这里一个不容易理解的是
$$
\begin{aligned} p(\boldsymbol{s}_{t - 1} \mid \mathcal{O}_{1:t - 1}) &= p(\boldsymbol{s}_{t - 1} \mid \mathcal{O}_{1:t - 2}, \mathcal{O}_{t - 1})\\  &= \frac{p(\mathcal{O}_{t - 1} \mid \boldsymbol{s}_{t - 1}, \mathcal{O}_{1:t-2}) p(\boldsymbol{s}_{t - 1} \mid \mathcal{O}_{1:t-2})}{p(\mathcal{O}_{t - 1} \mid \mathcal{O}_{1:t-2})}\\ &= \frac{p(\mathcal{O}_{t - 1} \mid \boldsymbol{s}_{t - 1}) p(\boldsymbol{s}_{t - 1} \mid \mathcal{O}_{1:t-2})}{p(\mathcal{O}_{t - 1} \mid \mathcal{O}_{1:t-2})}. \end{aligned}
$$

在计算 $\alpha_t(\boldsymbol{s}_t)$ 时，可以递归地计算从 $t = 2$ 到 $t = T$ 的 $\alpha_t(\boldsymbol{s}_t)$。

### 2.4 State margin and Forward/Backward message intersection
事实上，此时能够恢复状态边际分布 $p(\boldsymbol{s}_t \mid \mathcal{O}_{1:T})$，只需考虑
$$
\begin{aligned} p(\boldsymbol{s}_t \mid \mathcal{O}_{1:T}) &= \frac{p(\boldsymbol{s}_t, \mathcal{O}_{1:T})}{p(\mathcal{O}_{1:T})}\\ &= \frac{p(\mathcal{O}_{t:T} \mid \boldsymbol{s}_t) p(\boldsymbol{s}_t, \mathcal{O}_{1:t - 1})}{p(\mathcal{O}_{1:T})}\\ &\propto \beta_t(\boldsymbol{s}_t) \alpha_t(\boldsymbol{s}_t) p(\mathcal{O}_{1:t - 1}) \propto \beta_t(\boldsymbol{s}_t) \alpha_t(\boldsymbol{s}_t). \end{aligned}
$$
考虑 $p(\boldsymbol{s}_t) \propto \beta_t(\boldsymbol{s}_t) \alpha_t(\boldsymbol{s}_t)$ 背后的直觉，这里可以把前向信息与反向信息视作两个锥形：
- 后向的锥：表示了那些能够以高概率到达目标的状态；
- 前向的锥：表示了那些能够从初始状态出发以高奖励到达的状态。

![](17-3.png)

状态边际分布就是这两个锥的交集，不难发现这一交集两侧窄，中间宽。这与人类的行为是类似的，根据行为学上的研究，人的行为的状态也有类似的状态分布，在行为轨迹靠中间的位置通常可能有着更大的偏差。
![](17-4.png)

关于前向信息的推断方法会在逆强化学习中有着重要的应用。

### 2.5 Summary of Control as Inference
到目前为止，我们给出了最优控制的概率图模型，并将控制视作是推断的一种方式：
- 后向的过程类似于一种软版本的价值迭代；
- 可以利用反向信息恢复策略，就像是利用价值函数恢复策略一样； 
- 可以利用前向信息和反向信息恢复状态边际分布。

## 3 Control as Variational Inference
在之前的推断方法中，我们会通过准确的方式进行计算，然而现实中的问题更加复杂，并且不知道转移动态，需要通过样本来近似，因此需要使用近似的推断。

接下来会使用上一节介绍的[[Concepts#27 变分推断（Variational Inference，VI）|变分推断（Variational Inference，VI）]]方法来从刚才介绍的[[#2 Control as Inference]]中推导出的无模型的强化学习算法。但在介绍变分推断之前，我们先来解决之前一节中提到的问题，也就是过高估计问题。
### 3.1 Optimism problem
在反向信息中，从 $t = T - 1$ 到 $t = 1$：
$$
\beta_t(\boldsymbol{s}_t, \boldsymbol{a}_t) = p(\mathcal{O}_{t} \mid \boldsymbol{s}_t, \boldsymbol{a}_t) \mathbb{E}_{\boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)}[\beta_{t + 1}(\boldsymbol{s}_{t + 1})]\
$$
$$
\beta_t(\boldsymbol{s}_t) = \mathbb{E}_{\boldsymbol{a}_t \sim p(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}[\beta_t(\boldsymbol{s}_t, \boldsymbol{a}_t)]
$$
对应于价值函数的更新为
$$
Q_t(\boldsymbol{s}_t, \boldsymbol{a}_t) = r(\boldsymbol{s}_t, \boldsymbol{a}_t) + \log \mathbb{E}\left[\exp(V_{t + 1}(\boldsymbol{s}_{t + 1}))\right]
$$
最后一项由于先取指数，再取期望，容易受到极端值的影响，过高估计，不妨考虑以下例子

例如，考虑一个人在 $\boldsymbol{s}_1$，其有两个动作分别对应于是否购买一个彩票，分别用 $\boldsymbol{a}_1 = 1,0$ 表示，假设购买彩票 $99\%$ 亏 $1000$ 元， $1\%$ 赚 $10000$ 元，于是有
$$
r(\boldsymbol{s}_1, \boldsymbol{a}_1 = 1, \boldsymbol{s}_2 = w) = 10000, \quad r(\boldsymbol{s}_1, \boldsymbol{a}_1 = 1, \boldsymbol{s}_2 = l) = -1000
$$
$$
r(\boldsymbol{s}_1, \boldsymbol{a}_1 = 0, \boldsymbol{s}_2 = n) = 0
$$
其中有动态：
$$
p(\boldsymbol{s}_2 = w \mid \boldsymbol{s}_1, \boldsymbol{a}_1 = 1) = 0.01, \quad p(\boldsymbol{s}_2 = l \mid \boldsymbol{s}_1, \boldsymbol{a}_1 = 1) = 0.99
$$
$$
p(\boldsymbol{s}_2 = n \mid \boldsymbol{s}_1, \boldsymbol{a}_1 = 0) = 1
$$
尽管在期望上购买彩票是亏损的，由于指数的存在，我们可能会认为这是一个好的选择。

#### 3.1.1 Why it doesn't occur in deterministic case?
对于 $p(\tau \mid \mathcal{O}_{1:T})$，其有形式
$$
p(\tau \mid \mathcal{O}_{1:T}) \propto p(\boldsymbol{s}_1) \prod_{t = 1}^{T} p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)  \exp\left(\sum_{t = 1}^T r(\boldsymbol{s}_t, \boldsymbol{a}_t)\right)
$$
在确定性情况下，由于
$$
p(\boldsymbol{s}_1) \prod_{t = 1}^{T} p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t) = \mathbb{I}(p(\tau) \neq 0)
$$
最大的后验概率 $p(\tau \mid \mathcal{O}_{1:T})$ 对应的轨迹 $\tau$ 自然需要满足 $p(\tau) \neq 0$，于是指示函数就可以忽略，也即最大的后验概率对应的轨迹就是最优轨迹，进而就得到了最优的策略。找到最优轨迹的过程就等价于最大化后验概率 $p(\tau \mid \mathcal{O}_{1:T})$ 的过程。

然而在随机性的情况下，后验概率 $p(\tau \mid \mathcal{O}_{1:T})$ 不仅受到这个轨迹 $\tau$ 的奖励影响，还需要和这条轨迹在动态上出现的概率有关，单纯最大化后验概率并没有平衡好奖励与动态概率之间的关系。
#### 3.1.2 Further Analysis
核心的问题可以从后验概率的展开式中看出
$$
p(\tau \mid \mathcal{O}_{1:T}) = p(\boldsymbol{s}_1 \mid \mathcal{O}_{1:T}) \prod_{t = 1}^{T - 1} p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \mathcal{O}_{1:T}) \prod_{t = 1}^{T} p(\boldsymbol{a}_t \mid \boldsymbol{s}_t, \mathcal{O}_{1:T})
$$
在这里 $p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \mathcal{O}_{1:T})$ 未必等于 $p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$，也就是允许动态依赖于 $\mathcal{O}_{1:T}$，也就是说将动态也变成了一个可以由最优性控制的因素。此时得到的就不是原始动态下的最优策略，而是某种最优动态下的最优策略，这显然不是我们想要的。

回到彩票的例子，考虑最大化后验概率 $p(\boldsymbol{s}_1, \boldsymbol{a}_1, \boldsymbol{s}_2 \mid \mathcal{O}_1)$，由于 $10000$ 的奖励会出现在指数上，因此
$$
p(\boldsymbol{s}_1, \boldsymbol{a}_1 = 1, \boldsymbol{s}_2 = w \mid \mathcal{O}_1) \approx 1
$$
另一方面同时有
$$
p(\boldsymbol{s}_1, \boldsymbol{a}_1 = 1, \boldsymbol{s}_2 = w \mid \mathcal{O}_1) = p(\boldsymbol{s}_2 \mid \boldsymbol{s}_1, \boldsymbol{a}_1 = 1, \mathcal{O}_1) p(\boldsymbol{a}_1 = 1 \mid \boldsymbol{s}_1, \mathcal{O}_1) p(\boldsymbol{s}_1\mid \mathcal{O}_1)
$$
因此后面的每一个概率都必须是接近 $1$ 的数。在真实的动态中的 $p(\boldsymbol{s}_2 \mid \boldsymbol{s}_1, \boldsymbol{a}_1 = 1)$ 明明是 $0.01$ 这样小的概率，但是在这里由于允许动态依赖于 $\mathcal{O}_1$，最大化后验概率时，就会将其偏移到 $1$。

回到例子本身，将彩票中奖率变为了由最优性控制的因素，并且认为最优时动态应该是次次都中奖，并以此认为我们的做法是应该每次都买彩票，但是真实的彩票中奖率不是我们能决定的，这样的策略在真实的动态下是有问题的。

根据上述的分析，我们发现优化的目标不能是之前的后验概率 $p(\tau \mid \mathcal{O}_{1:T})$，因为这会修改动态。

### 3.2 Control via variational inference
实际上可以将其转化为变分推断的问题，这里取 $\boldsymbol{x} = \mathcal{O}_{1:T}$ 和 $\boldsymbol{z} = \tau$，我们希望使用某个参数化的分布 $q(\boldsymbol{z})$ 来近似后验概率 $p(\boldsymbol{z} \mid \boldsymbol{x})$，同时这个 $q(\boldsymbol{z})$ 需要能够拆分为 
$$
p(\boldsymbol{s}_1) \prod_t p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t) q(\boldsymbol{a}_t \mid \boldsymbol{s}_t)
$$
也即不能篡改动态，并且只需要学习 $q(\boldsymbol{a}_t \mid \boldsymbol{s}_t)$，根据惯例可以用 $\pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t) = q(\boldsymbol{a}_t \mid \boldsymbol{s}_t)$ 来表示。

这对应于概率图模型：
![](17-5.png)

这里希望采样的数据中观测到的最优性变量都为 $1$ 的概率尽可能大，也即希望最大化对数似然 $\log p(\boldsymbol{x})$。基于上一节[[Lecture 16 Variational Inference and Generative Model]]中所学知识，我们知道对数似然与最大证据下界（ELBO）的关系:
$$
\log p(\boldsymbol{x}) \geq \mathbb{E}_{\boldsymbol{z} \sim q(\boldsymbol{z})} \left[\log p(\boldsymbol{x}, \boldsymbol{z}) - \log q(\boldsymbol{z})\right]
$$
利用展开式
$$
p(\boldsymbol{x}, \boldsymbol{z}) = p(\boldsymbol{z}) p(\boldsymbol{x} \mid \boldsymbol{z}) = p(\tau) \prod_{t = 1}^{T} p(\mathcal{O}_t \mid \boldsymbol{s}_t, \boldsymbol{a}_t) = p(\boldsymbol{s}_1) \prod_{t = 1}^{T} p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t) p(\mathcal{O}_t \mid \boldsymbol{s}_t, \boldsymbol{a}_t)
$$
$$
q(\boldsymbol{z}) = p(\boldsymbol{s}_1) \prod_t p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t) q(\boldsymbol{a}_t \mid \boldsymbol{s}_t)
$$
代回最大证据下界中有
$$
\begin{aligned} \log p(\mathcal{O}_{1:T}) &\geq \mathbb{E}_{(\boldsymbol{s}_{1:T}, \boldsymbol{a}_{1:T}) \sim q(\boldsymbol{z})} \left[\sum_{t = 1}^{T} \log p(\mathcal{O}_t \mid \boldsymbol{s}_t, \boldsymbol{a}_t) - \log q(\boldsymbol{a}_t \mid \boldsymbol{s}_t)\right]\\ &= \mathbb{E}_{(\boldsymbol{s}_{1:T}, \boldsymbol{a}_{1:T}) \sim q(\boldsymbol{z})} \left[\sum_{t = 1}^{T} r(\boldsymbol{s}_t, \boldsymbol{a}_t) - \log q(\boldsymbol{a}_t \mid \boldsymbol{s}_t)\right]\\ &= \sum_{t = 1}^{T} \mathbb{E}_{(\boldsymbol{s}_t, \boldsymbol{a}_t) \sim q(\boldsymbol{s}_t, \boldsymbol{a}_t)} \left[r(\boldsymbol{s}_t, \boldsymbol{a}_t) + \mathcal{H}(q(\cdot \mid \boldsymbol{s}_t))\right]. \end{aligned}\
$$
优化的目标是最大证据下界：
$$
\sum_{t = 1}^{T} \mathbb{E}_{(\boldsymbol{s}_t, \boldsymbol{a}_t) \sim q(\boldsymbol{s}_t, \boldsymbol{a}_t)} \left[r(\boldsymbol{s}_t, \boldsymbol{a}_t) + \mathcal{H}(q(\cdot \mid \boldsymbol{s}_t))\right]
$$
这其实就是常见的最大熵强化学习目标，也就是要最大化通常强化学习的奖励总和与动作熵的和。不难理解，根据上述概率模型进行变分推断得到的最优策略需要具有较好的探索性质。

### 3.3 Optimizing the variational lower bound
接下来考虑如何优化这个变分下界，一个简单的方式是推导出价值函数的更新规则。

基本情况：首先考虑基类情况，也即 $t = T$ 时，注意对于每一个给定的 $\boldsymbol{s}_T$，$q(\cdot \mid \boldsymbol{s}_T)$ 都是一个概率分布，故对于 $\forall \boldsymbol{s}_T$，考虑
$$
q(\cdot\mid \boldsymbol{s}_T) = \arg\max_{q(\cdot\mid \boldsymbol{s}_T)} \mathbb{E}_{\boldsymbol{a}_T \sim q(\boldsymbol{a}_T \mid \boldsymbol{s}_T)}[r(\boldsymbol{s}_T, \boldsymbol{a}_T)] + \mathcal{H}(q(\cdot \mid \boldsymbol{s}_T))
$$
这里引入归一化约束 $\int q(\boldsymbol{a}_T \mid \boldsymbol{s}_T) d\boldsymbol{a}_T = 1$，定义拉格朗日函数：
$$
\mathcal{L} = \int q(\boldsymbol{a}_T \mid \boldsymbol{s}_T) \left[ r(\boldsymbol{s}_T, \boldsymbol{a}_T) - \log q(\boldsymbol{a}_T \mid \boldsymbol{s}_T) \right] d\boldsymbol{a}_T + \lambda \left( 1 - \int q(\boldsymbol{a}_T \mid \boldsymbol{s}_T) d\boldsymbol{a}_T \right)
$$
对 $q(\boldsymbol{a}_T \mid \boldsymbol{s}_T)$ 求[[Concepts#23 变分导数 (Variational Derivative)|变分导数 (Variational Derivative)]]并令其为零：
$$
\frac{\delta \mathcal{L}}{\delta q(\boldsymbol{a}_T \mid \boldsymbol{s}_T)} = r(\boldsymbol{s}_T, \boldsymbol{a}_T) - \log q(\boldsymbol{a}_T \mid \boldsymbol{s}_T) - 1 - \lambda = 0
$$
解得
$$
q(\boldsymbol{a}_T \mid \boldsymbol{s}_T) = \exp(r(\boldsymbol{s}_T, \boldsymbol{a}_T) - 1 - \lambda)
$$
对所有可能的 $\boldsymbol{s}_T$ 进行整理即可得出：目标函数在 $q(\boldsymbol{a}_T \mid \boldsymbol{s}_T) \propto \exp(r(\boldsymbol{s}_T, \boldsymbol{a}_T))$ 时是最优的，而由于归一化约束，有
$$
q(\boldsymbol{a}_T \mid \boldsymbol{s}_T) = \frac{\exp(r(\boldsymbol{s}_T, \boldsymbol{a}_T))}{\int \exp(r(\boldsymbol{s}_T, \boldsymbol{a}_T)) \text{d}\boldsymbol{a}_T} = \exp(Q(\boldsymbol{s}_T, \boldsymbol{a}_T) - V(\boldsymbol{s}_T))
$$
其中利用了
$$
V(\boldsymbol{s}_T) = \log \int \exp(Q(\boldsymbol{s}_T, \boldsymbol{a}_T)) \text{d}\boldsymbol{a}_T
$$
于是
$$
\mathbb{E}_{\boldsymbol{s}_T \sim q(\boldsymbol{s}_T)}[ \mathbb{E}_{\boldsymbol{a}_T \sim q(\boldsymbol{a}_T \mid \boldsymbol{s}_T)}[r(\boldsymbol{s}_T, \boldsymbol{a}_T) - \log q(\boldsymbol{a}_T \mid \boldsymbol{s}_T)]] = \mathbb{E}_{\boldsymbol{s}_T \sim q(\boldsymbol{s}_T)}[\mathbb{E}_{\boldsymbol{a}_T \sim q(\boldsymbol{a}_T \mid \boldsymbol{s}_T)}[V(\boldsymbol{s}_T)]]
$$

一般情况：对于一般的时间步 $t$，对任意的 $\boldsymbol{s}_t$，有
$$
\begin{aligned} q(\cdot \mid \boldsymbol{s}_t) &= \arg\max_{q(\cdot \mid \boldsymbol{s}_t)} \mathbb{E}_{\boldsymbol{a}_t \sim q(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}[r(\boldsymbol{s}_t, \boldsymbol{a}_t) + \mathbb{E}_{\boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)}[V(\boldsymbol{s}_{t + 1})] + \mathcal{H}(q(\cdot \mid \boldsymbol{s}_t))]\\ \end{aligned}
$$
不难发现如果使用常规的贝尔曼更新：
$$
Q(\boldsymbol{s}_t, \boldsymbol{a}_t) = r(\boldsymbol{s}_t, \boldsymbol{a}_t) + \mathbb{E}_{\boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)}[V(\boldsymbol{s}_{t + 1})]
$$
而非之前的过高估计的更新，那么可以归约到基类：
$$
q(\cdot \mid \boldsymbol{s}_t) = \arg\max_{q(\cdot \mid \boldsymbol{s}_t)} \mathbb{E}_{\boldsymbol{a}_t \sim q(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}[Q(\boldsymbol{s}_t, \boldsymbol{a}_t)] + \mathcal{H}(q(\boldsymbol{a}_t \mid \boldsymbol{s}_t))
$$
从而得到一样的结果：
$$
q(\boldsymbol{a}_t \mid \boldsymbol{s}_t) = \exp(Q(\boldsymbol{s}_t, \boldsymbol{a}_t) - V(\boldsymbol{s}_t)) = \exp(A(\boldsymbol{s}_t, \boldsymbol{a}_t))
$$

整理可知当依据这一方式作为策略时，我们能够最大化变分下界。

### 3.4 Backward pass summary -variational
根据上述结果，可以得到变分推断情况下的反向传播（或者说价值函数的更新规则）：
从 $t = T - 1$ 到 $t = 1$，重复：
1. $Q_t(\boldsymbol{s}_t, \boldsymbol{a}_t) = r(\boldsymbol{s}_t, \boldsymbol{a}_t) + \mathbb{E}_{\boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)}[V_{t + 1}(\boldsymbol{s}_{t + 1})]$；
2. $V_t(\boldsymbol{s}_t) = \log \int \exp(Q_t(\boldsymbol{s}_t, \boldsymbol{a}_t)) \text{d}\boldsymbol{a}_t$。

这可以视作一个软价值迭代算法，其中把对 $Q$ 的最大化变成了一个相对软的最大化：
1. 令 $Q(\boldsymbol{s}, \boldsymbol{a}) \gets r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{s}' \sim p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})}[V(\boldsymbol{s}')]$；
2. 令 $V(\boldsymbol{s}) \gets \text{ soft max}_{\boldsymbol{a}} Q(\boldsymbol{s}, \boldsymbol{a})$。

这一算法有一些简单的变体：
- 添加折旧因子 $\gamma$。  
- 添加温度 $V_t(\boldsymbol{s}_t) = \alpha\log \int \exp(Q_t(\boldsymbol{s}_t, \boldsymbol{a}_t) / \alpha) \text{d}\boldsymbol{a}_t$。
- 对于无限时间跨度，只需要运行一个无限时间跨度的软价值迭代即可。

上述所有内容均可以参见：Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review, Sergey Levine, 2018.

## 4 Algorithms for RL as Inference
### 4.1 Q-learning with soft optimality
回顾标准的 Q 学习：
- $\phi \gets \phi + \alpha \nabla_\phi Q_\phi(\boldsymbol{s}, \boldsymbol{a}) (r(\boldsymbol{s}, \boldsymbol{a}) + \gamma V(\boldsymbol{s'}) - Q_{\phi}(\boldsymbol{s}, \boldsymbol{a}))$；
- 其中目标价值：$V(\boldsymbol{s'}) = \max_{\boldsymbol{a'}} Q_{\phi'}(\boldsymbol{s'}, \boldsymbol{a'})$。

而在软 Q 学习中：
- $\phi \gets \phi + \alpha \nabla_\phi Q_\phi(\boldsymbol{s}, \boldsymbol{a}) (r(\boldsymbol{s}, \boldsymbol{a}) + \gamma V(\boldsymbol{s'}) - Q_{\phi}(\boldsymbol{s}, \boldsymbol{a}))$；
- 其中目标价值：$V(\boldsymbol{s'}) = \log \int \exp(Q_{\phi'}(\boldsymbol{s'}, \boldsymbol{a'})) \text{d}\boldsymbol{a}'$；
- 使用策略：$\pi(\boldsymbol{a} \mid \boldsymbol{s}) = \exp(Q(\boldsymbol{s}, \boldsymbol{a}) - V(\boldsymbol{s}))$。

类似地可以得到一个软 Q 学习的算法流程：
重复以下过程:
1. 采取动作 $\boldsymbol{a}_i$ 观测到 $(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)$，将其添加到回放缓冲区 $\mathcal{R}$ 中；  
2. 从 $\mathcal{R}$ 中采样一个批次 $\{\boldsymbol{s}_j, \boldsymbol{a}_j, \boldsymbol{s}_j', r_j\}$； 
3. 计算 $y_j = r_j + \text{ soft max}_{\boldsymbol{a}'} Q_{\phi'}(\boldsymbol{s}_j', \boldsymbol{a}')$；
4. $\phi \gets \phi - \alpha \sum_j \frac{\text{d}Q_\phi}{\text{d}\phi}(\boldsymbol{s}_j, \boldsymbol{a}_j) (Q_\phi(\boldsymbol{s}_j, \boldsymbol{a}_j) - y_j)$；  
5. 更新 $\phi'$：$\phi' \gets \tau \phi + (1 - \tau) \phi'$ 或每 $N$ 步更新 $\phi' \gets \phi$。

不难注意到由于上述做法需要计算 $\log \int \exp(Q_{\phi'}(\boldsymbol{s'}, \boldsymbol{a'})) \text{d}\boldsymbol{a'}$，因此通常会在离散动作空间上使用。

### 4.2 Policy gradient with soft optimality
#### 4.2.1 Derivation of objective function
对于连续动作空间，可以使用策略梯度方法。回顾前面的推导可知，如果选择策略
$$
\pi(\boldsymbol{a} \mid \boldsymbol{s}) = \exp(Q(\boldsymbol{s}, \boldsymbol{a}) - V(\boldsymbol{s}))
$$
就能够最大化证据下界
$$
\sum_{t = 1}^{T} \mathbb{E}_{(\boldsymbol{s}_t, \boldsymbol{a}_t) \sim q(\boldsymbol{s}_t, \boldsymbol{a}_t)} \left[r(\boldsymbol{s}_t, \boldsymbol{a}_t) + \mathcal{H}(q(\cdot \mid \boldsymbol{s}_t))\right]
$$
但是对于连续状态空间来说，即使有了 $Q(\boldsymbol{s}, \boldsymbol{a})$，也没办法显式地计算 $\pi(\boldsymbol{a} \mid \boldsymbol{s})$。只能通过优化的方式使得策略逼近 $\exp(Q(\boldsymbol{s}, \boldsymbol{a}) - V(\boldsymbol{s}))$，对于 $\forall \boldsymbol{s}$，考虑最小化 KL 散度：
$$
D_{KL}\left(\pi(\boldsymbol{a} \mid \boldsymbol{s}) \bigg\| \frac{1}{Z} \exp(Q(\boldsymbol{s}, \boldsymbol{a}))\right) = -\mathbb{E}_{\pi(\boldsymbol{a} \mid \boldsymbol{s})}[Q(\boldsymbol{s}, \boldsymbol{a})] - \mathcal{H}(\pi(\cdot\mid\boldsymbol{s})) + const
$$
这等价于最大化目标函数
$$
\begin{aligned} J(\theta) &= \mathbb{E}_{\boldsymbol{s} \sim p_\theta(\boldsymbol{s})} \left[\mathbb{E}_{\boldsymbol{a} \sim \pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})} [r(\boldsymbol{s}, \boldsymbol{a})] + \mathcal{H}(\pi_\theta(\cdot \mid \boldsymbol{s}))\right]\\ \end{aligned}
$$
在策略梯度中，这需要通过轨迹来获取，也就是
$$
\begin{aligned} J(\theta) &= \sum_{t = 1}^T \mathbb{E}_{\boldsymbol{s}_t \sim p_\theta(\boldsymbol{s}_t)} \left[\mathbb{E}_{\boldsymbol{a}_t \sim \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} [r(\boldsymbol{s}_t, \boldsymbol{a}_t)] + \mathcal{H}(\pi_\theta(\cdot \mid \boldsymbol{s}_t))\right]\\ &= \sum_{t = 1}^T \mathbb{E}_{(\boldsymbol{s}_t, \boldsymbol{a}_t) \sim p_\theta(\boldsymbol{s}_t, \boldsymbol{a}_t)}[r(\boldsymbol{s}_t, \boldsymbol{a}_t) - \log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)]\\ &= \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[\sum_{t = 1}^{T} r(\boldsymbol{s}_t, \boldsymbol{a}_t) - \log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)\right]. \end{aligned}
$$
这个做法通常被称为“熵正则化” 策略梯度，能够防止过早的熵坍缩。在通常的策略梯度中，完全依赖策略的随机性来探索，一旦策略变得过于确定，就会导致探索几乎停止。

#### 4.2.2 Derivation of gradient
不妨进一步推导一下目标函数梯度的形式，首先注意到期望中的项也和 $\theta$ 有关，此时不难验证
$$
\nabla_\theta \int p_\theta(\tau) r_\theta(\tau) \text{d}\tau = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[(\nabla_\theta \log p_\theta(\tau)) r_\theta(\tau) + \nabla_\theta r_\theta(\tau)\right]
$$
代入有
$$
\begin{aligned} \nabla_\theta J(\theta) &\approx \frac{1}{N} \sum_{i = 1}^{N} \left(\sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t})\right) \left[\left(\sum_{t = 1}^T r(\boldsymbol{s}_{i,t}, \boldsymbol{a}_{i,t}) - \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t})\right) - 1\right]\\ &= \frac{1}{N} \sum_{i = 1}^{N} \sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t}) \left[\left(\sum_{t' = t}^T r(\boldsymbol{s}_{i,t'}, \boldsymbol{a}_{i,t'}) - \log \pi_\theta(\boldsymbol{a}_{i,t'} \mid \boldsymbol{s}_{i,t'})\right) - 1\right], \end{aligned}
$$
忽略掉最后 $-1$ 作为基线，就得到了策略梯度的形式。

事实上，可以进一步写作一个更加广泛的形式：
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t) \hat{A}(s_t, a_t)\right]
$$
这里的 $\hat{A}(s_t, a_t)$ 是一个任何一个优势估计器，包括广义优势估计（Generalized Advantage Estimation，GAE）等等，为了体现目标中的熵一项，需要做的只是在优势中减去各 $t'$ 时间步的 $- \log \pi_\theta(\boldsymbol{a}_{t'} \mid \boldsymbol{s}_{t'})$。

### 4.3 Policy gradient vs Q-learning
实际上刚才讨论的软 Q 学习与熵正则化策略梯度是非常接近的。
#### 4.3.1 Rewritting policy gradient with value function
注意到最后的梯度表达式中的
$$
\sum_{t' = t}^T r(\boldsymbol{s}_{i,t'}, \boldsymbol{a}_{i,t'}) - \log \pi(\boldsymbol{a}_{i,t'} \mid \boldsymbol{s}_{i,t'}) = r(\boldsymbol{s}_t, \boldsymbol{a}_t) + \left(\sum_{t' = t + 1}^{T} r(\boldsymbol{s}_{t'}, \boldsymbol{a}_{t'}) - \log \pi(\boldsymbol{a}_{t'} \mid \boldsymbol{s}_{t'})\right) - \log \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t)
$$
利用 $\log \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t) = Q(\boldsymbol{s}_t, \boldsymbol{a}_t) - V(\boldsymbol{s}_t)$，其中的 $\sum_{t' = t + 1}^{T} r(\boldsymbol{s}_{t'}, \boldsymbol{a}_{t'}) - \log \pi(\boldsymbol{a}_{t'} \mid \boldsymbol{s}_{t'})$ 经过离奇的近似之后可以近似为 $Q(\boldsymbol{s}_{t + 1}, \boldsymbol{a}_{t + 1})$：
$$
\begin{aligned} &\quad\,\, \sum_{t' = t + 1}^{T} r(\boldsymbol{s}_{i,t'}, \boldsymbol{a}_{i, t'}) - \log \pi(\boldsymbol{a}_{i, t'} \mid \boldsymbol{s}_{i, t'})\\ &= \sum_{t' = t+1}^{T} r(\boldsymbol{s}_{i,t'}, \boldsymbol{a}_{i,t'}) - \sum_{t' = t+1}^{T} Q(\boldsymbol{s}_{i,t'}, \boldsymbol{a}_{i,t'}) + \sum_{t' = t+1}^{T} V(\boldsymbol{s}_{i,t'})\\ &\approx \sum_{t' = t+1}^{T} r(\boldsymbol{s}_{i,t'}, \boldsymbol{a}_{i,t'}) - \left(\sum_{t' = t+1}^{T}r(\boldsymbol{s}_{i,t'}, \boldsymbol{a}_{i,t'}) + \sum_{t' = t+1}^{T - 1} V(\boldsymbol{s}_{i,t' + 1}) \right) + \sum_{t' = t+1}^{T} V(\boldsymbol{s}_{i,t'})\\ &= V(\boldsymbol{s}_{i,t + 1}) \approx Q(\boldsymbol{s}_{i,t+1}, \boldsymbol{a}_{i,t+1}). \end{aligned}
$$
最后的近似看起来很奇怪，但在期望角度是无偏的，因为收集的样本中动作的概率也和 $\exp(Q(\boldsymbol{s}, \boldsymbol{a}))$ 成正比，并且事实上由于这里是 on-policy，也没有别的 $\boldsymbol{a}_{i, t + 1}$ 可用。于是进一步代入 
$$
\log \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t) = Q(\boldsymbol{s}_t, \boldsymbol{a}_t) - V(\boldsymbol{s}_t)
$$
就得到了：
$$
\approx \frac{1}{N} \sum_{i} \sum_{t} \left(\nabla_\theta Q(\boldsymbol{a}_t, \boldsymbol{s}_t) - \nabla_\theta V(\boldsymbol{s}_t)\right) \left(r(\boldsymbol{s}_t, \boldsymbol{a}_t) + Q(\boldsymbol{s}_t, \boldsymbol{a}_t) - Q(\boldsymbol{s}_{t + 1}, \boldsymbol{a}_{t + 1}) - V(\boldsymbol{s}_t)\right)
$$
其中最后减去的 $V(\boldsymbol{s}_t)$ 不影响期望也可以当作一种基线忽略。

#### 4.3.2 Comparison with soft Q-learning
现在将其与软 Q 学习进行比较，在软 Q 学习中，有
$$
\nabla_\theta Q_\theta(\boldsymbol{s}, \boldsymbol{a}) (r(\boldsymbol{s}_t, \boldsymbol{a}_t) + \gamma ~\text{soft max}~ Q(\boldsymbol{s}_{t + 1}, \boldsymbol{a}_{t + 1}) - Q_{\theta}(\boldsymbol{s}_t, \boldsymbol{a}_t))
$$
二者的差异主要体现在：
1. 在策略梯度中，使用了 $\left(\nabla_\theta Q(\boldsymbol{a}_t, \boldsymbol{s}_t) - \nabla_\theta V(\boldsymbol{s}_t)\right)$，而在 Q 学习中使用了 $\nabla_\theta Q(\boldsymbol{a}_t, \boldsymbol{s}_t)$。按照原论文的说法，这是因为策略本身不足以恢复出 $Q$ 函数中额外的自由度，因为 $Q$ 函数可以增减一个常量而不影响常量。
2. 在 Q 学习中对 $Q(\boldsymbol{s}_{t + 1}, \boldsymbol{a}_{t + 1})$ 使用了软最大化，这实际上是一种 off-policy 纠正，如果使用 on-policy Q 学习，实际上也得像策略梯度中那样使用真实采样，等价于一样利用了无偏估计 $V(\boldsymbol{s}_{t + 1}) \approx Q(\boldsymbol{s}_{t + 1}, \boldsymbol{a}_{t + 1})$。

因此在一定的意义上，我们最大化熵意义下的策略梯度与软 Q 学习是等价的。

### 4.4 Soft Actor-Critic (SAC)
软演员-评论家（Soft Actor-Critic，SAC）是目前常用的连续动作空间 off-policy 算法，其中的基本思想与我们之前的讨论是一致的，类似于整合了软 Q 学习与熵正则化策略梯度的方法。由于与前面提到的两种算法不同，软演员-评论家是广泛应用于实际问题的算法，因此我们从更加实际的角度进行介绍。
####  4.4.1 Overview
软演员-评论家是一个 off-policy 演员-评论家算法，在下方论文的实现中，我们使用了一个演员网络 $\pi_\phi(\boldsymbol{a} \mid \boldsymbol{s})$ 以及四个评论家网络 $Q_{\theta_1}(\boldsymbol{s}, \boldsymbol{a})$，$Q_{\theta_2}(\boldsymbol{s}, \boldsymbol{a})$ $Q_{\bar{\theta}_1}(\boldsymbol{s}, \boldsymbol{a})$，$Q_{\bar{\theta}_2}(\boldsymbol{s}, \boldsymbol{a})$，前两个作为当前网络，后两个作为目标网络，使用两组的原因是为了减小 $Q$ 函数的高估问题（在基于 $Q$ 函数的深度强化学习中详细介绍过，也不难理解仅仅使用 $2$ 个评论家网络也是可行的，将目标网络用来选取 $\arg \max$，再用当前网络计算目标价值即可）。

####  4.4.2 Objective of Q-function
软演员-评论家的 $Q$ 函数的目标是最小化贝尔曼误差，也就是  
$$
J_Q(\theta) = \mathbb{E}_{(\boldsymbol{s}_t, \boldsymbol{a}_t) \sim \mathcal{D}} \left[\frac{1}{2}\left(Q_\theta(\boldsymbol{s}_t, \boldsymbol{a}_t) - (r(\boldsymbol{s}_t, \boldsymbol{a}_t) + \gamma \mathbb{E}_{\boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)}[V_{\bar{\theta}}(\boldsymbol{s}_{t + 1})])\right)^2\right]
$$
这里的 $V$ 需要利用 $Q$ 和 $\log \pi$ 来计算，代入就有（去掉外层的期望）
$$
\nabla_\theta J_Q(\theta) = \nabla_\theta Q_\theta(\boldsymbol{s}_t, \boldsymbol{a}_t) \left(Q_\theta(\boldsymbol{s}_t, \boldsymbol{a}_t) - (r(\boldsymbol{s}_t, \boldsymbol{a}_t) + \gamma Q_{\bar{\theta}}(\boldsymbol{s}_{t + 1}, \boldsymbol{a}_{t + 1}) - \alpha \log \pi_\phi(\boldsymbol{a}_{t + 1} \mid \boldsymbol{s}_{t + 1}))\right)
$$

#### 4.4.3 Objective of policy
软演员-评论家的常量的目标是最大化 Q 值与熵的和，如果写成损失就是最小化  
$$
J_\pi(\phi) = \mathbb{E}_{\boldsymbol{s}_t \sim \mathcal{D}} \left[\mathbb{E}_{\boldsymbol{a}_t \sim \pi_\phi(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}[\alpha \log \pi_\phi(\boldsymbol{a}_t \mid \boldsymbol{s}_t) - Q_\theta(\boldsymbol{s}_t, \boldsymbol{a}_t)]\right]
$$
通常的方法中，这里的 $Q$ 函数仅仅作为一种基线，因而不会有梯度流入 $Q$ 函数，而在软演员-评论家中，考虑使用易处理的策略类型，这样我们可以使用重参数化技巧，考虑  
$$
\boldsymbol{a}_t = f_\phi(\epsilon_t;\boldsymbol{s}_t)
$$
进而可以重写损失为
$$
J_\pi(\phi) = \mathbb{E}_{\boldsymbol{s}_t \sim \mathcal{D}, \epsilon_t \sim \mathcal{N}} \left[\alpha \log \pi_\phi(f_\phi(\epsilon_t;\boldsymbol{s}_t) \mid \boldsymbol{s}_t) - Q_\theta(\boldsymbol{s}_t, f_\phi(\epsilon_t;\boldsymbol{s}_t))\right]
$$
从而得到梯度的估计（去掉外层的期望）
$$
\nabla_\phi J_\pi(\phi) = \nabla_\phi \alpha\log \pi_\phi(f_\phi(\epsilon_t;\boldsymbol{s}_t) \mid \boldsymbol{s}_t) + (\nabla_{\boldsymbol{a}_t} \alpha \log \pi_\phi(\boldsymbol{a}_t \mid \boldsymbol{s}_t) - \nabla_{\boldsymbol{a}_t} Q_\theta(\boldsymbol{s}_t, \boldsymbol{a}_t)) \nabla_\phi f_\phi(\epsilon_t;\boldsymbol{s}_t)
$$

####  4.4.4 Automatic temperature adjustment
事实上，软演员-评论家算法容易受到温度 $\alpha$ 的影响，且由于通常随着训练的进行，策略的随机性应当随之下降，这里可以使用一个自动调整的方，构造一个约束优化问题：
$$
\max_{\pi_1,\ldots, \pi_T} \mathbb{E}\left[\sum_{t = 1}^T r(\boldsymbol{s}_t,\boldsymbol{a}_t)\right], \text{ s.t. } \mathcal{H}(\pi_t(\cdot\mid\boldsymbol{s}_t)) \geq \mathcal{H}_0
$$
这里 $\mathcal{H}_0$ 是一个理想的熵限制条件，论文中的选择是 $-\dim(\mathcal{A})$。

值得注意的是，对于连续概率分布，熵没有离散分布那样的明确含义（例如平均编码长度等），例如熵也不一定是正数，如一元高斯分布的熵是 $\log (\sqrt{2e\pi \sigma^2})$，如果 $\sigma$ 比较小完全可以是负数。而事实上，如果高斯分布的基础上还进行一个 $\tanh$ 变换（对于原论文中的 HalfCheetah-v1 需要将动作分量限制在 $[-1,1]$），则熵还会减小。
  
最终可以得到的结果是（可以参见 Policy Gradient Algorithms | Lil'Log 的详细推导），对于温度 $\alpha$ 优化如下的目标函数：
$$
J(\alpha) = \mathbb{E}_{\boldsymbol{a}_t \sim \pi_t} \left[-\alpha \log \pi_t(\boldsymbol{a}_t \mid \boldsymbol{s}_t) - \alpha \mathcal{H}_0\right]
$$
具体来说我们依然梯度下降来优化 $\alpha$ .

![](17-6.png)

参见：Soft Actor-Critic Algorithms and Applications, Haarnoja et al., 2018（有另一篇更早的论文也是关于软演员-评论家的：Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine, '2018）

### 4.5 Side Note on Energy-Based Policies
上述的几种包含策略的算法实际上都可以视作是基于能量的策略，我们会定义一个能量函数 $E(\boldsymbol{s}, \boldsymbol{a})$ 来衡量在状态 $\boldsymbol{s}$ 下采取动作 $\boldsymbol{a}$ 的代价或"不合理性"。策略由能量函数确定，其形式为
$$
\pi(\boldsymbol{a} \mid \boldsymbol{s}) = \frac{1}{Z(\boldsymbol{s})} \exp(-E(\boldsymbol{s}, \boldsymbol{a}))
$$
这里的 $Z$ 是归一化因子，使得
$$
\frac{1}{Z(\boldsymbol{s})}\int \exp(-E(\boldsymbol{s}, \boldsymbol{a})) \text{d}\boldsymbol{a} = 1
$$

实际上基于能量的策略等价于软价值函数，考虑 Q 函数 $Q(\boldsymbol{s}, \boldsymbol{a}): \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ 使得
$$
E(\boldsymbol{s}, \boldsymbol{a}) = -\frac{1}{\alpha}Q(\boldsymbol{s}, \boldsymbol{a})
$$
即可得到软价值函数的形式。

事实上，基于之前的推导还可以发现最大熵策略和软价值函数也是等价的，因此基于能量的策略、软 Q 学习与熵正则化策略梯度三者是等价的。

### 4.6 Benefits of soft optimality
软最优性的好处：
- 事实上这是对人类行为的一种更好的建模；
- 可以改善探索并避免熵坍缩，由于得到的策略有更高的随机性，这非常有利于微调的过程； 
- 相较于最大化操作可以更好地处理两个动作概率相近的情况；
- 对于各种干扰可以更加鲁棒，因为能够覆盖更广的状态，能从这些干扰中恢复；
- 可以逐渐降低温度，从而更加接近贪心策略，当奖励信号足够强时。

### 4.7 Example Methods and Applications
示例，基于能量的随机性策略

在基于能量的策略中，如果 $Q$ 函数（能量函数）有两个峰值，则两个峰值都会得到探索，直到完全搞清楚哪一个更好。
![](https://pic3.zhimg.com/v2-c0ec913e13c1d0437b2480aeba03138e_1440w.jpg)
不难发现，在常规方法下，如果这个 $Q$ 函数有两个峰值，有可能最终的结果是只有一个更高的会得到探索。
![](17-7.png)
与此同时，这是一种预训练中的有效做法，使用软 $Q$ 学习得到的策略具有较大的熵，探索能力更强，由于能够覆盖更多的状态，通常只需要更短的时间进行微调。
![](17-8.png)
相较于深度确定性策略梯度（Deep Deterministic Policy Gradient，DDPG），基于软最优性的算法能够更好地应用到预训练+微调的情境中（因为得到的策略有更大的随机性）。

参见：Reinforcement Learning with Deep Energy-Based Policies, Haarnoja et al., 2017

其他一些机器人的相关的应用：
- Haarnoja, Pong, Zhou, Dalal, Abbeel, L. Composable Deep Reinforcement Learning for Robotic Manipulation. '18  
- Haarnoja, Zhou, Ha, Tan, Tucker, L. Learning to Walk via Deep Reinforcement Learning. '19  

## 5 Summary
本节中，我们讨论了如何将强化学习视作概率图模型中的推断。
- 我们可以把价值函数视作是一种反向信息；
- 最大化奖励与熵的和（自然地，当奖励增大时，熵的相对影响会减小）；
- 使用变分推断来避免直接推断中的过高估计问题；
- 我们介绍了一系列利用软最优性的算法，例如软 Q 学习、熵正则化策略梯度与软演员 - 评论家。
- 这些算法更好地建模了人类行为，并且能够更好地处理探索问题； 
- 事实上，我们可以发现基于能量的策略、软 Q 学习、熵正则化策略梯度这三种方法是等价的。