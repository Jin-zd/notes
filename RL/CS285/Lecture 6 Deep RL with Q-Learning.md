# 1 Introduction to Problems in Q-Learning
我们之前推导的 Q 迭代 / Q 学习有哪些问题？
- 更新 $\phi$ 的方式实质上并不是梯度下降，而是半梯度下降。考虑利用梯度下降更新 $\phi$ 的时候，并没有梯度流入 $y_i = r(\boldsymbol{s}_i, \boldsymbol{a}_i) + \gamma \max_{\boldsymbol{a}_i'} Q_\phi(\boldsymbol{s}_i', \boldsymbol{a}_i')$ 一项。事实上，在 pytorch 等框架中实现时，我们会在计算 $y_i$ 设置为不计算梯度。
- 许多前后的样本之间是相关的，这与通常的监督学习的[[Concepts#1 独立同分布（i.i.d.）|独立同分布（i.i.d.）]]假设不符。我们很有可能不断地过拟合当前所在的小区域。

后者相对容易解决，不妨考虑以下两种方式：
- 参考演员-评论家算法中介绍的解决方法：进行并行，也就是多个工作线程同时进行采样，可以进一步分为同步和异步两种方式。
![](6-1.png)
- 使用回放缓冲区（replay buffer），也就是将之前的样本存储在一个缓冲中，每次从缓冲中采样进行更新，具体来说可以考虑如下带回放缓冲区的 Q 学习（Q-learning with replay buffer）算法：
	- 从缓冲 $\mathcal{B}$ 采样一个批量 $\{(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)\}$；
	- 令 $\phi \gets \arg \min_\phi \frac{1}{2} \left\|Q_\phi(\boldsymbol{s}_i, \boldsymbol{a}_i) - y_i\right\|^2$。
![](6-2.png)

此时我们的样本不再相关，而且由于我们每个批量有很多的样本，可以降低方差。

不过我们依然需要周期地利用一些策略来采样，于是我们可以使用 $\epsilon$ 贪心策略来进行探索。将所有整理起来就是：
带回放缓冲区的 Q 学习（Q-learning with replay buffer）：
1. 从环境中采样，$\{(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)\}$，存入缓冲区 $\mathcal{B}$；
2. 重复以下 $K$ 次；
3. 从缓冲区 $\mathcal{B}$ 采样一个批量 $\{(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)\}$；
4. 令 $\phi \gets \arg \min_\phi \frac{1}{2} \sum_{i = 1}^N \left\|Q_\phi(\boldsymbol{s}_i, \boldsymbol{a}_i) - \left[r(\boldsymbol{s}_i, \boldsymbol{a}_i) + \gamma \max_{\boldsymbol{a}_i'} Q_{\phi}(\boldsymbol{s}_i', \boldsymbol{a}_i')\right]\right\|^2$。

# 2 Target Networks
回顾前面我们提到的问题，即我们的更新方式不是梯度下降。然而我们通常不会使用全梯度下降，因为这样的方式在实际中效果并不如半梯度下降（缺乏 SGD 中的一些随机性）。我们不妨在半梯度下降的基础上进行改进。

让我们感觉不舒服的是，我们的目标 $y_i$ 依赖于 $Q_\phi$，我们每进行一次更新，我们的目标也会更新，就像我们在拟合一个移动的目标，这也容易导致训练不稳定。我们这里考虑引入目标网络，于是我们的算法变为：
1. 更新网络： $\phi' \gets \phi$；
2. 重复以下 $N$ 次：
3. 从环境中采样，$\{(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)\}$, 存入缓冲区 $\mathcal{B}$；
4. 重复以下 $K$ 次：
5. 从缓冲区 $\mathcal{B}$ 采样一个批量 $\{(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)\}$；
6. 令 $\phi \gets \phi - \nabla_\phi \frac{1}{2} \sum_{i = 1}^N \left\|Q_\phi(\boldsymbol{s}_i, \boldsymbol{a}_i) - \left[r(\boldsymbol{s}_i, \boldsymbol{a}_i) + \gamma \max_{\boldsymbol{a}_i'} Q_{\phi'}(\boldsymbol{s}_i', \boldsymbol{a}_i')\right]\right\|^2$。

通常 $K$ 的值会比较小，也就是说我们不会在每次更新中都进行多次的梯度下降，这与监督学习中是很接近的。而 $N$ 会是一个比较大的值（例如 $10000$），因为我们需要保证我们的目标网络是比较稳定的。

我们可以给出经典的深度 Q 学习（Deep Q-learning）算法：
1. 采取动作 $\boldsymbol{a}_i$，得到 $(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)$，添加到缓冲区 $\mathcal{B}$；
2. 从缓冲区 $\mathcal{B}$ 采样一个批量 $\{(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)\}$；
3. 利用目标网络计算目标 $y_i = r(\boldsymbol{s}_i, \boldsymbol{a}_i) + \gamma \max_{\boldsymbol{a}_i'} Q_{\phi'}(\boldsymbol{s}_i', \boldsymbol{a}_i')$；
4. $\phi \gets \phi - \nabla_\phi \frac{1}{2} \sum_{i = 1}^N \left\|Q_\phi(\boldsymbol{s}_i, \boldsymbol{a}_i) - y_i\right\|^2$；
5. 更新 $\phi'$：每 $N$ 步。

这实际上是上面的算法的一个特例，也就是 $K = 1$。

我们还有其他设计目标网络的方式，当前的方式的一个奇怪的地方在于，我们目标网络在一些时间点上会突然发生变化，在一些时候会保持不变，但是在一些地方会感觉明显在拟合移动目标。一个可能的解决方式是使用波利亚克平均，也称参数平均（Polyak Averaging），也就是：
$$
\phi' \gets \tau \phi + (1 - \tau) \phi'
$$
常见使用的超参数为 $\tau = 0.999$。
![](6-3.png)

# 3 A General View of Q-Learning

事实上，我们有一种更一般的方式来看待 Q 学习，我们可以将其视作这三个过程的组合：
- 过程1：数据收集（包括旧数据清理）；
- 过程2：目标更新；
- 过程3：Q 函数回归。 
![](6-4.png)
通常这三个过程同时进行，但它们具有不同的时间尺度（可以真的是并发进行的，也可以是不同频率的顺序执行）。

# 4 Improving Q-Learning
## 4.1 Double Q-Learning

我们的 Q 值准确吗？从相对值的角度来看，我们的 Q 值是准确的，至少能够区分出哪个动作更好。但是从绝对值的角度来看，我们的 Q 值是不准确的。实践发现，我们预测的 Q 值通常比真实值要高很多。这一现象称为 Q 学习中的高估问题（overestimation in Q-learning），考虑： 
$$
y_j = r_j + \gamma \max_{\boldsymbol{a}'_j} Q_{\phi'}(\boldsymbol{s}'_j, \boldsymbol{a}'_j).
$$
核心的问题是，对于两个随机变量 $X_1, X_2$，我们有
$$
\mathbb{E}[\max(X_1, X_2)] \geq \max(\mathbb{E}[X_1], \mathbb{E}[X_2])
$$
对于 Q 学习，因为我们的 $Q_{\phi'}$ 并不完美，故表现为有噪声的。我们这里取的 max 类似于不等式的左侧，实际上会高于真实的最大值。
![](6-5.png)
注意到 
$$
\max_{\boldsymbol{a}'} Q_{\phi'}(\boldsymbol{s}', \boldsymbol{a}') = Q_{\phi'}(\boldsymbol{s}', \arg\max_{\boldsymbol{a}'} Q_{\phi'}(\boldsymbol{s}', \boldsymbol{a}'))
$$
如果我们能够把选取 $\arg\max_{\boldsymbol{a}'}$ 与获取对应的值这两个过程分开，也就是使用两个”独立“的网络，由于两个网络参数不同，它们的噪声也会不同，就能很大程度上解决这一问题。这样的方式称为双 Q 学习（Double Q-learning），具体来说，考虑以下两个网络:：
$$
Q_{\phi_A}(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \gamma Q_{\phi_B}(\boldsymbol{s}', \arg\max_{\boldsymbol{a}'} Q_{\phi_A}(\boldsymbol{s}', \boldsymbol{a}'))
$$
$$
Q_{\phi_B}(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \gamma Q_{\phi_A}(\boldsymbol{s}', \arg\max_{\boldsymbol{a}'} Q_{\phi_B}(\boldsymbol{s}', \boldsymbol{a}'))
$$
在实际中，我们只需要利用已有的两个网络即可，也就是：
$$
y = r + \gamma Q_{\phi'}(\boldsymbol{s}', \arg\max_{\boldsymbol{a}'} Q_\phi(\boldsymbol{s}', \boldsymbol{a}'))
$$
这样的做法并非完美，因为我们的两个网络会周期性相等，但这已经足以基本解决高估的问题。

## 4.2 Multi-step Q-Learning
事实上在更新式 
$$
y_{j,t} = r_{j,t} + \gamma \max_{\boldsymbol{a}_{j, t + 1} } Q_{\phi'}(\boldsymbol{s}_{j, t + 1}, \boldsymbol{a}_{j, t + 1})
$$
中，当我们 $Q$ 值估计很差时，此时主要的价值信号都来自于 $r_{j,t}$；而在 $Q$ 值估计较好时，我们的价值信号主要来自于 $Q_{\phi'}(\boldsymbol{s}_{j, t + 1}, \boldsymbol{a}_{j, t + 1})$。Q 学习事实上会有很大的偏差，很小的方差，这一现象和演员-评论家算法是相似的，我们可以借鉴演员-评论家算法中的多步回报： 
$$
y_{j,t} = \sum_{t' = t}^{t + N -1} \gamma^{t - t'} r_{j,t'} + \gamma^N \max_{\boldsymbol{a}_{j, t + N} } Q_{\phi'}(\boldsymbol{s}_{j, t + N}, \boldsymbol{a}_{j, t + N})
$$
这样的处理可以实现更低的偏差，尤其是 Q 值估计较差时，而且这样的处理通常会加速学习过程。

然而，与演员-评论家中的多步回报和广义优势估计（GAE）一样的是，这样的做法只适用于 on-policy 的情况下才是有效的，因为我们的新策略可能不会再采取之前的动作，最终到达 $\boldsymbol{s}_{t + N}$。

**Example 1**. 在 $N = 1$ 的情形中，我们目标对应的原始形式是
$$
y_{j,t} = r_{j,t} + \gamma \mathbb{E}_{\boldsymbol{s} \sim p(\boldsymbol{s} \mid \boldsymbol{s}_{j, t}, \boldsymbol{a}_{j, t})} \left[\max_{\boldsymbol{a}} Q_{\phi'}(\boldsymbol{s}, \boldsymbol{a})\right]
$$
注意我们收集到的数据是 $(\boldsymbol{s}_{j, t}, \boldsymbol{a}_{j, t}, r_{j,t}, \boldsymbol{s}_{j, t + 1})$，其中的 $\boldsymbol{s}_{j, t + 1}$ 仅仅与环境有关，故我们得到的依然是一个无偏估计。

然而，在 $N > 1$ 的情形中，我们目标对应的原始形式包含一系列嵌套的期望，其中也包含$\boldsymbol{a}_{t + 1} \sim \pi_{\phi'}(\boldsymbol{a} \mid \boldsymbol{s}_{t + 1})$，这一期望针对的是最新策略。而我们的数据 $\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}, \boldsymbol{a}_{t + 1}, \ldots, \boldsymbol{s}_{t + N}$, $Q(\boldsymbol{s}_t, \boldsymbol{a}_t)$ 对应的是生成轨迹的策略，如果我们用这一轨迹估计当前的 $Q$ 值，那么这一估计就是有偏差的。

那么如何解决这个问题呢？主要有以下几种方式：
- 忽略这一问题，通常表现也很好；
- 人为加工数据，动态地选择 $N$ 使得所有数据都是 on-policy 的；
- 重要性采样。

# 5 Q-Learning with Continuous Actions
我们知道对于 Q 函数，在动作空间离散的情况下，我们会输入一个 $\boldsymbol{s} \in \mathcal{S}$，输出 $\mathcal{A}$ 上全体动作的 Q 值。而在连续动作空间情况下，事实上我们没有办法直接输出全体动作的 Q 值，因而只能考虑一个 $\mathcal{S} \times \mathcal{A}$ 到 Q 值的映射。  
但是随之而来的问题是，我们如何得到 $\arg\max$ 呢？
## 5.1 Optimization
我们可以使用优化方式来解决这个问题，例如 SGD。

另一个简单的做法是随机优化，也就是 
$$
\max_{\boldsymbol{a}} Q(\boldsymbol{s}, \boldsymbol{a}) \approx \max\{Q(\boldsymbol{s}, \boldsymbol{a}_1), \ldots, Q(\boldsymbol{s}, \boldsymbol{a}_N)\}
$$
其中 $\boldsymbol{a}_1, \ldots, \boldsymbol{a}_N$ 是从某个分布中采样的（例如均匀分布）。
- 优点：简单，容易并行。  
- 缺点：不准确，但是一些时候或许无需过于担心不准确的问题。  

更加准确的基于优化的方式是交叉熵方法（Cross-Entropy Method，CEM），以及协方差矩阵自适应进化策略（Covariance Matrix Adaptation Evolution Strategy，CMA-ES）这些方法我们会在[[Lecture 8 Optimal Control and Planning]]一节中详细介绍。
这些方式对于维度在 $40$ 左右以内的问题是有效的。

## 5.2 Easily maximizable Q-functions
另一个方式是使用容易优化的函数类，例如使用二次函数：
$$
Q_\phi(\boldsymbol{s}, \boldsymbol{a}) = -\frac{1}{2} (\boldsymbol{a} - \boldsymbol{\mu}_\phi(\boldsymbol{s}))^T P_\phi(\boldsymbol{s}) (\boldsymbol{a} - \boldsymbol{\mu}_\phi(\boldsymbol{s})) + V_\phi(\boldsymbol{s})
$$
上述使用二次函数的方式称为归一化优势函数（Normalized Advantage Functions，NAF），此时获取 argmax 与 max 都变得非常简单：
$$
\arg\max_{\boldsymbol{a}} Q(\boldsymbol{s}, \boldsymbol{a}) = \mu_\phi(\boldsymbol{s})
$$
$$
\max_{\boldsymbol{a}} Q(\boldsymbol{s}, \boldsymbol{a}) = V_\phi(\boldsymbol{s})
$$
- 优点：无需改变算法，算法效率也不会受到影响。
- 缺点：会损失一些表示能力。
![](6-6.png)

## 5.3 Learn an approximate maximizer
在这一方式中，我们学习一个近似最大化器，即训练另一个网络来获取 argmax，使得 
$$
\mu_\theta(\boldsymbol{s}) \approx \arg\max_{\boldsymbol{a}} Q_\phi(\boldsymbol{s}, \boldsymbol{a})
$$
一个例子是深度确定性策略梯度（Deep Deterministic Policy Gradient，DDPG）算法，由于 $\mu_\theta(\boldsymbol{s})$ 实际上可以理解为一个确定性的策略，故其实很像是一个确定性的演员-评论家算法。在这个算法中，我们设计新的目标：
$$
y_j = r_j + \gamma Q_{\phi'}(\boldsymbol{s}'_j, \mu_{\theta}(\boldsymbol{s}'_j))
$$
我们关心的是如何更新 $\theta$：不难明确我们的 $\mu_\theta$ 需要最大化 $Q_\phi$，换言之需要更新 $\theta$ 这一参数使得 $Q_\phi$ 增大，利用链式法则就有：
$$
\frac{\text{d}Q_{\phi}}{\text{d}\theta} = \frac{\text{d}Q_{\phi}}{\text{d}\boldsymbol{a}} \frac{\text{d}\boldsymbol{a}}{\text{d}\theta}
$$

我们可以得到 DDPG 算法如下：
1. 从环境中采样 $\{(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)\}$，存入缓冲区 $\mathcal{B}$；
2. 从缓冲区 $\mathcal{B}$ 采样一个批量 $\{(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)\}$；
3. 设置 $y_i = r(\boldsymbol{s}_i, \boldsymbol{a}_i) + \gamma Q_{\phi'}(\boldsymbol{s}_i', \mu_{\theta'}(\boldsymbol{s}_i'))$；
4. 令 $\phi \gets \phi - \alpha \sum_{i = 1}^N \nabla_\phi \left\|Q_\phi(\boldsymbol{s}_i, \boldsymbol{a}_i) - y_i\right\|^2$；
5. 令 $\theta \gets \theta + \beta \sum_{i = 1}^N \frac{d\mu}{d\theta}(\boldsymbol{s}_i) \nabla_{\boldsymbol{a}} Q_\phi(\boldsymbol{s}_i, \mu(\boldsymbol{s}_i))$；
6. 应用波参数平均更新 $\phi'$，$\theta'$。

可以注意到在 DDPG 算法中并没有用到我们之前介绍的很多稳定性技巧，例如双 Q 学习。从相关算法的发展历史看，DDPG 算法是最早的算法之一，而后续这类型的算法如双延迟深度确定性策略梯度算法（Twin Delayed DDPG，TD3），除了引入我们介绍的双 Q 学习外，还引入了延迟策略更新（换言之更新 $\mu_\theta$ 的频率更低），目标策略平滑（向 $\mu_\theta$ 拟合的目标中添加噪声，防止策略过度利用 $Q$ 函数中的误差）。

# 6 Implementation Tips and Examples
基本提示：
- Q 学习相对来说不容易稳定，通常需要现在简单的任务上进行测试，确定算法的正确性。
- 使用更大的回放缓冲区通常会增强算法的稳定性。
- Q 学习的训练通常需要比较长的时间，并且在训练开始的很长一段时间可能并不会比随机策略好。
- 开始时，可以使用较大的 $\epsilon$ （用于探索），逐渐减小.  

高级技巧：
贝尔曼误差梯度（Bellman error gradients）： 
$$
\nabla_\phi \frac{1}{2} \sum_{i = 1}^N \left\|Q_\phi(\boldsymbol{s}_i, \boldsymbol{a}_i) - y_i\right\|^2
$$
可能非常大，因此梯度裁剪或者使用[[Concepts#11 胡贝尔损失（Huber Loss）|胡贝尔损失（Huber Loss）]]是有必要的。如下是胡贝尔损失的定义：
$$
L(x) = \begin{cases}  \frac{1}{2} x^2, & \text{if } |x| \leq \delta,\\  \delta|x| - \frac{\delta^2}{2}, & \text{otherwise}.  \end{cases}
$$
- 通常双 Q 学习会非常有帮助，这一技巧容易实现而且没有缺点。
- 多步回报也非常有帮助，但是也可能引入更高的方差。
- 在训练不同阶段进行探索以及学习率调整，尝试 Adam 等优化器通常也非常有帮助。
- 尝试多个随机种子，不同的随机种子可能有很大的影响。

# 7 Summary
在本节中，我们：
- 在前一节的基础上改进了基于 Q 函数的方法。
- 引入目标网络来稳定训练过程。
- 通过双 Q 学习缓解 Q 函数的高估问题。
- 引入多步回报来减小偏差（但是可能增加方差）。  
- 介绍了理解 Q 学习的一般方式，也就是将其视作三个过程的组合。
- 介绍了在 Q 学习中处理连续动作的方法，主要包括优化、使用容易获取极值的 Q 函数、以及学习一个近似最大化器。
- 最后我们给出了一些实现的提示。 
