# 1 Introduction to Problems in Q-Learning

我们之前推导的 Q-iteration / Q-learning 有哪些问题吗?

-   更新 $\phi$ 的方式实质上并不是梯度下降, 而是半梯度下降 (考虑利用梯度下降更新 $\phi$ 的时候, 并没有梯度流入 $y_i = r(\boldsymbol{s}_i, \boldsymbol{a}_i) + \gamma \max_{\boldsymbol{a}_i'} Q_\phi(\boldsymbol{s}_i', \boldsymbol{a}_i')$ 一项, 事实上, 在 pytorch 等框架中实现时, 我们会在计算 $y_i$ 设置为不计算梯度)  
-   许多前后的 sample 之间是相关的, 这与通常的 supervised regression 的 i.i.d. 假设不符. 我们很有可能不断地过拟合当前所在的小区域.  

后者相对容易解决, 不妨考虑以下两种方式:

-   参考 [actor-critic 算法](https://zhida.zhihu.com/search?content_id=253873463&content_type=Article&match_order=1&q=actor-critic+%E7%AE%97%E6%B3%95&zhida_source=entity)中我们介绍的解决方法: 进行 parallel, 也就是多个 worker 同时进行采样, 可以进一步分为 synchronous 和 asynchronous 两种方式.

![](https://pica.zhimg.com/v2-a7e042d85e048eac8480ed396f0cf604_1440w.jpg)

-   使用 [replay buffer](https://zhida.zhihu.com/search?content_id=253873463&content_type=Article&match_order=1&q=replay+buffer&zhida_source=entity), 也就是将之前的 sample 存储在一个 buffer 中, 每次从 buffer 中采样进行更新, 具体来说可以考虑如下**Q-learning with replay buffer** 算法:

![](https://pic3.zhimg.com/v2-8072f3dddc7141e7a498470471dd36d6_1440w.jpg)

-   从 buffer $\mathcal{B}$ 采样一个 batch $\{(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)\}$,  
-   令 $\phi \gets \arg \min_\phi \frac{1}{2} \left\|Q_\phi(\boldsymbol{s}_i, \boldsymbol{a}_i) - y_i\right\|^2$.

此时我们的 samples 不再相关, 而且由于我们每个 batch 有很多的 sample, 可以降低 variance.

不过我们依然需要周期地利用一些 policy 来采样, 于是我们可以使用 $\epsilon$\-greedy policy 来进行 exploration. 将所有整理起来就是:

**full Q-learning with replay buffer**:
1.  从环境中采样, $\{(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)\}$, 存入 buffer $\mathcal{B}$,  
2.  重复以下 $K$ 次:  
3.  从 buffer $\mathcal{B}$ 采样一个 batch $\{(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)\}$,  
4.  令 $\phi \gets \arg \min_\phi \frac{1}{2} \sum_{i = 1}^N \left\|Q_\phi(\boldsymbol{s}_i, \boldsymbol{a}_i) - \left[r(\boldsymbol{s}_i, \boldsymbol{a}_i) + \gamma \max_{\boldsymbol{a}_i'} Q_{\phi}(\boldsymbol{s}_i', \boldsymbol{a}_i')\right]\right\|^2$.

# 2 Target Networks

回顾前面我们提到的问题, 即我们的更新方式不是梯度下降. 然而我们通常不会使用全梯度下降, 因为这样的方式在实际中效果并不如版梯度下降 (缺乏 SGD 中的一些随机性). 我们不妨在半梯度下降的基础上进行改进:

让我们感觉不舒服的是, 我们的 target $y_i$ 依赖于 $Q_\phi$, 我们每进行一次更新, 我们的 target 也会更新, 就像我们在拟合一个移动的目标, 这也容易导致训练不稳定. 我们这里考虑引入 [target network](https://zhida.zhihu.com/search?content_id=253873463&content_type=Article&match_order=1&q=target+network&zhida_source=entity), 于是我们的算法变为:
1.  更新网络: $\phi' \gets \phi$  
2.  重复以下 $N$ 次:  
3.  从环境中采样, $\{(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)\}$, 存入 buffer $\mathcal{B}$,  
4.  重复以下 $K$ 次:  
5.  从 buffer $\mathcal{B}$ 采样一个 batch $\{(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)\}$,  
6.  令 $\phi \gets \phi - \nabla_\phi \frac{1}{2} \sum_{i = 1}^N \left\|Q_\phi(\boldsymbol{s}_i, \boldsymbol{a}_i) - \left[r(\boldsymbol{s}_i, \boldsymbol{a}_i) + \gamma \max_{\boldsymbol{a}_i'} Q_{\phi'}(\boldsymbol{s}_i', \boldsymbol{a}_i')\right]\right\|^2$.

通常 $K$ 的值会比较小, 也就是说我们不会在每次更新中都进行多次的梯度下降, 这与监督学习中是很接近的. 而 $N$ 会是一个比较大的值 (例如 $10000$), 因为我们需要保证我们的 target network 是比较稳定的.

我们可以给出 **"classic" deep Q-learning** 算法:
1.  采取 action $\boldsymbol{a}_i$, 得到 $(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)$, 添加到 buffer $\mathcal{B}$,  
2.  从 buffer $\mathcal{B}$ 采样一个 batch $\{(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)\}$,  
3.  利用 target network 计算 target $y_i = r(\boldsymbol{s}_i, \boldsymbol{a}_i) + \gamma \max_{\boldsymbol{a}_i'} Q_{\phi'}(\boldsymbol{s}_i', \boldsymbol{a}_i')$,  
4.  $\phi \gets \phi - \nabla_\phi \frac{1}{2} \sum_{i = 1}^N \left\|Q_\phi(\boldsymbol{s}_i, \boldsymbol{a}_i) - y_i\right\|^2$.  
5.  更新 $\phi'$: 每 $N$ 步

这实际上是上面的算法的一个特例, 也就是 $K = 1$.

我们还有其他设计 target network 的方式, 当前的方式的一个奇怪的地方在于, 我们 target network 在一些时间点上会突然发生变化, 在一些时候会保持不变, 但是在一些地方会感觉明显在拟合移动目标. 一个可能的解决方式是使用 **polyak averaging**, 也就是 
$$
\phi' \gets \tau \phi + (1 - \tau) \phi'
$$
常见使用的超参数为 $\tau = 0.999$.

![](https://pic1.zhimg.com/v2-d3e9ce3bb82f96a261f8b23a14dbc8ce_1440w.jpg)

# 3 A General View of Q-Learning

事实上, 我们有一种更一般的方式来看待 Q-learning, 我们可以将其视作这三个过程的组合:
-   process 1: data collection (包括 evict old data)  
-   process 2: target update  
-   process 3: Q-function regression  

![](https://pic1.zhimg.com/v2-b27d25dfa5b67255f7bebb29efbd1b56_1440w.jpg)

通常这三个过程同时进行, 但它们具有不同的时间尺度 (可以真的是并发进行的, 也可以是不同频率的顺序执行).

# 4 Improving Q-Learning

## 4.1 Double Q-Learning

我们的 Q-value 准确吗? 从相对值的角度来看, 我们的 Q-value 是准确的, 至少能够区分出哪个 action 更好. 但是从绝对值的角度来看, 我们的 Q-value 是不准确的, 实践发现, 我们预测的 Q-value 通常比真实值要高很多. 这一现象称为 **overestimation in Q-learning**, 考虑 
$$
y_j = r_j + \gamma \max_{\boldsymbol{a}'_j} Q_{\phi'}(\boldsymbol{s}'_j, \boldsymbol{a}'_j).
$$ 
核心的问题是, 对于两个随机变量 $X_1, X_2$, 我们有 
$$
\mathbb{E}[\max(X_1, X_2)] \geq \max(\mathbb{E}[X_1], \mathbb{E}[X_2])
$$
对于 Q-learning, 因为我们的 $Q_{\phi'}$ 并不 perfect, 故表现的 "noisy", 我们这里取的 max 类似于不等式的左侧, 实际上会高于真实的最大值.

![](https://pic4.zhimg.com/v2-deb2eeca3cc5004fbabcb5a0c924608f_1440w.jpg)

注意到 
$$
\max_{\boldsymbol{a}'} Q_{\phi'}(\boldsymbol{s}', \boldsymbol{a}') = Q_{\phi'}(\boldsymbol{s}', \arg\max_{\boldsymbol{a}'} Q_{\phi'}(\boldsymbol{s}', \boldsymbol{a}'))
$$
如果我们能够把选取 $\arg\max_{\boldsymbol{a}'}$ 与获取对应的 value 这两个过程分开, 也就是使用两个"独立"的网络, 由于两个网络参数不同, 它们的 "noise" 也会不同, 就能很大程度上解决这一问题, 这样的方式称为 **[double Q-learning](https://zhida.zhihu.com/search?content_id=253873463&content_type=Article&match_order=1&q=double+Q-learning&zhida_source=entity)**, 具体来说, 考虑以下两个网络: 
$$
Q_{\phi_A}(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \gamma Q_{\phi_B}(\boldsymbol{s}', \arg\max_{\boldsymbol{a}'} Q_{\phi_A}(\boldsymbol{s}', \boldsymbol{a}'))
$$ 
$$
Q_{\phi_B}(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \gamma Q_{\phi_A}(\boldsymbol{s}', \arg\max_{\boldsymbol{a}'} Q_{\phi_B}(\boldsymbol{s}', \boldsymbol{a}'))
$$
在实际中, 我们只需要利用已有的两个网络即可, 也就是: 
$$
y = r + \gamma Q_{\phi'}(\boldsymbol{s}', \arg\max_{\boldsymbol{a}'} Q_\phi(\boldsymbol{s}', \boldsymbol{a}'))
$$
这样的做法并非完美, 因为我们的两个网络会周期性相等, 但这已经足以基本解决 overestimation 的问题.

## 4.2 Multi-step Q-Learning

事实上在更新式 
$$
y_{j,t} = r_{j,t} + \gamma \max_{\boldsymbol{a}_{j, t + 1} } Q_{\phi'}(\boldsymbol{s}_{j, t + 1}, \boldsymbol{a}_{j, t + 1})
$$
中, 当我们 $Q$ 值估计很差时, 此时主要的 value signal 都来自于 $r_{j,t}$, 而在 $Q$ 值估计较好时, 我们的 value signal 主要来自于 $Q_{\phi'}(\boldsymbol{s}_{j, t + 1}, \boldsymbol{a}_{j, t + 1})$. Q-learning 事实上会有很大的 bias, 很小的 variance, 这一现象和 actor-critic 算法是相似的, 我们可以借鉴 actor-critic 算法中的 n-step return 
$$
y_{j,t} = \sum_{t' = t}^{t + N -1} \gamma^{t - t'} r_{j,t'} + \gamma^N \max_{\boldsymbol{a}_{j, t + N} } Q_{\phi'}(\boldsymbol{s}_{j, t + N}, \boldsymbol{a}_{j, t + N})
$$
这样的处理可以实现更低的 bias, 尤其是 Q-value 估计较差时. 而且这样的处理通常会加速学习过程.

然而, 与 actor-critic 中的 n-step return 和 GAE 一样的是, 这样的做法只适用于 on-policy 的情况下才是有效的, 因为我们的新 policy 可能不会再采取之前的 action, 最终到达 $\boldsymbol{s}_{t + N}$.

**Example 1**. 在 $N = 1$ 的情形中, 我们 target 对应的原始形式是
$$
y_{j,t} = r_{j,t} + \gamma \mathbb{E}_{\boldsymbol{s} \sim p(\boldsymbol{s} \mid \boldsymbol{s}_{j, t}, \boldsymbol{a}_{j, t})} \left[\max_{\boldsymbol{a}} Q_{\phi'}(\boldsymbol{s}, \boldsymbol{a})\right]
$$
注意我们收集到的数据是_ $(\boldsymbol{s}_{j, t}, \boldsymbol{a}_{j, t}, r_{j,t}, \boldsymbol{s}_{j, t + 1})$_, 其中的_ $\boldsymbol{s}_{j, t + 1}$ 仅仅与环境有关, 故我们得到的依然是一个 unbiased estimate.

然而, 在 $N > 1$ 的情形中, 我们 target 对应的原始形式包含一系列嵌套的期望, 其中也包含$\boldsymbol{a}_{t + 1} \sim \pi_{\phi'}(\boldsymbol{a} \mid \boldsymbol{s}_{t + 1})$, 这一期望针对的是最新 policy. 而我们的数据 $\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}, \boldsymbol{a}_{t + 1}, \ldots, \boldsymbol{s}_{t + N}$, $Q(\boldsymbol{s}_t, \boldsymbol{a}_t)$ 对应的是生成轨迹的 policy, 如果我们用这一轨迹估计当前的 $Q$ 值, 那么这一估计就是 biased 的.

那么如何解决这个问题呢? 主要有以下几种方式:
-   忽略这一问题, 通常表现也很好  
-   人为加工 data, 动态地选择 $N$ 使得所有数据都是 on-policy 的  
-   importance sampling.  

# 5 Q-Learning with Continuous Actions

我们知道对于 Q function, 在 action space 离散的情况下, 我们会输入一个 $\boldsymbol{s} \in \mathcal{S}$ , 输出 $\mathcal{A}$ 上全体 actions 的 Q-value. 而在连续 action space 情况下, 事实上我们没有办法直接输出全体 actions 的 Q-value, 因而只能考虑一个 $\mathcal{S} \times \mathcal{A}$ 到 Q-value 的映射.  
  
但是随之而来的问题是, 我们如何得到 $\arg\max$ 呢?

## 5.1 Optimization

我们可以使用优化方式来解决这个问题, 例如 SGD.

另一个简单的做法是 stochastic optimization, 也就是 
$$
\max_{\boldsymbol{a}} Q(\boldsymbol{s}, \boldsymbol{a}) \approx \max\{Q(\boldsymbol{s}, \boldsymbol{a}_1), \ldots, Q(\boldsymbol{s}, \boldsymbol{a}_N)\}
$$
其中 $\boldsymbol{a}_1, \ldots, \boldsymbol{a}_N$ 是从某个分布中采样的 (例如均匀分布).
-   优点: 简单, 容易并行  
-   缺点: 不准确, 但是一些时候或许无需过于担心不准确的问题.  

更加准确的基于优化的方式是 CEM (cross-entropy method), 以及 CMA-ES (covariance matrix adaptation evolution strategy), 这些方法我们会在 **Optimal Control** 一节中详细介绍.

**Remark:** 这些方式对于维度在 $40$ 左右以内的问题是有效的.

## 5.2 Easily maximizable Q-functions

另一个方式是使用容易优化的 function class, 例如使用二次函数 
$$
Q_\phi(\boldsymbol{s}, \boldsymbol{a}) = -\frac{1}{2} (\boldsymbol{a} - \boldsymbol{\mu}_\phi(\boldsymbol{s}))^T P_\phi(\boldsymbol{s}) (\boldsymbol{a} - \boldsymbol{\mu}_\phi(\boldsymbol{s})) + V_\phi(\boldsymbol{s})
$$
上述使用二次函数的方式称为 **NAF (Normalized Advantage Functions)**, 此时获取 argmax 与 max 都变得非常简单, 
$$
\arg\max_{\boldsymbol{a}} Q(\boldsymbol{s}, \boldsymbol{a}) = \mu_\phi(\boldsymbol{s})
$$

$$
\max_{\boldsymbol{a}} Q(\boldsymbol{s}, \boldsymbol{a}) = V_\phi(\boldsymbol{s})
$$

-   优点: 无需改变算法, 算法效率也不会受到影响  
    
-   缺点: 会损失一些表示能力

![](https://pic3.zhimg.com/v2-802ef39f52352eacc0abb6343d2e9d56_1440w.jpg)

## 5.3 Learn an approximate maximizer

在这一方式中, 我们学习一个 approximate maximizer, 即训练另一个网络来获取 argmax, 使得 
$$
\mu_\theta(\boldsymbol{s}) \approx \arg\max_{\boldsymbol{a}} Q_\phi(\boldsymbol{s}, \boldsymbol{a})
$$

一个例子是 **[DDPG](https://zhida.zhihu.com/search?content_id=253873463&content_type=Article&match_order=1&q=DDPG&zhida_source=entity) (Deep Deterministic Policy Gradient)** 算法, 由于 $\mu_\theta(\boldsymbol{s})$ 实际上可以理解为一个 deterministic 的 policy, 故其实很像是一个 deterministic actor-critic 算法. 在这个算法中, 我们设计新的 target 
$$
y_j = r_j + \gamma Q_{\phi'}(\boldsymbol{s}'_j, \mu_{\theta}(\boldsymbol{s}'_j))
$$
我们关心的是如何更新 $\theta$: 不难明确我们的 $\mu_\theta$ 需要最大化 $Q_\phi$, 换言之需要更新 $\theta$ 这一参数使得 $Q_\phi$ 增大, 利用链式法则就有 
$$
\frac{\text{d}Q_{\phi}}{\text{d}\theta} = \frac{\text{d}Q_{\phi}}{\text{d}\boldsymbol{a}} \frac{\text{d}\boldsymbol{a}}{\text{d}\theta}
$$

我们可以得到 DDPG 算法如下:
1.  从环境中采样, $\{(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)\}$, 存入 buffer $\mathcal{B}$,  
2.  从 buffer $\mathcal{B}$ 采样一个 batch $\{(\boldsymbol{s}_i, \boldsymbol{a}_i, \boldsymbol{s}_i', r_i)\}$,  
3.  设置 $y_i = r(\boldsymbol{s}_i, \boldsymbol{a}_i) + \gamma Q_{\phi'}(\boldsymbol{s}_i', \mu_{\theta'}(\boldsymbol{s}_i'))$,  
4.  令 $\phi \gets \phi - \alpha \sum_{i = 1}^N \nabla_\phi \left\|Q_\phi(\boldsymbol{s}_i, \boldsymbol{a}_i) - y_i\right\|^2$,  
5.  令 $\theta \gets \theta + \beta \sum_{i = 1}^N \frac{d\mu}{d\theta}(\boldsymbol{s}_i) \nabla_{\boldsymbol{a}} Q_\phi(\boldsymbol{s}_i, \mu(\boldsymbol{s}_i))$.  
6.  应用 Polyak averaging 更新 $\phi', \theta'$.  

可以注意到在 DDPG 算法中并没有用到我们之前介绍的很多稳定性技巧, 例如 double Q-learning, 从相关算法的发展历史看, DDPG 算法是最早的算法之一, 而后续这类型的算法如**[TD3](https://zhida.zhihu.com/search?content_id=253873463&content_type=Article&match_order=1&q=TD3&zhida_source=entity) (Twin Delayed DDPG)**, 除了引入我们介绍的 double Q-learning 外, 还引入了 **delayed policy update** (换言之更新 $\mu_\theta$ 的频率更低), **target policy smoothing** (向 $\mu_\theta$ 拟合的目标中添加噪声, 防止 policy 过度利用 $Q$ -function 中的误差).

# 6 Implementation Tips and Examples

**Basic tips:**

-   Q-learning 相对来说不容易 stabilize, 通常需要现在简单的任务上进行测试, 确定算法的正确性. 
-   使用更大的 replay buffer 通常会增强算法的稳定性.  
-   Q-learning 的训练通常需要比较长的时间, 并且在训练开始的很长一段时间可能并不会比随机策略好  
-   开始时, 可以使用较大的 $\epsilon$ (用于 exploration), 逐渐减小.  

**Advanced tips:**

Bellman error gradients $\nabla_\phi \frac{1}{2} \sum_{i = 1}^N \left\|Q_\phi(\boldsymbol{s}_i, \boldsymbol{a}_i) - y_i\right\|^2$ 可能非常大, 因此 clip gradients 或者使用 **Huber loss** 是有必要的. 如下是 Huber loss 的定义: 
$$
L(x) = \begin{cases}  \frac{1}{2} x^2, & \text{if } |x| \leq \delta,\\  \delta|x| - \frac{\delta^2}{2}, & \text{otherwise}.  \end{cases}
$$
-   通常 Double Q-learning 会非常有帮助, 这一技巧容易实现而且没有缺点  
-   N-step returns 也非常有帮助, 但是也可能引入更高的 variance.  
-   在训练不同阶段进行 exploration 以及 learning rate 调整, 尝试 Adam 等 optimizer 通常也非常有帮助.  
-   尝试多个 random seeds, 不同的 random seed 可能有很大的影响.  

# 7 Summary

在本节中, 我们:

-   在前一节的基础上改进了基于 Q-function 的方法, 主要的改进如下:  
-   引入 target network 来稳定训练过程  
-   通过 double Q-learning 缓解 Q-function 的 overestimation 问题  
-   引入 multi-step returns 来减小 bias (但是可能增加 variance)  
-   介绍了理解 Q-learning 的一般方式, 也就是将其视作三个过程的组合  
-   介绍了在 Q-learning 中处理 continuous action 的方法, 主要包括优化, 使用容易获取极值的 Q-function, 以及学习一个 approximate maximizer  
-   最后我们给出了一些实现的 tips.  
