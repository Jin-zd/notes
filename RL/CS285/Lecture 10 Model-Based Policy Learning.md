## 1 From open-loop planning to close-loop policy learning
回顾上一节介绍的 MPC 算法：
1. 运行基本策略 $\pi_0(\boldsymbol{a}_t, \boldsymbol{s}_t)$，收集 $\mathcal{D} = \{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')_i\}$；
2. 学习动态模型 $f(\boldsymbol{s}, \boldsymbol{a})$ 来最小化 $\sum_{i} \|f(\boldsymbol{s}_i, \boldsymbol{a}_i) - \boldsymbol{s}'_i\|^2$；
3. 依据 $f(\boldsymbol{s}, \boldsymbol{a})$ 来进行规划； 
4. 执行第一个规划的动作，观测到新的状态 $\boldsymbol{s}'$（MPC）；
5. 添加 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')$ 到 $\mathcal{D}$，重复 3-5，每 $N$ 次回到 2。

这实际上构造的是一种开环控制器，在学习的动态上进行最优控制，得到一条动作序列。核心在于，尽管进行了重规划，表面上每次只执行一个动作，但是与闭环的核心区别是，在进行规划时，并不知道未来还会重规划。
![](10-1.png)
鉴于开环空中的次优性，我们接下来考虑闭环控制。前面已经证明闭环控制中的目标与无模型的强化学习在形式上是一致的：
$$
\pi = \arg\max_{\pi} \mathbb{E}_{\tau \sim p(\tau)} \left[\sum_{t} r(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]
$$
![](10-2.png)
这里需要考虑的一点是 $\pi$ 的形式，主要的策略形式有两种：
- 全局策略： $\pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t)$，在所有时间步上都使用相同的映射，一般通过神经网络等全局函数表示，具有较强的泛化能力，可以适用于较多场景，在执行时不需要每次进行规划，而是直接基于当前状态选择动作。
- 局部策略：通常指轨迹优化或 MPC 中计算得到的时变策略，例如 $\boldsymbol{u}_t = \boldsymbol{K}_t \boldsymbol{s}_t + \boldsymbol{k}_t$，该策略只在当前规划的轨迹及其附近区域内有效，重规划时需要重新计算，虽然可能在当前状态附近表现非常好，但泛化到其他状态时性能不一定理想。

在本节中，我们考虑学习一个用神经网络表示的全局策略。

时间反向传播：在基于模型的设置中，由于有了一个动态模型，我们目标函数（整条轨迹的奖励）理论上可以利用 策略与动态模型表示出来。一个相当朴素的想法是，能否直接反向传播到策略呢？

这在理论上是可以实现的：
- 确定性动态：可以直接计算 $\boldsymbol{s}_{t + 1} = f(\boldsymbol{s}_t, \boldsymbol{a}_t)$，从而完全使用已知函数表达出目标函数，进而直接反向传播。
- 随机动态：可以使用重参数化技巧。
![](10-3.png)

基于这种朴素的想法，可以得到基于模型的强化学习 2.0 版本：
1. 运行基本策略 $\pi_0(\boldsymbol{a}_t, \boldsymbol{s}_t)$，收集 $\mathcal{D} = \{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')_i\}$；
2. 学习动态模型 $f(\boldsymbol{s}, \boldsymbol{a})$ 来最小化 $\sum_{i} \|f(\boldsymbol{s}_i, \boldsymbol{a}_i) - \boldsymbol{s}'_i\|^2$；
3. 通过反向传播来优化策略 $\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)$；
4. 添加 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')$ 到 $\mathcal{D}$，重复 $2-4$。

这样的算法有一些问题：开始位置动作的梯度涉及到多个[[Concepts#19 雅可比矩阵 (Jacobian Matrix)|雅可比矩阵 (Jacobian Matrix)]]相乘，因此容易梯度爆炸或梯度消失，产生较为严重的数值问题。具体来说，由于现在可以学习一个模型，我们也可以给出反向传播（逐路径）梯度（假设奖励为状态依赖的）：
$$
\nabla_\theta J(\theta) = \sum_{t = 1}^{T} \frac{\text{d}\boldsymbol{a}_t}{\text{d}\theta} \frac{\text{d}\boldsymbol{s}_{t + 1}}{\text{d}\boldsymbol{a}_t} \left(\sum_{t' = t + 1}^{T} \frac{\text{d}r_{t'}}{\text{d}\boldsymbol{s}_{t'}} \left(\prod_{t'' = t + 2}^{t'} \left(\frac{\text{d}\boldsymbol{s}_{t''}}{\text{d}\boldsymbol{a}_{t'' - 1}} \frac{\text{d}\boldsymbol{a}_{t''}}{\text{d}\boldsymbol{s}_{t'' - 1}} + \frac{\text{d}\boldsymbol{s}_{t''}}{\text{d}\boldsymbol{s}_{t'' - 1}}\right)\right)\right)
$$
其中造成数值问题的核心就在于上式中的 $\prod_{t'' = t + 2}^{t'}$ 项。

由于这里的处理方式与策略梯度的处理方式存在一定的差异，故详细地给出推导过程：

_Proof._ 
首先明确单条轨迹的总奖励为 $J(\theta) = \sum_{t=1}^T r(\boldsymbol{s}_t)$ （这里设定奖励为状态依赖的），而动作由 $\boldsymbol{a}_t = \pi_\theta(\boldsymbol{s}_t)$ 生成（如果是随机策略，则还需要使用重参数化技巧）。
考虑如下几步：
Step 1，梯度展开：对每个时间步 $t$，参数 $\theta$ 通过动作 $\boldsymbol{a}_t$ 影响后续奖励 $r_{t+1}, r_{t+2}, \dots$。梯度需累积这些影响：
$$
\nabla_\theta J(\theta) = \sum_{t=1}^T \left( \frac{\text{d}\boldsymbol{a}_t}{\text{d}\theta} \cdot \sum_{t' = t + 1}^T \frac{\text{d}r_{t'}}{\text{d}\boldsymbol{a}_t} \right)
$$
Step 2，链式法则应用：对于每个后续奖励 $r_{t'}$ ($t' \geq t + 1$)，其导数 $\frac{\text{d}r_{t'}}{\text{d}\boldsymbol{a}_t}$ 需沿路径 $t \rightarrow \cdots \rightarrow t'$ 展开为以下三部分：
- $\boldsymbol{a}_t$ 影响 $\boldsymbol{s}_{t+1}$（导数 $\frac{\text{d}\boldsymbol{s}_{t+1}}{\text{d}\boldsymbol{a}_t}$）。
- $\boldsymbol{s}_{t''}$ 影响 $\boldsymbol{a}_{t''}$ 和 $\boldsymbol{s}_{t'' + 1}$，其中 $t + 1 \leq t'' \leq t' - 1$。
- $\boldsymbol{s}_{t'}$ 直接影响 $r_{t'}$（导数 $\frac{\text{d}r_{t'}}{\text{d}\boldsymbol{s}_{t'}}$）。 

其中第一部分与 $t'$ 没有关系，因此可以提到最外面，与 $\frac{\text{d}\boldsymbol{a}_t}{\text{d}\theta}$ 乘在一起。
第二部分需要进一步展开。考虑 $\boldsymbol{s}_{t''} = f(\boldsymbol{s}_{t'' - 1}, \boldsymbol{a}_{t'' - 1})$，总导数为直接的转移 $\frac{\text{d}\boldsymbol{s}_{t''}}{\text{d}\boldsymbol{s}_{t''-1}}$ 与经过动作中介的转移 $\frac{\text{d}\boldsymbol{s}_{t''}}{\text{d}\boldsymbol{a}_{t''-1}} \cdot \frac{\text{d}\boldsymbol{a}_{t''-1}}{\text{d}\boldsymbol{s}_{t''-1}}$ 之和。由于第二部分被影响的下标从 $t + 2$ 到 $t'$，故可以得到一个连乘项：
$$
\prod_{t''=t+2}^{t'} \left( \frac{\text{d}\boldsymbol{s}_{t''}}{\text{d}\boldsymbol{a}_{t''-1}} \frac{\text{d}\boldsymbol{a}_{t''-1}}{\text{d}\boldsymbol{s}_{t''-1}} + \frac{\text{d}\boldsymbol{s}_{t''}}{\text{d}\boldsymbol{s}_{t''-1}} \right)
$$
Step 3，整理：
因此整理即可得到结果：
$$
\nabla_\theta J(\theta) = \sum_{t=1}^T \frac{\text{d}\boldsymbol{a}_t}{\text{d}\theta} \frac{\text{d}\boldsymbol{s}_{t+1}}{\text{d}\boldsymbol{a}_t} \left( \sum_{t'=t+1}^T \frac{\text{d}r_{t'}}{\text{d}\boldsymbol{s}_{t'}} \prod_{t''=t+2}^{t'} \left( \frac{\text{d}\boldsymbol{s}_{t''}}{\text{d}\boldsymbol{a}_{t''-1}} \frac{\text{d}\boldsymbol{a}_{t''-1}}{\text{d}\boldsymbol{s}_{t''-1}} + \frac{\text{d}\boldsymbol{s}_{t''}}{\text{d}\boldsymbol{s}_{t''-1}} \right) \right)
$$

备注：
- 这里的出现的数值问题在现象上类似于轨迹优化的打靶法的问题（从数值问题严重的角度，这些数值问题严重的问题不适合使用二阶优化方法）。
- 由于当前策略在各时间步都由 $\theta$ 关联（换言之我们的策略不再是局部的），因此也不能够利用 LQR 类型方法中的动态规划。
- 从深度学习的角度，这样的问题类似于在 RNN 等时间反向传播（BPTT）的网络的梯度消失或爆炸问题。但无法通过引入 LSTM 等结构与技巧来解决这个问题，因为我们不能人为设置一个动态。我们的 $f$ 必须接近于真正的动态。这可能听起来有些奇怪，不妨考虑如下计算图（可以想象为是一个没有输入的 RNN 网络，$\boldsymbol{s}_t$ 是隐藏状态，$\boldsymbol{a}_t$ 是输出，$f$ 是 RNN 的转移函数， $\pi_\theta$ 是输出层）：
![](10-4.png)
那么真正的解决方案是什么呢？使用无模型算法，只是使用模型来生成合成数据。这可能看起来有些奇怪，但实际上是效果相对的最好的，实际上可以认为是基于模型的无模型强化学习加速。

## 2 Model-Free Learning with a Model

回顾我们熟悉的策略梯度算法：
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i = 1}^{N} \sum_{t = 1}^{T} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i, t} \mid \boldsymbol{s}_{i, t}) \hat{Q}^{\pi}_{i,t}
$$
这实际上可以视作是一个梯度估计器，这里的好处是这里避免了动态，但这里的避免是相对的，因为我们依然需要进行采样。但由于模型的存在，可以使用模型来生成样本，从而减少在真实环境中的采样次数。

可以给出基于模型的强化学习 2.5 版本（基于模型的强化学习，通过策略梯度）：
1. 运行基本策略 $\pi_0(\boldsymbol{a}_t, \boldsymbol{s}_t)$，收集 $\mathcal{D} = \{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')_i\}$；
2. 学习动态模型 $f(\boldsymbol{s}, \boldsymbol{a})$ 来最小化 $\sum_{i} \|f(\boldsymbol{s}_i, \boldsymbol{a}_i) - \boldsymbol{s}'_i\|^2$；
3. 利用 $f$ 与策略 $\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})$ 生成 $\{\tau_i\}$； 
4. 利用 $\{\tau_i\}$ 改进 $\pi_{\theta}(\boldsymbol{a} \mid \boldsymbol{s})$，重复 3- 4。
5. 运行 $\pi_{\theta}(\boldsymbol{a} \mid \boldsymbol{s})$ 来收集数据，添加到 $\mathcal{D}$，重复 2- 5。

这依然不是大多数人会实际使用的基于模型的强化学习算法，这个算法依然有一些问题：对于长序列，模型误差会积累，导致分布偏移，这样的偏移还可能与策略的偏移叠加在一起累积为更大的误差，这样误差积累的速度与[[Lecture 1 Imitation Learning]]中同样为 $O(\epsilon T^2)$。
![](10-5.png)
我们不能通过较短的 推演来解决这个问题，因为这实际上改变了问题的条件（例如时间跨度）。

一个可能的做法是，使用一些现实世界的推演，然后从这些推演中选取一些状态，从这些出发进行 短模型推演。此时
- 有更小的误差；  
- 能见到所有的时间步；  
- 得到的状态分布并不正确。

长序列的分布偏移（左），短序列改变了问题条件（中），复合的推演（右）：
![](10-6.png)

这一做法中的分布不匹配是因为对于这些复合的推演，其真实采样的部分来源于较早的策略，而模型采样的部分可能来自于较新的策略。

事实上，这可以进一步引申出在基于模型的强化学习中使用 on-policy 的策略梯度方法可能并不合适：
- 使用基于模型的方法的目的是为了减少真实环境中的采样，故现实世界的推演的频率应该尽可能的低。  
- 为了减小分布不匹配，模型推演的频率应该尽可能的高。  

因此使用 Q-learning 之类的 off-policy 方法是一个更好的选择，这就得到了基于模型的强化学习 3.0 版本，这也是人们会使用的基于模型的强化学习算法：
1. 运行基本策略 $\pi_0(\boldsymbol{a}_t, \boldsymbol{s}_t)$，收集 $\mathcal{D} = \{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')_i\}$；
2. 学习动态模型 $f(\boldsymbol{s}, \boldsymbol{a})$ 来最小化 $\sum_{i} \|f(\boldsymbol{s}_i, \boldsymbol{a}_i) - \boldsymbol{s}'_i\|^2$；
3. 选择 $\mathcal{D}$ 中的状态 $\boldsymbol{s}_i$，利用 $f$ 生成短的推演；
4. 同时利用真实数据与模型数据改进 $\pi_{\theta}(\boldsymbol{a} \mid \boldsymbol{s})$，使用 off-policy RL，重复 3- 4。
5. 运行 $\pi_{\theta}(\boldsymbol{a} \mid \boldsymbol{s})$ 来收集数据，添加到 $\mathcal{D}$，重复 2- 5。

## 3 Dyna-Style Algorithms

在之前的算法基于模型的强化学习 3.0 版本中，我们实际还没有给出一些算法的具体形式。符合这一框架的一个经典算法是 Dyna：
1. 给定状态 $\boldsymbol{s}$，使用探索策略选择动作 $\boldsymbol{a}$；
2. 观测到 $\boldsymbol{s}', r$，得到转移 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}', r)$；
3. 更新动态模型与奖励模型；
4. Q 函数更新：$Q(\boldsymbol{s}, \boldsymbol{a}) \gets Q(\boldsymbol{s}, \boldsymbol{a}) + \alpha \mathbb{E}_{\boldsymbol{s'}, r} \left[r + \max_{\boldsymbol{a}'} Q(\boldsymbol{s}', \boldsymbol{a}') - Q(\boldsymbol{s}, \boldsymbol{a})\right]$；
5. 重复以下步骤 $K$ 次：
6. 采样 $(\boldsymbol{s}, \boldsymbol{a}) \sim \mathcal{B}$；
7. Q 函数更新：$Q(\boldsymbol{s}, \boldsymbol{a}) \gets Q(\boldsymbol{s}, \boldsymbol{a}) + \alpha \mathbb{E}_{\boldsymbol{s'}, r} \left[r + \max_{\boldsymbol{a}'} Q(\boldsymbol{s}', \boldsymbol{a}') - Q(\boldsymbol{s}, \boldsymbol{a})\right]$，只是这里的 $\boldsymbol{s}', r$ 是从模型中采样得到的。

这里有很多设计抉择，例如我们选择了 $(\boldsymbol{s}, \boldsymbol{a}) \sim \mathcal{B}$，而不是选择 $\boldsymbol{s} \sim \mathcal{B}$，然后用最新的策略来选择 $\boldsymbol{a}$。值得注意的是，这里的算法一方面可以提高数据利用率，而模型的存在可以让 $\mathbb{E}_{\boldsymbol{s'}, r} \left[r + \max_{\boldsymbol{a}'} Q(\boldsymbol{s}', \boldsymbol{a}') - Q(\boldsymbol{s}, \boldsymbol{a})\right]$ 估计更加准确（原先是单点估计，方差很大）。

结合上述 Dyna 算法，可以得到一个更加普遍形式的算法：
1. 收集数据，包含一系列转移 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}', r)$，放入缓冲 $\mathcal{B}$；
2. 学习动态模型，同时也可以学习奖励模型；
3. 重复以下步骤 $K$ 次： 
4. 采样 $\boldsymbol{s} \sim \mathcal{B}$；
5. 选择 $\boldsymbol{a}$ （从 $\mathcal{B}, \pi$ 或者随机）；
6. 利用模型生成 $\boldsymbol{s}'$ （与可能有的 $r$）；
7. 使用无模型的强化学习算法在 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}', r)$ 上进行更新；
8. （可选）进行 $N$ 步基于模型的步骤（在 $\boldsymbol{s}'$ 的基础上继续向前 $N$ 步）

![](10-7.png)

红色的部分为基于模型生成的轨迹，黑色的部分为缓冲中的轨迹。

这里最后一步是可选的，这样的做法可以更加充分地利用模型，从而加速 off-policy RL 的学习，但是也有一些问题，例如模型的误差会积累，这样的误差会导致策略的偏移。

## 4 A General View of Model-accelerated off-policy RL
- 过程 1：数据收集（放入 “回放缓冲区” B）（包括淘汰旧数据）；
- 过程 2：目标更新；
- 过程 3：Q 函数回归；
- 过程 4：模型训练（使用真实数据）；
- 过程 5：模型数据收集（放入基于模型的转换缓冲区）（每次改进模型时清除）。

这五个进程以不同的周期进行着，类似于我们在 Q-Learning 中的几个过程一样。
![](10-8.png)

也有一些变种，这里使用了一些其他设计抉择，参见：
- Model-Based Acceleration (MBA): Gu et al. Continuous deep Q-learning with model-based acceleration, 2016  
- Model-Based Value Expansion (MVE): Feinberg et al. Model-based value expansion for efficient model-free reinforcement learning, 2018  
- Model-Based Policy Optimization (MBPO): Janner et al. When to trust your model: model-based policy optimization, 2019  

上述提到的几个算法总的来说遵循以下过程：
1. 采取动作 $\boldsymbol{a}$，得到 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}', r)$，添加到回放缓冲区 $\mathcal{B}$；
2. 从 $\mathcal{B}$ 中采样 $\{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}', r)\}$；
3. 用 $\{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}', r)\}$ 更新动态模型；
4. 从 $\mathcal{B}$ 中采样 $\{\boldsymbol{s}_j\}$；
5. 用 $\{\boldsymbol{s}_j\}$ 与 $\boldsymbol{a} = \pi(\boldsymbol{s})$ 进行基于模型的推演；
6. 使用所有推演中的转移更新 $Q$ 函数。

与 Dyna 相比，利用模型生成了更长的轨迹：

![](10-9.png)
备注：
- 这里的做法与 Dyna 类似，但是由于使用了基于模型的推演而不是仅仅利用模型生成单个 $\boldsymbol{s}'$，可以让本身无模型的强化学习更加采样高效。
- 由于模型本身的不准确，这样的做法也有一些问题，可以利用不确定性来避免过度利用模型，例如使用自举聚集法。
- 不难注意到，这里的基于模型的推演可能比较奇怪，因为其既不完全基于收集真实推演的策略，同时也不完全基于当前的策略，尽管在实际中通常没有太大的问题，但是这意味着我们不能长时间不更新数据。

## 5 Multi-Step Models & Successor Representations
回顾介绍的两类强化学习算法：基于模型的强化学习与无模型的强化学习。在无模型的强化学习中，我们需要通过不断与环境交互来评估与改进策略，这样的做法采样效率较低。在基于模型的强化学习中，我们借助一个学习的模型来更高效地利用数据，评估与改进策略，但是基于模型的强化学习也有一些问题，例如模型的误差会积累，这样的误差会导致策略的偏移。

在某种意义上，基于模型的强化学习与无模型的强化学习是两种极端：
- 无模型的强化学习：不试图学习任何关于动态的信息，而事实上可能有一部分信息学习起来并不困难。
- 基于模型的强化学习：试图学习所有关于动态的信息，从现实的角度考虑，在一个极其复杂的环境中，恐怕没办法学习到所有的信息。

在本小节中，我们将介绍一些中间的方法，既尝试学习一些动态的信息，同时也不会试图学习整个动态，这一类方法在迁移学习中得到了很大的重视。

### 5.1 Successor representations
在基于模型的强化学习中，学习的模型可以用来生成一些数据，进而用这些数据来改进策略。归根结底，我们的模型是用来评估策略的，进而给出策略改进的方向。

接下来考虑什么样的表示可以用来评估策略？

一个表示想要能够评估策略，也就是需要能够给出 
$$
J(\pi) = \mathbb{E}_{\boldsymbol{s} \sim p(\boldsymbol{s}_1)} \left[V^\pi(\boldsymbol{s}_1)\right]
$$
这里略作一些调整，考虑仅仅依赖于状态的奖励（而不是之前讨论的状态 - 动作奖励），可以得到 
$$
\begin{aligned}  V^\pi(\boldsymbol{s}_t) &= \sum^{\infty}_{t = t'} \gamma^{t - t'} \mathbb{E}_{p(\boldsymbol{s}_{t'} \mid \boldsymbol{s}_t)} \left[r(\boldsymbol{s}_{t'})\right]\\  &= \sum_{t = t'}^{\infty} \gamma^{t - t'} \sum_{\boldsymbol{s}} p(\boldsymbol{s}_{t'} = \boldsymbol{s} \mid \boldsymbol{s}_t) r(\boldsymbol{s})\\  &= \sum_{\boldsymbol{s}}\left(\sum_{t' = t}^{\infty} \gamma^{t' - t} p(\boldsymbol{s}_{t'} = \boldsymbol{s} \mid \boldsymbol{s}_t)\right) r(\boldsymbol{s}) \end{aligned}
$$
这里考虑未来状态 $\boldsymbol{s}_{future}$ 的概念，有两种方式理解：
1. 依据 $Geom(\gamma)$ 随机选择一个未来的时间步 $t'$，这个时间步的状态 $\boldsymbol{s}_{t'}$。
2. 在每一时间步都有 $1 - \gamma$ 的概率停止，停止时所在的状态。

记：
$$
p_\pi(\boldsymbol{s}_{future} = \boldsymbol{s} \mid \boldsymbol{s}_t) = (1 - \gamma) \sum_{t' = t}^{\infty} \gamma^{t' - t} p(\boldsymbol{s}_{t'} = \boldsymbol{s} \mid \boldsymbol{s}_t)
$$
$p_\pi(\boldsymbol{s}_{future} = \boldsymbol{s} \mid \boldsymbol{s}_t)$ 的理解方式对应于以下两种：
1. 依据 $Geom(\gamma)$ 随机选择一个未来的时间步 $t'$，然后评估 $\boldsymbol{s}_{t'}$ 恰好为 $\boldsymbol{s}$ 的概率。
2. 在每一时间步都有 $1 - \gamma$ 的概率停止，恰好停在 $\boldsymbol{s}$ 的概率。 

事实上，应用这一概念，可以得到 
$$
\begin{aligned}  V^\pi(\boldsymbol{s}_t) &= \frac{1}{1 - \gamma} \sum_{\boldsymbol{s}} p_\pi(\boldsymbol{s}_{future} = \boldsymbol{s} \mid \boldsymbol{s}_t) r(\boldsymbol{s})\\  &= \frac{1}{1 - \gamma} \sum_{\boldsymbol{s}} \mu_{\boldsymbol{s}}^\pi(\boldsymbol{s}_t) r(\boldsymbol{s})\\  &= \frac{1}{1 - \gamma} \mu^\pi(\boldsymbol{s}_t)^T \overrightarrow{r}. \end{aligned}
$$
其中记 $\mu^\pi_i(\boldsymbol{s}_t) = p_\pi(\boldsymbol{s}_{future} = i \mid \boldsymbol{s}_t)$，其整合为向量形式即为 $\mu^\pi(\boldsymbol{s}_t)$，这一向量形式的表示称为后继表示（Successor representation）。这一表示同时包含模型与策略（以价值函数的形式）的信息，其与奖励的点积则可以还原得到价值函数的信息。

事实上，可以对 $\mu(\boldsymbol{s}_t)$ 做类似于[[Concepts#20 贝尔曼备份 (Bellman Backup)|贝尔曼备份（Bellman backup）]]的更新（考虑奖励为 $(1 - \gamma) \delta(\boldsymbol{s}_t = i)$）： 
$$
\begin{aligned}  \mu_i^\pi(\boldsymbol{s}_t) &= (1 - \gamma) \sum_{t' = t}^{\infty} \gamma^{t' - t} p(\boldsymbol{s}_{t'} = i \mid \boldsymbol{s}_t)\\  &= (1 - \gamma) \delta(\boldsymbol{s}_t = i) + \gamma \sum_{\boldsymbol{s}} p(\boldsymbol{s} \mid \boldsymbol{s}_t) \mu_i(\boldsymbol{s})\\  &= (1 - \gamma) \delta(\boldsymbol{s}_t = i) + \gamma \mathbb{E}_{\boldsymbol{a}_t \sim \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t), \boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)} \left[\mu_i^\pi(\boldsymbol{s}_{t + 1})\right], \end{aligned}
$$
在实际中上述过程可以向量化，同时对所有 $i$ 操作，即 
$$
\mu^\pi(\boldsymbol{s}_t) = (1 - \gamma) \boldsymbol{e}_{\boldsymbol{s}_t} + \gamma \mathbb{E}_{\boldsymbol{a}_t \sim \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t), \boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)} \left[\mu^\pi(\boldsymbol{s}_{t + 1})\right]
$$
其中 $\boldsymbol{e}_{\boldsymbol{s}_t}$ 是一个独热编码向量。

备注：
在上述过程中我们引入了后继表示，这一概念的好处在于其同时包含了模型与价值函数的信息，同样给出了其具有的基本性质，然而值得注意的是：
- 尚不明确学习后继表示是否比无模型的强化学习更加容易；  
- 尚不明确如何扩展到更大的状态空间；
- 并不知道如何将其扩展到连续状态空间。

在接下来的讨论中，我们首先考虑如何扩展到更大的状态空间。

### 5.2 Successor features
不难发现, 如果状态空间本身很大，那么 $\mu^\pi(\boldsymbol{s}_t)$ 也会很大， $\mu^\pi(\boldsymbol{s}_t)$ 为 $|\mathcal{S}|$ 的集合上的分布。

事实上，可以对状态进行压缩，考虑一个映射 $\phi: \mathcal{S} \rightarrow \mathbb{R}^d$ （这可以是人为设计的，也可以通过[[Concepts#21 自动编码器 (Autoencoder，AE)|自动编码器 (Autoencoder，AE)]]等方式学习），考虑通过 
$$
r(\boldsymbol{s}) = \sum_j \phi_j(\boldsymbol{s}) w_j = \phi(\boldsymbol{s})^T \boldsymbol{w}
$$
来近似地表示奖励，这里的 $\boldsymbol{w}$ 是一个 $d$ 维向量。
那么 
$$
\begin{aligned}  V(\boldsymbol{s}_t) &= \frac{1}{1 - \gamma} \mu^\pi(\boldsymbol{s}_t)^T \overrightarrow{r}\\  &= \frac{1}{1 - \gamma} \mu^\pi(\boldsymbol{s}_t)^T \sum_j \overrightarrow{\phi}_j w_j\\  &= \frac{1}{1 - \gamma} \sum_j \mu^\pi(\boldsymbol{s}_t)^T \overrightarrow{\phi}_j w_j \end{aligned}
$$
这里记 $\mu^\pi(\boldsymbol{s}_t)^T \overrightarrow{\phi}_j = \psi_j^\pi(\boldsymbol{s}_t)$，对 $j$ 整理得到一个 $\mathbb{R}^d$ 上的向量 $\psi^\pi(\boldsymbol{s}_t)$，称 $\psi^\pi(\boldsymbol{s}_t)$ 为 后继特征（Successor features）。整理得到 
$$
V(\boldsymbol{s}_t) = \frac{1}{1 - \gamma} \sum_j \psi_j^\pi(\boldsymbol{s}_t) w_j = \frac{1}{1 - \gamma} \psi^\pi(\boldsymbol{s}_t)^T \boldsymbol{w}
$$
这个表达式中的 $\psi^\pi$ 与 $\boldsymbol{w}$ 都是通过学习得到的，其中 $\boldsymbol{w}$ 的学习就是一个监督学习的过程（在已知 $\phi$ 的情况下学习一个权重）。我们使用贝尔曼备份来学习 $\psi_j^\pi(\boldsymbol{s}_t)$，这里考虑给之前向量化的贝尔曼备份点乘上 $\overrightarrow{\phi}_j$：
$$
\psi_j^\pi(\boldsymbol{s}_t) = \phi_j(\boldsymbol{s}_t) + \gamma \mathbb{E}_{\boldsymbol{a}_t \sim \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t), \boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)} \left[\psi_j^\pi(\boldsymbol{s}_{t + 1})\right]
$$
事实上上一部分讨论的 $\mu^\pi$ 的贝尔曼备份可以视作是使用 $\phi_i(\boldsymbol{s}) = (1 - \gamma) \delta(\boldsymbol{s} = i)$ 且 $|\mathcal{S}| = d$ 的特例。

上述设计的后继特征类似于价值函数，也可以构造类似于 Q 函数的后继特征：如果依然考虑 $r(\boldsymbol{s}) \approx \phi(\boldsymbol{s})^T \boldsymbol{w}$，则 
$$
Q^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t) \approx \psi^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t)^T \boldsymbol{w}
$$
同样可以写出贝尔曼备份：
$$
\psi_j^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t) = \phi_j(\boldsymbol{s}_t) + \gamma \mathbb{E}_{\boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t), \boldsymbol{a}_{t + 1} \sim \pi(\boldsymbol{a}_{t + 1} \mid \boldsymbol{s}_{t + 1})} \left[\psi_j^\pi(\boldsymbol{s}_{t + 1}, \boldsymbol{a}_{t + 1})\right]
$$
实际上原论文中 $\phi$ 输入为 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{a}')$，$\psi$ 输入为 $(\boldsymbol{s}, \boldsymbol{a})$，在上述推导中我们做了一些简化，但完全不影响结果。

最后讨论其使用方式：
Idea 1：用于快速恢复 Q 函数：
- 训练 $\psi_j^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t)$ （利用贝尔曼备份）； 
- 获取 $\{\boldsymbol{s}_i, r_i\}$ 样本；
- 获得 $\boldsymbol{w} \gets \arg\min_{\boldsymbol{w}} \|\phi(\boldsymbol{s}_i)^T \boldsymbol{w} - r_i\|^2$；
- 恢复 $Q^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t) \approx \psi^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t)^T \boldsymbol{w}$。

而我们的策略就是 
$$
\pi'(\boldsymbol{s}) = \arg\max_{\boldsymbol{a}} \psi^\pi(\boldsymbol{s}, \boldsymbol{a})^T \boldsymbol{w}
$$
这里计算得到的并不是最优 Q 函数，实际上这是当前的策略的 Q 函数，因此 $\pi'$ 只是 $\pi$ 的一步策略迭代的结果。

Idea 2：恢复多个 Q 函数（更好的想法）  
- 训练一系列 $\psi_j^{\pi_k}(\boldsymbol{s}_t, \boldsymbol{a}_t)$ 对于一系列 $\pi_k$ （利用贝尔曼备份）；
- 获取 $\{\boldsymbol{s}_i, r_i\}$ 样本；
- 获得 $\boldsymbol{w} \gets \arg\min_{\boldsymbol{w}} \|\phi(\boldsymbol{s}_i)^T \boldsymbol{w} - r_i\|^2$；
- 恢复 $Q^{\pi_k}(\boldsymbol{s}_t, \boldsymbol{a}_t) \approx \psi^{\pi_k}(\boldsymbol{s}_t, \boldsymbol{a}_t)^T \boldsymbol{w}$。

我们的策略就是 
$$
\pi'(\boldsymbol{s}) = \arg\max_{\boldsymbol{a}} \max_k \psi^{\pi_k}(\boldsymbol{s}, \boldsymbol{a})^T \boldsymbol{w}
$$
换言之就是找到每一个状态中的最高奖励策略，因此在每一个状态都在 $k$ 个中最好的策略上进行了一步改进。

实际上，从迁移学习的角度来看，在马尔可夫决策过程的其他设定不变以及 $\phi: \mathcal{S} \rightarrow \mathbb{R}^d$ 不变的情况下，每一个不同的 $\boldsymbol{w}$ 指定了一个不同的任务，论文中给出了以下的理论结果：
Theorem 1. 考虑 $M^\phi$ 为当前设定下的任务空间，已经学习了 $M_j \in M^\phi, j = 1,\ldots,n$ 这些任务，它们对应的最优 Q 函数为 $Q_i^{\pi_j^\ast}(\boldsymbol{s}, \boldsymbol{a})$，学习得到的 Q 函数为 $\tilde{Q}_i^{\pi_j^\ast}(\boldsymbol{s}, \boldsymbol{a})$，且它们满足 $\forall \boldsymbol{s} \in \mathcal{S}, \boldsymbol{a} \in \mathcal{A}$，则
$$
\left|\tilde{Q}_i^{\pi_j^\ast}(\boldsymbol{s}, \boldsymbol{a}) - Q_i^{\pi_j^\ast}(\boldsymbol{s}, \boldsymbol{a})\right| < \epsilon
$$
对于一个新任务 $M_i \in M^\phi$，采用 
$$
\pi(\boldsymbol{s}) = \arg\max_{\boldsymbol{a}} \max_j \tilde{Q}_i^{\pi_j^\ast}(\boldsymbol{s}, \boldsymbol{a})
$$
的方式作直接的迁移，那么有 
$$
Q_i^{\pi^\ast_i}(\boldsymbol{s}, \boldsymbol{a}) \leq \frac{2}{1 - \gamma} (\phi_{\max} \min_j\|\boldsymbol{w}_i - \boldsymbol{j}\| + \epsilon)
$$
其中 $\phi_{\max} = \max_{\boldsymbol{s}, \boldsymbol{a}} \|\phi(\boldsymbol{s}, \boldsymbol{a})\|$，其中 $\|\cdot\|$ 是内积诱导的范数。

本部分介绍的后继特征参见：Barreto et al. Successor Features for Transfer in Reinforcement Learning. 2016

### 5.3 Continuous Successor Representations
在上一小节中，我们讨论了如何将后继表示扩展到更大的状态空间，接下来我们介绍如何将后继表示的思想扩展到连续状态空间。

回顾特征表示 $\mu_i(\boldsymbol{s}_t) = p(\boldsymbol{s}_{future} = i \mid \boldsymbol{s}_t)$，在连续状态空间中，我们要学习的变成了一个 $\mathcal{S}$ 到 $\mathcal{S}$ 上连续分布的映射，这一映射的学习是非常困难的。但是，其实有其他方式来估计这一概率密度。

Idea：将学习后继表示的过程转化为学习一个二元分类的过程，考虑 
$$
p^\pi(F = 1 \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{future})
$$
这里记 $F = 1$ 意味着如果从 $\boldsymbol{s}_t, \boldsymbol{a}_t$ 出发采用策略 $\pi$，$\boldsymbol{s}_{future}$ 是一个未来状态，值得注意的是，我们这里又用回了依赖动作的奖励。

考虑正样本 $\mathcal{D}_+ \sim p^\pi(\boldsymbol{s}_{future} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$ 从未来状态的分布中采样，负样本 $D_- \sim p^\pi(\boldsymbol{s})$ 从所有 $\pi$ 可能到达的状态中采样，于是基于最优分类器的性质，有： 
$$
p^\pi(F = 1 \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{future}) = \frac{p^\pi(\boldsymbol{s}_{future})}{p^\pi(\boldsymbol{s}_{future}) + p^\pi(\boldsymbol{s}_{future})}
$$
$$
p^\pi(F = 0 \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{future}) = \frac{p^\pi(\boldsymbol{s}_{future})}{p^\pi(\boldsymbol{s}_{future}) + p^\pi(\boldsymbol{s}_{future})}
$$
这里训练上述的分类器比直接学习后继表示更加容易。并且如果我们能够训练这一分类器，能够从这个分类器中恢复 $p^\pi(\boldsymbol{s}_{future} \mid \boldsymbol{s}_t, \boldsymbol{s}_t)$：
$$
\frac{p^\pi(F = 1 \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{future})}{p^\pi(F = 0 \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{future})} p^\pi(\boldsymbol{s}_{future}) = p^\pi(\boldsymbol{s}_{future} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)
$$
这里的后继表示估计的是概率密度。而由于 $p^\pi(\boldsymbol{s}_{future})$ 是一个与 $\boldsymbol{a}_t, \boldsymbol{s}_t$ 无关的常量，尽管这个量非常难计算，因此它不影响我们基于 $p^\pi(F = 1 \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{future}) / p^\pi(F = 0 \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{future})$ 选择动作等等.

我们的算法是如下的 C 学习算法（The C-Learning algorithm）：
1. 获取负样本：采样 $\boldsymbol{s} \sim p^\pi(\boldsymbol{s})$ （运行策略，从轨迹采样）  
2. 获取正样本：采样 $\boldsymbol{s} \sim p^\pi(\boldsymbol{s}_{future} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$ （基于前面提到的 $\boldsymbol{s}_{future}$ 的两种理解方式, 其中一种就是采样 $\boldsymbol{s}_{t'}$，其中 $t' = t + \Delta, \Delta \sim Geom(\gamma)$）
3. 训练分类器：更新 $p^\pi(F = 1 \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{future})$，使用随机梯度下降和交叉熵损失。  

这里介绍的版本是一个 on-policy 算法，同样可以推导出 off-policy 算法。
参见：Eysenbach, Salakhutdinov, Levine. C-Learning: Learning to Achieve Goals via Recursive Classification. 2020

## 6 Summary
在本节中，我们
- 介绍了如何使用模型进行闭环策略学习，介绍了时间反向传播方法可能存在的问题，并引出了模型加速强化学习的概念。
- 介绍了这一类模型加速强化学习的例子，如 Dyna。 
- 介绍了一种理解模型加速强化学习的普遍方式，并介绍了一些变种。
- 介绍了后继表示这种介于无模型与基于模型之间的方法，介绍了扩展的后继特征与 C 学习，并讨论了其在迁移学习中的应用。