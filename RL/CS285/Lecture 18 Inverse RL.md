## 1 Introduction to Inverse RL
在之前对强化学习的讨论中，我们通常人为设计奖励函数。然而，在一些问题中，奖励函数可能很难设计，而又有一些专家示范，我们希望从这些示范中学习奖励函数，再利用这个奖励函数进行强化学习，这样的方式被称为逆强化学习。在本节我们将应用上一节[[Lecture 17 Reframing Control as an Inference Problem]]中讨论的最优性模型来学习奖励函数。

### 1.1 Why we want try inverse RL?
- 从模仿学习的角度：我们之前讨论的模仿学习算法仅仅是复制人类的行为，不会理解其背后的原因，也不会推理其动作的结果。与之形成对比的是，尽管模仿也是人类学习的一种方式，但人类的模仿是基于理解他人意图的。通过学习 奖励函数，似乎可以更好地理解人类的行为。
![](18-1.png)

- 从强化学习的角度：在很多情况下，奖励函数可能非常复杂，例如自动驾驶的奖励函数，相对的，获取专家的行为可能相对容易。  
![](18-2.png)
### 1.2 Potential Problems in learning reward function from demonstrations
学习一个奖励函数可能是欠定的（类似于一个不定方程有多个解），有很多奖励函数都可以解释同一个行为：
- 最极端的情况是，奖励退化为"只要动作不在专家的动作中，就给予 $-\infty$ 的奖励"；  
- 奖励函数的自由度可能带来一些问题，并不是所有奖励函数都是合理的，由于算法缺乏语义知识，可能很难避免学到一些不合理的奖励函数。 

例如，自动驾驶中，假如在一些地方的示范中的驾驶速度快，一些地方的示范中的驾驶速度，真实的奖励函数应当和相关的交通法规相关，而非与 GPS 位置相关，这样的问题可能会给实际的部署带来问题。

### 1.3 Formalization
上述问题实质是一种歧义，需要有一些方式来解决它们。在消除这些歧义之前，我们先介绍一些正式的的定义，在逆强化学习中：
对于给定的马尔可夫决策过程，我有
- 状态：$\boldsymbol{s} \in \mathcal{S}$；
- 动作：$\boldsymbol{a} \in \mathcal{A}$；
- （有时）转移动态：$p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})$； 
- 样本：$\{\tau_i\}$ 来自专家策略 $\pi^\ast$。

而目标是学习一个奖励函数 $r_\psi(\boldsymbol{s}, \boldsymbol{a})$，在学习了 $r_\psi(\boldsymbol{s}, \boldsymbol{a})$ 后，可以用其来学习一个策略。

关于奖励函数的形式，可以考虑两种方式：
- 在深度强化学习之前，学习线性奖励函数：$$r_\psi(\boldsymbol{s}, \boldsymbol{a}) = \sum_{i} \psi_i f_i(\boldsymbol{s}, \boldsymbol{a}) = \psi^T \boldsymbol{f}(\boldsymbol{s},\boldsymbol{a})$$这也是逆强化学习中的特征匹配的方式，会在下一部分进行介绍；
- 在现在深度学习中，通常会学习一个神经网络奖励函数 $r_\psi(\boldsymbol{s}, \boldsymbol{a})$。

## 2 Feature matching IRL
这里先介绍一个深度强化学习时代前的一个方法，逆强化学习中的特征匹配。
考虑线性奖励函数
$$
r_\psi(\boldsymbol{s}, \boldsymbol{a}) = \psi^T \boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})
$$
这里 $\boldsymbol{f}$ 是某种特征提取函数。一个直观的想法是，考虑学习一个 奖励函数 $r_\psi$ 使得
$$
\mathbb{E}_{\pi^{r_\psi}}[\boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})] = \mathbb{E}_{\pi^\ast}[\boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})]
$$
这里 $\pi^{r_\psi}$ 是 $r_\psi$ 下的最优策略，可以通过在 $r_\psi$ 下进行强化学习得到， $\pi^\ast$ 是专家策略。换句话说，我们希望奖励函数指导下的策略与专家策略产生的状态-动作对在期望下有一样的特征，这也就是特征匹配的由来。这里的期望都可以分别通过训练的策略采样估计以及专家示范估计。

然而，这样的处理方式依然是欠定的，有很多奖励函数都能够对应于同一个最优策略的期望。如果有多个奖励函数都满足要求，就缺少一个更新方向。

这里采用支持向量机中最大间隔原则的思想。在支持向量机中，当数据线性可分时，存在着无穷多的分离超平面，通过最大化间隔，可以得到唯一的超平面。这里同样可以考虑最大化间隔，具体来说是最大化奖励函数下，其他策略和专家策略之间的距离：
$$
\max_{\psi,m} m, \text{ s.t. } \psi^T \mathbb{E}_{\pi^\ast}[\boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})] \geq \max_{\pi \in \Pi}\psi^T \mathbb{E}_{\pi}[\boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})] + m
$$
这是非常直观的，如果我们的 专家策略 不能显著优于其他策略，那么怎么还叫专家呢？回顾支持向量机中的技巧，这可以转化为优化问题：
$$
\min_\psi \frac{1}{2} \|\psi\|^2 \text{ s.t. } \psi^T \mathbb{E}_{\pi^\ast}[\boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})] \geq \max_{\pi \in \Pi} \psi^T \mathbb{E}_{\pi^{r_\psi}}[\boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})] + 1
$$
然而这里有一个致命的问题，可能存在这距离最优策略本身就很接近的其他策略，此时也会把接近专家策略的奖励也下调，这并不是我们想要的。为了处理策略过于接近的问题，我们会将 $1$ 替换为 $D(\pi \parallel \pi^\ast)$：
$$
\min_\psi \frac{1}{2} \|\psi\|^2 \text{ s.t. } \psi^T \mathbb{E}_{\pi^\ast}[\boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})] \geq \max_{\pi \in \Pi} \psi^T \mathbb{E}_{\pi^{r_\psi}}[\boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})] + D(\pi \parallel \pi^\ast)
$$
这里的 $D$ 用于表示两个分布之间的差异，可以是 KL 散度，也可以是特征在期望上的差距。

这种方法存在的问题：
- 最大化间隔的做法依然是启发式的；  
- 这并没有建模出专家可能包含的一些次优性（处理这一点的已有方法类似于支持向量机中数据无法线性可分时，添加一些松弛变量，但这样的做法还是启发式的）；
- 这类复杂的约束优化问题对深度学习来说很难解决。

进一步阅读：
- Abbeel & Ng: Apprenticeship learning via inverse reinforcement learning  
- Ratliff et al: Maximum margin planning  

## 3 The MaxEnt IRL algorithm
在上一节中，我们讨论了将人类行为建模为一个概率图模型方式，这一建模方式能够很好地处理人类行为中的次优性，在这个模型中有一系列最优性变量 $\mathcal{O}_t$：
$$
p(\mathcal{O}_t \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \psi) = \exp(r_\psi(\boldsymbol{s}_t, \boldsymbol{a}_t))
$$
同时在模型中，给定最优性以及奖励下，轨迹的分布可以表示为
$$
p(\tau \mid \mathcal{O}_{1:T}, \psi) \propto p(\tau) \exp\left(\sum_{t = 1}^{T} r_\psi(\boldsymbol{s}_t, \boldsymbol{a}_t)\right)
$$
在确定性的情况下，最大化累计奖励的目标就可以直接转化为最大化这个后验概率。

但在这里，我们希望学习一个奖励函数 $r_\psi$，这可以理解为是人类行为中的某种偏好，在这种偏好下人类好的行为的轨迹分布是 $p(\tau \mid \mathcal{O}_{1:T}, \psi)$。因此可以通过学习 $r_\psi$ 以最大化专家示范的似然来使得行为得到最好的解释。
### 3.1 Derivation of the objective function
不妨记 $\{\tau_i\}$ 为从专家策略 $\pi^\ast$ 中采样得到的轨迹，则最大似然估计的目标可以写作
$$
\max_\psi \frac{1}{N} \sum_i \log p(\tau_i \mid \mathcal{O}_{1:T}, \psi) = \max_\psi \frac{1}{N} \sum_i \sum_{t = 1}^{T} r_\psi(\boldsymbol{s}_{i, t}, \boldsymbol{a}_{i, t}) - \log Z
$$
这里的 $Z$ 称为分片函数，有形式
$$
Z = \int p(\tau) \exp(r_\psi(\tau))\text{d}\tau
$$
通常来说对于深度神经网络来说，还是使用梯度下降来优化这个目标函数，这里的梯度可以写作
$$
\nabla_\psi \mathcal{L} = \frac{1}{N} \sum_{i = 1}^N \sum_{t = 1}^{T} \nabla_\psi r_\psi(\boldsymbol{s}_{i, t}, \boldsymbol{a}_{i, t}) - \frac{1}{Z} \int p(\tau) \exp(r_\psi(\tau)) \nabla_\psi r_\psi(\tau) \text{d}\tau
$$
这里第一项可以视作专家策略 $\pi^\ast$ 下的期望，而第二项可以视作是 $p(\tau \mid \mathcal{O}_{1:T}, \psi)$ 下的期望，于是可以转化为
$$
\nabla_\psi \mathcal{L} = \mathbb{E}_{\tau \sim \pi^\ast(\tau)}\left[\nabla_\psi r_\psi(\tau)\right] - \mathbb{E}_{\tau \sim p(\tau \mid \mathcal{O}_{1:T}, \psi)}\left[\nabla_\psi r_\psi(\tau)\right]
$$

### 3.2 Estimating the expectation
考虑如何计算上述梯度中的后一项期望，首先注意到化简
$$
\begin{aligned} \mathbb{E}_{\tau \sim p(\tau \mid \mathcal{O}_{1:T}, \psi)}\left[\nabla_\psi r_\psi(\tau)\right] &= \mathbb{E}_{\tau \sim p(\tau \mid \mathcal{O}_{1:T}, \psi)}\left[\nabla_\psi \sum_{t = 1}^{T} r_\psi(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]\\ &= \sum_{t = 1}^{T} \mathbb{E}_{(\boldsymbol{s}_t, \boldsymbol{a}_t) \sim p(\boldsymbol{s}_t, \boldsymbol{a}_t \mid \mathcal{O}_{1:T}, \psi)}\left[\nabla_\psi r_\psi(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]. \end{aligned}
$$
这里 $p(\boldsymbol{s}_t, \boldsymbol{a}_t \mid \mathcal{O}_{1:T}, \psi) = p(\boldsymbol{a}_t \mid \boldsymbol{s}_t, \mathcal{O}_{1:T}, \psi) p(\boldsymbol{s}_t \mid \mathcal{O}_{1:T}, \psi)$, 基于上一节课的讨论，我们知道：
- 乘积第一项 $p(\boldsymbol{a}_t \mid \boldsymbol{s}_t, \mathcal{O}_{1:T}, \psi) = \beta(\boldsymbol{s}_t, \boldsymbol{a}_t) / \beta(\boldsymbol{s}_t)$；
- 乘积第二项 $p(\boldsymbol{s}_t \mid \mathcal{O}_{1:T}, \psi) \propto \alpha(\boldsymbol{s}_t) \beta(\boldsymbol{s}_t)$。

于是整一项 $p(\boldsymbol{s}_t, \boldsymbol{a}_t \mid \mathcal{O}_{1:T}, \psi) \propto \beta(\boldsymbol{s}_t, \boldsymbol{a}_t) \alpha(\boldsymbol{s}_t)$。

这里令 $\mu_t(\boldsymbol{s}_t, \boldsymbol{a}_t) \propto \beta(\boldsymbol{s}_t, \boldsymbol{a}_t) \alpha(\boldsymbol{s}_t)$，如果忽略常数就可以得到
$$
\begin{aligned} \mathbb{E}_{\tau \sim p(\tau \mid \mathcal{O}_{1:T}, \psi)}\left[\nabla_\psi r_\psi(\tau)\right] &= \sum_{t = 1}^T \iint \mu_t(\boldsymbol{s}_t, \boldsymbol{a}_t) \nabla_\psi r_\psi(\boldsymbol{s}_t, \boldsymbol{a}_t) \text{d}\boldsymbol{s}_t \text{d}\boldsymbol{a}_t\\ &= \sum_{t = 1}^{T} \overrightarrow{\mu}_t^T \nabla_\psi \overrightarrow{r}_\psi. \end{aligned}
$$
这里的向量是对于全体 $\boldsymbol{s}_t, \boldsymbol{a}_t$ 的向量化。

### 3.3 The MaxEnt IRL algorithm (known dynamics, small state-action space)
于是可以给出最大熵逆强化学习（MaxEnt IRL）的算法：
1. 对给定的 $\psi$，计算反向信息 $\beta(\boldsymbol{s}_t, \boldsymbol{a}_t)$；
2. 对给定的 $\psi$，计算前向信息 $\alpha(\boldsymbol{s}_t)$；
3. 计算 $\mu_t(\boldsymbol{s}_t, \boldsymbol{a}_t) \propto \beta(\boldsymbol{s}_t, \boldsymbol{a}_t) \alpha(\boldsymbol{s}_t)$； 
4. 计算 $\nabla_\psi \mathcal{L} = \frac{1}{N} \sum_{i = 1}^{N} \sum_{t = 1}^{T} \nabla_\psi r_\psi(\boldsymbol{s}_{i, t}, \boldsymbol{a}_{i, t}) - \sum_{t = 1}^{T} \int \int \mu_t(\boldsymbol{s}_t, \boldsymbol{a}_t) \nabla_\psi r_\psi(\boldsymbol{s}_t, \boldsymbol{a}_t) \text{d}\boldsymbol{s}_t \text{d}\boldsymbol{a}_t$；
5. 梯度更新 $\psi \gets \psi + \eta \nabla_\psi \mathcal{L}$。

注意：这种算法利用了之前对次优性的建模，解决了之前的歧义问题。
- 这样的做法需要计算反向与前向信息，基于上一节的内容，这依赖于已知动态，同时由于需要对所有的状态-动作对进行计，故对大规模连续状态空间的问题并不适用；
- 为什么称这个算法为最大熵方法呢？在奖励线性时，可以证明这个算法最大化了熵：$$\max_\psi \mathcal{H}(\pi^{r_\psi}) \text{ s.t. } \mathbb{E}_{\pi^{r_\psi}}[\boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})] = \mathbb{E}_{\pi^\ast}[\boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})]$$这类似于一种最大熵推断的方式，此时除了有数据支持的部分，不做任何额外的推断，最大化了剩下的不确定性。

## 4 Approximations in High Dimensions
### 4.1 Drawbacks of MaxEnt IRL
在刚才介绍的最大熵逆强化学习中，需要：
- 在内层循环中计算（软）最优策略（直接更新的是 $\psi$，但实际上需要得到 $\pi^{r_\psi}$）；
- 枚举所有的 $\boldsymbol{s}_t, \boldsymbol{a}_t$ 对（回顾之前的双重"积分"）。

这需要非常大的计算量，然而在实际的问题中，可能有：
- 大规模与连续状态空间；
- 状态可能只能通过样本获得（没办法简单遍历一遍）；
- 未知动态。 

因此无法将这个算法直接应用到实际问题中，需要一些近似的方法来解决这个问题。

### 4.2 Unknown dynamics & large state/ action spaces
在未知动态（无法计算信息从而计算 $\mu_t(\boldsymbol{s}_t, \boldsymbol{a}_t)$），也没办法遍历所有的状态-动作的情况下，如何计算
$$
\mathbb{E}_{\tau \sim p(\tau \mid \mathcal{O}_{1:T}, \psi)}\left[\nabla_\psi r_\psi(\tau)\right]
$$
呢？这里依赖的是无模型的强化学习中核心的方法，采样。

回顾梯度表达式
$$
\nabla_\psi \mathcal{L} = \mathbb{E}_{\tau \sim \pi^\ast(\tau)}\left[\nabla_\psi r_\psi(\tau)\right] - \mathbb{E}_{\tau \sim p(\tau \mid \mathcal{O}_{1:T}, \psi)}\left[\nabla_\psi r_\psi(\tau)\right]
$$
前一个期望就是专家的示范，而后者是我们需要解决的。

想法：使用任何最大熵强化学习（也就是上一节介绍的软 Q 学习，软演员-评论家等）算法，学习 $p(\boldsymbol{a}_t \mid \boldsymbol{s}_t, \mathcal{O}_{1:T}, \psi)$，然后运行这个算法获得轨迹 $\{\tau_j\}$，通过样本估计梯度
$$
\nabla_\psi \mathcal{L} \approx \frac{1}{N} \sum_{i = 1}^{N} \nabla_\psi r_\psi(\tau_i) - \frac{1}{M} \sum_{j = 1}^{M} \nabla_\psi r_\psi(\tau_j)
$$
然而由于上述估计的无偏性基于 $\{\tau_j\}$ 来源于当前的 $\pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t, \mathcal{O}_{1:T}, \psi)$，这就意味着需要在每次更新 $\psi$ 时都重新学习一个 策略，这显然是不可接受的。

想法：使用 "懒惰" 策略优化的方，也就是仅仅改进这个策略，而没有重新学习 $\pi^\psi$ 并从 $p(\boldsymbol{a}_t \mid \boldsymbol{s}_t, \mathcal{O}_{1:T}, \psi)$ 采样，然而由于分布不匹配，期望会是有偏差的，需要一些方式来解决这个问题。

一个可能的方式是重要性采样。假设目前有采样的策略为 $\pi$，记这一策略下轨迹 $\tau$ 的概率为 $\pi(\tau)$，由于原先期望的概率分布为 $p(\tau \mid \mathcal{O}_{1:T}, \psi) \propto p(\tau) \exp(r_\psi(\tau_j))$，可以得到重要性比率：
$$
w_j = \frac{p(\tau) \exp(r_\psi(\tau_j))}{C\pi(\tau_j)} = \frac{p(\boldsymbol{s}_1) \prod_t p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t) \exp(r_\psi(\boldsymbol{s}_t, \boldsymbol{a}_t))}{C p(\boldsymbol{s}_1) \prod_t p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t) \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} = \frac{\exp(r_\psi(\boldsymbol{s}_t, \boldsymbol{a}_t))}{C\prod_{t} \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}
$$
这里 $C = p(\mathcal{O}_{1:T})$ 是常数，之后用一些技巧去除，剩余的都是可以直接计算的形式：其中的 $r_\psi$ 对应于当前的 $\psi$，可以直接查询对应的状态-动作对，而 $\pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t)$ 也可以通过查询当前的策略得到。

由于 $\sum_{j = 1}^M w_j$ 应该与 $M$ 相差不会太大，这里考虑用 $\sum_{j = 1}^M w_j$ 来替代 $C$，于是问题转化为
$$
\nabla_\psi \mathcal{L} \approx \frac{1}{N} \sum_{i = 1}^{N} \nabla_\psi r_\psi(\tau_i) - \frac{1}{\sum_{j} w_j} \sum_{j = 1}^{M} w_j\nabla_\psi r_\psi(\tau_j)
$$
这里的常数 $C$ 就可以被消掉了。

最后考虑一个关键的问题，为什么要使用最大熵强化学习来训练策略呢？不难理解由于奖励和策略会同时训练，我们不希望策略过早陷入一些特定的行为中，同时也能够进行增量式的更新。

### 4.3 Guided cost learning algorithm
在经过这样的近似后，我们的算法可以应用在实际的问题中，一个实际的例子是如下的引导成本学习（Guided cost learning algorithm）算法：

假设已经有了专家示范 $s$ 存储在 $\mathcal{D}_{demo}$ 中，有如下的算法：
1. 初始化策略 $\pi$；
2. 重复以下过程：
3. 利用策略收集轨迹 $\{\tau_j\}$；
4. 将轨迹添加到缓冲 $\mathcal{D}_{sample}$ 中； 
5. 利用 $\mathcal{D}_{demo}$ 和 $\mathcal{D}_{sample}$ 更新奖励函数 $r_\psi$；
6. 利用 $\{\tau_j\}$（当前轨迹）更新策略 $\pi$。

![](18-3.png)


参见：Finn et al. ICML '16. Guided Cost Learning. Sampling based method for MaxEnt IRL that handles unknown dynamics and deep reward functions.

## 4 IRL and GANs
这类最大熵逆强化学习的方法与生成对抗网络（Generative Adversarial Networks，GANs）有很强的联系，这一联系可以启发我们设计更好的逆强化学习算法。

### 4.1 Intuition
实际上前面的过程像是一个奖励和策略之间的博弈：
- 奖励：通过目标：$$\nabla_\psi \mathcal{L} \approx \frac{1}{N} \sum_{i = 1}^{N} \nabla_\psi r_\psi(\tau_i) - \frac{1}{\sum_j w_j} \sum_{j = 1}^{M} w_j\nabla_\psi r_\psi(\tau_j)$$使得专家示范看起来好，而我们的策略看起来不好。
- 策略：依据奖励优化自身，使其更好，也就是优化：$$\nabla_\theta \mathcal{L} \approx \frac{1}{M} \sum_{j = 1}^{M} \nabla_\theta \log \pi(\boldsymbol{a}_j \mid \boldsymbol{s}_j) r_\psi(\tau_j)$$ 使得其更难从专家示范中区分。

换言之，策略试图通过提升其轨迹的奖励来"骗过" 奖励函数，而奖励函数试图找到新的奖励形式来区分策略的轨迹与专家示范。

### 4.2 Generative Adversarial Networks
事实上这可以联系到生成对抗网络（Generative Adversarial Networks，GANs），在生成对抗网络中，生成器 $p_\theta(\boldsymbol{x} \mid \boldsymbol{z})$ 试图生成数据，而判别器 $D_\psi = p_\psi(\text{real} \mid \boldsymbol{x})$ 试图区分生成的数据与真实数据。
$$
\psi = \arg\max_\psi \frac{1}{N} \sum_{\boldsymbol{x} \sim p^\ast(\boldsymbol{x})} \log D_\psi(\boldsymbol{x}) + \frac{1}{M} \sum_{\boldsymbol{x} \sim p_\theta(\boldsymbol{x})} \log(1 - D_\psi(\boldsymbol{x}))
$$
$$
\theta = \arg\max_\theta \frac{1}{M} \sum_{\boldsymbol{x} \sim p_\theta(\boldsymbol{x})} \log D_\psi(\boldsymbol{x})
$$
这里 $p^\ast(\boldsymbol{x})$ 是数据的真实分布，$p_\theta(\boldsymbol{x})$ 是生成器生成的数据。

![](18-4.png)

### 4.3 Inverse RL as GAN
不难发现逆强化学习与生成对抗网络之间有着很强的联系，可以把逆强化学习看作是一个生成对抗网络。也就是说，训练一个判别器来区分专家的轨迹与策略生成的轨迹。

回顾在生成对抗网络中，最优判别器是
$$
D^\ast(\boldsymbol{x}) = \frac{p^\ast(\boldsymbol{x})}{p^\ast(\boldsymbol{x}) + p_\theta(\boldsymbol{x})}
$$
在逆强化学习中，最优策略对应的轨迹满足
$$
p_\theta(\tau) = p(\tau) \exp(r_\psi(\tau)) / Z
$$
这里的 $Z$ 是分片函数：
$$
Z = \int p(\tau) \exp(r_\psi(\tau)) \text{d}\tau
$$
因而可以使用如下方式来参数化判别器：
$$
\begin{aligned} D_\psi(\tau) = \frac{p(\tau) \frac{1}{Z}\exp(r(\tau))}{p_\theta(\tau) + p(\tau) \frac{1}{Z}\exp(r(\tau))} &= \frac{p(\tau) \frac{1}{Z'}\exp(r(\tau))}{p(\tau)\prod_t \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t) + \frac{1}{Z'} p(\tau) \exp(r(\tau))}\\ &= \frac{\frac{1}{Z'} \exp(r(\tau))}{\prod_t \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t) + \frac{1}{Z'} \exp(r(\tau))}. \end{aligned}
$$
这里 $p(\tau)$ 为专家轨迹的分布，$p_\theta(\tau)$ 为策略生成的轨迹的分布，它们满足
$$
p_\theta(\tau) = p(\tau) \left(\prod_t \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)\right) \bigg/ \left(\prod_t \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t)\right)
$$
因此这里的第二个等号等价于在分子分母中同时乘上 $\prod_t \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t)$，并且将其吸收进 $Z'$ 中。

基于上述的记号，类似于生成对抗网络中，优化判别器的方式可以写作
$$
\psi = \arg\max_\psi \mathbb{E}_{\tau \sim \pi^\ast(\tau)}\left[\log D_\psi(\tau)\right] + \mathbb{E}_{\tau \sim \pi_\theta}\left[\log(1 - D_\psi(\tau))\right]
$$
在这一写法中没有重要性采样出现，因为它们都会成为 $D_\psi$ 表达式中 $Z'$ 的一部分。值得注意的是这里的 $Z'$ 依然需要处理，可以对其用样本进行估。具体来说，可以在上述目标中对 $Z'$ 求偏导并令其等于 $0$ 再估计。

基于生成对抗网络的训练方式可以得到一个对应的最大熵逆强化学习算法：
- 生成器 / 策略 $\pi_\theta(\tau)$，对应于生成对抗网络中的生成器，可以从中采样轨迹；
- 专家示范 $p^\ast(\tau)$，对应于生成对抗网络训练中的真实图片；  
- 训练生成器 / 策略的过程：$$\nabla_\theta \mathcal{L} \approx \frac{1}{M} \sum_{j = 1}^{M} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_j \mid \boldsymbol{s}_j) r_\psi(\tau_j)$$  
- 训练奖励函数的过程：$$\psi = \arg\max_\psi \mathbb{E}_{\tau \sim \pi^\ast(\tau)}\left[\log D_\psi(\tau)\right] + \mathbb{E}_{\tau \sim \pi_\theta}\left[\log(1 - D_\psi(\tau))\right]$$$$D_\psi(\tau) = \frac{\frac{1}{Z} \exp(r(\tau))}{\prod_t \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t) + \frac{1}{Z} \exp(r(\tau))}$$

![](18-5.png)

实质上，可以证明上述利用生成对抗网络的方式进行逆强化学习的方法是等价于之前介绍的最大熵逆强化学习的方法.

参见：Finn\*, Christiano\* et al. "A Connection Between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models."

### 4.4 Generalization via Inverse RL
在 "Learning Robust Rewards with Adversarial Inverse Reinforcement Learning, Fu et al., 2017" 中，作者将逆强化学习中学习的奖励函数设置为仅基于状态。

此时在一个设置下的专家示范中学习奖励函数，在另一个不同条件下（修改质量等），依然可以通过这个奖励函数实现一些有意义的行为。这在某种意义上"理解"了专家的意图（是向一个方向走），达成了某种泛化的效果。
![](18-6.png)

### 4.5 Regular Discriminator
在前面的讨论中，我们已经注意到逆强化学习问题学习 $\psi$ 的过程也可以写作生成对抗网络中优化判别器的形式：
$$
\psi = \arg\max_\psi \mathbb{E}_{\tau \sim \pi^\ast(\tau)}\left[\log D_\psi(\tau)\right] + \mathbb{E}_{\tau \sim \pi_\theta}\left[\log(1 - D_\psi(\tau))\right]
$$
此时从 $\psi$ 的目标表面上看，我们摆脱了重要性采样比率等麻烦的东西，但是如果这里的 $D_\psi$ 还是由 $r_\psi$ 参数化的，那么还是没法避免计算这些东西。

一个直接的想法是，能否使用一个常规判别器？也就是使用 $D_\psi$ 为一个常规的二分类网络，用来区分轨迹是否来自专家示范。然而正如生成对抗网络训练终止后判别器会输出 $0.5$，此时也无法恢复 奖励函数。不过这不影响恢复一个接近专家的策略，因此事实上这属于模仿学习的一种算法。
![](18-7.png)

注意：
- 这里的问题有更少的可变量，更加容易优化；
- 但是判别器在收敛时什么也不知道（正如生成对抗网络中训练结束后判别器会被丢弃），同时也无法恢复奖励函数。 

## 5 Summary
在本节中，我们：
- 介绍了逆强化学习的基本概念，以及其中存在的欠定性问题；
- 介绍了深度学习时代前的逆强化学习方法：特征匹配的逆强化学习，这依然是一种启发式的做法，不适用于深度学习；
- 介绍了最大熵逆强化学习算法： 
	- 推导过程基于上一节介绍的概率图模型，通过计算反向信息和前向信息来计算 $\mu_t(\boldsymbol{s}_t, \boldsymbol{a}_t)$，进而优化奖励函数；  
	- 其严格的做法需要知道动态，同时需要遍历所有的状态-动作对，以及不断训练最新的奖励的最优策略，这在实际中不可行；
	- 通常常用的做法是通过一些近似，使用采样的方式来估计梯度，并且使用"懒惰"策略优化的方式来避免重新训练策略，一个应用的例子是引导代价学习算法。
- 介绍了逆强化学习和生成对抗网络之间的紧密联系，以及如何通过生成对抗网络相关的范式进行逆强化学习，并介绍了一些应用以及转化为模仿学习的方法。 

进一步阅读：
经典论文：
- Abbeel & Ng ICML '04. Apprenticeship Learning via Inverse Reinforcement Learning.  
- Good introduction to inverse reinforcement learning Ziebart et al. AAAI '08. Maximum Entropy Inverse Reinforcement Learning. Introduction to probabilistic method for inverse reinforcement learning.  

现代论文：
- Finn et al. ICML '16. Guided Cost Learning. Sampling based method for MaxEnt IRL that handles unknown dynamics and deep reward functions.  
- Wulfmeier et al. arXiv '16. Deep Maximum Entropy Inverse Reinforcement Learning.  
- MaxEnt inverse RL using deep reward functions Ho & Ermon NIPS '16. Generative Adversarial Imitating Learning. Inverse RL method using generative adversarial networks.  
- Fu, Luo, Levine ICLR '18. Learning Robust Rewards with Adversarial Inverse Reinforcement Learning.