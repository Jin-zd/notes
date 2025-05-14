## 1 Introduction to Inverse RL

在之前对 RL 的讨论中, 我们通常人为设计 reward function. 然而, 在一些问题中, reward function 可能很难设计, 而我们又有一些 expert demonstration, 我们希望从这些 demonstration 中学习 reward function, 再利用这个 reward function 进行强化学习, 这样的方式被称为 **inverse reinforcement learning (inverse RL)**. 在本节我们将应用上一节 **Reframing Control as an Inference Problem** 中讨论的 optimality model 来学习 reward function.

### Why we want try inverse RL?

-   **从 imitation learning 的角度**: 我们之前讨论的 imitation learning 算法仅仅是**复制人类的行为**, 不会理解其背后的原因, 也不会推理其 actions 的结果. 与之形成对比的是, 尽管 imitation 也是人类学习的一种方式, 但人类的 imitation 是基于理解他人意图的. 通过学习 reward function, 我们似乎可以更好地理解人类的行为.

![](https://pic1.zhimg.com/v2-e54aa4b44e22b1fe15a816a3c4d0814e_1440w.jpg)

人类的 imitation learning 能够理解他人的意图

-   **从 reinforcement learning 的角度**: 在很多情况下, reward function 可能非常复杂, 例如自动驾驶的 reward function, 相对的, 获取专家的行为可能相对容易.  
    

![](https://pic4.zhimg.com/v2-ad09b973cc11b7d0fb98426340fdecc3_1440w.jpg)

现实问题中的 reward 可能非常复杂

### Potential Problems in learning reward function from demonstrations:

学习一个 reward function 可能是欠定的 (类似于一个不定方程有多个解), 有很多 reward functions 都可以解释同一个行为.

-   最极端的情况是, reward 退化为"只要 action 不在 expert 的 action 中, 就给予 $-\infty$ 的 reward".  
    
-   reward function 的自由度可能带来一些问题, 并不是所有 reward function 都是合理的. 由于我们的算法缺乏 semantic knowledge, 可能很难避免学到一些不合理的 reward function.  
    

**Example 1**. _自动驾驶中, 假如在一些地方的 demonstration 中的驾驶速度快, 一些地方的 demonstration 中的驾驶速度慢, 真实的 reward function 应当和相关的交通法规相关, 而非与 GPS 位置相关. 这样的问题可能会给实际的部署带来问题._

### 1.1 Formalization

上述问题实质是一种 ambiguity, 我们需要有一些方式来解决它们. 在消除这些 ambiguity 之前, 我们先介绍一些 formal 的定义. 在 inverse RL 中:

对于给定的 MDP, 我们有

-   **states**: $\boldsymbol{s} \in \mathcal{S}$,  
    
-   **actions**: $\boldsymbol{a} \in \mathcal{A}$,  
    
-   **(sometimes) transition dynamics**: $p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})$,  
    
-   **samples**: $\{\tau_i\}$ from expert policy $\pi^\ast$.  
    

而目标是学习一个 reward function $r_\psi(\boldsymbol{s}, \boldsymbol{a})$. 在我们学习了 $r_\psi(\boldsymbol{s}, \boldsymbol{a})$ 后, 我们可以用其来学习一个 policy.

关于 reward function 的形式, 我们可以考虑两种方式:

-   在深度强化学习之前, 我们会学习 linear reward function: $r_\psi(\boldsymbol{s}, \boldsymbol{a}) = \sum_{i} \psi_i f_i(\boldsymbol{s}, \boldsymbol{a}) = \psi^T \boldsymbol{f}(\boldsymbol{s},\boldsymbol{a}).\\$ 这也是 [feature matching IRL](https://zhida.zhihu.com/search?content_id=255981177&content_type=Article&match_order=1&q=feature+matching+IRL&zhida_source=entity) 的方式, 我们会在下一部分进行介绍.  
    
-   在现在深度学习中, 我们通常会学习一个 neural network reward function $r_\psi(\boldsymbol{s}, \boldsymbol{a})$.  
    

## 2 Feature matching IRL

这里我们先介绍一个深度强化学习时代前的一个方法, **feature matching IRL**.

考虑 linear reward function $r_\psi(\boldsymbol{s}, \boldsymbol{a}) = \psi^T \boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a}).\\$这里 $\boldsymbol{f}$ 是某种特征提取函数. 一个直观的想法是, 我们考虑学习一个 reward function $r_\psi$ 使得 $\mathbb{E}_{\pi^{r_\psi}}[\boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})] = \mathbb{E}_{\pi^\ast}[\boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})],\\$这里 $\pi^{r_\psi}$ 是 $r_\psi$ 下的最优策略, 可以通过在 $r_\psi$ 下进行强化学习得到, $\pi^\ast$ 是 expert policy. 换句话说, 我们希望 reward function 指导下的 policy 与 expert policy 产生的 state-action pair 在期望下有一样的 feature. 这也就是 **feature matching** 的由来. 这里的期望都可以分别通过训练的 policy 采样估计以及 expert demonstration 估计.

然而, 这样的处理方式依然是欠定的, 有很多 reward function 都能够对应于同一个 optimal policy 的期望. 如果有多个 reward function 都满足要求, 我们就缺少一个更新方向!

这里我们采用 **[SVM](https://zhida.zhihu.com/search?content_id=255981177&content_type=Article&match_order=1&q=SVM&zhida_source=entity)** 中 **maximum margin principle** 的思想. 在 SVM 中, 当数据线性可分时, 存在着无穷多的 separating hyperplane, 通过最大化 margin, 我们可以得到唯一的 hyperplane. 这里同样可以考虑最大化 margin, 具体来说是最大化 reward function 下, 其他 policy 和 expert policy 之间的距离: $\max_{\psi,m} m, \text{ s.t. } \psi^T \mathbb{E}_{\pi^\ast}[\boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})] \geq \max_{\pi \in \Pi}\psi^T \mathbb{E}_{\pi}[\boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})] + m.\\$这是非常 intuitive 的, 如果我们的 expert policy 不能显著优于其他 policy, 那么怎么还叫 expert 呢? 回顾 SVM 中的 trick, 这可以转化为优化问题: $\min_\psi \frac{1}{2} \|\psi\|^2 \text{ s.t. } \psi^T \mathbb{E}_{\pi^\ast}[\boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})] \geq \max_{\pi \in \Pi} \psi^T \mathbb{E}_{\pi^{r_\psi}}[\boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})] + 1.\\$

然而这里有一个致命的问题, 可能存在这距离 optimal policy 本身就很接近的其他 policy, 此时我们也会把接近 expert policy 的 reward 也下调, 这并不是我们想要的. 为了处理 policy 过于接近的问题, 我们会将 $1$ 替换为 $D(\pi \parallel \pi^\ast)$: $\min_\psi \frac{1}{2} \|\psi\|^2 \text{ s.t. } \psi^T \mathbb{E}_{\pi^\ast}[\boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})] \geq \max_{\pi \in \Pi} \psi^T \mathbb{E}_{\pi^{r_\psi}}[\boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})] + D(\pi \parallel \pi^\ast),\\$ 这里的 $D$ 用于表示两个分布之间的差异, 可以是 KL 散度, 也可以是 feature 在期望上的差距.

### Problems with this approach

-   最大化 margin 的做法依然是 heuristic 的.  
    
-   这并没有建模出 expert 可能包含的一些 [suboptimality](https://zhida.zhihu.com/search?content_id=255981177&content_type=Article&match_order=1&q=suboptimality&zhida_source=entity) (处理这一点的已有方法类似于 SVM 中数据无法线性可分时, 我们添加一些 slack variables, 但这样的做法还是 heuristic 的)  
    
-   这类复杂的约束优化问题对深度学习来说很难解决.  
    

**further reading**:

-   Abbeel & Ng: Apprenticeship learning via inverse reinforcement learning  
    
-   Ratliff et al: Maximum margin planning  
    

## 3 The [MaxEnt IRL](https://zhida.zhihu.com/search?content_id=255981177&content_type=Article&match_order=1&q=MaxEnt+IRL&zhida_source=entity) algorithm

在上一节中, 我们讨论了将人类行为建模为一个 probabilistic graphical model 的方式, 这一建模方式能够很好地处理人类行为中的 suboptimality. 在这个模型中有一系列 optimality variables $\mathcal{O}_t$, $p(\mathcal{O}_t \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \psi) = \exp(r_\psi(\boldsymbol{s}_t, \boldsymbol{a}_t)),\\$ 同时在模型中, 给定 optimality 以及 reward 下, trajectory 的分布可以表示为 $p(\tau \mid \mathcal{O}_{1:T}, \psi) \propto p(\tau) \exp\left(\sum_{t = 1}^{T} r_\psi(\boldsymbol{s}_t, \boldsymbol{a}_t)\right),\\$ 在 deterministic 的情况下, 我们最大化累计 reward 的目标就可以直接转化为最大化这个后验概率.

但在这里, 我们希望学习一个 reward function $r_\psi$, 这可以理解为是人类行为中的某种偏好, 在这种偏好下人类好的行为的轨迹分布是 $p(\tau \mid \mathcal{O}_{1:T}, \psi)$. 因此我们可以通过学习 $r_\psi$ 以最大化 expert demonstration 的 likelihood 来使得行为得到最好的解释.

### 3.1 Derivation of the objective function

不妨记 $\{\tau_i\}$ 为从 expert policy $\pi^\ast$ 中采样得到的轨迹, 则 **MLE** 的 objective 可以写作 $\max_\psi \frac{1}{N} \sum_i \log p(\tau_i \mid \mathcal{O}_{1:T}, \psi) = \max_\psi \frac{1}{N} \sum_i \sum_{t = 1}^{T} r_\psi(\boldsymbol{s}_{i, t}, \boldsymbol{a}_{i, t}) - \log Z.\\$ 这里的 $Z$ 称为 **partition function**, 有形式 $Z = \int p(\tau) \exp(r_\psi(\tau))\text{d}\tau.\\$

通常来说对于深度神经网络来说, 我们还是使用梯度下降来优化这个目标函数, 这里的梯度可以写作 $\nabla_\psi \mathcal{L} = \frac{1}{N} \sum_{i = 1}^N \sum_{t = 1}^{T} \nabla_\psi r_\psi(\boldsymbol{s}_{i, t}, \boldsymbol{a}_{i, t}) - \frac{1}{Z} \int p(\tau) \exp(r_\psi(\tau)) \nabla_\psi r_\psi(\tau) \text{d}\tau.\\$

这里第一项可以视作 expert policy $\pi^\ast$ 下的期望, 而第二项可以视作是 $p(\tau \mid \mathcal{O}_{1:T}, \psi)$ 下的期望, 于是可以转化为 $\nabla_\psi \mathcal{L} = \mathbb{E}_{\tau \sim \pi^\ast(\tau)}\left[\nabla_\psi r_\psi(\tau)\right] - \mathbb{E}_{\tau \sim p(\tau \mid \mathcal{O}_{1:T}, \psi)}\left[\nabla_\psi r_\psi(\tau)\right].\\$

### 3.2 Estimating the expectation

我们考虑如何计算上述梯度中的后一项期望, 首先注意到化简 $\begin{aligned} \mathbb{E}_{\tau \sim p(\tau \mid \mathcal{O}_{1:T}, \psi)}\left[\nabla_\psi r_\psi(\tau)\right] &= \mathbb{E}_{\tau \sim p(\tau \mid \mathcal{O}_{1:T}, \psi)}\left[\nabla_\psi \sum_{t = 1}^{T} r_\psi(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]\\ &= \sum_{t = 1}^{T} \mathbb{E}_{(\boldsymbol{s}_t, \boldsymbol{a}_t) \sim p(\boldsymbol{s}_t, \boldsymbol{a}_t \mid \mathcal{O}_{1:T}, \psi)}\left[\nabla_\psi r_\psi(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]. \end{aligned}\\$ 这里 $p(\boldsymbol{s}_t, \boldsymbol{a}_t \mid \mathcal{O}_{1:T}, \psi) = p(\boldsymbol{a}_t \mid \boldsymbol{s}_t, \mathcal{O}_{1:T}, \psi) p(\boldsymbol{s}_t \mid \mathcal{O}_{1:T}, \psi)$, 基于上一节课的讨论, 我们知道:

-   乘积第一项 $p(\boldsymbol{a}_t \mid \boldsymbol{s}_t, \mathcal{O}_{1:T}, \psi) = \beta(\boldsymbol{s}_t, \boldsymbol{a}_t) / \beta(\boldsymbol{s}_t)$,  
    
-   乘积第二项 $p(\boldsymbol{s}_t \mid \mathcal{O}_{1:T}, \psi) \propto \alpha(\boldsymbol{s}_t) \beta(\boldsymbol{s}_t)$.  
    

于是整一项 $p(\boldsymbol{s}_t, \boldsymbol{a}_t \mid \mathcal{O}_{1:T}, \psi) \propto \beta(\boldsymbol{s}_t, \boldsymbol{a}_t) \alpha(\boldsymbol{s}_t)$.

这里我们令 $\mu_t(\boldsymbol{s}_t, \boldsymbol{a}_t) \propto \beta(\boldsymbol{s}_t, \boldsymbol{a}_t) \alpha(\boldsymbol{s}_t)$, 如果忽略常数就可以得到 $\begin{aligned} \mathbb{E}_{\tau \sim p(\tau \mid \mathcal{O}_{1:T}, \psi)}\left[\nabla_\psi r_\psi(\tau)\right] &= \sum_{t = 1}^T \iint \mu_t(\boldsymbol{s}_t, \boldsymbol{a}_t) \nabla_\psi r_\psi(\boldsymbol{s}_t, \boldsymbol{a}_t) \text{d}\boldsymbol{s}_t \text{d}\boldsymbol{a}_t\\ &= \sum_{t = 1}^{T} \overrightarrow{\mu}_t^T \nabla_\psi \overrightarrow{r}_\psi. \end{aligned}\\$ 这里的向量是对于全体 $\boldsymbol{s}_t, \boldsymbol{a}_t$ 的向量化.

### 3.3 The MaxEnt IRL algorithm (known dynamics, small state-action space)

于是我们可以给出 **MaxEnt IRL** 的算法:

1.  对给定的 $\psi$, 计算 backward messages $\beta(\boldsymbol{s}_t, \boldsymbol{a}_t)$  
    
2.  对给定的 $\psi$, 计算 forward messages $\alpha(\boldsymbol{s}_t)$.  
    
3.  计算 $\mu_t(\boldsymbol{s}_t, \boldsymbol{a}_t) \propto \beta(\boldsymbol{s}_t, \boldsymbol{a}_t) \alpha(\boldsymbol{s}_t)$.  
    
4.  计算 $\nabla_\psi \mathcal{L} = \frac{1}{N} \sum_{i = 1}^{N} \sum_{t = 1}^{T} \nabla_\psi r_\psi(\boldsymbol{s}_{i, t}, \boldsymbol{a}_{i, t}) - \sum_{t = 1}^{T} \int \int \mu_t(\boldsymbol{s}_t, \boldsymbol{a}_t) \nabla_\psi r_\psi(\boldsymbol{s}_t, \boldsymbol{a}_t) \text{d}\boldsymbol{s}_t \text{d}\boldsymbol{a}_t$.  
    
5.  梯度更新 $\psi \gets \psi + \eta \nabla_\psi \mathcal{L}$.  
    

**Remark:** 这种算法利用了之前对 suboptimality 的建模, 解决了之前的 ambiguity 问题.

**Note:**

-   这样的做法需要计算 backward 与 forward messages, 基于上一节的内容, 这依赖于**已知 dynamic**, 同时由于我们需要对所有的 state-action 对进行计算, 故对大规模连续状态空间的问题并不适用.  
    
-   为什么称这个算法为 max entropy method 呢? 在 reward 线性时, 我们可以证明这个算法最大化了 entropy: $\max_\psi \mathcal{H}(\pi^{r_\psi}) \text{ s.t. } \mathbb{E}_{\pi^{r_\psi}}[\boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})] = \mathbb{E}_{\pi^\ast}[\boldsymbol{f}(\boldsymbol{s}, \boldsymbol{a})].\\$ 这类似于一种 max entropy inference 的方式, 此时除了我们有数据支持的部分, 我们不做任何额外的 inference, 最大化了剩下的不确定性.  
    

## 4 Approximations in High Dimensions

### 4.1 Drawbacks of MaxEnt IRL

在刚才介绍的 MaxEnt IRL 中, 我们需要:

-   在内层循环中计算 (soft) optimal policy (直接更新的是 $\psi$, 但实际上我们需要得到 $\pi^{r_\psi}$)  
    
-   枚举所有的 $\boldsymbol{s}_t, \boldsymbol{a}_t$ 对 (回顾之前的双重"积分")  
    

这需要非常大的计算量, 然而在实际的问题中, 我们可能有:

-   大规模与连续状态空间  
    
-   状态可能只能通过样本获得 (也就没办法简单遍历一遍)  
    
-   未知 dynamics  
    

因此我们无法将这个算法直接应用到实际问题中, 我们需要一些近似的方法来解决这个问题.

### 4.2 Unknown dynamics & large state/ action spaces

在未知 dynamics (无法计算 messages 从而计算 $\mu_t(\boldsymbol{s}_t, \boldsymbol{a}_t)$), 也没办法遍历所有的 state-action 的情况下, 我们如何计算 $\mathbb{E}_{\tau \sim p(\tau \mid \mathcal{O}_{1:T}, \psi)}\left[\nabla_\psi r_\psi(\tau)\right]\\$ 呢? 这里依赖的是 model-free RL 中核心的方法, 采样!

回顾我们的梯度表达式 $\nabla_\psi \mathcal{L} = \mathbb{E}_{\tau \sim \pi^\ast(\tau)}\left[\nabla_\psi r_\psi(\tau)\right] - \mathbb{E}_{\tau \sim p(\tau \mid \mathcal{O}_{1:T}, \psi)}\left[\nabla_\psi r_\psi(\tau)\right].\\$ 前一个期望就是专家的 demonstration, 而后者是我们需要解决的.

**Idea:** 使用任何 **max entropy RL** (也就是上一节介绍的 **soft Q-learning, SAC** 等) 算法, 学习 $p(\boldsymbol{a}_t \mid \boldsymbol{s}_t, \mathcal{O}_{1:T}, \psi)$, 然后运行这个算法获得轨迹 $\{\tau_j\}$, 通过样本估计梯度 $\nabla_\psi \mathcal{L} \approx \frac{1}{N} \sum_{i = 1}^{N} \nabla_\psi r_\psi(\tau_i) - \frac{1}{M} \sum_{j = 1}^{M} \nabla_\psi r_\psi(\tau_j).\\$

然而由于上述估计的无偏性基于 $\{\tau_j\}$ 来源于当前的 $\pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t, \mathcal{O}_{1:T}, \psi)$, 这就意味着我们需要在每次更新 $\psi$ 时都重新学习一个 policy, 这显然是不可接受的.

**Idea:** 使用 "lazy" policy optimization 的方式, 也就是仅仅 improve 这个 policy, 而没有重新学习 $\pi^\psi$ 并从 $p(\boldsymbol{a}_t \mid \boldsymbol{s}_t, \mathcal{O}_{1:T}, \psi)$ 采样. 然而由于分布不匹配, 期望会是 biased 的. 我们需要一些方式来解决这个问题:

一个可能的方式是 **importance sampling**, 假设我们目前有 sample 的 policy 为 $\pi$, 记这一 policy 下轨迹 $\tau$ 的概率为 $\pi(\tau)$, 由于原先期望的概率分布为 $p(\tau \mid \mathcal{O}_{1:T}, \psi) \propto p(\tau) \exp(r_\psi(\tau_j))$, 我们可以得到 importance ratio: $w_j = \frac{p(\tau) \exp(r_\psi(\tau_j))}{C\pi(\tau_j)} = \frac{p(\boldsymbol{s}_1) \prod_t p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t) \exp(r_\psi(\boldsymbol{s}_t, \boldsymbol{a}_t))}{C p(\boldsymbol{s}_1) \prod_t p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t) \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t)} = \frac{\exp(r_\psi(\boldsymbol{s}_t, \boldsymbol{a}_t))}{C\prod_{t} \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t)},\\$ 这里 $C = p(\mathcal{O}_{1:T})$ 是常数, 之后我们用一些 trick 去除, 剩余的都是可以直接计算的形式: 其中的 $r_\psi$ 对应于当前的 $\psi$, 我们可以直接 query 对应的 state-action pair. 而 $\pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t)$ 也可以通过 query 当前的 policy 得到.

由于 $\sum_{j = 1}^M w_j$ 应该与 $M$ 相差不会太大, 这里考虑用 $\sum_{j = 1}^M w_j$ 来替代 $C$, 于是问题转化为 $\nabla_\psi \mathcal{L} \approx \frac{1}{N} \sum_{i = 1}^{N} \nabla_\psi r_\psi(\tau_i) - \frac{1}{\sum_{j} w_j} \sum_{j = 1}^{M} w_j\nabla_\psi r_\psi(\tau_j).\\$ 这里的常数 $C$ 就可以被消掉了.

最后我们考虑一个关键的问题, 为什么要使用 max entropy RL 来训练 policy 呢? 不难理解由于我们的 reward 和 policy 会同时训练, 我们不希望 policy 过早陷入一些特定的行为中, 同时也能够进行 incremental 的更新.

### 4.3 Guided cost learning algorithm

在经过这样的近似后, 我们的算法可以应用在实际的问题中, 一个实际的例子是如下的 **guided cost learning algorithm**:

假设我们已经有了 expert demonstrations 存储在 $\mathcal{D}_{demo}$ 中, 我们有如下的算法:

1.  初始化 policy $\pi$.  
    
2.  重复以下过程:  
    

3.  利用 policy 收集轨迹 $\{\tau_j\}$.  
    
4.  将轨迹添加到 buffer $\mathcal{D}_{sample}$ 中.  
    
5.  利用 $\mathcal{D}_{demo}$ 和 $\mathcal{D}_{sample}$ 更新 reward function $r_\psi$:  
    
6.  利用 $\{\tau_j\}$ (当前轨迹) 更新 policy $\pi$.  
    

![](https://pica.zhimg.com/v2-d1324830f21bf473370bb2e9e253f68e_1440w.jpg)

Guided Cost Learning

参见: Finn et al. ICML '16. Guided Cost Learning. Sampling based method for MaxEnt IRL that handles unknown dynamics and deep reward functions.

## 4 IRL and GANs

这类 MaxEnt IRL 的方法与 **GANs (Generative Adversarial Networks)** 有很强的联系. 这一联系可以启发我们设计更好的 IRL 算法.

### 4.1 Intuition

实际上前面的过程像是一个 reward 和 policy 之间的博弈:

-   **reward**: 通过 objective $\nabla_\psi \mathcal{L} \approx \frac{1}{N} \sum_{i = 1}^{N} \nabla_\psi r_\psi(\tau_i) - \frac{1}{\sum_j w_j} \sum_{j = 1}^{M} w_j\nabla_\psi r_\psi(\tau_j).\\$ 使得 expert demonstration 看起来好, 而我们的 policy 看起来不好.  
    
-   **policy**: 依据 reward 优化自身, 使其更好, 也就是优化 $\nabla_\theta \mathcal{L} \approx \frac{1}{M} \sum_{j = 1}^{M} \nabla_\theta \log \pi(\boldsymbol{a}_j \mid \boldsymbol{s}_j) r_\psi(\tau_j)\\$ 使得其更难从 expert demonstration 中区分.  
    

换言之, policy 试图通过提升其轨迹的 reward 来"骗过" reward function, 而 reward function 试图找到新的 reward 形式来区分 policy 的轨迹与 expert demonstration.

### 4.2 Generative Adversarial Networks

事实上这可以联系到 **Generative Adversarial Networks (GANs)**, 在 GAN 中, **generator** $p_\theta(\boldsymbol{x} \mid \boldsymbol{z})$ 试图生成数据, 而 **discriminator** $D_\psi = p_\psi(\text{real} \mid \boldsymbol{x})$ 试图区分生成的数据与真实数据. $\psi = \arg\max_\psi \frac{1}{N} \sum_{\boldsymbol{x} \sim p^\ast(\boldsymbol{x})} \log D_\psi(\boldsymbol{x}) + \frac{1}{M} \sum_{\boldsymbol{x} \sim p_\theta(\boldsymbol{x})} \log(1 - D_\psi(\boldsymbol{x})).\\$ $\theta = \arg\max_\theta \frac{1}{M} \sum_{\boldsymbol{x} \sim p_\theta(\boldsymbol{x})} \log D_\psi(\boldsymbol{x}).\\$ 这里 $p^\ast(\boldsymbol{x})$ 是数据的真实分布, $p_\theta(\boldsymbol{x})$ 是生成器生成的数据.

![](https://pic4.zhimg.com/v2-c6a0953fab6b84e9ba9f7ea32d1b83b3_1440w.jpg)

GAN

### 4.3 Inverse RL as GAN

不难发现 IRL 与 GAN 之间有着很强的联系, 我们可以把 IRL 看作是一个 GAN. 也就是说, 我们训练一个 discriminator 来区分 expert 的轨迹与我们 policy 生成的轨迹.

回顾在 GAN 中, 我们的 optimal discriminator 是 $D^\ast(\boldsymbol{x}) = \frac{p^\ast(\boldsymbol{x})}{p^\ast(\boldsymbol{x}) + p_\theta(\boldsymbol{x})}.\\$

在 IRL 中, optimal policy 对应的轨迹满足 $p_\theta(\tau) = p(\tau) \exp(r_\psi(\tau)) / Z$, 这里的 $Z$ 是 partition function $Z = \int p(\tau) \exp(r_\psi(\tau)) \text{d}\tau,\\$ 因而我们可以使用如下方式来参数化我们的 discriminator: $\begin{aligned} D_\psi(\tau) = \frac{p(\tau) \frac{1}{Z}\exp(r(\tau))}{p_\theta(\tau) + p(\tau) \frac{1}{Z}\exp(r(\tau))} &= \frac{p(\tau) \frac{1}{Z'}\exp(r(\tau))}{p(\tau)\prod_t \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t) + \frac{1}{Z'} p(\tau) \exp(r(\tau))}\\ &= \frac{\frac{1}{Z'} \exp(r(\tau))}{\prod_t \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t) + \frac{1}{Z'} \exp(r(\tau))}. \end{aligned}\\$ 这里 $p(\tau)$ 为 expert 轨迹的分布, $p_\theta(\tau)$ 为我们 policy 生成的轨迹的分布, 它们满足 $p_\theta(\tau) = p(\tau) \left(\prod_t \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)\right) \bigg/ \left(\prod_t \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t)\right),\\$ 因此这里的第二个等号等价于在分子分母中同时乘上 $\prod_t \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t)$, 并且将其吸收进 $Z'$ 中.

基于上述的记号, 类似于 GAN 中, 我们优化 discriminator 的方式可以写作 $\psi = \arg\max_\psi \mathbb{E}_{\tau \sim \pi^\ast(\tau)}\left[\log D_\psi(\tau)\right] + \mathbb{E}_{\tau \sim \pi_\theta}\left[\log(1 - D_\psi(\tau))\right].\\$

在这一写法中没有 importance sampling 出现, 因为它们都会成为 $D_\psi$ 表达式中 $Z'$ 的一部分. 值得注意的是这里的 $Z'$ 依然需要处理, 我们可以对其用样本进行估算. 具体来说, 可以在上述 objective 中对 $Z'$ 求偏导并令其等于 $0$ 再估计.

基于 GAN 的训练方式可以得到一个对应的 **MaxEnt IRL 算法**:

-   generator/ policy $\pi_\theta(\tau)$, 对应于 GAN 中的 generator, 我们可以从中采样轨迹.  
    
-   expert demonstration $p^\ast(\tau)$, 对应于 GAN 训练中的真实图片.  
    
-   训练 generator/policy 的过程 $\nabla_\theta \mathcal{L} \approx \frac{1}{M} \sum_{j = 1}^{M} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_j \mid \boldsymbol{s}_j) r_\psi(\tau_j).\\$  
    
-   训练 reward function 的过程 $\psi = \arg\max_\psi \mathbb{E}_{\tau \sim \pi^\ast(\tau)}\left[\log D_\psi(\tau)\right] + \mathbb{E}_{\tau \sim \pi_\theta}\left[\log(1 - D_\psi(\tau))\right].\\$ $D_\psi(\tau) = \frac{\frac{1}{Z} \exp(r(\tau))}{\prod_t \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t) + \frac{1}{Z} \exp(r(\tau))}.\\$

![](https://picx.zhimg.com/v2-4ac2752b16cc204dae29d0f0c9306011_1440w.jpg)

Inverse RL as GAN

实质上, 我们可以证明上述利用 GAN 的方式进行 IRL 的方法是等价于我们之前介绍的 MaxEnt IRL 的方法.

参见: Finn\*, Christiano\* et al. "A Connection Between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models."

### 4.4 Generalization via Inverse RL

**Example 2**. _在 "Learning Robust Rewards with Adversarial Inverse Reinforcement Learning, Fu et al., 2017" 中, 作者将 IRL 中学习的 reward function 设置为仅基于 state._

_此时我们在一个 setting 下的 expert demonstration 中学习 reward function. 在另一个不同条件下 (修改质量等), 我们依然可以通过这个 reward function 实现一些有意义的行为. 这在某种意义上"理解"了 expert 的意图 (是向一个方向走), 达成了某种 generalization 的效果._

![](https://pic2.zhimg.com/v2-3c9ab72ee173b33181196d058287cd41_1440w.jpg)

Generalization via Inverse RL

### 4.5 Regular Discriminator

在前面的讨论中, 我们已经注意到 IRL 问题学习 $\psi$ 的过程也可以写作 GAN 中优化 discriminator 的形式: $\psi = \arg\max_\psi \mathbb{E}_{\tau \sim \pi^\ast(\tau)}\left[\log D_\psi(\tau)\right] + \mathbb{E}_{\tau \sim \pi_\theta}\left[\log(1 - D_\psi(\tau))\right].\\$ 此时从 $\psi$ 的 objective 表面上看, 我们摆脱了 IS ratio 等麻烦的东西, 但是如果这里的 $D_\psi$ 还是由 $r_\psi$ 参数化的, 那么我们还是没法避免计算这些东西.

一个直接的想法是, 我们能否使用一个 regular discriminator? 也就是我们使用 $D_\psi$ 为一个常规的二分类网络, 用来区分轨迹是否来自 expert demonstration. 然而正如 GAN 训练终止后 discriminator 会输出 $0.5$, 此时也无法恢复 reward function. 不过这不影响 recover 一个接近 expert 的 policy, 因此事实上这属于 **imitation learning** 的一种算法.

![](https://pic3.zhimg.com/v2-46bdacff4eb6c0b7d82165c183192f06_1440w.jpg)

Generative Adversarial Imitation Learning

**Remark:**

-   这里的问题有更少的可变量, 更加容易优化.  
    
-   但是 discriminator 在收敛时什么也不知道 (正如 GAN 中训练结束后 discriminator 会被丢弃), 同时也无法恢复 reward function.  
    

## 5 Summary

在本节中, 我们:

-   介绍了 inverse RL 的基本概念, 以及其中存在的欠定性问题.  
    
-   介绍了深度学习时代前的 IRL 方法: **Feature matching IRL**, 这依然是一种 heuristic 的做法, 不适用于深度学习.  
    
-   介绍了 **MaxEnt IRL** 算法:  
    

-   推导过程基于上一节介绍的概率图模型, 通过计算 backward messages 和 forward messages 来计算 $\mu_t(\boldsymbol{s}_t, \boldsymbol{a}_t)$, 进而优化 reward function.  
    
-   其严格的做法需要知道 dynamics, 同时需要遍历所有的 state-action 对, 以及不断训练最新的 reward 的 optimal policy, 这在实际中不可行.  
    
-   通常常用的做法是通过一些近似, 使用采样的方式来估计梯度, 并且使用 "lazy" policy optimization 的方式来避免重新训练 policy. 一个应用的例子是 **guided cost learning algorithm**.  
    

-   介绍了 IRL 和 GAN 之间的紧密联系, 以及如何通过 GAN 相关的 formulation 进行 IRL, 并介绍了一些应用以及转化为 imitation learning 的方法.  
    

### Further reading:

### Classic Papers:

-   Abbeel & Ng ICML '04. Apprenticeship Learning via Inverse Reinforcement Learning.  
    
-   Good introduction to inverse reinforcement learning Ziebart et al. AAAI '08. Maximum Entropy Inverse Reinforcement Learning. Introduction to probabilistic method for inverse reinforcement learning.  
    

### Modern Papers:

-   Finn et al. ICML '16. Guided Cost Learning. Sampling based method for MaxEnt IRL that handles unknown dynamics and deep reward functions.  
    
-   Wulfmeier et al. arXiv '16. Deep Maximum Entropy Inverse Reinforcement Learning.  
    
-   MaxEnt inverse RL using deep reward functions Ho & Ermon NIPS '16. Generative Adversarial Imitation Learning. Inverse RL method using generative adversarial networks.  
    
-   Fu, Luo, Levine ICLR '18. Learning Robust Rewards with Adversarial Inverse Reinforcement Learning.