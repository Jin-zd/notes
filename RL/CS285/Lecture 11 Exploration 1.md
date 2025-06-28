## 1 Introduction
为什么一些看起来对人类更简单的游戏，例如 Montezuma's Revenge，对于强化学习算法来说却是一个很大的挑战？

这是因为探索问题的困难性。这一问题的困难性主要体现在以下两个方面：
- 任务的时间延续性  
- 对规则的未知程度  
![](11-1.png)
### 1.1 How extended the task is
一方面，当任务的时间延续性增加时，采用常规的探索策略很可能无法探索到有效的奖励信号。例如在 Montezuma's Revenge 中，需要完成一系列非常长序列的任务，而这些序列并不包含中间奖励。

我们可以试图添加一些中间奖励来帮助智能体更好地学习。在上述例子中，我们可以添加一些中间的奖励，例如被骷髅杀死。但是这样的做法也会带来一些问题，例如被骷髅杀死如果设置正的奖励很可能导致智能体最终学会最快地被骷髅杀死，然而设置负的奖励则可能导致智能体不会尝试通过骷髅，导致无法到达更远的地方。

### 1.2 How unknown the rules are
另一方面，不同于人类对游戏规则有先验的理解（例如上图中的骷髅图标，对于人类来说我们知道要避免的，而钥匙的位置是我们想要到达的），强化学习算法并不知道这一些，只能够通过试错来学习。

不难发现，无论是任务的时间延续性增加，还是规则的未知程度增加，都会给我们的探索问题带来更大的挑战，也就需要更加有效的探索策略。

## 2 Exploration and exploitation
### 2.1 Definition and Examples
两种对探索问题的潜在定义方式：
- 一个智能体如何才能发现一些高奖励的策略，如果这些策略需要一个很长时间的复杂行为，且每一个行为都没有奖励。
- 一个智能体如何决定是继续尝试一些新的行为，还是继续做已知最优的行为。

事实上二者都是探索问题：
- 利用：选择已知可获取最高奖励的行为。  
- 探索：选择之前没有尝试过的行为，期望能够发现更高奖励的行为。

### 2.2 Optimal Exploration
不难理解探索是一个非常困难的问题，最高的目标显然是推导出一个最优的探索策略。（当然，我们必须先定义什么是最优的，通常情况下有利用遗憾值与贝叶斯最优策略两种方式，在之后的讨论中会详细介绍。）

自然地，在不同难度的问题上，可以得到不同程度的理论结果，从获取理论结果的难度上分类，从简单到困难依次为：
- 多臂老虎机（Multi-armed bandits）：相当于单步的无状态马尔可夫决策过程。
- 上下文老虎机（Contextual bandits）：相当于单步马尔可夫决策过程。
- 小型、有限的马尔可夫决策过程（Small，finite MDPs）：例如易处理的规划、基于模型的强化学习。
- 大型、无限的马尔可夫决策过程，连续空间（Large，infinite MDPs，continuous space）：例如深度强化学习。

想法：尽管对于复杂的问题难以得到具有很强理论保证的算法，但是我们接下来会从简单的问题出发介绍一些可行的探索策略并尝试将这些方法应用到更复杂的问题中。

接下来用多臂老虎机作为例子，逐步引出探索问题的一些基本方法。

### 2.3 Multi-arm Bandits
![](11-2.png)

**Definition 1**. _Multi-armed bandits（多臂老虎机）_
在多臂老虎机中，有 $n$ 个旋臂，对应于动作空间 $\mathcal{A}$ 中的动作 $\{a_1, a_2, \ldots, a_n\}$。每一个旋臂都有一个奖励 $r(a_i)$，其独立服从于某个分布 $p(r \mid a_i)$，我们的目标是最大化累积奖励。

不妨假设 $r(a_i) = p_{\theta_i}(r_i)$，其中 $\theta_i$ 是概率分布的参数。可以给 $\theta_i$ 一个先验分布 $p(\theta)$，并记置信状态（对状态的估计）为 $\hat{p}(\theta)$。

为了简化起见，考虑静态多臂老虎机， 即 $\theta_i$ 是固定的，上述过程可以视作一个单步的部分可观测马尔可夫决策过程（POMDP）：
1. 状态空间：仅有单个状态 $\theta = [\theta_1, \ldots, \theta_n]$；
2. 动作空间：$\mathcal{A} = \{a_1, a_2, \ldots, a_n\}$；
3. 观测空间：$\mathbb{R}$ （如果假设奖励是连续的）；
4. 转移模型：$\theta_{t + 1} = \theta_t$ （静态多臂老虎机）；
5. 观测模型：$p(r \mid a_i, \theta) = p_{\theta_i}(r)$；
6. 奖励：$r(a_i) \sim p_{\theta_i}(r)$。

当求解这个部分可观测马尔可夫决策过程时，就得到了最优的探索策略。这样的做法可能有一些大材小用，实际可以使用更加简单的策略，它们在渐近性能（大 O 符号）的意义下和最优策略是一样的。以下是衡量探索算法的好坏的标准。

**Definition 2**. _Regret（遗憾值）_
对于一个探索策略，定义遗憾值为 $Reg(T) = T\cdot E\left[r(a^\ast)\right] - \sum_{t = 1}^{T} r(a_t)$，前者是最优动作的期望奖励，后者是智能体实际获得的奖励。

一个探索策略的好坏由其遗憾值来衡量，我们希望遗憾值越小越好。

## 3 Three Classes of Exploration Methods
接下来介绍几种简单的探索策略，可以理论证明它们在渐近性能的意义下是最优的，尽管它们的实际表现可能存在着一定的差异，可以将它们作为复杂问题的探索策略的启发。
### 3.1 Optimistic exploration
对每一个动作 $a$ 记录 $\hat{\mu}_a$，考虑乐观估计 $a = \arg\max_a \hat{\mu}_a + C \sigma_a$，这里的 $\sigma_a$ 是动作 $a$ 的某种方差估计。

直觉：要尝试每一个旋臂，能够足够确信它不够好。
可以证明该算法的遗憾值是 $O(\log T)$，对这个问题在理论上是最优的。

### 3.2 Probability matching/ posterior sampling
在这一做法中，会保留一个置信状态 $\hat{p}(\theta_1, \ldots, \theta_n)$。

想法：假设置信状态是正确的，使用 $\theta_1,\ldots, \theta_n \sim \hat{p}(\theta_1, \ldots, \theta_n)$，依据此选择最优动作。
这样的方式比直接求解部分可观测的马尔可夫决策过程要简单很多，这样的算法称为后验采样（Posterior sampling），也称为汤普森采样（Thompson sampling）。
分析这一算法的理论性质是很困难的，但是实际中表现很好，具体可见 An Empirical Evaluation of Thompson Sampling. Chapelle, Li。

### 3.3 Information gain
在这一类方法中，考虑如何最大化信息增益，在多臂老虎机问题中，我们希望最大化对 $\theta$ 的信息增益。由于实际的信息增益是由于动作而产生的，故可以定义一定的动作 $a$ 下的信息增益。

**Definition 3**. _information gain（信息增益）_
定义观测到 $y$ 后（这里理解为是对随机变量的单个观测）的信息增益为 
$$
IG(z,y) = \mathcal{H}(\hat{p}(z)) - \mathcal{H}(\hat{p}(z) \mid y)
$$

不难基于信息论的知识发现其与互信息（Mutual information）之间的联系：
**Proposition 1**. $\mathbb{E}_y\left[IG(z,y)\right] = \mathcal{I}(z,y)$

**Definition 4**. _information gain（信息增益）_
定义动作 $a$ 下的信息增益为 
$$
IG(z,y \mid a) = \mathbb{E}_{y}\left[\mathcal{H}(\hat{p}(z)) - \mathcal{H}(\hat{p}(z) \mid y) \mid a\right]
$$
这里 $y$ 的期望是基于 $p(y \mid a)$。

例如，应用在多臂老虎机问题中的算法例子是 Learning to Optimize via Information-Directed Sampling. Russo et al. 具体来说：
- 观测的变量是 $y = r_a$，需要估计的是 $z = \theta_a$，其中 $\theta_a$ 是动作 $a$ 的奖励的参数；
- 记 $g(a) = IG(\theta_a, r_a \mid a)$ 为 $a$ 的信息增益；
- 记 $\Delta(a) = \mathbb{E}[r(a^\ast) - r(a)]$ 为 $a$ 的预期的次优性；
- 选择 $a$ 基于 $\arg\min_a \frac{\Delta(a)^2}{g(a)}$。

不难观察发现这一项也平衡了利用和探索。

## 4 Overview of Exploration in RL
在更加复杂的情形中，没办法得出这些简单问题中能得到的理论保证，但是可能会基于这些简单问题中的一些见解来设计探索策略。可以将其应用到强化学习中，以下是对这一些做法的总览：
- 乐观探索；
- 新状态 = 良好状态；
- 需要通过对状态访问计数等方式来估计状态是否足够新；
- 通常通过探索奖励来实现；
- 汤普森采样风格算法；
- 学习一个 Q 函数或策略上的（信念）分布；
- 从信念分布中采样，并依据这一采样进行决策；
- 信息增益风格算法；
- 考虑访问新状态的信息增益。

## 5 Optimistic exploration in RL
置信上界（Upper Confidence Bound，UCB）算法是多臂老虎机问题中的一种经典策略。它旨在解决探索与利用之间的权衡问题。简单来说，UCB 算法在每次选择时，会考虑每个选项（“臂”）的当前估计收益，并加上一个基于其不确定性的“置信上界”。这使得算法能够优先选择那些目前看起来最好的臂，同时也能给那些探索不足的臂提供尝试的机会。

UCB 算法的核心思想是“悲观的乐观主义”。它不只看一个臂的平均奖励，还会给那些被选择次数较少的臂一个“额外加成”，这个加成随着选择次数的增加而减小。 每次选择一个臂时，UCB 算法会计算每个臂的 UCB 值，并选择具有最高 UCB 值的臂。UCB 值由两部分组成： $$UCB_t(a) = \bar{Q}_t(a) + C \sqrt{\frac{\ln t}{N_t(a)}}$$其中：
- $UCB_t(a)$：在时间步 $t$ 时，臂 $a$ 的上置信界值。 
- $\bar{Q}_t(a)$：臂 $a$ 在时间步 $t$ 之前获得的平均奖励（即当前对该臂收益的估计）。这代表了“利用”的部分，因为它倾向于选择平均收益高的臂。
- $N_t(a)$：臂 $a$ 在时间步 $t$ 之前被选择的次数。
- $t$：当前的总试验次数。
- $C$：一个探索参数（或常数），用于调整探索的程度。$C$ 值越大，探索的倾向性越强。

考虑如下的 UCB形式：
$$
a = \arg\max_a \hat{\mu}_a + \sqrt{\frac{2\ln T}{N(a)}}
$$
这里后一项是探索奖励。

值得注意的是，多臂老虎机的特殊性在于其只有单个状态，对于强化学习中的探索，不仅要对动作有探索奖励，同时对状态也要有探索奖励，也就是通常会考虑 $N(\boldsymbol{s}, \boldsymbol{a})$ 或 $N(\boldsymbol{s})$ 形式的计数。

### 5.1 Count-based exploration
直觉：一个状态或状态-动作对被访问的次数越少，需要添加的探索奖励越大。

基于这一想法，可以定义添加了探索奖励的附加奖励函数：
$$
r^+(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \mathcal{B}(N(\boldsymbol{s}))
$$
这里的 $\mathcal{B}(N(\boldsymbol{s}))$ 是一个随着 $N(\boldsymbol{s})$ 增加而减小的函数，此时我们使用 $r^+(\boldsymbol{s}, \boldsymbol{a})$ 作为附加奖励函数，进行探索。

注意：
- 优点：很容易添加到任何的强化学习算法中。  
- 缺点：需要调整出一个合适的 $\mathcal{B}(N(\boldsymbol{s}))$。

### 5.2 Counting in complex problems
在复杂的问题中，重复访问一个状态的可能性是很小的，而在连续状态空间中，不可能到达同一个状态多次。
![](11-3.png)

在这种情况下，我们需要设计其他的方法来估计 $N(\boldsymbol{s})$。然而值得注意的是，状态之间的相似性各有不同，可以拟合一个密度模型 $p_\theta(\boldsymbol{s})$ （或 $p_\theta(\boldsymbol{s}, \boldsymbol{a})$）。如果一个状态与已见到的状态相似，那么即使其没有被访问过，也可以有很大的 $p_\theta(\boldsymbol{s})$。

密度和计数通常还是有一定差异的，如果能够将密度转化为一个伪计数，那么就可以沿用原先对探索奖励的设计。实际上，我们能够把 $p_\theta(\boldsymbol{s})$ 转化为伪计数 $N(\boldsymbol{s})$，这里的做法基于 Unifying Count-Based Exploration and Intrinsic Motivation. Bellemare et al. 2016：

在访问 $\boldsymbol{s}$ 前后，可以列出以下两个方程： 
$$
P(\boldsymbol{s}) = \frac{N(\boldsymbol{s})}{n},\, P'(\boldsymbol{s}) = \frac{N(\boldsymbol{s}) + 1}{n + 1}
$$

可以让 $p_\theta(\boldsymbol{s})$ 与 $p_{\theta'}(\boldsymbol{s})$ 也遵循同样的这两个方程：
1. 利用当前见过的所有状态 $\mathcal{D}$ 拟合一个密度模型 $p_\theta(\boldsymbol{s})$；
2. 走一步 $i$ 观测到 $\boldsymbol{s}_i$；
3. 用 $\mathcal{D} \cup \boldsymbol{s}_i$ 拟合一个新的密度模型 $p_{\theta'}(\boldsymbol{s})$；
4. 使用 $p_{\theta}(s)$ 与 $p_{\theta'}(s)$ 来更新 $\hat{N}(\boldsymbol{s})$；
5. 设置 $r^+(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \mathcal{B}(\hat{N}(\boldsymbol{s}))$。

其中第 4 步进行的方式如下：
联立 
$$
\begin{cases}
p_\theta(\boldsymbol{s}_i) = \frac{\hat{N}(\boldsymbol{s}_i)}{\hat{n}}, \\ p_{\theta'}(\boldsymbol{s}_i) = \frac{\hat{N}(\boldsymbol{s}_i) + 1}{\hat{n} + 1}
\end{cases}
$$
可以解出 $\hat{N}(\boldsymbol{s}_i)$ 与 $\hat{n}$ 两个未知数：
$$
\begin{cases}  \hat{N}(\boldsymbol{s}_i) = \hat{n} p_\theta(\boldsymbol{s}_i)\\  \hat{n} = \frac{1 - p_{\theta'}(\boldsymbol{s}_i)}{p_{\theta'}(\boldsymbol{s}_i) - p_\theta(\boldsymbol{s}_i)}  \end{cases}  
$$
![](11-4.png)

### 5.3 Choice of bonus function
首先考虑使用什么样的附加奖励函数 $\mathcal{B}$，我们有以下几种有效的选择：
- UCB：$\mathcal{B}(N(\boldsymbol{s})) = \sqrt{\frac{2\ln T}{N(\boldsymbol{s})}}$  
- MBIE-EB (Strehl & Littman, 2008)：$\mathcal{B}(N(\boldsymbol{s})) = \sqrt{\frac{1}{N(s)}}$  
- BEB (Kolter & Ng, 2009)：$\mathcal{B}(N(\boldsymbol{s})) = \frac{1}{N(\boldsymbol{s})}$  

### 5.4 Choice of density model
接下来考虑使用什么样的密度模型 $p_\theta$：由于这里的目标仅仅是一个输出密度的模型，并不需要能够从中采样或者生成数据，因此可以使用一些简单的模型，例如 CTS model: condition each pixel on its top-left neighbor。

其他的一些模型：stochastic networks，compression length，EX2。

## 6 More Novelty-Seeking Exploration
由于只需要能够输出分数，还可以使用以下的一些更加新颖的思路。
### 6.1 Counting with hashes
想法：依然使用计数，但是我们利用一个哈希函数 $\phi(\boldsymbol{s})$ 将 $\boldsymbol{s}$ 映射到一个 $k$-bit 的哈希值，并尽可能使得相似状态有相似的哈希值。

具体来说，可以利用[[Concepts#21 自动编码器 (Autoencoder，AE)|自动编码器 (Autoencoder，AE)]]学习一个编码器，在此基础上进行降采样将 $\phi(\boldsymbol{s})$ 转化为只有 01 的哈希值。
![](11-5.png)
参见：Tang et al. "Exploration: A Study of Count-Based Exploration"

### 6.2 Implicit density modeling with exemplar model
想法：可以利用分类器来进密度估计，如果一个状态很容易与已见过的状态区分，则其是新颖的，如果一个状态与过去状态难以区分, 则说明其有很高的密度。

具体来说，对于每一个观测到的状态 $\boldsymbol{s}$，拟合一个分类器将其与所有过去的状态 $\mathcal{D}$ 区分开，利用分类器误差来获得密度。我们认为 $\{\boldsymbol{s}\}$ 是正类别，而 $\mathcal{D}$ 是负类别。

**Definition 5**. _exemplar model（范例模型）_
对于数据集 $X = \{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n\}$，范例模型是一系列模型，$\{D_{\boldsymbol{x}_1}, \ldots, D_{\boldsymbol{x}_n}\}$，其中 $D_{\boldsymbol{x}_i}$ 是一个二元分类器，用于区分 $\boldsymbol{x}_i$ 与其他数据。

**Theorem 1**. 对于离散分布，可以使用
$$
p_\theta(\boldsymbol{s}) = \frac{1 - D_{\boldsymbol{s}}(\boldsymbol{s})}{D_{\boldsymbol{s}}(\boldsymbol{s})}
$$ 
表示状态 $\boldsymbol{s}$ 的密度，其中 $D_{\boldsymbol{s}}(\boldsymbol{s})$ 表示的是将 $\boldsymbol{s}$ 归类为正类别概率。

_Proof._ 
不妨假设负样本的概率为 $q(\boldsymbol{s})$，这其实就是想要估计的密度。很显然最优分类器将 $\boldsymbol{s}$ 归类为正类别的概率为 
$$
D_{\boldsymbol{s}}(\boldsymbol{s}) = \frac{p(\boldsymbol{s})}{p(\boldsymbol{s}) + q(\boldsymbol{s})}
$$
可以解得 
$$
q(\boldsymbol{s}) = \frac{1 - D_{\boldsymbol{s}}(\boldsymbol{s})}{D_{\boldsymbol{s}}(\boldsymbol{s})} p(\boldsymbol{s})
$$
注意由于正类的概率密度函数是一个点分布，因此 $p(\boldsymbol{s}) = 1$，也就是 
$$
p_\theta(\boldsymbol{s}) = \frac{1 - D_{\boldsymbol{s}}(\boldsymbol{s})}{D_{\boldsymbol{s}}(\boldsymbol{s})}
$$
然而很显然对于连续分布，由于点分布的概率密度认为是 $\infty$，我们需要进行一些正则化处理，例如将点分布换成一个很小的高斯分布。

由于对每一个状态都需要训练一个分类器，这一方法可能会很昂贵，可以考虑使用摊销模型（Amortized model），例如只训练一个模型，并且将范例作为条件。

参见：EX2: Exploration with Exemplar Models for Deep Reinforcement Learning. Ostrovski et al. 2017

### 6.3 Heuristic estimation of counts via errors
想法：在训练神经网络时，对于那些出现概率大或密度高的数据，其上的误差会很小（否则总损失就会很大），而在密度小的数据上的误差会很大。

具体来说，不妨考虑我们有一个目标函数 $f^\ast(\boldsymbol{s}, \boldsymbol{a})$，给定缓冲 $\mathcal{D} = \{(\boldsymbol{s}_i, \boldsymbol{a}_i)\}$，拟合一个 $\hat{f}_\theta(\boldsymbol{s}, \boldsymbol{a})$。可以使用 $\mathcal{E} = \|\hat{f}_\theta(\boldsymbol{s}, \boldsymbol{a}) - f^\ast(\boldsymbol{s}, \boldsymbol{a})\|^2$ 作为附加奖励，这一值越大，说明这一状态动作对越新颖。

接下来可以考虑 $f^\ast(\boldsymbol{s}, \boldsymbol{a})$ 的选择问题：
- 一个通常的选择是令 $f^\ast(\boldsymbol{s}, \boldsymbol{a}) = \boldsymbol{s}'$，也就是预测下一个状态（这与信息增益有一定的联系）。
- 更简单的选择是 $f^\ast(\boldsymbol{s}, \boldsymbol{a}) = f_\phi(\boldsymbol{s}, \boldsymbol{a})$，其中 $\phi$ 是一个随机参数向量。根据前面的直觉，这里的 $f_\phi$ 的形式完全不重要，只要它是一个在 $\mathcal{S} \times \mathcal{A}$ 上不容易简单拟合的函数即可。
![](11-6.png)
参见：Burda et al. Exploration by Random Network Distillation. 2018

## 7 Posterior Sampling in Deep RL
前面介绍了非常多种乐观探索 / 寻求新颖探索的方法，接下来考虑在多臂老虎机中引入的第二个方法： 汤普森采样（Thompson sampling）。

### 7.1 Introduction
回顾汤普森采样：$\theta_1,\ldots, \theta_n \sim \hat{p}(\theta_1, \ldots, \theta_n)$，$a = \arg\max_a \mathbb{E}_{\theta_a}\left[r(a)\right]$

想法：在多臂老虎机中 $\hat{p}(\theta_1, \ldots, \theta_n)$ 可以视作是奖励 $s$ 上的分布，这在马尔可夫决策过程中的近似是 Q 函数。
具体来说，考虑以下的过程：
1. 从 $p(Q)$ 中采样一个 Q 函数 $Q$；  
2. 依据 $Q$ 做一个回合的动作；  
3. 更新 $p(Q)$。

这里使用 Q 函数的好处在于，由于 Q 学习是 off-policy，我们并不关心收集数据的 Q 函数是什么。
但尚未解决的问题是，应该如何表示 $p(Q)$ 呢？

### 7.2 Bootstrap
回忆在基于模型的强化学习中为了衡量模型的不确定性以避免滥用动态模型，我们引入了自举集成模型方法。类似地，我们可以使用集成 Q 函数来表示 $p(Q)$。

基于之前讨论过的做法：
- 给定数据集 $\mathcal{D}$，利用有放回采样得到 $N$ 个数据集 $\mathcal{D}_1, \ldots, \mathcal{D}_N$，于是可以训练 $N$ 个模型 $f_{\theta'}$。在深度学习中的技巧是使用同一个数据集，但是使用不同的初始化； 
- 为了采样 $p(\theta)$，只需要采样一个索引 $i$ 并使用 $f_{\theta_i}$；
- 这依然非常昂贵，更进一步的做法是（参见后面提到的论文）使用一个共享网络，但是用多个不同的头（最后的层）。
![](11-7.png)

### 7.3 Comparison with $\epsilon$\-greedy
相较于仅有单个 $Q$ 函数的算法，这一算法引入随机性通过 $p(Q)$ 实现了探索，而 $\epsilon$-greedy 则是在现有 $Q$ 的基础上通过随机性进行探索。这两种方式都引入了随机性，然而它们的表现可能会有很大的不同。

其背后的原因主要在于：
- 在 $\epsilon$\-greedy 中，在一个回合中并不会坚持固定的某种行为，而是会来回震荡，这可能会导致无法走到一个有价值的位置。
- 利用随机 Q 函数进行探索时，可以在整个回合中坚持一个一致的策略。
![](11-8.png)
在这个潜艇的游戏中，不同 Q 函数可以对应不同的整局游戏一致的策略，例如倾向于往上，而  $\epsilon$-greedy 则会上下震荡。

注意：
1. 不难发现这个算法相当简单，对原始的奖励函数不需要做任何的修改，只需要使用集成 Q 函数。  
2. 然而，遗憾的是，这个做法的表现不如基于计数的方法 / 伪计数，例如在 Montezuma's Revenge 中完全没有用。

参见：Osband et al. Deep Exploration via Bootstrapped DQN. 2016

## 8 Information Gain in Deep RL
在介绍完前两种探索策略，即乐观探索与汤普森采样后，我们接下来考虑第三种探索策略：信息增益。

首先需要考虑的是，在多臂老虎机中未知的只有一个状态 $\theta$，但在深度强化学习中未知的有很多，选取什么作为信息增益的指标呢？主要可以考虑的有：
- 使用奖励$r(\boldsymbol{s}, \boldsymbol{a})$：在奖励稀疏时并不是很有用； 
- 使用 $p(\boldsymbol{s})$：那么又回到了某种基于计数的探索；
- 使用动态 $p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})$ 是一个很好的选择，但这样的做法同样是启发式的，更多是一种经验性和直觉性的做法。

然而，对于复杂的问题来说，直接使用信息增益，无论估计什么都是难以处理的，因此通常会使用一些近似。

### 8.1 Prediction gain
考虑一个密度模型 $p_\theta(\boldsymbol{s})$，是当见到一个新状态 $\boldsymbol{s}$ 之后更新了参数 $\theta$ 到 $\theta'$，用 $\log p_{\theta'}(\boldsymbol{s}) - \log p_\theta(\boldsymbol{s})$ 作为一种信息增益的粗略估计。

如果 $\log p_{\theta'}(\boldsymbol{s}) - \log p_\theta(\boldsymbol{s})$ 很大，也就意味着原先在 $\boldsymbol{s}$ 附近的不确定性很大，而现在对 $\boldsymbol{s}$ 的不确定性降低了很多。

注意：这一做法比较粗略，同时也与基于计数的探索有着较强的联系。

参见：
- Schmidhuber. (1991). A possibility for implementing curiosity and boredom in model-building neural controllers（定义了“无聊感”的概念，当某个状态被充分探索，预测误差趋近于零时，智能体对该状态失去兴趣，内在奖励降低，转而探索其他区域）
- Bellemare, Srinivasan, Ostroviski, Schaul, Saxton, Munos. (2016). Unifying Count Based Exploration and Intrinsic Motivation （介绍了信息增益与基于计数的探索之间的一些联系）

### 8.2  Variational information maximizing exploration（VIME）
不难得出以下的结论：
Proposition 2. 
$$
\mathbb{E}_y \left[IG(z, y)\right] = \mathbb{E}_y \left[D_{KL}(p(z\mid y) \parallel p(z))\right]
$$
这意味着 $y$ 中包含的信息越多，那么 $p(z\mid y)$ 与 $p(z)$ 的 KL 散度就越大。

在变分信息最大化探索（Variational information maximizing exploration，VIME）中，学习一个参数为 $\theta$ 的动态模型 $p_\theta(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$，并且考虑对参数 $z = \theta$ 的信息增益，我们的观测则是 $y = (\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1})$。于是想要最大化的就是 
$$
D_{KL}(p(\theta \mid h, \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}) \parallel p(\theta \mid h))
$$
这里的 $h$ 是历史转移。

想法：不难发现如果一个转移让对 $\theta$ 的置信度有更大的变化，则其更加的提供有用信息。

首先明确我们需要对模型的不确定性也就是 $\theta$ 进行一个表示，这里使用在基于模型的强化学习中提到的贝叶斯神经网络。首先使用独立性假设 $p(\theta \mid h) = \prod_i p(\theta_i \mid h)$，然后使用一个高斯分布来表示 $p(\theta_i \mid h)$，也就是 $p(\theta_i \mid h) = \mathcal{N}(\mu_{\phi,i}, \sigma_{\phi,i})$，这里的 $\phi$ 是参数，换言之会使用一个贝叶斯神经网络 $q_\phi(\theta)$ 来近似 $p(\theta \mid h)$。

那么应该如何保证近似的有效性呢？可以使用[[Concepts#27 变分推断（Variational Inference，VI）|变分推断（Variational Inference，VI）]]，具体来说，考虑最小化 KL 散度：
$$
\begin{aligned}  D_{KL}(q_\phi(\theta) \parallel p(\theta \mid h)) &= \int q_\phi(\theta) \log \frac{q_\phi(\theta)}{p(\theta \mid h)} \text{d}\theta\\  &= \int q_\phi(\theta) \log \frac{q_\phi(\theta) p(h)}{p(\theta, h)} \text{d}\theta\\  &= \log p(h) - \int q_\phi(\theta) \log \frac{p(\theta, h)}{q_\phi(\theta)} \text{d}\theta \end{aligned}
$$
如果熟悉变分推断的话，会发现前者 $\log p(h)$ 是证据，而 $\int q_\phi(\theta) \log \frac{p(\theta, h)}{q_\phi(\theta)} \text{d}\theta$ 是证据下界（Evidence lower bound，ELBO）。如果假设证据不变，任务就转化为最大化证据下界，也即最小化 $D_{KL}(q_\phi(\theta) \parallel p(\theta) p(h \mid \theta))$。优化这个证据下界的过程需要使用随机梯度变分贝叶斯（Stochastic gradient variational Bayes，SGVB）算法，我们暂不展开。

在有了一个有效的近似之后，信息增益就可以近似表示为
$$
D_{KL}(q_{\phi'}(\theta) \parallel q_\phi(\theta))
$$
其中 $\phi'$ 是观测到一个新样本后更新后的参数。

整个训练流程大致如下：
1. 对于 $n = 1,\ldots,N$：
2. 重复以下 $K$ 次：
3. 从环境中采样一个转移 $(\boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{t + 1}, r_t)$，并添加到缓冲 $\mathcal{R}$；  
4. 通过 $\phi_{n + 1}$ 与新样本近似 $\phi_{n + 1}'$，基于此计算信息增益作为附加奖励项；
5. 利用 $r_t$ 与附加奖励项构造获取新的奖励； 
6. 从缓冲 $\mathcal{R}$ 中采样一个小批次，通过最小化 ELBO 训练贝叶斯神经网络 $q_{\phi_n}(\theta)$ 至 $q_{\phi_{n + 1}}(\theta)$；
7. 利用构造的奖励以及任意的强化学习算法训练策略。

近似信息增益的好处是有数学上更强的保证，但是其缺点是模型通常会更加复杂，更难得到有效使用。

### 8.3 Exploration with model errors
这一部分介绍的方法并没有利用信息增益，考虑 VIME 中的 $D_{KL}(q_{\phi'}(\theta \mid h) \parallel q_\phi(\theta \mid h))$，可以理解为惩罚观测一个状态产生的梯度。这里的直觉其实和基于模型误差的探索很像，在误差很大的地方的状态产生的梯度自然也会很大。

以下是一些示例：
Stadie, Levine, Abbeel (2015). Incentivizing Exploration in Reinforcement Learning with Deep Predictive Models.
- 用自动编码器来编码图片观测； 
- 使用隐码空间上的预测模型；
- 利用模型误差作为探索附加奖励。

更多的变种可见 Formal Theory of Creativity, Fun, and Intrinsic Motivation：
- 利用模型误差作为探索附加奖励；
- 利用模型梯度作为探索附加奖励。
## 9 Summary
在本节中，我们：
- 引出了探索问题；  
- 介绍了多臂老虎机中常见的三种探索策略：乐观探索、汤普森采样、信息增益；
- 将三种常见的探索策略推广到了深度强化学习中，具体来说： 
	- 乐观探索：
		- 使用计数与伪计数作为探索附加奖励；
		- 除了转化为计数，还可以使用一些更加新颖的方法，例如哈希，范例模型。
	- 汤普森采样风格的算法：
		- 通过自举集成维持一个模型的概率分布；  
		- 一个选择是选择 Q 函数上的分布。  
	- 信息增益风格算法：
		- 这一类方法比较复杂，通常需要使用一些近似；
		- 可以使用变分推断近似来估计信息增益：VIME。