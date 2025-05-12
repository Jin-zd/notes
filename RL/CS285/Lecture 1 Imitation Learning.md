# 0 Terminology & Notation

-   $\boldsymbol{o}_t$：**observation**。通常情况下，observation 是我们能够观测到的，但是并不包含所有的信息
-   $\boldsymbol{s}_t$：**state**。不同于 observation，state 是一个对当前状态完整的描述，包含了所有的信息. 通常情况下，state 是不可见的，但是我们可以通过 observation 来近似地估计 state
-   $\boldsymbol{a}_t$：**action**，$\boldsymbol{a}_t$ 会影响未来的观测，可以是连续的，也可以是离散的
-   $\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{o}_t)$：**policy**。通常是一个概率分布（确定性的 policy 是一个特例）
-   $\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)$：**policy (fully observed)**
-   $t$：**time step** 
这些量之间可以利用以下概率图来表示 (在部分可观测的情况下)：

![](https://picx.zhimg.com/v2-9770e66d350346f6cc252a2a648c74f3_1440w.jpg)

通常我们认为各个状态之间满足马尔可夫性质，接下来我们介绍的一部分算法将会依赖于这一性质，而在另一部分算法中则并不依赖，因此在一些时候我们会互换两种记号。

# 1 Imitation Learning

广义来说，**模仿学习（imitation learning）** 是指从专家的行为中学习。通常情况下，专家的行为是通过一些方式收集的，例如人类的行为。我们接下来从一个例子出发展现模仿学习的一些特点：

从人类驾驶的数据中获取每一时间步的 $\boldsymbol{o}_t，\boldsymbol{a}_t$，并将这些数据作为训练数据，在其上使用监督学习学习 $\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{o}_t)$，这样的一种方式也被称为**行为克隆（behavior cloning）**。

这样的算法能够起作用吗? 不妨仅考虑一个轨迹，我们的模型总会有一些误差，一旦出现误差则意味着到达了一个不熟悉的状态，这样的误差会被积累，直到我们到达极不熟悉的状态，这样的状态下我们的模型可能会完全失效。
![](https://pic2.zhimg.com/v2-63a4aaeb8d71a813e5086bcfd7fb6e33_1440w.jpg)

在通常的监督学习任务中，我们似乎不会遇到这种问题，这是因为监督学习中的数据[[Concepts#1 独立同分布（i.i.d.）| 独立同分布]]的，然而在当前的问题中，我们的数据是一个序列，前一个决策的微小不同都会影响后续的观测.
然而也有一些方法解决部分问题，例如引入一些 "假数据"：在车辆的左前和右前方分别放置两个摄像头，分别标记 $\boldsymbol{a}_t$ 为右转和左转，这样当我们的模型偏离时，可以利用这些数据来矫正。然而这样的方法是针对于特定的问题，并不是一个通用的方法。
![](https://picx.zhimg.com/v2-a880ee47080caab2bbd48cca4ac3bf27_1440w.jpg)
在我们了解相关例子后，我们接下来从以下角度继续介绍：
- 通过行为克隆实现的模仿学习并不一定能保证有效：这与监督学习的情况不同，原因在于独立同分布条件不再满足，我们接下来从理论角度解释这一点。
- 我们可以通过一些具体方法来解决这一问题，但是这些方法可能并不是通用的：
	- 采用更聪明的方式来收集数据
	- 使用更加复杂的模型
	- 使用多任务学习
	- 改进算法：DAgger

# 2 Theory of why behavior cloning fails
首先做一些记号上的约定：记数据集的分布：$p_{data}(\boldsymbol{o}_t)$，模型实际操作时数据的分布：$p_{\pi_\theta}(\boldsymbol{o}_t)$。
基于我们对监督学习的理解，很显然训练的过程在
$$
\max_{\theta} \mathbb{E}_{\boldsymbol{o}_t \sim p_{data}(\boldsymbol{o}_t)} \left[\log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{o}_t)\right]
$$
问题的核心在于在 $p_{data}$ 上表现好的策略未必在 $p_{\pi_\theta}$ 上表现好。一个策略是好是坏不能仅仅通过
$$
\mathbb{E}_{\boldsymbol{o}_t \sim p_{data}(\boldsymbol{o}_t)} \left[\log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{o}_t)\right]
$$
来决定，相对的我们应该使用一个 cost 函数（注意这里我们混用了状态 $\boldsymbol{s}_t$ 与观测 $\boldsymbol{o}_t$）：
$$
c(\boldsymbol{s}_t,\boldsymbol{a}_t) = \begin{cases}  0,& \text{if } \boldsymbol{a}_t = \pi^\ast(\boldsymbol{s}_t) \\  1,& \text{otherwise} \end{cases}
$$
因为我们实际关心的是希望在 $p_{\pi_\theta}$ 上表现好，故目标是最小化
$$
\mathbb{E}_{\boldsymbol{s}_t \sim p_{\pi_\theta}(\boldsymbol{s}_t)} \left[c(\boldsymbol{s}_t，\boldsymbol{a}_t)\right]
$$
这里我们考虑决策的时间跨度为 $T$ 的情况，我们首先考虑一个简单的情况，假设
$$
\pi_\theta(\boldsymbol{a} \neq \pi^\ast(\boldsymbol{s}) \mid \boldsymbol{s}) \leq \epsilon,\forall \boldsymbol{s} \in \mathcal{D}_{train}
$$
这意味着我们的模型在训练集上的表现是很好的。此时我们可以得到如下的上界（先放缩到犯错概率始终为 $\epsilon$ 的情况）：
$$
\begin{aligned}  \mathbb{E}\left[\sum_t c(\boldsymbol{s}_t,\boldsymbol{a}_t)\right] &\leq \epsilon T + (1 - \epsilon) (\epsilon (T - 1) + (1 - \epsilon) (\epsilon (T - 2) + \ldots)) \\  &= O(\epsilon T^2)\\ \end{aligned}
$$
推导过程是相对直观的，每一个时间步我们分成是否犯错两种情况，一旦犯错则最多后续的时间步全部犯错。这里的结果意味着当时间跨度 $T$ 很大时，我们的 cost 会变得非常大。

基于对监督学习的认识，这样的情况并不普遍，因为监督学习中的目标不是仅在训练数据上表现好，而是在训练数据所在的分布上表现好。由于类似的推导过程在 RL 中也非常常用，我们做如下具体推导：
假设
$$
\pi_\theta(\boldsymbol{a} \neq \pi^\ast(\boldsymbol{s}) \mid \boldsymbol{s}) \leq \epsilon,\forall \boldsymbol{s} \sim p_{train}
$$
同样还是先把犯错概率放缩到 $\epsilon$，那么
$$
p_\theta(\boldsymbol{s}_t) = (1 - \epsilon)^t p_{\text{train}}(\boldsymbol{s}_t) + (1 - (1 - \epsilon)^t) p_{\text{mistake}}(\boldsymbol{s}_{t})
$$
这里的 $p_{\text{mistake}}$ 会是一个很复杂的分布，我们不去考虑它的具体形式。前一项代表着到 $t$ 时间步之前我们还没有犯错，后一项代表着我们在某一时间步犯错了。移项得到
$$
|p_\theta(\boldsymbol{s}_t) - p_{\text{train}}(\boldsymbol{s}_t)| \leq (1 - (1 - \epsilon)^t) |p_{\text{mistake}}(\boldsymbol{s}_{t}) - p_{\text{train}}(\boldsymbol{s}_{t})|
$$
这里的记号 $|\cdot|$ 与表示[[Concepts#2 总变差距离（Total Variation Distance）|总变差距离(Total Variation Distance)]]。利用总变差距离的性质可知 (如果是连续情况则将求和换成积分)：
$$
\sum_{\boldsymbol{s}_t} |p_{\text{mistake}}(\boldsymbol{s}_{t}) - p_{\text{train}}(\boldsymbol{s}_{t})| = 2 d_{TV}(p_{\text{mistake}}，p_{\text{train}}) \leq 2
$$
再利用 $(1 - \epsilon)^t \geq 1 - \epsilon t$，就可以得到
$$
\sum_{\boldsymbol{s}_t} |p_\theta(\boldsymbol{s}_t) - p_{\text{train}}(\boldsymbol{s}_t)| \leq 2 \epsilon t
$$
另一方面注意 $p_{\text{train}}(\boldsymbol{s}_t) c_t(\boldsymbol{s}_t)$ 乘积中如果前者为非 $0$ 才对求和有贡献，而此时说明 $\boldsymbol{s}_t$ 在 $p_{\text{train}}$ 内，故后者可以放缩到 $\epsilon$，而所有非零的前者求和 (积分) 为 $1$，于是
$$
\sum_{\boldsymbol{s}_t} p_{\text{train}}(\boldsymbol{s}_t) c_t(\boldsymbol{s}_t) \leq \epsilon.
$$
最后再考虑总的 cost：
$$
\begin{aligned}  
\sum_{t}\mathbb{E}_{p_\theta(\boldsymbol{s}_t)} \left[c_t\right] 
&= \sum_{t} \sum_{\boldsymbol{s}_t} p_\theta(\boldsymbol{s}_t) c_t(\boldsymbol{s}_t)\\  
&= \sum_{t} \sum_{\boldsymbol{s}_t} (p_{\text{train}}(\boldsymbol{s_t}) + p_\theta(\boldsymbol{s}_t) - p_{\text{train}}(\boldsymbol{s_t}) )c_t(\boldsymbol{s}_t)\\  
&\leq \sum_{t} \sum_{\boldsymbol{s}_t} p_{\text{train}}(\boldsymbol{s_t}) c_t(\boldsymbol{s}_t) + \sum_{t} \sum_{\boldsymbol{s}_t} |p_\theta(\boldsymbol{s}_t) - p_{\text{train}}(\boldsymbol{s_t})| c_{\text{max}}\\  
&\leq \sum_{t} \epsilon + \sum_{t} 2 \epsilon t\\  
&\leq \epsilon T + 2 \epsilon T^2 \\
& = O(\epsilon T^2) 
\end{aligned}
$$
这一种推导是比较悲观的，因为在现实中我们能够通过一些方式从错误中恢复过来，例如 driving 例子中的两个摄像头，但是并不是所有情况都能这样。从这一点来看，事实上训练数据中的错误越多一定程度上是越好的，因为这样我们能够更好地学习如何从错误中恢复。

# 3 How to address the problem

## 3.1 Be smart about data collection

行为克隆简单的地方在于，如果训练数据中有很多关于错误的数据，那么我们可以通过学习如何从错误中恢复来提高我们的模型。主要有以下两种方法（这两种方式其实在一定程度上是相同的）：
1.  人为引入错误，这有可能会损坏我们的模型，但是总的来说这些小的错误可以忽视，但能让模型学会如何从错误中恢复。
2.  使用数据加强，添加一些错误数据。  

接下来我们从几个例子出发展示一下上述提到的两种方法.
- **Case study 1：trail following** 在 trail following 的例子中，我们的目标是让无人机在 trail 间飞行，这里收集数据的方式类似于前面驾驶的例子，也使用了 3 个摄像头。详见：A Machine Learning Approach to Visual Perception of Forest Trails for Mobile Robots，2016
- **Case study 2：imitation with a cheap robot** 人为引入错误的例子，尽管有时候 robot 会由于人为引入的数据而出现错误，但是也能从这些错误中恢复过来。详见：Vision-Based Multi-Task Manipulation for Inexpensive Robots Using End-To-End Learning from Demonstration，2017
## 3.2 Use a powerful model
很显然如果我们能够让 $\epsilon$ 变的非常小，以至于很长的时间跨度 $T$ 都不足以让 cost 过大，那么在一定程度上是可以接受的。是哪些原因限制了我们更好地拟合专家呢？

**Non-Markovian behavior** 人类的行为可能是非马尔科夫的，当前决策通常不是仅仅基于当前的观测，而是可能有很长的”上下文“。
解决这一问题的方式是使用一些序列式的模型，也可以同时将连续多个观测作为输入。这样的方式引入了更多的信息，但可能产生一些问题 (**causal confusion**)，例如会导致模型误解相关性与因果关系。在引入多个观测前，事件发生的因果关系是清晰的，而多个时间的事件同时展现则可能带来混淆，例如模型可能认为车辆减速是因为踩刹车的原因，而不是因为前方有车。

![](https://picx.zhimg.com/v2-da155b386a34d8f1efc1e932295c657b_1440w.jpg)
**Multimodal behavior** 面对同一状态/观测，人类的行为可能是多样的，例如前方有一棵树，那么人类可能选择左转或右转，而如果我们使用了高斯分布来拟合这一行为，那么我们可能会得到一个平均值，也就是直行。
![](https://pic3.zhimg.com/v2-1a7d33e926c47787669735aef968dfb0_1440w.jpg)

这里的解决方法主要有两个，一个是**使用更加有表示能力的连续分布**（例如 mixture of Gaussians，latent variable models，diffusion models），另一种则是**进行离散化**。

使用更加有表示能力的连续分布：
- **混合高斯模型（mixture of Gaussians）**：用一系列的 $w_1，\mu_1，\Sigma_1，\ldots，w_k，\mu_k，\Sigma_k$ 来表示一个混合高斯分布，这样我们就可以表示多个模态。我们可以让模型输出这些参数，并且通过这些参数表达负对数似然。
- **隐变量模型（latent variable models）**：隐变量模型理论上可以表示任意的分布，只要模型足够复杂。一个例子是条件变分自编码器（Conditional VAE）。
- **扩散模型（diffusion models）**：扩散模型可以用于生成图像，应用在动作生成上，考虑 $T$ 步的扩散过程，考虑 $a_{t,0}$ 是真实的 action，$a_{t,i + 1}$ 是 $a_{t，i}$ + noise，类似地让模型从 $a_{t，i}$ 预测 $a_{t，i - 1}$ （实际是预测被添加的噪声）。于是就可以得到一个能够生成出动作的模型。  

离散化：
- 对于单一维度的动作，这样的离散化很简单，然而对于高维，动作空间会指数增加。解决这一问题的一个方法是自回归离散化（autoregressive discretization），使用一些序列式模型 (RNN，GPT)。我们假设要生成 $3$ 个维度的动作。
- 在训练时，我们先将观测编码输入序列式模型目标为 $a_{t,0}$，接下来输入真实的 $a_{t,0}$ 并以 $a_{t,1}$ 为目标，以此类推。在测试时，我们不输入真实动作，而是使用模型生成的该维度。
- 这为什么有效？在第 $1,2,3$ 步，我们依次尝试学习分布：$p(a_{t,0} \mid \boldsymbol{s}_t)$，$p(a_{t,1} \mid a_{t,0},\boldsymbol{s}_t)$，$p(a_{t,2} \mid a_{t,1}，a_{t,0},\boldsymbol{s}_t)$，以此类推. 如果这些分布都能有效学习到，那么我们就学习到了$$
p(\boldsymbol{a}_t \mid \boldsymbol{s}_t) = p(a_{t,0},a_{t,1},a_{t,2} \mid \boldsymbol{s}_t) = p(a_{t,0} \mid \boldsymbol{s}_t) p(a_{t,1} \mid a_{t,0},\boldsymbol{s}_t) p(a_{t,2} \mid a_{t,1},a_{t,0},\boldsymbol{s}_t)
$$![](https://pica.zhimg.com/v2-cde098b2a894b926dac8715cc26aae2a_1440w.jpg)
对于上述涉及到的技巧，我们用以下几个例子来展示：
- **Case study 3：imitation with diffusion models** 每一次利用扩散策略生成一个较小的轨迹。详见：Chi et al. Diffusion Policy：Visuomotor Policy Learning via Action Diffusion. 2023
- **Case study 4：imitation with latent variable** 训练时使用前面描述的条件变分自编码器，测试时使随机采样出 $z$ 来决定模式。详见：Zhao et al. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. 2023
- **Case study 5：imitation with Transformer RT-1**：读入文本指令以及历史观测，利用 Transformer 生成动作（转化为了 seq2seq 问题）。详见：Brohan et al. RT-1：Robotics Transformer. 2023

## 3.3 Multi-task learning
同时学习多个任务，可以让模仿学习更加容易。同样考虑驾驶的例子，我们考虑同时学习到达 $\boldsymbol{p}_1，\boldsymbol{p}_2，\ldots,\boldsymbol{p}_n$ 等多个地点，使用策略 $\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s},\boldsymbol{p})$。这样我们能够覆盖更多的数据，包括完成单个任务时不会遇到的情况。

**Goal-conditioned behavior cloning** 训练 $\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s},\boldsymbol{g})$
在训练时间：我们的每一个轨迹可以提供多个可能潜在欠佳的轨迹 (Expert Relabeling)，具体来说，对于轨迹 $\boldsymbol{s}_0，\boldsymbol{a}_0，\boldsymbol{s}_1，\boldsymbol{a}_1，\ldots，\boldsymbol{s}_T$， 我们可以尝试最大化：
$$
\log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t,\boldsymbol{g} = \boldsymbol{s}_{t + k})
$$
这在数据少的情况下是非常有用的。
详见：Goal-conditioned Imitation Learning，2019

**Example：Learning Latent Plans from Play**

![](https://pic2.zhimg.com/v2-5d1a83f54a61090ee95460a6a335a271_1440w.jpg)

![](https://pic1.zhimg.com/v2-feeb2600cba3e9247c41ae76bbf31b4e_1440w.jpg)

这一篇工作中的主要流程如下：
1.  收集数据：没有给出特定的目标，而是完成随机的操作，这样我们可以覆盖更多的情况，可以很好地避免超出分布的问题.  
2.  利用 Goal-conditioned behavior cloning 训练  
3.  完成一些特定的目标  


我们还可以考虑如下的方式：从随机策略出发，收集随机目标的数据，然后将这些数据作为对应任务的 表示，并用这些数据改进策略，不断地迭代。这样的一种方式可能已经超越了单纯的模仿学习。
![](https://pic4.zhimg.com/v2-0f2895848dd62134973948113dc0fe97_1440w.jpg)

详见：Learning to Reach Goals via Iterated Supervised Learning

Goal-conditioned BC 可以应用在大规模的数据上。可以使用多种机器人上的数据，甚至能够在未见过的机器上完成 (Navigation Task)。
![](https://pic2.zhimg.com/v2-0025b3df97cbc6ee2d9ee5fd947cde85_1440w.jpg)
详见：GNM：A General Navigation Model to Drive Any Robot

相关的工作还有 Hindsight Experience Replay，2017 但这里使用的是 offline RL，我们将会在之后涉及。

## 3.4 DAgger：Dataset Aggregation
我们前面问题的核心是分布偏移，也就是 $p_{data}(\boldsymbol{o}_t) \neq p_{\pi_\theta}(\boldsymbol{o}_t)$。我们能否让 $p_{data}(\boldsymbol{o}_t)$ 更接近 $p_{\pi_\theta}(\boldsymbol{o}_t)$ 呢？

目标：收集训练数据，使得 $p_{data}(\boldsymbol{o}_t) \approx p_{\pi_\theta}(\boldsymbol{o}_t)$
1.  训练 $\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{o}_t)$ 在人工数据集 $\mathcal{D}$ 上；
2.  运行 $\pi_\theta$ 获取数据集 $\mathcal{D}_\pi$；
3.  让人类标注 $\mathcal{D}_\pi$ 得到正确的动作；
4.  将 $\mathcal{D} \cup \mathcal{D}_\pi$ 作为新的数据集，重复 1-4。

可以证明最后两个分布将会收敛到一致。
然而这个方法的问题主要就在第 3 步，因为人类标注数据的方式可能与人类自然完成任务的方式不同，导致标注的数据很难达到很高质量。

# 4 Summary
模仿学习的问题是什么？
- 人需要提供大量的数据。
- 人并不擅长提供一些特定的动作，例如复杂的机械臂动作。
- 人可以从自己学习，机器能够实现这一点吗？