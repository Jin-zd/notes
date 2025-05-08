# 1 Derivation of Policy Gradient
策略梯度（Policy Gradient）算法是一种直接对策略进行优化的方法，也就是直接对策略的参数 $\theta$ 进行优化。
策略梯度是一种无模型方法（model-free methods），在这种方法中，我们通常不假设知道 $p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$ 以及 $p(\boldsymbol{s}_1)$，也不尝试学习关于环境的模型，而是通过环境的交互（sampling）来获取关于环境的信息。

记目标函数为 
$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[\sum_{t} r(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]
$$
由于我们不知道 $p_\theta(\tau)$ 中关于环境的概率，故无法直接计算这个期望，但是可以通过采样进行估计，也就是
$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[\sum_{t} r(\boldsymbol{s}_t, \boldsymbol{a}_t)\right] \approx \frac{1}{N} \sum_{i} \sum_{t} r(\boldsymbol{s}_{i,t}, \boldsymbol{a}_{i,t})
$$
但实际上相比目标函数的具体值，我们更关心的是梯度。为了记号简便，我们记 $r(\tau) = \sum_{t} r(\boldsymbol{s}_t, \boldsymbol{a}_t)$，于是我们有
$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[r(\tau)\right] = \int p_\theta(\tau) r(\tau) \text{d}\tau
$$
从而梯度可以写作
$$
\nabla_\theta J(\theta) = \int \nabla_\theta p_\theta(\tau) r(\tau) \text{d}\tau
$$
这个式子存在多个难以处理的点。首先是 $p_\theta(\tau)$ 本身依赖于环境的状态转移，其次则是表达式并非是一个期望，给我们的估计带来困难，这里有一个常见的技巧是
$$
p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) = p_\theta(\tau) \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} = \nabla_\theta p_\theta(\tau)
$$
于是可以进一步化简为
$$
\nabla_\theta J(\theta) = \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) r(\tau) \text{d}\tau = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[\nabla_\theta \log p_\theta(\tau) r(\tau)\right]
$$
我们得到了理想的期望形式，此时就可以通过采样来估计梯度。但第一个问题尚未解决，也就是 $p_\theta(\tau)$ 的计算问题。不难注意到，$\log p_\theta(\tau)$ 可以写作
$$
\log p_\theta(\tau) = \log p(\boldsymbol{s}_1) + \sum_{t} \log p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t) + \sum_{t} \log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)
$$
其中与环境有关的部分都与 $\theta$ 无关，故取关于 $\theta$ 的梯度是
$$
\nabla_\theta \log p_\theta(\tau) = \sum_{t} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)
$$
整理即可得到了直接策略微分的形式
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[\left(\sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)\right) \left(\sum_{t = 1}^T r(\boldsymbol{s}_t, \boldsymbol{a}_t)\right)\right]
$$
用样本来估计就写作
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i = 1}^N \left(\sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t})\right) \left(\sum_{t = 1}^T r(\boldsymbol{s}_{i,t}, \boldsymbol{a}_{i,t})\right)
$$
以上即是 **REINFORCE** 算法，包含了以下几个步骤：
1.  利用 $\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)$ 采样 $\{\tau^i\}$  
2.  计算 $\nabla_\theta J(\theta) = \frac{1}{N} \sum_{i} \left(\sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t})\right) \left(\sum_{t = 1}^T r(\boldsymbol{s}_{i,t}, \boldsymbol{a}_{i,t})\right)$  
3.  更新 $\theta \gets \theta + \alpha \nabla_\theta J(\theta)$  

# 2 Understanding Policy Gradient
不难发现，策略梯度的形式与最大似然估计（MLE）中的梯度非常相似。
策略梯度：
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i = 1}^{N} \left(\sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t})\right) \left(\sum_{t = 1}^T r(\boldsymbol{s}_{i,t}, \boldsymbol{a}_{i,t})\right)
$$
最大似然：
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i = 1}^{N} \left(\sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t})\right)
$$
不难理解，最大似然只会最大化出现的轨迹的概率，而策略梯度在整体上会增大奖励高的轨迹的概率，减小奖励低的轨迹的概率，这可以理解为是试错的正式化。

## 2.1 Example: Gaussian policy
在连续的动作空间 $\mathcal{A}$ 中，我们可以通过预先设定的策略的形式来将神经网络输出的结果转化为一个动作的概率分布：考虑我们的策略是一个高斯分布，也就是 $\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t) = \mathcal{N}(f_{nn}(\boldsymbol{s}_t), \Sigma)$，其中均值由神经网络 $f_{nn}$ 给出，方差是一个固定的值 $\Sigma$。那么我们可以给出似然的形式：
$$
\log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t) = - \frac{1}{2} (\boldsymbol{a}_t - f_{nn}(\boldsymbol{s}_t))^\top \Sigma^{-1} (\boldsymbol{a}_t - f_{nn}(\boldsymbol{s}_t)) + const
$$
从而梯度
$$
\nabla_\theta \log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t) = -\Sigma^{-1} (\boldsymbol{a}_t - f_{nn}(\boldsymbol{s}_t)) \nabla_\theta f_{nn}(\boldsymbol{s}_t)
$$

## 2.2 Partial observability
考虑部分可观测的情况，我们不妨依然考虑 $\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{o}_t)$，即当前的动作依赖于当前的观测。不妨记整条轨迹为 $\tau_{\boldsymbol{o}}$，那么利用链式法则，我们可以得到
$$
\begin{aligned}  \log p(\tau_{\boldsymbol{o}}) &= \log p(\boldsymbol{o}_1) + \log p(\boldsymbol{a}_1 \mid \boldsymbol{o}_1) + \log p(\boldsymbol{o}_2 \mid \boldsymbol{o}_1, \boldsymbol{a}_1) + \log p(\boldsymbol{a}_2 \mid \boldsymbol{o}_{1:2}, \boldsymbol{a}_1) + \cdots\\  &= \log p(\boldsymbol{o}_1) + \sum_{t} \log p(\boldsymbol{a}_t \mid \boldsymbol{o}_{1:t}, \boldsymbol{a}_{1:t - 1}) + \sum_{t} \log p(\boldsymbol{o}_{t + 1} \mid \boldsymbol{o}_{1:t}, \boldsymbol{a}_{1:t})\\  &= \log p(\boldsymbol{o}_1) + \sum_{t} \log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{o}_t) + \sum_{t} \log p(\boldsymbol{o}_{t + 1} \mid \boldsymbol{o}_{1:t}, \boldsymbol{a}_{1:t}), \end{aligned}
$$
由于无论环境转移是否依赖于过去的观测，即无论观测是否满足马尔可夫性质，我们都可以得到
$$
\nabla_\theta \log p(\tau_{\boldsymbol{o}}) = \sum_{t} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{o}_t)
$$
于是就有
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i = 1}^N \left(\sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{o}_{i,t})\right) \left(\sum_{t = 1}^T r(\boldsymbol{o}_{i,t}, \boldsymbol{a}_{i,t})\right)
$$
当然, 一个可能的设计选项是由于我们的当前观测无法完全反映环境的状态，我们可以使用 $\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{o}_{1:t})$ 等形式的策略，并使用一些序列式的网络结构来融合过去的观测。

## 2.3 Problems in policy gradient
我们可以看到，策略梯度的形式是
$$
\nabla_\theta J(\theta) = \frac{1}{N} \sum_{i = 1}^{N} \nabla_\theta \log p_\theta(\tau) r(\tau)
$$
这一形式其实蕴含了一个很大的问题，也就是方差。不妨考虑如下情况：如果奖励都是很大的值，那么梯度就会很大，极易受采样结果的影响，造成非常大的方差。我们接下来介绍的很多改进都是为了解决这个问题。
![](https://pica.zhimg.com/v2-bd01f61cbf420cbb4bd7713e1f64a122_1440w.jpg)
# 3 Reducing Variance
## 3.1 Causality
我们可以基于causality这种普适的性质来减小方差。

**Definition 1**. _Causality_
$t'$ 时刻的 policy 不应该影响 $t$ 时刻的奖励，如果 $t < t'$。

与马尔可夫性质不同之处在于，只要我们时间流向是单向的，那么就可以保证 causality。
利用 causality 可以将我们的策略梯度进行如下改写:
$$
\begin{aligned}  \nabla_\theta J(\theta) &\approx \frac{1}{N} \sum_{i = 1}^{N} \nabla_\theta \left(\sum_{t = 1}^T \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t})\right) \left(\sum_{t = 1}^T r(\boldsymbol{s}_{i,t}, \boldsymbol{a}_{i,t})\right)\\  &= \frac{1}{N} \sum_{i = 1}^{N} \sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t}) \left(\sum_{t' = 1}^T r(\boldsymbol{s}_{i,t'}, \boldsymbol{a}_{i,t'})\right)\\  &= \frac{1}{N} \sum_{i = 1}^{N} \sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t}) \left(\sum_{t' = t}^T r(\boldsymbol{s}_{i,t'}, \boldsymbol{a}_{i,t'})\right).\\ \end{aligned}
$$
由于此时我们估计的值整体减小了，故这一操作可以减小方差。在应用 causality 后，我们用 $\hat{Q}_{i, t} = \sum_{t' = t}^T r(\boldsymbol{s}_{i,t'}, \boldsymbol{a}_{i,t'})$ 表示未来的奖励。

如果要从数学上证明 causality 转化的正确性，注意到
$$
\begin{aligned}  
&\mathbb{E}_{\tau\sim p_\theta(\tau)} \left[\nabla_\theta \log\pi_\theta(\boldsymbol{a}_t|\boldsymbol{s}_t) \cdot \left(\sum_{0 < j < t}r(\boldsymbol{s}_j,\boldsymbol{a}_j)\right)\right]\\  
&= \mathbb{E}_{\boldsymbol{s}_{1:t}, \boldsymbol{a}_{1:t-1}}\left[\mathbb{E}_{\boldsymbol{s}_{t + 1:T}, \boldsymbol{a}_{t:T}}\left[\nabla_\theta\log\pi_\theta(\boldsymbol{a}_t|\boldsymbol{s}_t)|\boldsymbol{s}_1, \boldsymbol{a}_1,\ldots,\boldsymbol{s}_{t-1},\boldsymbol{a}_{t-1},\boldsymbol{s}_t\right]\cdot\left(\sum_{0 < j < t} r(\boldsymbol{s}_j,\boldsymbol{a}_j)\right)\right]\\  
&= \mathbb{E}_{\boldsymbol{s}_{1:t}, \boldsymbol{a}_{1:t-1}}\left[\mathbb{E}_{\boldsymbol{a}_t \sim \pi_\theta(\boldsymbol{a}_t|\boldsymbol{s}_t)}\left[\nabla_\theta\log\pi_\theta(\boldsymbol{a}_t|\boldsymbol{s}_t)\right]\cdot\left(\sum_{0 < j < t} r(\boldsymbol{s}_j,\boldsymbol{a}_j)\right)\right]\\  
&= 0
\end{aligned}
$$
其中最后的等号利用了 $\mathbb{E}_{\boldsymbol{a}_t \sim \pi_\theta(\boldsymbol{a}_t|\boldsymbol{s}_t)}\left[\nabla_\theta\log\pi_\theta(\boldsymbol{a}_t|\boldsymbol{s}_t)\right] = \nabla_\theta \int \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)\text{d}\boldsymbol{a}_t = \nabla_\theta 1 = 0$。

## 3.2 Baselines
另一个常见的方法是引入 baseline，直觉上引入 $b = \frac{1}{N} \sum_{i = 1}^{N} r(\tau)$，这样我们就可以让奖励的期望接近 $0$，故每次更新时只有优于平均的奖励才会增大概率，从而减小了方差。在引入 baseline 后，得到的梯度为
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i = 1}^{N} \nabla_\theta \log p_\theta(\tau) \left(r(\tau) - b\right)
$$
此时我们对 $\nabla_\theta J(\theta)$ 的估计依然是无偏的，只需观察到
$$
\begin{aligned}  \mathbb{E}\left[\nabla_\theta \log p_\theta(\tau) b\right] &= \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) b \text{d}\tau\\  &= b \int \nabla_\theta p_\theta(\tau) \text{d}\tau\\  &= b \nabla_\theta \int p_\theta(\tau) \text{d}\tau\\  &= b \nabla_\theta 1 = 0. \end{aligned}
$$
事实上平均奖励并不是最好的 baseline，但是已经足够好了。接下来我们分析一下什么是最好的 baseline。有：
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[\nabla_\theta \log p_\theta(\tau) \left(r(\tau) - b\right)\right]
$$
于是
$$
\text{var}= \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[\left(\nabla_\theta \log p_\theta(\tau) \left(r(\tau) - b\right)\right)^2\right] - \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[\nabla_\theta \log p_\theta(\tau) \left(r(\tau) - b\right)\right]^2
$$
其中第二项我们已经在分析期望时证明其与 $b$ 的取值无关，于是我们只需要分析第一项。这里记 $g(\tau) = \nabla_\theta \log p_\theta(\tau)$，于是
$$
\begin{aligned}  \frac{\partial \text{var}}{\partial b} &= \frac{\partial}{\partial b} \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[g(\tau)^2 \left(r(\tau) - b\right)^2\right]\\   &= \frac{\partial}{\partial b} \left(\mathbb{E}\left[g(\tau)^2 r(\tau)^2\right] - \mathbb{E}\left[2 g(\tau)^2 r(\tau) b\right] + \mathbb{E}\left[g(\tau)^2 b^2\right]\right)\\  &= -2 \mathbb{E}\left[g(\tau)^2 r(\tau)\right] + 2 b \mathbb{E}\left[g(\tau)^2\right] = 0. \end{aligned}
$$
于是解得
$$
b = \frac{\mathbb{E}\left[g(\tau)^2 r(\tau)\right]}{\mathbb{E}\left[g(\tau)^2\right]}
$$
此时的 baseline 是一个奖励的加权平均，其中权值还依赖于策略的梯度。我们通常不使用这个最优 baseline，而是使用平均奖励。

# 4 Off-policy Policy Gradient
我们的策略梯度算法是 **on-policy** 的，因为我们的梯度
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[\nabla_\theta \log p_\theta(\tau) r(\tau)\right]
$$
中的期望是基于当前的策略的，因此每一次更新都需要重新采样。这样的方式对于 Deep RL 来说是不理想的，因为我们每一次梯度更新策略的改变量很小，而每次数据的采样通常都是很昂贵且耗时的。
这里我们介绍一种 **off-policy** 的方法，也就是**重要性采样（importance sampling）**。

**Definition 2**. _importance sampling（重要性采样）_
给定两个分布 $p(x)$ _与_ $q(x)$，如果我们只有 $q(x)$ 分布下的样本，而我们想要计算 $p(x)$ 下的期望，那么我们可以通过重要性采样来计算：
$$
\begin{aligned}  \mathbb{E}_{x \sim p(x)}\left[f(x)\right] &= \int p(x) f(x) \text{d}x\\  &= \int \frac{p(x)}{q(x)} q(x) f(x) \text{d}x\\  &= \mathbb{E}_{x \sim q(x)}\left[\frac{p(x)}{q(x)} f(x)\right].  \end{aligned}
$$
这里假设有样本的分布是 $p_\theta(\tau)$，而当前的策略参数是 $\theta'$, 对应轨迹分布是 $p_{\theta'}(\tau)$，那么目标函数可以写作
$$
J(\theta') = \mathbb{E}_{\tau \sim p_{\theta}(\tau)} \left[\frac{p_{\theta'}(\tau)}{p_\theta(\tau)} r(\tau)\right]
$$
接下来表达出期望中的这个比例，注意到由于初始分布与状态转移分布都是相互抵消的，我们有
$$
\frac{p_{\theta'}(\tau)}{p_\theta(\tau)} = \prod_{t = 1}^T \frac{\pi_{\theta'}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}{\pi_{\theta}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}
$$
然而我们更关心的是梯度
$$
\nabla_{\theta'} J(\theta') = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[\frac{p_{\theta'}(\tau)}{p_\theta(\tau)} \nabla_{\theta'} \log p_{\theta'}(\tau) r(\tau)\right]
$$
整理一下，off-policy 的策略梯度为
$$
\begin{aligned}  
&\nabla_{\theta'} J(\theta') \\
&= \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[\prod_{t = 1}^T \frac{\pi_{\theta'}(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}{\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)}\left(\sum_{t = 1}^T \nabla_{\theta'} \log \pi_{\theta'}(\boldsymbol{a}_t \mid \boldsymbol{s}_t) \right) \left(\sum_{t = 1}^T r(\boldsymbol{s}_t, \boldsymbol{a}_t)\right)\right]\\  
&= \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[\sum_{t = 1}^{T} \nabla_{\theta'} \log \pi_{\theta'}(\boldsymbol{a}_t \mid \boldsymbol{s}_t) \left(\prod_{t' = 1}^{t} \frac{\pi_{\theta'}(\boldsymbol{a}_{t'} \mid \boldsymbol{s}_{t'})}{\pi_\theta(\boldsymbol{a}_{t'} \mid \boldsymbol{s}_{t'})}\right) \left(\sum_{t' = t}^T r(\boldsymbol{s}_{t'}, \boldsymbol{a}_{t'}) \left(\prod_{t'' = t}^{t'} \frac{\pi_{\theta'}(\boldsymbol{a}_{t''} \mid \boldsymbol{s}_{t''})}{\pi_\theta(\boldsymbol{a}_{t''} \mid \boldsymbol{s}_{t''})}\right)\right)\right]
\end{aligned}
$$
这里后一个等号基于 causality，即未来的重要性采样比率不应该影响过去的奖励。

如果我们忽略掉后一个重要性采样比率，那么我们就得到了 **策略迭代算法（Policy Iteration Algorithm）** 的形式（先用当前采样得到的样本估计累积奖励，相当于策略评估，然后基于这个累积奖励加上当前动作对应的梯度调整策略参数，相当于策略改进）：
$$
\mathbb{E}_{\tau \sim p_\theta(\tau)} \left[\sum_{t = 1}^{T} \nabla_{\theta'} \log \pi_{\theta'}(\boldsymbol{a}_t \mid \boldsymbol{s}_t) \left(\prod_{t' = 1}^{t} \frac{\pi_{\theta'}(\boldsymbol{a}_{t'} \mid \boldsymbol{s}_{t'})}{\pi_\theta(\boldsymbol{a}_{t'} \mid \boldsymbol{s}_{t'})}\right) \left(\sum_{t' = t}^T r(\boldsymbol{s}_{t'}, \boldsymbol{a}_{t'}) \right)\right]
$$
可以证明，虽然这个式子不再是梯度，但是依然可以改善策略。

## 4.1 A first-order approximation for IS

这里的问题是， $\left(\prod_{t' = 1}^{t} \frac{\pi_{\theta'}(\boldsymbol{a}_{t'} \mid \boldsymbol{s}_{t'})}{\pi_\theta(\boldsymbol{a}_{t'} \mid \boldsymbol{s}_{t'})}\right)$ 一项是 exponential in $T$ 的，可能造成很大的方差。我们考虑把原先 $\boldsymbol{s}, \boldsymbol{a}$ 视作一个增强状态的方式进行重写：在 on-policy 中，我们有
$$
\nabla_{\theta'} J(\theta') \approx \frac{1}{N} \sum_{i = 1}^N \sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t}) \hat{Q}_{i,t}
$$
我们这里考虑不再从整个轨迹采样，而是在 $\pi_\theta(\boldsymbol{s}_t, \boldsymbol{a}_t)$ 这个边缘分布上采样，即 $(\boldsymbol{s}_{i,t}, \boldsymbol{a}_{i,t}) \sim \pi_\theta(\boldsymbol{s}_t, \boldsymbol{a}_t)$，故在 off-policy 中，我们有
$$
\begin{aligned}  \nabla_{\theta'} J(\theta') &\approx \frac{1}{N} \sum_{i = 1}^N \sum_{t = 1}^T \frac{\pi_{\theta'}(\boldsymbol{s}_{i,t'}, \boldsymbol{a}_{i,t'})}{\pi_\theta(\boldsymbol{s}_{i,t'}, \boldsymbol{a}_{i,t'})} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t}) \hat{Q}_{i,t}\\  &= \frac{1}{N} \sum_{i = 1}^N \sum_{t = 1}^T \frac{\pi_{\theta'}(\boldsymbol{s}_{i,t'})}{\pi_\theta(\boldsymbol{s}_{i,t'})} \frac{\pi_{\theta'}(\boldsymbol{a}_{t'} \mid \boldsymbol{s}_{t'})}{\pi_\theta(\boldsymbol{a}_{t'} \mid \boldsymbol{s}_{t'})} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t}) \hat{Q}_{i,t},\\ \end{aligned}
$$
这里化简式中的第一种写法是无法计算的，因为在我们不知道状态转移和初始分布的情况下，是无法写出关于状态-动作的边缘分布的。但重写得到的第二种写法中，如果忽略关于状态的边缘分布，就得到
$$
\frac{1}{N} \sum_{i = 1}^N \sum_{t = 1}^T \frac{\pi_{\theta'}(\boldsymbol{a}_{t'} \mid \boldsymbol{s}_{t'})}{\pi_\theta(\boldsymbol{a}_{t'} \mid \boldsymbol{s}_{t'})} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t}) \hat{Q}_{i,t}
$$
得到了一个可以计算的形式，相较于原先 exponential in $T$ 的形式，我们这里相当于只保留了对应时间 $t$ 的重要性采样比率。在 **Advanced Policy Gradient** 一节中我们会知道为什么这是合理的。

# 5 Policy Gradient Summary

在本节中, 我们:
-   介绍了 policy gradient 的基本概念, 给出了 policy gradient 的形式及推导过程.  
-   分析了 partial observability 的情况, 提出了 policy gradient 中 variance 过大的问题.  
-   介绍了减小 variance 的方法, 包括 causality 和 baseline.  
-   介绍了 off-policy 的 policy gradient.  
    

# 6 Introduction to Advanced Policy Gradients

本小节中, 我们初步介绍一些 policy gradient 的进阶算法及其相关概念, 为后续一节 **Advanced Policy Gradient** 做准备.

## 6.1 Policy gradient with constraints

首先考虑常规梯度下降 $\theta \gets \theta + \alpha \nabla_\theta J(\theta)$ 的问题. 这里的 $\alpha$ 是我们的学习率, 不难发现, 如果我们的学习率很小, 则我们需要很长时间收敛, 或者可能陷入局部极值, 如果学习率很大, 则有可能因为更新幅度过大而错过最优解.

一个直观的解决方案是, 我们对更新的幅度进行一个限制, 考虑

$$
\theta' \gets \arg\max_{\theta'} (\theta' - \theta)^T \nabla_\theta J(\theta) \text{ s.t. } \|\theta' - \theta\|^2 \leq \epsilon
$$
我们在参数空间中划分了一个球, 各个参数在参数空间中的步长是相等的.

上述约束问题可以通过拉格朗日乘子法解得, 我们构造一个拉格朗日函数

$$
\mathcal{L}(\theta', \lambda) = (\theta' - \theta)^T \nabla_\theta J(\theta) - \lambda (\|\theta' - \theta\|^2 - \epsilon)
$$
我们直接应用 KKT (Karush-Kuhn-Tucker) 条件, 可以得到问题的解为

$$
\theta' = \theta + \frac{\sqrt{\epsilon}}{\|\nabla_\theta J(\theta)\|} \nabla_\theta J(\theta)
$$
换言之这里使用的是动态学习率 $\alpha = \sqrt{\epsilon} / \|\nabla_\theta J(\theta)\|$.

作为一个解决约束优化问题的例子, 也为了直观展现 KKT condition 的含义, 我们给出如下利用拉格朗日对偶的过程 (如果你很熟悉这个过程直接跳过即可):

_Proof._ **Step 1: 将原问题转化为带约束的 minimax 问题** 不难发现原问题 $(P)$: $\max_{\theta'} (\theta' - \theta)^T \nabla_\theta J(\theta) \text{ s.t. } \|\theta' - \theta\|^2 \leq \epsilon$ 等价于

$$
\max_{\theta'} \min_{\lambda, \lambda \geq 0} \mathcal{L}(\theta', \lambda)
$$

**Step 2: 使用 Sion's Minimax Theorem** 由于 $\mathcal{L}(\theta', \lambda)$ 对 $\theta'$ 是凹函数, 对 $\lambda$ 是凸函数 (线性也是凸函数), 利用 Sion's Minimax Theorem 我们有

$$
\max_{\theta'} \min_{\lambda, \lambda \geq 0} \mathcal{L}(\theta', \lambda) = \min_{\lambda, \lambda \geq 0} \max_{\theta'} \mathcal{L}(\theta', \lambda)
$$
且这两个问题的解是等价的.

**Step 3: 求解内层极值得到对偶问题** 再考虑带参数的最值问题 $\max_{\theta'} \mathcal{L}(\theta', \lambda)$, 这里对 $\theta'$ 求偏导并令其为 $0$ 可得

$$
\nabla_\theta J(\theta) - 2\lambda(\theta' - \theta) = 0 \Rightarrow \theta' = \theta + \frac{1}{2\lambda} \nabla_\theta J(\theta)
$$
于是得到了对偶问题 $(D)$,

$$
\min_{\lambda, \lambda \geq 0} \mathcal{L}\left(\theta + \frac{1}{2\lambda} \nabla_\theta J(\theta), \lambda\right)
$$

**Step 4: 求解对偶问题** 代入可得这个对偶问题的解为 $\lambda = \|\nabla_\theta J(\theta)\| / 2\sqrt{\epsilon}$, 代入得到

$$
\theta' = \theta + \frac{\sqrt{\epsilon}}{\|\nabla_\theta J(\theta)\|} \nabla_\theta J(\theta)
$$

## 6.2 Natural Policy Gradient

上述的方法是在参数空间中进行的, 隐含着一些问题, 不妨考虑如下的例子:

考虑一维的状态空间和动作空间, 令 $r(\boldsymbol{s}_t, \boldsymbol{a}_t) = -\boldsymbol{s}_t^2 - \boldsymbol{a}_t^2$. 并且考虑高斯策略, 即

$$
\pi_{\theta}(\boldsymbol{a}_t \mid \boldsymbol{s}_t) \sim \mathcal{N}(k \boldsymbol{s}, \sigma)
$$

![](https://pic2.zhimg.com/v2-db24422e19c8dbe9c0c57299e7eb7ed1_1440w.jpg)

故可以得到

$$
\log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t) = -\frac{1}{2\sigma^2} (\boldsymbol{a}_t - k \boldsymbol{s}_t)^2 + const
$$
我们考虑参数 $\theta = (k, \sigma)$, 不难发现,

$$
\frac{\partial J(\theta)}{\partial \sigma} \sim \sigma^{-3}, \frac{\partial J(\theta)}{\partial k} \sim \sigma^{-2}
$$
当 $\sigma$ 接近 $0$ 时, 梯度沿着 $\sigma$ 方向的分量远大于沿着 $k$ 的分量, 因此收敛到最优解的速度会非常慢. 这可以类比到优化问题中的 condition number, 也就是这个优化问题是 poor conditioned (条件数非常大).

![](https://pic4.zhimg.com/v2-6259597dccd6407b9697dd58f67d830f_1440w.jpg)

使用常规 policy gradient 的问题

![](https://pic4.zhimg.com/v2-9d5c4617f85d5f6694522cab9bd23817_1440w.jpg)

类比优化问题

实质上, 这一问题的核心在于, 我们的梯度下降是在参数空间中进行的, 而参数空间中不同参数的相同变化对 policy 的影响是不同的. 在前面的例子中, $\sigma$ 接近 $0$ 时, 梯度的绝大多数分量都在 $\sigma$ 方向, 也就是说 $\sigma$ 的变化对 policy 的影响远大于 $k$ 的变化的影响.

我们希望每个参数的步长在对 policy 的意义下相等, 而不是在参数空间中相等. 也就是我们不是先在参数空间画出一个范围, 而是先依据 policy (概率分布) 的空间中画出一个范围, 然后再在这个范围内优化 $\theta'$. 这里我们需要一种新的距离度量, 这种距离度量不应受到参数化的影响, 一个很好的选择是 KL divergence.

我们这里可以得到一个新的约束优化问题
$$
\theta' \gets \arg\max_{\theta'} (\theta' - \theta)^T \nabla_\theta J(\theta) \text{ s.t. }D_{KL}(\pi_{\theta'} \parallel \pi_\theta) \leq \epsilon
$$
然而, 直接求解这个问题是非常困难的: 虽然 $D_{KL}$ 是一个凸函数, 但是这是对于分布 $\pi_{\theta'}$ 而言而非 $\theta'$ 而言, 这使得我们难以像处理前面的约束问题那样直接求解. 这里我们可以考虑一个近似的方法, 利用 Taylor 展开, 我们有
$$
D_{KL}(\pi_{\theta'} \parallel \pi_\theta) \approx \frac{1}{2} (\theta' - \theta)^T F(\theta) (\theta' - \theta)
$$
这里 $F(\theta)$ 为 Fisher 信息矩阵, 此时我们近似地得到如下的约束优化问题
$$
\theta' \gets \arg\max (\theta' - \theta)^T \nabla_\theta J(\theta) \text{ s.t. } \|\theta' - \theta\|_{F(\theta)}^2 \leq \epsilon
$$
我们可以通过拉格朗日乘子法求解, 定义
$$
\mathcal{L}(\theta', \lambda) = (\theta' - \theta)^T \nabla_\theta J(\theta) - \lambda \left(\frac{1}{2}(\theta' - \theta)^T F(\theta) (\theta' - \theta) - \epsilon\right)
$$
基于 KKT condition, 考虑
$$
\begin{cases}  \frac{\partial \mathcal{L}}{\partial \theta'} = \nabla_\theta J(\theta) - \lambda F(\theta) (\theta' - \theta) = 0\\  \frac{\partial \mathcal{L}}{\partial \lambda} = \frac{1}{2}(\theta' - \theta)^T F(\theta) (\theta' - \theta) - \epsilon = 0 \end{cases}
$$
于是解得
$$
\theta' = \theta + \alpha F^{-1}(\theta) \nabla_\theta J(\theta)
$$
其中动态学习率为
$$
\alpha = \sqrt{\frac{2\epsilon}{(\nabla_\theta J(\theta))^T F^{-1}(\theta) \nabla_\theta J(\theta)}}
$$
于是我们可以给出 **natural gradient** 算法的基本流程:
1.  利用 $\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)$ 采样一系列轨迹 $\mathcal{D}$  
2.  估计 advantage function $\hat{A}(\boldsymbol{s}_t, \boldsymbol{a}_t)$  
3.  计算 policy gradient $\nabla_\theta J(\theta) = \frac{1}{N} \sum_{i = 1}^N \sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t}) \hat{A}(\boldsymbol{s}_{i,t}, \boldsymbol{a}_{i,t})$  
4.  利用估计 $F(\theta) \approx \frac{1}{N} \sum_{i = 1}^N \sum_{t = 1}^T \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t}) \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i,t} \mid \boldsymbol{s}_{i,t})^T$ 计算 Fisher 信息矩阵 $F(\theta)$.  
5.  利用 $\theta' = \theta + \alpha F^{-1}(\theta) \nabla_\theta J(\theta)$ 更新参数, 其中 $\alpha$ 是动态计算的学习率.  

上述算法中我们尚未覆盖的是 $F(\theta)$ 的计算方式, 这里基于的是
$$
F(\theta) = \mathbb{E}_{\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})} \left[\nabla_\theta \log \pi_\theta(\boldsymbol{a} \mid \boldsymbol{s}) \nabla_\theta \log \pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})^T\right]
$$
我们用样本的均值来估计这个期望就可以得到 $F(\theta)$ 的估计.

![](https://picx.zhimg.com/v2-6900826967f76aac58d7fdd5aa87d86d_1440w.jpg)

使用 Natural policy gradient 的效果
**Remark:**
-   我们基于 policy 的分布来定义了一个新的距离度量, 在这一度量的近似约束下进行优化.  
-   我们引入了动态的学习率.  
-   我们的 Fisher 信息矩阵使用了二阶近似, 含有一些误差, 也可能导致我们的约束条件不被满足, 因此并不能保证我们的更新是最优的.  
-   动态学习率的计算是基于 Fisher 信息矩阵的逆矩阵, 这个计算复杂度是 $O(n^3)$ 的, 相对昂贵.  

在 **Advanced Policy Gradient** 一节中, 我们将会介绍一些更加高级的 policy gradient 算法, 进一步改进 natural gradient 的算法.  