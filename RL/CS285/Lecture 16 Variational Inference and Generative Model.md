变分推断的应用非常广泛，在强化学习中的应用例如我们已经讨论过的基于模型的强化学习中学习一个状态空间模型，探索中的变分信息最大化探索（VIME），以及之后将会介绍的逆强化学习。

## 1 Probabilistic latent variable models
### 1.1 Probabilistic models
对于随机变量 $x$，可以用 $p(x)$ 表示其概率分布，这也可以视作其的一个概率模型。类似地，条件概率分布 $p(y\mid x)$ 也可以视作是一个概率模型，例如我们已经反复讨论过的 $\pi(\boldsymbol{a} \mid \boldsymbol{s})$。

从统计的角度，在上述例子中：
- $p(x)$ 中 $x$ 是查询变量（希望推断的变量），没有证据变量（作为观测数据出现的变量）。
- $p(y\mid x)$ 中 $x$ 是证据变量，$y$ 是查询变量。

![](16-1.png)

### 1.2 Latent variable models
Definition 1. _latent variable models（隐变量模型）_
称一个概率模型为隐变量模型，如果其包含了一个或多个隐变量，也就是既不是查询变量也不是证据变量的变量。

不难发现为了得到实际关心的模型，我们需要将隐变量通过边缘化去除（通过求和或积分）。

例如：
- 一个表示 $p(x)$ 的隐变量模型是混合模型，例如高斯混合模型，也就是表示成$p(x) = \sum_{z} p(x \mid z) p(z)$，其中 $p(x \mid z)$ 是高斯分布。
![](16-2.png)

- 一个表示 $p(y\mid x)$ 的隐变量模型则可以表示为 $p(y\mid x) = \sum_{z} p(y\mid x, z) p(z)$ （假设 $z$ 不依赖于 $x$），例如混合密度模型。在[[Lecture 1 Imitation Learning]]中，我们简单介绍了如何利用这样的模型解决多模态的问题。 

### 1.3 Components of latent variable models
不妨拆解出一个隐变量模型的主要组成（以表示 $p(x)$ 为例）：
- 我们关心的变量 $x$ 及其分布 $p(x)$，通常这个分布是一个复杂的分布，因此会想用一些方式来近似这个分布；
- 隐变量 $z$ 及其分布 $p(z)$，通常这个分布是简单的；
- 我们会学习一个带参数的条件分布 $p(x\mid z)$，通常这个分布的形式是简单的，但是其参数可能是一个输入 $z$ 的神经网络，例如 $\mathcal{N}(\mu_{nn}(z), \sigma_{nn}(z))$。

这是一种很有用的工具，因为我们能够将一些难以表示的分布表示成一些简单的可以参数化的分布的组合：
$$
p(x) = \int p(x\mid z) p(z) \text{d}z
$$
![](16-3.png)


例如，类似的，如果想要表示一个复杂的条件分布 $p(y\mid x)$，可以使用 $z \sim \mathcal{N}(0, I)$，学习一个形式简单的 $p(y\mid x, z)$，最终得到一个复杂的 $p(y\mid x)$（这其实就是 CVAE 的基本思想）。

![](16-4.png)
应用：
- 在基于模型的强化学习中，我们提到为了处理复杂的观测，可以学习一个有结构的状态空间模型，这在本节中会进一步讨论；
- 在下一节中，我们会讨论利用强化学习 / 控制 + 变分推断的方式来建模人类的行为；
- 我们也会使用生成模型/ 变分推断来进行探索。

一个值得注意的点是，生成模型不一定是隐变量模型，但是通常情况下由于生成模型需要表示一个复杂的分布，因此使用隐变量模型是一个自然的选择。

### 1.4 How to train latent variable models
首先是一些基本要素：
- 模型：$p_\theta(x)$ （虽然我们可能显式地表示 $p_\theta(x\mid z)$，但是实际有意义的是 $p_\theta(x)$）；
- 数据：$\mathcal{D} = \{x_1,\ldots,x_N\}$。

训练这个模型的目标是最大化似然：
$$
\theta^\ast = \arg\max_\theta \frac{1}{N}\sum_{i} \log p_\theta(x_i)
$$
由于包含了隐变量 $z$，最大似然会进一步转化为
$$
\theta^\ast = \arg\max_\theta \frac{1}{N}\sum_{i} \log \int p_\theta(x_i\mid z) p(z) \text{d}z
$$
然而中间的积分是棘手的，即使 $z$ 是离散的，这个问题通常也会极难优化。

直觉：猜测最有可能的 $z$，并计算其对数似然，现实中由于有很多可能的 $z$，所以考虑使用 $p(z\mid x_i)$ 的期望，优化期望对数似然：
$$
\theta^\ast = \arg\max_\theta \frac{1}{N}\sum_{i} \mathbb{E}_{z\sim p(z\mid x_i)} \left[\log p_\theta(x_i, z)\right]
$$
能够进行这样的变换的原因，以及处理这里的 $p(z\mid x_i)$ 的方法都基于接下来讨论的变分推断。

## 2 Variational inference
这一节中，我们介绍如何使用变分推断来最大化数据化的似然。在正式介绍前，先进行一些推导。

注意到对于单个样本 $x_i$，对数似然可以进行以下的转化：
$$
\begin{aligned} \log p(x_i) &= \log \int_z p(x_i \mid z) p(z) \text{d}z\\ &= \log \int_z p(x_i \mid z) p(z) \frac{q_i(z)}{q_i(z)} \text{d}z\\ &= \log \mathbb{E}_{z \sim q_i(z)} \left[\frac{p(x_i \mid z) p(z)}{q_i(z)}\right]\\ &\geq \mathbb{E}_{z \sim q_i(z)} \left[\log \frac{p(x_i \mid z) p(z)}{q_i(z)}\right]\\ &= \mathbb{E}_{z \sim q_i(z)} \left[\log p(x_i \mid z) + \log p(z)\right] + \mathcal{H}(q_i), \end{aligned}\\
$$
这样的转化对于任意的 $q_i(z)$ 都是成立的，记最终得到的下界为证据下界（ Evidence lower bound，ELBO）：
$$
\mathcal{L}_i(p, q_i) := \mathbb{E}_{z \sim q_i(z)} \left[\log p(x_i \mid z) + \log p(z)\right] + \mathcal{H}(q_i)
$$

### 2.1 Interpretation of ELBO
有以下的一些方式来理解证据下界：
1. 由于$$\mathcal{L}_i(p, q_i) = \mathbb{E}_{z \sim q_i(z)} \left[\log p(x_i \mid z) + \log p(z)\right] + \mathcal{H}(q_i)$$最大化第一项可以让我们在 $z$ 概率大的地方有很高的 $p(x_i, z)$，第二项熵的存在能够让我们对 $q_i(z)$ 的分布尽可能地宽。
2. 从另一个角度来看，证据下界可以写作$$\mathcal{L}_i(p, q_i) = \log p(x_i) - D_{KL}(q_i(z) \parallel p(z\mid x_i))$$这意味着, $q_i(z)$ 需要近似 $p(z\mid x_i)$，这可以通过 KL 散度来实现。由于证据下界是原始目标的下界，这也告诉我们如果减小 KL 散度，就可以让下界变得更紧。

### 2.2 EM algorithm
对于较为简单的隐变量模型，例如混合高斯模型，可以直接让 $q_i(z)$ 等于精确的 $p(z \mid x_i)$，这样第二项 KL 散度就等于 $0$。具体来说，重复以下过程（期望最大化算法，EM）：
1. 更新$$q_i(z) \gets p(z \mid x_i) = p(z) p_\theta(x_i \mid z) \big/ \sum_{z} p(z) p_\theta(x_i \mid z)$$
2. 利用如下方式更新 $\theta$：$$\theta' \gets \arg\max_\theta \frac{1}{N}\sum_{i} \mathbb{E}_{z\sim p(z\mid x_i)} \left[\log p_\theta(x_i, z)\right]$$

可以简单分析这样更新的单调性质：不妨考虑第 $k$ 次更新后的参数为 $\theta_k$，第 $k$ 次更新后的 $q_i(z)$ 记为 $q_i^k(z)$，并且先更新 $\theta_k$，再更新 $q_i^k$，于是
$$
\log p_{\theta_k}(x_i) \geq \mathcal{L}_i(p_{\theta_k}, q_i^{k - 1}) \geq \mathcal{L}_i(p_{\theta_{k - 1}}, q_i^{k - 1}) = \log p_{\theta_{k - 1}}(x_i)
$$
这里几个不等式的解释如下：
1. 由于总是先更新 $\theta_k$，因此当更新 $\theta_{k - 1}$ 到 $\theta_k$ 时，$q_i$ 依然处在 $q_i^k$，第一个不等号对应于证据下界的定义；
2. 第二个不等号基于更新 $\theta_k$ 的过程等价于最大化证据下界。
3. 最后一个等号对应于将 $q_i^{k - 2}$ 更新到 $q_i^{k - 1}$ 时，会让 $q_i^{k - 1}(z) \gets p(z \mid x_i)$，此时 KL 散度为 $0$，于是 $\mathcal{L}_i(p_{\theta_{k - 1}}, q_i^{k - 1}) = \log p_{\theta_{k - 1}}(x_i)$。

基于单调性质以及似然函数存在上界，基于分析学中的单调收敛原理可知期望最大化算法会收敛于某个极值点。

### 2.3 General latent variable models
对于更加普遍的情形，并不能简单地令 $q_i(z) = p(z\mid x_i)$，但是可以利用一个易处理的分布 $q_i(z)$ 来近似 $p(z\mid x_i)$，这是变分推断的实质。基于证据下界的第二种解释可以得到一个更加普遍的目标：在最大化 $\mathcal{L}_i(p, q_i)$ 的同时通过更新 $q_i$ 最小化 $D_{KL}(q_i(z) \parallel p(z\mid x_i))$，这可以写成随机梯度下降的形式：
- 对于每一个 $x_i$（或小批次），计算 $\mathcal{L}_i(p, q_i)$；
- 采样 $z \sim q_i(z)$，利用 $\nabla_\theta \mathcal{L}_i(p, q_i) = \nabla_\theta \mathbb{E}_{z \sim q_i(z)}\left[\log p_\theta(x_i \mid z)\right] \approx \nabla_\theta \log p_\theta(x_i \mid z)$ 来更新 $\theta$；
- 更新 $q_i$ 来最大化 $\mathcal{L}_i(p, q_i)$（相当于是最小化 KL 散度）。

考虑最后一步如何进行，不妨假设 $q_i(z) = \mathcal{N}(\mu_i, \sigma_i)$，那么需要计算梯度
$$
\nabla_{\mu_i} \mathcal{L}_{i}(p, q_i), \nabla_{\sigma_i} \mathcal{L}_{i}(p, q_i)
$$
然而此时需要的参数数量 $|\theta| + (|\mu_i| + |\sigma_i|) \times N$ 是非常大的（注意每一个样本都需要对应于一个 $q_i$）。

相比于学习一系列 $q_i$，如果学习一个 $q(z\mid x_i)$ 的网络，那么就似乎解决了这个问题。通常会使用参数 $\phi$ 的神经网络 $q_\phi(z\mid x)$ 来表示，也就是说，会有一个 $p_\theta(x\mid z)$，作为解码器，一个 $q_\phi(z\mid x)$ 作为编码器，这就是摊销变分推断背后的想法。
![](16-5.png)


## 3 Amortized variational inference
### 3.1 Derivation
在之前的讨论中，对数似然与证据下界的关系为：
$$
\log p(x_i) \geq \mathbb{E}_{z \sim q_i(z)} \left[\log p(x_i \mid z) + \log p(z)\right] - \mathbb{E}_{z \sim q_i(z)} \left[\log q_i(z)\right]
$$
转化为摊销形式就得到：
$$
\log p(x_i) \geq \mathbb{E}_{z \sim q_\phi(z\mid x_i)} \left[\log p_\theta(x_i \mid z) + \log p(z)\right] + \mathcal{H}(q_\phi(z\mid x_i))
$$
此时证据下界可以写作：
$$
\mathcal{L}_i(p_\theta, q_\phi) = \mathbb{E}_{z \sim q_\phi(z\mid x_i)} \left[\log p_\theta(x_i \mid z) + \log p(z)\right] + \mathcal{H}(q_\phi(z\mid x_i))
$$
这可以通过如下随机梯度下降的形式进行更新：
- 对于每一个 $x_i$（或小批次），计算 $\mathcal{L}_i(p_\theta(x_i \mid z), q_\phi(z\mid x_i))$；
- 采样 $z \sim q_\phi(z\mid x_i)$；
- 计算 $\nabla_\theta \mathcal{L} \approx \nabla_\theta \log p_\theta(x_i \mid z)$；
- $\theta \gets \theta + \alpha \nabla_\theta \mathcal{L}$；
- $\phi \gets \phi + \alpha \nabla_\phi \mathcal{L}$。

### 3.2 Computing gradients
这里要考虑的是计算 $\nabla_\phi \mathcal{L}$，依然考虑使用
$$
q_\phi(z\mid x_i) = \mathcal{N}(\mu_\phi(x_i), \sigma_\phi(x_i))
$$
高斯分布的熵有一个解析的形式：
$$
\mathcal{H}(q_\phi(z\mid x_i)) = \frac{1}{2}\log \left((2\pi e)^d \det(\sigma_\phi(x_i))\right)
$$
因此只需要考虑前一项 
$$
\mathbb{E}_{z \sim q_\phi(z\mid x_i)} \left[\log p(x_i \mid z) + \log p(z)\right]
$$
这里的困难在于 $z$ 同时出现在采样分布中，也出现在期望中，不妨记做
$$
J(\phi) = \mathbb{E}_{z \sim q_\phi(z\mid x_i)} \left[r(x_i, z)\right]
$$
不难发现这其实和[[Lecture 3 Policy Gradients]]中的讨论是一样的，回忆在策略梯度中的轨迹来自于 $\tau \sim p_\theta(\tau)$，而 $\tau$ 也在期望中，在策略梯度中我们使用的技巧是：
$$
\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[r(\tau)\right] = \int \nabla_\theta p_\theta(\tau)r(\tau) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\log p_\theta(\tau) r(\tau)\right]
$$
类似的我们可以估计
$$
\nabla J(\phi) \approx \frac{1}{M} \sum_j \nabla_\phi \log q_\phi(z_j \mid x_i) r(x_i, z_j)
$$
这样的做法称作 REINFORCE。

然而这样的做法并不是最佳的选择，尽管这一估计是无偏的，但是这里的方差
$$
\text{Var}\left[\nabla_\phi \log q_\phi(z_j \mid x_i) r(x_i, z_j)\right] \geq \text{Var}\left[\nabla_\phi \log q_\phi(z_j \mid x_i)\right] \text{Var}\left[r(x_i, z_j)\right]
$$
通常会很大，因此需要生成很多样本。

然而值得注意的是，这里的问题与策略梯度中有决定性的不同，在策略梯度中，$\tau \sim p_\theta(\tau)$ 是未知的，必须要采样，而这里的 $z \sim q_\phi(z\mid x_i)$ 是已知的，而且通常还有一个解析的形式（例如高斯分布），可以使用更加有效的方法，例如重参数化技巧。

由于
$$
q_\phi(z\mid x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi(x))
$$
可以将
$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon
$$
其中 $\epsilon \sim \mathcal{N}(0, I)$。注意这里将 $\phi$ 从有随机性的 $z$ 中分离出来，于是
$$
J(\phi) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} \left[r(x_i, \mu_\phi(x_i) + \sigma_\phi(x_i) \odot \epsilon)\right]
$$
于是估计的方式是，从 $\mathcal{N}(0, I)$ 中采样 $M$ 个 $\epsilon$ （实际上单个样本估计的就相当好）：
$$
\nabla_\phi J(\phi) \approx \frac{1}{M} \sum_{j} \nabla_\phi r(x_i, \mu_\phi(x_i) + \sigma_\phi(x_i) \odot \epsilon_j)
$$
这与前面的估计相比方差会小很多。

可以进一步近似隐变量模型的目标：
$$
\begin{aligned} \mathcal{L}_i &= \mathbb{E}_{z \sim q_\phi(z\mid x_i)} \left[\log p_\theta(x_i \mid z)\right] - D_{KL}(q_\phi(z\mid x_i) \parallel p(z))\\ &= \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} \left[\log p_\theta(x_i \mid \mu_\phi(x_i) + \sigma_\phi(x_i) \odot \epsilon)\right] - D_{KL}(q_\phi(z\mid x_i) \parallel p(z))\\ &\approx \log p_\theta(x_i \mid \mu_\phi(x_i) + \sigma_\phi(x_i) \odot \epsilon) - D_{KL}(q_\phi(z\mid x_i) \parallel p(z)). \end{aligned}
$$
再取负号就可以转化为[[Concepts#18 变分自编码器 (Variational Autoencoder, VAE)|变分自编码器 (Variational Autoencoder, VAE)]]的损失函数，前者对应于重构损失，后者对应于 KL 散度。
![](16-6.png)

### 3.3 Policy gradient vs reparameterization trick
最后对比一下策略梯度（REINFORCE）与重参数化技巧的区别：
- 策略梯度（REINFORCE）：  
	- 可以处理离散或连续的隐变量；
	- 有很高的方差，需要使用很多样本以及更小的学习率。
- 重参数化技巧：  
	- 只能处理连续的隐变量；  
	- 容易实现，且有很低的方差。 

## 4 Generative models: variational autoencoders
### 4.1 Variational autoencoders
首先介绍[[Concepts#18 变分自编码器 (Variational Autoencoder, VAE)|变分自编码器 (Variational Autoencoder, VAE)]]。
- 编码器：$q_\phi(z\mid x)$，输入是 $x$，输出一个高斯分布的参数 $\mu_\phi(x), \sigma_\phi(x)$；
- 解码器：$p_\theta(x\mid z)$，输入是 $z$，输出是 $x$。

目标：
$$
\max_{\theta, \phi} \frac{1}{N} \sum_{i} \log p_\theta(x_i \mid \mu_\phi(x_i) + \sigma_\phi(x_i) \odot \epsilon) - D_{KL}(q_\phi(z\mid x_i) \parallel p(z))
$$

变分自编码器的用法：
如果采样 $z \sim p(z)$，那么通过解码器 $p_\theta(x\mid z)$ 就可以得到一个 $x \sim p(x)$。
![](16-7.png)
为什么上述采样是有效的呢？注意损失中的 KL 散度项会驱使输入的 $x$ 编码的 $z$  尽可能覆盖隐藏状态，以达到与 $p(z)$ 相同的方差，因此从 $p(z)$ 中采样的 $z$ 都会对应于数据分布中的一些点。

可以将变分自编码器用于强化学习中。这里我们依然考虑完全可观测的情况，此时变分自编码器作为一种表征学习的方法，换言之我们将 $z$ 作为 $\boldsymbol{s}$ 的一种更加高效的表示。

考虑 Montezuma's Revenge，我们可以将 $z$ 作为一个表示，这里的直觉是 $z$ 会对应于一些有用的信息，例如位置，门的状态，而不是每个像素具体的颜色以及完全无用的信息。尽管现实中变分自编码器未必一定能够完整地表示出所有的信息，但是它会尽可能地表示出有用的信息。

具体来说，可以如此修改 Q 学习的采样算法：
- 收集 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}', r)$ 并加入 $\mathcal{R}$；
- 利用 $\mathcal{R}$ 更新 $p_\theta(\boldsymbol{s} \mid z)$ 与 $q_\phi(z\mid \boldsymbol{s})$；
- 利用 $\mathcal{R}$ 的数据经过 变分自编码器得到 $z, z'$，更新 $Q(z,\boldsymbol{a})$。

### 4.2 Conditional models
下面介绍条件变分自编码器：
此时不再建模 $p(x)$，而是建模 $p(y \mid x)$，这里考虑
$$
\mathcal{L}_i = \mathbb{E}_{z \sim q_\phi(z\mid x_i, y_i)} \left[\log p_\theta(y_i \mid x_i, z) + \log p(z \mid x_i)\right] + \mathcal{H}(q_\phi(z\mid x_i, y_i))
$$
通常情况下会使用 $p(z) = \mathcal{N}(0, I)$ 而不是 $p(z \mid x_i)$ 作为先验。

一个实用的例子是将其作为策略，具体来说，$x$ 是观测，$y$ 是动作，$z \sim \mathcal{N}(0, I)$。具体来说：
- 编码器输入是 $x_i, y_i$，输出是 $\mu_\phi(x_i, y_i), \sigma_\phi(x_i, y_i)$；
- 解码器输入是 $x_i, z$，输出是 $y_i$。

![](16-8.png)

条件变分自编码器的使用：在使用时，从 $p(z)$ (或 $p(z \mid x_i)$) 中采样 $z$，然后用 $p(y_i \mid x_i, z)$ 得到 $y_i$。
这样的模型通常用于模仿学习，因为在模仿学习中，我们可能需要建模一些多模态的行为。

参见：Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.

### 4.3 State space models
在部分可观测的情况下，通常会把整个 $\boldsymbol{z}_1, \ldots, \boldsymbol{z}_T$ 作为隐藏状态 $z$，把观测序列 $\boldsymbol{o}_1, \ldots, \boldsymbol{o}_T$ 作为 $x$。

从以下几个角度来介绍状态空间模型中的一些设计：
- 先验：我们希望先验不再是常规变分自编码器中那样的各维度相互独立，而是具有隐含的动态，也就是$$p(\boldsymbol{z}) = p(\boldsymbol{z}_1) \prod_{t} p(\boldsymbol{z}_{t + 1} \mid \boldsymbol{z}_t, \boldsymbol{a}_t)$$可以使用 $p(\boldsymbol{z}_1) = \mathcal{N}(0, I)$，而 $p(\boldsymbol{z}_{t + 1} \mid \boldsymbol{z}_t, \boldsymbol{a}_t)$ 则是学习得到的。
- 解码器：我们希望解码器处理各时间步的观测是独立的，也就是$$p(\boldsymbol{o} \mid \boldsymbol{z}) = \prod_{t} p(\boldsymbol{o}_t \mid \boldsymbol{z}_t)$$
![](16-9.png)
- 编码器：我们通常不能假设各时间步的信息是独立的，因为单个观测实际上不足以表示整个隐藏状态。在基于模型的强化学习中讨论过了多种选择，其中一个是$$q_\phi(\boldsymbol{z} \mid \boldsymbol{o}) = \prod_{t} q_\phi(\boldsymbol{z}_t \mid \boldsymbol{o}_{1:t})$$通常可以利用 LSTM，Transformer 等序列模型来表示 $q_\phi(\boldsymbol{z}_t \mid \boldsymbol{o}_{1:t})$。
![](16-10.png)

### 4.4 Applications
一些状态空间模型的实际应用：
- 学习一个状态空间模型，并在潜在空间进行规划，参见：
	- Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images.  
	- SOLAR: Deep Structured Representations for Model-Based Reinforcement Learning.  
	- Learning Latent Dynamics for Planning from Pixels.  
- 学习一个状态空间模型，并在潜在空间运行强化学习算法，参见：
	- Stochastic Latent Actor-Critic: Deep Reinforcement Learning with a Latent Variable Model.
	- Dream to Control: Learning Behaviors by Latent Imagination.