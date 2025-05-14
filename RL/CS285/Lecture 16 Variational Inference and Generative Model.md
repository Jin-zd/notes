Variational Inference 的应用非常广泛, 在 RL 中的应用例如我们已经讨论过的 **model-based RL** 中学习一个 **state space model**, **exploration** 中的 **VIME**, 以及之后将会介绍的 **inverse RL**.

## 1 Probabilistic [latent variable models](https://zhida.zhihu.com/search?content_id=255508224&content_type=Article&match_order=1&q=+latent+variable+models&zhida_source=entity)

### 1.1 Probabilistic models

对于随机变量 $x$, 我们可以用 $p(x)$ 表示其概率分布, 这也可以视作其的一个 **probabilistic model**. 类似地, 条件概率分布 $p(y\mid x)$ 也可以视作是一个 probabilistic model. 例如我们已经反复讨论过的 $\pi(\boldsymbol{a} \mid \boldsymbol{s})$.

从统计的角度, 在上述例子中,

-   $p(x)$ 中 $x$ 是 **query variable** (我们希望推断的变量), 没有 **evidence variable** (作为观测数据出现的变量).  
    
-   $p(y\mid x)$ 中 $x$ 是 evidence variable, $y$ 是 query variable.  
    

![](https://pic4.zhimg.com/v2-95c7930e437b5762f308cf7bfdd1f2cf_1440w.jpg)

probabilistic model

### 1.2 Latent variable models

**Definition 1**. _latent variable models_

_我们称一个 probabilistic model 为 **latent variable model**, 如果其包含了一个或多个 latent variables, 也就是既不是 query variable 也不是 evidence variable 的变量._

不难发现为了得到我们实际关心的 model, 我们需要将 latent variables 通过边缘化去除 (通过求和或积分).

**Example 1**.

-   _一个表示_ $p(x)$ _的 latent variable model 是 [mixture model](https://zhida.zhihu.com/search?content_id=255508224&content_type=Article&match_order=1&q=+mixture+model&zhida_source=entity), 例如 **Gaussian mixture model**. 也就是表示成_ $p(x) = \sum_{z} p(x \mid z) p(z)$_, 其中_ $p(x \mid z)$ _是高斯分布._

![](https://pic3.zhimg.com/v2-f881dfb3bdf09a8e671872f8eb04bf08_1440w.jpg)

mixture of Gaussian

-   _一个表示_ $p(y\mid x)$ _的 latent variable model 则可以表示为_ $p(y\mid x) = \sum_{z} p(y\mid x, z) p(z)$ _(假设_ $z$ _不依赖于_ $x$_), 例如 mixture of density model. 在 imitation learning 中, 我们简单介绍了如何利用这样的模型解决 multi-modality 的问题._  
    

### 1.3 Components of latent variable models

我们不妨拆解出一个 latent variable model 的主要组成 (以表示 $p(x)$ 为例):

-   我们关心的变量 $x$ 及其分布 $p(x)$, 通常这个分布是一个复杂的分布, 因此我们会想用一些方式来近似这个分布.  
    
-   latent variable $z$ 及其分布 $p(z)$, 通常这个分布是简单的  
    
-   我们会**学习一个带参数的条件分布** $p(x\mid z)$ 通常这个分布的**形式是简单的**, 但是其参数可能是一个输入 $z$ 的神经网络, 例如 $\mathcal{N}(\mu_{nn}(z), \sigma_{nn}(z))$.  
    

这是一种很有用的工具, 因为我们能够将一些难以表示的分布表示成一些简单的可以参数化的分布的组合. $p(x) = \int p(x\mid z) p(z) \text{d}z.\\$

![](https://pica.zhimg.com/v2-d262957837b93c106a6e2a4afaf9ffa0_1440w.jpg)

latent variable

**Example 2**. _类似的, 如果我们想要表示一个复杂的条件分布_ $p(y\mid x)$_, 我们可以使用_ $z \sim \mathcal{N}(0, I)$_, 我们学习一个形式简单的_ $p(y\mid x, z)$_, 最终得到一个复杂的_ $p(y\mid x)$ _(这其实就是 CVAE 的基本思想)._

![](https://pic2.zhimg.com/v2-6f08a5acbe139cc6ae9ba8024cfe4565_1440w.jpg)

latent space model 可以用来建模 multimodal 的行为

### Application

-   在 model-based RL 中, 我们提到为了处理复杂的 observation, 我们可以学习一个有结构的 **state space model**, 这在本节中会进一步讨论.  
    
-   在下一节中, 我们会讨论利用 **RL / control + variational inference** 的方式来建模人类的行为.  
    
-   我们也会使用 [generative models](https://zhida.zhihu.com/search?content_id=255508224&content_type=Article&match_order=1&q=+generative+models&zhida_source=entity)/ variational inference 来进行 **exploration**.  
    

一个值得注意的点是, generative models 不一定是 latent variable models, 但是通常情况下由于 generative models 需要表示一个复杂的分布, 因此使用 latent variable models 是一个自然的选择.

### 1.4 How to train latent variable models

首先是一些基本要素:

-   model: $p_\theta(x)$ (虽然我们可能显式地表示 $p_\theta(x\mid z)$, 但是实际有意义的是 $p_\theta(x)$)  
    
-   data: $\mathcal{D} = \{x_1,\ldots,x_N\}$  
    

我们训练这个 model 的目标是最大化 likelihood: $\theta^\ast = \arg\max_\theta \frac{1}{N}\sum_{i} \log p_\theta(x_i),\\$ 由于我们包含了 latent variable $z$, 我们的 MLE 会进一步转化为 $\theta^\ast = \arg\max_\theta \frac{1}{N}\sum_{i} \log \int p_\theta(x_i\mid z) p(z) \text{d}z.\\$ 然而中间的积分是 intractable 的, 即使 $z$ 是离散的, 这个问题通常也会极难优化.

**intuition:** 猜测最有可能的 $z$, 并计算其 log likelihood, 现实中由于有很多可能的 $z$, 所以我们考虑使用 $p(z\mid x_i)$ 的期望, 优化 expected log likelihood: $\theta^\ast = \arg\max_\theta \frac{1}{N}\sum_{i} \mathbb{E}_{z\sim p(z\mid x_i)} \left[\log p_\theta(x_i, z)\right].\\$ 我们能够进行这样的变换的原因, 以及我们处理这里的 $p(z\mid x_i)$ 的方法都基于接下来讨论的 **variational inference**.

## 2 Variational inference

这一节中, 我们介绍如何使用 **variational inference** 来最大化数据化的似然. 在正式介绍前, 我们先进行一些推导:

注意到对于单个样本 $x_i$, 我们的对数似然可以进行以下的转化: $\begin{aligned} \log p(x_i) &= \log \int_z p(x_i \mid z) p(z) \text{d}z\\ &= \log \int_z p(x_i \mid z) p(z) \frac{q_i(z)}{q_i(z)} \text{d}z\\ &= \log \mathbb{E}_{z \sim q_i(z)} \left[\frac{p(x_i \mid z) p(z)}{q_i(z)}\right]\\ &\geq \mathbb{E}_{z \sim q_i(z)} \left[\log \frac{p(x_i \mid z) p(z)}{q_i(z)}\right]\\ &= \mathbb{E}_{z \sim q_i(z)} \left[\log p(x_i \mid z) + \log p(z)\right] + \mathcal{H}(q_i), \end{aligned}\\$ 这样的转化对于任意的 $q_i(z)$ 都是成立的, 我们记最终得到的下界为 **evidence lower bound (ELBO)**: $\mathcal{L}_i(p, q_i) := \mathbb{E}_{z \sim q_i(z)} \left[\log p(x_i \mid z) + \log p(z)\right] + \mathcal{H}(q_i).\\$

### 2.1 Interpretation of ELBO

我们有以下的一些方式来理解 ELBO:

1.  由于 $\mathcal{L}_i(p, q_i) = \mathbb{E}_{z \sim q_i(z)} \left[\log p(x_i \mid z) + \log p(z)\right] + \mathcal{H}(q_i),\\$ 最大化第一项可以让我们在 $z$ 概率大的地方有很高的 $p(x_i, z)$, 第二项 entropy 的存在能够让我们对 $q_i(z)$ 的分布尽可能地宽.  
    
2.  从另一个角度来看, ELBO 可以写作 $\mathcal{L}_i(p, q_i) = \log p(x_i) - D_{KL}(q_i(z) \parallel p(z\mid x_i)),\\$ 这意味着, $q_i(z)$ 需要近似 $p(z\mid x_i)$, 这可以通过 KL 散度来实现. 由于 ELBO 是原始目标的下界, 这也告诉我们如果我们减小 KL 散度, 我们就可以让下界变得更紧.  
    

### 2.2 [EM algorithm](https://zhida.zhihu.com/search?content_id=255508224&content_type=Article&match_order=1&q=EM+algorithm&zhida_source=entity)

对于较为简单的 latent variable models, 例如 mixture of Gaussian, 我们可以直接让 $q_i(z)$ 等于精确的 $p(z \mid x_i)$, 这样第二项 KL divergence 就等于 $0$. 具体来说, 我们重复以下过程:

1.  更新 $q_i(z) \gets p(z \mid x_i) = p(z) p_\theta(x_i \mid z) \big/ \sum_{z} p(z) p_\theta(x_i \mid z)$  
    

利用如下方式更新 $\theta$: $\theta' \gets \arg\max_\theta \frac{1}{N}\sum_{i} \mathbb{E}_{z\sim p(z\mid x_i)} \left[\log p_\theta(x_i, z)\right].\\$

我们可以简单分析这样更新的单调性质: 不妨考虑第 $k$ 次更新后的参数为 $\theta_k$, 第 $k$ 次更新后的 $q_i(z)$ 记为 $q_i^k(z)$, 并且我们先更新 $\theta_k$, 再更新 $q_i^k$, 于是 $\log p_{\theta_k}(x_i) \geq \mathcal{L}_i(p_{\theta_k}, q_i^{k - 1}) \geq \mathcal{L}_i(p_{\theta_{k - 1}}, q_i^{k - 1}) = \log p_{\theta_{k - 1}}(x_i).\\$ 这里几个不等式的解释如下:

1.  由于我们总是先更新 $\theta_k$, 因此当我们更新 $\theta_{k - 1}$ 到 $\theta_k$ 时, $q_i$ 依然处在 $q_i^k$, 第一个不等号对应于 ELBO 的定义.  
    
2.  第二个不等号基于我们更新 $\theta_k$ 的过程等价于最大化 ELBO.  
    
3.  最后一个等号对应于我们将 $q_i^{k - 2}$ 更新到 $q_i^{k - 1}$ 时, 会让 $q_i^{k - 1}(z) \gets p(z \mid x_i)$, 此时 KL 散度为 $0$, 于是 $\mathcal{L}_i(p_{\theta_{k - 1}}, q_i^{k - 1}) = \log p_{\theta_{k - 1}}(x_i)$.  
    

基于单调性质以及似然函数存在上界, 基于分析学中的单调收敛原理可知 EM 算法会收敛于某个极值点.

### 2.3 General latent variable models

对于更加 general 的情形, 我们并不能简单地令 $q_i(z) = p(z\mid x_i)$, 但是我们可以利用一个 tractable 的分布 $q_i(z)$ 来近似 $p(z\mid x_i)$ , 这是 **variational inference** 的实质! 基于 ELBO 的第二种 interpretation 可以得到一个更加 general 的目标: **在最大化** $\mathcal{L}_i(p, q_i)$ **的同时通过更新** $q_i$ **最小化** $D_{KL}(q_i(z) \parallel p(z\mid x_i))$, 这可以写成 SGD 的形式:

-   对于每一个 $x_i$ (或 mini-batch), 我们计算 $\mathcal{L}_i(p, q_i)$  
    
-   采样 $z \sim q_i(z)$, 利用 $\nabla_\theta \mathcal{L}_i(p, q_i) = \nabla_\theta \mathbb{E}_{z \sim q_i(z)}\left[\log p_\theta(x_i \mid z)\right] \approx \nabla_\theta \log p_\theta(x_i \mid z)$ 来更新 $\theta$  
    
-   更新 $q_i$ 来最大化 $\mathcal{L}_i(p, q_i)$ (相当于是最小化 KL 散度)  
    

我们考虑最后一步如何进行, 不妨假设 $q_i(z) = \mathcal{N}(\mu_i, \sigma_i)$, 那么我们需要计算梯度 $\nabla_{\mu_i} \mathcal{L}_{i}(p, q_i), \nabla_{\sigma_i} \mathcal{L}_{i}(p, q_i).\\$ 然而此时我们需要的参数数量 $|\theta| + (|\mu_i| + |\sigma_i|) \times N$ 是非常大的 (注意每一个 sample 都需要对应于一个 $q_i$).

相比于学习一系列 $q_i$, 如果我们学习一个 $q(z\mid x_i)$ 的网络, 那么我们就似乎解决了这个问题, 通常我们会使用参数 $\phi$ 的神经网络 $q_\phi(z\mid x)$ 来表示. 也就是说, 我们会有一个 $p_\theta(x\mid z)$, 作为 **decoder**, 一个 $q_\phi(z\mid x)$ 作为 **encoder**. 这就是 **amortized variational inference** 背后的 idea.

![](https://pic4.zhimg.com/v2-4c7f2a8481bc3a65c3cc66f41ee2f505_1440w.jpg)

amortized variational inference

## 3 Amortized variational inference

### 3.1 Derivation

在之前的讨论中, 我们对数似然与 ELBO 的关系为: $\log p(x_i) \geq \mathbb{E}_{z \sim q_i(z)} \left[\log p(x_i \mid z) + \log p(z)\right] - \mathbb{E}_{z \sim q_i(z)} \left[\log q_i(z)\right],\\$ 转化为 amortized 形式就得到: $\log p(x_i) \geq \mathbb{E}_{z \sim q_\phi(z\mid x_i)} \left[\log p_\theta(x_i \mid z) + \log p(z)\right] + \mathcal{H}(q_\phi(z\mid x_i)).\\$

此时我们的 ELBO 可以写作: $\mathcal{L}_i(p_\theta, q_\phi) = \mathbb{E}_{z \sim q_\phi(z\mid x_i)} \left[\log p_\theta(x_i \mid z) + \log p(z)\right] + \mathcal{H}(q_\phi(z\mid x_i)).\\$ 这可以通过如下 SGD 的形式进行更新:

-   对于每一个 $x_i$ (或 mini-batch), 我们计算 $\mathcal{L}_i(p_\theta(x_i \mid z), q_\phi(z\mid x_i))$  
    
-   采样 $z \sim q_\phi(z\mid x_i)$  
    
-   计算 $\nabla_\theta \mathcal{L} \approx \nabla_\theta \log p_\theta(x_i \mid z)$  
    
-   $\theta \gets \theta + \alpha \nabla_\theta \mathcal{L}$  
    
-   $\phi \gets \phi + \alpha \nabla_\phi \mathcal{L}$  
    

### 3.2 Computing gradients

这里要考虑的是计算 $\nabla_\phi \mathcal{L}$. 我们依然考虑使用 $q_\phi(z\mid x_i) = \mathcal{N}(\mu_\phi(x_i), \sigma_\phi(x_i))$, 高斯分布的 entropy 有一个解析的形式: $\mathcal{H}(q_\phi(z\mid x_i)) = \frac{1}{2}\log \left((2\pi e)^d \det(\sigma_\phi(x_i))\right).\\$ 因此我们只需要考虑前一项 $\mathbb{E}_{z \sim q_\phi(z\mid x_i)} \left[\log p(x_i \mid z) + \log p(z)\right],\\$ 这里的困难在于 $z$ 同时出现在采样分布中, 也出现在期望中, 我们不妨记做 $J(\phi) = \mathbb{E}_{z \sim q_\phi(z\mid x_i)} \left[r(x_i, z)\right],\\$ 不难发现这其实和我们在 policy gradient 中的讨论是一样的, 回忆我们在 policy gradient 中的 trajectory 来自于 $\tau \sim p_\theta(\tau)$, 而 $\tau$ 也在期望中, 在 policy gradient 中我们使用的 trick 是: $\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[r(\tau)\right] = \int \nabla_\theta p_\theta(\tau)r(\tau) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\log p_\theta(\tau) r(\tau)\right].\\$

类似的我们可以估计 $\nabla J(\phi) \approx \frac{1}{M} \sum_j \nabla_\phi \log q_\phi(z_j \mid x_i) r(x_i, z_j),\\$这样的做法称作 **REINFORCE**.

然而这样的做法并不是最佳的选择, 尽管这一估计是无偏的, 但是这里的方差 $\text{Var}\left[\nabla_\phi \log q_\phi(z_j \mid x_i) r(x_i, z_j)\right] \geq \text{Var}\left[\nabla_\phi \log q_\phi(z_j \mid x_i)\right] \text{Var}\left[r(x_i, z_j)\right],\\$ 通常会很大, 因此我们需要生成很多样本.

然而值得注意的是, 这里的问题与 policy gradient 中有决定性的不同, 在 policy gradient 中, 我们的 $\tau \sim p_\theta(\tau)$ 是未知的, 我们必须要采样, 而这里的 $z \sim q_\phi(z\mid x_i)$ 是已知的, 而且通常还有一个解析的形式 (例如高斯分布). 我们可以使用更加有效的方法, 例如 **reparameterization trick**:

由于 $q_\phi(z\mid x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi(x))$, 我们可以将 $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$, 其中 $\epsilon \sim \mathcal{N}(0, I)$. 注意这里我们将 $\phi$ 从有随机性的 $z$ 中分离出来, 于是 $J(\phi) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} \left[r(x_i, \mu_\phi(x_i) + \sigma_\phi(x_i) \odot \epsilon)\right],\\$ 于是估计的方式是: 从 $\mathcal{N}(0, I)$ 中采样 $M$ 个 $\epsilon$ (实际上单个样本估计的就相当好) $\nabla_\phi J(\phi) \approx \frac{1}{M} \sum_{j} \nabla_\phi r(x_i, \mu_\phi(x_i) + \sigma_\phi(x_i) \odot \epsilon_j).\\$ 这与前面的估计相比方差会小很多.

**Note:** 我们可以进一步近似 latent variable model 的目标: $\begin{aligned} \mathcal{L}_i &= \mathbb{E}_{z \sim q_\phi(z\mid x_i)} \left[\log p_\theta(x_i \mid z)\right] - D_{KL}(q_\phi(z\mid x_i) \parallel p(z))\\ &= \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} \left[\log p_\theta(x_i \mid \mu_\phi(x_i) + \sigma_\phi(x_i) \odot \epsilon)\right] - D_{KL}(q_\phi(z\mid x_i) \parallel p(z))\\ &\approx \log p_\theta(x_i \mid \mu_\phi(x_i) + \sigma_\phi(x_i) \odot \epsilon) - D_{KL}(q_\phi(z\mid x_i) \parallel p(z)). \end{aligned} \\$ 再取负号就可以转化为 VAE 的 loss function, 前者对应于 reconstruction loss, 后者对应于 KL divergence.

![](https://pic3.zhimg.com/v2-963a2ff2e458bd5f96053b18e4cdbdca_1440w.jpg)

VAE 模型结构

### 3.3 Policy gradient vs reparameterization trick

我们最后对比一下 policy gradient (REINFORCE) 与 reparameterization trick 的区别:

-   **policy gradient (REINFORCE)**:  
    

-   可以处理离散或连续的 latent variables  
    
-   有很高的 variance, 需要使用很多样本以及更小的 learning rate  
    

-   **reparameterization trick**:  
    

-   只能处理连续的 latent variables  
    
-   容易实现, 且有很低的 variance  
    

## 4 Generative models: variational autoencoders

### 4.1 Variational autoencoders

首先我们介绍 **variational autoencoders (VAEs)**.

-   **encoder**: $q_\phi(z\mid x)$, 输入是 $x$, 输出一个高斯分布的参数 $\mu_\phi(x), \sigma_\phi(x)$.  
    
-   **decoder**: $p_\theta(x\mid z)$, 输入是 $z$, 输出是 $x$.  
    

**objective**: $\max_{\theta, \phi} \frac{1}{N} \sum_{i} \log p_\theta(x_i \mid \mu_\phi(x_i) + \sigma_\phi(x_i) \odot \epsilon) - D_{KL}(q_\phi(z\mid x_i) \parallel p(z)).\\$

### Usage of VAE

如果我们采样 $z \sim p(z)$, 那么通过 decoder $p_\theta(x\mid z)$ 我们就可以得到一个 $x \sim p(x)$.

![](https://picx.zhimg.com/v2-7222c70f3b0303ddb0b63b0a3a88d833_1440w.jpg)

从 VAE 中采样

为什么上述采样是有效的呢? 注意 loss 中的 KL 散度项会驱使输入的 $x$ 编码的$z$ 尽可能覆盖 hidden space, 以达到与 $p(z)$ 相同的 variance, 因此从 $p(z)$ 中采样的 $z$ 都会对应于数据分布中的一些点.

我们可以将 VAE 用于 RL 中, 这里我们依然考虑 **full observation** 的情况. 此时 VAE 作为一种 **representation learning** 的方法. 换言之我们将 $z$ 作为 $\boldsymbol{s}$ 的一种更加高效的表示.

**Example 3**. _考虑 Montezuma's Revenge, 我们可以将_ $z$ _作为一个表示, 这里的 intuition 是_ $z$ _会对应于一些有用的信息, 例如位置, 门的状态, 而不是每个像素具体的颜色以及完全无用的信息. 尽管现实中 VAE 未必一定能够完整地表示出所有的信息, 但但是它会尽可能地表示出有用的信息._

具体来说, 我们可以如此修改 Q-learning 的 sample algorithm:

-   收集 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}', r)$ 并加入 $\mathcal{R}$  
    
-   利用 $\mathcal{R}$ 更新 $p_\theta(\boldsymbol{s} \mid z)$ 与 $q_\phi(z\mid \boldsymbol{s})$  
    
-   利用 $\mathcal{R}$ 的数据经过 VAE 得到 $z, z'$, 更新 $Q(z,\boldsymbol{a})$.  
    

### 4.2 Conditional models

此时我们不再建模 $p(x)$, 而是建模 $p(y \mid x)$, 这里我们考虑 $\mathcal{L}_i = \mathbb{E}_{z \sim q_\phi(z\mid x_i, y_i)} \left[\log p_\theta(y_i \mid x_i, z) + \log p(z \mid x_i)\right] + \mathcal{H}(q_\phi(z\mid x_i, y_i)).\\$ **Note:** 通常情况下我们会使用 $p(z) = \mathcal{N}(0, I)$ 而不是 $p(z \mid x_i)$ 作为先验.

一个实用的例子是将其作为 policy, 具体来说, 我们的 $x$ 是 observation, $y$ 是 action, $z \sim \mathcal{N}(0, I)$. 具体来说,

-   encoder 输入是 $x_i, y_i$, 输出是 $\mu_\phi(x_i, y_i), \sigma_\phi(x_i, y_i)$.  
    
-   decoder 输入是 $x_i, z$, 输出是 $y_i$.

![](https://picx.zhimg.com/v2-078724344852b8c6b944d3877ca9b98d_1440w.jpg)

CVAE 结构

**Usage of CVAE:** 在使用时, 我们从 $p(z)$ (或 $p(z \mid x_i)$) 中采样 $z$, 然后使用 $p(y_i \mid x_i, z)$ 得到 $y_i$.

**Example 4**. _这样的模型通常用于 imitation learning. 因为在 imitation learning 中, 我们可能需要建模一些 multimodal 的行为._

_参见: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware._

### 4.3 State space models

在 partial observation 的情况下, 我们通常会把整个 $\boldsymbol{z}_1, \ldots, \boldsymbol{z}_T$ 作为 hidden state $z$, 把观测序列 $\boldsymbol{o}_1, \ldots, \boldsymbol{o}_T$ 作为 $x$.

我们从以下几个角度来介绍 state space models 中的一些设计:

### Prior:

我们希望 prior 不再是常规 VAE 中那样的各维度相互独立, 而是具有隐含的 dynamics, 也就是 $p(\boldsymbol{z}) = p(\boldsymbol{z}_1) \prod_{t} p(\boldsymbol{z}_{t + 1} \mid \boldsymbol{z}_t, \boldsymbol{a}_t).\\$ 我们可以使用 $p(\boldsymbol{z}_1) = \mathcal{N}(0, I)$, 而 $p(\boldsymbol{z}_{t + 1} \mid \boldsymbol{z}_t, \boldsymbol{a}_t)$ 则是学习得到的.

### Decoder:

我们希望 decoder 处理各时间步的观测是独立的, 也就是 $p(\boldsymbol{o} \mid \boldsymbol{z}) = \prod_{t} p(\boldsymbol{o}_t \mid \boldsymbol{z}_t).\\$

![](https://pica.zhimg.com/v2-af40613e95a75fac759e1a6ad5c168ac_1440w.jpg)

state space model 的 decoder

### Encoder:

我们通常不能假设各时间步的信息是独立的, 因为单个观测实际上不足以表示整个 hidden state. 我们在 model-based RL 中讨论过了多种选择, 其中一个是 $q_\phi(\boldsymbol{z} \mid \boldsymbol{o}) = \prod_{t} q_\phi(\boldsymbol{z}_t \mid \boldsymbol{o}_{1:t}).\\$ 我们通常可以利用 LSTM, transformer 等序列式模型来表示 $q_\phi(\boldsymbol{z}_t \mid \boldsymbol{o}_{1:t})$.

![](https://pic1.zhimg.com/v2-250b43191cd8abac34eeebd42b1d1814_1440w.jpg)

state space model 中的 encoder

### 4.4 Applications

一些 state space model 的实际应用:

### Representation learning and model-based RL

-   学习一个 state space model, 并在 latent space 进行 planning, 参见:

-   Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images.  
    
-   SOLAR: Deep Structured Representations for Model-Based Reinforcement Learning.  
    
-   Learning Latent Dynamics for Planning from Pixels.  
    

-   学习一个 state space model, 并在 latent space 运行 RL 算法, 参见:

-   Stochastic Latent Actor-Critic: Deep Reinforcement Learning with a Latent Variable Model.
-   Dream to Control: Learning Behaviors by Latent Imagination.