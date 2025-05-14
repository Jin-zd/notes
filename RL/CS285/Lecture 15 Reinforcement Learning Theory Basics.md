在本节中, 我们将简单介绍 [Reinforcement Learning Theory](https://zhida.zhihu.com/search?content_id=255437491&content_type=Article&match_order=1&q=Reinforcement+Learning+Theory&zhida_source=entity) 是什么样的, 例如其关注的问题, 通常的假设, 以及我们对 RL theory 的正确期望. 在此基础上我们会介绍一些基本的工具, 以及 [model-based RL](https://zhida.zhihu.com/search?content_id=255437491&content_type=Article&match_order=1&q=model-based+RL&zhida_source=entity), [model-free RL](https://zhida.zhihu.com/search?content_id=255437491&content_type=Article&match_order=1&q=model-free+RL&zhida_source=entity) 在一系列假设下可以获得的一些理论结果.

## 1 Introduction

### 1.1 Problems we asks in RL theory

在 RL theory 中我们通常会问什么问题?

第一类问题是关于 **Learning** 的, 主要有以下的形式 (记号的含义会在后面详细介绍):

**Example 1**. _如果我们使用了某个算法, 使用了_ $N$ _个 samples, 迭代_ $k$ _次, 我们 learning 的结果能有多好?_

_不妨假设我们使用 [Q-learning](https://zhida.zhihu.com/search?content_id=255437491&content_type=Article&match_order=1&q=Q-learning&zhida_source=entity), 对于一定的误差_ $\epsilon$_, 我们能否说明如果_ $N \geq f(\epsilon)$_, 我们能以概率_ $1 - \delta$ _有_ $\|\hat{Q}_k - Q^\ast \|_\infty \leq \epsilon?\\$ _这里_ $\hat{Q}_k$ _是我们在第_ $k$ _次迭代后 (学到的) Q-function,_ $Q^\ast$ _是 optimal 的 Q-function._

_类似地还可以是_ $\|Q^{\pi_k} - Q^\ast\|_\infty \leq \epsilon.\\$_这里_ $Q^{\pi_k}$ _是 policy_ $\pi_k$ _对应的真实 Q-function, 这是实际在问 expected reward 上的差异, 换言之 regret._

第二类问题是关于 **Exploration** 的, 主要有以下的形式:

**Example 2**. _如果我使用了某个特定的 exploration algorithm, 我们的 regret 会有多少高?_

_一个可能得到的结果形如:_ $\text{Reg}(T) \leq O\left(\sqrt{T \cdot N \cdot \log \frac{NT}{\delta}}\right) + \delta T.\\$_我们不去深究这个公式的具体形式._

在本节中我们主要关注第一类也就是 learning 相关的问题.

### 1.2 Assumptions in RL theory

在 RL theory 中, 我们通常需要使用很强的假设. 对于很弱的假设, 通常我们推不出什么有用的结论. 但是过强的假设通常又会偏离现实太远. RL theory 需要在这两者之间取得平衡, 使用能够产生有趣结果但是又不能偏离现实太远的假设.

在 **Exploration** 类型的问题中, 我们通常考虑 worst case, 目标是证明需要的时间是关于 $|S|, |A|, 1/(1 - \gamma)$ 的 polynomial (这通常比较 pessimistic).

在 **Learning** 类型的问题中, 我们通常会忽略掉 exploration 的问题, 考虑我们需要多少 samples 能够有效地学习一个 policy, 这主要依赖于以下可能的假设:

-   **"generative model" assumption**: 假设我们能够从 $p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})$ 中采样, 对于任意的 $\boldsymbol{s}, \boldsymbol{a}$.  
    
-   **"[oracle exploration](https://zhida.zhihu.com/search?content_id=255437491&content_type=Article&match_order=1&q=oracle+exploration&zhida_source=entity)"**: 对于每一个 $(\boldsymbol{s}, \boldsymbol{a})$, 我们从 $p(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})$ 中采样 $N$ 次.  
    

很显然在现实的 RL 中是做不到的, 但是这能让我们研究在 exploration 简单的情况下的 learning.

### 1.3 What we expect from RL theory?

在 RL theory 中, 我们通常应当期望得到什么? 这一点通常容易被误解:

1.  证明我们的 RL 算法每次都会工作很好? 对于当前的 deep RL 来说通常甚至无法保证收敛! 因此这是不可能的.  
    
2.  实际上, 在 RL theory 中, 通常我们在强假设下通过精确的理论来得到**不精确的定性结果**, 例如理解 errors 与 discount, state space, iteration, samples 之间一种定性的关系. 于此同时我们尽可能使得这些假设足够合理, 使得他们有可能在实际中成立 (尽管并没有这样的保证), 从而给出一个对可能结果的粗略指引, 例如当我们 state space 变多时, 我们应该迭代更多还是更少?  
    
3.  当我们听到 "provable guarantees" 时, 通常背后涉及的很多假设都不现实.  
    

## 2 Analysis on Model-Based RL

在这一小节中, 我们将介绍一些关于 model-based RL 的理论分析, 具体来说就是我们 model 误差对于我们学习到的 Q-function 以及 policy 的影响.

### 2.1 Assumptions and goals

在这里我们考虑以下假设:

1.  **"oracle exploration"** 假设: 对于每一个 $(\boldsymbol{s}, \boldsymbol{a})$, 我们从 $P(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})$ 中采样 $N$ 次.  
    
2.  根据估计/ 真实的环境, 我们可以完美估计一个 policy 对应的 Q-function. (而不考虑现实中 fitted Q-iteration 不收敛的问题)  
    

由于我们有上述的 "神谕" 给我们提供所有 $(\boldsymbol{s}, \boldsymbol{a})$ 上的数据, 且我们能够得到精确的 Q-function, 我们可以得到一个很简单的 "model based" algorithm:

1.  估计 dynamic: $\hat{P}(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a}) = \frac{\# (\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')}{N}$  
    
2.  给定 $\pi$, 使用 $\hat{P}$ 来估计 $\hat{Q}^{\pi}$  
    

此时我们考虑的就是 $\hat{P}$ 的 imperfect 会带给 Q-function estimation 的 error. 具体来说, 我们会考虑以下几个 Q-function 之间的关系:

-   $Q^{\pi}$: 某个 policy $\pi$ 在真实环境 $P$ 下对应的 Q-function.  
    
-   $\hat{Q}^\pi$: 某个 policy $\pi$ 在估计的环境 $\hat{P}$ 下的 Q-function.  
    
-   $Q^\ast$ 表示真实环境 $P$ 下的 optimal Q-function.  
    
-   $\hat{Q}^\ast$ 表示估计的环境 $\hat{P}$ 下的 optimal Q-function.  
    
-   $Q^{\hat{\pi}}$ 表示**估计的环境** $\hat{P}$ 下的 optimal Q-function 对应的 (argmax) policy 在**真实环境**下的 Q-function.  
    

简而言之, 上述概念中 $Q$ 上方的 hat 表示 Q-function 是否来源于估计的环境 $\hat{P}$, 而右上角的角标则代表这个 Q-function 对应于哪个 policy.

而我们通常会考虑以下三组 Q-function 之间的关系:

### 1\. $Q^{\pi}$ 和 $\hat{Q}^{\pi}$ 的关系:

也就是说, 我们会考虑同样一个 policy, 在真实环境与估计环境中 policy evaluation 得到的 Q-function 之间的差异.

### 2\. $Q^{\ast}$ 和 $\hat{Q}^{\ast}$ 的关系:

也就是说, 我们会考虑真实环境与估计环境中学到的 optimal Q-function 之间的差异.

### 3\. $Q^{\ast}$ 和 $Q^{\hat{\pi}}$ 的关系:

也就是说, 我们会考虑真实环境中学到的 optimal Q-function 与**估计环境中** optimal policy **在真实环境中**的 Q-function 之间的差异. 这实际上才是真正意义上的, 我们在估计环境中学到的 policy 与 optimal policy 在 expected reward 上的差异.

而这三组 Q-function 之间的差异我们通过以下方式来分析, 以第一组关系为例, 我们希望展示的是, 对于一定的差异 $\epsilon > 0$, 如果 $N \geq f(\epsilon, \delta)$, 则以以概率 $1 - \delta$, 有 $\|Q^{\pi} - \hat{Q}^{\pi}\|_\infty \leq \epsilon.\\$

实际上对其中第一个问题的分析就可以给出我们分析后续问题的很好的工具.

在 supervised learning 的理论中, 我们有很多相关的工具来处理, 我们可以考虑将这些工具迁移到 RL 的分析中.

### 2.2 Concentration inequalities

在 supervised learning 中, 我们通常会使用 concentration inequalities 来分析我们的估计的误差. 一个重要的 inequality 是 Hoeffding's inequality.

**Theorem 1**. _Hoeffding's inequality_

_如果_ $X_1, \ldots, X_N$ _是独立同分布的随机变量, 其又均值_ $\mu$_, 记_ $\bar{X}_n = n^{-1}\sum_{i = 1}^n X_i$_, 且_ $a \leq X_i \leq b$_, 则对于任意_ $\epsilon > 0$_, 有_ $P\left(\bar{X}_n \geq \mu + \epsilon\right) \leq \exp\left(-\frac{2n\epsilon^2}{(b - a)^2}\right),\\$ _类似地,_ $P\left(\bar{X}_n \leq \mu - \epsilon\right) \leq \exp\left(-\frac{2n\epsilon^2}{(b - a)^2}\right).\\$

**Remark:** 这个定理有很多种理解方式:

1.  这个定理描绘了通过 samples 估计均值时, 我们的估计和真实值的差异. 这是一个很强的结果, 因为我们出现过大误差的概率随着 $n$ 的增加而指数下降.  
    
2.  对于一定的样本数 $n$ 以及我们容许的出错概率不超过 $\delta$, 则可能的误差 $\epsilon$ 不超过 $\frac{b - a}{\sqrt{2n}} \sqrt{\log\frac{2}{\delta}}$, 这可以利用 $\delta \leq 2\exp(-2n\epsilon^2/(b - a)^2)$ 得到. 这意味着 $\epsilon$ 的上界正比于 $1/\sqrt{n}$.  
    
3.  对于一定的 $\epsilon$ 以及 $\delta$, 我们需要 $(b - a)^2\log(2/\delta)/2\epsilon^2$ 个样本来保证出现超过 $\epsilon$ 的误差的概率小于 $\delta$.  
    

在 RL theory 中, 我们需要考虑的是多类变量而不是均值, 我们可以得出离散分布的 concentration inequalities.

**Theorem 2**. _Concentration for discrete distribution_

_如果_ $X_1, \ldots, X_N$ _是独立同分布的离散随机变量, 依照分布_ $q$ _取值在_ $\{1,\ldots,d\}$_. 我们记_ $q$ _为一个向量_ $\overrightarrow{q} = [P(z = j)]_{j = 1}^d$_. 记我们通过_ $X_1, \ldots, X_N$ _的样本估计_ $q$ _为_ $[\hat{q}]_j = \sum_{i = 1}^{N} \mathbb{I}(X_i = j)/N$_. 于是对于_ $\forall \epsilon > 0$_:_ $P\left(\|\overrightarrow{q} - \hat{q}\|_2 \geq \frac{1}{\sqrt{N}} + \epsilon\right) \leq \exp(-N\epsilon^2),\\$ _这可以推出 (直接利用_ $1$_\-norm 和_ $2$_\-norm 的关系):_ $P\left(\|\overrightarrow{q} - \hat{q}\|_1 \geq \sqrt{d} \left(\frac{1}{\sqrt{N}} + \epsilon\right)\right) \leq \exp(-N\epsilon^2).\\$

后一个推论可以被用于 total variation distance 的估计 (两个分布的 total variation distance 是 $1$\-norm 的一半).

对于上述结论, 我们可以得出类似 Hoeffding inequality 的一系列理解方式, 其中一个就是对于样本数 $N$ 和我们容许的出错概率 $\delta$, 可以解得 $\epsilon \leq \frac{1}{\sqrt{N}} \sqrt{\log\frac{1}{\delta}}.\\$ 这在我们原先 RL 中的意义是, 我们以 $1 - \delta$ 的概率, 有 $\|\hat{P}(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a}) - P(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})\|_1 \leq \sqrt{|S|} \left(\frac{1}{\sqrt{N}} + \epsilon\right) \leq \sqrt{\frac{|S|}{N}} + \sqrt{\frac{|S| \log 1/\delta}{N}} \leq c \sqrt{\frac{|S| \log 1/\delta}{N}}.\\$ 注意 $N$ 是仅仅估计一个 $(\boldsymbol{s}, \boldsymbol{a})$ 下的 dynamic 所需的样本数, 因此我们需要的总样本数是 $|S||A|N$.

### 2.3 Relating $P$ and Q-function

接下来我们需要介绍一些针对于 RL 问题的引理. 这些引理的共同目的是将 $\hat{P}$ 的估计 error 同最终 $\hat{Q}^\pi$ 的 error 关联起来.

在关联 error 之前, 我们先考虑将 $P$ 与 $Q^\pi$ 关联起来, 这里有两种关联方式:

### 通过 **transition** 关联:

$Q^\pi(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \mathbb{E}_{\boldsymbol{s}' \sim P(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})} \left[V^\pi(\boldsymbol{s}')\right].\\$ 写作概率的形式就是 $Q^\pi(\boldsymbol{s}, \boldsymbol{a}) = r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \sum_{\boldsymbol{s}'} P(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a}) V^\pi(\boldsymbol{s}').\\$ 进一步使用 vector 形式表示, 则有 $Q^\pi = r + \gamma P V^\pi.\\$ 这里 $Q^\pi, r$ 是 $|S| |A|$ 的向量, $P$ 是 $|S| |A| \times |S|$ 的矩阵, $V^\pi$ 是 $|S|$ 的向量.

### 利用 **policy 下的期望** 关联:

$V^\pi = \Pi Q^\pi,\\$ 其中 $\Pi$ 是一个 $|S| \times |S| |A|$ 的矩阵, 与 policy $\pi(\boldsymbol{a}\mid \boldsymbol{s})$ 有关.

### 将两种关联方式结合:

结合上述利用 transition 与 policy 的两种表示方式得到 $Q^\pi = r + \gamma P^\pi Q^\pi,\\$ 其中 $P^\pi = P \Pi$. 进一步化简得到 (可证明 $I - \gamma P^\pi$ 是可逆的) $Q^\pi = (I - \gamma P^\pi)^{-1} r.\\$

类似地我们可以得到 $\hat{Q}^\pi = (I - \gamma \hat{P}^\pi)^{-1} r,\\$ 于是我们就成功地将 $P$ 与 Q-function 建立起了联系. 进而我们可以考察它们 error 之间的联系.

### 2.4 Relating errors in $P$ and Q-function

这里考虑两个 lemma:

**Lemma 1**. _simulation lemma_

$Q^\pi - \hat{Q}^\pi = \gamma (I - \gamma P^\pi)^{-1} (P - \hat{P}) V^\pi.\\$

**Remark:** 一个直观的理解方式是: 这里 $(I - \gamma P^\pi)^{-1}$ 是一个 evaluation operator, 而 $P - \hat{P}$ 是 difference in probabilities. 一个理解方式是, 想象 $V^\pi$ 是一个 pseudo-reward, 其先被 dynamic 的差异作用, 再通过 evaluation 就得到了 Q-function 的差异.

_Proof._$\begin{aligned} Q^\pi - \hat{Q}^\pi &= Q^\pi - (I - \gamma \hat{P}^\pi)^{-1} r\\ &= (I - \gamma \hat{P}^\pi)^{-1} (I - \gamma \hat{P}^\pi) Q^\pi - (I - \gamma \hat{P}^\pi)^{-1} r\\ &= (I - \gamma \hat{P}^\pi)^{-1} (I - \gamma \hat{P}^\pi) Q^\pi - (I - \gamma \hat{P}^\pi)^{-1} (I - \gamma P^\pi) Q^\pi\\ &= (I - \gamma \hat{P}^\pi)^{-1} ((I - \gamma \hat{P}^\pi) - (I - \gamma P^\pi)) Q^\pi\\ &= \gamma (I - \gamma P^\pi)^{-1} (P^\pi - \hat{P}^\pi) V^\pi\\ &= \gamma (I - \gamma P^\pi)^{-1} (P\Pi - \hat{P}\Pi) Q^\pi\\ &= \gamma (I - \gamma P^\pi)^{-1} (P - \hat{P}) \Pi Q^\pi\\ &= \gamma (I - \gamma P^\pi)^{-1} (P - \hat{P}) V^\pi. \end{aligned}\\$

**Lemma 2**. _给定_ $P^\pi$ _和任何的_ $v \in \mathbb{R}^{|S||A|}$_, 我们有_ $\|(I - \gamma P^\pi)^{-1} v\|_\infty \leq \|v\|_\infty / (1 - \gamma).\\$

**Remark:** 这意味着将 evaluation operator 作用在一个向量上, 结果的 infinity norm 不会超过原向量的 infinity norm 除以 $1 - \gamma$. 换言之对应于 "reward" $v$ 的 Q-function 每次最多增大到 $1/(1 - \gamma)$ 倍.

**Note:** 注意这里的 $1/(1 - \gamma)$ 来源于无穷级数的求和, 这通常可以视作是某种 effective horizon.

_Proof._

令 $w = (I - \gamma P^\pi)^{-1} v$, 则有 $\begin{aligned} \|v\|_\infty &= \|(I - \gamma P^\pi) w\|_\infty\\ &\geq \|w\|_\infty - \gamma \|P^\pi w\|_\infty\\ &\geq \|w\|_\infty - \gamma \|w\|_\infty\\ &= (1 - \gamma) \|w\|_\infty. \end{aligned}\\$ 前一个不等号利用了 norm 的三角不等式, 后者利用了 transition matrix 的 infinity norm 不超过 $1$. $\square$

### 2.5 Putting it all together

将上述两个 lemma 结合, 我们将 $(P - \hat{P}) V^\pi$ 代入第二个 lemma 中的 $v$, 于是 $\begin{aligned} \|Q^\pi - \hat{Q}^\pi\|_\infty &= \gamma \|(I - \gamma P^\pi)^{-1} (P - \hat{P}) V^\pi\|_\infty\\ &\leq \frac{\gamma}{1 - \gamma} \|(P - \hat{P}) V^\pi\|_\infty\\ &\leq \frac{\gamma}{1 - \gamma} \|(P - \hat{P})\|_\infty \|V^\pi\|_\infty\\ &\leq \frac{\gamma}{1 - \gamma} \left(\max_{\boldsymbol{s}, \boldsymbol{a}}\|P(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a}) - \hat{P}(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})\|_1\right) \|V^\pi(\boldsymbol{s})\|_\infty. \end{aligned}\\$ 这里最后的不等号利用了 $|S||A| \times |S|$ 这一矩阵的 $\infty$\-norm 的定义, 也就是各行绝对值和的最大值.

而同时我们可以 bound $\|V^\pi\|_\infty$: $\sum_{t = 0}^{\infty} \gamma^t r_t \leq \frac{1}{1 - \gamma} R_{\max} = \frac{1}{1 - \gamma} R_{\max}.\\$ 通常我们假设 $R_{\max} = 1$, 于是 于是进一步化简 $\begin{aligned} \|Q^\pi - \hat{Q}^\pi\|_\infty &\leq \frac{\gamma}{(1 - \gamma)^2}\left(\max_{\boldsymbol{s}, \boldsymbol{a}}\|P(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a}) - \hat{P}(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})\|_1\right)\\ &\leq \frac{\gamma}{(1 - \gamma)^2} c_2 \sqrt{\frac{|S| \log |S||A|/\delta}{N}}. \end{aligned}\\$ 第二个不等式利用了我们之前证明的 concentration inequality, 这里有两点需要注意:

1.  对于每一个 $(\boldsymbol{s}, \boldsymbol{a})$ 我们的 concentration 前都有一个 $c$ 的系数, 但是由于有 max 操作, 我们可以选择其中最大系数的那个作为 $c_2$.  
    
2.  注意我们这里根号内的 $\log$ 中多出了 $|S||A|$, 这是为什么? 注意之前的 bound 是对于单个 $(\boldsymbol{s}, \boldsymbol{a})$ 犯错概率不超过 $\delta$ 得到的, 由于此时有 $|S||A|$ 个这样的 pair, 只有让每一个 $(\boldsymbol{s}, \boldsymbol{a})$ 的概率不超过 $\delta/|S||A|$, 再利用 union bound, 才能保证整体犯错概率不超过 $\delta$.  
    

最终的结果就是 $\|Q^\pi - \hat{Q}^\pi\|_\infty \leq \frac{c_2 \gamma}{(1 - \gamma)^2} \sqrt{\frac{|S| \log 1/\delta}{N}}.\\$ 回顾我们 RL Theory 的目标不是给出一个定量的理论保证, 而是给出了一个定性的结果, 我们从这些角度来理解这个结果:

1.  error 会随着 samples 的增加而减小, error 正比于 $\sqrt{1/N}$.  
    
2.  error 会随着 horizon $1/(1 - \gamma)$ 而二次增加.  
    

在给定每个 $(\boldsymbol{s}, \boldsymbol{a})$ 有 $N$ 个样本的情况下, 如果我们需要以 $1 - \delta$ 的概率保证 $\|Q^\pi - \hat{Q}^\pi\|_\infty \leq \epsilon$, 那么我们的 $\epsilon = \frac{c_2\gamma}{(1 - \gamma)^2} \sqrt{\frac{|S| \log 1/\delta}{N}}.\\$

### 2.6 Extensions to other errors

我们前面分析了 $\|Q^\pi - \hat{Q}^\pi\|_\infty$ 误差的 bound, 我们同样可以考虑 $\|Q^\ast - \hat{Q}^\ast\|_\infty$ 和 $\|Q^\ast - Q^{\hat{\pi}}\|_\infty$.

### 1\. $Q^\ast$ 和 $\hat{Q}^\ast$ 的关系:

利用 $|\sup_x f(x) - \sup_x g(x)| \leq \sup_x |f(x) - g(x)|,\\$ 有 $\|Q^\ast - \hat{Q}^\ast\|_\infty = \|\sup_\pi Q^\pi - \sup_\pi \hat{Q}^\pi\|_\infty \leq \sup_\pi \|Q^\pi - \hat{Q}^\pi\|_\infty.\\$

### 2\. $Q^\ast$ 和 $Q^{\hat{\pi}}$ 的关系:

首先利用三角不等式得到 $\|Q^\ast - Q^{\hat{\pi}}\|_\infty = \|Q^\ast - \hat{Q}^{\hat{\pi}} + \hat{Q}^{\hat{\pi}} - Q^{\hat{\pi}}\|_\infty \leq \|Q^\ast - \hat{Q}^{\hat{\pi}}\|_\infty + \|\hat{Q}^{\hat{\pi}} - Q^{\hat{\pi}}\|_\infty,\\$

对于第一项, 由于 $\hat{Q}^{\hat{\pi}}$ 含义为 **估计环境** 下的 optimal policy 在 **估计环境** 中的 Q-function, 实际也就是 $\hat{Q}^\ast$, 因此直接应用刚刚推导的结果, 就得到了 $\epsilon$ 的 bound.

对于第二项, 由于我们 $\|Q^\pi - \hat{Q}^\pi\|_\infty$ 的结果对于任何 policy $\pi$ 都成立, 故可以替换为 $\hat{\pi}$, 于是第二项可以 bound 到 $\epsilon$.

综上, 就可以得到 $\|Q^\ast - Q^{\hat{\pi}}\|_\infty$ 的 bound 为 $2\epsilon$.

## 3 Analysis on Model-free RL

### 3.1 Abstract

上一部分我们主要考虑了在 model-based 的 setting, 具体来说在忽略 exploration 与假设可以学习到环境对应的 optimal Q-function 的情况下, Q-function 的 error 与样本数以及 horizon 等因素的关系.

这一部分我们讨论 model-free 相关的问题. 在 **value function methods** 中, 我们已经证明了对于一般的 fitted Q-iteration 不存在收敛的理论保证. 但接下来我们考虑一个更加理想的模型, 在这个模型中可以进行一些理论的分析, 得出 Q-function 的 error 的 bound.

### Update in real environment:

对于 exact 的 Q-iteration, 我们进行 bellman 更新 $Q_{t + 1} \gets T Q_t,\\$ 其中 $T$ 是 Bellman operator, 也就是 $TQ = r + \gamma P \max_a Q.\\$

### Update we actually do:

但是现实中我们并不知道 $T$, 我们实际的更新是一种近似的 fitted Q-iteration: $\hat{Q}_{k + 1} \gets \arg\min_{\hat{Q}} \|\hat{Q} - \hat{T} \hat{Q}_k\|,\\$ 这里的 $\hat{T}$ 是一个 approximate 的 Bellman operator, 也就是 $\hat{T}Q = \hat{r} + \gamma \hat{P} \max_a Q.\\$ 我们的采样得到的数据共同决定了这里的 $\hat{T}, \hat{r}, \hat{P}$, 其中 $\hat{r}$ 与 $\hat{P}$ 分别估计为 $\hat{r}(\boldsymbol{s}, \boldsymbol{a}) = \frac{1}{N(\boldsymbol{s}, \boldsymbol{a})} \sum_{i} \delta((\boldsymbol{s}_i, \boldsymbol{a}_i) = (\boldsymbol{s}, \boldsymbol{a})) r_i, \quad \hat{P}(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a}) = \frac{N(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')}{N(\boldsymbol{s}, \boldsymbol{a})}\\$ 注意这不是 model, 我们在 Q-iteration 的时候如果从已有数据中进行更新, 那么实际上就在进行 $\hat{T}$ 所对应的更新.

### Error analysis:

1.  **sampling error**: 在 $\hat{r}, \hat{P}$ 估计中具有 sampling error, 我们的 $\hat{T}$ 与 $T$ 不同.  
    
2.  **approximation error**: 在 $\hat{Q}_{k + 1}$ 的更新中, 我们的 $\hat{Q}$ 也只是一个 approximation, 与真正的最小值 $\hat{T} \hat{Q}_k$ 有差异.  
    

### Assumption:

这里我们还没有决定使用什么样的 norm, 我们已经知道使用 $2$\-norm时无法得到关于收敛的保证. 因此这里假设每次更新在 **infinity norm** 的意义下进行最小化, 同时假设每一个 $\hat{Q}_{k + 1}$ 在 infinity norm 下都是 bounded 的.

我们接下来尝试分析的是, 当 $k \to \infty$ 时, $\hat{Q}_k$ 是否会收敛到 $Q^\ast$. 如果是, 那么我们希望得到一个关于 $\hat{Q}_k$ 与 $Q^\ast$ 之间的 error 与其他因素的关系.

### 3.2 Sampling error

首先我们分析上述提到的 sampling error, 也就是 $|\hat{T} Q(\boldsymbol{s}, \boldsymbol{a}) - TQ(\boldsymbol{s}, \boldsymbol{a})|$, 注意到 $\begin{aligned} &|\hat{T} Q(\boldsymbol{s}, \boldsymbol{a}) - TQ(\boldsymbol{s}, \boldsymbol{a})| \\ &= |\hat{r}(\boldsymbol{s}, \boldsymbol{a}) - r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \left(\mathbb{E}_{\hat{P}(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})} \left[\max_{a'} Q(\boldsymbol{s}', \boldsymbol{a}')\right] - \mathbb{E}_{P(s' \mid s, a)} \left[\max_{a'} Q(\boldsymbol{s}', \boldsymbol{a}')\right]\right)\\ &\leq |\hat{r}(\boldsymbol{s}, \boldsymbol{a}) - r(\boldsymbol{s}, \boldsymbol{a})| + \gamma \left|\mathbb{E}_{\hat{P}(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})} \left[\max_{a'} Q(\boldsymbol{s}', \boldsymbol{a}')\right] - \mathbb{E}_{P(s' \mid s, a)} \left[\max_{a'} Q(\boldsymbol{s}', \boldsymbol{a}')\right]\right|\\ \end{aligned}\\$ 第一项是 reward 的估计误差, 我们使用 Hoeffding inequality 可以得到 $|\hat{r}(\boldsymbol{s}, \boldsymbol{a}) - r(\boldsymbol{s}, \boldsymbol{a})| \leq 2R_{\max}\sqrt{\frac{\log 1/\delta}{2N}}.\\$ 对于第二部分, 我们转化为 $\begin{aligned} &\sum_{\boldsymbol{s}'} (\hat{P}(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a}) - P(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})) \max_{\boldsymbol{a}'} Q(\boldsymbol{s}', \boldsymbol{a}')\\  &\leq \sum_{\boldsymbol{s}'} |\hat{P}(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a}) - P(\boldsymbol{s}' \mid \boldsymbol{s}, \boldsymbol{a})| \max_{\boldsymbol{s}',\boldsymbol{a}'} Q(\boldsymbol{s}', \boldsymbol{a}')\\ &= \|\hat{P}(\cdot \mid \boldsymbol{s}, \boldsymbol{a}) - P(\cdot \mid \boldsymbol{s}, \boldsymbol{a})\|_1 \|Q\|_\infty\\ &\leq c \|Q\|_\infty \sqrt{\frac{ \log 1/\delta}{N}}. \end{aligned}\\$

于是 $|\hat{T} Q(\boldsymbol{s}, \boldsymbol{a}) - TQ(\boldsymbol{s}, \boldsymbol{a})| \leq 2R_{\max}\sqrt{\frac{\log 1/\delta}{2N}} + c \|Q\|_\infty \sqrt{\frac{ \log 1/\delta}{N}}.\\$ 此时类似于之前的分析, 为了保证总犯错概率不超过 $\delta$, 我们可以在前后两个 bound 分别分配 $\delta/2$ 的犯错概率, 之后再对 $(\boldsymbol{s}, \boldsymbol{a})$ 或者 $\boldsymbol{s}$ 取 union bound, 于是我们可以得到 $\|\hat{T} Q - TQ\|_\infty \leq 2R_{\max} c_1 \sqrt{\frac{\log |S||A|/\delta}{2N}} + c_2 \|Q\|_\infty \sqrt{\frac{\log |S|/\delta}{N}}.\\$

### 3.3 Approximation error

这一部分我们来分析 approximation error, 这里我们假设 approximation error 有如下的 bound: $\|\hat{Q}_{k + 1} - \hat{T} \hat{Q}_k\|_\infty \leq \epsilon_k.\\$ 由于这里的 bound 是对 $\infty$\-norm 的, 因此这是一个很强的 assumption, 在现实中的 supervised learning 中, 我们通常无法得到这样的保证.

我们接下来考虑 $\hat{Q}_k$ 与 $Q^\ast$ 之间的 error, 此时利用 $Q^\ast$ 是 Bellman backup 的 fixed point 就有: $$$\begin{aligned} \|\hat{Q}_k - Q^\ast\|_\infty &= \|\hat{Q}_k - T \hat{Q}_{k - 1} + T \hat{Q}_{k - 1} - Q^\ast\|_\infty\\ &\leq \|\hat{Q}_k - T \hat{Q}_{k - 1}\|_\infty + \|T \hat{Q}_{k - 1} - TQ^\ast\|_\infty\\ &\leq \|\hat{Q}_k - T \hat{Q}_{k - 1}\|_\infty + \|T \hat{Q}_{k - 1} - TQ^\ast\|_\infty\\ &\leq \|\hat{Q}_k - T \hat{Q}_{k - 1}\|_\infty + \gamma \| \hat{Q}_{k - 1} - Q^\ast\|_\infty. \end{aligned}\\$$$ 其中后一项利用 $T$ 是 contraction operator.

于是我们展开这个递归式: $\begin{aligned} \|\hat{Q}_k - Q^\ast\|_\infty &\leq \|\hat{Q}_k - T \hat{Q}_{k - 1}\|_\infty + \gamma \|\hat{Q}_{k - 1} - Q^\ast\|_\infty\\ &\leq \|\hat{Q}_k - T \hat{Q}_{k - 1}\|_\infty + \gamma \|\hat{Q}_{k - 1} - T \hat{Q}_{k - 2}\|_\infty + \gamma^2 \|\hat{Q}_{k - 2} - Q^\ast\|_\infty\\ &\leq \sum_{i = 0}^{k - 1}\gamma^i \|\hat{Q}_{k - i} - T \hat{Q}_{k - i - 1}\|_\infty + \gamma^k \|\hat{Q}_0 - Q^\ast\|_\infty. \end{aligned}\\$ 一个 implication 是, 我们迭代次数越多, 我们关于 initialization "忘记"的越多.

取 $k \to \infty$ 的极限, 进一步简化得到 $\lim_{k \to \infty} \|\hat{Q}_k - Q^\ast\|_\infty \leq \sum_{i = 0}^\infty \gamma^i \max_k \|\hat{Q}_{k + 1} - T \hat{Q}_{k}\|_\infty.\\$ 同样也是 horizon 越大, 我们可能的 error 也会越大.

### 3.4 Putting it all together

根据上述推导, $\hat{Q}_k$ 与 $Q^\ast$ 之间的 error 可以写作 $\begin{aligned} \lim_{k \to \infty} \|\hat{Q}_k - Q^\ast\|_\infty &\leq \frac{1}{1 - \gamma} \max_k \|\hat{Q}_k - T \hat{Q}_{k - 1}\|_\infty,\\ \end{aligned}\\$ 其中 $\begin{aligned} \|\hat{Q}_k - T \hat{Q}_{k - 1}\|_\infty &= \frac{1}{1 - \gamma} \|\hat{Q}_k - \hat{T} \hat{Q}_{k - 1} + \hat{T} \hat{Q}_{k - 1} - T \hat{Q}_{k - 1}\|_\infty\\ &\leq \frac{1}{1 - \gamma} (\|\hat{Q}_k - \hat{T} \hat{Q}_{k - 1}\|_\infty + \|\hat{T} \hat{Q}_{k - 1} - T \hat{Q}_{k - 1}\|_\infty)\\ \end{aligned}\\$ 其中前一项对应于 approximation error, 后一项对应于 sampling error.

由于 $\|Q\|_\infty$ 是 $O(R_{\max}/(1 - \gamma))$ 的, 于是我们可以得到 $\begin{aligned} \lim_{k \to \infty} \|\hat{Q}_k - Q^\ast\|_\infty &= O\left(\frac{\|\epsilon\|_\infty}{1 - \gamma} + \frac{c_1 R_{\max}}{1 - \gamma} \sqrt{\frac{\log |S||A|/\delta}{2N}} + \frac{c_2 R_{\max}}{(1 - \gamma)^2} \sqrt{\frac{\log |S|/\delta}{N}}\right).\\ \end{aligned}\\$

**Remark:** 上述过程是一个相对粗略的分析, 其中的一个 implication 是我们的 error 中依然存在一个关于 horizon 是二次增大的项:

## 4 Summary

在本节中, 我们

-   简单介绍了 RL theory 关心的问题, 通常的假设, 以及我们对 RL theory 的正确期望.  
    
-   介绍了分析 RL theory 的基本工具例如 concentration inequalities.  
    
-   介绍了 "oracle exploration" 假设下 model-based RL 的 error 与 horizon, samples 数量等因素的关系.  
    
-   介绍了 model-free RL 的 error 的主要组成和 bound.