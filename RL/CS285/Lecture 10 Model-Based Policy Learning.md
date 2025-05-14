## 1 From open-loop planning to close-loop policy learning

回顾上一节介绍的 [MPC](https://zhida.zhihu.com/search?content_id=254436995&content_type=Article&match_order=1&q=MPC&zhida_source=entity) 算法:

1.  运行 base policy $\pi_0(\boldsymbol{a}_t, \boldsymbol{s}_t)$, 收集 $\mathcal{D} = \{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')_i\}$,  
    
2.  学习 dynamic model $f(\boldsymbol{s}, \boldsymbol{a})$ 来最小化 $\sum_{i} \|f(\boldsymbol{s}_i, \boldsymbol{a}_i) - \boldsymbol{s}'_i\|^2$  
    
3.  依据 $f(\boldsymbol{s}, \boldsymbol{a})$ 来进行 plan  
    
4.  执行第一个规划的 action, 观测到新的状态 $\boldsymbol{s}'$ (MPC)  
    
5.  添加 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')$ 到 $\mathcal{D}$, 重复 3-5; 每 $N$ 次回到 2.  
    

这实际上构造的是一种 open-loop controller, 在学习的 dynamics 上进行 optimal control, 得到一条 action sequence. 核心在于, 尽管进行了 replanning, 表面上每次只执行一个 action, 但是与 close-loop 核心的区别是, 在进行 plan 时, **我们并不知道未来还会 replan!**

![](https://pic3.zhimg.com/v2-5b770ef87221e8e7d224b11a6d98fbe2_1440w.jpg)

open-loop control

鉴于 open-loop control 的 suboptimality, 我们接下来考虑 close-loop control, 我们前面已经证明 close-loop control 中的目标与 model-free RL 在形式上是一致的: $\pi = \arg\max_{\pi} \mathbb{E}_{\tau \sim p(\tau)} \left[\sum_{t} r(\boldsymbol{s}_t, \boldsymbol{a}_t)\right].\\$

![](https://pic3.zhimg.com/v2-6de7f615227c70a4aea43aeb67d98fba_1440w.jpg)

close-loop control

这里需要考虑的一点是 $\pi$ 的形式: 主要的策略形式有两种:

-   **global policy**: $\pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t)$, 在所有时间步上都使用相同的映射, 一般通过神经网络等全局函数表示, 具有较强的泛化能力, 可以适用于较多场景. 在执行时不需要每次进行 planning, 而是直接基于当前状态选择动作.  
    
-   **[local policy](https://zhida.zhihu.com/search?content_id=254436995&content_type=Article&match_order=1&q=local+policy&zhida_source=entity)**: 通常指 trajectory optimization 或 MPC 中计算得到的 time-varying policy, 例如 $\boldsymbol{u}_t = \boldsymbol{K}_t \boldsymbol{s}_t + \boldsymbol{k}_t$, 该策略只在当前 planning 的 trajectory 及其附近区域内有效, 重规划时需要重新计算. 虽然可能在当前状态附近表现非常好, 但泛化到其他状态时性能不一定理想.  
    

在本节中, 我们考虑学习一个用神经网络表示的 global policy.

### 1.1 Backpropagation through time

在 model-based 的 setting 中, 由于我们有了一个 dynamics model, 我们目标函数 (整条轨迹的 reward)理论上可以利用 policy 与 dynamics model 表示出来. 一个相当 naive 的想法是, 我们能否直接反向传播到 policy 呢?

这在理论上是可以实现的:

-   **deterministic dynamics**: 我们可以直接计算 $\boldsymbol{s}_{t + 1} = f(\boldsymbol{s}_t, \boldsymbol{a}_t)$, 从而完全使用已知函数表达出目标函数, 进而直接反向传播.  
    
-   **stochastic dynamics**: 我们可以使用 reparameterization trick.  
    

![](https://pic4.zhimg.com/v2-902bd568a328a9c10ae0509768238157_1440w.jpg)

backpropagate through time

基于这种 naive 的想法, 我们可以得到 model-based reinforcement learning version 2.0:

1.  运行 base policy $\pi_0(\boldsymbol{a}_t, \boldsymbol{s}_t)$, 收集 $\mathcal{D} = \{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')_i\}$,  
    
2.  学习 dynamic model $f(\boldsymbol{s}, \boldsymbol{a})$ 来最小化 $\sum_{i} \|f(\boldsymbol{s}_i, \boldsymbol{a}_i) - \boldsymbol{s}'_i\|^2$  
    
3.  通过反向传播来 optimize policy $\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{s}_t)$.  
    
4.  添加 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')$ 到 $\mathcal{D}$, 重复 $2-4$.  
    

这样的算法有一些问题: 开始位置的 actions 的梯度涉及到多个 Jacobian 相乘, 因此容易梯度爆炸或梯度消失, 产生较为严重的数值问题. 具体来说, 由于我们现在可以学习一个 model, 我们也可以给出 Backprop (pathwise) gradient (**假设 reward 为 state-dependent**): $\nabla_\theta J(\theta) = \sum_{t = 1}^{T} \frac{\text{d}\boldsymbol{a}_t}{\text{d}\theta} \frac{\text{d}\boldsymbol{s}_{t + 1}}{\text{d}\boldsymbol{a}_t} \left(\sum_{t' = t + 1}^{T} \frac{\text{d}r_{t'}}{\text{d}\boldsymbol{s}_{t'}} \left(\prod_{t'' = t + 2}^{t'} \left(\frac{\text{d}\boldsymbol{s}_{t''}}{\text{d}\boldsymbol{a}_{t'' - 1}} \frac{\text{d}\boldsymbol{a}_{t''}}{\text{d}\boldsymbol{s}_{t'' - 1}} + \frac{\text{d}\boldsymbol{s}_{t''}}{\text{d}\boldsymbol{s}_{t'' - 1}}\right)\right)\right).\\$ 其中造成数值问题的核心就在于上式中的 $\prod_{t'' = t + 2}^{t'}$ 项.

由于这里的处理方式与 policy gradient 的处理方式存在一定的差异, 故我们详细地给出推导过程:

_Proof._ 首先明确单条 trajectory 的总 reward 为 $J(\theta) = \sum_{t=1}^T r(\boldsymbol{s}_t)$ (这里设定 reward 为 state-dependent), 而 action 由 $\boldsymbol{a}_t = \pi_\theta(\boldsymbol{s}_t)$ 生成 (如果是 stochastic policy, 则我们还需要使用 reparameterization trick). 我们考虑如下几步:

**Step 1: 梯度展开**: 对每个时间步 $t$, 参数 $\theta$ 通过动作 $\boldsymbol{a}_t$ 影响后续奖励 $r_{t+1}, r_{t+2}, \dots$. 梯度需累积这些影响: $\nabla_\theta J(\theta) = \sum_{t=1}^T \left( \frac{\text{d}\boldsymbol{a}_t}{\text{d}\theta} \cdot \sum_{t' = t + 1}^T \frac{\text{d}r_{t'}}{\text{d}\boldsymbol{a}_t} \right).  \\$

**Step 2: 链式法则应用**: 对于每个后续奖励 $r_{t'}$ ($t' \geq t + 1$), 其导数 $\frac{\text{d}r_{t'}}{\text{d}\boldsymbol{a}_t}$ 需沿路径 $t \rightarrow \cdots \rightarrow t'$ 展开为以下三部分:

-   $\boldsymbol{a}_t$ 影响 $\boldsymbol{s}_{t+1}$(导数 $\frac{\text{d}\boldsymbol{s}_{t+1}}{\text{d}\boldsymbol{a}_t}$).  
    
-   $\boldsymbol{s}_{t''}$ 影响 $\boldsymbol{a}_{t''}$ 和 $\boldsymbol{s}_{t'' + 1}$, 其中 $t + 1 \leq t'' \leq t' - 1$.  
    
-   $\boldsymbol{s}_{t'}$ 直接影响 $r_{t'}$ (导数 $\frac{\text{d}r_{t'}}{\text{d}\boldsymbol{s}_{t'}}$).  
    

其中第一部分与 $t'$ 没有关系, 因此可以提到最外面, 与 $\frac{\text{d}\boldsymbol{a}_t}{\text{d}\theta}$ 乘在一起.

第二部分需要进一步展开, 考虑 $\boldsymbol{s}_{t''} = f(\boldsymbol{s}_{t'' - 1}, \boldsymbol{a}_{t'' - 1})$, 总导数为直接的转移 $\frac{\text{d}\boldsymbol{s}_{t''}}{\text{d}\boldsymbol{s}_{t''-1}}$ 与经过动作中介的转移 $\frac{\text{d}\boldsymbol{s}_{t''}}{\text{d}\boldsymbol{a}_{t''-1}} \cdot \frac{\text{d}\boldsymbol{a}_{t''-1}}{\text{d}\boldsymbol{s}_{t''-1}}$ 之和. 由于我们第二部分被影响的下标从 $t + 2$ 到 $t'$, 故我们可以得到一个连乘项: $\prod_{t''=t+2}^{t'} \left( \frac{\text{d}\boldsymbol{s}_{t''}}{\text{d}\boldsymbol{a}_{t''-1}} \frac{\text{d}\boldsymbol{a}_{t''-1}}{\text{d}\boldsymbol{s}_{t''-1}} + \frac{\text{d}\boldsymbol{s}_{t''}}{\text{d}\boldsymbol{s}_{t''-1}} \right).  \\$

**Step 3: 整理**

因此整理即可得到我们的结果: $\nabla_\theta J(\theta) = \sum_{t=1}^T \frac{\text{d}\boldsymbol{a}_t}{\text{d}\theta} \frac{\text{d}\boldsymbol{s}_{t+1}}{\text{d}\boldsymbol{a}_t} \left( \sum_{t'=t+1}^T \frac{\text{d}r_{t'}}{\text{d}\boldsymbol{s}_{t'}} \prod_{t''=t+2}^{t'} \left( \frac{\text{d}\boldsymbol{s}_{t''}}{\text{d}\boldsymbol{a}_{t''-1}} \frac{\text{d}\boldsymbol{a}_{t''-1}}{\text{d}\boldsymbol{s}_{t''-1}} + \frac{\text{d}\boldsymbol{s}_{t''}}{\text{d}\boldsymbol{s}_{t''-1}} \right) \right).  \\$ ◻

**Remark:**

-   这里的出现的数值问题在现象上类似于 trajectory optimization 的 shooting method 的问题. (从数值问题严重的角度, 这些数值问题严重的问题不适合使用二阶优化方法)  
    
-   由于当前策略在各时间步都由 $\theta$ 关联 (换言之我们的 policy 不再是 local 的), 因此也不能够利用 LQR 类型方法中的 dynamic programming.  
    
-   从 DL 的角度, 这样的问题类似于在 RNN 等 backpropagate through time (BPTT) 的网络的梯度消失或爆炸的问题. 但我们无法通过引入 LSTM 等结构与技巧来解决这个问题, 因为我们不能人为设置一个 dynamics, 我们的 $f$ 必须接近于真正的 dynamics. 这可能听起来有些奇怪, 不妨考虑如下计算图 (你可以想象为是一个没有输入的 RNN 网络, $\boldsymbol{s}_t$ 是 hidden state, $\boldsymbol{a}_t$ 是输出, $f$ 是 RNN 的 transition function, $\pi_\theta$ 是输出层):  
    

![](https://pic2.zhimg.com/v2-5d4bb62f98d6a2c1aacc9bf8dc44b331_1440w.jpg)

我们无法人为设计 dynamic

那么我们真正的解决方案是什么呢? 使用 model-free 算法, 只是使用 model 来生成合成数据. 这可能看起来有些奇怪, 但实际上是效果相对的最好的, 实际上可以认为是 model-based acceleration of model-free RL.

## 2 Model-Free Learning with a Model

回顾我们熟悉的 policy gradient 算法: $\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i = 1}^{N} \sum_{t = 1}^{T} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_{i, t} \mid \boldsymbol{s}_{i, t}) \hat{Q}^{\pi}_{i,t}.\\$ 这实际上可以视作是一个 gradient estimator, 这里的好处是这里避免了 dynamics, 但这里的避免是相对的, 因为我们依然需要进行 sample. 但由于 model 的存在, 我们可以使用 model 来生成 sample, 从而减少在真实环境中的采样次数.

我们可以给出 model-based reinforcement learning version 2.5 (**Model-based RL via policy gradient**):

1.  运行 base policy $\pi_0(\boldsymbol{a}_t, \boldsymbol{s}_t)$, 收集 $\mathcal{D} = \{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')_i\}$,  
    
2.  学习 dynamic model $f(\boldsymbol{s}, \boldsymbol{a})$ 来最小化 $\sum_{i} \|f(\boldsymbol{s}_i, \boldsymbol{a}_i) - \boldsymbol{s}'_i\|^2$  
    
3.  利用 $f$ 与 policy $\pi_\theta(\boldsymbol{a} \mid \boldsymbol{s})$ 生成 $\{\tau_i\}$  
    
4.  利用 $\{\tau_i\}$ 改进 $\pi_{\theta}(\boldsymbol{a} \mid \boldsymbol{s})$, 重复 3- 4.  
    
5.  运行 $\pi_{\theta}(\boldsymbol{a} \mid \boldsymbol{s})$ 来收集数据, 添加到 $\mathcal{D}$, 重复 2- 5.  
    

这依然不是大多数人会实际使用的 model-based RL 算法, 这个算法依然有一些问题: 对于**长序列**, 我们的 model 误差会积累, 导致 distribution shift, 这样的偏移还可能与 policy 的偏移叠加在一起累积为更大的误差. 这样误差积累的速度与 imitation learning 中同样为 $O(\epsilon T^2)$.

![](https://picx.zhimg.com/v2-f483bd5d8588d1927d267a28f3d490df_1440w.jpg)

distribution shift

我们不能通过较短的 rollouts 来解决这个问题, 因为这实际上改变了问题的条件 (例如 horizon).

一个可能的做法是, 使用一些 real world rollouts, 然后从这些 rollouts 中选取一些 states, 从这些出发进行 short model rollouts. 此时

-   我们有更小的误差  
    
-   我们能见到所有的时间步  
    
-   得到的 state distribution 并不正确

![](https://pic1.zhimg.com/v2-08fc7e0f4fb152971d245f235af460aa_1440w.jpg)

长序列的 distribution shift (左), 短序列改变了问题条件 (中), 复合的 rollouts (右)

这一做法中的 distribution mismatch 是因为对于这些复合的 rollouts, 其真实采样的部分来源于较早的 policy, 而 model 采样的部分可能来自于较新的 policy.

事实上, 这可以进一步引申出在 model-based RL 中使用 on-policy 的 policy gradient 方法可能并不合适:

-   使用 model-based 的目的是为了减少真实环境中的采样, 故我们的 real world rollouts 的频率应该尽可能的低  
    
-   为了减小 distribution mismatch, 我们的 model rollouts 的频率应该尽可能的高  
    

因此使用 Q-learning 之类的 off-policy 方法是一个更好的选择, 这就得到了 model-based reinforcement learning version 3.0, 这也是人们会使用的 model-based RL 算法:

1.  运行 base policy $\pi_0(\boldsymbol{a}_t, \boldsymbol{s}_t)$, 收集 $\mathcal{D} = \{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}')_i\}$,  
    
2.  学习 dynamic model $f(\boldsymbol{s}, \boldsymbol{a})$ 来最小化 $\sum_{i} \|f(\boldsymbol{s}_i, \boldsymbol{a}_i) - \boldsymbol{s}'_i\|^2$  
    
3.  选择 $\mathcal{D}$ 中的 state $\boldsymbol{s}_i$, 利用 $f$ 生成短的 rollouts.  
    
4.  同时利用 real data 与 model data 改进 $\pi_{\theta}(\boldsymbol{a} \mid \boldsymbol{s})$, 使用 off-policy RL, 重复 3- 4.  
    
5.  运行 $\pi_{\theta}(\boldsymbol{a} \mid \boldsymbol{s})$ 来收集数据, 添加到 $\mathcal{D}$, 重复 2- 5.  
    

## 3 Dyna-Style Algorithms

在之前的算法 model-based RL v3.0 中, 我们实际还没有给出一些算法的具体形式. 符合这一框架的一个 classic 算法是 **Dyna**:

1.  给定 state $\boldsymbol{s}$, 使用 exploration policy 选择 action $\boldsymbol{a}$  
    
2.  观测到 $\boldsymbol{s}', r$, 得到转移 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}', r)$  
    
3.  更新 dynamic model 与 reward model  
    
4.  Q-update: $Q(\boldsymbol{s}, \boldsymbol{a}) \gets Q(\boldsymbol{s}, \boldsymbol{a}) + \alpha \mathbb{E}_{\boldsymbol{s'}, r} \left[r + \max_{\boldsymbol{a}'} Q(\boldsymbol{s}', \boldsymbol{a}') - Q(\boldsymbol{s}, \boldsymbol{a})\right]$  
    
5.  重复以下步骤 $K$ 次:  
    

6.  采样 $(\boldsymbol{s}, \boldsymbol{a}) \sim \mathcal{B}$  
    
7.  Q-update: $Q(\boldsymbol{s}, \boldsymbol{a}) \gets Q(\boldsymbol{s}, \boldsymbol{a}) + \alpha \mathbb{E}_{\boldsymbol{s'}, r} \left[r + \max_{\boldsymbol{a}'} Q(\boldsymbol{s}', \boldsymbol{a}') - Q(\boldsymbol{s}, \boldsymbol{a})\right]$, 只是这里的 $\boldsymbol{s}', r$ 是从 model 中采样得到的.  
    

这里有很多 design choice, 例如我们选择了 $(\boldsymbol{s}, \boldsymbol{a}) \sim \mathcal{B}$, 而不是选择 $\boldsymbol{s} \sim \mathcal{B}$, 然后用最新的 policy 来选择 $\boldsymbol{a}$. 值得注意的是, 这里的算法一方面可以提高数据利用率, 而 model 的存在可以让 $\mathbb{E}_{\boldsymbol{s'}, r} \left[r + \max_{\boldsymbol{a}'} Q(\boldsymbol{s}', \boldsymbol{a}') - Q(\boldsymbol{s}, \boldsymbol{a})\right]$ 估计更加准确 (原先是单点估计, 方差很大).

结合上述 Dyna 算法, 我们可以得到一个更加 general 形式的算法:

1.  收集数据, 包含一系列 transitions $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}', r)$, 放入 buffer $\mathcal{B}$  
    
2.  学习 dynamic model, (同时也可以学习 reward model)  
    
3.  重复以下步骤 $K$ 次:  
    

4.  采样 $\boldsymbol{s} \sim \mathcal{B}$  
    
5.  选择 $\boldsymbol{a}$ (从 $\mathcal{B}, \pi$ 或者 random)  
    
6.  利用 model 生成 $\boldsymbol{s}'$ (与可能有的 $r$)  
    
7.  使用 model-free RL 算法在 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}', r)$ 上进行更新  
    

8.  (optional) 进行 $N$ 步 model-based steps (在 $\boldsymbol{s}'$ 的基础上继续向前 $N$ 步)

![](https://pica.zhimg.com/v2-e9c86a7af7e8f5fbee7cb4c551714fe0_1440w.jpg)

红色的部分为基于 model 生成的轨迹, 黑色的部分为 buffer 中的轨迹

这里最后一步是可选的, 这样的做法可以让我们更加充分地利用 model, 从而加速 off-policy RL 的学习, 但是也有一些问题, 例如 model 的误差会积累, 这样的误差会导致 policy 的偏移.

## 4 A General View of Model-accelerated off-policy RL

-   process 1: data collection (放入 "replay buffer" $\mathcal{B}$) (包括 evict old data)  
    
-   process 2: target update  
    
-   process 3: Q-function regression  
    
-   process 4: model training (用 real data)  
    
-   process 5: model data collection (放入 buffer of model-based transitions) (每次改进 model 清除)  
    

这五个进程以不同的周期进行着, 类似于我们在 Q-Learning 中的几个 process 一样.  

![](https://pic4.zhimg.com/v2-aa97e4e9aed500e13d6dfbbd258192db_1440w.jpg)

### 4.1 Some Variants

也有一些变种, 这里使用了一些其他 design choice, 参见:

-   Model-Based Acceleration (MBA): Gu et al. Continuous deep Q-learning with model-based acceleration, 2016  
    
-   Model-Based Value Expansion (MVE): Feinberg et al. Model-based value expansion for efficient model-free reinforcement learning, 2018  
    
-   Model-Based Policy Optimization (MBPO): Janner et al. When to trust your model: model-based policy optimization, 2019  
    

上述提到的几个算法总的来说遵循以下过程:

1.  采取 action $\boldsymbol{a}$, 得到 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}', r)$, 添加到 replay buffer $\mathcal{B}$  
    
2.  从 $\mathcal{B}$ 中采样 $\{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}', r)\}$  
    
3.  用 $\{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}', r)\}$ 更新 dynamic model  
    
4.  从 $\mathcal{B}$ 中采样 $\{\boldsymbol{s}_j\}$  
    
5.  用 $\{\boldsymbol{s}_j\}$ 与 $\boldsymbol{a} = \pi(\boldsymbol{s})$ 进行model-based rollout  
    
6.  使用所有 rollout 中的 transitions 更新 $Q$ function  
    

![](https://pic1.zhimg.com/v2-516fc169b544036d35c6cbb85ad8112c_1440w.jpg)

与 Dyna 相比, 我们利用 model 生成了更长的轨迹

**Remark:**

-   这里的做法与 Dyna 类似, 但是由于使用了 model-based rollout 而不是仅仅利用 model 生成单个 $\boldsymbol{s}'$, 可以让本身的 model-free RL 更加 sample efficient.  
    
-   由于 model 本身的不准确, 这样的做法也有一些问题, 我们可以利用 uncertainty 来避免过度利用模型, 例如使用 bootstrap ensemble.  
    
-   不难注意到, 我们这里的 model-based 的 rollouts 可能比较奇怪, 因为其既不完全基于收集 real rollouts 的 policy, 同时也不完全基于当前的 policy, 尽管在实际中通常没有太大的问题, 但是这意味着我们不能长时间不更新 data.  
    

## 5 Multi-Step Models & Successor Representations

回顾我们介绍的两类强化学习算法: model-free RL 与 model-based RL. 在 model-free RL 中, 我们需要通过不断与环境交互来评估与改进 policy, 这样的做法 sample efficiency 较低. 在 model-based RL 中, 我们借助一个学习的 model 来更高效地利用数据, 评估与改进 policy. 但是 model-based RL 也有一些问题, 例如 model 的误差会积累, 这样的误差会导致 policy 的偏移.

在某种意义上, model-free RL 与 model-based RL 是两种极端:

-   **model-free RL**: 我们不试图学习任何关于 dynamic 的信息, 而事实上可能有一部分信息学习起来并不困难.  
    
-   **model-based RL**: 我们试图学习所有关于 dynamic 的信息, 从现实的角度考虑, 在一个极其复杂的环境中, 我们恐怕没办法学习到所有的信息.  
    

在本小节中, 我们将介绍一些中间的方法, 既尝试学习一些 dynamic 的信息, 同时也不会试图学习整个 dynamic. 这一类方法在迁移学习中得到了很大的重视.

### 5.1 Successor representations

在 model-based RL 中, 我们学习的 model 可以用来生成一些数据, 进而用这些数据来改进 policy. **归根结底, 我们的 model 是用来 evaluate policy 的**, 进而给出 policy 改进的方向. 我们接下来考虑什么样的表示可以用来 evaluate policy?

一个 representation 想要能够 evaluate policy, 也就是需要能够给出 $J(\pi) = \mathbb{E}_{\boldsymbol{s} \sim p(\boldsymbol{s}_1)} \left[V^\pi(\boldsymbol{s}_1)\right]\\$ 这里我们略作一些调整, 考虑仅仅依赖于 state 的 reward (而不是之前讨论的 state-action reward), 我们可以得到 $\begin{aligned}  V^\pi(\boldsymbol{s}_t) &= \sum^{\infty}_{t = t'} \gamma^{t - t'} \mathbb{E}_{p(\boldsymbol{s}_{t'} \mid \boldsymbol{s}_t)} \left[r(\boldsymbol{s}_{t'})\right]\\  &= \sum_{t = t'}^{\infty} \gamma^{t - t'} \sum_{\boldsymbol{s}} p(\boldsymbol{s}_{t'} = \boldsymbol{s} \mid \boldsymbol{s}_t) r(\boldsymbol{s})\\  &= \sum_{\boldsymbol{s}}\left(\sum_{t' = t}^{\infty} \gamma^{t' - t} p(\boldsymbol{s}_{t'} = \boldsymbol{s} \mid \boldsymbol{s}_t)\right) r(\boldsymbol{s}) \end{aligned} \\$ 这里我们考虑 **future state** $\boldsymbol{s}_{future}$ 的概念, 我们有两种方式理解:

1.  依据 $Geom(\gamma)$ 随机选择一个未来的时间步 $t'$, 这个时间步的 state $\boldsymbol{s}_{t'}$.  
    
2.  在每一时间步都有 $1 - \gamma$ 的概率停止, 我们停止时所在的 state.  
    

记 $p_\pi(\boldsymbol{s}_{future} = \boldsymbol{s} \mid \boldsymbol{s}_t) = (1 - \gamma) \sum_{t' = t}^{\infty} \gamma^{t' - t} p(\boldsymbol{s}_{t'} = \boldsymbol{s} \mid \boldsymbol{s}_t),\\$ $p_\pi(\boldsymbol{s}_{future} = \boldsymbol{s} \mid \boldsymbol{s}_t)$ 的理解方式对应于以下两种:

1.  依据 $Geom(\gamma)$ 随机选择一个未来的时间步 $t'$, 然后 evaluate $\boldsymbol{s}_{t'}$ 恰好为 $\boldsymbol{s}$ 的概率.  
    
2.  在每一时间步都有 $1 - \gamma$ 的概率停止, 我们恰好停在 $\boldsymbol{s}$ 的概率.  
    

事实上, 应用这一概念, 我们可以得到 $\begin{aligned}  V^\pi(\boldsymbol{s}_t) &= \frac{1}{1 - \gamma} \sum_{\boldsymbol{s}} p_\pi(\boldsymbol{s}_{future} = \boldsymbol{s} \mid \boldsymbol{s}_t) r(\boldsymbol{s})\\  &= \frac{1}{1 - \gamma} \sum_{\boldsymbol{s}} \mu_{\boldsymbol{s}}^\pi(\boldsymbol{s}_t) r(\boldsymbol{s})\\  &= \frac{1}{1 - \gamma} \mu^\pi(\boldsymbol{s}_t)^T \overrightarrow{r}. \end{aligned}\\$ 其中我们记 $\mu^\pi_i(\boldsymbol{s}_t) = p_\pi(\boldsymbol{s}_{future} = i \mid \boldsymbol{s}_t)$, 其整合为向量形式即为 $\mu^\pi(\boldsymbol{s}_t)$, 这一向量形式的表示称为 **successor representation**. 这一表示同时包含 model 与 policy (以 value function 的形式) 的信息. 其与 reward 的点积则可以还原得到 value function 的信息.

事实上, 我们可以对 $\mu(\boldsymbol{s}_t)$ 做类似于 Bellman backup 的更新 (考虑 reward 为 $(1 - \gamma) \delta(\boldsymbol{s}_t = i)$): $\begin{aligned}  \mu_i^\pi(\boldsymbol{s}_t) &= (1 - \gamma) \sum_{t' = t}^{\infty} \gamma^{t' - t} p(\boldsymbol{s}_{t'} = i \mid \boldsymbol{s}_t)\\  &= (1 - \gamma) \delta(\boldsymbol{s}_t = i) + \gamma \sum_{\boldsymbol{s}} p(\boldsymbol{s} \mid \boldsymbol{s}_t) \mu_i(\boldsymbol{s})\\  &= (1 - \gamma) \delta(\boldsymbol{s}_t = i) + \gamma \mathbb{E}_{\boldsymbol{a}_t \sim \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t), \boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)} \left[\mu_i^\pi(\boldsymbol{s}_{t + 1})\right], \end{aligned} \\$ 在实际中上述过程可以向量化, 同时对所有 $i$ 操作, 即 $\mu^\pi(\boldsymbol{s}_t) = (1 - \gamma) \boldsymbol{e}_{\boldsymbol{s}_t} + \gamma \mathbb{E}_{\boldsymbol{a}_t \sim \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t), \boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)} \left[\mu^\pi(\boldsymbol{s}_{t + 1})\right].\\$ 其中 $\boldsymbol{e}_{\boldsymbol{s}_t}$ 是一个 one-hot 向量.

**Remark:** 在上述过程中我们引入了 successor representation, 这一概念的好处在于其同时包含了 model 与 value function 的信息, 我们同样给出了其具有的基本性质, 然而值得注意的是:

-   我们尚不明确 学习 successor representation 是否比 model-free RL 更加容易  
    
-   尚不明确如何 scale 到更大的 state space  
    
-   并不知道如何将其扩展到 continuous state space  
    

在接下来的讨论中, 我们首先考虑如何扩展到更大的 state space.

### 5.2 Successor features

不难发现, 如果 state space 本身很大, 那么我们的 $\mu^\pi(\boldsymbol{s}_t)$ 也会很大, 我们的 $\mu^\pi(\boldsymbol{s}_t)$ 为 $|\mathcal{S}|$ 的集合上的分布.

事实上, 我们可以对 state 进行压缩, 考虑一个映射 $\phi: \mathcal{S} \rightarrow \mathbb{R}^d$ (这可以是人为设计的, 也可以通过 AE 等方式学习), 考虑通过 $r(\boldsymbol{s}) = \sum_j \phi_j(\boldsymbol{s}) w_j = \phi(\boldsymbol{s})^T \boldsymbol{w}\\$ 来近似地表示 reward, 这里的 $\boldsymbol{w}$ 是一个 $d$ 维向量.

那么 $\begin{aligned}  V(\boldsymbol{s}_t) &= \frac{1}{1 - \gamma} \mu^\pi(\boldsymbol{s}_t)^T \overrightarrow{r}\\  &= \frac{1}{1 - \gamma} \mu^\pi(\boldsymbol{s}_t)^T \sum_j \overrightarrow{\phi}_j w_j\\  &= \frac{1}{1 - \gamma} \sum_j \mu^\pi(\boldsymbol{s}_t)^T \overrightarrow{\phi}_j w_j \end{aligned}\\$ 这里记 $\mu^\pi(\boldsymbol{s}_t)^T \overrightarrow{\phi}_j = \psi_j^\pi(\boldsymbol{s}_t)$, 对 $j$ 整理得到一个 $\mathbb{R}^d$ 上的向量 $\psi^\pi(\boldsymbol{s}_t)$, 我们称 $\psi^\pi(\boldsymbol{s}_t)$ 为 **successor features**. 整理得到 $V(\boldsymbol{s}_t) = \frac{1}{1 - \gamma} \sum_j \psi_j^\pi(\boldsymbol{s}_t) w_j = \frac{1}{1 - \gamma} \psi^\pi(\boldsymbol{s}_t)^T \boldsymbol{w}.\\$ 这个表达式中的 $\psi^\pi$ 与 $\boldsymbol{w}$ 都是通过学习得到的, 其中 $\boldsymbol{w}$ 的学习就是一个监督学习的过程 (在已知 $\phi$ 的情况下学习一个权重). 我们使用 Bellman backup 来学习 $\psi_j^\pi(\boldsymbol{s}_t)$, 这里考虑给之前向量化的 Bellman backup 点乘上 $\overrightarrow{\phi}_j$: $\psi_j^\pi(\boldsymbol{s}_t) = \phi_j(\boldsymbol{s}_t) + \gamma \mathbb{E}_{\boldsymbol{a}_t \sim \pi(\boldsymbol{a}_t \mid \boldsymbol{s}_t), \boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)} \left[\psi_j^\pi(\boldsymbol{s}_{t + 1})\right].\\$ 事实上上一部分我们讨论的 $\mu^\pi$ 的 Bellman backup 可以视作是使用 $\phi_i(\boldsymbol{s}) = (1 - \gamma) \delta(\boldsymbol{s} = i)$ 且 $|\mathcal{S}| = d$ 的特例.

上述设计的 successor features 类似于 value function, 我们也可以构造类似于 Q-function 的 successor features: 如果依然考虑 $r(\boldsymbol{s}) \approx \phi(\boldsymbol{s})^T \boldsymbol{w}$, 则 $Q^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t) \approx \psi^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t)^T \boldsymbol{w}\\$ 我们同样可以写出 Bellman backup: $\psi_j^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t) = \phi_j(\boldsymbol{s}_t) + \gamma \mathbb{E}_{\boldsymbol{s}_{t + 1} \sim p(\boldsymbol{s}_{t + 1} \mid \boldsymbol{s}_t, \boldsymbol{a}_t), \boldsymbol{a}_{t + 1} \sim \pi(\boldsymbol{a}_{t + 1} \mid \boldsymbol{s}_{t + 1})} \left[\psi_j^\pi(\boldsymbol{s}_{t + 1}, \boldsymbol{a}_{t + 1})\right].\\$ **Side Note:** 实际上原论文中 $\phi$ 输入为 $(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{a}')$, $\psi$ 输入为 $(\boldsymbol{s}, \boldsymbol{a})$, 在上述推导中我们做了一些简化, 但完全不影响结果.

最后我们讨论其使用方式:

-   **Idea 1: 用于快速恢复 Q-function**  
    
-   训练 $\psi_j^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t)$ (利用 Bellman backups)  
    
-   获取 $\{\boldsymbol{s}_i, r_i\}$ samples  
    
-   get $\boldsymbol{w} \gets \arg\min_{\boldsymbol{w}} \|\phi(\boldsymbol{s}_i)^T \boldsymbol{w} - r_i\|^2$  
    
-   恢复 $Q^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t) \approx \psi^\pi(\boldsymbol{s}_t, \boldsymbol{a}_t)^T \boldsymbol{w}$  
    

而我们的策略就是 $\pi'(\boldsymbol{s}) = \arg\max_{\boldsymbol{a}} \psi^\pi(\boldsymbol{s}, \boldsymbol{a})^T \boldsymbol{w}.\\$ **Note:** 这里计算得到的并不是 optimal Q-function, 实际上这是当前的 policy 的 Q-function, 因此 $\pi'$ 只是 $\pi$ 的一步 policy iteration 的结果.

-   **Idea 2: recover many Q-functions** (更好的想法)  
    
-   训练一系列 $\psi_j^{\pi_k}(\boldsymbol{s}_t, \boldsymbol{a}_t)$ 对于一系列 $\pi_k$ (利用 Bellman backups)  
    
-   获取 $\{\boldsymbol{s}_i, r_i\}$ samples  
    
-   get $\boldsymbol{w} \gets \arg\min_{\boldsymbol{w}} \|\phi(\boldsymbol{s}_i)^T \boldsymbol{w} - r_i\|^2$  
    
-   恢复 $Q^{\pi_k}(\boldsymbol{s}_t, \boldsymbol{a}_t) \approx \psi^{\pi_k}(\boldsymbol{s}_t, \boldsymbol{a}_t)^T \boldsymbol{w}$  
    

我们的策略就是 $\pi'(\boldsymbol{s}) = \arg\max_{\boldsymbol{a}} \max_k \psi^{\pi_k}(\boldsymbol{s}, \boldsymbol{a})^T \boldsymbol{w}.\\$ 换言之就是找到每一个 state 中的最高 reward policy, 因此我们在每一个状态都在 $k$ 个中最好的 policy 上进行了一步改进.

**Side Note:** 实际上, 从 transfer learning 的角度来看, 在 MDP 的其他设定不变以及 $\phi: \mathcal{S} \rightarrow \mathbb{R}^d$ 不变的情况下, 每一个不同的 $\boldsymbol{w}$ 指定了一个不同的 task, 论文中给出了以下的理论结果:

**Theorem 1**. _考虑 $M^\phi$ 为当前设定下的任务空间, 我们考虑已经学习了 $M_j \in M^\phi, j = 1,\ldots,n$ 这些任务, 它们对应的 optimal Q-functioin 为 $Q_i^{\pi_j^\ast}(\boldsymbol{s}, \boldsymbol{a})$, 学习得到的 Q-function 为 $\tilde{Q}_i^{\pi_j^\ast}(\boldsymbol{s}, \boldsymbol{a})$, 且它们满足 $\forall \boldsymbol{s} \in \mathcal{S}, \boldsymbol{a} \in \mathcal{A}$_ _$\left|\tilde{Q}_i^{\pi_j^\ast}(\boldsymbol{s}, \boldsymbol{a}) - Q_i^{\pi_j^\ast}(\boldsymbol{s}, \boldsymbol{a})\right| < \epsilon\\$ 对于一个新任务 $M_i \in M^\phi$, 我们采用 $\pi(\boldsymbol{s}) = \arg\max_{\boldsymbol{a}} \max_j \tilde{Q}_i^{\pi_j^\ast}(\boldsymbol{s}, \boldsymbol{a})\\$ 的方式作直接的迁移, 那么我们有 $Q_i^{\pi^\ast_i}(\boldsymbol{s}, \boldsymbol{a}) \leq \frac{2}{1 - \gamma} (\phi_{\max} \min_j\|\boldsymbol{w}_i - \boldsymbol{j}\| + \epsilon)\\$ 其中 $\phi_{\max} = \max_{\boldsymbol{s}, \boldsymbol{a}} \|\phi(\boldsymbol{s}, \boldsymbol{a})\|$, 其中 $\|\cdot\|$ 是内积诱导的范数._

本部分介绍的 successor features 参见: Barreto et al. Successor Features for Transfer in Reinforcement Learning. 2016

### 5.3 Continuous Successor Representations

在上一小节中, 我们讨论了如何将 successor representation 扩展到更大的 state space, 接下来我们介绍如何将 successor representation 的思想扩展到 continuous state space.

回顾我们 successor representation $\mu_i(\boldsymbol{s}_t) = p(\boldsymbol{s}_{future} = i \mid \boldsymbol{s}_t)$, 在连续 state space 中, 我们要学习的变成了一个 $\mathcal{S}$ 到 $\mathcal{S}$ 上连续分布的映射, 这一映射的学习是非常困难的. 但是, 我们其实有其他方式来估计这一概率密度:

**Idea:** 我们把学习 successor representation 的过程转化为学习一个 binary classifier 的过程, 考虑 $p^\pi(F = 1 \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{future}).\\$ 这里记 $F = 1$ 意味着如果从 $\boldsymbol{s}_t, \boldsymbol{a}_t$ 出发采用 policy $\pi$, $\boldsymbol{s}_{future}$ 是一个 future state, 值得注意的是, 我们这里又用回了依赖 action 的 reward.

考虑正样本 $\mathcal{D}_+ \sim p^\pi(\boldsymbol{s}_{future} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$ 从 future state 的分布中采样, 负样本 $D_- \sim p^\pi(\boldsymbol{s})$ 从所有 $\pi$ 可能到达的 state 中采样, 于是基于 optimal classifier 的性质, 有: $p^\pi(F = 1 \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{future}) = \frac{p^\pi(\boldsymbol{s}_{future})}{p^\pi(\boldsymbol{s}_{future}) + p^\pi(\boldsymbol{s}_{future})}\\$$p^\pi(F = 0 \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{future}) = \frac{p^\pi(\boldsymbol{s}_{future})}{p^\pi(\boldsymbol{s}_{future}) + p^\pi(\boldsymbol{s}_{future})}\\$ 这里训练上述的 classfier 比直接学习 successor representation 更加容易. 并且如果我们能够训练这一 classifier, 我们能够从这个 classifier 中恢复 $p^\pi(\boldsymbol{s}_{future} \mid \boldsymbol{s}_t, \boldsymbol{s}_t)$: $\frac{p^\pi(F = 1 \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{future})}{p^\pi(F = 0 \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{future})} p^\pi(\boldsymbol{s}_{future}) = p^\pi(\boldsymbol{s}_{future} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)\\$ 这里的 successor representation 估计的是概率密度. 而由于 $p^\pi(\boldsymbol{s}_{future})$ 是一个与 $\boldsymbol{a}_t, \boldsymbol{s}_t$ 无关的常量, 尽管这个量非常难计算, 因此它不影响我们基于 $p^\pi(F = 1 \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{future}) / p^\pi(F = 0 \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{future})$ 选择 action 等等.

我们的算法是如下的 **The C-Learning algorithm**:

1.  **获取负样本**: 采样 $\boldsymbol{s} \sim p^\pi(\boldsymbol{s})$ (运行 policy, 从 trajectory 采样)  
    
2.  **获取正样本**: 采样 $\boldsymbol{s} \sim p^\pi(\boldsymbol{s}_{future} \mid \boldsymbol{s}_t, \boldsymbol{a}_t)$ (基于前面提到的 $\boldsymbol{s}_{future}$ 的两种理解方式, 其中一种就是采样 $\boldsymbol{s}_{t'}$, 其中 $t' = t + \Delta, \Delta \sim Geom(\gamma)$)  
    
3.  **训练分类器**: 更新 $p^\pi(F = 1 \mid \boldsymbol{s}_t, \boldsymbol{a}_t, \boldsymbol{s}_{future})$, 使用 SGD 和 cross entropy loss.  
    

**Side Note:** 这里介绍的版本是一个 on-policy 算法, 我们同样可以推导出 off-policy 算法.

参见: Eysenbach, Salakhutdinov, Levine. C-Learning: Learning to Achieve Goals via Recursive Classification. 2020

## 6 Summary

在本节中, 我们

-   介绍了如何使用 model 进行 close-loop policy 学习, 介绍了 backpropagate through time 方法可能存在的问题, 并引出了 model-accelerated RL 的概念.  
    
-   介绍了这一类 model-accelerated RL 的例子, 如 **Dyna**.  
    
-   介绍了一种理解 model-accelerated RL 的 general 方式, 并介绍了一些变种.  
    
-   介绍了 **successor representation** 这种介于 model-free 与 model-based 之间的方法, 介绍了扩展的 **successor features** 与 **C-Learning**, 并讨论了其在 transfer learning 中的应用.