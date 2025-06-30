## 1 Brief Review

我们首先简单回顾一下过去介绍的内容:

-   从根本上来说, 我们之前介绍的是 learning-based control method, 这包括 imitation learning (learning from demonstration) 和 reinforcement learning (learning from reward).  
    
-   之后我们介绍了 RL 中的一些 "classic" model-free RL 算法, 例如 policy gradient 和 value-based methods以及介于它们之间的 actor-critic 算法. 同时我们也介绍了这几类算法的对应例子, 例如 TRPO 和 PPO, DQN 以及 SAC.  
    
-   之后我们介绍了另一类可能的算法: model-based control (例如 LQR), 这些方法未必是 learning-based methods, 但它们也可和 learning-based methods 结合, 得到 model-based RL w/o policy (例如 MPC). 而这样的方式也可也和 RL 的 policy learning 结合, 得到 model-based RL with policy (例如 Dyna-Q).  
    
-   我们剩下讨论的一些内容在某种意义上正交于我们之前讨论的内容, 例如 exploration, unsupervised RL (如 skill discovery). 我们还介绍了一些工具例如 probabilistic inference, 以及利用其建立的 control as inference 的框架, 在这基础上我们可以得到 inverse RL 的算法.  
    
-   我们同样还介绍了 offline RL, RL with sequence model, meta-learning 等等其他内容.  
    

![](https://pic2.zhimg.com/v2-56d66cee64d25163ccabcd79d77924af_1440w.jpg)

课程内容的思维导图

## 2 Challenges in Deep Reinforcement Learning

### Challenges with core algorithms:

-   Stability: 算法是否收敛?  
    
-   Efficiency: 收敛需要多少时间/sample?  
    
-   Generalization: 在收敛后, 其泛化性怎么样?  
    

### Challenges with assumptions:

-   问题的 formulation 是否合理, 这个问题是否可以建模为一个 RL 问题? 还是说有更好的假设?  
    
-   supervision 的来源是什么? 例如对于 imitation learning, supervision 来自于 demonstration, 而对于 RL 来说则是来自于 reward.  
    

接下来我们从以上提到的一些角度进行讨论.

### 2.1 Stability and hyperparameter tuning

对于 RL 来说, 我们的数据需要自己获取, 而不是 i.i.d. 的, 同时我们的目标是优化一个 objective 而不是简单地学习 ground truth. 这些额外的调整意味着我们的 RL 对一系列超参数更加敏感, 这给我们设计稳定的 RL 算法带来了挑战.

### Value function based methods

对基于 Q-learning estimation 的算法通常并不能保证收敛, 同时我们需要有非常多的超参数来实现稳定, 例如 target network delay, replay buffer size, clipping, learning rate 等等.

当然深度学习中的很多技巧例如 更大规模的网络, normalization, data augmentation (参见 Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels, Ilya Kostrikov, Denis Yarats, Rob Fergus) 都可以让训练更稳定.

Some Open Problems: 尽管基于 classic machine learning theory 会认为深度学习会出现严重的过拟合, 但是实际上深度学习通常能够表现地很好, 这说明存在着某种潜在的 regularizing effect, 例如 SGD 本身. 然而 value-based method 事实上不是 gradient descent, 因此一个可能的研究方向就是这样的一种 "magic" 是否在 value-based methods 中同样存在.

### Policy gradient methods

对于 policy gradient 相关的算法, 我们通常有更加深入的研究, 也有一些收敛性的保证. 但是代价是我们有更高的 variance. 其他的 RL 算法通常都有来自于 value function 的 bias, 而 policy 则有很大的 variance 而没有 bias. 这意味着我们需要大量的 samples 来进行训练, 同样我们需要考虑的超参数有 learning rate, batch size, baseline design.

### Model-based RL

这一类算法表面上看这会是一个稳定的选择, 因为 model 学习的过程是 supervised learning. 然而, 这一类方法中 model 在训练中不断改变, 这里的问题是, model 变得准确不能直接使得 policy 更好:

虽然如果我们的 model 是 perfect 的, 此时可以得到 optimal policy, 但是 model 变好并不一样. 有可能 model 以一种特定的方式改善, 例如可能在某一些地方准确性略微下降, 而这给 policy 带来灾难性的后果, 或者说, model 的不同 error 虽然在训练 model 的监督学习意义下是一样的, 但是对 policy 影响并不一致.

在 model-based policy learning 一节中我们介绍了使用 backpropagate through time 的方法来更新 policy, 但这种方法效果并不好, 因此我们通常还是会使用 model-free 的算法, 将 model 作为一种加速的方式.

model based methods 还有一些其他的问题, 例如 policy 可能会利用 model 的一些错误, 在这个意义上这种方法像是一种对抗性的过程.

### 2.2 The challenge with sample complexity

不同的算法在 sample efficiency 上有很大差异

-   gradient-free methods (例如 NES, CMA 等)  
    
-   fully online methods (例如 A3C), 这样的方法大约需要 $10^8$ 个 steps, 需要现实中大约 $$$15$$$ 天.  
    
-   policy gradient methods (例如 TRPO), 这样的算法通常需要 $10^7$ 个 steps, 需要现实中大约 $1.5$ 天.  
    
-   replay buffer value estimation methods (Q-learning, DDPG, NAF, SAC 等), 这样的方法大约需要 $$$10^6$$$ 个 steps, 对于一些简单的任务可能只需要几小时.  
    
-   model-based deep RL (PETS, guided policy search)  
    
-   model-based "shallow" RL (例如 PILCO), 但是使用的是没有办法扩展到大规模的方法, 例如 Gaussian Process.

![](https://pic4.zhimg.com/v2-0c6acc09506d7bb68813b776418ab3c3_1440w.jpg)

不同算法在 sample efficiency 上的巨大差异

直观来说, sample efficiency 自然是越高越好, 那么为什么我们还会选择使用那些不那么 efficient 的算法呢?

-   很多时候我们可以并行收集数据, 例如对于 robot task, 我们可以使用多个机械臂, 对于模拟环境更是如此. 在一些情况下, 收集数据的成本可能低于训练一个 model 的成本.

但是, 在 real-world learning 以及一些特定的问题 (比如 AI4S) 中, 收集数据的成本可能很高, 此时算法的 sample efficiency 就变得非常重要.

尽管整体来说, 目前各种算法的 sampling efficiency 也在不断改进, 但是当我们尝试广度更大的问题时, 这会成为一个更加严重的问题.

### 2.3 Scaling up deep RL & generalization

我们会发现一个很有意思的现象:

-   在目前的深度学习中, 我们会在大规模的数据上训练, 并且通常强调任务的 diversity, 使用 generalization 能力作为评估指标.  
    
-   在 RL 中, 通常我们只会在一个小规模的数据上训练, 且我们主要关注单任务上的效果, 也只会用 performance 来评估.  
    

为什么我们不能简单像一般的深度学习那样扩大 RL 的规模呢? 这实际上和 RL 本身的 workflow 有密切的关系:

-   对于 supervised learning 来说, 通常只需要从真实世界中获取数据一次, 之后就需要通过一个算法/模型训练即可. 如果我们不满意非常简单, 无论是调整超参数还是调整模型和算法, 都只需要重新在数据集上训练就行.

![](https://picx.zhimg.com/v2-b996c12bb182206347f366ba75fcdff7_1440w.jpg)

监督学习的常见 workflow

-   对于典型的 RL 来说, 数据来源于与环境的反复交互, 通常我们每一次调整算法都需要重复这一漫长的过程, 且现实中还有一个 outer loop 是人. 如果扩大 RL 的规模则会让整个过程更加不现实.

![](https://picx.zhimg.com/v2-af7a0d9fc72e8d1d0c124ba3e1411ff9_1440w.jpg)

通常情况下 RL 的实际 workflow

因此对于 RL 的改进不能仅仅局限在具体的方法上, 也可也从 workflow 的角度使得整个 RL 更加可行. 本课程中介绍了很多相关的研究方向:

-   offline RL: 如果我们能从一个预先收集好的 dataset 中学习, 那么整个 RL 的 workflow 就会变得和 supervised learning 一样, 即使我们修改了算法, 也不需要重新收集数据.  
    
-   meta-learning: 这同样是一种 workflow 的改进. 如果我们能够得到一个 meta-learner, 那么我们可以从过去的经验中更快的学习. 另一方面, pretraining + finetuning 也可以看作是 meta-learning 的一种方式.  
    

刚才我们主要侧重在 scaling up 上, 而在 generalization 层面, 我们依然有很长的路要走. 目前如果训练一个 humanoid 在完全平坦的平面上奔跑, 换算在真实世界中的时间需要 6 天. 但现实世界远比这样的环境复杂, 难道对于每一种环境, 我们都要专门训练一次吗?

为了完成多样的任务, 一个可能的思路是我们之前讨论过的 transfer learning 和 meta-learning 的方法. 一个简单的例子是 multi-task learning: 在这样的算法中, 可能会有更严重的 variance 和 sample efficiency 问题, 一方面我们可以考虑直接求解一个 augmented MDP, 也可也考虑专门设计一些算法来处理 multi-task 的问题.

![](https://pic1.zhimg.com/v2-aadb387fd5a15613d2ddb54fe5592bda_1440w.jpg)

通过在多个 MDP 的初始状态分布中采样来选择执行的任务, 以实现 multi-task learning, 这样的做法可能会有更严重的 variance 和 sample efficiency 问题

### 2.4 Assumptions: Where does the supervision come from?

我们都清楚 RL 的建模, 在 RL 中我们会有一个 reward function, 这是 supervision 的来源. 但是很多时候归根到底 reward 是哪里来的? 是我们设计的.

在这种思路下, 如果我们要进行 multi-task learning, 那么我们就得给每个任务设计一个 reward, 在一些问题中 reward 非常简单, 但是在更多情况下, reward 作为一种 supervision 是很难设计的, 例如让 robot 倒水, 一个稀疏的 reward 是 naive 的, 但问题是我们的 agent 从中往往无法学到任何东西.

实际上很多时候 reward 并不是我们唯一的选择, 很多时候可以有其他的 supervision 的来源:

-   Demonstration: 参见 Muelling, K et al. (2013). Learning to Select and Generalize Striking Movements in Robot Table Tennis.  
    
-   Language: 参见 Andreas et al. (2018). Learning with latent language.  
    
-   Human preference: 参见 Christiano et al. (2017). Deep reinforcement learning from human preferences  
    

还有一些可能的方式, 例如能否自动的生成 objective? (利用 automatic skill discovery)

关于 supervision 的选择, 有一些值得思考的问题: 我们的 supervision 需要告诉 agent what to do 还是 how to do?

-   对于 demonstration 来说, 我们的 supervision 不仅回答了 what, 也回答了 how, 例如我们给出了一个 robot 倒水的 demonstration, 那么 robot 不仅知道了我们要倒水, 还知道了怎么倒水.  
    
-   对于 reward function 来说, 我们的 supervision 通常仅仅回答了 what, 例如仅当 robot 倒水成功时才会给出 reward, 但是并没有告诉 robot 怎么倒水. 但是如果我们给出了一个很好的 reward function, 那么也可以部分地回答了 how 的问题.  
    
-   这其实有一个 tradeoff, 我们希望我们的算法能够找到更好的 solution, 那么我们不应该有过多细节的 supervision. 但是如果我们的 supervision 过于 high-level, 也可能导致整个 learning 变得非常困难.  
    

从上述关于 supervision 的讨论中我们其实想要引出一个更加重要的事情: 很多时候我们不要拘泥于 RL 本身已有的很多 formulation, 它们并不是完全不能改变的. 我们需要仔细考虑它们是否真的适合 problem setting, 比如我们可以考虑的是:

-   数据是什么?  
    
-   goal 是什么? (reward/ demonstration, preference)  
    
-   supervision 是什么? 这可能并不等同于 goal, 有可能我们只是想提供一些 hint, 用一些 demonstration 作为指导而不是 goal, 这是一个研究的 open area.  
    

## 3 Philosophical Perspective on Deep RL

这一部分我们将提供一些 Perspective, 关于 Deep RL 理解的一些 philosophy:

### 3.1 Reinforcement Learning as an Engineering Tool

我们实际上可以将 RL 视作一种 engineering tool.

通常情况下, 对于一个控制系统的工程问题来说, 我们会在纸上写下一系列关于系统的数学方程, 然后求解方程组来从想要的结果中反推出控制方式. 但是这样的反推过程并不容易, 例如对于复杂的系统, 我们可能会有一个复杂的方程组.

![](https://pic1.zhimg.com/v2-63c8f1c76fb7633ac9695f9d0db2adf2_1440w.jpg)

engineering a control system

但是虽然反推控制方式非常困难, 但是我们可以把这些方程组写进一个 simulator 中, 通过数值的方式来计算复杂的系统将会如何演化, 这就像是我们在 RL 中的 simulator.

当我们在这样的 simulator 中运行 RL 算法时, 我们实质上就在尝试从这些方程中反推出控制方式, 但是我们并不是通过人来求解这个问题, 而是通过 machine learning 的方式来进行.

因此我们的 RL 提供了一种强大的 engineering tool, 也就是一切我们能够 simulate 的都可以 control.

-   原先的 engineering 方式是, 我们会建模问题, 并进行 control.  
    
-   而 RL 给我们带来了另一种 engineering 的方式: 我们建模问题, 并进行 simulation, 然后运行 RL 算法, 得出控制的方式.  
    

Remark:

-   从这一角度理解的启示是, 我们要开发更加高效的 simulator, 并且开发能够更好地利用 simulation 的 RL 算法.  
    
-   但是存在的问题是我们依然需要 simulate.  
    

### 3.2 Reinforcement Learning and the Real World

在这一部分中, 我们想提出的是在真实世界中进行 RL 才是我们的真正目标, 具体来说我们先考虑到以下几个例子:

Example 1. _Moravec's paradox_

_一个研究 RL 的 motivation 是这样的 Moravec's paradox:_

_AI 可以在棋类比赛上胜过世界冠军, 但是并不是真的 robot 在下棋, 而是由现实中的人代替 AI 进行实际操作. 这是一个非常怪异的现象, 我们的 model 的"智能"足够在棋盘上赢过世界冠军, 却不能够完成任何一个人类都可以做到的实际的棋子移动._

![](https://pic3.zhimg.com/v2-0997205abc8394d88b469ed9421477d4_1440w.jpg)

Moravec&#39;s paradox

_这看似是一个悖论, 但如果从另一角度看, 则是完全 make sense 的: 在一个纯粹的智力游戏上胜过人类, 可能并没有我们想象的那么困难. 之所以我们在这方面比不过 AI 只是因为我们并不擅长它们, 与之相比我们只是远远地擅长移动我们的身体, 以至于我们觉得后者是理所当然的._

_Moravec's paradox 看起来是关于 AI 的 statement, 实际上可以理解为是关于 physical universe 的 statement, 也就是现实世界是一个 "hard" universe:_

-   _在 easy universe 例如棋类游戏中, motor control 和 perception 问题是不存在的._  
    
-   _在我们所在的 hard universe 中, motor control 和 perception 是非常困难的._  
    

Example 2. _另一个例子是考虑一个人在荒岛上生存, 这个问题中有一系列特点:_

-   _极少的外部 supervision 告诉我们应当做什么_  
    
-   _大量意料之外的情境需要 adaptions_  
    
-   _必须要自发的找到解决问题的方案_  
    
-   _我们必须要存活足够长的时间来发现它们, 与 Atari games 不同的是, 在现实中我们只有一条命._  
    

_从这个角度出发, 现实世界这个 hard universe 的另一个困难在于, 现实世界中充满了 variability 和 unexpected 的东西, 在训练数据中"永远不可能"出现的那些东西在现实中随时有可能发生._

_换言之, 在 easy universe 中:_

-   _success = high reward (optimal control)_  
    
-   _closed world, rules are Known_  
    
-   _lots of simulation_  
    
-   _Main question: can RL algorithms optimize really well_  
    

_在 hard universes 中:_

-   _success = survival (good enough control)_  
    
-   _open world, everything must come from data_  
    
-   _no simulation (rules are unknown)_  
    
-   _Main question: can RL generalize and adapt._  
    

上述的几个例子直观展现了现实世界这个 hard universe 中的问题与我们通常 RL 研究中的 easy universe 中的问题的不同. 很显然, 我们的最终目标应当是 RL 能够在现实世界这样的 hard universe 中取得好的效果. 但是在目前的 RL 研究中, 我们通常只考虑 easy universe 中的任务, 我们需要更多地尝试解决这些 hard universe 中的问题.

但是如果要解决现实世界中的问题, 我们需要考虑一些问题:

-   如何告诉 RL agent 需要做什么呢? 现实中没有 scores. 如果我们的目标是存活, 那么这样的 feedback 过于 delayed 了  
    
-   如何在持续的环境中完全自主的学习? 在现实中我们不可能 reset world 从头再来.  
    
-   在环境改变时我们如何保持 robust?  
    
-   使用过去经验和数据 generalize 的正确方式是什么?  
    
-   什么是利用 prior experience 进行 bootstrap exploration 的正确方式?  
    

接下来我们从几个 robotic task 的例子来给出一些解决上述问题的启发:

Example 3. _Other ways to communicate objective_

_除了 reward function 之外, 有没有其他方式来传达目标呢? 实际上, 一个可行的做法是 learning from preference. 这里我们没有固定的 reward function, 而是像 RLHF 那样让人类评价哪个 action 更符合指令._

![](https://pic2.zhimg.com/v2-b1745e407c4c32df90f17aa5f76eb1e3_1440w.jpg)

Deep reinforcement learning from human preferences

_参见: Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei. Deep reinforcement learning from human preferences. 2017._

Example 4. _Learn fully autonomously_

_在现实中如果我们部署一个 robot manipulation task, 那么我们需要构建一整套系统 (或者人为复位), 使得我们的机械臂可以复位然后不断尝试, 这是相当繁琐的, 并且实质上并不利于 policy 的泛化性._

_一个可行的思路是考虑 multi-task, 特别的处理方式是让任务之间相互转化, 这里以倒咖啡为例:_

_Task 1 很显然是"倒咖啡"._

_但是如果失败了呢? 那么通常情况下需要一个人把杯子复位然后再试一遍, 但是我们可以让 Task 2 是"捡起杯子"._

_而如果成功了呢? 可以让 Task 3 是"把杯子放到一边, 换一个杯子", 这个过程中如果失败了, 那么有得到 Task 4 "把洒了的咖啡清理掉". 以此类推. 如果我们同时学习多个任务, 那么我们的每一次失败就给了我们一个学习全新任务的机会._

_这里的例子是我们设计了一个多阶段且可以相互转化的任务, 使得每一阶段如果 agent 失败了, 他都马上有一个新的任务可以练习, 就不需要人为复位._

_参见:_

-   _Nagabandi, Konolige, Levine, Kumar. Deep Dynamics Models for Learning Dexterous Manipulation. CoRL 2019_

![](https://picx.zhimg.com/v2-d72c86a54a95d40e31a3b4c4645cd517_1440w.jpg)

Deep Dynamics Models for Learning Dexterous Manipulation

-   _Gupta, Yu, Zhao, Kumar, Rovinsky, Xu, Devlin, Levine. Reset-Free Reinforcement Learning via Multi-Task Learning: Learning Dexterous Manipulation Behaviors without Human Intervention. 2021._

![](https://picx.zhimg.com/v2-4b01bcf4840ec500cf17c224d6b938ed_1440w.jpg)

Reset-Free Reinforcement Learning via Multi-Task Learning: Learning Dexterous Manipulation Behaviors without Human Intervention

Example 5. _How bootsrap exploration from experience?_

_一个角度是, 在现实中并不是什么 action 都可以 explore 的, 例如一些行为可能会给 agent 本身带来极为严重的伤害. 而如何避免这些 action 就需要一些 prior knowledge._

_另一个角度是, 如果我们想要训练一个机械臂进行 grasping, 那么如果我们使用完全随机的 actions 作为初始化, 它仅仅只是随机地乱动, 很难有效地 exploration. 但是如果其已经有了完成一系列任务其他的 experience, 那么我们可以用这些任务的数据构造一个 behavioral prior, 具体来说, 可以是从这几个任务的 policy 中进行随机采样. 尽管其做的事情并不完全是我们想要的, 但是这是一种明显更加高效的 exploration 方式._

![](https://pic3.zhimg.com/v2-83d9d9bbbb24bff7ea8ebb17394deea8_1440w.jpg)

Parrot: Data-driven behavioral priors for reinforcement learning

_参见: Singh\*, Hui\*, Zhou, Yu, Rhinehart, Levine. Parrot: Data-driven behavioral priors for reinforcement learning. 2020_

自然现在我们清楚了在现实世界中进行 RL 是一件非常有挑战的事情, 并且也需要考虑到很多问题. 但是这是值得的:

如果我们想要看到有趣的 emergent behavior, 那么我们需要设计一个足够复杂的 world, 使得其中能够容纳这些 novel solutions. 现实世界的 RL 可能很困难, 但是值得的, 因为我们能够期望那些可能的 emergent behavior.

### 3.3 Reinforcement Learning as "Universal" Learning

在这一个视角中, 我们将 RL 视作一种 "universal" learning 的方式.

目前 LLM 的成功是基于大量的无标注预训练数据 + 少量标注的数据, 其背后的 knowledge 很大程度来自于我们通过自监督学习学到的分布 $p_\theta(\boldsymbol{x})$ . 目前这样的训练方式取得很大成功, 很大程度上因为我们能够从大量廉价的数据中学习到知识. 当然一个挑战是这些数据不能都是垃圾, 也就是我们不能学习一个 low quality 的数据分布. 实际上, 一个更好的做法是使用 RL:

machine learning 的目的可以理解是为了产生 adaptable 和 complex 的 decisions. (这可能不那么显然, 对于 image classification 来说, 其背后的 decision 是预测 label 后发生的事情, 例如识别到了险情, decision 是否报警.)

实际上 RL 是一种更好地利用 low quality data 的方式, 当我们获取到大量的 low quality data 时, 建模其密度分布可能并不是很好的选择, 我们从中需要学习到的不是在 world 中如何做, 而是在 world 中能够做到什么, 换言之在学习 how the world works (dynamic) 的信息. 而之后我们再通过 reward function 等学习到关于 task 的信息, 并从关于 world 的知识中找出那些 best possible 的 decision.

![](https://pic4.zhimg.com/v2-88fb18c53a14651f47575ce92a2c6b01_1440w.jpg)

RL as universal learning

具体来说, 我们可以得到如下的学习方式:

-   利用大规模数据进行 offline RL, 通过 human-defined skills, goal-conditioned RL, self-supervised skill discovery 等等方式进行预训练. 这一方式中我们学习关于 world (dynamic) 的知识.  
    
-   之后我们在特定的 downstream tasks 上进行微调. 这一方式中我们学习关于 task 的知识 (reward function).

![](https://pic3.zhimg.com/v2-5047f83401ca38e61ad1068ea5c201d4_1440w.jpg)

收集大规模数据 + offline RL + downstream finetune

一个利用 offline RL 作为 recipe 的例子如下:

Example 6. _Use offline RL train large language models_

_通常情况下我们使用 RLHF 等方式来对齐 LLM, 尽管其可以让回答更加符合人类偏好, 但是这样一种 single-step 的建模并不擅长于完成整个对话目标._

_也许我们可以用 offline RL 进行改进, 首先通过 LLM 获取一系列合成的对话轨迹, 然后再这些数据上进行 offline RL, 通过 model-based RL 等方式更好理解人的需求 (例如建模一个 POMDP), 通过更短的多个对话来更加符合用户要求._

![](https://pic3.zhimg.com/v2-9960dcbe591fa81d0325d88c2bc4cb32_1440w.jpg)

Zero-Shot Goal-Directed Dialogue via RL on Imagined Conversations

_参见: Hong, Levine, Dragan. Zero-Shot Goal-Directed Dialogue via RL on Imagined Conversations. 2023._

## 4 Back to the Bigger Picture

### 4.1 Why we need Deep RL?

回到这门课程介绍的部分中我们引出 deep RL 的原因:

-   Learning 是智能的基础  
    
-   RL 是 reason about decision making 的方式  
    
-   deep models 允许 RL 算法学习到复杂的映射, 通过端到端的方式解决复杂的问题.  
    

### 4.2 What's missing for an intelligent system?

我们目前为止还没办法实现一个真正的 intelligent system, 其真正重要的是什么呢?

-   从 Yann LeCun's cake 的角度, 通过不同的 learning 方法, 我们能够获得的 supervision 的量有很大差异, 因此重要的是  
    

-   Unsupervised or self-supervised learning  
    
-   Learn a world model (predict the future)  
    
-   Generative modeling of the world  
    
-   Lots to do even before you accomplish your goal!

![](https://picx.zhimg.com/v2-ede02afeb557cf76c75b846234725bb9_1440w.jpg)

Yann LeCun&#39;s cake

-   supervision 还可以来自于其他的地方, 例如 imitation 以及理解其他 agents.  
    
-   也许 RL 本身已经足够了, 虽然说 reward 本身可能很稀疏, 但是叠加上复杂 dynamic 后得到的 value function backup 已经携带了足够的信息.  
    
-   或许以上都是需要的.  
    

### 4.3 How should we answer these questions?

这一部分是 Levine 对 RL 科研的思考, 也可以理解为是 Levine 对 RL 科研的一些建议:

-   选择正确的问题: 问一问这有机会解决一个重要的问题吗? 保持对不确定性的乐观态度是一个很好地 exploration 策略.  
    
-   不要害怕改变问题的 statement: 仅仅是冲击 benchmark 并不会遇见很多这些有意义的挑战.  
    
-   Application matters: 很多时候将方法应用到挑战性的 real-world 中可以告诉我们缺失了哪些重要的东西. RL 的历史上有很长一段时间忽略了这件事情.  
    
-   Think big and start small!