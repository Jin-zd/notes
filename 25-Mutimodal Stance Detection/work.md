# 2024-11-28

虚假信息检测多是通过检索外部知识来判断图像和文本的不一致性。
与虚假信息检测不同，立场检测数据集中的文本和图像往往是信息互补的，是一致。

需要通过文本-图像联合检索外部知识。
外部知识来源：互联网、某些特定的知识图谱。
一种方法是分别搜索获得信息，再将信息融合后使用；
另一种方法是融合文本和图像进行搜索，获得信息直接使用。

分别搜索，
对于知识图谱，文本直接嵌入为向量进行搜索，或者提取实体之后搜索；图像嵌入为向量进行搜索，或者使用目标检测获取实体后搜索，或者使用图生文技术，生成图像的描述，再提取实体后搜索。
对于互联网，直接调用搜索引擎的 api，但是对于搜索到的信息还要进行进一步处理（判别、提取等）。

融合搜索，主要是要考虑怎么融合文本和图像的信息，
采用多细粒度融合（受论文 FMC: Multimodal fake news detection based on multi-granularity feature fusion and contrastive learning 启发），具体而言，就是使用多模态模型编码文本和图像作为粗粒度，单模态模型分别编码文本和图像作为细粒度，再使用注意力来获得加权特征，同样对知识图谱中的键使用多模态模型编码，使用加权特征来查询。


分别搜索获得两个模态的搜索结果，两个结果怎么融合呢？
一篇论文（Knowledge-Aware Multimodal Pre-Training for Fake News Detection）提出的方法是构建相关性图，文本实体和图像实体均作为图的节点，相似性作为图的边，获得的图作为知识的表达。
还有什么方法呢？
- 在原来论文的基础上，将图替换为异构图，显式的区分节点和边的类型，使用异构图神经网络学习不同模态检索到的知识之间的关系（图模型具体架构还需要设计，对边权重设置注意力或者近似度等）。
- 改进相似度计算方式，目前构建图的节点的相似度是直接采用特征之间的点积等作为相似度度量，可以构建孪生神经网络进行特征嵌入来判断相似度。
不用图？不用图的话，
- 使用预训练模型将每对文本-图像检索出的信息编码为向量，输入预训练模型前可以先通过注意力进行加权。


# 2024-12-09

[Learning to Prompt Segment Anything Models](https://arxiv.org/abs/2401.04651)中指出：

> Text prompt encoders in SAMs (e.g., CLIP in SAM and UniCL/Florence in SEEM) are largely pre-trained with object-centric imagetext data, where the text data is dominated by the description of foreground objects, leading to well-learnt foreground text knowledge but relatively poorly-learnt background text knowledge. Consequently, learning semantic prompts with such text prompt encoders can benefit from the well-learnt text knowledge, but may also suffer from the side effects from the poorly-learnt text knowledge.

表明 SAM 的提示编码器在 foreground objects 上性能更优，而且由于编码器的预训练方式（largely pre-trained with object-centric imagetext data），采用非对象的提示可能效果不会很好。

目前大多数针对 SAM 的提示策略都是基于非文本信息（主要是目标检测框）进行的 auto-prompt，对于文本提示的生成还是依靠人为标注图像中的实体内容。

- [AM-SAM: Automated Prompting and Mask Calibration for Segment Anything Model](https://arxiv.org/abs/2410.09714)采用 YOLOv8 目标检测器来自动生成边界框作为提示。
- [EVF-SAM: Early Vision-Language Fusion for Text-Prompted Segment Anything Model](https://arxiv.org/abs/2406.20076)探究的是如何更好的使用文本提示，而文本提示的来源还是人工标注。
- [AutoProSAM: Automated Prompting SAM for 3D Multi-Organ Segmentation](https://arxiv.org/abs/2308.14936)中将图像编码器最后一个注意力模块和空间适配器操作后的输出特征图作为提示进行输入，实现的是模型的自我提示。
- [GeoSAM: Fine-tuning SAM with Multi-Modal Prompts for Mobility Infrastructure Segmentation](https://arxiv.org/abs/2311.11319)中使用“You are a creative assistant, skilled in providing detailed visual descriptions of objects as seen in aerial imagery.”的角色提示来让 GPT-4 生成对图像的文本描述作为 SAM 的提示之一，这样生成的提示还是聚焦于图像中的对象。

其他针对 SAM 的提示的研究大多就是这几个方向：使用边界框（点）、融合外部提示和自主提示。
并且提示都是倾向于图像中的对象，而不是图像的其他一些属性。
所以，对于依据图像属性来进行文本提示的设计这方面并没有什么已有的工作作为参考。


论文 [Multi-modal Stance Detection: New Datasets and Model](https://arxiv.org/abs/2402.14298)中对于编码后的文本和图像向量只是做了简单的拼接，可能是使用多模态信息时分类准确率下降的原因。

度量学习（metric learning）通过学习适当的距离度量来优化样本之间相似度。
文本模态和图像模态的信息需要进行模态对齐，可以结合 contrastive learning 或者 metric learning 来学习模态间的对齐关系。

- [A cross-modal deep metric learning model for disease diagnosis based on chest x-ray images](https://link.springer.com/article/10.1007/s11042-023-14790-7)中提出的模型的映射模块通过学习一个映射函数，将视觉和语义信息映射到同一个低维空间，损失函数的设计目标是使正样本与类别原型之间的距离小于负样本与类别原型之间的距离，与三重损失函数设计类似。
- [Multi-modal Semantic Understanding with Contrastive Cross-modal Feature Alignment](https://arxiv.org/abs/2403.06355)中将来自相同文本-图像对的特征作为正样本，来自不同文本-图像对的特征作为负样本，构建对比函数，通过优化对比函数来对齐不同的模态。
- [DELAN: Dual-Level Alignment for Vision-and-Language Navigation by Cross-Modal Contrastive Learning](https://arxiv.org/abs/2404.01994)中将要对齐的模态的特征嵌入进行点积运算获得相似度，多个对齐模态的相似度使用注意力进行聚合获得最终的损失函数，优化损失函数进行多模态对齐。


# 2024-12-20

- sam 生成描述和属性描述进行分割的效果？
- 复现 MMFakeBench: A Mixed-Source Multimodal Misinformation Detection Benchmark for LVLMs
- 寻找更好利用多个模态信息的 metric learning 方法


Meta 开源的 SAM 的代码中没有使用文本提示的接口，使用了论文 Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks 的框架对 SAM 进行文本提示分割。
- 在纯文本图片上，SAM 的分割效果很差，使用描述性提示的话，只有捕捉到了强调的一些对象因素（LVLM 不能保证每次一定能生成这样的描述），SAM 才能较好的将其分割出来，使用属性提示基本是不可分割的；
- 针对人物动作图片，无论哪种提示，SAM 都倾向于分割出文本中提到的对象，Brightness，Colorfulness，Scene type 这三个属性似乎作用不大，分割时背景被提示涉及才会被抓取，不然基本都是忽略的；
- 物体图片，完全取决于物体在提示中是否被提及。
不管怎么样，只有物体或者动作在文本提示中被提及，SAM 才有可能将其分割出来。这还是应证之前的说法：SAM 的提示编码器是针对大量的对象提示进行预训练的，更倾向于捕捉提示中的对象的信息，其他的一些提示 SAM 关注的相当少。


MMFakeBench: A Mixed-Source Multimodal Misinformation Detection Benchmark for LVLMs 
数据集尝试使用 Multi Modal Stance 的数据集进行测试。
方法为智能体提示决策，将混合来源检测分解为三个阶段：文本真实性检查、视觉真实性检查和跨模态一致性推理。在每个阶段，MMD-Agent 指示 LVLM 生成多角度推理，并将模型动作与外部环境中的补充信息相结合，以进行一致的决策。


关于多模态信息的使用，即如何利用现有的多模态信息更好的进行分类，除了模态对齐外，可以进行模态选择，但直接针对这方面的研究工作相当少。
- [Efficient Modality Selection in Multimodal Learning](https://jmlr.org/papers/volume25/23-0439/23-0439.pdf)提出了两种方法，分别是贪婪搜索和基于特征重要性的排序，论文总体比较偏理论分析。

# 2025-02-28

- 如何抽取/识别图像中的事件背景，以及文本中的事件背景
- 事件知识怎么提取整合（内部和外部知识）


 Event-Radar: Event-driven Multi-View Learning for Multimodal Fake  News Detection 中对事件的提取：
 - 词元化文本，并使用 Faster R-CNN 对从图像中提取对象，然后在 CLIP 中对齐文本和图像，以获得维度相等的特征向量。构建多模态图，其中的节点是文本和图像的特征向量，边的权重基于特征向量之间的余弦相似度。
- 使用 Stanford NLP 和 TextSmart NLP 工具进行命名实体识别，提取 subject, object 和 location。将这些实体与文本的 CLS 词元连接，形成文本事件子图。
- 从图像中提取与识别出的 subject, object 和 location 最相似的部分，并将它们与图像的 CLS 词元连接，形成图像事件子图。
主要是先将文本和图像嵌入，再固定提取需要的实体与词元相连构成事件图。


Reinforced Adaptive Knowledge Learning for Multimodal Fake News Detection 中对知识图的构建：
- 实体嵌入：使用 TAGME 提取文本实体，并将其映射到知识图谱中；使用 Faster R-CNN 识别图像对象，并通过预训练的实体链接模型（pre-trained visual entity linkage model）将其嵌入到知识图谱中。
- 节点建模：使用马尔可夫决策过程（MDP）对知识图谱中的实体连接进行建模；根据特定的子图生成策略，为文本和图像生成各自的子图。
这里是提取文本和图像中的实体，嵌入知识图谱后采用强化学习的方法进行建模。


MMIDR: Teaching Large Language Model to Interpret Multimodal Misinformation via Knowledge Distillation 中对可解释性的工作：
- 使用 OCR 提取图像中的文本信息，并使用 Blip-2 生成图像的描述性文本。
- 使用 Google 反向图片搜索（Google Reverse Image Search）检索相似图像，并获取它们的标题和描述作为文本证据。
- 使用 Google 可编程搜索引擎（Google Programmable Search Engine）检索基于文本的搜索结果，并对检索到的图像应用之前的图像处理方法，以获取基于图像的证据。
- 设计带标签的提示模板，引导教师模型生成对虚假信息检测结果的解释。开发指令微调（instruction-tuning）提示模板，并使用 LoRA 训练学生模型。
主要目的是收集信息构建提示模板指导教师模型生成解释。



A Survey on Deep Learning Event Extraction:  Approaches and Applications
介绍了基于 CNN、RNN、LSTM、GCN、Transformer 的特征提取识别文本触发词和论元的方法。

Multi-Grained Gradual Inference Model for  Multimedia Event Extraction：
解决现有多媒体事件提取 (MEE) 方法在处理复杂多媒体文档时难以精确提取事件和论元的问题；
利用现有的 ResNet50 和 GloVe + LSTM 分别提取文本和图像信息，构建文本事件图和图像事件图，再对每个图使用 GCN 卷积预测图节点的事件类型和论元类型，卷积后的图进行对齐融合，再进行引导性推理训练后，最终输出多媒体文档中每个事件的类型和论元。

Cross-Modal Contrastive Learning for Event  Extraction：
通过对比文本-视频和事件-视频表征来从多模态数据中提取事件。
使用 BERT 模型将输入句子编码为上下文表征，结合 CRF 层标注句子中的触发词及其事件类型，进一步标注论元及其角色；使用 Transformer 编码器层计算视频的表征，Transformer 解码器层生成事件类型序列；使用两个可学习的网络将文本和视频的全局表征投影到共享的嵌入空间中；训练好的模型可以输出文本中的论元和触发词，视频中的事件类型序列。

Training Multimedia Event Extraction With Generated Images and Captions：
使用预训练的 Transformer 模型提取文本和图像特征，使用交叉注意力融合文本和视觉特征；将词的文本特征和融合后的特征输入分类器识别触发词，进一步判断其在事件中的作用；同样，对图像的 CLS token 特征和融合后的特征做相同的操作；最后根据不同的任务添加不同的分类器进一步输出触发词或者论元。


Cross-modal event extraction via Visual Event Grounding and Semantic Relation Filling： 
通过自行精心构建特征和融合各种特征进行图像和文本的融合。

![[Pasted image 20250312160250.png]]


上述关于多模态事件提取任务的方法，倾向于编码文本和图像信息后，采用不同的方式（对比学习、图卷积等）进行特征融合，再根据融合特征进行触发词（触发对象）、论元的提取；也有是设计不同的特征嵌入针对不同的事件提取问题。所使用的文本编码器和图像编码器多是 BERT 和 CNN 之类的模型，没有涉及到 LLM。


有关于文本事件提取的工作涉及到了 LLM，
Towards Event Extraction with Massive Types: LLM-based Collaborative Annotation and Partitioning Extraction：
该模型针对LLM处理大量事件类型时的上下文长度限制问题，提出了分区提取方法。
利用相似度模型，将句子中可能出现的候选事件类型减少到一个小的子集，将候选事件类型分成几个分区，以进一步缩短每个分区的提示长度，利用 LLM 根据分区提示进行事件触发词和论元的提取。


结合 SAM 识别图像中的事件：
- 结合其他模型给出图像中事件的描述，将该描述作为文本提示输入 SAM 进行提示分割
- 目标检测获取图像中的实体，再将实体提示输入 SAM 进行分割
- 参考 Event-Radar: Event-driven Multi-View Learning for Multimodal Fake  News Detection，可以固定 （主语、谓语、地点）之类的要素，直接提示 SAM 分割出这些子图像

识别文本中的事件：
- 命名实体识别获取实体对应的词汇
- 限定词提取，直接从句子中提取（地点，人物，动作）相对应的词

仅从文本和图像中直接获取的事件信息基本上就是这些，需要更多有关文本和图像中所涉及的事件信息，需要使用文本和图像来搜索过滤互联网信息。


# 2025-03-12
- 多模态大模型能够裁剪出细粒度的区域
- 通过 prompt 的方式去截取：使用描述作为提示对图像进行分割；做个判断，如果是纯文本信息就使用 OCR。
- 已经得到了裁剪之后的细粒度的多个图片，对每个小图片进行描述，或者用他们去进行事件提取，这些小图片之间的关系。
- 最终，图像方面事件关系（不仅包括外部知识，也包括图像中的语义信息）图


之前提到的几个事件提取模型都没有开源代码。


大模型提示 SAM 对图像进行分割：
描述均由 Qwen-2.5-Max 生成，提示为 Give a one-sentence description of the image.
- Two formally dressed individuals are signing documents, with a background screen displaying welcome messages in multiple languages.
![[grounded_sam2_florence2_phrase_grounding.jpg|500]]
- A panel of five individuals is seated at a conference table, with flags and an emblem displayed behind them.
![[Pasted image 20250317155049.png|500]]
- Three Union Craft Brewing glasses are stacked on a table, with an outdoor view of red umbrellas and trees in the background.
![[Pasted image 20250317155532.png|500]]
- A delicate necklace with a rectangular pendant featuring intricate wave-like engravings is displayed on a wooden surface.
![[Pasted image 20250317161516.png|500]]
- Promotional poster for 'The Port Arthur Tragedy' episode of Crimes That Shook Australia , airing on the Crime & Investigation Network.
![[Pasted image 20250317155822.png|500]]
- The image illustrates how other common items like eggs, milk, and oranges would cost if their prices had risen as rapidly as healthcare costs since 1945.
![[Pasted image 20250317160838.png|500]]
- Two silhouetted figures in suits engage in a tug-of-war over a rope labeled "GILEAD" and "ABBVIE," with the word "Anthem" on a briefcase nearby, set against a vibrant orange and yellow background.
![[Pasted image 20250317161905.png|500]]
- The image shows a bar graph from Gartner illustrating the top priorities of infrastructure acquisition strategies, with 'Integrated infrastructure systems' leading at 42%.
![[Pasted image 20250317162234.png|500]]

提示 SAM 分割的质量主要取决于提示的质量。对于人物场景图像，SAM 能根据提示进行细致的分割；对于文字传达主要信息的图像，SAM 很难根据提示把握图像的重点含义，有些这类图像甚至无法进行分割。


对于分割子图后，进一步探索子图之间的联系或者利用子图进行事件提取：
使用这张图的分割为例，
![[grounded_sam2_florence2_phrase_grounding.jpg|500]]
分割出的子图为：
<div style="display:flex; justify-content:center;"> 
	<img src="Pasted image 20250317172645.png" width="150" style="margin-right:2px;"> 
	<img src="Pasted image 20250317172702.png" width="150" style="margin-right:2px;"> 
	<img src="Pasted image 20250317172732.png" width="180" style="margin-right:2px;"> 
	<img src="Pasted image 20250317172747.png" width="180" style="margin-right:2px;"> 
</div>
<div style="display:flex; justify-content:center;"> 
	<img src="Pasted image 20250317172830.png" width="150" style="margin-right:2px;"> 
	<img src="Pasted image 20250317172843.png" width="150" style="margin-right:2px;"> 
</div>
图像是有部分重复的，第一张被识别为 a background screen，第二张被识别为 multiple languages，对这几张图片依旧使用 Qwen2.5-Max 进行描述：
1：A woman in a suit is looking down at a device, with a multilingual "Welcome" sign displayed on screens behind her.
2：A woman in a suit is looking down at a device, with a multilingual "Welcome" sign displayed on screens behind her.
3：A person is writing in a pink notebook with a pen.
4：A person is writing in a pink notebook with a pen, while a microphone and other objects are visible nearby.
5：An older man in a suit and glasses is signing a document at a table, with others standing nearby.
6：A woman in a suit is signing a document while seated at a desk, with a multilingual welcome sign displayed on screens behind her.

这里的 1 和 2 两张图被模型视作了同一张图像，获得了相同的描述，分割时，前一张图片被视为 mutiple language，后一张图片被视为 background screen。


# 2025-03-20
- 提取图像 caption，联合文本，使用 gpt-4o-mini 测试 MTSE 和 MCCQ 两个子数据集。
- 构建代码，进行命名实体识别和  wiki 搜索知识并进行总结。
- 看论文，认知学相关，如何构建经验池，怎么去学习经验池
**![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfzNBNu9Bjq1yQpxS4YxLOF420LS2LhJ1PNh_Wmd5oQ6hbcf_UVkCOM6DLruQnJyluR3wVzi77SrnK5VmlpdDLoqDzfECgm61kGk-8QP3MB1JkXnLlDOZYfbdN0wlX8LpjXJDEJzQ?key=4N7O3hWn3duZbtC2qbU5W7d6)**

gpt-4o-mini 在 mtse 和 mccq 上，结合图像的 caption 和文本的测试结果：
in-target mtse 59.46
in-target mccq 53.68
zero-shot mtse 63.56

实体提取和知识总结均使用 Qwen2.5-7B。


**ExpeL: LLM Agents Are Experiential Learners**
**目的**：
- 解决针对特定任务对 LLM 进行微调是资源密集型的，并且可能会削弱模型的泛化能力的问题
- 无需参数更新的从经验中学习的需要。

**经验获取**：
- 在多次试验中，智能体被赋予一个训练任务、少量样本和对过去的反思（初始为空）；
- 智能体将少量样本与其当前轨迹拼接作为上下文来执行任务，使用 ReAct 作为基础规划算法；
- 在某次试验中，当智能体完成任务或者达到最大步数时，智能体的经验池会吸收轨迹；
- 若任务成功，智能体将进入下一个任务。若任务失败，智能体将查看其失败的轨迹并进行自我反思，并将其与之前的反思拼接起来。在下一次重试中，智能体会在其上下文中扩展反思。

这种试错式的经验积累方式不仅提高了在评估过程中回忆经验时获得更多积极案例的可能性，还使得能够收集到用于在见解提取过程中进行比较的成功/失败案例对。

**从经验中学习**：
- 给智能体的指令可以分解为任务规范和少量示例。
- 可以利用智能体从以往经验中提取的见解来增强任务规范，让指令遵循型 LLM 紧密遵循这些规范。
- 对于少量示例，可以允许智能体从其经验池中检索出最相关的 top-k 示例来辅助其决策。
- 经验回忆使用 Faiss 向量库作为经验池，利用 kNN 检索器和 all-mpnet-base-v2 嵌入器来获取与评估任务具有最大内积任务相似性的前 k 条成功轨迹。使用任务相似性作为检索排名的优势是，若智能体重复执行一个任务，或者执行一个与经验池中已有的成功轨迹相似的任务，智能体只需紧密模仿成功轨迹，从而减轻能力外推的负担。
- 见解提取让智能体比较同一任务的成功轨迹和失败轨迹，让智能体在不同任务的成功轨迹集合中识别模式。向智能体提供经验池中的成功失败轨迹对或者成功轨迹，智能体可以进行添加、编辑、反对、赞成操作来构建见解集合。该设计选择增强了过程的稳健性，因为即使是成功的轨迹也可能是次优的，并可能误导生成的见解。

**任务推理**：对于每个任务，见解集合与经验池中最高任务相似度的 k 个成功轨迹会被作为少样本上下文示例提供给智能体。

**迁移学习**：使用从原始任务中提取的见解以及目标任务中的少量样本，对这些见解进行“微调”，使其更适合应用于目标任务。

**实验结果**:
- 在从训练集中收集的经验中提取了见解之后，该智能体随后获得了重新评估其最后几步的整个轨迹并明确结束任务的能力，而不是表现出其在提供解决方案方面的无能。
- 从现有少量样本中提取见解的智能体与 ReAct 智能体相比没有任何优势，这表明经验对于 ExpeL 学习至关重要。
- 基于任务相似性的检索表现最佳。基于原因相似性的检索仍然具有优势，但性能略有下降，可能是由于在单一轨迹中动态变化的少量样本导致的不稳定性。


**A-MEM: Agentic Memory for LLM Agents**
**目的**：
- 当前的记忆系统能够实现基本的存储和检索，但缺乏复杂的记忆组织架构；
- 虽然图数据库为记忆系统提供了结构化的组织方式，但它们对预定义模式和关系的依赖从根本上限制了其适应性；
- 当前系统只能在其预设框架内对这些信息进行分类和链接，无法随着知识的发展建立创新的联系或发展新的组织模式；
- 论文提出的方法将 Zettelkasten 的结构化组织原则与基于智能体的决策制定的灵活性相结合，能够以主体性的方式动态地组织记忆，实现了更具适应性和情境感知能力的记忆管理。

**记录构建**：
- 当一个智能体与其环境互动时，构建结构化的记忆记录，捕捉显式信息和 LLM 生成的上下文理解。
- 一个记忆记录包含：原始互动内容、互动时间戳、LLM 生成的捕捉关键概念的关键词、LLM 生成的用于分类的标签、LLM 生成的提供丰富语义理解的上下文描述和具有语义关联的链接记忆集合；
- 为了实现高效的检索和链接，通过文本编码器（Sentence-bert）计算出密集向量表示，表示记忆记录中所有的文本组件。

将 LLM 生成的语义组件与密集向量表示相结合，既提供了人类可解释的上下文，又实现了计算效率高的相似性匹配。

**连接生成**：
- 当记忆记录被加入系统时，首先使用其语义嵌入进行相似性检索（余弦相似度），选取 top-k 个最相似的记忆记录；
- 基于这些候选最相似的记忆，提示 LLM 根据它们的潜在共同属性分析潜在联系，更新连接集合。

A-MEM 即使在庞大的记忆集合中，也能快速识别潜在的联系，无需进行穷尽比较。

**记忆演变**：
- A-MEM 根据检索到的记忆的文本信息及其与新记忆的关系来演变这些记忆。
- 对于每个记忆的最相似集合中的每个记忆，系统会决定是否更新其上下文、关键词和标签。

这种进化式方法能够实现持续的更新和新的连接，模拟人类的学习过程。

**相关记忆检索**：
- 在每次交互中，A-MEM 执行上下文感知的记忆检索，为智能体提供相关的过往信息；
- 针对当前交互中的查询文本，首先使用与记忆记录相同的文本编码器计算其密集向量表示，随后使用余弦相似度计算查询嵌入与所有现有记忆记录之间的相似度得分；
- 根据相似度得分，从历史记忆存储中检索出最相关的 k 个记忆片段，以构建一个上下文合适的提示。

检索到的上下文通过将当前互动与记忆系统中存储的相关过去经验和知识联系起来，丰富了智能体的推理过程。

**实验**：
- 虽然较大的 k 值通常通过提供更丰富的历史背景来提升性能，但超出一定阈值后，收益会逐渐减少，这表明背景丰富度与有效信息处理之间存在权衡。
- 消融实验表明连接生成模块是记忆组织的关键基础，但记忆演变模块为记忆结构提供了必要的改进。
- 对记忆嵌入的 t-SNE 可视化表明，A-MEM 始终展现出更连贯的聚类模式，展现出良好的结构化组织。


# 2025-03-28
- 使用 gpt-4o-mini 测试两三个数据集上 summary 和原始文本以及图像 caption 拼接后的性能
- 实现第一部分，文本提取-检索-总结，图像标题概述-检索-总结，图像根据实体进行分割
- 实现第二部分，根据第一部分得到的文本图像信息，使用 LLM 进行推理，分别获得文本、图像和多模态的判断结果
![[cc495bd511d20517ceebd6ad641b84a.png|500]]


text，image caption，text knowledge summary，image knowledge summary 一起输入给 gpt-4o-mini，在 MRUC 上，in-target 的 F1 为 58.99，zero-shot 的 F1 为 54.74；在 MTWQ 上，in-target 的 F1 为 57.50，zero-shot 的 F1 为 58.82.
![[Pasted image 20250407104614.png]]
![[Pasted image 20250407104705.png]]

论文 ExpeL: LLM Agents Are Experiential Learners 框架中的 Trajectory 是动作-反馈的交替序列，智能体初始会被输入一个与任务相关的问题，然后智能体会根据问题去采取一个行动，这个行动会被输入给环境，环境根据行动给予智能体一个反馈，智能体根据这些信息进行下一步行动，如此交替直到获得答案。为了让智能体严格按照这样的形式进行操作，代码中对每个任务设计了好几个上下文提示，并且智能体基模型使用了gpt，以获得较好的指令遵循能力。
这里的智能体的行动并不是任意的行动，对每个不同的任务有不同的环境，不同的环境只有几个被设定好的智能体能采取的动作，比如在 hotpotqa 中，模型只能采取 finish（表示得出结果），search（表示需要搜索相关信息），lookup（在已有的搜索结果或内容中寻找特定关键词）：
```
Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?

Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.

Action 1: Search[Milhouse]

Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.

Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".

Action 2: Lookup[named after]

Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.

Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.

Action 3: Finish[Richard Nixon]
```
对于立场检测，这里不太合适构建类似的动作-反馈序列轨迹，在第一部分主要的信息已经被提取整合，第二部分很难设计与环境交互的动作。但可以设计思维链提示，引导模型进行推理，比如
```
【文本分析】分析文本内容的语气、情绪和措辞，有无明显立场词汇。
【文本实体知识分析】根据文本背景知识判断文本实体可能的倾向性。
【图像分析】分析图像是否含有暗示性元素。
【图像实体知识分析】根据图像背景知识解释图像表达的含义。
【多模态融合】将文本与图像信息融合，判断是否存在一致的立场倾向，或是否互相矛盾。
【立场判断】基于以上分析，总结帖子整体的立场倾向，并说明理由。
```
目前的代码实现中没有使用思维链提示，而是直接使用一段提示得出结果：
```
Classification task: Determine the stance in the tweet text and image towards " {dataset_config.topic_text[targets[i]]}" based on the following information.

Tweet text: "{sentences[i]}"
Text knowledge: "{text_knowledges[i]}"
Image caption: “{images[i]}”
Image knowledge: "{image_knowledges[i]}"

Your task is to classify the stance as one of the following:
{', '.join([f'{v}: {k}' for v, k in idx2label.items()])}

Analysis instructions:
1. Carefully examine both the tweet text and the knowledge from text
2. Identify any explicit or implicit stance towards the target
3. Select the most appropriate stance classification number

IMPORTANT: Respond ONLY with the single number corresponding to your classification.
DO NOT include any explanation, words, punctuation, or other characters.
```


# 2025-04-07
训练过程：
1. CoT reflect：究竟需要反思什么？递进关系
    1. 为什么xxx 模态能够判断正确，xxx 模态判断错误？
    2. 如何更好地利用两个模态去做立场检测 $R$
2. Experience pool
    1. 现在可以这么做：输入 reflect 的结果 $R$，从 domain 去检索到与当前最相似的 reflect $R_{sim}$（text-embedding-3-small 直接用 openai 的 apikey 去调用）做一次更新，将 $R_{sim}$  和 $R$ 进行融合，然后存到 domain。如果没有相似，threshold<0.8，就直接把 $R$ 存进去，不需要考虑融合。
    2. _**之后的优化：如何如建模经验池，链、树、图等方式去呈现。存的经验可以不只是文本经验，也可以的多模态**_
    3. 对于步骤a，是 train 的过程，有多少训练的数据，就会重复多少轮。epoch，迭代次数（一次）
    4. 最后会得到一个经验池 $P$，然后用 Qwen-VL，基于经验池，去对 test 进行立场检测。

测试过程：
将 text image 的 information 和 summary 四块输入 Evaluator LLM，我们将这四个和经验池中的经验再次做相似度匹配，取出组相似的top n个经验，直接拼接在一起去进行stance detection。

目的：
1. 看一下经验池是否真的能够总结出什么时候该用模态信息，并且有用。
2. 实验上可以考虑研究一下在Reflect的过程中CoT的方法是否有效，就比如我只用a或b作为reflect


训练过程：
对于某一个数据，首先获得三个智能体的答案，一个仅传入文本和文本知识，一个仅传入分割后的图像和图像知识，一个将所有信息输入，提示示例如下：
```
Classification task: Determine the stance towards "{data['target']}" based on the following information.

Tweet text: "{data['text_information']}"
Text knowledge: "{data['text_summary']}"
Image information: [The image is provided separately for your analysis]
Image knowledge: "{data['visual_summary']}"

Your task is to classify the stance as one of the following:
{stance_map}

Analysis instructions:
1. Carefully examine all the provided information.
2. Identify any explicit or implicit stance towards the target
3. Select the most appropriate stance classification number

IMPORTANT: Respond ONLY with the single number corresponding to your classification.
DO NOT include any explanation, words, punctuation, or other characters.
```
获得三个智能体的答案后，将由 Evaluator 进行反思，第一步反思结合已有的信息和三个智能体的答案，第一步的反思提示：
```
Analysis Task: Evaluate the performance of different modalities in stance detection.

Target: "{data['target']}"
Tweet text: "{data['text_information']}"
Text knowledge: "{data['text_summary']}"
Image information: [The image is provided separately for your analysis]
Image knowledge: "{data['visual_summary']}"

Correct stance label: {data['label']} ({idx2stance[data['label']]})

Classification results:
- Text-only modality: {results[0]}, {results[0] == data['label']};
- Image-only modality: {results[1]}, {results[1] == data['label']};
- Multi-modal fusion: {results[2]}, {results[2] == data['label']}.

Please analyze the following:
1. Which specific information in the text or image was most indicative of the correct stance?
2. What information did I miss or misinterpret that led to incorrect predictions?
3. How did the information from different modalities complement or contradict each other?

Provide a detailed analysis focusing on the information content and interpretation, not on model architecture.
```
随后，结合第一步反思的输出，Evaluator 进行第二步的反思，反思提示：
```
Improvement Task: Based on the previous analysis, provide concise recommendations for better utilization of multi-modal information.

Previous analysis:
{step_one_response}

As an AI system analyzing this case, provide concise reflections on:
1. How should I better weigh and interpret text vs. image information when they provide different signals?
2. What specific clues or patterns should I pay more attention to when determining stance?
3. What reasoning approaches would help me avoid the mistakes made in this case?

IMPORTANT: Focus only on how you as an AI can better interpret and reason with the given information. Do NOT suggest system redesigns, algorithm changes, or data collection strategies. Be extremely concise with 3-5 short bullet points total.
```
第二步反思的输出作为 Evaluator 的最终反思输出。
Evaluator 反思结果会被加入经验池中，对一个被加入经验池的反思，首先会使用 text-embedding-3-small 进行嵌入，然后会在经验池中检索与该经验最相似的经验，如果余弦相似度小于 0.8，直接将该经验的嵌入存入，否则，将新的经验与旧的经验进行融合，将融合后的经验存入经验池。这里的向量嵌入目前使用 Faiss 存储。融合提示：
```
Please merge the following two similar experience descriptions into a unified, comprehensive description.
Please retain all important information, eliminate redundant content, and ensure the merged description is fluent, natural, and complete.
If there is any image-related information, please integrate it as well.

New experience:
{experience}

Existing similar experience:
{similar_experience}

Please output a merged experience description that is more comprehensive and useful than either individual experience alone.
```
至此，一个数据的训练过程完成。

评估过程：
对于一个评估数据，将文本、文本知识、图像知识拼接，获取嵌入后查询经验池，获得最相似的 top_k（设置 3） 个经验，随后结合信息构建提示：
```
Classification task: Determine the stance towards "{data['target']}" based on the following information.

Tweet text: "{data['text_information']}"
Text knowledge: "{data['text_summary']}"
Image information: [The image is provided separately for your analysis]
Image knowledge: "{data['visual_summary']}"

Previous experiences:
{experiences_text}

Your task is to classify the stance as one of the following:
{stance_map}

Analysis instructions:
1. Carefully examine all the provided information.
2. Consider the previous experiences as reference.
3. Identify any explicit or implicit stance towards the target
4. Select the most appropriate stance classification number

IMPORTANT: Respond ONLY with the single number corresponding to your classification.
DO NOT include any explanation, words, punctuation, or other characters.
```
Evaluator 根据该提示给出答案。



Experience示例1:
```
Comprehensive Guide for Better Utilization of Multi-Modal Information

When dealing with multi-modal information, such as text and images, it is crucial to adopt a systematic approach to ensure accurate and reliable evaluations. Here’s a comprehensive guide to help you navigate through the complexities of combining text and image data:

#### Weighting Text vs. Image Information
- **Prioritize Text**: Generally, text is more reliable and direct compared to images, especially when it comes to stance detection. Text offers explicit statements and avoids potential visual misinterpretations. For instance, if the text explicitly mentions "Chloroquine," be cautious of images showing "Hydroxychloroquine" and ensure they align with the text's content.
- **Consider Image for Supplementary Context**: While text should be prioritized, images can provide supplementary context and supporting details, particularly when they offer relevant background information. However, be wary of images that contradict the text, as they may introduce misleading information. Use image analysis to complement text when it enhances understanding, but avoid over-reliance on images if they are irrelevant. Develop algorithms that filter out information that is contextually irrelevant. This can involve using domain-specific knowledge to identify topics and ensure that only relevant modalities contribute to the stance prediction.

#### Clues and Patterns for Stance Determination
- **Explicit Statements**: Focus on clear, unambiguous language in the text. Look for explicit mentions of ongoing research, clinical trials, or neutral statements that neither confirm nor deny the efficacy of the subject. Avoid assumptions based on visual elements unless they are directly supported by the text.
- **Complementary Information**: When text and image headlines contradict each other, prioritize the text content, as it typically offers more detailed and verifiable information. Cross-reference key phrases and facts from both sources to ensure consistency and avoid misinterpretations.
- **Tone and Language**: Analyze the overall tone and language used in the text to infer the stance. Look for expert opinions and official approvals (e.g., FDA, WHO) as they carry significant weight. Pay attention to numerical data or statistics mentioned in the text that support the claim.
- **Contextual Understanding**: Consider the broader context in which the text and images are presented. Understand the purpose and intent behind the content to better interpret the information.
- **Expert Opinions**: Pay attention to expert opinions and endorsements from reputable organizations. These can provide valuable insights and lend credibility to the information.
- **Visual Cues**: Analyze body language and attire in images for potential indicators of stance, but cross-reference these with textual evidence to avoid misinterpretations.
- **Temporal Considerations**: Be mindful of the time frame in which the text and images were created. Newer information may supersede older claims, so ensure you are using the most current data.


#### Reasoning Approaches to Avoid Mistakes
- **Cross-Reference for Consistency**: Ensure all claims in the text are supported by evidence or context. Cross-reference information from both text and image to avoid over-reliance on one modality. Maintain a critical stance by questioning assumptions and seeking additional evidence where necessary.
- **Logical Reasoning**: Use logical reasoning to reconcile conflicting signals by considering the source credibility and the broader context. Apply critical thinking to question implicit stances and seek explicit confirmations. Cross-reference textual claims with visual data to ensure consistency. Seek additional scientific evidence or references to validate causal relationships mentioned in the text.
- **Balanced Analysis**: Ensure a balanced analysis by considering multiple perspectives and sources before concluding a stance. Be cautious of making assumptions about causality based solely on visual data; always look for supporting textual evidence.
- **Specific Evidence and References**: Focus on specific evidence and references provided in the text to substantiate claims. Look for direct statements or explicit comparisons that clearly indicate a stance.

#### Cross-Referencing and Consistency
- **Consistent Analysis**: Cross-reference information across modalities to ensure consistency. Avoid drawing conclusions solely from one modality without sufficient supporting evidence from the other.
- **Critical Stance**: Maintain a critical stance by questioning assumptions and seeking additional evidence where necessary. Apply logical reasoning to reconcile conflicting signals and ensure a balanced analysis.

#### Hierarchical Reasoning Approach
- **Begin with Text Analysis**: Start by analyzing the text for explicit stance indicators. Look for clear, unambiguous language and direct statements that provide explicit information.
- **Consider Image for Supplementary Context**: Once the text has been analyzed, use image analysis to complement the text when it enhances understanding. Focus on indirect or contextual clues that might relate to the stance on the drugs, even if not explicitly stated.
- **Cross-Reference for Consistency**: Finally, cross-reference the findings from the text and image analysis to ensure consistency and avoid over-reliance on one modality. Use a multi-step reasoning process where initial predictions from each modality are cross-verified, incorporating domain knowledge to better interpret ambiguous signals. Implement a confidence score system to prioritize more reliable sources of information.
```


Experience示例2:
```
Comprehensive Recommendations for Better Utilization of Multi-Modal Information:
1. **Weighting Text Over Image When Explicit Positive Language Exists**:
   - Prioritize text content if it contains explicit positive language, as it often carries stronger influence over stance determination. For example, if the text states that a policy has "impressive results," the image should not contradict this claim.

2. **Attention to Clues in Text for Stance Determination**:
   - Focus on keywords like "beneficial," "positive results," and "impressive" in text to quickly identify a favorable stance. For example, if the text mentions "beneficial outcomes," the image should show positive reactions or results.

3. **Contextual Interpretation of Visual Data**:
   - Ensure visual data includes clear conclusions or statements that directly support or oppose the stance being analyzed. Lack of such clarity may lead to misinterpretations. For example, if the text discusses a negative impact, the image should not show positive outcomes.

4. **Weighting Text and Image Information**:
   - Prioritize text information as it often contains explicit statements that directly influence stance determination. Use images primarily for contextual verification or to identify potential contradictions, ensuring they provide relevant textual or visual cues. Avoid over-reliance on visual cues alone; prioritize text for detailed explanations and context. For example, if the text mentions a controversial issue, the image should not show a neutral or ambiguous scene.

5. **Identifying Relevance**:
   - Focus on identifying and interpreting only relevant information from each modality. Irrelevant or unrelated content in images should be disregarded to avoid misleading multi-modal fusion. Use metadata or context clues to distinguish between relevant and irrelevant information. For example, if the text discusses a specific event, the image should show related events or symbols.

6. **Contextualizing Image Content**:
   - If the image contains unrelated content, ensure it is clearly labeled as such and does not influence the interpretation of the text. Use metadata or context clues to distinguish between relevant and irrelevant information. For example, if the text discusses a specific policy, the image should not show unrelated scenes.

7. **Clues and Patterns to Pay Attention To**:
   - Look for explicit mentions of stance-related terms (e.g., "favor," "support"). Analyze visual cues like handshakes or gestures that may symbolize cooperation or support. Consider the overall context and tone of the content.

```
经验示例1主要内容：
文本与图像信息的权重：一般情况下优先考虑文本，因其更可靠直接，尤其在立场检测方面。但图像可提供补充背景信息，需警惕与文本矛盾的图像，避免过度依赖无关图像，开发算法过滤无关信息。
立场确定的线索和模式：关注文本中的明确表述，优先文本内容当文本和图像标题矛盾时，分析文本的语气、语言、上下文，关注专家意见和视觉线索，同时考虑时间因素。
避免错误的推理方法：交叉参考文本和图像以确保一致性，避免过度依赖单一模态，采用多步推理过程，对每个模态的初步预测进行交叉验证，使用置信度评分系统优先考虑更可靠的信息源。


经验示例2主要内容：
信息权重与利用方式：当文本存在明确积极语言时，优先以文本内容主导立场判断，因其对立场确定影响力更强。图像主要用于语境验证，避免过度依赖视觉线索，应以文本获取详细解释与语境。
相关性判断：从文本和图像模态中精准识别相关信息，摒弃不相关内容，防止误导。利用元数据或语境线索区分，若图像有不相关内容，需明确标注，避免干扰文本解读。
线索挖掘：关注文本中如 “有益”“支持” 等立场相关的关键词，同时留意图像里像握手等象征合作支持的视觉线索，结合整体语境与语气进行分析。


类似经验示例1的经验，存在“开发对应的算法”、“置信度评分”、“寻求专家意见”等不和场景的应对方式；类似经验二的经验，相对较好的捕捉了判断时应该注意的问题。
目前的普遍存在的问题是经验大多过长，相似表述的经验多一些，可能整体的表述不一样，但是每个表述中的几个小点是一个意思。



训练过程示例1：
文本内容：
```
Sometimes you see what's perfectly unspoiled..... StayHome lockdowneffect Hydroxychloroquine COVID19  Stay At Home Save Lives
```
文本知识：
```
stayhome: Social distancing, also known as physical distancing, involves maintaining a physical distance from others and avoiding large gatherings to reduce the spread of contagious diseases like COVID-19. It includes measures like school and workplace closures, isolation, and restrictions on movement to suppress disease transmission and avoid overburdening healthcare systems.

hydroxychloroquine: Hydroxychloroquine, sold as Plaquenil, is used to prevent and treat malaria and manage conditions like rheumatoid arthritis and lupus. Common side effects include vomiting and blurred vision, while severe risks include allergic reactions and irregular heart rate. It has been studied for COVID-19 but found ineffective and potentially harmful.

covid19: Coronavirus disease 2019 (COVID-19) is caused by SARS-CoV-2 and has led to a global pandemic. Symptoms include fever, fatigue, cough, and breathing difficulties, with some people experiencing long-term effects. Transmission occurs through respiratory droplets and contact with contaminated surfaces. Vaccines and supportive care are key preventive and treatment measures.
```
图像知识：
```
golden poppies: Eschscholzia californica, a native North American plant, blooms in vibrant colors and is used as an ornamental and culinary plant, officially adopted as California's state flower in 1903.

nmisty mountain sky: Nüwa, a Chinese mythological figure, created humanity and repaired the heavens, symbolizing creation and restoration in Chinese culture.

nserene beauty: Daniel DeVito, an American actor and filmmaker, is known for his roles in classic films like \"Taxi\" and \"It's Always Sunny in Philadelphia,\" and has produced notable works such as \"Pulp Fiction.\"
```
目标：
```
The stance on the use of \"Chloroquine\" and \"Hydroxychloroquine\" for the treatment or prevention from the coronavirus is:
```
正确答案：2（Neutral）
三个模态答案（文本、图像、多模态）：1（Against）、2（Neutral）、1（Against）
反思：
```
### Concise Recommendations for Better Utilization of Multi-Modal Information. 
**Weighting Text vs. Image Information**:
- Prioritize text content over images unless the image provides critical context or contradicts the text.
- Use image analysis to complement text when it enhances understanding, but avoid over-reliance on images if they are irrelevant.

**Clues and Patterns for Stance Determination**:
- Focus on explicit statements about the topic (e.g., \"hydroxychloroquine,\" \"COVID-19\").
- Look for contextual clues that support or contradict the stance (e.g., \"StayHome,\" \"StayAtHomeSaveLives\").
- Consider the overall tone and intent of the message.

**Reasoning Approaches to Avoid Mistakes**: 
- Ensure that both modalities are analyzed before making a stance decision.\n   - Apply a hierarchical approach where text takes precedence unless the image provides essential context.
- Avoid labeling stances based solely on irrelevant or ambiguous information.
```
这里的反思主要强调了优先利用文本，要将文本和图像结合起来。存在未针对该样本进行总结的问题。

训练过程示例2：
文本：
```
Trump on his assertion of the drug Hydroxychloroquine as being the answer to treat patients with COVID19 backpedaling NotADoctor Trump NotTested liability
```
文本知识：
```
Hydroxychloroquine: Used to prevent and treat malaria and manage conditions like rheumatoid arthritis, lupus, and porphyria cutanea tarda; effective but associated with potential side effects and risks, including retinopathy and irregular heart rate.

COVID-19: A highly contagious disease caused by SARS-CoV-2, characterized by symptoms ranging from mild to severe, with older individuals at higher risk. Transmission occurs via respiratory droplets and close contact, and prevention includes vaccines, social distancing, and hygiene practices.

Trumpet: A brass instrument commonly used in classical and jazz ensembles, produced by vibrating lips in a mouthpiece, with variations in pitch and construction, including the cornet, which differs in bore shape and valve design.
```
图像知识：
```
Basketball court: A rectangular playing surface with polished wood floors indoors, typically maple, and 10-foot-high rims on each basket; outdoor courts are made from paving materials like concrete or asphalt.
```
答案：1（Against）
三个模态答案（文本、图像、多模态）：1（Against）、2（Neutral）、1（Against）
反思：
```
### Concise Recommendations for Better Utilization of Multi-Modal Information\. **Weighting Text vs. Image Information**:
- Prioritize text information over image when it provides clear, relevant signals.
- Implement a confidence threshold for image relevance; if the image is highly irrelevant, consider disregarding it entirely.

**Clues for Determining Stance**:
- Focus on key phrases and sentiments expressed in the text.
- Look for explicit statements or contradictions that align with the stance labels.

**Avoiding Mistakes**:
- Ensure robust text analysis to capture nuanced stances.
- Use image relevance filters to prevent misclassification due to irrelevant visuals.
```
这个反思仍然也强调了优先考虑文本，同时提到了使用“图像相关性过滤器”这个奇怪的建议。

整个训练数据中有超过100条反思提到了“Prioritize text information over image”。



根据训练结果，目前存在的问题：
- 融合后的经验文本长度过长，重复度高了一些（应该是源于融合前的反思相似度就比较高）
- 生成的反思不够贴近于当前训练的数据，比较空泛，应该是因为在第二步反思时并未传入这个数据的信息，模型第二部反思完全依赖于第一步反思的结果，对样本的针对性降低了。
- 生成的反思有时候会包含一些无法在对话中做到的方法（设计算法之类的），这里应该可以在提示中指定模型的角色来解决。
- wiki 搜索结果有错误，某些情况下有词条搜索不到。这里应该是某些情况下网络异常导致搜索返回空字符串，需要添加 wiki 搜索重试机制，重新生成文本和图像知识。


测试结果：MCCQ in-target 400 条数据，F1 为 34.62。论文中 900 条数据 Qwen-VL 的 F1 为 50.51。
测试样本示例1：
文本：
```
Chloroquine vs. Hydroxychloroquine!   Prof. Pauline Byakika Kibwika shares the myths behind the two drugs with regards COVID-19 treatment.  Prof. Byakika is an Associate Professor of Medicine at MakerereU College of Health Sciences
```
文本知识：
```
Chloroquine: An antiparasitic medication primarily used to treat malaria by increasing haemoglobin levels toxic to the parasite. It's also used for amebiasis, rheumatoid arthritis, and lupus, though its use for COVID-19 is controversial and not recommended by the NIH.

COVID-19: A contagious disease caused by the SARS-CoV-2 coronavirus, characterized by fever, fatigue, cough, and breathing difficulties. Symptoms can range from mild to severe, with older individuals at higher risk. Transmission occurs via respiratory droplets and close contact, and vaccines are available for prevention.

Hydroxychloroquine: An antimalarial medication used to treat malaria and autoimmune diseases like rheumatoid arthritis and lupus. Common side effects include vomiting and blurred vision; severe risks include retinopathy and irregular heart rate. Its use for COVID-19 is controversial due to potential side effects.

Makerere University College of Health Sciences**: A constituent college of Makerere University in Uganda, offering undergraduate and postgraduate courses in biomedical sciences, health sciences, human medicine, and public health.

Prof. Pauline Byakika Kibwika: A Ugandan specialist physician, internist, epidemiologist, academic, and researcher. She was appointed as the vice chancellor of Mbarara University of Science and Technology in August 2024, succeeding as vice president of the Uganda Medical Association from 2017 to 2019.
```
图像知识：
```
Expert: Expect is a Tcl extension for automating interactions with text-terminal programs, originally developed in 1990 for Unix and now available on multiple platforms.

Covid-19: Coronavirus disease 2019 (COVID-19) is a contagious illness caused by SARS-CoV-2, with symptoms ranging from mild to severe, and transmission primarily through respiratory droplets and contact with contaminated surfaces.

Chloroquine: Chloroquine is an antimalarial drug that increases haeme levels in the blood, killing parasites and treating malaria, amebiasis, rheumatoid arthritis, and lupus. It has potential side effects and is not recommended for COVID-19 treatment.
```
预测：0（Favor），正确：2（Neutral）

搜索的经验：
经验一总结：
信息权衡：文本优先，因其含核心信息；图像补充，不矛盾，不过度依赖。
确定立场：关注文本明确陈述、图像视觉元素和上下文，留意情感线索。
推理方法：做上下文整合和一致性检查，区分显隐线索。
避免错误：分析上下文，识别语气，核查事实。
额外建议：选可信来源，抓上下文线索，结合多模态证据。
信息整合：图像与文本结合，图像不与文本立场矛盾。
立场确定：文本为立场检测重点，图像补充；关注短语关键词和间接指标。
避免错误推理：结合多模态数据，用领域知识解读模糊信号，确保立场一致。

经验二总结：
信息权衡：优先用文本做立场检测，因其更可靠直接；图像可作补充背景，但要防与文本矛盾及过滤无关信息。
立场线索：关注文本明确表述，结合互补信息，分析语气、上下文、专家意见和视觉线索，留意时间因素。
推理方法：交叉参考信息保一致，逻辑推理调冲突，平衡分析多视角，依文本具体证据下结论。
一致性：多模态交叉参考，保持批判，找额外证据。
推理步骤：先文本找立场指标，再图像补背景，最后交叉参考，用多步推理和置信度评分系统。

经验三总结：
信息权重：文本信息优先于模糊图像，含明确立场的文本权重更高，依据临床试验结果中立且图像无新信息的情况，优先采信文本证据，独立验证外部因素可信度。
确定立场线索：找明确观点表述，分析文本上下文，关注语言情感指标和图像线索，警惕误读，留意矛盾，区分主次信息。
推理防错方法：保证图文立场一致或有上下文，逻辑推理连接模态，多模态融合重实证，评估外部因素，先分析文本情感再用图像验证，结合领域知识，逻辑检查，用概率模型处理矛盾，开发辨真伪算法，分析上下文解矛盾，关注核心信息，交叉验证，分析图像相关性，交叉参考信息，避免误读图像。

三个经验有很多相似的地方，感觉模型在最后判断的时候没能很好利用这些信息。
其他一些样本也是类似的经验（原本经验池里的经验同质化就比较严重）。

问题：
- 经验大多是泛泛而谈，不够有针对性
- 相似度匹配的问题，目前使用文本拼接进行经验搜索



# 2025-04-15
![[Pasted image 20250416105003.png]]
1. 优化目前的提示设计
	- 三个智能体不仅要返回结果，还需要返回给出这个结果的原因
	- 第一步反思需要从三个模态的角度得到结果，即分析为什么xxx模态能够判断正确（错误）
	- 第二步的反思提示中需要加入原始信息，而不仅仅是利用第一步反思得到的结果
2. 模块重构
	- 经验池以经验为值，（文本、文本知识、图像、图像知识）为键。
	- 在三个智能体进行推理之前进行经验池检索，分为文本相似度和图像相似度进行检索，两者取平均获得最终相似度，选取大于阈值的 top_k 个相似经验作为上下文辅助智能体推理；若无大于相似度阈值的经验，直接进行推理。
	- 经验池的更新中，智能体反思后可以获得一个经验，使用该经验去更新之前检索获得的 top_k 个经验（即进行融合）；如果之前没有获得相似经验，就直接将该经验存入经验池。

仅优化提示设计后的 400 条 MCCQ in-target 测试的 F1 为 28，相较于未更改提示下降了 6 个点左右。
原因可能在于：
更改提示设计之前，辅助推理的经验很长，模型可能更本没从中提取什么信息；
更改提示之后，辅助推理的经验更有针对性、更加简明扼要，模型采用文本信息和经验文本之间的相似度进行搜索获得的经验并不能匹配当前任务（之前不管怎么搜索获得的经验都很长，经验匹不匹配根本没什么意义），导致模型根据经验误判（有时候几条经验中还有冲突的部分）。


# 2025-04-21
1. wiki的例子具有立场，比如有些药物有问题（影响不是很大
    1. Text summary 生成情感中心的summarization
2. Experience pool设置
    1. 基于经验去判断用哪个模态？决定使用哪个模态，然后再基于确定要使用的模态，输入，再得到结果。
    2. 出现了againt和favor neutral的立场（待定）。
3. 微调小模型
    1. 在训练的时候，一个样例会有一个reflection的输出，由此可以得到一个训练集：包括了文本图像的特征+summary，也得到了一个reflection。
    2. 在测试的时候，可以先把文本图像特征+summary 喂给训练好的LLM，然后生成reflection，再联合这五个特征喂给KEBERT+ViT
4. 思考一下要不要微调LLM

|            | MRUC  |       | MTWQ  |       | MCCQ  |
| :--------: | :---: | :---: | :---: | :---: | :---: |
|            |  RUS  |  UKR  |  MOC  |  TOC  |  CQ   |
|    TMPT    | 43.56 | 59.24 | 55.68 | 46.82 | 67.67 |
|  Original  | 36.92 | 45.93 | 55.60 | 36.61 | 53.87 |
|   LLaVA    | 36.73 | 44.89 | 49.03 | 38.64 | 54.27 |
| 中性的Summary | 34.78 | 45.02 | 55.52 | 38.11 | 52.19 |
| 修改测试方式（2a） | 32.71 | 47.69 | 52.60 | 32.71 | 56.41 |
|   微调小模型    | 38.85 | 52.58 | 55.70 | 47.02 | 62.71 |

# 2025-04-28
1. 在paper的基础上，text 中加入回复的经验进行微调
2. TMPT+COT/ TMPT+EX

完成Qwen-VL 微调+Ex 的微调，在2个数据集，分别是zero-shot和in-target两种设置上。

Qwen-VL+Ex: mccq in_target 51.21
Qwen-VL: mccq in_target 55.63


# 2025-05-03
1. 纯文本微调
2. 图像caption+文本 微调
3. visual caption、visual summary、text information、text summary、联合experience pool得到的文本经验

|                | MRUC  |       | MTWQ  |       | MCCQ  |
| -------------- | ----- | ----- | ----- | ----- | ----- |
| Qwen Only Text | 40.31 | 48.82 | 48.82 | 40.30 | 68.57 |
| Qwen All       | 39.54 | 52.31 | 55.76 | 40.99 | 60.13 |
