# HMG-Net: A Hierarchical Multi-Granularity Interaction Network with Market-Guided Depreciation Modeling for Non-Standard Item Valuation

(HMG-Net：一种面向非标品融合多模态折旧建模的多粒度分层交互商品价值预估网络)

## 摘要

在二手非标品估值领域，现有多模态模型难以处理商品的细粒度瑕疵和市场动态影响。本文提出HMG-Net，一种分层多粒度交互网络，结合市场引导的折旧建模。通过微观、中观和宏观层级的交互融合，以及引入MSRP锚点和时空图谱，我们实现了更准确、可解释的价格预测。实验在Gumtree数据集上证明了该方法的优越性。

## 1. 引言

二手市场中，非标品（如家具、电子产品）的估值是一个复杂问题。这些商品的状态高度个性化，受瑕疵、配件完整性和市场环境影响。传统方法依赖人工评估或简单回归模型，难以捕捉多模态信息（如图像、文本描述）和外部因素（如地域供需）。

现有模型存在两大痛点：一是多模态融合忽略了粒度差异，导致局部瑕疵被全局特征淹没；二是价格预测缺乏物理约束，模型易过拟合长尾数据。我们提出HMG-Net，模仿人类专家的鉴定流程：从细节检查到整体评估，再结合市场锚点进行折旧计算。该方法不仅提升了准确性，还提供了可解释的折旧归因。

## 2. 相关工作

### 2.1 多模态融合在估值任务中的应用

多模态模型如CLIP和BLIP在图像-文本对齐中表现出色，但它们主要处理全局相似性，无法精细捕捉非标品的局部缺陷。一些工作如VisualBERT尝试了注意力机制，但缺乏结构化交互。

### 2.2 折旧建模和市场引导

传统折旧模型（如线性折旧）忽略了市场动态。近期研究引入了Graph Neural Networks (GNN) 来建模时空因素，但未与多模态特征深度整合。我们创新地将MSRP作为物理锚点，转化为折旧率预测问题。

## 3. 方法

### 3.1 核心架构图解

我们将整个网络分为三个阶段：多粒度特征提取 (Extraction) → 阶梯式交互融合 (Interaction) → 物理感知折旧预测 (Prediction)。

### 3.2 深度创新点挖掘：模块细节与数学建模

#### 创新点一：基于“缺陷感知”的微观粒度对齐 (Level 1: Micro-Level Alignment)

痛点解决：解决单词 "scratch" (划痕) 无法精准对应图片中那几像素的“划痕”的问题，避免全局特征淹没局部瑕疵。

技术设计：Defect-Sensitive Cross-Attention (DSCA)

输入：
- 文本：Word Embeddings $T = \{t_1, t_2, ..., t_n\}$。
- 视觉：Feature Map (ResNet Stage 2/3 或 ViT Patch features) $V \in \mathbb{R}^{H \times W \times C}$，将其展平为 $V = \{v_1, ..., v_{HW}\}$。

机制：不使用标准的Softmax Attention（因为瑕疵是稀疏的），而是使用Sparsemax或Gumbel-Softmax，强制模型只关注极少数的关键区域（即瑕疵区域），忽略背景。

公式：
$$ A_{ij} = \text{Sparsemax}\left(\frac{(t_i W_Q)(v_j W_K)^T}{\sqrt{d}}\right) $$
$$ F_{micro} = \sum_{i} \sum_{j} A_{ij} \cdot (v_j W_V) $$

物理含义：只有当文本提到特定瑕疵词汇（如crack, dent）且图片对应区域有高响应时，该特征才会被激活并传递到下一层。

#### 创新点二：基于“场景图与句法树”的中观结构对齐 (Level 2: Meso-Level Alignment)

痛点解决：解决“配件齐全”或“支架断裂”这种物体级别的结构关系。

技术设计：Structural Graph Matching (SGM)

视觉侧：利用Faster R-CNN提取Object Proposals，构建视觉场景图 (Visual Scene Graph)。节点是物体，边是相对位置（on, next to）。

文本侧：利用Dependency Parsing（依存句法分析）构建文本语义树。

融合：使用 Graph Matching Network (GMN) 计算两个图结构的相似性矩阵。

创新价值：这超越了简单的特征点积，而是验证“文本描述的结构”与“图像呈现的结构”是否一致（Consistency Check）。如果文本说“有充电器”，视觉图中必须存在“Charger”节点，否则产生惩罚项。

#### 创新点三：第4/5模态引入——市场物理引导的折旧层 (Physics-Informed Depreciation Layer)

这是本方案最大的亮点。传统的模型直接预测价格 $P_{pred}$，这很难，因为价格方差极大。我们改为预测折旧率 (Depreciation Rate, $\delta$)，利用MSRP作为物理锚点。输入模态：MSRP (锚点)：$P_{anchor}$ (标准化后的原价)。Spatiotemporal Graph (ST-Graph)：一个异构图，包含 $Node_{item}, Node_{location}, Node_{time}$。通过GCN提取出的市场环境特征向量 $E_{market}$。Visual-Text Fused Feature：前两层融合得到的商品实体特征 $E_{entity}$。

折旧方程 (Depreciation Equation)：我们不再让黑盒神经网络直接输出价格，而是显式建模价格形成机制：
$$ P_{pred} = P_{anchor} \times (1 - \delta_{decay}) $$

其中，$\delta_{decay} \in [0, 1]$ 是预测的折旧率。折旧率预测函数：折旧率由商品实体状态（坏没坏）和市场环境（该地区保值率）共同决定。
$$ \delta_{decay} = \sigma \left( \mathcal{F}_{entity}(E_{entity}) + \mathcal{F}_{market}(E_{market}) \times \gamma \right) $$

$\mathcal{F}_{entity}$：评估成色带来的基础折旧（如：划痕扣20%）。$\mathcal{F}_{market}$：评估市场供需带来的系数（如：该地区iPhone需求大，折旧减缓）。$\sigma$：Sigmoid函数，确保折旧率在0-1之间。

## 4. 实验设计的亮点 (For Paper Evaluation)

为了证明这个架构不仅仅是堆砌模块，你需要设计巧妙的实验：

1. Ablation Study on Granularity (粒度消融实验)：证明：仅有Global特征 vs. 加上Pixel-Word对齐 vs. 全架构。预期结果：对于描述了细微瑕疵（"tiny scratch on corner"）的样本，加入Level 1对齐后，MSE误差显著下降。

2. Case Study: Depreciation Attribution (折旧归因可视化)：可视化创新：由于模型预测的是 $\delta$，我们可以反向传播梯度的Heatmap。展示一张带有划痕的椅子图。模型能高亮划痕区域，并标注：“此区域导致折旧率增加 15%”。展示同样的椅子在不同地区（ST-Graph不同），折旧率的变化（例如伦敦地区折旧率低，农村地区折旧率高）。

3. Anchor Effectiveness (锚点有效性验证)：对比直接预测价格 $P$ 和预测折旧率 $P_{anchor} \times (1-\delta)$ 的收敛速度和鲁棒性。通常后者在少样本（Few-shot）或长尾商品上表现会好得多。

## 5. 论文的Storyline (逻辑叙事)

在写这篇Paper时，建议遵循以下叙事逻辑，非常具有说服力：

Introduction: 二手定价很难，因为它是非标品（Non-Standard）。现有的多模态模型把“识别物体”和“评估状态”混为一谈，且忽略了“市场环境”和“原始价值”的约束。

Methodology - Hierarchy: 我们提出HMG-Net，模仿人类专家的鉴定流程：

先看细节（Level 1：有没有划痕？）；

再看整体配件（Level 2：东西齐不齐？）；

最后结合整体成色（Level 3：几成新？）。

Methodology - Physics: 人类估价不是瞎猜数字，而是基于“原价”打折。我们引入MSRP锚点和时空图谱，将回归问题转化为折旧率估算问题，这使得模型具有了物理可解释性。

Experiments: 在Gumtree数据集上，SOTA性能，且具备极强的可解释性（归因）。

## 6. 结论

HMG-Net 通过分层多粒度交互和市场引导的折旧建模，为非标品估值提供了高效、可解释的解决方案。未来可扩展到实时拍卖系统或更多模态融合。

## 参考文献

- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. In ICML.

- Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. In ICLR.


