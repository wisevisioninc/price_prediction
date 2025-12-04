# HMG-Net：一种面向非标品融合多模态折旧建模的多粒度分层交互商品价值预估网络

**Paper Title:** HMG-Net: A Hierarchical Multi-Granularity Interaction Network with Market-Guided Depreciation Modeling for Non-Standard Item Valuation  

---

## 📖 核心架构图解 (Conceptual Architecture)

将单纯的“感知问题”（看起来值多少钱）转化为了 **“多模态感知 + 推理 + 市场动力学”** 的综合问题：

> **看起来的状态 + 市场供需 + 原始价值 = 最终定价**

我们将整个网络分为三个阶段：

1.  **多粒度特征提取 (Extraction):** 提取像素/单词级、物体/句子级、全局/篇章级特征。
2.  **阶梯式交互融合 (Interaction):** 自底向上进行分层对齐 (Micro $\to$ Meso $\to$ Macro)。
3.  **物理感知折旧预测 (Prediction):** 基于 MSRP 锚点预测折旧率。

$$\text{Extraction} \rightarrow \text{Interaction} \rightarrow \text{Prediction}$$

---

## 🚀 深度创新点与数学建模

### 1. 微观粒度对齐：缺陷感知交叉注意力 (Level 1: Micro-Level Alignment)

**痛点解决：** 解决单词 "scratch" (划痕) 无法精准对应图片中那几像素的“划痕”的问题，避免全局特征淹没局部瑕疵。  
**技术设计：** **Defect-Sensitive Cross-Attention (DSCA)**

**输入：**
* 文本：使用miniLM模型进行特征提取，生成Word Embeddings $T = \{t_1, t_2, ..., t_n\}$。miniLM是一种高效的轻量级语言模型，提供紧凑且语义丰富的嵌入表示，通过蒸馏技术从大型模型中继承知识，减少计算开销同时保持高语义准确性。
* 视觉：使用Vision Transformer (ViT)提取Patch features，将Feature Map展平为 $V = \{v_1, ..., v_{HW}\}$。ViT通过自注意力机制捕捉全局和局部视觉依赖，提升对细微瑕疵的敏感度，并支持高效的预训练转移学习。

**机制与公式：**
不使用标准的 Softmax Attention（因为瑕疵是稀疏的），而是使用 **Sparsemax** 强制模型只关注极少数的关键区域（即瑕疵区域），忽略背景。为了增强对齐创新，我们引入动态阈值调整，使得注意力分数根据瑕疵严重度自适应缩放。此外，添加跨模态Transformer层，进一步融合文本和视觉嵌入，实现更深层的语义对齐。

$$
A_{ij} = \text{Sparsemax}\left(\frac{(t_i W_Q)(v_j W_K)^T}{\sqrt{d}} \cdot \alpha_{severity}\right)
$$

其中，$\alpha_{severity}$ 是基于预训练瑕疵检测器的动态缩放因子。

$$
F_{micro} = \text{Transformer}_{cross}( \sum_{i} \sum_{j} A_{ij} \cdot (v_j W_V), T )
$$

*物理含义：* 只有当文本提到特定瑕疵词汇（如 crack, dent）且图片对应区域有高响应时，该特征才会被激活并传递到下一层。通过miniLM和ViT的结合，以及跨模态Transformer，实现更精确的跨模态语义对齐，提升融合的创新深度和模型的泛化能力。

### 2. 中观结构对齐：场景图与句法树匹配 (Level 2: Meso-Level Alignment)

**痛点解决：** 解决“配件齐全”或“支架断裂”这种物体级别的结构关系验证。  
**技术设计：** **Structural Graph Matching (SGM)**

* **视觉侧：** 利用 DINO v3 结合 ViT 提取Object Proposals，构建**视觉场景图 (Visual Scene Graph)**。节点是物体，边是相对位置。DINO v3作为先进的Transformer-based检测器，提供更准确和高效的对象检测，使图节点更具语义丰富性，并通过去噪锚框机制捕捉细粒度关系。
* **文本侧：** 利用 Dependency Parsing 结合 miniLM 构建**文本语义树**。miniLM提供更先进的句子级嵌入，提高句法树的语义准确性，并支持上下文-aware的解析。
* **融合：** 使用 **Graph Matching Network (GMN)** 计算结构一致性，进行 Consistency Check。为了增加创新深度，我们引入多层图注意力融合（Graph Attention Fusion），允许跨模态图节点动态交互，并融入Gated Fusion机制以平衡模态贡献：

$$
S_{match} = \text{GMN}(G_{visual}, G_{text}) + \text{GAT}(E_{visual}, E_{text})
$$

$$
F_{meso} = \text{GatedFusion}(S_{match}, E_{visual} \odot E_{text})
$$

其中，GAT 表示 Graph Attention Network，用于增强特征对齐的鲁棒性；$\odot$ 表示元素-wise乘法。

### 3. 物理引导的折旧层 (Physics-Informed Depreciation Layer)

**这是本方案最大的亮点。** 传统的模型直接预测价格 $P_{pred}$ 很难（方差极大）。我们改为利用 **MSRP (建议零售价)** 作为物理锚点，预测 **折旧率 (Depreciation Rate, $\delta$)**。

**输入模态：**
* **MSRP (锚点)：** $P_{anchor}$ (标准化后的原价)。
* **时空图谱 (ST-Graph)：** 包含 $Node_{item}, Node_{location}, Node_{time}$ 的异构图，提取市场环境特征 $E_{market}$。
* **实体特征：** 前两层融合得到的商品特征 $E_{entity}$，通过增强的多模态融合（如ViT和miniLM的联合表示，以及跨模态Transformer）提供更丰富的输入。

**折旧方程 (Depreciation Equation)：**

我们显式建模价格形成机制：

$$
P_{pred} = P_{anchor} \times (1 - \delta_{decay})
$$

其中，$\delta_{decay} \in [0, 1]$ 是预测的折旧率。

**折旧率预测函数：**

折旧率由**商品实体状态**（坏没坏）和**市场环境**（该地区保值率）共同决定：

$$
\delta_{decay} = \sigma \left( \mathcal{F}_{entity}(E_{entity}) + \mathcal{F}_{market}(E_{market}) \times \gamma \right)
$$

* $\mathcal{F}_{entity}$：评估成色带来的基础折旧（如：划痕扣 20%）。
* $\mathcal{F}_{market}$：评估市场供需带来的系数（如：该地区 iPhone 需求大，折旧减缓）。
* $\sigma$：Sigmoid 函数，确保折旧率在 0-1 之间。

---

## 🧪 实验设计亮点 (For Evaluation)

为了证明架构的有效性，设计以下关键实验：

* **粒度消融实验 (Ablation Study on Granularity):**
    * 对比：`仅Global特征` vs `加上 Pixel-Word 对齐` vs `全架构`。
    * *预期：* 对于描述了细微瑕疵（"tiny scratch on corner"）的样本，加入 Level 1 对齐后，MSE 误差显著下降。
* **折旧归因可视化 (Depreciation Attribution):**
    * 反向传播梯度至输入层。
    * *展示：* 模型能高亮图片中的划痕区域，并标注：“此区域导致折旧率增加 15%”。
    * *展示：* 同样的商品在不同地区（ST-Graph 不同），折旧率的变化。
* **锚点有效性验证 (Anchor Effectiveness):**
    * 对比直接预测价格 $P$ 和预测折旧率 $P_{anchor} \times (1-\delta)$ 的收敛速度和鲁棒性。后者在少样本（Few-shot）或长尾商品上表现更好。

---

## 📝 论文叙事逻辑 (Storyline)

1.  **Introduction:** 二手定价难在“非标品”。现有模型混淆了“识别”与“状态评估”，且忽略了“市场”和“原价”。
2.  **Methodology - Hierarchy:** 模仿人类专家流程：先看细节（Level 1 划痕），再看配件（Level 2 完整度），最后看整体（Level 3 成色）。
3.  **Methodology - Physics:** 引入 MSRP 锚点和时空图谱，将回归问题转化为**折旧率估算问题**，赋予模型物理可解释性。
4.  **Experiments:** 在 Gumtree 数据集上实现 SOTA，并具备极强的可解释性。

---

## 📚 Citation

If you use this code or ideas, please cite:

```bibtex
@article{HMGNet202X,
  title={HMG-Net: A Hierarchical Multi-Granularity Interaction Network with Market-Guided Depreciation Modeling},
  author={Your Name},
  journal={arXiv preprint},
  year={202X}
}
```

### Related Works
- Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." ICML.
- Kipf, T. N., & Welling, M. (2017). "Semi-Supervised Classification with Graph Convolutional Networks." ICLR.
- Li, W., et al. (2022). "Multimodal Data Guided Spatial Feature Fusion." CVPR.
- Zhang, Y., et al. (2023). "Multimodal AI Framework for High-Potential Product Listings." NeurIPS.
- Wang, Z., et al. (2024). "IPL: Leveraging Multimodal Large Language Models for Intelligent Product Listing." ACL.
- Chen, X., et al. (2023). "Zero-Shot Retrieval for Scalable Visual Search." ECCV.
- Liu, J., et al. (2022). "Cross-Platform E-Commerce Product Categorization." WWW Conference.
