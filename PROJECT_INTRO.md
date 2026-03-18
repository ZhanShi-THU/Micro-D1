# 项目介绍：基于 DINOv3/dinotxt 与 Qwen 的模块化显微图像 VQA 系统

## 1. 项目背景

当前主流大规模多模态模型通常可以被理解为“语言模型 + 视觉编码器”的组合。这样的结构在开放域图文理解中已经取得了很强的能力，但它也存在一个明显问题：模型的整体性能往往受限于视觉分支的上限。尤其是在显微图像、病理图像或其他细粒度科学视觉场景中，视觉编码器如果对局部结构、细微形态和复杂组织关系的建模能力不足，语言模型再强，也难以弥补输入表征的缺陷。

另一方面，自监督视觉模型，特别是 DINOv3 这一类强视觉 backbone，在纯视觉表征上已经展现出明显优势。它们擅长学习高质量的 dense visual tokens，能够更稳定地覆盖局部结构和细粒度特征。但这类模型本身并不具备语言接口，无法直接接入 LLM，也不能天然参与语言生成或视觉问答任务。

因此，这个项目试图解决的核心问题是：

**能否在不从头联合训练整个多模态模型的前提下，把一个更强的视觉 backbone 接到一个现成的大语言模型上，构造出一个成本更低、结构更干净的显微图像 VQA 系统？**

## 2. 核心思路

这个项目的切入点不是重新设计一个复杂的视觉-语言融合框架，而是尽可能利用现有模型中已经存在的能力，把跨模态问题拆成几个相对独立的子问题。

整体思路可以概括为三步：

1. 使用 DINOv3 提供强视觉表征。
2. 使用 `dinotxt` 视觉对齐头，把视觉特征先拉到一个更接近文本语义的空间。
3. 使用一个非常轻量的 adapter，把这些视觉特征映射到 Qwen 解码器的表示空间，并直接作为前缀 token 输入语言模型。

这个项目的出发点并不是“让语言模型从头学会看图”，而是：

**先让视觉特征变得更接近语言可以处理的语义形式，再交给语言模型完成后续推理和生成。**

## 3. 当前代码中的真实实现

从当前代码实现看，这个系统是一个非常明确的模块化结构，而不是黑盒式端到端 VLM。

### 3.1 视觉编码器：DINOv3 + DINOText 对齐头

视觉分支定义在：

- [`models/vision_encoder.py`](/home/user/Project_files/project/models/vision_encoder.py)

这里的流程是：

1. 输入图像经过 DINOv3 backbone。
2. 提取 patch-level visual tokens。
3. 将这些 token 输入一个 DINOText 风格的 alignment head。

alignment head 的权重来自项目中已有的 checkpoint：

- [`pth/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth`](/home/user/Project_files/project/pth/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth)

这一步的意义在于：项目并没有直接把原始 DINOv3 token 生硬地接到 LLM 上，而是先借助一个已经具备视觉-文本对齐先验的中间层，把视觉特征推进到一个更适合后续语言建模处理的表示空间。

### 3.2 轻量 adapter：只做表示空间适配

中间 adapter 定义在：

- [`models/adapter.py`](/home/user/Project_files/project/models/adapter.py)

当前实现非常克制，只包含：

- `LayerNorm`
- `Linear`
- `Dropout`

也就是说，这个项目没有引入 cross-attention、Q-Former、fusion transformer 之类的复杂结构。adapter 的职责不是重新学习视觉语义，而是把已经对齐过的视觉表示进一步映射到 Qwen 解码器的 hidden size 上，让信息能够以尽量小的改动流入语言模型。

这个设计反映了一个很明确的实验假设：

**如果视觉特征本身已经具备较强的语义结构和一定的跨模态先验，那么跨模态融合本身不一定需要复杂模块。**

### 3.3 与语言模型的融合：视觉前缀拼接

多模态主模型定义在：

- [`models/modular_vlm.py`](/home/user/Project_files/project/models/modular_vlm.py)

当前融合方式非常直接：

1. 图像经过视觉编码器和 adapter，得到 visual embeddings。
2. 文本 token 经过 Qwen 输入 embedding 层，得到 text embeddings。
3. 在 sequence 维度上把 visual embeddings 拼接到 text embeddings 前面。
4. 通过 `inputs_embeds` 直接送入 LLM 解码器。

这意味着视觉 token 和文本 token 进入的是同一个 transformer 自注意力空间，二者不依赖额外的 fusion module，而是在 decoder 内部统一建模。

从工程设计上，这是一种非常干净的实现：

- 不改写 Qwen 的主体结构
- 不依赖额外的跨模态模块
- 不需要重建整个多模态训练框架

它更接近“外接视觉前缀的语言解码器”方案，而不是原生 Qwen3-VL 视觉塔的替换实现。

## 4. 训练策略

当前训练入口在：

- [`train.py`](/home/user/Project_files/project/train.py)

配置文件在：

- [`configs/qwen3_dinov3.yaml`](/home/user/Project_files/project/configs/qwen3_dinov3.yaml)

当前代码实现的是一个两阶段的最小侵入训练策略：

### Phase 1

- 冻结 DINOv3 backbone
- 冻结 DINOText alignment head
- 冻结 Qwen LLM
- 只训练 adapter

这一阶段的目标不是让系统从头学习跨模态语义，而是完成分布和维度上的适配，让视觉 token 能够稳定流入 Qwen 解码器。

### Phase 2

- 保持视觉部分冻结
- 继续训练 adapter
- 可选解冻 LLM 最后若干层

这一阶段允许模型在较低成本下做轻量级语言侧适配，但仍然避免了大规模端到端联合训练。

整体来看，这个训练策略非常符合项目的核心定位：

**不是从零训练一个新 VLM，而是在已有强视觉模型、已有对齐先验和已有强语言模型之间，用极少的可训练参数做桥接。**

## 5. 当前任务与数据范围

这个项目当前已经不是一个抽象方法草图，而是落在了具体的数据和任务上：

- 任务类型：显微图像多项选择 VQA
- 统一数据根目录：
  - [`/data1/staging_datasets/unified_vqa`](/data1/staging_datasets/unified_vqa)
- 主配置默认训练集：
  - [`/data1/staging_datasets/unified_vqa/manifests/merged/train.jsonl`](/data1/staging_datasets/unified_vqa/manifests/merged/train.jsonl)
- 主配置默认测试集：
  - [`/data1/staging_datasets/unified_vqa/manifests/merged/test.jsonl`](/data1/staging_datasets/unified_vqa/manifests/merged/test.jsonl)

目前统一数据集已经整合了：

- `microbench`
- `microvqa`
- `mms`
- `mmsci++`

并且当前本地 split 状态是：

- `microbench`：95% train / 5% test
- `mmsci++`：train only
- `microvqa`：test only
- `mms`：test only

当前模型的默认监督目标不是自由生成式 caption，而是输出标准化多项选择答案：

```text
The answer is (X)
```

其中 `X` 是零起点索引。

## 6. 当前项目已经证明了什么

从当前代码和流程来看，这个项目已经明确证明了以下几件事在工程上是可行的：

1. 可以把 DINOv3 这种强视觉模型接入 Qwen 解码器，而不依赖原生多模态视觉塔。
2. 可以利用 DINOText 风格 alignment head 作为跨模态桥接先验，而不是完全从随机初始化开始学视觉-语言映射。
3. 可以仅通过一个非常轻量的 adapter 和 prefix 拼接方式完成多模态接入。
4. 可以在统一显微图像 VQA 数据集上，用较低侵入性的训练流程对这套结构进行实验。

也就是说，这个项目已经构成了一个结构逻辑完整、训练路径清晰、数据接口统一的研究原型。

## 7. 当前项目还没有完全证明什么

为了让项目定位更准确，也需要明确它当前的边界。

### 7.1 它还不是完整论文结论

当前系统虽然已经能跑通数据、训练和评测链路，但它还没有在代码层面内置完整的研究对照与消融，例如：

- 与原生 Qwen3-VL 视觉分支的严格 baseline 对照
- 不使用 `dinotxt` alignment head 的消融
- 与更复杂 fusion 模块的对照
- 不同冻结策略、不同解冻层数的系统比较

所以它目前更接近一个强研究原型，而不是一个已经完成全部实验论证的最终论文系统。

### 7.2 `dinotxt` 提供的是强先验，不是彻底解决跨模态问题

从方法论上说，`dinotxt` head 很关键，因为它显著缩小了视觉表示与语言表示之间的鸿沟。但从当前实现角度，更准确的表述应该是：

**它提供了一个很强的跨模态对齐先验，而不是让视觉特征直接变成语言模型天然可理解的“文本 token”。**

这也是为什么项目中仍然需要 adapter 和后续训练，而不是完全零适配直接推理。

### 7.3 当前验证范围是显微图像 MCQ VQA

项目当前真正落地和默认支持的是显微图像多选问答，而不是开放域通用多模态对话系统。因此它更适合被描述为：

**一种面向细粒度科学视觉任务的模块化多模态实验框架。**

## 8. 这个项目真正的价值

这个项目的价值，不只是“换了一个更强视觉 backbone”，而是提出并实现了一种很清晰的构建范式：

### 8.1 视觉能力、跨模态对齐能力和语言生成能力被显式拆分

在这里：

- DINOv3 负责视觉理解
- DINOText alignment head 负责提供跨模态先验
- adapter 负责空间适配
- Qwen 负责语言建模与输出

这种解耦使得系统结构更容易分析，也更容易替换其中某一部分。

### 8.2 训练成本显著低于从头训练一个 VLM

由于项目默认冻结大部分大模型参数，只训练 adapter 或少量 LLM 顶层，整个训练成本远低于大规模联合训练。对于科研探索和领域实验，这种路线更现实。

### 8.3 更适合研究“强视觉 backbone 是否真的能提升下游多模态理解”

很多原生 VLM 的视觉部分本身就是性能瓶颈，因此很难判断“语言模型推理能力不够”和“视觉输入不够好”哪个是主要问题。这个项目把视觉分支替换成更强的自监督视觉模型，并尽量保持下游结构简单，有助于更干净地研究这个问题。

## 9. 一句话总结

如果用一句话描述这个项目，可以这样说：

**这是一个面向显微图像多项选择 VQA 的模块化多模态研究原型，它利用 DINOv3 的强视觉表征、dinotxt 提供的跨模态对齐先验，以及一个极简 adapter，把外部视觉 token 接入 Qwen 解码器，从而在不重训整个 VLM 的前提下构造出一个低侵入、低成本、结构清晰的多模态系统。**

## 10. 当前推荐工作流

如果按当前代码和数据状态使用这个项目，推荐工作流是：

1. 确保使用 `microvqa` Conda 环境。
2. 使用统一后的数据目录：
   - [`/data1/staging_datasets/unified_vqa`](/data1/staging_datasets/unified_vqa)
3. 使用默认配置：
   - [`configs/qwen3_dinov3.yaml`](/home/user/Project_files/project/configs/qwen3_dinov3.yaml)
4. 训练：

```bash
cd /home/user/Project_files/project
conda run -n microvqa python train.py --config configs/qwen3_dinov3.yaml
```

5. 评测：

```bash
cd /home/user/Project_files/project
conda run -n microvqa python eval.py --config configs/qwen3_dinov3.yaml mcq
```

如果后续需要把这份介绍进一步压缩成摘要、README 前言、开题说明或论文方法概述，也可以在这份文本基础上继续改写。
