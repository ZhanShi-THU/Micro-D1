# qwen3_dinov3.yaml 参数详解

## project_name & seed
| 参数 | 值 | 含义 |
| --- | --- | --- |
| project_name | "qwen3_dinov3_modular_vlm" | 项目名称 |
| seed | 42 | 随机种子 |

## model — 模型结构配置
大部分参数与 pretrain_llava.yaml 相同，区别在于：

| 参数 | 值 | 含义 |
| --- | --- | --- |
| use_deepstack_injection | true | Phase2/3 开启：启用 DeepStack 注入 |
| adapter_init_checkpoint | null | Phase2/3 加载 Phase1 权重：预训练 adapter 权重路径 |
| llm_quantization | "none" | 主训练默认不量化，便于后续解冻部分 LLM 层 |

## data — 显微 VQA 数据配置
| 参数 | 值 | 含义 |
| --- | --- | --- |
| unified_root | "...unified_vqa" | 显微 VQA 数据根目录 |
| train_manifest | "...train.jsonl" | 训练集清单 |
| val_manifest | null | 验证集（暂无） |
| test_manifest | "...test.jsonl" | 测试集清单 |
| summary_path | "...summary.json" | 数据集统计信息 |
| image_root | null | 图片根目录（manifest 用相对路径） |
| image_size | 448 | 图片 resize 到 448×448（比 Phase1 的 336 大） |
| max_text_length | 512 | token 序列最大长度（VQA prompt 更长） |

## training — 训练配置
与 pretrain_llava.yaml 区别：

| 参数 | 值 | 含义 |
| --- | --- | --- |
| batch_size | 2 | 每卡 batch size（更大因为训更多模块） |
| warmup_steps | 100 | 预热步数（更少，因为数据更小） |
| save_every | 1000 | 保存间隔（数据集小，1000 步约 1 epoch） |
| use_run_subdir | true | 每次训练在 output_dir 下创建独立 run 子目录 |
| run_name | null | 可手动指定 run 名；为空时自动生成时间戳目录 |

## training.phases — 多阶段训练配置（核心区别）
这是 qwen3_dinov3.yaml 独有的，用于显微数据微调的多阶段策略：

```yaml
phases:
  phase1:          # 阶段1：Adapter warmup
    name: "adapter_warmup"
    freeze_modules:
      - "vision_backbone"    # 冻结 DINOv3
      - "alignment_head"    # 冻结 dinotxt head
      - "llm_base"          # 冻结 Qwen LLM
    trainable_modules:
      - "adapter"           # 只训练 Adapter
    llm_tune_last_n_layers: 0
    learning_rate: 1.0e-4
    weight_decay: 0.01

  phase2:          # 阶段2：可选的 partial LLM 微调
    name: "optional_partial_llm_finetune"
    freeze_modules:
      - "vision_backbone"    # 仍冻结 DINOv3
      - "alignment_head"    # 仍冻结 dinotxt head
    trainable_modules:
      - "adapter"           # 继续训练 Adapter
      - "llm_last_layers"  # + 微调 Qwen 最后 N 层
    llm_tune_last_n_layers: 2    # 微调 Qwen 最后 2 层
    learning_rate: 5.0e-5         # 更小的学习率（LLM 用较小 LR）
    weight_decay: 0.01
```

### phase 参数详解
| 参数 | 含义 |
| --- | --- |
| freeze_modules | 明确冻结的模块列表 |
| trainable_modules | 明确可训练的模块列表 |
| llm_tune_last_n_layers | 额外解冻 Qwen 最后 N 层 transformer |

### Phase 选择方式
```bash
# 训练 phase1（只训 adapter）
python train.py --config configs/qwen3_dinov3.yaml --phase phase1

# 训练 phase2（adapter + 部分 LLM）
python train.py --config configs/qwen3_dinov3.yaml --phase phase2
```

## 两个 yaml 的关键区别
| 配置项 | pretrain_llava.yaml | qwen3_dinov3.yaml |
| --- | --- | --- |
| use_deepstack_injection | false | true |
| adapter_init_checkpoint | null | 可加载 Phase1 权重 |
| image_size | 336 | 448 |
| batch_size | 1 | 2 |
| data | LLaVA Pretrain | 显微 VQA |
| training.phases | 无 | 有（phase1/phase2） |
| wandb | 有 | 无 |

### 整体流程
Phase1 用 pretrain_llava.yaml 训好 adapter → Phase2/3 用 qwen3_dinov3.yaml 并设置 adapter_init_checkpoint 加载 Phase1 权重。

---

### 总结
1. qwen3_dinov3.yaml 核心新增了 `training.phases` 多阶段训练配置，支持分阶段训练 Adapter 和部分 LLM 层；
2. 相较于 pretrain_llava.yaml，该配置调整了 image_size、batch_size 等参数适配显微 VQA 数据，并开启了 DeepStack 注入；
3. 训练需分阶段执行，先通过 pretrain_llava.yaml 完成 Phase1 训练，再加载权重进行后续阶段的微调。

### 关于量化和后续阶段
- Phase1 的量化方案只适用于“冻结 Qwen、只训 adapter”的 warm-up。
- `qwen3_dinov3.yaml` 默认保持 `llm_quantization: "none"`，因为如果后面要解冻 `llm_tune_last_n_layers`，量化权重并不适合直接走当前这套全参数训练流程。
- 训练输出现在会按 run 单独落盘，并附带 `resolved_config.yaml` 与 `run_info.json`，避免不同实验日志和 checkpoint 混在一起。
