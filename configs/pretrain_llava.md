# model — 模型结构配置
| 参数 | 值 | 含义 |
| --- | --- | --- |
| vision_backbone | "dinov3_vitl16" | 视觉编码器：DINOv3 ViT-L/16 |
| vision_source | "torch_hub" | 视觉编码器来源：torch.hub |
| vision_repo | "facebookresearch/dinov3" | torch.hub 的 repo ID |
| vision_model_name | "dinov3_vitl16" | 模型名称 |
| vision_pretrained | false | 是否加载预训练权重（实际从 vision_checkpoint_path 加载） |
| vision_checkpoint_path | "...pth" | DINOv3 预训练权重路径 |
| alignment_head_weights | "...pth" | dinotxt alignment head 权重路径 |
| llm_base | "...qwen3-vl-8b" | Qwen3-VL-8B 的本地路径 |
| llm_quantization | "8bit" | Phase1 默认对冻结的 Qwen 使用 8-bit 量化加载 |
| llm_quant_compute_dtype | "bf16" | 4-bit 路线时的计算 dtype；保留为 bf16 也便于后续切换 |
| llm_quant_double_quant | true | 4-bit 路线时是否开启 double quant；8-bit 下会被忽略 |
| embed_dim_dino | 1024 | DINOv3 输出特征维度 |
| alignment_dim | 1024 | dinotxt head 输出维度（与 DINOv3 相同） |
| hidden_size_qwen | 4096 | Qwen 文本嵌入维度 |
| adapter_hidden_dim | 2048 | Adapter 中间层维度（MLP: 1024→2048→4096） |
| use_deepstack_injection | false | Phase1 关闭：是否启用 DeepStack 注入 |
| deepstack_num_layers | 4 | DeepStack 注入层数 |
| adapter_dropout | 0.0 | Adapter dropout 概率 |

# data — 数据配置
| 参数 | 值 | 含义 |
| --- | --- | --- |
| train_manifest | "...pretrain.jsonl" | 训练数据清单（JSONL 格式） |
| val_manifest | "...train_val.jsonl" | 验证集清单（从原始 pretrain JSONL 切分得到） |
| image_root | null | 图片根目录（manifest 中的 image 是绝对路径，所以为 null） |
| val_image_root | null | 验证集图片根目录；默认与训练一致 |
| image_size | 336 | 图片 resize 到 336×336 |
| max_text_length | 320 | token 序列最大长度（prompt + target） |
| num_workers | 8 | DataLoader 的 worker 进程数 |

# training — 训练配置
| 参数 | 值 | 含义 |
| --- | --- | --- |
| output_dir | "./outputs/pretrain_llava" | checkpoint 输出目录 |
| use_run_subdir | true | 每次训练在 output_dir 下自动新建独立 run 子目录 |
| run_name | null | 可手动指定本次 run 名称；为 null 时自动按时间戳生成 |
| batch_size | 1 | 每卡 batch size（Phase1 只训练 adapter，显存占用小） |
| num_epochs | 1 | 训练轮数 |
| learning_rate | 1.0e-4 | 学习率 |
| weight_decay | 0.01 | 权重衰减 |
| warmup_steps | 1000 | 学习率预热步数 |
| gradient_accumulation_steps | 8 | 梯度累积步数（effective batch = 1×8 = 8） |
| max_grad_norm | 1.0 | 梯度裁剪最大范数 |
| mixed_precision | "bf16" | 混合精度：bf16 / fp16 / none |
| log_every | 10 | 每多少步打印日志 |
| eval_every | 200 | 每多少个 optimizer step 跑一次验证 |
| eval_max_batches | 64 | 单次验证最多评估多少个 batch |
| eval_batch_size | 1 | 验证时每卡 batch size |
| save_every | 1000 | 每多少步保存 checkpoint |
| device | "cuda" | 训练设备 |

# training.wandb — 可视化配置
| 参数 | 值 | 含义 |
| --- | --- | --- |
| enabled | true | 是否启用 Weights & Biases |
| project | "microvqa" | wandb 项目名 |
| entity | null | wandb team/个人名 |
| name | null | run 名称（null 则自动生成） |
| mode | null | online / offline / disabled |
| tags | ["phase1", "llava-pretrain"] | run 标签 |

---

### 几个关键点：
- `batch_size=1 + gradient_accumulation_steps=8 = effective batch 8`，适合 Adapter 训练
- `image_size=336`（你改的，比原来的 448 小，可减少显存）
- 现在 Phase1 也支持 `val_manifest`，训练中会定期记录 `val/loss` 到终端、`train_log.jsonl` 和 wandb
- `use_deepstack_injection=false` — Phase1 只训 Adapter，DeepStack 后续开启
- `llm_quantization=8bit` — 现在默认把冻结的 Qwen 以量化方式加载，优先解决单卡显存占用；如果环境里没装 `bitsandbytes`，启动时会直接报清晰错误
- 每次启动会把日志、checkpoint、`resolved_config.yaml`、`run_info.json` 写进单独的 run 目录；如果使用 `--resume`，则会继续写回原来的 run 目录
