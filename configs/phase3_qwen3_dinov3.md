# Phase 3 配置说明

`phase3_qwen3_dinov3.yaml` 现在默认用于最终的显微 reasoning-supervised 定向微调。

`phase3_qwen3_dinov3_lora128.yaml` 与它使用同一套 reasoning 数据和训练策略，唯一的核心差别是 LoRA 配置从：

- `r=64, lora_alpha=128`

切换为：

- `r=128, lora_alpha=256`

因此现在不再保留“默认版走旧 answer-only，lora128 版走 reasoning”这种分叉。

## 模型策略

- 冻结 `DINOv3 backbone` 的前 20 个 blocks
- 解冻 `DINOv3 backbone.blocks[20:24]`
- 继续训练 `dinotxt alignment head`
- 继续训练 `adapter`
- `qwen3-vl-8b` 使用 `4bit QLoRA`
- `DeepStack injection` 保持开启

默认 LoRA 设置：

- `r=64`
- `lora_alpha=128`
- `lora_dropout=0.05`
- `target_modules=[q_proj, k_proj, v_proj, o_proj]`

如果要跑高 rank 版本，直接改用：

- `configs/phase3_qwen3_dinov3_lora128.yaml`

## 数据配置

Phase 3 默认改为使用 reasoning-supervised manifests：

- `data.train_manifest=/data1/staging_datasets/phase3_reasoning/mmsci_reasoning/train.jsonl`
- `data.val_manifest=/data1/staging_datasets/phase3_reasoning/mmsci_reasoning/val.jsonl`

输入策略：

- `image_preprocessing=pad_preserve`
- `image_size=448`
- `max_text_length=512`
- `prompt_style=reasoning`

这条主路径不再以纯 `The answer is (X)` 监督作为默认训练目标，而是使用：

```text
{reason}

The answer is (X)
```

因此更适合作为后续大规模 reasoning 数据的默认入口。

当前默认配置已经不再保留旧的 `answer_only + merged unified_vqa train/val` 训练主路径。

需要注意：

- `train_phase3.py` 现在只接受 `data.prompt_style=reasoning`
- `answer_only` 只继续作为 evaluation 侧的可选 prompt，用于对比和消融
- 如果在 Phase 3 训练配置里把 `prompt_style` 改回 `answer_only`，训练入口会直接报错

## 验证与 best checkpoint

Phase 3 训练过程中默认记录：

- `val/loss`

默认策略：

- `save_every=1000`
- `eval_every=5`
- `eval_accuracy_every=0`
- `eval_accuracy_max_samples=64`
- `eval_max_new_tokens=64`

默认 best checkpoint 改为按 `val loss` 更新，文件名为：

- `phase3_best_loss.pt`

同时仍保留：

- 周期保存：`phase3_step_<step>.pt`
- 最终保存：`phase3_final.pt`

如果确实需要在训练中同步看生成式指标，可以手动把 `eval_accuracy_every` 改成一个较大的值，再配合较小的 `eval_accuracy_max_samples` 使用；更推荐的主路径仍然是训练时看 `val/loss`，训练后用 `evaluation/` 目录下的离线脚本统一跑准确率。

## Phase 2 初始化

Phase 3 从完整 `Phase 2` checkpoint 初始化，支持加载：

- `adapter`
- `vision_alignment_head`
- `lora_state`

如果继续训过的 `Phase 3` checkpoint，也会额外恢复：

- `vision_backbone_top_blocks`
- `optimizer`
- `scheduler`
- `global_step`
- `optimizer_step`

## 常用命令

### 从 Phase 2 checkpoint 启动

```bash
cd /home/user/Project_files/project

accelerate launch --num_processes 8 train_phase3.py \
  --config configs/phase3_qwen3_dinov3.yaml \
  --phase2-checkpoint /path/to/phase2_reasoning_compatible_checkpoint.pt
```

### 中断后续训

```bash
cd /home/user/Project_files/project

accelerate launch --num_processes 8 train_phase3.py \
  --config configs/phase3_qwen3_dinov3.yaml \
  --resume /path/to/phase3_step_3000.pt
```
