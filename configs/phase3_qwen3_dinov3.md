# Phase 3 配置说明

`phase3_qwen3_dinov3.yaml` 用于最终的显微 `Unified VQA` 任务定向微调。

## 模型策略

- 冻结 `DINOv3 backbone` 的前 20 个 blocks
- 解冻 `DINOv3 backbone.blocks[20:24]`
- 继续训练 `dinotxt alignment head`
- 继续训练 `adapter`
- `qwen3-vl-8b` 使用 `4bit QLoRA`
- `DeepStack injection` 保持开启

默认 LoRA 设置：

- `r=128`
- `lora_alpha=256`
- `lora_dropout=0.05`
- `target_modules=[q_proj, k_proj, v_proj, o_proj]`

## 数据配置

Phase 3 默认只使用 `Unified VQA`：

- `data.train_manifest`
- `data.val_manifest`
- `data.test_manifest`

主路径默认不混入 `VQAv2` 或 `LLaVA-Instruct`。

输入策略：

- `image_preprocessing=pad_preserve`
- `image_size=448`
- `max_text_length=192`
- `prompt_style=answer_only`

`answer_only` 会去掉 `Think step by step`，只保留 `The answer is (X)` 的输出格式约束。代码层面仍保留了 `reasoning` 与 `answer_only` 两种 prompt 风格，后续可以继续做对照实验。

注意：

- 当前仓库默认 `Unified VQA` 只有 `train/test` merged manifest。
- `Phase 3` 训练要求显式提供 `data.val_manifest`，因为 best checkpoint 依赖 `val accuracy`。
- 不建议直接拿 `test_manifest` 充当长期训练的验证集。

## 验证与 best checkpoint

Phase 3 同时记录：

- `val/loss`
- `val/accuracy`
- `val_accuracy/by_dataset/<dataset>`

默认策略：

- `save_every=1000`
- `eval_every=250`
- `eval_accuracy_every=250`
- `eval_accuracy_max_samples=512`
- `eval_max_new_tokens=16`

best checkpoint 只按 `val accuracy` 更新，文件名为：

- `phase3_best_accuracy.pt`

同时仍保留：

- 周期保存：`phase3_step_<step>.pt`
- 最终保存：`phase3_final.pt`

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

### 从 Phase 2b checkpoint 启动

```bash
cd /home/user/Project_files/project

accelerate launch --num_processes 8 train_phase3.py \
  --config configs/phase3_qwen3_dinov3.yaml \
  --phase2-checkpoint /path/to/phase2_vqa_best.pt
```

### 中断后续训

```bash
cd /home/user/Project_files/project

accelerate launch --num_processes 8 train_phase3.py \
  --config configs/phase3_qwen3_dinov3.yaml \
  --resume /path/to/phase3_step_3000.pt
```
