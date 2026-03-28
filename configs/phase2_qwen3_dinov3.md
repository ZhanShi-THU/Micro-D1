# Phase 2 配置说明

`phase2_qwen3_dinov3.yaml` 现在默认使用单阶段 mixed 训练路径，不再把 Phase 2 强制拆成 `instruct -> vqa` 两段：

- `LLaVA-Instruct-150K`
- `VQAv2`
- `ScienceQA`

三者会在同一个逻辑 epoch 内按权重混合采样。

## 模型策略

- 冻结 `DINOv3 backbone`
- 解冻 `dinotxt alignment head`
- 继续训练 `adapter`
- `qwen3-vl-8b` 采用 `4bit QLoRA`
- `DeepStack injection` 保持开启

默认 LoRA 设置：

- `r=64`
- `lora_alpha=128`
- `lora_dropout=0.05`
- `target_modules=[q_proj, k_proj, v_proj, o_proj]`

## 数据配置

`phase2.datasets` 中为每个数据源单独配置：

- `name`
- `train_manifest`
- `val_manifest`
- `sampling_weight`
- 可选 `image_root` / `val_image_root`

当前默认混合比例是：

- `llava_instruct: 0.35`
- `vqav2: 0.55`
- `scienceqa: 0.10`

并通过 `phase2.mixed_samples_per_epoch` 控制每个逻辑 epoch 的总采样量。默认值设为 `560000`，大致对齐旧两阶段训练合计看到的样本规模，同时给 `ScienceQA` 一定的重复曝光，但不让它主导整个 Phase 2。

`image_preprocessing` 目前支持三种模式：

- `pad_preserve`
  - 先按比例缩放，让长边贴齐目标尺寸
  - 再对短边居中 padding 成正方形
  - 更适合显微图像或长宽比差异较大的样本
- `resize`
  - 直接拉伸到 `(image_size, image_size)`
  - 与旧版本 Phase 2 行为一致
- `qwen_hybrid`
  - 保持宽高比
  - 将长边约束到有限分桶，如 `[384, 448, 512]`
  - 将高宽对齐到 `patch_size=16`
  - 在 collate 阶段按 batch 内最大高宽做 padding
  - 适合做 Qwen3-inspired 几何预处理实验，同时保持 DINO 侧可控的 token 波动

`LLaVA` 和 `VQAv2` 继续复用项目已有的 caption/instruction 样式 manifest：

```json
{
  "image": "/abs/path/to/image.jpg",
  "text": "prompt text",
  "target_text": "supervised answer"
}
```

`ScienceQA` 使用 microvqa-style manifest，包含：

```json
{
  "image": "/abs/path/to/image.png",
  "question": "question text\nContext: optional hint",
  "choices": ["A", "B", "C", "D"],
  "correct_index": 0,
  "target_text": "The answer is (0)"
}
```

## 检查点约定

Phase 2 checkpoint 会保存：

- `adapter`
- `vision_alignment_head`
- `lora_state`
- `optimizer`
- `scheduler`
- `global_step`
- `optimizer_step`
- `stage`

默认保存策略说明：

- `save_every` 按固定步数做一次周期保存
- `best.pt` 不再从训练一开始就频繁覆盖，只有当 `loss <= training.best_checkpoint_min_loss` 后，才会按 best loss 更新

当前默认策略更偏向减少写盘开销：

- `save_every=2000`
- `save_steps=[]`
- `save_loss_thresholds=[]`
- `best_checkpoint_min_loss=2.5`

验证相关的默认配置：

- `eval_every=250`
- `eval_max_batches=64`
- `eval_batch_size=1`

只要当前 stage 配置了 `val_manifest`，训练过程中就会周期性计算 `val/loss`，并写入 wandb 与 `train_log.jsonl`。这样你可以直接观察 `train/loss` 与 `val/loss` 的背离，判断是否开始过拟合。

如果你想恢复旧行为，可以把 `best_checkpoint_min_loss` 设成 `null`，或者重新填写 `save_steps` / `save_loss_thresholds`。

使用方式：

- 默认 `mixed` 阶段直接从 Phase 1 adapter checkpoint 初始化
- 不再要求先训练 `phase2a` 再训练 `phase2b`
- 中断续训仍然使用 `--resume`

## 常用命令

### Phase 2: mixed 单阶段训练

```bash
cd /home/user/Project_files/project

accelerate launch --num_processes 8 train_phase2.py \
  --config configs/phase2_qwen3_dinov3.yaml \
  --adapter-checkpoint /path/to/phase1_adapter.pt
```

### 中断后续训

```bash
cd /home/user/Project_files/project

accelerate launch --num_processes 8 train_phase2.py \
  --config configs/phase2_qwen3_dinov3.yaml \
  --resume /path/to/phase2_mixed_step_3000.pt
```
