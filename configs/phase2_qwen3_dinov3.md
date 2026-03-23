# Phase 2 配置说明

`phase2_qwen3_dinov3.yaml` 用于“深度语义融合”阶段，默认路径为：

- `stage=instruct`：先用 `LLaVA-Instruct-150K` 做语义融合热身
- `stage=vqa`：再用 `VQAv2` 做问答强化

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

`phase2.stages.instruct` 与 `phase2.stages.vqa` 分别维护两段 curriculum 的 manifest 和长度设置：

- `instruct`
  - 预期使用 `LLaVA-Instruct-150K`
  - 支持额外配置 `val_manifest` / `val_image_root`，用于训练时验证
  - 默认 `max_text_length=384`
- `vqa`
  - 预期使用 `VQAv2`
  - 也支持单独配置 `val_manifest`
  - 默认 `max_text_length=320`
  - 默认混入 `20%` 的 `LLaVA-Instruct-150K` 样本，避免后段纯 VQA 训练过度覆盖前段的开放式视觉语言能力

两段都复用项目已有的 caption/instruction 样式 manifest：

```json
{
  "image": "/abs/path/to/image.jpg",
  "text": "prompt text",
  "target_text": "supervised answer"
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

- `instruct` 阶段从 Phase 1 adapter checkpoint 初始化
- `vqa` 阶段从 `instruct` 阶段输出的 Phase 2 checkpoint 初始化
- 中断续训使用 `--resume`

## 常用命令

### Phase 2a: instruct 热身

```bash
cd /home/user/Project_files/project

accelerate launch --num_processes 8 train_phase2.py \
  --config configs/phase2_qwen3_dinov3.yaml \
  --stage instruct \
  --adapter-checkpoint /path/to/phase1_adapter.pt
```

### Phase 2b: VQAv2 强化

```bash
cd /home/user/Project_files/project

accelerate launch --num_processes 8 train_phase2.py \
  --config configs/phase2_qwen3_dinov3.yaml \
  --stage vqa \
  --phase2-checkpoint /path/to/phase2_instruct_final.pt
```

### 中断后续训

```bash
cd /home/user/Project_files/project

accelerate launch --num_processes 8 train_phase2.py \
  --config configs/phase2_qwen3_dinov3.yaml \
  --resume /path/to/phase2_vqa_step_3000.pt
```

## VQA 混合策略

`stage=vqa` 默认不再是纯 `VQAv2`，而是：

- 主数据集：`VQAv2`
- 辅助数据集：`LLaVA-Instruct-150K`
- 默认辅助占比：`0.2`

也就是大约 `80% VQAv2 + 20% instruct` 的训练逻辑。实现上会保证每个逻辑 epoch 都完整覆盖主数据集，再按比例补充辅助数据集样本。

如果你想改这个比例，直接修改 `phase2.stages.vqa.auxiliary_fraction` 即可。
