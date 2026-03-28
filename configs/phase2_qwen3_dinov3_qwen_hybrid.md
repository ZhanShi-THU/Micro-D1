# Phase 2 Qwen-Hybrid 配置说明

`phase2_qwen3_dinov3_qwen_hybrid.yaml` 是 Phase 2 的 Qwen3-inspired 几何预处理实验配置。

与基线配置相比，它的核心变化是：

- `image_preprocessing=qwen_hybrid`
- `dynamic_buckets=[384, 448, 512]`
- `patch_size=16`
- `dynamic_batch_padding=true`

这套配置不会引入 Qwen3 原生视觉塔，也不会切换到 Qwen 的图像归一化；它只借用：

- 宽高比保持
- patch 对齐
- 有界动态分辨率
- batch 内局部 padding

仍然保持：

- `DINOv3 backbone`
- `dinotxt alignment head`
- `adapter`
- `Qwen3-VL 4bit QLoRA`

推荐用途：

- 与 `configs/phase2_qwen3_dinov3.yaml` 做 Phase 2a / 2b 的成对对照实验
- 使用 MicroVQA gold detailed suite 比较：
  - `EU`
  - `HG`
  - `EP`
  - `microvqa_macro_accuracy`

建议的最小实验矩阵：

- `phase2a_pad_preserve`
- `phase2a_qwen_hybrid`
- `phase2b_pad_preserve`
- `phase2b_qwen_hybrid`
