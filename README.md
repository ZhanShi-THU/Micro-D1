# Modular VLM For Unified Microscopy VQA

This project trains and evaluates a modular vision-language model for microscopy visual question answering on a unified multiple-choice benchmark.

The current codebase is built around one canonical dataset bundle:

- `/data1/staging_datasets/unified_vqa`

and one main configuration:

- [`configs/qwen3_dinov3.yaml`](/home/user/Project_files/project/configs/qwen3_dinov3.yaml)

## Overview

The model is intentionally modular:

- A DINOv3 vision backbone extracts patch tokens.
- A DINOText-style alignment head maps visual features into an aligned token space.
- A lightweight trainable adapter projects aligned visual tokens into the Qwen embedding space.
- The Qwen3-VL text backbone consumes visual tokens as a prefix sequence followed by text tokens.
- The same adapter output is also shared into the first several Qwen text layers through DeepStack-style residual injection.

The project does not use the official Qwen3-VL vision tower in the modular path. It uses only the Qwen3-VL text embeddings, decoder stack, and language head.

For a shape-by-shape walkthrough of the full image/text pipeline, see [`ARCHITECTURE_DATAFLOW.md`](/home/user/Project_files/project/ARCHITECTURE_DATAFLOW.md).

The training target is multiple-choice VQA in the format:

```text
The answer is (X)
```

where `X` is a zero-based choice index.

## Current Dataset Contract

The project consumes JSONL manifests whose records include:

- `sample_id`
- `source_dataset`
- `split`
- `image_path`
- `question`
- `choices`
- `correct_index`
- `correct_answer`
- `target_text`
- `metadata`

The canonical local bundle lives at:

- [`/data1/staging_datasets/unified_vqa`](/data1/staging_datasets/unified_vqa)

The current local split state is:

- `microbench`: `95% train / 5% test`, grouped by image to avoid leakage
- `mmsci++`: train only
- `microvqa`: test only
- `mms`: test only

The default config points to:

- [`/data1/staging_datasets/unified_vqa/manifests/merged/train.jsonl`](/data1/staging_datasets/unified_vqa/manifests/merged/train.jsonl)
- [`/data1/staging_datasets/unified_vqa/manifests/merged/test.jsonl`](/data1/staging_datasets/unified_vqa/manifests/merged/test.jsonl)

## Repository Layout

- [`train.py`](/home/user/Project_files/project/train.py)
  Main training entrypoint.
- [`eval.py`](/home/user/Project_files/project/eval.py)
  Main evaluation entrypoint. Includes a unified `mcq` evaluator for the final benchmark.
- [`data/dataset.py`](/home/user/Project_files/project/data/dataset.py)
  Dataset loader used by training and evaluation.
- [`data/unified_vqa.py`](/home/user/Project_files/project/data/unified_vqa.py)
  Shared utilities for unified manifests, canonical paths, and deterministic dataset re-splitting.
- [`models/modular_vlm.py`](/home/user/Project_files/project/models/modular_vlm.py)
  The multimodal model wrapper that combines vision encoder, adapter, and Qwen decoder.
- [`models/vision_encoder.py`](/home/user/Project_files/project/models/vision_encoder.py)
  DINOv3 feature extraction plus DINOText-style alignment head loading.
- [`models/adapter.py`](/home/user/Project_files/project/models/adapter.py)
  Trainable visual adapter into Qwen embedding space.
- [`scripts/prepare_unified_vqa.py`](/home/user/Project_files/project/scripts/prepare_unified_vqa.py)
  Builds the self-contained unified dataset bundle from source datasets.
- [`scripts/finalize_unified_vqa.py`](/home/user/Project_files/project/scripts/finalize_unified_vqa.py)
  Rebuilds merged manifests and validates paths.
- [`scripts/rehome_unified_vqa.py`](/home/user/Project_files/project/scripts/rehome_unified_vqa.py)
  Migrates an older bundle into the canonical self-contained layout.
- [`scripts/split_unified_vqa.py`](/home/user/Project_files/project/scripts/split_unified_vqa.py)
  Applies generic deterministic train/test re-splitting to one or more datasets already present in the unified bundle.
- [`scripts/deploy_local_models.py`](/home/user/Project_files/project/scripts/deploy_local_models.py)
  Prepares local model directories for DINOv3 and Qwen3-VL.
- [`data/README_unified_vqa.md`](/home/user/Project_files/project/data/README_unified_vqa.md)
  Dataset-pipeline-specific notes.

## Environment

The recommended environment on this machine is the existing Conda environment:

- `microvqa`

Most project commands should be run as:

```bash
conda run -n microvqa python ...
```

Minimal Python dependencies are listed in [`requirements.txt`](/home/user/Project_files/project/requirements.txt):

- `torch`
- `torchvision`
- `transformers>=4.50.0`
- `huggingface_hub>=0.36.0`
- `datasets`
- `PyYAML`
- `Pillow`

## Local Model Dependencies

The default config expects:

- Qwen3-VL at [`local_models/qwen3-vl-8b`](/home/user/Project_files/project/local_models/qwen3-vl-8b)
- DINOv3 backbone checkpoint at [`pth/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`](/home/user/Project_files/project/pth/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth)
- DINOText alignment weights at [`pth/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth`](/home/user/Project_files/project/pth/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth)

To prepare local model paths:

```bash
cd /home/user/Project_files/project
conda run -n microvqa python scripts/deploy_local_models.py
```

By default this script:

- symlinks DINOv3 from `/home/user/Project_files/microvqa/models/dinov3_vit`
- downloads Qwen3-VL-8B-Instruct into `local_models/qwen3-vl-8b`

## Data Preparation

### 1. Build the unified bundle

```bash
cd /home/user/Project_files/project
conda run -n microvqa python scripts/prepare_unified_vqa.py \
  --output-root /data1/staging_datasets/unified_vqa
```

This writes:

- `manifests/by_dataset/*.jsonl`
- `manifests/by_split/*.jsonl`
- `manifests/merged/train.jsonl`
- `manifests/merged/test.jsonl`
- `manifests/merged/all.jsonl`
- `manifests/summary.json`
- exported images under `images/...`
- copied source annotations under `sources/...`

### 2. Prepare and split in one pass

`prepare_unified_vqa.py` accepts a generic re-splitting interface:

```bash
cd /home/user/Project_files/project
conda run -n microvqa python scripts/prepare_unified_vqa.py \
  --output-root /data1/staging_datasets/unified_vqa \
  --dataset-train-ratio microbench=0.95 \
  --seed 42
```

The interface is:

```text
--dataset-train-ratio DATASET=RATIO
```

Examples:

- `--dataset-train-ratio microbench=0.95`
- `--dataset-train-ratio microvqa=0.80`

When image-level identifiers are available, splitting is done by image-group rather than by record, so multiple questions from the same image stay on the same side of the split.

### 3. Re-split an existing unified bundle

If the bundle already exists and you only want to change splits:

```bash
cd /home/user/Project_files/project
conda run -n microvqa python scripts/split_unified_vqa.py \
  --output-root /data1/staging_datasets/unified_vqa \
  --dataset-train-ratio microbench=0.95 \
  --seed 42
```

### 4. Rebuild indexes and validate paths

```bash
cd /home/user/Project_files/project
conda run -n microvqa python scripts/finalize_unified_vqa.py \
  --output-root /data1/staging_datasets/unified_vqa
```

### 5. Rehome an older bundle

If you have an older bundle whose manifests still point to `/data1/mms_data` or `/data1/mmsci++`:

```bash
cd /home/user/Project_files/project
conda run -n microvqa python scripts/rehome_unified_vqa.py \
  --output-root /data1/staging_datasets/unified_vqa
```

## Training

The default training config is already wired to the canonical unified bundle.

Run:

```bash
cd /home/user/Project_files/project
conda run -n microvqa python train.py --config configs/qwen3_dinov3.yaml
```

The current training loop supports two phases:

- `phase1`
  Adapter-only warmup.
- `phase2`
  Adapter plus optional tuning of the last few LLM layers.

You can override the active phase:

```bash
cd /home/user/Project_files/project
conda run -n microvqa python train.py \
  --config configs/qwen3_dinov3.yaml \
  --phase phase1
```

Training outputs are written under the configured `training.output_dir`, currently:

- `./outputs/qwen3_dinov3`

Saved checkpoints include:

- adapter weights
- optimizer state
- scheduler state
- optionally trainable LLM layer state for phase 2

## Evaluation

### Unified multiple-choice evaluation

This is the main evaluation path for the current final dataset version:

```bash
cd /home/user/Project_files/project
conda run -n microvqa python eval.py --config configs/qwen3_dinov3.yaml mcq
```

`eval.py` uses global options before subcommands, so `--config` and `--device` must appear before `mcq`, `finegrained`, or `efficiency`.

This evaluator reports:

- overall accuracy
- per-dataset accuracy
- macro accuracy across datasets

Optional arguments:

- `--manifest`
  Override the test manifest
- `--adapter-checkpoint`
  Evaluate a trained adapter checkpoint
- `--baseline-model-path`
  Override the baseline Qwen model path
- `--limit`
  Evaluate only the first N samples

### Fine-grained multiple-choice evaluation

This mode keeps the older bucketed analysis:

```bash
cd /home/user/Project_files/project
conda run -n microvqa python eval.py --config configs/qwen3_dinov3.yaml finegrained
```

### Efficiency evaluation

This mode compares checkpoint progress against a caption-style target metric:

```bash
cd /home/user/Project_files/project
conda run -n microvqa python eval.py \
  --config configs/qwen3_dinov3.yaml \
  efficiency \
  --manifest /path/to/caption_manifest.jsonl \
  --checkpoint-dir /path/to/checkpoints
```

Use `efficiency` only when you actually have a caption-style evaluation manifest. The current unified microscopy VQA benchmark is primarily intended for the `mcq` evaluator.

## Tests

Lightweight checks included in this repository:

- [`tests/test_unified_vqa_prepare.py`](/home/user/Project_files/project/tests/test_unified_vqa_prepare.py)
- [`tests/test_dinotxt_head_smoke.py`](/home/user/Project_files/project/tests/test_dinotxt_head_smoke.py)
- [`tests/test_qwen3_smoke.py`](/home/user/Project_files/project/tests/test_qwen3_smoke.py)
- [`tests/test_vision_encoder_smoke.py`](/home/user/Project_files/project/tests/test_vision_encoder_smoke.py)

Example:

```bash
cd /home/user/Project_files/project
conda run -n microvqa python -m unittest tests.test_unified_vqa_prepare
```

## Notes And Constraints

- `correct_index` is zero-based throughout the project.
- The model target format is always `The answer is (X)`.
- The data loader reads absolute `image_path` values directly from the manifest by default.
- For datasets with multiple questions per image, grouped splitting is preferred to avoid train/test leakage.
- `microbench` dominates the benchmark by size, so for serious reporting you should inspect both overall accuracy and per-dataset accuracy.

## Related Documentation

- [`data/README_unified_vqa.md`](/home/user/Project_files/project/data/README_unified_vqa.md)
- [`/data1/staging_datasets/unified_vqa/README_local_layout.md`](/data1/staging_datasets/unified_vqa/README_local_layout.md)
