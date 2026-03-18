# Unified VQA Data Pipeline

This project now supports a unified multiple-choice manifest for:

- `mms`
- `mmsci++`
- `microvqa`
- `microbench`

The preparation entrypoint is:

```bash
cd /home/user/Project_files/project
conda run -n microvqa python scripts/prepare_unified_vqa.py --output-root /data1/staging_datasets/unified_vqa
```

`prepare_unified_vqa.py` now accepts the same generic re-splitting interface used by `split_unified_vqa.py`, so you can prepare and split in one pass:

```bash
cd /home/user/Project_files/project
conda run -n microvqa python scripts/prepare_unified_vqa.py \
  --output-root /data1/staging_datasets/unified_vqa \
  --dataset-train-ratio microbench=0.95 \
  --seed 42
```

The script writes:

- `manifests/by_dataset/<dataset>.jsonl`
- `manifests/by_split/<dataset>_<split>.jsonl`
- `manifests/merged/<split>.jsonl`
- `manifests/merged/all.jsonl`
- exported images for datasets that store image bytes instead of file paths
- copied `mms` / `mmsci++` images and source annotations into `images/...` and `sources/...` so the bundle is self-contained

Unified manifest fields:

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

The existing `ImageTextDataset` can read these manifests directly because they keep the `question`, `choices`, `correct_index`, and `image_path` keys.

For an already-built legacy bundle whose manifests still point to `/data1/mms_data` or `/data1/mmsci++`, run:

```bash
cd /home/user/Project_files/project
conda run -n microvqa python scripts/rehome_unified_vqa.py --output-root /data1/staging_datasets/unified_vqa
```

To re-split any dataset already present in the unified bundle, use the generic split interface:

```bash
cd /home/user/Project_files/project
conda run -n microvqa python scripts/split_unified_vqa.py \
  --output-root /data1/staging_datasets/unified_vqa \
  --dataset-train-ratio microbench=0.95 \
  --seed 42
```

The interface is generic across datasets. Repeat `--dataset-train-ratio DATASET=RATIO` to re-split multiple datasets in one run. When image-level identifiers are available, records are grouped by image to avoid leaking different questions from the same image across train and test.
