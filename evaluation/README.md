# Evaluation Layout

This directory keeps Phase 1/2/3 validation and ablation entrypoints together so results do not get mixed with training code.

## Main Entrypoints

- `cli.py`
  - the original single-run evaluator, still available through top-level `eval.py`
- `run_microvqa_suite.py`
  - detailed MicroVQA gold-standard runner with EU / HG / EP accuracy and macro average
- `run_unified_accuracy_suite.py`
  - simplified unified runner that only reports overall accuracy
- `suites/microvqa_gold_detailed.yaml`
  - detailed MicroVQA evaluation template
- `suites/microvqa_phase2_preprocessing_gold.yaml`
  - Phase 2 preprocessing experiment template for detailed MicroVQA gold evaluation
- `suites/unified_accuracy_suite.yaml`
  - simple unified evaluation template
- `suites/unified_phase2_preprocessing_accuracy.yaml`
  - Phase 2 preprocessing experiment template for unified overall-accuracy checks

## Recommended Validation Stack

For MicroVQA gold-standard checks, compare:

- `phase1_adapter_warmup`
- `phase2a_instruct`
- `phase2b_vqa`
- `phase3_final`

For Phase 3 checks, the default path is now the reasoning-supervised configuration.
The first useful comparison is no longer prompt ablations inside the old answer-only setup, but:

- standard `phase2b` initialization vs stronger `phase2b` checkpoints
- `lora64` vs `lora128`
- `pad_preserve` vs future `qwen_hybrid` carry-over
- default reasoning Phase 3 vs future larger reasoning datasets

For Phase 2 preprocessing ablations, the recommended starting matrix is:

- `phase2a_pad_preserve`
- `phase2a_qwen_hybrid`
- `phase2b_pad_preserve`
- `phase2b_qwen_hybrid`

## Output Convention

Each suite writes into its own output root:

- `<output_root>/<run_name>/predictions.csv`
- `<output_root>/<run_name>/summary.json`
- `<output_root>/leaderboard.csv`
- `<output_root>/suite_summary.json`

That keeps phase progression, prompt ablations, and preprocessing ablations separated cleanly.

## Example

```bash
cd /home/user/Project_files/project

python evaluation/run_microvqa_suite.py \
  --suite-config evaluation/suites/microvqa_gold_detailed.yaml
```

```bash
cd /home/user/Project_files/project

python evaluation/run_unified_accuracy_suite.py \
  --suite-config evaluation/suites/unified_accuracy_suite.yaml
```
