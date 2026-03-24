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
- `suites/unified_accuracy_suite.yaml`
  - simple unified evaluation template

## Recommended Validation Stack

For MicroVQA gold-standard checks, compare:

- `phase1_adapter_warmup`
- `phase2a_instruct`
- `phase2b_vqa`
- `phase3_final`

For Phase 3 ablations, the first set worth testing is:

- `phase3_no_backbone_unfreeze`
- `phase3_resize_448`
- `phase3_pad_preserve_448`
- `phase3_answer_only`
- `phase3_reasoning_prompt`
- `phase2b_best` vs `phase2b_final` initialization

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
