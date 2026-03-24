# Evaluation Layout

This directory keeps Phase 1/2/3 validation and ablation entrypoints together so results do not get mixed with training code.

## Main Entrypoints

- `cli.py`
  - the original single-run evaluator, still available through top-level `eval.py`
- `run_microvqa_suite.py`
  - packed suite runner for comparing multiple checkpoints on the same MicroVQA manifest
- `suites/microvqa_phase_progression.yaml`
  - template for phase progression and ablation experiments

## Recommended Validation Stack

For basic progression checks, compare:

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
  --suite-config evaluation/suites/microvqa_phase_progression.yaml
```
