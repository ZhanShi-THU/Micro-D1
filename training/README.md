# Training Layout

This directory groups the stage-specific training entrypoints:

- `phase1_pretrain.py`
- `phase2.py`
- `phase3.py`
- `legacy_train.py`

Top-level compatibility wrappers are still kept:

- `train_pretrain.py`
- `train_phase2.py`
- `train_phase3.py`
- `train.py`

That means old commands keep working, while the project structure stays easier to navigate.
