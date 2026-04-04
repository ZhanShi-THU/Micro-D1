#!/usr/bin/env bash
set -euo pipefail

# Auto-generated Phase 3 component ablation commands.

# Full Model (Micro-D1)
# Reusing existing full-model checkpoint:
# /home/user/Project_files/project/outputs/phase3_qwen3_dinov3_qwen_hybrid_lora128/qwen3_dinov3_phase3_reasoning_qwen_hybrid_lora128_conservative_20260402_123905/phase3_step_200.pt

# w/o DeepStack
# accelerate launch --num_processes 2 train_phase3.py --config configs/ablations/phase3_component_ablation/phase3_qwen3_dinov3_qwen_hybrid_lora128_no_deepstack.yaml --phase2-checkpoint /home/user/Project_files/project/outputs/phase2_qwen3_dinov3_qwen_hybrid_lora128/qwen3_dinov3_phase2_mixed_qwen_hybrid_lora128_mixed_20260401_220905/phase2_mixed_best.pt

# w/o alignment head (dino-txt)
accelerate launch --num_processes 2 train_phase3.py --config configs/ablations/phase3_component_ablation/phase3_qwen3_dinov3_qwen_hybrid_lora128_no_alignment.yaml --phase2-checkpoint /home/user/Project_files/project/outputs/phase2_qwen3_dinov3_qwen_hybrid_lora128/qwen3_dinov3_phase2_mixed_qwen_hybrid_lora128_mixed_20260401_220905/phase2_mixed_best.pt

# w/o adapter (linear)
accelerate launch --num_processes 2 train_phase3.py --config configs/ablations/phase3_component_ablation/phase3_qwen3_dinov3_qwen_hybrid_lora128_linear_adapter.yaml --phase2-checkpoint /home/user/Project_files/project/outputs/phase2_qwen3_dinov3_qwen_hybrid_lora128/qwen3_dinov3_phase2_mixed_qwen_hybrid_lora128_mixed_20260401_220905/phase2_mixed_best.pt

# Evaluate all ablation runs
python3 evaluation/run_microvqa_suite.py --suite-config evaluation/suites/phase3_component_ablation.yaml --cache-mode off
