#!/usr/bin/env bash
set -euo pipefail

python3 evaluation/run_microvqa_suite.py \
  --suite-config evaluation/suites/phase3_no_deepstack_quick.yaml \
  --cache-mode off
