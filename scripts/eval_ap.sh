#!/usr/bin/env bash
# Instance-segmentation AP: Table II of the paper.
#
# Every cell is scored under two submission rules, top-1 and flatten-600, at a uniform confidence of
# 1.0, and at the weight Algorithm 1 estimates for that setting -- the same weight Table I prints.
# Requires data/features together with that pipeline's masks and ground truth; see data/DATASETS.md.
# Roughly thirty minutes per ScanNet200 block over 312 scenes on one CPU.
#
#   bash scripts/eval_ap.sh                       every block
#   bash scripts/eval_ap.sh --list                what the blocks are
#   bash scripts/eval_ap.sh --block o3d_replica   one of them
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/tables/table2.py "$@"
