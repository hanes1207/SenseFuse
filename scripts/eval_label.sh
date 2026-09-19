#!/usr/bin/env bash
# Labelling accuracy: Table I of the paper, over every setting it reports.
#
# Needs data/features, and for the rows named below the masks or ground truth of the pipeline in
# question; data/DATASETS.md states which. No GPU is used.
#
#   bash scripts/eval_label.sh                    every cell of Table I, beside its printed value
#   bash scripts/eval_label.sh --dataset replica  one dataset
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/tables/table1.py "$@"
