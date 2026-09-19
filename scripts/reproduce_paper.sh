#!/usr/bin/env bash
# Every quantitative result the paper reports, recomputed and checked against the printed value.
#
#   bash scripts/reproduce_paper.sh
#
# What this reproduces is the paper's numbers, not its typesetting: the LaTeX source is not part of
# this repository, and `SenseFuse.pdf` is the published article. Each step recomputes a table from
# `data/features` and the pipeline's own inputs, prints every cell beside the digit the paper
# carries, and reports the largest difference over the table. Steps 2 to 5 additionally hold that
# difference against the value this repository measures and exit non-zero if it has moved, so a
# silent regression there cannot pass as a successful run.
#
# Steps 1 to 5 take minutes. Tables III and V and Table I's Open-YOLO3D ScanNet200 rows need only
# data/features; Table I's other rows and Table IV also read the masks, ground truth or dumps of the
# pipeline in question, as data/DATASETS.md sets out. Step 6 is Table II, which is instance AP: it
# scores every prediction against each benchmark's own ground truth, needs that benchmark's masks,
# and takes some hours. Steps 7 and 8 print the Section IV-D ablations and the quantities behind
# Figures 3, 5 and 6 beside the claims the text makes about them.
set -euo pipefail
cd "$(dirname "$0")/.."

LOGS="$(mktemp -d)"
trap 'rm -rf "$LOGS"' EXIT
SUMMARY=()
FAILED=0

# Recompute one table, then hold its largest difference against what this repository has measured.
check () {
    local step="$1" allow="$2" label="$3" pattern="$4"; shift 4
    echo
    echo "== ${step}/8  ${label}"
    "$@" 2>&1 | tee "${LOGS}/${step}.log"
    local got
    got="$(grep -oE "${pattern}[0-9.]+" "${LOGS}/${step}.log" | grep -oE '[0-9.]+$' | tail -1)"
    if [ -z "${got}" ]; then
        SUMMARY+=("FAIL  ${label}: it printed no summary line")
        FAILED=1
    elif awk -v g="${got}" -v a="${allow}" 'BEGIN{exit !(g <= a + 1e-9)}'; then
        SUMMARY+=("ok    ${label}: largest difference ${got}, allowed ${allow}")
    else
        SUMMARY+=("FAIL  ${label}: largest difference ${got}, allowed ${allow}")
        FAILED=1
    fi
}

echo "== 1/8  gate: is the first row of Table I still 48.44 -> 52.97 at w = 0.3522?"
python main.py --gate                      # asserts, so a mismatch stops the run here
SUMMARY+=("ok    gate: Table I's first row")

check 2 0.01  "Table I: labelling accuracy, eleven settings, every column" \
      "printed precision: " python scripts/tables/table1.py
check 3 0.001 "Table III: pair complementarity over four frozen encoders" \
      "printed value: " python scripts/tables/table3.py
check 4 0.01  "Table IV: the cost of correcting for cross-modal correlation" \
      "printed value: " python scripts/tables/table4.py
check 5 0.05  "Table V: the fusion rule against its alternatives" \
      "printed value: " python scripts/tables/table5.py

echo
echo "== 6/8  Table II: instance-segmentation AP, eleven blocks"
python scripts/tables/table2.py

echo
echo "== 7/8  Section IV-D: the estimator ablations, each beside the value the text states"
python scripts/tables/ablations.py --all

echo
echo "== 8/8  Figures 3, 5 and 6: the quantities the figures are drawn from"
python scripts/tables/figures.py --all

echo
echo "===================== summary ====================="
printf '%s\n' "${SUMMARY[@]}"
if [ "${FAILED}" -ne 0 ]; then
    echo "At least one table no longer reproduces. The lines marked FAIL say which."
    exit 1
fi
echo "Every check above passed."
