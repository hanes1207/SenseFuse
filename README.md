# SenseFuse: Label-Free Fusion of Image and Shape Encoders for Open-Vocabulary 3D Instance Segmentation

We present **SenseFuse**, a label-free fusion framework for open-vocabulary 3D instance segmentation that enhances mask labeling accuracy by combining 2D image and 3D shape encoders.
Based on our empirical observation that 2D image and 3D shape encoders exhibit largely disjoint failure patterns and rarely share identical wrong labels, SenseFuse fuses their score matrices using a single scene-level scalar weight. 
This weight is optimized directly in milliseconds by maximizing a label-free sensitivity metric—requiring **no ground-truth labels, no parameter training, and no fine-tuning**.
On ScanNet200, SenseFuse recovers **99%** of the accuracy gain achievable with an oracle weight (and a median of **93%** across ScanNet200, Replica, and ScanNet++).

Paper: [arXiv:2609.20475](https://arxiv.org/abs/2609.20475) · [`SenseFuse.pdf`](SenseFuse.pdf) (8 pages).

```python
from main import accuracy
r = accuracy("openyolo3d", "scannet200", head="eva")
r["acc_2d"], r["w"], r["acc_fused"]        # 48.44   0.3522   52.97
```

## Installation

```sh
git clone https://github.com/hanes1207/SenseFuse.git
cd SenseFuse

conda create -n sensefuse python=3.11
conda activate sensefuse
conda install pytorch==2.5.1 torchvision pytorch-cuda=12.4 -c pytorch -c nvidia

pip install -r requirements.txt

# a CPU build of PyTorch is enough to reproduce every table; CUDA is needed only to rebuild
# data/features with scripts/extract
```

## Data Preparation

All required datasets and inputs should be placed within the `data/` directory. 
For a comprehensive overview of the expected directory structure and external data sources, refer to **[`data/DATASETS.md`](data/DATASETS.md)**. 

To facilitate integration with existing storage structures, every path can be overridden using environment variables, allowing you to utilize existing local copies without duplicating data.

### Pre-packaged Assets & Features

*   **`features/` (~900 MB)**: All feature files were generated in-house using frozen encoders on published mask proposals (reproduction scripts available in [`scripts/extract`](scripts/extract/README.md)). 
The entire feature archive can be rebuilt from scratch or extended to custom datasets.
*   **`assets/` (~2.4 MB)**: Pre-packaged assets include text embedding anchors, ground-truth-to-proposal matchings, and ScanNet++ class mapping tables. 
All benchmark metrics reported in the paper can be recomputed directly without re-running any feature extraction pipelines.

### Data Sources & Environment Variables

| Directory | Description | Source | Environment Variable |
| :--- | :--- | :--- | :--- |
| `features/` | **Per-mask embeddings**: Text anchors, EVA02-E/CLIP-L crop embeddings, and Uni3D shape embeddings per proposal. | Generated via [`scripts/extract`](scripts/extract/README.md) | `SENSEFUSE_FEATS` |
| `openyolo3d/` | The Open-YOLO3D checkout: its released ScanNet200 and Replica masks, its Replica evaluator, and its two prompt configurations. | [Open-YOLO3D](https://github.com/aminebdj/OpenYOLO3D) | `OPENYOLO3D_ROOT` |
| `openmask3d/` | Class-agnostic proposals and ScanNet200 instance GT. Includes `{masks, instance_gt}`. | [OpenMask3D](https://github.com/OpenMask3D/openmask3d) | `OPENMASK3D_ROOT` |
| `openmask3d_features/` | Per-scene CLIP features extracted from OpenMask3D (2D head). | Obtained by re-running OpenMask3D | `SENSEFUSE_OM_REPRO` |
| `open3dis_released/` | Released ScanNet200 results from Open3DIS. | [Open3DIS](https://github.com/VinAIResearch/Open3DIS) | `SENSEFUSE_O3D_DL` |
| `scannetpp*/` | ScanNet++ ground truth and stage-B scoring dumps. | [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) (signed TOU) + Open3DIS | `SENSEFUSE_O3D`, `SENSEFUSE_SCANNETPP_EVAL`, `SENSEFUSE_SCANNETPP_DUMPS` |
| `open3dis*/` | Open3DIS proposals and pre-collapse per-proposal scores. | Obtained by re-running [Open3DIS](https://github.com/VinAIResearch/Open3DIS) | `OPEN3DIS_ROOT`, `SENSEFUSE_O3D_COSINE`, `SENSEFUSE_O3D_SPP_PROPOSALS` |
| `uni3d/` | Uni3D-giant `model.pt` and EVA02-E `open_clip_pytorch_model.bin`. | [Uni3D](https://github.com/baaivision/Uni3D) | `UNI3D_ROOT` |

### Evaluation Requirements by Table

While `features/` and `assets/` are sufficient for core metrics, reproducing specific baseline comparisons from the paper requires additional dependencies.

| Target Evaluation (Paper) | Additional Data Requirements (Beyond `features/` & `assets/`) |
| :--- | :--- |
| Tables III, V, and Table I (Open-YOLO3D ScanNet200) | *None* |
| Tables I and IV (OpenMask3D) | `openmask3d_features/` (Requires re-running OpenMask3D) |
| Table I (Replica) | the Open-YOLO3D checkout: its Replica masks and ground truth, and its evaluator |
| Table I (ScanNet++) | Stage-B dumps under `scannetpp_dumps/` |
| Tables I and IV (Open3DIS) | Pipeline dumps under `open3dis*/` |
| Table II (Instance AP) | Respective pipeline masks and benchmark ground truth (AP is evaluated at the scene level, not via stored matchings), plus each row's own requirement above &mdash; the OpenMask3D rows still need `openmask3d_features/` |

## Evaluation

Ensure all datasets are properly formatted and accessible as detailed in [`data/DATASETS.md`](data/DATASETS.md) prior to running evaluations.

You can reproduce all benchmark tables and figures simultaneously using the main shell script, or execute them individually:

```sh
# Reproduce all tables
bash scripts/reproduce_paper.sh

# Reproduce individual components
python main.py --gate                      # Table I (first row, executes in seconds)
python scripts/tables/table1.py            # Table I
bash scripts/eval_ap.sh                    # Table II (Note: takes several hours; --list shows the blocks, --block picks one)
python scripts/tables/table3.py            # Table III
python scripts/tables/table4.py            # Table IV
python scripts/tables/table5.py            # Table V
python scripts/tables/ablations.py --all   # Section IV-D (Ablation Studies)
python scripts/tables/figures.py --all     # Figures 3, 5, and 6
```

## Repository Structure

*   **`main.py`**: Main entry point for computing accuracy (`accuracy()`) and instance AP (`--task ap`).
*   **`scripts/`**: Scripts for reproducing paper results and extracting features.
    *   `reproduce_paper.sh`: Master script to reproduce all tables and ablations.
    *   `eval_label.sh` / `eval_ap.sh`: Evaluation scripts for Table I (label accuracy) and Table II (instance AP).
    *   `tables/`: Individual generator scripts for each table and Section IV-D ablations.
    *   `extract/`: Feature extraction pipelines (one per encoder) to generate `data/features`.
*   **`models/`**: Core SenseFuse algorithms and fusion logic.
    *   `law.py`: The closed-form weight calibration `w = pi_3 / (pi_2 + pi_3)` on centered score rows.
    *   `estimator.py`: Implementation of **Algorithm 1** (label-free weight estimation from soft responsibilities).
    *   `rules.py`: Alternative fusion rules evaluated in Table V.
*   **`utils/`**: Data loaders, evaluation metrics, and configuration modules.
    *   `paths.py` / `settings.py`: Centralized path management and interfaces for the 11 experimental settings.
    *   `scannet200.py` / `replica.py` / `open3dis.py`: Dataset-specific loaders and score matrix handlers.
    *   `ap.py` / `ap_scannetpp.py`: Instance AP evaluation logic matching official submission rules.
    *   `benchmark/`: Official evaluators from three of the baseline projects, carried as released under their own licences (see [`LICENSE`](LICENSE)); Open-YOLO3D's is read from its own checkout.
*   **`assets/`**: Pre-packaged text anchors and ground-truth matchings (~2.4 MB).
*   **`data/DATASETS.md`**: Documentation detailing the expected data layout and external sources.

## Acknowledgements

We sincerely thank the authors of [Uni3D](https://github.com/baaivision/Uni3D), [Open-YOLO3D](https://github.com/aminebdj/OpenYOLO3D), [OpenMask3D](https://github.com/OpenMask3D/openmask3d), [Open3DIS](https://github.com/VinAIResearch/Open3DIS), and [EVA-CLIP](https://github.com/baaivision/EVA) for open-sourcing their code, pre-trained models, and evaluation benchmarks.

## Citation

```bibtex
@article{han2026sensefuse,
  title   = {SenseFuse: Label-Free Fusion of Image and Shape Encoders
             for Open-Vocabulary 3D Instance Segmentation},
  author  = {Han, Euiseok and Ton, Tri and Kim, Hwanhee and
             Ryu, Seungyeon and Yoo, Chang D.},
  journal = {arXiv preprint arXiv:2609.20475},
  year    = {2026}
}
```
