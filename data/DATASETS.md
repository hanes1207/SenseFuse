# Data Setup

Data files are not tracked by Git. This document outlines the expected directory layout and the source of each component. 
To avoid duplicating existing datasets, all paths can be dynamically configured via environment variables defined in `utils/paths.py`.

> 🔥 **Core Requirement:** `features/` alone reproduces the Open-YOLO3D ScanNet200 accuracy (`python main.py`). Each other setting adds one directory: `openyolo3d/` for Replica, `openmask3d_features/` for OpenMask3D, `open3dis*/` and `scannetpp*/` for Open3DIS. Instance AP additionally needs each pipeline's own masks and the benchmark's ground truth, since it is scored by each benchmark's official evaluator.

As noted, `features/` contains embeddings generated in-house by running frozen encoders over published baseline masks. 
It is not a downloaded feature bank. The extraction scripts are provided in [`scripts/extract`](../scripts/extract/README.md) to ensure full reproducibility from raw scans.

---

## Expected Directory Layout

```text
./data
├── features/                       # Per-mask embeddings [built in-house; rebuild with scripts/extract]
│   ├── evaoy_s{0..3}.npz               # EVA02-E crop embeddings (Open-YOLO3D ScanNet200 masks)
│   ├── s2doy_s{0..3}.npz               # CLIP ViT-L-14-336 crop embeddings (Same masks)
│   ├── oyuni_s{0..1}.npz               # Uni3D shape embeddings (Same masks)
│   ├── yolovotes/                      # Open-YOLO3D's YOLO-World vote histograms
│   ├── o3drel_uni/                     # Uni3D embeddings over Open3DIS released masks
│   ├── replica/                        # Replica dump and 48-class text anchors
│   ├── yolovotes_replica/              # Replica vote histograms (8 scenes)
│   └── yolovotes_scannetpp/            # ScanNet++ vote histograms (50 scenes)
├── openyolo3d/                     # [[github.com/aminebdj/OpenYOLO3D](https://github.com/aminebdj/OpenYOLO3D)] checkout
│   ├── output/scannet200/scannet200_masks/     # Released class-agnostic masks
│   ├── output/replica/replica_masks/
│   ├── replica/ground_truth/
│   ├── evaluate/                               # its Replica evaluator, read here rather than copied
│   └── pretrained/config_{scannet200,replica}.yaml   # its YOLO-World prompt lists
├── openmask3d/                     # [[github.com/OpenMask3D/openmask3d](https://github.com/OpenMask3D/openmask3d)]
│   ├── masks/                                  # Class-agnostic proposals
│   └── instance_gt/validation/                 # ScanNet200 instance GT (<scene>.txt)
├── openmask3d_features/            # Re-run of OpenMask3D's feature extraction stage
│   └── features/, replica_features/
├── open3dis_released/              # [[github.com/VinAIResearch/Open3DIS](https://github.com/VinAIResearch/Open3DIS)]
│   └── Result_OpenVocab_ISBNet-GSAM/final_result_hier_agglo/
├── scannetpp/data/                 # [kaldir.vc.in.tum.de/scannetpp] (Signed TOU required)
├── scannetpp_open3dis_3d/          # Open3DIS preprocessed ScanNet++ release
│   ├── Scannetpp/Scannetpp_3D/val/{groundtruth,isbnet_clsagnostic_scannetpp}/
│   └── Scannetpp/Scannetpp_2D_5interval/val/   # iPhone frames (vote extraction only)
├── scannetpp_dumps/                # Intermediate dumps for ScanNet++ & Open3DIS rows
├── open3dis/                       # [[github.com/VinAIResearch/Open3DIS](https://github.com/VinAIResearch/Open3DIS)] repository checkout
│   └── data/replica/, exp/version_8scenes/     # Replica proposals
├── open3dis_cosine/                # Open3DIS ScanNet200 re-run (dumped before collapse)
│   └── {dense,denseraw,uni}/
├── open3dis_scannetpp_proposals/   # Hierarchical-agglomerative ScanNet++ proposals
├── uni3d/                          # [[github.com/baaivision/Uni3D](https://github.com/baaivision/Uni3D)] checkout: the code imports from it
│   └── models/                                 # model.pt (Uni3D-giant), open_clip_pytorch_model.bin (EVA02-E), point_encoder.py
├── scannet200/{train,val}/         # Preprocessed Mask3D-format meshes (<scene>.ply)
└── scannet_v2/scans/               # Raw ScanNet export (color, depth, poses, intrinsics)
```

## Data Sources & Environment Variables

You can relocate any component by setting its corresponding environment variable instead of modifying the code. Setting `SENSEFUSE_DATA` changes the root directory for the entire tree.

| Directory | Source | Environment Variable |
| :--- | :--- | :--- |
| `features/` | Generated in-house over published masks; rebuild with `scripts/extract`. | `SENSEFUSE_FEATS` |
| `openyolo3d/` | [Open-YOLO3D](https://github.com/aminebdj/OpenYOLO3D) checkout (masks via its `scripts/get_class_agn_masks.sh`). Its `evaluate/` and `pretrained/` are read in place, not copied into this repository. | `OPENYOLO3D_ROOT` |
| `openmask3d/` | [OpenMask3D](https://github.com/OpenMask3D/openmask3d) mask stage & ScanNet200 instance GT. | `OPENMASK3D_ROOT` |
| `openmask3d_features/` | Obtained by re-running OpenMask3D's feature stage. *(Note: Assumes equal color/depth resolutions. Using ScanNet's native color resolution scales SAM prompts by half).* | `SENSEFUSE_OM_REPRO` |
| `open3dis_released/` | [Open3DIS](https://github.com/VinAIResearch/Open3DIS) released ScanNet200 results. | `SENSEFUSE_O3D_DL` |
| `scannetpp/` | [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) official release (requires signed TOU). | `SENSEFUSE_SCANNETPP_EVAL` |
| `scannetpp_open3dis_3d/` | Open3DIS preprocessed ScanNet++ GT and 2D frames. | `SENSEFUSE_O3D` |
| `scannetpp_dumps/` | Stage-B dumps and Mask3D proposals for ScanNet++ evaluation. | `SENSEFUSE_SCANNETPP_DUMPS` |
| `open3dis/` | Open3DIS checkout for Replica proposals. | `OPEN3DIS_ROOT` |
| `open3dis_cosine/` | Open3DIS re-run on ScanNet200 (per-proposal features dumped pre-collapse). | `SENSEFUSE_O3D_COSINE` |
| `open3dis_scannetpp_proposals/`| ScanNet++ hierarchical-agglomerative proposals from Open3DIS re-run. | `SENSEFUSE_O3D_SPP_PROPOSALS` |
| `uni3d/` | [Uni3D](https://github.com/baaivision/Uni3D) checkout, added to `sys.path` for its point encoder: `models/model.pt` & EVA02-E `models/open_clip_pytorch_model.bin` (~12 GB). | `UNI3D_ROOT` |
| `scannet200/` | Preprocessed Mask3D-format meshes (Read *only* during feature extraction). | `SCANNET200_ROOT` |
| `scannet_v2/` | Raw [ScanNet](http://www.scan-net.org/) export (Read *only* during feature extraction). | `SCANNET_V2_ROOT` |

**System Temporary Directory:** Official evaluators generate report files that this repository does not read back.
The location for these scratch files is defined by `SENSEFUSE_TMPDIR`, which defaults to your system's temporary directory.