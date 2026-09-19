"""Text anchors for the encoders whose anchors are not committed
-> ``features/bigg_text_200.npy`` and ``features/replica/bigg_text_48.npy``.

Both heads are scored as cosines against class-name embeddings, so an anchor set belongs to the
tower that produced it. EVA02-E's and CLIP ViT-L's anchors are a few megabytes and are committed
under ``assets/fusion``; OpenShape's teacher is OpenCLIP ViT-bigG, whose anchors are written here
instead, because only Table III reads them.

The ensemble is the twenty templates used throughout: each template is embedded, normalised, and the
mean renormalised, which is the protocol Uni3D evaluates under.

    python scripts/extract/text_anchors.py --dataset scannet200
    python scripts/extract/text_anchors.py --dataset replica
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from _lib import ply  # noqa: F401  (the package import also sets sys.path)
from utils.paths import FEATS

TEMPLATES = ["{}", "a {}", "the {}", "a {} in a scene", "there is a {} in the room",
             "a photo of a {}", "a photo of the {}", "a photo of one {}",
             "a close-up photo of a {}", "a cropped photo of a {}", "a bright photo of a {}",
             "a dark photo of a {}", "a blurry photo of a {}", "a photo of a small {}",
             "a photo of a large {}", "a {} in a living room", "a {} in a bedroom",
             "a {} in an office", "an indoor photo of a {}", "a 3d render of a {}"]


def class_names(dataset):
    if dataset == "scannet200":
        import scannet200_constants as C200
        return list(C200.CLASS_LABELS_200), f"{FEATS}/bigg_text_200.npy"
    from evaluate.replica.eval_semantic_instance import CLASS_LABELS
    return ([n.replace("-", " ") for n in CLASS_LABELS],
            f"{FEATS}/replica/bigg_text_48.npy")


def main():
    import open_clip
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scannet200", choices=("scannet200", "replica"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    names, out = class_names(a.dataset)

    m, _, _ = open_clip.create_model_and_transforms("ViT-bigG-14", pretrained="laion2b_s39b_b160k")
    m.visual = None                       # only the text tower is needed, and it is the smaller half
    m = m.to(a.device).eval()
    tok = open_clip.get_tokenizer("ViT-bigG-14")
    with torch.no_grad():
        per = [F.normalize(m.encode_text(tok([t.format(n) for n in names]).to(a.device)).float(),
                           dim=-1) for t in TEMPLATES]
        T = F.normalize(torch.stack(per).mean(0), dim=-1).cpu().numpy()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, T)
    print(f"wrote {out}: {T.shape}", flush=True)


if __name__ == "__main__":
    main()
