"""Selection of the image crops from which a mask's 2D embedding is formed.

The procedure is OpenMask3D's: a mask is projected into every posed frame, the frames are ranked by
the number of its points that are visible, the top ``top_k`` are retained, and each is cut at two
paddings around the projected extent. The crops of a mask are encoded and averaged by the caller,
which is what makes one selection serve both image encoders in this repository.

Visibility is occlusion-aware. A projected point is kept only when the sensor depth at its pixel
agrees with its camera-frame depth to within ``vis_threshold`` metres, which removes the points seen
through a wall or through the object standing in front. Selection is pure CPU work and returns
preprocessed crop tensors; the encoder forward pass is batched over a whole scene by the caller.
"""
import numpy as np

from .frames import project


def build_clipL(dev):
    """CLIP ViT-L-14-336, the second of the two 2D heads.

    The OpenAI weights were trained with QuickGELU, so the ``-quickgelu`` tag is required: the plain
    tag loads the same weights under a different activation and degrades silently.
    """
    import open_clip
    m, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14-336-quickgelu", pretrained="openai")
    return m.to(dev).eval(), preprocess


def _depth_path(color_path):
    return color_path.replace("/color/", "/depth/").rsplit(".", 1)[0] + ".png"


def mask_crops(pts_raw, frames, preprocess, img_cache,
               top_k=5, scales=(0.0, 0.3), min_vis=15, depth_min=0.1, depth_max=8.0, vis_cap=3000,
               depth_cache=None, vis_threshold=0.2, return_groups=False):
    """Preprocessed crops of one mask, over the ``top_k`` frames that see it best.

    ``pts_raw`` are the mask's points in the pose frame (:func:`_lib.ply.read_raw`), ``frames`` the
    dictionary :func:`_lib.frames.load_frames` returns, ``preprocess`` the encoder's own transform,
    and ``img_cache`` a per-scene dictionary that keeps each JPEG decoded once. Passing
    ``depth_cache`` as a dictionary enables occlusion-aware visibility.

    ``return_groups`` returns ``(frame_id, n_visible, [crops])`` in the order the selection ranked
    the frames, rather than the flat list of crops.
    """
    import imageio.v2 as imageio
    from PIL import Image
    if len(pts_raw) > vis_cap:
        pts_raw = pts_raw[np.random.default_rng(0).choice(len(pts_raw), vis_cap, replace=False)]
    cand = []
    cw, ch = frames["color_size"]
    for li, fid in enumerate(frames["fids"]):
        fid = int(fid)
        if fid not in frames["colors"]:
            continue
        uv, z = project(pts_raw, frames["poses"][li], frames["intr"])
        vis = ((z >= depth_min) & (z <= depth_max) & (uv[:, 0] >= 0) & (uv[:, 0] < cw) &
               (uv[:, 1] >= 0) & (uv[:, 1] < ch))
        if depth_cache is not None:                        # occlusion-aware visibility
            if fid not in depth_cache:
                try:
                    depth_cache[fid] = imageio.imread(
                        _depth_path(frames["colors"][fid])).astype(np.float32) / 1000.0
                except Exception:
                    depth_cache[fid] = None
            dm = depth_cache[fid]
            if dm is not None:
                dh, dw = dm.shape
                du = np.clip((uv[:, 0] * dw / cw).astype(np.int64), 0, dw - 1)
                dv = np.clip((uv[:, 1] * dh / ch).astype(np.int64), 0, dh - 1)
                sd = dm[dv, du]                            # sensor depth (m) at the projected pixel
                vis = vis & (sd > 0) & (np.abs(sd - z) <= vis_threshold)
        if vis.sum() >= min_vis:
            cand.append((int(vis.sum()), fid, uv, vis))
    if not cand:
        return []
    cand.sort(key=lambda c: c[0], reverse=True)
    crops, groups = [], []
    for _, fid, uv, vis in cand[:top_k]:
        fid = int(fid)
        if fid not in img_cache:                            # decode each JPEG once per scene
            img_cache[fid] = Image.open(frames["colors"][fid]).convert("RGB")
        img = img_cache[fid]
        W, H = img.size
        u, v = uv[vis, 0], uv[vis, 1]
        x0, x1, y0, y1 = float(u.min()), float(u.max()), float(v.min()), float(v.max())
        bw, bh = x1 - x0, y1 - y0
        here = []
        for pad in scales:
            a = max(0, int(x0 - pad * bw))
            b = min(W, int(x1 + pad * bw))
            c = max(0, int(y0 - pad * bh))
            d = min(H, int(y1 + pad * bh))
            if b - a < 4 or d - c < 4:
                continue
            here.append(preprocess(img.crop((a, c, b, d))))
        crops += here
        groups.append((fid, int(vis.sum()), here))
    return groups if return_groups else crops
