"""Restore the Pillow constants detectron2 still references.

detectron2 0.6 uses Image.LINEAR (transform.py:46) as a default argument, which Pillow removed in
10.0 in favour of Image.Resampling.BILINEAR. The default is evaluated at import time, so the
module cannot even load. Patching the alias back is safer than editing the vendored package:
LINEAR and BILINEAR were the same filter, so this restores the old spelling, not old behaviour.
Import this before anything that imports detectron2.
"""
from PIL import Image
for old, new in (("LINEAR", "BILINEAR"), ("CUBIC", "BICUBIC"), ("ANTIALIAS", "LANCZOS")):
    if not hasattr(Image, old):
        setattr(Image, old, getattr(Image.Resampling, new))
