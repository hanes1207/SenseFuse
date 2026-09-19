"""Shared components of the feature writers in ``scripts/extract``.

Importing this package places the repository root on ``sys.path``, so that a writer executed as
``python scripts/extract/<name>.py`` resolves ``utils.paths`` and the vendored benchmark sources in
``utils/benchmark`` exactly as ``main.py`` does.

The modules here hold only what more than one writer needs: the two readers of the raw scans
(:mod:`~scripts.extract._lib.ply`, :mod:`~scripts.extract._lib.frames`), the crop selection shared
by both image encoders (:mod:`~scripts.extract._lib.crops`), and the Uni3D encoder together with its
preprocessing (:mod:`~scripts.extract._lib.uni3d`). Everything specific to one encoder stays in the
writer that uses it.
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import utils  # noqa: E402,F401  (places utils/benchmark's four sources on sys.path)
