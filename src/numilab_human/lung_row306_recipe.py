"""Fail-closed aligned filtering for row306 face provenance."""
from __future__ import annotations
import numpy as np

def filter_parallel_face_rows(faces, origins, kinds, keep_mask):
    """Apply one caller-selected row mask to faces and parallel lineage arrays."""
    f, o, k, keep = map(np.asarray, (faces, origins, kinds, keep_mask))
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must have shape (face_count, 3)")
    if o.ndim != 1 or k.ndim != 1:
        raise ValueError("face provenance arrays must be one-dimensional")
    if len(o) != len(f) or len(k) != len(f):
        raise ValueError("face and provenance array lengths must match")
    if keep.dtype != np.dtype(bool) or keep.ndim != 1 or len(keep) != len(f):
        raise ValueError("keep_mask must be a boolean vector matching face_count")
    return f[keep].copy(), o[keep].copy(), k[keep].copy()
