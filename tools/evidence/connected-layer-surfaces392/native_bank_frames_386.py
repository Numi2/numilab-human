#!/usr/bin/env python3
"""Source-bank query transforms from emitted native matrices, never re-normalized."""
from __future__ import annotations
import numpy as np


def native_current_world_to_body(record):
    if 'current' not in record:
        raise ValueError('native body record has no current frame')
    state=record['current']
    raw=np.asarray(state.get('local_to_world_row_major_f32', []),dtype=np.float64)
    origin_raw=np.asarray(state.get('position_world_m', []),dtype=np.float64)
    if raw.shape!=(9,) or not np.isfinite(raw).all():
        raise ValueError('native current frame requires nine finite emitted matrix entries')
    if origin_raw.shape!=(3,) or not np.isfinite(origin_raw).all():
        raise ValueError('native current frame requires a finite world origin')
    matrix=raw.astype(np.float32).astype(np.float64).reshape(3,3)
    if not np.array_equal(matrix.reshape(-1),raw):
        raise ValueError('native current frame matrix is not exact emitted Float32 data')
    origin=origin_raw.astype(np.float32).astype(np.float64)
    bound=8*float(np.finfo(np.float32).eps)
    orth=float(np.max(np.abs(matrix.T@matrix-np.eye(3))))
    determinant=float(np.linalg.det(matrix))
    if orth>bound or determinant<=0 or abs(determinant-1)>bound:
        raise ValueError('native current matrix exceeds the existing Float32 SO(3) envelope')
    inverse=np.linalg.inv(matrix)
    if not np.isfinite(inverse).all():
        raise ValueError('native current matrix inverse is nonfinite')
    return inverse,origin,{'native_matrix_orthogonality_error':orth,
                           'native_matrix_determinant':determinant,
                           'native_matrix_so3_bound':bound,
                           'frame_method':'native_all_bodies_f32_local_to_world_matrix_and_inverse'}
