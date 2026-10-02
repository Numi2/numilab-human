"""Focused topology checks for the one-element endocardial FEC candidate."""

import numpy as np

from tools.audit_cardiac_fec_layer_topology import _boundary_face_fec_mask


def test_face_centroid_selects_one_boundary_cell_not_a_vertex_ring():
    cells = np.array([
        [0, 1, 2, 3],       # endocardial face shared below, so not exterior
        [4, 5, 6, 7],       # one endocardial vertex only
        [8, 9, 10, 11],     # boundary face at Z=.6, opposite vertex at Z=.9
        [12, 13, 14, 15],   # boundary face above the Z=.7 limit
        [0, 1, 2, 16],      # shares the first cell's endocardial face
    ], dtype='<u4')
    rho = np.ones(17, dtype='<f8')
    z = np.full(17, 0.5, dtype='<f8')
    rho[[0, 1, 2, 4, 8, 9, 10, 12, 13, 14]] = 0.0
    z[[8, 9, 10]] = 0.6
    z[11] = 0.9
    z[[12, 13, 14]] = 0.8

    selected, face_count = _boundary_face_fec_mask(cells, rho, z, 0.7)

    assert np.array_equal(selected, [False, False, True, False, False])
    assert face_count == 1
