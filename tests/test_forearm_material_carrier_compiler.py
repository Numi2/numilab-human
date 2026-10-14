import importlib.util
from pathlib import Path
import struct

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "carrier_compiler", Path(__file__).parents[1] / "tools/prepare_forearm_material_carriers.py")
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)


def test_barycentric_material_term_uses_parent_offset():
    old = np.asarray([.1, .2, .3], dtype='<f4')
    deltas = {10: np.asarray([.001, 0, 0], dtype='<f4'),
              99: np.asarray([0, .002, 0], dtype='<f4')}
    point = np.asarray(old + deltas[10], dtype='<f4')
    compiler.validate_parent_point(point, old, deltas, 99, 10, 2)
    with pytest.raises(ValueError, match="material parent"):
        compiler.validate_parent_point(point, old, deltas, 99, 10, 0)


def test_undeclared_displacement_is_rejected():
    old = np.asarray([0, 0, 0], dtype='<f4')
    with pytest.raises(ValueError):
        compiler.validate_parent_point(np.asarray([.00001, 0, 0], dtype='<f4'),
                                       old, {}, 99, 10, 2)


def test_new_normal_does_not_mutate_original_carrier():
    old = struct.pack('<6f4I4f', 0, 0, 0, 1, 0, 0, 7, 0, 0, 0, 1, 0, 0, 0)
    child = np.asarray([0, 1, 0], dtype='<f4')
    assert compiler.carrier_normal(0, old, child) == old[12:24]
    assert compiler.carrier_normal(2, old, child) == child.tobytes()
    assert struct.unpack_from('<3f', old, 12) == (1, 0, 0)
    assert len(old) == 56


def lineage():
    return {'new_vertices': {'99': {
        'nearest_source_face': {'stable_id': 132, 'face_index': 4,
                                'vertex_gids': [10, 11, 12],
                                'barycentric': [.2, .3, .5],
                                'inferred_source_offset_from_face_base_m': [0, .00025, 0]},
        'material_parents': [{'global_gid': 10, 'coefficient': .4},
                             {'global_gid': 11, 'coefficient': .6}]}}}


def test_lineage_pins_parent_identities_and_positive_partition():
    grouped = {99: [(10, 92, 2, .4, None), (11, 92, 2, .6, None)]}
    compiler.validate_lineage(lineage(), grouped)
    wrong = lineage()
    wrong['new_vertices']['99']['material_parents'][0]['global_gid'] = 12
    with pytest.raises(ValueError, match="parent identity"):
        compiler.validate_lineage(wrong, grouped)
    wrong = lineage()
    wrong['new_vertices']['99']['nearest_source_face']['barycentric'] = [0, 0, 0]
    with pytest.raises(ValueError, match="barycentric"):
        compiler.validate_lineage(wrong, grouped)
    with pytest.raises(ValueError, match="missing new child"):
        compiler.validate_lineage({'new_vertices': {}}, grouped)
