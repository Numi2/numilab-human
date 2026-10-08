from __future__ import annotations

import importlib.util
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
FEATURE_1084 = HERE.parent / 'native-lung-exact-classifier-index-1084/native_feature_index.py'
ADAPTER_1085 = HERE / 'native_feature_index_1085.py'
SOURCE_RULE = Path('/Users/n/numi-human-lung-contact-classifier-001/src/numilab_human/lung_contact_classification.py')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fake_audit_base():
    def classify(a, b, ra, rb, points, pose, maps, lobe_maps):
        return 'unclassified_cross_intersection' if 311 in (a, b) else 'legacy_lobe_lobe_result'

    def pkey(point):
        return tuple(int(value) for value in point)

    base = SimpleNamespace(classify=classify, pkey=pkey)
    adapter = load(ADAPTER_1085, 'native_feature_index_1085_under_test')
    adapter.install(base, feature_1084=load(FEATURE_1084, 'native_feature_index_1084_under_test'),
                    source_rule=load(SOURCE_RULE, 'lung_contact_rule_under_test'))
    return base


def case(boundary_edges, points, *, db_equals_lb=True, valid=True):
    # Two separate candidate lobe triangles avoid accidental mapped-face adjacency.
    vertices = np.asarray([
        [10, 0, 0], [11, 0, 0], [10, 1, 0],
        [12, 0, 0], [13, 0, 0], [12, 1, 0],
    ], dtype=np.float64)
    lobe_faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    d_vertices = np.asarray([[20, 0, 0], [21, 0, 0], [20, 1, 0]], dtype=np.float64)
    d_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    pose = {305: {'v': vertices, 'f': lobe_faces}, 311: {'v': d_vertices, 'f': d_faces}}
    edges = {tuple(sorted((tuple(a), tuple(b)))) for a, b in boundary_edges}
    map_doc = {'valid': valid, 'd2l': {0: 0}, 'l2d': {0: 0}, 'db': edges,
               'lb': edges if db_equals_lb else edges | {((99, 99, 99), (100, 100, 100))}}
    # Base exact-record format: (triangle keys, bbox lo, bbox hi, row, vertex ids).
    lobe_record = ((), (), (), 1, (3, 4, 5))
    d_record = ((), (), (), 0, (0, 1, 2))
    return pose, {305: map_doc}, lobe_record, d_record, points


class NativeFeatureIndex1085Tests(unittest.TestCase):
    def test_exact_boundary_vertex_uses_source_label_and_legacy_native_label(self):
        base = fake_audit_base()
        boundary = (((0, 0, 0), (4, 0, 0)), ((0, 0, 0), (0, 4, 0)))
        pose, maps, lobe_record, d_record, points = case(boundary, [(0, 0, 0)])
        label = base.classify(305, 311, lobe_record, d_record, points, pose, maps, {})
        self.assertEqual(label, 'exact_fullunion_boundary_contact')

    def test_off_boundary_point_remains_unclassified(self):
        base = fake_audit_base()
        boundary = (((0, 0, 0), (4, 0, 0)), ((0, 0, 0), (0, 4, 0)))
        pose, maps, lobe_record, d_record, points = case(boundary, [(1, 1, 0)])
        assert base.classify(305, 311, lobe_record, d_record, points, pose, maps, {}) == 'unclassified_cross_intersection'

    def test_points_on_different_boundary_edges_do_not_authorize_contact(self):
        base = fake_audit_base()
        boundary = (((0, 0, 0), (4, 0, 0)), ((0, 0, 0), (0, 4, 0)))
        pose, maps, lobe_record, d_record, points = case(boundary, [(2, 0, 0), (0, 2, 0)])
        assert base.classify(305, 311, lobe_record, d_record, points, pose, maps, {}) == 'unclassified_cross_intersection'

    def test_noninteger_lattice_witness_fails_closed_without_rounding(self):
        base = fake_audit_base()
        boundary = (((0, 0, 0), (4, 0, 0)),)
        pose, maps, lobe_record, d_record, points = case(boundary, [(Fraction(1, 2), Fraction(0), Fraction(0))])
        assert base.classify(305, 311, lobe_record, d_record, points, pose, maps, {}) == 'unclassified_cross_intersection'

    def test_reverse_d311_argument_order_uses_same_native_classification(self):
        base = fake_audit_base()
        boundary = (((0, 0, 0), (4, 0, 0)),)
        pose, maps, lobe_record, d_record, points = case(boundary, [(2, 0, 0)])
        forward = base.classify(305, 311, lobe_record, d_record, points, pose, maps, {})
        reverse = base.classify(311, 305, d_record, lobe_record, points, pose, maps, {})
        assert forward == reverse == 'exact_fullunion_boundary_contact'

    def test_native_map_mismatch_fails_closed(self):
        base = fake_audit_base()
        boundary = (((0, 0, 0), (4, 0, 0)),)
        pose, maps, lobe_record, d_record, points = case(boundary, [(0, 0, 0)], db_equals_lb=False)
        assert base.classify(305, 311, lobe_record, d_record, points, pose, maps, {}) == 'unclassified_current_map_mismatch'

    def test_non_d_lobe_behavior_is_delegated_to_1084(self):
        base = fake_audit_base()
        assert base.classify(305, 306, (), (), (), {}, {}, {}) == 'legacy_lobe_lobe_result'


def test_exact_boundary_vertex_uses_source_label_and_legacy_native_label():
    NativeFeatureIndex1085Tests().test_exact_boundary_vertex_uses_source_label_and_legacy_native_label()


def test_off_boundary_point_remains_unclassified():
    NativeFeatureIndex1085Tests().test_off_boundary_point_remains_unclassified()


def test_points_on_different_boundary_edges_do_not_authorize_contact():
    NativeFeatureIndex1085Tests().test_points_on_different_boundary_edges_do_not_authorize_contact()


def test_noninteger_lattice_witness_fails_closed_without_rounding():
    NativeFeatureIndex1085Tests().test_noninteger_lattice_witness_fails_closed_without_rounding()


def test_reverse_d311_argument_order_uses_same_native_classification():
    NativeFeatureIndex1085Tests().test_reverse_d311_argument_order_uses_same_native_classification()


def test_native_map_mismatch_fails_closed():
    NativeFeatureIndex1085Tests().test_native_map_mismatch_fails_closed()


def test_non_d_lobe_behavior_is_delegated_to_1084():
    NativeFeatureIndex1085Tests().test_non_d_lobe_behavior_is_delegated_to_1084()
