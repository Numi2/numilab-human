#!/usr/bin/env python3
import json
import unittest
from pathlib import Path

import analyze

REPORT = Path('/Users/n/numi-human-resting-evidence-20261005/native-lung-seam-patch-linkage-933/report.json')


class PatchLinkageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(REPORT.read_text())
        cls.events = cls.report['events']
        cls.census = cls.report['canonical_events']

    def test_complete_canonical_event_set_is_topologically_linked(self):
        self.assertEqual(len(self.events), 93)
        self.assertEqual(sum(event['raw_native_intersection_point_count'] for event in self.events), 184)
        self.assertTrue(all(event['raw_native_intersection_span_m'] >= 0.0 for event in self.events))
        self.assertTrue(all(any(side['linked_by_exact_source_topology'] for side in event['owner_face_relations']) for event in self.events))
        self.assertEqual(self.census['patch_topology_relation_counts'], {
            'event_face_is_reciprocal_patch_face': 6,
            'event_face_shares_exact_patch_boundary_edge': 47,
            'event_face_shares_exact_patch_boundary_vertex': 40,
        })
        self.assertEqual(self.census['patch_topology_relation_counts_by_pair'], {
            '305-308': {
                'event_face_shares_exact_patch_boundary_edge': 43,
                'event_face_shares_exact_patch_boundary_vertex': 38,
            },
            '306-307': {
                'event_face_shares_exact_patch_boundary_edge': 4,
                'event_face_shares_exact_patch_boundary_vertex': 2,
            },
            '307-309': {'event_face_is_reciprocal_patch_face': 6},
        })

    def test_source_side_and_quantization_counts_are_retained(self):
        self.assertEqual(self.census['source_side_counts'], {'positive': 90, 'negative': 3, 'ambiguous': 0})
        for event in self.events:
            analyze.assert_within_quantization_bound(
                event['opposite_source_side_intrusion_m'], event['pair_half_ulp_normal_bound_m'])
        self.assertAlmostEqual(self.census['maximum_opposite_source_side_intrusion_m'], 1.07895476442188e-8, places=18)
        self.assertAlmostEqual(self.census['maximum_intrusion_over_pair_half_ulp_bound'], 0.4562124196328977, places=14)
        self.assertEqual(len(set(tuple(x.values()) for x in self.census['source_side_epsilon_sensitivity'].values())), 1)

    def test_only_source_vertex_or_edge_contacts_are_allowed(self):
        self.assertEqual(self.census['unique_source_face_pair_count'], 33)
        self.assertEqual(self.census['unique_source_exact_vertex_or_edge_pairs'], 11)
        self.assertEqual(self.census['unique_source_disjoint_pairs'], 22)
        self.assertEqual(len(self.report['reusable_new_capture_contract']['known_unallowed_source_face_pair_catalog']), 33)
        self.assertEqual(self.census['source_overlap_beyond_allowed_vertex_edge_contacts'], 0)

    def test_source_side_classifier_fails_closed_on_ambiguous_ranges(self):
        eps = analyze.SIDE_EPSILON_M
        self.assertEqual(analyze.classify_source_side({'min_m': -eps, 'max_m': 2 * eps}), 1)
        self.assertEqual(analyze.classify_source_side({'min_m': -2 * eps, 'max_m': eps}), -1)
        self.assertEqual(analyze.classify_source_side({'min_m': -2 * eps, 'max_m': 2 * eps}), 0)
        self.assertEqual(analyze.classify_source_side({'min_m': -eps / 2, 'max_m': eps / 2}), 0)
        with self.assertRaises(ValueError):
            analyze.opposite_source_side_intrusion(0, {'min_m': -1e-9, 'max_m': 1e-9})

    def test_beyond_bound_intrusion_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze.assert_within_quantization_bound(1.01e-8, 1.0e-8)
        analyze.assert_within_quantization_bound(0.99e-8, 1.0e-8)

    def test_unallowed_source_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze.assert_source_pair_allowed({
                'candidate_source_intersection': True,
                'candidate_source_intersection_allowed_vertex_edge': False,
            })
        analyze.assert_source_pair_allowed({
            'candidate_source_intersection': True,
            'candidate_source_intersection_allowed_vertex_edge': True,
        })
        analyze.assert_source_pair_allowed({'candidate_source_intersection': False})

    def test_event_without_exact_patch_face_edge_or_vertex_link_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze.assert_event_patch_linked([
                {'face_is_reciprocal_patch': False, 'patch_boundary_edge_count': 0, 'patch_boundary_vertex_count': 0},
                {'face_is_reciprocal_patch': False, 'patch_boundary_edge_count': 0, 'patch_boundary_vertex_count': 0},
            ])
        analyze.assert_event_patch_linked([
            {'face_is_reciprocal_patch': False, 'patch_boundary_edge_count': 0, 'patch_boundary_vertex_count': 1},
            {'face_is_reciprocal_patch': False, 'patch_boundary_edge_count': 0, 'patch_boundary_vertex_count': 0},
        ])

    def test_full_new_capture_scan_coverage_requires_every_source_face(self):
        contract = self.report['reusable_new_capture_contract']
        coverage = {
            'source_nha_sha256': self.report['inputs']['source_nha']['sha256'],
            'all_faces_considered': True,
            'self_rows': contract['required_self_rows_all_faces'],
            'self_face_counts': contract['source_row_face_counts'],
            'cross_pairs': contract['required_declared_cross_owner_pairs_all_faces'],
            'cross_face_counts': {
                f'{a}-{b}': [contract['source_row_face_counts'][str(a)], contract['source_row_face_counts'][str(b)]]
                for a, b in contract['required_declared_cross_owner_pairs_all_faces']
            },
            'step': 10000,
            'pack_sha256': 'a' * 64,
            'receipt_sha256': 'b' * 64,
        }
        analyze.validate_full_scan_coverage(coverage, self.report['inputs']['source_nha']['sha256'], contract['source_row_face_counts'])
        incomplete = dict(coverage)
        incomplete['self_rows'] = incomplete['self_rows'][:-1]
        with self.assertRaises(ValueError):
            analyze.validate_full_scan_coverage(incomplete, self.report['inputs']['source_nha']['sha256'], contract['source_row_face_counts'])
        incomplete = dict(coverage)
        incomplete['cross_pairs'] = incomplete['cross_pairs'][:-1]
        with self.assertRaises(ValueError):
            analyze.validate_full_scan_coverage(incomplete, self.report['inputs']['source_nha']['sha256'], contract['source_row_face_counts'])
        wrong_source = dict(coverage)
        wrong_source['source_nha_sha256'] = 'wrong-source'
        with self.assertRaises(ValueError):
            analyze.validate_full_scan_coverage(wrong_source, self.report['inputs']['source_nha']['sha256'], contract['source_row_face_counts'])
        wrong_hash = dict(coverage)
        wrong_hash['pack_sha256'] = 'bad'
        with self.assertRaises(ValueError):
            analyze.validate_full_scan_coverage(wrong_hash, self.report['inputs']['source_nha']['sha256'], contract['source_row_face_counts'])

    def test_native_self_intersection_witness_is_rejected(self):
        analyze.reject_native_self_witnesses([])
        with self.assertRaises(ValueError):
            analyze.reject_native_self_witnesses([{'stable_id': 307, 'face_a': 1, 'face_b': 2}])

    def test_new_or_unknown_witness_is_rejected_until_reviewed(self):
        pair_report = json.loads(Path('/Users/n/numi-human-resting-evidence-20261005/native-lung-seam-structural-audit-927/source-native-hit-pair-comparison.json').read_text())
        known = {(tuple(row['owners']), tuple(row['face_ids'])) for row in pair_report['face_pairs']}
        analyze.reject_unreviewed_witnesses([{'owners': [305, 308], 'face_ids': [24552, 26582]}], known)
        with self.assertRaises(ValueError):
            analyze.reject_unreviewed_witnesses([{'owners': [305, 308], 'face_ids': [1, 2]}], known)
        with self.assertRaises(ValueError):
            analyze.reject_unreviewed_witnesses([{'owners': [305, 311], 'face_ids': [1, 2]}], known)

    def test_report_marks_scope_and_uncertainty(self):
        self.assertEqual(self.report['status'], 'PASS_SCOPED_GEOMETRIC_CLASSIFICATION')
        text = ' '.join(self.report['interpretation_and_limits']).lower()
        self.assertIn('not proof of anatomical intention', text)
        self.assertIn('not a bound on respiratory deformation', text)
        self.assertIn('all 93 exact native intersections remain', text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
