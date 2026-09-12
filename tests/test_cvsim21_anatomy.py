from __future__ import annotations

import copy
import hashlib
import math
import unittest
from unittest.mock import patch

from numilab_human import cvsim21 as source
from numilab_human import cvsim21_anatomy as anatomy
from numilab_human.model import ImportError as HumanImportError


VERTICES = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
FACES = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]


def digest(value: object) -> str:
    return hashlib.sha256(source.canonical(value) + b'\n').hexdigest()


def multiply(a: list, b: list) -> list:
    return [[math.fsum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


class CVSim21AnatomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline, cls.baseline_lowering = source.compile_source()
        cls.baseline_bytes = source.canonical(cls.baseline)
        cls.config = source.read_json(anatomy.CONFIG)
        cls.native, cls.manifest = anatomy.compile_registration()

    def test_exact_four_cavity_ids_are_the_only_physical_record_changes(self) -> None:
        expected = {15: 'FMA:11359', 16: 'FMA:9291', 19: 'FMA:9465', 20: 'FMA:9466'}
        restored = copy.deepcopy(self.native)
        for i, (actual, original) in enumerate(zip(restored['compartments'], self.baseline['compartments'], strict=True)):
            if i in expected:
                self.assertEqual(actual['anatomical_region_id'], expected[i])
                actual['anatomical_region_id'] = original['anatomical_region_id']
            self.assertEqual(actual, original)
        self.assertNotEqual(restored['model_id'], self.baseline['model_id'])
        self.assertNotEqual(restored['authored_graph_sha256'], self.baseline['authored_graph_sha256'])
        self.assertNotEqual(restored['source_graph_sha256'], self.baseline['source_graph_sha256'])
        for key in ('model_id', 'authored_graph_sha256', 'source_graph_sha256'):
            restored[key] = self.baseline[key]
        self.assertEqual(source.canonical(restored), self.baseline_bytes)

    def test_source_volume_owners_mass_and_physiological_parameters_are_unchanged(self) -> None:
        self.assertEqual(len(self.native['compartments']), 21)
        self.assertEqual(len(self.native['connections']), 24)
        owners = [c['physical_volume_owner_id'] for c in self.native['compartments']]
        self.assertEqual(owners, [c['physical_volume_owner_id'] for c in self.baseline['compartments']])
        self.assertEqual(len(set(owners)), 21)
        self.assertFalse(self.manifest['hydraulic_parameters_changed'])
        self.assertEqual(self.manifest['source_blood_volume_m3'], self.baseline_lowering['total_blood_volume_m3'])
        self.assertEqual(self.manifest['added_mechanical_mass_kg'], 0)
        self.assertEqual(self.manifest['withdrawn_mechanical_mass_kg'], 0)
        self.assertEqual(self.manifest['blood_in_gross_body_mass'], 'unresolved')
        self.assertEqual(self.native['qualification'], 'source_model_variant')
        for binding in self.manifest['bindings']:
            row = self.baseline['compartments'][binding['source_index']]
            self.assertEqual(binding['physical_volume_owner_id'], row['physical_volume_owner_id'])
            self.assertEqual(binding['compartment_stable_identifier'], row['stable_identifier'])
            self.assertEqual(binding['hydraulic_initial_volume_m3'], row['initial_volume_m3'])
            self.assertEqual(binding['hydraulic_reference_volume_m3'], row['reference_volume_m3'])
            self.assertIsNone(binding['geometric_scale_fit'])
            self.assertIsNone(binding['reference_cardiac_phase'])
            self.assertIsNone(binding['world_or_body_frame_registration'])
            self.assertIsNone(binding['mechanical_mass_owner'])
            self.assertIsNone(binding['geometry_moments']['physical_volume_m3'])
            self.assertIsNone(binding['geometry_moments']['mechanical_mass_kg'])
        for claim in ('atlas_to_body_registration', 'hydraulic_volume_matches_anatomical_volume',
                      'blood_tissue_mass_partition', 'pressure_deformation_coupling',
                      'physiological_calibration', 'standing_walking'):
            self.assertIs(self.manifest['qualification'][claim], False)

    def test_source_compiler_remains_byte_identical_after_registration(self) -> None:
        after, lowering = source.compile_source()
        self.assertEqual(source.canonical(after), self.baseline_bytes)
        self.assertEqual(source.canonical(lowering), source.canonical(self.baseline_lowering))
        self.assertEqual(digest(after), 'eeb6ebc5dad5cb413587038532ac5badc3d3e8aa419cca111604239e7f692818')
        self.assertTrue(all(c['anatomical_region_id'].startswith('source_aggregate:CVSim21:') for c in after['compartments']))

    def test_provenance_hashes_bind_geometry_native_and_full_registration_preimage(self) -> None:
        manifest = self.manifest
        self.assertEqual(manifest['native_content_sha256'], digest(self.native))
        self.assertEqual(manifest['source_native_sha256'], digest(self.baseline))
        self.assertEqual(manifest['cavity_geometry_sha256'], digest(manifest['cavity_geometry']))
        self.assertEqual(self.native['authored_graph_sha256'], digest(manifest['identity_preimage']))
        preimage = manifest['identity_preimage']
        self.assertEqual(preimage['configuration'], self.config)
        self.assertEqual(preimage['cavity_geometry_sha256'], manifest['cavity_geometry_sha256'])
        self.assertEqual(preimage['source_native_sha256'], manifest['source_native_sha256'])
        self.assertEqual(preimage['bindings'], manifest['bindings'])
        changed = copy.deepcopy(preimage)
        changed['bindings'][0]['semantic_id'] = 'FMA:7096'
        self.assertNotEqual(digest(changed), self.native['authored_graph_sha256'])
        self.assertEqual(manifest['unregistered_compartment_indices'], [i for i in range(21) if i not in {15, 16, 19, 20}])

    def test_registration_is_deterministic_and_preserves_caller_configuration(self) -> None:
        config = copy.deepcopy(self.config)
        before = source.canonical(config)
        native, manifest = anatomy.compile_registration(config=config)
        self.assertEqual(source.canonical(config), before)
        self.assertEqual(source.canonical(native), source.canonical(self.native))
        self.assertEqual(source.canonical(manifest), source.canonical(self.manifest))

    def test_unsupported_modes_mass_claims_and_source_overrides_reject_before_extraction(self) -> None:
        cases = [('schema', 'unrecognized'), ('id', 'arbitrary_model'),
                 ('volume_coordinates', 'rescale_to_atlas'), ('volume_coordinates', []),
                 ('source_manifest_sha256', '0' * 64), ('association', 'same_subject_calibrated'),
                 ('association', True), ('mechanical_mass_policy', 'add_blood_to_existing_mass'),
                 ('mechanical_mass_policy', 'subtract_5150mL_from_rigid_mass'),
                 ('geometry_source', {}), ('geometry_source', None),
                 ('hydraulic_overrides', {'initial_volume_m3': 1}),
                 ('density_kg_per_m3', 1060), ('atlas_to_body_registration', True),
                 ('source_mode', {'ABReflexOn': True}), ('blood_tissue_mass_partition', True)]
        for key, value in cases:
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key, value=value), patch.object(anatomy.geometry, 'extract_cavity_surfaces') as extraction:
                with self.assertRaises(HumanImportError):
                    anatomy.compile_registration(config=config)
                extraction.assert_not_called()
        for key in self.config['geometry_source']:
            config = copy.deepcopy(self.config)
            config['geometry_source'][key] = '0' * 64
            with self.subTest(geometry_hash=key), patch.object(anatomy.geometry, 'extract_cavity_surfaces') as extraction:
                with self.assertRaises(HumanImportError):
                    anatomy.compile_registration(config=config)
                extraction.assert_not_called()
        for malformed in ([], 'not a configuration', True, float('nan')):
            with self.subTest(malformed=malformed), self.assertRaises(HumanImportError):
                anatomy.compile_registration(config=malformed)

    def test_distinct_source_volume_variant_remains_a_distinct_unmodified_hydraulic_model(self) -> None:
        config = copy.deepcopy(self.config)
        config['volume_coordinates'] = 'heldt_table_aligned'
        registered, manifest = anatomy.compile_registration(config=config)
        base_config = source.read_json(source.CONFIG)
        base_config['volume_coordinates'] = 'heldt_table_aligned'
        baseline, _ = source.compile_source(config=base_config)
        for actual, expected in zip(registered['compartments'], baseline['compartments'], strict=True):
            actual = copy.deepcopy(actual)
            actual['anatomical_region_id'] = expected['anatomical_region_id']
            self.assertEqual(actual, expected)
        self.assertEqual(registered['connections'], baseline['connections'])
        self.assertEqual(manifest['source_native_sha256'], digest(baseline))
        self.assertNotEqual(registered['authored_graph_sha256'], self.native['authored_graph_sha256'])
        self.assertEqual(manifest['source_blood_volume_m3'], self.manifest['source_blood_volume_m3'])


class GeometryMomentTests(unittest.TestCase):
    def assertVectorClose(self, actual: list, expected: list, *, relative: float = 2e-12, absolute: float = 2e-14) -> None:
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected, strict=True):
            if isinstance(b, list):
                self.assertVectorClose(a, b, relative=relative, absolute=absolute)
            else:
                self.assertTrue(math.isclose(a, b, rel_tol=relative, abs_tol=absolute), f'{a!r} != {b!r}')

    def test_unit_tetrahedron_matches_independent_polynomial_integrals(self) -> None:
        result = anatomy.geometry_moments(VERTICES, FACES)
        self.assertAlmostEqual(result['signed_volume_m3'], 1 / 6, places=15)
        self.assertVectorClose(result['centroid_source_frame_m'], [1/4] * 3)
        self.assertVectorClose(result['first_volume_moment_m4'], [1/24] * 3)
        self.assertVectorClose(result['second_volume_moment_m5'], [[1/60 if i == j else 1/120 for j in range(3)] for i in range(3)])
        self.assertVectorClose(result['central_second_volume_moment_m5'], [[1/160 if i == j else -1/480 for j in range(3)] for i in range(3)])
        self.assertVectorClose(result['inertia_per_unit_density_m5'], [[1/80 if i == j else 1/480 for j in range(3)] for i in range(3)])
        self.assertIsNone(result['physical_volume_m3'])
        self.assertIsNone(result['density_kg_per_m3'])
        self.assertIsNone(result['mechanical_mass_kg'])
        self.assertIs(result['self_intersection_qualified'], False)
        self.assertIs(result['interdomain_disjointness_qualified'], False)

    def test_translation_rotation_and_scale_obey_moment_tensor_transform(self) -> None:
        axis = [x / math.sqrt(14) for x in (1, 2, 3)]
        angle = 0.7
        skew = [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        rotation = [[math.cos(angle) * (i == j) + (1 - math.cos(angle))*axis[i]*axis[j]
                     + math.sin(angle)*skew[i][j] for j in range(3)] for i in range(3)]
        identity = [[float(i == j) for j in range(3)] for i in range(3)]
        central = [[1/160 if i == j else -1/480 for j in range(3)] for i in range(3)]
        cases = [(identity, 1, [13., -7., 23.]), (rotation, 1, [0., 0., 0.]),
                 (identity, 3.25, [0., 0., 0.]), (rotation, 0.2, [13., -7., 23.])]
        for rotate, scale, translation in cases:
            vertices = [[translation[i] + scale * math.fsum(rotate[i][j]*p[j] for j in range(3))
                         for i in range(3)] for p in VERTICES]
            result = anatomy.geometry_moments(vertices, FACES)
            volume = scale**3 / 6
            center = [translation[i] + scale * math.fsum(rotate[i][j]/4 for j in range(3)) for i in range(3)]
            transpose = [[rotate[j][i] for j in range(3)] for i in range(3)]
            rotated = multiply(multiply(rotate, central), transpose)
            expected_central = [[scale**5 * x for x in row] for row in rotated]
            with self.subTest(scale=scale, translation=translation):
                self.assertTrue(math.isclose(result['absolute_signed_volume_m3'], volume, rel_tol=2e-12))
                self.assertVectorClose(result['centroid_source_frame_m'], center)
                self.assertVectorClose(result['first_volume_moment_m4'], [volume*x for x in center])
                self.assertVectorClose(result['central_second_volume_moment_m5'], expected_central)
                self.assertVectorClose(result['second_volume_moment_m5'], [[expected_central[i][j] + volume*center[i]*center[j] for j in range(3)] for i in range(3)])
                self.assertVectorClose(result['inertia_per_unit_density_m5'], [[(sum(expected_central[k][k] for k in range(3)) if i == j else 0) - expected_central[i][j] for j in range(3)] for i in range(3)])

    def test_positive_tensor_admission_is_invariant_under_extreme_finite_uniform_scale(self) -> None:
        reference = anatomy.geometry_moments(VERTICES, FACES)
        for scale in (1e-50, 1e-30, 1e30, 1e40, 1e50):
            with self.subTest(scale=scale):
                result = anatomy.geometry_moments([[x * scale for x in p] for p in VERTICES], FACES)
                self.assertTrue(math.isclose(result['absolute_signed_volume_m3'] / scale**3,
                                             reference['absolute_signed_volume_m3'], rel_tol=2e-12))
                self.assertVectorClose([x / scale for x in result['centroid_source_frame_m']],
                                       reference['centroid_source_frame_m'])
                self.assertVectorClose([x / scale**4 for x in result['first_volume_moment_m4']],
                                       reference['first_volume_moment_m4'])
                for key in ('second_volume_moment_m5', 'central_second_volume_moment_m5',
                            'inertia_per_unit_density_m5'):
                    self.assertVectorClose([[x / scale**5 for x in row] for row in result[key]], reference[key])

    def test_global_winding_reversal_only_changes_signed_volume_and_winding(self) -> None:
        positive = anatomy.geometry_moments(VERTICES, FACES)
        negative = anatomy.geometry_moments(VERTICES, [list(reversed(face)) for face in FACES])
        self.assertEqual(negative['source_winding'], 'negative')
        self.assertAlmostEqual(negative['signed_volume_m3'], -positive['signed_volume_m3'], places=15)
        for key in ('absolute_signed_volume_m3', 'centroid_source_frame_m', 'first_volume_moment_m4',
                    'second_volume_moment_m5', 'central_second_volume_moment_m5', 'inertia_per_unit_density_m5'):
            if isinstance(positive[key], list):
                self.assertVectorClose(negative[key], positive[key])
            else:
                self.assertAlmostEqual(negative[key], positive[key], places=15)

    def test_open_inconsistent_disconnected_pinched_and_zero_volume_surfaces_rejected(self) -> None:
        wrong_winding = copy.deepcopy(FACES)
        wrong_winding[0].reverse()
        shifted = [[p[0]+3, p[1], p[2]] for p in VERTICES]
        pinched_vertices = VERTICES + [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
        pinched_map = [0, 4, 5, 6]
        cases = [(VERTICES, FACES[:-1]), (VERTICES, wrong_winding),
                 (VERTICES + shifted, FACES + [[i+4 for i in face] for face in FACES]),
                 (pinched_vertices, FACES + [[pinched_map[i] for i in face] for face in FACES]),
                 ([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]], FACES),
                 (VERTICES + [[2., 2., 2.]], FACES), (VERTICES, FACES + [FACES[0]]),
                 ([[float('nan'), 0., 0.]] + VERTICES[1:], FACES),
                 ([[0., 0., 0.], [1e308, 0., 0.], [0., 1e308, 0.], [0., 0., 1e308]], FACES)]
        for vertices, faces in cases:
            with self.subTest(vertex_count=len(vertices), faces=faces), self.assertRaises(HumanImportError):
                anatomy.geometry_moments(vertices, faces)


if __name__ == '__main__':
    unittest.main()
