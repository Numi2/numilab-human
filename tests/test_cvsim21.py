"""Source identity/admission and coordinate mapping; no Python physics stepping."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

from numilab_human import cvsim21 as source
from numilab_human.model import ImportError as HumanImportError


class CVSim21AuthoringTests(unittest.TestCase):
    def setUp(self):
        self.config = source.read_json(source.CONFIG)

    def test_complete_absolute_source_graph(self):
        native, manifest = source.compile_source(config=self.config)
        self.assertEqual((len(native['compartments']), len(native['connections'])), (21, 24))
        self.assertEqual([r['stable_identifier'] for r in native['compartments'] + native['connections']], list(range(1, 46)))
        self.assertEqual(len({r['physical_volume_owner_id'] for r in native['compartments']}), 21)
        self.assertTrue(all(r['storage_kind'] == 'absolute_volume' and r['initial_volume_m3'] > 0 for r in native['compartments']))
        self.assertAlmostEqual(math.fsum(r['initial_volume_m3'] for r in native['compartments']), .00515, places=14)
        self.assertAlmostEqual(math.fsum(r['reference_volume_m3'] for r in native['compartments']), .004166, places=14)
        self.assertEqual(manifest['parameters']['parameter_count'], 153)
        self.assertEqual(native['qualification'], 'source_model_variant')
        self.assertEqual(native['species'], [])
        self.assertEqual(native['tissue_reservoirs'], [])
        self.assertEqual(native['exchanges'], [])
        self.assertEqual(manifest['native_content_sha256'], hashlib.sha256(source.canonical(native) + b'\n').hexdigest())

    def test_rational_cardiac_clock_and_native_law_counts(self):
        native, _ = source.compile_source()
        chambers = [r for r in native['compartments'] if r['pressure_law'] == 'cosine_pulse_elastance']
        self.assertEqual(len(chambers), 4)
        for row in chambers:
            self.assertEqual((row['period_numerator_seconds'], row['period_denominator'], row['period_seconds']), ('6', '7', 0.0))
            self.assertLessEqual(row['phase_delay'] + row['activation_end'], 1)
        self.assertEqual(sum(r['pressure_law'] == 'atan_compliance' for r in native['compartments']), 3)
        self.assertEqual(sum(r['flow_law'] == 'one_way_resistance' for r in native['connections']), 5)
        self.assertEqual(sum(r['flow_law'] == 'starling_resistance' for r in native['connections']), 1)

    def test_table_variant_is_only_paired_volume_translation(self):
        raw, raw_manifest = source.compile_source(config=self.config)
        self.config['volume_coordinates'] = 'heldt_table_aligned'
        aligned, manifest = source.compile_source(config=self.config)
        self.assertNotEqual(raw['authored_graph_sha256'], aligned['authored_graph_sha256'])
        self.assertEqual(raw['connections'], aligned['connections'])
        for i, (a, b) in enumerate(zip(raw['compartments'], aligned['compartments'])):
            expected = 184e-6 if i == 2 else -184e-6 if i == 5 else 0
            for key in a:
                if key in {'reference_volume_m3', 'initial_volume_m3'}:
                    self.assertAlmostEqual(b[key] - a[key], expected, places=16)
                else:
                    self.assertEqual(a[key], b[key])
        self.assertEqual(manifest['total_blood_volume_m3'], raw_manifest['total_blood_volume_m3'])
        self.assertEqual(len(manifest['explicit_variants']), 4)

    def test_no_anatomical_mapping_or_mechanical_mass_inferred(self):
        native, manifest = source.compile_source()
        self.assertTrue(all(r['anatomical_region_id'].startswith('source_aggregate:') for r in native['compartments']))
        self.assertTrue(all(r['anatomy_registration'] is None and r['mechanical_mass_owner'] is None for r in manifest['compartment_bindings']))
        self.assertEqual(len(manifest['parameters']['source_inconsistencies']), 6)

    def test_incidence_is_closed_connected_and_matches_regional_beds(self):
        native, _ = source.compile_source()
        adjacency = {i: set() for i in range(1, 22)}
        for edge in native['connections']:
            self.assertNotEqual(edge['from'], edge['to'])
            adjacency[edge['from']].add(edge['to'])
            adjacency[edge['to']].add(edge['from'])
        seen, pending = set(), [1]
        while pending:
            current = pending.pop()
            if current not in seen:
                seen.add(current)
                pending.extend(adjacency[current] - seen)
        self.assertEqual(len(seen), 21)
        # Renal and splanchnic venous outflow joins abdominal veins, not pulmonary circulation.
        self.assertEqual((native['connections'][10]['from'], native['connections'][10]['to']), (9, 14))
        self.assertEqual((native['connections'][13]['from'], native['connections'][13]['to']), (11, 14))

    def test_unknown_configuration_and_physiological_overrides_rejected(self):
        for key, value in [('heart_rate', 80), ('species', []), ('calibrated', True), ('mass_density', 1060)]:
            with self.subTest(key=key):
                config = deepcopy(self.config)
                config[key] = value
                with self.assertRaises(HumanImportError):
                    source.compile_source(config=config)

    def test_mode_changes_rejected(self):
        for key, value in [('ABReflexOn', True), ('CPReflexOn', True), ('tiltTestOn', True), ('ABReflexOn', 0), ('clock', 'original_discrete_sa_node')]:
            with self.subTest(key=key, value=value):
                config = deepcopy(self.config)
                config['source_mode'][key] = value
                with self.assertRaises(HumanImportError):
                    source.compile_source(config=config)

    def test_numerical_controls_cannot_change_units_or_claim_provenance(self):
        for key, value in [('unit', 'mmHg'), ('role', 'calibration'), ('value', True), ('value', float('nan')), ('value', 0)]:
            with self.subTest(key=key, value=value):
                config = deepcopy(self.config)
                config['numerical_settings']['pressure_scale_pa'][key] = value
                with self.assertRaises(HumanImportError):
                    source.compile_source(config=config)

    def test_source_and_initial_hash_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copied = root / 'source'
            shutil.copytree(source.SOURCE, copied)
            path = copied / '21-comp-backend/sim/equation.c'
            path.write_bytes(path.read_bytes() + b'\n')
            with self.assertRaisesRegex(HumanImportError, 'SHA256'):
                source.compile_source(directory=copied)
            initial = root / 'initial.json'
            data = source.read_json(source.INITIAL)
            data['volume_mL'][0] += 1
            initial.write_text(json.dumps(data))
            with self.assertRaisesRegex(HumanImportError, 'SHA256'):
                source.compile_source(initial_path=initial)

    def test_redirected_initial_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'initial.json'
            path.symlink_to(source.INITIAL)
            with self.assertRaisesRegex(HumanImportError, 'redirected'):
                source.compile_source(initial_path=path)

    def test_duplicate_and_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            for data in ['{"schema":1,"schema":2}', '{"value":NaN}']:
                path.write_text(data)
                with self.assertRaises(HumanImportError):
                    source.read_json(path)

    def test_repeated_authoring_is_byte_deterministic(self):
        a = source.compile_source()
        b = source.compile_source()
        self.assertEqual(tuple(map(source.canonical, a)), tuple(map(source.canonical, b)))


if __name__ == '__main__':
    unittest.main()
