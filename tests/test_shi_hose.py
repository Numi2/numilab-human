from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET

from numilab_human.model import ImportError as HumanImportError
from numilab_human.shi_hose import (
    CONFIG, FILES, SOURCE, SOURCE_LOCK_SHA256, _units, canonical, compile_source, main, read_json,
)


class ShiHoseSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = read_json(CONFIG)

    def test_curated_files_and_full_parameter_coverage(self) -> None:
        native, manifest = compile_source(config=self.config)
        self.assertEqual(len(FILES), 15)
        self.assertEqual(manifest["source_parameter_count"], 59)
        self.assertEqual(len({(p["file"], p["symbol"]) for p in manifest["parameters"]}), 59)
        self.assertEqual(native["source_graph_sha256"], SOURCE_LOCK_SHA256)
        self.assertEqual(hashlib.sha256((SOURCE / "source-lock.json").read_bytes()).hexdigest(), SOURCE_LOCK_SHA256)
        self.assertEqual(native["schema"], "HumanPack.physiology-native.v2")
        self.assertEqual(native["qualification"], "source_model_reproduction")
        self.assertIn("not_qualified_by_compilation", manifest["scientific_status"])

    def test_source_units_and_literal_constants_are_preserved(self) -> None:
        native, manifest = compile_source(config=self.config)
        self.assertEqual(manifest["source_unit_si_multipliers"]["UnitP"], 133)
        self.assertEqual(manifest["source_unit_si_multipliers"]["UnitV"], 1e-6)
        nodes = {row["id"]: row for row in native["compartments"]}
        edges = {row["id"]: row for row in native["connections"]}
        self.assertEqual(nodes["LV"]["source_pi"], 3.14159)
        self.assertEqual(nodes["LA"]["source_pi"], 3.14159)
        self.assertNotEqual(nodes["LA"]["source_pi"], math.pi)
        self.assertEqual((nodes["LA"]["activation_start"], nodes["LA"]["activation_end"]), (0.92, 0.09))
        self.assertEqual((nodes["LV"]["activation_start"], nodes["LV"]["activation_end"]), (0.3, 0.45))
        self.assertAlmostEqual(edges["RV_outflow"]["orifice_coefficient_m3_per_s_sqrt_pa"], 350e-6 / math.sqrt(133), delta=1e-19)

    def test_source_dynamic_initial_volume_is_not_reference_volume(self) -> None:
        native, _ = compile_source(config=self.config)
        nodes = {row["id"]: row for row in native["compartments"]}
        self.assertEqual(nodes["LV"]["initial_volume_m3"], 0.0005)
        self.assertAlmostEqual(nodes["LV"]["reference_volume_m3"], 5e-6, delta=1e-20)
        self.assertEqual(nodes["RV"]["initial_volume_m3"], 0.0005)
        self.assertAlmostEqual(nodes["RV"]["reference_volume_m3"], 10e-6, delta=1e-20)
        self.assertEqual(nodes["LV"]["reference_pressure_pa"], 133)

    def test_vascular_storage_has_no_fabricated_absolute_volume(self) -> None:
        native, _ = compile_source(config=self.config)
        storage = [row for row in native["compartments"] if row["storage_kind"] == "storage_displacement"]
        self.assertEqual(len(storage), 6)
        for row in storage:
            self.assertEqual(row["reference_volume_m3"], 0)
            self.assertEqual(row["reference_pressure_pa"], 0)
            self.assertEqual(row["pressure_law"], "linear_compliance")
            self.assertEqual(row["initial_species_mol"], [])
            self.assertEqual(row["source_pi"], 0)
        veins = [row for row in storage if row["id"] in {"Pvn", "Svn"}]
        self.assertEqual([row["initial_volume_m3"] for row in veins], [0, 0])
        self.assertEqual(native["species"], [])
        self.assertEqual(native["tissue_reservoirs"], [])
        self.assertEqual(native["exchanges"], [])
        self.assertTrue(all(row["anatomical_region_id"].startswith("CellML:shi_hose_2009:") for row in native["compartments"]))

    def test_connected_source_cycle_and_exact_zero_storage_elimination(self) -> None:
        native, manifest = compile_source(config=self.config)
        self.assertEqual(len(native["compartments"]), 10)
        self.assertEqual(len(native["connections"]), 10)
        self.assertEqual(len(manifest["source_connections"]), 14)
        self.assertEqual([row["stable_identifier"] for row in native["compartments"]], list(range(1, 11)))
        self.assertEqual([row["stable_identifier"] for row in native["connections"]], list(range(11, 21)))
        outgoing = {row["from"]: row["to"] for row in native["connections"]}
        visited, cursor = set(), 1
        while cursor not in visited:
            visited.add(cursor)
            cursor = outgoing[cursor]
        self.assertEqual(visited, set(range(1, 11)))
        edges = {row["id"]: row for row in native["connections"]}
        self.assertEqual(edges["Sat_outflow"]["resistance_pa_s_per_m3"], 142310000)
        self.assertEqual(edges["Pat_outflow"]["resistance_pa_s_per_m3"], 41230000)
        derivations = {row["id"]: row for row in manifest["derivations"]}
        self.assertEqual(derivations["Sat_outflow"]["source_components"], ["Sat", "Sar", "Scp"])
        self.assertEqual(derivations["Pat_outflow"]["source_components"], ["Pat", "Par", "Pcp"])

    def test_algebraic_initial_flows_preserve_reverse_venous_flow_and_open_pulmonary_valve(self) -> None:
        native, _ = compile_source(config=self.config)
        edges = {row["id"]: row for row in native["connections"]}
        self.assertAlmostEqual(edges["Pvn_outflow"]["initial_flow_m3_per_s"], -0.005978611452314101, delta=2e-18)
        self.assertAlmostEqual(edges["Svn_outflow"]["initial_flow_m3_per_s"], -4.7828891618512803e-5, delta=2e-20)
        self.assertAlmostEqual(edges["RV_outflow"]["initial_flow_m3_per_s"], 0.0015652475842498528, delta=1e-18)
        for name in ("Pas", "Pat", "Sas", "Sat"):
            self.assertEqual(edges[name + "_outflow"]["initial_flow_m3_per_s"], 0)
            self.assertGreater(edges[name + "_outflow"]["inertance_pa_s2_per_m3"], 0)

    def test_config_key_order_is_irrelevant(self) -> None:
        first = compile_source(config=self.config)
        reordered = json.loads(json.dumps(self.config, sort_keys=True))
        second = compile_source(config=reordered)
        self.assertEqual(tuple(map(canonical, first)), tuple(map(canonical, second)))

    def test_numerical_settings_change_identity_without_overriding_source_parameters(self) -> None:
        before, source_before = compile_source(config=self.config)
        self.config["numerical_settings"]["residual_tolerance"]["value"] = 2e-5
        after, source_after = compile_source(config=self.config)
        self.assertNotEqual(before["authored_graph_sha256"], after["authored_graph_sha256"])
        self.assertEqual(before["source_graph_sha256"], after["source_graph_sha256"])
        self.assertEqual(source_before["parameters"], source_after["parameters"])
        for kind in ("compartments", "connections"):
            left = [{k: v for k, v in row.items() if "tolerance" not in k} for row in before[kind]]
            right = [{k: v for k, v in row.items() if "tolerance" not in k} for row in after[kind]]
            self.assertEqual(left, right)

    def test_transport_is_rejected_without_absolute_vascular_baselines(self) -> None:
        for field in ("species", "tissue_reservoirs", "exchanges"):
            with self.subTest(field=field):
                config = copy.deepcopy(self.config)
                config[field] = [{"id": "oxygen"}]
                with self.assertRaisesRegex(HumanImportError, "absolute vascular volumes"):
                    compile_source(config=config)

    def test_physiological_override_and_baseline_invention_rejected(self) -> None:
        for key in ("reference_volume_m3", "pressure_unit_multiplier", "heart_rate", "FMA_region_override"):
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config[key] = 1
                with self.assertRaisesRegex(HumanImportError, "physiological overrides"):
                    compile_source(config=config)

    def test_invalid_numerical_settings_rejected(self) -> None:
        for value in (0, -1, True, float("nan"), float("inf"), "1e-5"):
            with self.subTest(value=value):
                config = copy.deepcopy(self.config)
                config["numerical_settings"]["residual_tolerance"]["value"] = value
                with self.assertRaises(HumanImportError):
                    compile_source(config=config)

    def test_source_mutation_and_missing_import_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "source"
            shutil.copytree(SOURCE, copy)
            (copy / "Units.cellml").write_bytes((copy / "Units.cellml").read_bytes().replace(b'"133"', b'"133.322"'))
            with self.assertRaisesRegex(HumanImportError, "source hash mismatch"):
                compile_source(directory=copy, config=self.config)
            shutil.copy2(SOURCE / "Units.cellml", copy / "Units.cellml")
            (copy / "TempRLC.cellml").unlink()
            with self.assertRaisesRegex(HumanImportError, "missing/redirected curated source"):
                compile_source(directory=copy, config=self.config)

    def test_resealed_forged_source_manifest_cannot_redefine_the_curated_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "source"
            shutil.copytree(SOURCE, copy)
            lock = read_json(copy / "source-lock.json")
            lock["revision"] = "0" * 40
            (copy / "source-lock.json").write_bytes(canonical(lock))
            with self.assertRaisesRegex(HumanImportError, "curated revision"):
                compile_source(directory=copy, config=self.config)

    def test_source_units_reject_cycles_and_unsupported_offsets(self) -> None:
        for xml in ('<units name="a"><unit units="b"/></units><units name="b"><unit units="a"/></units>',
                    '<units name="a"><unit units="pascal" offset="10"/></units>'):
            model = ET.fromstring('<model xmlns="http://www.cellml.org/cellml/1.1#">' + xml + '</model>')
            with self.assertRaises(HumanImportError):
                _units(model)

    def test_duplicate_source_configuration_fields_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text('{"schema":"one","sch\\u0065ma":"two"}')
            with self.assertRaisesRegex(HumanImportError, "duplicate"):
                read_json(path)

    def test_cli_writes_distinct_immutable_native_payload_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "model.json"
            self.assertEqual(main(["--output", str(output)]), 0)
            self.assertTrue(output.with_suffix(".manifest.json").is_file())
            self.assertEqual(main(["--output", str(output)]), 0)
            original = output.read_bytes()
            modified = directory / "modified-config.json"
            self.config["numerical_settings"]["residual_tolerance"]["value"] = 2e-5
            modified.write_bytes(canonical(self.config))
            self.assertEqual(main(["--config", str(modified), "--output", str(output)]), 1)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(main(["--output", str(directory / "same.json"), "--manifest", str(directory / "same.json")]), 1)

    def test_numi_overlay_is_executable_and_resolves_module_outside_repository(self) -> None:
        overlay = SOURCE.parents[2] / ".numi/commands/human-cardiac"
        self.assertTrue(os.access(overlay, os.X_OK))
        with tempfile.TemporaryDirectory() as temporary:
            env = dict(os.environ, NUMI_HUMAN_PYTHON=sys.executable)
            result = subprocess.run([str(overlay), "--help"], cwd=temporary, env=env, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--source-directory", result.stdout)
            described = subprocess.run([str(overlay), "--numi-describe"], cwd=temporary, env=env, text=True, capture_output=True, check=False)
            self.assertEqual(described.returncode, 0, described.stderr)
            self.assertIn("Shi/Hose", described.stdout)


if __name__ == "__main__":
    unittest.main()
