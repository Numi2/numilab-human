from __future__ import annotations

import copy
import argparse
import json
import tempfile
import unittest
from pathlib import Path

from numilab_human.model import ImportError as HumanImportError
from numilab_human.physiology import ROOT, _run, canonical, compile_graph, read_json, validate_graph


class PhysiologyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = read_json(ROOT / "config/physiology-passive-fixture.v1.json")

    def compile(self, graph: dict | None = None) -> dict:
        return compile_graph(self.graph if graph is None else graph, sources=ROOT / "Sources")

    def rejected(self, message: str) -> None:
        with self.assertRaisesRegex(HumanImportError, message):
            self.compile()

    def test_native_plan_has_deterministic_global_ids_and_conservation_endpoints(self) -> None:
        native = self.compile()
        self.assertEqual(native["qualification"], "fixture_only")
        self.assertEqual([s["stable_identifier"] for s in native["species"]], [1])
        self.assertEqual([c["stable_identifier"] for c in native["compartments"]], [2, 3])
        self.assertEqual(native["compartments"][0]["anatomical_region_id"], "FMA:7204")
        self.assertEqual((native["connections"][0]["from"], native["connections"][0]["to"]), (2, 3))
        exchange = native["exchanges"][0]
        self.assertEqual((exchange["compartment"], exchange["tissue_reservoir"], exchange["species"]), (3, 5, 1))
        self.assertEqual(native["connections"][0]["flow_residual_tolerance"], native["residual_tolerance"])
        self.assertEqual(native["compartments"][0]["volume_residual_tolerance"], native["residual_tolerance"])
        self.assertEqual(native["species"][0]["amount_residual_tolerance"], native["residual_tolerance"])

    def test_reordered_authoring_has_identical_native_bytes(self) -> None:
        expected = canonical(self.compile())
        for name in ("regions", "species", "parameter_sources", "compartments", "connections", "tissue_reservoirs", "exchanges"):
            self.graph[name].reverse()
        for region in self.graph["regions"]:
            region["member_ids"].reverse()
        self.assertEqual(canonical(self.compile()), expected)

    def test_parameter_provenance_changes_compatibility_identity(self) -> None:
        before = self.compile()
        self.graph["connections"][0]["resistance_pa_s_per_m3"]["provenance"]["description"] += " Different derivation."
        after = self.compile()
        self.assertNotEqual(before["authored_graph_sha256"], after["authored_graph_sha256"])
        self.assertEqual(before["connections"], after["connections"])

    def test_complete_actual_anatomy_template_retains_missing_parameters_and_overlaps(self) -> None:
        graph = read_json(ROOT / "config/physiology-organ-network-template.v1.json")
        result = validate_graph(graph, sources=ROOT / "Sources")
        self.assertEqual(len(graph["regions"]), 18)
        self.assertTrue(result["calibration_required"])
        self.assertGreater(len(result["missing_parameters"]), 200)
        self.assertTrue(result["overlapping_source_memberships"])
        self.assertIn("FMA:7101", {r["semantic_id"] for r in graph["regions"]})
        with self.assertRaisesRegex(HumanImportError, "requires resolved parameters"):
            self.compile(graph)

    def test_duplicate_physical_volume_owner_rejected_across_blood_and_tissue(self) -> None:
        self.graph["tissue_reservoirs"][0]["physical_volume_owner_id"] = self.graph["compartments"][0]["physical_volume_owner_id"]
        self.rejected("duplicate physical volume owner")

    def test_duplicate_record_ids_rejected_for_every_table(self) -> None:
        for table in ("regions", "species", "compartments", "connections", "tissue_reservoirs", "exchanges"):
            with self.subTest(table=table):
                graph = copy.deepcopy(self.graph)
                graph[table].append(copy.deepcopy(graph[table][0]))
                with self.assertRaisesRegex(HumanImportError, "duplicate .* ID"):
                    self.compile(graph)

    def test_forged_source_member_rejected(self) -> None:
        self.graph["regions"][0]["member_ids"] = ["FJ000000"]
        self.rejected("forged source membership")

    def test_forged_source_concept_rejected(self) -> None:
        self.graph["regions"][0]["concept_id"] = "FMA7088"
        self.graph["regions"][0]["semantic_id"] = "FMA:7088"
        self.rejected("forged source membership")

    def test_forged_source_hash_rejected(self) -> None:
        self.graph["anatomy_source"]["tables"][0]["sha256"] = "0" * 64
        self.rejected("forged or stale anatomy source")

    def test_missing_source_table_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(HumanImportError, "missing anatomy source"):
                compile_graph(self.graph, sources=Path(temporary))

    def test_changed_source_bytes_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "partof_element_parts.txt").write_text("FMA7204\tright kidney\tFJ0000\n")
            with self.assertRaisesRegex(HumanImportError, "source hash mismatch"):
                compile_graph(self.graph, sources=path)

    def test_unresolved_region_rejected(self) -> None:
        self.graph["compartments"][0]["anatomical_region_id"] = "absent"
        self.rejected("unresolved anatomical region")

    def test_duplicate_source_members_rejected(self) -> None:
        members = self.graph["regions"][0]["member_ids"]
        members.append(members[0])
        self.rejected("duplicate anatomy member")

    def test_nonfinite_and_boolean_parameters_rejected(self) -> None:
        for value in (float("nan"), float("inf"), -float("inf"), True, "1e-6"):
            with self.subTest(value=value):
                self.graph["compartments"][0]["initial_volume_m3"]["value"] = value
                self.rejected("finite|numeric")

    def test_wrong_units_rejected(self) -> None:
        self.graph["compartments"][0]["initial_volume_m3"]["unit"] = "mL"
        self.rejected("SI unit m3")

    def test_negative_resistance_and_amount_rejected(self) -> None:
        self.graph["connections"][0]["resistance_pa_s_per_m3"]["value"] = -1
        self.rejected("sign")
        self.setUp()
        self.graph["compartments"][0]["initial_species_mol"]["tracer"]["value"] = -1
        self.rejected("sign")

    def test_unknown_parameter_source_rejected(self) -> None:
        self.graph["compartments"][0]["initial_volume_m3"]["provenance"] = {"kind": "source", "source_id": "missing", "record_id": "table1"}
        self.rejected("unresolved .* parameter source")

    def test_missing_uncertainty_rejected(self) -> None:
        del self.graph["compartments"][0]["initial_volume_m3"]["uncertainty"]
        self.rejected("fields")

    def test_fixture_cannot_be_relabelled_biological_or_uncalibrated(self) -> None:
        self.graph["qualification"] = "qualified"
        self.rejected("cannot promote")
        self.graph["qualification"] = "uncalibrated"
        self.rejected("fixture_only")

    def test_disconnected_compartment_rejected(self) -> None:
        extra = copy.deepcopy(self.graph["compartments"][0])
        extra.update(id="isolated", physical_volume_owner_id="blood:isolated")
        self.graph["compartments"].append(extra)
        self.rejected("disconnected hydraulic topology")

    def test_missing_hydraulic_endpoint_rejected(self) -> None:
        self.graph["connections"][0]["to"] = "outside"
        self.rejected("conservation endpoints")

    def test_duplicate_hydraulic_edge_rejected(self) -> None:
        duplicate = copy.deepcopy(self.graph["connections"][0])
        duplicate.update(id="reverse", **{"from": duplicate["to"], "to": duplicate["from"]})
        self.graph["connections"].append(duplicate)
        self.rejected("duplicate hydraulic connection")

    def test_missing_exchange_conservation_endpoint_rejected(self) -> None:
        for field in ("compartment", "tissue_reservoir", "species"):
            with self.subTest(field=field):
                graph = copy.deepcopy(self.graph)
                graph["exchanges"][0][field] = "missing"
                with self.assertRaisesRegex(HumanImportError, "conservation endpoints"):
                    self.compile(graph)

    def test_unexchanged_tissue_rejected(self) -> None:
        self.graph["exchanges"] = []
        self.rejected("no conservative exchange")

    def test_missing_species_amount_rejected(self) -> None:
        self.graph["tissue_reservoirs"][0]["initial_species_mol"] = {}
        self.rejected("every species exactly")

    def test_unknown_fields_and_duplicate_json_keys_rejected(self) -> None:
        self.graph["compartments"][0]["hidden_inflow_m3_per_s"] = 1
        self.rejected("fields")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            for payload in ('{"schema":"one","schema":"two"}',
                            '{"schema":"one","sch\\u0065ma":"two"}',
                            '{"anatomy_source":{"tables":[{"sha256":"one","sha256":"two"}]}}',
                            '{"sources":{"bodyparts3d_4":{"version":"4.0","version":"forged"}}}'):
                with self.subTest(payload=payload):
                    path.write_text(payload)
                    with self.assertRaisesRegex(HumanImportError, "duplicate JSON key"):
                        read_json(path)

    def test_malformed_structural_types_are_typed_rejections(self) -> None:
        for path, value in ((["qualification"], []), (["regions", 0, "hierarchy"], []),
                            (["regions", 0, "concept_id"], {}),
                            (["compartments", 0, "anatomical_region_id"], []),
                            (["compartments", 0, "initial_volume_m3", "uncertainty", "status"], [])):
            with self.subTest(path=path):
                graph = copy.deepcopy(self.graph)
                current = graph
                for field in path[:-1]:
                    current = current[field]
                current[path[-1]] = value
                with self.assertRaises(HumanImportError):
                    self.compile(graph)

    def test_cli_output_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            graph = directory / "graph.json"
            graph.write_bytes(canonical(self.graph))
            args = argparse.Namespace(graph=graph, sources=ROOT / "Sources", source_lock=ROOT / "sources.lock.json",
                                      output=directory / "native.json", validate_only=False)
            self.assertEqual(_run(args), 0)
            before = args.output.read_bytes()
            self.graph["id"] = "new_identity"
            graph.write_bytes(canonical(self.graph))
            with self.assertRaisesRegex(HumanImportError, "immutable"):
                _run(args)
            self.assertEqual(args.output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
