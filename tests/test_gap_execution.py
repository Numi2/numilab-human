from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from numilab_human.gap_execution import (
    ROOT, command, dependency_layers, ledger_rows, materialize, read_json, validate_registry,
)
from numilab_human.model import ImportError as HumanImportError
from numilab_human.target_coverage import MANDATORY, canonical_bytes, digest, materialize as coverage_materialize


class GapExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.registry = read_json(ROOT / "config/human-gap-execution.v1.json")
        for path in {self.registry["ledger"], *(p for row in self.registry["workstreams"] for p in row["references"])}:
            destination = self.root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / path).read_bytes())

    def coverage(self, *, source: bool = False) -> dict:
        entries = {}
        if source:
            model = self.root / "model.osim"
            model.write_text('<Model name="fixture"><NovelTissue name="unassigned"/></Model>')
            entries["fixture"] = {"model_file": model.name,
                                  "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                                  "revision": "fixture-only", "license": "fixture-only"}
            entries["missing"] = {"model_file": "missing.osim", "sha256": "0" * 64,
                                  "revision": "fixture-only", "license": "fixture-only"}
        lock = self.root / "sources.lock.json"
        lock.write_text(json.dumps({"schema": "numi.human.source-lock.v1", "sources": entries}))
        return coverage_materialize(sources=self.root, source_lock=lock, repository_root=self.root,
                                    supplements=False)

    def test_live_ledger_and_entire_mandatory_catalog_are_accounted_for(self) -> None:
        report = materialize(self.registry, root=self.root)
        self.assertEqual(report["counts"]["workstreams"], len(ledger_rows(self.root / self.registry["ledger"])))
        self.assertEqual(report["counts"]["mandatory_targets"], sum(map(len, MANDATORY.values())))
        for row in report["workstreams"]:
            self.assertEqual(row["ledger"], ledger_rows(self.root / self.registry["ledger"])[row["ledger_workstream"]])
            self.assertEqual(row["task_assessment"], "not_assessed")
        self.assertEqual(report["integrated_qualification"], "not_assessed")
        self.assertEqual(report["evidence_reference_validation"], "not_performed")
        self.assertEqual(report["coverage"]["binding_status"], "not_supplied")
        self.assertIsNone(report["counts"]["unmapped_source_leaves"])

    def test_stable_content_and_independent_frontier(self) -> None:
        first = materialize(self.registry, root=self.root)
        self.assertEqual(canonical_bytes(first), canonical_bytes(materialize(self.registry, root=self.root)))
        self.assertEqual(first["report_sha256"], digest({k: v for k, v in first.items() if k != "report_sha256"}))
        self.assertTrue({"runtime.precision", "behavior.telemetry", "source.acquire", "calibration.acquire",
                         "performance.baseline"} <= set(first["dependency_layers"][0]))
        indices = {task: index for index, layer in enumerate(first["dependency_layers"]) for task in layer}
        for row in self.registry["workstreams"]:
            for task in row["tasks"]:
                self.assertTrue(all(indices[parent] < indices[task["id"]] for parent in task["depends_on"]))

    def test_missing_or_added_ledger_row_is_rejected(self) -> None:
        missing = copy.deepcopy(self.registry)
        missing["workstreams"].pop()
        with self.assertRaisesRegex(HumanImportError, "every ledger"):
            validate_registry(missing, root=self.root)
        path = self.root / self.registry["ledger"]
        path.write_text(path.read_text().replace("| Source foundation |", "| New source scope |", 1))
        with self.assertRaisesRegex(HumanImportError, "every ledger"):
            materialize(self.registry, root=self.root)

    def test_target_omission_and_unknown_target_are_rejected(self) -> None:
        missing = copy.deepcopy(self.registry)
        target = "mandatory:biological_sensing/visual"
        for row in missing["workstreams"]:
            row["target_ids"] = [item for item in row["target_ids"] if item != target]
        with self.assertRaisesRegex(HumanImportError, "omits mandatory"):
            validate_registry(missing, root=self.root)
        bad = copy.deepcopy(self.registry)
        bad["workstreams"][0]["target_ids"].append("mandatory:invented/success")
        with self.assertRaisesRegex(HumanImportError, "unknown mandatory"):
            validate_registry(bad, root=self.root)

    def test_cycle_unknown_parent_duplicate_identity_and_bad_category_are_rejected(self) -> None:
        for mutation, message in (("cycle", "cycle"), ("unknown", "unknown prerequisite"),
                                  ("duplicate", "duplicate execution task"), ("category", "category")):
            with self.subTest(mutation=mutation):
                value = copy.deepcopy(self.registry)
                tasks = value["workstreams"][0]["tasks"]
                if mutation == "cycle":
                    tasks[0]["depends_on"] = [tasks[1]["id"]]
                elif mutation == "unknown":
                    tasks[0]["depends_on"] = ["absent"]
                elif mutation == "duplicate":
                    tasks[1]["id"] = tasks[0]["id"]
                else:
                    tasks[0]["category"] = "passed"
                with self.assertRaisesRegex(HumanImportError, message):
                    validate_registry(value, root=self.root)

    def test_metadata_cannot_inject_status_or_qualification(self) -> None:
        for location in ("registry", "workstream", "task"):
            value = copy.deepcopy(self.registry)
            target = value if location == "registry" else value["workstreams"][0]
            if location == "task":
                target = target["tasks"][0]
            target["status"] = "proved"
            with self.assertRaisesRegex(HumanImportError, "fields differ"):
                materialize(value, root=self.root)

    def test_live_status_is_quoted_without_promoting_tasks(self) -> None:
        path = self.root / self.registry["ledger"]
        path.write_text(path.read_text().replace("| partial |", "| proved |", 1))
        report = materialize(self.registry, root=self.root)
        self.assertEqual(report["workstreams"][0]["ledger"]["status"], "proved")
        self.assertEqual(report["workstreams"][0]["task_assessment"], "not_assessed")
        self.assertEqual(report["integrated_qualification"], "not_assessed")

    def test_source_leaves_and_missing_registers_are_retained_unassigned(self) -> None:
        coverage = self.coverage(source=True)
        before = canonical_bytes(coverage)
        report = materialize(self.registry, root=self.root, coverage=coverage)
        self.assertEqual(before, canonical_bytes(coverage))
        self.assertEqual(report["coverage"]["manifest_sha256"], coverage["manifest_sha256"])
        expected = {leaf["leaf_sha256"] for leaf in coverage["leaves"] if leaf["kind"] != "mandatory_target"}
        self.assertEqual(expected, {leaf["leaf_sha256"] for leaf in report["coverage"]["unmapped_source_leaves"]})
        self.assertTrue(expected)
        self.assertEqual(report["coverage"]["source_assignment_status"], "unmapped_source_leaves")
        self.assertTrue(report["coverage"]["unresolved_current_registers"])
        self.assertTrue(all(link["leaf_sha256"] for link in report["target_links"]))

    def test_forged_or_historical_coverage_is_rejected_by_existing_owner(self) -> None:
        value = self.coverage()
        value["integrated_qualification"] = "proved"
        value["manifest_sha256"] = digest({k: v for k, v in value.items() if k != "manifest_sha256"})
        with self.assertRaisesRegex(HumanImportError, "promote"):
            materialize(self.registry, root=self.root, coverage=value)
        value = self.coverage()
        value["compiler"] = "numilab-human.target-coverage.1"
        with self.assertRaisesRegex(HumanImportError, "historical"):
            materialize(self.registry, root=self.root, coverage=value)

    def test_missing_external_and_changed_references_are_not_hidden(self) -> None:
        for path in ("Docs/absent.md", "../outside.md", "/tmp/outside.md"):
            value = copy.deepcopy(self.registry)
            value["workstreams"][0]["references"] = [path]
            with self.assertRaises(HumanImportError):
                materialize(value, root=self.root)
        before = materialize(self.registry, root=self.root)
        path = self.root / "Docs/DEVELOPMENT_ROADMAP.md"
        path.write_text(path.read_text() + "\nchanged fixture reference\n")
        after = materialize(self.registry, root=self.root)
        self.assertNotEqual(before["report_sha256"], after["report_sha256"])
        self.assertEqual(after["evidence_reference_validation"], "not_performed")

    def test_immutable_output_and_existing_cli_registration(self) -> None:
        registry = self.root / "registry.json"
        registry.write_text(json.dumps(self.registry))
        output = self.root / "report.json"
        args = argparse.Namespace(registry=registry, repository_root=self.root, coverage=None, output=output)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(command(args), 0)
            self.assertEqual(command(args), 0)
        output.write_text('{"forged":true}')
        with self.assertRaisesRegex(HumanImportError, "immutable"):
            command(args)
        self.assertEqual(output.read_text(), '{"forged":true}')
        from numilab_human.cli import parser
        parsed = parser().parse_args(["gap-execution", "--registry", str(registry)])
        self.assertIs(parsed.handler, command)

    def test_reference_change_during_materialization_is_rejected(self) -> None:
        from numilab_human import gap_execution
        actual = gap_execution.file_digest
        calls = {}
        def changing(path):
            calls[path] = calls.get(path, 0) + 1
            return actual(path) if calls[path] == 1 else "0" * 64
        with patch.object(gap_execution, "file_digest", side_effect=changing):
            with self.assertRaisesRegex(HumanImportError, "changed"):
                materialize(self.registry, root=self.root)


if __name__ == "__main__":
    unittest.main()
