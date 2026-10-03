from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from numilab_human import gap_execution
from numilab_human.gap_reference_audit import ROOT, _links_in_markdown, command, materialize


class GapReferenceAuditTests(unittest.TestCase):
    def test_inline_markdown_links_images_directories_and_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs = root / "Docs"
            docs.mkdir()
            (docs / "target.md").write_text("# Evidence title\n", encoding="utf-8")
            (docs / "figure.png").write_bytes(b"image fixture")
            (docs / "media").mkdir()
            source = docs / "source.md"
            source.write_text(
                "# Source\n"
                "[target](target.md#evidence-title) ![figure](figure.png) "
                "[folder](media/) [external](https://example.org/x)\n"
                "`[inline code](missing.md)`\n"
                "```md\n[example](also-missing.md)\n```\n",
                encoding="utf-8",
            )

            links, external = _links_in_markdown(source, root)

            self.assertEqual([row["status"] for row in links], ["resolved"] * 3)
            self.assertEqual([row["target_type"] for row in links], ["file", "file", "directory"])
            self.assertEqual(links[1]["kind"], "image")
            self.assertEqual(len(external), 1)

    def test_missing_local_target_and_anchor_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.md"
            source.write_text("[missing](gone.md) [bad anchor](source.md#absent)\n", encoding="utf-8")

            links, external = _links_in_markdown(source, root)

            self.assertEqual([row["status"] for row in links],
                             ["missing_or_outside_repository", "missing_anchor"])
            self.assertEqual(external, [])

    def test_full_live_registry_reference_graph_resolves(self) -> None:
        registry_path = ROOT / "config/human-gap-execution.v1.json"
        registry = gap_execution.read_json(registry_path)

        report = materialize(registry, root=ROOT, registry_path=registry_path)

        self.assertEqual(report["status"], "passed_local_reference_graph")
        self.assertEqual(report["counts"]["unresolved_local_targets"], 0)
        self.assertGreater(report["counts"]["registered_markdown_documents"], 100)
        self.assertGreater(report["counts"]["json_references_parsed"], 100)
        self.assertGreater(report["counts"]["external_links_not_fetched"], 0)
        self.assertEqual(report["audit_sha256"], gap_execution.digest(
            {key: value for key, value in report.items() if key != "audit_sha256"}
        ))

    def test_external_registry_is_rejected(self) -> None:
        registry_path = ROOT / "config/human-gap-execution.v1.json"
        registry = gap_execution.read_json(registry_path)
        with self.assertRaisesRegex(gap_execution.HumanImportError, "inside the repository"):
            materialize(registry, root=ROOT, registry_path=Path("/tmp/registry.json"))

    def test_owner_cli_registers_reference_audit(self) -> None:
        from numilab_human.cli import parser

        parsed = parser().parse_args(["gap-reference-audit"])

        self.assertIs(parsed.handler, command)


if __name__ == "__main__":
    unittest.main()
