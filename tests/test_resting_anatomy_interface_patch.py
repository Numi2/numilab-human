"""Focused source-bound test for the resting diaphragm/lung interface compiler."""
from __future__ import annotations

import json
import math
from pathlib import Path
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from numilab_human import resting_anatomy
from numilab_human.resting_anatomy_interface_patch import (
    BASE,
    LOBE_IDS,
    parse_payload,
    sha,
    topology_report,
)


EXPECTED_NHANATOMY_SHA256 = "82bec56c17d71a7075a694e81c5fb0d310b7190125f9d4c6c8603e6d6fda51e7"
EXPECTED_DIAPHRAGM_AREA_M2 = 0.019892791918944348


class RestingAnatomyInterfacePatchTests(unittest.TestCase):
    def test_owner_cli_builds_pinned_reciprocal_diaphragm_lung_interface(self):
        """Exercise the owner entry point against the retained exact source inputs."""
        base_payload = BASE / "resting-thorax.nhanatomy"
        base_receipt = BASE / "resting-anatomy-receipt.json"
        required = (base_payload, base_receipt, BASE / "resting-anatomy-manifest.json")
        if not all(path.is_file() for path in required):
            self.skipTest("pinned Mini anatomy source package is not installed")

        with tempfile.TemporaryDirectory(prefix="resting-anatomy-interface-") as temp:
            output = Path(temp) / "resting-anatomy-interface"
            stdout = io.StringIO()
            with patch.object(sys, "argv", [
                "numilab-human-resting-anatomy",
                "--output", str(output),
                "--patch-diaphragm-lung-interfaces",
                "--base-payload", str(base_payload),
                "--base-receipt", str(base_receipt),
            ]), redirect_stdout(stdout):
                resting_anatomy.main()
            result = json.loads(stdout.getvalue())

            payload = output / "resting-thorax.nhanatomy"
            receipt_path = output / "resting-anatomy-receipt.json"
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(sha(payload), EXPECTED_NHANATOMY_SHA256)
            self.assertEqual(result["payload"]["sha256"], EXPECTED_NHANATOMY_SHA256)
            self.assertEqual(result["preserved_non_target_surface_count"], 457)
            self.assertEqual(receipt["functional_bindings"]["anatomy_payload_sha256"], EXPECTED_NHANATOMY_SHA256)
            geometry = receipt["functional_bindings"]["respiratory_geometry_binding"]
            self.assertEqual(geometry["lung_motion_model"], "basal_superior_sweep_v1")
            self.assertTrue(math.isclose(
                geometry["diaphragm_effective_area_m2"], EXPECTED_DIAPHRAGM_AREA_M2,
                rel_tol=1e-12, abs_tol=0.0,
            ))

            _, rows = parse_payload(payload)
            self.assertTrue(set(LOBE_IDS).issubset(rows))
            for stable_id in LOBE_IDS:
                topology = topology_report(rows[stable_id]["faces"])
                self.assertEqual(topology["boundary_edge_count"], 0)
                self.assertEqual(topology["nonmanifold_edge_count"], 0)
                self.assertEqual(topology["orientation_error_edge_count"], 0)

            diaphragm_topology = topology_report(rows[311]["faces"])
            self.assertEqual(diaphragm_topology["boundary_edge_count"], 10)
            self.assertEqual(diaphragm_topology["boundary_loop_count"], 1)
            self.assertEqual(diaphragm_topology["boundary_branch_vertex_count"], 0)
            self.assertEqual(diaphragm_topology["nonmanifold_edge_count"], 0)
            self.assertEqual(diaphragm_topology["orientation_error_edge_count"], 0)
            self.assertEqual(
                receipt["provenance"]["source_id_map"]["311"]["repair"][
                    "natural_source_aperture_identity"
                ],
                "not assigned by source file",
            )


if __name__ == "__main__":
    unittest.main()
