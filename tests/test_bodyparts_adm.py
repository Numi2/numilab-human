from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from numilab_human.bodyparts_adm import (
    MEMBERS, SCHEMA, compile_adm_inference, compile_adm_payload,
)
from numilab_human.model import ImportError as HumanImportError


def _obj(vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]) -> bytes:
    lines = [f"v {x} {y} {z}" for x, y, z in vertices]
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    return ("\n".join(lines) + "\n").encode()


class BodyPartsAdmTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, str]:
        source = root / "Sources"
        source.mkdir()
        members: dict[str, bytes] = {}
        for side, sign in (("right", -1.0), ("left", 1.0)):
            ids = MEMBERS[side]
            muscle_vertices = []
            for y in (0.5, 49.5):
                for ordinal in range(12):
                    muscle_vertices.append(
                        (sign * (10.0 + 0.1 * ordinal), y, -2.0 + 0.35 * ordinal)
                    )
            members[ids["muscle"]] = _obj(
                muscle_vertices,
                [(0, 1, 2), (9, 10, 11), (12, 13, 14), (21, 22, 23)],
            )
            for role, y in (("origin_bone", 0.0), ("insertion_bone", 50.0)):
                members[ids[role]] = _obj(
                    [
                        (sign * 20.0, y, -10.0),
                        (0.0, y, -10.0),
                        (0.0, y, 10.0),
                        (sign * 20.0, y, 10.0),
                    ],
                    [(0, 1, 2), (0, 2, 3)],
                )
        archive = source / "isa_BP3D_4.0_obj_99.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            for member_id, value in sorted(members.items()):
                stream.writestr(f"fixture/{member_id}.obj", value)
        archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
        anchors = []
        for side_index, side in enumerate(("right", "left")):
            for endpoint_index, role in enumerate(("origin_bone", "insertion_bone")):
                member_id = MEMBERS[side][role]
                anchors.append({
                    "source": {
                        "member_id": member_id,
                        "member_sha256": hashlib.sha256(members[member_id]).hexdigest(),
                    },
                    "target": {
                        "core_body_index": side_index * 10 + endpoint_index,
                        "name": f"{role}_{side}",
                    },
                    "registration": {
                        "status": "fixture_registration",
                        "source_obj_mm_to_core_inertial_body_m": [
                            [0.001, 0.0, 0.0, 0.0],
                            [0.0, 0.001, 0.0, 0.0],
                            [0.0, 0.0, 0.001, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        "attachment_surface_refinement": {"applied": True},
                    },
                })
        receipt = root / "registration.json"
        receipt.write_text(json.dumps({"anchors": anchors}))
        return source, receipt, archive_hash

    def test_compiles_replayable_bilateral_nonvisual_witness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, receipt, archive_hash = self._fixture(Path(directory))
            with patch("numilab_human.bodyparts_adm.EXPECTED_ARCHIVE_SHA256", archive_hash):
                artifact = compile_adm_inference(source, receipt)
                replay = compile_adm_inference(source, receipt)
        self.assertEqual(artifact, replay)
        self.assertEqual(artifact["schema"], SCHEMA)
        self.assertEqual(len(artifact["hands"]), 2)
        self.assertTrue(artifact["bilateral_validation"]["passed"])
        self.assertLessEqual(artifact["bilateral_validation"]["maximum_residual_mm"], 5.0)
        for hand in artifact["hands"]:
            self.assertEqual(hand["endpoint_cluster_overlap_count"], 0)
            self.assertGreater(hand["source_endpoint_separation_mm"], 20.0)
            self.assertEqual(hand["origin"]["cluster"]["selected_vertex_count"], 8)
            self.assertEqual(hand["insertion"]["cluster"]["selected_vertex_count"], 8)
        self.assertAlmostEqual(
            artifact["force_capacity_sensitivity"]["maximum_isometric_force_n"]["nominal"],
            29.748,
            places=12,
        )
        payload = compile_adm_payload(artifact)
        self.assertEqual(len(payload), 52 + 2 * 64)
        self.assertEqual(payload[:8], b"NHADM1\0\0")

    def test_fails_closed_on_registration_member_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, receipt, archive_hash = self._fixture(Path(directory))
            value = json.loads(receipt.read_text())
            value["anchors"][0]["source"]["member_sha256"] = "00" * 32
            receipt.write_text(json.dumps(value))
            with patch("numilab_human.bodyparts_adm.EXPECTED_ARCHIVE_SHA256", archive_hash):
                with self.assertRaisesRegex(HumanImportError, "member hash drifted"):
                    compile_adm_inference(source, receipt)


if __name__ == "__main__":
    unittest.main()
