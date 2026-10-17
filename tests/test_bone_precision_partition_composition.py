import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from numilab_human import composed_bone_body_frame as owner, model


def _artifact(path, raw, manifest):
    path.mkdir()
    payload = path / "bodyparts3d-myosim-major-bones.nhbones"
    payload.write_bytes(raw)
    header = owner._BONE_HEADER.unpack_from(raw) if len(raw) >= owner._BONE_HEADER.size else (0, 0, 0, 0, 0)
    manifest = json.loads(json.dumps(manifest))
    manifest["payload"] = {
        "file": payload.name,
        "sha256": model.sha256(payload),
        "bytes": len(raw),
        "vertex_count": header[3],
        "index_count": header[4],
    }
    mp = path / "bodyparts3d-myosim-major-bones.manifest.json"
    mp.write_text(json.dumps(manifest))
    return path, raw, manifest


class PrecisionAfterPartitionCompositionTests(unittest.TestCase):
    def make_chain(self, root):
        base, base_raw, base_manifest = _artifact(
            root / "base", b"base", {"source": {"identity": "pinned"}}
        )
        partition_decl = {
            "input_artifact": str(base),
            "input_manifest_sha256": model.sha256(
                base / "bodyparts3d-myosim-major-bones.manifest.json"
            ),
            "input_payload_sha256": model.sha256(
                base / "bodyparts3d-myosim-major-bones.nhbones"
            ),
        }
        parent, parent_raw, parent_manifest = _artifact(
            root / "partition",
            b"partition",
            {
                "source": {"identity": "pinned"},
                "source_partition_retriangulation": partition_decl,
            },
        )
        precision_decl = {
            "input_artifact": str(parent),
            "input_manifest_sha256": model.sha256(
                parent / "bodyparts3d-myosim-major-bones.manifest.json"
            ),
            "input_payload_sha256": model.sha256(
                parent / "bodyparts3d-myosim-major-bones.nhbones"
            ),
            "parameters": {"fixture": True},
            "proof": {"fixture": "precision"},
        }
        outer, outer_raw, outer_manifest = _artifact(
            root / "precision",
            b"precision",
            {
                **parent_manifest,
                "source_precision_retriangulation": precision_decl,
            },
        )
        return (base, base_raw, base_manifest), (parent, parent_raw, parent_manifest), (
            outer, outer_raw, outer_manifest
        )

    def test_precision_outer_replays_partition_parent_first(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            base, parent, outer = self.make_chain(root)
            base_path, base_raw, base_manifest = base
            parent_path, parent_raw, parent_manifest = parent
            outer_path, outer_raw, outer_manifest = outer

            with (
                patch.object(
                    owner,
                    "validate_bone_source_precision_retriangulation",
                    return_value=(
                        parent_path.resolve(), parent_manifest, parent_raw,
                        {"status": "PRECISION_REPLAY"},
                    ),
                ) as precision,
                patch.object(
                    owner,
                    "validate_bone_source_partition",
                    return_value=(
                        base_path.resolve(), base_manifest, base_raw,
                        {"status": "PARTITION_REPLAY"},
                    ),
                ) as partition,
                patch.object(
                    owner,
                    "_validate_composed_rib_reframe_source003",
                    return_value={
                        "status": "BASE_REPLAY",
                        "validation_sha256": "0" * 64,
                    },
                ) as base_replay,
            ):
                result = owner.validate_composed_rib_reframe(
                    outer_path, outer_manifest, outer_raw
                )

            precision.assert_called_once_with(
                outer_path.resolve(), outer_manifest, outer_raw
            )
            partition.assert_called_once_with(
                parent_path.resolve(), parent_manifest, parent_raw
            )
            base_replay.assert_called_once_with(
                base_path.resolve(), base_manifest, base_raw
            )
            self.assertEqual(
                result["source_precision_retriangulation"]["status"],
                "PRECISION_REPLAY",
            )
            self.assertEqual(
                result["source_partition_retriangulation"]["status"],
                "PARTITION_REPLAY",
            )

    def test_precision_outer_rejects_drifted_inherited_partition_pin(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, outer = self.make_chain(root)
            outer_path, outer_raw, outer_manifest = outer
            changed = json.loads(json.dumps(outer_manifest))
            changed["source_partition_retriangulation"][
                "input_manifest_sha256"
            ] = "f" * 64
            with self.assertRaisesRegex(
                model.ImportError, "bone partition input manifest is missing or drifted"
            ):
                owner.validate_composed_rib_reframe(
                    outer_path, changed, outer_raw
                )

    def test_bone_envelope_loader_selects_outer_precision_over_partition(self):
        from tests import test_bone_opposed_source_face_retirement as bone_fixture
        from numilab_human import model as human_model

        raw = bone_fixture.fixture()
        _, rows, _, _ = owner._decode_bones(raw)
        stable_id = rows[0][5]
        with TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base"
            base.mkdir()
            base_payload = base / "bodyparts3d-myosim-major-bones.nhbones"
            base_payload.write_bytes(raw)
            base_manifest = {
                "payload": {"file": base_payload.name, "sha256": model.sha256(base_payload)},
            }
            base_manifest_path = base / "bodyparts3d-myosim-major-bones.manifest.json"
            base_manifest_path.write_text(json.dumps(base_manifest))

            parent = root / "partition-parent"
            parent.mkdir()
            parent_payload = parent / "bodyparts3d-myosim-major-bones.nhbones"
            parent_payload.write_bytes(raw)
            partition_decl = {
                "input_artifact": str(base),
                "input_manifest_sha256": model.sha256(base_manifest_path),
            }
            parent_manifest = {
                "payload": {"file": parent_payload.name, "sha256": model.sha256(parent_payload)},
                "source_partition_retriangulation": partition_decl,
            }
            (parent / "bodyparts3d-myosim-major-bones.manifest.json").write_text(
                json.dumps(parent_manifest)
            )

            outer = root / "precision-child"
            outer.mkdir()
            outer_payload = outer / "bodyparts3d-myosim-major-bones.nhbones"
            outer_payload.write_bytes(raw)
            precision_decl = {
                "input_artifact": str(parent),
                "input_manifest_sha256": model.sha256(
                    parent / "bodyparts3d-myosim-major-bones.manifest.json"
                ),
            }
            outer_manifest = {
                "payload": {"file": outer_payload.name, "sha256": model.sha256(outer_payload)},
                "source_partition_retriangulation": partition_decl,
                "source_precision_retriangulation": precision_decl,
            }
            (outer / "bodyparts3d-myosim-major-bones.manifest.json").write_text(
                json.dumps(outer_manifest)
            )

            surfaces = {"fixture": [{"stable_id": stable_id, "triangles": []}]}
            proof = {"typed_identity": [51005, stable_id], "status": "PRECISION_REPLAY"}
            original_loader = human_model._numi_human_bone_envelope_surfaces

            def parent_only_loader(artifact, source_sha):
                self.assertEqual(Path(artifact).resolve(), parent.resolve())
                return surfaces, {"source": "partition-parent"}, parent_payload

            with (
                patch.object(owner, "validate_composed_rib_reframe",
                             return_value={"validation_sha256": "a" * 64}) as composed,
                patch.object(owner, "validate_bone_source_precision_retriangulation",
                             return_value=(parent.resolve(), parent_manifest, raw, proof)) as precision,
                patch.object(owner, "validate_bone_source_partition",
                             side_effect=AssertionError("wrong outer operation selected")) as partition,
                patch.object(human_model, "_numi_human_bone_envelope_surfaces",
                             side_effect=parent_only_loader),
            ):
                got_surfaces, got_metadata, got_path = original_loader(outer, "b" * 64)

            composed.assert_called_once()
            precision.assert_called_once()
            partition.assert_not_called()
            self.assertEqual(got_path.resolve(), outer_payload.resolve())
            self.assertEqual(got_metadata["source_precision_retriangulation"], proof)
            self.assertTrue(got_surfaces["fixture"][0]["triangles"])


if __name__ == "__main__":
    unittest.main()
