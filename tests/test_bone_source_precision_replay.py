import hashlib
import json
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from numilab_human import composed_bone_body_frame as owner, model
from tests import test_bone_opposed_source_face_retirement as bone_fixture
from tests import test_resting_lung_edge_repair as lung_fixture


def fixture():
    raw = bone_fixture.fixture()
    h, rows, vo, io = owner._decode_bones(raw)
    points, faces = lung_fixture.RestingLungEdgeRepairTests._octahedron()
    vertex_streams, index_streams, records = [], [], []
    vc = ic = 0
    for old in rows:
        if old[5] == 49:
            vertices, triangles = points, faces
        else:
            vertices = np.frombuffer(raw, "<f4", count=old[2]*6,
                                     offset=vo+old[1]*24).reshape(-1, 6)
            triangles = np.frombuffer(raw, "<u4", count=old[4],
                                      offset=io+old[3]*4).reshape(-1, 3)-old[1]
        record = list(old)
        record[1:5] = [vc, len(vertices), ic, triangles.size]
        records.append(record)
        vertex_streams.append(np.asarray(vertices, dtype="<f4").tobytes())
        index_streams.append(np.asarray(triangles+vc, dtype="<u4").tobytes())
        vc += len(vertices)
        ic += triangles.size
    header = list(h)
    header[3:5] = [vc, ic]
    return (owner._BONE_HEADER.pack(*header)
            + b"".join(owner._BONE_RECORD.pack(*r) for r in records)
            + b"".join(vertex_streams) + b"".join(index_streams))


PARAMS = dict(stable_id=49, keep_vertex=2, remove_vertex=0, flip_edges=[],
              max_endpoint_displacement_m=2e-5, minimum_altitude_m=1e-12,
              max_abs_volume_delta_m3=1e-12)


class BonePrecisionReplayTests(unittest.TestCase):
    def case(self, root):
        source = root/"source"
        source.mkdir()
        raw = fixture()
        p = source/"bodyparts3d-myosim-major-bones.nhbones"
        p.write_bytes(raw)
        manifest = {"payload": {"file": p.name, "sha256": model.sha256(p),
                                "bytes": len(raw), "index_count": owner._BONE_HEADER.unpack_from(raw)[4]},
                    "source": {"identity": "untouched"}, "body_local_bakes": {"proof": "retained"}}
        mp = source/"bodyparts3d-myosim-major-bones.manifest.json"
        mp.write_text(json.dumps(manifest))
        result, proof = owner.repair_bone_source_precision_edges(raw, **PARAMS)
        manifest["payload"].update(sha256=hashlib.sha256(result).hexdigest(),
                                   bytes=len(result), index_count=owner._BONE_HEADER.unpack_from(result)[4])
        manifest["source_precision_retriangulation"] = {
            "input_artifact": str(source), "input_manifest_sha256": model.sha256(mp),
            "input_payload_sha256": model.sha256(p), "parameters": PARAMS.copy(), "proof": proof}
        return result, manifest, raw, source

    def test_preserves_attributes_rest_owners_and_other_bone_faces(self):
        raw = fixture()
        result, proof = owner.repair_bone_source_precision_edges(raw, **PARAMS)
        _, before, bv, bi = owner._decode_bones(raw)
        _, after, av, ai = owner._decode_bones(result)
        self.assertEqual(raw[bv:bi], result[av:ai])
        for old, new in zip(before, after):
            self.assertEqual(old[:3], new[:3])
            self.assertEqual(old[5:], new[5:])
            if old[5] != 49:
                self.assertEqual(raw[bi+old[3]*4:bi+(old[3]+old[4])*4],
                                 result[ai+new[3]*4:ai+(new[3]+new[4])*4])
        self.assertEqual(len(proof["output_face_parent_origins"]), after[48][4]//3)
        self.assertFalse(proof["anatomical_admission"])

    def test_replays_authenticated_input(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            result, manifest, raw, source = self.case(root)
            got = owner.validate_bone_source_precision_retriangulation(root/"out", manifest, result)
            self.assertEqual(got[0], source.resolve())
            self.assertEqual(got[2], raw)

    def test_rejects_unrelated_manifest_changes(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            result, manifest, *_ = self.case(root)
            manifest["source"]["identity"] = "renamed"
            with self.assertRaisesRegex(model.ImportError, "unrelated manifest"):
                owner.validate_bone_source_precision_retriangulation(root/"out", manifest, result)

    def test_rejects_source_drift_output_drift_and_false_ancestry(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            result, manifest, _, source = self.case(root)
            with self.assertRaisesRegex(model.ImportError, "bytes or source ancestry"):
                owner.validate_bone_source_precision_retriangulation(root/"out", manifest, result[:-1]+b"!")
            manifest["source_precision_retriangulation"]["proof"]["native_admission"] = True
            with self.assertRaisesRegex(model.ImportError, "bytes or source ancestry"):
                owner.validate_bone_source_precision_retriangulation(root/"out", manifest, result)
            (source/"bodyparts3d-myosim-major-bones.nhbones").write_bytes(b"drift")
            with self.assertRaisesRegex(model.ImportError, "source payload"):
                owner.validate_bone_source_precision_retriangulation(root/"out", manifest, result)

    def test_rejects_untyped_identity_and_bad_flip_ancestry(self):
        for changes in ({"stable_id": True}, {"flip_edges": [[False, 1]]}):
            with self.subTest(changes=changes), self.assertRaises(model.ImportError):
                owner.repair_bone_source_precision_edges(fixture(), **{**PARAMS, **changes})


if __name__ == "__main__":
    unittest.main()
