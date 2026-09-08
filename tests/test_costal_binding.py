import copy
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from numilab_human.costal_binding import compile_binding, HEADER, MATRIX
from numilab_human.model import (
    ImportError, _NUMI_HUMAN_COSTAL_CARTILAGE_MEMBERS,
    _NUMI_HUMAN_STERNAL_ATTACHMENT_MEMBERS,
)


class CostalBindingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.hash = lambda data: hashlib.sha256(data).hexdigest()
        source = {"revision": "source-test", "archive_sha256": "34" * 32}
        pose = {"core_body_index": 0, "source_body_id": 1, "name": "torso",
                "default_com_position_world_m": [0., 0., 1.],
                "default_inertial_quaternion_world_xyzw": [0., 0., 0., 1.]}
        # Structural authoring fixture. Numerical mesh/body admission remains
        # in the native decoder and independent native qualification program.
        rigid = struct.pack("<8s10I32s", b"NHRIGID2", 1, 1, 1, 1, 0, 7, 6, 0, 0, 0,
                            bytes.fromhex("34" * 32)) + bytes(224 + 160 + 64*6 + 4*13 + 32 - 80)
        rigid = rigid[:-32] + struct.pack("<I7f", 0, 0., 0., 1., 0., 0., 0., 1.)
        (self.root / "rigid.bin").write_bytes(rigid)
        cart = struct.pack("<8s6I2f32s", b"NHCART1\0", 1, 14, 14, 14, 14, 0,
                           1100., .004, bytes.fromhex("56" * 32))
        cart += b"".join(struct.pack("<8s10I4f", row[0].encode(), *([0]*10), *([0.]*4))
                         for row in _NUMI_HUMAN_COSTAL_CARTILAGE_MEMBERS)
        cart += bytes(28*14 + 20*14)
        (self.root / "cart.bin").write_bytes(cart)
        self.human = {"source": source,
                      "core_tree": {"source_body_records": [pose]},
                      "payloads": {"rigid": {"file": "rigid.bin", "bytes": len(rigid), "sha256": self.hash(rigid)}}}
        self.cartilage = {"payload": {"file": "cart.bin", "bytes": len(cart), "sha256": self.hash(cart)},
                          "source": {"bodyparts3d_archive": {"sha256": "56"*32},
                                     "sternal_members": [], "rib_members": []}}
        self.registration = {"source": {"myosim": {"source": source,
                            "payloads": {"rigid": {"sha256": "78"*32}}}},
                             "coordinate_system": {"global_source_mm_to_myosim_world_m": [
                                 [.001,0,0,.01],[0,.001,0,.02],[0,0,.001,.03],[0,0,0,1]]},
                             "anchors": []}
        for group, members in (("sternal_members", _NUMI_HUMAN_STERNAL_ATTACHMENT_MEMBERS),
                               ("rib_members", [r[4] for r in _NUMI_HUMAN_COSTAL_CARTILAGE_MEMBERS])):
            for member in members:
                digest = self.hash(member.encode())
                self.cartilage["source"][group].append({"member_id": member, "obj_sha256": digest})
                self.registration["anchors"].append({"source": {"member_id": member, "member_sha256": digest},
                                                      "target": copy.deepcopy(pose)})

    def run_binding(self):
        for name, data in (("cart.json", self.cartilage), ("human.json", self.human), ("registration.json", self.registration)):
            (self.root / name).write_text(json.dumps(data))
        return compile_binding(cartilage_manifest=self.root / "cart.json", human_manifest=self.root / "human.json",
                               registration=self.root / "registration.json", output=self.root / "out")

    def test_common_frame_refresh_preserves_current_source_and_deterministic_bytes(self):
        result = self.run_binding()
        first = (self.root / "out/costal-tissue.nhtbind").read_bytes()
        self.assertEqual(len(first), 424)
        self.assertEqual(HEADER.unpack_from(first)[8].hex(), self.human["payloads"]["rigid"]["sha256"])
        self.assertEqual(MATRIX.unpack_from(first, HEADER.size)[:4], (1., 0., 0., .01))
        self.assertTrue(result["registration"]["source_body_pose_continuity_checked"])
        self.assertFalse(result["qualification"]["accepted_root_integration"])
        self.run_binding()
        self.assertEqual(first, (self.root / "out/costal-tissue.nhtbind").read_bytes())

    def test_missing_manubrium_is_not_silently_substituted(self):
        self.registration["anchors"].pop(0)
        with self.assertRaisesRegex(ImportError, "missing source bone registration"):
            self.run_binding()

    def test_stale_source_pose_cannot_be_refreshed_by_hash_replacement(self):
        self.registration["anchors"][0]["target"]["default_com_position_world_m"][0] += .001
        with self.assertRaisesRegex(ImportError, "pose drifted"):
            self.run_binding()

    def test_changed_payload_bytes_fail_before_emission(self):
        (self.root / "rigid.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ImportError, "bytes disagree"):
            self.run_binding()
        self.assertFalse((self.root / "out").exists())

    def test_reflection_shear_and_invalid_scale_fail(self):
        baseline = copy.deepcopy(self.registration)
        for row, column, value in ((0, 0, -.001), (0, 1, .0001), (0, 0, float("nan"))):
            self.registration = copy.deepcopy(baseline)
            self.registration["coordinate_system"]["global_source_mm_to_myosim_world_m"][row][column] = value
            with self.assertRaises(ImportError):
                self.run_binding()

    def test_source_bone_order_and_hash_are_binding_authority(self):
        baseline = copy.deepcopy(self.cartilage)
        self.cartilage["source"]["rib_members"].reverse()
        with self.assertRaisesRegex(ImportError, "correspondence"):
            self.run_binding()
        self.cartilage = baseline
        self.registration["anchors"][0]["source"]["member_sha256"] = "ab" * 32
        with self.assertRaisesRegex(ImportError, "mesh identity"):
            self.run_binding()

    def test_different_source_model_cannot_reuse_registration(self):
        self.human["source"] = {"revision": "different"}
        with self.assertRaisesRegex(ImportError, "different source"):
            self.run_binding()

    def test_split_thorax_requires_volumetric_mass_correspondence(self):
        other = copy.deepcopy(self.human["core_tree"]["source_body_records"][0])
        other.update(core_body_index=1, source_body_id=2, name="rib")
        self.human["core_tree"]["source_body_records"].append(other)
        self.registration["anchors"][-1]["target"] = other
        # An invented articulated donor is rejected at rigid-byte admission,
        # before it can be used to hide a missing volume ownership map.
        with self.assertRaisesRegex(ImportError, "source body manifest coverage"):
            self.run_binding()

    def test_manifest_cannot_relabel_the_rigid_reference_pose(self):
        self.human["core_tree"]["source_body_records"][0]["default_com_position_world_m"][0] = .01
        for anchor in self.registration["anchors"]:
            anchor["target"]["default_com_position_world_m"][0] = .01
        with self.assertRaisesRegex(ImportError, "pose manifest disagrees"):
            self.run_binding()


if __name__ == "__main__":
    unittest.main()
