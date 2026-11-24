import json
import tempfile
import unittest
from pathlib import Path

from live_pose_frames import current_donors


class LivePoseFramesTest(unittest.TestCase):
    def setUp(self):
        storage = tempfile.TemporaryDirectory()
        self.addCleanup(storage.cleanup)
        self.case = Path(storage.name).resolve()
        self.live = self.case / "native-initial-bodies.json"
        self.source = self.case / "registered-initial-bodies.json"
        env = {"NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP": str(self.source)}
        (self.case / "launch.json").write_text(json.dumps({"environment": env}))
        for path, height in ((self.live, 0.12), (self.source, 0.13)):
            pose = {"position_world_m": [0, 0, height], "orientation_world_xyzw": [0, 0, 0, 1]}
            rows = [{"body_index": 0, "current": pose}]
            path.write_text(json.dumps(dict(schema="numi.human.native-skin-pose-snapshot.v1",
                    registration_fingerprint32=17, binding_count=1, vertex_count=9,
                    payload_abi=5, bodies=rows, all_bodies=rows)))

    def test_uses_live_pose(self):
        file, rows = current_donors(self.case)
        self.assertEqual(file, self.live)
        self.assertEqual(rows[0]["current"]["position_world_m"][2], 0.12)

    def test_rejects_same_capture(self):
        self.source.unlink()
        self.source.symlink_to(self.live)
        with self.assertRaises(ValueError):
            current_donors(self.case)

    def test_rejects_mismatched_source(self):
        snapshot = json.loads(self.source.read_bytes())
        snapshot["registration_fingerprint32"] += 1
        self.source.write_text(json.dumps(snapshot))
        with self.assertRaises(ValueError):
            current_donors(self.case)

    def test_rejects_inconsistent_native_rows(self):
        snapshot = json.loads(self.live.read_bytes())
        snapshot["all_bodies"][0]["current"]["position_world_m"][0] = 1
        self.live.write_text(json.dumps(snapshot))
        with self.assertRaises(ValueError):
            current_donors(self.case)


if __name__ == "__main__":
    unittest.main()
