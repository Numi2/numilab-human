"""Pinned-source export checks; run with the existing .venv-myosim interpreter."""

from pathlib import Path
import unittest

from numilab_human.myosim_export import export_fullbody

try:
    import mujoco
    from myo_sim.build.compose import build_model
except ImportError:
    mujoco = None


@unittest.skipUnless(mujoco is not None, "requires the pinned MyoSim source environment")
class MyoSimSourceInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.value = export_fullbody(Path(__file__).resolve().parents[1] / "Sources")
        cls.model = build_model("myofullbody")

    def test_source_tendon_partition_and_unlowered_ligaments(self):
        value, model = self.value, self.model
        records = value["nonmuscle_tendons"]
        self.assertEqual(len(records), 8)
        self.assertEqual(len(value["muscles"]), 416)
        self.assertEqual(len(value["sites"]), 1815)
        self.assertEqual(len(value["wrap_geometries"]), 143)
        self.assertEqual({entry["id"] for entry in records}, set(range(model.ntendon)) - {
            int(model.actuator_trnid[index, 0]) for index in range(model.nu)
        })
        self.assertEqual({entry["name"] for entry in records}, {
            name + suffix for suffix in ("", "_l")
            for name in ("s_glenohum_ligament", "m_glenohum_ligament", "i_glenohum_ligament", "coracohum_ligament")
        })
        for entry in records:
            with self.subTest(tendon=entry["name"]):
                self.assertEqual(entry["native_mechanics_status"], "not_lowered")
                parameters = entry["compiled_mujoco_parameters"]
                self.assertEqual(parameters["tendon_stiffness"], 0.0)
                self.assertEqual(parameters["tendon_damping"], 0.0)
                self.assertEqual(parameters["tendon_actuatorid"], -1)
                self.assertEqual(parameters["tendon_lengthspring"], model.tendon_lengthspring[entry["id"]].tolist())
                self.assertEqual(len(entry["route"]), 3)
                self.assertEqual(len(entry["sites"]), 3)
                self.assertEqual(len(entry["wrap_geometries"]), 1)
                sites = {site["id"] for site in entry["sites"]}
                self.assertIn(entry["route"][1]["side_site_source_id"], sites)
                for site in entry["sites"]:
                    self.assertEqual(site["position_body_m"], model.site_pos[site["id"]].tolist())


if __name__ == "__main__":
    unittest.main()
