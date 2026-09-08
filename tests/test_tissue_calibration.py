import math
import tempfile
import unittest
from pathlib import Path

from numilab_human import tissue_calibration as tc
from numilab_human.model import ImportError as HumanImportError


class TissueCalibrationTests(unittest.TestCase):
    def test_protocol_rejects_wrong_units_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.txt"
            prefix = "<Mach-1 File>\n<Zero Load>\n<DATA>\n"
            path.write_text(prefix + "Time, s\tPosition (z), mm\tFz, N\n0\t0\t0\n")
            with self.assertRaisesRegex(HumanImportError, "units"):
                tc.parse_mach1(path)
            path.write_text(prefix + "Time, s\tPosition (z), mm\tFz, gf\n0\t0\t0\n")
            with self.assertRaisesRegex(HumanImportError, "incomplete"):
                tc.parse_mach1(path)

    def test_time_weighting_is_invariant_to_irregular_sampling(self):
        coarse = [(0., 0., 0.), (1., 0., 2.), (10., 0., 20.)]
        dense = [(float(i), 0., float(2*i)) for i in range(11)]
        self.assertAlmostEqual(tc.time_mean(coarse, 7., 2), 13.)
        self.assertAlmostEqual(tc.time_mean(dense, 7., 2), 13.)
        with self.assertRaises(HumanImportError):
            tc.time_mean(coarse, 11., 2)

    def test_lateral_traction_free_nh_limits(self):
        area = math.pi * .005**2 / 4
        for strain in (.001, .05, .15):
            axial = 1 - strain
            # lambda=0 gives no lateral strain; incompressible limit J=1.
            self.assertAlmostEqual(tc.nh_force_basis(strain, 0.) / area,
                                   1 / axial - axial, places=12)
            self.assertAlmostEqual(tc.nh_force_basis(strain, .4999999) / area,
                                   axial**-2 - axial, places=6)
        # Infinitesimal modulus E=2*mu*(1+nu).
        self.assertAlmostEqual(tc.nh_force_basis(1e-7) / (area*1e-7), 2.9, places=5)

    def test_fit_uses_only_supplied_training_and_preserves_units(self):
        mu = 80000.
        training = [dict(compression_strain=e, force_N=mu*tc.nh_force_basis(e))
                    for e in (.04, .08, .12)]
        fitted = tc.fit_mu(training)
        self.assertAlmostEqual(fitted, mu)
        held = [dict(compression_strain=.10, force_N=2*mu*tc.nh_force_basis(.10))]
        self.assertAlmostEqual(tc.metrics(held, fitted)["force_nrmse_relative_to_measured_rms"], .5)
        self.assertAlmostEqual(tc.GF_TO_N*1000, 9.80665)
        self.assertTrue(set(tc.TRAIN).isdisjoint(tc.HELD_OUT))
        self.assertEqual(len({x.rsplit("-", 1)[0] for x in (*tc.TRAIN, *tc.HELD_OUT)}), 1)

    def test_otms_rejects_mismatched_identity_and_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.xml"
            text = ('<OTMS><Sample>plug_timestamp</Sample><Thickness units="mm">'
                    '[2,2,2,2,2]</Thickness><Statistics units="mm"><average>2</average>'
                    '</Statistics></OTMS>')
            path.write_text(text)
            self.assertEqual(tc.thickness(path, "plug"), .002)
            with self.assertRaises(HumanImportError):
                tc.thickness(path, "another-plug")
            path.write_text(text.replace("<average>2", "<average>3"))
            with self.assertRaises(HumanImportError):
                tc.thickness(path, "plug")

    def test_source_hash_gate_and_candidate_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source"
            path.write_bytes(b"raw")
            with self.assertRaisesRegex(HumanImportError, "SHA256"):
                tc.verify_file(path, {"bytes": 3, "sha256": "0"*64})
        text = tc.material_text(80000., "0"*64)
        self.assertIn("UNQUALIFIED", text)
        self.assertNotIn(" identifiable", text)
        self.assertIn("lambda : Pa = 720000", text)

    def test_strain_uses_protocol_contact_reference_not_fast_ramp_crossing(self):
        blocks = [tc.Block(command, {}, 2, (0., 100., 10.), (1., 100., 10.), [])
                  for command in tc.COMMANDS]
        for index in (1, 14):
            blocks[index].header["Stop criteria, gf:"] = "10.000000"
        blocks[8].header = {"Number of Cycles:": "1000", "Freqency, Hz:": "2.00000"}
        blocks[17].header["Position, mm:"] = "99.7000"
        # This faster ramp crosses 10 gf before the protocol reference.
        blocks[18].rows = [(0., 99.7, 0.), (1., 100., 20.), (2., 100.1, 30.)]
        for ramp, strain in ((18, .05), (23, .10), (28, .15)):
            position = 100. + strain * 2.  # 2 mm specimen.
            for index, duration in zip(range(ramp+1, ramp+5), (10, 100, 1000, 690)):
                block = blocks[index]
                block.header["Wait:"] = f"00:00:{duration}"
                block.first = (0., position, 30.)
                block.last = (float(duration), position, 29.)
                block.rows = [block.first, block.last]
        report = tc.observations(blocks, .002)
        self.assertEqual(report["reference_position_mm"], 100.)
        for point, expected in zip(report["points"], (.05, .10, .15)):
            self.assertAlmostEqual(point["compression_strain"], expected)
        blocks[17].header["Position, mm:"] = "99.6"
        with self.assertRaisesRegex(HumanImportError, "reference"):
            tc.observations(blocks, .002)


if __name__ == "__main__":
    unittest.main()
