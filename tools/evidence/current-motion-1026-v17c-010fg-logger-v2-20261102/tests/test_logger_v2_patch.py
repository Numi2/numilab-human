import subprocess
import tempfile
from pathlib import Path
import unittest

OWNER_ROOT = Path("/Users/numi/Work/skin-fat-native-integration-209")
OWNER_REVISION = "f75955f08d6edf0f85027eac7614017ac9fbc996"
OWNER_RELATIVE = "apps/numilab_human_myosim_visual_probe.mm"
PATCH = Path(__file__).resolve().parents[1] / "logger" / "endpoint-capture-v2.patch"


class LoggerV2PatchTests(unittest.TestCase):
    def test_patch_applies_only_the_v2_endpoint_fields_to_pinned_base(self):
        source_ref = f"{OWNER_REVISION}:{OWNER_RELATIVE}"
        baseline = subprocess.check_output(
            ["git", "-C", str(OWNER_ROOT), "show", source_ref],
            env={"GIT_NO_LAZY_FETCH": "1"},
        )
        text = baseline.decode("utf-8")
        changes = (
            (
                '                <<"resting_FEM_endpoint_capture=begin version=1 steps="<<stepCount\n',
                '                <<"resting_FEM_endpoint_capture=begin version=2 steps="<<stepCount\n',
            ),
            (
                '                        <<",inverse_mass="<<v.w\n'
                '                        <<",rest_x="<<rest.x<<",rest_y="<<rest.y<<",rest_z="<<rest.z\n',
                '                        <<",inverse_mass="<<v.w\n'
                '                        <<",initial_rest_x="<<initialNode.restAndFixed.x\n'
                '                        <<",initial_rest_y="<<initialNode.restAndFixed.y\n'
                '                        <<",initial_rest_z="<<initialNode.restAndFixed.z\n'
                '                        <<",rest_x="<<rest.x<<",rest_y="<<rest.y<<",rest_z="<<rest.z\n',
            ),
            (
                '                        <<",rest_x="<<rest.x<<",rest_y="<<rest.y<<",rest_z="<<rest.z\n'
                '                        <<",reference_mode="<<current.referenceDisplacementAndMode.w\n',
                '                        <<",rest_x="<<rest.x<<",rest_y="<<rest.y<<",rest_z="<<rest.z\n'
                '                        <<",initial_reference_mode="<<initialNode.referenceDisplacementAndMode.w\n'
                '                        <<",initial_reference_u_hi_x="<<initialNode.referenceDisplacementAndMode.x\n'
                '                        <<",initial_reference_u_hi_y="<<initialNode.referenceDisplacementAndMode.y\n'
                '                        <<",initial_reference_u_hi_z="<<initialNode.referenceDisplacementAndMode.z\n'
                '                        <<",initial_reference_u_lo_x="<<initialNode.referenceDisplacementLow.x\n'
                '                        <<",initial_reference_u_lo_y="<<initialNode.referenceDisplacementLow.y\n'
                '                        <<",initial_reference_u_lo_z="<<initialNode.referenceDisplacementLow.z\n'
                '                        <<",reference_mode="<<current.referenceDisplacementAndMode.w\n',
            ),
        )
        expected = text
        for old, new in changes:
            self.assertEqual(expected.count(old), 1)
            expected = expected.replace(old, new, 1)

        with tempfile.TemporaryDirectory(prefix="logger-v2-patch-") as temporary:
            target = Path(temporary) / OWNER_RELATIVE
            target.parent.mkdir(parents=True)
            target.write_bytes(baseline)
            result = subprocess.run(
                ["patch", "--batch", "--forward", "-p1"],
                cwd=temporary,
                input=PATCH.read_bytes(),
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout.decode() + result.stderr.decode())
            self.assertEqual(target.read_bytes(), expected.encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
