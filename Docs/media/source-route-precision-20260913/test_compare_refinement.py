"""Admission controls using retained fixtures; these tests never run physics."""
import copy
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

import compare_refinement as comparison

BASE = Path(__file__).parent


def receipts():
    return {step: json.loads((BASE / name / "execution.json").read_text())
            for step, name in comparison.RUNS}


class RefinementAdmissionTests(unittest.TestCase):
    def test_retained_inputs_pass(self):
        comparison.validate_inputs(BASE)

    def test_exit_code_and_schema_are_strict(self):
        for field, value in (("returncode", False), ("returncode", 0.0),
                             ("schema", "unknown"), ("source_unchanged", 1)):
            data = receipts()
            data[25][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                comparison.compatible_receipts(data)

    def test_before_after_identity_disagreement_rejected(self):
        for section in ("source", "artifacts"):
            data = receipts()
            first = next(iter(data[25][section + "_after"]))
            data[25][section + "_after"][first]["sha256"] = "0" * 64
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, "inconsistent retained"):
                comparison.compatible_receipts(data)

    def test_mixed_sources_rejected_even_when_unchanged(self):
        data = receipts()
        path = next(iter(data[25]["source_before"]))
        for when in ("before", "after"):
            data[25]["source_" + when][path]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source/build/static-input"):
            comparison.compatible_receipts(data)

    def test_each_static_input_and_binary_identity_is_bound(self):
        baseline = receipts()
        env = baseline[25]["request"]["environment"]
        for path in [env[k] for k in comparison.STATIC_FILES] + baseline[25]["request"]["binaries"]:
            data = copy.deepcopy(baseline)
            for when in ("before", "after"):
                data[25]["artifacts_" + when][path]["sha256"] = "0" * 64
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "source/build/static-input"):
                comparison.compatible_receipts(data)

    def test_environment_step_and_physics_settings_are_bound(self):
        for key, value in (("NUMANX_PREPARED_TIMESTEP_US", "50"),
                           ("NUMANX_PREPARED_DURATION_US", "800"),
                           ("NUMANX_HUMAN_SOURCE_FP", "0000000000000001"),
                           ("NEW_PHYSICS_SETTING", "1"), ("NM_HUMAN_SUPPORT_TRACE_ROOT", "9")):
            data = receipts()
            data[25]["request"]["environment"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                comparison.compatible_receipts(data)

    def test_diagnostic_trace_selector_is_explicitly_allowed(self):
        data = receipts()
        data[50]["request"]["environment"]["NM_HUMAN_SUPPORT_TRACE_ROOT"] = "5"
        comparison.compatible_receipts(data)

    def copy_inputs(self, temporary):
        root = Path(temporary)
        for step, name in comparison.RUNS:
            (root / name).mkdir()
            shutil.copyfile(BASE / name / "execution.json", root / name / "execution.json")
            fixture = f"precision-fixture-{step}us"
            shutil.copytree(BASE / fixture, root / fixture)
        return root

    def rewrite_fixture(self, root, name, raw):
        """Refresh outer digests so semantic controls must catch the mutation."""
        directory = root / "precision-fixture-25us"
        (directory / name).write_bytes(raw)
        compiler = json.loads((directory / "execution.json").read_text())
        compiler[name] = comparison.identity(raw)
        (directory / "execution.json").write_text(json.dumps(compiler))
        receipt_path = root / "precision-25us-001" / "execution.json"
        receipt = json.loads(receipt_path.read_text())
        for when in ("before", "after"):
            for path in receipt["artifacts_" + when]:
                if Path(path).name == name:
                    receipt["artifacts_" + when][path] = comparison.identity(raw)
        receipt_path.write_text(json.dumps(receipt))

    def test_fixture_schema_and_clock_mutations_rejected(self):
        for mutation in ("schema", "nhinit_clock", "package_clock", "wrong_fixture_path"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = self.copy_inputs(temporary)
                directory = root / "precision-fixture-25us"
                if mutation == "schema":
                    metadata = json.loads((directory / "prepared.json").read_text())
                    metadata["schema"] = "unknown"
                    self.rewrite_fixture(root, "prepared.json", json.dumps(metadata).encode())
                elif mutation == "nhinit_clock":
                    raw = bytearray((directory / "prepared.nhinit").read_bytes())
                    struct.pack_into("<Q", raw, 96, 50000)
                    self.rewrite_fixture(root, "prepared.nhinit", bytes(raw))
                elif mutation == "package_clock":
                    name = "prepared-25000ns.nmatterpack"
                    raw = bytearray((directory / name).read_bytes())
                    struct.pack_into("<f", raw, 260, 0.00005)
                    struct.pack_into("<Q", raw, 80, comparison.fnv(raw[88:280]))
                    self.rewrite_fixture(root, name, bytes(raw))
                else:
                    path = root / "precision-25us-001" / "execution.json"
                    data = json.loads(path.read_text())
                    data["request"]["environment"]["NUMANX_INITIAL_STATE"] = receipts()[50]["request"]["environment"]["NUMANX_INITIAL_STATE"]
                    path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    comparison.validate_inputs(root)

    def test_changed_initial_state_cannot_hide_behind_updated_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copy_inputs(temporary)
            directory = root / "precision-fixture-25us"
            raw = bytearray((directory / "prepared.nhinit").read_bytes())
            raw[160] ^= 1
            self.rewrite_fixture(root, "prepared.nhinit", bytes(raw))
            metadata = json.loads((directory / "prepared.json").read_text())
            metadata["initial_state_fp"] = f"{comparison.fnv(raw, 14695981039346656037):016x}"
            self.rewrite_fixture(root, "prepared.json", json.dumps(metadata).encode())
            path = root / "precision-25us-001" / "execution.json"
            data = json.loads(path.read_text())
            data["request"]["environment"]["NUMANX_INITIAL_STATE_FP"] = metadata["initial_state_fp"]
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "authored payload differs beyond timestep"):
                comparison.validate_inputs(root)

    def test_changed_package_physics_cannot_hide_behind_updated_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copy_inputs(temporary)
            directory = root / "precision-fixture-25us"
            name = "prepared-25000ns.nmatterpack"
            raw = bytearray((directory / name).read_bytes())
            raw[264] ^= 1  # Dispatch numerical limit, outside the timestep word.
            struct.pack_into("<Q", raw, 80, comparison.fnv(raw[88:280]))
            self.rewrite_fixture(root, name, bytes(raw))
            with self.assertRaisesRegex(ValueError, "authored payload differs beyond timestep"):
                comparison.validate_inputs(root)


if __name__ == "__main__":
    unittest.main()
