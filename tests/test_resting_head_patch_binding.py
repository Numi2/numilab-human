import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from numilab_human import resting_head_patch_binding as binding


class HeadPatchInputTests(unittest.TestCase):
    def test_member_pin_authenticates_bytes_independently_of_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "source.zip"
            member = "partof/FJ1831.obj"
            raw = b"# File ID : FJ1831\nv 0 0 0\n"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr(member, raw)
            pins = {str(archive): "member-only-verified-not-whole-archive-hash",
                    str(archive) + "::" + member: hashlib.sha256(raw).hexdigest()}
            checked = binding._checked_inputs(pins)
            self.assertEqual(checked[str(archive) + "::" + member][1], raw)
            pins[str(archive) + "::" + member] = "0" * 64
            with self.assertRaisesRegex(ValueError, "source SHA differs"):
                binding._checked_inputs(pins)

    def test_parent_pin_is_unique(self):
        checked = {"one": ("a" * 64, b"one"), "two": ("a" * 64, b"one")}
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            binding._one_input(checked, "a" * 64)

    def _report(self, changed=None):
        report = {
            "schema": "numi.human.head343-374-source-patch-fragment-complex-gate.v1",
            "status": "PREFLIGHT_ONLY_COMPLETE_NO_EXACT_AUDITS",
            "source_patch_complex": {
                "identities": [[51010, 343], [51010, 374]],
                "input_pins": {"capture_pack_sha256": "a" * 64,
                               "capture_receipt_sha256": "b" * 64}}}
        if changed:
            changed(report)
        return report

    def _bind(self, report, *, digest=None, endpoint=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            raw = json.dumps(report).encode()
            path.write_bytes(raw)
            return binding.bind_head_source_patch_report(
                {}, endpoint or {"pack_sha256": "a" * 64,
                                 "capture_receipt_sha256": "b" * 64},
                path, digest or hashlib.sha256(raw).hexdigest())

    def test_changed_report_bytes_rejected_before_binding(self):
        with self.assertRaisesRegex(ValueError, "report SHA differs"):
            self._bind(self._report(), digest="0" * 64)

    def test_different_capture_rejected_before_geometry(self):
        with self.assertRaisesRegex(ValueError, "different accepted capture"):
            self._bind(self._report(), endpoint={"pack_sha256": "c" * 64,
                                                "capture_receipt_sha256": "b" * 64})

    def test_bare_ids_cannot_alias_typed_anatomy(self):
        with self.assertRaisesRegex(ValueError, "typed identities differ"):
            self._bind(self._report(lambda r: r["source_patch_complex"].update(
                identities=[[51005, 343], [51010, 374]])))

    def test_failed_report_cannot_admit_contacts(self):
        with self.assertRaisesRegex(ValueError, "incomplete or failed"):
            self._bind(self._report(lambda r: r.update(status="FAIL_FIRST_SELF")))


if __name__ == "__main__":
    unittest.main()
