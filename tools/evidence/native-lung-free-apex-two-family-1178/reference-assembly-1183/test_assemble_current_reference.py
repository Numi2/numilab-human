import copy, importlib.util, json, unittest
from pathlib import Path

DRIVER_PATH=Path("/Users/n/numi-human-resting-evidence-20261005/final-native-scene-preflight-936/current-reference-1183/assemble_current_reference.py")
SPEC=importlib.util.spec_from_file_location("reference_driver_1183", DRIVER_PATH)
DRIVER=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(DRIVER)
REF=DRIVER.REF
def load(name): return json.loads((REF/name).read_text())
class ReferenceValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta=load("run-metadata.json")
        cls.inv=load("invocation.json")
        cls.ver=json.loads(DRIVER.VERIFY.read_text())
        for p,h in DRIVER.PINS.items():
            if DRIVER.sha(p)!=h: raise AssertionError("pinned input mismatch: "+str(p))
    def valid(self, metadata=None, invocation=None, verification=None, pins=None):
        return DRIVER.validate_reference_metadata(
            copy.deepcopy(metadata if metadata is not None else self.meta),
            copy.deepcopy(invocation if invocation is not None else self.inv),
            copy.deepcopy(verification if verification is not None else self.ver),
            dict(pins if pins is not None else DRIVER.PINS), REF)
    def test_actual_pinned_reference_passes(self):
        self.valid()
    def test_reference_cli_reports_1173_identity_not_931(self):
        import contextlib, io, sys
        old=sys.argv
        try:
            sys.argv=[str(DRIVER_PATH), "--reference-self-test"]
            out=io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(DRIVER.main(), 0)
            result=json.loads(out.getvalue())
            self.assertEqual(result["status"], "pinned_verified1173_reference_pass")
            self.assertEqual(result["reference_run_path"], str(REF))
            self.assertEqual(result["accepted_terminal_step"], 10000)
            self.assertFalse(result["historical_931_metadata_reconstructed"])
        finally:
            sys.argv=old
    def test_failed_native_metadata_rejected(self):
        m=copy.deepcopy(self.meta); m["exit_code"]=1
        with self.assertRaisesRegex(RuntimeError,"did not complete"): self.valid(metadata=m)
    def test_invocation_mismatch_rejected(self):
        inv=copy.deepcopy(self.inv); inv["argv"]=inv["argv"]+["--unexpected"]
        with self.assertRaisesRegex(RuntimeError,"metadata/invocation mismatch"): self.valid(invocation=inv)
    def test_wrong_terminal_receipt_hash_rejected(self):
        v=copy.deepcopy(self.ver); v["terminal"]["receipt_sha256"]="0"*64
        with self.assertRaisesRegex(RuntimeError,"terminal file identity mismatch"): self.valid(verification=v)
    def test_advanced_terminal_rejected(self):
        v=copy.deepcopy(self.ver); v["terminal"]["accepted_step"]=10001
        with self.assertRaisesRegex(RuntimeError,"terminal accepted-state proof incomplete"): self.valid(verification=v)
    def test_unsealed_terminal_rejected(self):
        v=copy.deepcopy(self.ver); v["terminal"]["no_additional_physical_or_controller_step"]=False
        with self.assertRaisesRegex(RuntimeError,"terminal accepted-state proof incomplete"): self.valid(verification=v)
    def test_hash_pin_mismatch_rejected(self):
        key=REF/"native.log"; original=DRIVER.PINS[key]; DRIVER.PINS[key]="0"*64
        try:
            with self.assertRaisesRegex(RuntimeError,"reference input changed"): DRIVER.main()
        finally:
            DRIVER.PINS[key]=original
if __name__=="__main__": unittest.main(verbosity=2)
