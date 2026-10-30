import importlib.util
import tempfile
from pathlib import Path

MODULE = Path(__file__).with_name("canonical_d_operational_targets.py")
spec = importlib.util.spec_from_file_location("h706_targets", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def check_legacy_contract():
    result = module.select_paired_inner_surface({}, {})
    assert result == {"path": None, "sha256": None, "gate_name": "changed-D-vs-I014"}

def check_current_binding_and_mismatch():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "I-current.npz"
        p.write_bytes(b"authenticated current inner test")
        digest = module.sha(p)
        descriptor = {"path": str(p), "sha256": digest}
        proof = {"paired_inner_surface": descriptor}
        gates = {"paired_inner_surface": dict(descriptor)}
        result = module.select_paired_inner_surface(proof, gates)
        assert result == {"path": str(p), "sha256": digest, "gate_name": "changed-D-vs-current-I"}
        assert result["sha256"] == module.sha(p)
        bad_gates = {"paired_inner_surface": {"path": str(p), "sha256": "0" * 64}}
        try:
            module.select_paired_inner_surface(proof, bad_gates)
        except ValueError as exc:
            assert "hash mismatch" in str(exc) or "not bound" in str(exc)
        else:
            raise AssertionError("mismatched current inner hash was accepted")
        try:
            module.select_paired_inner_surface({"paired_inner_surface": {"path": str(p), "sha256": "0" * 64}}, gates)
        except ValueError as exc:
            assert "hash mismatch" in str(exc)
        else:
            raise AssertionError("proof with wrong current inner hash was accepted")

if __name__ == "__main__":
    check_legacy_contract()
    check_current_binding_and_mismatch()
    print("PASS legacy I014 selector retained; current inner hash binding and mismatch rejected")
