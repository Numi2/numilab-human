#!/usr/bin/env python3
"""Rebind retained cardiac source proof by byte equality; no predicates or state updates."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
OWNER = Path("/Users/n/numi-human-final-integration-publish-1397")
D = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
sys.path.insert(0, str(OWNER / "src"))
from numilab_human.resting_anatomy_interface_patch import parse_payload
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok, why):
    if not ok:
        raise ValueError(why)
CERT = D / "cardiac-current-1c0c37-certificate.json"
RECEIPT = D / "head-source-identity-migration-002/resting-anatomy-receipt.json"
EXPECTED_CERT = "b5624cf41a2c113457619036a38f63653d896e729ac7a37132bf44dcc511308e"
def main():
    expected = EXPECTED_CERT
    require(sha(CERT) == expected, "certificate pin differs")
    require(sha(RECEIPT) == "89d55e1f6dbcacf7a6d985c56e0de40c7bdbec66ecd9542c1eaa2851c61d0992",
            "current receipt pin differs")
    cert = json.loads(CERT.read_text())
    receipt = json.loads(RECEIPT.read_text())
    require(cert["status"] == "pass" and cert["inputs_unchanged"] is True,
            "existing certificate did not pass on immutable inputs")
    proof = cert["proof"]
    require(proof["interiors_disjoint"] is True and proof["shared_interface_opposite_chain"] is True
            and proof["biological_valve_interface"] is False and proof["mechanical_mass_assigned"] is False,
            "certificate proof scope differs")
    old = Path(cert["inputs"]["payload"]["path"])
    current = Path(receipt["payload"]["path"])
    if not current.is_absolute():
        current = RECEIPT.parent / current
    require(sha(old) == cert["inputs"]["payload"]["sha256"], "certified payload changed")
    require(sha(current) == receipt["payload"]["sha256"] ==
            "1e6da4fa8c550334f31a6b74fda7aed6f26b0c436bff0b9ad168fe97d41cbb23",
            "current anatomy payload changed")
    _, old_rows = parse_payload(old)
    _, rows = parse_payload(current)
    compared = []
    for sid, name in ((318, "right_atrium"), (319, "right_ventricle")):
        a, b = old_rows[sid], rows[sid]
        bound = cert["source_binding"][name]
        require(a["body_index"] == b["body_index"] == bound["body_index"] == 20
                and a["layer"] == b["layer"] == bound["layer"] == 9
                and a["flags"] == b["flags"], "typed ownership changed")
        identity = receipt["provenance"]["source_id_map"][str(sid)]
        require(identity["source_member"] == bound["source_member"]
                and identity["source_sha256"] == bound["source_sha256"], "source member identity changed")
        v = np.ascontiguousarray(b["vertices6"], dtype="<f4")
        f = np.ascontiguousarray(b["faces"], dtype="<u4")
        require(v.tobytes() == np.ascontiguousarray(a["vertices6"], dtype="<f4").tobytes()
                and f.tobytes() == np.ascontiguousarray(a["faces"], dtype="<u4").tobytes(),
                "certified geometry or normals changed")
        compared.append({"stable_id": sid, "body_index": 20, "layer": 9,
                         "source_member": bound["source_member"], "source_sha256": bound["source_sha256"],
                         "vertex_count": len(v), "face_count": len(f),
                         "vertices6_f32_sha256": hashlib.sha256(v.tobytes()).hexdigest(),
                         "faces_u32_sha256": hashlib.sha256(f.tobytes()).hexdigest(),
                         "exact_source_geometry_and_normals_equal": True})
    result = {"status": "SOURCE_GEOMETRY_EQUIVALENT_EXISTING_CERTIFICATE", "passed": True,
              "certificate": {"path": str(CERT), "sha256": expected},
              "current_receipt": {"path": str(RECEIPT), "sha256": sha(RECEIPT)},
              "current_payload": {"path": str(current), "sha256": sha(current)},
              "certified_payload": {"path": str(old), "sha256": sha(old)},
              "rows": compared, "existing_proof_reused_by_exact_geometry_equality": True,
              "geometry_predicates_run": False, "physical_or_controller_steps_advanced": 0,
              "native_motion_admission": False, "biological_valve_interface": False,
              "mechanical_mass_assigned": False, "blood_state_added": False,
              "scope": "Source-neutral RA/RV partition only. Historical raw-source overlap notes remain retained and do not describe these byte-identical partitioned rows. Current native motion and joint anatomy still require separate verification.",
              "verification_owner": {"path": str(Path(parse_payload.__code__.co_filename)),
                                     "sha256": sha(Path(parse_payload.__code__.co_filename))},
              "driver": {"path": str(Path(__file__)), "sha256": sha(Path(__file__))}}
    out = D / "cardiac-current-source-equivalence-001"
    out.mkdir(exist_ok=False)
    (out / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "report": str(out / "report.json"),
                      "sha256": sha(out / "report.json")}))
if __name__ == "__main__":
    main()
