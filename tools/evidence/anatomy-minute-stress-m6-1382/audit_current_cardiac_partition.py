from pathlib import Path
import hashlib, json, sys, time
import numpy as np

SRC = Path("/Users/numi/numi-human-anatomy-finish-1378/src")
PKG = SRC / "numilab_human"
PAYLOAD = Path("/Users/numi/numi-human-resting-evidence-20261005/native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-thorax.nhanatomy")
RECEIPT = Path("/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276/back412-skin-reference-native-021-003/resting-anatomy-receipt.json")
OUT = Path("/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
START = OUT / "cardiac-current-1c0c37-start.json"
REPORT = OUT / "cardiac-current-1c0c37-certificate.json"

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write_new(p, obj):
    with Path(p).open("x") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")

sys.path.insert(0, str(SRC))
from numilab_human import cardiac_cavity_intersections as predicates
from numilab_human import cardiac_partition_certificate as cert
from numilab_human.resting_anatomy_interface_patch import parse_payload

receipt = json.loads(RECEIPT.read_text())
prov = receipt["provenance"]
source_map = prov["source_id_map"]
inputs = {
    "payload": {"path": str(PAYLOAD), "sha256": sha(PAYLOAD)},
    "current_native_receipt": {"path": str(RECEIPT), "sha256": sha(RECEIPT)},
    "certificate_source": {"path": str(Path(cert.__file__)), "sha256": sha(cert.__file__)},
    "intersection_predicate_source": {"path": str(Path(predicates.__file__)), "sha256": sha(predicates.__file__)},
    "payload_parser_source": {"path": str(PKG / "resting_anatomy_interface_patch.py"), "sha256": sha(PKG / "resting_anatomy_interface_patch.py")},
}
assert inputs["payload"]["sha256"] == "1c0c37af76ab3f8e86870fd6cd3abab00b7bcdae51fe934e3461722ca306c241"
assert receipt["payload"]["input_payload_sha256"] == inputs["payload"]["sha256"]
assert prov["cardiac_cavity_geometry_sha256"] == "849bd41e2fa7e172e8cda2c6889658eda203c4c5e240b3a2adb7c5649c9243c1"
for sid, member, fma in [(318, "FJ2424", "FMA11359"), (319, "FJ2423", "FMA9291")]:
    row = source_map[str(sid)]
    assert row["source_member"] == member and row["source_owner_metadata"] is None
    assert row["layer"] == 9 and row["body_index"] == 20
    assert row["source_sha256"]
h, rows = parse_payload(PAYLOAD)
regions = {}
face_rows = {}
for sid, name in [(318, "right_atrium"), (319, "right_ventricle")]:
    row = rows[sid]
    assert row["body_index"] == 20 and row["layer"] == 9
    p = row["vertices6"][:, :3].astype(np.float64)
    f = row["faces"]
    tris = [tuple(tuple(float(x) for x in p[j]) for j in face) for face in f]
    regions[name] = tris
    face_rows[name] = {tuple(sorted(tri)): k for k, tri in enumerate(tris)}
shared_keys = face_rows["right_atrium"].keys() & face_rows["right_ventricle"].keys()
interface = [regions["right_atrium"][face_rows["right_atrium"][key]] for key in sorted(shared_keys)]
start = {
    "status": "running",
    "scope": "Direct exact Float32 current-payload classification for stable IDs 318/319 only. Uses the current source-neutral partition certificate; no mesh/runtime mutation or physiology claim.",
    "inputs": inputs,
    "source_binding": {
        "canonical_four_surface_geometry_sha256": prov["cardiac_cavity_geometry_sha256"],
        "right_atrium": {k: source_map["318"].get(k) for k in ("name", "source_member", "source_sha256", "body_index", "layer")},
        "right_ventricle": {k: source_map["319"].get(k) for k in ("name", "source_member", "source_sha256", "body_index", "layer")},
    },
    "payload_rows": {
        name: {"vertex_count": int(rows[sid]["vertices6"].shape[0]), "triangle_count": int(len(rows[sid]["faces"]))}
        for sid, name in [(318, "right_atrium"), (319, "right_ventricle")]
    },
    "exact_shared_triangle_count": len(shared_keys),
    "old_1182_certificate_scope": "Its c10dce4609... payload pin is not reused as evidence for current 1c0c37... geometry.",
}
write_new(START, start)
t = time.monotonic()
try:
    proof = cert.audit_shared_interface_partition(regions, interface, classify_connected_patches=True)
    report = {
        "status": "pass",
        "scope": "Current 1c0c37 NHA payload, direct stable-ID 318/319 source-neutral cavity geometry check only; geometrically shared face complex is not a biological valve plane or clinical validation.",
        "inputs": inputs,
        "source_binding": start["source_binding"],
        "payload_rows": start["payload_rows"],
        "exact_shared_triangle_count": len(shared_keys),
        "proof": proof,
        "elapsed_s": time.monotonic() - t,
        "inputs_unchanged": all(sha(Path(v["path"])) == v["sha256"] for v in inputs.values()),
    }
    write_new(REPORT, report)
    print(json.dumps({"status": "pass", "report": str(REPORT), "report_sha256": sha(REPORT), "elapsed_s": report["elapsed_s"],
                      "interiors_disjoint": proof.get("interiors_disjoint"), "cross_domain_contact_pairs": proof.get("cross_domain_contact_pairs"),
                      "shared_triangles": len(shared_keys)}, sort_keys=True), flush=True)
except Exception as e:
    failure = {"status": "failed", "error": repr(e), "inputs": inputs, "exact_shared_triangle_count": len(shared_keys),
               "elapsed_s": time.monotonic() - t,
               "inputs_unchanged": all(sha(Path(v["path"])) == v["sha256"] for v in inputs.values())}
    write_new(OUT / "cardiac-current-1c0c37-failure.json", failure)
    print(json.dumps(failure, sort_keys=True), flush=True)
    raise

