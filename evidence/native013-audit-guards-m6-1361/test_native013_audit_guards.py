#!/usr/bin/env python3
"""Synthetic fail-closed tests plus read-only prepared-bundle preflight for native013 guards."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
AUDITS = HERE
BUNDLE = Path("/Users/n/numi-human-fds132-m6-1350/attempt-156-right-candidate150-carrier-assets-001")
ORIGINALS = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
NAMES = (
    "audit_bilateral_native_013_successor_001.py",
    "audit_bilateral_external_native_013_successor_001.py",
)

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def extracted_guards(path: Path):
    tree = ast.parse(path.read_text())
    wanted = {"validate_compact_row", "shared_global_ids"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise AssertionError(f"guard definitions missing from {path}")
    ns = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns

def rejected(call, label: str):
    try:
        call()
    except ValueError:
        return {"case": label, "rejected": True}
    raise AssertionError(f"guard accepted malformed input: {label}")

def main():
    rows = []
    for name in NAMES:
        path = AUDITS / name
        guards = extracted_guards(path)
        validate = guards["validate_compact_row"]
        shared_ids = guards["shared_global_ids"]

        a_ids = np.array([10, 20, 30], dtype=np.int64)
        a_faces = np.array([[0, 1, 2]], dtype=np.int64)
        a_global = np.array([[10, 20, 30]], dtype=np.int64)
        b_ids = np.array([20, 30, 40], dtype=np.int64)
        b_faces = np.array([[0, 1, 2]], dtype=np.int64)
        b_global = np.array([[20, 30, 40]], dtype=np.int64)
        validate(a_ids, a_faces, a_global)
        validate(b_ids, b_faces, b_global)
        if shared_ids(a_ids, b_ids, 2) != [20, 30]:
            raise AssertionError("valid shared identity fixture did not pass")

        failures = [
            rejected(lambda: validate([10, 10, 30], a_faces, a_global), "duplicate_global_ids"),
            rejected(lambda: validate(a_ids, a_faces, [[10, 20, 31]]), "global_local_face_mapping_mismatch"),
            rejected(lambda: validate(a_ids, [[0, 1, 3]], [[10, 20, 30]]), "local_face_out_of_range"),
            rejected(lambda: shared_ids(a_ids, [50, 60, 70], 2), "empty_shared_intersection"),
            rejected(lambda: shared_ids(a_ids, b_ids, 3), "wrong_shared_cardinality"),
        ]

        prep = json.loads((BUNDLE / "preparation.json").read_text())
        actual = {}
        geometry_hashes = {}
        for sid in (131, 133):
            item = prep["geometry_rows"][str(sid)]
            geom_path = Path(item["path"])
            observed_geometry_sha = sha(geom_path)
            if observed_geometry_sha != item["sha256"]:
                raise AssertionError(f"prepared geometry hash mismatch for {sid}")
            geometry_hashes[str(sid)] = observed_geometry_sha
            with np.load(geom_path, allow_pickle=False) as z:
                ids, faces = validate(z["global_vertex_ids"], z["faces_local"], z["faces_global"])
                if z["vertices6"].shape[0] != len(ids):
                    raise AssertionError(f"vertex count mismatch for {sid}")
                actual[sid] = ids.tolist()
        expected = int(prep["shared_interface"]["candidate_global_vertex_count"])
        if expected != 2070 or len(shared_ids(actual[131], actual[133], expected)) != 2070:
            raise AssertionError("prepared shared identity cardinality did not match 2070")

        rows.append({
            "successor": name,
            "sha256": sha(path),
            "synthetic_valid_mapping": "passed",
            "synthetic_rejections": failures,
            "prepared_row_counts": {str(k): len(v) for k, v in actual.items()},
            "prepared_shared_identity_count": len(shared_ids(actual[131], actual[133], expected)),
            "prepared_shared_metadata_count": expected,
            "prepared_geometry_sha256": geometry_hashes,
        })

    report = {
        "schema": "numi.human.native013-audit-guard-regression-1361.v1",
        "status": "passed_guard_regressions_and_read_only_preflight",
        "scope": "Synthetic compact-ID/shared-count guard tests and read-only validation of the prepared candidate150 bundle. No native capture, GPU launch, or anatomy admission.",
        "successors": rows,
        "pins": {
            "main_original_sha256": sha(ORIGINALS / "audit_bilateral_native_013.py"),
            "external_original_sha256": sha(ORIGINALS / "audit_bilateral_external_native_013.py"),
            "bundle_preparation_sha256": sha(BUNDLE / "preparation.json"),
        },
    }
    out = HERE / "guard_regression.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "report": str(out), "sha256": sha(out)}))

if __name__ == "__main__":
    main()
