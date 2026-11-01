#!/usr/bin/env python3
"""Fail-closed consistency check for a 433 candidate and its 217 child mesh.

This checks the exact files consumed by the existing 436 conservative mass
mapper. It does not run mass mapping, native motion, or physical admission.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "numi.h706.current-mass-input-consistency-1023.v1"
CANDIDATE_MODE = "numi-human.current-typed-volume-candidate.v1"
LABEL_HEADER = ["child_tet_row", "material_index", "geometric_donor_body",
                "anatomical_region_id", "stable_source_face_id"]
HERE = Path(__file__).resolve().parent
PHYSICAL_TRANSFER = HERE.parent.parent
MASS_OWNER_DEFAULT = PHYSICAL_TRANSFER / "run_refined_mass_gate_436.py"
SOURCE330_DEFAULT = PHYSICAL_TRANSFER / "native-physical-parent-mass-330" / "native-source-identity-control-003"
SOURCE330_FILES = (
    "compiled-parent-cell-mass.csv",
    "compiled-parent-cell-mass.csv.nodes.csv",
    "compiled-parent-cell-mass.csv.tets.csv",
    "compiled-parent-cell-mass.csv.labels.csv",
    "compiled-parent-cell-mass.csv.weights.csv",
    "result.json",
)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def pin(path: Path, expected: str, label: str) -> dict[str, str]:
    p = Path(path).resolve()
    if not p.is_file():
        raise ValueError(f"{label} is missing: {p}")
    got = sha256(p)
    if got != expected:
        raise ValueError(f"{label} SHA mismatch: expected {expected}, got {got}")
    return {"path": str(p), "sha256": got}

def _one_string(value: Any, label: str) -> str:
    a = np.asarray(value)
    if a.size != 1:
        raise ValueError(f"{label} must be a scalar string lane")
    v = a.reshape(-1)[0]
    if isinstance(v, bytes):
        v = v.decode("utf-8")
    if not isinstance(v, (str, np.str_)):
        raise ValueError(f"{label} must contain text")
    return str(v)

def _read_text_rows(path: Path) -> list[list[str]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if fields and fields[0] == "#":
                continue
            rows.append(fields)
    if not rows:
        raise ValueError(f"{path.name} is empty")
    return rows

def read_node(path: Path) -> np.ndarray:
    rows = _read_text_rows(path)
    head = rows[0]
    if len(head) != 4:
        raise ValueError("mesh.node header must be: node_count 3 0 0")
    n, dim, attrs, markers = map(int, head)
    if n < 1 or dim != 3 or attrs != 0 or markers != 0 or len(rows) != n + 1:
        raise ValueError("mesh.node header/domain is not the canonical 3D untagged node table")
    xyz = np.empty((n, 3), dtype=np.float64)
    for i, row in enumerate(rows[1:]):
        if len(row) != 4 or int(row[0]) != i + 1:
            raise ValueError(f"mesh.node row ID/domain mismatch at row {i}")
        xyz[i] = [float(x) for x in row[1:]]
    if not np.isfinite(xyz).all():
        raise ValueError("mesh.node contains nonfinite coordinates")
    return xyz

def read_ele(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = _read_text_rows(path)
    head = rows[0]
    if len(head) != 3:
        raise ValueError("mesh.ele header must be: tet_count 4 1")
    n, corners, attrs = map(int, head)
    if n < 1 or corners != 4 or attrs != 1 or len(rows) != n + 1:
        raise ValueError("mesh.ele header/domain is not the canonical 4-node, one-material table")
    tets = np.empty((n, 4), dtype=np.int64)
    mats = np.empty(n, dtype=np.int64)
    for i, row in enumerate(rows[1:]):
        if len(row) != 6 or int(row[0]) != i + 1:
            raise ValueError(f"mesh.ele row ID/width mismatch at row {i}")
        tets[i] = [int(x) - 1 for x in row[1:5]]
        mats[i] = int(row[5])
    if np.any(tets < 0):
        raise ValueError("mesh.ele has nonpositive one-based node IDs")
    return tets, mats

def read_labels(path: Path, n: int) -> np.ndarray:
    expected = np.empty((n, 5), dtype=np.int64)
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != LABEL_HEADER:
            raise ValueError("child.labels.csv header differs from the exact five-column 850/433 schema")
        count = 0
        for row in reader:
            if count >= n:
                raise ValueError("child.labels.csv has more rows than mesh.ele")
            values = [int(row[k]) for k in LABEL_HEADER]
            if values[0] != count:
                raise ValueError(f"child.labels.csv child_tet_row is not contiguous at {count}")
            expected[count] = values
            count += 1
    if count != n:
        raise ValueError(f"child.labels.csv has {count} rows; expected {n}")
    return expected

def verify(candidate_path: Path, candidate_sha: str, volume_dir: Path,
           export_receipt_path: Path, export_receipt_sha: str,
           current_state_path: Path, current_state_sha: str,
           source_proof_path: Path, source_proof_sha: str,
           source330_dir: Path = SOURCE330_DEFAULT,
           mass_owner_path: Path = MASS_OWNER_DEFAULT) -> dict[str, Any]:
    candidate_path = Path(candidate_path).resolve()
    volume_dir = Path(volume_dir).resolve()
    export_receipt_path = Path(export_receipt_path).resolve()
    current_state_path = Path(current_state_path).resolve()
    source_proof_path = Path(source_proof_path).resolve()
    source330_dir = Path(source330_dir).resolve()
    mass_owner_path = Path(mass_owner_path).resolve()

    candidate_pin = pin(candidate_path, candidate_sha, "433 candidate")
    export_pin = pin(export_receipt_path, export_receipt_sha, "433 receipt")
    state_pin = pin(current_state_path, current_state_sha, "current state")
    proof_pin = pin(source_proof_path, source_proof_sha, "808 source proof")
    if not mass_owner_path.is_file():
        raise ValueError("frozen 436 mass owner is missing")
    mass_owner_pin = {"path": str(mass_owner_path), "sha256": sha256(mass_owner_path)}

    state = json.loads(current_state_path.read_text())
    if state.get("physical_admission") is not False or state.get("native_steps") != 0:
        raise ValueError("current state is not source-only")
    mass_status = state.get("mass_mapping", state.get("mass_reconciliation", "PENDING"))
    if mass_status not in ("NOT_RUN", "PENDING", "PENDING_VOLUME_CHANGE_RECONCILIATION"):
        raise ValueError(f"current state has incompatible mass status {mass_status!r}")

    source_proof = json.loads(source_proof_path.read_text())
    if (source_proof.get("status") != "PASS_CURRENT_EXACT_SOURCE_EVIDENCE" or
        source_proof.get("physical_admission") is not False or source_proof.get("native_steps") != 0):
        raise ValueError("808 source proof is not passing source-only evidence")
    proof_state = source_proof.get("state", {})
    if (Path(proof_state.get("path", "")).resolve() != current_state_path or
        proof_state.get("sha256") != current_state_sha):
        raise ValueError("808 proof does not bind the exact current state")
    all185 = source_proof.get("all185", {})
    if (all185.get("record_count") != 938272 or
        all185.get("changed_self_count") != 0 or
        all185.get("changed_vs_all185_count") != 0):
        raise ValueError("808 proof does not establish the complete unchanged all185 source context")

    mesh_node = (volume_dir / "mesh.node").resolve()
    mesh_ele = (volume_dir / "mesh.ele").resolve()
    labels_csv = (volume_dir / "child.labels.csv").resolve()
    for p in (mesh_node, mesh_ele, labels_csv):
        if not p.is_file():
            raise ValueError(f"433 volume output is missing: {p}")

    receipt = json.loads(export_receipt_path.read_text())
    if (receipt.get("schema") != "numi-human.registered-volume-parent-identity-source-control.v1" or
        receipt.get("physical_admission") is not False or receipt.get("native_steps") != 0):
        raise ValueError("433 receipt is not the expected source-only export")
    input_hashes = receipt.get("input_sha256", {})
    if input_hashes.get(str(candidate_path)) != candidate_sha:
        raise ValueError("433 receipt does not bind the supplied candidate NPZ")
    output_hashes = receipt.get("output_sha256", {})
    output_pins = {}
    for name, path in (("mesh.node", mesh_node), ("mesh.ele", mesh_ele),
                       ("child.labels.csv", labels_csv)):
        got = sha256(path)
        if output_hashes.get(str(path)) != got:
            raise ValueError(f"433 receipt does not bind the supplied {name}")
        output_pins[name] = {"path": str(path), "sha256": got}
    operational = receipt.get("candidate_operational_reference_map", {})
    if (operational.get("candidate_npz_sha256") != candidate_sha or
        operational.get("candidate_node_sha256") != output_pins["mesh.node"]["sha256"] or
        operational.get("candidate_element_sha256") != output_pins["mesh.ele"]["sha256"] or
        operational.get("candidate_labels_sha256") != output_pins["child.labels.csv"]["sha256"]):
        raise ValueError("433 operational-reference record does not bind candidate and exported mesh")

    with np.load(candidate_path, allow_pickle=False) as z:
        required = ("candidate_mapping_mode", "points_reference_world_m_f64",
                    "volume_tetrahedra", "material_index", "geometric_donor_body",
                    "anatomical_region_id", "stable_source_face_id",
                    "current_typed_role_asset_pins_json")
        missing = [k for k in required if k not in z.files]
        if missing:
            raise ValueError(f"candidate NPZ lacks required current typed mass lanes {missing}")
        if _one_string(z["candidate_mapping_mode"], "candidate_mapping_mode") != CANDIDATE_MODE:
            raise ValueError("candidate NPZ is not the authenticated current typed volume candidate")
        points = np.asarray(z["points_reference_world_m_f64"], dtype=np.float64)
        tets = np.asarray(z["volume_tetrahedra"], dtype=np.int64)
        if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or
            tets.ndim != 2 or tets.shape[1] != 4 or len(tets) < 1 or
            np.any(tets < 0) or np.any(tets >= len(points))):
            raise ValueError("candidate point/tet domains are malformed")
        sem = {}
        for k in ("material_index", "geometric_donor_body",
                  "anatomical_region_id", "stable_source_face_id"):
            a = np.asarray(z[k], dtype=np.int64)
            if a.shape != (len(tets),):
                raise ValueError(f"candidate semantic lane {k} has the wrong domain")
            sem[k] = a.copy()
        role_pins = json.loads(_one_string(z["current_typed_role_asset_pins_json"],
                                           "current_typed_role_asset_pins_json"))

    role_surfaces = source_proof.get("source_surfaces", {})
    if set(role_pins) != {"O", "I", "D"} or set(role_surfaces) != {"O", "I", "D"}:
        raise ValueError("candidate or 808 proof does not have the exact O/I/D source-role set")
    for role in ("O", "I", "D"):
        rp = role_pins[role]
        sp = role_surfaces[role]
        if (Path(rp.get("path", "")).resolve() != Path(sp.get("path", "")).resolve() or
            rp.get("sha256") != sp.get("sha256")):
            raise ValueError(f"candidate {role} source-role pin differs from passing 808 proof")
        if input_hashes.get(str(Path(rp["path"]).resolve())) != rp["sha256"]:
            raise ValueError(f"433 receipt omits or changes the candidate {role} source pin")

    exported_points = read_node(mesh_node)
    exported_tets, exported_material = read_ele(mesh_ele)
    exported_labels = read_labels(labels_csv, len(exported_tets))
    expected_points = points.astype(np.float32).astype(np.float64)
    if exported_points.shape != expected_points.shape or not np.array_equal(exported_points, expected_points):
        raise ValueError("433 mesh.node coordinates differ from candidate Float32 reference coordinates")
    if exported_tets.shape != tets.shape or not np.array_equal(exported_tets, tets):
        raise ValueError("433 mesh.ele connectivity/order differs from candidate volume_tetrahedra")
    if not np.array_equal(exported_material, sem["material_index"]):
        raise ValueError("433 mesh.ele material column differs from candidate material_index")
    expected_labels = np.column_stack([
        np.arange(len(tets), dtype=np.int64),
        sem["material_index"], sem["geometric_donor_body"],
        sem["anatomical_region_id"], sem["stable_source_face_id"]])
    if not np.array_equal(exported_labels, expected_labels):
        raise ValueError("433 child.labels.csv differs from candidate material/donor/region/source-face labels")

    source_files = {}
    for name in SOURCE330_FILES:
        p = source330_dir / name
        if not p.is_file():
            raise ValueError(f"source330 control003 file is missing: {p}")
        source_files[name] = {"path": str(p), "sha256": sha256(p)}
    control = json.loads((source330_dir / "result.json").read_text())
    if (control.get("mass_mapping_pass") is not True or
        control.get("source_identity_remap_pass") is not True or
        control.get("source_prepared_replacement_pass") is not True or
        control.get("source_capture_only") is not True or
        control.get("physical_steps") != 0 or
        control.get("repaired_volume_admitted") is not False or
        control.get("motion_qualified") is not False):
        raise ValueError("source330 control003 does not match the frozen source-only control contract")

    return {
        "schema": SCHEMA,
        "status": "PASS_433_MASS_INPUTS_BITWISE_BOUND_NOT_RUN",
        "candidate": candidate_pin,
        "current_state": state_pin,
        "source_proof_808": proof_pin,
        "source_proof_status": source_proof["status"],
        "export_receipt_433": export_pin,
        "exported_volume": output_pins,
        "candidate_rows": {"points": int(len(points)), "tetrahedra": int(len(tets))},
        "source330_control003": {
            "directory": str(source330_dir), "files": source_files,
            "control_result": {
                "mass_mapping_pass": True, "source_identity_remap_pass": True,
                "source_prepared_replacement_pass": True, "source_capture_only": True,
                "physical_steps": 0, "repaired_volume_admitted": False,
                "motion_qualified": False}},
        "mass_mapper_436": mass_owner_pin,
        "mass_mapping": "NOT_RUN",
        "physical_admission": False,
        "native_steps": 0,
        "limitations": [
            "This preflight binds candidate arrays to the exact 433 exported TetGen mesh and semantic labels.",
            "It does not execute the 436 mass mapper, native dynamics, or physical admission.",
            "The 436 result must be reviewed independently and remains distinct from motion qualification."
        ]
    }

def run_controls() -> dict[str, Any]:
    from tempfile import TemporaryDirectory
    from unittest import TestCase
    class Controls(TestCase):
        pass
    def write_mesh(d: Path, points: np.ndarray, tets: np.ndarray, sem: dict[str, np.ndarray]):
        vd = d / "volume"; vd.mkdir()
        with (vd / "mesh.node").open("w") as f:
            f.write(f"{len(points)} 3 0 0\n")
            for i, xyz in enumerate(points, 1):
                f.write(str(i) + " " + " ".join(format(float(x), ".17g") for x in xyz) + "\n")
        with (vd / "mesh.ele").open("w") as f:
            f.write(f"{len(tets)} 4 1\n")
            for i, tet in enumerate(tets, 1):
                f.write(str(i) + " " + " ".join(str(int(x) + 1) for x in tet) +
                        " " + str(int(sem["material_index"][i-1])) + "\n")
        with (vd / "child.labels.csv").open("w", newline="") as f:
            w = csv.writer(f, lineterminator="\n"); w.writerow(LABEL_HEADER)
            for i in range(len(tets)):
                w.writerow([i, *(int(sem[k][i]) for k in
                    ("material_index", "geometric_donor_body", "anatomical_region_id", "stable_source_face_id"))])
        return vd
    def build_case(d: Path):
        state = d / "state.json"
        state.write_text(json.dumps({"physical_admission": False, "native_steps": 0,
                                     "mass_mapping": "NOT_RUN"}))
        role_files = {}
        for role in ("O", "I", "D"):
            p = d / f"{role}.npz"
            p.write_bytes(role.encode("ascii"))
            role_files[role] = {"path": str(p.resolve()), "sha256": sha256(p)}
        proof = d / "source-proof.json"
        proof.write_text(json.dumps({"status": "PASS_CURRENT_EXACT_SOURCE_EVIDENCE",
            "physical_admission": False, "native_steps": 0,
            "state": {"path": str(state.resolve()), "sha256": sha256(state)},
            "all185": {"record_count": 938272, "changed_self_count": 0,
                       "changed_vs_all185_count": 0},
            "source_surfaces": role_files}))
        pts = np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
        tets = np.asarray([[0,1,2,3]], dtype=np.int64)
        sem = {"material_index": np.asarray([15]), "geometric_donor_body": np.asarray([2]),
               "anatomical_region_id": np.asarray([7]), "stable_source_face_id": np.asarray([9])}
        candidate = d / "candidate.npz"
        np.savez_compressed(candidate, candidate_mapping_mode=np.asarray([CANDIDATE_MODE]),
            points_reference_world_m_f64=pts.astype(np.float64), volume_tetrahedra=tets,
            **sem, current_typed_role_asset_pins_json=np.asarray([json.dumps(role_files, sort_keys=True)]))
        volume = write_mesh(d, pts.astype(np.float64), tets, sem)
        receipt = d / "433-receipt.json"
        out = {str((volume / x).resolve()): sha256(volume / x)
               for x in ("mesh.node", "mesh.ele", "child.labels.csv")}
        report = {"schema": "numi-human.registered-volume-parent-identity-source-control.v1",
            "physical_admission": False, "native_steps": 0,
            "input_sha256": {str(candidate.resolve()): sha256(candidate),
                **{v["path"]: v["sha256"] for v in role_files.values()}},
            "output_sha256": out,
            "candidate_operational_reference_map": {
                "candidate_npz_sha256": sha256(candidate),
                "candidate_node_sha256": out[str((volume / "mesh.node").resolve())],
                "candidate_element_sha256": out[str((volume / "mesh.ele").resolve())],
                "candidate_labels_sha256": out[str((volume / "child.labels.csv").resolve())]}}
        receipt.write_text(json.dumps(report))
        source330 = d / "source330"
        source330.mkdir()
        for name in SOURCE330_FILES:
            (source330 / name).write_text("fixture\n")
        (source330 / "result.json").write_text(json.dumps({
            "mass_mapping_pass": True, "source_identity_remap_pass": True,
            "source_prepared_replacement_pass": True, "source_capture_only": True,
            "physical_steps": 0, "repaired_volume_admitted": False,
            "motion_qualified": False}))
        owner = d / "436.py"; owner.write_text("# fixture owner\n")
        return {"state": state, "proof": proof, "candidate": candidate,
                "volume": volume, "receipt": receipt, "source330": source330, "owner": owner}

    def call(case):
        return verify(case["candidate"], sha256(case["candidate"]), case["volume"],
            case["receipt"], sha256(case["receipt"]), case["state"], sha256(case["state"]),
            case["proof"], sha256(case["proof"]), case["source330"], case["owner"])
    tests = []
    with TemporaryDirectory() as tmp:
        d = Path(tmp); c = build_case(d)
        good = call(c)
        if good["status"] != "PASS_433_MASS_INPUTS_BITWISE_BOUND_NOT_RUN":
            raise AssertionError("valid synthetic inputs were not accepted")
        tests.append("exact candidate/433 mesh/label roundtrip accepted")
        original = (c["volume"] / "mesh.node").read_text()
        (c["volume"] / "mesh.node").write_text(original.replace("1 0 0 0", "1 0.0001 0 0", 1))
        try: call(c)
        except ValueError: tests.append("tampered node coordinate rejected")
        else: raise AssertionError("tampered node coordinate accepted")
        (c["volume"] / "mesh.node").write_text(original)
        original_ele = (c["volume"] / "mesh.ele").read_text()
        (c["volume"] / "mesh.ele").write_text(original_ele.replace("1 2 3 4 15", "1 3 2 4 15", 1))
        try: call(c)
        except ValueError: tests.append("tampered tet ordering rejected")
        else: raise AssertionError("tampered tet ordering accepted")
        (c["volume"] / "mesh.ele").write_text(original_ele)
        original_labels = (c["volume"] / "child.labels.csv").read_text()
        (c["volume"] / "child.labels.csv").write_text(original_labels.replace("0,15,2,7,9", "0,16,2,7,9", 1))
        try: call(c)
        except ValueError: tests.append("tampered material/source label rejected")
        else: raise AssertionError("tampered labels accepted")
        (c["volume"] / "child.labels.csv").write_text(original_labels)
        state0 = c["state"].read_text()
        c["state"].write_text(json.dumps({"physical_admission": True, "native_steps": 0,
                                          "mass_mapping": "NOT_RUN"}))
        try: call(c)
        except ValueError: tests.append("physical-admission state rejected")
        else: raise AssertionError("admitted state accepted as source-only")
        c["state"].write_text(state0)
        c["proof"].write_text(json.dumps({"status": "FAIL", "physical_admission": False,
                                          "native_steps": 0}))
        try: call(c)
        except (ValueError, KeyError): tests.append("failed or stale source proof rejected")
        else: raise AssertionError("failed source proof accepted")
    return {"status": "PASS_433_MASS_INPUT_CONSISTENCY_CONTROLS",
            "tests": tests, "physical_admission": False, "native_steps": 0,
            "mass_mapping": "NOT_RUN"}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--candidate", type=Path)
    ap.add_argument("--candidate-sha256")
    ap.add_argument("--volume-dir", type=Path)
    ap.add_argument("--export-receipt", type=Path)
    ap.add_argument("--export-receipt-sha256")
    ap.add_argument("--current-state", type=Path)
    ap.add_argument("--current-state-sha256")
    ap.add_argument("--source-proof-808", type=Path)
    ap.add_argument("--source-proof-808-sha256")
    ap.add_argument("--source330-dir", type=Path, default=SOURCE330_DEFAULT)
    ap.add_argument("--mass-owner-436", type=Path, default=MASS_OWNER_DEFAULT)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.self_test:
        print(json.dumps(run_controls(), indent=2, sort_keys=True))
        return 0
    required = (a.candidate, a.candidate_sha256, a.volume_dir,
        a.export_receipt, a.export_receipt_sha256, a.current_state,
        a.current_state_sha256, a.source_proof_808, a.source_proof_808_sha256, a.out)
    if any(x is None for x in required):
        ap.error("candidate, candidate SHA, volume dir, 433 receipt and SHA, state and SHA, 808 proof and SHA, and output are required")
    out = Path(a.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    result = verify(a.candidate, a.candidate_sha256, a.volume_dir,
        a.export_receipt, a.export_receipt_sha256, a.current_state,
        a.current_state_sha256, a.source_proof_808, a.source_proof_808_sha256,
        a.source330_dir, a.mass_owner_436)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "out": str(out),
                      "candidate_rows": result["candidate_rows"],
                      "mass_mapping": "NOT_RUN", "physical_admission": False,
                      "native_steps": 0}, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
