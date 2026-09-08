"""Compile pinned source scalar joint-limit compliance; never step physics."""
from __future__ import annotations

import hashlib
import math
import struct

from .model import ImportError


def compile_joint_limits(exported: dict, manifest: dict) -> tuple[dict, bytes]:
    if exported.get("source") != manifest.get("source"):
        raise ImportError("joint-limit source provenance disagrees with the native map")
    digest = manifest.get("source", {}).get("archive_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ImportError("joint-limit source archive requires a canonical SHA256")
    policy = exported.get("model", {}).get("joint_limit_solver")
    if not isinstance(policy, dict) or any(policy.get(key) != value for key, value in {
        "schema": "numi.human.mujoco-scalar-limit-solver.v1",
        "mujoco_version": "3.12.0", "integrator": "Euler",
        "diagexact": False, "enabled": True,
    }.items()) or any(type(policy.get(key)) is not bool for key in ("refsafe", "diagexact", "enabled")):
        raise ImportError("NHLIM1 requires pinned MuJoCo 3.12 classic scalar-limit metadata")

    def scalar(value):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ImportError("joint-limit parameter must be finite")
        try:
            rounded = struct.unpack("<f", struct.pack("<f", value))[0]
        except (OverflowError, struct.error) as error:
            raise ImportError("joint-limit parameter is outside FP32") from error
        if not math.isfinite(rounded):
            raise ImportError("joint-limit parameter is outside FP32")
        if value != 0 and abs(rounded) < 1.1754943508222875e-38:
            raise ImportError("joint-limit parameter is not a normal FP32 value")
        return float(value)

    def vector(value, count):
        if not isinstance(value, list) or len(value) != count:
            raise ImportError("joint-limit parameter dimension mismatch")
        return [scalar(x) for x in value]

    core = manifest["core_tree"]
    nq, nv = core["nq"], core["nv"]
    if type(nq) is not int or type(nv) is not int or not 6 < nv <= 160 or nq != nv + 1:
        raise ImportError("NHLIM1 requires the scalar floating-root native layout")
    bindings = {}
    native_dofs = set()
    for row in core["source_joint_map"]:
        source_id, v, q = row["source_joint_id"], row["core_v_index"], row["core_q_index"]
        if any(type(x) is not int for x in (source_id, v, q)) or source_id < 0 or source_id >= 2**32 - 1 or \
                not 6 <= v < nv or q != v + 1 or source_id in bindings or v in native_dofs:
            raise ImportError("joint-limit native/source identity is invalid or duplicated")
        if type(row.get("source_limited")) is not bool:
            raise ImportError("joint-limit native source coverage must be explicit")
        bindings[source_id] = row
        native_dofs.add(v)
    records, identities, seen = [], [], set()
    joints = exported.get("joints")
    if not isinstance(joints, list) or any(not isinstance(row, dict) or
            type(row.get("id")) is not int or not 0 <= row["id"] < 2**32 - 1 for row in joints):
        raise ImportError("source joint identity is invalid")
    for joint in sorted(joints, key=lambda row: row["id"]):
        source_id = joint["id"]
        if type(source_id) is not int or source_id in seen:
            raise ImportError("source joint identity is duplicated or invalid")
        seen.add(source_id)
        if type(joint.get("limited")) is not bool:
            raise ImportError("source joint limit flag is not explicit")
        if source_id in bindings and bindings[source_id]["source_limited"] != joint["limited"]:
            raise ImportError("source joint limit coverage changed")
        if not joint["limited"]:
            continue
        if source_id not in bindings or joint["type"] not in (2, 3):
            raise ImportError("limited source joint has no scalar native binding")
        binding = bindings[source_id]
        limits = vector(joint["range"], 2)
        if joint["name"] != binding["source_name"] or joint["type"] != binding["source_type"] or \
                limits != binding["source_range"] or binding.get("core_limit_status") not in {
                    "enforced", "retained_in_manifest_not_enforced_at_source_default"}:
            raise ImportError("source joint range/type identity disagrees with native mechanics")
        solref = vector(joint["limit_solref"], 2)
        solimp = vector(joint["limit_solimp"], 5)
        margin = scalar(joint["limit_margin"])
        weight = scalar(joint["limit_dof_invweight0"])
        if limits[0] > limits[1] or weight < 1.1754943508222875e-38 or \
                ((solref[0] > 0) != (solref[1] > 0)):
            raise ImportError("joint-limit range, inverse weight or solref is invalid")
        records.append(struct.pack("<4I16f", binding["core_q_index"], binding["core_v_index"],
            source_id, 0, *limits, margin, weight, *solref, 0, 0, *solimp, 0, 0, 0))
        identities.append({"source_joint_id": source_id, "source_name": joint["name"],
            "core_q_index": binding["core_q_index"], "core_v_index": binding["core_v_index"],
            "legacy_core_limit_status": binding["core_limit_status"]})
    expected = {key for key, value in bindings.items() if value["source_limited"]}
    if {row["source_joint_id"] for row in identities} != expected or not records:
        raise ImportError("NHLIM1 source limit inventory is incomplete")
    payload = struct.pack("<8s10I32s", b"NHLIM1\0\0", 1, nq, nv, len(records), 80,
        len(records), 1, int(policy["refsafe"]), 0, 0,
        bytes.fromhex(digest)) + b"".join(records)
    return {"schema": "numi.human.joint-limit-source-compliance-payload.v1",
        "file": "myosim-fullbody-joint-limits.nhlim", "payload_abi": 1,
        "record_bytes": 80, "joint_count": len(records), "maximum_row_count": 2 * len(records),
        "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
        "policy_id": 1, "solver": policy, "joints": identities}, payload


def main():
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-export", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.source_manifest.read_text())
    rigid = manifest["payloads"]["rigid"]
    raw = (args.source_manifest.parent / rigid["file"]).read_bytes()
    if len(raw) != rigid["bytes"] or hashlib.sha256(raw).hexdigest() != rigid["sha256"]:
        raise ImportError("joint-limit native rigid payload drifted")
    metadata, payload = compile_joint_limits(json.loads(args.source_export.read_text()), manifest)
    metadata["source_manifest_sha256"] = hashlib.sha256(args.source_manifest.read_bytes()).hexdigest()
    metadata["source_export_sha256"] = hashlib.sha256(args.source_export.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / metadata["file"]).write_bytes(payload)
    (args.output / "joint-limits.manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: value for key, value in metadata.items() if key != "joints"}))


if __name__ == "__main__":
    main()
