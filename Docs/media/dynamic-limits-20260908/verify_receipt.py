#!/usr/bin/env python3
"""Check immutable artifacts and independently decode all source limit rows."""
import gzip
import hashlib
import json
import re
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent


def require(value, detail):
    if not value:
        raise ValueError(detail)


def audit_payload(payload, exported, manifest):
    require(len(payload) >= 80, "truncated NHLIM1 envelope")
    header = struct.unpack_from("<8s10I32s", payload)
    magic, abi, nq, nv, count, stride, source_count, policy, flags, r0, r1, digest = header
    source = {j["id"]: j for j in exported["joints"] if j["limited"]}
    mapping = {j["source_joint_id"]: j for j in manifest["core_tree"]["source_joint_map"]}
    require(magic == b"NHLIM1\0\0" and abi == 1 and stride == 80 and policy == 1 and r0 == r1 == 0,
            "NHLIM1 ABI/policy mismatch")
    require(nq == manifest["core_tree"]["nq"] and nv == manifest["core_tree"]["nv"], "native dimensions drifted")
    require(count == source_count == len(source) and len(payload) == 80 + count * stride, "source inventory incomplete")
    require(flags == int(exported["model"]["joint_limit_solver"]["refsafe"]), "REFSAFE changed")
    require(exported["source"] == manifest["source"] and digest.hex() == manifest["source"]["archive_sha256"], "source identity changed")
    seen, legacy_omitted = set(), 0
    for index in range(count):
        offset = 80 + 80 * index
        q, v, identity, reserved = struct.unpack_from("<4I", payload, offset)
        require(identity in source and identity not in seen and reserved == 0, "source row identity invalid")
        seen.add(identity)
        joint, native = source[identity], mapping[identity]
        require(q == native["core_q_index"] and v == native["core_v_index"] and q == v + 1, "native coordinate drifted")
        require(joint["range"] == native["source_range"] and joint["name"] == native["source_name"], "authored range identity drifted")
        values = [*joint["range"], joint["limit_margin"], joint["limit_dof_invweight0"],
                  *joint["limit_solref"], 0, 0, *joint["limit_solimp"], 0, 0, 0]
        require(payload[offset + 16:offset + 80] == struct.pack("<16f", *values), "source parameters changed")
        legacy_omitted += native["core_limit_status"] == "retained_in_manifest_not_enforced_at_source_default"
    require(seen == set(source), "source rows omitted")
    return {"joint_count": count, "maximum_rows": 2 * count, "legacy_omitted": legacy_omitted}


def main():
    receipt = json.loads((HERE / "receipt.json").read_text())
    require(receipt["schema"] == "numi.human.source-limit-receipt.v1", "receipt schema")
    for entry in receipt["artifacts"]:
        path = (HERE / entry["path"]).resolve()
        require(path.is_relative_to(HERE) and path.is_file(), "artifact escaped bundle")
        raw = path.read_bytes()
        require(len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"], entry["path"])
    source_bytes = gzip.decompress((HERE / "source-export.json.gz").read_bytes())
    manifest_bytes = gzip.decompress((HERE / "source-manifest.json.gz").read_bytes())
    require(hashlib.sha256(source_bytes).hexdigest() == receipt["source_export_sha256"], "source export changed")
    require(hashlib.sha256(manifest_bytes).hexdigest() == receipt["source_manifest_sha256"], "native map changed")
    inventory = audit_payload((HERE / "myosim-fullbody-joint-limits.nhlim").read_bytes(),
                              json.loads(source_bytes), json.loads(manifest_bytes))
    require(inventory == {"joint_count": 122, "maximum_rows": 244, "legacy_omitted": 6}, "full source coverage changed")
    details = (HERE / "native-clean-ctest-details.log").read_text()
    records = [json.loads(line) for line in details.splitlines() if line.startswith('{"status":"passed","candidates":')]
    require(len(records) == 1 and records[0]["candidates"] == 101 and records[0]["finite_differences"] == 79, "source GPU coverage")
    ctest = (HERE / "native-clean-tests.log").read_text()
    require("100% tests passed out of 4" in ctest and len(re.findall(r"Test\s+#\d+:.*Passed", ctest)) == 4,
            "native regressions failed")
    tokens = []
    for label in ("transaction-clean", "transaction-replay"):
        launch = json.loads((HERE / (label + "-launch.json")).read_text())
        require(launch["returncode"] == 0, "joint transaction failed")
        for owner in ("native", "brain"):
            require(launch["source_state"][owner]["status"] == "", "qualification source was dirty")
            require(launch["source_state"][owner]["revision"] == receipt[owner + "_commit"], "qualification revision changed")
        log = (HERE / (label + ".log")).read_text()
        require("Executed 2 tests, with 0 failures" in log, "native admission/transaction tests incomplete")
        words = re.findall(r"GateB physical-token words=([^\n]+)", log)
        require(len(words) == 9, "accepted/rejected physical-state evidence incomplete")
        generations = [int(word.split(',')[4], 16) for word in words]
        require(generations == [1, 2, 3, 4, 4, 5, 6, 7, 8] and words[3] == words[4],
                "rejected candidate changed accepted physical identity")
        tokens.append(words)
    require(tokens[0] == tokens[1], "physical replay drift")
    for key in ("standing", "walking", "internal_equilibrium", "dynamic_contact", "prepared_pose_loaded_limits",
                "costal_requalification", "calibration", "performance", "full_release"):
        require(receipt["qualification"][key] is False, "unsupported promotion: " + key)
    require(receipt["costal_attempt"]["disposition"] == "interrupted_competing_workload", "costal scope changed")
    print(json.dumps({"status": "bounded_source_limits_passed", **inventory,
                      "accepted_roots": 8, "physical_replay": "exact", "full_release": False}))


if __name__ == "__main__":
    main()
