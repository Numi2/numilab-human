"""Bind strict frozen-pose residuals to the matching native and source joint rows."""
import hashlib
import json
from pathlib import Path
import struct

trial = Path("strict-equilibrium-witness")
launch = json.loads((trial / "launch.json").read_text())
rigid_path = Path(launch["argv"][1])
rigid = rigid_path.read_bytes()
rigid_hash = hashlib.sha256(rigid).hexdigest()
assert launch["pins"][str(rigid_path)] == rigid_hash
manifest_path = Path("/Users/n/numi-human-separation-20261018/C/publication-source-001/docs/media/numi-human-passive-fit-v4/source-manifest.json")
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
assert rigid[:8] == b"NHRIGID2"
assert rigid[48:80].hex() == manifest["source"]["archive_sha256"]
abi, _, source_bodies, core_bodies, joints, nq, nv = struct.unpack_from("<7I", rigid, 8)
assert abi == 1 and nq == manifest["model"]["source_nq"] and nv == manifest["model"]["source_nv"]
assert source_bodies == manifest["model"]["source_body_count"]
bodies = manifest["core_tree"]["body_order"]
assert len(bodies) == core_bodies
joint_start = 224 + 160 * core_bodies
dof_start = joint_start + 144 * joints
joint_names = {int(row["core_joint_index"]): row["source_name"]
               for row in manifest["core_tree"]["source_joint_map"]}
rows = []
for line in (trial / "stdout.log").read_text().splitlines():
    if not line.startswith("strict_equilibrium_residual "):
        continue
    fields = dict(item.split("=", 1) for item in line.split()[1:])
    v, q = int(fields.pop("v_index")), int(fields.pop("q_index"))
    _, joint, native_q, native_v = struct.unpack_from("<4I", rigid, dof_start + v * 64)
    assert (q, v) == (native_q, native_v)
    child = None
    if joint != 0xffffffff:
        assert joint < joints
        child, = struct.unpack_from("<I", rigid, joint_start + joint * 144 + 4)
    rows.append({
        "velocity_index": v,
        "coordinate_index": q,
        "joint_index": joint,
        "source_joint": joint_names.get(joint),
        "child_body": bodies[child] if child is not None else None,
        **{key: float(value) for key, value in fields.items()},
    })
assert len(rows) == nv
rows.sort(key=lambda row: abs(row["acceleration"]), reverse=True)
result = {
    "rigid_sha256": rigid_hash,
    "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "source_archive_sha256": rigid[48:80].hex(),
    "residual_rows_by_absolute_acceleration": rows,
}
(trial / "residual-owners.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(rows[:8]))
