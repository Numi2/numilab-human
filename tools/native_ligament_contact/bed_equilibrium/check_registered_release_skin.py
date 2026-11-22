"""Measure the presented native outer skin against its actual registered physical coordinates."""
import csv
import hashlib
import json
import mmap
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np

reader, trial = map(Path, sys.argv[1:3])
sys.path.insert(0, str(reader.parent / "src"))
primitives = runpy.run_path(str(reader))["primitives"]
launch = json.loads((trial / "launch.json").read_text())
env = launch["environment"]
prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def geometry_file(key, hash_key):
    path = Path(env[prefix + key])
    if digest(path) != env[prefix + hash_key]:
        raise ValueError("Changed registered source " + key)
    return path

face_path = geometry_file("SURFACE_O_FACES_CSV", "SURFACE_O_FACES_SHA256")
faces = np.loadtxt(face_path, dtype="i4", delimiter=",", skiprows=1, usecols=(1,2,3))
source = geometry_file("SURFACE_O_NODES_CSV", "SURFACE_O_NODES_SHA256")
source_xyz = np.loadtxt(source, dtype="f4", delimiter=",", skiprows=1, usecols=(2,3,4))
outer = np.full(len(source_xyz), -1, dtype="i4")
with geometry_file("COMBINED_NODE_MAP_CSV", "COMBINED_NODE_MAP_SHA256").open() as f:
    for row in csv.DictReader(f):
        if row["registered_surface_layer"] == "O":
            i = int(row["registered_surface_vertex_row"])
            if not (0 <= i < len(outer)) or outer[i] != -1:
                raise ValueError("Duplicate or invalid registered outer vertex")
            outer[i] = int(row["joint_node_id"]) - 1
if outer.min() < 0 or len(np.unique(outer)) != len(outer):
    raise ValueError("Incomplete material-to-surface map")

def coordinates(path):
    with path.open() as f:
        n,d,attributes,flags = map(int, next(f).split())
        records = np.loadtxt(f)
    if d != 3 or attributes or flags or records.shape != (n,4) or not np.array_equal(records[:,0], np.arange(1,n+1)):
        raise ValueError("Unexpected node file " + str(path))
    return records[outer,1:]

def compare(raw, coords, poses):
    ((_,_,xyz,tri),) = primitives(raw, poses, skin_only=True)
    if not np.array_equal(tri, faces) or xyz.shape != coords.shape:
        raise ValueError("Rendered skin has different topology")
    delta = np.linalg.norm(xyz.astype("f8") - coords,axis=1)
    ref = np.linalg.norm(xyz.astype("f8") - source_xyz,axis=1)
    return dict(vertices=len(xyz),faces=len(faces),same_indices=True,
                maximum_physical_error_m=float(delta.max()),rms_physical_error_m=float(np.sqrt(np.mean(delta**2))),
                beyond_two_microns=int((delta>2e-6).sum()),worst_vertex=int(delta.argmax()),
                maximum_from_original_outer_m=float(ref.max()))

pose_path = Path(env["NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"])
pose_id = digest(pose_path)
if pose_id != launch["pins"].get(str(pose_path)) or "pose_sha256="+pose_id not in (trial/"stdout.log").read_text():
    raise ValueError("Initial shape lacks matched native install")
snapshot = json.loads((trial/"registered-initial-bodies.json").read_text())
poses = {r["body_index"]: dict(position_m=r["current"]["position_world_m"],quaternion_xyzw=r["current"]["orientation_world_xyzw"]) for r in snapshot["all_bodies"]}
pack = trial/"initial-material-display.mrvpack"
with pack.open("rb") as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as raw:
    info = dict(initial=compare(raw,coordinates(pose_path),poses),initial_pack_sha256=hashlib.sha256(raw).hexdigest(),initial_pose_sha256=pose_id)
accept = json.loads((trial/"admission.json").read_text())
if accept.get("terminal_pack_sha256"):
    n=accept["accepted_steps"]
    receipt=json.loads((trial/"native/accepted-geometry"/f"step-{n}.receipt.json").read_text())
    raw=subprocess.check_output(["/opt/homebrew/bin/zstd","-dcq",str(trial/"terminal.mrvpack.zst")])
    id=hashlib.sha256(raw).hexdigest()
    if info["initial_pack_sha256"] != receipt["base_pack_file_sha256"]:
        raise ValueError("Opening frame does not belong to accepted native transaction")
    if id != receipt["pack_file_sha256"] or id != accept["terminal_pack_sha256"] or receipt["accepted_step"]!=n:
        raise ValueError("Unmatched accepted capture")
    step_delta=np.linalg.norm(coordinates(trial/"physical-accepted.node")-coordinates(pose_path),axis=1)
    info.update(maximum_physical_outer_motion_m=float(step_delta.max()), rms_physical_outer_motion_m=float(np.sqrt(np.mean(step_delta**2))))
    info.update(accepted=compare(raw,coordinates(trial/"physical-accepted.node"),{r["body_index"]:r for r in receipt["accepted_body_poses"]}),
                accepted_step=n,accepted_pack_sha256=id,accepted_nodes_sha256=digest(trial/"physical-accepted.node"))
(trial/"release-skin-parity.json").write_text(json.dumps(info,indent=2)+"\n")
print(json.dumps(info))
