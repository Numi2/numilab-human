"""Replay retained typed bindings and carrier witnesses for the exact refit."""
from contextlib import contextmanager
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np

run = Path(__file__).resolve().parent
root = run.parent
mesh = root / "constitutive-witness-repaired-tetgen"
state_path = run / "p1/current-paired-volume-state.json"
state = json.loads(state_path.read_text())

@contextmanager
def open_retained(path):
    if path.is_file():
        with path.open("rb") as stream:
            yield stream
        return
    with subprocess.Popen(["/opt/homebrew/bin/zstd", "-qdc", str(path)+".zst"],
                          stdout=subprocess.PIPE) as archive:
        yield archive.stdout
        if archive.wait() != 0:
            raise RuntimeError(f"Could not restore {path}")


def sha(path):
    with open_retained(Path(path)) as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def read_json(path):
    return json.loads(path.read_text())

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")

def replace_selected(source, destination, transform):
    with open_retained(source) as source_bytes, io.TextIOWrapper(source_bytes, newline="") as inp, \
         destination.open("w", newline="") as out:
        header = next(inp)
        names = next(csv.reader([header]))
        out.write(header)
        writer = csv.DictWriter(out, names, lineterminator="\n")
        for line in inp:
            row = transform(names, line)
            if row is None:
                out.write(line)
            else:
                writer.writerow(row)

def decode(names, line):
    return dict(zip(names, next(csv.reader([line])), strict=True))

moved = read_json(root / "constitutive-witness-surface-delta.json")["moved_nodes"]
geometry = read_json(mesh / "receipt.json")
assert sha(mesh/"mesh.node") == geometry["mesh_node_sha256"]
assert sha(mesh/"mesh.ele") == geometry["mesh_ele_sha256"]
points = np.loadtxt(mesh/"mesh.node", skiprows=1, usecols=(1, 2, 3))
assert np.array_equal(points, points.astype(np.float32))
tets = np.loadtxt(mesh/"mesh.ele", skiprows=1, usecols=(1, 2, 3, 4), dtype=np.int32)-1
affected = np.flatnonzero(np.any(np.isin(tets,moved), axis=1))
centroids = dict(zip(map(int, affected), np.sum(points[tets[affected]] * .25, axis=1), strict=True))
assert len(centroids) == read_json(run/"refit-density-closure.json")["affected_volume_rows"]

def load_parent(key, columns, dtype=float):
    pin = state["source330_identity"][key]
    assert sha(pin["path"]) == pin["sha256"]
    return np.loadtxt(pin["path"], usecols=columns, dtype=dtype, delimiter=",", skiprows=1)

parent_nodes = load_parent("source330_nodes", (2, 3, 4))
parent_tets = load_parent("source330_tets", (3, 4, 5, 6), np.int32)
registration = root.parent/"captured-fat-sliver-495/proposal/compiler/volume-export/registered"
frames = {}
with (registration/"donor-frames.csv").open() as stream:
    for row in csv.DictReader(stream):
        frame = np.array([[float(row[f"current_R{i}{j}"]) for j in range(3)] for i in range(3)])
        origin = np.array([float(row[f"current_{a}"]) for a in "xyz"])
        frames[int(row["donor"])] = (int(row["binding"]), origin, frame, np.linalg.inv(frame))
prior = root/"paired-repaired-registration-live-parent"
old = read_json(prior/"receipt.json")
assert all(sha(prior/name) == digest for name,digest in old["outputs"].items())
sidecars = run/"registration"
sidecars.mkdir()
surface = {role: {} for role in "OID"}
nodes_seen = set()
map_rows = 0
residual_max = 0.
def update_volume(names, line):
    n = int(line.partition(",")[0])-1
    if n not in moved:
        return None
    global map_rows, residual_max
    nodes_seen.add(n)
    map_rows += 1
    row = decode(names, line)
    binding, origin, rotation, inverse = frames[int(row["donor_body"])]
    assert binding == int(row["donor_binding"])
    local = inverse @ (points[n]-origin)
    assert np.max(np.abs(origin + rotation@local - points[n])) < 1e-12
    beta = np.asarray([float(row[f"parent_bary{i}"]) for i in range(4)])
    anchors = parent_nodes[parent_tets[int(row["parent_tet_row"])]]
    residual = points[n] - beta @ anchors
    distance = float(np.linalg.norm(residual))
    assert np.isfinite(distance) and distance <= .02
    residual_max = max(residual_max, distance)
    for i, a in enumerate("xyz"):
        row[f"current_{a}_m"] = repr(float(points[n,i]))
        row[f"donor_local_{a}_m"] = repr(float(local[i]))
        row[f"parent_current_residual_{a}_m"] = repr(float(residual[i]))
    row["parent_registration_distance_m"] = repr(distance)
    role = row["registered_surface_layer"]
    if role in surface:
        key = int(row["registered_surface_vertex_row"])
        assert surface[role].setdefault(key, row) == row or all(
            surface[role][key][k] == row[k] for k in ("donor_body","parent_tet_row",
                "current_x_m","current_y_m","current_z_m","parent_registration_distance_m"))
    return row

replace_selected(prior/"combined-node-map.csv", sidecars/"combined-node-map.csv", update_volume)
assert nodes_seen == set(moved)
role_rows = 0
def update_reg(names, line):
    role, vertex, _ = line.split(",",2)
    changed = surface[role].get(int(vertex))
    if changed is None:
        return None
    global role_rows
    role_rows += 1
    row = decode(names,line)
    for a in "xyz":
        row[f"parent_residual_{a}_m"] = changed[f"parent_current_residual_{a}_m"]
    row["parent_registration_distance_m"] = changed["parent_registration_distance_m"]
    assert row["parent_tet_row"] == changed["parent_tet_row"]
    return row
replace_selected(prior/"node-parent-registration.csv", sidecars/"node-parent-registration.csv", update_reg)
assert role_rows == sum(map(len,surface.values()))
for role, changed in surface.items():
    src = prior/f"registered-{role}.csv"
    dst = sidecars/src.name
    if not changed:
        dst.hardlink_to(src)
        continue
    with open_retained(src) as stream:
        p = np.loadtxt(stream, delimiter=",", skiprows=1, usecols=(2,3,4))
    for row, data in changed.items():
        p[row] = [float(data[f"current_{a}_m"]) for a in "xyz"]
    faces = np.loadtxt(registration/f"registered-{role}-faces.csv", delimiter=",",
                       skiprows=1, usecols=(1,2,3), dtype=np.int32)
    normals = np.zeros_like(p)
    q = p[faces]
    cross = np.cross(q[:,1]-q[:,0], q[:,2]-q[:,0])
    ring = np.unique(faces[np.any(np.isin(faces,list(changed)),axis=1)])
    for corner in faces.T:
        np.add.at(normals, corner, cross)
    lengths = np.linalg.norm(normals[ring],axis=1)
    assert np.all(lengths > 0.)
    normal = dict(zip(map(int,ring),normals[ring]/lengths[:,None],strict=True))
    def update_surface(names, line):
        row_id = int(line.partition(",")[0])
        if row_id not in normal:
            return None
        row = decode(names,line)
        for i,a in enumerate("xyz"):
            row["n"+a] = repr(float(normal[row_id][i]))
            if row_id in changed:
                data = changed[row_id]
                assert row["donor"] == data["donor_body"]
                row["w"+a] = data[f"current_{a}_m"]
                row["l"+a] = data[f"donor_local_{a}_m"]
        return row
    replace_selected(src,dst,update_surface)
for name in ("all-body-current-rest.csv", "source-domain-pins.tsv", "facet-ancestry-current.csv"):
    (sidecars/name).hardlink_to(prior/name)
save_json(sidecars/"receipt.json", {"current_parent_state_sha256":sha(state_path),
    "updated_node_map_rows":map_rows, "updated_role_rows":role_rows,
    "changed_nodes":moved, "maximum_changed_parent_residual_m":residual_max,
    "outputs":{p.name:sha(p) for p in sorted(sidecars.iterdir())}})

old = read_json(root/"paired-live-parent-flow-witness/receipt.json")
closure = read_json(run/"refit-density-closure.json")
assert old["verified_flow_sha256"] == closure["source_sha256"]
limit = read_json(root/"archived-compensation-independent-check.json")["transport_distance_limit_m"]
witnesses = run/"witnesses"
witnesses.mkdir()
result = []
for asset in old["roles"]:
    path, packet = (Path(asset[name + "_path"]) for name in ("witness","carrier"))
    assert sha(path) == asset["witness_sha256"] and sha(packet) == asset["carrier_sha256"]
    with np.load(packet) as contents:
        fields = dict(contents)
    if "regional_compensation_child_rows" in fields:
        assert np.intersect1d(affected,fields["regional_compensation_child_rows"]).size == 0
    replacement = witnesses/path.name
    changed = primary = 0
    maximum = 0.
    with gzip.open(path,"rt",newline="") as inp, replacement.open("wb") as buf, \
         gzip.GzipFile(fileobj=buf,filename="",mode="wb",mtime=0,compresslevel=5) as archive:
        with io.TextIOWrapper(archive,newline="") as out:
            header = next(inp)
            out.write(header)
            cols = {name:i for i,name in enumerate(next(csv.reader([header])))}
            writer = csv.writer(out,lineterminator="\n")
            key = cols["child_global_row"]
            for line in inp:
                n = int(line.split(",",key+1)[key])
                if n not in centroids:
                    out.write(line)
                    continue
                row = next(csv.reader([line]))
                source = np.array([float(row[cols[f"source_{a}_m"]]) for a in "xyz"])
                center = centroids[n]
                distance = float(np.linalg.norm(source-center))
                assert np.isfinite(distance) and distance <= limit
                maximum = max(maximum,distance)
                for i,a in enumerate("xyz"):
                    row[cols[f"child_centroid_{a}_m"]] = repr(float(center[i]))
                row[cols["distance_m"]] = repr(distance)
                writer.writerow(row)
                changed += 1
                if row[0] == "carrier":
                    idx = int(row[cols["packet_carrier_row"]])
                    assert fields["carrier_target_child_rows"][idx] == int(row[cols["child_local_row"]])
                    fields["carrier_target_distance_m"][idx] = distance
                    primary += 1
    if primary:
        packet = witnesses/packet.name
        np.savez_compressed(packet,**fields)
    result.append(dict(role=asset["role"], witness_path=str(replacement),
        witness_sha256=sha(replacement),carrier_path=str(packet),carrier_sha256=sha(packet),
        changed_witness_rows=changed,changed_primary_rows=primary,maximum_changed_distance_m=maximum))
save_json(witnesses/"receipt.json", {"current_state_sha256":sha(state_path),
    "verified_flow_sha256":closure["candidate_sha256"],"changed_centroids":len(centroids),
    "transport_distance_limit_m":limit,"roles":result})
print(json.dumps(result, indent=2))
