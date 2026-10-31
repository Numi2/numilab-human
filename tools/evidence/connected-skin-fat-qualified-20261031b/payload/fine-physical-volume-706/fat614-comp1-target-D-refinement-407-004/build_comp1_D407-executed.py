#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, time, importlib.util, sys
import numpy as np

H = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
OUT = H / "fat614-comp1-target-D-refinement-407-004"
OUT.mkdir(parents=True, exist_ok=False)

D_PATH = H / "d-shortedge-quotient-385-run006/typed-rebind-003/candidate-D-typed-current.npz"
I_PATH = H / "paired-I-interface-399-root/iteration002/typed365/candidate-I-typed-current.npz"
O_PATH = H / "joint-native-OI-ears-279-root/O-batch006-dp001/typed-280-003/candidate-O-typed-current.npz"
V406_PATH = H / "current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz"
VOLD_PATH = H / "fat614-yy-volume-replay-004/run-014-current-target-rebound-v3/fat-D307-I377-YY-components-current-target-rebound.npz"
R026_PATH = H / "fat614-yy-volume-replay-004/run-026-single-Y-diagnostic/receipt.json"
R028_PATH = H / "fat614-yy-volume-replay-004/run-028-single-Y-boundary-refinement-witness/report.json"
CORR_PATH = H / "fat614-yy-volume-replay-004/run-031-boundary-f32-kernel/boundary-node-correction.npz"
CORR_REPORT_PATH = H / "fat614-yy-volume-replay-004/run-031-boundary-f32-kernel/report.json"
NODE_PATH = H / "fat614-yy-volume-replay-004/run-026-single-Y-diagnostic/cavity.1.node"
FACE_PATH = H / "fat614-yy-volume-replay-004/run-026-single-Y-diagnostic/cavity.1.face"
SCHEMA_PATH = H / "quality-warning-connected-union-038-typed-metadata-005-001/D-run-001/candidate-D-connected-union-typed.npz"
AUDIT_SCRIPT = H / "current-volume-boundary-remesh-032/derived-D-pY-edge-split-001/exact-audit-001/audit_changed_d_surface_exact_appended.py"
CONFIG_TEMPLATE = H / "d-shortedge-quotient-385-run006/run-config.json"
PY = Path("/Users/numi/numi-human-prep-venv-20261005/bin/python")

PINS = {
    D_PATH: "9a5ca486d7b2025a7a2140798155df79c208871703f1eaf75171761b388ba7eb",
    I_PATH: "fec15d6b59e9094952845cbeaaf418b91a4b2f5d0405d1c58b7e3e2e752ec406",
    O_PATH: "4e0c46662721fed33d4bd4efb390c6668ff67581d2484fcd9ba9058834fa7a20",
    V406_PATH: "5dcebfd751dedcb1891559352e9d0e673ea3a8e15a8edeae8179c6d04e847211",
    VOLD_PATH: "8915c3c7a58f8622d0568da5006bf4b7b0f2832f802b65e29bd144e95f11001d",
    R026_PATH: "4cc147c3213297865bd29e4eb5e4cc30109b03e5d439602940bc97eaa28e774a",
    R028_PATH: "e5a30bfeb03bbae4a28917cb512f275da454b664286e4d8e787dc3caff534ea3",
    CORR_PATH: "817cde16f36b0ee0b576a87b9f8374f98284520360a60126bb4af5561f60d72f",
    CORR_REPORT_PATH: "8e9496fd001d477193c5931a3e7ae7c765b98d4f577102e85c31cc4d57d3f1c8",
    NODE_PATH: "f2e0b13f47ef30438f0ef9c7c2d849d85bd4df0caf4c2b39404b6a04ffb1e552",
    FACE_PATH: "539d128fbfb242be7bd7f3e60b173050aa21e12bcb77f1354ab51118ee959a42",
    SCHEMA_PATH: "d0f6cda10552a55be2d6ac5a0740e0142293cdf55177f1ce853a3eb5624374da",
}
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k], copy=True) for k in z.files}
def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
def coord_key(x):
    return np.ascontiguousarray(np.asarray(x, np.float32)).tobytes()
def cyclic(t):
    t = tuple(map(int, t))
    return min((t, (t[1], t[2], t[0]), (t[2], t[0], t[1])))
def reverse_cyclic(t):
    t = tuple(map(int, t))
    return cyclic((t[0], t[2], t[1]))
def directed_patch_boundary(faces):
    from collections import defaultdict
    edges = defaultdict(list)
    for tri in np.asarray(faces, np.int64):
        a, b, c = map(int, tri)
        for u, v in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted((u, v)))].append((u, v))
    out = {}
    for e, d in edges.items():
        if len(d) == 1:
            out[e] = tuple(d)
        elif len(d) == 2:
            if d[0][0] == d[1][0]:
                out[e] = tuple(sorted(d))
        else:
            out[e] = tuple(sorted(d))
    return out
def closest_triangle(p, tri):
    # Ericson closest-point construction; returns barycentrics and nearest point.
    a, b, c = np.asarray(tri, np.float64)
    p = np.asarray(p, np.float64)
    ab = b - a; ac = c - a; ap = p - a
    d1 = ab @ ap; d2 = ac @ ap
    if d1 <= 0 and d2 <= 0: return np.array([1.,0.,0.]), a
    bp = p - b; d3 = ab @ bp; d4 = ac @ bp
    if d3 >= 0 and d4 <= d3: return np.array([0.,1.,0.]), b
    vc = d1*d4 - d3*d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1/(d1-d3); return np.array([1-v,v,0.]), a+v*ab
    cp = p - c; d5 = ab @ cp; d6 = ac @ cp
    if d6 >= 0 and d5 <= d6: return np.array([0.,0.,1.]), c
    vb = d5*d2 - d1*d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2/(d2-d6); return np.array([1-w,0,w]), a+w*ac
    va = d3*d6 - d5*d4
    if va <= 0 and d4-d3 >= 0 and d5-d6 >= 0:
        w = (d4-d3)/((d4-d3)+(d5-d6)); return np.array([0.,1-w,w]), b+w*(c-b)
    den = 1.0/(va+vb+vc); v = vb*den; w = vc*den; u = 1-v-w
    return np.array([u,v,w]), u*a+v*b+w*c

for path, want in PINS.items():
    got = sha(path)
    if got != want:
        raise ValueError(f"pin mismatch {path}: {got} != {want}")
D = load_npz(D_PATH); I = load_npz(I_PATH); O = load_npz(O_PATH)
V406 = load_npz(V406_PATH); Vold = load_npz(VOLD_PATH)
r026 = json.loads(R026_PATH.read_text()); r028 = json.loads(R028_PATH.read_text())
corr_report = json.loads(CORR_REPORT_PATH.read_text()); corr = load_npz(CORR_PATH)

assert r026["status"] == "DIAGNOSTIC_ONLY" and r026["physical_admission"] is False and r026["native_steps"] == 0
assert r028["status"] == "PASS_LOCAL_BOUNDARY_REFINEMENT_RECONSTRUCTION_DIAGNOSTIC_ONLY"
assert corr_report["status"] == "PASS_LOCAL_F32_KERNEL_CANDIDATE_ONLY"
assert corr_report["physical_admission"] is False and corr_report["native_steps"] == 0
assert corr_report["corrected_float32_exact_sign_counts"] == {"-1":0,"0":0,"1":2153}

P = np.asarray(D["points_world_m_f32"], np.float32)
P64 = np.asarray(D["points_world_m"], np.float64)
F = np.asarray(D["faces"], np.int64)
VoldP = np.asarray(Vold["points_world_m_f32"], np.float32)
VnewP = np.asarray(V406["points_world_m_f32"], np.float32)
D_to_v406 = np.asarray(V406["fat_current_target_rebind_D_point_to_volume_rows"], np.int64)
assert len(D_to_v406) == len(P) and np.all(D_to_v406 >= 0)
assert np.array_equal(P, VnewP[D_to_v406])
coord_to_D = {coord_key(x): i for i, x in enumerate(P)}
if len(coord_to_D) != len(P): raise ValueError("current D point F32 keys are not unique")
oldV_to_D = np.asarray([coord_to_D.get(coord_key(x), -1) for x in VoldP], np.int64)

# Reidentify the five old comp1 target parents by exact current D coordinates and face support.
parent_face_rows = []
parent_volume_face_rows = []
parent_volume_point_rows = []
for entry in r028["missing_input_target_faces"]:
    assert entry["kind"] == "typed-target"
    gv = np.asarray(entry["global_point_rows"], np.int64)
    dr = oldV_to_D[gv]
    if np.any(dr < 0) or len(set(map(int, dr))) != 3:
        raise ValueError(f"old target face points do not map uniquely into current D: {gv.tolist()} -> {dr.tolist()}")
    hit = np.flatnonzero(np.all(np.sort(F, axis=1) == np.sort(dr), axis=1))
    if len(hit) != 1: raise ValueError(f"current D parent face map count={len(hit)} for {dr.tolist()}")
    parent_face_rows.append(int(hit[0])); parent_volume_point_rows.append(gv.tolist())
    group = np.asarray(V406["desired_boundary_source_group"], np.int64)
    sfrow = np.asarray(V406["desired_boundary_source_face_rows"], np.int64)
    trg = np.flatnonzero((group == int(entry["target_source"]["group"])) &
                         (sfrow == int(hit[0])))
    if len(trg) != 1: raise ValueError(f"current target D face row map count={len(trg)} for D face {int(hit[0])}")
    parent_volume_face_rows.append(int(trg[0]))
parent_face_rows = np.asarray(parent_face_rows, np.int64)
parent_volume_face_rows = np.asarray(parent_volume_face_rows, np.int64)
if len(np.unique(parent_face_rows)) != 5: raise ValueError("expected five distinct current D parent faces")
if set(parent_face_rows.tolist()) != {234144, 232293, 232188, 230875, 230877}:
    raise ValueError(f"unexpected current D parent rows: {parent_face_rows.tolist()}")
# The rebound desired boundary must be the reverse-oriented Fat copy of each D parent face.
def require_reverse_target(drow, targetrow):
    dtri = D_to_v406[F[int(drow)]]
    ttri = np.asarray(V406["desired_boundary_faces"][int(targetrow)], np.int64)
    if cyclic(ttri) != reverse_cyclic(dtri):
        raise ValueError(f"current Fat/D orientation relationship changed for D row {drow}: {dtri.tolist()} / {ttri.tolist()}")
for dr, tr in zip(parent_face_rows, parent_volume_face_rows):
    require_reverse_target(int(dr), int(tr))

# Authenticate the actual TetGen table rows against the frozen report.
node_table = np.loadtxt(NODE_PATH, skiprows=1, dtype=np.float64)
face_table = np.loadtxt(FACE_PATH, skiprows=1, dtype=np.float64)
node_ids = node_table[:, 0].astype(np.int64)
node_xyz = node_table[:, 1:4].astype(np.float32)
node_by_id = {int(q): node_xyz[i] for i, q in enumerate(node_ids)}
if len(node_by_id) != len(node_ids): raise ValueError("duplicate TetGen node ID")
face_ids = face_table[:, 0].astype(np.int64)
face_nodes = face_table[:, 1:4].astype(np.int64)
face_by_id = {int(q): tuple(map(int, face_nodes[i])) for i, q in enumerate(face_ids)}
if len(face_by_id) != len(face_ids): raise ValueError("duplicate TetGen face ID")
for rec in r028["new_output_boundary_faces"]:
    if tuple(map(int, rec["tetgen_node_ids"])) != face_by_id[int(rec["tetgen_face_id"])]:
        raise ValueError(f"TetGen face row disagrees with report at face {rec['tetgen_face_id']}")
    for gv, nid in zip(rec["input_global_point_rows"], rec["tetgen_node_ids"]):
        if gv is not None and not np.array_equal(node_by_id[int(nid)].view(np.uint32), VoldP[int(gv)].view(np.uint32)):
            raise ValueError(f"TetGen retained point node {nid} differs from pinned old volume row {gv}")
raw_A = np.asarray(node_by_id[1026], np.float32)
raw_B = np.asarray(node_by_id[1027], np.float32)
raw_C = np.asarray(node_by_id[1028], np.float32)
corrected_A = np.asarray(corr["corrected_f32"], np.float32)
if not np.array_equal(raw_A.view(np.uint32), np.asarray(corr["original_emitted_f32"], np.float32).view(np.uint32)):
    raise ValueError("run031 correction does not start from the exact run026 node1026 bits")
if not np.array_equal(raw_B.view(np.uint32), np.asarray(r026["output_boundary_steiner_nodes"][1]["xyz_f32"], np.float32).view(np.uint32)):
    raise ValueError("node1027 differs from the run026 receipt")
if not np.array_equal(raw_C.view(np.uint32), np.asarray(r026["output_boundary_steiner_nodes"][2]["xyz_f32"], np.float32).view(np.uint32)):
    raise ValueError("node1028 differs from the run026 receipt")
new_points32 = np.asarray([corrected_A, raw_B, raw_C], np.float32)
new_points64 = new_points32.astype(np.float64)
npoint = len(P); nface = len(F)
if np.any(np.all(P[:, None, :] == new_points32[None, :, :], axis=2)):
    raise ValueError("a generated D point already exists in the current D point set")
for role, arr in (("O", O["points_world_m_f32"]), ("I", I["points_world_m_f32"])):
    if np.any(np.all(np.asarray(arr, np.float32)[:, None, :] == new_points32[None, :, :], axis=2)):
        raise ValueError(f"a generated D point exactly duplicates a current {role} point")
if not (len(r028["new_output_boundary_faces"]) == 11 and r028["boundary_steiner_node_ids"] == [1026,1027,1028]):
    raise ValueError("unexpected comp1 11-face / 3-boundary-point patch")

# Map the eleven local TetGen boundary facets into the current D point namespace.
child_faces = []
for rec in r028["new_output_boundary_faces"]:
    tri = []
    for gv, nid in zip(rec["input_global_point_rows"], rec["tetgen_node_ids"]):
        if gv is None:
            nid = int(nid)
            if nid not in (1026,1027,1028): raise ValueError(f"unexpected nonboundary TetGen node {nid}")
            tri.append(npoint + (nid - 1026))
        else:
            row = int(oldV_to_D[int(gv)])
            if row < 0: raise ValueError(f"retained TetGen vertex {gv} has no current D row")
            tri.append(row)
    child_faces.append(tri)
child_faces = np.asarray(child_faces, np.int64)
if len({tuple(sorted(map(int,t))) for t in child_faces}) != 11: raise ValueError("duplicate mapped TetGen child face")
# A single fixed-coordinate 2-2 face flip removes the only joint-apex obstruction
# while preserving the exact oriented boundary of the 11-face patch.
old_child_faces = child_faces.copy()
child_faces[2] = np.asarray([npoint+2, 102543, npoint], np.int64)   # C, old D row 102543, A
child_faces[8] = np.asarray([102543, npoint+2, 102286], np.int64)   # old D row 102543, C, old D row 102286
patch_boundary_before = directed_patch_boundary(old_child_faces)
patch_boundary_after = directed_patch_boundary(child_faces)
parent_patch_boundary = directed_patch_boundary(F[parent_face_rows])
if patch_boundary_before != parent_patch_boundary:
    raise ValueError("raw TetGen child patch directed boundary differs from current D parent patch")
if patch_boundary_after != parent_patch_boundary:
    raise ValueError("2-2 child face flip did not preserve the exact oriented parent patch boundary")
if len({tuple(sorted(map(int,t))) for t in child_faces}) != 11:
    raise ValueError("duplicate child faces after local diagonal flip")

# Measure per-point geometric support error against the five current D parent triangles.
supports = []
for pi, point in enumerate(new_points32.astype(np.float64)):
    options = []
    for face_row in parent_face_rows:
        tri = P[F[int(face_row)]].astype(np.float64)
        bary, nearest = closest_triangle(point, tri)
        dist = float(np.linalg.norm(point - nearest))
        options.append((dist, int(face_row), bary, nearest))
    dist, face_row, bary, nearest = min(options, key=lambda z: (z[0], z[1]))
    if np.min(bary) < -1e-12 or abs(float(np.sum(bary))-1.0) > 1e-12:
        raise ValueError(f"invalid closest-triangle barycentric support for new point {pi}")
    supports.append({"new_point_row": npoint+pi, "tetgen_node_id": 1026+pi,
                     "nearest_current_D_parent_face_row": face_row,
                     "barycentric_on_parent_face": bary.tolist(),
                     "nearest_parent_surface_point_world_m": nearest.tolist(),
                     "f32_point_world_m": point.tolist(),
                     "distance_to_parent_surface_m": dist})
# Keep original face order except the five explicitly retired parents; append eleven derived faces.
keep_face_rows = np.setdiff1d(np.arange(nface, dtype=np.int64), np.sort(parent_face_rows), assume_unique=True)
Fout = np.vstack((F[keep_face_rows], child_faces)).astype(np.int64)
new_face_rows = np.arange(len(keep_face_rows), len(Fout), dtype=np.int64)
Pout32 = np.vstack((P, new_points32)).astype(np.float32)
Pout64 = np.vstack((P64, new_points64)).astype(np.float64)
point_parent = np.r_[np.arange(npoint, dtype=np.int64), np.full(3, -1, np.int64)]
face_parent = np.r_[keep_face_rows, np.full(11, -1, np.int64)]
if not np.array_equal(Pout32[point_parent >= 0], P[point_parent[point_parent >= 0]]):
    raise ValueError("retained D point prefix changed")
# Whole-surface duplicate/topology invariants and unchanged directed boundary.
tri_keys = [tuple(sorted(map(int,t))) for t in Fout]
if len(set(tri_keys)) != len(tri_keys): raise ValueError("whole D candidate has duplicate triangle")
old_db = directed_patch_boundary(F)
new_db = directed_patch_boundary(Fout)
if old_db != new_db: raise ValueError("D whole-surface directed boundary changed")
# Every edge has one or two faces and interior edges oppose; compare topology counts.
from collections import defaultdict
edge_dirs = defaultdict(list)
for tri in Fout:
    a,b,c = map(int,tri)
    for u,v in ((a,b),(b,c),(c,a)): edge_dirs[tuple(sorted((u,v)))].append((u,v))
bad_edges = [(e,d) for e,d in edge_dirs.items() if len(d) > 2 or (len(d)==2 and d[0][0]==d[1][0])]
if bad_edges: raise ValueError(f"nonmanifold or same-direction D edges after patch: {len(bad_edges)}")
# Joint native apex threshold on every new child and the complete current D surface.
J_path = H / "audit_joint_native_apex_272_root.py"
spec = importlib.util.spec_from_file_location("joint272_comp1_refinement", J_path)
J = importlib.util.module_from_spec(spec); sys.modules[spec.name] = J; spec.loader.exec_module(J)
child_score = np.asarray(J.score(Pout32, child_faces), np.float64)
all_score = np.asarray(J.score(Pout32, Fout), np.float64)
if len(child_score) != 11 or np.any(child_score < J.GATE):
    raise ValueError(f"generated child faces fail joint native apex gate: {child_score.tolist()}")
if np.any(all_score < J.GATE):
    raise ValueError(f"whole D surface has joint apex failures after patch: {int(np.count_nonzero(all_score<J.GATE))}")
tri = Pout32[child_faces].astype(np.float64)
aa = tri[:,1]-tri[:,0]; bb = tri[:,2]-tri[:,0]; cc = tri[:,2]-tri[:,1]
length = np.maximum.reduce((np.linalg.norm(aa,axis=1),np.linalg.norm(bb,axis=1),np.linalg.norm(cc,axis=1)))
qface = np.linalg.norm(np.cross(aa,bb),axis=1)/(length*length)
if not np.all(np.isfinite(qface)) or np.any(qface <= 0): raise ValueError("degenerate generated D child face")
# Flatten the complete typed source union and full original face ancestry of all five retired parents.
lanes = ("layer","body","bone","face_id","original_id","identity0","identity1")
ancestry_lanes = ("face_ancestry_source_face_rows","face_ancestry_source_original_ids","face_ancestry_parent_face_rows")
parent_unions = []
parent_ancestry = []
for r in parent_face_rows:
    lo, hi = map(int, D["quality_star_patch_union_offsets"][int(r):int(r)+2])
    if hi > lo:
        parent_unions.extend(tuple(int(D[f"quality_star_patch_union_face_source_{lane}"][j]) for lane in lanes) for j in range(lo,hi))
    else:
        parent_unions.append(tuple(int(D[f"face_source_{lane}"][int(r)]) for lane in lanes))
    alo, ahi = map(int, D["face_ancestry_offsets"][int(r):int(r)+2])
    parent_ancestry.extend(tuple(int(D[k][j]) for k in ancestry_lanes) for j in range(alo,ahi))
parent_unions = sorted(set(parent_unions))
parent_ancestry = sorted(set(parent_ancestry))
if not parent_unions or not parent_ancestry: raise ValueError("retired D parents lack typed union or full face ancestry")
for lane in ("layer","body","bone"):
    vals = {int(D[f"face_source_{lane}"][int(r)]) for r in parent_face_rows}
    if len(vals) != 1: raise ValueError(f"cannot assign one derived primary {lane} descriptor: {vals}")
primary = {lane: int(D[f"face_source_{lane}"][int(parent_face_rows[0])]) for lane in ("layer","body","bone")}
all_orig = np.concatenate((np.asarray(D["face_source_original_id"],np.int64),
                           np.asarray(D["quality_star_patch_union_face_source_original_id"],np.int64)))
synthetic_id = int(np.max(all_orig)) + 1
schema_src = load_npz(SCHEMA_PATH)
schema_key = "face_source_identity_schema_by_original_id_json"
schema_raw = np.asarray(schema_src[schema_key])
if schema_raw.size != 1: raise ValueError("base identity registry is not scalar")
schema = json.loads(str(schema_raw.item()))
if str(synthetic_id) in schema: raise ValueError("derived original ID collides with existing registry key")
union_json = json.dumps([list(x) for x in parent_unions], separators=(",",":"), ensure_ascii=True)
union_sha = hashlib.sha256(union_json.encode()).hexdigest()
ancestry_json = json.dumps([list(x) for x in parent_ancestry], separators=(",",":"), ensure_ascii=True)
ancestry_sha = hashlib.sha256(ancestry_json.encode()).hexdigest()
patch_id = "D-comp1-current-target-steiner-boundary-refinement-407"
schema_addition = {
    "role":"DERIVED_PATCH",
    "derived_operation":"TetGen boundary Steiner target refinement with one in-patch 2-2 diagonal flip",
    "face_count":11,
    "identity_keys":["face_source_identity0","face_source_identity1"],
    "parent_union_tuple_count":len(parent_unions),
    "parent_union_tuple_sha256":union_sha,
    "parent_support_semantics":"All eleven generated D faces carry the complete deduplicated seven-lane typed union and full flattened source-face ancestry of the five retired target parents. This is patch-level support; no exact per-triangle source-face overlap is claimed.",
    "source_lineage":{"parent_candidate_path":str(D_PATH),"parent_candidate_sha256":sha(D_PATH),
                      "parent_face_rows":parent_face_rows.tolist(),
                      "current_fat_target_face_rows":parent_volume_face_rows.tolist()}
}
schema[str(synthetic_id)] = schema_addition
schema_json = json.dumps(schema, sort_keys=True, separators=(",",":"), ensure_ascii=True)
schema_digest = hashlib.sha256(schema_json.encode()).hexdigest()
# Rebuild output CSR, retaining all unchanged rows byte-for-byte in value and tuple order.
base_union_offsets = np.asarray(D["quality_star_patch_union_offsets"], np.int64)
base_anc_offsets = np.asarray(D["face_ancestry_offsets"], np.int64)
parent_face_set = set(map(int,parent_face_rows))
all_union_rows=[]; all_anc_rows=[]
for outrow, parentrow in enumerate(face_parent):
    if parentrow >= 0:
        r=int(parentrow)
        lo,hi=map(int,base_union_offsets[r:r+2])
        if hi>lo:
            tuples=[tuple(int(D[f"quality_star_patch_union_face_source_{lane}"][j]) for lane in lanes) for j in range(lo,hi)]
        else:
            tuples=[tuple(int(D[f"face_source_{lane}"][r]) for lane in lanes)]
        alo,ahi=map(int,base_anc_offsets[r:r+2])
        anc=[tuple(int(D[k][j]) for k in ancestry_lanes) for j in range(alo,ahi)]
    else:
        tuples=parent_unions; anc=parent_ancestry
    all_union_rows.append(tuples);all_anc_rows.append(anc)
union_offsets=np.zeros(len(Fout)+1,np.int64); union_flat=[]
anc_offsets=np.zeros(len(Fout)+1,np.int64); anc_flat=[]
for i,(tuples,anc) in enumerate(zip(all_union_rows,all_anc_rows)):
    union_flat.extend(tuples); union_offsets[i+1]=len(union_flat)
    anc_flat.extend(anc);anc_offsets[i+1]=len(anc_flat)
union_array=np.asarray(union_flat,np.int64).reshape((-1,7))
anc_array=np.asarray(anc_flat,np.int64).reshape((-1,3))
# The secondary CSR is only for the 11 generated rows; take it from the full
# patch union directly so preceding retained rows cannot shift its values.
new_union_offsets=np.r_[0,np.cumsum([len(parent_unions)]*11)].astype(np.int64)
new_union_array=np.tile(np.asarray(parent_unions,np.int64).reshape((-1,7)),(11,1))
# Primary face descriptors, preserving all unmodified primary fields exactly.
primary_fields={}
for lane in lanes:
    old=np.asarray(D[f"face_source_{lane}"])
    suffix=np.asarray([primary.get(lane, -1)]*11, dtype=old.dtype)
    primary_fields[f"face_source_{lane}"]=np.concatenate((old[keep_face_rows],suffix))
for j,outrow in enumerate(new_face_rows):
    primary_fields["face_source_face_id"][outrow] = -1
    primary_fields["face_source_original_id"][outrow] = synthetic_id
    primary_fields["face_source_identity0"][outrow] = int(outrow)
    primary_fields["face_source_identity1"][outrow] = 0
point_input = np.asarray(D["quality_star_point_output_to_input_row"],np.int64)
face_input = np.asarray(D["quality_star_face_output_to_input_row"],np.int64)
point_input_out=np.r_[point_input,np.full(3,-1,np.int64)]
face_input_out=np.r_[face_input[keep_face_rows],np.full(11,-1,np.int64)]
face_input_source_out=face_input_out.copy()
# New faces use the complete five-parent patch union and full ancestry. Preserve exact point lineage only for retained points.
newrows = new_face_rows
new_parent_point_rows = point_parent[Fout[newrows]]
new_input_point_rows = point_input_out[Fout[newrows]]
if np.any(new_parent_point_rows < -1) or np.any(new_input_point_rows < -1):
    raise ValueError("invalid derived new-face point ancestry map")
# Operation source document is written before NPZ to avoid a circular hash.
op_path=OUT/(patch_id+".json")
operation={
 "schema":"numi.D-current-target-steiner-patch-refinement-operation.v1",
 "status":"DERIVED_BOUNDARY_PATCH_PROPOSAL_FULL_CONTEXT_PENDING",
 "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
 "operation":patch_id,"role":"D","parent_candidate":{"path":str(D_PATH),"sha256":sha(D_PATH)},
 "current_full_context":{"O":{"path":str(O_PATH),"sha256":sha(O_PATH)},"I":{"path":str(I_PATH),"sha256":sha(I_PATH)}},
 "current_fat_target":{"path":str(V406_PATH),"sha256":sha(V406_PATH),
                       "parent_target_face_rows":parent_volume_face_rows.tolist(),
                       "target_face_source_group":1,
                       "target_face_source_D_face_rows":parent_face_rows.tolist()},
 "retired_parent_face_rows":parent_face_rows.tolist(),
 "parent_face_triangles_D_rows":F[parent_face_rows].tolist(),
 "new_output_face_rows":newrows.tolist(),
 "new_output_faces_D_point_rows":child_faces.tolist(),
 "child_face_retriangulation":{"source_tetgen_output_face_count":11,
    "source_run028_face_indices_reflipped":[2,8],
    "flipped_diagonal":{"retired_edge_D_rows":[npoint,102286],"new_edge_D_rows":[npoint+2,102543]},
    "replacement_faces_D_rows":[child_faces[2].tolist(),child_faces[8].tolist()],
    "reason":"The raw diagnostic triangulation had one generated face below the unchanged joint native-apex gate. This single in-patch 2-2 flip preserves the exact oriented patch boundary and raises the minimum generated-face score above the gate."},
 "new_points":[
   {"output_D_point_row":npoint+i,"tetgen_node_id":1026+i,
    "world_f32_m":new_points32[i].tolist(),"world_f64_m":new_points64[i].tolist(),
    "raw_tetgen_world_f32_m":np.asarray([raw_A,raw_B,raw_C][i],np.float32).tolist(),
    "coordinate_adjustment":{"source":"run031 exact-F32 local kernel correction" if i==0 else "none",
       "ulp_offsets_xyz":np.asarray(corr["ulp_offsets_xyz"],np.int64).tolist() if i==0 else [0,0,0],
       "distance_from_raw_emitted_m":float(np.linalg.norm(new_points32[i].astype(np.float64)-np.asarray([raw_A,raw_B,raw_C][i],np.float32).astype(np.float64)))},
    "source_surface_support":supports[i]}
   for i in range(3)],
 "complete_typed_parent_union":parent_unions,"parent_union_sha256":union_sha,
 "complete_source_face_ancestry_union":parent_ancestry,"source_face_ancestry_union_sha256":ancestry_sha,
 "synthetic_original_id":synthetic_id,"synthetic_identity_schema_addition":schema_addition,
 "source_diagnostics":{
  "run026_receipt":{"path":str(R026_PATH),"sha256":sha(R026_PATH)},
  "run028_report":{"path":str(R028_PATH),"sha256":sha(R028_PATH)},
  "run031_correction":{"path":str(CORR_PATH),"sha256":sha(CORR_PATH),"report_path":str(CORR_REPORT_PATH),"report_sha256":sha(CORR_REPORT_PATH)},
  "run026_node_table":{"path":str(NODE_PATH),"sha256":sha(NODE_PATH)},
  "run026_face_table":{"path":str(FACE_PATH),"sha256":sha(FACE_PATH)},
  "original_current_target_volume":{"path":str(VOLD_PATH),"sha256":sha(VOLD_PATH)},
 },
 "scope_semantics":"The inserted points are inferred construction coordinates from the bounded TetGen diagnostic. Their nearest source-triangle barycentric support and residual are recorded; they are not exact points on the original target surface and no measured single-face identity is asserted. The 2-2 face flip changes only the local triangulation of the same patch boundary.",
 "checks":{"current_D_parent_face_crosswalk_exact":True,"current_target_source_rows_exact":True,
  "all_3_new_points_unique_vs_current_D_O_I":True,"all_11_new_faces_non_degenerate":True,
  "patch_oriented_boundary_equal_to_retired_parent_patch":True,"whole_D_directed_boundary_unchanged":True,
  "whole_D_duplicate_faces":0,"whole_D_bad_edges":0,
  "joint_native_apex_gate":float(J.GATE),"minimum_new_face_joint_apex_score":float(child_score.min()),
  "new_face_joint_apex_failures":int(np.count_nonzero(child_score<J.GATE)),
  "whole_D_joint_apex_failures":int(np.count_nonzero(all_score<J.GATE)),
  "full_O_I_D_walls_all185_exact_context":"PENDING",
  "actual_697_load":"PENDING","volume_target_transfer":"PENDING"},
 "physical_admission":False,"native_steps":0
}
save_json(op_path,operation); op_sha=sha(op_path)
# Assemble the output typed D candidate.
payload={k:np.asarray(v) for k,v in D.items()}
payload.update({
 "points_world_m":Pout64,"points_world_m_f32":Pout32,"faces":Fout,
 "point_output_to_parent_rows":point_parent,"face_output_to_parent_rows":face_parent,
 "retired_parent_face_rows":np.sort(parent_face_rows),
 "changed_parent_face_rows":newrows,
 "new_faces_parent_point_rows":new_parent_point_rows,
 "new_faces_input_point_rows":new_input_point_rows,
 "new_faces_output_point_rows":Fout[newrows],
 "new_face_patch_union_offsets":new_union_offsets,
 "new_face_patch_union_face_source_layer":new_union_array[:,0],
 "new_face_patch_union_face_source_body":new_union_array[:,1],
 "new_face_patch_union_face_source_bone":new_union_array[:,2],
 "new_face_patch_union_face_source_face_id":new_union_array[:,3],
 "new_face_patch_union_face_source_original_id":new_union_array[:,4],
 "new_face_patch_union_face_source_identity0":new_union_array[:,5],
 "new_face_patch_union_face_source_identity1":new_union_array[:,6],
 "quality_star_point_output_to_input_row":point_input_out,
 "quality_star_face_output_to_input_row":face_input_out,
 "quality_star_patch_union_offsets":union_offsets,
 "face_ancestry_offsets":anc_offsets,
 "face_source_identity_schema_by_original_id_json":np.asarray([schema_json],dtype="U"),
 "derived_boundary_split_operation_json_path":np.asarray([str(op_path)],dtype="U"),
 "derived_boundary_split_operation_json_sha256":np.asarray([op_sha],dtype="U64"),
 "derived_boundary_split_parent_face_rows":parent_face_rows,
 "derived_boundary_split_current_target_face_rows":parent_volume_face_rows,
 "derived_boundary_split_output_face_rows":newrows,
 "derived_boundary_split_point_rows":np.arange(npoint,npoint+3,dtype=np.int64),
 "derived_boundary_split_point_source_support_face_rows":np.asarray([x["nearest_current_D_parent_face_row"] for x in supports],np.int64),
 "derived_boundary_split_point_source_barycentric":np.asarray([x["barycentric_on_parent_face"] for x in supports],np.float64),
 "derived_boundary_split_point_surface_residual_m":np.asarray([x["distance_to_parent_surface_m"] for x in supports],np.float64),
 "derived_boundary_split_patch_parent_union_tuples":np.asarray(parent_unions,np.int64),
 "derived_boundary_split_patch_parent_union_sha256":np.asarray([union_sha],dtype="U64"),
 "derived_boundary_split_face_ancestry_union_tuples":np.asarray(parent_ancestry,np.int64),
 "derived_boundary_split_face_ancestry_union_sha256":np.asarray([ancestry_sha],dtype="U64"),
 "immediate_parent_candidate_sha256":np.asarray([sha(D_PATH)],dtype="U64"),
 "source_current_candidate_path":np.asarray([str(D_PATH)],dtype="U"),
 "source_current_candidate_sha256":np.asarray([sha(D_PATH)],dtype="U64"),
 "typed_parent_candidate_path":np.asarray([str(D_PATH)],dtype="U"),
 "typed_parent_candidate_sha256":np.asarray([sha(D_PATH)],dtype="U64"),
})
for lane in lanes:
    payload[f"face_source_{lane}"]=primary_fields[f"face_source_{lane}"]
    payload[f"quality_star_patch_union_face_source_{lane}"]=union_array[:,lanes.index(lane)]
for idx,lane in enumerate(ancestry_lanes):
    payload[lane]=anc_array[:,idx]
# Override all retained primary rows through exact parent selection.
if not np.array_equal(payload["face_source_layer"][:len(keep_face_rows)],D["face_source_layer"][keep_face_rows]):
    raise ValueError("retained face primary types changed")
# Check every output row's complete flattened source-face ancestry against the
# exact retained parent row or the full derived patch parent union.
for outrow, parentrow in enumerate(face_parent):
    if parentrow < 0:
        expected = parent_ancestry
    else:
        alo, ahi = map(int, base_anc_offsets[int(parentrow):int(parentrow)+2])
        expected = [tuple(int(D[k][j]) for k in ancestry_lanes) for j in range(alo, ahi)]
    alo, ahi = map(int, anc_offsets[outrow:outrow+2])
    actual = [tuple(map(int, anc_array[j])) for j in range(alo, ahi)]
    if actual != expected:
        raise ValueError(f"source-face ancestry CSR mismatch at output face {outrow}")
# Verify all source IDs now have a declared registry entry.
source_ids=set(map(str,np.unique(payload["face_source_original_id"])))
missing=sorted(source_ids-set(schema))
if missing: raise ValueError(f"identity registry lacks candidate source IDs: {missing[:8]}")
out_npz=OUT/"candidate-D-comp1-current-target-refinement-typed.npz"
np.savez_compressed(out_npz,**payload)
reread=load_npz(out_npz)
if not np.array_equal(reread["points_world_m_f32"],Pout32) or not np.array_equal(reread["faces"],Fout):
    raise ValueError("candidate NPZ readback geometry mismatch")
if not np.array_equal(reread["point_output_to_parent_rows"],point_parent) or not np.array_equal(reread["face_output_to_parent_rows"],face_parent):
    raise ValueError("candidate NPZ readback maps mismatch")
# Verify the reconstituted typed CSR row-by-row against the retained parent plus full patch union.
for oi,pr in enumerate(face_parent):
    if pr >= 0:
        lo,hi=map(int,union_offsets[oi:oi+2]); actual=[tuple(map(int,row)) for row in union_array[lo:hi]]
        plo,phi=map(int,base_union_offsets[int(pr):int(pr)+2])
        if phi>plo:
            expected=[tuple(int(D[f"quality_star_patch_union_face_source_{lane}"][j]) for lane in lanes) for j in range(plo,phi)]
        else:
            expected=[tuple(int(D[f"face_source_{lane}"][int(pr)]) for lane in lanes)]
        if actual != expected: raise ValueError(f"retained typed union changed at output face {oi}")
    else:
        lo,hi=map(int,union_offsets[oi:oi+2])
        actual=[tuple(map(int,row)) for row in union_array[lo:hi]]
        if actual != parent_unions: raise ValueError(f"derived patch union incomplete at output face {oi}")
# Candidate operation file has all parent face union and full ancestry; report is explicitly not physical.
report={
 "schema":"h706.current-D-comp1-target-refinement-407.v1",
 "status":"PASS_TYPED_D_PATCH_LOCAL_TOPOLOGY_AND_JOINT_APEX",
 "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
 "candidate":{"path":str(out_npz),"sha256":sha(out_npz)},
 "operation":{"path":str(op_path),"sha256":op_sha},
 "inputs":{str(p):{"sha256":sha(p)} for p in PINS},
 "current_surface_baselines":{"O":{"path":str(O_PATH),"sha256":sha(O_PATH)},
  "I":{"path":str(I_PATH),"sha256":sha(I_PATH)},"D":{"path":str(D_PATH),"sha256":sha(D_PATH)}},
 "current_fat_target":{"path":str(V406_PATH),"sha256":sha(V406_PATH),
  "five_parent_target_face_rows":parent_volume_face_rows.tolist(),"source_group":1},
 "parent_D_face_rows":parent_face_rows.tolist(),
 "output_face_rows":newrows.tolist(),"point_rows":list(range(npoint,npoint+3)),
 "point_surface_support_residual_m":[float(x["distance_to_parent_surface_m"]) for x in supports],
 "parent_union_tuple_count":len(parent_unions),"parent_union_tuple_sha256":union_sha,
 "source_face_ancestry_union_count":len(parent_ancestry),"source_face_ancestry_union_sha256":ancestry_sha,
 "geometry":{"points_before_after":[len(P),len(Pout32)],"faces_before_after":[len(F),len(Fout)],
  "all_old_point_bits_unchanged":bool(np.array_equal(Pout32[:len(P)].view(np.uint32),P.view(np.uint32))),
  "new_faces":child_faces.tolist(),"patch_oriented_boundary_unchanged":True,
  "whole_D_oriented_boundary_unchanged":True,"whole_D_bad_edges":0,"whole_D_duplicate_faces":0,
  "joint_apex_gate":float(J.GATE),"new_face_joint_apex_min":float(child_score.min()),
  "new_face_joint_apex_failures":int(np.count_nonzero(child_score<J.GATE)),
  "whole_D_joint_apex_failures":int(np.count_nonzero(all_score<J.GATE)),
  "new_face_qface_min":float(qface.min())},
 "current_context_audit":"PENDING",
 "actual_697_loader":"PENDING",
 "volume_patch":"owned by repair_method; must conform to exact 11-face split-and-flip target",
 "limitations":["This is an inferred local target refinement, not exact support on the original surface; point-to-parent-face residuals are explicit.","The 2-2 diagonal flip changes local target triangulation but preserves the exact oriented patch boundary.","Full O/I/D, both-wall and all185 intersection audit and actual 697 loader remain pending.","Volume transfer, mass mapping, native qualification, physical admission and native steps are not claimed."]
}
report_path=OUT/"report.json";save_json(report_path,report)
# Current exact-audit config rooted at the actual O/I/D state.
cfg=json.loads(CONFIG_TEMPLATE.read_text())
cfg["current_candidates"]={
 "O":{"path":str(O_PATH),"sha256":sha(O_PATH)},
 "I":{"path":str(I_PATH),"sha256":sha(I_PATH)},
 "D":{"path":str(D_PATH),"sha256":sha(D_PATH)}
}
cfg["typed_ancestry_parents"]=dict(cfg.get("typed_ancestry_parents",{}))
cfg["typed_ancestry_parents"]["D"]={"path":str(D_PATH),"sha256":sha(D_PATH)}
cfg["batch_id"]="D-comp1-current-target-steiner-boundary-refinement-407"
cfg_path=OUT/"run-config-current-OI-D.json";save_json(cfg_path,cfg)
print(json.dumps({
 "status":report["status"],"candidate_path":str(out_npz),"candidate_sha256":sha(out_npz),
 "report_path":str(report_path),"report_sha256":sha(report_path),
 "operation_path":str(op_path),"operation_sha256":op_sha,
 "config_path":str(cfg_path),"config_sha256":sha(cfg_path),
 "parent_D_face_rows":parent_face_rows.tolist(),"current_target_face_rows":parent_volume_face_rows.tolist(),
 "new_face_rows":newrows.tolist(),"new_point_rows":list(range(npoint,npoint+3)),
 "new_face_apex_min":float(child_score.min()),"whole_D_apex_bad":int(np.count_nonzero(all_score<J.GATE)),
 "point_residuals_m":[float(x["distance_to_parent_surface_m"]) for x in supports],
 "typed_union_tuples":len(parent_unions),"ancestry_tuples":len(parent_ancestry)
},indent=2),flush=True)

