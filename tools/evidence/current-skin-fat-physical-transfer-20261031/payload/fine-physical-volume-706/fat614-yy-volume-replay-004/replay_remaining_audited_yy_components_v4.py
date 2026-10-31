#!/usr/bin/env python3
"""Versioned exact replay of remaining audited D target-edge splits into Fat614.

Source-only construction proposal. Retains the frozen comp2/6 replay, adds the
seven remaining prepared component fills, and does not claim mass or admission.
"""
from pathlib import Path
from collections import Counter, defaultdict
from fractions import Fraction
import argparse, hashlib, importlib.util, json, sys, time
import numpy as np

H = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
OUT = H / "fat614-yy-volume-replay-004"
RUN = OUT / "run-010-composed-boundary-components-v4"
BASE = H / "fat614-residual-pass-001/remeshed/connected-fixed-surface-volume-proposal.npz"
PART = H / "fat614-compact-cavities-004-budget1-satcap-002/connected-fixed-surface-volume-proposal.npz"
PREP = H / "fat614-compact-cavities-004-anchor-stars/connected-cavity-shells.npz"
REMR = H / "fat614-compact-cavities-004-remesh-002/report.json"
DESC336 = H / "fat-inferred-identity-mask-336-root/fat-run001/fixed-boundary-volume-proposal.npz"
DESC336_REPORT = H / "fat-inferred-identity-mask-336-root/fat-run001/report.json"
DESC336_INDEPENDENT = H / "fat-inferred-identity-mask-336-root/independent326001/receipt.json"
D602 = H / "fat614-minimal-f32-target-split-001/candidate-D-minimally-rounded-yy-edge-split.npz"
D602_AUDIT = H / "fat614-minimal-f32-target-split-001/exact-audit-001/result-001/report.json"
DFINAL = H / "fat614-minimal-f32-target-split-002/run-005/candidate-D-remaining-yy-target-edge-splits.npz"
DFINAL_REPORT = H / "fat614-minimal-f32-target-split-002/run-005/report.json"
DFINAL_AUDIT = H / "fat614-minimal-f32-target-split-002/exact-audit-002/report.json"
DFINAL_697 = H / "fat614-minimal-f32-target-split-002/run-005/actual-697-adapter-001/receipt.json"
D_RUN = H / "fat614-minimal-f32-target-split-002/run-005"
PARENT = H / "fat614-yy-volume-replay-002/fat-comp2-comp6-rounded-boundary-points-volume-proposal.npz"
PARENT_REPORT = H / "fat614-yy-volume-replay-002/report.json"
CASE_ROOT = H / "fat614-compact-cavities-004-remesh-002"
COMP7_CASE = OUT / "run-007-comp7-current-plc"
COMP3_CORRECTED = OUT / "run-004-interior-node3386-v4/component-0003-f32-corrected.npz"
COMP3_CORRECTION_REPORT = OUT / "run-004-interior-node3386-v4/report.json"
COMP10_CORRECTED = OUT / "run-006-interior-155-55-sat/component-0010-interior-corrected.npz"
COMP10_CORRECTION_REPORT = OUT / "run-006-interior-155-55-sat/report.json"
COMP7_CORRECTED = COMP7_CASE / "component-0007-current-target-volume-proposal.npz"
COMP7_RUN = COMP7_CASE / "run.json"
COMP7_PREP = COMP7_CASE / "prepare.json"
COMP7_COORD_CORRECTION = COMP7_CASE / "run-008-interior-steiner-ulp-search/coordinate-correction-v3.json"
COMP7_VALIDATOR = COMP7_CASE / "validate_comp7_existing_current_target_tetgen_output_v7.py"
COMPONENTS = (3, 4, 5, 7, 8, 10, 11)
EXPECTED = {
    "base_c21b": ("c21b708fa591476b1c5ab5f33329362ac8086c6429c73a117e63100555984e8b", BASE),
    "partial_4602": ("4602edc3b1db3617ab610c21f91309db10892c0e28ff7d6ba5c381491d13df2c", PART),
    "prepared_shells": ("5810a3139b619630b8658c407a3b22cde1939ba9c917d273852622743ce5fe44", PREP),
    "remesh_report": ("f2d82ac0d677e63755c43c76ffaf9ed65e63ca73b3f3b49b9da7d5e8758934e0", REMR),
    "descendant_336": ("2fc5aa8930fe50f940b071a954c0393b8e41afee514d64ace81e1c419e2b1905", DESC336),
    "descendant_336_report": ("2c2861083dc2e38ab4d9c3d9c62f05504c224c3159a3e89e9b6cac02412637a0", DESC336_REPORT),
    "descendant_336_independent": ("992dadb5f07d1c5babb0e324f43343d93dbace447c3ed3e67fa87163bd32ca9b", DESC336_INDEPENDENT),
    "parent_d602": ("602036c9b164bd1eec83b344675b8cabf453665920e45b558b50fe8115c67493", D602),
    "parent_d602_audit": ("79533b3f38c4a202463c90c0b426a561641500c1465c6558406cde6077971bb2", D602_AUDIT),
    "final_d_run005": ("4ce01a4bd397e7cf2b3c469ee33a3937d9173ff7d585a2687a1a09cf98377ab2", DFINAL),
    "final_d_run005_report": ("3fef3f823f952afc67b9c481d9ce452414a8c4d7aa3c2420fca5f0f461da9e9c", DFINAL_REPORT),
    "final_d_complete_audit": ("8eaf3426ff00f12691592c9e1f7ce923f6d011c2a0a5dcb009056d194e289854", DFINAL_AUDIT),
    "final_d_697_loader": ("c8bc0ca2293d242672ce575622bcbf86d75bfdf82704c093678644e57599e858", DFINAL_697),
    "comp3_corrected_candidate": ("963b5f24f92ae26239ea54ad3d057b34ad5ebb7bbd764cb441de59dfd9ffe525", COMP3_CORRECTED),
    "comp3_correction_report": ("31efe817d1fdefe945cbaa504f678081f433264df80aeda5d6779e7cc461c412", COMP3_CORRECTION_REPORT),
    "comp10_corrected_candidate": ("7c14d90dc0270ce94c85acb585a2b82a9626993456249ba0848e5a1ee32499f6", COMP10_CORRECTED),
    "comp10_correction_report": ("cadb62d27e058b325cae395b34e9be98e73ddd4fdafc242d6566da7a83f95bfa", COMP10_CORRECTION_REPORT),
    "comp7_corrected_candidate": ("d50de579de2f0a7bf2913d3c7e1b3b125935c52e9bb690479f4ae66cb5080e15", COMP7_CORRECTED),
    "comp7_validation_run": ("f47de02a48dda9649b156a7da7790b35c3c365bf3b48016c90b5f6d31d22ef5a", COMP7_RUN),
    "comp7_current_target_prepare": ("18a60a9c3683765f82a27479e770d780368f8ca4cd6d86aeb814a97c3068e36e", COMP7_PREP),
    "comp7_coordinate_correction": ("864b863fb70e27a54a20287841de737f4ecf9177229b6768c2e2cc08c9d674ca", COMP7_COORD_CORRECTION),
    "comp7_validator_source": ("371fb573e1319c5e4357d1cd11ed56dcaf01144553dd02d95226866d10fdb3b9", COMP7_VALIDATOR),
    "fat_parent_d4906": ("d4906b176f78283f4073f9bd1e8d202a7301b57550a7b1b71fb101e76c4d236f", PARENT),
    "fat_parent_d4906_report": ("05562cfe70678bc591252e60540006b5294b8a48a3bbdca6ee0a99ea055a8550", PARENT_REPORT),
}
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
def read_json(path):
    s = Path(path).read_text()
    return json.JSONDecoder().raw_decode(s)[0]
def f32key(x):
    return np.asarray(x, np.float32).view(np.uint32).tobytes()
def canonical_face(f):
    return tuple(sorted(map(int, f)))
def face_counter(faces):
    ident = m.boundary_identity(np.asarray(faces, np.int64).reshape(-1, 3))
    return Counter(tuple(map(int, row)) for row in ident)
def counter_delta(actual, wanted):
    got = face_counter(actual); want = face_counter(wanted)
    missing = want - got; extra = got - want
    return {
        "missing_oriented_records": int(sum(missing.values())),
        "extra_oriented_records": int(sum(extra.values())),
        "missing_samples": [list(k) for k in sorted(missing)[:12]],
        "extra_samples": [list(k) for k in sorted(extra)[:12]],
    }
def table(path):
    lines = [x.strip() for x in Path(path).read_text().splitlines()
             if x.strip() and not x.lstrip().startswith("#")]
    count = int(lines[0].split()[0])
    rows = np.asarray([[float(v) for v in line.split()] for line in lines[1:]], np.float64)
    if len(rows) != count:
        raise ValueError(f"{path}: declared {count} rows but read {len(rows)}")
    return rows
def exact_float64_det_sign(points64, tet):
    q = [[Fraction.from_float(float(x)) for x in points64[int(row)]] for row in tet]
    a = [q[1][i] - q[0][i] for i in range(3)]
    b = [q[2][i] - q[0][i] for i in range(3)]
    c = [q[3][i] - q[0][i] for i in range(3)]
    det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])
    return 1 if det > 0 else (-1 if det < 0 else 0)
def exact_flux(points, faces):
    return int(r.exact_volume(np.asarray(points, np.float32), np.asarray(faces, np.int64)))
EXACT_INTEGER_POINT_CACHE = {}
def exact_tet_volume(points, tets):
    ip = EXACT_INTEGER_POINT_CACHE.get(id(points))
    if ip is None:
        ip = m.m.integer_points(np.asarray(points, np.float32))
        EXACT_INTEGER_POINT_CACHE[id(points)] = ip
    return [int(m.m.exact_det(ip, t)) for t in np.asarray(tets, np.int64)]
def split_faces_multi(faces, edge, split_rows, point_array, parent_faces, label):
    """Split each authenticated oriented parent triangle along an edge at ordered F32 points."""
    fs = np.asarray(faces, np.int64).reshape(-1, 3)
    u, v = map(int, edge)
    parent_keys = {canonical_face(f) for f in np.asarray(parent_faces, np.int64).reshape(-1, 3)}
    if len(parent_keys) != len(parent_faces):
        raise ValueError(f"{label}: duplicate requested parent triangles")
    hits = [f for f in fs if canonical_face(f) in parent_keys]
    if len(hits) != len(parent_keys) or len({canonical_face(f) for f in hits}) != len(parent_keys):
        raise ValueError(f"{label}: did not find each parent face exactly once")
    A = np.asarray(point_array[u], np.float64)
    B = np.asarray(point_array[v], np.float64)
    vec = B - A
    denom = float(np.dot(vec, vec))
    if denom <= 0:
        raise ValueError(f"{label}: degenerate edge")
    params = []
    for row in map(int, split_rows):
        P = np.asarray(point_array[row], np.float64)
        t = float(np.dot(P - A, vec) / denom)
        if not 0.0 < t < 1.0:
            raise ValueError(f"{label}: split point {row} outside open segment")
        params.append((t, row))
    params.sort()
    if len({t for t, _ in params}) != len(params):
        raise ValueError(f"{label}: duplicate edge split parameter")
    out = []
    hit_keys = set(parent_keys)
    for face in fs:
        if canonical_face(face) not in hit_keys:
            out.append(list(map(int, face)))
            continue
        f = list(map(int, face))
        found = None
        for i in range(3):
            a, b, w = f[i], f[(i + 1) % 3], f[(i + 2) % 3]
            if {a, b} == {u, v}:
                found = (a, b, w)
                break
        if found is None:
            raise ValueError(f"{label}: parent triangle does not contain requested edge")
        a, b, w = found
        ordered = [row for _, row in params]
        if (a, b) == (v, u):
            ordered.reverse()
        chain = [a] + ordered + [b]
        out.extend([[chain[j], chain[j + 1], w] for j in range(len(chain) - 1)])
    return np.asarray(out, np.int64).reshape(-1, 3), hits
def parse_poly_nodes(path):
    lines = [x.strip() for x in Path(path).read_text().splitlines()
             if x.strip() and not x.lstrip().startswith("#")]
    head = lines[0].split()
    n = int(head[0])
    if len(head) < 4 or int(head[1]) != 3:
        raise ValueError("unexpected .poly point header")
    rec = []
    for line in lines[1:1+n]:
        q = line.split()
        if len(q) < 4:
            raise ValueError("malformed .poly point row")
        rec.append((int(q[0]), np.asarray([float(q[1]), float(q[2]), float(q[3])], np.float32)))
    if len(rec) != n:
        raise ValueError("truncated .poly point section")
    return rec
def loadmod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Resolve all artifact pins before loading geometry or writing outputs.
for name, (want, path) in EXPECTED.items():
    got = sha(path)
    if got != want:
        raise ValueError(f"{name} SHA mismatch: expected {want}, got {got}")
for op_path in sorted(D_RUN.glob("edge-split-operation-*.json")):
    if op_path.stat().st_size == 0:
        raise ValueError(f"empty D operation record: {op_path}")
d_run_report = read_json(DFINAL_REPORT)
if d_run_report["outputs"]["candidate"]["sha256"] != EXPECTED["final_d_run005"][0]:
    raise ValueError("run-005 producer report does not bind the exact final D candidate")
d_audit = read_json(DFINAL_AUDIT)
if (d_audit.get("status") != "PASS_EXACT_CHANGED_D_VS_COMPLETE_OID_WALLS_ALL185_AND_APEX"
        or d_audit.get("physical_admission") is not False or d_audit.get("native_steps") != 0
        or d_audit["candidate_D"]["sha256"] != EXPECTED["final_d_run005"][0]):
    raise ValueError("final D exact audit is not the pinned source-only full-context PASS")
d_loader = read_json(DFINAL_697)
if (d_loader.get("status") != "PASS_ACTUAL_697_TYPED_D_LOADER_METADATA_VIEW"
        or d_loader.get("physical_admission") is not False or d_loader.get("native_steps") != 0
        or d_loader["source_candidate"]["sha256"] != EXPECTED["final_d_run005"][0]):
    raise ValueError("final D 697 loader receipt does not bind the exact audited source")
parent_report = read_json(PARENT_REPORT)
if (parent_report.get("status") != "PASS_LOCAL_COMP2_COMP6_VOLUME_TOPOLOGY_PROPOSAL_PENDING_COMPLETE_BOUNDARY_MASS_AND_NATIVE_GATES"
        or parent_report.get("physical_admission") is not False
        or parent_report["whole_candidate"]["candidate_sha256"] != EXPECTED["fat_parent_d4906"][0]):
    raise ValueError("d4906 is not the expected frozen source-only partial")
desc336_receipt = read_json(DESC336_INDEPENDENT)
if (desc336_receipt.get("status") != "PASS_INDEPENDENT_INFERRED_INTERIOR_RETIREMENT_REPLAY"
        or desc336_receipt.get("physical_admission") is not False or desc336_receipt.get("native_steps") != 0):
    raise ValueError("336 inherited independent source-only receipt failed")
desc336_data = load_npz(DESC336)
d4906_data = load_npz(PARENT)
def replacement_group_signatures(z):
    po = np.asarray(z["replacement_parent_offsets"], np.int64)
    to = np.asarray(z["replacement_tet_offsets"], np.int64)
    parents = np.asarray(z["replacement_parent_cell_rows"], np.int64)
    tets = np.asarray(z["replacement_tetrahedra"], np.int64)
    if len(po) != len(to) or po[0] != 0 or po[-1] != len(parents) or to[0] != 0 or to[-1] != len(tets):
        raise ValueError("replacement operation CSR is malformed")
    out = {}
    for i in range(len(po) - 1):
        pk = tuple(sorted(map(int, parents[po[i]:po[i + 1]])))
        tk = tuple(sorted(tuple(sorted(map(int, t))) for t in tets[to[i]:to[i + 1]]))
        if pk in out:
            raise ValueError("duplicate operation parent set")
        out[pk] = tk
    return out
identity336_ops = replacement_group_signatures(desc336_data)
d4906_ops = replacement_group_signatures(d4906_data)
if len(identity336_ops) != 97 or any(d4906_ops.get(pk) != tk for pk, tk in identity336_ops.items()):
    raise ValueError("d4906 does not contain every 336 identity operation exactly by parent-cell set and tetrahedron corner sets")
if not np.array_equal(np.asarray(desc336_data["points_world_m"], np.float32),
                      np.asarray(load_npz(PART)["points_world_m"], np.float32)):
    raise ValueError("336 point coordinates differ from shared 4602 parent")

# Load already-existing computation and audited owner modules.
r = loadmod("fat614_remesh296_007_for_replay004", H / "remesh_fat614_compact_cavities_296_007.py")
m = r.m
caps = loadmod("fat614_caps288_for_replay004", H / "direct_connected_skin_caps_288_root.py")
owner = H.parent / "deep-boundary-checkpoint-349/payload/exact-owner/numilab_human/cardiac_cavity_intersections.py"
sys.path.insert(0, str(owner.parent.parent))
exact = __import__("numilab_human.cardiac_cavity_intersections", fromlist=["_audit_pair"])

base = load_npz(BASE); part = load_npz(PART); prep = load_npz(PREP)
desc336 = load_npz(DESC336); desc = load_npz(PARENT)
d602 = load_npz(D602); dfinal = load_npz(DFINAL)
pbase = np.asarray(base["points_world_m"], np.float32)
ppart = np.asarray(part["points_world_m"], np.float32)
pdesc = np.asarray(desc["points_world_m"], np.float32)
tbase = np.asarray(base["tetrahedra"], np.int64)
tpart = np.asarray(part["tetrahedra"], np.int64)
tdesc = np.asarray(desc["tetrahedra"], np.int64)
m_desc = np.asarray(desc["material_indices"], np.int32)
if not np.array_equal(pbase, ppart) or not np.array_equal(pbase, pdesc[:len(pbase)]):
    raise ValueError("c21b, partial-4602 and d4906 original point prefixes differ")
if len(desc["replacement_tetrahedra"]) != len(tdesc) - len(desc["retained_input_cell_rows"]):
    raise ValueError("d4906 replacement tail length mismatch")
if not np.array_equal(desc["replacement_tetrahedra"], tdesc[len(desc["retained_input_cell_rows"]):]):
    raise ValueError("d4906 replacement tail differs from full candidate tet ordering")

# Establish source-base -> partial output -> d4906 direct output maps. These are exact row identities.
part_source_to_row = {int(src): i for i, src in enumerate(np.asarray(part["retained_input_cell_rows"], np.int64))}
if len(part_source_to_row) != len(part["retained_input_cell_rows"]):
    raise ValueError("partial-4602 retained-source map is not unique")
d4906_input_to_output = {int(src): i for i, src in enumerate(np.asarray(desc["retained_input_cell_rows"], np.int64))}
if len(d4906_input_to_output) != len(desc["retained_input_cell_rows"]):
    raise ValueError("d4906 direct retained map is not unique")
existing_parent_rows = set(map(int, np.asarray(desc["replacement_parent_cell_rows"], np.int64)))
old_target = np.asarray(desc["desired_boundary_faces"], np.int64)
actual_before = m.boundary(tdesc)
if face_counter(actual_before) != face_counter(np.asarray(desc["boundary_faces"], np.int64)):
    raise ValueError("d4906 actual boundary_faces does not replay from exact tet incidence")

# Add the nine audited D split points after all d4906 points, preserving the existing point prefix.
point_list = [x.copy() for x in pdesc]
point_index = {f32key(x): i for i, x in enumerate(pdesc)}
if len(point_index) != len(pdesc):
    raise ValueError("d4906 has duplicate F32 point coordinates")
d602points = np.asarray(d602["points_world_m_f32"], np.float32)
dfinalpoints = np.asarray(dfinal["points_world_m_f32"], np.float32)
if len(dfinalpoints) <= len(d602points) or not np.array_equal(dfinalpoints[:len(d602points)], d602points):
    raise ValueError("run-005 D point prefix is not exact to its D602 parent")
drow_to_volume = {}
for drow, x in enumerate(d602points):
    k = f32key(x)
    if k not in point_index:
        raise ValueError(f"D602 point {drow} is missing from d4906 volume points")
    drow_to_volume[drow] = int(point_index[k])
d_run_ops = list(d_run_report["operations"])
if len(d_run_ops) != 9:
    raise ValueError(f"expected nine run-005 split operations, got {len(d_run_ops)}")
ops_by_component = defaultdict(list)
for op in d_run_ops:
    if op["parent_candidate_sha256"] != EXPECTED["parent_d602"][0]:
        raise ValueError("run-005 operation is not based on the pinned D602 surface")
    ops_by_component[int(op["component_id"])].append(op)
if set(ops_by_component) != set(COMPONENTS):
    raise ValueError(f"run-005 component set mismatch: {sorted(ops_by_component)}")
for drow in range(len(d602points), len(dfinalpoints)):
    x = dfinalpoints[drow]
    k = f32key(x)
    if k in point_index:
        raise ValueError(f"new D point {drow} aliases an existing d4906 point")
    volume_row = len(point_list)
    point_index[k] = volume_row
    drow_to_volume[drow] = volume_row
    point_list.append(x.copy())
for drow in range(len(dfinalpoints)):
    if drow not in drow_to_volume:
        raise ValueError(f"unmapped final D point row {drow}")
p_work = np.asarray(point_list, np.float32)

# The final D target edges and parent-face rows are read from the pinned report, not inferred from component IDs.
edge_groups = defaultdict(lambda: {"d_rows": [], "parent_face_rows": set(), "component_ids": set(), "operation_rows": []})
for op in d_run_ops:
    edge = tuple(map(int, op["edge_global_volume_point_rows"]))
    drow = int(op["new_output_D_point_row"])
    if drow not in range(len(d602points), len(dfinalpoints)):
        raise ValueError(f"run-005 D row {drow} is not newly appended")
    info = edge_groups[tuple(sorted(edge))]
    info["d_rows"].append(drow)
    info["component_ids"].add(int(op["component_id"]))
    info["operation_rows"].append(op)
    for patch in op["face_patches"]:
        info["parent_face_rows"].add(int(patch["parent_face_row"]))
for edge, info in edge_groups.items():
    if len(info["d_rows"]) != len(set(info["d_rows"])):
        raise ValueError(f"duplicate D split row on edge {edge}")
    if not info["parent_face_rows"]:
        raise ValueError(f"no authenticated D parent faces on edge {edge}")
    for drow in info["d_rows"]:
        if set(map(int, d_run_ops[[int(q["new_output_D_point_row"]) for q in d_run_ops].index(drow)]["edge_global_volume_point_rows"])) != set(edge):
            raise ValueError(f"D point {drow} edge mismatch")

# The full updated D face set must embed, reversed for the Fat interface, in the planned desired boundary.
def d_face_to_volume(face):
    try:
        return [drow_to_volume[int(i)] for i in face]
    except KeyError as e:
        raise ValueError(f"D face references unmapped point row {e}")
d_fat_target = np.asarray([[f[0], f[2], f[1]] for f in
                           np.asarray([d_face_to_volume(f) for f in np.asarray(dfinal["faces"], np.int64)], np.int64)],
                          np.int64)
# D is the inner Fat interface, whose physical Fat boundary orientation is reversed.
desired_before = face_counter(old_target)
for edge, info in edge_groups.items():
    parent_global = []
    for dface_row in sorted(info["parent_face_rows"]):
        if dface_row < 0 or dface_row >= len(d602["faces"]):
            raise ValueError(f"bad D602 parent face row {dface_row}")
        parent_global.append([drow_to_volume[int(q)] for q in d602["faces"][dface_row]])
    if any(Counter(canonical_face(f) for f in old_target)[canonical_face(f)] != 1 for f in parent_global):
        raise ValueError(f"D edge {edge}: D602 parent triangle not present once in d4906 desired target")
    info["parent_global_faces"] = np.asarray(parent_global, np.int64)
    info["volume_split_rows"] = [drow_to_volume[int(q)] for q in info["d_rows"]]

# Verify all source row maps and local TetGen output boundaries in preflight mode, without SAT/global exact work.
cell_offsets = np.asarray(prep["cell_offsets"], np.int64)
point_offsets = np.asarray(prep["point_offsets"], np.int64)
face_offsets = np.asarray(prep["face_offsets"], np.int64)
old_target_keys = {canonical_face(f) for f in old_target}
case_data = {}
deleted_output_rows = []
all_source_rows = []
for ci in COMPONENTS:
    lo, hi = map(int, cell_offsets[ci:ci+2])
    source_rows = np.asarray(prep["input_cell_rows"][lo:hi], np.int64)
    if len(np.unique(source_rows)) != len(source_rows):
        raise ValueError(f"component {ci}: duplicate source input rows")
    try:
        partial_rows = np.asarray([part_source_to_row[int(x)] for x in source_rows], np.int64)
    except KeyError as e:
        raise ValueError(f"component {ci}: missing from partial-4602 retained map: {e}")
    overlap = set(map(int, partial_rows)) & existing_parent_rows
    if overlap:
        raise ValueError(f"component {ci}: source rows overlap existing d4906 replacement operation; closure required: {sorted(overlap)[:12]}")
    try:
        out_rows = np.asarray([d4906_input_to_output[int(x)] for x in partial_rows], np.int64)
    except KeyError as e:
        raise ValueError(f"component {ci}: partial row not directly retained by d4906: {e}")
    if len(out_rows) != len(source_rows) or np.any(out_rows >= len(desc["retained_input_cell_rows"])):
        raise ValueError(f"component {ci}: bad d4906 direct output row map")
    if not np.array_equal(tbase[source_rows], tpart[partial_rows]):
        raise ValueError(f"component {ci}: partial-4602 source tet tuple mismatch")
    if not np.array_equal(tpart[partial_rows], tdesc[out_rows]):
        raise ValueError(f"component {ci}: d4906 did not preserve source tet row identity")
    if np.any(np.asarray(desc["material_indices"])[out_rows] != 16):
        raise ValueError(f"component {ci}: selected old cells are not material 16")
    if np.any(np.asarray(desc["material_indices"])[out_rows] != np.asarray(part["material_indices"])[partial_rows]):
        raise ValueError(f"component {ci}: material map differs across d4906 direct rows")
    grows = np.asarray(prep["global_point_rows"][int(point_offsets[ci]):int(point_offsets[ci+1])], np.int64)
    local_faces = np.asarray(prep["local_faces"][int(face_offsets[ci]):int(face_offsets[ci+1])], np.int64)
    shell_global = grows[local_faces]
    if len(np.unique(grows)) != len(grows):
        raise ValueError(f"component {ci}: duplicate shell points")
    comp_ops = ops_by_component[ci]
    comp_drows = [int(o["new_output_D_point_row"]) for o in comp_ops]
    comp_parent_faces = sorted({int(p["parent_face_row"]) for o in comp_ops for p in o["face_patches"]})
    target_parent_global = []
    for fr in comp_parent_faces:
        tri = np.asarray([drow_to_volume[int(q)] for q in d602["faces"][fr]], np.int64)
        target_parent_global.append(tri)
    target_parent_global = np.asarray(target_parent_global, np.int64)
    target_counts = Counter(canonical_face(f) for f in shell_global)
    for f in target_parent_global:
        if target_counts[canonical_face(f)] != 1:
            raise ValueError(f"component {ci}: D split parent face is not present once in prepared cavity shell")
    case = COMP7_CASE if ci == 7 else CASE_ROOT / f"component-{ci:04d}"
    required = [case / "cavity.poly", case / "cavity.1.node", case / "cavity.1.ele", case / "cavity.1.face"]
    if any(not q.is_file() for q in required):
        raise ValueError(f"component {ci}: frozen TetGen outputs missing")
    poly_nodes = parse_poly_nodes(case / "cavity.poly")
    local_source_keys = {f32key(pdesc[int(row)]): int(row) for row in grows}
    if len(local_source_keys) != len(grows):
        raise ValueError(f"component {ci}: duplicate source point coordinate keys")
    allowed_poly_rows = set(map(int, grows))
    if ci == 7:
        allowed_poly_rows.update(int(drow_to_volume[int(q)]) for q in comp_drows)
    poly_map = []
    for nid, x in poly_nodes:
        k = f32key(x)
        if k not in point_index:
            raise ValueError(f"component {ci}: .poly node {nid} lacks exact volume coordinate")
        row = int(point_index[k]); poly_map.append(row)
        if row not in allowed_poly_rows:
            raise ValueError(f"component {ci}: .poly node maps outside prepared source plus authorized current-D split point")
    if len(poly_map) != len(allowed_poly_rows) or set(poly_map) != allowed_poly_rows:
        raise ValueError(f"component {ci}: .poly point set differs from prepared points plus authorized D split points")
    node = table(case / "cavity.1.node")
    ele = table(case / "cavity.1.ele")
    face_table = table(case / "cavity.1.face")
    node_ids = node[:, 0].astype(np.int64)
    node_xyz64 = np.asarray(node[:, 1:4], np.float64)
    node_xyz_raw = np.asarray(node_xyz64, np.float32)
    node_xyz = node_xyz_raw.copy()
    coordinate_correction = None
    if ci == 3:
        z = load_npz(COMP3_CORRECTED); rr = read_json(COMP3_CORRECTION_REPORT)
        nid = int(z["corrected_tetgen_node_id"][0]); old = np.asarray(rr["old_point_f32"], np.float32)
        new = np.asarray(z["corrected_point_world_m"][0], np.float32)
        matches = np.flatnonzero(node_ids == nid)
        if len(matches) != 1 or not np.array_equal(node_xyz_raw[matches[0]].view(np.uint32), old.view(np.uint32)):
            raise ValueError("component 3 corrected TetGen node does not match its frozen raw Float32 coordinate")
        if not np.array_equal(np.asarray(rr["new_point_f32"], np.float32).view(np.uint32), new.view(np.uint32)):
            raise ValueError("component 3 corrected candidate/report coordinates differ")
        node_xyz[matches[0]] = new; coordinate_correction = {"node_id": nid, "old_f32": old.tolist(), "new_f32": new.tolist(), "source_report_sha256": EXPECTED["comp3_correction_report"][0], "candidate_sha256": EXPECTED["comp3_corrected_candidate"][0]}
    elif ci == 10:
        z = load_npz(COMP10_CORRECTED); rr = read_json(COMP10_CORRECTION_REPORT); cc = rr["components"]["10"]
        nid = int(z["corrected_tetgen_node_id"][0]); old = np.asarray(cc["old_point_f32"], np.float32)
        new = np.asarray(z["corrected_point_world_m"][0], np.float32)
        matches = np.flatnonzero(node_ids == nid)
        if len(matches) != 1 or not np.array_equal(node_xyz_raw[matches[0]].view(np.uint32), old.view(np.uint32)):
            raise ValueError("component 10 corrected TetGen node does not match its frozen raw Float32 coordinate")
        if not np.array_equal(np.asarray(cc["new_point_f32"], np.float32).view(np.uint32), new.view(np.uint32)):
            raise ValueError("component 10 corrected candidate/report coordinates differ")
        node_xyz[matches[0]] = new; coordinate_correction = {"node_id": nid, "old_f32": old.tolist(), "new_f32": new.tolist(), "source_report_sha256": EXPECTED["comp10_correction_report"][0], "candidate_sha256": EXPECTED["comp10_corrected_candidate"][0]}
    elif ci == 7:
        rr = read_json(COMP7_COORD_CORRECTION)
        nid = int(rr["tetgen_output_node_id"]); old = np.asarray(rr["original_f32_coordinate"], np.float32)
        new = np.asarray(rr["corrected_f32_coordinate"], np.float32)
        matches = np.flatnonzero(node_ids == nid)
        if len(matches) != 1 or not np.array_equal(node_xyz_raw[matches[0]].view(np.uint32), old.view(np.uint32)):
            raise ValueError("component 7 corrected TetGen node does not match its frozen raw Float32 coordinate")
        z = load_npz(COMP7_CORRECTED)
        zrows = np.flatnonzero(np.asarray(z["tetgen_output_node_ids"], np.int64) == nid)
        if len(zrows) != 1 or not np.array_equal(np.asarray(z["points_world_m"][zrows[0]], np.float32).view(np.uint32), new.view(np.uint32)):
            raise ValueError("component 7 local candidate does not contain the receipt-pinned corrected node")
        node_xyz[matches[0]] = new; coordinate_correction = {"node_id": nid, "old_f32": old.tolist(), "new_f32": new.tolist(), "source_report_sha256": EXPECTED["comp7_coordinate_correction"][0], "candidate_sha256": EXPECTED["comp7_corrected_candidate"][0], "validation_run_sha256": EXPECTED["comp7_validation_run"][0]}
    ele_node_ids = ele[:, 1:5].astype(np.int64)
    face_node_ids = face_table[:, 1:4].astype(np.int64)
    if len(np.unique(node_ids)) != len(node_ids) or len(np.unique(ele[:, 0])) != len(ele):
        raise ValueError(f"component {ci}: duplicate TetGen IDs")
    if ele.shape[1] < 6 or np.any(ele[:, 5].astype(np.int64) != 16):
        raise ValueError(f"component {ci}: TetGen material attr differs from 16")
    if not set(map(int, ele_node_ids.ravel())).issubset(set(map(int, node_ids))):
        raise ValueError(f"component {ci}: element references unknown output node")
    if not set(map(int, face_node_ids.ravel())).issubset(set(map(int, node_ids))):
        raise ValueError(f"component {ci}: face references unknown output node")
    id_to_local = {int(q): i for i, q in enumerate(node_ids)}
    input_node_ids = {int(q) for q, _ in poly_nodes}
    output_node_pos = {int(node_ids[i]): node_xyz[i] for i in range(len(node_ids))}
    if not input_node_ids.issubset(set(output_node_pos)):
        raise ValueError(f"component {ci}: TetGen dropped a PLC point")
    for nid, xyz in poly_nodes:
        if nid not in output_node_pos or not np.array_equal(output_node_pos[nid].view(np.uint32), xyz.view(np.uint32)):
            raise ValueError(f"component {ci}: TetGen moved input PLC point {nid}")
    face_node_ids_set = set(map(int, face_node_ids.ravel()))
    used_node_ids = set(map(int, ele_node_ids.ravel())) | face_node_ids_set
    unused_ids = sorted(set(map(int, node_ids)) - used_node_ids)
    new_d_keys = {f32key(dfinalpoints[row]): row for row in comp_drows}
    found_boundary = {}
    for i, nid in enumerate(node_ids):
        if int(nid) not in face_node_ids_set:
            continue
        k = f32key(node_xyz[i])
        if k in new_d_keys:
            found_boundary[int(new_d_keys[k])] = int(nid)
    if set(found_boundary) != set(comp_drows):
        raise ValueError(f"component {ci}: D split output boundary nodes differ from run-005 points: {found_boundary}")
    case_data[ci] = {
        "source_rows": source_rows, "partial_rows": partial_rows, "output_rows": out_rows,
        "grows": grows, "shell_global": shell_global, "local_faces": local_faces,
        "target_parent_global": target_parent_global, "case": case, "node": node,
        "ele": ele, "face_table": face_table, "node_ids": node_ids, "node_xyz": node_xyz, "node_xyz_raw": node_xyz_raw, "node_xyz64": node_xyz64,
        "coordinate_correction": coordinate_correction,
        "ele_node_ids": ele_node_ids, "face_node_ids": face_node_ids, "id_to_local": id_to_local,
        "used_node_ids": used_node_ids, "unused_ids": unused_ids, "found_D_nodes": found_boundary,
    }
    deleted_output_rows.extend(map(int, out_rows))
    all_source_rows.extend(map(int, source_rows))
    print(f"PREFLIGHT component={ci} old_cells={len(source_rows)} TetGen_cells={len(ele)} "
          f"D_points={len(comp_drows)} boundary_node_ids={found_boundary} parents={len(comp_parent_faces)}")
if len(set(deleted_output_rows)) != len(deleted_output_rows):
    raise ValueError("component output rows overlap")
if len(set(all_source_rows)) != len(all_source_rows):
    raise ValueError("component source rows overlap")
if "--preflight" in sys.argv:
    print("PREFLIGHT_PASS source rows, row maps, D target faces, and TetGen boundary nodes validated")
    raise SystemExit(0)

# All output node/element maps are now rebound to one append-only volume point namespace.
for ci in COMPONENTS:
    c = case_data[ci]
    node_ids = c["node_ids"]; node_xyz = c["node_xyz"]
    id_to_local = c["id_to_local"]
    used = c["used_node_ids"]
    if len({f32key(node_xyz[i]) for i, nid in enumerate(node_ids) if int(nid) in used}) != len(used):
        raise ValueError(f"component {ci}: duplicate used TetGen node coordinates")
    node_global = np.full(len(node_ids), -1, np.int64)
    for i, x in enumerate(node_xyz):
        if int(node_ids[i]) not in used:
            continue
        k = f32key(x)
        if k in point_index:
            row = int(point_index[k])
        else:
            row = len(point_list)
            point_index[k] = row
            point_list.append(np.asarray(x, np.float32).copy())
        node_global[i] = row
    for drow, nid in c["found_D_nodes"].items():
        i = id_to_local[int(nid)]
        if int(node_global[i]) != int(drow_to_volume[int(drow)]):
            raise ValueError(f"component {ci}: TetGen D boundary node does not map to authenticated D row {drow}")
    elem_rows = np.asarray([[id_to_local[int(n)] for n in row] for row in c["ele_node_ids"]], np.int64)
    out_tets = node_global[elem_rows]
    face_rows = np.asarray([[id_to_local[int(n)] for n in row] for row in c["face_node_ids"]], np.int64)
    out_faces = node_global[face_rows]
    if np.any(out_tets < 0) or np.any(out_faces < 0):
        raise ValueError(f"component {ci}: TetGen output map references a discarded node")
    c["node_global"] = node_global
    c["out_tets"] = out_tets
    c["out_tets_local"] = elem_rows
    c["out_faces"] = out_faces
    c["new_D_global_rows"] = np.asarray([drow_to_volume[int(x)] for x in sorted(c["found_D_nodes"])], np.int64)
    local = np.asarray(c["shell_global"], np.int64)
    used_edges = []
    for edge, info in edge_groups.items():
        if ci not in info["component_ids"]:
            continue
        local, _ = split_faces_multi(local, edge, info["volume_split_rows"], np.asarray(point_list, np.float32),
                                     info["parent_global_faces"], f"component-{ci}-local")
        used_edges.append(edge)
    if not used_edges:
        raise ValueError(f"component {ci}: no target-edge split was applied")
    c["expected_local"] = local
    c["used_edges"] = used_edges
    if len(c["out_tets"]) == 0:
        raise ValueError(f"component {ci}: empty TetGen replacement")
# Point namespace is frozen before any exact measure or collision check.
new_points = np.asarray(point_list, np.float32)
EXACT_INTEGER_POINT_CACHE[id(pdesc)] = m.m.integer_points(pdesc)
EXACT_INTEGER_POINT_CACHE[id(new_points)] = m.m.integer_points(new_points)
if not np.array_equal(new_points[:len(pdesc)].view(np.uint32), pdesc.view(np.uint32)):
    raise ValueError("d4906 volume point bytes changed")
if not np.array_equal(new_points[len(pdesc):len(pdesc)+len(dfinalpoints)-len(d602points)],
                       dfinalpoints[len(d602points):]):
    raise ValueError("appended D target point order or bytes changed")

# Split the full Fat desired boundary on every authenticated D target edge.
desired_updated = np.asarray(old_target, np.int64)
parent_face_overlap = set()
for edge, info in sorted(edge_groups.items()):
    pkeys = {canonical_face(f) for f in info["parent_global_faces"]}
    overlap = parent_face_overlap & pkeys
    if overlap:
        raise ValueError(f"target-face parents overlap across split edges: {sorted(overlap)[:4]}")
    parent_face_overlap |= pkeys
    desired_updated, _ = split_faces_multi(desired_updated, edge, info["volume_split_rows"], new_points,
                                           info["parent_global_faces"], f"whole-target-{edge}")
if len(desired_updated) != len(old_target) + 2 * sum(len(v["volume_split_rows"]) for v in edge_groups.values()):
    raise ValueError("full desired-boundary face count does not match the authenticated edge splits")
d_fat_target = np.asarray([[int(f[0]), int(f[2]), int(f[1])]
                           for f in [ [drow_to_volume[int(q)] for q in face] for face in dfinal["faces"] ]],
                          np.int64)
d_target_missing = face_counter(d_fat_target) - face_counter(desired_updated)
if d_target_missing:
    raise ValueError(f"updated desired boundary does not contain full reversed run-005 D target: {sum(d_target_missing.values())} faces")
# Check each local TetGen product against its complete local target/cut shell, exact orientation, volume and self-intersection.
desired_face_keys = {canonical_face(q) for q in desired_updated}
component_results = {}
new_tets_by_component = []
deleted_output_rows = []
new_parent_groups = []
new_tet_groups = []
for ci in COMPONENTS:
    c = case_data[ci]
    deleted_output_rows.extend(map(int, c["output_rows"]))
    new_parent_groups.append(np.sort(c["partial_rows"]).astype(np.int64))
    new_tet_groups.append(np.asarray(c["out_tets"], np.int64))
    new_tets_by_component.append(np.asarray(c["out_tets"], np.int64))
    expected = np.asarray(c["expected_local"], np.int64)
    raw_f32_det = exact_tet_volume(new_points, c["out_tets"])
    zero_rows = [i for i, value in enumerate(raw_f32_det) if int(value) == 0]
    if zero_rows:
        raise ValueError(f"component {ci}: TetGen output has exact F32 zero-volume cells {zero_rows[:12]}")
    raw64_signs = [exact_float64_det_sign(c["node_xyz64"], row) for row in c["out_tets_local"]]
    if any(sign == 0 for sign in raw64_signs):
        raise ValueError(f"component {ci}: original .1.node Float64 has exact zero-volume cells")
    emitted_f32_det = exact_tet_volume(c["node_xyz_raw"], c["out_tets_local"])
    if any(int(x) == 0 for x in emitted_f32_det):
        raise ValueError(f"component {ci}: raw TetGen emitted Float32 has exact zero-volume cells")
    raw_f32_signs = [1 if int(value) > 0 else -1 for value in emitted_f32_det]
    candidate_pre_normalize_det = raw_f32_det
    candidate_pre_normalize_signs = [1 if int(value) > 0 else -1 for value in candidate_pre_normalize_det]
    normalize_rows = [i for i, sign in enumerate(candidate_pre_normalize_signs) if sign < 0]
    if normalize_rows:
        c["out_tets"] = np.asarray(c["out_tets"], np.int64).copy()
        c["out_tets"][normalize_rows, 2], c["out_tets"][normalize_rows, 3] = (c["out_tets"][normalize_rows, 3].copy(), c["out_tets"][normalize_rows, 2].copy())
    new_det = exact_tet_volume(new_points, c["out_tets"])
    if any(int(x) <= 0 for x in new_det):
        raise ValueError(f"component {ci}: exact F32-positive orientation normalization did not remove every sign defect")
    c["raw64_signs"] = np.asarray(raw64_signs, np.int8)
    c["raw_f32_signs"] = np.asarray(raw_f32_signs, np.int8)
    c["candidate_pre_normalize_signs"] = np.asarray(candidate_pre_normalize_signs, np.int8)
    c["orientation_normalized_rows"] = np.asarray(normalize_rows, np.int64)
    c["orientation_normalized_element_ids"] = np.asarray([c["ele"][i, 0] for i in normalize_rows], np.int64)
    local_boundary = m.boundary(c["out_tets"])
    if face_counter(local_boundary) != face_counter(expected):
        raise ValueError(f"component {ci}: normalized TetGen tetrahedron boundary differs from exact updated local shell")
    face_file_counts = Counter(canonical_face(f) for f in c["out_faces"])
    expected_face_counts = Counter(canonical_face(f) for f in expected)
    if face_file_counts != expected_face_counts:
        raise ValueError(f"component {ci}: frozen TetGen .1.face differs from local target/cut shell")
    if face_file_counts != Counter(canonical_face(f) for f in local_boundary):
        raise ValueError(f"component {ci}: .1.face does not match tetrahedral boundary")
    local_records, _ = exact.float32_surface_records(new_points, expected)
    shell_self = exact._audit_pair(local_records, local_records, same_surface=True)
    if int(shell_self.get("count", 0)) != 0:
        raise ValueError(f"component {ci}: exact local shell self-intersections {shell_self.get('count')}")
    old_tets = tdesc[c["output_rows"]]
    old_det = exact_tet_volume(pdesc, old_tets)
    if any(int(x) <= 0 for x in old_det):
        raise ValueError(f"component {ci}: old selected source has exact nonpositive F32 tets")
    old_boundary = m.boundary(old_tets)
    old_volume = sum(old_det)
    old_flux = exact_flux(pdesc, old_boundary)
    if old_volume != old_flux:
        raise ValueError(f"component {ci}: old local tet volume differs from its exact boundary flux")
    new_volume = sum(new_det)
    new_flux = exact_flux(new_points, expected)
    if new_volume != new_flux:
        raise ValueError(f"component {ci}: new local tet volume differs from exact target/cut shell flux")
    delta = int(new_volume - old_volume)
    if delta != int(new_flux - old_flux):
        raise ValueError(f"component {ci}: tet volume delta differs from revised shell flux delta")
    ndet, nq = m.m.measurements(new_points, c["out_tets"])
    native_fail = int(np.sum(~m.m.passes(ndet, nq)))
    face_count_old = len(old_boundary); face_count_new = len(expected)
    c["old_boundary"] = old_boundary
    c["old_volume"] = int(old_volume); c["old_flux"] = int(old_flux)
    c["new_volume"] = int(new_volume); c["new_flux"] = int(new_flux); c["delta"] = delta
    c["new_det_count"] = len(new_det); c["native_failures"] = native_fail
    c["shell_self_count"] = int(shell_self.get("count", 0))
    c["old_boundary_face_count"] = face_count_old; c["new_boundary_face_count"] = face_count_new
    component_results[str(ci)] = {
        "source_c21b_rows_count": int(len(c["source_rows"])),
        "source_c21b_rows_sha256": hashlib.sha256(np.asarray(c["source_rows"], dtype="<i8").tobytes()).hexdigest(),
        "partial_4602_parent_rows_count": int(len(c["partial_rows"])),
        "partial_4602_parent_rows_sha256": hashlib.sha256(np.asarray(c["partial_rows"], dtype="<i8").tobytes()).hexdigest(),
        "d4906_output_rows_count": int(len(c["output_rows"])),
        "d4906_output_rows_sha256": hashlib.sha256(np.asarray(c["output_rows"], dtype="<i8").tobytes()).hexdigest(),
        "old_cells": int(len(old_tets)), "new_cells": int(len(c["out_tets"])), "material": 16,
        "target_edges": [list(map(int, e)) for e in c["used_edges"]],
        "D_target_point_rows": sorted(map(int, c["found_D_nodes"])),
        "D_volume_point_rows": c["new_D_global_rows"].tolist(),
        "TetGen_new_boundary_node_ids": {str(k): int(v) for k, v in c["found_D_nodes"].items()},
        "TetGen_unused_output_node_ids": c["unused_ids"],
        "TetGen_interior_points_added": int(sum(1 for row in c["node_global"] if row >= len(pdesc) + len(dfinalpoints) - len(d602points))),
        "local_shell_faces_before_after": [int(len(c["shell_global"])), int(len(expected))],
        "local_tet_boundary_matches_updated_shell": True,
        "TetGen_face_file_matches_unoriented_shell": True,
        "exact_shell_self_intersections": int(c["shell_self_count"]),
        "new_cells_exact_F32_positive": int(c["new_det_count"]),
        "TetGen_raw_Float64_exact_sign_counts": {str(k): int(v) for k, v in Counter(map(int, c["raw64_signs"])).items()},
        "TetGen_raw_emitted_F32_exact_sign_counts": {str(k): int(v) for k, v in Counter(map(int, c["raw_f32_signs"])).items()},
        "raw_Float64_vs_emitted_F32_exact_sign_disagreement_count": int(np.sum(c["raw64_signs"] != c["raw_f32_signs"])),
        "post_coordinate_correction_pre_normalization_F32_sign_counts": {str(k): int(v) for k, v in Counter(map(int, c["candidate_pre_normalize_signs"])).items()},
        "receipt_pinned_interior_coordinate_correction": c["coordinate_correction"],
        "orientation_normalization_corner_swaps": c["orientation_normalized_element_ids"].tolist(),
        "orientation_normalization_changes_geometry_or_connectivity": False,
        "new_native_quality_failures": native_fail,
        "old_exact_volume_det6": int(c["old_volume"]),
        "old_exact_boundary_flux_det6": int(c["old_flux"]),
        "new_exact_volume_det6": int(c["new_volume"]),
        "new_exact_boundary_flux_det6": int(c["new_flux"]),
        "exact_local_volume_delta_det6": int(c["delta"]),
        "complete_internal_cut_face_count": int(sum(1 for f in expected if canonical_face(f) not in desired_face_keys)),
    }

# Resolve the selected current source rows and revised outputs; only direct retained cells are retired.
deleted_output_rows = np.asarray(sorted(set(deleted_output_rows)), np.int64)
direct_retained = np.asarray(desc["retained_input_cell_rows"], np.int64)
if np.any(deleted_output_rows >= len(direct_retained)):
    raise ValueError("selected component row maps outside d4906 direct retained output block")
if len(deleted_output_rows) != sum(len(case_data[ci]["output_rows"]) for ci in COMPONENTS):
    raise ValueError("duplicate selected d4906 output row removal")
keep = np.ones(len(tdesc), bool)
keep[deleted_output_rows] = False
if not np.all(keep[len(direct_retained):]):
    raise ValueError("a pending component unexpectedly overlaps an existing d4906 replacement block")
new_retained = direct_retained[keep[:len(direct_retained)]]
new_tets_flat = np.vstack([np.asarray(case_data[ci]["out_tets"], np.int64) for ci in COMPONENTS])
new_tets = np.vstack([tdesc[keep], new_tets_flat])
new_materials = np.r_[m_desc[keep], np.full(len(new_tets_flat), 16, np.int32)]
if not np.array_equal(new_tets[len(new_retained):],
                      np.vstack([np.asarray(desc["replacement_tetrahedra"], np.int64), new_tets_flat])):
    raise ValueError("candidate replacement tail order differs from map arrays")
old_parent_flat = np.asarray(desc["replacement_parent_cell_rows"], np.int64)
old_parent_offsets = np.asarray(desc["replacement_parent_offsets"], np.int64)
old_replace_flat = np.asarray(desc["replacement_tetrahedra"], np.int64)
old_tet_offsets = np.asarray(desc["replacement_tet_offsets"], np.int64)
if old_parent_offsets[0] != 0 or old_parent_offsets[-1] != len(old_parent_flat):
    raise ValueError("d4906 replacement-parent CSR malformed")
if old_tet_offsets[0] != 0 or old_tet_offsets[-1] != len(old_replace_flat):
    raise ValueError("d4906 replacement-tet CSR malformed")
new_parent_counts = [len(x) for x in new_parent_groups]
new_tet_counts = [len(x) for x in new_tet_groups]
parent_offsets = list(map(int, old_parent_offsets))
tet_offsets = list(map(int, old_tet_offsets))
parent_total = int(parent_offsets[-1]); tet_total = int(tet_offsets[-1])
for a, b in zip(new_parent_counts, new_tet_counts):
    parent_total += a; tet_total += b
    parent_offsets.append(parent_total); tet_offsets.append(tet_total)
new_parent_flat = np.concatenate([old_parent_flat] + new_parent_groups).astype(np.int64)
new_replacement_flat = np.vstack([old_replace_flat] + new_tet_groups)
parent_offsets = np.asarray(parent_offsets, np.int64)
tet_offsets = np.asarray(tet_offsets, np.int64)
if len(parent_offsets) != len(tet_offsets):
    raise ValueError("replacement operation parent/tet CSR count differs")
if not np.array_equal(new_tets[len(new_retained):], new_replacement_flat):
    raise ValueError("appended replacement operation map does not reproduce candidate cell order")
# Every PART output parent row must be represented once, either directly retained or by one replacement op.
all_parent_rows = list(map(int, new_retained)) + list(map(int, new_parent_flat))
if len(all_parent_rows) != len(tpart) or len(set(all_parent_rows)) != len(tpart) or set(all_parent_rows) != set(range(len(tpart))):
    raise ValueError("updated retained/operation parent maps do not form an exact PART-domain partition")
if len(np.unique(np.sort(new_tets, axis=1), axis=0)) != len(new_tets):
    raise ValueError("duplicate tetrahedra after extension")
if np.any(np.min(new_tets, axis=1) == np.max(new_tets, axis=1)):
    raise ValueError("degenerate tetrahedron after extension")
if np.any(new_materials != 16):
    raise ValueError("unexpected material labels in Fat candidate")

# Bidirectional oriented cut-set equality. Exclude only the old and revised external target faces.
old_global_boundary_keys = {canonical_face(f) for f in actual_before}
old_target_keys = {canonical_face(f) for f in old_target}
new_target_keys = {canonical_face(f) for f in desired_updated}
old_selected_boundary = m.boundary(tdesc[deleted_output_rows])
old_internal_cut = np.asarray(
    [f for f in old_selected_boundary
     if canonical_face(f) not in old_global_boundary_keys and canonical_face(f) not in old_target_keys],
    np.int64).reshape(-1, 3)
new_local_shells = np.vstack([case_data[ci]["expected_local"] for ci in COMPONENTS])
new_internal_cut = np.asarray(
    [f for f in new_local_shells if canonical_face(f) not in new_target_keys],
    np.int64).reshape(-1, 3)
old_cut_counter = face_counter(old_internal_cut)
new_cut_counter = face_counter(new_internal_cut)
cut_missing = new_cut_counter - old_cut_counter
cut_extra = old_cut_counter - new_cut_counter
if cut_missing or cut_extra:
    raise ValueError(
        f"bidirectional internal cut mismatch: output_only={sum(cut_missing.values())} "
        f"source_only={sum(cut_extra.values())}; output={list(cut_missing.items())[:10]}, "
        f"source={list(cut_extra.items())[:10]}")
unselected = tdesc[keep]
unselected_oriented = face_counter(m.boundary(unselected))
def require_opposite_cut(cuts, label):
    for key, count in face_counter(cuts).items():
        opp = key[:3] + (-key[3],)
        available = int(unselected_oriented[opp])
        if available < count:
            raise ValueError(f"{label}: cut face lacks untouched opposite tet: {key} need={count} have={available}")
require_opposite_cut(old_internal_cut, "old")
require_opposite_cut(new_internal_cut, "new")
# Ensure no extra new target face is silently introduced beyond the authenticated D split.
target_counter = face_counter(desired_updated)
expected_d_counter = face_counter(d_fat_target)
if expected_d_counter - target_counter:
    raise ValueError("revised desired target omitted a D face after reversal")
# Whole candidate closure and exact global flux.
whole_boundary = m.boundary(new_tets)
old_whole_diff = counter_delta(actual_before, old_target)
new_whole_diff = counter_delta(whole_boundary, desired_updated)
global_old_flux = exact_flux(pdesc, actual_before)
global_new_flux = exact_flux(new_points, whole_boundary)
sum_local_delta = sum(int(case_data[ci]["delta"]) for ci in COMPONENTS)
if global_new_flux - global_old_flux != sum_local_delta:
    raise ValueError("global exact boundary-flux delta differs from all seven local cavity volume deltas")
# Exact F32 SAT against every untouched source cell and between new patches.
tree = caps.TetBVH(pdesc, tdesc)
all_new_patch_tets = np.vstack([case_data[ci]["out_tets"] for ci in COMPONENTS])
sat = caps.check_intersections(new_points, tdesc, all_new_patch_tets, deleted_output_rows, [], tree, max_pairs=10_000_000)
if not sat.get("pass"):
    raise ValueError("exact retained/interpatch SAT failed: " + str(sat))
# Full candidate native metric census is diagnostic only; native execution/admission remains off.
before_det, before_q = m.m.measurements(pdesc, tdesc)
after_det, after_q = m.m.measurements(new_points, new_tets)
quality_before = int(np.sum(~m.m.passes(before_det, before_q)))
quality_after = int(np.sum(~m.m.passes(after_det, after_q)))
f64_nonpositive_after = int(np.sum(after_det <= 0))

# Save a separate immutable run; never overwrite parent artifacts.
if RUN.exists():
    raise FileExistsError(f"refusing to overwrite frozen output directory {RUN}")
RUN.mkdir(parents=True, exist_ok=False)
outnpz = RUN / "fat-d-run005-boundary-transfer-volume-proposal-v4.npz"
payload = {k: v for k, v in desc.items()}
payload.update({
    "points_world_m": new_points,
    "tetrahedra": new_tets,
    "material_indices": new_materials,
    "boundary_faces": whole_boundary,
    "desired_boundary_faces": desired_updated,
    "retained_input_cell_rows": new_retained,
    "replacement_parent_offsets": parent_offsets,
    "replacement_parent_cell_rows": new_parent_flat,
    "replacement_tet_offsets": tet_offsets,
    "replacement_tetrahedra": new_replacement_flat,
    "fat614_boundary_transfer_004_component_ids": np.asarray(COMPONENTS, np.int64),
    "fat614_boundary_transfer_004_deleted_d4906_output_rows": deleted_output_rows,
    "fat614_boundary_transfer_004_source_c21b_rows": np.concatenate([case_data[ci]["source_rows"] for ci in COMPONENTS]),
    "fat614_boundary_transfer_004_partial_4602_parent_rows_by_component_offsets": np.r_[0, np.cumsum(new_parent_counts)].astype(np.int64),
    "fat614_boundary_transfer_004_partial_4602_parent_rows": np.concatenate(new_parent_groups),
    "fat614_boundary_transfer_004_tetgen_output_tet_offsets": np.r_[0, np.cumsum(new_tet_counts)].astype(np.int64),
    "fat614_boundary_transfer_004_tetgen_output_tets_global_rows": new_tets_flat,
    "fat614_boundary_transfer_004_tetgen_raw_float64_exact_signs": np.concatenate([case_data[ci]["raw64_signs"] for ci in COMPONENTS]),
    "fat614_boundary_transfer_004_tetgen_raw_f32_exact_signs": np.concatenate([case_data[ci]["raw_f32_signs"] for ci in COMPONENTS]),
    "fat614_boundary_transfer_004_tetgen_orientation_normalized_element_ids": np.concatenate([case_data[ci]["orientation_normalized_element_ids"] for ci in COMPONENTS]),
    "fat614_boundary_transfer_004_tetgen_orientation_normalized_offsets": np.r_[0, np.cumsum([len(case_data[ci]["orientation_normalized_element_ids"]) for ci in COMPONENTS])].astype(np.int64),
    "fat614_boundary_transfer_004_tetgen_output_node_offsets": np.r_[0, np.cumsum([len(case_data[ci]["node_ids"]) for ci in COMPONENTS])].astype(np.int64),
    "fat614_boundary_transfer_004_tetgen_node_global_rows": np.concatenate([case_data[ci]["node_global"] for ci in COMPONENTS]),
    "fat614_boundary_transfer_004_d_source_point_rows": np.asarray(sorted(drow_to_volume), np.int64),
    "fat614_boundary_transfer_004_d_global_volume_point_rows": np.asarray([drow_to_volume[q] for q in sorted(drow_to_volume)], np.int64),
    "fat614_boundary_transfer_004_d_point_world_m_f32": np.asarray([new_points[drow_to_volume[q]] for q in sorted(drow_to_volume)], np.float32),
    "fat614_boundary_transfer_004_old_internal_cut_faces": old_internal_cut,
    "fat614_boundary_transfer_004_new_internal_cut_faces": new_internal_cut,
    "fat614_boundary_transfer_004_expected_local_shell_offsets": np.r_[0, np.cumsum([len(case_data[ci]["expected_local"]) for ci in COMPONENTS])].astype(np.int64),
    "fat614_boundary_transfer_004_expected_local_shell_faces": np.vstack([case_data[ci]["expected_local"] for ci in COMPONENTS]),
})
np.savez_compressed(outnpz, **payload)
# Freeze hashes of each raw TetGen output table used.
inputs = {}
for name, (want, path) in EXPECTED.items():
    inputs[name] = {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}
input_table_records = {}
for ci in COMPONENTS:
    case = COMP7_CASE if ci == 7 else CASE_ROOT / f"component-{ci:04d}"
    for suffix in ("cavity.poly", "cavity.1.node", "cavity.1.ele", "cavity.1.face", "tetgen.stdout.log", "tetgen.stderr.log"):
        p = case / suffix
        if not p.is_file():
            continue
        rec = {"path": str(p), "sha256": sha(p), "size_bytes": p.stat().st_size}
        input_table_records[f"component{ci}_{suffix}"] = rec
op_records = []
for rec in d_run_report["outputs"]["operations"]:
    p = Path(rec["path"])
    got = sha(p)
    if got != rec["sha256"]:
        raise ValueError(f"run-005 operation hash mismatch: {p}")
    op_records.append({"path": str(p), "sha256": got, "size_bytes": p.stat().st_size})
def array_sha(a):
    x = np.ascontiguousarray(a)
    return hashlib.sha256(x.tobytes()).hexdigest()
report = {
    "schema": "numi.fat614.run005-audited-d-target-boundary-transfer.v4",
    "producer_script_path": str(Path(__file__)),
    "producer_script_sha256": sha(Path(__file__)),
    "status": "PASS_LOCAL_AUDITED_D_BOUNDARY_TRANSFER_TOPOLOGY_PROPOSAL_PENDING_COMPONENTS_9_12_MASS_AND_NATIVE_GATES",
    "physical_admission": False,
    "native_steps": 0,
    "mass_mapping_status": "PENDING_SOURCE330_CONSERVATIVE_REBIND",
    "source_only": True,
    "inputs": inputs,
    "run005_operation_records": op_records,
    "frozen_tetgen_table_records": input_table_records,
    "identity_336_branch_inherited_by_d4906": {
        "candidate_sha256": EXPECTED["descendant_336"][0],
        "independent_source_only_receipt_sha256": EXPECTED["descendant_336_independent"][0],
        "replacement_operation_groups_exactly_replayed_inside_d4906": int(len(identity336_ops)),
        "replacement_parent_rows_exactly_replayed": int(sum(len(pk) for pk in identity336_ops)),
        "replacement_tetrahedra_exact_corner_sets_replayed": int(sum(len(tk) for tk in identity336_ops.values())),
        "no_duplicate_identity_operations_added": True,
    },
    "corrected_component_sources": {
        "component3_candidate_sha256": EXPECTED["comp3_corrected_candidate"][0],
        "component3_report_sha256": EXPECTED["comp3_correction_report"][0],
        "component10_candidate_sha256": EXPECTED["comp10_corrected_candidate"][0],
        "component10_report_sha256": EXPECTED["comp10_correction_report"][0],
        "component7_candidate_sha256": EXPECTED["comp7_corrected_candidate"][0],
        "component7_validation_run_sha256": EXPECTED["comp7_validation_run"][0],
        "component7_current_target_prepare_sha256": EXPECTED["comp7_current_target_prepare"][0],
        "component7_coordinate_correction_sha256": EXPECTED["comp7_coordinate_correction"][0],
        "component7_validator_source_sha256": EXPECTED["comp7_validator_source"][0],
        "coordinate_changes_are_limited_to_receipt_pinned_interior_steiner_nodes": True,
    },
    "parent_d4906": {
        "candidate_sha256": EXPECTED["fat_parent_d4906"][0],
        "report_sha256": EXPECTED["fat_parent_d4906_report"][0],
        "cell_count": int(len(tdesc)), "point_count": int(len(pdesc)),
        "input_target_preserved_before_D_extension": True,
    },
    "D_surface": {
        "candidate_sha256": EXPECTED["final_d_run005"][0],
        "full_exact_audit_sha256": EXPECTED["final_d_complete_audit"][0],
        "actual_697_loader_sha256": EXPECTED["final_d_697_loader"][0],
        "run005_operation_count": len(d_run_ops),
        "D_points_added_to_fat": int(len(dfinalpoints) - len(d602points)),
        "edge_group_count": len(edge_groups),
        "target_parent_face_count": int(sum(len(v["parent_face_rows"]) for v in edge_groups.values())),
        "D_interface_reversed_for_Fat": True,
    },
    "components": component_results,
    "whole_candidate": {
        "candidate_path": str(outnpz),
        "candidate_sha256": sha(outnpz),
        "candidate_size_bytes": outnpz.stat().st_size,
        "points": int(len(new_points)), "cells": int(len(new_tets)),
        "desired_boundary_faces_before_after": [int(len(old_target)), int(len(desired_updated))],
        "actual_boundary_faces_before_after": [int(len(actual_before)), int(len(whole_boundary))],
        "oriented_boundary_mismatch_before": old_whole_diff,
        "oriented_boundary_mismatch_after": new_whole_diff,
        "complete_boundary_matches_updated_target": (
            new_whole_diff["missing_oriented_records"] == 0 and new_whole_diff["extra_oriented_records"] == 0),
        "target_full_D_interface_is_present_with_Fat_orientation": True,
        "old_internal_cut_face_count": int(sum(old_cut_counter.values())),
        "new_internal_cut_face_count": int(sum(new_cut_counter.values())),
        "bidirectional_oriented_internal_cut_sets_identical": True,
        "old_and_new_internal_cut_faces_each_opposed_by_untouched_tets": True,
        "exact_global_boundary_flux_delta_det6": int(global_new_flux - global_old_flux),
        "sum_exact_local_cavity_volume_deltas_det6": int(sum_local_delta),
        "global_flux_delta_equals_sum_local_deltas": True,
        "retained_and_interpatch_exact_F32_SAT": sat,
        "quality_failures_before": quality_before,
        "quality_failures_after": quality_after,
        "quality_failure_delta": quality_after - quality_before,
        "F64_metric_nonpositive_after": f64_nonpositive_after,
        "replacement_parent_partition_covers_partial_4602_domain": True,
    },
    "unresolved": {
        "other_failed_or_unreplayed_components": [9, 12],
        "prior_tetgen_return_code_200_attempts": "accepted only through independent exact geometric validation in this replay",
        "remaining_global_boundary_mismatch_is_preserved_and_reported": True,
        "source330_mass_mapping": "NOT_RUN",
        "native_execution": "NOT_RUN",
        "physical_admission": False,
    },
    "gates": {
        "exact_F32_positive_new_cells": True,
        "TetGen_orientation_normalization_only_reorders_corners": True,
        "Float64_vs_emitted_F32_raw_signs_recorded": True,
        "local_oriented_shells_exact": True,
        "local_exact_volume_flux": True,
        "bidirectional_internal_cut_identity": True,
        "untouched_tets_oppose_all_cut_faces": True,
        "exact_retained_interpatch_SAT": True,
        "full_audited_D_target_used": True,
        "full_boundary_complete": new_whole_diff["missing_oriented_records"] == 0 and new_whole_diff["extra_oriented_records"] == 0,
        "mass_mapping": "NOT_RUN",
        "native_execution": "NOT_RUN",
        "physical_admission": False,
    },
    "scope": (
        "Source-only Fat volume topology proposal extending frozen d4906 (which independently "
        "contains all 97 verified 336 identity operations plus comp2/6) with seven audited D target-edge "
        "cavity fills. Components 3 and 10 use their verified one-ULP interior F32 corrections; component 7 "
        "uses its rebuilt current-D PLC and verified interior-Steiner F32 correction. Components 9 and 12 "
        "and their resulting full-boundary mismatch remain unresolved. Mass, native execution and physical "
        "admission are not claimed."
    ),
}
report_path = RUN / "report.json"
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "status": report["status"],
    "candidate_path": str(outnpz),
    "candidate_sha256": sha(outnpz),
    "report_path": str(report_path),
    "report_sha256": sha(report_path),
    "points": len(new_points), "cells": len(new_tets),
    "components_old_new": {k: [v["old_cells"], v["new_cells"]] for k, v in component_results.items()},
    "boundary_mismatch_before_after": [old_whole_diff, new_whole_diff],
    "internal_cut_count": [sum(old_cut_counter.values()), sum(new_cut_counter.values())],
    "global_flux_delta": int(global_new_flux - global_old_flux),
    "sum_local_delta": int(sum_local_delta),
    "quality_failures": [quality_before, quality_after],
    "sat": sat,
}, indent=2))

