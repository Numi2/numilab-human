from pathlib import Path
import copy, gzip, hashlib, importlib.util, itertools, json, subprocess, time
import numpy as np

E = Path("/Users/n/numi-human-resting-evidence-20261005")
ROOT = E / "native-lung-free-apex-two-family-candidate-1177/source-trial-024"
OUT = ROOT / "row310-copy-self-audit-retry004"
BASE_NHA = E / "native-lung-free-apex-composition-1159-eightops-attempt1/final/resting-thorax.nhanatomy"
BASE_REPORT = E / "native-lung-free-apex-composition-1159-eightops-attempt1/composition-report.json"
V8 = E / "native-lung-final-selected-composition-dryrun-1113/provisional-1105-1106-v8"
PUB = Path("/Users/n/numi-human-final-compose-publication-001/tools/evidence/native-lung-final-selected-composition-1113/compose_candidate_v8.py")
RUNNER = E / "native-lung-late-pose-audit-runner-1171/audit_lung_cycle_1159.py"
SCAN = E / "native-lung-free-apex-two-family-candidate-1177/source-trial-017/scan_24_states_017.py"
CANDIDATE = ROOT / "candidate-rows.npz"
TRIAL = ROOT / "trial-report.json"
PROBE_REPORT = ROOT / "production-probe-24/probe-24-report.json"
LINEAGE = V8 / "row310-face-lineage.npy"
PROBE = Path("/Users/n/numi-human-resting-evidence-20261005/native-respiratory-kernel-saved-points-1119/probe")
METALLIB = Path("/Users/n/numi-human-retired-alias-visibility-build-018-attempt2/matter/shaders/HumanRespiration.metallib")
EXPECTED_NHA_SHA = "c10dce4609be99fdc569801c2705e23c46120b610c8aa86c631165d38baf4713"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def bits(x):
    return tuple(int(v) for v in np.asarray(x, dtype="<f4").view("<u4"))

def same_winding_bits(a, b):
    return any(tuple(a[(i + shift) % 3] for i in range(3)) == tuple(b[i] for i in range(3)) for shift in range(3))

if OUT.exists():
    raise SystemExit("fresh output required")
OUT.mkdir()
c = load(PUB, "frozen_v8_composer_copy_audit")
scan = load(SCAN, "scan_017_helpers_for_024_copy_audit")
runner = load(RUNNER, "late_pose_audit_1171_copy_audit")
if sha(BASE_NHA) != EXPECTED_NHA_SHA:
    raise SystemExit("base NHA changed")
head, base_rows = c.parse_payload(BASE_NHA)
trial = json.loads(TRIAL.read_text())
with np.load(CANDIDATE) as z:
    cand_rows = {}
    for sid in (305, 308, 310):
        cand_rows[sid] = {"vertices6": z[f"row{sid}_vertices6"].copy(), "faces": z[f"row{sid}_faces"].copy()}
lineage = np.load(LINEAGE, allow_pickle=False)
changed_rows = trial["changed_rows"]
changed310 = set(map(int, changed_rows["310"]["face_ids"]))
if lineage.shape != (len(cand_rows[310]["faces"]), 2):
    raise SystemExit("row310 lineage does not cover candidate face rows")
for sid in (305, 308, 310):
    if not np.array_equal(cand_rows[sid]["faces"], base_rows[sid]["faces"]):
        raise SystemExit("candidate changed source face order row%d" % sid)

copy_specs = [
    {"owner": 308, "owner_vertex": 10628, "pleura_vertex": 101392},
    {"owner": 305, "owner_vertex": 21520, "pleura_vertex": 21523},
    {"owner": 305, "owner_vertex": 21522, "pleura_vertex": 21525},
]
source_copy_checks = []
for item in copy_specs:
    sid, vid, pvid = item["owner"], item["owner_vertex"], item["pleura_vertex"]
    old_owner = base_rows[sid]["vertices6"][vid, :3]
    old_copy = base_rows[310]["vertices6"][pvid, :3]
    new_owner = cand_rows[sid]["vertices6"][vid, :3]
    new_copy = cand_rows[310]["vertices6"][pvid, :3]
    if bits(old_owner) != bits(old_copy) or bits(new_owner) != bits(new_copy):
        raise SystemExit("row310 owner coordinate copy mismatch %s" % item)
    source_copy_checks.append({
        **item,
        "old_coordinate_bits_equal": True,
        "candidate_coordinate_bits_equal": True,
        "old_xyz": old_owner.astype(float).tolist(),
        "candidate_xyz": new_owner.astype(float).tolist(),
    })

changed_face_copy_rows = []
triangle_point_pairs = []
for fi in sorted(changed310):
    owner_sid, owner_fi = map(int, lineage[fi])
    if owner_sid not in (305, 308) or owner_fi not in changed_rows[str(owner_sid)]["face_ids"]:
        raise SystemExit("changed row310 face lacks expected changed owner lineage: %d -> %d:%d" % (fi, owner_sid, owner_fi))
    tri310_ids = cand_rows[310]["faces"][fi]
    tri_owner_ids = cand_rows[owner_sid]["faces"][owner_fi]
    tri310 = [bits(cand_rows[310]["vertices6"][int(v), :3]) for v in tri310_ids]
    tri_owner = [bits(cand_rows[owner_sid]["vertices6"][int(v), :3]) for v in tri_owner_ids]
    if not same_winding_bits(tri310, tri_owner):
        raise SystemExit("row310 changed face is not exact same-wound owner copy %d -> %d:%d" % (fi, owner_sid, owner_fi))
    shift = next(s for s in range(3) if all(tri310[i] == tri_owner[(i + s) % 3] for i in range(3)))
    for i in range(3):
        pleura_vid = int(tri310_ids[i])
        owner_vid = int(tri_owner_ids[(i + shift) % 3])
        triangle_point_pairs.append({
            "label": "face310:%d:%d:%d" % (fi, i, owner_sid),
            "pleura_vertex": pleura_vid,
            "owner": owner_sid,
            "owner_vertex": owner_vid,
            "xyz_f32": cand_rows[310]["vertices6"][pleura_vid, :3].astype(float).tolist(),
            "pleura_source_bits": list(tri310[i]),
            "owner_source_bits": list(tri_owner[(i + shift) % 3]),
        })
    changed_face_copy_rows.append({
        "row310_face": fi,
        "owner_row": owner_sid,
        "owner_face": owner_fi,
        "same_winding_float32_triangle_copy": True,
    })

if len(changed_face_copy_rows) != len(changed310):
    raise SystemExit("not all changed row310 faces were validated")

# Exact source-coordinate self admission for only the changed row310 star.
# This is separate from the six-owner native audit because row310 is a derived external-union copy.
step = 0
early = scan.COHORTS["early8"]["run"]
cfg = {
    "base": str(runner.BASE),
    "run": early,
    "out": OUT / "context",
    "nha_path": BASE_NHA,
    "nha_sha": EXPECTED_NHA_SHA,
    "map_reports": [],
    "lobe_lineage_report": scan.LREPORT,
    "d_map_composition_report": scan.DREPORT,
    "geometry_only_area_mismatch": False,
    "workers": 1,
    "probe_report": None,
    "registered_arm": None,
}
base, adapters, ctx = runner.prepare_worker_context(cfg)
v310 = cand_rows[310]["vertices6"]
f310 = cand_rows[310]["faces"]
records, bad = base.exact_records(v310[:, :3], f310)
changed_records = [r for r in records if int(r[3]) in changed310]
seen = set()
self_hits = []
unallowed = 0
for x, y in base.pred()._aabb_candidate_pairs(changed_records, records, same_surface=False):
    if int(x[3]) == int(y[3]):
        continue
    pair = tuple(sorted((int(x[3]), int(y[3]))))
    if pair in seen:
        continue
    pts = base.pred().triangle_intersection_points(x[0], y[0])
    if not pts:
        continue
    seen.add(pair)
    common = set(map(int, x[4])) & set(map(int, y[4]))
    shared_points = {tuple(x[0][x[4].index(i)]) for i in common}
    allowed = len(common) in (1, 2) and all(base.pred()._allowed_shared_point(p, shared_points) for p in pts)
    label = "allowed_indexed_adjacency" if allowed else "unallowed_self_intersection"
    unallowed += int(not allowed)
    self_hits.append({"faces": list(pair), "class": label, "point_count": len(pts)})
if set(changed310) & set(map(int, bad)):
    raise SystemExit("degenerate row310 changed face")
if unallowed:
    raise SystemExit("row310 changed-star source self intersection")

# Query the unchanged production Metal point mapper with old owner/copy points and
# every changed row310 face corner paired with its exact lobe-owner corner.
point_records = []
old_owner_indices = []
copy_output_pairs = []
for item in copy_specs:
    sid, vid, pvid = item["owner"], item["owner_vertex"], item["pleura_vertex"]
    for role, row, vertex in [
        ("old_owner", base_rows[sid], vid),
        ("old_pleura_copy", base_rows[310], pvid),
    ]:
        point_records.append({"label": "%s:%d:%d" % (role, sid, vertex), "xyz": row["vertices6"][vertex, :3].astype("<f4").tolist()})
    old_owner_indices.append(len(point_records) - 2)
    copy_output_pairs.append([len(point_records) - 2, len(point_records) - 1])
    for role, row, vertex in [
        ("candidate_owner", cand_rows[sid], vid),
        ("candidate_pleura_copy", cand_rows[310], pvid),
    ]:
        point_records.append({"label": "%s:%d:%d" % (role, sid, vertex), "xyz": row["vertices6"][vertex, :3].astype("<f4").tolist()})
    copy_output_pairs.append([len(point_records) - 2, len(point_records) - 1])

face_corner_pair_indices = []
for pair in triangle_point_pairs:
    source = np.asarray(pair["xyz_f32"], dtype="<f4")
    start = len(point_records)
    point_records.append({"label": pair["label"] + ":lobe", "xyz": source.tolist()})
    point_records.append({"label": pair["label"] + ":row310", "xyz": source.tolist()})
    face_corner_pair_indices.append([start, start + 1])
    copy_output_pairs.append([start, start + 1])
points = np.asarray([x["xyz"] for x in point_records], dtype="<f4")
pointfile = OUT / "source-points.f32.bin"
pointfile.write_bytes(points.tobytes())
index_path = OUT / "source-points-index.json"
index_path.write_text(json.dumps(point_records, indent=2, sort_keys=True) + "\n")

probe_reference = json.loads(PROBE_REPORT.read_text())
if sha(PROBE) != probe_reference["probe"]["sha256"] or sha(METALLIB) != probe_reference["metallib"]["sha256"]:
    raise SystemExit("production point probe or Metal library changed")
base_receipt = Path(probe_reference["base_receipt"]["path"])
if sha(base_receipt) != probe_reference["base_receipt"]["sha256"]:
    raise SystemExit("probe base receipt changed")
cohort_specs = {
    "early8": (E / "final-native-scene-preflight-936/skin-927-lung-1159-viewer-018-v015-attempt1/native-run", None),
    "baseline8": (Path("/Users/n/numi-human-retained-delivery-20261009/native-integrated-resting-study-1170/trials/resting-baseline"), "control"),
    "drive_half8": (Path("/Users/n/numi-human-retained-delivery-20261009/native-integrated-resting-study-1170/trials/resting-drive-half"), "treatment"),
}
cohort_results = {}
for cohort, (run_path, arm) in cohort_specs.items():
    cohort_doc = probe_reference["cohorts"][cohort]
    steps = cohort_doc["steps"]
    receipts = [Path(x["path"]) for x in cohort_doc["accepted_receipts"]]
    params = Path(cohort_doc["scene"]) / "skin-source-anatomy-parameters.bin"
    if [int(x.stem.split("-")[1].split(".")[0]) for x in receipts] != list(map(int, steps)):
        raise SystemExit("accepted receipt schedule mismatch " + cohort)
    probe_out = OUT / cohort
    argv = [
        str(PROBE), str(METALLIB), str(params), str(base_receipt),
        str(pointfile), str(probe_out), *map(str, receipts),
    ]
    completed = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (probe_out / "stdout.txt").write_text(completed.stdout)
    if completed.returncode != 0:
        raise SystemExit("production probe failed " + cohort + "\n" + completed.stdout)
    cfg = dict(cfg)
    cfg["run"] = run_path
    cfg["registered_arm"] = arm
    base_for_run, _, run_ctx = runner.prepare_worker_context(cfg)
    if list(map(int, run_ctx["steps"])) != list(map(int, steps)):
        raise SystemExit("registered capture schedule mismatch " + cohort)
    pose_rows = []
    equality_checks = 0
    for step in steps:
        output_path = probe_out / ("step-%d.receipt.xyz-f32.bin" % int(step))
        mapped = np.fromfile(output_path, dtype="<f4")
        if mapped.size != len(point_records) * 3:
            raise SystemExit("point output length mismatch " + str(output_path))
        mapped = mapped.reshape((-1, 3))
        if any(not np.array_equal(mapped[a].view("<u4"), mapped[b].view("<u4")) for a, b in copy_output_pairs):
            raise SystemExit("production mapped copy differs at %s:%d" % (cohort, step))
        pose, pack = base_for_run.row_pose(run_ctx["run_path"], int(step), run_ctx["nha_rows"])
        for item, index in zip(copy_specs, old_owner_indices):
            sid, vid = item["owner"], item["owner_vertex"]
            actual = pose[sid]["v"][vid, :3]
            if not np.array_equal(actual.view("<u4"), mapped[index].view("<u4")):
                raise SystemExit("old owner point kernel parity failed %s:%d:%d" % (cohort, step, sid))
        equality_checks += len(copy_output_pairs)
        pose_rows.append({
            "step": int(step),
            "accepted_pack_path": str(pack),
            "accepted_pack_sha256": sha(pack),
            "probe_output_path": str(output_path),
            "probe_output_sha256": sha(output_path),
            "old_owner_points_match_actual_capture": True,
            "candidate_row310_copy_and_lobe_owner_outputs_bit_equal": True,
            "mapped_copy_pair_count": len(copy_output_pairs),
        })
    cohort_results[cohort] = {
        "scene_path": str(run_ctx["run_path"]),
        "accepted_steps": list(map(int, steps)),
        "parameters_sha256": sha(params),
        "accepted_capture_identity": "validated by the 1171 accepted standalone/registered context",
        "steps": pose_rows,
        "exact_duplicate_map_pairs_checked_per_pose": len(copy_output_pairs),
    }

report = {
    "schema": "numi.human.lung-row310-external-union-copy-self-audit.v1",
    "status": "complete_source_lineage_and_saved_point-kernel_copy_check",
    "scope": "The native six-owner scan excludes derived row310 by design. This report checks changed row310 faces against its exact external-union owner lineage, strict source self-intersection admission for the changed row310 star, and deterministic production-kernel equality for identical lobe/row310 source points across the same 24 accepted states. Row310 is not present in accepted MRVPack surface rows, so no claim of a separately captured row310 world mesh is made.",
    "inputs": {
        "base_nha_path": str(BASE_NHA), "base_nha_sha256": sha(BASE_NHA),
        "trial_report_path": str(TRIAL), "trial_report_sha256": sha(TRIAL),
        "candidate_npz_path": str(CANDIDATE), "candidate_npz_sha256": sha(CANDIDATE),
        "row310_lineage_path": str(LINEAGE), "row310_lineage_sha256": sha(LINEAGE),
        "composer_path": str(PUB), "composer_sha256": sha(PUB),
        "audit_runner_path": str(RUNNER), "audit_runner_sha256": sha(RUNNER),
        "point_probe_path": str(PROBE), "point_probe_sha256": sha(PROBE),
        "metallib_path": str(METALLIB), "metallib_sha256": sha(METALLIB),
        "exact_predicate_path": str(scan.RUNNER_PATH), "exact_predicate_sha256": sha(scan.RUNNER_PATH),
        "prior_composition_report_path": str(BASE_REPORT), "prior_composition_report_sha256": sha(BASE_REPORT),
        "point_file_path": str(pointfile), "point_file_sha256": sha(pointfile),
        "point_index_path": str(index_path), "point_index_sha256": sha(index_path),
    },
    "owner_copy_vertices": source_copy_checks,
    "changed_row310_face_lineage": changed_face_copy_rows,
    "row310_source_self_admission": {
        "changed_face_count": len(changed310),
        "degenerate_changed_face_count": 0,
        "exact_changed_star_self_hit_count": len(self_hits),
        "unallowed_self_intersections": unallowed,
        "hit_classes": {"allowed_indexed_adjacency": len(self_hits)},
        "events": self_hits,
        "scope": "candidate source-coordinate triangles, changed row310 faces against all row310 source faces",
    },
    "production_kernel_copy_check": {
        "probe_invocation_points": len(point_records),
        "source_copy_pairs_per_pose": len(copy_output_pairs),
        "source_owner_actual_capture_parity": True,
        "row310_captured_in_native_mrvpack": False,
        "cohorts": cohort_results,
    },
}
out_path = OUT / "report.json"
out_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
print(json.dumps({
    "report_path": str(out_path),
    "report_sha256": sha(out_path),
    "source_copied_faces": len(changed_face_copy_rows),
    "row310_self_hits": len(self_hits),
    "row310_unallowed_self": unallowed,
    "row310_degenerate": len(set(changed310) & set(map(int, bad))),
    "cohort_steps": {k: len(v["steps"]) for k, v in cohort_results.items()},
    "duplicate_map_pairs_per_pose": len(copy_output_pairs),
}, indent=2))
