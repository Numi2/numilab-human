#!/usr/bin/env python3
from pathlib import Path
import gc, hashlib, importlib.util, json, platform, signal, sys, time
import numpy as np

P = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
OUT = Path(__file__).resolve().parent / "run-001"
O_PATH = P / "surface-transfer-checkpoint-392/payload/O/candidate.npz"
I_PATH = P / "geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/dnew-vs-i296-inventory-001/i328-cavity-audit-001/fill-003/I-f32-conditioning-367/condition-288-pair-connected-remesh-001-run-002/result/candidate-f32-typed.npz"
D_PATH = P / "geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/fine-I-bone-barrier-run-024/full-native-self-028/current-collar-plan-032/engine-20-ops-one-sided624-run-006/candidate-f32-linked-remesh-typed.npz"
SKIN_CLOSED = P / "paired-fine-headwall-rim-635/run-013/output/fine-skin-closed-boundary.npz"
SKIN_RIM = P / "paired-fine-headwall-rim-635/run-013/output/fine-skin-rim-oriented-faces.npz"
SKIN_MAP = P / "paired-fine-headwall-rim-635/run-013/output/fine-rim-source-map.npz"
SKIN_REPORT = P / "paired-fine-headwall-rim-635/run-013/output/report.json"
FAT_CLOSED = P / "paired-fine-headwall-rim-635/run-010/output/fine-fat-closed-boundary.npz"
FAT_RIM = P / "paired-fine-headwall-rim-635/run-010/output/fine-fat-rim-oriented-faces.npz"
FAT_MAP = P / "paired-fine-headwall-rim-635/run-010/output/fine-rim-source-map.npz"
FAT_REPORT = P / "paired-fine-headwall-rim-635/run-010/output/report.json"
DI_REPORT = P / "whole-D-fine-I-interface-627/run-001/report.json"
LOADER = P / "geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py"
EXPECTED = {
    O_PATH: "2958a704db4b8b68422640a5e991c6b51af25f19d525c49adc8a742f0c207d7a",
    I_PATH: "5d46ee99df2613e9f30a43075af95a6402beffe26392cd9f017d8c1a154bf410",
    D_PATH: "c17794a1012f76924dcd9e4e754bb026c774deb5a3d37e2f1c79635ccfe9dc83",
    SKIN_CLOSED: "352dc5e1ebfeca7f8c7dc13075dbcf0abc8d15a842f52057ba7744f998d5fa23",
    SKIN_RIM: "2e46017738573dacd4c31c6ff7002c09427026db671ebed98a3a65a1154f94f9",
    FAT_CLOSED: "b02a064f088e625dfb11c98fd2859e13bd271224bd13d050c7b848c217d856b2",
    FAT_RIM: "6f2141cc5998b01e2fb1183bb70ed26ffefdbb6bd3f3090b0570fa17143e7ffd",
    SKIN_REPORT: "24ed76b7725d1b097e92d1859662b9725532e651abea5052f42dc0f14fb87b8c",
    FAT_REPORT: "65ca93243fbec108e48ee8151f80994721ee65f3e689d7f8fdaf141388eb9e75",
    DI_REPORT: "ea5ef3a3879f4b9e78ae49f0c4dfe8c699e718825cba20b573f03bf03bba22f3",
}
EXACT_OWNER_PATH = Path("/Users/numi/Work/numi-human-connected-surface-069/D/publication-206-stage-001/tools/cardiac_cavity_intersections.py")
EXACT_OWNER_SHA = "f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d"
LOADER_SHA = "f1d697b42fc9d97472fbd70cec2d8adfe178bb591d3513680609396ec4c204ad"
START = None

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

def progress(stage, **fields):
    item = {"stage": stage, "elapsed_s": time.monotonic() - START, **fields}
    write_json(OUT / "progress.json", item)
    print(json.dumps(item, sort_keys=True), flush=True)

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: np.asarray(z[key]) for key in z.files}

def validate_points(points, label):
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"{label}: invalid finite point array")
    emitted = points.astype(np.float32).astype(np.float64)
    if not np.array_equal(points.astype(np.float64), emitted):
        raise ValueError(f"{label}: coordinates are not exact Float32 emissions")
    return emitted

def load_open(path, role):
    z = load_npz(path)
    points = validate_points(z["points_world_m"], f"{role} source")
    faces = np.asarray(z["faces"], dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"{role}: invalid face array")
    if np.any(faces < 0) or np.any(faces >= len(points)):
        raise ValueError(f"{role}: face row outside point domain")
    return points, faces

def reconstruct_closed(role):
    if role == "skin":
        closed_path, rim_path, map_path = SKIN_CLOSED, SKIN_RIM, SKIN_MAP
        left_path, right_path = O_PATH, I_PATH
        left_name, right_name = "O", "I"
        report_path = SKIN_REPORT
    elif role == "fat":
        closed_path, rim_path, map_path = FAT_CLOSED, FAT_RIM, FAT_MAP
        left_path, right_path = I_PATH, D_PATH
        left_name, right_name = "I", "D"
        report_path = FAT_REPORT
    else:
        raise ValueError(role)

    left_points, left_faces = load_open(left_path, left_name)
    right_points, right_faces = load_open(right_path, right_name)
    closed = load_npz(closed_path)
    rim_file = load_npz(rim_path)
    source_map = load_npz(map_path)
    points_key = "points_world_m_f64"
    points = np.asarray(closed[points_key], dtype=np.float64)
    if not np.array_equal(points.astype(np.float32).astype(np.float64), points):
        raise ValueError(f"{role}: closed boundary points are not canonical F32")
    rim_original = np.asarray(closed["oriented_rim_faces"], dtype=np.int64)
    rim_companion = np.asarray(rim_file["oriented_rim_faces"], dtype=np.int64)
    source_field = "fine_skin_rim_faces" if role == "skin" else "fine_fat_rim_faces"
    if not np.array_equal(rim_original, rim_companion):
        raise ValueError(f"{role}: closed rim faces differ from oriented rim artifact")
    if not np.array_equal(rim_original, source_map[source_field]):
        raise ValueError(f"{role}: rim face map differs from closed-boundary artifact")

    source_points = np.concatenate((left_points, right_points), axis=0)
    if role == "skin":
        expected_left = left_faces
        expected_right = right_faces[:, [0, 2, 1]] + len(left_points)
        saved_left = np.asarray(closed["fine_O_faces"], dtype=np.int64)
        saved_right = np.asarray(closed["reversed_fine_I_faces"], dtype=np.int64)
    else:
        expected_left = left_faces
        expected_right = right_faces[:, [0, 2, 1]] + len(left_points)
        saved_left = np.asarray(closed["fine_I_faces"], dtype=np.int64)
        saved_right = np.asarray(closed["reversed_fine_D_faces"], dtype=np.int64)
    if not np.array_equal(saved_left, expected_left):
        raise ValueError(f"{role}: closed package left source faces changed")
    if not np.array_equal(saved_right, expected_right):
        raise ValueError(f"{role}: closed package right source faces changed")
    full_faces = np.vstack((expected_left, expected_right, rim_original))
    used = np.unique(full_faces)
    if np.any(full_faces < 0) or np.any(full_faces >= len(source_points)):
        raise ValueError(f"{role}: source face index outside combined point rows")
    if not np.array_equal(points, source_points[used].astype(np.float64)):
        raise ValueError(f"{role}: compact boundary points do not reconstruct from source arrays")
    inverse = np.full(len(source_points), -1, dtype=np.int64)
    inverse[used] = np.arange(len(used), dtype=np.int64)
    compact_faces = inverse[full_faces]
    if not np.array_equal(np.asarray(closed["faces"], dtype=np.int64), compact_faces):
        raise ValueError(f"{role}: closed boundary face rows do not reconstruct from source faces")
    rim_compact = inverse[rim_original]
    if np.any(rim_compact < 0):
        raise ValueError(f"{role}: rim references an unused compact point")
    report = json.loads(report_path.read_text())
    expected_status = "PASS_FINE_PAIRED_SKIN_RIM_GEOMETRY_DIAGNOSTIC" if role == "skin" else "PASS_FINE_PAIRED_RIM_GEOMETRY_DIAGNOSTIC"
    if report.get("status") != expected_status:
        raise ValueError(f"{role}: frozen source-boundary report status mismatch")
    report_closed_key = "closed_skin_boundary" if role == "skin" else "closed_fat_boundary"
    if not report.get(report_closed_key, {}).get("oriented_edge_closed"):
        raise ValueError(f"{role}: frozen source-boundary report does not certify closed oriented shell")
    return points, rim_compact, np.asarray(rim_file["source_headwall_face_rows"], dtype=np.int64)

def get_pair(name):
    if name == "O-I":
        return (*load_open(O_PATH, "O"), *load_open(I_PATH, "I")), {
            "A": "complete open fine O; original O face rows",
            "B": "complete open fine I; original I face rows",
            "a_face_source_rows": None, "b_face_source_rows": None}
    if name == "O-D":
        return (*load_open(O_PATH, "O"), *load_open(D_PATH, "D")), {
            "A": "complete open fine O; original O face rows",
            "B": "complete open fine D; original D face rows",
            "a_face_source_rows": None, "b_face_source_rows": None}
    if name == "skin-rim-D":
        rim_points, rim_faces, rim_source_rows = reconstruct_closed("skin")
        d_points, d_faces = load_open(D_PATH, "D")
        return (rim_points, rim_faces, d_points, d_faces), {
            "A": "90 authenticated fine-skin headwall rim faces in compact closed-skin point domain",
            "B": "complete open fine D; original D face rows",
            "a_face_source_rows": rim_source_rows, "b_face_source_rows": None}
    if name == "fat-rim-O":
        rim_points, rim_faces, rim_source_rows = reconstruct_closed("fat")
        o_points, o_faces = load_open(O_PATH, "O")
        return (rim_points, rim_faces, o_points, o_faces), {
            "A": "90 authenticated fine-fat headwall rim faces in compact closed-fat point domain",
            "B": "complete open fine O; original O face rows",
            "a_face_source_rows": rim_source_rows, "b_face_source_rows": None}
    raise ValueError(name)

def build_records(pred, name):
    (a_points, a_faces, b_points, b_faces), info = get_pair(name)
    ar, _ = pred.float32_surface_records(a_points, a_faces, source_seam_groups=())
    br, _ = pred.float32_surface_records(b_points, b_faces, source_seam_groups=())
    return ar, br, info, len(a_points), len(a_faces), len(b_points), len(b_faces)

def iter_aabb(pred, second_records, first_index):
    # Query order follows the 627 exact owner convention; distinct source
    # surfaces use same_surface=False, so coincident contacts are never waived.
    for second, first in pred._query_surface_aabb(second_records, first_index):
        yield first, second

def count_candidates(pred, a_records, b_records):
    index = pred._prepare_surface_aabb(a_records)
    return sum(1 for _ in iter_aabb(pred, b_records, index))

def first_exact_crossing(pred, first, second):
    exact = pred._audit_record_pairs(((first, second),), same_surface=False)
    if not exact["count"]:
        return None
    pts = pred.triangle_intersection_points(first[0], second[0])
    return {
        "first_face_row": int(first[3]),
        "second_face_row": int(second[3]),
        "first_triangle_lattice": [[str(v) for v in p] for p in first[0]],
        "second_triangle_lattice": [[str(v) for v in p] for p in second[0]],
        "exact_intersection_points_lattice": [
            [str(v) for v in p] for p in pts
        ],
    }

def audit_pair(pred, name, preflight_entry):
    ar, br, info, ap, af, bp, bf = build_records(pred, name)
    if af != preflight_entry["A_faces"] or bf != preflight_entry["B_faces"]:
        raise ValueError(f"{name}: row counts changed between preflight and exact pass")
    idx = pred._prepare_surface_aabb(ar)
    scanned = 0
    witness = None
    for first, second in iter_aabb(pred, br, idx):
        scanned += 1
        witness = first_exact_crossing(pred, first, second)
        if witness is not None:
            if info["a_face_source_rows"] is not None:
                witness["first_source_headwall_face_row"] = int(info["a_face_source_rows"][witness["first_face_row"]])
            break
        if scanned % 250000 == 0:
            progress("exact-narrow-phase", pair=name, aabb_candidates_scanned=scanned,
                     aabb_candidate_total=preflight_entry["aabb_candidate_pairs"])
    result = {
        "pair": name,
        "status": "FAILED_FIRST_EXACT_CROSSING" if witness else "PASS_ZERO_EXACT_PAIRS",
        "decision_complete": True,
        "exhaustive": witness is None,
        "aabb_candidate_pairs_total": int(preflight_entry["aabb_candidate_pairs"]),
        "aabb_candidate_pairs_exactly_scanned": scanned,
        "exact_pair_count": 1 if witness else 0,
        "first_witness": witness,
        "A": info["A"], "B": info["B"],
        "A_point_rows": ap, "A_faces": af, "B_point_rows": bp, "B_faces": bf,
        "source_row_mapping": "A rim row maps through pinned source_headwall_face_rows when applicable; all open surface rows are original source face rows",
    }
    del ar, br, idx
    gc.collect()
    return result

def main():
    global START
    START = time.monotonic()
    OUT.mkdir(exist_ok=False)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("bounded four-interface exact audit timeout")))
    signal.alarm(360)
    inputs = list(EXPECTED) + [SKIN_MAP, FAT_MAP, LOADER, EXACT_OWNER_PATH]
    pins = {str(path): sha(path) for path in inputs}
    for path, expected in EXPECTED.items():
        if pins[str(path)] != expected:
            raise ValueError(f"pinned input changed: {path}")
    if pins[str(LOADER)] != LOADER_SHA:
        raise ValueError("627 loader source changed")
    if pins[str(EXACT_OWNER_PATH)] != EXACT_OWNER_SHA:
        raise ValueError("exact predicate source changed")
    exact_owner_sha = pins[str(EXACT_OWNER_PATH)]
    loader = load_module(LOADER, "cross_surface_loader654")
    pred = loader.load_exact_owner()
    if sha(pred.__file__) != EXACT_OWNER_SHA:
        raise ValueError("loader resolved a different exact predicate owner")
    di = json.loads(DI_REPORT.read_text())
    if (di.get("status") != "PASS_SURFACE_PAIR_GATE" or not di.get("complete")
        or di.get("exact_triangle_pair_count") != 0
        or di.get("D_sha256") != EXPECTED[D_PATH] or di.get("I_sha256") != EXPECTED[I_PATH]):
        raise ValueError("retained D-I open-surface proof no longer matches frozen inputs")
    for role, report_path in (("skin", SKIN_REPORT), ("fat", FAT_REPORT)):
        # Reconstructing verifies the closed-shell artifacts against their
        # original open source rows and retained paired-rim map.
        reconstruct_closed(role)
        gc.collect()

    wrapper = str(P / "run_geometry_serial.py")
    python = "/Users/numi/numi-human-prep-venv-20261005/bin/python"
    write_json(OUT / "launch.json", {
        "schema": "numi.complete-cross-surface-pair-audit-654.launch.v1",
        "execution_host": "numi",
        "command": [python, wrapper, python, str(Path(__file__).resolve())],
        "python": sys.version,
        "platform": platform.platform(),
        "script_sha256": sha(__file__),
        "input_pins": pins,
        "retained_D_I_report_sha256": EXPECTED[DI_REPORT],
        "exact_owner_sha256": exact_owner_sha,
        "physical_admission": False,
        "runtime_created": 0,
        "accepted_physical_steps": 0,
        "expected_pairs": ["O-I", "O-D", "skin-rim-D", "fat-rim-O"],
    })

    names = ["O-I", "O-D", "skin-rim-D", "fat-rim-O"]
    preflight = {
        "schema": "numi.complete-cross-surface-pair-audit-654.preflight.v1",
        "input_pins": pins,
        "retained_D_I_pass": {
            "report_path": str(DI_REPORT),
            "sha256": EXPECTED[DI_REPORT],
            "exact_pair_count": 0,
            "AABB_candidate_pairs": int(di["aabb_candidate_pairs"]),
            "scope": di["scope"],
        },
        "pairs": {},
        "exact_narrow_phase_started": False,
        "physical_admission": False,
    }
    for name in names:
        ar, br, info, ap, af, bp, bf = build_records(pred, name)
        candidate_count = count_candidates(pred, ar, br)
        preflight["pairs"][name] = {
            "A": info["A"], "B": info["B"],
            "A_point_rows": ap, "A_faces": af, "B_point_rows": bp, "B_faces": bf,
            "aabb_candidate_pairs": candidate_count,
            "all_AABB_candidates_counted_before_any_new_exact_audit": True,
        }
        progress("pair-aabb-preflight-complete", pair=name, A_faces=af,
                 B_faces=bf, aabb_candidate_pairs=candidate_count)
        del ar, br
        gc.collect()
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise ValueError(f"input changed before exact stage: {path}")
    preflight["exact_narrow_phase_started"] = True
    write_json(OUT / "preflight.json", preflight)
    progress("preflight-pinned-and-saved-before-exact-audits",
             pair_candidate_counts={k: v["aabb_candidate_pairs"] for k, v in preflight["pairs"].items()})

    results = []
    for name in names:
        result = audit_pair(pred, name, preflight["pairs"][name])
        results.append(result)
        write_json(OUT / "partial-results.json", {
            "schema": "numi.complete-cross-surface-pair-audit-654.partial.v1",
            "completed_pairs": results,
            "physical_admission": False,
            "runtime_created": 0,
            "accepted_physical_steps": 0,
        })
        progress("pair-decision-recorded", pair=name, status=result["status"],
                 exact_pair_count=result["exact_pair_count"],
                 exhaustive=result["exhaustive"])
        if result["exact_pair_count"]:
            # The task requires retaining the first genuine exact witness and
            # stopping further pair work instead of iterating geometry fixes.
            break
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise ValueError(f"input changed during exact stage: {path}")
    status = "PASS_ALL_REQUESTED_CROSS_LAYER_PAIRS" if len(results) == len(names) and all(
        x["status"] == "PASS_ZERO_EXACT_PAIRS" and x["exhaustive"] for x in results
    ) else "CROSS_LAYER_PAIR_WITNESS_OR_INCOMPLETE"
    report = {
        "schema": "numi.complete-cross-surface-pair-audit-654.result.v1",
        "status": status,
        "complete": len(results) == len(names),
        "input_pins": pins,
        "retained_D_I_report_sha256": EXPECTED[DI_REPORT],
        "preflight_path": str(OUT / "preflight.json"),
        "preflight_sha256": sha(OUT / "preflight.json"),
        "pairs": results,
        "physical_admission": False,
        "runtime_created": 0,
        "accepted_physical_steps": 0,
        "scope": "exact source open O-I and O-D plus authenticated skin-headwall-rim-D and fat-headwall-rim-O; distinct-surface contacts receive no shared-vertex/edge waiver; no runtime or physical steps",
        "elapsed_s": time.monotonic() - START,
    }
    write_json(OUT / "report.json", report)
    write_json(OUT / "witnesses.json", {
        "schema": "numi.complete-cross-surface-pair-audit-654.witnesses.v1",
        "witnesses": [
            {"pair": x["pair"], **x["first_witness"]}
            for x in results if x["first_witness"] is not None
        ],
    })
    print(json.dumps({
        "status": status,
        "complete": report["complete"],
        "preflight": {k: v["aabb_candidate_pairs"] for k, v in preflight["pairs"].items()},
        "results": [{"pair": x["pair"], "status": x["status"],
                     "aabb_total": x["aabb_candidate_pairs_total"],
                     "aabb_scanned": x["aabb_candidate_pairs_exactly_scanned"],
                     "exact": x["exact_pair_count"], "exhaustive": x["exhaustive"]}
                    for x in results],
        "elapsed_s": report["elapsed_s"],
        "runtime_created": 0, "accepted_physical_steps": 0,
    }, sort_keys=True), flush=True)
    signal.alarm(0)

if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        out = OUT
        if out.exists():
            error = {"status": "INCOMPLETE", "complete": False, "error": repr(exc),
                     "elapsed_s": None if START is None else time.monotonic() - START,
                     "physical_admission": False, "runtime_created": 0, "accepted_physical_steps": 0}
            write_json(out / "failure.json", error)
            print(json.dumps(error), file=sys.stderr, flush=True)
        raise
