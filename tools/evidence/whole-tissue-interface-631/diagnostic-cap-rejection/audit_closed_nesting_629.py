#!/usr/bin/env python3
"""Exact closure-cap intersections and component nesting for final open D."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import argparse, csv, hashlib, importlib.util, itertools, json, math, signal, sys, time
import numpy as np

P = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
D = P / "geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/fine-I-bone-barrier-run-024/full-native-self-028/current-collar-plan-032/engine-20-ops-one-sided624-run-006/candidate-f32-linked-remesh-typed.npz"
OPERATIONS = D.parent / "accepted-patch-operations.json"
D_CAP_SOURCE = P / "geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/occupied-fat-run007-001/closed-inputs/D-closed-run007-f32.npz"
I = P / "geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/dnew-vs-i296-inventory-001/i328-cavity-audit-001/fill-003/I-f32-conditioning-367/condition-288-pair-connected-remesh-001-run-002/result/candidate-f32-typed.npz"
I_CLOSED = P / "geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/occupied-fat-run007-001/closed-inputs/I-closed-current-f32.npz"
I_SELF = I.parent / "exact-native-387-001/self.json"
D_SELF = P / "complete-locality-self-replay-622/joint15-run-002/report.json"
D_I_OPEN = P / "whole-D-fine-I-interface-627/run-001/report.json"
EXACT_OWNER = P / "geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py"
EXPECTED = {
    "D": "c17794a1012f76924dcd9e4e754bb026c774deb5a3d37e2f1c79635ccfe9dc83",
    "operations": "6baddfa15e7fe40dd608b97eeca686059f25d00ade19ec583e2c429879923183",
    "D_cap_source": "03396a361c3b5c65a6886379241cfb2d69aac4ce64de2779e91f465469844e86",
    "I": "5d46ee99df2613e9f30a43075af95a6402beffe26392cd9f017d8c1a154bf410",
    "I_closed": "c6045dbca6954a65c8c53be1393fc79eca7e6d84f7c9fcbc7f0306cf4b30fcec",
    "I_self": "af94d841f904d25e99713d88581a9bf6f0d227616ca3ebe93f4cda4f664ace27",
    "D_self": "5661751e02258fdfa5696afa2619b07e425a699f3790af3f5e9ef28a0595ce3d",
    "D_I_open": "ea5ef3a3879f4b9e78ae49f0c4dfe8c699e718825cba20b573f03bf03bba22f3",
    "exact_owner_loader": "f1d697b42fc9d97472fbd70cec2d8adfe178bb591d3513680609396ec4c204ad",
    "exact_predicate_source": "f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d",
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def loadmod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def edge_audit(faces):
    incidence = Counter()
    direction = Counter()
    for tri in np.asarray(faces, dtype=np.int64):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(a), int(b)
            incidence[tuple(sorted((a, b)))] += 1
            direction[(a, b)] += 1
    hist = Counter(incidence.values())
    same = sum(1 for (a, b), n in incidence.items()
               if n == 2 and (direction[(a, b)] != 1 or direction[(b, a)] != 1))
    return {
        "edge_incidence_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
        "boundary_edge_count": int(hist.get(1, 0)),
        "nonmanifold_edge_count": int(sum(v for k, v in hist.items() if k > 2)),
        "same_direction_shared_edge_count": int(same),
    }

def directed_boundary(faces):
    incidence = Counter()
    directed = Counter()
    for tri in np.asarray(faces, dtype=np.int64):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(a), int(b)
            incidence[tuple(sorted((a, b)))] += 1
            directed[(a, b)] += 1
    out = []
    for (a, b), n in incidence.items():
        if n == 1:
            out.append((a, b) if directed[(a, b)] == 1 else (b, a))
    return np.asarray(sorted(out), dtype=np.int64).reshape(-1, 2)

def records(pred, points, faces, face_start, vertex_offset=0):
    raw, _ = pred.float32_surface_records(np.asarray(points, dtype=np.float64), faces, source_seam_groups=())
    if len(raw) != len(faces):
        raise ValueError("exact owner changed face cardinality")
    return [
        (r[0], r[1], r[2], int(face_start)+j, tuple(vertex_offset+int(v) for v in r[4]))
        for j, r in enumerate(raw)
    ]

def components_and_orientation(points, faces):
    faces = np.asarray(faces, dtype=np.int64)
    parent = np.arange(len(faces), dtype=np.int64)
    size = np.ones(len(faces), dtype=np.int32)
    def find(x):
        x = int(x)
        while int(parent[x]) != x:
            parent[x] = parent[int(parent[x])]
            x = int(parent[x])
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
    edge_owner = {}
    for row, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = tuple(sorted((int(a), int(b))))
            previous = edge_owner.get(key)
            if previous is None:
                edge_owner[key] = row
            else:
                union(row, previous)
    grouped = {}
    for row, tri in enumerate(faces):
        root = find(row)
        item = grouped.setdefault(root, {"face_rows": [], "vertices": set()})
        item["face_rows"].append(row)
        item["vertices"].update(map(int, tri))
    result, face_groups = [], []
    for root, item in sorted(grouped.items()):
        face_rows = np.asarray(item["face_rows"], dtype=np.int64)
        vertex_rows = sorted(item["vertices"])
        reference = np.asarray(points[vertex_rows[0]], dtype=np.float64)
        tri_points = np.asarray(points[faces[face_rows]], dtype=np.float64) - reference
        terms = np.einsum("ij,ij->i", tri_points[:, 0], np.cross(tri_points[:, 1], tri_points[:, 2])) / 6.0
        signed_volume = math.fsum(map(float, terms))
        result.append({
            "root_face_row": int(root),
            "face_count": int(len(face_rows)),
            "vertex_count": int(len(vertex_rows)),
            "representative_vertex_row": int(vertex_rows[0]),
            "signed_volume_m3": signed_volume,
            "orientation_sign": "positive" if signed_volume > 0 else "negative" if signed_volume < 0 else "zero",
        })
        face_groups.append(face_rows)
    return result, face_groups

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("output already exists")
    started = time.monotonic()
    source_paths = {
        "D": D, "operations": OPERATIONS, "D_cap_source": D_CAP_SOURCE,
        "I": I, "I_closed": I_CLOSED, "I_self": I_SELF, "D_self": D_SELF,
        "D_I_open": D_I_OPEN, "exact_owner_loader": EXACT_OWNER,
    }
    pins = {key: sha(path) for key, path in source_paths.items()}
    for key, expected in EXPECTED.items():
        if key == "exact_predicate_source":
            continue
        if pins.get(key) != expected:
            raise SystemExit(f"pinned input changed: {key} {pins.get(key)} != {expected}")
    opsdoc = json.loads(OPERATIONS.read_text())
    if (opsdoc.get("candidate_sha256") != "9790ed1fa49bacff36ea169c7b8354344e40f1df33f604c006bcfb7991723a2a"
        or len(opsdoc.get("accepted_transactions", [])) != 15):
        raise SystemExit("wrong 15-operation source packet")
    dself = json.loads(D_SELF.read_text())
    iself = json.loads(I_SELF.read_text())
    diopen = json.loads(D_I_OPEN.read_text())
    if dself.get("status") != "PASS_SELF_GATE" or dself.get("candidate_sha256") != pins["D"] or dself.get("exact_triangle_pair_count") != 0:
        raise SystemExit("D full-self certificate is not a zero-pair pass for this exact candidate")
    if not iself.get("complete") or iself.get("candidate_sha256") != pins["I"] or iself.get("exact_triangle_pair_count") != 0:
        raise SystemExit("I full-self certificate is not a zero-pair pass for this exact asset")
    if diopen.get("status") != "PASS_SURFACE_PAIR_GATE" or diopen.get("D_sha256") != pins["D"] or diopen.get("I_sha256") != pins["I"] or diopen.get("exact_triangle_pair_count") != 0:
        raise SystemExit("open-D/open-I certificate is not a zero-pair pass for these exact assets")

    with np.load(D, allow_pickle=False) as z:
        dp = np.asarray(z["points_world_m"], dtype=np.float32)
        df = np.asarray(z["faces"], dtype=np.int64)
    with np.load(D_CAP_SOURCE, allow_pickle=False) as z:
        dcp = np.asarray(z["points_world_m_f64"], dtype=np.float64)
        dclosed = np.asarray(z["faces_f64"], dtype=np.int64)
        d_open_count = int(z["open_face_count"][0])
        dcap_count = int(z["cap_face_count"][0])
        dcap = np.asarray(z["cap_faces_from_run036_closed_parent_rows"], dtype=np.int64)
    with np.load(I, allow_pickle=False) as z:
        ip = np.asarray(z["points_world_m"], dtype=np.float32)
        iff = np.asarray(z["faces"], dtype=np.int64)
    with np.load(I_CLOSED, allow_pickle=False) as z:
        icp = np.asarray(z["points_world_m_f64"], dtype=np.float64)
        iclosed = np.asarray(z["faces_f64"], dtype=np.int64)
        i_open_count = int(z["open_face_count"][0])
        icap_count = int(z["cap_face_count"][0])
        icap = iclosed[i_open_count:i_open_count+icap_count]

    if d_open_count + dcap_count != len(dclosed) or dcap_count != len(dcap):
        raise SystemExit("D cap source face-row partition malformed")
    if not np.array_equal(dp, dcp.astype(np.float32)):
        raise SystemExit("final D rows differ from cap source point rows")
    if not np.array_equal(dcap, dclosed[d_open_count:]):
        raise SystemExit("D cap array does not match the pinned closed-source suffix")
    if not np.array_equal(directed_boundary(df), directed_boundary(dclosed[:d_open_count])):
        raise SystemExit("final D directed boundary differs from pinned cap source boundary")
    dclosed_faces = np.vstack((df, dcap)).astype(np.int64)
    dtopo = edge_audit(dclosed_faces)
    if dtopo["boundary_edge_count"] or dtopo["nonmanifold_edge_count"] or dtopo["same_direction_shared_edge_count"]:
        raise SystemExit("final D plus caps is not closed and consistently oriented")
    if not np.array_equal(ip, icp.astype(np.float32)) or not np.array_equal(iff, iclosed[:i_open_count]):
        raise SystemExit("closed I source does not preserve the exact open fine-I rows")
    if i_open_count != len(iff) or icap_count != 41:
        raise SystemExit("unexpected I open/cap face counts")
    itopo = edge_audit(iclosed)
    if itopo["boundary_edge_count"] or itopo["nonmanifold_edge_count"] or itopo["same_direction_shared_edge_count"]:
        raise SystemExit("I plus caps is not closed and consistently oriented")
    dcomponents, dcomponent_rows = components_and_orientation(dp, dclosed_faces)
    icomponents, icomponent_rows = components_and_orientation(ip, iclosed)
    dcap_vertices = set(map(int, dcap.reshape(-1)))
    icap_vertices = set(map(int, icap.reshape(-1)))
    def noncap_component_representatives(components, groups, open_faces, cap_vertices, owner):
        for index, (component, face_rows) in enumerate(zip(components, groups)):
            open_rows = face_rows[face_rows < len(open_faces)]
            candidates = set(map(int, open_faces[open_rows].reshape(-1))) - cap_vertices if len(open_rows) else set()
            if not candidates:
                component["representative_vertex_row"] = None
                component["representative_selection"] = "no_noncap_open_surface_vertex"
            else:
                component["representative_vertex_row"] = min(candidates)
                component["representative_selection"] = "smallest_vertex_on_open_surface_excluding_temporary_cap_vertices"
    noncap_component_representatives(dcomponents, dcomponent_rows, df, dcap_vertices, "D")
    noncap_component_representatives(icomponents, icomponent_rows, iff, icap_vertices, "I")
    if any(c["representative_vertex_row"] is None for c in dcomponents+icomponents):
        raise SystemExit("a closed component lacks a non-cap representative for nesting")
    if not dcomponents or not icomponents:
        raise SystemExit("closed surfaces contain no edge-connected components")

    out.mkdir(parents=True, exist_ok=False)
    owner_module = loadmod("closed_nesting_629_exact_owner_loader", EXACT_OWNER)
    pred = owner_module.load_exact_owner()
    if sha(pred.__file__) != EXPECTED["exact_predicate_source"]:
        raise SystemExit("loaded exact predicate source does not match pin")
    pins["exact_predicate_source"] = sha(pred.__file__)

    d_open = records(pred, dp, df, 0)
    d_caps = records(pred, dp, dcap, len(df))
    ivert_offset = len(dp) + 2_000_000
    i_open = records(pred, ip, iff, -1_000_000_000_000, ivert_offset)
    i_caps = records(pred, ip, icap, -1_000_000_000_000-len(iff), ivert_offset)
    i_closed_records = i_open + i_caps
    i_closed_index = pred._prepare_surface_aabb(i_closed_records)
    d_open_index = pred._prepare_surface_aabb(d_open)

    i_open_index = pred._prepare_surface_aabb(i_open)
    i_cap_index = pred._prepare_surface_aabb(i_caps)
    d_cap_index = pred._prepare_surface_aabb(d_caps)
    cap_raw = {
        "D_caps_vs_D_open": pred._audit_pair_prepared_first(d_open_index, d_caps, same_surface=True),
        "D_cap_self": pred._audit_record_pairs(itertools.combinations(d_caps, 2), same_surface=True),
        "I_caps_vs_I_open": pred._audit_pair_prepared_first(i_open_index, i_caps, same_surface=True),
        "I_cap_self": pred._audit_record_pairs(itertools.combinations(i_caps, 2), same_surface=True),
        "D_caps_vs_I_open": pred._audit_pair_prepared_first(i_open_index, d_caps, same_surface=False),
        "I_caps_vs_D_open": pred._audit_pair_prepared_first(d_open_index, i_caps, same_surface=False),
        "D_caps_vs_I_caps": pred._audit_pair_prepared_first(i_cap_index, d_caps, same_surface=False),
    }
    cap_checks = {}
    pair_arrays = {}
    capcap_pairs = np.asarray(cap_raw["D_caps_vs_I_caps"]["triangle_pairs"], dtype=np.int64).reshape(-1, 2)
    capcap_classifications = []
    def exact_coplanar_cap_pair(d_face_id, i_face_id):
        d_local = int(d_face_id) - len(df)
        i_local = int(i_face_id) - (-1_000_000_000_000-len(iff))
        if not (0 <= d_local < len(dcap) and 0 <= i_local < len(icap)):
            raise ValueError("cap-cap pair record ID does not map into pinned cap rows")
        dtri = dcap[d_local]
        itri = icap[i_local]
        dkeys = [pred.float32_point_lattice_key(dp[int(v)]) for v in dtri]
        ikeys = [pred.float32_point_lattice_key(ip[int(v)]) for v in itri]
        a,b,c = dkeys
        u=tuple(b[k]-a[k] for k in range(3)); v=tuple(c[k]-a[k] for k in range(3))
        normal=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])
        if normal == (0,0,0):
            return False
        return all(sum(normal[k]*(q[k]-a[k]) for k in range(3)) == 0 for q in ikeys)
    for name, raw in cap_raw.items():
        pairs = np.asarray(raw["triangle_pairs"], dtype=np.int64).reshape(-1, 2)
        pair_arrays[name] = pairs
        cap_checks[name] = {
            "exact_pair_count": int(raw["count"]),
            "allowed_shared_topology_pairs": int(raw.get("allowed_shared_vertex_or_edge_pairs", 0)),
            "aabb_candidate_pairs": int(raw.get("aabb_candidate_pairs", 0)),
        }
    for first_id, second_id in capcap_pairs:
        ids = (int(first_id), int(second_id))
        d_ids = [x for x in ids if x >= 0]
        i_ids = [x for x in ids if x < 0]
        if len(d_ids) != 1 or len(i_ids) != 1:
            raise SystemExit("D/I artificial-cap face namespace collision")
        coplanar = exact_coplanar_cap_pair(d_ids[0], i_ids[0])
        capcap_classifications.append({
            "D_cap_face_row": d_ids[0]-len(df),
            "I_cap_face_row": i_ids[0]-(-1_000_000_000_000-len(iff)),
            "classification": "pinned_temporary_coplanar_cut_interface" if coplanar else "unclassified_non_coplanar_cap_intersection",
            "anatomical_contact_waiver": False,
        })
    unclassified_capcap = sum(x["classification"] != "pinned_temporary_coplanar_cut_interface" for x in capcap_classifications)
    cap_checks["D_caps_vs_I_caps"]["temporary_coplanar_cut_interface_pairs"] = len(capcap_classifications)-unclassified_capcap
    cap_checks["D_caps_vs_I_caps"]["unclassified_pairs"] = int(unclassified_capcap)
    np.savez_compressed(out / "closure-cap-exact-pairs.npz", **pair_arrays)
    (out / "closure-cap-cap-classification.json").write_text(json.dumps({
        "schema":"temporary-closure-cap-cap-classification.v1",
        "D_cap_source_sha256":pins["D_cap_source"],
        "I_closed_source_sha256":pins["I_closed"],
        "pair_count":len(capcap_classifications),
        "pairs":capcap_classifications,
        "anatomical_contact_waiver":False,
    },indent=2,sort_keys=True)+"\\n")
    caps_clear = all(v["exact_pair_count"] == 0 for name,v in cap_checks.items() if name != "D_caps_vs_I_caps")
    caps_clear = caps_clear and unclassified_capcap == 0

    nesting = {
        "status": "NOT_RUN_CAP_PAIR_GATE_FAILED" if not caps_clear else "NOT_RUN",
        "all_D_shells_inside_I_filled_region": False,
        "all_component_orientation_nesting_valid": False,
        "all_I_void_shells_outside_D_filled_region": False,
        "queries": [], "D_component_relations": [], "I_component_relations": [],
    }
    if caps_clear:
        denom = 1 << 149
        def locate(points, prepared, vertex_row):
            lattice = pred.float32_point_lattice_key(points[int(vertex_row)])
            point = tuple(Fraction(int(x), denom) for x in lattice)
            return pred.point_location_prepared(point, prepared)
        prepared_i_all = pred.prepare_float32_point_location_meters(ip, iclosed)
        prepared_d_all = pred.prepare_float32_point_location_meters(dp, dclosed_faces)
        queries = []
        for component in dcomponents:
            row = component["representative_vertex_row"]
            queries.append({
                "query_kind": "D_component_in_I_filled_region",
                "component_root_face_row": component["root_face_row"],
                "representative_vertex_row": row,
                "orientation_sign": component["orientation_sign"],
                "location": locate(dp, prepared_i_all, row),
            })
        for component in icomponents:
            if component["orientation_sign"] == "negative":
                row = component["representative_vertex_row"]
                queries.append({
                    "query_kind": "I_void_component_in_D_filled_region",
                    "component_root_face_row": component["root_face_row"],
                    "representative_vertex_row": row,
                    "orientation_sign": component["orientation_sign"],
                    "location": locate(ip, prepared_d_all, row),
                })
        def parity_relations(points, faces, components, groups, owner):
            relations = []
            for index, component in enumerate(components):
                other_rows = np.concatenate([g for j,g in enumerate(groups) if j != index]) if len(groups) > 1 else np.empty(0,dtype=np.int64)
                if len(other_rows):
                    prepared_other = pred.prepare_float32_point_location_meters(points, faces[other_rows])
                    location = locate(points, prepared_other, component["representative_vertex_row"])
                else:
                    location = {"location":"outside","ray":None,"crossings":0,"attempts":0,"reason":"no_other_components"}
                expected_location = "outside" if component["orientation_sign"] == "positive" else "inside"
                relations.append({
                    "owner": owner, "component_root_face_row": component["root_face_row"],
                    "orientation_sign": component["orientation_sign"],
                    "expected_location_against_other_components": expected_location,
                    "exact_location": location,
                })
            return relations
        d_rel = parity_relations(dp, dclosed_faces, dcomponents, dcomponent_rows, "D")
        i_rel = parity_relations(ip, iclosed, icomponents, icomponent_rows, "I")
        d_queries = [q for q in queries if q["query_kind"] == "D_component_in_I_filled_region"]
        void_queries = [q for q in queries if q["query_kind"] == "I_void_component_in_D_filled_region"]
        all_d_inside = bool(d_queries) and all(q["location"].get("location") == "inside" for q in d_queries)
        all_i_voids_outside = all(q["location"].get("location") == "outside" for q in void_queries)
        orientation_nesting_valid = (
            all(c["orientation_sign"] != "zero" for c in dcomponents+icomponents)
            and any(c["orientation_sign"] == "positive" for c in dcomponents)
            and any(c["orientation_sign"] == "positive" for c in icomponents)
            and all(q["exact_location"].get("location") == q["expected_location_against_other_components"] for q in d_rel+i_rel)
        )
        passed = all_d_inside and all_i_voids_outside and orientation_nesting_valid
        nesting = {
            "status": "PASS_EXACT_COMPONENT_POINT_LOCATION" if passed else "FAIL_OR_INDETERMINATE_COMPONENT_LOCATION",
            "all_D_shells_inside_I_filled_region": all_d_inside,
            "all_component_orientation_nesting_valid": orientation_nesting_valid,
            "all_I_void_shells_outside_D_filled_region": all_i_voids_outside,
            "D_components": dcomponents, "I_components": icomponents,
            "queries": queries, "D_component_relations": d_rel, "I_component_relations": i_rel,
            "solid_interpretation": "positive shells bound filled regions; negative shells are cavities and must nest inside positive shells; exact representative queries against other components verify parity",
            "predicate": "existing exact rational ray-parity point_location_prepared",
        }

    result = {
        "schema": "closed-D-I-cap-intersections-and-exact-component-nesting.629.v1",
        "source_pins": pins,
        "candidate": {"path": str(D), "sha256": pins["D"], "open_faces": int(len(df)), "vertices": int(len(dp))},
        "prior_surface_certificates": {
            "D_full_self": {"path": str(D_SELF), "sha256": pins["D_self"], "status": dself["status"], "exact_pairs": int(dself["exact_triangle_pair_count"])},
            "I_full_self": {"path": str(I_SELF), "sha256": pins["I_self"], "exact_pairs": int(iself["exact_triangle_pair_count"])},
            "open_D_vs_open_I": {"path": str(D_I_OPEN), "sha256": pins["D_I_open"], "status": diopen["status"], "exact_pairs": int(diopen["exact_triangle_pair_count"])},
        },
        "closure_assets": {
            "D": {"path": str(D_CAP_SOURCE), "sha256": pins["D_cap_source"], "caps": int(len(dcap)), "edge_audit": dtopo},
            "I": {"path": str(I_CLOSED), "sha256": pins["I_closed"], "caps": int(icap_count), "edge_audit": itopo},
        },
        "component_topology_and_orientation": {"D": dcomponents, "I": icomponents},
        "closure_cap_exact_checks": cap_checks,
        "closure_pair_archive": str(out / "closure-cap-exact-pairs.npz"), "cap_cap_classification": str(out / "closure-cap-cap-classification.json"),
        "closed_nesting": nesting,
        "scope": "Closure-cap self/cross checks, separately classified pinned coplanar temporary D/I cut-cap interfaces, and exact component orientation/nesting point location; no bone rerun, Boolean volume, mass, or physics.",
        "physical_admission": False, "runtime": 0, "accepted_steps": 0,
        "status": ("PASS_CLOSED_CAPS_AND_COMPONENT_NESTING_ONLY"
                   if caps_clear and nesting["all_D_shells_inside_I_filled_region"] and nesting["all_component_orientation_nesting_valid"] and nesting["all_I_void_shells_outside_D_filled_region"]
                   else "CLOSED_CAP_INTERSECTION_OR_NESTING_REVIEW_REQUIRED"),
        "elapsed_s": time.monotonic()-started,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    (out / "progress.json").unlink(missing_ok=True)
    print(json.dumps({
        "status": result["status"], "D_component_count": len(dcomponents), "I_component_count": len(icomponents),
        "D_component_signs": [x["orientation_sign"] for x in dcomponents],
        "I_component_signs": [x["orientation_sign"] for x in icomponents],
        "cap_checks": cap_checks, "nesting_status": nesting["status"],
        "nesting_locations": [q.get("exact_location", q.get("location", {})).get("location") for q in nesting.get("queries", [])],
        "physical_admission": False, "runtime": 0, "accepted_steps": 0,
        "elapsed_s": result["elapsed_s"], "result": str(out/"result.json"),
    }, indent=2), flush=True)

if __name__ == "__main__":
    main()
