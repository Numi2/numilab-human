"""Versioned v22 builder: v21 source/identity gates plus optional role-balanced LP objective."""
"""Source-only INNER_BACK_LAYER_REPAIR builder with exact same-archive identity handling and row-scaled shared-feature SAT."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np
H=Path(__file__).resolve().parent
def imp(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
T=imp("coupled648T",H/"transfer_surface_contractions_cross_tissue_470_root.py");M=T.m
_Q_PATH=H/"629-quality-lp/quality_lp_629_v18_role_balanced_shared_feature.py"
_Q_EXPECTED_SHA="d88d9451670d7db685314c30354430f7123e176c8d3f2b540b018586e552b850"
if hashlib.sha256(_Q_PATH.read_bytes()).hexdigest()!=_Q_EXPECTED_SHA:raise RuntimeError("pinned v17b numeric-ULP shared-feature helper changed")
Q=imp("coupled721Qv16rowscaledshared",_Q_PATH)
B=imp("coupled721Base197",H/"fixed-boundary-quality-topology-196/run_fixed_boundary_dp_batch_197.py")
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def pin(p):return {"path":str(p),"sha256":sha(p)}
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def source_point_step_payload(parent_archive,candidate_points_f32,source_global_rows,parent_entry,trust_cap_m):
 """Apply shared global Float32 rows and rebase point-only source ancestry to its actual parent."""
 data=dict(parent_archive);old=np.asarray(data["points_world_m_f32"])
 candidate=np.asarray(candidate_points_f32);rows=np.asarray(source_global_rows,dtype=np.int64);faces=np.asarray(data["faces"],dtype=np.int64)
 cap=float(trust_cap_m)
 if old.dtype!=np.float32 or old.ndim!=2 or old.shape[1]!=3:raise ValueError("source points must be Float32 [N,3]")
 if candidate.dtype!=np.float32 or candidate.ndim!=2 or candidate.shape[1]!=3 or not np.isfinite(candidate).all():raise ValueError("candidate points must be finite Float32 [N,3]")
 if rows.shape!=(len(old),) or faces.ndim!=2 or faces.shape[1]!=3 or np.any(faces<0) or np.any(faces>=len(old)):raise ValueError("source point-step topology shape mismatch")
 if not np.isfinite(cap) or cap<=0 or np.any(rows<-1) or np.any(rows>=len(candidate)):raise ValueError("invalid source point-step map or trust cap")
 referenced=np.zeros(len(old),dtype=bool);referenced[np.unique(faces)]=True;hit=(rows>=0)&referenced
 new=old.copy();new[hit]=candidate[rows[hit]]
 if not np.isfinite(new).all():raise ValueError("nonfinite source candidate coordinate")
 displacement=np.abs(new.astype(np.float64)-old.astype(np.float64)).sum(axis=1)
 if np.any(displacement[hit]>cap):raise ValueError("source point step exceeds per-node L1 trust cap")
 changed=np.flatnonzero(np.any(new.view(np.uint32)!=old.view(np.uint32),axis=1))
 data["points_world_m_f32"]=new
 world=np.asarray(data["points_world_m"]).copy()
 if world.shape!=old.shape:raise ValueError("source Float64/Float32 point arrays disagree")
 world[changed]=new[changed].astype(world.dtype)
 if not np.isfinite(world).all():raise ValueError("nonfinite source Float64 coordinate")
 data["points_world_m"]=world
 data["typed_parent_candidate_path"]=np.asarray([str(parent_entry["path"])])
 data["typed_parent_candidate_sha256"]=np.asarray([str(parent_entry["sha256"])])
 data["source_input_face_rows"]=np.arange(len(faces),dtype=np.int64)
 data["source_input_vertex_quotient"]=np.arange(len(old),dtype=np.int64)
 for key in list(data):
  if key.startswith("source_") and key.endswith("_operation_rows"):
   dtype=np.asarray(data[key]).dtype
   if dtype.kind not in "iu":dtype=np.dtype(np.int64)
   data[key]=np.full(len(faces),-1,dtype=dtype)
 return data,changed

def _load_archive(path):
 with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}

def _source_witness_context(surfaces,smap):
 rolepoint={}
 for role,a in surfaces.items():
  pp=np.asarray(a["points_world_m_f32"],np.float32)
  for row,xyz in enumerate(pp):
   key=T.key(xyz)
   if key in rolepoint:raise ValueError("source witness point identity is ambiguous across roles")
   rolepoint[key]=(role,int(row))
 wall_faces={}
 for volume,marker,roles in (("Skin",3,("O","I")),("Fat",6,("I","D"))):
  path,expected=B.SHELL[volume]
  if sha(path)!=expected:raise ValueError(volume+" witness wall source pin mismatch")
  shell=_load_archive(path);sp=np.asarray(shell["points_world_m"],np.float32);sf=np.asarray(shell["faces"],np.int64);fm=np.asarray(shell["face_marker"],np.int64)
  rows=np.flatnonzero(fm==marker)
  if len(rows)!=90:raise ValueError(volume+" witness wall face count mismatch")
  mapped=[]
  for tri in sf[rows]:
   out=[]
   for sr in tri:
    hit=rolepoint.get(T.key(sp[int(sr)]))
    if hit is None or hit[0] not in roles:raise ValueError(volume+" wall vertex no longer maps to its source role")
    role,pr=hit;gr=int(smap[role][pr])
    if gr<0:raise ValueError("wall source row is outside the shared volume map")
    out.append(gr)
   mapped.append(out)
  wall_faces[volume]=np.asarray(mapped,np.int64)
 return rolepoint,wall_faces

def _oriented_face_key(tri):
    """Canonical cyclic rotation key; reversed winding remains a different face."""
    a, b, c = map(int, tri)
    if len({a, b, c}) != 3:
        raise ValueError("degenerate oriented face key")
    return min((a, b, c), (b, c, a), (c, a, b))


def _unoriented_face_key(tri):
    return tuple(sorted(map(int, tri)))


def _int64_array_sha256(values, domain):
    arr = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
    h = hashlib.sha256()
    h.update(("h706." + domain + ".i64le.v1\0").encode("ascii"))
    h.update(np.asarray(arr.shape, dtype="<i8").tobytes())
    h.update(arr.tobytes())
    return h.hexdigest()

def _face_array_sha256(faces):
    return _int64_array_sha256(faces, "face-array")


def _typed_archive_scalar_text(value, field_name, archive_path):
    """Read a pinned typed-parent string stored as 0-D or one-item 1-D text."""
    arr = np.asarray(value)
    if arr.dtype.kind not in ("U", "S"):
        raise ValueError("typed lineage field must use a text dtype: " + field_name + " in " + str(archive_path))
    if not (arr.ndim == 0 or (arr.ndim == 1 and arr.shape == (1,))):
        raise ValueError("typed lineage field must contain exactly one scalar text value: " + field_name + " in " + str(archive_path))
    raw = arr.reshape(-1)[0]
    if isinstance(raw, (bytes, np.bytes_)):
        text = bytes(raw).decode("utf-8", "strict")
    else:
        text = str(raw)
    if not text:
        raise ValueError("typed lineage scalar text must not be empty: " + field_name + " in " + str(archive_path))
    return text


def _build_current_face_lookup(faces, point_count, role):
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3 or (f.size and (f.min() < 0 or f.max() >= point_count)):
        raise ValueError(role + ": malformed current face array")
    oriented = {}
    unoriented = {}
    for row, tri in enumerate(f):
        key = _oriented_face_key(tri)
        ukey = _unoriented_face_key(tri)
        if key in oriented:
            raise ValueError(role + ": ambiguous duplicate current oriented face " + str(key))
        if ukey in unoriented:
            raise ValueError(role + ": duplicate current face support with conflicting or duplicate orientation " + str(ukey))
        oriented[key] = int(row)
        unoriented[ukey] = int(row)
    return oriented, unoriented


def _build_report_face_namespace(report_faces, changed_rows, face_base, wall_base, expected_total):
    role_order = ("D", "O", "I")
    ranges = {}
    for role in role_order:
        lo = int(face_base[role])
        hi = lo + len(report_faces[role])
        ranges[role] = {"base": lo, "end_exclusive": hi, "face_count": len(report_faces[role])}
    if ranges["D"]["end_exclusive"] > face_base["O"] or ranges["O"]["end_exclusive"] > face_base["I"] or ranges["I"]["end_exclusive"] > wall_base["Skin"]:
        raise ValueError("630 frozen source face namespaces overlap")
    if wall_base["Skin"] + 90 > wall_base["Fat"]:
        raise ValueError("630 frozen wall face namespaces overlap")
    max_context = max(max(ranges[r]["end_exclusive"] - 1 for r in role_order),
                      wall_base["Skin"] + 89, wall_base["Fat"] + 89)
    changed_start = max_context + 1_000_000
    idmap = {}
    next_id = changed_start
    for role in role_order:
        rows = np.asarray(changed_rows[role], dtype=np.int64)
        if len(np.unique(rows)) != len(rows):
            raise ValueError("630 changed candidate face rows are duplicated")
        for row in rows:
            idmap[next_id] = ("source_old", role, int(row))
            next_id += 1
    if int(expected_total) != sum(len(changed_rows[r]) for r in role_order) or next_id - changed_start != int(expected_total):
        raise ValueError("630 changed-face ID namespace count mismatch")
    return ranges, idmap, changed_start, max_context


def _resolve_report_face_id(fid, role_ranges, face_base, wall_base, idmap, changed_start, expected_total):
    fid = int(fid)
    if fid in idmap:
        return idmap[fid]
    if fid < 0:
        return ("bone", -1 - fid)
    for role in ("D", "O", "I"):
        lo = int(face_base[role])
        hi = int(role_ranges[role]["end_exclusive"])
        if lo <= fid < hi:
            return ("source_old", role, fid - lo)
    for volume in ("Skin", "Fat"):
        lo = int(wall_base[volume])
        if lo <= fid < lo + 90:
            return ("wall", volume, fid - lo)
    if changed_start <= fid < changed_start + int(expected_total):
        raise ValueError("630 changed-face ID is missing from changed-row decode map")
    raise ValueError("630 witness face ID outside the pinned source/wall/all185 namespaces: " + str(fid))


def _compose_point_map_to_ancestor(child_path, child_sha, ancestor_path, ancestor_sha, sha_cache):
    """Authenticate output->parent vertex maps until the exact report baseline archive."""
    cur = Path(child_path).resolve()
    target = Path(ancestor_path).resolve()
    if sha_cache.get(cur) is None:
        sha_cache[cur] = sha(cur)
    if sha_cache[cur] != child_sha:
        raise ValueError("current source archive pin mismatch while tracing point lineage: " + str(cur))
    with np.load(cur, allow_pickle=False) as z:
        if "points_world_m_f32" not in z:
            raise ValueError("lineage archive missing Float32 points: " + str(cur))
        n = len(z["points_world_m_f32"])
    original_n = n
    point_map = np.arange(n, dtype=np.int64)
    depth = 0
    seen = set()
    while cur != target:
        if cur in seen:
            raise ValueError("cycle in typed point-parent chain at " + str(cur))
        seen.add(cur)
        with np.load(cur, allow_pickle=False) as z:
            required = ("typed_parent_candidate_path", "typed_parent_candidate_sha256", "source_input_vertex_quotient")
            if any(k not in z for k in required):
                raise ValueError("lineage step lacks explicit parent/point map: " + str(cur))
            parent = Path(_typed_archive_scalar_text(z["typed_parent_candidate_path"], "typed_parent_candidate_path", cur)).resolve()
            expected_parent_sha = _typed_archive_scalar_text(z["typed_parent_candidate_sha256"], "typed_parent_candidate_sha256", cur)
            child_q = np.asarray(z["source_input_vertex_quotient"], dtype=np.int64)
            if child_q.shape != (n,) or np.any(child_q < -1):
                raise ValueError("lineage point map malformed: " + str(cur))
        if parent == cur:
            raise ValueError("self-parent before report baseline: " + str(cur))
        if not parent.is_file():
            raise FileNotFoundError("typed point-parent archive missing: " + str(parent))
        actual_parent_sha = sha_cache.get(parent)
        if actual_parent_sha is None:
            actual_parent_sha = sha(parent)
            sha_cache[parent] = actual_parent_sha
        if actual_parent_sha != expected_parent_sha:
            raise ValueError("typed point-parent hash mismatch: " + str(parent))
        with np.load(parent, allow_pickle=False) as z:
            if "points_world_m_f32" not in z:
                raise ValueError("typed parent missing Float32 point domain: " + str(parent))
            parent_n = len(z["points_world_m_f32"])
        valid_q = child_q >= 0
        if np.any(child_q[valid_q] >= parent_n):
            raise ValueError("typed point map references parent row outside domain: " + str(cur))
        valid_map = point_map >= 0
        composed = np.full_like(point_map, -1)
        composed[valid_map] = child_q[point_map[valid_map]]
        point_map = composed
        cur = parent
        n = parent_n
        depth += 1
        if depth > 128:
            raise ValueError("typed point-parent chain exceeded bounded depth")
    if sha_cache.get(target) != ancestor_sha:
        raise ValueError("report baseline archive hash mismatch at point-lineage target")
    with np.load(target, allow_pickle=False) as z:
        target_n = len(z["points_world_m_f32"])
    if point_map.shape != (original_n,):
        raise ValueError("composed point map lost current output domain")
    if np.any(point_map >= target_n):
        raise ValueError("composed point map references report baseline outside domain")
    return point_map, depth


def _report_face_row_to_current(role, old_row, report_faces, old_to_current,
                                current_oriented, current_unoriented, map_cache):
    key0 = (role, int(old_row))
    if key0 in map_cache:
        return map_cache[key0]
    f = np.asarray(report_faces, dtype=np.int64)
    row = int(old_row)
    if row < 0 or row >= len(f):
        raise ValueError(role + ": report face row outside authenticated report archive")
    old_tri = tuple(map(int, f[row]))
    mapped = []
    missing_point_rows = []
    for old_point_row in old_tri:
        current_row = old_to_current.get(int(old_point_row))
        if current_row is None:
            missing_point_rows.append(int(old_point_row))
        else:
            mapped.append(int(current_row))
    if missing_point_rows:
        result = {
            "current_row": None,
            "reason": "report_triangle_vertex_has_no_active_current_point_row",
            "old_oriented_triangle": list(old_tri),
            "missing_report_point_rows": missing_point_rows,
        }
        map_cache[key0] = result
        return result
    if len(set(mapped)) != 3:
        raise ValueError(role + ": report face collapses under authenticated point-row map")
    okey = _oriented_face_key(mapped)
    current_row = current_oriented.get(okey)
    if current_row is not None:
        result = {
            "current_row": int(current_row),
            "reason": None,
            "old_oriented_triangle": list(old_tri),
            "mapped_current_oriented_triangle": list(mapped),
        }
        map_cache[key0] = result
        return result
    ukey = _unoriented_face_key(mapped)
    if ukey in current_unoriented:
        raise ValueError(role + ": report triangle exists in current source with reversed orientation; refusing to drop it")
    result = {
        "current_row": None,
        "reason": "exact_oriented_triangle_absent_from_current_source",
        "old_oriented_triangle": list(old_tri),
        "mapped_current_oriented_triangle": list(mapped),
    }
    map_cache[key0] = result
    return result


def _witness_separation_rows(report_paths, state, surfaces, smap, points, output_dir=None, input_state_pin=None, skip_fixed_o_pairs=False):
    paths = [Path(x).resolve() for x in report_paths]
    if not paths:
        empty = {"reports": [], "exact_pairs": 0, "constraints": 0, "frozen_source_nodes": 0,
                 "deduplicated_pairs": 0, "stale_face_pairs_dropped": 0}
        return None, empty

    role_order = ("D", "O", "I")
    face_base = {r: int(B.FACE_BASE[r]) for r in role_order}
    if face_base != {"D": 0, "O": 300000, "I": 600000}:
        raise ValueError("frozen 630 source face namespace changed")
    wall_base = {"Skin": 900000, "Fat": 901000}
    current_faces = {r: np.asarray(surfaces[r]["faces"], dtype=np.int64) for r in role_order}
    current_face_lookup = {}
    current_face_sha = {}
    for role in role_order:
        arr = current_faces[role]
        current_face_lookup[role] = _build_current_face_lookup(arr, len(surfaces[role]["points_world_m_f32"]), role)
        current_face_sha[role] = _face_array_sha256(arr)

    rolepoint, wall_global = _source_witness_context(surfaces, smap)
    all_pairs = []
    seen_pairs = {}
    report_receipts = []
    dropped_pairs = []
    pair_origins = {}
    sha_cache = {}
    archive_cache = {}
    lineage_cache = {}
    archive_bytes_equal_cache = {}
    total_raw_pairs = 0
    total_stale_pairs = 0
    total_duplicate_pairs = 0
    total_skipped_fixed_pairs = 0
    all_skipped_fixed_pair_details = []
    bone_requests = set()
    bone_report_signature = None
    bone_report = None

    def checked_archive(path, expected_sha):
        p = Path(path).resolve()
        actual = sha_cache.get(p)
        if actual is None:
            actual = sha(p)
            sha_cache[p] = actual
        if actual != expected_sha:
            raise ValueError("witness report source archive pin mismatch: " + str(p))
        if p not in archive_cache:
            with np.load(p, allow_pickle=False) as z:
                identity_keys = ("points_world_m_f32", "points_world_m", "faces")
                if any(k not in z for k in identity_keys):
                    raise ValueError("witness source archive missing F32/full-precision points or faces: " + str(p))
                identity_arrays = {}
                for k in identity_keys:
                    arr = np.ascontiguousarray(z[k])
                    identity_arrays[k] = {
                        "dtype": arr.dtype.str,
                        "shape": list(arr.shape),
                        "sha256": hashlib.sha256(arr.tobytes(order="C")).hexdigest(),
                    }
                pdata = {
                    "point_count": int(len(z["points_world_m_f32"])),
                    "faces": np.array(z["faces"], dtype=np.int64, copy=True),
                    "identity_arrays": identity_arrays,
                }
                for k in ("typed_parent_candidate_path", "typed_parent_candidate_sha256", "source_input_vertex_quotient"):
                    if k in z:
                        pdata[k] = np.array(z[k], copy=True) if k.endswith("quotient") else _typed_archive_scalar_text(z[k], k, p)
            archive_cache[p] = pdata
        return p, archive_cache[p]

    def archive_bytes_equal(path_a, path_b):
        a, b = Path(path_a).resolve(), Path(path_b).resolve()
        if a == b:
            return True
        key = tuple(sorted((str(a), str(b))))
        if key in archive_bytes_equal_cache:
            return archive_bytes_equal_cache[key]
        same = a.stat().st_size == b.stat().st_size
        if same:
            with a.open("rb") as fa, b.open("rb") as fb:
                while True:
                    ba, bb = fa.read(1024 * 1024), fb.read(1024 * 1024)
                    if ba != bb:
                        same = False
                        break
                    if not ba:
                        break
        archive_bytes_equal_cache[key] = bool(same)
        return bool(same)

    def current_to_report_baseline(role, base_path, base_sha):
        entry = state["source_surfaces"][role]
        key = (role, str(Path(entry["path"]).resolve()), str(Path(base_path).resolve()), base_sha)
        if key not in lineage_cache:
            lineage_cache[key] = _compose_point_map_to_ancestor(
                entry["path"], entry["sha256"], base_path, base_sha, sha_cache)
        return lineage_cache[key]

    for report_path in paths:
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        report_sha = sha(report_path)
        report = json.loads(report_path.read_text())
        if report.get("schema") != "h706.combined-current-exact-surface-diff-audit.v1":
            raise ValueError("unrecognized 630 witness report schema: " + str(report_path))
        if set(report.get("roles", {})) != set(role_order):
            raise ValueError("630 witness report must include exactly D/O/I source roles")

        report_faces = {}
        report_point_maps = {}
        report_role_info = {}
        changed_rows = {}
        for role in role_order:
            rr = report["roles"][role]
            bp, bdata = checked_archive(rr["baseline"]["path"], rr["baseline"]["sha256"])
            cp, cdata = checked_archive(rr["candidate"]["path"], rr["candidate"]["sha256"])
            bf = bdata["faces"]
            cf = cdata["faces"]
            if bf.ndim != 2 or bf.shape[1] != 3 or not np.array_equal(bf, cf):
                raise ValueError(role + ": frozen witness must be a point-only report with identical baseline/candidate face rows")
            if bdata["point_count"] != cdata["point_count"]:
                raise ValueError(role + ": witness candidate changed the point-row domain")
            if rr.get("baseline_face_count") != len(bf) or rr.get("candidate_face_count") != len(cf):
                raise ValueError(role + ": report face-count receipt disagrees with pinned source arrays")
            baseline_sha = str(rr["baseline"]["sha256"])
            candidate_sha = str(rr["candidate"]["sha256"])
            same_path_and_pin = (cp == bp and candidate_sha == baseline_sha)
            same_full_archive_bytes = same_path_and_pin or (
                candidate_sha == baseline_sha and archive_bytes_equal(bp, cp))
            identity_basis = None
            if same_full_archive_bytes:
                if bdata["identity_arrays"] != cdata["identity_arrays"]:
                    raise ValueError(role + ": same-pinned witness source arrays differ in F32/full-precision points or faces")
                identity_basis = ("same_resolved_path_and_sha256" if same_path_and_pin
                                  else "byte_identical_full_archive_sha256_and_content")
            else:
                if Path(cdata.get("typed_parent_candidate_path", "")).resolve() != bp:
                    raise ValueError(role + ": changed witness point candidate is not directly parented to its pinned baseline")
                if cdata.get("typed_parent_candidate_sha256") != baseline_sha:
                    raise ValueError(role + ": changed witness candidate parent hash is not the pinned baseline")
                cand_q = np.asarray(cdata.get("source_input_vertex_quotient", []), dtype=np.int64)
                if cand_q.shape != (cdata["point_count"],) or not np.array_equal(cand_q, np.arange(cdata["point_count"], dtype=np.int64)):
                    raise ValueError(role + ": changed point-only witness candidate lacks identity input-vertex map")
            if bf.size and (bf.min() < 0 or bf.max() >= bdata["point_count"]):
                raise ValueError(role + ": report baseline face vertex outside pinned point domain")
            rows = np.asarray(rr["changed_candidate_face_rows"], dtype=np.int64)
            if rows.ndim != 1 or (len(rows) and (rows.min() < 0 or rows.max() >= len(bf))) or len(np.unique(rows)) != len(rows):
                raise ValueError(role + ": witness changed face rows invalid or duplicated")
            if same_full_archive_bytes and len(rows):
                raise ValueError(role + ": byte-identical baseline/candidate report declares changed faces")
            report_faces[role] = bf
            changed_rows[role] = rows

            cur_faces = current_faces[role]
            report_point_maps[role] = None
            report_role_info[role] = {
                "baseline": {"path": str(bp), "sha256": rr["baseline"]["sha256"], "face_count": int(len(bf))},
                "candidate": {"path": str(cp), "sha256": rr["candidate"]["sha256"], "face_count": int(len(cf))},
                "candidate_baseline_identity_archive": bool(same_full_archive_bytes),
                "candidate_baseline_identity_basis": identity_basis,
                "candidate_baseline_array_fingerprints": bdata["identity_arrays"] if same_full_archive_bytes else None,
                "current_face_count": int(len(cur_faces)),
                "current_face_array_sha256": current_face_sha[role],
                "current_source_archive": state["source_surfaces"][role],
                "oriented_face_key_semantics": "cyclic rotations equivalent; reverse winding distinct",
            }

        wall_receipts = report.get("wall_receipts", [])
        wall_by_role = {x.get("role"): x for x in wall_receipts}
        if set(wall_by_role) != {"Skin", "Fat"}:
            raise ValueError("630 witness report wall receipt roles are incomplete")
        for volume in ("Skin", "Fat"):
            exp_path, exp_sha = B.SHELL[volume]
            item = wall_by_role[volume]
            if (Path(item.get("path", "")).resolve() != Path(exp_path).resolve()
                    or item.get("sha256") != exp_sha or int(item.get("face_count", -1)) != 90):
                raise ValueError(volume + ": witness report wall namespace/pin mismatch")

        expected_total = int(report.get("changed_face_total", -1))
        current_namespace, idmap, changed_start, max_context = _build_report_face_namespace(
            report_faces, changed_rows, face_base, wall_base, expected_total)

        bone = report.get("all185_receipt")
        if not isinstance(bone, dict):
            raise ValueError("all185 report receipt missing")
        range_digest = hashlib.sha256(json.dumps(bone.get("ranges", []), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        signature = (bone.get("native_sha256"), bone.get("pinned_manifest_sha256"), range_digest)
        if signature[:2] != (B.BONE_NATIVE, B.BONE_MANIFEST):
            raise ValueError("all185 witness source pin mismatch")
        if bone_report_signature is not None and signature != bone_report_signature:
            raise ValueError("witness reports use inconsistent all185 identity namespaces or range tables")
        bone_report_signature = signature
        bone_report = bone

        # Decode exact report pairs before tracing role ancestry. During the
        # inner-back fit, O has no free vertices. O/O and O/bone witnesses
        # cannot constrain a variable, so retain their pins and role labels
        # while skipping their stale lineage and ineffective halfspaces.
        section_pairs = {}
        decoded_pairs = {}
        skipped_fixed_pair_indices = set()
        skipped_fixed_pair_details = []
        roles_requiring_point_map = set()
        raw_in_report = 0
        for section in ("changed_vs_complete_retained_OID_both_walls_all185", "changed_OID_joint_self_pairs"):
            info = report.get(section, {})
            pairs = np.asarray(info.get("triangle_pairs", []), dtype=np.int64).reshape((-1, 2))
            if len(pairs) != int(info.get("count", -1)):
                raise ValueError("630 exact witness pair list is incomplete")
            section_pairs[section] = pairs
            raw_in_report += len(pairs)
            for pair_index, (a, b) in enumerate(pairs):
                endpoints = tuple(_resolve_report_face_id(
                    x, current_namespace, face_base, wall_base, idmap, changed_start, expected_total
                ) for x in (a, b))
                decoded_pairs[(section, pair_index)] = endpoints
                is_o_a = endpoints[0][0] == "source_old" and endpoints[0][1] == "O"
                is_o_b = endpoints[1][0] == "source_old" and endpoints[1][1] == "O"
                is_bone_a = endpoints[0][0] == "bone"
                is_bone_b = endpoints[1][0] == "bone"
                inactive_o_fixed_pair = bool(skip_fixed_o_pairs and (
                    (is_o_a and (is_o_b or is_bone_b)) or (is_o_b and is_bone_a)
                ))
                if inactive_o_fixed_pair:
                    skipped_fixed_pair_indices.add((section, pair_index))
                    skipped_fixed_pair_details.append({
                        "report_path": str(report_path), "report_sha256": report_sha,
                        "section": section, "pair_index": int(pair_index),
                        "raw_face_ids": [int(a), int(b)],
                        "decoded_identities": [list(endpoints[0]), list(endpoints[1])],
                        "reason": "both endpoints are fixed O/O or fixed O/all185 bone under INNER_BACK_LAYER_REPAIR",
                    })
                    continue
                for endpoint in endpoints:
                    if endpoint[0] == "source_old":
                        roles_requiring_point_map.add(endpoint[1])

        total_skipped_fixed_pairs += len(skipped_fixed_pair_indices)
        all_skipped_fixed_pair_details.extend(skipped_fixed_pair_details)
        skipped_roles = {
            role for item in skipped_fixed_pair_details for ident in item["decoded_identities"]
            for role in ([ident[1]] if ident[0] == "source_old" else [])
        }
        for role in role_order:
            if role in roles_requiring_point_map:
                point_map, depth = current_to_report_baseline(role, report_role_info[role]["baseline"]["path"],
                                                               report_role_info[role]["baseline"]["sha256"])
                cur_faces = current_faces[role]
                active_current = np.unique(cur_faces.reshape(-1))
                mapped_active = point_map[active_current]
                valid_active = mapped_active >= 0
                valid_old_rows = mapped_active[valid_active]
                if len(np.unique(valid_old_rows)) != len(valid_old_rows):
                    raise ValueError(role + ": active current vertices map ambiguously into a witness baseline point domain")
                report_point_maps[role] = {
                    int(old): int(cur) for cur, old in zip(active_current[valid_active], valid_old_rows)
                }
                report_role_info[role].update({
                    "current_to_report_baseline_point_map_sha256": _int64_array_sha256(point_map, "point-map"),
                    "typed_point_lineage_depth": int(depth),
                    "current_active_point_rows": int(len(active_current)),
                    "active_current_rows_mapped_to_report": int(np.count_nonzero(valid_active)),
                    "active_current_rows_unmapped": int(np.count_nonzero(~valid_active)),
                    "active_point_map_injective": True,
                    "point_lineage_status": "AUTHENTICATED_FOR_ACTIVE_WITNESS_ENDPOINTS",
                })
            else:
                report_point_maps[role] = {}
                report_role_info[role].update({
                    "current_to_report_baseline_point_map_sha256": None,
                    "typed_point_lineage_depth": None,
                    "current_active_point_rows": int(len(np.unique(current_faces[role].reshape(-1)))),
                    "active_current_rows_mapped_to_report": None,
                    "active_current_rows_unmapped": None,
                    "active_point_map_injective": None,
                    "point_lineage_status": ("SKIPPED_FIXED_FIXED_O_WITNESSES_ONLY" if role in skipped_roles
                                             else "NOT_REQUIRED_NO_ACTIVE_WITNESS_ENDPOINTS"),
                })

        map_cache = {}
        stale_in_report = 0
        duplicate_in_report = 0
        skipped_in_report = 0
        pairs_with_origins = []
        referenced_report_rows = {r: set() for r in role_order}
        mapped_report_rows = {r: set() for r in role_order}
        missing_report_rows = {r: set() for r in role_order}
        for section in ("changed_vs_complete_retained_OID_both_walls_all185", "changed_OID_joint_self_pairs"):
            pairs = section_pairs[section]
            for pair_index, (a, b) in enumerate(pairs):
                raw_ids = [int(a), int(b)]
                endpoints = list(decoded_pairs[(section, pair_index)])
                if (section, pair_index) in skipped_fixed_pair_indices:
                    skipped_in_report += 1
                    for ident in endpoints:
                        if ident[0] == "source_old":
                            referenced_report_rows[ident[1]].add(int(ident[2]))
                    continue
                mapped_endpoints = []
                missing = []
                endpoint_notes = []
                for endpoint_index, ident in enumerate(endpoints):
                    if ident[0] != "source_old":
                        mapped_endpoints.append(ident)
                        endpoint_notes.append({"report_identity": list(ident), "current_identity": list(ident)})
                        continue
                    _, role, old_row = ident
                    referenced_report_rows[role].add(int(old_row))
                    mapped = _report_face_row_to_current(
                        role, old_row, report_faces[role], report_point_maps[role],
                        current_face_lookup[role][0], current_face_lookup[role][1], map_cache)
                    if mapped["current_row"] is None:
                        missing_report_rows[role].add(int(old_row))
                        missing.append({"endpoint_index": endpoint_index, "role": role, "report_face_row": old_row, **mapped})
                        endpoint_notes.append({"report_identity": ["source", role, old_row], "current_identity": None, "missing": mapped})
                    else:
                        mapped_report_rows[role].add(int(old_row))
                        current_ident = ("source", role, int(mapped["current_row"]))
                        mapped_endpoints.append(current_ident)
                        endpoint_notes.append({"report_identity": ["source", role, old_row], "current_identity": list(current_ident), "mapped_current_oriented_triangle": mapped["mapped_current_oriented_triangle"]})
                origin = {"report_path": str(report_path), "report_sha256": report_sha,
                          "section": section, "pair_index": int(pair_index), "raw_face_ids": raw_ids,
                          "endpoints": endpoint_notes}
                if missing:
                    stale_in_report += 1
                    dropped_pairs.append({**origin, "status": "DROPPED_STALE_PAIR_SOURCE_TRIANGLE_ABSENT", "missing_endpoints": missing})
                    continue
                if len(mapped_endpoints) != 2:
                    raise ValueError("internal pair endpoint mapping accounting mismatch")
                if mapped_endpoints[0] == mapped_endpoints[1]:
                    raise ValueError("distinct report witness endpoints coalesced to the same current face; refusing silent drop")
                for endpoint in mapped_endpoints:
                    if endpoint[0] == "bone":
                        bone_requests.add(int(endpoint[1]))
                canonical_pair = tuple(sorted(mapped_endpoints))
                if canonical_pair in seen_pairs:
                    total_duplicate_pairs += 1
                    duplicate_in_report += 1
                    pair_origins[canonical_pair].append(origin)
                    continue
                seen_pairs[canonical_pair] = len(all_pairs)
                pair_origins[canonical_pair] = [origin]
                all_pairs.append((mapped_endpoints[0], mapped_endpoints[1]))
                pairs_with_origins.append(origin)
        total_raw_pairs += raw_in_report
        total_stale_pairs += stale_in_report
        for role in role_order:
            report_role_info[role]["referenced_report_face_rows"] = int(len(referenced_report_rows[role]))
            report_role_info[role]["mapped_report_face_rows"] = int(len(mapped_report_rows[role]))
            report_role_info[role]["missing_report_face_rows"] = int(len(missing_report_rows[role]))
        report_receipts.append({
            "path": str(report_path), "sha256": report_sha,
            "context_pairs": int(report["changed_vs_complete_retained_OID_both_walls_all185"]["count"]),
            "joint_self_pairs": int(report["changed_OID_joint_self_pairs"]["count"]),
            "raw_pair_count": raw_in_report, "retained_unique_pairs_from_this_report": len(pairs_with_origins),
            "stale_source_triangle_pairs_dropped": stale_in_report, "duplicate_mapped_pairs_deduplicated": duplicate_in_report,
            "inactive_fixed_fixed_pairs_skipped": skipped_in_report,
            "active_roles_requiring_point_map": sorted(roles_requiring_point_map),
            "changed_face_total": expected_total, "source_roles": report_role_info,
            "frozen_namespace": {"face_base": face_base, "role_ranges": current_namespace,
                                  "wall_base": wall_base, "wall_face_count": 90,
                                  "max_context_id": max_context, "changed_start_id": changed_start,
                                  "changed_end_exclusive": changed_start + expected_total},
            "wall_receipts": wall_receipts,
        })

    bone_triangles = {}
    if bone_requests:
        native_path = B.ROOT / "geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz"
        manifest_path = B.ROOT / "geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/all185-bank-operand-manifest.json"
        if sha(native_path) != B.BONE_NATIVE or sha(manifest_path) != B.BONE_MANIFEST:
            raise ValueError("all185 witness source pin mismatch")
        if bone_report is None or bone_report.get("native_sha256") != B.BONE_NATIVE or bone_report.get("pinned_manifest_sha256") != B.BONE_MANIFEST:
            raise ValueError("all185 report/namespace pin mismatch")
        ranges = bone_report.get("ranges", [])
        if not ranges:
            raise ValueError("all185 witness range receipt missing")
        with np.load(native_path, allow_pickle=False) as z:
            native_points = np.array(z["points_world_f32"], copy=True)
            native_faces = np.array(z["faces"], copy=True)
        manifest = json.loads(manifest_path.read_text())
        operand = {(int(x["body"]), int(x["bone"])): x for x in manifest["operands"]}
        bank_cache = {}
        for idx in sorted(bone_requests):
            hits = [r for r in ranges if int(r["negative_face_index_start"]) <= idx < int(r["negative_face_index_end"])]
            if len(hits) != 1:
                raise ValueError("all185 witness index has no unique exact owner range: " + str(idx))
            rg = hits[0]
            local = idx - int(rg["negative_face_index_start"])
            domain = rg["domain"]
            if domain == "source_order_native_f32_replay":
                row = idx
                if row < 0 or row >= len(native_faces):
                    raise ValueError("native all185 face index outside archive")
                tri = native_points[native_faces[row]].astype(np.float64)
            elif domain == "pinned_f64_exact":
                key = (int(rg["body"]), int(rg["bone"]))
                if key not in operand:
                    raise ValueError("all185 witness body/bone absent from manifest")
                entry = operand[key]
                op = Path(entry["operand_path"])
                if sha(op) != entry["operand_sha256"]:
                    raise ValueError("all185 witness Float64 operand pin mismatch")
                if key not in bank_cache:
                    bank = _load_archive(op)
                    bank_cache[key] = (np.asarray(bank["points_world_m_f64"], np.float64), np.asarray(bank["faces_f64"], np.int64))
                bp, bf = bank_cache[key]
                if local < 0 or local >= len(bf):
                    raise ValueError("Float64 all185 local face index outside operand")
                tri = bp[bf[local]]
            else:
                raise ValueError("unknown all185 witness coordinate domain")
            if tri.shape != (3, 3) or not np.isfinite(tri).all():
                raise ValueError("all185 witness triangle is malformed")
            bone_triangles[idx] = tri

    src_a, src_b, source_pair_identities = [], [], []
    fixed_a, fixed_xyz, fixed_pair_identities = [], [], []
    source_pair_keys = 0
    fixed_pair_keys = 0

    def global_source_face(ident):
        _, role, row = ident
        gr = np.asarray(smap[role][current_faces[role][row]], dtype=np.int64)
        if gr.shape != (3,) or np.any(gr < 0) or np.any(gr >= len(points)):
            raise ValueError("current witness triangle has unmapped volume vertices: " + str(ident))
        return gr

    for ia, ib in all_pairs:
        if ia[0] == "bone":
            ia, ib = ib, ia
        fa = global_source_face(ia) if ia[0] == "source" else wall_global[ia[1]][int(ia[2])]
        if ib[0] == "bone":
            fixed_a.append(fa)
            fixed_xyz.append(bone_triangles[int(ib[1])])
            fixed_pair_identities.append((ia, ib))
            fixed_pair_keys += 1
        else:
            fb = global_source_face(ib) if ib[0] == "source" else wall_global[ib[1]][int(ib[2])]
            if np.any(fb < 0) or np.any(fb >= len(points)):
                raise ValueError("wall witness triangle has unmapped global vertices")
            src_a.append(fa)
            src_b.append(fb)
            source_pair_identities.append((ia, ib))
            source_pair_keys += 1

    sep_results = []
    if src_a:
        aa = np.asarray(src_a, np.int64)
        bb = np.asarray(src_b, np.int64)
        pr = np.column_stack((np.arange(len(aa)), np.arange(len(aa))))
        sep_results.append(Q.prepare_triangle_separation_rows(points, aa, bb, pr))
    if fixed_a:
        aa = np.asarray(fixed_a, np.int64)
        pb = np.concatenate(fixed_xyz, axis=0)
        bb = np.arange(len(pb), dtype=np.int64).reshape((-1, 3))
        pr = np.column_stack((np.arange(len(aa)), np.arange(len(aa))))
        sep_results.append(Q.prepare_triangle_separation_rows(points, aa, bb, pr, points_b=pb))
    constraints = []
    frozen = set()
    prep_parts = []
    for result in sep_results:
        constraints.extend(result["constraints"])
        frozen.update(result["preparation"].get("freeze_source_nodes", []))
        prep_parts.append(result["preparation"])

    weak_numeric_rows=[]
    baseline_constraint_violations=[]
    for ri,row in enumerate(constraints):
        nodes=np.asarray(row["point_rows"],np.int64)
        vectors=np.asarray(row["vectors"],np.float64)
        origin=np.asarray(row["origin"],np.float64)
        lhs=float(np.einsum("ij,ij->",vectors,
            np.asarray(points,dtype=np.float32)[nodes].astype(np.float64)-origin[None,:]))
        upper=float(row["upper_bound"])
        if lhs>upper:
            baseline_constraint_violations.append({"constraint_row":int(ri),"pair_id":int(row.get("pair_id",-1)),
                "face_pair":row.get("face_pair"),"lhs_m":lhs,"upper_bound_m":upper,"excess_m":lhs-upper,
                "kind":row.get("kind"),"shared_feature_size":row.get("shared_feature_size")})
        if "baseline_numeric_allowance_m" in row:
            pid=int(row.get("pair_id",-1))
            weak_numeric_rows.append({
                "constraint_row":int(ri),"pair_id":pid,"pair_identities":[list(x) for x in source_pair_identities[pid]] if 0<=pid<len(source_pair_identities) else None,
                "face_pair":row.get("face_pair"),"point_rows":nodes.tolist(),
                "shared_feature_rows":row.get("shared_feature_rows"),"axis_name":row.get("shared_feature_axis_name"),
                "baseline_lhs_m":float(row["baseline_float64_lhs_m"]),
                "original_upper_bound_m":float(row["baseline_float64_old_upper_m"]),
                "emitted_upper_bound_m":upper,"allowance_m":float(row["baseline_numeric_allowance_m"]),
                "rule":row["baseline_numeric_allowance_rule"]})
    if baseline_constraint_violations:
        raise ValueError("witness baseline still violates emitted halfspaces: "+json.dumps(baseline_constraint_violations[:8],sort_keys=True))

    retained_pair_details = []
    for canonical_pair, origins in pair_origins.items():
        retained_pair_details.append({"identities": [list(canonical_pair[0]), list(canonical_pair[1])], "origins": origins})
    rejected_pair_details = []
    rejected_detail_omissions = 0
    for label, prep, identities in (("source_source", sep_results[0] if src_a else None, source_pair_identities),):
        if prep is None:
            continue
        pinfo = prep["preparation"]
        details = pinfo.get("rejected_pair_details", [])
        rejected_detail_omissions += max(0, int(pinfo.get("rejected_pairs", 0)) - len(details))
        for item in details:
            pid = int(item["pair_id"])
            if pid < 0 or pid >= len(identities):
                raise ValueError("separator rejected-pair source identity index is invalid")
            idents = identities[pid]
            rejected_pair_details.append({"preparation_group": label, "pair_id": pid,
                                          "identities": [list(idents[0]), list(idents[1])],
                                          "origins": pair_origins[tuple(sorted(idents))], **item})
    fixed_result_index = 1 if src_a else 0
    fixed_result = sep_results[fixed_result_index] if fixed_a else None
    if fixed_result is not None:
        pinfo = fixed_result["preparation"]
        details = pinfo.get("rejected_pair_details", [])
        rejected_detail_omissions += max(0, int(pinfo.get("rejected_pairs", 0)) - len(details))
        for item in details:
            pid = int(item["pair_id"])
            if pid < 0 or pid >= len(fixed_pair_identities):
                raise ValueError("separator rejected-pair fixed identity index is invalid")
            idents = fixed_pair_identities[pid]
            rejected_pair_details.append({"preparation_group": "source_fixed", "pair_id": pid,
                                          "identities": [list(idents[0]), list(idents[1])],
                                          "origins": pair_origins[tuple(sorted(idents))], **item})
    remap_receipt = {
        "schema": "h706.witness-face-rebase-v23d-fixed-fixed-o-skip-scalar-text.v1",
        "status": "PASS_EXACT_ORIENTED_FACE_MAP_AND_PAIR_ACCOUNTING",
        "producer": pin(Path(__file__)),
        "current_state": {"input_state_pin": input_state_pin,
                           "source_surface_pins": {r: state["source_surfaces"][r] for r in role_order},
                           "current_face_array_sha256": current_face_sha},
        "reports": report_receipts,
        "pair_accounting": {
            "raw_report_pairs": int(total_raw_pairs),
            "stale_source_triangle_pairs_dropped": int(total_stale_pairs),
            "duplicate_mapped_pairs_deduplicated": int(total_duplicate_pairs),
            "inactive_fixed_fixed_o_pairs_skipped": int(total_skipped_fixed_pairs),
            "retained_unique_pairs": int(len(all_pairs)),
            "identity": int(total_raw_pairs) == int(total_stale_pairs + total_duplicate_pairs + total_skipped_fixed_pairs + len(all_pairs)),
        },
        "retained_pairs": retained_pair_details,
        "dropped_pairs": dropped_pairs,
        "skipped_fixed_fixed_o_pairs": all_skipped_fixed_pair_details,
        "inactive_fixed_pair_policy": {
            "enabled": bool(skip_fixed_o_pairs),
            "skipped_pair_classes": ["O/O", "O/all185-bone"],
            "outer_O_fixed_by_inner_back_layer_policy": bool(skip_fixed_o_pairs),
            "active_source_roles_still_mapped": sorted(set().union(*[
                set(x.get("active_roles_requiring_point_map", [])) for x in report_receipts
            ]) if report_receipts else set()),
        },
        "halfspace_preparation": {
            "source_source_pairs": int(source_pair_keys), "source_fixed_pairs": int(fixed_pair_keys),
            "usable_separating_pairs": sum(int(x.get("usable_separating_pairs", 0)) for x in prep_parts),
            "rejected_pairs": sum(int(x.get("rejected_pairs", 0)) for x in prep_parts),
            "linear_halfspace_rows": int(len(constraints)),
            "rejected_pair_details": rejected_pair_details,
            "rejected_pair_details_omitted_by_owner_limit": int(rejected_detail_omissions),
            "freeze_source_nodes": sorted(map(int, frozen)),
            "baseline_constraint_violation_count_after_numeric_allowance":int(len(baseline_constraint_violations)),
            "weak_shared_feature_numeric_ulp_adjustment_count":int(len(weak_numeric_rows)),
            "weak_shared_feature_numeric_ulp_adjustments":weak_numeric_rows,
            "weak_shared_feature_max_numeric_allowance_m":float(max((r["allowance_m"] for r in weak_numeric_rows),default=0.0)),
        },
        "physical_admission": False,
        "native_steps": 0,
    }
    if len(all_skipped_fixed_pair_details) != total_skipped_fixed_pairs:
        raise ValueError("fixed/fixed O pair receipt count does not match skipped pair total")
    remap_path = None
    if output_dir is not None:
        out = Path(output_dir)
        remap_path = out / "witness-face-remap.json"
        remap_path.write_text(json.dumps(remap_receipt, indent=2, sort_keys=True) + "\n")
    by = {}
    for ri, row in enumerate(constraints):
        for node in np.unique(np.asarray(row["point_rows"], np.int64)):
            by.setdefault(int(node), []).append(ri)
    preparation = {
        "exact_pair_witnesses": int(len(all_pairs)),
        "inactive_fixed_fixed_o_pairs_skipped": int(total_skipped_fixed_pairs),
        "source_source_pairs": int(source_pair_keys),
        "source_fixed_pairs": int(fixed_pair_keys),
        "usable_separating_pairs": sum(int(x.get("usable_separating_pairs", 0)) for x in prep_parts),
        "rejected_pairs": sum(int(x.get("rejected_pairs", 0)) for x in prep_parts),
        "linear_halfspace_rows": len(constraints),
        "baseline_constraint_violation_count_after_numeric_allowance":int(len(baseline_constraint_violations)),
        "weak_shared_feature_numeric_ulp_adjustment_count":int(len(weak_numeric_rows)),
        "weak_shared_feature_max_numeric_allowance_m":float(max((r["allowance_m"] for r in weak_numeric_rows),default=0.0)),
        "freeze_source_nodes": sorted(map(int, frozen)),
        "retained_gap_fraction": .5,
        "reports": report_receipts,
        "face_identity": "role plus exact cyclic-oriented vertex triangle mapped through authenticated typed point ancestry",
        "stale_face_policy": "drop only report pairs whose exact oriented source triangle is absent from current role surface; reversed-current triangle is an error",
        "pair_accounting": remap_receipt["pair_accounting"],
        "remap_receipt": ({"path": str(remap_path.resolve()), "sha256": sha(remap_path)} if remap_path else None),
    }
    info = {
        "reports": report_receipts,
        "exact_pairs": len(all_pairs),
        "constraints": len(constraints),
        "frozen_source_nodes": len(frozen),
        "deduplicated_pairs": total_duplicate_pairs,
        "stale_face_pairs_dropped": total_stale_pairs,
        "source_source_pairs": source_pair_keys,
        "source_fixed_pairs": fixed_pair_keys,
        "usable_separating_pairs": preparation["usable_separating_pairs"],
        "rejected_pairs": preparation["rejected_pairs"],
        "pair_accounting": remap_receipt["pair_accounting"],
        "remap_receipt": preparation["remap_receipt"],
    }
    return {"constraints": constraints, "row_indices_by_point": by, "preparation": preparation}, info


def expand_triangle_ring_once(bad_global_rows, faces_by_role, source_global_rows_by_role,
                              fixed_global_rows=(), witness_frozen_global_rows=()):
    """Add one direct incident-triangle ring around mutable bad-corner surface vertices."""
    bad = set(map(int, np.asarray(bad_global_rows, dtype=np.int64).reshape(-1)))
    fixed = set(map(int, fixed_global_rows))
    witness = set(map(int, witness_frozen_global_rows))
    excluded = fixed | witness
    seeds = bad - excluded
    ring = set()
    role_receipts = {}
    for role in ("D", "O", "I"):
        if role not in faces_by_role or role not in source_global_rows_by_role:
            raise ValueError("one-ring source role maps must include D, O, and I")
        faces = np.asarray(faces_by_role[role], dtype=np.int64)
        global_rows = np.asarray(source_global_rows_by_role[role], dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3 or global_rows.ndim != 1:
            raise ValueError(role + ": malformed source triangle/global-row arrays")
        if faces.size and (faces.min() < 0 or faces.max() >= len(global_rows)):
            raise ValueError(role + ": source face index outside source point map")
        seed_local = np.flatnonzero(np.isin(global_rows, np.fromiter(seeds, dtype=np.int64)))
        if len(seed_local) == 0:
            role_receipts[role] = {"seed_source_rows": 0, "incident_source_faces": 0,
                                   "ring_global_rows": 0}
            continue
        incident = np.any(np.isin(faces, seed_local), axis=1)
        face_rows = np.flatnonzero(incident)
        local_ring = np.unique(faces[incident].reshape(-1)) if len(face_rows) else np.empty(0, np.int64)
        ring_global = global_rows[local_ring]
        if len(ring_global) and np.any(ring_global < 0):
            bad_local = local_ring[np.flatnonzero(ring_global < 0)].tolist()
            raise ValueError(role + ": incident triangle contains unmapped global point rows " + str(bad_local[:12]))
        ring.update(map(int, ring_global))
        role_receipts[role] = {"seed_source_rows": int(len(seed_local)),
                               "incident_source_faces": int(len(face_rows)),
                               "ring_global_rows": int(len(np.unique(ring_global)))}
    expanded = (bad | ring) - excluded
    info = {
        "schema": "h706.source-triangle-one-ring-quality-free-set.v1",
        "seed_bad_corner_global_rows": int(len(bad)),
        "mutable_bad_corner_seeds": int(len(seeds)),
        "excluded_fixed_rows": int(len(fixed)),
        "excluded_witness_frozen_rows": int(len(witness)),
        "one_ring_global_rows_before_exclusions": int(len(ring)),
        "new_ring_only_rows": int(len(ring - bad)),
        "fixed_or_witness_rows_reexcluded_after_expansion": int(len(ring & excluded)),
        "final_movable_rows": int(len(expanded)),
        "role_incidence": role_receipts,
        "expansion_rule": "exactly one set of source triangles incident to initially mutable bad-cell corners; no recursive flooding",
        "all_affected_tets_remain_in_global_quality_constraints": True,
    }
    return np.asarray(sorted(expanded), dtype=np.int64), info



def expand_inner_back_layer_once(bad_global_rows, faces_by_role, source_global_rows_by_role,
                                  tetrahedra, actual_back_fat_rows, fixed_global_rows=(),
                                  witness_frozen_global_rows=(), inner_scope_global_rows=()):
    """Free only bad I/D surface corners, their direct I/D triangle ring, and
    actual-back Fat points in the bad-cell one-star. O/back-core/rim/routes and
    every unselected point remain fixed. All volume cells remain in the LP.
    """
    bad=set(map(int,np.asarray(bad_global_rows,np.int64).reshape(-1)))
    back=set(map(int,np.asarray(actual_back_fat_rows,np.int64).reshape(-1)))
    fixed=set(map(int,fixed_global_rows));witness=set(map(int,witness_frozen_global_rows))
    excluded=fixed|witness
    surface=set()
    for role in ("I","D"):
        gr=np.asarray(source_global_rows_by_role[role],np.int64)
        surface.update(map(int,gr[gr>=0]))
    seeds=(bad & (surface|back))-excluded
    ring=set();role_receipts={}
    for role in ("I","D"):
        faces=np.asarray(faces_by_role[role],np.int64)
        gr=np.asarray(source_global_rows_by_role[role],np.int64)
        if faces.ndim!=2 or faces.shape[1]!=3 or (faces.size and (faces.min()<0 or faces.max()>=len(gr))):
            raise ValueError(role+": malformed source faces or point map")
        local_seed=np.flatnonzero(np.isin(gr,np.fromiter(seeds,dtype=np.int64)))
        incident=np.any(np.isin(faces,local_seed),axis=1) if len(local_seed) else np.zeros(len(faces),bool)
        incident_faces=np.flatnonzero(incident)
        local_ring=np.unique(faces[incident].reshape(-1)) if len(incident_faces) else np.empty(0,np.int64)
        mapped=gr[local_ring]
        if len(mapped) and np.any(mapped<0):raise ValueError(role+": inner ring includes unmapped global point rows")
        ring.update(map(int,mapped))
        role_receipts[role]={"seed_source_rows":int(len(local_seed)),
                              "incident_source_faces":int(len(incident_faces)),
                              "ring_global_rows":int(len(np.unique(mapped)))}
    t=np.asarray(tetrahedra,np.int64)
    if t.ndim!=2 or t.shape[1]!=4:raise ValueError("volume tetrahedra malformed")
    if seeds:
        seedarr=np.fromiter(seeds,dtype=np.int64)
        incident_tets=np.any(np.isin(t,seedarr),axis=1)
        star_nodes=set(map(int,np.unique(t[incident_tets])))
    else:
        incident_tets=np.zeros(len(t),bool);star_nodes=set()
    back_ring=back & star_nodes
    free=(seeds|ring|back_ring)-excluded
    scope=set(map(int,inner_scope_global_rows))
    info={"schema":"h706.inner-back-layer-quality-free-set.v1",
          "policy":"INNER_BACK_LAYER_REPAIR",
          "seed_bad_corner_rows":int(len(bad)),
          "eligible_bad_I_D_or_actual_back_seeds":int(len(seeds)),
          "I_D_surface_seed_rows":int(len(seeds&surface)),
          "actual_back_Fat_seed_rows":int(len(seeds&back)),
          "direct_I_D_triangle_ring_rows":int(len(ring)),
          "actual_back_Fat_rows_in_seed_cell_stars":int(len(back_ring)),
          "released_inner_scope_rows":int(len(free&scope)),
          "fixed_global_rows":int(len(fixed)),
          "witness_frozen_rows":int(len(witness)),
          "free_rows":int(len(free)),
          "O_surface_rows_never_released":True,
          "rims_routes_and_registered_attachments_remain_fixed":True,
          "all_affected_tets_remain_in_global_quality_constraints":True,
          "role_incidence":role_receipts}
    return np.asarray(sorted(free),np.int64),info

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--state",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--trust-cap-m",type=float,default=2e-5);ap.add_argument("--seconds",type=float,default=30);ap.add_argument("--allow-external-boundary",action="store_true");ap.add_argument("--inner-back-layer-repair",action="store_true");ap.add_argument("--witness-report",type=Path,action="append",default=[]);ap.add_argument("--witness-preflight-only",action="store_true");ap.add_argument("--objective-mode",choices=("legacy","role_equal","role_equal_nearpass"),default="legacy");args=ap.parse_args()
 if not 0 < args.trust_cap_m <= .0005: raise ValueError('trust cap must be positive and at most 0.5 mm')
 if args.inner_back_layer_repair:
  if args.allow_external_boundary:raise ValueError("INNER_BACK_LAYER_REPAIR cannot release the external boundary")
  if args.trust_cap_m>5.e-5:raise ValueError("INNER_BACK_LAYER_REPAIR maximum displacement is 50 micrometres")
 assert 0<args.seconds<=30.,"v12 total SLP time limit must be at most 30 seconds"
 st=json.loads(args.state.read_text());st=st.get("current_verified_state_20261031",st);out=args.output;out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 volumes={};surfaces={};spkeys={};allpoints=[];global_lookup={};vmap={};smap={}
 for role,e in st["source_surfaces"].items():
  assert sha(e["path"])==e["sha256"];a=load(e["path"]);surfaces[role]=a
  spkeys[role]={T.key(a["points_world_m_f32"][r]) for r in np.unique(a["faces"])}
 if args.witness_preflight_only:
  parts=[];preflight_map={};cursor=0
  for role in ("D","O","I"):
   rp=np.asarray(surfaces[role]["points_world_m_f32"],np.float32);parts.append(rp);preflight_map[role]=np.arange(cursor,cursor+len(rp),dtype=np.int64);cursor+=len(rp)
  preflight_points=np.concatenate(parts,axis=0)
  _,preflight_info=_witness_separation_rows(args.witness_report,st,surfaces,preflight_map,preflight_points,out,pin(args.state),
                                                    skip_fixed_o_pairs=bool(args.inner_back_layer_repair))
  print("WITNESS_PREFLIGHT_ONLY",json.dumps(preflight_info,separators=(",",":")),flush=True)
  return
 for vr in ("Skin","Fat"):
  e=st[vr+"_current"];assert sha(e["path"])==e["sha256"];a=load(e["path"]);volumes[vr]=a;p=a["points_world_m"];mp=[]
  for point in p:
   key=T.key(point)
   if vr=="Fat" and key in spkeys["I"] and key in global_lookup:row=global_lookup[key]
   else:row=len(allpoints);allpoints.append(point);global_lookup.setdefault(key,row)
   mp.append(row)
  vmap[vr]=np.asarray(mp,np.int64)
 p=np.asarray(allpoints,np.float32);t=np.vstack([vmap[vr][volumes[vr]["tetrahedra"]] for vr in ("Skin","Fat")]);mat=np.concatenate([volumes[vr]["material_indices"] for vr in ("Skin","Fat")])
 quality_group_ids=np.concatenate([np.zeros(len(volumes["Skin"]["tetrahedra"]),dtype=np.int8),np.ones(len(volumes["Fat"]["tetrahedra"]),dtype=np.int8)])
 d,q=M.measurements(p,t);initial=M.passes(d,q);assert np.all(d>0)
 bad_corner_rows=np.unique(t[~initial]).astype(np.int64)
 scope=H/"current-back412-scope-597-root/run003/output-001/protected-back412-aabb-current-O558-I50-D264-rows.npz"
 assert sha(scope)=="1abbf3c5887cfa507c303e845dafbb78d247e6b7a4893722a5d81ed202fd2676"
 masks=load(scope);routes=json.loads((H/"verified-volume-preparation-010-merged038005/rim/protected149-current-routes.json").read_text())["surface_routes"]
 fixed=set();anchor_counts={"protected_scope_rows":0,"source_rim_rows":0,"route149_exact_rows":0,"actual_back412_fat_rows":0}
 for role,a in surfaces.items():
  pp=a["points_world_m_f32"];sf=a["faces"];mp=np.asarray([global_lookup.get(T.key(x),-1) for x in pp],np.int64);smap[role]=mp
  if role=="O" or (role=="D" and not args.allow_external_boundary and not args.inner_back_layer_repair):fixed.update(map(int,mp[mp>=0]))
  scope_rows=set(map(int,masks[role+"_vertex_rows"]))
  if not (args.inner_back_layer_repair and role in ("I","D")):
   protected=set(scope_rows)
  else:
   protected=set()
  anchor_counts["protected_scope_rows"]+=sum(1 for r in scope_rows if 0<=r<len(mp) and mp[r]>=0)
  ed=np.sort(np.concatenate([sf[:,[0,1]],sf[:,[1,2]],sf[:,[2,0]]]),axis=1);ue,ct=np.unique(ed,axis=0,return_counts=True);rim_rows=set(map(int,ue[ct==1].ravel()));protected.update(rim_rows);anchor_counts["source_rim_rows"]+=sum(1 for r in rim_rows if mp[r]>=0)
  for r in routes:
   if r["role"]==role:
    hit=np.flatnonzero(np.all(pp==np.asarray(r["target_world_m_f32"],np.float32),axis=1));assert len(hit)==1;protected.add(int(hit[0]));anchor_counts["route149_exact_rows"]+=int(mp[int(hit[0])]>=0)
  fixed.update(int(mp[r]) for r in protected if mp[r]>=0)
 back=load(H/"paired-I-current620-root/actual-back412-fat-point-mask.npz")["affected_volume_point_rows"]
 back_global=np.asarray(vmap["Fat"][back],np.int64)
 anchor_counts["actual_back412_fat_rows"]=int(len(back))
 if not args.inner_back_layer_repair:fixed.update(map(int,back_global))
 separation_rows,witness_info=_witness_separation_rows(args.witness_report,st,surfaces,smap,p,out,pin(args.state),
                                                              skip_fixed_o_pairs=bool(args.inner_back_layer_repair))
 witness_frozen=np.empty(0,np.int64)
 if separation_rows is not None:witness_frozen=np.asarray(separation_rows["preparation"].get("freeze_source_nodes",[]),np.int64)
 if args.inner_back_layer_repair:
  inner_scope_global=set()
  for role in ("I","D"):
   rr=np.asarray(masks[role+"_vertex_rows"],np.int64);mapped_scope=smap[role][rr];inner_scope_global.update(map(int,mapped_scope[mapped_scope>=0]))
  free_rows,one_ring_info=expand_inner_back_layer_once(
   bad_corner_rows,{r:np.asarray(surfaces[r]["faces"],np.int64) for r in ("D","I")},smap,t,
   back_global,fixed_global_rows=fixed,witness_frozen_global_rows=witness_frozen,
   inner_scope_global_rows=inner_scope_global)
 else:
  free_rows,one_ring_info=expand_triangle_ring_once(
   bad_corner_rows,{r:np.asarray(surfaces[r]["faces"],np.int64) for r in ("D","O","I")},smap,
   fixed_global_rows=fixed,witness_frozen_global_rows=witness_frozen)
 free=np.zeros(len(p),bool);free[free_rows]=True
 print("WITNESS_CONDITIONING",json.dumps(witness_info,separators=(",",":")),flush=True)
 print("LP_INPUT",json.dumps({"global_points":len(p),"cells":len(t),"free_nodes":int(free.sum()),"bad":int((~initial).sum()),"fixed_outer_and_deep":(not args.allow_external_boundary and not args.inner_back_layer_repair),"inner_back_layer_repair":bool(args.inner_back_layer_repair),"protected_anchors":anchor_counts,"fixed_global_rows":len(fixed),"one_ring_expansion":one_ring_info,"trust_cap_m":args.trust_cap_m,"objective_mode":args.objective_mode},separators=(",",":")),flush=True)
 candidate,diag=Q.propose_quality_repair(p,t,mat,free,args.trust_cap_m,time_limit_s=args.seconds,linear_margin=.001,max_backtracks=12,separation_rows=separation_rows,objective_mode=args.objective_mode,quality_group_ids=quality_group_ids)
 (out/"lp-diagnostics.json").write_text(json.dumps(diag,indent=2)+"\n")
 compact_keys=("status","solver_status","bad_before","bad_after","accepted_components","inner_iteration_count","strict_quality_count_progress","strict_bad_score_progress","bad_score_before","bad_score_after","per_step_trust_cap_m","total_trust_cap_m","good_cell_regressions","exact_F32_positive_cells","max_free_node_l1_displacement_bound_m")
 print("LP_STATUS",json.dumps({k:diag.get(k) for k in compact_keys},separators=(",",":")),flush=True)
 if not diag["status"].startswith("PASS"):return
 nd,nq=M.measurements(candidate,t);newgood=M.passes(nd,nq);assert np.all(nd>0) and not np.any(initial&~newgood)
 assert np.array_equal(candidate[list(fixed)].view("u4"),p[list(fixed)].view("u4"))
 if args.inner_back_layer_repair:
  free_mask=np.zeros(len(p),bool);free_mask[free_rows]=True
  if not np.array_equal(candidate[~free_mask].view("u4"),p[~free_mask].view("u4")):
   raise ValueError("INNER_BACK_LAYER_REPAIR changed a point outside its declared bad-corner/ring/back-star free set")
 result=dict(st);source_surfaces={}
 for role,a in surfaces.items():
  payload,changed=source_point_step_payload(a,candidate,smap[role],st["source_surfaces"][role],args.trust_cap_m)
  if args.inner_back_layer_repair and role=="O" and len(changed):raise ValueError("INNER_BACK_LAYER_REPAIR moved the outer O surface")
  if args.inner_back_layer_repair and role not in ("I","D") and len(changed):raise ValueError("INNER_BACK_LAYER_REPAIR moved an undeclared source role")
  if not len(changed):source_surfaces[role]=st["source_surfaces"][role];continue
  new=payload["points_world_m_f32"]
  dest=out/("candidate-"+role+"-typed-current.npz");np.savez_compressed(dest,**payload);source_surfaces[role]=pin(dest)
 for vr,a in volumes.items():
  old=a["points_world_m"];new=candidate[vmap[vr]];tt=a["tetrahedra"];dd,qq=M.measurements(new,tt);bad=int((~M.passes(dd,qq)).sum());assert np.all(dd>0)
  payload=dict(a);payload["points_world_m"]=new
  if "points_world_m_f32" in payload:payload["points_world_m_f32"]=new.copy()
  n=len(tt);payload.update(current_geometry_parent_path=np.asarray([st[vr+"_current"]["path"]]),current_geometry_parent_sha256=np.asarray([st[vr+"_current"]["sha256"]]),output_cell_to_input_cell=np.arange(n,dtype=np.int64),input_cell_to_output_cell=np.arange(n,dtype=np.int64),retained_input_cell_rows=np.arange(n,dtype=np.int64),replacement_parent_offsets=np.asarray([0],np.int64),replacement_parent_cell_rows=np.empty(0,np.int64),replacement_tet_offsets=np.asarray([0,0],np.int64),replacement_tetrahedra=np.empty((0,4),np.int64),input_node_to_output_node=np.arange(len(new),dtype=np.int64),output_node_to_input_node=np.arange(len(new),dtype=np.int64),output_point_to_input_point_row=np.arange(len(new),dtype=np.int64),output_point_parent_offsets=np.zeros(len(new)+1,np.int64),output_point_parent_input_cell_rows=np.empty(0,np.int64))
  for role,e in source_surfaces.items():
   payload["typed_"+role+"_candidate_path"]=np.asarray([e["path"]]);payload["typed_"+role+"_candidate_sha256"]=np.asarray([e["sha256"]])
  dest=out/(vr.lower()+"-coupled-quality-point-proposal.npz");np.savez_compressed(dest,**payload)
  result[vr+"_current"]={**st[vr+"_current"],**pin(dest),"native_quality_failures":bad,"nonpositive":0,"geometry_changed_points":int(np.any(new.view("u4")!=old.view("u4"),axis=1).sum())}
  print("QUALITY",vr,bad,flush=True)
 result.update(source_surfaces=source_surfaces,physical_admission=False,native_steps=0,surface_crossings="NOT_YET_AUDITED",source_state=pin(args.state),source_triangle_one_ring_expansion=one_ring_info,coupled_point_repair={"producer":pin(Path(__file__)),"maximum_displacement_m":args.trust_cap_m,"quality_step_policy":"up to 10 continuous SLP steps at total trust cap / 10 each","fixed_outer_and_deep":(not args.allow_external_boundary and not args.inner_back_layer_repair),"inner_back_layer_repair":bool(args.inner_back_layer_repair),"inner_back_layer_policy":"INNER_BACK_LAYER_REPAIR" if args.inner_back_layer_repair else None,"outer_O_scope_fixed":True,"I_D_scope_released_only_inside_bad_corners_and_one_triangle_ring":bool(args.inner_back_layer_repair),"actual_back_Fat_points_released_only_inside_bad_cell_star":bool(args.inner_back_layer_repair),"protected_anchors":anchor_counts,"fixed_global_rows":len(fixed),"quality_helper":{"path":str(_Q_PATH),"sha256":_Q_EXPECTED_SHA},"quality_objective_mode":args.objective_mode,"quality_objective_group_ids":"Skin=0,Fat=1","all_source_roles_rebased_to_actual_point_parent":True,"all_roles_follow_shared_global_F32_rows":True,"source_roles":sorted(surfaces),"diagnostics":pin(out/"lp-diagnostics.json"),"exact_witness_conditioning":witness_info})
 for k in ("operations","independent_replay","exact_context"):result.pop(k,None)
 (out/"state.json").write_text(json.dumps(result,indent=2)+"\n")
 changed=[r for r in source_surfaces if source_surfaces[r]["sha256"]!=st["source_surfaces"][r]["sha256"]]
 if args.inner_back_layer_repair and (not changed or not set(changed).issubset({"I","D"})):
  raise ValueError("INNER_BACK_LAYER_REPAIR requires a changed I/D source surface and forbids O changes")
 bounded={}
 for r in changed:
  entry={"maximum_displacement_m":args.trust_cap_m}
  if args.inner_back_layer_repair:
   if r not in ("I","D"):raise ValueError("INNER_BACK_LAYER_REPAIR config may declare I/D only")
   entry["inner_back_layer_repair"]=True
  bounded[r]=entry
 config={"schema":"h706.combined-current-exact-surface-diff-config.v1","baseline_candidates":st["source_surfaces"],"candidate_surfaces":source_surfaces,"changed_roles":changed,"bounded_point_repair":bounded}
 (out/"context-config.json").write_text(json.dumps(config,indent=2)+"\n")
 print("STATE",str(out/"state.json"),"elapsed_s",time.monotonic()-start,flush=True)
if __name__=="__main__":main()
