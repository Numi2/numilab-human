# Versioned v26e derivative: witness-directed expanded rows and one-ULP correction; prior gates retained.
# Parent source SHA256: c944380189b71d43e6c91c429c359c229cecbf157ced9e7693e3560e09fa92eb (v26b)
# Count-stall solver block donor SHA256: 1676e6f9138fe5eed1b4a697ec0986d983371e361acfe7379ab271a866278b5f
# Back-lineage owner SHA256: 68124ffcffcc2bf5147de081e50f87739089c3be5c2370be8ae3b27bef4e922a
"""Versioned v26e builder: witness-directed expansion and bounded ULP correction."""
"""Source-only target residual fit with per-bad-cluster bounded local SLP patches."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np
H=Path(__file__).resolve().parents[2]
def imp(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
T=imp("coupled648T",H/"transfer_surface_contractions_cross_tissue_470_root.py");M=T.m
_Q_PATH=H/"629-quality-lp/quality_lp_629_v18_role_balanced_shared_feature.py"
_Q_EXPECTED_SHA="d88d9451670d7db685314c30354430f7123e176c8d3f2b540b018586e552b850"
if hashlib.sha256(_Q_PATH.read_bytes()).hexdigest()!=_Q_EXPECTED_SHA:raise RuntimeError("pinned v17b numeric-ULP shared-feature helper changed")
Q=imp("coupled721Qv16rowscaledshared",_Q_PATH)
B=imp("coupled721Base197",H/"fixed-boundary-quality-topology-196/run_fixed_boundary_dp_batch_197.py")
_JOINT_PATH=H/"coupled-anchor-collapse-780-root/ordinary-allrole-v26e-witness-ulp-replay-controls/joint_small_patch_slsqp_936_v26e_witness_ulp.py"
_JOINT_EXPECTED_SHA="6422b020e80d6518655acea2581228c78c04e36fab2cdb7edd29bea981588dfe"
if hashlib.sha256(_JOINT_PATH.read_bytes()).hexdigest()!=_JOINT_EXPECTED_SHA:
 raise RuntimeError("pinned joint small-patch SLSQP helper changed")
JOINT=imp("coupled_quality_joint_small_patch_936",_JOINT_PATH)
_F32_BACKTRACK_PATH=H/"quality-f32-backtracking-923-root/quality_f32_backtracking_923.py"
_F32_BACKTRACK_SHA="8e7959e82ffc5adddf2dd24dcdcbb1b4550dbce81d62ca628f4f389688860a4c"
if hashlib.sha256(_F32_BACKTRACK_PATH.read_bytes()).hexdigest()!=_F32_BACKTRACK_SHA:
 raise RuntimeError("pinned Float32 full-star backtracking helper changed")
F32_BACKTRACK=imp("quality_f32_backtracking_923_for_v26b4",_F32_BACKTRACK_PATH)
_ULP_PATH=Path(__file__).with_name("quality_f32_witness_ulp_948.py")
_ULP_SHA="7769ca5565b341575f56f4a4a6b50c9556d2188c90383542212ffed783f7b89a"
if hashlib.sha256(_ULP_PATH.read_bytes()).hexdigest()!=_ULP_SHA:raise RuntimeError("pinned witness ULP helper changed")
ULP=imp("quality_f32_witness_ulp_948_for_v26e",_ULP_PATH)
_COW_NPZ_PATH=H/"npz-cow-append-925-root/npz_cow_append_925.py"
_COW_NPZ_SHA="4799f33d73a37072498565b6ff9c20d181916f0158e253fe5c7e9056ba2c0270"
if hashlib.sha256(_COW_NPZ_PATH.read_bytes()).hexdigest()!=_COW_NPZ_SHA:
 raise RuntimeError("pinned APFS clone NPZ writer changed")
COW_NPZ=imp("npz_cow_append_925_for_v26b4",_COW_NPZ_PATH)
_COW_COMPLETE_PATH=H/"npz-cow-append-926b-root/npz_cow_append_complete_926b.py"
_COW_COMPLETE_SHA="98dd06cafb9a38af4a8c1c7a803ed967d6effff73fd7d1528df7bedcfb40418f"
if hashlib.sha256(_COW_COMPLETE_PATH.read_bytes()).hexdigest()!=_COW_COMPLETE_SHA:
 raise RuntimeError("pinned complete-payload APFS NPZ adapter changed")
COW_COMPLETE=imp("npz_cow_append_complete_926b_for_v26b5",_COW_COMPLETE_PATH)
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def pin(p):return {"path":str(p),"sha256":sha(p)}
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

_BACK_LINEAGE_PATH=H/"coupled-anchor-collapse-780-root/coupled_anchor_collapse_795_outer_role.py"
_BACK_LINEAGE_SHA="68124ffcffcc2bf5147de081e50f87739089c3be5c2370be8ae3b27bef4e922a"
if sha(_BACK_LINEAGE_PATH)!=_BACK_LINEAGE_SHA:raise RuntimeError("authenticated back-lineage owner changed")
BACK=imp("coupled_anchor_collapse_795_for_v25",_BACK_LINEAGE_PATH)
_BACK_PROTECTION_PATH=H/"quality-partial-cavity-684-root/paired_connected_boundary_patch_856_extended_250um_construction.py"
_BACK_PROTECTION_SHA="25a2bf8cbb2ff776745cb3a68bb3ffdfe4b294d8a555f247ce1bc53b476c4897"
if sha(_BACK_PROTECTION_PATH)!=_BACK_PROTECTION_SHA:raise RuntimeError("authenticated back-protection owner changed")
BACK_PROTECTION=imp("paired_boundary_patch_854_for_v25j",_BACK_PROTECTION_PATH)

def _typed_archive_scalar_text(value, field_name, archive_path):
 arr=np.asarray(value)
 if arr.dtype.kind not in "US" or arr.shape not in ((),(1,)):
  raise ValueError(str(archive_path)+": "+field_name+" must be one scalar text value")
 out=str(arr.reshape(-1)[0])
 if not out:raise ValueError(str(archive_path)+": "+field_name+" is empty")
 return out

def authenticated_actual_back_fat_rows(state_path,state,surfaces,volumes,outdir):
 """Use the authenticated 854 owner; preserve all historical I-back records even when quotiented."""
 volume_caches={}
 for vr in ("Skin","Fat"):
  a=volumes[vr]
  points=np.asarray(a.get("points_world_m_f32",a["points_world_m"]),np.float32)
  volume_caches[vr]={"points":points,"coordinate_index":BACK_PROTECTION.point_key_index(points)}
 protected=BACK_PROTECTION.protected_scope(Path(state_path),state,"I",volume_caches,Path(outdir),fixture=False)
 fat_sources=protected["Fat"]["report"]["sources"]
 try: d_record=fat_sources["D"]
 except (KeyError,TypeError) as exc:raise ValueError("855 owner omitted current D-back lineage") from exc
 d_source_rows=np.unique(np.asarray(d_record.get("back_rows",[]),np.int64))
 d_lineage=d_record.get("back_lineage",{})
 if int(d_lineage.get("historical_back_records",-1))!=1673 or not d_lineage.get("current_transition_displacement_pass",False):
  raise ValueError("855 owner did not authenticate all 1673 D-back records/current transition")
 if np.any(d_source_rows<0) or np.any(d_source_rows>=len(surfaces["D"]["points_world_m_f32"])):
  raise ValueError("855 D-back current source rows outside D point domain")
 dmap=BACK_PROTECTION.exact_point_map(np.asarray(surfaces["D"]["points_world_m_f32"],np.float32),
  volume_caches["Fat"]["points"],"current D back/Fat exact map",
  active_rows=BACK_PROTECTION.active_surface_point_rows(surfaces["D"]),required_rows=d_source_rows,
  volume_index=volume_caches["Fat"]["coordinate_index"])
 d_fat_by_source=np.asarray(dmap[d_source_rows],np.int64)
 if np.any(d_fat_by_source<0) or len(np.unique(d_fat_by_source))!=len(np.unique(d_source_rows)):
  raise ValueError("855 current D-back source-to-Fat map is incomplete or noninjective")
 try: record=fat_sources["actual_fat_back_union"]
 except (KeyError,TypeError) as exc:raise ValueError("854 owner omitted the authenticated Fat/I-back record ledger") from exc
 historical=np.asarray(record["historical_source_rows"],np.int64)
 current_i=np.asarray(record["historical_to_current_source_rows"],np.int64)
 current_fat_by_record=np.asarray(record["historical_to_current_volume_rows"],np.int64)
 historical_fat_by_record=np.asarray(record["historical_direct_volume_rows_or_minus_one"],np.int64)
 direct=np.asarray(record["historical_rows"],np.int64)
 current=np.asarray(record["current_rows"],np.int64)
 union=np.asarray(record["current_historical_protected_volume_row_union"],np.int64)
 nrecords=int(record.get("historical_record_count",-1))
 if nrecords!=790 or historical.shape!=(790,) or len(np.unique(historical))!=790:
  raise ValueError("854 owner did not authenticate exactly 790 distinct historical I-back records")
 for name,rows in (("current I source representatives",current_i),
                   ("current Fat representatives by historical record",current_fat_by_record),
                   ("historical direct Fat coordinate matches",historical_fat_by_record)):
  if rows.shape!=(790,):raise ValueError("854 owner record-domain mismatch for "+name)
 if np.any(current_i<0) or np.any(current_i>=len(surfaces["I"]["points_world_m_f32"])):
  raise ValueError("854 owner left a historical I-back record without a current source representative")
 if np.any(current_fat_by_record<0) or np.any(current_fat_by_record>=len(volume_caches["Fat"]["points"])):
  raise ValueError("854 owner left a historical I-back record without a current Fat representative")
 if np.any((historical_fat_by_record < -1) | (historical_fat_by_record >= len(volume_caches["Fat"]["points"]))):
  raise ValueError("854 owner emitted an invalid direct historical Fat row")
 # Every historical record maps to exactly one current I row and one exact Fat row.
 # Multiple historical records may share a representative after a typed quotient.
 pair=np.column_stack((current_i,current_fat_by_record))
 if len(np.unique(pair,axis=0))!=len(np.unique(current_i)):
  raise ValueError("854 owner current I-to-Fat representatives are inconsistent")
 calculated_direct=np.unique(historical_fat_by_record[historical_fat_by_record>=0])
 calculated_current=np.unique(current_fat_by_record)
 calculated_union=np.union1d(calculated_current,calculated_direct)
 if not np.array_equal(np.sort(current),calculated_current):raise ValueError("854 current Fat representative set differs from record ledger")
 if not np.array_equal(np.sort(direct),calculated_direct):raise ValueError("854 direct historical Fat set differs from record ledger")
 if not np.array_equal(np.sort(union),calculated_union):raise ValueError("854 protected Fat union differs from record ledger")
 if int(record.get("lineage_mapped_volume_record_count",-1))!=790:
  raise ValueError("854 lineage did not cover every historical record")
 missing=int(np.count_nonzero(historical_fat_by_record<0))
 if int(record.get("historical_coordinates_directly_missing_count",-1))!=missing:
  raise ValueError("854 historical direct-match/orphan count is inconsistent")
 if int(record.get("historical_coordinates_directly_present_count",-1))!=790-missing:
  raise ValueError("854 historical direct-match count is inconsistent")
 coverage={"historical_record_count":790,"historical_reference_rows":historical.tolist(),
  "lineage_mapped_record_count":790,"uncovered_historical_record_count":0,
  "current_i_source_rows_by_record":current_i.tolist(),
  "current_unique_i_source_representatives":int(len(np.unique(current_i))),
  "collapsed_historical_record_count":int(790-len(np.unique(current_i))),
  "current_fat_volume_rows_by_record":current_fat_by_record.tolist(),
  "current_unique_fat_representatives":int(len(calculated_current)),
  "historical_direct_fat_rows_by_record_or_minus_one":historical_fat_by_record.tolist(),
  "historical_coordinate_match_record_count":int(790-missing),
  "historical_coordinate_orphan_record_count":missing,
  "historical_unique_direct_fat_rows":int(len(calculated_direct)),
  "current_plus_historical_unique_fat_protected_rows":int(len(calculated_union)),
  "source_owner_report":record}
 return {"fat_rows":calculated_union,"current_fat_rows":calculated_current,
  "historical_fat_rows":calculated_direct,"current_i_rows":current_i,
  "historical_i_rows":historical,"current_fat_by_record":current_fat_by_record,
  "historical_fat_by_record":historical_fat_by_record,"coverage":coverage,
  "lineage":record.get("source_lineage"),
  "scope_mask":protected["Fat"]["report"].get("back_scope"),
  "scope_report":protected["Fat"]["report"].get("back_scope_report"),
  "scope_reference_I":record.get("source_lineage",{}).get("reference"),
  "current_I":state["source_surfaces"]["I"],
  "D_back_lineage":{"historical_records":1673,"current_unique_source_rows":int(len(d_source_rows)),
      "current_unique_Fat_rows":int(len(np.unique(d_fat_by_source))),"owner_lineage":d_lineage},
  "current_d_source_rows":np.unique(d_source_rows).astype(np.int64),
  "current_d_fat_rows":np.unique(d_fat_by_source).astype(np.int64),
  "current_d_fat_by_source_row":d_fat_by_source.astype(np.int64),
  "source_owner":{"path":str(_BACK_PROTECTION_PATH),"sha256":_BACK_PROTECTION_SHA},
  "protection_report":protected["Fat"]["report"]}

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

def cow_payload_delta(parent_path, full_output_payload):
 """Diff a complete candidate NPZ mapping against its parent for clone-append."""
 parent_path=Path(parent_path)
 out=dict(full_output_payload)
 if any(not isinstance(k,str) or not k for k in out):raise ValueError("NPZ output keys must be nonempty strings")
 deleted=[];changed={};unchanged=[]
 with np.load(parent_path,allow_pickle=False) as parent:
  parent_keys=set(parent.files);output_keys=set(out)
  deleted=sorted(parent_keys-output_keys)
  for key in sorted(output_keys):
   value=np.asarray(out[key])
   if value.dtype.hasobject:raise TypeError("object array forbidden for NPZ key "+key)
   if key not in parent_keys:
    changed[key]=value
   else:
    previous=np.asarray(parent[key])
    if not COW_NPZ._same_array(previous,value):changed[key]=value
    else:unchanged.append(key)
  if set(changed)&set(deleted):raise AssertionError("COW update/delete sets overlap")
  if len(unchanged)+len(changed)!=len(output_keys):raise AssertionError("COW array diff does not partition output members")
 return changed,deleted,{"parent_member_count":len(parent_keys),"output_member_count":len(output_keys),
  "unchanged_member_count":len(unchanged),"changed_or_new_keys":sorted(changed),"deleted_keys":deleted,
  "comparison":"independent dtype, shape, Fortran-contiguity and order-A byte comparison via pinned 925 helper"}

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
            parent = Path(_npz_text_scalar(z,"typed_parent_candidate_path")).resolve()
            expected_parent_sha = _npz_text_scalar(z,"typed_parent_candidate_sha256")
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



def _npz_text_scalar(z, key):
    """Read exactly one Unicode/bytes text value from a 0-D or 1-D NPZ field."""
    if key not in z:
        raise ValueError("typed archive missing text field: " + str(key))
    value = np.asarray(z[key])
    if value.ndim not in (0, 1) or value.size != 1 or value.dtype.kind not in ("U", "S"):
        raise ValueError("typed archive text field must be a single 0-D/1-D text value: " + str(key))
    item = value.reshape(-1)[0]
    if value.dtype.kind == "S":
        result = bytes(item).decode("utf-8", errors="strict")
    else:
        result = str(item)
    if not result or "\x00" in result:
        raise ValueError("typed archive text field is empty or contains NUL: " + str(key))
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
                        pdata[k] = np.array(z[k], copy=True) if k.endswith("quotient") else _npz_text_scalar(z,k)
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
        "schema": "h706.witness-face-rebase-v23c-fixed-fixed-o-skip.v1",
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


def select_bad_cells_without_ordinary_free_corner(tetrahedra, bad_cell_rows, ordinary_free_rows):
    """Return native-bad cell rows with zero corners in the ordinary v25 free mask."""
    t=np.asarray(tetrahedra,dtype=np.int64)
    cells=np.asarray(bad_cell_rows,dtype=np.int64).reshape(-1)
    free=np.asarray(ordinary_free_rows,dtype=np.int64).reshape(-1)
    if t.ndim!=2 or t.shape[1]!=4 or (cells.size and (cells.min()<0 or cells.max()>=len(t))):
        raise ValueError("bad-cell roster is outside tetrahedron array")
    if free.size and (free.min()<0 or free.max()>=int(t.max())+1):
        raise ValueError("ordinary free-row roster is outside tetrahedron point domain")
    if not len(cells):return np.empty(0,np.int64)
    return cells[~np.any(np.isin(t[cells],free),axis=1)]

def expand_inner_back_remainder_once(selected_bad_cell_rows, faces_by_role,
                                      source_global_rows_by_role, tetrahedra,
                                      actual_back_fat_rows, fixed_global_rows=(),
                                      witness_frozen_global_rows=()):
    """Release only selected zero-ordinary-free bad-cell corners, their I/D
    source-face one-ring, selected volume-interior corners, and authenticated
    actual-back Fat nodes in the selected-corner tet star.
    """
    t=np.asarray(tetrahedra,dtype=np.int64)
    cells=np.asarray(selected_bad_cell_rows,dtype=np.int64).reshape(-1)
    if t.ndim!=2 or t.shape[1]!=4 or (cells.size and (cells.min()<0 or cells.max()>=len(t))):
        raise ValueError("selected bad-cell roster is malformed")
    if not len(cells):
        return np.empty(0,np.int64),{"schema":"h706.inner-back-remainder-free-set.v1",
            "selected_bad_cells":0,"free_global_rows":0,"empty_selection":True,
            "outer_O_surface_rows_never_released":True,
            "all_affected_tets_remain_in_global_quality_constraints":True}
    selected_corners=set(map(int,np.unique(t[cells].reshape(-1))))
    back=set(map(int,np.asarray(actual_back_fat_rows,dtype=np.int64).reshape(-1)))
    fixed=set(map(int,fixed_global_rows));witness=set(map(int,witness_frozen_global_rows))
    excluded=fixed|witness
    faces_by_role=dict(faces_by_role);maps=dict(source_global_rows_by_role)
    surface_rows=set();inner_surface_rows=set()
    for role in ("O","I","D"):
        if role not in maps or role not in faces_by_role:
            raise ValueError("remainder repair requires O/I/D source maps")
        gr=np.asarray(maps[role],np.int64);faces=np.asarray(faces_by_role[role],np.int64)
        if gr.ndim!=1 or faces.ndim!=2 or faces.shape[1]!=3 or (faces.size and (faces.min()<0 or faces.max()>=len(gr))):
            raise ValueError(role+": malformed source faces or global map")
        used_local=np.unique(faces.reshape(-1))
        used_global=gr[used_local]
        if np.any(used_global<0):raise ValueError(role+": source-face row is unmapped")
        surface_rows.update(map(int,gr[gr>=0]))
        if role in ("I","D"):inner_surface_rows.update(map(int,used_global))
    seed=(selected_corners&inner_surface_rows)-excluded
    ring=set();role_receipts={}
    for role in ("I","D"):
        gr=np.asarray(maps[role],np.int64);faces=np.asarray(faces_by_role[role],np.int64)
        seed_rows=np.fromiter(seed,dtype=np.int64) if seed else np.empty(0,np.int64)
        local_seed=np.flatnonzero(np.isin(gr,seed_rows))
        incident=np.any(np.isin(faces,local_seed),axis=1) if len(local_seed) else np.zeros(len(faces),bool)
        face_rows=np.flatnonzero(incident)
        local_ring=np.unique(faces[incident].reshape(-1)) if len(face_rows) else np.empty(0,np.int64)
        mapped=gr[local_ring]
        if len(mapped) and np.any(mapped<0):raise ValueError(role+": I/D one-ring has an unmapped source row")
        ring.update(map(int,mapped))
        role_receipts[role]={"selected_seed_source_rows":int(len(local_seed)),
            "incident_source_faces":int(len(face_rows)),"direct_ring_global_rows":int(len(np.unique(mapped)))}
    # Volume-interior selected bad corners are movable; every source-domain point
    # stays governed by the explicit O/I/D free-set rules below.
    interior=(selected_corners-surface_rows)-excluded
    # Back nodes may be one tet beyond a selected bad tet. Include the full
    # tet-star of its selected corners, then intersect only the authenticated 790-row set.
    corner_arr=np.fromiter(selected_corners,dtype=np.int64)
    star_cells=np.flatnonzero(np.any(np.isin(t,corner_arr),axis=1))
    star_nodes=set(map(int,np.unique(t[star_cells].reshape(-1)))) if len(star_cells) else set()
    back_ring=(back&star_nodes)-excluded
    free=(seed|ring|interior|back_ring)-excluded
    o_rows=set(map(int,np.asarray(maps["O"],np.int64)))
    if free&o_rows:raise ValueError("inner-back remainder free-set released an O source point")
    info={"schema":"h706.inner-back-remainder-free-set.v1",
        "policy":"zero-ordinary-free native-bad cells; selected bad I/D corners plus direct I/D source-face ring, selected interior corners, and authenticated actual-back Fat rows in selected-corner tet star",
        "selected_bad_cells":int(len(cells)),"selected_bad_corner_rows":int(len(selected_corners)),
        "I_D_selected_seed_rows":int(len(seed)),"I_D_direct_ring_rows_before_exclusions":int(len(ring)),
        "selected_interior_corner_rows":int(len(interior)),
        "authenticated_actual_back_Fat_rows_in_selected_corner_star":int(len(back_ring)),
        "selected_corner_star_tetrahedra":int(len(star_cells)),
        "fixed_global_rows":int(len(fixed)),"witness_frozen_rows":int(len(witness)),
        "free_global_rows":int(len(free)),"O_source_rows_never_released":True,
        "rims_routes_and_registered_attachments_remain_fixed":True,
        "all_global_native_good_cell_constraints_preserved":True,
        "role_incidence":role_receipts}
    return np.asarray(sorted(free),np.int64),info


def expand_ordinary_outer_o_once(bad_global_rows, o_faces, o_global_rows,
                                  fixed_global_rows=(), witness_frozen_global_rows=()):
    """Release only bad-cell O corners plus their direct O-triangle one-ring.

    The caller independently keeps protected O scope rows, source rims, routes,
    witness-frozen rows, and every unselected global point fixed.
    """
    bad=set(map(int,np.asarray(bad_global_rows,np.int64).reshape(-1)))
    gr=np.asarray(o_global_rows,np.int64)
    faces=np.asarray(o_faces,np.int64)
    if gr.ndim!=1 or faces.ndim!=2 or faces.shape[1]!=3:
        raise ValueError("ordinary O source face/point map malformed")
    if faces.size and (faces.min()<0 or faces.max()>=len(gr)):
        raise ValueError("ordinary O source face outside point map")
    if np.any(gr[np.unique(faces)]<0):
        raise ValueError("ordinary O incident surface has unmapped source points")
    fixed=set(map(int,fixed_global_rows))
    witness=set(map(int,witness_frozen_global_rows))
    excluded=fixed|witness
    o_rows=set(map(int,gr[gr>=0]))
    seeds=(bad & o_rows)-excluded
    seed_local=np.flatnonzero(np.isin(gr,np.fromiter(seeds,dtype=np.int64))) if seeds else np.empty(0,np.int64)
    incident=np.any(np.isin(faces,seed_local),axis=1) if len(seed_local) else np.zeros(len(faces),bool)
    incident_faces=np.flatnonzero(incident)
    ring_local=np.unique(faces[incident].reshape(-1)) if len(incident_faces) else np.empty(0,np.int64)
    ring=set(map(int,gr[ring_local]))
    free=(seeds|ring)-excluded
    info={"schema":"h706.ordinary-outer-o-local-quality-free-set.v1",
          "policy":"ORDINARY_OUTER_O_BAD_CORNER_PLUS_ONE_SOURCE_TRIANGLE_RING",
          "bad_global_rows":len(bad),"O_surface_global_rows":len(o_rows),
          "mutable_bad_O_corner_seeds":len(seeds),
          "incident_O_source_faces":len(incident_faces),
          "direct_O_triangle_ring_global_rows":len(ring),
          "free_global_rows":len(free),
          "fixed_rows_reexcluded":len((seeds|ring)&fixed),
          "witness_rows_reexcluded":len((seeds|ring)&witness),
          "protected_scope_rims_and_routes_fixed":True,
          "all_affected_tets_remain_in_global_quality_constraints":True}
    return np.asarray(sorted(free),np.int64),info



def _bad_face_components(tetrahedra,bad_cell_rows):
    """Connected components of bad tets via shared triangular faces only."""
    t=np.asarray(tetrahedra,np.int64);rows=np.asarray(bad_cell_rows,np.int64).reshape(-1)
    if not len(rows):return []
    if rows.min()<0 or rows.max()>=len(t):raise ValueError("bad-cell component row outside tet array")
    parent=np.arange(len(rows),dtype=np.int64);rank=np.zeros(len(rows),dtype=np.int8)
    def find(x):
        x=int(x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=int(parent[x])
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra==rb:return
        if rank[ra]<rank[rb]:ra,rb=rb,ra
        parent[rb]=ra
        if rank[ra]==rank[rb]:rank[ra]+=1
    face_owner={}
    omit=np.asarray([[1,2,3],[0,2,3],[0,1,3],[0,1,2]],np.int64)
    for local,cell in enumerate(rows):
        tet=t[int(cell)]
        for ix in omit:
            face=tuple(sorted(map(int,tet[ix])))
            previous=face_owner.get(face)
            if previous is None:face_owner[face]=local
            else:union(local,previous)
    groups={}
    for local,cell in enumerate(rows):groups.setdefault(find(local),[]).append(int(cell))
    out=[np.asarray(sorted(v),np.int64) for v in groups.values()]
    out.sort(key=lambda a:int(a[0]))
    return out

def _point_cell_incidence(tetrahedra,point_count):
    """Compact point-to-cell CSR built once for the bounded local batch."""
    t=np.asarray(tetrahedra,np.int64)
    flat=t.reshape(-1)
    if flat.size and (flat.min()<0 or flat.max()>=int(point_count)):
        raise ValueError("tet incidence point row outside point domain")
    counts=np.bincount(flat,minlength=int(point_count))
    offsets=np.empty(int(point_count)+1,np.int64);offsets[0]=0
    np.cumsum(counts,out=offsets[1:])
    order=np.argsort(flat,kind="stable")
    cellids=np.repeat(np.arange(len(t),dtype=np.int32),4)
    by_node=cellids[order]
    return offsets,by_node

def _incident_cells_by_rows(rows,offsets,by_node):
    rr=np.asarray(rows,np.int64).reshape(-1)
    if not len(rr):return np.empty(0,np.int64)
    parts=[by_node[int(offsets[int(r)]):int(offsets[int(r)+1])] for r in rr]
    parts=[x for x in parts if len(x)]
    if not parts:return np.empty(0,np.int64)
    return np.unique(np.concatenate(parts)).astype(np.int64,copy=False)

def _build_remainder_patches(all_bad_cell_rows,selected_bad_cell_rows,faces_by_role,
                              source_global_rows_by_role,tetrahedra,actual_back_fat_rows,
                              fixed_global_rows,witness_frozen_global_rows,point_count):
    t=np.asarray(tetrahedra,np.int64)
    selected=set(map(int,np.asarray(selected_bad_cell_rows,np.int64)))
    offsets,by_node=_point_cell_incidence(t,point_count)
    bad_components=_bad_face_components(t,all_bad_cell_rows)
    fixed=set(map(int,fixed_global_rows));witness=set(map(int,witness_frozen_global_rows));excluded=fixed|witness
    back=set(map(int,np.asarray(actual_back_fat_rows,np.int64)))
    all_surface=set();inner_used=set()
    faces={r:np.asarray(faces_by_role[r],np.int64) for r in ("O","I","D")}
    maps={r:np.asarray(source_global_rows_by_role[r],np.int64) for r in ("O","I","D")}
    for role in ("O","I","D"):
        all_surface.update(map(int,maps[role][maps[role]>=0]))
        if role in ("I","D"):
            active=np.unique(faces[role].reshape(-1))
            mapped=maps[role][active]
            if mapped.size and np.any(mapped<0):raise ValueError(role+" active face point is unmapped")
            inner_used.update(map(int,mapped))
    patches=[]
    for cluster_index,cluster_bad in enumerate(bad_components):
        seeds_cells=np.asarray([int(x) for x in cluster_bad if int(x) in selected],np.int64)
        if not len(seeds_cells):continue
        selected_corners=set(map(int,np.unique(t[seeds_cells].reshape(-1))))
        seed=(selected_corners&inner_used)-excluded
        ring=set();ring_receipt={}
        for role in ("I","D"):
            gr=maps[role];ff=faces[role]
            local_seed=np.flatnonzero(np.isin(gr,np.fromiter(seed,dtype=np.int64))) if seed else np.empty(0,np.int64)
            incident=np.any(np.isin(ff,local_seed),axis=1) if len(local_seed) else np.zeros(len(ff),bool)
            local=np.unique(ff[incident].reshape(-1)) if np.any(incident) else np.empty(0,np.int64)
            mapped=gr[local]
            if mapped.size and np.any(mapped<0):raise ValueError(role+" patch ring has unmapped point")
            ring.update(map(int,mapped))
            ring_receipt[role]={"seed_source_rows":int(len(local_seed)),
                "incident_source_faces":int(np.count_nonzero(incident)),
                "direct_ring_global_rows":int(len(np.unique(mapped)))}
        interior=(selected_corners-all_surface)-excluded
        star_cells=_incident_cells_by_rows(np.fromiter(selected_corners,dtype=np.int64),offsets,by_node)
        star_nodes=set(map(int,np.unique(t[star_cells].reshape(-1)))) if len(star_cells) else set()
        back_star=(back&star_nodes)-excluded
        free=(seed|ring|interior|back_star)-excluded
        free_rows=np.asarray(sorted(free),np.int64)
        affected=_incident_cells_by_rows(free_rows,offsets,by_node)
        if not len(free_rows) or not len(affected):
            continue
        if free_rows.size and np.any(free_rows<0):raise ValueError("patch has negative free point row")
        patches.append({"patch":int(len(patches)),"bad_component":int(cluster_index),
            "seed_bad_cells":seeds_cells,"bad_cluster_cells":cluster_bad,
            "free_rows":free_rows,"affected_cells":affected,
            "selected_bad_corners":int(len(selected_corners)),
            "inner_seed_rows":int(len(seed)),"direct_ring_rows":int(len(ring)),
            "interior_rows":int(len(interior)),"back_star_rows":int(len(back_star)),
            "star_cell_count":int(len(star_cells)),"ring_receipt":ring_receipt})
    patch_union=np.unique(np.concatenate([x["free_rows"] for x in patches])) if patches else np.empty(0,np.int64)
    roster={"schema":"h706.target-bad-face-component-local-patches.v1",
        "all_bad_cells":int(len(np.asarray(all_bad_cell_rows).reshape(-1))),
        "bad_face_components":int(len(bad_components)),"seed_components":int(len(patches)),
        "patches":[{"patch":x["patch"],"bad_component":x["bad_component"],
            "seed_bad_cells":int(len(x["seed_bad_cells"])),"bad_cluster_cells":int(len(x["bad_cluster_cells"])),
            "free_rows":int(len(x["free_rows"])),"affected_cells":int(len(x["affected_cells"])),
            "back_star_rows":x["back_star_rows"]} for x in patches],
        "patch_free_union_rows":int(len(patch_union)),
        "point_cell_incidence_built_once":True,
        "bad_component_connectivity":"shared triangular faces among native-bad tetrahedra",
        "all_cells_outside_each_patch_affected_tet_set_remain_unchanged":True}
    return patches,roster,offsets,by_node

def _build_ordinary_block_local_patches(all_bad_cell_rows, selected_bad_cell_rows,
                                         ordinary_free_rows, faces_by_role,
                                         source_global_rows_by_role, tetrahedra,
                                         point_count):
    """Assign ordinary bad-corner/source-ring rows to complete-star patches."""
    t=np.asarray(tetrahedra,np.int64)
    all_bad=np.asarray(all_bad_cell_rows,np.int64).reshape(-1)
    selected=np.asarray(selected_bad_cell_rows,np.int64).reshape(-1)
    ordinary=np.unique(np.asarray(ordinary_free_rows,np.int64).reshape(-1))
    allowed=set(map(int,ordinary))
    offsets,by_node=_point_cell_incidence(t,point_count)
    all_components=_bad_face_components(t,all_bad)
    selected_set=set(map(int,selected))
    faces={r:np.asarray(faces_by_role[r],np.int64) for r in ("O","I","D")}
    maps={r:np.asarray(source_global_rows_by_role[r],np.int64) for r in ("O","I","D")}
    patches=[]
    for component_id, component in enumerate(all_components):
        seed_cells=np.asarray([int(c) for c in component if int(c) in selected_set],np.int64)
        if not len(seed_cells):
            continue
        seed_rows=set(map(int,np.unique(t[seed_cells].reshape(-1))))
        candidate_rows=set(seed_rows)
        ring_by_role={}
        for role in ("O","I","D"):
            gr=maps[role];ff=faces[role]
            local_seed=np.flatnonzero(np.isin(gr,np.fromiter(seed_rows,dtype=np.int64))) if seed_rows else np.empty(0,np.int64)
            incident=np.any(np.isin(ff,local_seed),axis=1) if len(local_seed) else np.zeros(len(ff),bool)
            local=np.unique(ff[incident].reshape(-1)) if np.any(incident) else np.empty(0,np.int64)
            mapped=gr[local]
            if mapped.size and np.any(mapped<0):
                raise ValueError(role+" ordinary patch source ring has unmapped vertices")
            ring=set(map(int,mapped))
            candidate_rows.update(ring)
            ring_by_role[role]={"seed_source_rows":int(len(local_seed)),
                "incident_source_faces":int(np.count_nonzero(incident)),
                "direct_ring_rows":int(len(ring))}
        free=np.asarray(sorted(candidate_rows & allowed),np.int64)
        if not len(free):
            continue
        affected=_incident_cells_by_rows(free,offsets,by_node)
        if not len(affected):
            raise ValueError("ordinary patch free row has no incident cell")
        patches.append({"patch":len(patches),"bad_component":int(component_id),
            "seed_bad_cells":seed_cells,"bad_cluster_cells":component,
            "free_rows":free,"affected_cells":affected,"ring_by_role":ring_by_role})

    # Preserve bad-face components as independent sequential patches. Their free
    # rows and complete incident-cell stars may overlap; the existing solver
    # restarts each patch from the latest accepted points, rechecks all current
    # and immutable-good incident cells, and tracks each node's cumulative L1.
    # Merging on healthy one-ring overlap would create large torso-sized LPs.
    result=[]
    patches.sort(key=lambda x:(-len(x["seed_bad_cells"]),int(x["bad_component"])))
    for i,item in enumerate(patches):
        item["patch"]=i
        result.append(item)
    observed=np.unique(np.concatenate([x["free_rows"] for x in result])) if result else np.empty(0,np.int64)
    if not np.array_equal(observed,ordinary):
        missing=np.setdiff1d(ordinary,observed);extra=np.setdiff1d(observed,ordinary)
        raise AssertionError(f"ordinary patch roster differs from free mask: missing={len(missing)} extra={len(extra)}")
    selected_covered=set()
    for item in result:
        selected_covered.update(map(int,np.intersect1d(item["affected_cells"],selected)))
    if selected_covered!=selected_set:
        raise AssertionError(f"ordinary patch roster omitted selected bad cells: {len(selected_set-selected_covered)}")
    free_overlap=0;cell_overlap=0;seen_free=set();seen_cells=set()
    for item in result:
        rows=set(map(int,item["free_rows"]));cells=set(map(int,item["affected_cells"]))
        free_overlap+=len(rows&seen_free);cell_overlap+=len(cells&seen_cells)
        seen_free.update(rows);seen_cells.update(cells)
    free_occurrences=sum(len(x["free_rows"]) for x in result)
    affected_occurrences=sum(len(x["affected_cells"]) for x in result)
    free_overlap=free_occurrences-int(len(observed))
    unique_affected=np.unique(np.concatenate([x["affected_cells"] for x in result])) if result else np.empty(0,np.int64)
    affected_overlap=affected_occurrences-int(len(unique_affected))
    roster={"schema":"h706.ordinary-all-role-block-local-patches.v2",
        "all_bad_cells":int(len(all_bad)),"selected_bad_cells":int(len(selected)),
        "initial_bad_face_components":int(len(all_components)),
        "bad_face_component_patches":int(len(result)),
        "ordinary_free_rows":int(len(ordinary)),
        "overlap_counts":{"repeated_free_row_occurrences":int(free_overlap),
            "repeated_affected_cell_occurrences":int(affected_overlap)},
        "patches":[{"patch":int(x["patch"]),"bad_component":int(x["bad_component"]),
            "seed_bad_cells":int(len(x["seed_bad_cells"])),"free_rows":int(len(x["free_rows"])),
            "affected_cells":int(len(x["affected_cells"])),"ring_by_role":x["ring_by_role"]} for x in result],
        "full_point_incidence_stars":True,
        "overlapping_free_rows_allowed_and_sequentially_revalidated":True,
        "overlapping_affected_cells_allowed_and_sequentially_revalidated":True,
        "overlap_counts":{"repeated_free_row_occurrences":int(free_overlap),
            "repeated_affected_cell_occurrences":int(cell_overlap)},
        "all_selected_bad_cells_covered":True,"native_thresholds_unchanged":True}
    return result,roster,offsets,by_node


def _source_face_incidence(faces_by_role, source_global_rows_by_role, point_count):
    """Build one sparse global-point to source-face index per source role."""
    result={}
    for role in ("O","I","D"):
        faces=np.asarray(faces_by_role[role],np.int64)
        gr=np.asarray(source_global_rows_by_role[role],np.int64)
        mapped=gr[faces]
        if mapped.size and (np.any(mapped<0) or np.any(mapped>=point_count)):
            raise ValueError(role+": unified roster has an unmapped face vertex")
        flat=mapped.reshape(-1)
        face_rows=np.repeat(np.arange(len(faces),dtype=np.int64),3)
        order=np.argsort(flat,kind="stable")
        counts=np.bincount(flat,minlength=point_count) if len(flat) else np.zeros(point_count,np.int64)
        offsets=np.concatenate((np.zeros(1,np.int64),np.cumsum(counts,dtype=np.int64)))
        result[role]={"offsets":offsets,"face_rows":face_rows[order],"global_faces":mapped}
    return result

def _incident_source_faces(index, global_rows):
    rows=np.asarray(sorted(set(map(int,global_rows))),np.int64)
    if not len(rows):return np.empty(0,np.int64)
    off=index["offsets"];flat=index["face_rows"]
    pieces=[flat[int(off[r]):int(off[r+1])] for r in rows if int(off[r+1])>int(off[r])]
    return np.unique(np.concatenate(pieces)) if pieces else np.empty(0,np.int64)

def build_unified_free_sets(bad_cell_rows, faces_by_role, source_global_rows_by_role,
                            tetrahedra, actual_back_fat_rows, ordinary_fixed_rows,
                            inner_fixed_rows, witness_frozen_rows=()):
    """Union ordinary bad-corner/ring freedom with I/D-inner and actual-back stars for every bad cell."""
    t=np.asarray(tetrahedra,np.int64);cells=np.asarray(bad_cell_rows,np.int64).reshape(-1)
    if t.ndim!=2 or t.shape[1]!=4 or (len(cells) and (cells.min()<0 or cells.max()>=len(t))):
        raise ValueError("unified bad-cell roster is malformed")
    bad_corners=np.unique(t[cells].reshape(-1)) if len(cells) else np.empty(0,np.int64)
    ordinary,ordinary_info=expand_triangle_ring_once(
        bad_corners,faces_by_role,source_global_rows_by_role,
        fixed_global_rows=ordinary_fixed_rows,witness_frozen_global_rows=witness_frozen_rows)
    inner,inner_info=expand_inner_back_remainder_once(
        cells,faces_by_role,source_global_rows_by_role,t,actual_back_fat_rows,
        fixed_global_rows=inner_fixed_rows,witness_frozen_global_rows=witness_frozen_rows)
    union=np.union1d(ordinary,inner).astype(np.int64)
    inner_info={**inner_info,"policy":"I/D inner surface corner/ring and authenticated actual-back Fat points in the complete one-star of every baseline-bad cell"}
    return ordinary,inner,union,ordinary_info,inner_info

def _build_unified_block_local_patches(all_bad_cell_rows, ordinary_free_rows,
        inner_free_rows, faces_by_role, source_global_rows_by_role, tetrahedra,
        actual_back_fat_rows, fixed_global_rows, witness_frozen_global_rows, point_count):
    """Partition every bad-face component; combined freedoms share complete sequentially rechecked stars."""
    t=np.asarray(tetrahedra,np.int64);all_bad=np.asarray(all_bad_cell_rows,np.int64).reshape(-1)
    allowed=set(map(int,np.union1d(ordinary_free_rows,inner_free_rows)))
    fixed=set(map(int,fixed_global_rows));witness=set(map(int,witness_frozen_global_rows));excluded=fixed|witness
    back=set(map(int,np.asarray(actual_back_fat_rows,np.int64).reshape(-1)))
    faces={r:np.asarray(faces_by_role[r],np.int64) for r in ("O","I","D")}
    maps={r:np.asarray(source_global_rows_by_role[r],np.int64) for r in ("O","I","D")}
    all_surface=set();inner_used=set()
    for role in ("O","I","D"):
        all_surface.update(map(int,maps[role][maps[role]>=0]))
        if role in ("I","D"):
            active=np.unique(faces[role].reshape(-1));mapped=maps[role][active]
            if mapped.size and np.any(mapped<0):raise ValueError(role+": unified active face point is unmapped")
            inner_used.update(map(int,mapped))
    face_incidence=_source_face_incidence(faces,maps,point_count)
    cell_offsets,cell_by_node=_point_cell_incidence(t,point_count)
    components=_bad_face_components(t,all_bad)
    patches=[];component_receipts=[]
    for component_id,component in enumerate(components):
        seed_cells=np.asarray(component,np.int64)
        corners=set(map(int,np.unique(t[seed_cells].reshape(-1))))
        candidate=set(corners);all_ring=set();ring_counts={}
        for role in ("O","I","D"):
            incident=_incident_source_faces(face_incidence[role],corners)
            if len(incident):
                ring=set(map(int,np.unique(face_incidence[role]["global_faces"][incident].reshape(-1))))
                candidate.update(ring);all_ring.update(ring)
            else:ring=set()
            ring_counts[role]={"incident_faces":int(len(incident)),"ring_rows":int(len(ring))}
        inner_seed=(corners&inner_used)-excluded
        inner_ring=set();inner_ring_counts={}
        for role in ("I","D"):
            incident=_incident_source_faces(face_incidence[role],inner_seed)
            mapped=set(map(int,np.unique(face_incidence[role]["global_faces"][incident].reshape(-1)))) if len(incident) else set()
            inner_ring.update(mapped);candidate.update(mapped)
            role_rows=set(map(int,maps[role][maps[role]>=0]))
            inner_ring_counts[role]={"seed_rows":int(len(inner_seed&role_rows)),
                "incident_faces":int(len(incident)),"ring_rows":int(len(mapped))}
        interior=(corners-all_surface)-excluded
        corner_arr=np.asarray(sorted(corners),np.int64)
        star_cells=_incident_cells_by_rows(corner_arr,cell_offsets,cell_by_node)
        star_nodes=set(map(int,np.unique(t[star_cells].reshape(-1)))) if len(star_cells) else set()
        back_star=(back&star_nodes)-excluded
        candidate.update(inner_seed);candidate.update(interior);candidate.update(back_star)
        free_rows=np.asarray(sorted(candidate&allowed),np.int64)
        if not len(free_rows):
            component_receipts.append({"bad_component":int(component_id),"bad_cells":int(len(seed_cells)),
                "free_rows":0,"status":"BLOCKED_NO_ALLOWED_FREE_ROW"})
            continue
        affected=_incident_cells_by_rows(free_rows,cell_offsets,cell_by_node)
        if not len(affected):raise ValueError("unified free row has no complete tet star")
        patches.append({"patch":len(patches),"bad_component":int(component_id),
            "seed_bad_cells":seed_cells,"bad_cluster_cells":seed_cells,
            "free_rows":free_rows,"affected_cells":affected,
            "selected_bad_corners":int(len(corners)),"ordinary_ring_rows":int(len(all_ring)),
            "inner_seed_rows":int(len(inner_seed)),"inner_ring_rows":int(len(inner_ring)),
            "interior_rows":int(len(interior)),"back_star_rows":int(len(back_star)),
            "star_cell_count":int(len(star_cells)),"ring_by_role":ring_counts,
            "inner_ring_by_role":inner_ring_counts})
        component_receipts.append({"bad_component":int(component_id),"bad_cells":int(len(seed_cells)),
            "free_rows":int(len(free_rows)),"affected_cells":int(len(affected)),"status":"PATCHED"})
    observed=np.unique(np.concatenate([x["free_rows"] for x in patches])) if patches else np.empty(0,np.int64)
    expected=np.asarray(sorted(allowed),np.int64)
    if not np.array_equal(observed,expected):
        raise AssertionError("unified patch union differs from ordinary+inner mask: missing="+
            str(len(np.setdiff1d(expected,observed)))+" extra="+str(len(np.setdiff1d(observed,expected))))
    eligible=set()
    if len(all_bad) and len(expected):
        hit=np.any(np.isin(t[all_bad],expected),axis=1)
        eligible.update(map(int,all_bad[hit]))
    covered=set()
    for patch in patches:
        covered.update(map(int,np.intersect1d(patch["affected_cells"],all_bad)))
    if eligible!=covered:
        raise AssertionError("unified roster omitted a bad cell incident to a free point")
    patches.sort(key=lambda x:(-len(x["seed_bad_cells"]),int(x["bad_component"])))
    for i,p in enumerate(patches):p["patch"]=i
    occ_free=sum(len(x["free_rows"]) for x in patches);occ_cells=sum(len(x["affected_cells"]) for x in patches)
    unique_cells=np.unique(np.concatenate([x["affected_cells"] for x in patches])) if patches else np.empty(0,np.int64)
    roster={"schema":"h706.unified-inner-back-ordinary-block-local-patches.v1",
        "all_bad_cells":int(len(all_bad)),"bad_face_components":int(len(components)),
        "patched_components":int(len(patches)),"ordinary_free_rows":int(len(np.unique(ordinary_free_rows))),
        "inner_back_free_rows":int(len(np.unique(inner_free_rows))),"free_union_rows":int(len(expected)),
        "bad_cells_with_at_least_one_free_corner":int(len(eligible)),
        "bad_cells_without_a_free_corner":int(len(all_bad)-len(eligible)),
        "all_free_corner_bad_cells_covered_by_full_star":True,
        "overlapping_free_rows_allowed_and_sequentially_revalidated":True,
        "overlapping_affected_cells_allowed_and_sequentially_revalidated":True,
        "overlap_counts":{"repeated_free_row_occurrences":int(occ_free-len(expected)),
            "repeated_affected_cell_occurrences":int(occ_cells-len(unique_cells))},
        "bad_components":component_receipts,
        "patches":[{"patch":int(x["patch"]),"bad_component":int(x["bad_component"]),
            "seed_bad_cells":int(len(x["seed_bad_cells"])),"free_rows":int(len(x["free_rows"])),
            "affected_cells":int(len(x["affected_cells"])),"ordinary_ring_rows":x["ordinary_ring_rows"],
            "inner_seed_rows":x["inner_seed_rows"],"inner_ring_rows":x["inner_ring_rows"],
            "back_star_rows":x["back_star_rows"],"ring_by_role":x["ring_by_role"],
            "inner_ring_by_role":x["inner_ring_by_role"]} for x in patches],
        "point_cell_incidence_built_once":True,"source_face_incidence_built_once_per_role":True,
        "bad_component_connectivity":"shared triangular faces among baseline native-bad tetrahedra",
        "all_global_native_and_exact_gates_remain_required":True,"native_thresholds_unchanged":True}
    return patches,roster,cell_offsets,cell_by_node

def _global_log_deficit(q,d):
    return Q._log_deficit(np.asarray(q,dtype=np.float64)/1.e-5,
                          np.asarray(d,dtype=np.float64)/6.e-18)

def _cell_log_deficit(q,d):
    """Per-cell terms that sum exactly to the existing global native log-deficit."""
    qr=np.asarray(q,dtype=np.float64)/1.e-5
    vr=np.asarray(d,dtype=np.float64)/6.e-18
    if qr.shape!=vr.shape or not np.isfinite(qr).all() or not np.isfinite(vr).all():
        raise ValueError("nonfinite or mismatched per-cell quality score arrays")
    floor=np.finfo(np.float64).tiny
    return np.maximum(0.,-np.log(np.maximum(qr,floor)))+np.maximum(0.,-np.log(np.maximum(vr,floor)))

def _global_log_deficit_by_role(q,d,quality_group_ids):
    groups=np.asarray(quality_group_ids)
    if groups.shape!=np.asarray(q).shape:raise ValueError("quality group domain differs from score arrays")
    score=_cell_log_deficit(q,d)
    out={"Skin":float(score[groups==0].sum(dtype=np.float64)),"Fat":float(score[groups==1].sum(dtype=np.float64))}
    if np.any((groups!=0)&(groups!=1)):raise ValueError("unexpected quality group id")
    return out

def _schedule_unified_patches_fat_first(patches,roster,quality_group_ids,q,d):
    """Prioritize Fat bad-face patches under one shared deadline; keep joint stars intact."""
    groups=np.asarray(quality_group_ids);cell_score=_cell_log_deficit(q,d)
    if groups.ndim!=1 or groups.shape!=cell_score.shape:raise ValueError("Fat-first patch priority cell domain mismatch")
    if np.any((groups!=0)&(groups!=1)):raise ValueError("Fat-first scheduler requires Skin=0/Fat=1 groups")
    original={int(x["bad_component"]):dict(x) for x in roster.get("patches",[])}
    for patch in patches:
        cells=np.asarray(patch["seed_bad_cells"],np.int64)
        if np.any(cells<0) or np.any(cells>=len(groups)):raise ValueError("patch seed cell outside quality groups")
        fat=cells[groups[cells]==1];skin=cells[groups[cells]==0]
        patch["seed_bad_fat_cells"]=int(len(fat));patch["seed_bad_skin_cells"]=int(len(skin))
        patch["fat_seed_log_deficit"]=float(cell_score[fat].sum(dtype=np.float64))
        patch["skin_seed_log_deficit"]=float(cell_score[skin].sum(dtype=np.float64))
    patches.sort(key=lambda x:(-int(x["seed_bad_fat_cells"]),-float(x["fat_seed_log_deficit"]),
                               -int(x["seed_bad_skin_cells"]),int(x["bad_component"])))
    policy=("Fat-seeded patches first; descending Fat seed bad-cell count, then summed Fat per-cell native log deficit; ties use Skin seed count then stable component id. Mixed Skin/Fat patches remain joint and retain complete cross-material stars.")
    roster["schema"]="h706.unified-inner-back-ordinary-block-local-patches.v2-fat-first"
    roster["scheduler_policy"]=policy
    schedule=[];records=[]
    for rank,patch in enumerate(patches):
        patch["patch"]=rank;patch["fat_priority_rank"]=rank
        r=dict(original[int(patch["bad_component"])])
        r["patch"]=rank
        for key in ("seed_bad_fat_cells","seed_bad_skin_cells","fat_seed_log_deficit","skin_seed_log_deficit"):
            r[key]=patch[key]
        r["fat_priority_rank"]=rank;records.append(r)
        schedule.append({"rank":rank,"bad_component":int(patch["bad_component"]),
            "fat_seed_bad_cells":patch["seed_bad_fat_cells"],"fat_seed_log_deficit":patch["fat_seed_log_deficit"],
            "skin_seed_bad_cells":patch["seed_bad_skin_cells"]})
    roster["patches"]=records;roster["fat_priority_order"]=schedule
    roster["Fat_seeded_patches_scheduled_before_Skin_only_patches"]=True
    return patches,roster


def _single_node_quality_candidate(points,tetrahedra,node,cell_rows,remaining_cap_m,
                                   current_good,baseline_good,separation_rows,deadline,*,
                                   return_float64_candidate=False):
    """Dimensionless 3-variable SLSQP fit; strict Float32/native replay is authoritative."""
    from scipy.optimize import minimize
    p=np.asarray(points,np.float32);t=np.asarray(tetrahedra,np.int64);rows=np.asarray(cell_rows,np.int64)
    tloc=t[rows];node=int(node);cap=float(remaining_cap_m)
    if cap<=1.e-12 or not len(rows):return None,{"status":"NO_REMAINING_TRUST_OR_CELLS"}
    x0=p[node].astype(np.float64)
    d0,q0=Q._M.measurements(p,tloc)
    d0=np.asarray(d0,np.float64);q0=np.asarray(q0,np.float64)
    good=np.asarray(current_good[rows],bool)
    base=np.asarray(baseline_good[rows],bool)
    if not np.any(~good):return None,{"status":"NO_CURRENT_BAD_INCIDENT_CELL"}
    corners=np.argmax(tloc==node,axis=1)
    if np.any(~np.any(tloc==node,axis=1)):raise AssertionError("single-node incident roster lost its node")
    grad=Q.determinant_gradients(p,tloc)
    grad_node=np.asarray(grad[np.arange(len(rows)),corners],np.float64)
    xyz=p[tloc].astype(np.float64)
    edge_pairs=np.asarray([(i,j) for i in range(4) for j in range(i+1,4)],np.int64)
    edge0=np.linalg.norm(xyz[:,edge_pairs[:,0],:]-xyz[:,edge_pairs[:,1],:],axis=2)
    dynamic=np.zeros(edge0.shape,dtype=bool);other=np.zeros(edge0.shape,dtype=np.int64)
    for ei,(a,b) in enumerate(edge_pairs):
        ma=corners==a;mb=corners==b
        dynamic[:,ei]=ma|mb
        other[:,ei]=np.where(ma,b,a)
    rowix=np.arange(len(rows),dtype=np.int64)
    otherxyz=xyz[rowix[:,None],other]
    qratio0=q0/1.e-5;volratio0=d0/6.e-18
    qfloor=np.minimum(qratio0,1.001);vfloor=np.minimum(volratio0,1.001)
    positive_floor=np.full(len(rows),1.e-30,dtype=np.float64)
    longest0=np.max(edge0,axis=1)
    det_scale=np.maximum.reduce((np.abs(d0),1.e-5*np.maximum(longest0,1.e-30)**3,
                                 np.full(len(rows),6.e-18),np.full(len(rows),1.e-30)))
    bad=~good
    def checktime():
        if time.monotonic()>=deadline:raise TimeoutError("single-node patch budget exhausted")
    def ratio_terms(y):
        """Return native ratios and analytic derivatives w.r.t. normalized y=delta/cap."""
        checktime()
        y=np.asarray(y,dtype=np.float64)
        delta=cap*y
        x=x0+delta
        det=d0+np.einsum("ij,j->i",grad_node,delta)
        lens=edge0.copy()
        if np.any(dynamic):
            dynlen=np.linalg.norm(x[None,:]-otherxyz,axis=2)
            lens=np.where(dynamic,dynlen,lens)
        longest=np.max(lens,axis=1)
        safe_longest=np.maximum(longest,1.e-30)
        qden=1.e-5*safe_longest**3
        qr=det/qden
        vr=det/6.e-18
        det_grad_y=grad_node*cap
        # The max edge is locally smooth except at ties; argmax supplies a valid
        # active-edge derivative away from that measure-zero tie set.
        max_edge=np.argmax(lens,axis=1)
        is_dynamic=dynamic[rowix,max_edge]
        dlongest=np.zeros((len(rows),3),dtype=np.float64)
        chosen=np.flatnonzero(is_dynamic)
        if len(chosen):
            edge_vec=x[None,:]-otherxyz[chosen,max_edge[chosen]]
            dlongest[chosen]=cap*edge_vec/safe_longest[chosen,None]
        qgrad=det_grad_y/qden[:,None]-(3.*det/(1.e-5*safe_longest**4))[:,None]*dlongest
        return det,qr,vr,det_grad_y,qgrad
    def objective(z):
        det,qr,vr,_,_=ratio_terms(z[:3])
        return float(-np.sum(np.minimum(qr[bad],1.001)+np.minimum(vr[bad],1.001),dtype=np.float64))
    def objective_jac(z):
        _,qr,vr,det_grad_y,qgrad=ratio_terms(z[:3])
        active_q=bad&(qr<1.001)
        active_v=bad&(vr<1.001)
        grad_y=-np.sum(qgrad[active_q],axis=0)-np.sum(det_grad_y[active_v]/6.e-18,axis=0)
        return np.concatenate((grad_y,np.zeros(3,dtype=np.float64)))
    def ineq(z):
        det,qr,vr,_,_=ratio_terms(z[:3])
        y=z[:3];u=z[3:]
        return np.concatenate((u-y,u+y,np.asarray([1.-float(np.sum(u))]),
            det/det_scale-positive_floor/det_scale,qr[good]-qfloor[good],vr[good]-vfloor[good]))
    def ineq_jac(z):
        det,qr,vr,det_grad_y,qgrad=ratio_terms(z[:3])
        n=len(rows);ng=int(np.count_nonzero(good))
        J=np.zeros((7+n+2*ng,6),dtype=np.float64)
        I=np.eye(3,dtype=np.float64)
        J[0:3,0:3]=-I;J[0:3,3:6]=I
        J[3:6,0:3]=I;J[3:6,3:6]=I
        J[6,3:6]=-1.
        cursor=7
        J[cursor:cursor+n,0:3]=det_grad_y/det_scale[:,None]
        cursor+=n
        J[cursor:cursor+ng,0:3]=qgrad[good]
        cursor+=ng
        J[cursor:cursor+ng,0:3]=det_grad_y[good]/6.e-18
        return J
    z0=np.zeros(6,dtype=np.float64)
    bounds=[(-1.,1.)]*3+[(0.,1.)]*3
    result=minimize(objective,z0,method="SLSQP",jac=objective_jac,bounds=bounds,
        constraints=[{"type":"ineq","fun":ineq,"jac":ineq_jac}],
        options={"maxiter":30,"ftol":1.e-10,"disp":False})
    raw_x64=x0+cap*np.asarray(result.x[:3],dtype=np.float64)
    if not np.isfinite(raw_x64).all():return None,{"status":"NONFINITE_SLSQP"}
    if return_float64_candidate:
        raw_candidate=p.astype(np.float64).copy();raw_candidate[node]=raw_x64
        raw_step=float(np.abs(raw_x64-x0).sum(dtype=np.float64))
        if raw_step>cap+1.e-15:return None,{"status":"FLOAT64_NODE_STEP_EXCEEDS_REMAINING_CAP","step_l1_m":raw_step}
        return raw_candidate,{"status":"RAW_FLOAT64_NODE_CANDIDATE_FOR_F32_BACKTRACK",
            "optimizer_success":bool(result.success),"optimizer_message":str(result.message),
            "iterations":int(result.nit),"optimizer_delta_m":(cap*np.asarray(result.x[:3],np.float64)).tolist(),
            "float64_step_l1_m":raw_step,"variable_parameterization":"normalized_delta_over_cap",
            "constraint_row_scaling":"L1_over_cap_and_det_over_per_cell_native_scale"}
    x32=np.asarray(raw_x64,np.float32)
    if not np.isfinite(x32).all():return None,{"status":"NONFINITE_SLSQP"}
    candidate=p.copy();candidate[node]=x32
    # F32 rounding recovery: move the largest overshooting coordinate one ULP
    # at a time toward the immutable current point; trust cap remains exact.
    round_steps=0
    while round_steps<16:
        node_l1=float(np.abs(candidate[node].astype(np.float64)-p[node].astype(np.float64)).sum())
        if node_l1<=cap:break
        delta=np.abs(candidate[node].astype(np.float64)-p[node].astype(np.float64))
        j=int(np.argmax(delta))
        candidate[node,j]=np.nextafter(np.float32(candidate[node,j]),np.float32(p[node,j]))
        round_steps+=1
    node_l1=float(np.abs(candidate[node].astype(np.float64)-p[node].astype(np.float64)).sum())
    if node_l1>cap:return None,{"status":"F32_NODE_STEP_EXCEEDS_REMAINING_CAP","step_l1_m":node_l1,"rounding_steps":round_steps}
    moved=not np.array_equal(candidate[node].view(np.uint32),p[node].view(np.uint32))
    if not moved:return None,{"status":"NO_F32_NODE_MOVE","optimizer_success":bool(result.success),
        "optimizer_message":str(result.message),"iterations":int(result.nit),
        "optimizer_delta_m":(cap*np.asarray(result.x[:3],np.float64)).tolist(),
        "objective_before":float(objective(z0)),"objective_after":float(objective(result.x)),
        "variable_parameterization":"normalized_delta_over_cap","constraint_row_scaling":"L1_over_cap_and_det_over_per_cell_native_scale"}
    d,q=Q._M.measurements(candidate,tloc)
    g=np.asarray(Q._M.passes(d,q),bool)
    if np.any(np.asarray(d)<=0) or not np.isfinite(d).all() or not np.isfinite(q).all():
        return None,{"status":"F32_NONPOSITIVE_OR_NONFINITE"}
    if np.any(good&~g) or np.any(base&~g):
        return None,{"status":"F32_GOOD_CELL_REGRESSION"}
    exact=Q._exact_positive(candidate,tloc)
    if any(int(x)<=0 for x in exact):return None,{"status":"EXACT_NONPOSITIVE"}
    if Q._separation_violation_rows(candidate,separation_rows):
        return None,{"status":"SEPARATION_WITNESS_REGRESSION"}
    l1=float(np.abs(candidate[node].astype(np.float64)-p[node].astype(np.float64)).sum())
    if l1>cap+1.e-15:return None,{"status":"F32_NODE_STEP_EXCEEDS_REMAINING_CAP","step_l1_m":l1}
    return candidate,{"status":"F32_NODE_CANDIDATE","optimizer_success":bool(result.success),
        "optimizer_message":str(result.message),"iterations":int(result.nit),
        "step_l1_m":l1,"f32_cap_rounding_steps":int(round_steps),"bad_before":int(np.count_nonzero(~good)),
        "bad_after":int(np.count_nonzero(~g)),"objective_before":float(objective(z0)),
        "objective_after":float(objective(result.x)),
        "variable_parameterization":"normalized_delta_over_cap",
        "constraint_row_scaling":"L1_over_cap_and_det_over_per_cell_native_scale",
        "determinant_row_scale_m3":{"minimum":float(det_scale.min()),"maximum":float(det_scale.max())}}

def _backtracking_replay_packet(current,candidate64,tetrahedra,cells,free_rows,caps,
                                  current_good,baseline_good,baseline_points,alpha,
                                  separation_rows,progress_reference):
    t=np.asarray(tetrahedra,np.int64);cells=np.asarray(cells,np.int64)
    free=np.asarray(free_rows,np.int64);local_t_global=t[cells]
    sep_records=[];extra=[]
    if separation_rows is not None:
        sepobj=Q._normalise_separation_rows(separation_rows);selected=set()
        for n in free:selected.update(sepobj["row_indices_by_point"].get(int(n),()))
        for ri in sorted(selected):
            row=sepobj["constraints"][int(ri)];nodes=np.asarray(row["point_rows"],np.int64)
            extra.extend(map(int,nodes))
            sep_records.append((nodes,np.asarray(row["vectors"],np.float64),
                np.asarray(row["origin"],np.float64),float(row["upper_bound"])))
    point_rows=np.unique(np.r_[local_t_global.reshape(-1),np.asarray(extra,np.int64)])
    lut=np.full(len(current),-1,np.int64);lut[point_rows]=np.arange(len(point_rows),dtype=np.int64)
    local_t=lut[local_t_global]
    free_local=lut[free]
    local_sep=[]
    for nodes,vectors,origin,upper in sep_records:
        local_sep.append({"point_rows":lut[nodes].astype(int).tolist(),"vectors":vectors.tolist(),
            "origin":origin.tolist(),"upper_bound":upper})
    return {"schema":"h706.v26b4.backtrack-independent-replay.v1",
        "global_cell_rows":cells.astype(int).tolist(),"global_point_rows":point_rows.astype(int).tolist(),
        "current_points_f32_bits":np.asarray(current[point_rows],np.float32).view(np.uint32).astype(np.uint32).tolist(),
        "baseline_points_f32_bits":np.asarray(baseline_points[point_rows],np.float32).view(np.uint32).astype(np.uint32).tolist(),
        "candidate64_points_m":np.asarray(candidate64[point_rows],np.float64).tolist(),
        "accepted_points_f32_bits":np.asarray(current[point_rows],np.float32).view(np.uint32).astype(np.uint32).tolist(),
        "tetrahedra_local":local_t.astype(int).tolist(),"free_local_rows":free_local.astype(int).tolist(),
        "l1_caps_m":np.asarray(caps,np.float64).tolist(),
        "current_passed":np.asarray(current_good[cells],bool).tolist(),
        "baseline_passed":np.asarray(baseline_good[cells],bool).tolist(),
        "separation_halfspaces":local_sep,"accepted_alpha":float(alpha),
        "progress_reference":progress_reference}

def _backtrack_f32_candidate(current,candidate64,tetrahedra,cells,free_rows,caps,
                             current_det,current_quality,current_good,baseline_good,
                             baseline_points,separation_rows,global_bad,global_score,optimizer_status):
    t=np.asarray(tetrahedra,np.int64);cells=np.asarray(cells,np.int64);free=np.asarray(free_rows,np.int64)
    before_bad_local=int(np.count_nonzero(~np.asarray(current_good,bool)[cells]))
    before_score_local=_global_log_deficit(np.asarray(current_quality)[cells],np.asarray(current_det)[cells])
    reference={"global_bad_before":int(global_bad),"global_score_before":float(global_score),
        "local_bad_before":before_bad_local,"local_score_before":before_score_local}
    def progress(trial,d,q,alpha):
        good=np.asarray(Q._M.passes(d,q),bool)
        global_bad_after=int(global_bad-before_bad_local+np.count_nonzero(~good))
        local_score_after=_global_log_deficit(q,d)
        global_score_after=float(global_score-before_score_local+local_score_after)
        tolerance=max(1.e-8,1.e-6*float(global_score))
        count_gain=global_bad_after<int(global_bad)
        score_gain=(global_bad_after==int(global_bad) and float(global_score)-global_score_after>tolerance)
        return {"accepted":bool(count_gain or score_gain),"global_bad_before":int(global_bad),
            "global_bad_candidate":global_bad_after,"global_score_before":float(global_score),
            "global_score_candidate":global_score_after,"local_bad_before":before_bad_local,
            "local_bad_candidate":int(np.count_nonzero(~good)),"local_score_before":before_score_local,
            "local_score_candidate":local_score_after,"tolerance":tolerance,
            "count_gain":bool(count_gain),"score_gain":bool(score_gain)}
    def separation(points):
        return Q._separation_violation_rows(points,separation_rows,point_filter=free)
    candidate,receipt=F32_BACKTRACK.backtrack_f32_candidate(
        np.asarray(current,np.float32),np.asarray(candidate64,np.float64),t[cells],free,np.asarray(caps,np.float64),
        np.asarray(current_good,bool)[cells],np.asarray(baseline_good,bool)[cells],
        lambda pts,tt:Q._M.measurements(pts,tt),lambda d,q:Q._M.passes(d,q),progress=progress,
        exact_positive=lambda pts,tt:Q._exact_positive(pts,tt),separation_violations=separation,
        immutable_baseline_points_f32=np.asarray(baseline_points,np.float32),optimizer_status=optimizer_status,
        q_min=1.e-5,volume_min_m3=1.e-18,max_halvings=6)
    if candidate is None:
        first=receipt.get("first_f32_good_regression")
        if isinstance(first,dict) and first.get("candidate64_passed_same_native_predicate_on_regressed_rows",False):
            alpha=float(first.get("alpha",0.))
            if 0.<alpha<=1.:
                trial64=np.asarray(current,np.float32).astype(np.float64)
                trial64[free]=np.asarray(current,np.float32)[free].astype(np.float64)+alpha*(
                    np.asarray(candidate64,np.float64)[free]-np.asarray(current,np.float32)[free].astype(np.float64))
                trial32=np.asarray(trial64,np.float32)
                witness=first.get("regressed_good_cell_witness",{})
                witness_rows=sorted(set(int(n) for r in witness.get("rows",[]) for n in r.get("tet_point_rows",[])))
                repaired,ulp_receipt=ULP.search_witness_ulp_correction(
                    np.asarray(current,np.float32),trial32,t[cells],free,np.asarray(caps,np.float64),
                    np.asarray(current_good,bool)[cells],np.asarray(baseline_good,bool)[cells],
                    lambda pts,tt:Q._M.measurements(pts,tt),lambda d,q:Q._M.passes(d,q),
                    exact_positive=lambda pts,tt:Q._exact_positive(pts,tt),
                    separation_violations=separation,witness_point_rows=witness_rows,alpha=alpha,
                    progress=progress,max_changed_coordinates=4,max_states=256,
                    max_ulp_steps_per_coordinate=1)
                receipt["ulp_quantization"]=ulp_receipt
                if repaired is not None:
                    candidate=repaired
                    receipt["status"]="PASS_BOUNDED_F32_ULP_CORRECTED_PROPOSAL"
                    receipt["alpha"]=alpha
                    receipt["maximum_l1_step_m"]=float(np.abs(candidate[free].astype(np.float64)-
                        np.asarray(current,np.float32)[free].astype(np.float64)).sum(axis=1).max(initial=0.))
    if candidate is not None:
        receipt["independent_replay"]=_backtracking_replay_packet(
            current,candidate64,t,cells,free,caps,current_good,baseline_good,baseline_points,
            receipt["alpha"],separation_rows,reference)
        receipt["independent_replay"]["accepted_points_f32_bits"]=(
            np.asarray(candidate[receipt["independent_replay"]["global_point_rows"]],np.float32)
            .view(np.uint32).astype(np.uint32).tolist())
    return candidate,receipt

def _joint_seed_candidates_v26c(seed_bad_cells,patch_free,points,tetrahedra,current_good,current_det,current_quality,path_l1,cap,offsets,by_node,max_free_nodes=16):
    """Return the best single-seed and one expanded authorized vertex-star candidate.

    Expanded rows are a bounded subset of the complete one-ring of a current bad
    seed tetrahedron, intersected with the patch's preauthorized free rows. The
    selected rows remain connected to the seed through tetrahedron incidence.
    """
    t=np.asarray(tetrahedra,np.int64);p=np.asarray(points,np.float32);good=np.asarray(current_good,bool)
    q=np.asarray(current_quality,np.float64);d=np.asarray(current_det,np.float64);paths=np.asarray(path_l1,np.float64)
    patch_rows=set(map(int,np.asarray(patch_free,np.int64)))
    singles=[];expanded=[]
    for cell0 in sorted(set(map(int,np.asarray(seed_bad_cells,np.int64).reshape(-1)))):
        cell=int(cell0)
        if cell<0 or cell>=len(t) or good[cell]:continue
        seed_vertices=np.unique(t[cell]).astype(np.int64)
        seed_free=np.asarray([int(v) for v in seed_vertices if int(v) in patch_rows and cap-paths[int(v)]>1.e-12],np.int64)
        if not 2<=len(seed_free)<=4:continue
        deficit=_global_log_deficit(q[np.asarray([cell])],d[np.asarray([cell])])
        singles.append((-deficit,(cell,),seed_free,"seed-corners"))
        seed_star=np.asarray(_incident_cells_by_rows(seed_vertices,offsets,by_node),np.int64)
        if not len(seed_star):continue
        near_nodes=np.unique(t[seed_star].reshape(-1))
        seed_set=set(map(int,seed_vertices))
        eligible=[int(v) for v in near_nodes if int(v) not in seed_set
                  and int(v) in patch_rows and cap-paths[int(v)]>1.e-12]
        if len(eligible)<4:continue
        seed_center=p[seed_vertices].astype(np.float64).mean(axis=0)
        ranked=[]
        for node in eligible:
            node_cells=np.asarray(_incident_cells_by_rows(np.asarray([node],np.int64),offsets,by_node),np.int64)
            bad_incident=int(np.count_nonzero(~good[node_cells]))
            distance=float(np.linalg.norm(p[node].astype(np.float64)-seed_center))
            ranked.append((-bad_incident,distance,node))
        ranked.sort()
        extras=[int(x[2]) for x in ranked[:max(0,int(max_free_nodes)-len(seed_free))]]
        free=np.asarray(sorted(set(map(int,seed_free))|set(extras)),np.int64)
        if 8<=len(free)<=int(max_free_nodes):expanded.append((-deficit,(cell,),free,"expanded-seed-vertex-star"))
    singles.sort(key=lambda x:(x[0],x[1]))
    expanded.sort(key=lambda x:(x[0],len(x[2]),x[1]))
    if expanded and singles:return [singles[0],expanded[0]]
    if expanded:return [expanded[0]]
    return singles[:2]

def _witness_directed_expansion_v26e(seed_cells,seed_free,previous_f32_report,
        old_expanded_rows,patch_rows,points,tetrahedra,current_good,path_l1,cap,
        offsets,by_node,max_free_nodes=16):
    """Replace arbitrary nearest extras with authorized vertices from actual F32 regression witnesses."""
    t=np.asarray(tetrahedra,np.int64);p=np.asarray(points,np.float32);good=np.asarray(current_good,bool)
    patch=set(map(int,np.asarray(patch_rows,np.int64)));seed_cells=tuple(sorted(set(map(int,seed_cells))))
    seed_free=set(map(int,np.asarray(seed_free,np.int64)))
    if not 2<=len(seed_free)<=4:return None,{"status":"SKIP_INVALID_SEED_FREE_CORNERS"}
    back=previous_f32_report.get("f32_backtracking",{}) if isinstance(previous_f32_report,dict) else {}
    first=back.get("first_f32_good_regression",{}) if isinstance(back,dict) else {}
    witness=first.get("regressed_good_cell_witness",{}) if isinstance(first,dict) else {}
    rows=witness.get("rows",[]) if isinstance(witness,dict) else []
    if not rows or not bool(first.get("candidate64_passed_same_native_predicate_on_regressed_rows",False)):
        return None,{"status":"NO_RAW64_PASSING_GOOD_CELL_WITNESS"}
    seed_vertices=set(map(int,np.asarray(t[list(seed_cells)],np.int64).reshape(-1)))
    seed_star=np.asarray(_incident_cells_by_rows(np.asarray(sorted(seed_vertices),np.int64),offsets,by_node),np.int64)
    star_vertices=set(map(int,np.unique(t[seed_star]).tolist())) if len(seed_star) else set()
    remaining=float(cap)-np.asarray(path_l1,np.float64)
    support={}
    for ri,row in enumerate(rows):
        tet_nodes=set(map(int,row.get("tet_point_rows",[])))
        candidate_nodes=tet_nodes & patch & star_vertices
        for node in candidate_nodes:
            if node in seed_free or node<0 or node>=len(remaining) or remaining[node]<=1.e-12:continue
            ent=support.setdefault(int(node),{"witness_rows":[],"min_candidate32_q_margin":float("inf"),
                "min_candidate64_q_margin":float("inf")})
            ent["witness_rows"].append(int(ri))
            ent["min_candidate32_q_margin"]=min(ent["min_candidate32_q_margin"],
                float(row.get("candidate32_q_margin",row.get("float32_minus_float64",{}).get("normalized_det",0.))))
            ent["min_candidate64_q_margin"]=min(ent["min_candidate64_q_margin"],
                float(row.get("candidate64_q_margin",0.)))
    slots=max(0,int(max_free_nodes)-len(seed_free))
    ranked=sorted(support,key=lambda n:(-len(support[n]["witness_rows"]),
        support[n]["min_candidate32_q_margin"],-support[n]["min_candidate64_q_margin"],n))
    chosen=ranked[:slots];selected=set(seed_free)|set(chosen)
    old_extras=[int(x) for x in np.asarray(old_expanded_rows,np.int64) if int(x) not in selected
        and int(x) in patch and remaining[int(x)]>1.e-12 and int(x) in star_vertices]
    old_extras=sorted(set(old_extras))
    for n in old_extras:
        if len(selected)>=8:break
        selected.add(n)
    if len(selected)<8:
        seed_center=p[np.asarray(sorted(seed_vertices),np.int64)].astype(np.float64).mean(axis=0)
        fallback=[int(n) for n in star_vertices if int(n) in patch and int(n) not in selected and remaining[int(n)]>1.e-12]
        fallback.sort(key=lambda n:(float(np.linalg.norm(p[n].astype(np.float64)-seed_center)),n))
        for n in fallback:
            if len(selected)>=8:break
            selected.add(n)
    if not 8<=len(selected)<=int(max_free_nodes):
        return None,{"status":"SKIP_WITNESS_EXPANSION_OUTSIDE_8_TO_16","seed_free_rows":sorted(seed_free),
            "witness_candidates":ranked,"selected_rows":sorted(selected)}
    free=np.asarray(sorted(selected),np.int64)
    return free,{"status":"WITNESS_DIRECTED_EXPANSION_READY",
        "selection_policy":"retain 2-4 seed corners; prioritize authorized complete-seed-star free rows incident to actual raw-F64-pass/F32-regressed good-cell witnesses; fill nearest extras only to reach eight; cap sixteen",
        "seed_bad_cells":list(seed_cells),"seed_free_rows":sorted(seed_free),
        "witness_local_cell_rows":[int(r.get("local_cell_row",-1)) for r in rows],
        "witness_point_rows":sorted(set(int(n) for r in rows for n in r.get("tet_point_rows",[]) if int(n) in support)),
        "witness_candidate_rows":ranked,"selected_witness_extra_rows":chosen,
        "selected_rows":free.tolist(),"replaced_nearest_rows":sorted(set(map(int,old_expanded_rows))-set(map(int,free))),
        "support_by_selected_row":{str(n):support[n] for n in chosen if n in support},
        "raw64_witness_predicate_passed":True}

def propose_remainder_block_local(points,tetrahedra,material_indices,quality_group_ids,
                                  baseline_det,baseline_quality,baseline_passed,
                                  patches,trust_cap_m,seconds,patch_seconds,
                                  offsets,by_node,separation_rows=None,
                                  objective_mode="role_equal_nearpass",
                                  node_fallbacks_per_patch=8, ordinary_mode=False,
                                  joint_small_patch_fallback=False,
                                  joint_fallbacks_per_patch=2,
                                  joint_patch_seconds=.95,
                                  joint_maxiter=25,
                                  extended_250um_construction_trial=False):
    p0=np.asarray(points,np.float32);t=np.asarray(tetrahedra,np.int64)
    mat=np.asarray(material_indices);groups=np.asarray(quality_group_ids)
    cap=float(trust_cap_m);budget=float(seconds);patch_budget=float(patch_seconds)
    start=time.monotonic();deadline=start+budget
    extended=bool(extended_250um_construction_trial)
    if extended and ordinary_mode:
        raise ValueError("extended 250um construction cap cannot use ordinary 500um mode")
    if extended:
        if not 5.e-5 < cap <= 2.5e-4:
            raise ValueError("explicit 250um construction solver requires a cap above 50um and at most 250um")
        cap_limit=2.5e-4
    else:
        cap_limit=5.e-4 if ordinary_mode else 5.e-5
    if cap<=0 or cap>cap_limit or budget<=0 or budget>30 or patch_budget<=0:
        raise ValueError("invalid bounded block-local trust/time budget")
    if not 0<=int(joint_fallbacks_per_patch)<=2 or not 0< float(joint_patch_seconds)<=1.0 or not 1<=int(joint_maxiter)<=40:
        raise ValueError("invalid bounded joint SLSQP fallback budget")
    base_d=np.asarray(baseline_det,dtype=np.float64);base_q=np.asarray(baseline_quality,dtype=np.float64)
    base_good=np.asarray(baseline_passed,bool)
    if base_d.shape!=(len(t),) or base_q.shape!=(len(t),) or base_good.shape!=(len(t),):
        raise ValueError("block-local immutable baseline metric shape mismatch")
    current=p0.copy();cur_d=base_d.copy();cur_q=base_q.copy();cur_good=base_good.copy()
    cur_bad=int(np.count_nonzero(~cur_good));base_bad=cur_bad
    cur_score=_global_log_deficit(cur_q,cur_d);base_score=cur_score
    base_role_score=_global_log_deficit_by_role(cur_q,cur_d,groups)
    if abs(sum(base_role_score.values())-base_score)>max(1.e-8,1.e-10*max(1.,abs(base_score))):
        raise AssertionError("per-role log-deficit scores do not sum to unchanged global score")
    path_l1=np.zeros(len(p0),dtype=np.float64)
    reports=[];accepted=0;accepted_nodes=0;accepted_joint=0;attempted=0;skipped=0
    touched=set()
    for patch in patches:
        if time.monotonic()>=deadline:break
        patch_start=time.monotonic()
        patch_deadline=min(deadline,patch_start+patch_budget)
        patch_rows=np.asarray(patch["free_rows"],np.int64)
        blocked=set()
        patch_accepted=False
        local_summary={"patch":int(patch["patch"]),"bad_component":int(patch["bad_component"]),
            "seed_bad_cells":int(len(patch["seed_bad_cells"])),"bad_cluster_cells":int(len(patch["bad_cluster_cells"])),
            "free_rows":int(len(patch_rows)),"affected_cells_initial":int(len(patch["affected_cells"])),
            "native_bad_count_before_patch":int(cur_bad)}
        for priority_key in ("fat_priority_rank","seed_bad_fat_cells","seed_bad_skin_cells","fat_seed_log_deficit","skin_seed_log_deficit"):
            if priority_key in patch:local_summary[priority_key]=patch[priority_key]
        # Reserve a small slice of the SAME patch deadline only where a seed
        # bad tet has 2-4 currently movable corners. The nonlinear fallback is
        # reached only after the block LP and every one-node attempt stall.
        joint_seed_candidates=[]
        joint_reserve=0.0
        if joint_small_patch_fallback and joint_fallbacks_per_patch>0:
            joint_seed_candidates=_joint_seed_candidates_v26c(
                patch["seed_bad_cells"],patch_rows,current,t,cur_good,cur_d,cur_q,path_l1,cap,
                offsets,by_node,max_free_nodes=16)
            joint_seed_candidates=joint_seed_candidates[:int(joint_fallbacks_per_patch)]
            if joint_seed_candidates:
                joint_reserve=min(float(joint_patch_seconds),patch_budget*.65,max(0.,patch_deadline-time.monotonic()))
        primary_deadline=max(time.monotonic(),patch_deadline-joint_reserve)
        lp_last_status=None
        lp_attempted=False
        for retry in range(4):
            remain=primary_deadline-time.monotonic()
            if remain<=.04:break
            active_rows=np.asarray([int(r) for r in patch_rows if int(r) not in blocked and cap-path_l1[int(r)]>1.e-12],np.int64)
            if not len(active_rows):break
            local_cells=_incident_cells_by_rows(active_rows,offsets,by_node)
            if not len(local_cells):break
            free_mask=np.zeros(len(p0),dtype=bool);free_mask[active_rows]=True
            remaining=float(np.min(cap-path_l1[active_rows]))
            if remaining<=1.e-12:break
            call_cap=min(cap,remaining)
            call_budget=min(patch_budget,remain)
            try:
                lp_attempted=True
                proposal,diag=Q.propose_quality_repair(
                    current,t[local_cells],mat[local_cells],free_mask,call_cap,
                    time_limit_s=call_budget,recovery_time_limit_s=call_budget,
                    linear_margin=1.e-3,max_backtracks=12,max_inner_iterations=10,
                    separation_rows=separation_rows,objective_mode=objective_mode,
                    quality_group_ids=groups[local_cells])
            except Exception as exc:
                local_summary.update({"status":"PATCH_SOLVER_EXCEPTION","exception":repr(exc),"retry":retry})
                lp_last_status="PATCH_SOLVER_EXCEPTION"
                break
            lp_last_status=str(diag.get("status",diag.get("solver_status","UNKNOWN")))
            if not np.array_equal(proposal[~free_mask].view(np.uint32),current[~free_mask].view(np.uint32)):
                raise AssertionError("local SLP moved a point outside this patch free set")
            moved=np.flatnonzero(np.any(proposal[active_rows].view(np.uint32)!=current[active_rows].view(np.uint32),axis=1))
            moved_rows=active_rows[moved]
            if len(moved_rows):
                step_l1=np.abs(proposal[moved_rows].astype(np.float64)-current[moved_rows].astype(np.float64)).sum(axis=1)
                offenders=moved_rows[path_l1[moved_rows]+step_l1>cap+1.e-15]
            else:
                step_l1=np.empty(0,np.float64);offenders=np.empty(0,np.int64)
            if len(offenders):
                blocked.update(map(int,offenders))
                local_summary.update({"status":"RETRY_EXCLUDING_CUMULATIVE_TRUST_ROWS",
                    "retry":retry,"newly_blocked_rows":len(offenders),
                    "remaining_patch_rows":int(len(active_rows)-len(offenders))})
                continue
            proposed_cells=local_cells
            new_d,new_q=Q._M.measurements(proposal,t[proposed_cells])
            new_good=np.asarray(Q._M.passes(new_d,new_q),bool)
            if np.any(~np.isfinite(new_d)) or np.any(~np.isfinite(new_q)) or np.any(new_d<=0):
                local_summary.update({"status":"REJECT_NONPOSITIVE_OR_NONFINITE","retry":retry})
                break
            if np.any(cur_good[proposed_cells]&~new_good) or np.any(base_good[proposed_cells]&~new_good):
                local_summary.update({"status":"REJECT_CURRENT_OR_IMMUTABLE_GOOD_REGRESSION","retry":retry,
                    "current_good_regressions":int(np.count_nonzero(cur_good[proposed_cells]&~new_good)),
                    "immutable_good_regressions":int(np.count_nonzero(base_good[proposed_cells]&~new_good))})
                break
            if hasattr(Q,"_exact_positive"):
                exact=Q._exact_positive(proposal,t[proposed_cells])
                if any(float(x)<=0 for x in exact):
                    local_summary.update({"status":"REJECT_EXACT_NONPOSITIVE","retry":retry})
                    break
            before_bad_local=int(np.count_nonzero(~cur_good[proposed_cells]))
            after_bad_local=int(np.count_nonzero(~new_good))
            new_global_bad=cur_bad-before_bad_local+after_bad_local
            before_score_local=_global_log_deficit(cur_q[proposed_cells],cur_d[proposed_cells])
            after_score_local=_global_log_deficit(new_q,new_d)
            new_global_score=cur_score-before_score_local+after_score_local
            tolerance=max(1.e-8,1.e-6*cur_score)
            count_progress=new_global_bad<cur_bad
            score_progress=(new_global_bad==cur_bad and cur_score-new_global_score>tolerance)
            if not (count_progress or score_progress):
                local_summary.update({"status":"REJECT_NO_GLOBAL_LEX_PROGRESS","retry":retry,
                    "local_bad_before":before_bad_local,"local_bad_after":after_bad_local,
                    "global_bad_before":cur_bad,"global_bad_candidate":new_global_bad,
                    "global_score_before":cur_score,"global_score_candidate":new_global_score,
                    "solver_status":diag.get("solver_status")})
                break
            current=proposal.copy()
            cur_d[proposed_cells]=new_d;cur_q[proposed_cells]=new_q;cur_good[proposed_cells]=new_good
            cur_bad=new_global_bad;cur_score=new_global_score
            if len(moved_rows):path_l1[moved_rows]+=step_l1
            touched.update(map(int,proposed_cells))
            patch_accepted=True;accepted+=1
            local_summary.update({"status":"ACCEPT_COUNT_REDUCTION" if count_progress else "ACCEPT_TIED_SCORE_REDUCTION",
                "retry":retry,"local_bad_before":before_bad_local,"local_bad_after":after_bad_local,
                "global_bad_before":int(cur_bad+(before_bad_local-after_bad_local)),
                "global_bad_after":int(cur_bad),"global_score_after":cur_score,
                "solver_status":diag.get("solver_status"),"solver_method":diag.get("solver_method"),
                "solver_elapsed_s":diag.get("elapsed_s"),
                "moved_rows":int(len(moved_rows)),
                "maximum_cumulative_path_L1_m":float(path_l1.max(initial=0.)),
                "max_step_L1_m":float(step_l1.max(initial=0.)),
                "local_native_bad_after":after_bad_local})
            break
        lp_accepted=bool(patch_accepted)
        local_summary["native_bad_count_after_block_lp"]=int(cur_bad)
        local_summary["block_lp_retry"]={"attempted":bool(lp_attempted),"last_status":lp_last_status,
            "accepted":lp_accepted,"native_bad_count_before":int(local_summary["native_bad_count_before_patch"]),
            "native_bad_count_after":int(cur_bad),
            "remaining_same_patch_deadline_s":max(0.,patch_deadline-time.monotonic())}
        # If a coupled patch LP is blocked by a small set of cells, try a
        # sequence of three-coordinate local optimizations. Each one-node
        # feasible set keeps every currently good incident tet above its native
        # predicate and every incident determinant positive.
        node_reports=[]
        if time.monotonic()<primary_deadline and node_fallbacks_per_patch>0:
            node_priority=[]
            for node in np.asarray(patch["free_rows"],np.int64):
                node=int(node);remaining_node=cap-path_l1[node]
                if remaining_node<=1.e-12:continue
                node_cells=_incident_cells_by_rows(np.asarray([node],np.int64),offsets,by_node)
                if not len(node_cells):continue
                bad_here=int(np.count_nonzero(~cur_good[node_cells]))
                if not bad_here:continue
                score_here=_global_log_deficit(cur_q[node_cells],cur_d[node_cells])
                node_priority.append((-bad_here,-score_here,node))
            node_priority.sort()
            for _,__,node in node_priority[:int(node_fallbacks_per_patch)]:
                if time.monotonic()>=primary_deadline:break
                node_cells=_incident_cells_by_rows(np.asarray([node],np.int64),offsets,by_node)
                if not len(node_cells) or not np.any(~cur_good[node_cells]):continue
                remaining_node=float(cap-path_l1[node])
                try:
                    node_candidate,node_diag=_single_node_quality_candidate(
                        current,t,node,node_cells,remaining_node,cur_good,base_good,separation_rows,primary_deadline,
                        return_float64_candidate=True)
                    if node_candidate is not None and node_diag.get("status")=="RAW_FLOAT64_NODE_CANDIDATE_FOR_F32_BACKTRACK":
                        raw64=np.asarray(node_candidate,np.float64)
                        node_candidate,backtrack_diag=_backtrack_f32_candidate(
                            current,raw64,t,node_cells,np.asarray([node],np.int64),np.asarray([remaining_node],np.float64),
                            cur_d,cur_q,cur_good,base_good,p0,separation_rows,cur_bad,cur_score,node_diag)
                        node_diag={**node_diag,"optimizer_candidate_status":node_diag["status"],
                            "status":"F32_BACKTRACKED_CANDIDATE" if node_candidate is not None else "NO_ACCEPTED_F32_BACKTRACK_ALPHA",
                            "f32_backtracking":backtrack_diag}
                except TimeoutError:
                    node_reports.append({"node":int(node),"status":"NODE_FALLBACK_TIME_LIMIT"})
                    break
                except Exception as exc:
                    node_reports.append({"node":int(node),"status":"NODE_FALLBACK_EXCEPTION","exception":repr(exc)})
                    continue
                if node_candidate is None:
                    node_reports.append({"node":int(node),**node_diag})
                    continue
                step_l1=float(np.abs(node_candidate[node].astype(np.float64)-current[node].astype(np.float64)).sum())
                if path_l1[node]+step_l1>cap+1.e-15:
                    node_reports.append({"node":int(node),"status":"CUMULATIVE_PATH_CAP","step_l1_m":step_l1})
                    continue
                nd,nq=Q._M.measurements(node_candidate,t[node_cells])
                ng=np.asarray(Q._M.passes(nd,nq),bool)
                if np.any(nd<=0) or np.any(cur_good[node_cells]&~ng) or np.any(base_good[node_cells]&~ng):
                    node_reports.append({"node":int(node),"status":"GLOBAL_LOCAL_REPLAY_REJECT"})
                    continue
                before_local_bad=int(np.count_nonzero(~cur_good[node_cells]))
                after_local_bad=int(np.count_nonzero(~ng))
                candidate_global_bad=cur_bad-before_local_bad+after_local_bad
                before_local_score=_global_log_deficit(cur_q[node_cells],cur_d[node_cells])
                after_local_score=_global_log_deficit(nq,nd)
                candidate_global_score=cur_score-before_local_score+after_local_score
                tol=max(1.e-8,1.e-6*cur_score)
                count_gain=candidate_global_bad<cur_bad
                score_gain=(candidate_global_bad==cur_bad and cur_score-candidate_global_score>tol)
                if not (count_gain or score_gain):
                    node_reports.append({"node":int(node),"status":"NO_GLOBAL_LEX_PROGRESS",
                        "global_bad_before":cur_bad,"global_bad_candidate":candidate_global_bad,
                        "score_before":cur_score,"score_candidate":candidate_global_score})
                    continue
                current=node_candidate.copy()
                cur_d[node_cells]=nd;cur_q[node_cells]=nq;cur_good[node_cells]=ng
                cur_bad=candidate_global_bad;cur_score=candidate_global_score;path_l1[node]+=step_l1
                touched.update(map(int,node_cells));patch_accepted=True;accepted_nodes+=1
                node_record={"node":int(node),"global_bad_after":cur_bad,"score_after":cur_score,"step_l1_m":step_l1,
                    "cumulative_path_l1_m":float(path_l1[node]),**node_diag}
                node_record["optimizer_candidate_status"]=node_record.pop("status",None)
                node_record["status"]="ACCEPT_COUNT_REDUCTION" if count_gain else "ACCEPT_TIED_SCORE_REDUCTION"
                node_reports.append(node_record)
                if cur_bad==0:break
            if node_reports:
                local_summary["single_node_fallback"]={"attempt_count":len(node_reports),
                    "accepted_count":sum(1 for x in node_reports if str(x.get("status","")).startswith("ACCEPT_")),
                    "nodes":node_reports}
        local_summary["native_bad_count_after_one_node_fallback"]=int(cur_bad)
        joint_reports=[]
        no_patch_count_gain=(cur_bad==int(local_summary["native_bad_count_before_patch"]))
        if (joint_small_patch_fallback and joint_seed_candidates and lp_attempted
                and node_reports and no_patch_count_gain and time.monotonic()<patch_deadline):
            joint_candidate_signatures=set()
        for seed_index,(_,seed_cells,seed_free_rows,candidate_kind) in enumerate(joint_seed_candidates):
                remain=patch_deadline-time.monotonic()
                if remain<=.02:break
                remaining_candidates=max(1,len(joint_seed_candidates)-seed_index)
                call_budget=min(float(joint_patch_seconds),remain/remaining_candidates)
                if call_budget<=0:break
                remaining_caps=np.maximum(0.,cap-path_l1)
                call_seed_cells=np.asarray(seed_cells,np.int64)
                call_free_rows=np.asarray([int(r) for r in seed_free_rows if remaining_caps[int(r)]>1.e-12],np.int64)
                witness_selection=None
                if candidate_kind=="expanded-seed-vertex-star" and joint_reports:
                    prior_report=joint_reports[-1]
                    seed_vertices=np.unique(t[call_seed_cells]).astype(np.int64)
                    seed_free=np.asarray([int(v) for v in seed_vertices if int(v) in set(map(int,patch_rows))
                        and remaining_caps[int(v)]>1.e-12],np.int64)
                    witness_free,witness_selection=_witness_directed_expansion_v26e(
                        call_seed_cells,seed_free,prior_report,call_free_rows,patch_rows,current,t,cur_good,
                        path_l1,cap,offsets,by_node,16)
                    if witness_free is not None:
                        call_free_rows=np.asarray(witness_free,np.int64)
                        seed_free_rows=call_free_rows.copy()
                        candidate_kind="expanded-regressed-good-witness"
                per_seed_free=np.asarray([np.count_nonzero(np.isin(t[int(c)],call_free_rows)) for c in call_seed_cells],np.int64)
                valid_size=(2<=len(call_free_rows)<=4) if candidate_kind=="seed-corners" else (8<=len(call_free_rows)<=16)
                if (not valid_size or len(call_seed_cells)<1 or len(call_seed_cells)>16
                        or np.any(cur_good[call_seed_cells]) or np.any((per_seed_free<2)|(per_seed_free>4))):
                    joint_reports.append({"seed_cell":int(seed_cells[0]),"seed_bad_cells":list(map(int,seed_cells)),
                        "free_rows":call_free_rows.tolist(),"candidate_kind":candidate_kind,
                        "status":"SKIP_REMAINING_FREE_CORNERS_OUTSIDE_ALLOWED_SEED_OR_EXPANDED_RANGE"})
                    continue
                joint_star=np.asarray(_incident_cells_by_rows(call_free_rows,offsets,by_node),np.int64)
                signature=(tuple(map(int,call_seed_cells)),tuple(map(int,call_free_rows)),tuple(sorted(map(int,joint_star))))
                if signature in joint_candidate_signatures:
                    joint_reports.append({"seed_cell":int(seed_cells[0]),"seed_bad_cells":list(map(int,seed_cells)),
                        "free_rows":call_free_rows.tolist(),"candidate_kind":candidate_kind,
                        "selection_receipt":witness_selection,"status":"SKIP_DUPLICATE_ROW_AND_COMPLETE_STAR_SIGNATURE"})
                    continue
                joint_candidate_signatures.add(signature)
                accepted_node_statuses=[str(x.get("status")) for x in node_reports if str(x.get("status","")).startswith("ACCEPT_")]
                prior={"lp_attempted":bool(lp_attempted),
                    "lp_status":"NO_NATIVE_BAD_COUNT_REDUCTION",
                    "lp_raw_status":lp_last_status,
                    "lp_accepted_score_only":bool(lp_accepted),
                    "node_fallback_attempted":bool(node_reports),
                    "node_fallback_accepted":int(len(accepted_node_statuses)),
                    "node_fallback_accepted_statuses":accepted_node_statuses,
                    "node_statuses":[str(x.get("status","UNKNOWN")) for x in node_reports],
                    "native_bad_count_before_patch":int(local_summary["native_bad_count_before_patch"]),
                    "block_lp_bad_count_after":int(local_summary["native_bad_count_after_block_lp"]),
                    "node_bad_count_after":int(cur_bad),
                    "native_bad_count_before_joint":int(cur_bad),
                    "block_lp_count_reduction":False,
                    "node_fallback_count_reduction":False,
                    "native_bad_count_reduction_since_patch_start":False}
                if not (int(prior["native_bad_count_before_patch"])==int(prior["block_lp_bad_count_after"])==int(prior["node_bad_count_after"])==int(prior["native_bad_count_before_joint"])):
                    raise AssertionError("joint count-stall eligibility ledger disagrees")
                try:
                    joint_candidate,joint_diag=JOINT.try_joint_small_patch_slsqp(
                        current,t,call_free_rows,call_seed_cells,cur_good,base_good,remaining_caps,prior,
                        separation_rows=separation_rows,allowed_free_rows=patch_rows,
                        incidence_offsets=offsets,incidence_by_node=by_node,
                        maxiter=int(joint_maxiter),time_limit_s=min(3.0,call_budget),return_float64_candidate=True)
                except Exception as exc:
                    joint_candidate=None;joint_diag={"status":"JOINT_HELPER_EXCEPTION","exception":repr(exc)}
                if joint_candidate is not None and joint_diag.get("status")=="RAW_FLOAT64_CANDIDATE_FOR_F32_BACKTRACK":
                    raw64=np.asarray(joint_candidate,np.float64)
                    joint_cells=np.asarray(joint_diag.get("affected_cells",[]),np.int64)
                    joint_candidate,backtrack_diag=_backtrack_f32_candidate(
                        current,raw64,t,joint_cells,call_free_rows,remaining_caps[call_free_rows],
                        cur_d,cur_q,cur_good,base_good,p0,separation_rows,cur_bad,cur_score,joint_diag)
                    joint_diag={**joint_diag,"optimizer_candidate_status":"RAW_FLOAT64_CANDIDATE_FOR_F32_BACKTRACK",
                        "status":"F32_BACKTRACKED_CANDIDATE" if joint_candidate is not None else "NO_ACCEPTED_F32_BACKTRACK_ALPHA",
                        "f32_backtracking":backtrack_diag}
                record={"seed_cell":int(seed_cells[0]),"seed_bad_cells":list(map(int,seed_cells)),"candidate_kind":candidate_kind,"free_rows":call_free_rows.tolist(),"selection_receipt":witness_selection,"prior_failures":prior,**joint_diag}
                if joint_candidate is None:
                    joint_reports.append(record)
                    continue
                joint_candidate=np.asarray(joint_candidate,np.float32)
                moved_rows=np.flatnonzero(np.any(joint_candidate.view(np.uint32)!=current.view(np.uint32),axis=1))
                if not set(map(int,moved_rows)).issubset(set(map(int,seed_free_rows))):
                    joint_reports.append({**record,"status":"REJECT_MOVED_POINT_OUTSIDE_SEED_PATCH"})
                    continue
                step_l1=np.abs(joint_candidate[moved_rows].astype(np.float64)-current[moved_rows].astype(np.float64)).sum(axis=1)
                if len(moved_rows) and np.any(path_l1[moved_rows]+step_l1>cap+1.e-15):
                    joint_reports.append({**record,"status":"REJECT_CUMULATIVE_PATH_CAP"})
                    continue
                proposed_cells=np.asarray(joint_diag.get("affected_cells",[]),np.int64)
                if not len(proposed_cells):
                    joint_reports.append({**record,"status":"REJECT_EMPTY_AFFECTED_STAR"})
                    continue
                new_d,new_q=Q._M.measurements(joint_candidate,t[proposed_cells])
                new_good=np.asarray(Q._M.passes(new_d,new_q),bool)
                if (not np.isfinite(new_d).all() or not np.isfinite(new_q).all() or np.any(new_d<=0)
                        or np.any(cur_good[proposed_cells]&~new_good) or np.any(base_good[proposed_cells]&~new_good)):
                    joint_reports.append({**record,"status":"REJECT_FULL_AFFECTED_STAR_NATIVE_REPLAY"})
                    continue
                exact=Q._exact_positive(joint_candidate,t[proposed_cells])
                if any(int(x)<=0 for x in exact):
                    joint_reports.append({**record,"status":"REJECT_FULL_AFFECTED_STAR_EXACT_F32_SIGN"})
                    continue
                before_local_bad=int(np.count_nonzero(~cur_good[proposed_cells]))
                after_local_bad=int(np.count_nonzero(~new_good))
                candidate_global_bad=cur_bad-before_local_bad+after_local_bad
                before_local_score=_global_log_deficit(cur_q[proposed_cells],cur_d[proposed_cells])
                after_local_score=_global_log_deficit(new_q, new_d)
                candidate_global_score=cur_score-before_local_score+after_local_score
                tolerance=max(1.e-8,1.e-6*cur_score)
                count_gain=candidate_global_bad<cur_bad
                score_gain=(candidate_global_bad==cur_bad and cur_score-candidate_global_score>tolerance)
                if not (count_gain or score_gain):
                    joint_reports.append({**record,"status":"REJECT_NO_GLOBAL_LEX_PROGRESS",
                        "global_bad_before":int(cur_bad),"global_bad_candidate":int(candidate_global_bad),
                        "global_score_before":float(cur_score),"global_score_candidate":float(candidate_global_score)})
                    continue
                current=joint_candidate.copy();cur_d[proposed_cells]=new_d;cur_q[proposed_cells]=new_q;cur_good[proposed_cells]=new_good
                cur_bad=candidate_global_bad;cur_score=candidate_global_score
                if len(moved_rows):path_l1[moved_rows]+=step_l1
                touched.update(map(int,proposed_cells));patch_accepted=True;accepted+=1;accepted_joint+=1
                joint_reports.append({**record,"status":"ACCEPT_COUNT_REDUCTION" if count_gain else "ACCEPT_TIED_SCORE_REDUCTION",
                    "global_bad_after":int(cur_bad),"global_score_after":float(cur_score),
                    "moved_rows":moved_rows.astype(int).tolist(),"max_cumulative_path_l1_m":float(path_l1.max(initial=0.))})
                if count_gain or cur_bad<int(local_summary["native_bad_count_before_patch"]):break
        if joint_reports:
            local_summary["joint_multi_node_slsqp_fallback"]={"attempt_count":len(joint_reports),
                "accepted_count":sum(1 for x in joint_reports if str(x.get("status","")).startswith("ACCEPT_")),
                "shared_patch_deadline_s":float(joint_patch_seconds),"reserved_slice_s":float(joint_reserve),
                "reports":joint_reports}
        if not patch_accepted:
            skipped+=1
            if "status" not in local_summary:local_summary["status"]="SKIP_NO_BUDGET_OR_FREE_ROWS"
        reports.append(local_summary)
        if cur_bad==0 or time.monotonic()>=deadline:break
    # One independent full-domain native replay proves the incremental local updates agree.
    final_d,final_q=Q._M.measurements(current,t);final_good=np.asarray(Q._M.passes(final_d,final_q),bool)
    if np.any(final_d<=0) or np.any(base_good&~final_good) or np.any(cur_good&~final_good):
        raise AssertionError("final block-local candidate violates global native/positivity gates")
    final_bad=int(np.count_nonzero(~final_good));final_score=_global_log_deficit(final_q,final_d)
    final_role_score=_global_log_deficit_by_role(final_q,final_d,groups)
    if abs(sum(final_role_score.values())-final_score)>max(1.e-8,1.e-10*max(1.,abs(final_score))):
        raise AssertionError("final per-role log-deficit scores do not sum to unchanged global score")
    if final_bad!=cur_bad or abs(final_score-cur_score)>max(1.e-7,1.e-8*max(1.,abs(final_score))):
        raise AssertionError("incremental block-local global ledger disagrees with full-domain replay")
    if path_l1.max(initial=0.)>cap+1.e-15:raise AssertionError("block-local cumulative path trust cap exceeded")
    count_progress=final_bad<base_bad
    score_progress=final_bad==base_bad and base_score-final_score>max(1.e-8,1.e-6*base_score)
    progress=bool(count_progress or score_progress)
    if not progress:current=p0.copy();final_bad=base_bad;final_score=base_score
    changed=np.flatnonzero(np.any(current.view(np.uint32)!=p0.view(np.uint32),axis=1))
    f32_events=[]
    for patch_report in reports:
        for node_report in patch_report.get("single_node_fallback",{}).get("nodes",[]):
            if "f32_backtracking" in node_report:f32_events.append(node_report["f32_backtracking"])
        for joint_report in patch_report.get("joint_multi_node_slsqp_fallback",{}).get("reports",[]):
            if "f32_backtracking" in joint_report:f32_events.append(joint_report["f32_backtracking"])
    accepted_f32_events=sum(1 for x in f32_events if x.get("status") in ("PASS_BOUNDED_F32_BACKTRACKED_PROPOSAL","PASS_BOUNDED_F32_ULP_CORRECTED_PROPOSAL"))
    return current,{"schema":"h706.target-remainder-block-local-slp.v3-joint-count-stall",
        "status":"PASS_F32_BLOCK_LOCAL_TARGET_QUALITY_PROPOSAL" if progress else "NO_LEXICOGRAPHIC_PROGRESS",
        "physical_admission":False,"native_steps":0,"bad_before":base_bad,"bad_after":final_bad,
        "bad_score_before":base_score,"bad_score_after":final_score,
        "quality_log_deficit_by_role_before":base_role_score,"quality_log_deficit_by_role_after":final_role_score,
        "fat_log_deficit_score_before":base_role_score["Fat"],"fat_log_deficit_score_after":final_role_score["Fat"],
        "skin_log_deficit_score_before":base_role_score["Skin"],"skin_log_deficit_score_after":final_role_score["Skin"],
        "strict_fat_log_deficit_score_progress":bool(base_role_score["Fat"]-final_role_score["Fat"]>max(1.e-8,1.e-6*base_role_score["Fat"])),
        "strict_skin_log_deficit_score_progress":bool(base_role_score["Skin"]-final_role_score["Skin"]>max(1.e-8,1.e-6*base_role_score["Skin"])),
        "strict_quality_count_progress":bool(count_progress),"strict_bad_score_progress":bool(score_progress),
        "accepted_components":int(accepted),"accepted_log_deficit_components":int(0),
        "local_patch_count":len(patches),"local_patch_attempts":reports,
        "visited_patch_count":int(len(reports)),"unvisited_patch_count":int(max(0,len(patches)-len(reports))),
        "per_patch_combined_solver_and_fallback_budget_s":patch_budget,
        "per_patch_deadline_policy":"one shared deadline per bad-face component covers block-LP retries and single-node fallbacks; later components receive a fresh slice until the global deadline",
        "accepted_local_patches":int(accepted),"accepted_single_node_fallbacks":int(accepted_nodes),
        "accepted_joint_small_patch_fallbacks":int(accepted_joint),
        "f32_candidate_backtracking_helper":{"path":str(_F32_BACKTRACK_PATH),"sha256":_F32_BACKTRACK_SHA},
        "f32_witness_ulp_quantization_helper":{"path":str(_ULP_PATH),"sha256":_ULP_SHA},
        "f32_backtracking_event_count":int(len(f32_events)),"accepted_f32_backtracked_proposals":int(accepted_f32_events),
        "joint_small_patch_fallback":{"enabled":bool(joint_small_patch_fallback),
            "helper":{"path":str(_JOINT_PATH),"sha256":_JOINT_EXPECTED_SHA},
            "attempts_per_patch":int(joint_fallbacks_per_patch),"reserve_seconds":float(joint_patch_seconds),"reserve_patch_budget_fraction":0.65,
            "maxiter":int(joint_maxiter),
            "eligibility":"after LP and node attempts when the current bad-face patch has no native bad-count reduction; prior score-only gains are retained; one 2-4 seed-corner solve is followed by at most one solve with 8-16 authorized movable rows from the seed tetrahedron one-ring, including neighboring currently-good cell corners as constrained variables",
            "expanded_free_node_range":[8,16],"expanded_rows_must_be_in_authorized_patch_and_complete_seed_vertex_star":True,
            "deadline":"reserved slice remains inside the same per-patch deadline; full star and exact Float32 local gates before ledger acceptance"},
        "skipped_local_patches":int(skipped),
        "local_bad_face_components":len({int(x["bad_component"]) for x in patches}),
        "affected_cell_sets_are_point_incident_only":True,
        "current_passed_cells_and_immutable_baseline_good_cells_never_regress":True,
        "cumulative_per_point_path_L1_cap_m":cap,"maximum_cumulative_path_L1_m":float(path_l1.max(initial=0.)),
        "final_full_domain_native_replay":True,"full_domain_cells":int(len(t)),
        "changed_point_rows":changed.astype(int).tolist(),
        "max_free_node_l1_displacement_bound_m":float(path_l1.max(initial=0.)) if progress else 0.,
        "total_trust_cap_m":cap,"objective_mode":str(objective_mode),
        "trust_policy":"bounded local patches; per-point cumulative L1 measured from immutable baseline across all accepted patches",
        "elapsed_s":time.monotonic()-start}

def validate_explicit_250um_construction_policy(trust_cap_m,allow_external_boundary=False):
 if allow_external_boundary:raise ValueError("250um construction trial cannot release the external boundary")
 if not 5.e-5 < float(trust_cap_m) <= 2.5e-4:raise ValueError("explicit 250um construction trial requires a cap above 50um and at most 250um")
 return {"inner_back_layer_extended_construction_trial":True,
         "maximum_displacement_m":float(trust_cap_m),
         "ordinary_outer_O_local_repair":True,
         "ordinary_outer_O_max_cap_m":5.e-4,
         "outer_O_scope_fixed":True,
         "outer_O_entire_surface_fixed":False,
         "outer_O_back_scope_fixed":True,
         "rims_and_routes_fixed":True,
         "physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"}


UNIFIED_250_POLICY_TEXT=(
 "all baseline-bad face components use ordinary bad-corner/source-ring freedoms; "
 "I/D additionally use declared inner-back construction freedoms and authenticated actual-back Fat rows in each bad-cell star; "
 "ordinary O freedom is limited to bad-corner/source-ring rows at the ordinary cap (at most 500um); "
 "O back scope, outer scope, source rims and all 149 routes remain fixed")

def construction_policy_metadata(args):
 extended=bool(args.unified_inner_back_ordinary_250um_construction_trial)
 unified=bool(args.unified_inner_back_ordinary_local_repair or extended)
 ordinary_o=bool(args.ordinary_outer_o_local_repair or args.ordinary_all_role_local_repair or unified)
 policy=(UNIFIED_250_POLICY_TEXT if unified else None)
 if args.unified_inner_back_ordinary_250um_construction_trial:
  validate_explicit_250um_construction_policy(args.trust_cap_m,args.allow_external_boundary)
  if not unified or args.trust_cap_m>2.5e-4:
   raise ValueError("extended I/D construction policy does not match unified mode/cap")
  if not ordinary_o:
   raise ValueError("extended construction mode must retain the ordinary O local policy")
 return {
  "inner_back_layer_repair":bool(args.inner_back_layer_repair or args.inner_back_remainder_repair),
  "inner_back_layer_extended_construction_trial":extended,
  "inner_back_layer_extended_construction_cap_m":float(args.trust_cap_m) if extended else None,
  "outer_O_scope_fixed":True,
  "outer_O_entire_surface_fixed":not ordinary_o,
  "outer_O_back_scope_fixed":True,
  "ordinary_outer_O_local_repair":ordinary_o,
  "ordinary_outer_O_maximum_displacement_m":float(args.trust_cap_m) if ordinary_o else None,
  "ordinary_outer_O_max_cap_m":5.e-4 if ordinary_o else None,
  "ordinary_outer_O_policy":("bad O corners plus one direct incident O source-triangle ring; O back scope, outer scope, rims and routes remain fixed; cap <= 500um" if ordinary_o else None),
  "unified_inner_back_ordinary_policy":policy,
 }

def bounded_point_repair_entries(args, changed_roles):
 changed=set(changed_roles);out={}
 unified=bool(args.unified_inner_back_ordinary_local_repair or args.unified_inner_back_ordinary_250um_construction_trial)
 extended=bool(args.unified_inner_back_ordinary_250um_construction_trial)
 for role in sorted(changed):
  entry={"maximum_displacement_m":float(args.trust_cap_m)}
  if unified:
   if role in ("I","D"):
    entry["inner_back_layer_extended_construction_trial" if extended else "inner_back_layer_repair"]=True
   elif role=="O":
    entry["ordinary_outer_o_local_repair"]=True
   else:raise ValueError("unknown bounded source role")
  elif args.ordinary_all_role_local_repair:
   pass
  elif args.ordinary_outer_o_local_repair and role=="O":
   entry["ordinary_outer_o_local_repair"]=True
  bounded_cap=(2.5e-4 if extended else (5.e-5 if unified else 5.e-4))
  if float(args.trust_cap_m)>bounded_cap:
   raise ValueError("source role cap exceeds the selected construction policy")
  out[role]=entry
 return out

def serialized_policy_contract(args, changed_roles):
 """Return the exact metadata/config fragments serialized by main()."""
 return (construction_policy_metadata(args),
         {"changed_roles":sorted(set(changed_roles)),
          "bounded_point_repair":bounded_point_repair_entries(args,changed_roles)})


def main():
 ap=argparse.ArgumentParser();ap.add_argument("--state",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--trust-cap-m",type=float,default=2e-5);ap.add_argument("--seconds",type=float,default=30);ap.add_argument("--allow-external-boundary",action="store_true");ap.add_argument("--inner-back-layer-repair",action="store_true");ap.add_argument("--inner-back-remainder-repair",action="store_true");ap.add_argument("--inner-back-remainder-preflight-only",action="store_true");ap.add_argument("--block-local-target-repair",action="store_true");ap.add_argument("--patch-seconds",type=float,default=1.5);ap.add_argument("--node-fallbacks-per-patch",type=int,default=8);ap.add_argument("--ordinary-outer-o-local-repair",action="store_true");ap.add_argument("--ordinary-all-role-local-repair",action="store_true");ap.add_argument("--unified-inner-back-ordinary-local-repair",action="store_true");ap.add_argument("--unified-inner-back-ordinary-250um-construction-trial",action="store_true");ap.add_argument("--ordinary-block-local-quality",action="store_true");ap.add_argument("--ordinary-preflight-only",action="store_true");ap.add_argument("--witness-report",type=Path,action="append",default=[]);ap.add_argument("--witness-preflight-only",action="store_true");ap.add_argument("--objective-mode",choices=("legacy","role_equal","role_equal_nearpass"),default="legacy");args=ap.parse_args()
 if sum(map(bool,(args.inner_back_layer_repair,args.inner_back_remainder_repair,args.ordinary_outer_o_local_repair,args.ordinary_all_role_local_repair,args.allow_external_boundary,args.unified_inner_back_ordinary_local_repair,args.unified_inner_back_ordinary_250um_construction_trial)))>1:raise ValueError("unified mixed mode is exclusive with other boundary policies")
 if not 0 < args.trust_cap_m <= .0005: raise ValueError('trust cap must be positive and at most 0.5 mm')
 if args.unified_inner_back_ordinary_250um_construction_trial:
  validate_explicit_250um_construction_policy(args.trust_cap_m,args.allow_external_boundary)
  # Reuse the proven unified local bad-corner/one-ring masks while keeping the
  # new source-policy declaration distinct from the legacy 50um policy.
  args.unified_inner_back_ordinary_local_repair=True
 elif args.inner_back_layer_repair or args.inner_back_remainder_repair or args.unified_inner_back_ordinary_local_repair:
  if args.allow_external_boundary:raise ValueError("inner-back policies cannot release the external boundary")
  if args.trust_cap_m>5.e-5:raise ValueError("legacy inner-back policies are capped at 50 micrometres")
 if args.ordinary_block_local_quality and not (args.ordinary_all_role_local_repair or args.unified_inner_back_ordinary_local_repair):raise ValueError("block-local quality requires ordinary all-role or unified mixed mode")
 if args.inner_back_remainder_preflight_only and not args.inner_back_remainder_repair:
  raise ValueError("remainder preflight requires --inner-back-remainder-repair")
 if args.block_local_target_repair and not args.inner_back_remainder_repair:
  raise ValueError("block-local target repair requires --inner-back-remainder-repair")
 if not 0<args.patch_seconds<=5.:raise ValueError("per-patch solve budget must be in (0,5] seconds")
 if not 0<=args.node_fallbacks_per_patch<=16:raise ValueError("node fallback count must be in [0,16]")
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
  _,preflight_info=_witness_separation_rows(args.witness_report,st,surfaces,preflight_map,preflight_points,out,pin(args.state))
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
 fixed=set();ordinary_reference_fixed=set();anchor_counts={"protected_scope_rows":0,"source_rim_rows":0,"route149_exact_rows":0,"actual_back412_fat_rows":0}
 for role,a in surfaces.items():
  pp=a["points_world_m_f32"];sf=a["faces"];mp=np.asarray([global_lookup.get(T.key(x),-1) for x in pp],np.int64);smap[role]=mp
  if (role=="O" and not (args.ordinary_outer_o_local_repair or args.ordinary_all_role_local_repair or args.unified_inner_back_ordinary_local_repair)) or (role=="D" and not args.allow_external_boundary and not args.inner_back_layer_repair and not args.inner_back_remainder_repair and not args.ordinary_all_role_local_repair and not args.unified_inner_back_ordinary_local_repair):fixed.update(map(int,mp[mp>=0]))
  scope_rows=set(map(int,masks[role+"_vertex_rows"]))
  ordinary_reference_fixed.update(int(mp[r]) for r in scope_rows if 0<=r<len(mp) and mp[r]>=0)
  if not ((args.inner_back_layer_repair or args.inner_back_remainder_repair or args.unified_inner_back_ordinary_local_repair) and role in ("I","D")):
   protected=set(scope_rows)
  else:
   protected=set()
  anchor_counts["protected_scope_rows"]+=sum(1 for r in scope_rows if 0<=r<len(mp) and mp[r]>=0)
  ed=np.sort(np.concatenate([sf[:,[0,1]],sf[:,[1,2]],sf[:,[2,0]]]),axis=1);ue,ct=np.unique(ed,axis=0,return_counts=True);rim_rows=set(map(int,ue[ct==1].ravel()));protected.update(rim_rows);ordinary_reference_fixed.update(int(mp[r]) for r in rim_rows if 0<=r<len(mp) and mp[r]>=0);anchor_counts["source_rim_rows"]+=sum(1 for r in rim_rows if mp[r]>=0)
  for r in routes:
   if r["role"]==role:
    hit=np.flatnonzero(np.all(pp==np.asarray(r["target_world_m_f32"],np.float32),axis=1));assert len(hit)==1;protected.add(int(hit[0]));ordinary_reference_fixed.add(int(mp[int(hit[0])])) if mp[int(hit[0])]>=0 else None;anchor_counts["route149_exact_rows"]+=int(mp[int(hit[0])]>=0)
  fixed.update(int(mp[r]) for r in protected if mp[r]>=0)
 if args.ordinary_all_role_local_repair or args.inner_back_remainder_repair or args.unified_inner_back_ordinary_local_repair:
  back_receipt=authenticated_actual_back_fat_rows(args.state,st,surfaces,volumes,out)
  back=np.asarray(back_receipt["fat_rows"],np.int64)
  back_i_rows=np.asarray(back_receipt["current_i_rows"],np.int64)
  d_source_rows=np.asarray(back_receipt["current_d_source_rows"],np.int64)
  d_fat_by_source=np.asarray(back_receipt["current_d_fat_by_source_row"],np.int64)
  if d_source_rows.shape!=d_fat_by_source.shape or np.any(d_source_rows<0) or np.any(d_fat_by_source<0):
   raise ValueError("855 authenticated D back-row arrays have inconsistent/incomplete domains")
 else:
  back=load(H/"paired-I-current620-root/actual-back412-fat-point-mask.npz")["affected_volume_point_rows"]
  back_receipt=None
  back_i_rows=np.empty(0,np.int64)
  d_source_rows=np.empty(0,np.int64)
  d_fat_by_source=np.empty(0,np.int64)
 back_global=np.asarray(vmap["Fat"][back],np.int64)
 anchor_counts["actual_back412_fat_rows"]=int(len(back))
 back_summary=None
 if args.ordinary_all_role_local_repair or args.inner_back_remainder_repair or args.unified_inner_back_ordinary_local_repair:
  back_asset=out/"actual-back-fat-protected-rows.npz"
  np.savez_compressed(back_asset,current_i_surface_rows=back_i_rows,
      current_fat_volume_rows=np.asarray(back_receipt["current_fat_rows"],np.int64),
      historical_fat_volume_rows=np.asarray(back_receipt["historical_fat_rows"],np.int64),
      current_plus_historical_fat_volume_rows=back,
      historical_i_reference_rows=np.asarray(back_receipt["historical_i_rows"],np.int64),
      current_i_surface_rows_by_historical_record=np.asarray(back_receipt["current_i_rows"],np.int64),
      current_fat_volume_rows_by_historical_record=np.asarray(back_receipt["current_fat_by_record"],np.int64),
      historical_fat_volume_rows_by_historical_record_or_minus_one=np.asarray(back_receipt["historical_fat_by_record"],np.int64),
      shared_global_fat_rows=back_global,
      shared_global_current_i_rows=np.asarray(smap["I"][back_i_rows],np.int64),
      current_d_surface_back_rows=d_source_rows,
      current_d_fat_volume_rows_by_current_source_row=d_fat_by_source,
      shared_global_current_d_rows=np.asarray(smap["D"][d_source_rows],np.int64))
  back_summary={"schema":"h706.actual-back-fat-lineage-protection.v1",
      "row_asset":pin(back_asset),"current_i_surface_rows":int(len(back_i_rows)),
      "current_fat_volume_rows":int(len(back_receipt["current_fat_rows"])),
      "historical_fat_volume_rows":int(len(back_receipt["historical_fat_rows"])),
      "current_plus_historical_unique_fat_rows":int(len(back)),
      "historical_record_coverage":back_receipt["coverage"],
      "lineage":back_receipt["lineage"],"scope_mask":back_receipt["scope_mask"],
      "scope_report":back_receipt["scope_report"],"scope_reference_I":back_receipt["scope_reference_I"],
      "current_I":back_receipt["current_I"],"source_owner":back_receipt["source_owner"],
      "protection_owner_report":back_receipt["protection_report"]}
 ordinary_reference_fixed.update(map(int,back_global))
 if back_receipt is not None:
  ordinary_reference_fixed.update(int(smap["I"][r]) for r in back_i_rows if 0<=r<len(smap["I"]) and smap["I"][r]>=0)
 if not args.inner_back_layer_repair and not args.inner_back_remainder_repair and not args.unified_inner_back_ordinary_local_repair:fixed.update(map(int,back_global))
 if args.ordinary_all_role_local_repair:
  current_i_back_global=np.asarray(smap["I"][back_i_rows],np.int64)
  if np.any(current_i_back_global<0):raise ValueError("current actual-back I rows do not map to shared points")
  fixed.update(map(int,current_i_back_global))
  anchor_counts["current_actual_back_I_rows"]=int(len(current_i_back_global))
 separation_rows,witness_info=_witness_separation_rows(args.witness_report,st,surfaces,smap,p,out,pin(args.state),skip_fixed_o_pairs=bool((args.inner_back_layer_repair or args.inner_back_remainder_repair) and not args.ordinary_outer_o_local_repair and not args.unified_inner_back_ordinary_local_repair))
 witness_frozen=np.empty(0,np.int64)
 if separation_rows is not None:witness_frozen=np.asarray(separation_rows["preparation"].get("freeze_source_nodes",[]),np.int64)
 free_parts=[];one_ring_info={"schema":"h706.coupled-quality-role-free-set.v24"}
 if args.inner_back_layer_repair:
  inner_scope_global=set()
  for role in ("I","D"):
   rr=np.asarray(masks[role+"_vertex_rows"],np.int64);mapped_scope=smap[role][rr];inner_scope_global.update(map(int,mapped_scope[mapped_scope>=0]))
  inner_free,inner_info=expand_inner_back_layer_once(
   bad_corner_rows,{r:np.asarray(surfaces[r]["faces"],np.int64) for r in ("D","I")},smap,t,
   back_global,fixed_global_rows=fixed,witness_frozen_global_rows=witness_frozen,
   inner_scope_global_rows=inner_scope_global)
  free_parts.append(inner_free);one_ring_info["inner_back_layer"]=inner_info
 if args.unified_inner_back_ordinary_local_repair:
  faces_by_role={r:np.asarray(surfaces[r]["faces"],np.int64) for r in ("O","I","D")}
  bad_cells=np.flatnonzero(~initial)
  ordinary_free,inner_free,unified_free,ordinary_info,inner_info=build_unified_free_sets(
   bad_cells,faces_by_role,smap,t,back_global,ordinary_reference_fixed,fixed,witness_frozen)
  patch_roster,patch_receipt,patch_offsets,patch_by_node=_build_unified_block_local_patches(
   bad_cells,ordinary_free,inner_free,faces_by_role,smap,t,back_global,fixed,witness_frozen,len(p))
  patch_roster,patch_receipt=_schedule_unified_patches_fat_first(
   patch_roster,patch_receipt,quality_group_ids,q,d)
  free_parts.append(unified_free)
  one_ring_info["unified_inner_back_ordinary"]={"policy":"all baseline-bad face components use the union of ordinary bad-corner/source-one-ring rows and I/D-inner plus actual-back Fat rows from every bad-corner star",
      "patch_priority_policy":patch_receipt["scheduler_policy"],"ordinary_mask":ordinary_info,"inner_mask":inner_info,"roster":patch_receipt}
 if args.ordinary_all_role_local_repair and args.ordinary_block_local_quality:
  ordinary_free,ordinary_info=expand_triangle_ring_once(
   bad_corner_rows,{r:np.asarray(surfaces[r]["faces"],np.int64) for r in ("D","O","I")},smap,
   fixed_global_rows=ordinary_reference_fixed,witness_frozen_global_rows=witness_frozen)
  ordinary_mask=np.zeros(len(p),dtype=bool);ordinary_mask[ordinary_free]=True
  bad_cells=np.flatnonzero(~initial)
  selected_cells=bad_cells[np.any(ordinary_mask[t[bad_cells]],axis=1)] if len(bad_cells) else np.empty(0,np.int64)
  block_roster,block_roster_receipt,block_offsets,block_by_node=_build_ordinary_block_local_patches(
   bad_cells,selected_cells,ordinary_free,
   {r:np.asarray(surfaces[r]["faces"],np.int64) for r in ("O","I","D")},smap,t,len(p))
  free_parts.append(ordinary_free)
  one_ring_info["ordinary_all_role_block_local"]={"mask":ordinary_info,
      "selected_bad_cells":int(len(selected_cells)),"roster":block_roster_receipt}
 if args.inner_back_remainder_repair:
  # Recreate the v25 ordinary all-role mask exactly, using authenticated current
  # back I/Fat lineage instead of the historical I620 point-row asset.
  ordinary_free,ordinary_info=expand_triangle_ring_once(
   bad_corner_rows,{r:np.asarray(surfaces[r]["faces"],np.int64) for r in ("D","O","I")},smap,
   fixed_global_rows=ordinary_reference_fixed,witness_frozen_global_rows=witness_frozen)
  ordinary_mask=np.zeros(len(p),dtype=bool);ordinary_mask[ordinary_free]=True
  bad_cells=np.flatnonzero(~initial)
  selected_cells=select_bad_cells_without_ordinary_free_corner(t,bad_cells,ordinary_free)
  remainder_free,remainder_info=expand_inner_back_remainder_once(
   selected_cells,{r:np.asarray(surfaces[r]["faces"],np.int64) for r in ("D","O","I")},
   smap,t,back_global,fixed_global_rows=fixed,witness_frozen_global_rows=witness_frozen)
  selected_hit=np.zeros(len(t),dtype=bool);selected_hit[selected_cells]=True
  covered=bad_cells[np.any(ordinary_mask[t[bad_cells]],axis=1)] if len(bad_cells) else np.empty(0,np.int64)
  remainder_info.update({"ordinary_reference_free_rows":int(len(ordinary_free)),
      "ordinary_reference_bad_cells_with_any_free_corner":int(len(bad_cells)-len(selected_cells)),
      "ordinary_reference_bad_cells_with_zero_free_corners":int(len(selected_cells)),
      "ordinary_reference_bad_cells_verified_covered":int(len(covered)),
      "ordinary_reference_policy":"v25 ordinary all-role local mask: baseline-bad volume corners plus one direct source-triangle ring over O/I/D; all scope/rims/routes and authenticated current/historical actual Fat back fixed",
      "selected_bad_cells_by_quality_role":{"Skin":int(np.count_nonzero((quality_group_ids[selected_cells]==0))),"Fat":int(np.count_nonzero((quality_group_ids[selected_cells]==1)))},
      "unselected_native_bad_cells":int(len(bad_cells)-len(selected_cells))})
  free_parts.append(remainder_free);one_ring_info["inner_back_remainder"]=remainder_info
  if args.block_local_target_repair:
   patch_roster,patch_receipt,patch_offsets,patch_by_node=_build_remainder_patches(
    np.flatnonzero(~initial),selected_cells,
    {r:np.asarray(surfaces[r]["faces"],np.int64) for r in ("O","I","D")},smap,t,
    back_global,fixed,witness_frozen,len(p))
   expected=np.unique(remainder_free)
   observed=np.unique(np.concatenate([x["free_rows"] for x in patch_roster])) if patch_roster else np.empty(0,np.int64)
   if not np.array_equal(expected,observed):raise AssertionError("block-local patch union differs from target mask")
   one_ring_info["block_local_patch_roster"]=patch_receipt
 if args.ordinary_outer_o_local_repair:
  outer_free,outer_info=expand_ordinary_outer_o_once(
   bad_corner_rows,np.asarray(surfaces["O"]["faces"],np.int64),smap["O"],
   fixed_global_rows=fixed,witness_frozen_global_rows=witness_frozen)
  free_parts.append(outer_free);one_ring_info["ordinary_outer_O"]=outer_info
 if not free_parts:
  free_rows,one_ring_info=expand_triangle_ring_once(
   bad_corner_rows,{r:np.asarray(surfaces[r]["faces"],np.int64) for r in ("D","O","I")},smap,
   fixed_global_rows=fixed,witness_frozen_global_rows=witness_frozen)
 else:
  free_rows=np.unique(np.concatenate(free_parts)).astype(np.int64)
  one_ring_info["free_global_rows_union"]=int(len(free_rows))
 free=np.zeros(len(p),bool);free[free_rows]=True
 if args.inner_back_remainder_preflight_only:
  if not args.inner_back_remainder_repair:raise ValueError("inner-back remainder preflight requires its repair mode")
  q=one_ring_info["inner_back_remainder"]
  report={"schema":"h706.inner-back-remainder-quality-preflight.v1","status":"PASS_READ_ONLY_NO_SOLVER",
      "input_state":pin(args.state),"builder_source":pin(Path(__file__)),"back_lineage":back_summary,
      "native_quality_failures_before":int(np.count_nonzero(~initial)),
      "nonpositive_cells_before":int(np.count_nonzero(d<=0)),"global_points":int(len(p)),
      "global_cells":int(len(t)),"free_global_point_rows":int(free.sum()),
      "fixed_global_point_rows":int((~free).sum()),
      "free_rows_by_surface_role":{r:int(np.count_nonzero(free[smap[r][np.unique(surfaces[r]["faces"].reshape(-1))]])) for r in ("O","I","D")},
      "fixed_anchors":anchor_counts,"ordinary_reference":q["ordinary_reference_policy"],
      "eligible_residual_cells":{k:q[k] for k in ("ordinary_reference_bad_cells_with_any_free_corner","ordinary_reference_bad_cells_with_zero_free_corners","ordinary_reference_bad_cells_verified_covered","selected_bad_cells_by_quality_role","unselected_native_bad_cells")},
      "target_free_set":q,"witness_conditioning":witness_info,"cap_m":float(args.trust_cap_m),
      "native_threshold_unchanged":True,"physical_admission":False,"native_steps":0,
      "mass_reconciliation":"PENDING","LP_invocations":0,"source_audits":0}
  np.savez_compressed(out/"free-fixed-global-row-mask.npz",free_rows=np.flatnonzero(free),
      fixed_rows=np.flatnonzero(~free),ordinary_reference_free_rows=ordinary_free,
      selected_bad_cell_rows=selected_cells)
  (out/"preflight.json").write_text(json.dumps(report,indent=2)+"\n")
  compact={"status":report["status"],"native_quality_failures_before":report["native_quality_failures_before"],
      "global_cells":report["global_cells"],"free_global_point_rows":report["free_global_point_rows"],
      "eligible_residual_cells":report["eligible_residual_cells"],"fixed_anchors":anchor_counts}
  print("INNER_BACK_REMAINDER_PREFLIGHT",json.dumps(compact,separators=(",",":")),flush=True)
  return
 if args.ordinary_preflight_only:
  if not (args.ordinary_all_role_local_repair or args.unified_inner_back_ordinary_local_repair):raise ValueError("ordinary preflight requires an all-role or unified mode")
  report={"schema":"h706.ordinary-all-role-local-quality-preflight.v1","status":"PASS_READ_ONLY_NO_SOLVER","input_state":pin(args.state),"builder_source":pin(Path(__file__)),"back_lineage":back_summary,"native_quality_failures_before":int(np.count_nonzero(~initial)),"nonpositive_cells_before":int(np.count_nonzero(d<=0)),"global_points":int(len(p)),"global_cells":int(len(t)),"free_global_point_rows":int(free.sum()),"fixed_global_point_rows":int((~free).sum()),"free_rows_by_surface_role":{r:int(np.count_nonzero(free[smap[r][np.unique(surfaces[r]["faces"].reshape(-1))]])) for r in ("O","I","D")},"fixed_anchors":anchor_counts,"one_ring_expansion":one_ring_info,"witness_conditioning":witness_info,"cap_m":float(args.trust_cap_m),"construction_policy":construction_policy_metadata(args),"policy":("ordinary bad-corner/source-ring freedoms plus I/D-inner and authenticated actual-back Fat full-star freedoms for all bad cells; O ordinary local rows remain movable under the ordinary cap, while O back scope, outer scope, rims and routes stay fixed" if args.unified_inner_back_ordinary_local_repair else "baseline-bad volume corners plus one direct source-triangle ring over O/I/D; current/historical actual Fat back, all scope, rims and registered routes fixed"),"native_threshold_unchanged":True,"physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"}
  np.savez_compressed(out/"free-fixed-global-row-mask.npz",free_rows=np.flatnonzero(free),fixed_rows=np.flatnonzero(~free))
  (out/"preflight.json").write_text(json.dumps(report,indent=2)+"\n")
  print("PREFLIGHT",json.dumps({k:report[k] for k in ("status","native_quality_failures_before","nonpositive_cells_before","global_points","global_cells","free_global_point_rows","fixed_global_point_rows","free_rows_by_surface_role","fixed_anchors")},separators=(",",":")),flush=True)
  return
 print("WITNESS_CONDITIONING",json.dumps(witness_info,separators=(",",":")),flush=True)
 print("LP_INPUT",json.dumps({"global_points":len(p),"cells":len(t),"free_nodes":int(free.sum()),"bad":int((~initial).sum()),"fixed_outer_and_deep":(not args.allow_external_boundary and not args.inner_back_layer_repair and not args.inner_back_remainder_repair and not args.ordinary_all_role_local_repair and not args.unified_inner_back_ordinary_local_repair),"inner_back_layer_repair":bool(args.inner_back_layer_repair or args.inner_back_remainder_repair),"block_local_target_repair":bool(args.block_local_target_repair),"protected_anchors":anchor_counts,"fixed_global_rows":len(fixed),"one_ring_expansion":one_ring_info,"trust_cap_m":args.trust_cap_m,"objective_mode":args.objective_mode},separators=(",",":")),flush=True)
 if args.block_local_target_repair:
  candidate,diag=propose_remainder_block_local(
   p,t,mat,quality_group_ids,d,q,initial,patch_roster,args.trust_cap_m,args.seconds,
   args.patch_seconds,patch_offsets,patch_by_node,separation_rows=separation_rows,
   objective_mode=args.objective_mode,node_fallbacks_per_patch=args.node_fallbacks_per_patch,
   joint_small_patch_fallback=True)
 elif args.unified_inner_back_ordinary_local_repair:
  candidate,diag=propose_remainder_block_local(
   p,t,mat,quality_group_ids,d,q,initial,patch_roster,args.trust_cap_m,args.seconds,
   args.patch_seconds,patch_offsets,patch_by_node,separation_rows=separation_rows,
   objective_mode=args.objective_mode,node_fallbacks_per_patch=args.node_fallbacks_per_patch,
   joint_small_patch_fallback=True,
   ordinary_mode=False,
   extended_250um_construction_trial=bool(args.unified_inner_back_ordinary_250um_construction_trial))
 elif args.ordinary_block_local_quality:
  candidate,diag=propose_remainder_block_local(
   p,t,mat,quality_group_ids,d,q,initial,block_roster,args.trust_cap_m,args.seconds,
   args.patch_seconds,block_offsets,block_by_node,separation_rows=separation_rows,
   objective_mode=args.objective_mode,node_fallbacks_per_patch=args.node_fallbacks_per_patch,
   joint_small_patch_fallback=True,
   ordinary_mode=True)
 else:
  candidate,diag=Q.propose_quality_repair(p,t,mat,free,args.trust_cap_m,time_limit_s=args.seconds,linear_margin=.001,max_backtracks=12,separation_rows=separation_rows,objective_mode=args.objective_mode,quality_group_ids=quality_group_ids)
 (out/"lp-diagnostics.json").write_text(json.dumps(diag,indent=2)+"\n")
 compact_keys=("status","solver_status","bad_before","bad_after","accepted_components","inner_iteration_count","strict_quality_count_progress","strict_bad_score_progress","bad_score_before","bad_score_after","fat_log_deficit_score_before","fat_log_deficit_score_after","skin_log_deficit_score_before","skin_log_deficit_score_after","strict_fat_log_deficit_score_progress","per_step_trust_cap_m","total_trust_cap_m","good_cell_regressions","exact_F32_positive_cells","max_free_node_l1_displacement_bound_m")
 print("LP_STATUS",json.dumps({k:diag.get(k) for k in compact_keys},separators=(",",":")),flush=True)
 if not diag["status"].startswith("PASS"):return
 nd,nq=M.measurements(candidate,t);newgood=M.passes(nd,nq);assert np.all(nd>0) and not np.any(initial&~newgood)
 assert np.array_equal(candidate[list(fixed)].view("u4"),p[list(fixed)].view("u4"))
 if args.inner_back_layer_repair or args.inner_back_remainder_repair or args.ordinary_outer_o_local_repair or args.ordinary_all_role_local_repair or args.unified_inner_back_ordinary_local_repair:
  free_mask=np.zeros(len(p),bool);free_mask[free_rows]=True
  if not np.array_equal(candidate[~free_mask].view("u4"),p[~free_mask].view("u4")):
   raise ValueError("coupled local repair changed a point outside its declared source-face free set")
 result=dict(st);source_surfaces={};cow_writer_receipts={};cow_payload_deltas={}
 for role,a in surfaces.items():
  payload,changed=source_point_step_payload(a,candidate,smap[role],st["source_surfaces"][role],args.trust_cap_m)
  if (args.inner_back_layer_repair or args.inner_back_remainder_repair) and not args.ordinary_outer_o_local_repair and not args.unified_inner_back_ordinary_local_repair and role=="O" and len(changed):raise ValueError("INNER_BACK_LAYER_REPAIR moved the outer O surface")
  if (args.inner_back_layer_repair or args.inner_back_remainder_repair) and not args.ordinary_outer_o_local_repair and not args.unified_inner_back_ordinary_local_repair and role not in ("I","D") and len(changed):raise ValueError("INNER_BACK_LAYER_REPAIR moved an undeclared source role")
  if not len(changed):source_surfaces[role]=st["source_surfaces"][role];continue
  new=payload["points_world_m_f32"]
  dest=out/("candidate-"+role+"-typed-current.npz")
  delta,deleted,delta_report=cow_payload_delta(st["source_surfaces"][role]["path"],payload)
  cow_receipt=COW_COMPLETE.write_npz_clone_complete(st["source_surfaces"][role]["path"],dest,payload)
  cow_payload_deltas["source_"+role]=delta_report
  receipt_path=out/(role.lower()+"-source-cow-writer-receipt.json")
  receipt_path.write_text(json.dumps(cow_receipt,indent=2,sort_keys=True)+"\n")
  cow_writer_receipts["source_"+role]=pin(receipt_path);source_surfaces[role]=pin(dest)
 for vr,a in volumes.items():
  old=a["points_world_m"];new=candidate[vmap[vr]];tt=a["tetrahedra"];dd,qq=M.measurements(new,tt);bad=int((~M.passes(dd,qq)).sum());assert np.all(dd>0)
  payload=dict(a);payload["points_world_m"]=new
  if "points_world_m_f32" in payload:payload["points_world_m_f32"]=new.copy()
  n=len(tt);payload.update(current_geometry_parent_path=np.asarray([st[vr+"_current"]["path"]]),current_geometry_parent_sha256=np.asarray([st[vr+"_current"]["sha256"]]),output_cell_to_input_cell=np.arange(n,dtype=np.int64),input_cell_to_output_cell=np.arange(n,dtype=np.int64),retained_input_cell_rows=np.arange(n,dtype=np.int64),replacement_parent_offsets=np.asarray([0],np.int64),replacement_parent_cell_rows=np.empty(0,np.int64),replacement_tet_offsets=np.asarray([0,0],np.int64),replacement_tetrahedra=np.empty((0,4),np.int64),input_node_to_output_node=np.arange(len(new),dtype=np.int64),output_node_to_input_node=np.arange(len(new),dtype=np.int64),output_point_to_input_point_row=np.arange(len(new),dtype=np.int64),output_point_parent_offsets=np.zeros(len(new)+1,np.int64),output_point_parent_input_cell_rows=np.empty(0,np.int64))
  for role,e in source_surfaces.items():
   payload["typed_"+role+"_candidate_path"]=np.asarray([e["path"]]);payload["typed_"+role+"_candidate_sha256"]=np.asarray([e["sha256"]])
  dest=out/(vr.lower()+"-coupled-quality-point-proposal.npz")
  delta,deleted,delta_report=cow_payload_delta(st[vr+"_current"]["path"],payload)
  cow_receipt=COW_COMPLETE.write_npz_clone_complete(st[vr+"_current"]["path"],dest,payload)
  cow_payload_deltas["volume_"+vr]=delta_report
  receipt_path=out/(vr.lower()+"-volume-cow-writer-receipt.json")
  receipt_path.write_text(json.dumps(cow_receipt,indent=2,sort_keys=True)+"\n")
  cow_writer_receipts["volume_"+vr]=pin(receipt_path)
  result[vr+"_current"]={**st[vr+"_current"],**pin(dest),"native_quality_failures":bad,"nonpositive":0,"geometry_changed_points":int(np.any(new.view("u4")!=old.view("u4"),axis=1).sum())}
  print("QUALITY",vr,bad,flush=True)
 result.update(source_surfaces=source_surfaces,physical_admission=False,native_steps=0,surface_crossings="NOT_YET_AUDITED",source_state=pin(args.state),source_triangle_one_ring_expansion=one_ring_info,coupled_point_repair={"producer":pin(Path(__file__)),"maximum_displacement_m":args.trust_cap_m,"quality_step_policy":"up to 10 continuous SLP steps at total trust cap / 10 each","fixed_outer_and_deep":(not args.allow_external_boundary and not args.inner_back_layer_repair and not args.inner_back_remainder_repair and not args.ordinary_all_role_local_repair and not args.unified_inner_back_ordinary_local_repair),"inner_back_layer_repair":bool(args.inner_back_layer_repair or args.inner_back_remainder_repair),"inner_back_layer_policy":"INNER_BACK_LAYER_REPAIR" if (args.inner_back_layer_repair or args.inner_back_remainder_repair) else None,"inner_back_layer_extended_construction_trial":bool(args.unified_inner_back_ordinary_250um_construction_trial),"inner_back_layer_extended_construction_cap_m":float(args.trust_cap_m) if args.unified_inner_back_ordinary_250um_construction_trial else None,"outer_O_scope_fixed":True,"outer_O_entire_surface_fixed":not bool(args.ordinary_outer_o_local_repair or args.ordinary_all_role_local_repair or args.unified_inner_back_ordinary_local_repair),"ordinary_outer_O_local_repair":bool(args.ordinary_outer_o_local_repair or args.unified_inner_back_ordinary_local_repair),"ordinary_outer_O_policy":"unified bad-corner/source-ring outer O freedoms; outer scope/rims/routes remain fixed" if args.unified_inner_back_ordinary_local_repair else ("bad O corners plus one direct incident O source-triangle ring; scope/rims/routes remain fixed" if args.ordinary_outer_o_local_repair else None),"ordinary_all_role_local_repair":bool(args.ordinary_all_role_local_repair),"unified_inner_back_ordinary_local_repair":bool(args.unified_inner_back_ordinary_local_repair),"ordinary_block_local_quality_fit":bool(args.ordinary_block_local_quality),"ordinary_block_patch_policy":"bad-face-connected patches run sequentially; overlapping free nodes and full stars are allowed only with current/baseline-good revalidation and cumulative per-node trust cap" if args.ordinary_block_local_quality else None,"ordinary_all_role_policy":"baseline-bad volume corners plus one direct source-triangle ring over O/I/D; current/historical actual Fat back, all scope, rims and registered routes remain fixed" if args.ordinary_all_role_local_repair else None,"unified_inner_back_ordinary_policy":"all baseline-bad face components receive ordinary bad-corner/source-ring freedoms unioned with I/D-inner source corners/rings and authenticated actual-back Fat rows in each bad-corner tet star; O back scope, outer scope, rims and routes remain fixed" if args.unified_inner_back_ordinary_local_repair else None,"actual_back_Fat_lineage_protection":back_summary if (args.ordinary_all_role_local_repair or args.inner_back_remainder_repair or args.unified_inner_back_ordinary_local_repair) else None,
"inner_back_remainder_repair":bool(args.inner_back_remainder_repair),"block_local_target_repair":bool(args.block_local_target_repair),
"inner_back_remainder_policy":"only native-bad cells with zero ordinary-v25-free corners; selected bad I/D corners plus one I/D source-face ring, selected volume-interior corners, and authenticated actual-back Fat rows in selected-corner tet star; O source/back, rims/routes fixed" if args.inner_back_remainder_repair else None,"I_D_scope_released_only_inside_bad_corners_and_one_triangle_ring":bool(args.inner_back_layer_repair or args.inner_back_remainder_repair or args.unified_inner_back_ordinary_local_repair),"actual_back_Fat_points_released_only_inside_bad_cell_star":bool(args.inner_back_layer_repair or args.inner_back_remainder_repair or args.unified_inner_back_ordinary_local_repair),"protected_anchors":anchor_counts,"fixed_global_rows":len(fixed),"quality_helper":{"path":str(_Q_PATH),"sha256":_Q_EXPECTED_SHA},"joint_small_patch_slsqp_helper":{"path":str(_JOINT_PATH),"sha256":_JOINT_EXPECTED_SHA},"joint_small_patch_fallback":diag.get("joint_small_patch_fallback",{"enabled":False}),"quality_objective_mode":args.objective_mode,"quality_objective_group_ids":"Skin=0,Fat=1","all_source_roles_rebased_to_actual_point_parent":True,"all_roles_follow_shared_global_F32_rows":True,"source_roles":sorted(surfaces),"diagnostics":pin(out/"lp-diagnostics.json"),"exact_witness_conditioning":witness_info})
 for k in ("operations","independent_replay","exact_context"):result.pop(k,None)
 changed=[r for r in source_surfaces if source_surfaces[r]["sha256"]!=st["source_surfaces"][r]["sha256"]]
 result["npz_cow_append_writer"]=pin(_COW_COMPLETE_PATH)
 result["npz_cow_underlying_npz_writer"]=pin(_COW_NPZ_PATH)
 result["npz_cow_writer_receipts"]=cow_writer_receipts
 result["npz_cow_payload_deltas"]=cow_payload_deltas
 result["npz_cow_append_policy"]="Complete desired NPZ payloads are diffed by the pinned 926b adapter using pinned 925 dtype/shape/order-A byte equality; only changed/new arrays and deletions are emitted through APFS clone-only updates; no full-copy fallback"
 result["coupled_point_repair"]["f32_candidate_backtracking_helper"]={"path":str(_F32_BACKTRACK_PATH),"sha256":_F32_BACKTRACK_SHA}
 result["coupled_point_repair"]["f32_witness_ulp_quantization_helper"]={"path":str(_ULP_PATH),"sha256":_ULP_SHA}
 result["coupled_point_repair"]["f32_candidate_backtracking_policy"]="raw float64 single-node, 2-4 seed-corner and 8-16 authorized seed-vertex-star SLSQP candidates are checked at alpha 1,1/2,...,1/64 on complete incident stars; after alpha rejection, at most four one-ULP coordinate choices on actual regressed-good-cell witness points are tested; existing F32 native, exact sign, separation, cap, and role count/score gates remain unchanged"
 result["coupled_point_repair"].update(construction_policy_metadata(args))
 (out/"state.json").write_text(json.dumps(result,indent=2)+"\n")
 if (args.inner_back_layer_repair or args.inner_back_remainder_repair) and not args.ordinary_outer_o_local_repair and (not changed or not set(changed).issubset({"I","D"})):
  raise ValueError("INNER_BACK_LAYER_REPAIR requires a changed I/D source surface and forbids O changes")
 if args.ordinary_outer_o_local_repair and (not changed or not set(changed).issubset({"O","I","D"})):
  raise ValueError("ordinary O local repair requires a changed O/I/D source surface")
 if (args.ordinary_all_role_local_repair or args.unified_inner_back_ordinary_local_repair) and (not changed or not set(changed).issubset({"O","I","D"})):
  raise ValueError("ordinary all-role repair requires changed O/I/D source rows")
 bounded=bounded_point_repair_entries(args,changed)
 config={"schema":"h706.combined-current-exact-surface-diff-config.v1","baseline_candidates":st["source_surfaces"],"candidate_surfaces":source_surfaces,"changed_roles":changed,"bounded_point_repair":bounded}
 (out/"context-config.json").write_text(json.dumps(config,indent=2)+"\n")
 print("STATE",str(out/"state.json"),"elapsed_s",time.monotonic()-start,flush=True)
if __name__=="__main__":main()
