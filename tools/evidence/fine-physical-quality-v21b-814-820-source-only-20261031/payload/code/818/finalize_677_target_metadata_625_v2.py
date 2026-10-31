#!/usr/bin/env python3
"""Metadata-only paired target/map finalizer for a frozen 677 source-diagonal batch.

Rebuilds Skin and Fat desired oriented chains from current typed O/I/D sources
and authenticated collar walls. It changes no source surface or volume geometry.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np

SCHEMA = "numi.quality677.paired-typed-target-metadata-finalizer.v2-active-source-points"
GEOM_KEYS = ("points_world_m", "points_world_m_f32", "tetrahedra", "material_indices", "boundary_faces")
ALIAS_KEYS = (
    "desired_boundary_faces", "desired_boundary_source_rows", "desired_boundary_source_group",
    "desired_boundary_source_face_rows", "desired_boundary_degenerate_source_rows",
    "mapped_current_boundary_faces", "proposed_current_desired_boundary_faces",
    "proposed_current_desired_boundary_source_group", "proposed_current_desired_boundary_source_rows",
    "proposed_current_desired_boundary_source_face_rows",
)

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def array_sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def read_json(path: Path):
    return json.loads(Path(path).read_text())

def scalar_text(a) -> str:
    x = np.asarray(a)
    if x.size != 1:
        raise ValueError("expected scalar string pin")
    return str(x.reshape(-1)[0])

def exact_point_map(surface_points, volume_points, label: str, required_rows=None) -> np.ndarray:
    sp = np.asarray(surface_points, dtype=np.float32)
    vp = np.asarray(volume_points, dtype=np.float32)
    if sp.ndim != 2 or sp.shape[1] != 3 or vp.ndim != 2 or vp.shape[1] != 3:
        raise ValueError(f"{label}: malformed point arrays")
    if not np.isfinite(sp).all() or not np.isfinite(vp).all():
        raise ValueError(f"{label}: non-finite point")
    vkeys = np.ascontiguousarray(vp).view(np.uint32).reshape(-1, 3)
    if len(np.unique(vkeys, axis=0)) != len(vp):
        raise ValueError(f"{label}: duplicate current volume F32 coordinate rows")
    lookup = {row.tobytes(): i for i, row in enumerate(vkeys)}
    skeys = np.ascontiguousarray(sp).view(np.uint32).reshape(-1, 3)
    out = np.fromiter((lookup.get(row.tobytes(), -1) for row in skeys), dtype=np.int64, count=len(sp))
    required = np.arange(len(sp), dtype=np.int64) if required_rows is None else np.unique(np.asarray(required_rows, np.int64))
    if required.size and (required.min() < 0 or required.max() >= len(sp)):
        raise ValueError(f"{label}: required point row outside typed source domain")
    miss = required[out[required] < 0]
    if len(miss):
        raise ValueError(f"{label}: {len(miss)} active typed points lack exact volume rows; examples {miss[:8].tolist()}")
    valid = out >= 0
    if np.any(valid) and not np.array_equal(vp[out[valid]].view(np.uint32), sp[valid].view(np.uint32)):
        raise ValueError(f"{label}: mapped F32 point bytes differ")
    return out

def mapped_surface_faces(source: dict, volume_points: np.ndarray, label: str):
    if "points_world_m_f32" not in source or "faces" not in source:
        raise ValueError(f"{label}: typed source lacks points_world_m_f32/faces")
    p = np.asarray(source["points_world_m_f32"], np.float32)
    f = np.asarray(source["faces"], np.int64)
    if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0 or f.min() < 0 or f.max() >= len(p):
        raise ValueError(f"{label}: invalid face index domain")
    mp = exact_point_map(p, volume_points, label, required_rows=np.unique(f))
    mf = mp[f]
    if np.any(mf < 0):
        raise ValueError(f"{label}: a face-used point is unmapped")
    if np.any((mf[:, 0] == mf[:, 1]) | (mf[:, 1] == mf[:, 2]) | (mf[:, 2] == mf[:, 0])):
        raise ValueError(f"{label}: exact crosswalk collapsed a face")
    return mp, mf

def directed_chain(faces: np.ndarray) -> Counter:
    out = Counter()
    for face in np.asarray(faces, np.int64).reshape(-1, 3):
        q = tuple(map(int, face))
        key = tuple(sorted(q))
        inversions = int(q[0] > q[1]) + int(q[0] > q[2]) + int(q[1] > q[2])
        out[key] += 1 if inversions % 2 == 0 else -1
        if out[key] == 0:
            del out[key]
    return out

def assert_closed_unique(faces: np.ndarray, label: str):
    f = np.asarray(faces, np.int64)
    if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
        raise ValueError(f"{label}: empty/malformed triangle set")
    canonical = np.sort(f, axis=1)
    if len(np.unique(canonical, axis=0)) != len(f):
        raise ValueError(f"{label}: duplicate unoriented triangle")
    edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]), axis=0)
    keys = np.sort(edges, axis=1)
    signs = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    k, s = keys[order], signs[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(k[1:] != k[:-1], axis=1))]
    counts = np.diff(np.r_[starts, len(k)])
    sums = np.add.reduceat(s, starts)
    bad = (counts != 2) | (sums != 0)
    if np.any(bad):
        examples = k[starts[bad]][:8].tolist()
        raise ValueError(f"{label}: target is not a closed oriented 2-chain; {int(bad.sum())} bad edges, examples {examples}")

def wall_faces(role: str, volume_points: np.ndarray, wall_csv: Path, collar: dict):
    col = "skin_sidewall_faces" if role == "Skin" else "fat_sidewall_faces"
    want = "skin_O_to_I" if role == "Skin" else "fat_I_to_D"
    with Path(wall_csv).open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["role"] == want]
    if len(rows) != 90:
        raise ValueError(f"{role}: expected 90 authenticated {want} wall rows, got {len(rows)}")
    ids = np.asarray([int(r["wall_face_row"]) for r in rows], np.int64)
    expected = np.arange(0, 90, dtype=np.int64) if role == "Skin" else np.arange(90, 180, dtype=np.int64)
    if not np.array_equal(ids, expected):
        raise ValueError(f"{role}: wall source row block is not the expected authenticated range")
    local = np.asarray([[int(r[f"output_node{i}"]) for i in range(3)] for r in rows], np.int64)
    cf = np.asarray(collar[col], np.int64)
    cp = np.asarray(collar["points_world_m"], np.float32)
    if cf.shape != (90, 3) or not np.array_equal(local, cf):
        raise ValueError(f"{role}: CSV wall faces differ from authenticated collar faces")
    if local.min() < 0 or local.max() >= len(cp):
        raise ValueError(f"{role}: wall face point outside collar domain")
    xyz = cp[local]
    wall_points = xyz.reshape(-1, 3)
    wm = exact_point_map(wall_points, volume_points, f"{role} walls", required_rows=np.arange(len(wall_points), dtype=np.int64)).reshape(90, 3)
    if not np.array_equal(np.asarray(volume_points, np.float32)[wm].view(np.uint32), xyz.view(np.uint32)):
        raise ValueError(f"{role}: wall coordinate crosswalk differs")
    return wm, ids, local, xyz

def build_target(role: str, volume_points: np.ndarray, surfaces: dict[str, dict], wall_csv: Path, collar: dict):
    # Existing orientation contract: Skin = O - I + O-to-I walls;
    # Fat = I - D + I-to-D walls.
    if role == "Skin":
        om, of = mapped_surface_faces(surfaces["O"], volume_points, "Skin/O")
        im, inf = mapped_surface_faces(surfaces["I"], volume_points, "Skin/I")
        wf, wall_ids, wlocal, wxyz = wall_faces(role, volume_points, wall_csv, collar)
        target = np.vstack((of, inf[:, [0, 2, 1]], wf)).astype(np.int64, copy=False)
        groups = np.concatenate((np.zeros(len(of), np.int8), np.ones(len(inf), np.int8), np.full(len(wf), 2, np.int8)))
        rows = np.concatenate((np.arange(len(of), dtype=np.int64), np.arange(len(inf), dtype=np.int64), wall_ids))
        point_maps = {"O": om, "I": im}
    elif role == "Fat":
        im, inf = mapped_surface_faces(surfaces["I"], volume_points, "Fat/I")
        dm, df = mapped_surface_faces(surfaces["D"], volume_points, "Fat/D")
        wf, wall_ids, wlocal, wxyz = wall_faces(role, volume_points, wall_csv, collar)
        target = np.vstack((inf, df[:, [0, 2, 1]], wf)).astype(np.int64, copy=False)
        groups = np.concatenate((np.zeros(len(inf), np.int8), np.ones(len(df), np.int8), np.full(len(wf), 2, np.int8)))
        rows = np.concatenate((np.arange(len(inf), dtype=np.int64), np.arange(len(df), dtype=np.int64), wall_ids))
        point_maps = {"I": im, "D": dm}
    else:
        raise ValueError(f"unsupported region {role}")
    assert_closed_unique(target, role + " typed target")
    if len(groups) != len(target) or len(rows) != len(target):
        raise ValueError(f"{role}: source metadata lengths do not match target order")
    return target, groups, rows, point_maps, wall_ids, wlocal, wxyz

def sync_volume_metadata(volume: dict[str, np.ndarray], role: str, target: np.ndarray,
                         groups: np.ndarray, rows: np.ndarray, point_maps: dict[str, np.ndarray],
                         wall_local: np.ndarray, wall_xyz: np.ndarray,
                         parent_path: Path, parent_sha: str, source_pins: dict):
    out = dict(volume)
    n = len(np.asarray(volume["points_world_m_f32"], np.float32))
    for key in ALIAS_KEYS:
        if key in volume:
            out["metadata_finalizer_previous_" + key] = np.asarray(volume[key]).copy()
    actual = np.asarray(volume["boundary_faces"], np.int64)
    out.update({
        "desired_boundary_faces": np.asarray(target, np.int64).copy(),
        "desired_boundary_source_group": np.asarray(groups, np.int8).copy(),
        "desired_boundary_source_rows": np.asarray(rows, np.int64).copy(),
        "desired_boundary_source_face_rows": np.asarray(rows, np.int64).copy(),
        "desired_boundary_degenerate_source_rows": np.empty(0, np.int64),
        "mapped_current_boundary_faces": actual.copy(),
        "proposed_current_desired_boundary_faces": np.asarray(target, np.int64).copy(),
        "proposed_current_desired_boundary_source_group": np.asarray(groups, np.int8).copy(),
        "proposed_current_desired_boundary_source_rows": np.asarray(rows, np.int64).copy(),
        "proposed_current_desired_boundary_source_face_rows": np.asarray(rows, np.int64).copy(),
        "output_point_to_input_point_row": np.arange(n, dtype=np.int64),
        "input_node_to_output_node": np.arange(n, dtype=np.int64),
        "output_node_to_input_node": np.arange(n, dtype=np.int64),
        "output_point_parent_offsets": np.zeros(n + 1, np.int64),
        "output_point_parent_input_cell_rows": np.empty(0, np.int64),
        "source_interior_vertex_quotient": np.arange(n, dtype=np.int64),
        "metadata_finalizer_schema": np.asarray(SCHEMA),
        "metadata_finalizer_region": np.asarray(role),
        "metadata_finalizer_status": np.asarray("REBOUND_TO_CURRENT_TYPED_SURFACES_AND_WALLS_METADATA_ONLY"),
        "metadata_finalizer_parent_candidate_path": np.asarray(str(parent_path)),
        "metadata_finalizer_parent_candidate_sha256": np.asarray(parent_sha),
        "metadata_finalizer_point_map_status": np.asarray("IDENTITY_NO_POINT_CHANGE_THIS_STEP"),
        "metadata_finalizer_point_parent_status": np.asarray("NO_NEW_POINTS_IN_THIS_STEP"),
        "metadata_finalizer_source_asset_sha256_json": np.asarray(json.dumps(source_pins, sort_keys=True)),
        "target_metadata_physical_admission": np.asarray(False),
        "target_metadata_native_steps": np.asarray(0, np.int64),
    })
    if len(rows) != len(target) or not np.array_equal(out["desired_boundary_faces"], out["proposed_current_desired_boundary_faces"]):
        raise ValueError(f"{role}: target alias synchronization failed")
    if role == "Skin":
        out["skin_current_target_rebind_schema"] = np.asarray(SCHEMA)
        out["skin_current_target_rebind_O_point_to_volume_rows"] = np.asarray(point_maps["O"], np.int64)
        out["skin_current_target_rebind_I_point_to_volume_rows"] = np.asarray(point_maps["I"], np.int64)
        out["skin_current_target_rebind_wall_faces_volume_rows"] = np.asarray(target[-90:], np.int64)
        out["skin_current_target_rebind_wall_faces_collar_point_rows"] = np.asarray(wall_local, np.int64)
        out["skin_current_target_rebind_wall_face_points_world_m_f32"] = np.asarray(wall_xyz, np.float32)
    else:
        out["fat_current_target_rebind_schema"] = np.asarray(SCHEMA)
        out["fat_current_target_rebind_I_point_to_volume_rows"] = np.asarray(point_maps["I"], np.int64)
        out["fat_current_target_rebind_D_point_to_volume_rows"] = np.asarray(point_maps["D"], np.int64)
        out["fat_current_target_rebind_wall_faces_volume_rows"] = np.asarray(target[-90:], np.int64)
        out["fat_current_target_rebind_wall_faces_collar_point_rows"] = np.asarray(wall_local, np.int64)
        out["fat_current_target_rebind_wall_face_points_world_m_f32"] = np.asarray(wall_xyz, np.float32)
    return out

def self_test():
    orphan_surface=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[4,4,4]],np.float32)
    orphan_volume=orphan_surface[:4].copy()
    orphan_map=exact_point_map(orphan_surface,orphan_volume,"orphan-control",required_rows=np.asarray([0,1,2,3]))
    assert orphan_map.tolist()==[0,1,2,3,-1]
    try:
        exact_point_map(orphan_surface,orphan_volume,"orphan-negative-control")
    except ValueError:
        pass
    else:
        raise AssertionError("unreferenced mismatch must not pass when marked required")
    p = np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], np.float32)
    tet = np.asarray([[0,1,2,3]], np.int64)
    b = np.asarray([[1,2,3],[0,3,2],[0,1,3],[0,2,1]], np.int64)
    assert_closed_unique(b, "fixture")
    stale = np.asarray([[0,1,3]], np.int64)
    v = {"points_world_m":p.copy(),"points_world_m_f32":p.copy(),"tetrahedra":tet.copy(),
         "material_indices":np.asarray([16],np.int32),"boundary_faces":b.copy(),
         "desired_boundary_faces":stale.copy(),"mapped_current_boundary_faces":stale.copy(),
         "proposed_current_desired_boundary_faces":stale.copy(),
         "input_node_to_output_node":np.asarray([1,0,2,3],np.int64),
         "output_node_to_input_node":np.asarray([1,0,2,3],np.int64),
         "output_point_to_input_point_row":np.asarray([1,0,2,3],np.int64),
         "output_point_parent_offsets":np.asarray([0,1,1,1,1],np.int64),
         "output_point_parent_input_cell_rows":np.asarray([0],np.int64),
         "source_interior_vertex_quotient":np.asarray([1,0,2,3],np.int64)}
    groups=np.zeros(4,np.int8); rows=np.arange(4,dtype=np.int64)
    sync=sync_volume_metadata(v,"Skin",b,groups,rows,{"O":np.arange(4),"I":np.arange(4)},
         np.zeros((90,3),np.int64),np.zeros((90,3),np.float32),Path("/parent.npz"),"a"*64,{})
    assert np.array_equal(sync["desired_boundary_faces"],b)
    assert np.array_equal(sync["mapped_current_boundary_faces"],b)
    assert np.array_equal(sync["proposed_current_desired_boundary_faces"],b)
    for k in ("input_node_to_output_node","output_node_to_input_node","output_point_to_input_point_row","source_interior_vertex_quotient"):
        assert np.array_equal(sync[k],np.arange(4))
    assert np.array_equal(sync["output_point_parent_offsets"],np.zeros(5,np.int64))
    assert len(sync["output_point_parent_input_cell_rows"])==0
    assert np.array_equal(sync["metadata_finalizer_previous_desired_boundary_faces"],stale)
    assert np.array_equal(v["desired_boundary_faces"],stale)
    print("PASS tiny target alias, immediate-map, and parent-history controls")

def run(state_path: Path, config_path: Path, output: Path):
    state_path, config_path, output = Path(state_path), Path(config_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    state=read_json(state_path); parent_ref=state["source_state"]; parent_state_path=Path(parent_ref["path"])
    if sha_file(parent_state_path)!=parent_ref["sha256"]: raise ValueError("source_state pin mismatch")
    parent_state=read_json(parent_state_path); cfg=read_json(config_path)
    for name in ("wall_csv","collar","wall_receipt"):
        if sha_file(Path(cfg[name]["path"]))!=cfg[name]["sha256"]: raise ValueError(f"wall input pin mismatch: {name}")
    wall_receipt=read_json(Path(cfg["wall_receipt"]["path"]))
    if wall_receipt.get("status")!="PASS_COLLAR_AND_PROTECTED149_REINDEX_SOURCE_ONLY" or wall_receipt.get("physical_admission") is not False:
        raise ValueError("wall receipt is not the authenticated source-only collar pass")
    wall_csv,collar_path=Path(cfg["wall_csv"]["path"]),Path(cfg["collar"]["path"]); collar=load_npz(collar_path)
    surface_data={}
    source_pins={"677_state":{"path":str(state_path),"sha256":sha_file(state_path)},
      "source_state":{"path":str(parent_state_path),"sha256":parent_ref["sha256"]},
      "wall_csv":{"path":str(wall_csv),"sha256":sha_file(wall_csv)},
      "collar":{"path":str(collar_path),"sha256":sha_file(collar_path)},
      "wall_receipt":{"path":cfg["wall_receipt"]["path"],"sha256":cfg["wall_receipt"]["sha256"]}}
    for role in ("O","I","D"):
        entry=state["source_surfaces"][role]; path=Path(entry["path"])
        if sha_file(path)!=entry["sha256"]: raise ValueError(f"current typed {role} pin mismatch")
        surf=load_npz(path)
        if "source_surfaces" in parent_state:
            parent_entry=parent_state["source_surfaces"][role]; parent_path=Path(parent_entry["path"])
            if sha_file(parent_path)!=parent_entry["sha256"]: raise ValueError(f"{role}: direct typed parent file hash mismatch")
            if scalar_text(surf["typed_parent_candidate_path"])!=parent_entry["path"] or scalar_text(surf["typed_parent_candidate_sha256"])!=parent_entry["sha256"]:
                raise ValueError(f"{role}: direct typed parent pin mismatch")
            parent_surf=load_npz(parent_path)
            if not np.array_equal(surf["points_world_m_f32"].view(np.uint32),parent_surf["points_world_m_f32"].view(np.uint32)):
                raise ValueError(f"{role}: points changed despite identity immediate point map")
            if not np.array_equal(surf["source_input_face_rows"],np.arange(len(surf["faces"]),dtype=np.int64)):
                raise ValueError(f"{role}: immediate face-row map is not identity")
            for key in ("face_source_original_id","face_source_identity0","face_source_identity1"):
                if key in surf and surf[key].shape!=(len(surf["faces"]),): raise ValueError(f"{role}: source identity field length mismatch: {key}")
            for key in ("quality_star_patch_union_offsets","face_ancestry_offsets"):
                if key in surf and surf[key].shape!=(len(surf["faces"])+1,): raise ValueError(f"{role}: source ancestry CSR length mismatch: {key}")
        surface_data[role]=surf;source_pins[role]={"path":str(path),"sha256":entry["sha256"]}
    output.mkdir(parents=True); out_state=dict(state)
    out_state["metadata_finalizer"]={"schema":SCHEMA,"producer_path":str(Path(__file__).resolve()),"producer_sha256":sha_file(Path(__file__)),
        "parent_state":parent_ref,"wall_receipt":cfg["wall_receipt"],"geometry_changed":False,"physical_admission":False,
        "native_steps":0,"surface_crossings":"PENDING_SEPARATE_678_CONTEXT_AUDIT"}
    reports={}
    for role in ("Skin","Fat"):
        entry=state[role+"_current"]; inpath=Path(entry["path"])
        if sha_file(inpath)!=entry["sha256"]: raise ValueError(f"{role}: 677 volume pin mismatch")
        vol=load_npz(inpath); points=np.asarray(vol["points_world_m_f32"],np.float32)
        parent_vol_entry=parent_state.get(role+"_current")
        if not parent_vol_entry: raise ValueError(f"parent state missing {role} current volume")
        parent_vol_path=Path(parent_vol_entry["path"])
        if sha_file(parent_vol_path)!=parent_vol_entry["sha256"]: raise ValueError(f"{role}: prior volume pin mismatch")
        parent_vol=load_npz(parent_vol_path)
        if not np.array_equal(points.view(np.uint32),np.asarray(parent_vol["points_world_m_f32"],np.float32).view(np.uint32)):
            raise ValueError(f"{role}: point geometry changed in 677")
        if not np.array_equal(vol["points_world_m"].view(np.uint32),points.view(np.uint32)):
            raise ValueError(f"{role}: world and emitted F32 point views differ")
        for sr in ("O","I","D"):
            key="typed_"+sr+"_candidate_sha256"; pathkey="typed_"+sr+"_candidate_path"
            if key in vol and scalar_text(vol[key])!=state["source_surfaces"][sr]["sha256"]: raise ValueError(f"{role}: volume typed source hash mismatch for {sr}")
            if pathkey in vol and scalar_text(vol[pathkey])!=state["source_surfaces"][sr]["path"]: raise ValueError(f"{role}: volume typed source path mismatch for {sr}")
        target,groups,rows,pmap,wall_ids,wall_local,wall_xyz=build_target(role,points,surface_data,wall_csv,collar)
        actual=np.asarray(vol["boundary_faces"],np.int64)
        if directed_chain(target)!=directed_chain(actual): raise ValueError(f"{role}: current typed source target differs from actual 677 boundary chain")
        if len(target)!=len(actual): raise ValueError(f"{role}: target and actual boundary face counts differ")
        srcpins=dict(source_pins);srcpins[role+"_volume_parent"]={"path":str(inpath),"sha256":entry["sha256"]}
        outvol=sync_volume_metadata(vol,role,target,groups,rows,pmap,wall_local,wall_xyz,inpath,entry["sha256"],srcpins)
        outpath=output/(role.lower()+"-paired-target-metadata-finalized.npz");np.savez_compressed(outpath,**outvol)
        with np.load(outpath,allow_pickle=False) as z:
            for key in GEOM_KEYS:
                if array_sha(z[key])!=array_sha(vol[key]): raise ValueError(f"{role}: finalizer changed geometry array {key}")
            for key in ("replacement_parent_offsets","replacement_parent_cell_rows","replacement_tet_offsets","replacement_tetrahedra"):
                if key in vol and array_sha(z[key])!=array_sha(vol[key]): raise ValueError(f"{role}: finalizer changed cell ancestry {key}")
            if not np.array_equal(z["desired_boundary_faces"],target): raise ValueError(f"{role}: output target mismatch")
        out_state[role+"_current"]={**entry,"path":str(outpath),"sha256":sha_file(outpath),"metadata_parent_path":str(inpath),
          "metadata_parent_sha256":entry["sha256"],"target_face_count":int(len(target)),"target_boundary_match":True}
        reports[role]={"parent_candidate":{"path":str(inpath),"sha256":entry["sha256"]},"output_candidate":{"path":str(outpath),"sha256":sha_file(outpath)},
          "point_count":int(len(points)),"cell_count":int(len(vol["tetrahedra"])),"target_face_count":int(len(target)),
          "actual_boundary_face_count":int(len(actual)),"actual_equals_current_typed_target_oriented_chain":True,
          "source_group_counts":{str(int(x)):int(np.count_nonzero(groups==x)) for x in np.unique(groups)},
          "source_rows_aligned_to_target_order":True,"volume_point_maps_identity_for_this_step":True,"unmapped_unreferenced_source_point_rows":{k:int(np.count_nonzero(np.asarray(v)<0)) for k,v in pmap.items()},"new_point_parent_count":0,
          "geometry_changed":False,"physical_admission":False,"native_steps":0,
          "stale_parent_aliases_preserved_as_metadata_finalizer_previous_fields":True}
    out_state["metadata_finalizer"]["regions"]=reports
    state_out=output/"state.json";state_out.write_text(json.dumps(out_state,indent=2,sort_keys=True)+"\n")
    report={"schema":SCHEMA,"status":"PASS_PAIRED_TYPED_TARGET_METADATA_REBOUND_SOURCE_ONLY","producer_path":str(Path(__file__).resolve()),
      "producer_sha256":sha_file(Path(__file__)),"input_state":{"path":str(state_path),"sha256":sha_file(state_path)},
      "source_state":parent_ref,"wall_csv":source_pins["wall_csv"],"collar":source_pins["collar"],"wall_receipt":source_pins["wall_receipt"],
      "source_surfaces":{k:source_pins[k] for k in ("O","I","D")},"regions":reports,
      "output_state":{"path":str(state_out),"sha256":sha_file(state_out)},"geometry_changed":False,"native_steps":0,"mass_mapping":"NOT_RUN",
      "surface_crossings":"PENDING_SEPARATE_678_CONTEXT_AUDIT",
      "limitations":"Metadata-only target/source-row rebinding. No new source collision audit, native admission, or mass claim."}
    report_path=output/"metadata-finalizer-report.json";report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":report["status"],"report":str(report_path),"report_sha256":sha_file(report_path),
      "state":str(state_out),"regions":reports},indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--state",type=Path);ap.add_argument("--wall-config",type=Path)
    ap.add_argument("--output",type=Path);ap.add_argument("--self-test",action="store_true");a=ap.parse_args()
    if a.self_test:self_test();return
    if not (a.state and a.wall_config and a.output):ap.error("--state, --wall-config, and --output are required unless --self-test")
    run(a.state,a.wall_config,a.output)

if __name__=="__main__":
    main()

