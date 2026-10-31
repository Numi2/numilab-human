#!/usr/bin/env python3
"""Metadata-only SOURCE+VOLUME rebase for a frozen current quality state.

Outputs identity maps to each exact raw-current parent, preserves absolute
ancestry arrays byte-for-byte, clears per-step source operation rows, and
rebuilds current target aliases from the pinned typed sources and collar walls.
No geometry, collision, native-admission, or mass operation is performed.
"""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
HELPER = HERE / "finalize_677_target_metadata_625_v2.py"
HELPER_SHA256 = "af0333a7832cc136d15c5a24a169d69d041bc1e4bbf6813fd6b31f7740189756"
SCHEMA = "numi.quality709.current-source-volume-metadata-rebase.v2"
SGEOM = ("points_world_m", "points_world_m_f32", "faces")
VGEOM = ("points_world_m", "points_world_m_f32", "tetrahedra", "material_indices", "boundary_faces")
SUNIONS = ("quality_star_patch_union_", "face_ancestry_")
VSTEP_FIELDS = ("replacement_parent_offsets", "replacement_parent_cell_rows", "replacement_tet_offsets",
                "replacement_tetrahedra", "output_point_parent_offsets", "output_point_parent_input_cell_rows")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def ah(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def read(path): return json.loads(Path(path).read_text())
def val(a):
    a=np.asarray(a)
    if a.size!=1: raise ValueError("expected scalar pin")
    return str(a.reshape(-1)[0])
def settext(d,k,v):
    old=d.get(k); d[k]=np.asarray(v) if old is not None and np.asarray(old).ndim==0 else np.asarray([v])
def helper():
    if sha(HELPER)!=HELPER_SHA256: raise ValueError("pinned 677 target helper hash mismatch")
    s=importlib.util.spec_from_file_location("quality709_target_builder",HELPER)
    if s is None or s.loader is None: raise ImportError(str(HELPER))
    m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
def equal_geometry(a,b,keys,label):
    before={k:ah(a[k]) for k in keys if k in a};after={k:ah(b[k]) for k in keys if k in b}
    if before!=after: raise ValueError(f"{label}: geometry bytes changed")
    return before
def union_hashes(data,prefixes): return {k:ah(data[k]) for k in data if k.startswith(prefixes)}
def verify_unions(data,want,label):
    for k,h in want.items():
        if k not in data or ah(data[k])!=h: raise ValueError(f"{label}: ancestry changed: {k}")

def clear_source_ops(d,nfaces):
    out={}
    for k in list(d):
        if k.startswith("source_") and k.endswith("_operation_rows"):
            if np.asarray(d[k]).shape!=(nfaces,): raise ValueError(f"{k} does not index source faces")
            d[k]=np.full(nfaces,-1,np.int64);out[k]=nfaces
    for k in ("source_collapse_operation_rows","source_diagonal_flip_operation_rows"):
        if k not in d: d[k]=np.full(nfaces,-1,np.int64);out[k]=nfaces
    return out

def rebase_source(parent,role,parent_path,parent_sha):
    d=dict(parent);nf=len(d["faces"]);npnt=len(d["points_world_m_f32"])
    prior={k:val(d[k]) for k in ("typed_parent_candidate_path","typed_parent_candidate_sha256") if k in d}
    unions=union_hashes(d,SUNIONS)
    for k,n in (("source_input_face_rows",nf),("quality_star_face_output_to_input_row",nf),
                ("source_input_vertex_quotient",npnt),("quality_star_point_output_to_input_row",npnt),
                ("face_output_to_parent_rows",nf),("point_output_to_parent_rows",npnt),
                ("face_output_to_input_rows",nf),("point_output_to_input_rows",npnt)):
        if k in d or k in ("source_input_face_rows","quality_star_face_output_to_input_row",
                           "source_input_vertex_quotient","quality_star_point_output_to_input_row"):
            d[k]=np.arange(n,dtype=np.int64)
    cleared=clear_source_ops(d,nf)
    settext(d,"metadata_finalizer_709_previous_typed_parent_path",prior.get("typed_parent_candidate_path",""))
    settext(d,"metadata_finalizer_709_previous_typed_parent_sha256",prior.get("typed_parent_candidate_sha256",""))
    settext(d,"typed_parent_candidate_path",str(parent_path));settext(d,"typed_parent_candidate_sha256",parent_sha)
    settext(d,"metadata_finalizer_709_schema",SCHEMA)
    settext(d,"metadata_finalizer_709_status","IDENTITY_MAP_TO_RAW_TYPED_PARENT_NO_SOURCE_GEOMETRY_CHANGE")
    settext(d,"metadata_finalizer_709_absolute_union_sha256_json",json.dumps(unions,sort_keys=True))
    verify_unions(d,unions,role)
    return d,{"points":npnt,"faces":nf,"cleared_operation_maps":cleared,"absolute_union_hashes":unions,"previous_typed_parent":prior}

def rebase_volume(parent,role,parent_path,parent_sha,target,groups,rows,pmap,wall_local,wall_xyz,source_out,assets):
    d=dict(parent);npnt=len(parent["points_world_m_f32"]);nc=len(parent["tetrahedra"])
    actual=np.asarray(parent["boundary_faces"],np.int64)
    aliases=("desired_boundary_faces","desired_boundary_source_rows","desired_boundary_source_group",
             "desired_boundary_source_face_rows","desired_boundary_degenerate_source_rows","mapped_current_boundary_faces",
             "proposed_current_desired_boundary_faces","proposed_current_desired_boundary_source_group",
             "proposed_current_desired_boundary_source_rows","proposed_current_desired_boundary_source_face_rows")
    old_aliases={k:ah(parent[k]) for k in aliases if k in parent}
    old_step_maps={k:{"sha256":ah(parent[k]),"shape":list(np.asarray(parent[k]).shape),"dtype":str(np.asarray(parent[k]).dtype)}
                   for k in VSTEP_FIELDS if k in parent}
    old_union_metadata={k:val(parent[k]) for k in ("canonical_parent_union_schema","canonical_parent_union_operation_order",
                         "canonical_parent_union_physical_status") if k in parent}
    d.update(desired_boundary_faces=np.asarray(target,np.int64).copy(),desired_boundary_source_group=np.asarray(groups,np.int8).copy(),
      desired_boundary_source_rows=np.asarray(rows,np.int64).copy(),desired_boundary_source_face_rows=np.asarray(rows,np.int64).copy(),
      desired_boundary_degenerate_source_rows=np.empty(0,np.int64),mapped_current_boundary_faces=actual.copy(),
      proposed_current_desired_boundary_faces=np.asarray(target,np.int64).copy(),
      proposed_current_desired_boundary_source_group=np.asarray(groups,np.int8).copy(),
      proposed_current_desired_boundary_source_rows=np.asarray(rows,np.int64).copy(),
      proposed_current_desired_boundary_source_face_rows=np.asarray(rows,np.int64).copy(),
      output_cell_to_input_cell=np.arange(nc,dtype=np.int64),input_cell_to_output_cell=np.arange(nc,dtype=np.int64),
      retained_input_cell_rows=np.arange(nc,dtype=np.int64),input_node_to_output_node=np.arange(npnt,dtype=np.int64),
      output_node_to_input_node=np.arange(npnt,dtype=np.int64),output_point_to_input_point_row=np.arange(npnt,dtype=np.int64),
      source_interior_vertex_quotient=np.arange(npnt,dtype=np.int64),
      replacement_parent_offsets=np.asarray([0],np.int64),replacement_parent_cell_rows=np.empty(0,np.int64),
      replacement_tet_offsets=np.asarray([0,0],np.int64),replacement_tetrahedra=np.empty((0,4),np.int64),
      output_point_parent_offsets=np.zeros(npnt+1,np.int64),output_point_parent_input_cell_rows=np.empty(0,np.int64))
    if "canonical_parent_union_operation_order" in parent:
        settext(d,"canonical_parent_union_operation_order","metadata-only-identity-rebase-709-v2")
    cleared={}
    for k in list(d):
        if k.startswith("source_") and k.endswith("_operation_rows"):
            if np.asarray(d[k]).shape!=(nc,): raise ValueError(f"{role}: ambiguous volume op map {k}")
            d[k]=np.full(nc,-1,np.int64);cleared[k]=nc
    for r in ("O","I","D"):
        p,h=source_out[r];settext(d,f"typed_{r}_candidate_path",str(p));settext(d,f"typed_{r}_candidate_sha256",h)
    hashes={"schema":SCHEMA,"base_candidate":parent_sha,"typed_O":source_out["O"][1],"typed_I":source_out["I"][1],"typed_D":source_out["D"][1],
      "wall_csv":assets["wall_csv"]["sha256"],"collar":assets["collar"]["sha256"],"wall_receipt":assets["wall_receipt"]["sha256"]}
    paths={"base_candidate":str(parent_path),**{f"typed_{r}":str(source_out[r][0]) for r in ("O","I","D")},
      **{k:v["path"] for k,v in assets.items() if k in ("wall_csv","collar","wall_receipt")}}
    prefix="skin" if role=="Skin" else "fat"
    settext(d,f"{prefix}_current_target_rebind_schema",SCHEMA)
    settext(d,f"{prefix}_current_target_rebind_source_asset_sha256_json",json.dumps(hashes,sort_keys=True))
    settext(d,f"{prefix}_current_target_rebind_source_asset_paths_json",json.dumps(paths,sort_keys=True))
    if role=="Skin":
        d["skin_current_target_rebind_O_point_to_volume_rows"]=np.asarray(pmap["O"],np.int64)
        d["skin_current_target_rebind_I_point_to_volume_rows"]=np.asarray(pmap["I"],np.int64)
    else:
        d["fat_current_target_rebind_I_point_to_volume_rows"]=np.asarray(pmap["I"],np.int64)
        d["fat_current_target_rebind_D_point_to_volume_rows"]=np.asarray(pmap["D"],np.int64)
    d[f"{prefix}_current_target_rebind_wall_faces_volume_rows"]=np.asarray(target[-90:],np.int64)
    d[f"{prefix}_current_target_rebind_wall_faces_collar_point_rows"]=np.asarray(wall_local,np.int64)
    d[f"{prefix}_current_target_rebind_wall_face_points_world_m_f32"]=np.asarray(wall_xyz,np.float32)
    for k,v in (("current_geometry_parent_path",str(parent_path)),("current_geometry_parent_sha256",parent_sha),
        ("metadata_finalizer_parent_candidate_path",str(parent_path)),("metadata_finalizer_parent_candidate_sha256",parent_sha),
        ("metadata_finalizer_schema",SCHEMA),("metadata_finalizer_region",role),
        ("metadata_finalizer_status","IDENTITY_MAP_TO_RAW_VOLUME_PARENT_NO_VOLUME_GEOMETRY_CHANGE"),
        ("metadata_finalizer_point_map_status","IDENTITY_TO_RAW_CURRENT_VOLUME_PARENT"),
        ("metadata_finalizer_cell_map_status","IDENTITY_TO_RAW_CURRENT_VOLUME_PARENT"),
        ("metadata_finalizer_point_parent_status","EMPTY_CURRENT_STEP_PARENT_CSR_PRIOR_PARENT_PINNED"),
        ("metadata_finalizer_source_asset_sha256_json",json.dumps(assets,sort_keys=True)),
        ("metadata_finalizer_709_previous_immediate_map_sha256_json",json.dumps(old_step_maps,sort_keys=True)),
        ("metadata_finalizer_709_previous_canonical_parent_union_metadata_json",json.dumps(old_union_metadata,sort_keys=True)),
        ("metadata_finalizer_709_previous_target_alias_sha256_json",json.dumps(old_aliases,sort_keys=True))): settext(d,k,v)
    d["target_metadata_physical_admission"]=np.asarray(False);d["target_metadata_native_steps"]=np.asarray(0,np.int64)
    return d,{"points":npnt,"cells":nc,"target_faces":len(target),"cleared_operation_maps":cleared,
      "previous_immediate_map_hashes":old_step_maps,"previous_canonical_parent_union_metadata":old_union_metadata,
      "previous_target_alias_hashes":old_aliases}

def unwrap(raw):
    k="current_verified_state_20261031"
    return (k,raw[k]) if k in raw and isinstance(raw[k],dict) else (None,raw)

def check_walls(cfg):
    for k in ("wall_csv","collar","wall_receipt"):
        if sha(Path(cfg[k]["path"]))!=cfg[k]["sha256"]:raise ValueError(f"wall input pin mismatch: {k}")
    r=read(cfg["wall_receipt"]["path"])
    if r.get("status")!="PASS_COLLAR_AND_PROTECTED149_REINDEX_SOURCE_ONLY" or r.get("physical_admission") is not False:
        raise ValueError("unauthenticated collar/wall receipt")

def run(state_path,config_path,outdir):
    state_path=Path(state_path);config_path=Path(config_path);outdir=Path(outdir)
    if outdir.exists():raise FileExistsError(outdir)
    m=helper();raw=read(state_path);wrap,st=unwrap(raw);cfg=read(config_path);check_walls(cfg)
    wall_csv=Path(cfg["wall_csv"]["path"]);collar_path=Path(cfg["collar"]["path"]);collar=load(collar_path)
    outdir.mkdir(parents=True);input_state={"path":str(state_path),"sha256":sha(state_path)}
    assets={"input_state":input_state,"wall_csv":{"path":str(wall_csv),"sha256":sha(wall_csv)},
      "collar":{"path":str(collar_path),"sha256":sha(collar_path)},"wall_receipt":cfg["wall_receipt"],
      "producer":{"path":str(Path(__file__).resolve()),"sha256":sha(Path(__file__))},
      "target_helper":{"path":str(HELPER),"sha256":HELPER_SHA256}}
    sin={};sout={};sdata={};sreport={}
    for r in ("O","I","D"):
        e=st["source_surfaces"][r];p=Path(e["path"]);h=e["sha256"]
        if sha(p)!=h:raise ValueError(f"typed {r} source pin mismatch")
        a=load(p);b,detail=rebase_source(a,r,p,h)
        upstream={}
        if "source_current_candidate_path" in a and "source_current_candidate_sha256" in a:
            upstream={"path":val(a["source_current_candidate_path"]),"sha256":val(a["source_current_candidate_sha256"])}
        b["metadata_finalizer_709_source_asset_bindings_json"]=np.asarray(json.dumps({
          "schema":SCHEMA,"source_role":r,"input_state":input_state,
          "raw_typed_parent":{"path":str(p),"sha256":h},"upstream_source_current_candidate":upstream,
          "producer":assets["producer"],"target_helper":assets["target_helper"]},sort_keys=True))
        op=outdir/f"{r.lower()}-typed-identity-rebased.npz";np.savez_compressed(op,**b);saved=load(op)
        gh=equal_geometry(a,saved,SGEOM,r);verify_unions(saved,detail["absolute_union_hashes"],r+" saved")
        for k,n in (("source_input_face_rows",len(a["faces"])),("quality_star_face_output_to_input_row",len(a["faces"])),
          ("source_input_vertex_quotient",len(a["points_world_m_f32"])),("quality_star_point_output_to_input_row",len(a["points_world_m_f32"]))):
            if not np.array_equal(saved[k],np.arange(n,dtype=np.int64)):raise ValueError(f"{r}: nonidentity {k}")
        sin[r]={"path":str(p),"sha256":h};sout[r]=(op,sha(op));sdata[r]=saved
        sreport[r]={**detail,"input":sin[r],"output":{"path":str(op),"sha256":sout[r][1]},"geometry_hashes":gh,"geometry_bitwise_equal":True}
    assets["raw_typed_sources"]=sin;assets["rebased_typed_sources"]={r:{"path":str(p),"sha256":h} for r,(p,h) in sout.items()}
    outcore=dict(st);outcore["source_surfaces"]={r:{**st["source_surfaces"][r],"path":str(sout[r][0]),"sha256":sout[r][1],
      "metadata_parent_path":sin[r]["path"],"metadata_parent_sha256":sin[r]["sha256"]} for r in ("O","I","D")}
    vreport={}
    for r in ("Skin","Fat"):
        e=st[r+"_current"];p=Path(e["path"]);h=e["sha256"]
        if sha(p)!=h:raise ValueError(f"{r} volume source pin mismatch")
        a=load(p);points=np.asarray(a["points_world_m_f32"],np.float32)
        for sr in ("O","I","D"):
            kp=f"typed_{sr}_candidate_path";kh=f"typed_{sr}_candidate_sha256"
            if kp in a and val(a[kp])!=sin[sr]["path"]:raise ValueError(f"{r}: stale raw typed {sr} path")
            if kh in a and val(a[kh])!=sin[sr]["sha256"]:raise ValueError(f"{r}: stale raw typed {sr} hash")
        target,groups,rows,pmap,wallids,wlocal,wxyz=m.build_target(r,points,sdata,wall_csv,collar)
        actual=np.asarray(a["boundary_faces"],np.int64)
        if m.directed_chain(target)!=m.directed_chain(actual) or len(target)!=len(actual):
            raise ValueError(f"{r}: current typed target is not the exact raw volume boundary chain")
        roleassets={**assets,"volume_parent":{"path":str(p),"sha256":h}}
        b,detail=rebase_volume(a,r,p,h,target,groups,rows,pmap,wlocal,wxyz,sout,roleassets)
        op=outdir/f"{r.lower()}-volume-identity-rebased.npz";np.savez_compressed(op,**b);saved=load(op)
        gh=equal_geometry(a,saved,VGEOM,r)
        if not (np.array_equal(saved["replacement_parent_offsets"],np.asarray([0],np.int64)) and
          len(saved["replacement_parent_cell_rows"])==0 and np.array_equal(saved["replacement_tet_offsets"],np.asarray([0,0],np.int64)) and
          saved["replacement_tetrahedra"].shape==(0,4) and np.array_equal(saved["output_point_parent_offsets"],np.zeros(len(points)+1,np.int64)) and
          len(saved["output_point_parent_input_cell_rows"])==0): raise ValueError(f"{r}: current-step replacement/point-parent maps are not empty")
        for k,n in (("output_point_to_input_point_row",len(points)),("input_node_to_output_node",len(points)),
          ("output_node_to_input_node",len(points)),("source_interior_vertex_quotient",len(points)),
          ("output_cell_to_input_cell",len(a["tetrahedra"])),("input_cell_to_output_cell",len(a["tetrahedra"])),
          ("retained_input_cell_rows",len(a["tetrahedra"]))):
            if not np.array_equal(saved[k],np.arange(n,dtype=np.int64)):raise ValueError(f"{r}: nonidentity {k}")
        vreport[r]={**detail,"input":{"path":str(p),"sha256":h},"output":{"path":str(op),"sha256":sha(op)},
          "geometry_hashes":gh,"geometry_bitwise_equal":True,"target_matches_actual_oriented_chain":True,"native_steps":0}
        outcore[r+"_current"]={**e,"path":str(op),"sha256":sha(op),"metadata_parent_path":str(p),"metadata_parent_sha256":h,
          "target_face_count":len(target),"target_boundary_match":True,"metadata_only_geometry_identical":True}
    outcore["metadata_finalizer_709"]={"schema":SCHEMA,"producer":assets["producer"],"input_state":input_state,
      "geometry_changed":False,"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
      "source_surfaces":{r:sreport[r]["output"] for r in ("O","I","D")},
      "volume_regions":{r:vreport[r]["output"] for r in ("Skin","Fat")},
      "history":"Outputs pin exact raw current parents; previous immediate maps remain recoverable from those parents. Absolute ancestry arrays are byte-preserved."}
    if wrap:
        state_out_obj=dict(raw);state_out_obj[wrap]=outcore
    else:
        state_out_obj=outcore
    state_out=outdir/"state.json";state_out.write_text(json.dumps(state_out_obj,indent=2,sort_keys=True)+"\n")
    report={"schema":SCHEMA,"status":"PASS_SOURCE_VOLUME_METADATA_ONLY_IDENTITY_REBASE","producer":assets["producer"],
      "target_helper":assets["target_helper"],"input_state":input_state,
      "wall_assets":{k:assets[k] for k in ("wall_csv","collar","wall_receipt")},"raw_typed_sources":sin,
      "rebased_typed_sources":sreport,"volume_regions":vreport,
      "output_state":{"path":str(state_out),"sha256":sha(state_out)},"geometry_changed":False,
      "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
      "surface_collision_audit":"NOT_RUN_METADATA_ONLY; parent context evidence remains separately pinned",
      "limitations":"No new collision, native quality, source330 mass, or admission claim. Typed target is checked only against existing oriented volume boundary."}
    rp=outdir/"metadata-finalizer-709-report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":report["status"],"report":str(rp),"report_sha256":sha(rp),"state":str(state_out),
      "source_outputs":{r:v["output"] for r,v in sreport.items()},"volume_outputs":{r:v["output"] for r,v in vreport.items()}},indent=2))

def self_test():
    p=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
    f=np.asarray([[0,1,2],[0,3,1],[0,2,3],[1,3,2]],np.int64); off=np.asarray([0,4,8,12,16])
    s={"points_world_m_f32":p.copy(),"points_world_m":p.astype(np.float64),"faces":f.copy(),
      "source_input_vertex_quotient":np.asarray([1,0,2,3]),"source_input_face_rows":np.arange(4)[::-1],
      "quality_star_point_output_to_input_row":np.asarray([1,0,2,3]),"quality_star_face_output_to_input_row":np.arange(4)[::-1],
      "source_collapse_operation_rows":np.asarray([3,3,-1,-1]),"source_diagonal_flip_operation_rows":np.asarray([2,2,-1,-1]),
      "quality_star_patch_union_offsets":off,"quality_star_patch_union_source_original_ids":np.arange(16),
      "face_ancestry_offsets":off.copy(),"face_ancestry_parent_face_rows":np.arange(16)}
    z,sd=rebase_source(s,"T",Path("/raw/T.npz"),"a"*64)
    assert equal_geometry(s,z,SGEOM,"tiny source") and np.array_equal(z["source_input_face_rows"],np.arange(4))
    assert np.all(z["source_collapse_operation_rows"]==-1) and np.all(z["source_diagonal_flip_operation_rows"]==-1)
    assert val(z["typed_parent_candidate_path"])=="/raw/T.npz" and val(z["typed_parent_candidate_sha256"])=="a"*64
    verify_unions(z,sd["absolute_union_hashes"],"tiny source")
    v={"points_world_m":p.astype(np.float64),"points_world_m_f32":p.copy(),"tetrahedra":np.asarray([[0,1,2,3]]),
      "material_indices":np.asarray([16],np.int32),"boundary_faces":f.copy(),"desired_boundary_faces":f[::-1].copy(),
      "output_point_to_input_point_row":np.asarray([1,0,2,3]),"output_cell_to_input_cell":np.asarray([-1]),
      "input_cell_to_output_cell":np.asarray([-1]),"replacement_parent_offsets":np.asarray([0,1]),
      "replacement_parent_cell_rows":np.asarray([77]),"replacement_tet_offsets":np.asarray([0,1]),
      "replacement_tetrahedra":np.asarray([[0,1,2,3]]),"output_point_parent_offsets":np.asarray([0,1,1,1,1]),
      "output_point_parent_input_cell_rows":np.asarray([55]),"source_cell_operation_rows":np.asarray([8])}
    old_step_hashes={k:ah(v[k]) for k in VSTEP_FIELDS}
    outs={r:(Path("/out/"+r+".npz"),str(i)*64) for i,r in enumerate(("O","I","D"),1)}
    assets={k:{"path":"/"+k,"sha256":c*64} for k,c in (("wall_csv","b"),("collar","c"),("wall_receipt","d"))}
    w,vd=rebase_volume(v,"Fat",Path("/raw/F.npz"),"e"*64,f,np.zeros(4,np.int8),np.arange(4),
      {"I":np.arange(4),"D":np.arange(4)},np.empty((0,3),np.int64),np.empty((0,3),np.float32),outs,assets)
    assert equal_geometry(v,w,VGEOM,"tiny volume") and np.array_equal(w["output_cell_to_input_cell"],np.asarray([0]))
    assert np.array_equal(w["input_cell_to_output_cell"],np.asarray([0])) and np.array_equal(w["retained_input_cell_rows"],np.asarray([0]))
    assert np.array_equal(w["replacement_parent_offsets"],np.asarray([0],np.int64)) and w["replacement_parent_cell_rows"].size==0
    assert np.array_equal(w["replacement_tet_offsets"],np.asarray([0,0],np.int64)) and w["replacement_tetrahedra"].shape==(0,4)
    assert np.array_equal(w["output_point_parent_offsets"],np.zeros(5,np.int64)) and w["output_point_parent_input_cell_rows"].size==0
    assert np.array_equal(w["source_cell_operation_rows"],np.asarray([-1]))
    assert {k:ah(v[k]) for k in VSTEP_FIELDS}==old_step_hashes
    recorded=json.loads(val(w["metadata_finalizer_709_previous_immediate_map_sha256_json"]))
    assert {k:x["sha256"] for k,x in recorded.items()}==old_step_hashes
    assert val(w["current_geometry_parent_path"])=="/raw/F.npz" and val(w["current_geometry_parent_sha256"])=="e"*64
    assert val(w["typed_D_candidate_path"])=="/out/D.npz" and val(w["typed_D_candidate_sha256"])=="3"*64
    print("PASS 709 tiny identity-map, parent-hash, cleared-op, preserved-union and geometry controls")

def control_678(state_path,report_path):
    m=helper();st=read(state_path);prior=read(report_path)
    cfg={k:prior[k] for k in ("wall_csv","collar","wall_receipt")};check_walls(cfg)
    wall=Path(cfg["wall_csv"]["path"]);collar=load(Path(cfg["collar"]["path"]));surfs={};outs={}
    for r in ("O","I","D"):
        e=st["source_surfaces"][r];p=Path(e["path"]);a=load(p);b,d=rebase_source(a,r,p,e["sha256"])
        equal_geometry(a,b,SGEOM,"678 "+r);verify_unions(b,d["absolute_union_hashes"],"678 "+r)
        surfs[r]=b;outs[r]=(p,e["sha256"])
    for r in ("Skin","Fat"):
        e=st[r+"_current"];p=Path(e["path"]);a=load(p);pts=np.asarray(a["points_world_m_f32"],np.float32)
        target,g,rows,pmap,wid,wl,wxyz=m.build_target(r,pts,surfs,wall,collar)
        if m.directed_chain(target)!=m.directed_chain(a["boundary_faces"]):raise ValueError("678 target/boundary mismatch")
        assets={k:prior[k] for k in ("wall_csv","collar","wall_receipt")}
        assets["input_state"]=prior["input_state"];assets["producer"]={"path":"control-only","sha256":"f"*64}
        assets["target_helper"]={"path":str(HELPER),"sha256":HELPER_SHA256}
        b,d=rebase_volume(a,r,p,e["sha256"],target,g,rows,pmap,wl,wxyz,outs,assets)
        equal_geometry(a,b,VGEOM,"678 "+r)
        if len(b["replacement_parent_cell_rows"]) or len(b["replacement_tetrahedra"]) or len(b["output_point_parent_input_cell_rows"]):
            raise ValueError("678 volume current-step ancestry maps were not cleared")
        if not np.array_equal(b["replacement_parent_offsets"],np.asarray([0],np.int64)) or not np.array_equal(b["replacement_tet_offsets"],np.asarray([0,0],np.int64)):
            raise ValueError("678 current-step replacement CSR not empty")
        if val(b["current_geometry_parent_path"])!=str(p) or val(b["current_geometry_parent_sha256"])!=e["sha256"]:
            raise ValueError("678 exact raw volume parent pin mismatch")
        prior_maps=json.loads(val(b["metadata_finalizer_709_previous_immediate_map_sha256_json"]))
        if any(prior_maps[k]["sha256"]!=ah(a[k]) for k in VSTEP_FIELDS if k in a):
            raise ValueError("678 prior immediate maps are not recoverable from pinned parent")
    print("PASS 678 in-memory control: exact target chains; SOURCE/VOLUME geometry and absolute face ancestry byte-identical; current-step volume ancestry empty")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--state",type=Path);ap.add_argument("--wall-config",type=Path);ap.add_argument("--output",type=Path)
    ap.add_argument("--self-test",action="store_true");ap.add_argument("--control-678",action="store_true")
    ap.add_argument("--control-state",type=Path);ap.add_argument("--control-report",type=Path);a=ap.parse_args()
    if a.self_test:self_test();return
    if a.control_678:
        if not(a.control_state and a.control_report):ap.error("--control-678 requires --control-state and --control-report")
        control_678(a.control_state,a.control_report);return
    if not(a.state and a.wall_config and a.output):ap.error("--state, --wall-config, and --output required")
    run(a.state,a.wall_config,a.output)
if __name__=="__main__":main()
