#!/usr/bin/env python3
"""Build current role P1 support and 143 one-hop source-region support.

This is a source-only derivative of the existing 230 same-material point
support, integer P1 corner integration, and exact source330 shared-face
admissibility rules. It accepts a current authenticated volume/state and
optionally reuses a prior node-support cache only on exact F32 coordinate keys.
No mass allocation or physical admission is performed.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, sys, time, tempfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any
import numpy as np

HERE=Path(__file__).resolve().parent
H=HERE.parent
P=H.parent
PARENT=P/"native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv"
OWNER230=H/"current-semantic-atlas-088/export_authoring_registered_surface_sidecars_230_semantic_088.py"
OWNER230_SHA="121e3bf44d90ba3d7b82ccaa40a26a92d1675c6d277a4409ae95f8c05ad2485d"
OWNER217_BASE=P/"paired-head-reference-fit-684/run-063-registered-native-wall-frame-002/native-domain-217-source-003/build_native_domain_217.py"
OWNER217_BASE_SHA="d7f411c18f91905c6e1d91ab75592ec5782a8852892ac8a1fdb566ee2206df09"
OWNER217=H/"current-role-p1-support-938-1025-active-source-map/current_fine_217_owner_938_v3_active_source_rows.py"
OWNER217_SHA="e644a2852ab94fa1b4ce81d7abff41dd06d3aa0a7f955030a7c4f8d8253411fb"
OWNER937=H/"current-role-p1-support-938-1025-active-source-map/current_fine_volume_to_217_938_818c_849d_active_rows.py"
OWNER937_SHA="bce0755224226eb89c8e545e0a66f421c6e71aa7b0a479e00a8c39367e884e8f"
FULL86=Path(str(PARENT)+".weights.csv")
FULL86_SHA="98205ae7de44401544b7969184788f5c2ee39cd4404cd34b8037377e7ae63a76a"
PARENT_SHA="c7823c42d64a2154a34c87cc0d4d202f0d5256f79217e74f5f62bcf1d5b31679"
ROLE_MATERIAL={"Skin":15,"Fat":16}
ROLE_STATUS={"Skin":"PASS_SOURCE_ONLY_CURRENT_SKIN_P1_AND_AUTHENTICATED_ONE_HOP_SUPPORT",
             "Fat":"PASS_SOURCE_ONLY_CURRENT_FAT_P1_AND_AUTHENTICATED_ONE_HOP_SUPPORT"}
SCHEMA="numi.human.current-role-p1-semantic-onehop-support.v1"
LOCAL_FACES=np.asarray([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],dtype=np.int64)
MAX_PARENT_DISTANCE_M=0.02
AMBIGUITY_TOLERANCE_M=1.0e-8
DERIVED_BASE=0x80000000

def sha(path:Path|str)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def arrays_equal(a:np.ndarray,b:np.ndarray)->bool:
    x=np.asarray(a);y=np.asarray(b)
    if x.dtype!=y.dtype or x.shape!=y.shape:return False
    if np.issubdtype(x.dtype,np.floating):return bool(np.array_equal(x,y,equal_nan=True))
    return bool(np.array_equal(x,y))

def pin(path:Path|str)->dict[str,Any]:
    p=Path(path).resolve()
    return {"path":str(p),"sha256":sha(p),"size_bytes":p.stat().st_size}

def _scalar(a:np.ndarray)->str:
    if np.asarray(a).size!=1: raise ValueError("expected scalar NPZ metadata")
    return str(np.asarray(a).reshape(-1)[0])

def _state_core(state:dict)->dict:
    q=[state];seen=set()
    while q:
        x=q.pop(0)
        if not isinstance(x,dict) or id(x) in seen:continue
        seen.add(id(x))
        if isinstance(x.get("Skin_current"),dict) and isinstance(x.get("Fat_current"),dict):
            if x.get("physical_admission") is True or x.get("native_steps",0)!=0:
                raise ValueError("current state is not source-only")
            if x.get("mass_mapping",x.get("mass_reconciliation","PENDING")) not in ("PENDING","NOT_RUN","PENDING_VOLUME_CHANGE_RECONCILIATION"):
                raise ValueError("current state mass status is not pending")
            return x
        for k in ("current_verified_state_20261031","current","verified_state","state"):
            if isinstance(x.get(k),dict):q.append(x[k])
    raise ValueError("state lacks paired current role volume entries")

def _role_volume(path:Path,state_path:Path,role:str)->dict[str,np.ndarray|str]:
    state_path=state_path.resolve()
    state_doc=json.loads(state_path.read_text())
    st=_state_core(state_doc)
    entry=st[f"{role}_current"]
    with np.load(path,allow_pickle=False) as z:
        keys=set(z.files)
        pkey="points_world_m_f32" if "points_world_m_f32" in keys else "points_world_m"
        required=("tetrahedra","material_indices","canonical_parent_union_offsets",
          "canonical_parent_union_parent_rows","canonical_parent_union_current_state_path",
          "canonical_parent_union_current_state_sha256","canonical_parent_union_current_volume_path",
          "canonical_parent_union_current_volume_sha256","canonical_parent_union_base_state_path",
          "canonical_parent_union_base_state_sha256","canonical_parent_union_base_volume_path",
          "canonical_parent_union_base_volume_sha256","canonical_parent_union_parent_row_domain",
          "canonical_parent_union_physical_status")
        missing=[k for k in required if k not in keys]
        if missing:raise ValueError(f"current {role} volume lacks 834 metadata: {missing}")
        points=np.asarray(z[pkey]).copy()
        tets=np.asarray(z["tetrahedra"]).copy()
        mats=np.asarray(z["material_indices"]).copy()
        off=np.asarray(z["canonical_parent_union_offsets"]).copy()
        rows=np.asarray(z["canonical_parent_union_parent_rows"]).copy()
        meta={k:_scalar(z[k]) for k in required[4:]}
    if points.dtype!=np.float32 or points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError(f"{role} current volume must have finite exact Float32 world points")
    if not np.issubdtype(tets.dtype,np.integer) or tets.ndim!=2 or tets.shape[1]!=4:
        raise ValueError(f"{role} current volume tetrahedra are malformed")
    tets=np.asarray(tets,np.int64)
    if len(tets)==0 or int(tets.min())<0 or int(tets.max())>=len(points):raise ValueError(f"{role} current tetrahedron node rows invalid")
    if mats.shape!=(len(tets),) or not np.issubdtype(mats.dtype,np.integer) or not np.all(mats==ROLE_MATERIAL[role]):
        raise ValueError(f"{role} current material rows do not match the registered role")
    if off.shape!=(len(tets)+1,) or int(off[0])!=0 or int(off[-1])!=len(rows) or np.any(off[1:]<off[:-1]):
        raise ValueError(f"{role} 834 parent CSR is malformed")
    if np.any(rows<0):raise ValueError(f"{role} 834 parent CSR has negative base-domain rows")
    if np.any(np.diff(off)<=0):raise ValueError(f"{role} 834 parent CSR omits a current cell")
    if meta["canonical_parent_union_parent_row_domain"]!="canonical_base_volume_cell_rows":
        raise ValueError(f"{role} parent CSR is not in the explicit 834 base-volume domain")
    if meta["canonical_parent_union_physical_status"]!="SOURCE_ONLY_PHYSICAL_ADMISSION_FALSE":
        raise ValueError(f"{role} current volume is not source-only")
    if Path(meta["canonical_parent_union_current_state_path"]).resolve()!=state_path or meta["canonical_parent_union_current_state_sha256"]!=sha(state_path):
        raise ValueError(f"{role} current volume is bound to a different current state")
    if Path(meta["canonical_parent_union_current_volume_path"]).resolve()!=Path(entry.get("path","")).resolve() or meta["canonical_parent_union_current_volume_sha256"]!=entry.get("sha256"):
        raise ValueError(f"{role} current volume ancestry differs from paired-state input")
    if not Path(entry.get("path","")).is_file() or sha(entry["path"])!=entry.get("sha256"):
        raise ValueError(f"{role} paired-state input volume pin is stale")
    for field,sha_field in (("canonical_parent_union_base_state_path","canonical_parent_union_base_state_sha256"),
                            ("canonical_parent_union_base_volume_path","canonical_parent_union_base_volume_sha256")):
        bp=Path(meta[field])
        if not bp.is_file() or sha(bp)!=meta[sha_field]:
            raise ValueError(f"{role} 834 base ancestry pin is stale: {field}")
    base_volume=Path(meta["canonical_parent_union_base_volume_path"])
    with np.load(base_volume,allow_pickle=False) as bz:
        if "tetrahedra" not in bz:raise ValueError(f"{role} pinned 834 base volume has no tetrahedra")
        base_cell_count=len(bz["tetrahedra"])
    if len(rows) and int(rows.max())>=base_cell_count:
        raise ValueError(f"{role} 834 parent row exceeds its explicit canonical base-volume domain")
    return {"points":points,"tets":tets,"materials":np.asarray(mats,np.int32),
            "canonical_offsets":off,"canonical_rows":rows,"metadata":meta,
            "state":st,"state_pin":pin(state_path),"volume_pin":pin(path)}

def _point_key(points:np.ndarray)->np.ndarray:
    x=np.ascontiguousarray(np.asarray(points,dtype=np.float32)).copy()
    if x.ndim!=2 or x.shape[1]!=3:raise ValueError("point key requires Nx3")
    x[x==0]=0.0
    return x.view(np.dtype((np.void,12))).reshape(-1)

def _load_230(owner_path:Path):
    if owner_path.resolve()!=OWNER230.resolve() or sha(owner_path)!=OWNER230_SHA:
        raise ValueError("230 current same-material owner path/SHA differs from frozen owner")
    spec=importlib.util.spec_from_file_location("current_role_support_230_owner_938",owner_path)
    if spec is None or spec.loader is None:raise ImportError(str(owner_path))
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod

def _validate_current_834_pair(state_path:Path,report_path:Path,role:str,volume_path:Path)->dict:
    if sha(OWNER937)!=OWNER937_SHA:raise ValueError("current 937 canonical-volume bridge source changed")
    spec=importlib.util.spec_from_file_location("current_834_binding_owner_938",OWNER937)
    if spec is None or spec.loader is None:raise ImportError(str(OWNER937))
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    report=json.loads(report_path.read_text())
    roles=report.get("roles",{})
    paired={}
    for current_role in ("Skin","Fat"):
        out=roles.get(current_role,{}).get("output",{})
        path=Path(out.get("path",""))
        if not path.is_file() or sha(path)!=out.get("sha256"):
            raise ValueError(f"834 report has a missing or stale {current_role} canonical volume")
        paired[current_role]=path
    if paired[role].resolve()!=volume_path.resolve():
        raise ValueError(f"supplied current {role} volume differs from paired 834 canonical report")
    binding=mod.validate_834_binding(state_path,report_path,paired)
    return {"canonical_834_report":pin(report_path),"paired_canonical_volumes":{
        q:pin(paired[q]) for q in ("Skin","Fat")},
        "validated_role_union_hashes":{
        q:binding["roles"][q]["union_rows_sha256"] for q in ("Skin","Fat")}}


def _source_groups(parent:dict,role:str)->tuple[np.ndarray,dict]:
    mats=np.asarray(parent["materials"],np.int32);donors=np.asarray(parent["donors"],np.int32);regions=np.asarray(parent["regions"],np.int32)
    faces=np.asarray(parent["faces"])
    n=len(mats)
    if donors.shape!=(n,) or regions.shape!=(n,) or faces.shape!=(n,):raise ValueError("230 parent semantic lanes are not row aligned")
    if np.any(donors<0) or np.any(regions<0):raise ValueError("source330 donor/region IDs must be nonnegative")
    tuples=sorted({(int(mats[i]),int(donors[i]),int(regions[i])) for i in np.flatnonzero(mats==ROLE_MATERIAL[role])})
    if not tuples:raise ValueError(f"source330 has no parent groups for {role}")
    groups=np.asarray(tuples,np.int32).reshape((-1,3))
    lookup={x:i for i,x in enumerate(tuples)}
    return groups,lookup

def _cache_binding(receipt_path:Path,cache_support:Path,cache_volume:Path,source_owner:Path,
                   parent_meta:dict,role:str)->dict:
    receipt=json.loads(receipt_path.read_text())
    if receipt.get("physical_admission") is True or receipt.get("native_steps",0)!=0:
        raise ValueError("node cache receipt is not source-only")
    if receipt.get("mass_transport",receipt.get("mass_mapping","NOT_RUN")) not in ("NOT_RUN","PENDING"):
        raise ValueError("node cache receipt contains completed mass status")
    support_pin=pin(cache_support);volume_pin=pin(cache_volume)
    # Existing 531/597 cache receipts differ in shape. Require an explicit exact
    # output and geometry pin, plus the immutable 230/source330 owner pins.
    output_ok=False
    for key in ("node_support_npz","p1_143","p1_support","output"):
        q=receipt.get("outputs",{}).get(key)
        if isinstance(q,str) and Path(q).resolve()==Path(support_pin["path"]) and receipt.get("outputs",{}).get("node_support_sha256")==support_pin["sha256"]:
            output_ok=True
        elif isinstance(q,dict) and Path(q.get("path","")).resolve()==Path(support_pin["path"]) and q.get("sha256")==support_pin["sha256"]:
            output_ok=True
    if not output_ok:
        raise ValueError("cache receipt does not bind node support output")
    geo=receipt.get("geometry_target") or receipt.get("inputs",{}).get("geometry") or receipt.get("geometry")
    if not isinstance(geo,dict) or Path(geo.get("path","")).resolve()!=Path(volume_pin["path"]) or geo.get("sha256")!=volume_pin["sha256"]:
        raise ValueError("cache receipt does not bind its exact old point-volume domain")
    with np.load(cache_volume,allow_pickle=False) as vz:
        wanted_meta=("canonical_parent_union_current_state_path","canonical_parent_union_current_state_sha256",
          "canonical_parent_union_current_volume_path","canonical_parent_union_current_volume_sha256")
        if any(k not in vz.files for k in wanted_meta):raise ValueError("cached volume lacks its current state/input identity chain")
        vm={k:_scalar(vz[k]) for k in wanted_meta}
    old_state=Path(vm["canonical_parent_union_current_state_path"])
    old_input=Path(vm["canonical_parent_union_current_volume_path"])
    if not old_state.is_file() or sha(old_state)!=vm["canonical_parent_union_current_state_sha256"]:
        raise ValueError("cached volume current-state pin is missing or stale")
    old_core=_state_core(json.loads(old_state.read_text()))
    old_entry=old_core.get(f"{role}_current",{})
    if Path(old_entry.get("path","")).resolve()!=old_input.resolve() or old_entry.get("sha256")!=vm["canonical_parent_union_current_volume_sha256"]:
        raise ValueError("cached volume current input does not match its pinned paired-state role")
    if not old_input.is_file() or sha(old_input)!=vm["canonical_parent_union_current_volume_sha256"]:
        raise ValueError("cached volume current input is missing or stale")
    owner_found=False;parent_found=False
    input_hashes=receipt.get("input_sha256",{})
    for k,v in input_hashes.items():
        if Path(k).resolve()==source_owner.resolve() and v==sha(source_owner):owner_found=True
        if Path(k).resolve()==PARENT.resolve() and v==PARENT_SHA:parent_found=True
    for pkey,hkey in (("massPath","massSHA256"),("nodesPath","nodeSHA256"),
                      ("tetsPath","tetsSHA256"),("labelsPath","labelsSHA256")):
        expected_path=Path(parent_meta[pkey]).resolve();expected_hash=parent_meta[hkey]
        if not expected_path.is_file() or sha(expected_path)!=expected_hash:
            raise ValueError(f"current source330 sidecar is stale: {pkey}")
        if input_hashes.get(str(expected_path))!=expected_hash:
            raise ValueError(f"cache receipt lacks exact source330 sidecar pin: {pkey}")
    for key in ("semantic_owner","source330_mass","source330_parent_identity"):
        q=receipt.get("inputs",{}).get(key,{})
        if isinstance(q,dict):
            if Path(q.get("path","")).resolve()==source_owner.resolve() and q.get("sha256")==sha(source_owner):owner_found=True
            if Path(q.get("path","")).resolve()==PARENT.resolve() and q.get("sha256")==PARENT_SHA:parent_found=True
    if not owner_found or not parent_found:raise ValueError("cache receipt lacks pinned 230 and source330 input ownership")
    return {"receipt":pin(receipt_path),"support":support_pin,"volume":volume_pin}

def _load_cache(cache_support:Path,cache_volume:Path,receipt_path:Path,role:str,groups:np.ndarray,lookup:dict,
                parent:dict,source_owner:Path,parent_meta:dict)->tuple[dict,dict]:
    pins=_cache_binding(receipt_path,cache_support,cache_volume,source_owner,parent_meta,role)
    with np.load(cache_volume,allow_pickle=False) as z:
        pkey="points_world_m_f32" if "points_world_m_f32" in z.files else "points_world_m"
        points=np.asarray(z[pkey]).copy()
    if points.dtype!=np.float32 or points.ndim!=2 or points.shape[1]!=3:raise ValueError("cache volume points are not exact Float32 Nx3")
    with np.load(cache_support,allow_pickle=False) as z:cache={k:np.asarray(z[k]).copy() for k in z.files}
    needed=("group_tuples","node_group_offsets","node_group_ids","node_group_parent_tet_offsets","node_group_parent_tet_rows")
    miss=[k for k in needed if k not in cache]
    if miss:raise ValueError(f"node-support cache lacks aligned 230 CSR fields: {miss}")
    go=cache["node_group_offsets"];gi=cache["node_group_ids"];po=cache["node_group_parent_tet_offsets"];pr=cache["node_group_parent_tet_rows"];gt=np.asarray(cache["group_tuples"],np.int64)
    if go.shape!=(len(points)+1,) or int(go[0])!=0 or int(go[-1])!=len(gi) or np.any(go[1:]<go[:-1]):
        raise ValueError("cache node-group CSR row domain malformed")
    if po.shape!=(len(gi)+1,) or int(po[0])!=0 or int(po[-1])!=len(pr) or np.any(po[1:]<po[:-1]):
        raise ValueError("cache parent CSR malformed")
    if gt.ndim!=2 or gt.shape[1]!=3 or np.any(gi<0) or np.any(gi>=len(gt)):
        raise ValueError("cache group tuple IDs invalid")
    if len(_point_key(points))!=len(points) or len(np.unique(_point_key(points)))!=len(points):
        raise ValueError("cache volume has duplicate exact F32 point keys")
    if len(cache["node_origin_kind"])!=len(points) or len(cache["node_parent_registration_distance_m"])!=len(points):
        raise ValueError("cache registration diagnostics have wrong node domain")
    source_mats=np.asarray(parent["materials"],np.int32);source_donors=np.asarray(parent["donors"],np.int32);source_regions=np.asarray(parent["regions"],np.int32)
    cache_maps=[]
    for nr in range(len(points)):
        m={}
        for s in range(int(go[nr]),int(go[nr+1])):
            oldgid=int(gi[s]);ot=tuple(int(x) for x in gt[oldgid])
            if ot not in lookup:raise ValueError(f"cached source tuple {ot} is absent from current immutable 230 registry")
            gid=int(lookup[ot]);a=int(po[s]);b=int(po[s+1]);rows=np.asarray(pr[a:b],np.int64)
            if not len(rows) or int(rows.min())<0 or int(rows.max())>=len(source_mats):raise ValueError(f"cached row out of source330 range at node {nr}")
            if any((int(source_mats[r]),int(source_donors[r]),int(source_regions[r]))!=ot for r in rows):
                raise ValueError(f"cached parent rows do not authenticate tuple {ot}")
            if ot[0]!=ROLE_MATERIAL[role]:raise ValueError("cached node support crosses role material")
            m.setdefault(gid,set()).update(map(int,rows))
        cache_maps.append(m)
    return {"points":points,"maps":cache_maps,"origin":np.asarray(cache["node_origin_kind"],np.uint8),
            "distance":np.asarray(cache["node_parent_registration_distance_m"],np.float64)},pins

def build_node_support(points:np.ndarray,role:str,parent:dict,index,material_index,groups:np.ndarray,lookup:dict,
                       owner,cache:dict|None=None,max_distance:float=MAX_PARENT_DISTANCE_M,
                       tie_tolerance:float=AMBIGUITY_TOLERANCE_M,progress=None)->dict:
    n=len(points);maps=[{} for _ in range(n)];origin=np.zeros(n,np.uint8);dist=np.full(n,np.nan,np.float64)
    reused=0;cache_map=cache["maps"] if cache else None
    if cache:
        ck=_point_key(cache["points"]);tk=_point_key(points)
        cidx={k.tobytes():i for i,k in enumerate(ck)}
        for i,k in enumerate(tk):
            old=cidx.get(k.tobytes())
            if old is None:continue
            m=cache_map[old]
            if not m:continue
            d=float(cache["distance"][old])
            if not np.isfinite(d) or d>max_distance+1e-12:continue
            maps[i]={gid:set(rows) for gid,rows in m.items()}
            origin[i]=2;dist[i]=d;reused+=1
    query=np.flatnonzero(np.fromiter((not bool(m) for m in maps),dtype=np.bool_,count=n))
    queried=[];failures=[]
    pm=np.asarray(parent["materials"],np.int32);pd=np.asarray(parent["donors"],np.int32);pr=np.asarray(parent["regions"],np.int32)
    for k,nr0 in enumerate(query):
        nr=int(nr0)
        try:q=owner.closest_same_material(parent,index,material_index,ROLE_MATERIAL[role],points[nr].astype(np.float64),max_distance,tie_tolerance)
        except Exception as e:
            failures.append({"node_row":nr,"error":type(e).__name__+": "+str(e)})
            continue
        ties=sorted(set(map(int,q.get("ties",[]))));selected=int(q.get("tet",-1));d=float(q.get("distance",float("nan")))
        if not ties or selected not in ties or not np.isfinite(d) or d>max_distance+1e-12:
            failures.append({"node_row":nr,"error":"invalid_or_out_of_bound_230_witness","selected":selected,"ties":ties[:16],"distance_m":d})
            continue
        for row in ties:
            if row<0 or row>=len(pm):raise ValueError(f"230 node {nr} returned source parent out of range")
            tup=(int(pm[row]),int(pd[row]),int(pr[row]))
            if tup[0]!=ROLE_MATERIAL[role]:raise ValueError(f"230 node {nr} returned cross-material parent tuple {tup}")
            if tup not in lookup:raise ValueError(f"230 node {nr} tuple {tup} absent from source group registry")
            gid=int(lookup[tup]);maps[nr].setdefault(gid,set()).add(row)
        origin[nr]=3;dist[nr]=d;queried.append(d)
        if progress and (k+1)%10000==0:progress({"stage":"current_230_node_support","queries_done":k+1,"queries_total":len(query),"reused_exact_points":reused,"failures":len(failures)})
    if failures:
        raise ValueError(f"230 node support failed for {len(failures)}/{len(query)} current points; first={failures[0]}")
    for nr,m in enumerate(maps):
        if not m:raise ValueError(f"current node {nr} has no same-material 230 support")
        for gid,rows in m.items():
            tup=tuple(int(x) for x in groups[gid])
            if tup[0]!=ROLE_MATERIAL[role] or not rows:raise ValueError(f"invalid node support at {nr}, group {tup}")
            for r in rows:
                if (int(pm[r]),int(pd[r]),int(pr[r]))!=tup:raise ValueError(f"current node parent tuple mismatch at node {nr}")
    return {"maps":maps,"origin":origin,"distance":dist,"reused_exact_points":reused,
            "queried_points":int(len(query)),"query_distance_max_m":float(max(queried,default=0.0)),
            "query_distance_p99_m":float(np.quantile(queried,.99)) if queried else 0.0}

def integrate_p1_cells(tets:np.ndarray,node_maps:list[dict[int,set[int]]],groups:np.ndarray,parent:dict,role:str)->dict[str,np.ndarray]:
    ncell=len(tets);cell_go=[0];cell_gid=[];cell_num=[];cell_den=[];cg_po=[0];cg_pr=[];cg_fo=[0];cg_fr=[]
    selected=np.full(ncell,-1,np.int32);status=np.full(ncell,2,np.uint8);denom=np.zeros(ncell,np.int64);winner=np.zeros(ncell,np.int64)
    resolved=[];resolved_gid=[];scalar_faces=[];unique_po=[0];unique_pr=[];unique_fo=[0];unique_fr=[]
    parent_faces=np.asarray(parent["faces"],np.uint32);pm=np.asarray(parent["materials"],np.int32)
    for ci,tet in enumerate(tets):
        corners=[node_maps[int(v)] for v in tet]
        if any(not m for m in corners):raise ValueError(f"cell {ci} has an unsupported corner")
        scale=1
        for m in corners:scale=math.lcm(scale,len(m))
        scores=Counter()
        for m in corners:
            share=scale//len(m)
            for gid in m:scores[int(gid)]+=share
        d=4*scale
        if sum(scores.values())!=d:raise AssertionError("P1 partition of unity failed")
        denom[ci]=d
        top=max(scores.values());winners=sorted(g for g,v in scores.items() if v==top)
        if len(winners)==1:
            gid=int(winners[0]);selected[ci]=gid;status[ci]=1;winner[ci]=scores[gid];resolved.append(ci);resolved_gid.append(gid)
            prs=np.unique(np.concatenate([np.asarray(sorted(m[gid]),np.int64) for m in corners if gid in m])).astype(np.int32)
            fr=np.unique(parent_faces[prs]).astype(np.uint32)
            if not len(prs) or not len(fr):raise ValueError(f"unique P1 group lacks source330 ancestry at cell {ci}")
            if not np.all(pm[prs]==ROLE_MATERIAL[role]):raise ValueError(f"P1 unique group has cross-material parent at cell {ci}")
            unique_pr.extend(prs.tolist());unique_po.append(len(unique_pr));unique_fr.extend(fr.tolist());unique_fo.append(len(unique_fr))
            scalar_faces.append(int(fr[0]) if len(fr)==1 else DERIVED_BASE+gid)
        for gid in sorted(scores):
            prs=np.unique(np.concatenate([np.asarray(sorted(m[gid]),np.int64) for m in corners if gid in m])).astype(np.int32)
            fr=np.unique(parent_faces[prs]).astype(np.uint32)
            if not len(prs) or not len(fr):raise ValueError(f"P1 group lacks source330 ancestry at cell {ci}, group {gid}")
            if not np.all(pm[prs]==ROLE_MATERIAL[role]):raise ValueError(f"P1 group crosses role material at cell {ci}")
            cell_gid.append(int(gid));cell_num.append(int(scores[gid]));cell_den.append(int(d))
            cg_pr.extend(prs.tolist());cg_po.append(len(cg_pr));cg_fr.extend(fr.tolist());cg_fo.append(len(cg_fr))
        cell_go.append(len(cell_gid))
    unresolved=np.flatnonzero(selected<0).astype(np.int32)
    return {
      "resolved_child_tet_rows":np.asarray(resolved,np.int32),"resolved_group_ids":np.asarray(resolved_gid,np.int32),
      "stable_source_face_id":np.asarray(scalar_faces,np.uint32),"parent_source_tet_offsets":np.asarray(unique_po,np.int64),
      "parent_source_tet_rows":np.asarray(unique_pr,np.int32),"stable_source_face_offsets":np.asarray(unique_fo,np.int64),
      "stable_source_face_rows":np.asarray(unique_fr,np.uint32),"unresolved_child_tet_rows":unresolved,
      "unresolved_reason_codes":np.ones(len(unresolved),np.int16),"cell_selected_group_ids":selected,
      "cell_semantic_status":status,"cell_p1_score_denominator":denom,"cell_p1_winner_score_numerator":winner,
      "cell_group_offsets":np.asarray(cell_go,np.int64),"cell_group_ids":np.asarray(cell_gid,np.int32),
      "cell_group_p1_score_numerators":np.asarray(cell_num,np.int64),"cell_group_p1_score_denominators":np.asarray(cell_den,np.int64),
      "cell_group_parent_tet_offsets":np.asarray(cg_po,np.int64),"cell_group_parent_tet_rows":np.asarray(cg_pr,np.int32),
      "cell_group_stable_face_offsets":np.asarray(cg_fo,np.int64),"cell_group_stable_face_rows":np.asarray(cg_fr,np.uint32),
    }

def build_region_adjacency(parent:dict,role:str)->tuple[dict[int,set[int]],dict]:
    t=np.asarray(parent["tets"],np.int64);m=np.asarray(parent["materials"],np.int32);r=np.asarray(parent["regions"],np.int32)
    if t.ndim!=2 or t.shape[1]!=4 or m.shape!=(len(t),) or r.shape!=(len(t),):raise ValueError("source330 topology/semantic rows are misaligned")
    face_map=defaultdict(list)
    for row,tet in enumerate(t):
        for local in LOCAL_FACES:
            face_map[tuple(sorted(int(tet[k]) for k in local))].append(row)
    mat=ROLE_MATERIAL[role];adj=defaultdict(set);proofs={}
    for region in np.unique(r[m==mat]):adj[int(region)].add(int(region))
    for face,rows in face_map.items():
        if len(rows)>2:raise ValueError(f"nonmanifold source330 face {face}")
        if len(rows)!=2:continue
        a,b=map(int,rows)
        if m[a]!=mat or m[b]!=mat:continue
        ra,rb=int(r[a]),int(r[b])
        adj[ra].add(rb);adj[rb].add(ra)
        if ra!=rb:
            proofs.setdefault(tuple(sorted((ra,rb))),{"source_parent_tet_rows":[a,b],"shared_source_node_rows":list(face)})
    return dict(adj),proofs

def expand_onehop(cell_groups:dict[str,np.ndarray],groups:np.ndarray,adj:dict[int,set[int]],role:str,ncell:int)->dict[str,np.ndarray]:
    ids=np.asarray(cell_groups["cell_group_ids"],np.int64);offs=np.asarray(cell_groups["cell_group_offsets"],np.int64)
    child=np.repeat(np.arange(ncell,dtype=np.int64),np.diff(offs))
    if len(child)!=len(ids):raise ValueError("cell P1 group CSR mismatch before 143 closure")
    direct=set();factor=int(max([0,*adj.keys(),*(int(x[2]) for x in groups)])+1)
    for ci,gid in zip(child,ids):
        g=tuple(int(x) for x in groups[int(gid)])
        if g[0]!=ROLE_MATERIAL[role]:raise ValueError(f"cell {ci} direct group crosses role material")
        direct.add((int(ci),int(g[2])))
    expanded=set()
    for ci,region in direct:
        neigh=adj.get(region)
        if not neigh:raise ValueError(f"source330 one-hop owner has no direct region {region}")
        for r in neigh:expanded.add((ci,int(r)))
    if not direct.issubset(expanded):raise ValueError("143 closure lost a direct P1 region")
    rows=[[] for _ in range(ncell)]
    for ci,r in sorted(expanded):rows[ci].append(r)
    if any(not x for x in rows):raise ValueError("one-hop current child has no admissible source region")
    out=[0];idsout=[]
    for x in rows:idsout.extend(sorted(set(x)));out.append(len(idsout))
    offsets=np.asarray(out,np.int64);regions=np.asarray(idsout,np.int32)
    return {"cell_admissible_region_offsets":offsets,"cell_admissible_region_ids":regions,
            "admissibility_max_source_region_hops":np.asarray(1,np.int32)}

def _serialize_node_maps(points:np.ndarray,node_support:dict,groups:np.ndarray,parent:dict)->dict[str,np.ndarray]:
    po=[0];pr=[];fo=[0];fr=[];go=[0];gi=[]
    faces=np.asarray(parent["faces"],np.uint32)
    for m in node_support["maps"]:
        if not m:raise ValueError("cannot serialize unsupported current node")
        for gid in sorted(m):
            rows=np.asarray(sorted(set(map(int,m[gid]))),np.int64)
            tup=tuple(int(x) for x in groups[int(gid)])
            if not len(rows):raise ValueError("empty current node parent support")
            if np.any(rows<0) or np.any(rows>=len(faces)):raise ValueError("node support source parent out of range")
            if any((int(parent["materials"][r]),int(parent["donors"][r]),int(parent["regions"][r]))!=tup for r in rows):
                raise ValueError("node support parent tuple disagrees with role group")
            if tup[0] not in ROLE_MATERIAL.values():raise ValueError("node support tuple material invalid")
            gi.append(int(gid));pr.extend(rows.tolist());po.append(len(pr));fr.extend(np.unique(faces[rows]).tolist());fo.append(len(fr))
        go.append(len(gi))
    return {"points_world_m_f32":np.asarray(points,np.float32),
      "node_group_offsets":np.asarray(go,np.int64),"node_group_ids":np.asarray(gi,np.int32),
      "node_group_parent_tet_offsets":np.asarray(po,np.int64),"node_group_parent_tet_rows":np.asarray(pr,np.int32),
      "node_group_stable_face_offsets":np.asarray(fo,np.int64),"node_group_stable_face_rows":np.asarray(fr,np.uint32),
      "node_origin_kind":np.asarray(node_support["origin"],np.uint8),
      "node_parent_registration_distance_m":np.asarray(node_support["distance"],np.float64),
      "group_tuples":np.asarray(groups,np.int32)}

def _write_support_pair(base_arrays:dict[str,np.ndarray],onehop_arrays:dict[str,np.ndarray],
                        out_dir:Path,role:str)->tuple[Path,Path,str]:
    if out_dir.exists():raise FileExistsError(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True)
    base_path=out_dir/f"{role.lower()}-current-p1-support.npz"
    support_path=out_dir/f"{role.lower()}-current-p1-143-support.npz"
    np.savez_compressed(base_path,**base_arrays)
    base_sha=sha(base_path)
    full=dict(base_arrays)
    full.update(onehop_arrays)
    full["admissibility_source_support_sha256"]=np.asarray(base_sha)
    full["admissibility_max_source_region_hops"]=np.asarray(1,np.int32)
    np.savez_compressed(support_path,**full)
    for path,expected in ((base_path,base_arrays),(support_path,full)):
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!=set(expected):raise ValueError(f"serialized {role} support member set changed: {path.name}")
            for k,v in expected.items():
                if not arrays_equal(z[k],v):raise ValueError(f"serialized {role} support roundtrip mismatch: {k}")
    with np.load(support_path,allow_pickle=False) as z:
        if str(z["admissibility_source_support_sha256"].item())!=base_sha:
            raise ValueError("143 support does not bind exact base-P1 artifact")
        if int(z["admissibility_max_source_region_hops"])!=1:
            raise ValueError("143 support changed the one-hop limit")
    return base_path,support_path,base_sha


def _validate_role_ele(role:str,role_ele_path:Path,handoff_receipt_path:Path,
                       state_pin:dict,volume_pin:dict,canonical_report_pin:dict,ncells:int)->dict:
    receipt=json.loads(handoff_receipt_path.read_text())
    if receipt.get("schema")!="numi.h706.current-fine-217-combined-volume-handoff.v1" or receipt.get("status")!="PREPARED_EXISTING_217_COMBINED_CHILD_VOLUME_HANDOFF_NO_PHYSICAL_ADMISSION":
        raise ValueError("role element is not from the pinned source-only 937 217 handoff")
    if receipt.get("physical_admission") is not False or receipt.get("native_steps")!=0 or receipt.get("mass_mapping")!="NOT_RUN":
        raise ValueError("217 role element handoff has unexpected admission or mass status")
    if sha(OWNER217_BASE)!=OWNER217_BASE_SHA:raise ValueError("unmodified 217 source003 base owner changed")
    if sha(OWNER217)!=OWNER217_SHA:raise ValueError("pinned 938 active-I 217 owner derivative changed")
    owner_pin=receipt.get("owner",{})
    if Path(owner_pin.get("path","")).resolve()!=OWNER217.resolve() or owner_pin.get("sha256")!=OWNER217_SHA:
        raise ValueError("217 element handoff used a different active-I 217 owner")
    base_owner=owner_pin.get("derived_from",{})
    if Path(base_owner.get("path","")).resolve()!=OWNER217_BASE.resolve() or base_owner.get("sha256")!=OWNER217_BASE_SHA:
        raise ValueError("217 element handoff does not bind the exact unmodified source003 base owner")
    st=receipt.get("inputs",{}).get("current_state",{})
    if Path(st.get("path","")).resolve()!=Path(state_pin["path"]) or st.get("sha256")!=state_pin["sha256"]:
        raise ValueError("217 role element handoff is bound to a different current state")
    cr=receipt.get("inputs",{}).get("canonical_834_report",{})
    if Path(cr.get("path","")).resolve()!=Path(canonical_report_pin["path"]) or cr.get("sha256")!=canonical_report_pin["sha256"]:
        raise ValueError("217 element handoff is bound to a different 834 report")
    key=role.lower()+"_volume";vp=receipt.get("inputs",{}).get(key,{})
    if Path(vp.get("path","")).resolve()!=Path(volume_pin["path"]) or vp.get("sha256")!=volume_pin["sha256"]:
        raise ValueError("217 role element handoff is bound to a different canonical volume")
    mesh=receipt.get("role_meshes",{}).get(role,{})
    if mesh.get("active_cell_count")!=ncells or mesh.get("cell_order_preserved") is not True or mesh.get("tetrahedra_roundtrip_exact") is not True:
        raise ValueError("217 role element handoff does not preserve the current role cell domain")
    filename=role.lower()+".ele"
    out=receipt.get("outputs",{}).get(filename,{})
    resolved=role_ele_path.resolve()
    if resolved!=handoff_receipt_path.resolve().parent/filename:
        raise ValueError("role .ele path is not the exact output named by its handoff receipt")
    if not resolved.is_file() or out.get("sha256")!=sha(resolved):
        raise ValueError("role .ele bytes differ from the pinned 217 handoff")
    return {"handoff_receipt":pin(handoff_receipt_path),"role_ele":pin(resolved),
            "role_cell_count":int(ncells),"cell_order_preserved":True,
            "canonical_834_report":canonical_report_pin,"owner":{"path":str(OWNER217),"sha256":OWNER217_SHA,
              "derived_from":{"path":str(OWNER217_BASE),"sha256":OWNER217_BASE_SHA}}}


def build(role:str,current_state_path:Path,volume_path:Path,canonical_report_path:Path,
          parent_csv:Path,owner_path:Path,out_dir:Path,role_ele_path:Path,handoff_receipt_path:Path,
          cache_support:Path|None=None,cache_volume:Path|None=None,
          cache_receipt:Path|None=None,max_distance:float=MAX_PARENT_DISTANCE_M,
          tie_tolerance:float=AMBIGUITY_TOLERANCE_M)->dict:
    if role not in ROLE_MATERIAL:raise ValueError("role must be Skin or Fat")
    if parent_csv.resolve()!=PARENT.resolve() or sha(parent_csv)!=PARENT_SHA:raise ValueError("source330 current parent mass path/SHA differs from immutable identity control")
    if not FULL86.is_file() or sha(FULL86)!=FULL86_SHA:raise ValueError("immutable source330 full86 weights path/SHA differs from current support contract")
    if owner_path.resolve()!=OWNER230.resolve() or sha(owner_path)!=OWNER230_SHA:raise ValueError("current 230 owner path/SHA differs from frozen implementation")
    if out_dir.exists():raise FileExistsError(f"refusing to overwrite {out_dir}")
    if max_distance!=MAX_PARENT_DISTANCE_M or tie_tolerance!=AMBIGUITY_TOLERANCE_M:raise ValueError("current role support must retain 230 distance/tie policy")
    cache_args=(cache_support,cache_volume,cache_receipt)
    if any(x is not None for x in cache_args) and not all(x is not None for x in cache_args):
        raise ValueError("cache support, its exact volume, and its receipt must be supplied together")
    started=time.monotonic()
    current_pair=_validate_current_834_pair(current_state_path,canonical_report_path,role,volume_path)
    vol=_role_volume(volume_path,current_state_path,role)
    current=vol["points"];tets=vol["tets"];ncell=len(tets)
    if len(np.unique(_point_key(current)))!=len(current):raise ValueError("current volume has duplicate exact F32 point rows")
    reg=_load_230(owner_path)
    parent,index,material_index,parent_meta=reg.load_parent_current(parent_csv)
    pm=np.asarray(parent["materials"],np.int32);pd=np.asarray(parent["donors"],np.int32);pr=np.asarray(parent["regions"],np.int32)
    if pm.ndim!=1 or len(pm)!=len(pd) or len(pm)!=len(pr) or len(pm)!=len(parent["faces"]):raise ValueError("230 source330 semantic rows are misaligned")
    groups,lookup=_source_groups(parent,role)
    cache=None;cachepins=None
    if cache_support is not None:
        cache,cachepins=_load_cache(cache_support,cache_volume,cache_receipt,role,groups,lookup,parent,owner_path,parent_meta)
    def progress(item):print(json.dumps(item,sort_keys=True),flush=True)
    ns=build_node_support(current,role,parent,index,material_index,groups,lookup,reg,cache,max_distance,tie_tolerance,progress)
    node_arrays=_serialize_node_maps(current,ns,groups,parent)
    p1=integrate_p1_cells(tets,ns["maps"],groups,parent,role)
    adj,proofs=build_region_adjacency(parent,role)
    p1.update(expand_onehop(p1,groups,adj,role,ncell))
    volume_sha=sha(volume_path)
    base_arrays={**p1,**node_arrays,
      "geometry_sha256":np.asarray(volume_sha),
      "current_state_sha256":np.asarray(vol["state_pin"]["sha256"]),
      "current_state_path":np.asarray(vol["state_pin"]["path"]),
      "volume_sha256":np.asarray(volume_sha),
      "source330_mass_sha256":np.asarray(PARENT_SHA)}
    if base_arrays["cell_group_offsets"].shape!=(ncell+1,) or base_arrays["cell_group_parent_tet_offsets"].shape!=(len(base_arrays["cell_group_ids"])+1,):
        raise ValueError("current P1 output CSR domains failed")
    if base_arrays["node_group_offsets"].shape!=(len(current)+1,) or base_arrays["node_group_parent_tet_offsets"].shape!=(len(base_arrays["node_group_ids"])+1,):
        raise ValueError("current 230 node map CSR domains failed")
    onehop={k:p1[k] for k in ("cell_admissible_region_offsets","cell_admissible_region_ids",
                              "admissibility_max_source_region_hops")}
    if len(onehop["cell_admissible_region_offsets"])!=ncell+1:raise ValueError("143 output is not full current child domain")
    ele_binding=_validate_role_ele(role,role_ele_path,handoff_receipt_path,vol["state_pin"],vol["volume_pin"],
                                   current_pair["canonical_834_report"],ncell)
    out_dir.parent.mkdir(parents=True,exist_ok=True)
    base_path,support_path,base_sha=_write_support_pair(base_arrays,onehop,out_dir,role)
    arrays={**base_arrays,**onehop}
    status=ROLE_STATUS[role]
    direct_child=np.repeat(np.arange(ncell,dtype=np.int64),np.diff(arrays["cell_group_offsets"]))
    direct_regions=groups[arrays["cell_group_ids"],2].astype(np.int64)
    factor=int(max([0,*adj.keys(),*(int(x[2]) for x in groups)])+1)
    direct_region_slot_count=int(len(np.unique(direct_child*factor+direct_regions)))
    receipt={"schema":SCHEMA,"status":status,"physical_admission":False,"native_steps":0,
      "mass_transport":"NOT_RUN","role":role,"material_index":ROLE_MATERIAL[role],
      "current_state":vol["state_pin"],"current_volume":vol["volume_pin"],
      "source330_mass":pin(parent_csv),"source330_full86_weights":pin(FULL86),"source230_owner":pin(owner_path),
      "role_mesh_217":ele_binding,"canonical_current_pair":current_pair,"node_support_cache":cachepins,
      "inputPins":{"source330_parent":{
          "mass_sha256":parent_meta["massSHA256"],"nodes_sha256":parent_meta["nodeSHA256"],
          "tets_sha256":parent_meta["tetsSHA256"],"labels_sha256":parent_meta["labelsSHA256"],
          **({"full86_weights_sha256":sha(Path(str(parent_csv)+".weights.csv"))} if role=="Skin" else {})},
        "073_receipt":{"element_sha256":ele_binding["role_ele"]["sha256"]}},
      "counts":{"nodes":len(current),"cells":ncell,"source_groups":len(groups),
        "node_group_slots":int(len(arrays["node_group_ids"])),"node_parent_rows":int(len(arrays["node_group_parent_tet_rows"])),
        "p1_group_slots":int(len(arrays["cell_group_ids"])),"unique_max_cells":int(len(arrays["resolved_child_tet_rows"])),
        "tie_cells":int(len(arrays["unresolved_child_tet_rows"])),"direct_cell_region_slots":direct_region_slot_count,
        "admissible_cell_region_slots":int(len(arrays["cell_admissible_region_ids"])),"shared_face_region_edges":len(proofs),
        "reused_exact_cache_points":ns["reused_exact_points"],"queried_current_points":ns["queried_points"],
        "query_distance_max_m":ns["query_distance_max_m"],"query_distance_p99_m":ns["query_distance_p99_m"]},
      "source330":{"massSHA256":parent_meta["massSHA256"],"nodeSHA256":parent_meta["nodeSHA256"],
        "tetsSHA256":parent_meta["tetsSHA256"],"labelsSHA256":parent_meta["labelsSHA256"]},
      "source330_parent_identity":{"current_positive_tets":int(parent_meta.get("currentPositiveTets",len(pm))),
        "mass_csv_sha256":PARENT_SHA,"source_owner_sha256":OWNER230_SHA},
      "admissibility":{"policy":"same-material source330 tetrahedra sharing an exact three-node face; direct P1 regions are retained",
        "max_source_region_hops":1,"shared_face_region_edges":[{"regions":list(k),**v} for k,v in sorted(proofs.items())]},
      "canonical_parent_unions":{"row_domain":vol["metadata"]["canonical_parent_union_parent_row_domain"],
        "rows_sha256":hashlib.sha256(np.asarray(vol["canonical_rows"],dtype="<i8").tobytes()).hexdigest(),
        "source330_cell_ancestry_claim":False},
      "outputs":{"base_p1":{"path":str(base_path),"sha256":sha(base_path),"size_bytes":base_path.stat().st_size},
        "p1_plus_authenticated_143":{"path":str(support_path),"sha256":sha(support_path),"size_bytes":support_path.stat().st_size,
          "admissibility_source_support_sha256":base_sha,"max_source_region_hops":1},
        "support_npz":pin(support_path)},"elapsed_s":time.monotonic()-started,
      "limitations":["Source-only current role P1/143 support. No transport, mass allocation, native run or physical admission.",
        "Node support rows refer to the exact pinned source330 current parent domain; canonical child-cell unions remain in the explicit 834 base-volume domain.",
        "A supplied cache is reused only for exact normalized-F32 coordinate keys; every unmatched current point is queried through the pinned 230 owner.",
        "node_origin_kind code 2 means exact-F32 point support reused from a receipt-pinned cache; code 3 means a direct current-point query through 230."]}
    receipt_path=out_dir/"receipt.json";receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":status,"role":role,"support":str(support_path),"support_sha256":sha(support_path),
        "receipt":str(receipt_path),"receipt_sha256":sha(receipt_path),"counts":receipt["counts"],
        "physical_admission":False,"native_steps":0,"mass_transport":"NOT_RUN"},sort_keys=True),flush=True)
    return receipt

def run_controls()->dict:
    # P1 partition-of-unity and exact integer group score control.
    groups=np.asarray([[15,2,7],[15,3,8]],np.int32)
    maps=[{0:{0},1:{1}},{0:{0}},{0:{0},1:{1}},{1:{1}}]
    fake={"faces":np.asarray([11,12,13],np.uint32),"materials":np.asarray([15,15,15],np.int32)}
    a=integrate_p1_cells(np.asarray([[0,1,2,3]],np.int64),maps,groups,fake,"Skin")
    den=int(a["cell_p1_score_denominator"][0])
    assert int(a["cell_group_p1_score_numerators"].sum())==den
    assert den>0 and np.isclose(float(a["cell_group_p1_score_numerators"].sum()/den),1.0,rtol=0,atol=0)
    # Every linear nodal field retains its standard tetrahedral P1 average:
    # the corner partition is 1/4 per vertex, independent of semantic splitting.
    xyz=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float64)
    f=2*xyz[:,0]-3*xyz[:,1]+4*xyz[:,2]+7
    assert np.isclose(float(f.mean()),7.75,rtol=0,atol=1e-15)
    # Exact shared-face adjacency is same-material only; a nearby/different
    # material tetrahedron cannot extend the region graph.
    pt=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1],[3,4,5]],np.float32)
    parent={"tets":np.asarray([[0,1,2,3],[0,2,1,4],[0,3,4,5]],np.int64),
            "materials":np.asarray([15,15,16],np.int32),"regions":np.asarray([7,8,99],np.int32),
            "donors":np.asarray([2,2,3],np.int32),"faces":np.asarray([101,102,103],np.uint32)}
    adj,proof=build_region_adjacency(parent,"Skin")
    assert adj[7]=={7,8} and adj[8]=={8,7} and 99 not in adj.get(7,set())
    # The current node map reuses only an exact F32 point key; a new row goes
    # through the supplied 230 closest_same_material callable.
    node_groups,node_lookup=_source_groups(parent,"Skin")
    class Fake230:
        calls=0
        def closest_same_material(self,parent,index,material_index,material,point,max_distance,tie_tolerance):
            self.calls+=1
            assert material==15 and max_distance==MAX_PARENT_DISTANCE_M
            return {"ties":[1],"tet":1,"distance":0.005}
    fake=Fake230()
    current_pts=np.asarray([[0,0,0],[0.2,0.2,0.2]],np.float32)
    cached_gid=int(node_lookup[(15,2,7)])
    cache={"points":current_pts[:1].copy(),"maps":[{cached_gid:{0}}],"distance":np.asarray([0.01])}
    ns=build_node_support(current_pts,"Skin",parent,object(),object(),node_groups,node_lookup,
                          fake,cache=cache)
    assert ns["reused_exact_points"]==1 and ns["queried_points"]==1 and fake.calls==1
    assert ns["maps"][0]=={cached_gid:{0}}
    assert node_groups[int(next(iter(ns["maps"][1])))].tolist()==[15,2,8]
    # Wrong material row in a semantic group must be rejected during integration.
    badgroups=np.asarray([[16,2,7]],np.int32);badmaps=[{0:{0}}]*4
    badparent={"faces":np.asarray([11],np.uint32),"materials":np.asarray([16],np.int32)}
    try:integrate_p1_cells(np.asarray([[0,0,0,0]],np.int64),badmaps,badgroups,badparent,"Skin")
    except ValueError:pass
    else:raise AssertionError("cross-material support accepted")
    # Current state/current volume binding control.
    with tempfile.TemporaryDirectory() as td:
        d=Path(td);src=d/"raw.npz";state=d/"state.json";vol=d/"canon.npz"
        np.savez_compressed(src,points_world_m_f32=np.zeros((4,3),np.float32),tetrahedra=np.asarray([[0,1,2,3]]),
                            material_indices=np.asarray([15]))
        state.write_text(json.dumps({"Skin_current":pin(src),"Fat_current":pin(src),
                                    "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}))
        np.savez_compressed(vol,points_world_m_f32=np.zeros((4,3),np.float32),tetrahedra=np.asarray([[0,1,2,3]]),
          material_indices=np.asarray([15]),canonical_parent_union_offsets=np.asarray([0,1]),
          canonical_parent_union_parent_rows=np.asarray([0]),canonical_parent_union_current_state_path=np.asarray(str(state.resolve())),
          canonical_parent_union_current_state_sha256=np.asarray(sha(state)),
          canonical_parent_union_current_volume_path=np.asarray(str(src.resolve())),
          canonical_parent_union_current_volume_sha256=np.asarray(sha(src)),
          canonical_parent_union_base_state_path=np.asarray(str(state.resolve())),
          canonical_parent_union_base_state_sha256=np.asarray(sha(state)),
          canonical_parent_union_base_volume_path=np.asarray(str(src.resolve())),
          canonical_parent_union_base_volume_sha256=np.asarray(sha(src)),
          canonical_parent_union_parent_row_domain=np.asarray("canonical_base_volume_cell_rows"),
          canonical_parent_union_physical_status=np.asarray("SOURCE_ONLY_PHYSICAL_ADMISSION_FALSE"))
        v=_role_volume(vol,state,"Skin")
        assert len(v["tets"])==1
        with np.load(vol,allow_pickle=False) as z:q={k:z[k] for k in z.files}
        q["canonical_parent_union_current_state_sha256"]=np.asarray("0"*64)
        bad=d/"bad.npz";np.savez_compressed(bad,**q)
        try:_role_volume(bad,state,"Skin")
        except ValueError:pass
        else:raise AssertionError("stale current-state volume accepted")
    return {"status":"PASS_CURRENT_ROLE_P1_SUPPORT_CONTROLS","tests":["exact integer P1 partition of unity","affine tetrahedral nodal averaging","exact same-material shared-face 143 closure","cross-material P1 group rejected","exact-coordinate cache reuse and new-point 230 query","current-state volume pin mismatch rejected"],"physical_admission":False,"native_steps":0,"mass_transport":"NOT_RUN"}

def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--role",choices=("Skin","Fat"))
    ap.add_argument("--current-state",type=Path)
    ap.add_argument("--volume-npz",type=Path)
    ap.add_argument("--canonical-834-report",type=Path)
    ap.add_argument("--source330-parent-mass",type=Path,default=PARENT)
    ap.add_argument("--source230-owner",type=Path,default=OWNER230)
    ap.add_argument("--role-ele",type=Path)
    ap.add_argument("--handoff-receipt",type=Path)
    ap.add_argument("--cache-support-npz",type=Path)
    ap.add_argument("--cache-volume-npz",type=Path)
    ap.add_argument("--cache-receipt",type=Path)
    ap.add_argument("--output-dir",type=Path)
    ap.add_argument("--maximum-parent-distance-m",type=float,default=MAX_PARENT_DISTANCE_M)
    ap.add_argument("--ambiguity-tolerance-m",type=float,default=AMBIGUITY_TOLERANCE_M)
    a=ap.parse_args()
    if a.self_test:
        print(json.dumps(run_controls(),indent=2,sort_keys=True));return 0
    required=(a.role,a.current_state,a.volume_npz,a.canonical_834_report,a.output_dir,a.role_ele,a.handoff_receipt)
    if any(x is None for x in required):ap.error("role, current-state, volume-npz, canonical-834-report, role-ele, handoff-receipt, and output-dir are required")
    build(a.role,a.current_state,a.volume_npz,a.canonical_834_report,a.source330_parent_mass,a.source230_owner,a.output_dir,
          a.role_ele,a.handoff_receipt,a.cache_support_npz,a.cache_volume_npz,a.cache_receipt,
          a.maximum_parent_distance_m,a.ambiguity_tolerance_m)
    return 0
if __name__=="__main__":raise SystemExit(main())
