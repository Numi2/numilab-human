#!/usr/bin/env python3
"""Source-only bridge from canonical role NPZ volumes to existing 217 child tables.

No TetGen is invoked. The bridge preserves each role cell row and emits the
existing 217 .node/.ele, maps, and handoff receipt. Original source330 ancestry
is not inferred: canonical parent unions remain in the pinned per-role 834 NPZs.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, importlib.util, json, math, os, shutil, tempfile, time
from pathlib import Path
from typing import Any
import numpy as np

HERE=Path(__file__).resolve().parent
H=HERE.parent
P=H.parent
OWNER_BASE_PATH=P/"paired-head-reference-fit-684/run-063-registered-native-wall-frame-002/native-domain-217-source-003/build_native_domain_217.py"
OWNER_BASE_SHA="d7f411c18f91905c6e1d91ab75592ec5782a8852892ac8a1fdb566ee2206df09"
OWNER_PATH=H/"current-role-p1-support-938/current_fine_217_owner_938_v2_coordinate_alias_guard.py"
OWNER_SHA="1fc5d6278aa0aa9bd76105d4bf7caa2c1ada6431428e20fe16c28e4afad3f6e4"
OWNER834=H/"canonical-volume-parent-unions-834-nativepreflight/canonical_volume_parent_unions_834.py"
OWNER834_SHA="02120822bb280cc96b312a14f47b26ecaec275f5fcb50e82c2d2ec6c5f92b0f3"
OWNER849=H/"canonical-volume-parent-unions-849-nativepreflight/build_current_manifest_849.py"
OWNER849_SHA="860c1eba9ef9158a3fc5f5dd4fad5883bc9a43b9f4b3cf9a389bfe66c86636ce"
OWNER818=H/"bounded625-local-cavity-repair/finalize_818b_volume_only_current_source_prefix.py"
OWNER818_SHA="9b157b92721867465b1d9439846ee6f2c317c31816abaddc91e6ef766a39bdf4"
CANONICAL_REPORT_SCHEMA="numi.h706.current-volume-parent-union-rebuild.v1"
MANIFEST_SCHEMA="numi.h706.current-volume-parent-union-rebuild-input.v1"
MANIFEST_REPORT_SCHEMA="numi.h706.auto-current-volume-parent-union-manifest.v1"
PREFIX_SCHEMA="numi.quality818b.current-source-volume-metadata-prefix.v1"
HANDOFF_STATUS="PREPARED_EXISTING_217_COMBINED_CHILD_VOLUME_HANDOFF_NO_PHYSICAL_ADMISSION"
RECEIPT_SCHEMA="numi.h706.current-fine-217-combined-volume-handoff.v1"
ROLE_MATERIAL={"Skin":15,"Fat":16}
UNION_KEYS=("canonical_parent_union_offsets","canonical_parent_union_parent_rows")

def sha(path: str|Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def pin(path: str|Path)->dict[str,Any]:
    p=Path(path).resolve()
    return {"path":str(p),"sha256":sha(p),"size_bytes":p.stat().st_size}

def _scalar(a:np.ndarray)->str:
    if np.asarray(a).size!=1: raise ValueError("expected scalar metadata array")
    return str(np.asarray(a).reshape(-1)[0])

def import_owner():
    if sha(OWNER_BASE_PATH)!=OWNER_BASE_SHA: raise ValueError("pinned unmodified 217 source003 base owner changed")
    if sha(OWNER_PATH)!=OWNER_SHA: raise ValueError("pinned 938 active-I 217 owner derivative hash changed")
    spec=importlib.util.spec_from_file_location("current_fine_217_owner_937",OWNER_PATH)
    if spec is None or spec.loader is None: raise ImportError(str(OWNER_PATH))
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.configure_capacity_profile("full-reference-500k-nodes-2m-cells")
    return m

def read_state(path:Path)->dict:
    raw=json.loads(path.read_text())
    queue=[raw]; seen=set(); state=None
    while queue:
        q=queue.pop(0)
        if not isinstance(q,dict) or id(q) in seen: continue
        seen.add(id(q))
        if isinstance(q.get("Skin_current"),dict) and isinstance(q.get("Fat_current"),dict):
            state=q; break
        for k in ("current_verified_state_20261031","current","verified_state","state"):
            if isinstance(q.get(k),dict): queue.append(q[k])
    if state is None: raise ValueError("current verified state lacks paired Skin/Fat current volume entries")
    if state.get("physical_admission") is True or state.get("native_steps",0)!=0:
        raise ValueError("current state is not source-only")
    mass=state.get("mass_mapping",state.get("mass_reconciliation","PENDING"))
    if mass not in ("PENDING","NOT_RUN","PENDING_VOLUME_CHANGE_RECONCILIATION"):
        raise ValueError("current state has unexpected completed mass status")
    return state

def validate_834_binding(state_path:Path,report_path:Path,role_paths:dict[str,Path])->dict:
    state_path=state_path.resolve(); report=json.loads(report_path.read_text()); state=read_state(state_path)
    if sha(OWNER834)!=OWNER834_SHA: raise ValueError("pinned 834 source changed")
    if sha(OWNER849)!=OWNER849_SHA: raise ValueError("pinned 849 source changed")
    if sha(OWNER818)!=OWNER818_SHA: raise ValueError("pinned 818b source changed")
    if report.get("schema")!=CANONICAL_REPORT_SCHEMA or report.get("status")!="PASS_SOURCE_ONLY_CANONICAL_CURRENT_VOLUME_PARENT_UNIONS":
        raise ValueError("834 report is not a passing source-only canonical union report")
    if report.get("physical_admission") is not False or report.get("native_steps")!=0 or report.get("mass_mapping")!="NOT_RUN":
        raise ValueError("834 report has unexpected admission, execution, or mass status")
    current=report.get("current_state",{})
    if Path(current.get("path","")).resolve()!=state_path or current.get("sha256")!=sha(state_path):
        raise ValueError("834 report does not bind the supplied current state")
    m=report.get("manifest",{})
    mp=Path(m.get("path",""))
    if not mp.is_file() or sha(mp)!=m.get("sha256"): raise ValueError("834 ancestry manifest pin is invalid")
    manifest=json.loads(mp.read_text())
    if manifest.get("schema")!=MANIFEST_SCHEMA: raise ValueError("834 input manifest schema is not the 849 canonical-current manifest")
    if Path(manifest.get("current_state",{}).get("path","")).resolve()!=state_path or manifest.get("current_state",{}).get("sha256")!=sha(state_path):
        raise ValueError("834 manifest does not bind current state")
    validated={}
    for role,p in role_paths.items():
        item=report.get("roles",{}).get(role,{})
        out=item.get("output") or {}
        if Path(out.get("path","")).resolve()!=p.resolve() or out.get("sha256")!=sha(p):
            raise ValueError(f"834 report does not bind the supplied {role} canonical volume")
        state_entry=state.get(f"{role}_current",{})
        if not isinstance(state_entry,dict): raise ValueError(f"current state has no {role} volume")
        role_current=item.get("current_volume",{})
        if (Path(state_entry.get("path","")).resolve()!=Path(role_current.get("path","")).resolve()
                or state_entry.get("sha256")!=role_current.get("sha256")):
            raise ValueError(f"{role} 834 current-volume pin differs from the paired current state")
        current_volume_path=Path(role_current.get("path","")).resolve()
        if not current_volume_path.is_file() or sha(current_volume_path)!=role_current.get("sha256"):
            raise ValueError(f"{role} 834 current-volume input pin is missing or stale")
        # The canonicalized volume is the output of the 834 chain, while current state's
        # input volume is separately authenticated by the 834 report/manifest.
        with np.load(p,allow_pickle=False) as z:
            off=np.asarray(z["canonical_parent_union_offsets"],dtype="<i8")
            rows=np.asarray(z["canonical_parent_union_parent_rows"],dtype="<i8")
            basevol_path=_scalar(z["canonical_parent_union_base_volume_path"])
            basevol_sha=_scalar(z["canonical_parent_union_base_volume_sha256"])
            if off.shape!=(len(z["tetrahedra"])+1,): raise ValueError(f"{role} 834 CSR row count mismatch")
            oh=hashlib.sha256(off.tobytes()).hexdigest(); rh=hashlib.sha256(rows.tobytes()).hexdigest()
            if Path(_scalar(z["canonical_parent_union_current_state_path"])).resolve()!=state_path or _scalar(z["canonical_parent_union_current_state_sha256"])!=sha(state_path):
                raise ValueError(f"{role} canonical NPZ is bound to a different current state")
            role_report=item
            cur=role_report.get("current_volume",{})
            if Path(_scalar(z["canonical_parent_union_current_volume_path"])).resolve()!=Path(cur.get("path","")).resolve() or _scalar(z["canonical_parent_union_current_volume_sha256"])!=cur.get("sha256"):
                raise ValueError(f"{role} canonical NPZ current volume ancestry pins differ from 834 report")
            base=report.get("base_state",{})
            if Path(_scalar(z["canonical_parent_union_base_state_path"])).resolve()!=Path(base.get("path","")).resolve() or _scalar(z["canonical_parent_union_base_state_sha256"])!=base.get("sha256"):
                raise ValueError(f"{role} canonical NPZ base state differs from 834 report")
            basevol_path=_scalar(z["canonical_parent_union_base_volume_path"])
            basevol_sha=_scalar(z["canonical_parent_union_base_volume_sha256"])
        if item.get("canonical_parent_offsets_sha256")!=oh or item.get("canonical_parent_rows_sha256")!=rh:
            raise ValueError(f"{role} canonical CSR bytes differ from 834 report")
        if item.get("canonical_parent_offsets_entries")!=len(off) or item.get("canonical_parent_references")!=len(rows):
            raise ValueError(f"{role} canonical CSR counts differ from 834 report")
        if Path(item.get("base_volume",{}).get("path","")).resolve()!=Path(basevol_path).resolve():
            raise ValueError(f"{role} canonical base volume path differs from 834 report")
        if item.get("base_volume",{}).get("sha256")!=basevol_sha:
            raise ValueError(f"{role} canonical base volume SHA differs from 834 report")
        validated[role]={"canonical_volume":pin(p),"report_role":item,
          "union_offsets_sha256":oh,"union_rows_sha256":rh}
    return {"report":report,"roles":validated,"current_state":pin(state_path),"manifest":m}

def read_union_volume(path:Path,role:str,owner, max_cells:int|None=None,max_nodes:int|None=None)->dict[str,Any]:
    with np.load(path,allow_pickle=False) as z:
        keys=set(z.files)
        for k in ("tetrahedra","material_indices",*UNION_KEYS):
            if k not in keys: raise ValueError(f"{role} canonical NPZ lacks {k}")
        point_key="points_world_m_f32" if "points_world_m_f32" in keys else "points_world_m"
        if point_key not in keys: raise ValueError(f"{role} canonical NPZ lacks Float32 points")
        p=np.asarray(z[point_key])
        t=np.asarray(z["tetrahedra"])
        mat=np.asarray(z["material_indices"])
        off=np.asarray(z[UNION_KEYS[0]],np.int64)
        rows=np.asarray(z[UNION_KEYS[1]],np.int64)
        required_meta=("canonical_parent_union_base_state_path",
          "canonical_parent_union_base_state_sha256","canonical_parent_union_base_volume_path",
          "canonical_parent_union_base_volume_sha256","canonical_parent_union_current_state_path",
          "canonical_parent_union_current_state_sha256","canonical_parent_union_current_volume_path",
          "canonical_parent_union_current_volume_sha256","canonical_parent_union_parent_row_domain",
          "canonical_parent_union_physical_status")
        missing=[k for k in required_meta if k not in keys]
        if missing: raise ValueError(f"{role} canonical NPZ lacks 834 provenance: {missing}")
        metadata={k:_scalar(z[k]) for k in required_meta}
    if p.dtype!=np.float32 or p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():
        raise ValueError(f"{role} points must be finite, exact Float32 (not silently recast)")
    if not np.issubdtype(t.dtype,np.integer) or t.ndim!=2 or t.shape[1]!=4 or not len(t):
        raise ValueError(f"{role} tetrahedra have invalid shape/type")
    t=np.asarray(t,np.int64)
    if int(t.min())<0 or int(t.max())>=len(p): raise ValueError(f"{role} tet node row out of range")
    if np.any(mat.shape!=(len(t),)) or not np.issubdtype(mat.dtype,np.integer) or not np.all(mat==ROLE_MATERIAL[role]):
        raise ValueError(f"{role} material IDs are not exact uniform owner IDs")
    if off.shape!=(len(t)+1,) or off[0]!=0 or off[-1]!=len(rows) or np.any(off[1:]<off[:-1]):
        raise ValueError(f"{role} canonical parent CSR is malformed")
    if np.any(rows<0): raise ValueError(f"{role} canonical parent CSR contains invalid rows")
    if metadata["canonical_parent_union_parent_row_domain"]!="canonical_base_volume_cell_rows" or metadata["canonical_parent_union_physical_status"]!="SOURCE_ONLY_PHYSICAL_ADMISSION_FALSE":
        raise ValueError(f"{role} canonical parent CSR has unsupported row domain or admission status")
    # Verify cells pass the exact existing owner reader after its standard 9-significant
    # digit TetGen text roundtrip. Preserve row IDs by emitting original point-row IDs.
    used=np.unique(t.reshape(-1))
    if len(used)==0: raise ValueError(f"{role} has no active points")
    inverse=np.full(len(p),-1,dtype=np.int64); inverse[used]=np.arange(len(used),dtype=np.int64)
    active_t=inverse[t]
    if np.any(active_t<0): raise ValueError(f"{role} active-node compaction failed")
    max_nodes=owner.MAX_NODES if max_nodes is None else min(int(max_nodes),int(owner.MAX_NODES))
    max_cells=owner.MAX_CELLS if max_cells is None else min(int(max_cells),int(owner.MAX_CELLS))
    if len(used)>max_nodes or len(t)>max_cells: raise ValueError(f"{role} exceeds selected 217 capacity")
    if int(used[-1])+1>0xffffffff or len(t)>0xffffffff: raise ValueError(f"{role} source row IDs exceed TetGen UInt32 IDs")
    return {"points":p,"tets":t,"materials":np.asarray(mat,np.int32),"union_offsets":off,
      "union_rows":rows,"union_hashes":{"offsets_sha256":hashlib.sha256(off.astype('<i8').tobytes()).hexdigest(),
      "rows_sha256":hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest()},
      "metadata":metadata,"active_input_rows":used,"active_points":p[used].copy(),
      "active_tets":active_t,"dropped_point_rows":np.flatnonzero(inverse<0)}

def write_role_tables(owner,vol:dict[str,Any],role:str,node_path:Path,ele_path:Path)->dict:
    p=vol["active_points"]; t=vol["active_tets"]; mat=vol["materials"]
    used=vol["active_input_rows"]
    # Node IDs encode the original canonical NPZ node row + 1. This preserves the
    # row crosswalk through the existing combined-node map while compacting unused rows.
    node_ids=used+1
    with node_path.open("w") as f:
        f.write(f"{len(p)} 3 0 0\n")
        for node_id,xyz in zip(node_ids,p):
            f.write(f"{int(node_id)} {float(xyz[0]):.9g} {float(xyz[1]):.9g} {float(xyz[2]):.9g}\n")
    with ele_path.open("w") as f:
        f.write(f"{len(t)} 4 1\n")
        for r,(tet,m) in enumerate(zip(t,mat),1):
            ids=node_ids[tet]
            f.write(f"{r} {int(ids[0])} {int(ids[1])} {int(ids[2])} {int(ids[3])} {int(m)}\n")
    checked=owner.read_tetgen_mesh(node_path,ele_path,ROLE_MATERIAL[role])
    if not np.array_equal(checked["points"],p): raise ValueError(f"{role} .node Float32 roundtrip changed coordinates")
    if not np.array_equal(checked["tetrahedra"],t): raise ValueError(f"{role} .ele roundtrip changed tet order/orientation")
    if not np.array_equal(checked["element_ids"],np.arange(1,len(t)+1)): raise ValueError(f"{role} element IDs not canonical")
    if np.any(checked["element_reoriented"]): raise ValueError(f"{role} parser had to reorient source tets")
    if not np.array_equal(checked["material_indices"],mat): raise ValueError(f"{role} .ele material roundtrip changed")
    return {"active_node_count":len(p),"source_point_count":len(vol["points"]),
      "removed_unreferenced_point_rows":int(len(vol["dropped_point_rows"])),
      "active_cell_count":len(t),"cell_order_preserved":True,"tetrahedra_roundtrip_exact":True,
      "coordinates_roundtrip_exact_f32":True,"material_ids_roundtrip_exact":True,
      "canonical_parent_union":vol["union_hashes"],"canonical_base_state":vol["metadata"]["canonical_parent_union_base_state_path"],
      "canonical_base_state_sha256":vol["metadata"]["canonical_parent_union_base_state_sha256"],
      "ancestry_claim":"canonical-parent rows refer only to the explicitly pinned 834 base-volume domain; source330 ancestry is not inferred"}

def load_shell_npz(path:Path)->dict[str,Any]:
    with np.load(path,allow_pickle=False) as z:
        x={k:np.array(z[k],copy=True) for k in z.files}
    required=("points_world_m","faces","face_marker","registered_I_source_vertex_rows",
      "registered_I_shell_node_rows","registered_I_source_faces","registered_I_shell_faces",
      "registered_I_shell_face_rows","registered_I_source_face_rows","registered_I_source_exact_face_rows",
      "registered_I_source_face_ids","registered_I_source_ancestry_offsets","registered_I_source_ancestry_rows",
      "face_source_role","face_ancestry_offsets","face_ancestry_source_rows")
    missing=[k for k in required if k not in x]
    if missing: raise ValueError(f"shell NPZ lacks current 217 keys {missing}")
    x["face_ancestry"]={k:v for k,v in x.items() if k.startswith("face_")}
    return x

def validate_shared_i(owner,skin_shell:dict,fat_shell:dict,inner:dict)->None:
    ip=np.asarray(inner["points"],np.float32); faces=np.asarray(inner["faces"],np.int64)
    if ip.ndim!=2 or ip.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3 or not len(faces):
        raise ValueError("current I surface point/face domains are malformed")
    if np.any(faces<0) or np.any(faces>=len(ip)):
        raise ValueError("current I surface face references an invalid historical source row")
    active=np.unique(faces.reshape(-1))
    expected=np.arange(len(ip),dtype=np.int64)
    for name,s in (("Skin",skin_shell),("Fat",fat_shell)):
        sr=np.asarray(s["registered_I_source_vertex_rows"],np.int64)
        nr=np.asarray(s["registered_I_shell_node_rows"],np.int64)
        sf=np.asarray(s["registered_I_source_faces"],np.int64)
        if not np.array_equal(sr,expected) or nr.shape!=sr.shape:
            raise ValueError(f"{name} registered-I source vertex identity is incomplete/noncanonical")
        if sf.shape!=faces.shape or not np.array_equal(sf,faces):
            raise ValueError(f"{name} registered-I source face identity differs from current I surface")
        if np.any(nr < -1):
            raise ValueError(f"{name} I map uses an unsupported negative sentinel")
        points=np.asarray(s["points_world_m"],np.float32)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
            raise ValueError(f"{name} I shell point table is malformed")
        if np.any(nr>=len(points)):
            raise ValueError(f"{name} I source row points outside the shell node table")
        if np.any(nr[active]<0):
            bad=int(active[np.flatnonzero(nr[active]<0)[0]])
            raise ValueError(f"{name} face-referenced I source row {bad} has no exact shell node mapping")
        mapped=np.flatnonzero(nr>=0)
        if len(mapped):
            got=np.ascontiguousarray(points[nr[mapped]],dtype=np.float32)
            want=np.ascontiguousarray(ip[mapped],dtype=np.float32)
            if got.view(np.uint32).tobytes()!=want.view(np.uint32).tobytes():
                diff=np.flatnonzero(np.any(got.view(np.uint32).reshape(-1,3)!=
                                           want.view(np.uint32).reshape(-1,3),axis=1))
                bad=int(mapped[int(diff[0])])
                raise ValueError(f"{name} mapped I source row {bad} does not retain exact Float32 coordinate bytes")
    if not np.array_equal(skin_shell["registered_I_source_vertex_rows"],fat_shell["registered_I_source_vertex_rows"]):
        raise ValueError("Skin/Fat registered-I source row namespaces differ")
    if not np.array_equal(skin_shell["registered_I_source_faces"],fat_shell["registered_I_source_faces"]):
        raise ValueError("Skin/Fat registered-I face row namespaces differ")
    # Source-row identity and all active interface coordinates remain exact.
    # A -1 shell mapping is accepted only for a source-I row absent from every face.

def load_shell_receipt(receipt_path:Path,skin_path:Path,fat_path:Path,state:dict)->dict:
    r=json.loads(receipt_path.read_text())
    if r.get("physical_admission") is not False: raise ValueError("shell receipt must be source-only")
    out=r.get("outputs",{})
    for name,path in (("skin-shell.npz",skin_path),("fat-shell.npz",fat_path)):
        entry=out.get(name)
        expected=entry.get("sha256") if isinstance(entry,dict) else entry
        if expected!=sha(path): raise ValueError(f"shell receipt does not bind {name}")
    inputs=r.get("inputs",{})
    if not isinstance(inputs,dict) or not inputs: raise ValueError("shell receipt lacks current source input pins")
    # Existing 217 adapter receipts have either path-keyed pins or named pins.
    values=[]
    for key,value in inputs.items():
        if isinstance(value,dict) and "path" in value and "sha256" in value:
            q=Path(value["path"])
            if not q.is_file() or sha(q)!=value["sha256"]:
                raise ValueError(f"shell receipt source input changed: {key}")
            values.append((str(q.resolve()),value["sha256"]))
        elif isinstance(value,str) and len(value)==64:
            values.append((str(Path(key).resolve()),value))
    source_pins=state.get("source_surfaces",{})
    if not isinstance(source_pins,dict) or set(source_pins)!={"O","I","D"}: raise ValueError("current state source surfaces are incomplete")
    for role,p in source_pins.items():
        path=Path(p["path"]).resolve(); digest=p["sha256"]
        if not any(q==str(path) and d==digest for q,d in values):
            raise ValueError(f"shell receipt does not bind current {role} source surface")
    return r

def execute(skin_volume:Path,fat_volume:Path,skin_shell_path:Path,fat_shell_path:Path,
            inner_path:Path,state_path:Path,canonical_report_path:Path,
            shell_receipt_path:Path,out_path:Path,low_cell_cap:int|None=None,
            manifest_849_report_path:Path|None=None,prefix_818b_report_path:Path|None=None)->dict:
    state_path=state_path.resolve()
    for path in (skin_volume,fat_volume,skin_shell_path,fat_shell_path,inner_path,state_path,canonical_report_path,shell_receipt_path):
        if not path.is_file(): raise FileNotFoundError(path)
    if out_path.exists(): raise FileExistsError(f"refusing to overwrite {out_path}")
    owner=import_owner()
    state=read_state(state_path)
    binding=validate_834_binding(state_path,canonical_report_path,{"Skin":skin_volume,"Fat":fat_volume})
    chain_pins={}
    if manifest_849_report_path is not None:
        r849=json.loads(manifest_849_report_path.read_text())
        if r849.get("schema")!=MANIFEST_REPORT_SCHEMA or r849.get("status")!="PASS_SOURCE_ONLY_AUTODISCOVERED_CELL_ANCESTRY_MANIFEST" or r849.get("physical_admission") is not False or r849.get("native_steps")!=0:
            raise ValueError("849 report is not a passing source-only current ancestry report")
        if Path(r849.get("current_state",{}).get("path","")).resolve()!=state_path or r849.get("current_state",{}).get("sha256")!=sha(state_path):
            raise ValueError("849 report does not bind current state")
        if r849.get("manifest_sha256")!=binding["report"]["manifest"]["sha256"] or Path(r849.get("manifest_path","")).resolve()!=Path(binding["report"]["manifest"]["path"]).resolve():
            raise ValueError("849 report manifest differs from the 834 input manifest")
        chain_pins["849_report"]=pin(manifest_849_report_path)
    if prefix_818b_report_path is not None:
        r818=json.loads(prefix_818b_report_path.read_text())
        if r818.get("schema")!=PREFIX_SCHEMA or r818.get("status")!="PASS_CURRENT_SOURCE_VOLUME_METADATA_PREFIX" or r818.get("geometry_changed") is not False or r818.get("physical_admission") is not False or r818.get("native_steps")!=0 or r818.get("mass_mapping")!="NOT_RUN":
            raise ValueError("818b report is not a source-only geometry-identity metadata prefix")
        outstate=r818.get("output_state",{})
        if Path(outstate.get("path","")).resolve()!=state_path or outstate.get("sha256")!=sha(state_path):
            raise ValueError("818b report does not bind supplied current state")
        prod=r818.get("producer",{})
        if Path(prod.get("path","")).resolve()!=OWNER818.resolve() or prod.get("sha256")!=OWNER818_SHA:
            raise ValueError("818b report producer pin differs from frozen owner")
        chain_pins["818b_report"]=pin(prefix_818b_report_path)
    # Validate every inherited source input that generated the current 217 shell files.
    shell_receipt=load_shell_receipt(shell_receipt_path,skin_shell_path,fat_shell_path,state)
    skin_shell=load_shell_npz(skin_shell_path); fat_shell=load_shell_npz(fat_shell_path)
    inner=owner.load_surface(inner_path,"I")
    validate_shared_i(owner,skin_shell,fat_shell,inner)
    # Explicit limit is only a stricter fixture/operations cap; never increases owner cap.
    if low_cell_cap is not None and int(low_cell_cap)<1: raise ValueError("test/operational cell cap must be positive")
    out_path.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=out_path.name+".stage-",dir=out_path.parent))
    try:
        vols={"Skin":read_union_volume(skin_volume,"Skin",owner,max_cells=low_cell_cap),
              "Fat":read_union_volume(fat_volume,"Fat",owner,max_cells=low_cell_cap)}
        if len(vols["Skin"]["active_points"])+len(vols["Fat"]["active_points"])>owner.MAX_NODES:
            raise ValueError("combined active node capacity exceeds current 217 owner profile")
        if len(vols["Skin"]["active_tets"])+len(vols["Fat"]["active_tets"])>owner.MAX_CELLS:
            raise ValueError("combined cell capacity exceeds current 217 owner profile")
        role_reports={}
        for role,stem in (("Skin","skin"),("Fat","fat")):
            role_reports[role]=write_role_tables(owner,vols[role],role,stage/f"{stem}.node",stage/f"{stem}.ele")
        combined=owner.combine_tetgen_children(stage/"skin.node",stage/"skin.ele",
          stage/"fat.node",stage/"fat.ele",skin_shell,fat_shell,inner,stage,
          require_registered_surface_maps=True)
        if combined["combined_tetrahedra"]!=len(vols["Skin"]["active_tets"])+len(vols["Fat"]["active_tets"]):
            raise ValueError("217 combiner changed combined child cell count")
        # Verify 217 role cell map is a bijection and the parent-union row join remains exact.
        ep=stage/"skin-fat-combined-element-map.csv.gz"
        expected=len(vols["Skin"]["active_tets"])+len(vols["Fat"]["active_tets"])
        seen={"skin":0,"fat":0}
        with gzip.open(ep,"rt",encoding="utf-8",newline="") as f:
            rd=csv.DictReader(f)
            wanted=["joint_element_id","child_tet_row","source_partition","source_tetgen_element_id","material_index","source_orientation_reordered_to_positive"]
            if rd.fieldnames!=wanted: raise ValueError("217 element map header changed")
            for child,row in enumerate(rd):
                role=row["source_partition"].lower()
                if role not in seen or int(row["child_tet_row"])!=child: raise ValueError("217 cell map role/row invalid")
                expected_local=seen[role]
                if int(row["source_tetgen_element_id"])!=expected_local+1:
                    raise ValueError("217 cell map shifted a role-local row")
                local_role="Skin" if role=="skin" else "Fat"
                expected_mat=ROLE_MATERIAL[local_role]
                if int(row["material_index"])!=expected_mat: raise ValueError("217 material row changed")
                seen[role]+=1
        if seen!={"skin":len(vols["Skin"]["active_tets"]),"fat":len(vols["Fat"]["active_tets"])}:
            raise ValueError("217 element map does not bijectively cover both role domains")
        outputs={p.name:{"sha256":sha(p),"size_bytes":p.stat().st_size} for p in sorted(stage.iterdir()) if p.is_file()}
        report={"schema":RECEIPT_SCHEMA,"status":HANDOFF_STATUS,
          "classification":"existing-format 217 child-volume handoff from canonical current NPZ; no TetGen run",
          "construction_method":"exact Float32 role NPZ to existing .node/.ele plus pinned 217 combiner",
          "inputs":{"skin_volume":pin(skin_volume),"fat_volume":pin(fat_volume),
            "current_state":binding["current_state"],"canonical_834_report":pin(canonical_report_path),
            "canonical_834_manifest":binding["report"]["manifest"],"canonical_849_report":chain_pins.get("849_report"),"volume_prefix_818b_report":chain_pins.get("818b_report"),
            "skin_shell":pin(skin_shell_path),"fat_shell":pin(fat_shell_path),
            "inner_source_surface":pin(inner_path),"shell_assembly_receipt":pin(shell_receipt_path)},
          "owner":{"path":str(OWNER_PATH),"sha256":OWNER_SHA,"capacity_profile":owner.capacity_profile_receipt(),
            "derived_from":{"path":str(OWNER_BASE_PATH),"sha256":OWNER_BASE_SHA},
            "active_I_mapping_policy":"all face-referenced source-I rows map exactly; -1 is permitted only for rows absent from every current I face"},
          "role_meshes":role_reports,"canonical_parent_unions":{
            role:{"offsets_sha256":vols[role]["union_hashes"]["offsets_sha256"],
                  "rows_sha256":vols[role]["union_hashes"]["rows_sha256"],
                  "cell_rows":int(len(vols[role]["active_tets"])),
                  "base_state_path":vols[role]["metadata"]["canonical_parent_union_base_state_path"],
                  "base_state_sha256":vols[role]["metadata"]["canonical_parent_union_base_state_sha256"],
                  "source330_ancestry_claim":False} for role in ("Skin","Fat")},
          "combined_skin_fat_child_mesh":combined,"outputs":outputs,
          "tetgen_invoked":False,"physical_admission":False,"native_steps":0,
          "runtime":0,"mass_mapping":"NOT_RUN","limitations":[
            "Canonical parent CSR remains in the per-role input NPZs and joins through the existing 217 element map; its rows refer to the explicit 834 base-volume domain.",
            "This handoff does not assert original source330 cell ancestry, mass transfer, or native/physical admission."]}
        # The handoff receipt is pinned externally; it is intentionally excluded
        # from its own outputs map to avoid a self-referential digest.
        (stage/"adapter-receipt.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        os.replace(stage,out_path)
        report["receipt"]=pin(out_path/"adapter-receipt.json")
        return report
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise

def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skin-volume-npz",type=Path,required=True);ap.add_argument("--fat-volume-npz",type=Path,required=True)
    ap.add_argument("--skin-shell-npz",type=Path,required=True);ap.add_argument("--fat-shell-npz",type=Path,required=True)
    ap.add_argument("--inner-surface-npz",type=Path,required=True);ap.add_argument("--current-state",type=Path,required=True)
    ap.add_argument("--canonical-834-report",type=Path,required=True);ap.add_argument("--canonical-849-report",type=Path,required=True);ap.add_argument("--volume-prefix-818b-report",type=Path,required=True);ap.add_argument("--shell-assembly-receipt",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    result=execute(a.skin_volume_npz,a.fat_volume_npz,a.skin_shell_npz,a.fat_shell_npz,
      a.inner_surface_npz,a.current_state,a.canonical_834_report,a.shell_assembly_receipt,a.output,
      manifest_849_report_path=a.canonical_849_report,prefix_818b_report_path=a.volume_prefix_818b_report)
    print(json.dumps({"status":result["status"],"output":str(a.output),"nodes":result["combined_skin_fat_child_mesh"]["nodes"],
      "cells":result["combined_skin_fat_child_mesh"]["combined_tetrahedra"],"physical_admission":False,"native_steps":0},sort_keys=True))
    return 0
if __name__=="__main__": raise SystemExit(main())
