#!/usr/bin/env python3
"""Compose the current-parent I flip with retained-domain Skin/Fat proofs and independent replay."""
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
from itertools import combinations
import importlib.util, hashlib, json, sys, time
import numpy as np
_json_dumps_original=json.dumps
def _to_jsonable(v):
 if isinstance(v,np.ndarray): return v.tolist()
 if isinstance(v,np.generic): return v.item()
 if isinstance(v,dict): return {(_to_jsonable(k) if isinstance(k,np.generic) else k):_to_jsonable(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)): return [_to_jsonable(x) for x in v]
 return v
def _json_dumps_numpy_safe(obj,*args,**kwargs): return _json_dumps_original(_to_jsonable(obj),*args,**kwargs)
json.dumps=_json_dumps_numpy_safe

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
BASE=H/"quality-partial-cavity-684-root"
OWNER=BASE/"paired_connected_boundary_patch_batch_976_oriented_boundary_parity_root.py"
PARENT=H/"all-quality-984-retained-I-quad-qualified-root/verified-state.json"
BASE_PARENT=H/"all-quality-972-last-fat-fixed-775-qualified-root/verified-state.json"
PREVIEW=H/"all-quality-975-I-flip-65823-66707-root/skin-seed-5point-preview-977/skin-seed-preview-readback-977b.json"
PLAN=H/"all-quality-975-I-flip-65823-66707-root/retained-domain-979/plan-981b.json"
COW_PATH=H/"npz-cow-append-926b-root/npz_cow_append_complete_926b.py"
OUT=H/"all-quality-992-retained-I-quad-rebased984-root"
if OUT.exists(): raise FileExistsError(OUT)
OUT.mkdir(parents=True)

def sha(p): return hashlib.file_digest(Path(p).open("rb"),"sha256").hexdigest()
def pin(p): return {"path":str(Path(p).resolve()),"sha256":sha(p)}
def write_archive_exact(path,payload,label):
 np.savez_compressed(path,**payload)
 actual=B.load_npz(path)
 if set(payload)!=set(actual): raise ValueError(label+" compressed archive key set mismatch")
 for k,v in payload.items():
  want=np.asarray(v);got=np.asarray(actual[k])
  if want.dtype!=got.dtype or want.shape!=got.shape or not np.array_equal(np.ascontiguousarray(want).view(np.uint8),np.ascontiguousarray(got).view(np.uint8)):
   raise ValueError(label+" compressed archive byte-exact readback mismatch for "+k)
 return actual
def assert_payload_matches_archive(payload,actual,label):
 if set(payload)!=set(actual): raise ValueError(label+" reused archive key set mismatch")
 for k,v in payload.items():
  want=np.asarray(v);got=np.asarray(actual[k])
  if want.dtype!=got.dtype or want.shape!=got.shape or not np.array_equal(np.ascontiguousarray(want).view(np.uint8),np.ascontiguousarray(got).view(np.uint8)):
   raise ValueError(label+" reused archive byte-exact replay mismatch for "+k)
 return actual
REUSED_990=H/"all-quality-990-retained-I-quad-rebased984-root"
REUSED_990_EXPECTED={"candidate-I-typed-current.npz":"f0ff2260b397c45f60ad5c189b88bae182be1f9ff3ba84011658ade13c16c204",
 "skin-retained-seed-volume-proposal.npz":"5c9c2fb6f879dc1baa4769e0ae9baf377d006933a9dc94732f9c2f0f91ebb910",
 "fat-retained-seed-volume-proposal.npz":"63652916dbf2a76533b655f2480a0983f9394d483a61cf026b2fdbbc54a47ded"}
spec=importlib.util.spec_from_file_location("owner976_984",OWNER)
B=importlib.util.module_from_spec(spec);sys.modules[spec.name]=B;spec.loader.exec_module(B)
ENG=B.ENG;T=B.T;M=B.M;A=B.A
root,st=B._source_state(PARENT)
base_root,base_st=B._source_state(BASE_PARENT)
preview=json.loads(PREVIEW.read_text());plan=json.loads(PLAN.read_text())
if sha(OWNER)!=plan.get("owner",{}).get("sha256"): raise ValueError("976 parity-screen owner pin mismatch")
if sha(BASE_PARENT)!=plan.get("parent",{}).get("sha256"): raise ValueError("retained-domain plan base-parent pin mismatch")
if sha(PREVIEW)!=plan.get("preview",{}).get("sha256"): raise ValueError("retained-domain plan preview pin mismatch")
if plan.get("status")!="PASS_FULL_DOMAINS_FOUND_DELTA_RECOMPUTED": raise ValueError("corrected retained-domain plan is not PASS")
if preview.get("parent",{}).get("sha256")!=sha(BASE_PARENT) or not preview.get("solver_candidate_present"): raise ValueError("pinned 972 Skin preview is not exact/present")
if preview.get("source_role")!="I" or preview.get("volume_role")!="Skin": raise ValueError("current-parent preview roles are not the expected I/Skin seed")
if not preview.get("new_exact_positive") or preview.get("new_native_bad")!=0 or not preview.get("new_exact_sum_equals_boundary_flux"): raise ValueError("current-parent Skin preview lacks exact positive/native/flux proof")
tic=time.monotonic()
sources={r:B.load_npz(B.check_ref(st["source_surfaces"][r],"current typed "+r)) for r in ("O","I","D")}
surface=sources["I"];srcfaces=np.asarray(surface["faces"],np.int64);srcpoints=np.asarray(surface["points_world_m_f32"],np.float32)
old_source_faces=np.asarray(preview["source_old_faces"],np.int64);new_source_faces=np.asarray(preview["source_new_faces"],np.int64)
old_source_rows=[]
sf_sorted=np.sort(srcfaces,axis=1)
for f in old_source_faces:
 hits=np.flatnonzero(np.all(sf_sorted==np.sort(f),axis=1))
 if len(hits)!=1: raise ValueError("old source face does not uniquely remap to current I")
 old_source_rows.append(int(hits[0]))
old_source_rows=np.asarray(old_source_rows,np.int64)
base_surface=B.load_npz(B.check_ref(base_st["source_surfaces"]["I"],"pinned 972 typed I"))
for coord_key in ("points_world_m_f32","points_world_m"):
 if coord_key in surface or coord_key in base_surface:
  if coord_key not in surface or coord_key not in base_surface: raise ValueError("source coordinate view missing across 972/984")
  aa=np.asarray(surface[coord_key]);bb=np.asarray(base_surface[coord_key])
  if aa.dtype!=bb.dtype or aa.shape!=bb.shape or not np.array_equal(aa.view(np.uint8),bb.view(np.uint8)): raise ValueError("source "+coord_key+" bytes differ between 972 and 984")
base_srcfaces=np.asarray(base_surface["faces"],np.int64)
base_sorted=np.sort(base_srcfaces,axis=1)
base_source_rows=[]
for f in old_source_faces:
 hits=np.flatnonzero(np.all(base_sorted==np.sort(f),axis=1))
 if len(hits)!=1: raise ValueError("old source face does not uniquely map to pinned 972 I")
 base_source_rows.append(int(hits[0]))
base_source_rows=np.asarray(base_source_rows,np.int64)
if not np.array_equal(base_source_rows,np.asarray(plan["source_face_rows"],np.int64)): raise ValueError("retained-domain plan source row mismatch against its pinned 972 parent")
if not np.array_equal(base_srcfaces[base_source_rows],old_source_faces): raise ValueError("retained-domain plan source faces do not match pinned 972 orientation")
if not np.array_equal(srcfaces[old_source_rows],old_source_faces): raise ValueError("current 984 source face orientation/rows changed")
def face_keyset(fs): return {tuple(sorted(map(int,f))) for f in np.asarray(fs,np.int64)}
base_source_keys=face_keyset(base_srcfaces);current_source_keys=face_keyset(srcfaces)
prior_source_delta=base_source_keys.symmetric_difference(current_source_keys)
operation_source_keys=face_keyset(np.vstack((old_source_faces,new_source_faces)))
if prior_source_delta & operation_source_keys: raise ValueError("65823 source patch overlaps exact changed face keys from 984")
prior_source_vertices=set(int(x) for f in base_srcfaces if tuple(sorted(map(int,f))) in prior_source_delta for x in f)
operation_source_vertices=set(map(int,np.r_[old_source_faces.ravel(),new_source_faces.ravel()]))
if prior_source_vertices & operation_source_vertices: raise ValueError("65823 source patch shares vertices with the 984 source edit")
edge=np.asarray(plan["edge"],np.int64)
if not np.array_equal(old_source_faces,np.asarray(plan["source_old_faces"],np.int64)) or not np.array_equal(new_source_faces,np.asarray(plan["source_new_faces"],np.int64)): raise ValueError("current preview faces differ from retained-domain plan")
if not np.array_equal(srcfaces[old_source_rows],old_source_faces): raise ValueError("current source face orientation/rows changed")
source_face_keys={tuple(sorted(map(int,f))) for f in srcfaces}
for f in new_source_faces:
 if tuple(sorted(map(int,f))) in source_face_keys: raise ValueError("new source face already exists in current I")
edge_vertices=set(map(int,edge))
opposite_vertices=np.asarray([next(int(x) for x in face if int(x) not in edge_vertices) for face in old_source_faces],np.int64)
if any(sum(int(x) in edge_vertices for x in face)!=2 for face in old_source_faces): raise ValueError("source old face does not contain exactly the selected diagonal")
quadproof=B.quad_overlay_hausdorff_upper(srcpoints,old_source_faces,new_source_faces,edge,opposite_vertices)
if quadproof["oriented_boundary_chain_match"] is not True: raise ValueError("current source flip oriented chain is not preserved")
quadcap=float(B.MAX_INNER_BACK_HAUSDORFF_M)
if float(quadproof["hausdorff_upper_bound_m"])>quadcap: raise ValueError("source I quad overlay exceeds current inner-back topology cap")

volumes={vr:B.B.volume_cache(st[vr+"_current"]) for vr in ("Skin","Fat")}
base_volumes={vr:B.B.volume_cache(base_st[vr+"_current"]) for vr in ("Skin","Fat")}
source_maps={vr:B.active_map(surface,c,"I/"+vr) for vr,c in volumes.items()}
# Current face row to volume target row, with exact source/target orientation relation.
target_updates={};target_rows_by_role={};orient_by_role={}
for vr,c in volumes.items():
 cross=np.asarray(source_maps[vr],np.int64)
 if np.any(cross[np.asarray(new_source_faces,np.int64)]<0): raise ValueError(vr+" new I triangle has an unmapped active source vertex")
 oldvf=cross[old_source_faces];newvf=cross[new_source_faces]
 lookup=B.B.face_lookup(np.asarray(c["target"],np.int64));rows=[];rels=[]
 for f in oldvf:
  hits=lookup.get(tuple(sorted(map(int,f))),[])
  if len(hits)!=1: raise ValueError(vr+" old I face lacks one exact current target row")
  rows.append(int(hits[0]));rels.append(int(ENG.rel(np.asarray(c["target"],np.int64)[int(hits[0])],f)))
 if len(set(rels))!=1: raise ValueError(vr+" source/target winding relation differs")
 mapped_new=newvf.copy()
 if rels[0]<0: mapped_new=mapped_new[:,[0,2,1]]
 oldrowset=set(rows)
 for f in mapped_new:
  hits=lookup.get(tuple(sorted(map(int,f))),[])
  if any(int(x) not in oldrowset for x in hits): raise ValueError(vr+" new I target triangle collides with retained current boundary")
 target_updates[vr]=[(rows[j],mapped_new[j].copy()) for j in range(len(rows))]
 target_rows_by_role[vr]=rows;orient_by_role[vr]=rels
 if not np.array_equal(np.asarray(base_volumes[vr]["target"],np.int64)[np.asarray(plan["roles"][vr]["target_rows"],np.int64)],np.asarray(plan["roles"][vr]["old_target_faces"],np.int64)): raise ValueError(vr+" retained-domain plan old target faces do not match its pinned 972 rows")
 if not np.array_equal(np.asarray(c["target"],np.int64)[np.asarray(rows,np.int64)],np.asarray(plan["roles"][vr]["old_target_faces"],np.int64)): raise ValueError(vr+" exact old target faces do not remap to current 984 boundary")
 if not np.array_equal(mapped_new,np.asarray(plan["roles"][vr]["new_target_faces"],np.int64)): raise ValueError(vr+" new target point rows differ from retained-domain plan")

# Rebase selected and full-domain rows from plan parent972 onto actual parent984 by exact tet+material keys.
def cell_key(tet,material): return (int(material),)+tuple(sorted(map(int,tet)))
def keyed_rows(cache):
 ts=np.asarray(cache["tets"],np.int64);ms=np.asarray(cache["mat"])
 keys=[cell_key(t,m) for t,m in zip(ts,ms)]
 if len(set(keys))!=len(keys): raise ValueError("duplicate ordered-tet+material keys in parent cell domain")
 return keys,{k:i for i,k in enumerate(keys)}
rebase={};full_domains={};rebase_map={}
for vr,c in volumes.items():
 cb=base_volumes[vr]
 if not np.array_equal(np.asarray(c["points"],np.float32).view(np.uint32),np.asarray(cb["points"],np.float32).view(np.uint32)):
  raise ValueError(vr+" point coordinates differ between 972 key domain and 984 current parent")
 for coord_key in ("points_world_m_f32","points_world_m"):
  if coord_key in c["arrays"] or coord_key in cb["arrays"]:
   if coord_key not in c["arrays"] or coord_key not in cb["arrays"]: raise ValueError(vr+" coordinate view missing across 972/984")
   aa=np.asarray(c["arrays"][coord_key]);bb=np.asarray(cb["arrays"][coord_key])
   if aa.dtype!=bb.dtype or aa.shape!=bb.shape or not np.array_equal(aa.view(np.uint8),bb.view(np.uint8)): raise ValueError(vr+" "+coord_key+" bytes differ between 972 and 984")
 base_keys,base_index=keyed_rows(cb);cur_keys,cur_index=keyed_rows(c)
 plan_selected=np.asarray(plan["roles"][vr]["selected_seed_cells"],np.int64)
 plan_full=np.asarray(plan["roles"][vr]["full_domain_cell_rows"],np.int64)
 if np.any(plan_selected<0) or np.any(plan_selected>=len(base_keys)) or np.any(plan_full<0) or np.any(plan_full>=len(base_keys)): raise ValueError(vr+" plan rows outside pinned base972 cell domain")
 selected_keys=[base_keys[int(i)] for i in plan_selected];full_keys=[base_keys[int(i)] for i in plan_full]
 if len(set(selected_keys))!=len(selected_keys) or len(set(full_keys))!=len(full_keys): raise ValueError(vr+" planned selected/full key domain duplicates")
 missing=[k for k in full_keys if k not in cur_index]
 if missing: raise ValueError(vr+" a retained full-domain cell key is absent from current984: "+repr(missing[:3]))
 selected=np.sort(np.asarray([cur_index[k] for k in selected_keys],np.int64))
 full=np.asarray([cur_index[k] for k in full_keys],np.int64)
 changed_keys=set(base_index).symmetric_difference(set(cur_index))
 overlap=changed_keys.intersection(full_keys)
 if overlap: raise ValueError(vr+" retained repair full domain overlaps 984 changed cell keys: "+repr(list(overlap)[:3]))
 selected_map=[{"base_row":int(br),"current_row":int(cur_index[k]),"cell_key":[int(k[0]),*map(int,k[1:])]} for br,k in zip(plan_selected,selected_keys)]
 full_map=[{"base_row":int(br),"current_row":int(cur_index[k]),"cell_key":[int(k[0]),*map(int,k[1:])]} for br,k in zip(plan_full,full_keys)]
 cross=np.asarray(source_maps[vr],np.int64);ev=cross[edge]
 if np.any(ev<0): raise ValueError(vr+" current source edge is not mapped")
 if not np.array_equal(np.asarray(B.B.full_edge_star(c,int(ev[0]),int(ev[1])),np.int64),selected): raise ValueError(vr+" exact current 984 edge star differs from rebased selected seed keys")
 if len(np.unique(np.asarray(c["mat"])[selected]))!=1: raise ValueError(vr+" selected seed mixes materials")
 rebase[vr]={"selected_rows":selected,"full_rows":full,"base_selected_rows":plan_selected,"base_full_rows":plan_full,"selected_keys":selected_keys,"full_keys":full_keys}
 full_domains[vr]=full
 rebase_map[vr]={"selected_seed_cell_rows":selected_map,"full_domain_cell_rows":full_map,"base_cell_count":len(base_keys),"current_cell_count":len(cur_keys),"base_to_current_domain_status":"PASS_EXACT_TET_MATERIAL_KEYS","changed_cell_key_count_from_972_to_984":len(changed_keys),"changed_cell_keys_intersect_full_domain":0}
if not np.array_equal(np.asarray(preview["seed_cell_rows"],np.int64),np.asarray(plan["roles"]["Skin"]["selected_seed_cells"],np.int64)): raise ValueError("pinned 972 preview seed rows differ from 981b plan seed rows")
# Revalidate the local interface operation on current972, not by trusting 966 row IDs.
def exact_det_sum(points,tets):
 t=np.asarray(tets,np.int64)
 if not len(t): return 0
 ip={int(i):ENG.S.S.ip(np.asarray(points[int(i)],np.float32)) for i in np.unique(t.ravel())}
 return sum(ENG.S.S.det(*(ip[int(i)] for i in tet)) for tet in t)
local_records={};patch_volumes={}
for vr,c in volumes.items():
 fullrows=full_domains[vr];domain_tets=np.asarray(c["tets"][fullrows],np.int64)
 selected=rebase[vr]["selected_rows"];oldsub=np.asarray(c["tets"][selected],np.int64)
 newglobal=np.asarray(preview["new_tets"],np.int64).reshape((-1,4)) if vr=="Skin" else np.empty((0,4),np.int64)
 if not np.array_equal(newglobal,np.asarray(plan["roles"][vr]["replacement_tets"],np.int64).reshape((-1,4))): raise ValueError(vr+" current-parent replacement tets differ from retained-domain plan")
 oldfaces=np.asarray(c["target"],np.int64)[np.asarray(target_rows_by_role[vr],np.int64)]
 newfaces=np.asarray([ff for _,ff in target_updates[vr]],np.int64)
 # Faces use the common point namespace; exact coordinate arrays were byte/bit-checked in preflight.
 bdomain=np.asarray(T.m.boundary(domain_tets),np.int64)
 bsub=np.asarray(T.m.boundary(oldsub),np.int64)
 chain_sub=ENG.chain(bsub)
 for f in oldfaces:
  if chain_sub.get(tuple(sorted(map(int,f))),0)!=ENG.fsign(f): raise ValueError(vr+" old interface face not on selected seed boundary")
 replacements=[(oldfaces[j],newfaces[j]) for j in range(len(oldfaces))]
 newbsub=np.asarray(ENG.oriented_boundary_replace(bsub,replacements),np.int64)
 empty_retirement=(vr=="Fat" and len(oldsub)==1 and len(newglobal)==0 and len(newbsub)==0)
 if vr=="Fat" and not empty_retirement: raise ValueError("Fat empty seed retirement proof does not match exact local shell")
 if vr=="Skin":
  if not T.closed(newbsub) or T.manifold_topology(newbsub).get("bad_vertex_links") or T.manifold_topology(newbsub).get("boundary_components")!=1:
   raise ValueError("Skin replacement seed shell is not one closed manifold")
  verts=np.unique(oldsub.ravel())
  if not set(map(int,newglobal.ravel())).issubset(set(map(int,verts))): raise ValueError("Skin MILP output uses point outside retained seed")
  lmap=np.full(len(c["points"]),-1,np.int64);lmap[verts]=np.arange(len(verts))
  newlocal=np.asarray([lmap[x] for x in newglobal],np.int64)
  if ENG.chain(np.asarray(T.m.boundary(newlocal),np.int64))!=ENG.chain(lmap[newbsub]):
   raise ValueError("Skin local MILP tets fail exact new target chain")
  newsubsum=exact_det_sum(c["points"],newglobal)
 else:
  newsubsum=0
 oldsubsum=exact_det_sum(c["points"],oldsub)
 delta=int(newsubsum)-int(oldsubsum)
 olddomain=exact_det_sum(c["points"],domain_tets)
 if olddomain!=int(plan["roles"][vr]["full_domain_growth"][-1]["old_exact_det6_sum"]): raise ValueError(vr+" current expanded full-domain sum differs from pinned planner proof")
 if delta!=int(plan["roles"][vr]["exact_local_delta_det6"]): raise ValueError(vr+" current seed replacement exact delta differs from pinned planner proof")
 if int(plan["roles"][vr]["full_domain_growth"][-1]["new_boundary_flux"])-olddomain!=delta: raise ValueError(vr+" planner full-domain exact delta differs from recomputed sums")
 if abs(delta)/float(olddomain)>.01: raise ValueError(vr+" declared full-cavity 1% volume cap failed")
 fullnew=np.asarray(ENG.oriented_boundary_replace(bdomain,replacements),np.int64)
 if not T.closed(fullnew) or T.manifold_topology(fullnew).get("bad_vertex_links") or T.manifold_topology(fullnew).get("boundary_components")!=1:
  raise ValueError(vr+" declared full-cavity new target boundary is not a single closed manifold")
 fullflux=ENG.exact_boundary_flux(c["points"],fullnew)
 if fullflux-olddomain!=delta: raise ValueError(vr+" full cavity exact target flux delta mismatch")
 keep_local=np.asarray([int(x) not in set(map(int,selected)) for x in fullrows],bool)
 retained=domain_tets[keep_local]
 combined=np.vstack((retained,newglobal)).astype(np.int64)
 if ENG.chain(np.asarray(T.m.boundary(combined),np.int64))!=ENG.chain(fullnew):
  raise ValueError(vr+" retained full-domain cavity does not realize the complete changed target chain")
 combined_sum=exact_det_sum(c["points"],combined)
 if combined_sum!=fullflux: raise ValueError(vr+" retained full-domain exact tet sum differs from target flux")
 local_det,local_q=M.measurements(c["points"],combined)
 if np.any(local_det<=0) or not np.all(M.passes(local_det,local_q)): raise ValueError(vr+" retained declared cavity has a native failure")
 local_records[vr]={"selected_rows":selected,"full_rows":fullrows,"old_tets":oldsub,"new_tets":newglobal,
   "old_delta":delta,"full_old_sum":olddomain,"full_new_flux":fullflux,"full_new_sum":combined_sum,
   "empty_retirement":empty_retirement,"target_replacements":target_updates[vr],
   "old_bad":int(np.count_nonzero(~np.asarray(M.passes(c["det"][fullrows],c["q"][fullrows]),bool))),
   "new_bad":0,"local_cavity_all_native_pass":True}
 patch_volumes[vr]={"cell_rows":selected,"replacement_tets":newglobal,"exact_det6_delta":delta,
  "old_bad":int(np.count_nonzero(~np.asarray(M.passes(c["det"][selected],c["q"][selected]),bool))),
  "new_bad":int(np.count_nonzero(~np.asarray(M.passes(*M.measurements(c["points"],newglobal)),bool))) if len(newglobal) else 0,
  "material":int(np.asarray(c["mat"])[selected[0]]),
  "target_replacements":target_updates[vr],
  "candidate_operations":[{"orientation_relation":orient_by_role[vr][0],"inner_back_touched":False}],
  "inner_back_hausdorff_max_m":0.0,
  "local_volume_fraction":abs(delta)/float(olddomain),
  "global_volume_fraction":abs(delta)/float(ENG.exact_boundary_flux(c["points"],c["target"]))}

if sum(int(local_records[r]["old_delta"]) for r in ("Skin","Fat"))!=0:
 raise ValueError("paired I source flip exact signed deltas do not cancel")
args=SimpleNamespace(role="I",maximum_local_volume_fraction=.01,maximum_global_volume_fraction=1e-4)
quad=set(map(int,quadproof["cycle_rows"]))
candidate={"face_rows":old_source_rows.tolist(),"new_faces":new_source_faces.copy(),"edge":list(edge),"quad":quad,
 "source_patch_kind":"edge_2to2","patch_boundary_cycle":list(quadproof["cycle_rows"]),
 "quad_hausdorff":quadproof,"surface_hausdorff_limit_m":quadcap,
 "material_face_ring_added_cell_rows":{},"material_face_ring_diagnostics":{},
 "other_role_protected_volume_rows_touched":{"Skin":[],"Fat":[]},
 "stars":{"Skin":rebase["Skin"]["selected_rows"],"Fat":rebase["Fat"]["selected_rows"]}}
accepted=[{"group":[candidate],"volumes":patch_volumes,"selection":"retained seed subcavity with exact paired interface transfer"}]
surface_arrays,face_ops=B.compose_surface_batch(surface,accepted,st["source_surfaces"]["I"]["path"])
if ENG.topology_sig(np.asarray(surface_arrays["faces"],np.int64))!=ENG.topology_sig(srcfaces):
 raise ValueError("source topology signature changed")
if sha(COW_PATH)!="98dd06cafb9a38af4a8c1c7a803ed967d6effff73fd7d1528df7bedcfb40418f": raise ValueError("diagnostic COW emitter pin mismatch")
src_path=REUSED_990/"candidate-I-typed-current.npz"
if sha(src_path)!=REUSED_990_EXPECTED[src_path.name]: raise ValueError("retained 990 source archive hash mismatch")
source_read=B.load_npz(src_path);assert_payload_matches_archive(surface_arrays,source_read,"typed I candidate")
source_pin=pin(src_path)
role_refs={r:dict(st["source_surfaces"][r]) for r in ("O","I","D")};role_refs["I"]=source_pin
output_volumes={};volume_metrics={}
for vr in ("Skin","Fat"):
 va,vm=B.compose_volume_batch(vr,volumes[vr],accepted,source_ref=source_pin,all_source_refs=role_refs,args=args)
 # Explicit empty-target retirement identity, linked to source patch and exact signed volume transfer.
 if local_records[vr]["empty_retirement"]:
  prev_rows=np.asarray(va.get("connected_boundary_patch_empty_target_retirement_cell_rows",np.empty(0,np.int64)),np.int64)
  prev_faces=np.asarray(va.get("connected_boundary_patch_empty_target_retirement_source_face_rows",np.empty(0,np.int64)),np.int64)
  history=[]
  prior_json=np.asarray(va.get("connected_boundary_patch_empty_target_retirement_history_json",np.asarray([],dtype="U"))).reshape(-1)
  if len(prior_json): history.extend(map(str,prior_json))
  if len(prev_rows) or len(prev_faces):
   history.append(json.dumps({"immediate_parent_volume":pin(st[vr+"_current"]["path"]),"historical_input_cell_rows":prev_rows.tolist(),"historical_source_face_rows_in_parent_source_namespace":prev_faces.tolist()},sort_keys=True))
  history.append(json.dumps({"immediate_parent_volume":pin(st[vr+"_current"]["path"]),"current_input_cell_rows":np.asarray(rebase[vr]["selected_rows"],np.int64).tolist(),"current_source_face_rows_in_parent_source_namespace":old_source_rows.tolist()},sort_keys=True))
  va["connected_boundary_patch_empty_target_retirement_history_json"]=np.asarray(history,dtype="U")
  va["connected_boundary_patch_empty_target_retirement_cell_rows"]=np.asarray(rebase[vr]["selected_rows"],np.int64)
  va["connected_boundary_patch_empty_target_retirement_source_face_rows"]=old_source_rows.copy()
 output_volumes[vr]=va;volume_metrics[vr]=vm
volume_refs={}
for vr in ("Skin","Fat"):
 vp=REUSED_990/(vr.lower()+"-retained-seed-volume-proposal.npz")
 if sha(vp)!=REUSED_990_EXPECTED[vp.name]: raise ValueError(vr+" retained 990 archive hash mismatch")
 reused_volume=B.load_npz(vp);assert_payload_matches_archive(output_volumes[vr],reused_volume,vr+" volume candidate")
 output_volumes[vr]=reused_volume;volume_refs[vr]=pin(vp)
rebase_receipt_path=OUT/"exact-key-rebase-984.json"
rebase_receipt={"schema":"h706.retained-seed-exact-key-rebase.v1","status":"PASS_EXACT_CELL_AND_FACE_KEY_REBASE",
 "base_state":pin(BASE_PARENT),"current_parent_state":pin(PARENT),"base_source_candidate":pin(base_st["source_surfaces"]["I"]["path"]),
 "current_source_candidate":pin(st["source_surfaces"]["I"]["path"]),"retained_domain_plan":pin(PLAN),"operation_edge":list(map(int,edge)),
 "base_source_face_rows":base_source_rows.tolist(),"current_source_face_rows":old_source_rows.tolist(),
 "base_target_face_rows":{r:np.asarray(plan["roles"][r]["target_rows"],np.int64).tolist() for r in ("Skin","Fat")},
 "current_target_face_rows":{r:np.asarray(target_rows_by_role[r],np.int64).tolist() for r in ("Skin","Fat")},
 "prior_972_to_984_source_face_key_delta_count":len(prior_source_delta),"source_face_key_delta_intersection_with_operation":0,
 "prior_source_vertices_intersection_with_operation":0,"point_coordinate_views_bitwise_equal_972_to_984":True,
 "exact_cell_key_maps":rebase_map,"all_selected_and_full_domain_cell_keys_survive_984":True,
 "984_changed_volume_cell_keys_intersect_full_domain":0,
 "selection_by":"sorted point-row tetrahedron plus material integer; source and target faces by sorted point-row keys with exact orientation",
 "prior_empty_target_retirement_event":"preserved in immediate-parent984 NPZ and parent state chain; current retirement metadata in output names only the 989 immediate operation",
 "physical_admission":False,"mass_mapping":"NOT_RUN"}
rebase_receipt_path.write_text(json.dumps(rebase_receipt,indent=2,sort_keys=True)+"\n")
for vr in ("Skin","Fat"):
 local_records[vr]["new_bad"]=int(volume_metrics[vr]["native_bad_after"])
global_bad_before={vr:int(np.count_nonzero(volumes[vr]["bad"])) for vr in ("Skin","Fat")}
global_bad_after={vr:int(volume_metrics[vr]["native_bad_after"]) for vr in ("Skin","Fat")}
if any(global_bad_after[vr]>global_bad_before[vr] for vr in ("Skin","Fat")):
 raise ValueError("whole-role native bad count regressed")
if sum(global_bad_after.values())>=sum(global_bad_before.values()):
 raise ValueError("whole-role paired global native bad count did not strictly decrease")
# Independent readback replay starts from NPZs and parent arrays, not the compose inputs.
def independent_replay():
 psurf=B.load_npz(st["source_surfaces"]["I"]["path"])
 osurf=B.load_npz(src_path)
 oldsf=np.asarray(psurf["faces"],np.int64);newsf=oldsf.copy()
 newsf[old_source_rows]=new_source_faces
 if not np.array_equal(np.asarray(osurf["faces"],np.int64),newsf):raise ValueError("readback source geometry differs from reconstructed face edit")
 if not np.array_equal(np.asarray(psurf["points_world_m_f32"],np.float32).view(np.uint32),np.asarray(osurf["points_world_m_f32"],np.float32).view(np.uint32)):raise ValueError("readback I point bytes changed")
 if ENG.topology_sig(oldsf)!=ENG.topology_sig(newsf):raise ValueError("readback source topology signature differs")
 union=set();anc=set()
 for r in old_source_rows:
  union.update(tuple(map(int,x)) for x in A.rows_union(psurf,int(r)))
  anc.update(tuple(map(int,x)) for x in A.rows_anc(psurf,int(r)))
 if not union or not anc:raise ValueError("source patch parent ancestry is empty")
 outq=np.asarray(osurf["quality_star_face_output_to_input_row"],np.int64)
 uoff=np.asarray(osurf["quality_star_patch_union_offsets"],np.int64);aoff=np.asarray(osurf["face_ancestry_offsets"],np.int64)
 for r in range(len(oldsf)):
  if r in set(map(int,old_source_rows)):
   if outq[r]!=-1:raise ValueError("changed source face kept exact-parent row mapping")
   lo,hi=map(int,uoff[r:r+2]);got={tuple(int(osurf[k][j]) for k in ENG.TUN) for j in range(lo,hi)}
   if got!=union:raise ValueError("changed source face union mismatch")
   lo,hi=map(int,aoff[r:r+2]);got={tuple(int(osurf[k][j]) for k in ENG.ANC) for j in range(lo,hi)}
   if got!=anc:raise ValueError("changed source face ancestry mismatch")
  else:
   if not np.array_equal(oldsf[r],newsf[r]):raise ValueError("unrelated source face changed")
   if A.rows_union(psurf,r)!=A.rows_union(osurf,r) or A.rows_anc(psurf,r)!=A.rows_anc(osurf,r):raise ValueError("unrelated source face ancestry changed")
 results={}
 for vr in ("Skin","Fat"):
  pvol=B.load_npz(st[vr+"_current"]["path"]);ovol=output_volumes[vr]
  pt=np.asarray(pvol["tetrahedra"],np.int64);pm=np.asarray(pvol["material_indices"])
  sel=np.asarray(rebase[vr]["selected_rows"],np.int64);keep=np.ones(len(pt),bool);keep[sel]=False
  nt=np.asarray(preview["new_tets"],np.int64).reshape((-1,4)) if vr=="Skin" else np.empty((0,4),np.int64);nm=np.full(len(nt),int(pm[sel[0]]),dtype=pm.dtype)
  expected=np.vstack((pt[keep],nt)).astype(np.int64);expectedm=np.r_[pm[keep],nm]
  if not np.array_equal(np.asarray(ovol["tetrahedra"],np.int64),expected) or not np.array_equal(np.asarray(ovol["material_indices"]),expectedm):
   raise ValueError(vr+" independent whole-volume connectivity/material replay mismatch")
  if not np.array_equal(np.asarray(ovol["output_cell_to_input_cell"],np.int64),np.r_[np.flatnonzero(keep),np.full(len(nt),-1,np.int64)]):raise ValueError(vr+" direct output-cell map mismatch")
  inv=np.full(len(pt),-1,np.int64);inv[keep]=np.arange(int(keep.sum()),dtype=np.int64)
  if not np.array_equal(np.asarray(ovol["input_cell_to_output_cell"],np.int64),inv):raise ValueError(vr+" direct input-cell map mismatch")
  rpo=np.asarray(ovol["replacement_parent_offsets"],np.int64);rpr=np.asarray(ovol["replacement_parent_cell_rows"],np.int64)
  if not np.array_equal(rpo,np.r_[0,np.cumsum(np.full(len(nt),len(sel),np.int64))]) or not np.array_equal(rpr,np.tile(sel,len(nt))):raise ValueError(vr+" replacement parent CSR mismatch")
  if vr=="Fat":
   if len(nt)!=0 or not np.array_equal(np.asarray(ovol["connected_boundary_patch_empty_target_retirement_cell_rows"],np.int64),sel):raise ValueError("Fat explicit current empty-target retirement input row mismatch")
   if not np.array_equal(np.asarray(ovol["connected_boundary_patch_empty_target_retirement_source_face_rows"],np.int64),old_source_rows):raise ValueError("Fat current empty-retirement source-face ancestry mismatch")
   prior_rows=np.asarray(pvol.get("connected_boundary_patch_empty_target_retirement_cell_rows",np.empty(0,np.int64)),np.int64)
   history=np.asarray(ovol.get("connected_boundary_patch_empty_target_retirement_history_json",np.asarray([],dtype="U"))).reshape(-1)
   if len(prior_rows) and not any(str(st["Fat_current"]["sha256"]) in str(x) and prior_rows.tolist()==json.loads(str(x)).get("historical_input_cell_rows") for x in history): raise ValueError("Fat previous empty-target retirement event was not retained")
   if not len(history): raise ValueError("Fat current empty-target retirement history is absent")
  target0=np.asarray(pvol["desired_boundary_faces"],np.int64);want=target0.copy()
  for rr,ff in target_updates[vr]:want[int(rr)]=np.asarray(ff,np.int64)
  target1=np.asarray(ovol["desired_boundary_faces"],np.int64)
  if not np.array_equal(target1,want):raise ValueError(vr+" desired target face edit mismatch")
  actual=np.asarray(T.m.boundary(expected),np.int64)
  if ENG.chain(actual)!=ENG.chain(target1):raise ValueError(vr+" complete full-volume target chain mismatch")
  if len(np.unique(np.sort(expected,axis=1),axis=0))!=len(expected):raise ValueError(vr+" duplicate volume tets")
  dd,qq=M.measurements(np.asarray(pvol["points_world_m_f32"],np.float32),expected)
  if np.any(dd<=0):raise ValueError(vr+" independent whole-volume positivity replay failed")
  oldpass=np.asarray(M.passes(volumes[vr]["det"],volumes[vr]["q"]),bool)
  newpass=np.asarray(M.passes(dd,qq),bool)
  oldbad=int(np.count_nonzero(~oldpass));newbad=int(np.count_nonzero(~newpass))
  if newbad>oldbad:raise ValueError(vr+" whole-role bad count regressed")
  if np.any(oldpass[keep] & ~newpass[:int(keep.sum())]):raise ValueError(vr+" formerly-good retained cell regressed")
  if len(nt) and not np.all(newpass[-len(nt):]):raise ValueError(vr+" replacement cell fails native threshold")
  oldflux=ENG.exact_boundary_flux(volumes[vr]["points"],volumes[vr]["target"])
  newflux=ENG.exact_boundary_flux(volumes[vr]["points"],target1)
  delta=int(newflux-oldflux)
  if delta!=int(local_records[vr]["old_delta"]):raise ValueError(vr+" exact whole-target flux delta mismatch")
  # Independently rebuild the declared conforming subdomain with retained old cells.
  fullrows=np.asarray(rebase[vr]["full_rows"],np.int64);fullt=pt[fullrows];selected=set(map(int,sel))
  keepfull=np.asarray([int(x) not in selected for x in fullrows],bool)
  localcombined=np.vstack((fullt[keepfull],nt)).astype(np.int64)
  localboundary=np.asarray(T.m.boundary(localcombined),np.int64)
  targetfullold=np.asarray(T.m.boundary(fullt),np.int64)
  oldtarget_current=np.asarray(volumes[vr]["target"],np.int64)[np.asarray(target_rows_by_role[vr],np.int64)]
  targetfullnew=np.asarray(ENG.oriented_boundary_replace(targetfullold,[(oldtarget_current[j],target_updates[vr][j][1]) for j in range(2)]),np.int64)
  # The old target faces are already in the shared volume point namespace; verify local target chain separately.
  if ENG.chain(localboundary)!=ENG.chain(targetfullnew):raise ValueError(vr+" retained full-cavity local target chain mismatch")
  olddomain=exact_det_sum(volumes[vr]["points"],fullt);newdomain=exact_det_sum(volumes[vr]["points"],localcombined)
  flux=ENG.exact_boundary_flux(volumes[vr]["points"],targetfullnew)
  if newdomain!=flux or newdomain-olddomain!=delta:raise ValueError(vr+" retained full-cavity exact flux/volume replay mismatch")
  if abs(delta)/float(olddomain)>.01:raise ValueError(vr+" retained full-cavity 1% volume budget replay mismatch")
  results[vr]={"whole_native_bad_before":oldbad,"whole_native_bad_after":newbad,"all_replacement_cells_native_pass":True,
   "no_formerly_good_cell_regressions":True,"full_target_chain":True,"whole_exact_target_flux_delta":str(delta),"retained_domain_cells":len(fullrows)-len(sel),
   "retained_domain_exact_flux":str(flux),"retained_domain_exact_volume_delta":str(delta),"regional_volume_fraction":abs(delta)/float(olddomain),
   "replacement_parent_CSR":True,"empty_target_retirement":vr=="Fat"}
 # Source-to-volume identity for the changed I facets and full target boundary has already been checked.
 if sum(int(results[r]["whole_exact_target_flux_delta"]) for r in ("Skin","Fat"))!=0:raise ValueError("independent paired exact volume cancellation failed")
 if sum(results[r]["whole_native_bad_after"] for r in ("Skin","Fat"))>=sum(results[r]["whole_native_bad_before"] for r in ("Skin","Fat")):
  raise ValueError("independent paired whole-role native count did not strictly improve")
 return {"status":"PASS_INDEPENDENT_RETAINED_SEED_REPLAY","roles":results,"source_face_geometry":True,"source_face_ancestry":True,
   "paired_exact_volume_cancellation":True,"full_global_target_chains":True,"whole_native_quality":True,"physical_admission":False,"native_steps":0}
replay=independent_replay()
# Pin operation journal and candidate arrays. The Fat zero-child event is explicit, never represented as a silent delete.
journal=B.build_patch_journal(surface,volumes,accepted,("Skin","Fat"))
journal[0]["replacement_policy"]="replace only the closed source-flip seed subcavity; every other cell in the declared full conforming domain is retained unchanged"
journal[0]["declared_full_domain_cell_rows_by_role"]={vr:rebase[vr]["full_rows"].tolist() for vr in ("Skin","Fat")}
journal[0]["domain_growth_plan"]={"path":str(PLAN.resolve()),"sha256":sha(PLAN)}
journal[0]["immediate_parent_state_984"]=pin(PARENT)
journal[0]["base_plan_row_domain_972"]=pin(BASE_PARENT)
journal[0]["exact_key_rebase_receipt"]=pin(rebase_receipt_path)
journal[0]["cell_count_arithmetic_by_role"]={vr:{"parent_cells":len(volumes[vr]["tets"]),"retired_seed_cells":len(rebase[vr]["selected_rows"]),"new_replacement_cells":len(patch_volumes[vr]["replacement_tets"]),"output_cells":len(output_volumes[vr]["tetrahedra"]),"retained_nonseed_cells":len(volumes[vr]["tets"])-len(rebase[vr]["selected_rows"])} for vr in ("Skin","Fat")}
journal[0]["retained_domain_cell_rows_by_role"]={vr:sorted(set(map(int,rebase[vr]["full_rows"]))-set(map(int,rebase[vr]["selected_rows"]))) for vr in ("Skin","Fat")}
journal[0]["empty_target_retirement"]={"role":"Fat","input_cell_rows":rebase["Fat"]["selected_rows"].tolist(),"output_tetrahedra":[],"reason":"the oriented two-face I-interface replacement cancels the one-tet seed shell; the 18 nonseed cells in the declared Fat domain and every cell outside it are retained unchanged; exact changed full target boundary is independently replayed","exact_det6_delta":str(local_records["Fat"]["old_delta"]),"mass_mapping":"NOT_RUN"}
journal_path=OUT/"retained-seed-patch-journal.json";journal_path.write_text(json.dumps(journal,indent=2,sort_keys=True)+"\n")
replay_path=OUT/"independent-retained-seed-replay.json";replay_path.write_text(json.dumps(replay,indent=2,sort_keys=True)+"\n")
# Candidate state remains source-only and pending full 775/808 review plus physical mass reconciliation.
if "current_verified_state_20261031" in root:
 stateout=dict(root);so=dict(st);stateout["current_verified_state_20261031"]=so
else:
 so=dict(st);stateout=so
so["source_surfaces"]=role_refs
so["current_retained_seed_archive_serialization"]={"method":"reuse validated 990 compressed NPZ arrays","byte_exact_replay_against_fresh_composition":True,"reused_archive_directory":str(REUSED_990.resolve()),"COW_attempt":"FAILED_PARENT_FACE_ANCESTRY_FIELD_COMPARE","COW_attempt_sha256":sha(COW_PATH)}
if "retained_seed_interface_transfer" in so:
 so["parent_retained_seed_interface_transfer_history"]={"immediate_parent_state":pin(PARENT),"parent_operation":so["retained_seed_interface_transfer"]}
for vr in ("Skin","Fat"):
 old=so[vr+"_current"];newref={**old,**volume_refs[vr],"cells":len(output_volumes[vr]["tetrahedra"]),
  "native_quality_failures":int(volume_metrics[vr]["native_bad_after"]),"nonpositive":0,
  "boundary_source_binding_status":"PENDING_FRESH_775_808_SOURCE_AUDIT"}
 so[vr+"_current"]=newref;so[vr]=dict(newref)
for k in ("exact_context","independent_replay","independent_point_replay","independent_allrole_local_replay","independent_target_replay",
          "source_evidence","source_audit","source_crossing_audit","native_qualification","mass_mapping_receipt","boundary_source_binding_status","all185_775","actual_697_217"):
 so.pop(k,None)
so.update({"source_state":pin(PARENT),"source_audit_status":"PENDING_GLOBAL_185_AUDIT","source_crossing_audit_status":"PENDING_GLOBAL_185_AUDIT",
 "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
 "status":"SOURCE_ONLY_RETAINED_SEED_INTERFACE_TRANSFER_PENDING_775_808_AND_MASS",
 "retained_seed_interface_transfer":{"source_role":"I","source_owner":pin(OWNER),"source_face_rows":old_source_rows.tolist(),"source_edge":list(edge),
   "parent_state":pin(PARENT),"base_key_domain_state":pin(BASE_PARENT),"exact_key_rebase_receipt":pin(rebase_receipt_path),"retained_domain_plan":pin(PLAN),"current_skin_preview":pin(PREVIEW),"current_parent_source_flip":True,"rebased_onto_984_independent_I_flip":True,
   "source_quad_hausdorff_upper_m":float(quadproof["hausdorff_upper_bound_m"]),"source_quad_hausdorff_cap_m":quadcap,
   "retained_domain_replay":pin(replay_path),"patch_journal":pin(journal_path),
   "independent_replay":"PASS","full_global_185_audit":"REQUIRED_NOT_RUN","mass_mapping":"NOT_RUN",
   "skin_seed_parent_cells":rebase["Skin"]["selected_rows"].tolist(),"fat_empty_retirement_parent_cells":rebase["Fat"]["selected_rows"].tolist(),
   "domain_cell_arithmetic":{vr:{"parent_cell_count":len(volumes[vr]["tets"]),"retired_seed_cells":len(rebase[vr]["selected_rows"]),"new_replacement_cells":len(patch_volumes[vr]["replacement_tets"]),"output_cell_count":len(output_volumes[vr]["tetrahedra"]),"all_nonseed_parent_cells_retained":len(output_volumes[vr]["tetrahedra"])==len(volumes[vr]["tets"])-len(rebase[vr]["selected_rows"])+len(patch_volumes[vr]["replacement_tets"])} for vr in ("Skin","Fat")},
   "all_point_coordinates_unchanged":True,"outer_O_source_unchanged":True,"only_seed_cells_replaced_or_retired":True,"other_cells_in_full_domain_retained":True,
   "exact_paired_signed_volume_delta":"0","native_thresholds_unchanged":True}})
statepath=OUT/"candidate-state-proposal.json";statepath.write_text(json.dumps(stateout,indent=2,sort_keys=True)+"\n")
report={"schema":"h706.retained-seed-interface-transfer-rebased-984.v1","status":"PASS_INDEPENDENT_REPLAY_PENDING_FRESH_775_808",
 "candidate_workflow":"Only 2 Skin seed cells are replaced and 1 Fat cap seed cell is retired; every other parent tetrahedron is retained exactly.",
 "parent_state":pin(PARENT),"base_key_domain_state":pin(BASE_PARENT),"exact_key_rebase_receipt":pin(rebase_receipt_path),"retained_domain_plan":pin(PLAN),"current_skin_preview":pin(PREVIEW),"current_parent_source_flip":True,"rebased_onto_984_independent_I_flip":True,
 "source_candidate":source_pin,"volume_candidates":volume_refs,"state_candidate":pin(statepath),
 "journal":pin(journal_path),"independent_replay":pin(replay_path),"independent_replay_status":replay["status"],
 "source_quad_hausdorff":quadproof,"orientation_parity_control":pin(H/"all-quality-975-I-flip-65823-66707-root/oriented-boundary-parity-976/control-002.json"),"source_owner_976":pin(OWNER),"plan_correction_receipt":pin(PLAN.parent/"plan-981b-receipt.json"),"diagnostic_cow_writer":pin(COW_PATH),"reused_990_archives":{k:{"path":str((REUSED_990/k).resolve()),"sha256":sha(REUSED_990/k)} for k in REUSED_990_EXPECTED},"role_metrics":volume_metrics,
 "paired_exact_signed_delta":"0","paired_quality_bad_before":sum(x["whole_native_bad_before"] for x in replay["roles"].values()),
 "paired_quality_bad_after":sum(x["whole_native_bad_after"] for x in replay["roles"].values()),
 "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN","global_185_source_audit":"PENDING","exact_key_rebase_receipt":pin(rebase_receipt_path)}
rp=OUT/"report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,indent=2,sort_keys=True));print("candidate",statepath);print("elapsed_s",time.monotonic()-tic)

