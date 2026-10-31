#!/usr/bin/env python3
"""Rebind the current Fat target boundary from authenticated typed I, D, and rim-wall sources.

The candidate's points/connectivity are copied byte-for-byte. This produces a metadata-only
surface target and an exact source-row crosswalk; it does not repair the remaining boundary
mismatch or claim physical admission.
"""
from pathlib import Path
from collections import Counter
import csv, hashlib, importlib.util, json, sys
import numpy as np

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
BASE=H/"fat614-yy-volume-replay-004/run-011-rebased-377-007-components/fat-D307-I377-plus-YY-components-source-only.npz"
BASE_REPORT=H/"fat614-yy-volume-replay-004/run-011-rebased-377-007-components/report.json"
I=H/"paired-I-interface-377-root/qualification378001/typed365/candidate-I-typed-current.npz"
IR=H/"paired-I-interface-377-root/qualification378001/typed365/receipt.json"
D=H/"d-shortedge-quotient-351-run002/typed-rebind-003/candidate-D-typed-current.npz"
DR=H/"d-shortedge-quotient-351-run002/typed-rebind-003/697-adapter-001/receipt.json"
W=H/"verified-volume-preparation-010-merged038005/rim/wall-provenance-current.csv"
COLLAR=H/"verified-volume-preparation-010-merged038005/rim/collar-current.npz"
WR=H/"verified-volume-preparation-010-merged038005/rim/receipt.json"
OUT=H/"fat614-yy-volume-replay-004/run-014-current-target-rebound-v3"
EXPECTED={
 "base_candidate":("be8ba2f5a1170e692fa92ebe458d72e6bee5427abb18e1d2808b12ec2908f241",BASE),
 "base_report":("b4c83eee4a421238dfd160a37d0a53d9a95fd4fa715e5b370a98aa45f890968b",BASE_REPORT),
 "typed_I":("8e04da9ad1e9c04216dae1fde854a3d69c8e5bb4c7a7c4a4796754675ce7b747",I),
 "typed_I_receipt":("e9915b726a01ac55b754dac4ad368ca4b2c64b0e0b503547de81d1fedfede336",IR),
 "typed_D":("18517e7dc0b1b25d67eea9691ad98bdbcfd07d18b9e1fe45f25b95515ee99ade",D),
 "typed_D_697_receipt":("4f37239ebf651c90eb3af3f7b98acdf4f6f4b25cda42462f69374f70fd424fe6",DR),
 "wall_csv":("7efadaa02773ff36392da40bbe7d58957d84ddffa96cdc071988a680b828d2f4",W),
 "collar":("ccbc772935ded2bffc841a4c3b024f9ef949f0c302c1ff8cd0e3c232357055d7",COLLAR),
 "wall_receipt":("f735c54355c07aaa85f00fe09b662688c17c3f2ba159b5238e1b84e118f62caa",WR),
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def loadnpz(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def readjson(path):
 s=Path(path).read_text()
 return json.JSONDecoder().raw_decode(s)[0]
for name,(want,p) in EXPECTED.items():
 got=sha(p)
 if got!=want:raise ValueError(f"{name} pin mismatch: {got}")
ir=readjson(IR);dr=readjson(DR);wr=readjson(WR)
if ir.get("status")!="PASS_PAIRED_I_TYPED_ANCESTRY_AND_INDEPENDENT_VOLUME_REPLAY" or ir.get("candidate_sha256")!=EXPECTED["typed_I"][0] or ir.get("physical_admission") is not False:
 raise ValueError("typed I receipt is not the authenticated source-only pass")
if dr.get("status")!="PASS_ACTUAL_697_TYPED_D_LOADER_METADATA_VIEW" or dr.get("source_candidate",{}).get("sha256")!=EXPECTED["typed_D"][0] or dr.get("physical_admission") is not False:
 raise ValueError("typed D 697 receipt is not the authenticated source-only pass")
if wr.get("status")!="PASS_COLLAR_AND_PROTECTED149_REINDEX_SOURCE_ONLY" or wr.get("physical_admission") is not False:
 raise ValueError("wall receipt is not the pinned source-only collar pass")
# Geometry/incidence owner from the pinned rebase implementation.
s=importlib.util.spec_from_file_location("fat614_remesh296_for_target_rebind",H/"remesh_fat614_compact_cavities_296_007.py")
r=importlib.util.module_from_spec(s);sys.modules[s.name]=r;s.loader.exec_module(r);m=r.m
base=loadnpz(BASE);iz=loadnpz(I);dz=loadnpz(D);collar=loadnpz(COLLAR)
p=np.asarray(base["points_world_m"],np.float32)
t=np.asarray(base["tetrahedra"],np.int64)
mat=np.asarray(base["material_indices"],np.int32)
if p.shape!=(253220,3) or t.shape!=(841623,4):raise ValueError("unexpected rebased geometry domain")
# Require a unique exact F32 coordinate key for every volume row.
def key(x):return np.asarray(x,np.float32).view(np.uint32).tobytes()
vol={}
for row,x in enumerate(p):
 k=key(x)
 if k in vol:raise ValueError(f"duplicate volume F32 coordinate rows {vol[k]} and {row}")
 vol[k]=row
def surface_map(label,source):
 sp=np.asarray(source["points_world_m_f32"],np.float32)
 out=np.empty(len(sp),np.int64)
 missing=[]
 for row,x in enumerate(sp):
  q=vol.get(key(x))
  if q is None:missing.append(row)
  else:out[row]=q
 if missing:raise ValueError(f"{label}: {len(missing)} of {len(sp)} typed points lack exact current Fat volume F32 rows; examples {missing[:12]}")
 faces=np.asarray(source["faces"],np.int64)
 if faces.ndim!=2 or faces.shape[1]!=3 or int(faces.min())<0 or int(faces.max())>=len(sp):
  raise ValueError(f"{label}: malformed source face point domain")
 mapped=out[faces]
 if np.any(mapped[:,0]==mapped[:,1]) or np.any(mapped[:,1]==mapped[:,2]) or np.any(mapped[:,2]==mapped[:,0]):
  raise ValueError(f"{label}: exact F32 mapping collapses a source face")
 if not np.array_equal(p[mapped],sp[faces]):
  raise ValueError(f"{label}: mapped face coordinate bytes differ")
 return out,mapped
imap,ifaces=surface_map("I",iz)
dmap,dfaces=surface_map("D",dz)
# Fat target is current I, reversed current D, and unchanged directed wall faces.
d_fat=dfaces[:,[0,2,1]]
with W.open(newline="") as f: all_wall_rows=list(csv.DictReader(f))
rows=[q for q in all_wall_rows if q["role"]=="fat_I_to_D"]
if len(rows)!=90:raise ValueError(f"expected exactly 90 frozen directed Fat rim walls, got {len(rows)}")
wlocal=np.asarray([[int(q[f"output_node{i}"]) for i in range(3)] for q in rows],np.int64)
wall_ids=np.asarray([int(q["wall_face_row"]) for q in rows],np.int64)
if not np.array_equal(wall_ids,np.arange(90,180,dtype=np.int64)):raise ValueError("Fat wall row IDs are not the authenticated 90..179 block")
collar_points=np.asarray(collar["points_world_m"],np.float32)
collar_faces=np.asarray(collar["fat_sidewall_faces"],np.int64)
if collar_points.ndim!=2 or collar_points.shape[1]!=3 or collar_faces.shape!=(90,3):raise ValueError("pinned collar Fat wall geometry has an unexpected shape")
if not np.array_equal(wlocal,collar_faces):raise ValueError("wall CSV point rows differ from authenticated collar Fat sidewall rows")
if wlocal.min()<0 or wlocal.max()>=len(collar_points):raise ValueError("wall point row outside collar point namespace")
wall_xyz=collar_points[wlocal]
wfaces_volume=np.asarray([[vol.get(key(x),-1) for x in tri] for tri in wall_xyz],np.int64)
if np.any(wfaces_volume<0):raise ValueError("collar wall point lacks exact F32 current volume point")
if not np.array_equal(p[wfaces_volume].view(np.uint32),wall_xyz.view(np.uint32)):raise ValueError("mapped wall point coordinate bits differ from authenticated collar")
no=len(collar_points)-len(collar["donor_I_parent_node_rows"])-len(collar["donor_D_parent_node_rows"])
i_ring_rows=np.arange(no,no+len(collar["donor_I_parent_node_rows"]),dtype=np.int64)
d_ring_rows=np.arange(no+len(collar["donor_I_parent_node_rows"]),len(collar_points),dtype=np.int64)
i_volume_rows=set(map(int,imap))
d_volume_rows=set(map(int,dmap))
collar_i_volume={vol[key(x)] for x in collar_points[i_ring_rows]}
collar_d_volume={vol[key(x)] for x in collar_points[d_ring_rows]}
if not collar_i_volume<=i_volume_rows or not collar_d_volume<=d_volume_rows:raise ValueError("collar I/D ring does not map into the matching authenticated typed surface")
if not set(map(int,np.unique(wfaces_volume)))<=collar_i_volume|collar_d_volume:raise ValueError("Fat wall uses a collar point outside its authenticated I/D rings")
if any(len(set(map(int,f)))<3 for f in wfaces_volume):raise ValueError("degenerate mapped wall face")
target=np.vstack([ifaces,d_fat,wfaces_volume]).astype(np.int64,copy=False)
groups=np.concatenate([np.zeros(len(ifaces),np.int8),np.ones(len(d_fat),np.int8),np.full(len(wfaces_volume),2,np.int8)])
source_rows=np.concatenate([np.arange(len(ifaces),dtype=np.int64),np.arange(len(d_fat),dtype=np.int64),wall_ids])
if len(target)!=483534:raise ValueError(f"current target face count mismatch {len(target)} != 483534")
# Fail on duplicate oriented or unoriented source faces.
canonical=np.sort(target,axis=1)
if len(np.unique(canonical,axis=0))!=len(target):raise ValueError("duplicate unoriented face in I+reversed-D+wall target")
# Validate the complete directed surface chain: every edge appears exactly twice with opposite orientation.
edges=np.vstack([target[:,[0,1]],target[:,[1,2]],target[:,[2,0]]])
e_sorted=np.sort(edges,axis=1)
order=np.lexsort((e_sorted[:,1],e_sorted[:,0]))
es=e_sorted[order]
starts=np.r_[0,1+np.flatnonzero(np.any(es[1:]!=es[:-1],axis=1))]
counts=np.diff(np.r_[starts,len(es)])
signs=np.where(edges[:,0]<edges[:,1],1,-1)[order]
sums=np.add.reduceat(signs,starts)
bad=(counts!=2)|(sums!=0)
if np.any(bad):
 badkeys=es[starts[bad]]
 raise ValueError(f"current typed target I+D+walls is not a closed oriented 2-chain: {len(badkeys)} bad edges; examples {badkeys[:12].tolist()}")
# Recompute actual boundary and its exact oriented difference from the authenticated current target.
actual=m.boundary(t)
if not Counter(map(tuple,m.boundary_identity(actual)))==Counter(map(tuple,m.boundary_identity(np.asarray(base["boundary_faces"],np.int64)))):
 raise ValueError("rebound parent stored actual boundary does not replay")
def facecounter(faces):
 return Counter(tuple(map(int,x)) for x in m.boundary_identity(np.asarray(faces,np.int64).reshape(-1,3)))
got=facecounter(actual);want=facecounter(target);missing=want-got;extra=got-want
diff={"missing_oriented_records":int(sum(missing.values())),"extra_oriented_records":int(sum(extra.values())),
      "missing_samples":[list(k) for k in sorted(missing)[:24]],"extra_samples":[list(k) for k in sorted(extra)[:24]]}
# Exact fluxes make the residual explicit; no geometry changes are made here.
base_flux=int(r.exact_volume(p,actual));target_flux=int(r.exact_volume(p,target))
if OUT.exists():raise FileExistsError(OUT)
OUT.mkdir(parents=True)
out=OUT/"fat-D307-I377-YY-components-current-target-rebound.npz"
payload=dict(base)
payload["stale_5bf_desired_boundary_faces"]=np.asarray(base["desired_boundary_faces"],np.int64)
payload["stale_5bf_desired_boundary_source_rows"]=np.asarray(base.get("desired_boundary_source_rows",np.empty(0,np.int64)),np.int64)
payload["desired_boundary_faces"]=target
payload["desired_boundary_source_rows"]=source_rows
payload["desired_boundary_degenerate_source_rows"]=np.empty(0,np.int64)
payload["desired_boundary_source_group"]=groups
payload["desired_boundary_source_face_rows"]=source_rows.copy()
payload["fat_current_target_rebind_schema"]=np.asarray("numi.fat614.current-typed-I-D-wall-boundary-rebind.v1")
payload["fat_current_target_rebind_I_point_to_volume_rows"]=imap
payload["fat_current_target_rebind_D_point_to_volume_rows"]=dmap
payload["fat_current_target_rebind_wall_faces_volume_rows"]=wfaces_volume
payload["fat_current_target_rebind_wall_faces_collar_point_rows"]=wlocal
payload["fat_current_target_rebind_wall_face_points_world_m_f32"]=wall_xyz
payload["fat_current_target_rebind_source_asset_sha256_json"]=np.asarray(json.dumps({k:v[0] for k,v in EXPECTED.items()},sort_keys=True))
np.savez_compressed(out,**payload)
# Ensure this is strictly a target-metadata derivative.
with np.load(out,allow_pickle=False) as z:
 for k in ("points_world_m","points_world_m_f32","tetrahedra","material_indices","boundary_faces"):
  if array_sha(z[k])!=array_sha(base[k]):raise ValueError(f"target rebind changed geometry member {k}")
report={
 "schema":"numi.fat614.current-typed-I-D-wall-boundary-rebind.v1",
 "status":"PASS_TYPED_CURRENT_TARGET_REBOUND_SOURCE_ONLY_WITH_RESIDUAL_BOUNDARY_MISMATCH",
 "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN","geometry_changed":False,
 "producer_script_path":str(Path(__file__)),"producer_script_sha256":sha(Path(__file__)),
 "inputs":{k:{"path":str(p),"sha256":sha(p),"size_bytes":p.stat().st_size} for k,(_,p) in EXPECTED.items()},
 "typed_surface_counts":{"I_points":len(imap),"I_faces":len(ifaces),"D_points":len(dmap),"D_faces":len(dfaces),"D_faces_reversed_for_Fat":True,"directed_wall_faces":len(wfaces_volume),"target_faces":len(target)},
 "crosswalk":{"I_all_points_exact_F32_mapped":True,"D_all_points_exact_F32_mapped":True,"wall_CSV_exact_collar_face_match":True,"wall_collar_points_exact_F32_mapped_to_candidate":True,"collar_I_and_D_rings_map_to_matching_typed_surfaces":True,"I_point_map_sha256":array_sha(imap),"D_point_map_sha256":array_sha(dmap),"wall_faces_sha256":array_sha(wfaces_volume),"target_faces_sha256":array_sha(target),"target_source_groups_sha256":array_sha(groups),"target_source_rows_sha256":array_sha(source_rows)},
 "target_chain":{"closed_oriented_2_chain":True,"bad_edge_count":0,"duplicate_faces":0},
 "source_candidate":{"path":str(BASE),"sha256":EXPECTED["base_candidate"][0],"points":len(p),"cells":len(t),"actual_boundary_faces":len(actual)},
 "output_candidate":{"path":str(out),"sha256":sha(out),"size_bytes":out.stat().st_size,"points_sha256":array_sha(p),"tets_sha256":array_sha(t),"materials_sha256":array_sha(mat),"actual_boundary_sha256":array_sha(actual),"desired_boundary_faces":len(target)},
 "actual_vs_typed_target":{"missing_oriented_records":diff["missing_oriented_records"],"extra_oriented_records":diff["extra_oriented_records"],"missing_samples":diff["missing_samples"],"extra_samples":diff["extra_samples"],"boundary_complete":not bool(missing or extra)},
 "flux":{"actual_boundary_det6":base_flux,"typed_target_det6":target_flux,"target_minus_actual_det6":target_flux-base_flux},
 "limitations":"The typed target mapping is complete and its I + reversed D + directed wall chain is closed. The rebased volume geometry is byte-identical to its parent. The remaining oriented boundary mismatch is retained explicitly for pending cavity work; no mass, native, or physical admission is claimed."
}
rp=OUT/"report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":report["status"],"candidate":str(out),"candidate_sha256":sha(out),"report":str(rp),"report_sha256":sha(rp),"target_counts":report["typed_surface_counts"],"target_chain":report["target_chain"],"actual_vs_target":diff,"flux_delta":target_flux-base_flux},indent=2))

