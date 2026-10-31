from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
H=Path(__file__).resolve().parents[1]
P=H.parent
SRC=H/"complete-surface-quotient-214-root/run002/skin-complete-surface-quotient-remesh-input.npz"
SHELLS=H/"complete-surface-cavity-217-native/skin-run001/connected-cavity-shells.npz"
CAND217=H/"complete-surface-cavity-217-native/skin-run001/connected-fixed-surface-volume-proposal.npz"
CAND222=H/"complete-surface-cavity-222-root/run-001/skin-composed-candidate.npz"
OUT=H/"complete-surface-cavity-223-root/run-001"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
m=load("cav209_223",H/"transfer_surface_contractions_to_volume_209_root.py")
with np.load(SRC,allow_pickle=False) as z:v={k:z[k] for k in z.files}
with np.load(SHELLS,allow_pickle=False) as z:sh={k:z[k] for k in z.files}
with np.load(CAND217,allow_pickle=False) as z:c217={k:z[k] for k in z.files}
with np.load(CAND222,allow_pickle=False) as z:c222={k:z[k] for k in z.files}
# Pin the failed component-4 evidence and map its component shell to global point rows.
ci=4;po=sh["point_offsets"];fo=sh["face_offsets"];co=sh["cell_offsets"]
comp_pts=sh["global_point_rows"][po[ci]:po[ci+1]]
comp_faces=comp_pts[sh["local_faces"][fo[ci]:fo[ci+1]]]
comp_keys={tuple(sorted(map(int,f))) for f in comp_faces}
miss=v["desired_missing_faces"];extra=v["current_extra_boundary_faces"]
miss4=np.asarray([f for f in miss if tuple(sorted(map(int,f))) in comp_keys],np.int64)
# Separate the 19 DP-added facets into components joined by shared vertices.
par=list(range(len(miss4)))
def find(x):
 while par[x]!=x:par[x]=par[par[x]];x=par[x]
 return x
def union(a,b):
 a,b=find(a),find(b)
 if a!=b:par[b]=a
owner={}
for i,f in enumerate(miss4):
 for x in f:
  x=int(x)
  if x in owner:union(i,owner[x])
  else:owner[x]=i
cc={}
for i,f in enumerate(miss4):cc.setdefault(find(i),[]).append(f)
clusters=sorted((np.asarray(x,np.int64) for x in cc.values()),key=lambda x:(-len(x),tuple(map(int,np.unique(x)))))
# First target: largest disconnected DP facet component, 11 faces / 13 points.
want=clusters[0]
seed=154706
if len(want)!=11 or seed not in want or len(np.unique(want))!=13:raise SystemExit("component-4 DP cluster identity changed")
t=v["tetrahedra"]
extra_keys={tuple(sorted(map(int,f))) for f in extra}
inc_cache={}
def inc(x):
 x=int(x)
 if x not in inc_cache:inc_cache[x]=np.flatnonzero(np.any(t==x,axis=1))
 return inc_cache[x]
def bad_nodes(faces):
 ed=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);se=np.sort(ed,axis=1);order=np.lexsort((se[:,1],se[:,0]));ss=se[order]
 starts=np.r_[0,1+np.flatnonzero(np.any(ss[1:]!=ss[:-1],axis=1))];ct=np.diff(np.r_[starts,len(ss)])
 signs=np.where(ed[:,0]<ed[:,1],1,-1);sm=np.add.reduceat(signs[order],starts)
 return np.unique(ss[starts[(ct!=2)|(sm!=0)]])
def closure(faces):
 ed=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);se=np.sort(ed,axis=1);order=np.lexsort((se[:,1],se[:,0]));ss=se[order]
 starts=np.r_[0,1+np.flatnonzero(np.any(ss[1:]!=ss[:-1],axis=1))];ct=np.diff(np.r_[starts,len(ss)])
 signs=np.where(ed[:,0]<ed[:,1],1,-1);sm=np.add.reduceat(signs[order],starts)
 return bool(np.all(ct==2) and np.all(sm==0)),int(np.sum((ct!=2)|(sm!=0)))
def assemble(sel):
 bf=m.boundary(t[sel])
 bf=np.asarray([f for f in bf if tuple(sorted(map(int,f))) not in extra_keys],np.int64).reshape(-1,3)
 return np.vstack([bf,want])
sel=inc(seed).copy();history=[]
for it in range(12):
 faces=assemble(sel);closed,bad=closure(faces)
 history.append({"iteration":it,"cells":len(sel),"unique_points":int(len(np.unique(t[sel]))),"boundary_faces":len(faces),"closed_oriented":closed,"bad_edge_records":bad})
 if closed:break
 bn=bad_nodes(faces)
 if not len(bn):break
 more=np.unique(np.concatenate([inc(x) for x in bn]));new=np.setdiff1d(more,sel)
 if not len(new) or len(sel)+len(new)>1500:break
 sel=np.unique(np.r_[sel,new])
faces=assemble(sel)
if not closure(faces)[0]:raise SystemExit("bounded seed expansion did not close")
pts_global=np.unique(t[sel])
if len(pts_global)!=73 or len(sel)!=150 or len(faces)!=138:raise SystemExit("bounded shell dimensions changed")
idx=np.full(len(v["points_world_m"]),-1,np.int64);idx[pts_global]=np.arange(len(pts_global))
localfaces=idx[faces]
if np.any(localfaces<0) or len({tuple(sorted(map(int,f))) for f in localfaces})!=len(localfaces):raise SystemExit("invalid or duplicate shell face")
# Exact Float32 lattice signed volume.
ip=m.m.integer_points(v["points_world_m"][pts_global]);look={int(x):i+1 for i,x in enumerate(pts_global)};ip=[(0,0,0)]+ip
volume_det=sum(m.m.exact_det(ip,[0]+[look[int(x)] for x in f]) for f in faces)
if volume_det<=0:raise SystemExit("restored DP shell is not positive oriented")
# Every target facet is present exactly once, in its authored orientation.
face_lookup={tuple(sorted(map(int,f))):f for f in faces}
if any(tuple(sorted(map(int,f))) not in face_lookup or not np.array_equal(face_lookup[tuple(sorted(map(int,f)))],f) for f in want):raise SystemExit("desired DP facet orientation/coverage changed")
# This candidate patch must map onto untouched 217 and 222 rows.
ret217=c217["retained_input_cell_rows"]
if not np.array_equal(ret217,np.sort(ret217)):raise SystemExit("217 retained input rows are not ordered")
out217=np.searchsorted(ret217,sel)
if np.any(out217>=len(ret217)) or not np.array_equal(ret217[out217],sel):raise SystemExit("some selected 214 source cells were replaced in 217")
ret222=c222["retained_input_candidate_tet_rows"]
if not np.array_equal(ret222,np.sort(ret222)):raise SystemExit("222 retained 217 rows are not ordered")
if not np.all(np.isin(out217,ret222)):raise SystemExit("some selected 217 cells were replaced in 222")
# All mapped coordinates are exact input-prefix rows; no output vertex renumbering.
if not np.array_equal(c217["points_world_m"][:len(v["points_world_m"])],v["points_world_m"]):raise SystemExit("217 point prefix differs from 214 input")
if not np.array_equal(c222["points_world_m"],c217["points_world_m"]):raise SystemExit("222 point coordinates differ from 217")
# Pin expected oriented cell connectivity for selected retained cells.
if not np.array_equal(c217["tetrahedra"][out217],t[sel]):raise SystemExit("217 selected-cell connectivity is not source-identical")
out222=np.searchsorted(ret222,out217)
if not np.array_equal(c222["tetrahedron_output_to_input_candidate_row"][out222],out217):raise SystemExit("222 selected-cell output map is not exact")
np.savez_compressed(OUT/"component4-dp-cluster-11face-shell.npz",points_world_m=v["points_world_m"][pts_global],faces=localfaces,global_point_rows=pts_global,source_input_tet_rows=sel,source_217_output_tet_rows=out217,source_222_output_tet_rows=out222,desired_dp_faces_global=want,seed_global_point_row=np.asarray([seed],np.int64),selected_extra_boundary_faces_global=np.asarray([f for f in m.boundary(t[sel]) if tuple(sorted(map(int,f))) in extra_keys],np.int64))
report={"schema":"numi.skin.component4-dp-cluster-shell-223.v1","status":"PASS_CLOSED_DP_CLUSTER_SHELL_PREPARATION","physical_admission":False,"native_steps":0,"source_214_path":str(SRC),"source_214_sha256":sha(SRC),"component_shells_path":str(SHELLS),"component_shells_sha256":sha(SHELLS),"candidate_217_path":str(CAND217),"candidate_217_sha256":sha(CAND217),"candidate_222_path":str(CAND222),"candidate_222_sha256":sha(CAND222),"component_id":4,"component4_failed_cell_count":1478,"component4_dp_added_facets":19,"dp_face_cluster_count":3,"selected_dp_cluster_face_count":len(want),"selected_dp_cluster_global_faces":want.tolist(),"selected_dp_cluster_points":np.unique(want).tolist(),"seed_global_point_row":seed,"seed_expansion_history":history,"selected_source_214_cell_rows":sel.tolist(),"mapped_217_output_cell_rows":out217.tolist(),"mapped_222_output_cell_rows":out222.tolist(),"shell_global_point_rows":pts_global.tolist(),"shell_face_count":len(faces),"shell_unique_point_count":len(pts_global),"shell_closed_oriented":closure(faces)[0],"shell_edge_failure_count":closure(faces)[1],"shell_target_facets_exact_oriented":True,"shell_exact_positive_volume_det_lattice":str(volume_det),"shell_exact_positive_volume_m3":float(volume_det*(2.0**-447)/6),"selected_cells_materials":np.unique(v["material_indices"][sel]).tolist(),"selected_cells_retained_unchanged_in_217_and_222":True,"point_prefix_unchanged_in_217_and_222":True,"shell_path":str(OUT/"component4-dp-cluster-11face-shell.npz"),"shell_sha256":sha(OUT/"component4-dp-cluster-11face-shell.npz")}
(OUT/"shell-preflight.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:report[k] for k in ("status","selected_dp_cluster_face_count","seed_global_point_row","selected_dp_cluster_points","seed_expansion_history","shell_unique_point_count","shell_face_count","shell_closed_oriented","shell_edge_failure_count","shell_exact_positive_volume_m3","selected_cells_materials","selected_cells_retained_unchanged_in_217_and_222","shell_sha256")},indent=2))
