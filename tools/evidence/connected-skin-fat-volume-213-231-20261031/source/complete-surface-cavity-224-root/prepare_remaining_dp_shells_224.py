from pathlib import Path
import hashlib, importlib.util, json, sys, time
import numpy as np
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
SRC=H/"complete-surface-quotient-214-root/run002/skin-complete-surface-quotient-remesh-input.npz"
SHELLS=H/"complete-surface-cavity-217-native/skin-run001/connected-cavity-shells.npz"
C217=H/"complete-surface-cavity-217-native/skin-run001/connected-fixed-surface-volume-proposal.npz"
C222=H/"complete-surface-cavity-222-root/run-001/skin-composed-candidate.npz"
C224=H/"complete-surface-cavity-224-root/run-002/skin-component4-dp-cluster-composed-candidate.npz"
OUT=H/"complete-surface-cavity-224-root/run-004-dp-clusters-shells"
PINS={
 str(SRC):"42d438fba30b43243cafa1e9f33fa2b247404c1b979757b724c7e51132f62ec8",
 str(C217):"da043a9e566b4af38eb8a6cb07860da220f771be85b294309f655f530913d46b",
 str(C222):"cb4c57dd4f281edfbcf9361f1b848dc52a91793978d138c19540eff50efa079d",
 str(C224):"eb90c2ac505f426886f03ca7867335794618f791e88cad3b0e67e5165f914f74",
}
def sha(p):return hashlib.file_digest(open(p,"rb"),"sha256").hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
for path,pin in PINS.items():
 if sha(path)!=pin:raise SystemExit("pinned input changed: "+path)
if OUT.exists():raise SystemExit("refusing to overwrite "+str(OUT))
m=load("boundary_owner_small_dp224",H/"transfer_surface_contractions_to_volume_209_root.py")
q=load("quality_owner_small_dp224",H/"fixed_boundary_volume_flip_198_root.py")
with np.load(SRC,allow_pickle=False) as z:v={k:z[k] for k in z.files}
with np.load(SHELLS,allow_pickle=False) as z:sh={k:z[k] for k in z.files}
with np.load(C217,allow_pickle=False) as z:c217={k:z[k] for k in z.files}
with np.load(C222,allow_pickle=False) as z:c222={k:z[k] for k in z.files}
with np.load(C224,allow_pickle=False) as z:c224={k:z[k] for k in z.files}
ci=4;po=sh["point_offsets"];fo=sh["face_offsets"]
comp_pts=sh["global_point_rows"][po[ci]:po[ci+1]]
comp_faces=comp_pts[sh["local_faces"][fo[ci]:fo[ci+1]]]
comp_keys={tuple(sorted(map(int,f))) for f in comp_faces}
miss=v["desired_missing_faces"];extra=v["current_extra_boundary_faces"]
miss4=np.asarray([f for f in miss if tuple(sorted(map(int,f))) in comp_keys],np.int64)
par=list(range(len(miss4)))
def find(x):
 while par[x]!=x:
  par[x]=par[par[x]];x=par[x]
 return x
def union(a,b):
 a=find(a);b=find(b)
 if a!=b:par[b]=a
faceowner={}
for i,f in enumerate(miss4):
 for vv in f:
  vv=int(vv)
  if vv in faceowner:union(i,faceowner[vv])
  else:faceowner[vv]=i
cc={}
for i,f in enumerate(miss4):cc.setdefault(find(i),[]).append(f)
clusters=sorted((np.asarray(x,np.int64) for x in cc.values()),key=lambda x:(-len(x),tuple(map(int,np.unique(x)))))
if [len(x) for x in clusters] != [11,5,3]:raise SystemExit("component-4 DP cluster sizes changed")
t=v["tetrahedra"].astype(np.int64);npoints=len(v["points_world_m"])
flat=t.reshape(-1);counts=np.bincount(flat,minlength=npoints);order=np.argsort(flat,kind="stable");rowids=order//4;offsets=np.r_[0,np.cumsum(counts)]
def inc(x):return rowids[offsets[int(x)]:offsets[int(x)+1]]
extra_keys={tuple(sorted(map(int,f))) for f in extra}
def bad_nodes(faces):
 ed=np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
 se=np.sort(ed,axis=1);order=np.lexsort((se[:,1],se[:,0]));ss=se[order]
 starts=np.r_[0,1+np.flatnonzero(np.any(ss[1:]!=ss[:-1],axis=1))];ct=np.diff(np.r_[starts,len(ss)])
 signs=np.where(ed[:,0]<ed[:,1],1,-1);sm=np.add.reduceat(signs[order],starts)
 return np.unique(ss[starts[(ct!=2)|(sm!=0)]])
def close(sel,want):
 bf=m.boundary(t[sel])
 bf=np.asarray([f for f in bf if tuple(sorted(map(int,f))) not in extra_keys],np.int64).reshape(-1,3)
 faces=np.vstack((bf,want))
 ed=np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]));se=np.sort(ed,axis=1);order=np.lexsort((se[:,1],se[:,0]));ss=se[order]
 starts=np.r_[0,1+np.flatnonzero(np.any(ss[1:]!=ss[:-1],axis=1))];ct=np.diff(np.r_[starts,len(ss)])
 signs=np.where(ed[:,0]<ed[:,1],1,-1);sm=np.add.reduceat(signs[order],starts)
 return bool(np.all(ct==2) and np.all(sm==0)),int(np.sum((ct!=2)|(sm!=0))),faces,bf
def try_seed(seed,want):
 sel=np.unique(inc(seed));hist=[]
 for it in range(20):
  pts=np.unique(t[sel]);closed,bad,faces,bf=close(sel,want)
  hist.append({"iteration":it,"cells":int(len(sel)),"unique_points":int(len(pts)),"boundary_faces":int(len(faces)),"closed_oriented":closed,"bad_edge_records":bad})
  if closed:return sel,pts,faces,bf,hist
  if len(pts)>90:return None,None,None,None,hist
  bn=bad_nodes(faces)
  if not len(bn):return None,None,None,None,hist
  more=np.unique(np.concatenate([inc(x) for x in bn]))
  new=np.setdiff1d(more,sel)
  if not len(new):return None,None,None,None,hist
  nxt=np.unique(np.r_[sel,new])
  if len(np.unique(t[nxt]))>90:return None,None,None,None,hist
  sel=nxt
 return None,None,None,None,hist
ret217=c217["retained_input_cell_rows"];ret222=c222["retained_input_candidate_tet_rows"];ret224=c224["retained_input_candidate_tet_rows"]
OUT.mkdir(parents=True)
results=[]
for ci_cluster in (1,2):
 want=clusters[ci_cluster]
 candidates=[]
 for seed in map(int,np.unique(want)):
  sel,pts,faces,bf,hist=try_seed(seed,want)
  if sel is None:continue
  r217=np.searchsorted(ret217,sel)
  if np.any(r217>=len(ret217)) or not np.array_equal(ret217[r217],sel):continue
  r222=np.searchsorted(ret222,r217)
  if np.any(r222>=len(ret222)) or not np.array_equal(ret222[r222],r217):continue
  pos224=np.searchsorted(ret224,r222)
  if np.any(pos224>=len(ret224)) or not np.array_equal(ret224[pos224],r222):continue
  candidates.append((len(pts),len(sel),seed,sel,pts,faces,bf,hist,r217,r222))
 if not candidates:
  results.append({"cluster_index":ci_cluster,"target_face_count":len(want),"target_faces":want.tolist(),"status":"NO_CLOSED_SHELL_WITHIN_90_POINTS","physical_admission":False})
  continue
 candidates.sort(key=lambda x:(x[0],x[1],x[2]))
 nppts,ncells,seed,sel,pts_global,faces,bf,hist,r217,r222=candidates[0]
 point_index=np.full(npoints,-1,np.int64);point_index[pts_global]=np.arange(len(pts_global))
 local_faces=point_index[faces]
 if np.any(local_faces<0):raise SystemExit("shell face omitted from point map")
 volume_ip=q.integer_points(v["points_world_m"][pts_global]);global_local={int(g):i+1 for i,g in enumerate(pts_global)};volume_ip=[(0,0,0)]+volume_ip
 volume=sum(q.exact_det(volume_ip,[0]+[global_local[int(x)] for x in f]) for f in faces)
 if volume<=0:raise SystemExit("shell not positively oriented")
 target_keys={tuple(sorted(map(int,f))) for f in want}
 face_lookup={tuple(sorted(map(int,f))):f for f in faces}
 if any(tuple(sorted(map(int,f))) not in face_lookup or not np.array_equal(face_lookup[tuple(sorted(map(int,f)))],f) for f in want):raise SystemExit("target face orientation mismatch")
 name=f"component4-dp-cluster-{len(want)}face-shell.npz";path=OUT/name
 np.savez_compressed(path,points_world_m=v["points_world_m"][pts_global],faces=local_faces,global_point_rows=pts_global,source_input_tet_rows=sel,source_217_output_tet_rows=r217,source_222_output_tet_rows=r222,desired_dp_faces_global=want,seed_global_point_row=np.asarray([seed],np.int64),selected_extra_boundary_faces_global=np.asarray([f for f in bf if tuple(sorted(map(int,f))) in extra_keys],np.int64))
 results.append({"cluster_index":ci_cluster,"target_face_count":len(want),"target_faces":want.tolist(),"status":"PASS_CLOSED_DP_CLUSTER_SHELL_PREPARATION","physical_admission":False,"native_steps":0,"selected_seed_global_point_row":seed,"seed_candidates_closed_under_90_points":len(candidates),"selected_source_tet_rows":sel.tolist(),"mapped_217_output_tet_rows":r217.tolist(),"mapped_222_output_tet_rows":r222.tolist(),"selected_point_count":int(nppts),"selected_tet_count":int(ncells),"boundary_face_count":int(len(faces)),"closed_oriented":True,"bad_edge_records":0,"target_faces_oriented_and_present":len(target_keys)==len(want),"exact_positive_shell_volume_det_lattice":str(volume),"exact_positive_shell_volume_m3":float(volume*(2.0**-447)/6),"seed_expansion_history":hist,"shell_path":str(path),"shell_sha256":sha(path)})
report={"schema":"numi.skin.component4-remaining-dp-shells-224.v1","physical_admission":False,"native_steps":0,"source_214_path":str(SRC),"source_214_sha256":sha(SRC),"candidate_217_sha256":sha(C217),"candidate_222_sha256":sha(C222),"candidate_224_path":str(C224),"candidate_224_sha256":sha(C224),"component_id":4,"component4_dp_cluster_sizes":[len(x) for x in clusters],"max_shell_points":90,"clusters":results}
(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"status":"COMPLETE","clusters":[{k:v for k,v in x.items() if k not in ("selected_source_tet_rows","mapped_217_output_tet_rows","mapped_222_output_tet_rows","seed_expansion_history","target_faces")} for x in results]},indent=2))
