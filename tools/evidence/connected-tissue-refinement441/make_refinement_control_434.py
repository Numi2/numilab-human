from pathlib import Path
import numpy as np,csv,json,hashlib,shutil
P=Path(__file__).resolve().parent;B=P/"reduced-physical-boundary-400/attempt-002-parent-ancestry-walk";D=P/"refined-native-producer-433/interior-refinement-control-434";D.mkdir(exist_ok=False)
with np.load(B/"candidate.npz") as z:a={k:z[k] for k in z.files}
t=a["volume_tetrahedra"];p=a["points_reference_world_m_f64"].astype(np.float32).astype(np.float64)
b=p[t[:,1]]-p[t[:,0]];c=p[t[:,2]]-p[t[:,0]];e=p[t[:,3]]-p[t[:,0]];vol=np.einsum("ij,ij->i",b,np.cross(c,e))/6
choose=np.flatnonzero((a["material_index"]==16)&(a["geometric_donor_body"]==109));pi=int(choose[np.argmax(vol[choose])]);nid=len(p);source=t[pi]
cp=[];tt=[]
for row,tet in enumerate(t):
 if row!=pi:tt.append(tet);cp.append(row);continue
 for omit in range(4):
  face=[int(tet[k]) for k in range(4) if k!=omit];child=[nid,*face]
  q=np.vstack([p,p[source].mean(0).astype(np.float32).astype(np.float64)])
  x=q[child]
  if np.dot(x[1]-x[0],np.cross(x[2]-x[0],x[3]-x[0]))<0:child[2],child[3]=child[3],child[2]
  tt.append(child);cp.append(row)
cp=np.array(cp,np.int64)
a["volume_tetrahedra"]=np.asarray(tt,np.int64);a["child_parent_tet_row"]=cp
for k in ["material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"]:a[k]=a[k][cp]
for k in ["points_reference_world_m_f64","points_current_world_m_f32","points_donor_local_m_f64","full86_weights","target_current_world_m"]:
 x=a[k];q=x[source].mean(0)
 if k in ["points_reference_world_m_f64","points_current_world_m_f32"]:q=q.astype(np.float32).astype(x.dtype)
 a[k]=np.vstack([x,q])
a["parent_volume_node_row"]=np.append(a["parent_volume_node_row"],-1)
np.savez_compressed(D/"candidate.npz",**a)
with (D/"parent-reduced-boundary.node").open("w") as f:
 f.write(str(len(p)+1)+" 3 0 0\n")
 for i,q in enumerate(a["points_reference_world_m_f64"]):f.write(str(i+1)+" "+" ".join(format(float(x),".17g") for x in q)+"\n")
shutil.copyfile(B/"coarse-node-to-fine-surface.csv",D/"coarse-node-to-fine-surface.csv")
with (D/"refined-node-parent-ancestry.csv").open("w",newline="") as f:
 w=csv.writer(f);w.writerow(["volume_node_row","parent_tet_row",*["parent_bary"+str(i) for i in range(4)]]);w.writerow([nid,pi,.25,.25,.25,.25])
(D/"fixture.json").write_text(json.dumps({"scope":"one interior-node refinement registration/mass regression; source geometry remains unresolved; not physical admission","source_candidate_sha256":hashlib.sha256((B/"candidate.npz").read_bytes()).hexdigest(),"split_original_parent_tet":pi,"original_nodes":len(p),"nodes":len(p)+1,"cells":len(tt),"original_outer_boundary_unchanged":True,"physical_admission":False},indent=2)+"\n")
print(str(D))
