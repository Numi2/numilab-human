from pathlib import Path
import numpy as np,json,importlib.util,sys,hashlib
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=P/'fine-physical-volume-706';R=H/'tetgen-scale-relative-tolerance-059/fat009-run001';D=H/'fat009-star-kernel-pilot-062'
owner=P/'deep-boundary-checkpoint-349/payload/exact-owner/numilab_human/cardiac_cavity_intersections.py'
sys.path.insert(0,str(owner.parent.parent));m=__import__('numilab_human.cardiac_cavity_intersections',fromlist=['_audit_pair'])
with np.load(D/'patch-proposals.npz')as z:a={k:z[k]for k in z.files}
nodes=np.loadtxt(R/'fat.1.node',skiprows=1);ele=np.loadtxt(R/'fat.1.ele',skiprows=1,dtype=np.int64);p=np.vstack([nodes[:,1:4].astype(np.float32),a['new_interior_points_world_m']])
t=ele[:,1:5]-1;pat=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
results=[];selected=[];newparts=[];newmaterials=[]
for j,k in enumerate(a['accepted_patch_ids']):
 fan=a['new_tetrahedra_global_rows'][a['new_tet_offsets'][j]:a['new_tet_offsets'][j+1]]
 ids=a['replaced_cell_rows'][a['replaced_cell_offsets'][j]:a['replaced_cell_offsets'][j+1]]
 f=fan[:,pat].reshape(-1,3);_,ix,ct=np.unique(np.sort(f,axis=1),axis=0,return_index=True,return_counts=True);uf=f[ix]
 used=np.unique(uf);mp=np.full(len(p),-1,np.int64);mp[used]=np.arange(len(used));rec,_=m.float32_surface_records(p[used],mp[uf])
 ck=m._audit_pair(rec,rec,same_surface=True)
 labs=np.unique(ele[ids,5])
 passed=ck['count']==0 and len(labs)==1 and np.all(ct<=2)
 results.append({'patch':int(k),'new_unique_faces':len(uf),'exact_crossing_gate':ck,'nonmanifold_faces':int(np.sum(ct>2)),'materials':labs.tolist(),'passed':bool(passed)})
 if passed:selected.extend(ids.tolist());newparts.append(fan);newmaterials.extend([int(labs[0])]*len(fan))
mask=np.ones(len(t),bool);mask[selected]=False
ct=np.vstack([t[mask],*newparts]);cm=np.r_[ele[mask,5],newmaterials].astype(np.int32)
q=p[ct].astype(np.float64);det=np.einsum('ij,ij->i',np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,3]-q[:,0])
np.savez_compressed(D/'fat-partial-positive-proposal.npz',points_world_m=p,tetrahedra=ct,material_indices=cm,retained_source_element_ids=ele[mask,0],replaced_source_element_ids=ele[selected,0])
r={'status':'PARTIAL_LOCAL_VOLUME_REPAIR_EXACT_GATES_PASSED_NOT_PHYSICAL_ADMISSION','physical_admission':False,'runtime':0,'accepted_steps':0,'source_cells':len(t),'candidate_cells':len(ct),'replaced_cells':len(selected),'new_cells':sum(len(x)for x in newparts),'remaining_f32_zero_cells':int(np.sum(det==0)),'remaining_f32_negative_cells':int(np.sum(det<0)),'new_f32_zero_cells':int(np.sum(det[len(t)-len(selected):]==0)),'new_f32_negative_cells':int(np.sum(det[len(t)-len(selected):]<0)),'exact_gate_owner_sha256':hashlib.sha256(owner.read_bytes()).hexdigest(),'patches':results,'remaining_gates':['all F32 cells positive and consistent','original boundary facets restored','same-material ownership','source330 conservative physical mass mapping','native installation and accepted-state motion']}
(D/'exact-patch-verification.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items()if k!='patches'}))
