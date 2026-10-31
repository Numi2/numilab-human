from pathlib import Path
import importlib.util,itertools,json,hashlib,numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
S=H/'complete-surface-cavity-217-native/skin-run001/connected-fixed-surface-volume-proposal.npz'
M=H/'transfer_surface_contractions_to_volume_209_root.py'
Q=H/'complete-surface-cavity-220-root/solve_oriented_cavity_milp_220.py'
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
a=load('shellowner220',M);c=load('cavity220diag',Q)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with np.load(S,allow_pickle=False) as z:p=z['points_world_m'];t=z['tetrahedra'];mat=z['material_indices']
target=463367; corners=t[target]; rowsout=[]
for center in corners:
 cellrows=np.flatnonzero(np.any(t==center,axis=1)); localglobal=np.unique(t[cellrows]); inv=np.full(len(p),-1,np.int64);inv[localglobal]=np.arange(len(localglobal)); lp=p[localglobal]; lt=inv[t[cellrows]]; bglobal=a.boundary(t[cellrows]); b=inv[bglobal]
 candidates=np.asarray(list(itertools.combinations(range(len(lp)),4)),np.int64);q=lp[candidates].astype(np.float64);u,v,w=(q[:,j]-q[:,0]for j in (1,2,3));ds=np.einsum('ij,ij->i',u,np.cross(v,w));uncertain=np.flatnonzero(np.abs(ds)<=np.finfo(float).eps*64*np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)*np.linalg.norm(w,axis=1));xyz=[c.ip(x)for x in lp]
 for k in uncertain:
  d=c.det(*(xyz[int(v)]for v in candidates[k]));ds[k]=1e-300 if d>0 else -1e-300 if d<0 else 0.
 keep=ds!=0;candidates=candidates[keep];ds=ds[keep];neg=ds<0;candidates[neg,0],candidates[neg,1]=candidates[neg,1].copy(),candidates[neg,0].copy();ok,ndet,norm=c.native_quality_mask(lp,candidates);bad=int(np.count_nonzero(~ok));candidates=candidates[ok];faces=candidates[:,c.PAT].reshape(-1,3);fb=np.sort(faces,axis=1);n=len(lp);codes=(fb[:,0]*n+fb[:,1])*n+fb[:,2];order=np.argsort(codes);sc=codes[order];bc=np.sort(b,axis=1);bcode=(bc[:,0]*n+bc[:,1])*n+bc[:,2];sg=c.signs(faces);bsg=c.signs(b);missing=[]
 for i,(code,sign) in enumerate(zip(bcode,bsg)):
  lo=np.searchsorted(sc,code,'left');hi=np.searchsorted(sc,code,'right');okorient=0
  if hi>lo: okorient=int(np.count_nonzero(sg[order[lo:hi]]==sign))
  if not okorient:missing.append(i)
 rowsout.append({'center_global_point_row':int(center),'source_cell_rows':int(len(cellrows)),'point_rows':int(len(localglobal)),'boundary_faces':int(len(b)),'nondegenerate_candidates':int(np.count_nonzero(keep)),'quality_rejected':bad,'quality_admissible':int(len(candidates)),'boundary_faces_without_quality_admissible_incident_tet':missing,'uncoverable_boundary_face_count':len(missing),'materials':np.unique(mat[cellrows]).tolist()})
R={'schema':'numi.skin.cavity-220.boundary-quality-coverage.v1','source_path':str(S),'source_sha256':sha(S),'quality_owner_path':str(c.QUALITY_OWNER_PATH),'quality_owner_sha256':sha(c.QUALITY_OWNER_PATH),'target_negative_cell_row':int(target),'target_negative_corners':corners.tolist(),'corner_stars':rowsout,'physical_admission':False}
out=H/'complete-surface-cavity-220-root'/'quality-boundary-coverage-220.json';out.write_text(json.dumps(R,indent=2)+'\n');print(json.dumps(R,indent=2))
