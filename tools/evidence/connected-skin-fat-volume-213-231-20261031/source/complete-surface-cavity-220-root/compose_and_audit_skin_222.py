from pathlib import Path
import argparse,hashlib,importlib.util,json,time,numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
INPUT=H/'complete-surface-cavity-217-native/skin-run001/connected-fixed-surface-volume-proposal.npz'
INPUT_SHA='da043a9e566b4af38eb8a6cb07860da220f771be85b294309f655f530913d46b'
PATCH=H/'complete-surface-cavity-220-root/skin-negative-cell-463367-corner-154492/milp-run-001/global-patch-proposal.npz'
PATCH_SHA='2c88b1c65044d36a7b6c8450e93cc0cfa7f89fed09931c35cc10a752d5e95b2f'
PATCH_RECEIPT=PATCH.with_suffix('.receipt.json')
PATCH_RECEIPT_SHA='801cb8fd7646ac2f316306ece30fe118a2add8f862fb67cd8add0ceb2b263144'
OWNER=H/'transfer_surface_contractions_to_volume_209_root.py'
QUALITY=H/'fixed_boundary_volume_flip_198_root.py'
OWNER_SHA='5d90632e208af37d3a4f4bd6bcaaad3533eec1113e5145cdb3fecfba322dbdbd'
QUALITY_SHA='650794951cbcf338de9cde691a8f43377b9d023ccea78b876e87c847880243e0'
PAT=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],np.int64)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def exact_audit(p,t,q):
 npos=nneg=nzero=nunc=0;badrows=[];minnorm=float('inf');minvol=float('inf');quality_bad=[]
 for lo in range(0,len(t),100000):
  tt=t[lo:lo+100000];x=p[tt].astype(np.float64);u=x[:,1]-x[:,0];v=x[:,2]-x[:,0];w=x[:,3]-x[:,0]
  d=np.einsum('ij,ij->i',u,np.cross(v,w))
  # Same six-term determinant permanent used by the existing filtered exact audit.
  perm=(np.abs(u[:,0])*(np.abs(v[:,1]*w[:,2])+np.abs(v[:,2]*w[:,1]))+
        np.abs(u[:,1])*(np.abs(v[:,0]*w[:,2])+np.abs(v[:,2]*w[:,0]))+
        np.abs(u[:,2])*(np.abs(v[:,0]*w[:,1])+np.abs(v[:,1]*w[:,0])))
  suspect=np.flatnonzero(np.abs(d)<=32*np.finfo(np.float64).eps*perm)
  signs=np.sign(d).astype(np.int8)
  for j in suspect:
   ip=q.integer_points(p[tt[j]])
   exact=q.exact_det(ip,[0,1,2,3]);signs[j]=(exact>0)-(exact<0)
  nunc+=len(suspect);npos+=int(np.count_nonzero(signs>0));nneg+=int(np.count_nonzero(signs<0));nzero+=int(np.count_nonzero(signs==0))
  badrows.extend((lo+np.flatnonzero(signs<=0)).tolist())
  det,nd=q.measurements(p,tt);ok=q.passes(det,nd);quality_bad.extend((lo+np.flatnonzero(~ok)).tolist())
  minnorm=min(minnorm,float(np.min(nd)));minvol=min(minvol,float(np.min(det/6)))
 return {'sign_counts':{'positive':npos,'negative':nneg,'zero':nzero},'uncertain_filtered_exact_rechecks':nunc,'nonpositive_rows':badrows,'native_quality_failures':len(quality_bad),'native_quality_failure_rows':quality_bad,'native_minimum_normalized_determinant':minnorm,'native_minimum_volume_m3':minvol,'sign_filter':'32 epsilon times six-term determinant permanent; uncertain Float32 cells are checked by pinned 198 exact dyadic integer determinant'}
def boundary_closed(b):
 e=np.concatenate([b[:,[0,1]],b[:,[1,2]],b[:,[2,0]]]);s=np.sort(e,axis=1);order=np.lexsort((s[:,1],s[:,0]));ss=s[order];starts=np.r_[0,1+np.flatnonzero(np.any(ss[1:]!=ss[:-1],axis=1))];counts=np.diff(np.r_[starts,len(ss)]);sg=np.where(e[:,0]<e[:,1],1,-1);sums=np.add.reduceat(sg[order],starts)
 return bool(np.all(counts==2) and np.all(sums==0)),int(np.count_nonzero((counts!=2)|(sums!=0)))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();started=time.monotonic()
 if a.out.exists():raise SystemExit('refusing to overwrite '+str(a.out))
 for path,pin in [(INPUT,INPUT_SHA),(PATCH,PATCH_SHA),(PATCH_RECEIPT,PATCH_RECEIPT_SHA),(OWNER,OWNER_SHA),(QUALITY,QUALITY_SHA)]:
  if sha(path)!=pin:raise SystemExit('input hash mismatch: '+str(path))
 a.out.mkdir(parents=True)
 m=load('boundary_owner_209_for_222',OWNER);q=load('quality_owner_198_for_222',QUALITY)
 with np.load(INPUT,allow_pickle=False)as z:src={k:z[k]for k in z.files}
 with np.load(PATCH,allow_pickle=False)as z:patch={k:z[k]for k in z.files}
 p=src['points_world_m'];old_t=src['tetrahedra'].astype(np.int64);old_m=src['material_indices'].astype(np.int32);old_b=src['boundary_faces'].astype(np.int64);desired=src['desired_boundary_faces'].astype(np.int64)
 if p.dtype!=np.float32 or not np.isfinite(p).all():raise SystemExit('input coordinates are not finite F32')
 if old_t.ndim!=2 or old_t.shape[1]!=4 or old_m.shape!=(len(old_t),):raise SystemExit('input cell/material row shape mismatch')
 if old_t.min()<0 or old_t.max()>=len(p):raise SystemExit('input tet point index out of range')
 g=patch['global_point_rows'].astype(np.int64);new_t=patch['global_tetrahedra'].astype(np.int64);ret=patch['retired_source_cell_rows'].astype(np.int64);new_m=patch['material_indices'].astype(np.int32)
 if g.ndim!=1 or len(np.unique(g))!=len(g) or g.min()<0 or g.max()>=len(p):raise SystemExit('patch point map invalid')
 if patch['points_world_m'].dtype!=np.float32 or not np.array_equal(patch['points_world_m'],p[g]):raise SystemExit('patch point map does not preserve exact source F32 coordinates')
 if new_t.ndim!=2 or new_t.shape[1]!=4 or new_m.shape!=(len(new_t),) or new_t.min()<0 or new_t.max()>=len(p):raise SystemExit('patch tet/material rows invalid')
 if ret.ndim!=1 or len(np.unique(ret))!=len(ret) or ret.min()<0 or ret.max()>=len(old_t):raise SystemExit('retired source cell map invalid')
 center=154492
 if center not in g:raise SystemExit('patch point map omits its pinned star center')
 expected_ret=np.flatnonzero(np.any(old_t==center,axis=1))
 if not np.array_equal(ret,expected_ret):raise SystemExit('patch retired cells differ from complete source vertex star')
 if not np.all(old_m[ret]==15) or not np.all(new_m==15) or not np.all(patch['material_indices']==15):raise SystemExit('Skin material alignment failed')
 keep=np.ones(len(old_t),bool);keep[ret]=False;kept_rows=np.flatnonzero(keep)
 out_t=np.concatenate([old_t[kept_rows],new_t]);out_m=np.concatenate([old_m[kept_rows],new_m])
 out_to_in=np.concatenate([kept_rows,np.full(len(new_t),-1,np.int64)]);out_patch=np.concatenate([np.full(len(kept_rows),-1,np.int32),np.zeros(len(new_t),np.int32)])
 if np.any(out_t<0) or np.any(out_t>=len(p)) or out_m.shape!=(len(out_t),):raise SystemExit('composed rows invalid')
 sorted_t=np.sort(out_t,axis=1);_,dup_counts=np.unique(sorted_t,axis=0,return_counts=True);duplicate_count=int(np.sum(dup_counts>1))
 if duplicate_count:raise SystemExit(f'duplicate output tetrahedra: {duplicate_count}')
 try:out_b=m.boundary(out_t);boundary_error=None
 except Exception as exc:out_b=None;boundary_error=repr(exc)
 source_boundary_id=m.boundary_identity(old_b)
 if out_b is not None:
  output_boundary_id=m.boundary_identity(out_b);boundary_identity_same=bool(np.array_equal(source_boundary_id,output_boundary_id));boundary_rows_equal=bool(np.array_equal(old_b,out_b));closed,edge_fail=boundary_closed(out_b)
  # The stored 217 boundary itself must also be a consistent closed oriented chain.
  source_closed,source_edge_fail=boundary_closed(old_b)
 else:
  output_boundary_id=np.empty((0,4),np.int64);boundary_identity_same=False;boundary_rows_equal=False;closed=False;edge_fail=-1;source_closed,source_edge_fail=boundary_closed(old_b)
 quality_result=exact_audit(p,out_t,q)
 material_counts={str(int(k)):int(v) for k,v in zip(*np.unique(out_m,return_counts=True))}
 # The original boundary field remains the input contract. Compare exact canonical oriented row identity.
 report={'schema':'numi.skin.complete-surface-cavity-222.v1','status':'PASS_LOCAL_CAVITY_COMPOSITION' if boundary_identity_same and boundary_error is None and duplicate_count==0 and quality_result['sign_counts']['negative']==0 and quality_result['sign_counts']['zero']==0 else 'REJECT_COMPOSITION_GATE','physical_admission':False,'native_steps':0,'input_candidate_path':str(INPUT),'input_candidate_sha256':INPUT_SHA,'patch_path':str(PATCH),'patch_sha256':PATCH_SHA,'patch_receipt_path':str(PATCH_RECEIPT),'patch_receipt_sha256':PATCH_RECEIPT_SHA,'native_quality_owner_path':str(QUALITY),'native_quality_owner_sha256':QUALITY_SHA,'boundary_owner_path':str(OWNER),'boundary_owner_sha256':OWNER_SHA,'target_negative_input_tet_row':463367,'patch_star_center_global_point_row':center,'retired_input_tet_rows':len(ret),'new_patch_tets':len(new_t),'output_points':len(p),'output_tets':len(out_t),'output_material_counts':material_counts,'retired_material_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(old_m[ret],return_counts=True))},'duplicate_tetrahedra':duplicate_count,'boundary_extraction_error':boundary_error,'source_boundary_faces':len(old_b),'output_boundary_faces':0 if out_b is None else len(out_b),'source_boundary_closed_oriented_chain':source_closed,'source_boundary_edge_failures':source_edge_fail,'output_boundary_closed_oriented_chain':closed,'output_boundary_edge_failures':edge_fail,'full_boundary_canonical_oriented_identity':boundary_identity_same,'full_boundary_rows_exact_same_order':boundary_rows_equal,'source_vs_desired_boundary_face_delta':int(len(old_b)-len(desired)),'exact_F32_sign_and_native_quality':quality_result,'elapsed_s':time.monotonic()-started}
 if out_b is not None and boundary_identity_same:
  outnpz=a.out/'skin-composed-candidate.npz'
  np.savez_compressed(outnpz,points_world_m=p,tetrahedra=out_t,material_indices=out_m,boundary_faces=out_b,desired_boundary_faces=desired,retained_input_candidate_tet_rows=kept_rows,retired_input_candidate_tet_rows=ret,tetrahedron_output_to_input_candidate_row=out_to_in,tetrahedron_output_patch_ids=out_patch,patch_global_point_rows=g,patch_output_tetrahedra_global_node_rows=new_t,patch_material_indices=new_m)
  report['candidate_path']=str(outnpz);report['candidate_sha256']=sha(outnpz)
 (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='exact_F32_sign_and_native_quality'},indent=2));print(json.dumps({'exact_F32_sign_and_native_quality':report['exact_F32_sign_and_native_quality']},indent=2))
if __name__=='__main__':main()
