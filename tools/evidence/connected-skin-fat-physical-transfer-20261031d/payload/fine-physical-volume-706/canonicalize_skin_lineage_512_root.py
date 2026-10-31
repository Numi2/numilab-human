
"""Canonical current Skin lineage after existing independently replayed connectivity operations."""
from pathlib import Path
import json,hashlib,importlib.util,time
import numpy as np
H=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
T=mod('canon512',H/'transfer_surface_contractions_indexed_379_root.py')
out=H/'skin-canonical-lineage-512-root';out.mkdir(exist_ok=False);started=time.monotonic()
rp=H/'skin-fixed-boundary-batch-508-root/receipt.json';r=json.loads(rp.read_text())
assert r['status']=='PASS_CURRENT_CONNECTED_FIXED_BOUNDARY_SKIN_BATCH'
ip=Path(r['input']);fp=Path(r['candidate'])
assert sha(ip)==r['input_sha256'] and sha(fp)==r['candidate_sha256']
a=load(ip);p=a['points_world_m'];t=a['tetrahedra'].copy();mat=a['material_indices'].copy()
owner=np.arange(len(t),dtype=np.int64);unions={}
initialt=t.copy();initialm=mat.copy();steps=[]
def apply(keep,rows,po,nt,no,nmat,label):
 global t,mat,owner,unions
 keep=np.asarray(keep,np.int64);rows=np.asarray(rows,np.int64)
 mask=np.ones(len(t),bool);mask[rows]=False
 assert np.array_equal(keep,np.flatnonzero(mask)) and len(np.unique(rows))==len(rows)
 assert len(po)==len(no) and po[0]==no[0]==0 and po[-1]==len(rows) and no[-1]==len(nt)
 repunions=[]
 for k in range(len(po)-1):
  parent=rows[po[k]:po[k+1]];child=nt[no[k]:no[k+1]];u=set()
  assert len(np.unique(mat[parent]))==1
  for j in parent:u.update([int(owner[j])] if owner[j]>=0 else unions[int(j)])
  assert u and len(u)<=len(initialt)
  assert T.m.certify(p,t[parent],child)==(True,'PASS')
  repunions.extend([tuple(sorted(u))]*len(child))
 assert len(repunions)==len(nt)
 oldunions=unions
 unions={i:oldunions[int(j)] for i,j in enumerate(keep) if owner[j]<0}
 unions.update({len(keep)+k:u for k,u in enumerate(repunions)})
 owner=np.r_[owner[keep],np.full(len(nt),-1,np.int64)]
 t=np.vstack([t[keep],nt]);mat=np.r_[mat[keep],nmat]
 assert len(owner)==len(t) and len(unions)==np.count_nonzero(owner<0)
 steps.append({'stage':label,'retained':len(keep),'replacements':len(nt),'operations':len(po)-1})
for e in r['history']:
 before=Path(e['input']);assert sha(before)==e['input_sha256']
 x=load(before);assert np.array_equal(t,x['tetrahedra']) and np.array_equal(mat,x['material_indices'])
 proof=Path(e['independent_proof']);assert sha(proof)==e['independent_proof_sha256']
 pr=json.loads(proof.read_text());assert pr['physical_admission'] is False and pr['native_steps']==0
 report=Path(e['report']);assert sha(report)==e['report_sha256']
 prop=Path(e['candidate']);assert sha(prop)==e['candidate_sha256']
 if e['stage'].endswith('-edge'):
  er=json.loads(report.read_text())
  for h in er['history']:
   dp=report.parent/('pass-%03d'%h['pass'])/'exact-connectivity-delta.npz';assert sha(dp)==h['delta_sha256']
   z=load(dp);rows=z['replacement_parent_cell_rows'];keep=np.setdiff1d(np.arange(len(t)),rows)
   apply(keep,rows,z['replacement_parent_offsets'],z['replacement_tetrahedra'],z['replacement_tet_offsets'],z['replacement_material_indices'],e['stage']+'-pass'+str(h['pass']))
 else:
  z=load(prop);rows=z['replacement_parent_cell_rows'];nt=z['replacement_tetrahedra']
  if rows.ndim==2:
   po=np.arange(len(rows)+1,dtype=np.int64)*rows.shape[1]
   no=np.arange(len(nt)+1,dtype=np.int64)*nt.shape[1];rows=rows.ravel();nt=nt.reshape(-1,4)
  else:po=z['replacement_parent_offsets'];no=z['replacement_tet_offsets']
  newmat=np.concatenate([np.full(no[k+1]-no[k],mat[rows[po[k]]],np.int32) for k in range(len(po)-1)])
  apply(z['retained_input_cell_rows'],rows,po,nt,no,newmat,e['stage'])
 x=load(prop);assert np.array_equal(t,x['tetrahedra']) and np.array_equal(mat,x['material_indices'])
 assert np.array_equal(p.view('u4'),x['points_world_m'].view('u4'))
assert np.array_equal(t,load(fp)['tetrahedra']) and np.array_equal(mat,load(fp)['material_indices'])
bd=T.boundary(t);desired=a['desired_boundary_faces']
assert np.array_equal(T.boundary_identity(bd),T.boundary_identity(desired))
d,q=T.m.measurements(p,t);assert np.all(d>0) and np.count_nonzero(~T.m.passes(d,q))==r['bad_after']==1064
assert len(np.unique(np.sort(t,axis=1),axis=0))==len(t)
newrows=np.flatnonzero(owner<0);po=[0];parents=[]
for j in newrows:
 u=unions[int(j)];assert np.all(initialm[np.asarray(u)]==mat[j])
 parents.extend(u);po.append(len(parents))
assert np.array_equal(initialt[owner[owner>=0]],t[owner>=0])
payload={'points_world_m':p,'points_world_m_f32':p.copy(),'tetrahedra':t,'material_indices':mat,'boundary_faces':bd,'desired_boundary_faces':desired,
 'current_geometry_parent_path':np.asarray([str(ip)]),'current_geometry_parent_sha256':np.asarray([sha(ip)]),
 'typed_I_candidate_path':a['typed_I_candidate_path'],'typed_I_candidate_sha256':a['typed_I_candidate_sha256'],
 'typed_O_candidate_path':a['typed_O_candidate_path'],'typed_O_candidate_sha256':a['typed_O_candidate_sha256'],
 'output_cell_to_input_cell':owner,'replacement_parent_offsets':np.asarray(po,np.int64),'replacement_parent_cell_rows':np.asarray(parents,np.int64),
 'replacement_tet_offsets':np.asarray([0,len(newrows)],np.int64),'replacement_tetrahedra':t[newrows]}
path=out/'skin-current-O506-Iflip-physical-volume-proposal.npz';np.savez_compressed(path,**payload)
x=load(path);assert set(x)==set(payload) and all(np.array_equal(x[k],v) for k,v in payload.items())
receipt={'status':'PASS_CANONICAL_SKIN_CURRENT_TARGET_AND_TRANSITIVE_CELL_ANCESTRY','physical_admission':False,'native_steps':0,'input':str(ip),'input_sha256':sha(ip),'raw_final':str(fp),'raw_final_sha256':sha(fp),'candidate':str(path),'candidate_sha256':sha(path),'nodes':len(p),'cells':len(t),'native_quality_failures':1064,'nonpositive':0,'all_coordinates_bit_identical':True,'F32_views_synchronized':True,'source330_mass_immutable':True,'complete_boundary_faces':len(bd),'missing_target_faces':0,'extra_actual_faces':0,'transitive_new_cell_parent_unions':len(newrows),'parent_rows':len(parents),'all_unrelated_cells_exact':True,'native_motion':'NOT_RUN','mass_mapping':'COMPLETE_CONSERVATIVE_REBIND_REQUIRED','steps':steps,'chain_receipt':str(rp),'chain_receipt_sha256':sha(rp),'producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-started}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
