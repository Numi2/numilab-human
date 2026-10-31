#!/Users/numi/numi-human-prep-venv-20261005/bin/python
from pathlib import Path
import hashlib,importlib.util,json,sys,time
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
CAND=H/'skin-kernel-fan-531-root/run003-corrected/fan-output/kernel-fan-volume-proposal.npz'
CAND_SHA='055e2177e208187002a4a45ad12bada98111b1f1ba12e2b8d473935946aab5f3'
LINEAGE=H/'skin-kernel-fan-531-root/run003-corrected/fan-output/current531-row-lineage.npz'
LINEAGE_SHA='939b23184eef42c70c973959263375548599b41d4a8df8ee13d76bbb99695287'
VIEWS=H/'skin-kernel-fan-531-root/run003-corrected/canonical531-point-coordinate-views.npz'
VIEWS_SHA='b31e428609fac468258c2ee9880b4adee9b36c07e96238a17fb3760d7119ea20'
O=H/'outer-O-connected-quality-530-root/candidate/typed434001/candidate-O-typed-current.npz'; O_SHA='2f98f16f06baaefada41d4577594923ff5e9019a84a5e1c7f6cb44a1171f8901'
I=H/'paired-I-current-protected-528-root/typed463001/candidate-I-typed-current.npz'; I_SHA='6f59262c2ae70065576166dd84f767afb64352241e76009707f78447afe477d1'
HELPER=H/'remesh_cached_joint_witness_fans_297_root.py'; HELPER_SHA='f1b9f76cc2621fc5bc60d217486cfe511023c5c893cec3d97f332992074036e1'
OUT=H/'skin-kernel-fan-531-root/run003-corrected/quality-classification-002'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(n,p):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);sys.modules[n]=m;spec.loader.exec_module(m);return m
def canon_keys(f):
 return np.sort(np.asarray(f,np.int64),axis=1)
def main():
 tic=time.monotonic()
 for p,h in ((CAND,CAND_SHA),(LINEAGE,LINEAGE_SHA),(VIEWS,VIEWS_SHA),(O,O_SHA),(I,I_SHA),(HELPER,HELPER_SHA)):
  if sha(p)!=h:raise ValueError(f'hash mismatch {p}')
 mod=load('quality_owner_297',HELPER)
 OUT.mkdir(parents=True,exist_ok=False)
 with np.load(CAND,allow_pickle=False) as z:
  p=z['points_world_m'];t=z['tetrahedra'];mat=z['material_indices'];bd=z['boundary_faces'];want=z['desired_boundary_faces']
 with np.load(VIEWS,allow_pickle=False) as z:
  p1=z['points_world_m'];p2=z['points_world_m_f32']
 if p.dtype!=np.float32 or p1.dtype!=np.float32 or p2.dtype!=np.float32 or not np.array_equal(p.view(np.uint32),p1.view(np.uint32)) or not np.array_equal(p.view(np.uint32),p2.view(np.uint32)):
  raise ValueError('two point-coordinate views do not match candidate bits')
 if len(t)!=729953 or len(p)!=210298 or mat.shape!=(len(t),):raise ValueError('final candidate domain mismatch')
 if not np.array_equal(mod.m.boundary_identity(bd),mod.m.boundary_identity(want)) or not np.array_equal(mod.m.boundary_identity(bd),mod.m.boundary_identity(mod.m.boundary(t))):raise ValueError('candidate boundary no longer matches target')
 with np.load(O,allow_pickle=False) as z:op=z['points_world_m_f32'];of=z['faces']
 with np.load(I,allow_pickle=False) as z:ip=z['points_world_m_f32'];inf=z['faces']
 coord={}
 for row,x in enumerate(p):
  k=tuple(map(int,x.view(np.uint32)))
  if k in coord:coord[k]=-1
  else:coord[k]=row
 def remap_faces(sp,sf,label):
  used=np.unique(sf)
  mp=np.full(len(sp),-1,np.int64)
  misses=[];ambiguous=[]
  for q in used:
   k=tuple(map(int,sp[int(q)].view(np.uint32)));v=coord.get(k)
   if v is None:misses.append(int(q))
   elif v<0:ambiguous.append(int(q))
   else:mp[int(q)]=v
  if misses or ambiguous:raise ValueError(f'{label} surface point map missing={misses[:5]} duplicate_coordinate={ambiguous[:5]}')
  out=mp[sf]
  if np.any(out<0):raise ValueError(f'{label} mapped face has invalid node')
  return out
 ofv=remap_faces(op,of,'O');infv=remap_faces(ip,inf,'I')
 okeys=canon_keys(ofv); ikeys=canon_keys(infv); dkeys=canon_keys(want)
 def lexorder(a):return np.lexsort((a[:,2],a[:,1],a[:,0]))
 dsort=dkeys[lexorder(dkeys)];osort=okeys[lexorder(okeys)];isort=ikeys[lexorder(ikeys)]
 if len(np.unique(osort,axis=0))!=len(osort) or len(np.unique(isort,axis=0))!=len(isort):raise ValueError('typed O/I has duplicate faces')
 # Verify each role face belongs to the complete current Skin target by exact node IDs.
 def contains_sorted(hay,need):
  order=lexorder(hay);h=hay[order];q=lexorder(need);n=need[q]
  # Structured scalar keys permit lexicographic binary search without a face dictionary.
  dt=np.dtype([('a','<i8'),('b','<i8'),('c','<i8')]); hv=h.view(dt).reshape(-1); nv=n.view(dt).reshape(-1)
  ix=np.searchsorted(hv,nv); ok=ix<len(hv);ok[ok]=hv[ix[ok]]==nv[ok]
  ret=np.zeros(len(need),bool);ret[q]=ok;return ret
 if not np.all(contains_sorted(dkeys,okeys)) or not np.all(contains_sorted(dkeys,ikeys)):raise ValueError('typed O/I faces not covered by current Skin target')
 # Locate exactly one owning tet for every typed boundary triangle by joining against all 4 faces/tet.
 pat=np.asarray(mod.m.PAT,np.int64)
 allfaces=t[:,pat].reshape(-1,3);allkeys=canon_keys(allfaces)
 allowners=np.repeat(np.arange(len(t),dtype=np.int64),len(pat))
 omask=contains_sorted(dkeys,okeys);imask=contains_sorted(dkeys,ikeys)
 wallkeys=dkeys[~(omask|imask)]
 if len(wallkeys)!=90:raise ValueError(f'expected 90 rim wall faces, found {len(wallkeys)}')
 target=np.vstack([okeys,ikeys,wallkeys]);target_ids=np.arange(len(target),dtype=np.int64)
 joined=np.vstack([allkeys,target]); owners=np.concatenate([allowners,np.full(len(target),-1,np.int64)])
 target_idx=np.concatenate([np.full(len(allkeys),-1,np.int64),target_ids])
 order=lexorder(joined);jk=joined[order];jo=owners[order];jt=target_idx[order]
 eq=np.all(jk[1:]==jk[:-1],axis=1)
 pair=np.flatnonzero(eq)
 hits=np.full(len(target),-1,np.int64);hitcount=np.zeros(len(target),np.int8)
 l=order[pair];r=order[pair+1]
 left_target=target_idx[l]>=0;right_target=target_idx[r]>=0
 cross=left_target^right_target
 li=l[cross];ri=r[cross]
 ti=np.where(target_idx[li]>=0,target_idx[li],target_idx[ri]);oi=np.where(owners[li]>=0,owners[li],owners[ri])
 hits[ti]=oi;np.add.at(hitcount,ti,1)
 if np.any(hitcount!=1):raise ValueError(f'typed target face owner multiplicity invalid: {np.unique(hitcount,return_counts=True)}')
 oowner=hits[:len(okeys)];iowner=hits[len(okeys):len(okeys)+len(ikeys)];wowner=hits[len(okeys)+len(ikeys):]
 # The exact current native predicate is reused once over the frozen final cells.
 det,q=mod.m.m.measurements(p,t);passes=mod.m.m.passes(det,q);bad=np.flatnonzero(~passes).astype(np.int64)
 bad_o=np.zeros(len(t),bool);bad_i=np.zeros(len(t),bool);bad_w=np.zeros(len(t),bool)
 bad_o[np.unique(oowner)]=True;bad_i[np.unique(iowner)]=True;bad_w[np.unique(wowner)]=True
 cat=bad_o.astype(np.uint8)|(bad_i.astype(np.uint8)<<1)|(bad_w.astype(np.uint8)<<2)
 # Bitmask: O=1, I=2, 90-face rim wall=4; 0 is interior.
 bad_cat=cat[bad]
 counts={str(k):int(np.count_nonzero(bad_cat==k)) for k in range(8)}
 with np.load(LINEAGE,allow_pickle=False) as z:
  po=z['output_cell_parent_offsets'];pr=z['output_cell_parent_skin531_rows']
  if len(po)!=len(t)+1:raise ValueError('canonical531 cell-parent CSR does not cover output')
 parent_samples={}
 for code,name in ((0,'interior_no_O_I_or_wall'),(1,'O_only'),(2,'I_only'),(3,'O_and_I'),(4,'rim_wall_only'),(5,'O_and_wall'),(6,'I_and_wall'),(7,'O_I_and_wall')):
  rows=bad[bad_cat==code][:3]
  parent_samples[name]=[{'output_cell_row':int(r),'canonical531_parent_rows':pr[int(po[r]):int(po[r+1])].tolist()} for r in rows]
 np.savez_compressed(OUT/'bad-cell-classification.npz',bad_output_cell_rows=bad,bad_category_codes=bad_cat,typed_O_owner_cell_rows=oowner,typed_I_owner_cell_rows=iowner,typed_wall_owner_cell_rows=wowner)
 report={'schema':'numi.skin.quality-failure-surface-classification.531.v1','status':'PASS_TYPED_O_I_QUALITY_CENSUS','physical_admission':False,'native_steps':0,'candidate':str(CAND),'candidate_sha256':CAND_SHA,'candidate_cells':int(len(t)),'candidate_points':int(len(p)),'bad_cell_count':int(len(bad)),'typed_O_candidate':str(O),'typed_O_sha256':O_SHA,'typed_O_face_count':int(len(of)),'typed_I_candidate':str(I),'typed_I_sha256':I_SHA,'typed_I_face_count':int(len(inf)),'typed_surface_faces_with_unique_tet_owner':int(len(target)),'rim_wall_face_count':int(len(wallkeys)),'actual_boundary_vs_desired_oriented_identity':'PASS','typed_O_I_faces_covered_by_current_skin_target':'PASS','bad_cells_by_surface_incidence':counts,'category_definitions':{'0':'interior; no O, I, or rim-wall face','1':'O only','2':'I only','3':'O and I','4':'rim wall only','5':'O and rim wall','6':'I and rim wall','7':'O, I, and rim wall'},'bad_cell_examples_with_canonical531_parent_union':parent_samples,'point_views_sidecar':str(VIEWS),'point_views_sha256':VIEWS_SHA,'point_views_bit_identical':True,'lineage_sidecar':str(LINEAGE),'lineage_sha256':LINEAGE_SHA,'helper_sha256':HELPER_SHA,'classification_npz':str(OUT/'bad-cell-classification.npz'),'classification_npz_sha256':sha(OUT/'bad-cell-classification.npz'),'quality_predicate':'exact existing 297 owner m.m.measurements + m.m.passes on final candidate','limitation':'Diagnostic classification only; no mass/native/physical admission. O/I incidence is by exact F32 coordinate crosswalk and exact typed face owner cell. Category 0 includes interior and rim-wall cells.' ,'elapsed_s':time.monotonic()-tic}
 (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 report['report_sha256']=sha(OUT/'report.json');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
