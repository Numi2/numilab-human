"""Reapply existing202 interior-star repairs to current Skin; independently replay exact cavities."""
from pathlib import Path
import json,hashlib,importlib.util,types,time,numpy as np
H=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def module(name,rel):
 s=importlib.util.spec_from_file_location(name,H/rel);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
owner=module('current_202','fixed_boundary_volume_polygon_flip_202_root.py')
v=module('boundary_379','transfer_surface_contractions_indexed_379_root.py')
inp=H/'paired-I-interface-399-root/iteration002/skin-target369/skin-bound-current-I-target-volume-proposal.npz'
assert sha(inp)=='78ad7f990c400c14106a3336d7364b861c3bf1af4b0b11d21ae368a815ca59d2'
out=H/'current-skin-fixed-boundary-421-root';out.mkdir(exist_ok=False);start=time.monotonic()
with np.load(inp) as z:base={k:z[k] for k in z.files}
p=base['points_world_m'];t=base['tetrahedra'];mat=base['material_indices']
baseb=v.boundary_identity(v.boundary(t));assert np.array_equal(baseb,v.boundary_identity(base['desired_boundary_faces']))
hist=[];current=inp
for i in range(2):
 d=out/('pass-%03d'%i)
 owner.run(types.SimpleNamespace(input_npz=current,output=d,seconds=35,maximum_degree=8,maximum_operations=10000))
 rp=d/'report.json';r=json.loads(rp.read_text());raw=d/'fixed-boundary-volume-proposal.npz'
 with np.load(raw) as z:a={k:z[k] for k in z.files}
 assert np.array_equal(p.view(np.uint32),a['points_world_m'].view(np.uint32))
 oldrows=a['replacement_parent_cell_rows'];offs=a['replacement_parent_offsets'];new=a['replacement_tetrahedra'];no=a['replacement_tet_offsets']
 assert len(offs)==len(no)==r['passing_operations']+1
 assert len(np.unique(oldrows))==len(oldrows) and offs[0]==no[0]==0 and offs[-1]==len(oldrows) and no[-1]==len(new)
 seen=set();newmat=[]
 for j in range(len(offs)-1):
  rows=oldrows[offs[j]:offs[j+1]];nt=new[no[j]:no[j+1]];ot=t[rows]
  ns=set(map(int,ot.ravel()));assert not ns&seen;seen.update(ns)
  assert len(np.unique(mat[rows]))==1
  assert owner.m.certify(p,ot,nt)==(True,'PASS')
  assert np.count_nonzero(~owner.m.passes(*owner.m.measurements(p,nt)))<np.count_nonzero(~owner.m.passes(*owner.m.measurements(p,ot)))
  newmat.extend([int(mat[rows[0]])]*len(nt))
 mask=np.ones(len(t),bool);mask[oldrows]=False
 assert np.array_equal(a['retained_input_cell_rows'],np.flatnonzero(mask))
 assert np.array_equal(a['tetrahedra'],np.vstack((t[mask],new)))
 assert np.array_equal(a['material_indices'],np.r_[mat[mask],np.asarray(newmat,np.int32)])
 nt=a['tetrahedra'];nm=a['material_indices']
 assert np.array_equal(baseb,v.boundary_identity(v.boundary(nt)))
 keys=np.sort(nt,axis=1);idx=np.lexsort(tuple(keys[:,k] for k in (3,2,1,0)))
 assert not np.any(np.all(keys[idx][1:]==keys[idx][:-1],axis=1))
 det,quality=owner.m.measurements(p,nt);assert np.all(det>0)
 h={'pass':i,'source_candidate':str(current),'source_sha256':sha(current),'candidate':str(raw),'candidate_sha256':sha(raw),'report':str(rp),'report_sha256':sha(rp),'exact_local_replay':'PASS','full_oriented_target_boundary':'PASS','duplicate_tets':0,'operations':r['passing_operations'],'bad_before':r['bad_cells_before'],'bad_after':r['bad_cells_after']}
 hist.append(h);t,mat,current=nt,nm,raw
 if r['passing_operations']==0:break
# Preserve complete current target and authenticated interface identities; the replacement's cell ancestry is its compact ordered delta chain.
a.update(boundary_faces=base['boundary_faces'],desired_boundary_faces=base['desired_boundary_faces'])
for k in ['typed_I_candidate_path','typed_I_candidate_sha256','source_I_vertex_quotient','source_interface_collapse_rows','solver_infeasible_cavity_sha256']:
 a[k]=base[k]
a['current_geometry_parent_path']=np.asarray([str(inp)])
a['current_geometry_parent_sha256']=np.asarray([sha(inp)])
final=out/'skin-current-fixed-boundary-volume-proposal.npz'
np.savez_compressed(final,**a)
r={'status':'PASS_CURRENT_SKIN_FIXED_BOUNDARY_EXACT_CONNECTIVITY_REPLAY','physical_admission':False,'native_steps':0,'source':str(inp),'source_sha256':sha(inp),'candidate':str(final),'candidate_sha256':sha(final),'coordinates_unchanged':True,'complete_oriented_target_boundary_unchanged':True,'boundary_faces':len(baseb),'material_and_exact_volume_preserved':True,'quality_before':1513,'quality_after':hist[-1]['bad_after'],'operations':sum(x['operations'] for x in hist),'history':hist,'producer_sha256':sha(__file__),'operator_sha256':sha(H/'fixed_boundary_volume_polygon_flip_202_root.py'),'exact_predicate_sha256':sha(H/'fixed_boundary_volume_flip_198_root.py'),'elapsed_s':time.monotonic()-start,'mass_mapping':'NEW_CELL_MASS_MAPPING_REQUIRED','limitation':'Positive construction; residual native-quality failures are not admitted.'}
(out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
