
"""Canonical paired-I Skin geometry with immediate source-cell union ancestry."""
from pathlib import Path
import numpy as np,json,hashlib,importlib.util,time
H=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def npz(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
s=importlib.util.spec_from_file_location('canon529',H/'transfer_surface_contractions_indexed_379_root.py');T=importlib.util.module_from_spec(s);s.loader.exec_module(T)
start=time.monotonic();out=H/'skin-canonical-outer-O-531-root';out.mkdir(exist_ok=False)
base=H/'outer-O-connected-quality-530-root/candidate';r=json.loads((base/'run001/report.json').read_text())
assert r['passing_operations']==14 and r['status']=='PASS_BOUNDED_OUTER_O_QUOTIENT_PHYSICAL_PREFLIGHT'
rr=next(x for x in r['roles'] if x['role']=='Skin');ip=Path(rr['input']);assert sha(ip)==rr['input_sha256']
pr=json.loads((base/'typed434001/receipt.json').read_text());assert pr['status']=='PASS_BOUNDED_OUTER_O_TYPED_ANCESTRY_AND_INDEPENDENT_VOLUME_REPLAY'
br=json.loads((base/'qualification435001/skin-target437/receipt.json').read_text());assert br['status']=='PASS_EXACT_BOUNDED_O_QUOTIENT_PHYSICAL_TARGET_BIND_ONLY'
bp=Path(br['candidate']);assert sha(bp)==br['candidate_sha256'];old=npz(ip);bound=npz(bp)
p=old['points_world_m'];ot=old['tetrahedra'];om=old['material_indices'];nt=ot.copy();active=np.ones(len(ot),bool);owner=np.full(len(ot),-1,np.int64);seen=set()
for k,op in enumerate(rr['operations']):
 rows=np.asarray(op['source_cell_rows'],np.int64);assert len(rows)==len(np.unique(rows));assert not seen.intersection(rows);seen.update(map(int,rows))
 assert np.array_equal(rows,np.flatnonzero(np.any(ot==op['retired_volume_node'],axis=1)))
 assert len(np.unique(om[rows]))==1
 nt[rows]=np.where(ot[rows]==op['retired_volume_node'],op['kept_volume_node'],ot[rows])
 degen=np.any(np.diff(np.sort(nt[rows],axis=1),axis=1)==0,axis=1);assert degen.sum()==op['removed_cells']
 active[rows[degen]]=False;owner[rows]=k
keep=np.flatnonzero(active);t=nt[keep];mat=om[keep]
assert np.array_equal(t,bound['tetrahedra']) and np.array_equal(mat,bound['material_indices'])
assert np.array_equal(p.view('u4'),bound['points_world_m'].view('u4')) and np.array_equal(p.view('u4'),bound['points_world_m_f32'].view('u4'))
changed=owner[keep]>=0;mapping=keep.copy();mapping[changed]=-1;offsets=[0];parents=[]
for j in keep[changed]:
 rows=rr['operations'][int(owner[j])]['source_cell_rows'];parents.extend(rows);offsets.append(len(parents))
assert np.array_equal(ot[mapping[~changed]],t[~changed])
d,q=T.m.measurements(p,t);assert np.all(d>0) and int((~T.m.passes(d,q)).sum())==1024
bd=T.boundary(t);assert np.array_equal(T.boundary_identity(bd),T.boundary_identity(bound['desired_boundary_faces']))
payload={k:bound[k] for k in ['points_world_m','points_world_m_f32','tetrahedra','material_indices','boundary_faces','desired_boundary_faces','typed_I_candidate_path','typed_I_candidate_sha256','typed_O_candidate_path','typed_O_candidate_sha256']}
payload.update(current_geometry_parent_path=np.asarray([str(ip)]),current_geometry_parent_sha256=np.asarray([sha(ip)]),output_cell_to_input_cell=mapping,replacement_parent_offsets=np.asarray(offsets,np.int64),replacement_parent_cell_rows=np.asarray(parents,np.int64),replacement_tet_offsets=np.asarray([0,int(changed.sum())],np.int64),replacement_tetrahedra=t[changed],retained_input_cell_rows=keep,source_I_vertex_quotient=bound['source_I_vertex_quotient'],source_interface_collapse_rows=bound['source_interface_collapse_rows'],source_O_vertex_quotient=bound['source_O_vertex_quotient'])
cp=out/'skin-current-O530-I528-canonical-volume-proposal.npz';np.savez_compressed(cp,**payload)
x=npz(cp);assert set(x)==set(payload) and all(np.array_equal(x[k],v) for k,v in payload.items())
roleproof=next(x for x in pr['roles'] if x['role']=='Skin') if 'roles' in pr else None
receipt={'status':'PASS_CANONICAL_OUTER_O_SKIN_PARENT_UNIONS_AND_TARGET','physical_admission':False,'native_steps':0,'input':str(ip),'input_sha256':sha(ip),'bound_candidate':str(bp),'bound_candidate_sha256':sha(bp),'candidate':str(cp),'candidate_sha256':sha(cp),'nodes':len(p),'cells':len(t),'native_quality_failures':1024,'nonpositive':0,'boundary_faces':len(bd),'missing_target_faces':0,'extra_actual_faces':0,'all_coordinates_bit_identical':True,'all_point_coordinate_views_synchronized':True,'all_unrelated_cell_bytes_and_ancestry_retained':True,'new_cell_parent_unions':int(changed.sum()),'parent_rows':len(parents),'paired_typed_replay_path':str(base/'typed434001/receipt.json'),'paired_typed_replay_sha256':sha(base/'typed434001/receipt.json'),'per_layer_geometric_volume':'BOUNDED_DECLARED_O_QUOTIENT_DELTA_IN_TYPED479_REPLAY; SOURCE330_MASS_UNCHANGED','source_geometry_only':'All coordinates fixed; O connectivity changed through bounded authenticated contractions. Native gates unchanged.','source330_mass':'IMMUTABLE_CURRENT_MAPPING_REQUIRED','source_context_path':str(base/'qualification435001/audit354/report.json'),'source_context_sha256':sha(base/'qualification435001/audit354/report.json'),'producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
