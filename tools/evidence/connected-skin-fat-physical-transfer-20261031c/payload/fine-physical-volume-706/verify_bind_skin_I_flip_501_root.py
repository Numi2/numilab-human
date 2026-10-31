
from pathlib import Path
import numpy as np,json,hashlib,importlib.util,time
H=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
T=mod('verify501',H/'transfer_surface_contractions_indexed_379_root.py');R=mod('exact501',H/'interior-quality-conditioning-309-banks/condition_registered_interior_309.py')
out=H/'skin-I-flip-independent-501-root';out.mkdir(exist_ok=False);start=time.monotonic()
root=H/'skin-paired-I-flip-target-498-root';bind=root/'target-bind-report.json';br=json.loads(bind.read_text())
rr=root/'remesh499-S0/report.json';r=json.loads(rr.read_text())
assert r['script_sha256']==sha(H/'remesh_authenticated_skin_I_flip_499_root.py')
ip=Path(r['input']);raw=Path(r['output'])
assert sha(ip)==r['input_sha256']==br['candidate_sha256'] and sha(raw)==r['output_sha256']=='469a36543d9de898330366d109de1557446efeadf5b0e6b9ef27371f77416274'
a=load(ip);b=load(raw);p=a['points_world_m'];t=a['tetrahedra']
assert np.array_equal(p.view(np.uint32),b['points_world_m'].view(np.uint32))
assert len(r['components'])==r['passing_components']==1 and r['complete_boundary_matches_desired']
c=r['components'][0];assert c['status']=='PASS_CONNECTED_FIXED_SURFACE_CAVITY' and c['new_interior_points']==0
rep=root/'remesh499-S0'/('component-%04d'%c['id'])/'replacement.npz'
if not rep.exists():
 matches=list((root/'remesh499-S0').glob('*/replacement.npz'))
 assert len(matches)==1;rep=matches[0]
assert sha(rep)==c['replacement_sha256'];x=load(rep);rows=x['parent_cell_rows'];nt=x['tetrahedra']
keep=np.ones(len(t),bool);keep[rows]=False
assert np.array_equal(np.flatnonzero(keep),b['retained_input_cell_rows'])
expected=np.vstack([t[keep],nt]);assert np.array_equal(expected,b['tetrahedra'])
mat=np.r_[a['material_indices'][keep],np.full(len(nt),15,np.int32)]
assert np.array_equal(mat,b['material_indices']) and np.all(mat==15)
assert len(np.unique(np.sort(expected,axis=1),axis=0))==len(expected)
d,q=T.m.measurements(p,expected);assert np.all(d>0) and int((~T.m.passes(d,q)).sum())==1111
positive=R.exact_star_determinants(p,nt);assert all(x>0 for x in positive)
boundary=T.boundary(expected)
assert np.array_equal(T.boundary_identity(boundary),T.boundary_identity(b['boundary_faces']))
assert np.array_equal(T.boundary_identity(boundary),T.boundary_identity(a['desired_boundary_faces']))
oldsum=sum(R.exact_star_determinants(p,t[rows]));newsum=sum(positive)
assert c['oriented_boundary_exact'] and c['exact_new_shell_volume'] and c['exact_internal_face_pairs']==0 and c['exact_retained_and_prior_volume_intersections']['pass']
# Preserve accepted row ownership and declare complete per-replacement source ancestry.
parent_offsets=np.arange(len(nt)+1,dtype=np.int64)*len(rows)
b.update(points_world_m_f32=p.copy(),current_geometry_parent_path=np.asarray([str(ip)]),current_geometry_parent_sha256=np.asarray([sha(ip)]),
 typed_I_candidate_path=a['typed_I_candidate_path'],typed_I_candidate_sha256=a['typed_I_candidate_sha256'],
 typed_O_candidate_path=a['typed_O_candidate_path'],typed_O_candidate_sha256=a['typed_O_candidate_sha256'],
 output_cell_to_input_cell=np.r_[np.flatnonzero(keep),np.full(len(nt),-1,np.int64)],
 replacement_parent_offsets=parent_offsets,replacement_parent_cell_rows=np.tile(rows,len(nt)),
 replacement_tet_offsets=np.asarray([0,len(nt)],np.int64),replacement_tetrahedra=nt.copy())
canonical=out/'skin-current-O484-Iflip-physical-volume-proposal.npz';np.savez_compressed(canonical,**b)
with np.load(canonical) as z:
 for k in b:assert np.array_equal(z[k],b[k])
 assert np.array_equal(z['points_world_m'],z['points_world_m_f32'])
receipt={'status':'PASS_INDEPENDENT_PAIRED_I_FLIP_SKIN_VOLUME_TRANSFER','physical_admission':False,'native_steps':0,'input':str(ip),'input_sha256':sha(ip),'target_bind':str(bind),'target_bind_sha256':sha(bind),'remesh_report':str(rr),'remesh_report_sha256':sha(rr),'raw_candidate':str(raw),'raw_candidate_sha256':sha(raw),'candidate':str(canonical),'candidate_sha256':sha(canonical),'old_source_cell_rows':rows.tolist(),'replacement_cells':len(nt),'all_point_coordinates_bit_identical':True,'all_materials':15,'all_unrelated_cells_preserved':True,'exact_new_positive_signs':len(positive),'exact_signed_integer_det6_volume_change':str(newsum-oldsum),'signed_geometric_volume_change_m3':float(newsum-oldsum)*2.**-447/6,'complete_current_target_faces':len(boundary),'missing_target_faces':0,'extra_actual_faces':0,'native_quality_failures':1111,'source330_mass':'IMMUTABLE_CURRENT_CONSERVATIVE_REBIND_REQUIRED','parent_mapping':'new_cells carry complete original two-cell parent union; full source330 mass allocation pending','producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
