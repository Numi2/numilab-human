
"""Canonical Skin volume from exact fixed-boundary fan chain, retaining source identities and parent unions."""
from pathlib import Path
import numpy as np,json,hashlib,importlib.util,time
H=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def load(e):
 p=Path(e['path']);assert sha(p)==e['sha256']
 with np.load(p) as z:return {k:z[k] for k in z.files}
s=importlib.util.spec_from_file_location('canon538',H/'transfer_surface_contractions_indexed_379_root.py');T=importlib.util.module_from_spec(s);s.loader.exec_module(T)
rp=H/'skin-kernel-fan-531-root/final-chain-receipt.json';r=json.loads(rp.read_text());assert r['status']=='PASS_SOURCE_ONLY_SKIN_FAN_CHAIN_WITH_ZERO_NATIVE_OR_PHYSICAL_ADMISSION'
out=H/'skin-canonical-fan-chain-538-root';out.mkdir(exist_ok=False);start=time.monotonic()
base=load(r['base_candidate']);a=load(r['best_candidate']);lin=load(r['parent_cell_point_lineage']);v=load(r['both_point_coordinate_views'])
p=a['points_world_m'];t=a['tetrahedra'];mat=a['material_indices'];mp=lin['output_cell_to_skin531_input_cell_row'];po=lin['output_cell_parent_offsets'];pr=lin['output_cell_parent_skin531_rows'];pointmap=lin['output_point_to_skin531_input_point_row']
assert len(p)==210298 and len(t)==729953 and len(mp)==len(t)
assert np.array_equal(p[:len(base['points_world_m'])].view('u4'),base['points_world_m'].view('u4'))
assert np.array_equal(p.view('u4'),v['points_world_m'].view('u4')) and np.array_equal(p.view('u4'),v['points_world_m_f32'].view('u4'))
assert po.shape==(len(t)+1,) and po[0]==0 and po[-1]==len(pr) and np.all(np.diff(po)>0)
assert np.array_equal(t[mp>=0],base['tetrahedra'][mp[mp>=0]]) and np.array_equal(mat[mp>=0],base['material_indices'][mp[mp>=0]])
rows=np.flatnonzero(mp<0);off=[0];parents=[]
for row in rows:
 u=pr[po[row]:po[row+1]];assert len(u)==len(np.unique(u)) and np.all(base['material_indices'][u]==mat[row]);parents.extend(map(int,u));off.append(len(parents))
assert pointmap.shape==(len(p),) and np.array_equal(pointmap[:len(base['points_world_m'])],np.arange(len(base['points_world_m']))) and np.all(pointmap[len(base['points_world_m']):]==-1)
qpo=lin['output_point_parent_offsets'];qpr=lin['output_point_parent_skin531_cell_rows']
assert qpo.shape==(len(p)+1,) and qpo[0]==0 and qpo[-1]==len(qpr)
assert np.all(np.diff(qpo)[pointmap<0]>0)
bd=T.boundary(t);assert np.array_equal(T.boundary_identity(bd),T.boundary_identity(base['desired_boundary_faces']))
d,q=T.m.measurements(p,t);assert np.all(d>0) and int((~T.m.passes(d,q)).sum())==988
payload=dict(a);payload.update(points_world_m_f32=p.copy(),typed_I_candidate_path=base['typed_I_candidate_path'],typed_I_candidate_sha256=base['typed_I_candidate_sha256'],typed_O_candidate_path=base['typed_O_candidate_path'],typed_O_candidate_sha256=base['typed_O_candidate_sha256'],current_geometry_parent_path=np.asarray([r['base_candidate']['path']]),current_geometry_parent_sha256=np.asarray([r['base_candidate']['sha256']]),output_cell_to_input_cell=mp,replacement_parent_offsets=np.asarray(off,np.int64),replacement_parent_cell_rows=np.asarray(parents,np.int64),replacement_tet_offsets=np.asarray([0,len(rows)],np.int64),replacement_tetrahedra=t[rows],retained_input_cell_rows=mp[mp>=0],output_point_to_input_point_row=pointmap,output_point_parent_offsets=qpo,output_point_parent_input_cell_rows=qpr)
cp=out/'skin-current-O530-I528-fan-canonical-volume-proposal.npz';np.savez_compressed(cp,**payload)
b=load(pin(cp));assert set(b)==set(payload) and all(np.array_equal(b[k],x) for k,x in payload.items())
res={'status':'PASS_CANONICAL_FIXED_BOUNDARY_SKIN_FAN_CURRENT_VOLUME_AND_ANCESTRY','physical_admission':False,'native_steps':0,'candidate':str(cp),'candidate_sha256':sha(cp),'input':r['base_candidate'],'chain_receipt':pin(rp),'nodes':len(p),'cells':len(t),'native_quality_failures':988,'nonpositive':0,'boundary_faces':len(bd),'missing_target_faces':0,'extra_actual_faces':0,'all_prior_coordinates_bit_identical':True,'all_point_coordinate_views_synchronized':True,'inferred_points_added':len(p)-len(base['points_world_m']),'new_cell_parent_unions':len(rows),'parent_rows':len(parents),'exact_geometric_volume_delta':0,'source330_mass':'IMMUTABLE_COMPLETE_CURRENT_REBIND_REQUIRED','source_surfaces_unchanged':True,'typed_I':{'path':str(base['typed_I_candidate_path'][0]),'sha256':str(base['typed_I_candidate_sha256'][0])},'typed_O':{'path':str(base['typed_O_candidate_path'][0]),'sha256':str(base['typed_O_candidate_sha256'][0])},'producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start}
(out/'receipt.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res),flush=True)
