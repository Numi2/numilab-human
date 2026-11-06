from pathlib import Path
import sys,json,hashlib,collections
import numpy as np
w=Path('/Users/numi/Work/human-integrated-gaps-20261104');h=Path('/Users/numi/Work/human-checkpoint-owner-20261105');o=Path(__file__).resolve().parent
sys.path.insert(0,str(h/'tools'))
import simplify_human_skin_fat_boundary as mesh
from rebuild_human_interface_caps import exact_determinants
from build_human_conditioned_cavities import spectral_condition
r=json.loads((o/'proposal/report.json').read_text());v=Path(r['input']['volume']['path'])
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(v)==r['input']['volume']['sha256']
with np.load(v) as z:p=z['points_reference_world_m_f64'];t=z['volume_tetrahedra'];labels=np.column_stack([z[k] for k in ['material_index','geometric_donor_body','anatomical_region_id']])
path=o/'proposal/fixed-boundary-volume-proposal.npz';assert sha(path)==r['output']['sha256']
with np.load(path) as z:payload={k:z[k] for k in z.files}
assert str(payload['source_volume_sha256'].item())==sha(v)
retired=[];new=[];newlabels=[];offset=0;used=set();ratios=[]
for patch in r['patches']:
 rows=np.array(patch['old_rows']);cells=np.array(patch['new_cells']);old=t[rows]
 assert 2<=len(rows)<=32 and 2<=len(cells)<=64 and np.all(labels[rows]==labels[rows[0]])
 nodes=set(map(int,old.ravel()));assert not nodes&used;used|=nodes
 assert np.array_equal(np.unique(old),np.unique(cells)) and mesh.unchanged_cavity_boundary(old,cells)
 assert np.array_equal(payload['replacement_tetrahedra'][offset:offset+len(cells)],cells)
 for k in range(offset,offset+len(cells)):
  a,b=payload['replacement_parent_offsets'][k:k+2]
  assert np.array_equal(payload['replacement_parent_cell_rows'][a:b],rows)
 ids,inverse=np.unique(np.r_[old.ravel(),cells.ravel()],return_inverse=True)
 integer=exact_determinants(p[ids],inverse.reshape(-1,4));before=integer[:len(old)];after=integer[len(old):]
 assert min(before)>0 and min(after)>0 and sum(before)==sum(after)
 det,q=mesh.measure(p,cells);oldq=mesh.measure(p,old)[1]
 assert det.min()/6>1e-18 and q.min()>=1e-5 and q.min()>=oldq.min()*1.02
 a=spectral_condition(p,old);b=spectral_condition(p,cells);assert b.max()<=a.max()
 ratios.append(float(a.max()/b.max()));retired.extend(rows.tolist());new.extend(cells.tolist());newlabels.extend([labels[rows[0]]]*len(cells));offset+=len(cells)
assert offset==len(payload['replacement_tetrahedra']) and len(set(retired))==len(retired)
keep=np.ones(len(t),bool);keep[retired]=False;assert np.array_equal(payload['retained_input_cell_rows'],np.flatnonzero(keep))
final=np.concatenate([t[keep],np.array(new)]);finallabels=np.concatenate([labels[keep],np.array(newlabels)])
key=np.sort(final.astype(np.int64),axis=1);key=np.ascontiguousarray(key).view(np.dtype((np.void,32))).reshape(-1);assert len(np.unique(key))==len(key)
u,reference_receipt=mesh.load_reference_deformation(Path(r['input']['reference']['path']),r['input']['reference']['sha256'],p,t,labels)
direct=mesh.ReferenceDeformationGate(p,final,u,finallabels[:,1],{})
for region,values in direct.base.items():
 original=np.array(r['reference_deformation']['baseline'][str(region)])
 assert np.all(np.abs(values-original)<=.05*np.abs(original)),region
 np.testing.assert_allclose(values,r['reference_deformation']['accepted'][str(region)],rtol=1e-12,atol=1e-30)
result=dict(status='PASS_INDEPENDENT_EXACT_CAVITY_CSR_AND_FULL_REBUILT_REFERENCE_STRAIN',source_sha256=sha(v),proposal_sha256=sha(path),cavities=len(r['patches']),source_cells=len(t),replacement_mesh_cells=len(final),exact_boundary_and_integer_lattice_volumes=True,all_points_unchanged=True,no_duplicate_tetrahedra=True,complete_parent_ancestry_and_construction_labels=True,minimum_rebuilt_cell_quality=min(x['minimum_native_quality_after'] for x in r['patches']),median_local_condition_ratio=float(np.median(ratios)),physical_admission=False,native_steps=0,remaining=['conservative source-mass owner','fresh complete native force/strain comparison','sustained coupled motion and presentation'])
(o/'independent-checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
