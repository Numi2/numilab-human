"""Independent replay of existing333 fixed-boundary refinement output."""
from pathlib import Path
import numpy as np,json,hashlib,importlib.util,time
H=Path(__file__).resolve().parent
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
M=load('cert198refinement376',H/'fixed_boundary_volume_flip_198_root.py')
B=load('boundary376',H/'verify_face_flips_generic_374_root.py')
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
start=time.monotonic()
root=H/'reconstructed-cell-refinement-375-root/skin-run001'
r=json.loads((root/'report.json').read_text())
inp=Path(r['input']);candidate=Path(r['output'])
assert sha(inp)==r['input_sha256'] and sha(candidate)==r['output_sha256']
with np.load(inp) as z:p=z['points_world_m'];t=z['tetrahedra'];mat=z['material_indices'];target=z['desired_boundary_faces']
with np.load(candidate) as z:pp=z['points_world_m'];tt=z['tetrahedra'];mm=z['material_indices'];targetnew=z['desired_boundary_faces']
assert np.array_equal(p.view('u4'),pp[:len(p)].view('u4'))
assert np.array_equal(target,targetnew)
keep=np.ones(len(t),bool);news=[];newm=[];operations=[];seen=set()
for o in r['operations']:
 hits=[]
 for rp in root.glob('trial-*/report.json'):
  rr=json.loads(rp.read_text())
  if rr['old_cell_rows']==o['old_cell_rows'] and rr['node']==o['node']:hits.append(rp)
 assert len(hits)==1
 rp=hits[0];replacement=rp.parent/'replacement.npz'
 with np.load(replacement) as z:
  rows=z['old_cell_rows'];new=z['new_tetrahedra'];nm=z['material_indices'];added=z['new_global_point_rows'];coords=z['new_interior_points_world_m']
 assert not set(rows)&seen;seen.update(map(int,rows))
 assert len(np.unique(mat[rows]))==1 and np.all(nm==mat[rows[0]])
 assert np.array_equal(pp[added].view('u4'),coords.view('u4'))
 assert M.certify(pp,t[rows],new)==(True,'PASS')
 before=int(sum(~M.passes(*M.measurements(p,t[rows]))));after=int(sum(~M.passes(*M.measurements(pp,new))))
 assert before==o['old_bad'] and after==o['new_bad'] and after<before
 keep[rows]=False;news.append(new);newm.append(nm)
 operations.append({'replacement':str(replacement),'sha256':sha(replacement),'exact_cavity_boundary_sign_volume':'PASS','bad_before':before,'bad_after':after})
assert np.array_equal(np.vstack([t[keep],*news]),tt)
assert np.array_equal(np.concatenate([mat[keep],*newm]),mm)
oldboundary=B.boundary(t);newboundary=B.boundary(tt)
assert np.array_equal(oldboundary,newboundary)
d,q=M.measurements(pp,tt);assert np.all(d>0)
assert np.count_nonzero(~M.passes(d,q))==r['bad_after']
assert len(np.unique(np.sort(tt,axis=1),axis=0))==len(tt)
receipt={'status':'PASS_INDEPENDENT_RECONSTRUCTED_CELL_REFINEMENT_REPLAY','physical_admission':False,'native_steps':0,'mass_mapping_status':'NOT_RUN','input':str(inp),'input_sha256':sha(inp),'candidate':str(candidate),'candidate_sha256':sha(candidate),'report_sha256':sha(root/'report.json'),'original_point_bytes_unchanged':True,'full_connectivity_material_replay':True,'full_oriented_boundary_unchanged':True,'boundary_face_count':len(newboundary),'boundary_sha256':hashlib.sha256(newboundary.astype('<i8').tobytes()).hexdigest(),'quality_failures':r['bad_after'],'nonpositive':0,'operations':operations,'verifier_sha256':sha(__file__),'elapsed_s':time.monotonic()-start}
out=H/'reconstructed-cell-refinement-375-root/independent376001';out.mkdir(exist_ok=False)
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
