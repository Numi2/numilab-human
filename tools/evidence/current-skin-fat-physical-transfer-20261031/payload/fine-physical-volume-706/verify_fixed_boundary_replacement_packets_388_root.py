"""Replay existing fixed-boundary remesh replacement packets with source198 certificates."""
from pathlib import Path
import argparse,json,hashlib,numpy as np,importlib.util,time
H=Path(__file__).resolve().parent
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
M=load('cavity198control388',H/'fixed_boundary_volume_flip_198_root.py')
B=load('boundary315control388',H/'verify_face_flips_generic_374_root.py')
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--replacement',type=Path,action='append',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();start=time.monotonic()
with np.load(a.input) as z:p=z['points_world_m'];t=z['tetrahedra'];mat=z['material_indices'];target=z['desired_boundary_faces']
with np.load(a.candidate) as z:pp=z['points_world_m'];tt=z['tetrahedra'];mm=z['material_indices'];target2=z['desired_boundary_faces'];retained=z['retained_input_cell_rows']
assert np.array_equal(p.view('u4'),pp[:len(p)].view('u4')) and np.array_equal(target,target2)
keep=np.ones(len(t),bool);news=[];newmat=[];witnesses=[];addedset=set();used=set()
for path in a.replacement:
 with np.load(path) as z:
  rows=z['parent_cell_rows'] if 'parent_cell_rows' in z.files else z['old_cell_rows']
  nt=z['tetrahedra'] if 'tetrahedra' in z.files else z['new_tetrahedra']
  extra=z['new_global_point_rows'];xyz=z['new_interior_points_world_m']
 assert not used&set(map(int,rows));used.update(map(int,rows))
 assert np.all(rows>=0) and np.all(rows<len(t)) and len(np.unique(mat[rows]))==1
 assert np.array_equal(pp[extra].view('u4'),xyz.view('u4'))
 addedset.update(map(int,extra))
 assert M.certify(pp,t[rows],nt)==(True,'PASS')
 before=int(sum(~M.passes(*M.measurements(p,t[rows]))));after=int(sum(~M.passes(*M.measurements(pp,nt))));assert after<before
 keep[rows]=False;news.append(nt);newmat.append(np.full(len(nt),mat[rows[0]],mat.dtype))
 witnesses.append({'replacement':str(path),'sha256':sha(path),'old_cells':len(rows),'new_cells':len(nt),'bad_before':before,'bad_after':after,'exact_local_boundary_sign_volume':'PASS'})
assert addedset==set(range(len(p),len(pp)))
assert np.array_equal(np.flatnonzero(keep),retained)
assert np.array_equal(np.vstack([t[keep],*news]),tt) and np.array_equal(np.r_[mat[keep],*newmat],mm)
bd=B.boundary(t);bd2=B.boundary(tt);assert np.array_equal(bd,bd2)
assert len(np.unique(np.sort(tt,axis=1),axis=0))==len(tt)
d,q=M.measurements(pp,tt);assert np.all(d>0)
r={'status':'PASS_INDEPENDENT_FIXED_BOUNDARY_REPLACEMENT_PACKET_REPLAY','physical_admission':False,'native_steps':0,'mass_mapping_status':'NOT_RUN','input':str(a.input),'input_sha256':sha(a.input),'candidate':str(a.candidate),'candidate_sha256':sha(a.candidate),'full_point_connectivity_material_replay':'PASS','complete_oriented_boundary_unchanged':True,'boundary_face_count':len(bd2),'boundary_sha256':hashlib.sha256(bd2.astype('<i8').tobytes()).hexdigest(),'native_quality_failures':int(sum(~M.passes(d,q))),'nonpositive_cells':0,'operations':witnesses,'verifier_sha256':sha(__file__),'elapsed_s':time.monotonic()-start}
a.output.mkdir(parents=True,exist_ok=False);(a.output/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
