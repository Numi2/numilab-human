"""Independent exact-cavity replay for canonical fixed-boundary proposals."""
from pathlib import Path
import argparse,hashlib,json,importlib.util,time,numpy as np
H=Path(__file__).resolve().parent
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def imp(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=imp('cert198',H/'fixed_boundary_volume_flip_198_root.py')
v=imp('verify374',H/'verify_face_flips_generic_374_root.py')
ap=argparse.ArgumentParser();ap.add_argument('--receipt',type=Path,required=True);ap.add_argument('--receipt-sha256',required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
tic=time.monotonic();assert sha(args.receipt)==args.receipt_sha256
r=json.loads(args.receipt.read_text());assert r['status'].startswith('PASS')
for k in ('input','candidate','protected_keys'):
 if k in r:assert sha(r[k]['path'])==r[k]['sha256']
a=load(r['input']['path']);b=load(r['candidate']['path']);p=a['points_world_m'];t=a['tetrahedra'];mat=a['material_indices']
assert p.dtype==np.float32 and np.array_equal(p.view('u4'),b['points_world_m'].view('u4'))
if 'points' in a:assert np.array_equal(a['points'].view('u4'),b['points'].view('u4'))
protected=None
if 'protected_keys' in r:
 with np.load(r['protected_keys']['path']) as z:protected=set(map(int,z['current_output_cell_rows']))
def link(tt,vertices):
 from itertools import combinations
 out=set()
 for tet in tt:
  other=sorted(set(map(int,tet))-set(vertices))
  for k in range(len(other)+1):out.update(combinations(other,k))
 return out
seen=set();removed=[];new=[];repm=[];parent=[];offset=[0];before=after=0
for o in r['operations']:
 rows=np.asarray(o['old_rows'],np.int64);nt=np.asarray(o['new_rows'],np.int64)
 assert len(rows)>0 and len(nt)>0 and np.all((rows>=0)&(rows<len(t))) and len(np.unique(rows))==len(rows)
 verts=set(map(int,t[rows].ravel()));assert not verts&seen;seen.update(verts)
 assert set(map(int,nt.ravel()))<=verts and len(np.unique(mat[rows]))==1
 if protected is not None:assert set(map(int,rows))<=protected
 assert m.certify(p,t[rows],nt)==(True,'PASS')
 nb=int((~m.passes(*m.measurements(p,t[rows]))).sum());na=int((~m.passes(*m.measurements(p,nt))).sum())
 assert na<nb and nb==o['bad_before'] and na==o['bad_after']
 if 'retired_interior_point_row' in o:
  u=o['retained_point_row'];w=o['retired_interior_point_row']
  ru=np.flatnonzero(np.any(t==u,axis=1));rw=np.flatnonzero(np.any(t==w,axis=1))
  assert np.array_equal(np.sort(rows),rw) and np.array_equal(np.union1d(ru,rw),o['full_endpoint_star_rows'])
  assert w not in v.boundary(t)[:,:3] and u!=w
  tmp=np.where(t[rw]==w,u,t[rw]);ss=np.sort(tmp,axis=1);deleted=np.any(ss[:,1:]==ss[:,:-1],axis=1)
  assert np.array_equal(tmp[~deleted],nt)
  assert not np.any(m.passes(*m.measurements(p,t[rw]))[~deleted]&~m.passes(*m.measurements(p,nt)))
  edge=np.intersect1d(ru,rw,assume_unique=True)
  assert len(edge) and link(t[ru],(u,))&link(t[rw],(w,))==link(t[edge],(u,w))

 before+=nb;after+=na;removed.extend(rows.tolist());new.extend(nt.tolist());repm.extend([mat[rows[0]]]*len(nt))
 for _ in nt:parent.extend(rows.tolist());offset.append(len(parent))
keep=np.ones(len(t),bool);keep[removed]=False;nt=np.asarray(new,np.int64).reshape(-1,4)
assert len(removed)==len(set(removed))
assert np.array_equal(b['tetrahedra'],np.vstack((t[keep],nt)))
assert np.array_equal(b['material_indices'],np.r_[mat[keep],repm])
assert np.array_equal(b['retained_input_cell_rows'],np.flatnonzero(keep))
assert np.array_equal(b['replacement_tetrahedra'],nt)
assert np.array_equal(b['replacement_parent_offsets'],offset) and np.array_equal(b['replacement_parent_cell_rows'],parent)
mapping=np.r_[np.flatnonzero(keep),np.full(len(nt),-1,np.int64)];reverse=np.full(len(t),-1,np.int64);reverse[np.flatnonzero(keep)]=np.arange(keep.sum())
assert np.array_equal(mapping,b['output_cell_to_input_cell']) and np.array_equal(reverse,b['input_cell_to_output_cell'])
assert str(b['current_geometry_parent_path'][0])==r['input']['path'] and str(b['current_geometry_parent_sha256'][0])==r['input']['sha256']
bd=v.boundary(t);outbd=v.boundary(b['tetrahedra'])
assert np.array_equal(bd,outbd)
assert len(np.unique(np.sort(b['tetrahedra'],axis=1),axis=0))==len(b['tetrahedra'])
# Target orientation is owner-dependent, so compare its identity separately;
# complete old-to-new orientation equality above remains mandatory.
target=np.sort(a['desired_boundary_faces'],axis=1);target=target[np.lexsort((target[:,2],target[:,1],target[:,0]))]
assert np.array_equal(bd[:,:3],target)
d,q=m.measurements(p,b['tetrahedra']);od,oq=m.measurements(p,t)
assert np.all(d>0) and int((~m.passes(d,q)).sum())==int((~m.passes(od,oq)).sum())-before+after
for k in ('desired_boundary_faces','points_world_m','points_world_m_f32','points','source_O_path','source_O_sha256','source_I_path','source_I_sha256','source_D_path','source_D_sha256'):
 if k in a:assert np.array_equal(a[k],b[k]),k
rr={'status':'PASS_INDEPENDENT_CANONICAL_FIXED_CAVITY_REPLAY','producer':{'path':str(Path(__file__)),'sha256':sha(Path(__file__))},'proposal_receipt':{'path':str(args.receipt),'sha256':sha(args.receipt)},'input':r['input'],'candidate':r['candidate'],'operations':len(r['operations']),'cells':len(b['tetrahedra']),'native_quality_failures':int((~m.passes(d,q)).sum()),'nonpositive':0,'complete_boundary_faces':len(bd),'missing_target_faces':0,'extra_actual_faces':0,'complete_oriented_boundary_identical':True,'all_coordinates_fixed':True,'all_outside_cells_fixed':True,'per_cavity_exact_Float32_volume':True,'immediate_parent_union_replayed':True,'physical_admission':False,'native_steps':0,'elapsed_s':time.monotonic()-tic}
args.out.mkdir(exist_ok=False);(args.out/'receipt.json').write_text(json.dumps(rr,indent=2)+'\n');print(json.dumps(rr),flush=True)
