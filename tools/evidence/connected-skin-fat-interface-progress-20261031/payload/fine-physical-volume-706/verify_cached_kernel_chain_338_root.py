"""Independent complete-domain checks and exact local replay for connectivity candidates."""
from pathlib import Path
import argparse,hashlib,json,time,numpy as np,importlib.util
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('f32_cavity',H/'fixed_boundary_volume_flip_198_root.py');op=importlib.util.module_from_spec(s);s.loader.exec_module(op)
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def load_source(path):
 with np.load(path) as z:return z['points_world_m'],z['tetrahedra'],z['material_indices']
def boundary(t):
 faces=t[:,op.PAT].reshape(-1,3)
 sortedf=np.sort(faces,axis=1);order=np.lexsort((sortedf[:,2],sortedf[:,1],sortedf[:,0]));sf=sortedf[order]
 starts=np.r_[0,1+np.flatnonzero(np.any(sf[1:]!=sf[:-1],axis=1))];counts=np.diff(np.r_[starts,len(sf)])
 assert np.all(counts<=2)
 inversions=((faces[:,0]>faces[:,1]).astype(np.int8)+(faces[:,0]>faces[:,2]).astype(np.int8)+(faces[:,1]>faces[:,2]).astype(np.int8))
 signs=1-2*(inversions%2)
 summed=np.add.reduceat(signs[order],starts)
 assert np.all(summed[counts==2]==0)
 rows=starts[counts==1];b=np.c_[sf[rows],summed[counts==1]].astype('<i8')
 return len(b),hashlib.sha256(b.tobytes()).hexdigest()
def quality(p,t):
 d,q=op.measurements(p,t)
 assert np.all(d>0)
 return {'cells':len(t),'normalized_det_failures':int(sum(q<1e-5)),'absolute_volume_failures':int(sum(d/6<=1e-18)),'union_failures':int(sum(~op.passes(d,q))),'minimum_normalized_det':float(q.min()),'minimum_volume_m3':float(d.min()/6)}


ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--step',type=Path,action='append',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
a.output.mkdir(parents=True,exist_ok=False);st=time.monotonic()
p,t,mat=load_source(a.input);baseline=quality(p,t);bb=boundary(t);original=p.copy();current=a.input;chain=[];nops=0;casepins=[]
for folder in a.step:
 r=json.loads((folder/'report.json').read_text());assert r['input_sha256']==sha(current)
 f=Path(r['output']);assert sha(f)==r['output_sha256']
 with np.load(f) as z:pp=z['points_world_m'];tt=z['tetrahedra'];mm=z['material_indices']
 assert np.array_equal(pp[:len(p)].view(np.uint32),p.view(np.uint32))
 assert len(pp)==len(p)+len(r['operations'])
 allold=[];allnew=[];allmat=[]
 for k,o in enumerate(r['operations']):
  case=folder/('pass-%04d'%k)/'replacement.npz'
  with np.load(case) as z:rows=z['old_cell_rows'];new=z['new_tetrahedra'];npnt=z['new_point_world_m'];nm=z['material_indices']
  assert rows.tolist()==o['old_cell_rows'] and o['new_global_point_row']==len(p)+k
  assert np.array_equal(npnt.view(np.uint32),pp[len(p)+k].view(np.uint32))
  assert np.any(new==len(p)+k) and np.all((new<len(p))|(new==len(p)+k))
  assert len(np.unique(mat[rows]))==1 and np.all(nm==mat[rows[0]])
  ok,why=op.certify(pp,t[rows],new);assert ok,why
  bd,bq=op.measurements(pp,t[rows]);ad,aq=op.measurements(pp,new)
  assert sum(~op.passes(ad,aq))<sum(~op.passes(bd,bq)) and np.all(ad>0)
  allold.extend(rows.tolist());allnew.append(new);allmat.append(nm)
  casepins.append({'path':str(case),'sha256':sha(case)})
 assert len(set(allold))==len(allold)
 keep=np.ones(len(t),bool);keep[allold]=False
 nt=np.vstack([t[keep],*(allnew or [np.empty((0,4),np.int64)])])
 nm=np.concatenate([mat[keep],*allmat])
 assert np.array_equal(nt,tt) and np.array_equal(nm,mm)
 assert quality(pp,tt)['union_failures']==r['bad_after']
 chain.append({'input_sha256':sha(current),'report':str(folder/'report.json'),'report_sha256':sha(folder/'report.json'),'candidate':str(f),'candidate_sha256':sha(f),'operations':len(r['operations'])})
 nops+=len(r['operations']);p,t,mat=pp,tt,mm;current=f
assert np.array_equal(original.view(np.uint32),p[:len(original)].view(np.uint32))
fb=boundary(t);assert fb==bb
assert len(np.unique(np.sort(t,axis=1),axis=0))==len(t)
r={'status':'PASS_INDEPENDENT_CACHED_KERNEL_FAN_CHAIN_REPLAY','physical_admission':False,'native_steps':0,
 'input':str(a.input),'input_sha256':sha(a.input),'candidate':str(current),'candidate_sha256':sha(current),
 'original_point_prefix_bit_identical':True,'new_nodes':'inferred volume-interior construction points only',
 'complete_oriented_boundary_bit_identical':True,'boundary_face_count':fb[0],'oriented_boundary_sha256':fb[1],
 'nonmanifold_faces':0,'inconsistent_internal_face_orientation':0,'duplicate_cells':0,
 'exact_per_cavity_positive_F32_volume_and_boundary':True,'connectivity_and_material_full_replay':'PASS',
 'operations':nops,'baseline_quality':baseline,'final_quality':quality(p,t),'chain':chain,'case_payload_pins':casepins,
 'verifier_sha256':sha(Path(__file__)),'elapsed_s':time.monotonic()-st,
 'limitation':'native quality still incomplete; mass mapping, GPU accepted-state and motion are not qualified'}
(a.output/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('chain','case_payload_pins')}))
