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

ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--proposal',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
a.output.mkdir(parents=True,exist_ok=False);st=time.monotonic()
p,t,mat=load_source(a.input);baseq=quality(p,t);baseb=boundary(t)
r=json.loads(a.report.read_text());assert r['input_sha256']==sha(a.input)
assert r['output_sha256']==sha(a.proposal)
with np.load(a.proposal) as z:
 assert np.array_equal(p.view(np.uint32),z['points_world_m'].view(np.uint32))
 proposal=z['tetrahedra'];pm=z['material_indices']
 flat=z['replacement_parent_cell_rows'];po=z['replacement_parent_offsets'];nf=z['replacement_tetrahedra'];no=z['replacement_tet_offsets']
 assert np.array_equal(z['retained_input_cell_rows'],np.setdiff1d(np.arange(len(t)),flat))
assert len(np.unique(flat))==len(flat)
assert len(po)==len(no)==r['passing_operations']+1 and po[0]==no[0]==0 and po[-1]==len(flat) and no[-1]==len(nf)
nm=[]
for k in range(len(po)-1):
 rows=flat[po[k]:po[k+1]];nt=nf[no[k]:no[k+1]]
 assert len(np.unique(mat[rows]))==1
 ok,why=op.certify(p,t[rows],nt);assert ok,why
 bd,bq=op.measurements(p,t[rows]);ad,aq=op.measurements(p,nt)
 assert sum(~op.passes(ad,aq))<sum(~op.passes(bd,bq)) and np.all(ad>0)
 nm.extend([mat[rows[0]]]*len(nt))
keep=np.ones(len(t),bool);keep[flat]=False
out=np.vstack([t[keep],nf]);outmat=np.r_[mat[keep],np.asarray(nm,dtype=mat.dtype)]
assert np.array_equal(out,proposal) and np.array_equal(outmat,pm)
newq=quality(p,out);newb=boundary(out);assert newb==baseb
assert len(np.unique(np.sort(out,axis=1),axis=0))==len(out)
assert newq['union_failures']==r['bad_cells_after']<baseq['union_failures']
receipt={'status':'PASS_INDEPENDENT_INFERRED_INTERIOR_RETIREMENT_REPLAY','physical_admission':False,'native_steps':0,
 'input':str(a.input),'input_sha256':sha(a.input),'proposal':str(a.proposal),'proposal_sha256':sha(a.proposal),
 'report_sha256':sha(a.report),'all_coordinates_bit_identical':True,'complete_oriented_boundary_identical':True,
 'nonmanifold_faces':0,'inconsistent_internal_face_orientation':0,'duplicate_cells':0,
 'exact_cavity_F32_sign_volume_boundary':True,'full_connectivity_and_material_replay':'PASS',
 'operations':len(po)-1,'baseline_quality':baseq,'final_quality':newq,'boundary_face_count':newb[0],
 'oriented_boundary_sha256':newb[1],'verifier_sha256':sha(Path(__file__)),'elapsed_s':time.monotonic()-st}
(a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
