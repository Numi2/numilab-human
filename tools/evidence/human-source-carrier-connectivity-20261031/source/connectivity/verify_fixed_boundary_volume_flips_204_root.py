"""Independent complete-domain checks and exact local replay for connectivity candidates."""
from pathlib import Path
import argparse,hashlib,json,time,numpy as np,importlib.util
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('f32_cavity',H/'fixed_boundary_volume_flip_198_root.py');op=importlib.util.module_from_spec(s);s.loader.exec_module(op)
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def load_source(role):
 p=H/'combined-child-handoff-142'/role.lower()
 nd=np.loadtxt(p.with_suffix('.node'),skiprows=1);ids=nd[:,0].astype(np.int64);xyz=nd[:,1:4].astype(np.float32)
 ee=np.loadtxt(p.with_suffix('.ele'),skiprows=1);t=np.searchsorted(ids,ee[:,1:5].astype(np.int64));mat=ee[:,5].astype(np.int32)
 assert np.array_equal(ids[t],ee[:,1:5].astype(np.int64))
 return xyz,t,mat
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
ap=argparse.ArgumentParser();ap.add_argument('--role',choices=['Skin','Fat'],required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();p,t,mat=load_source(a.role);basequality=quality(p,t);baseboundary=boundary(t)
if a.role=='Skin':
 prefix=[H/'fixed-boundary-volume-flips-198-root/skin-run001',H/'fixed-boundary-volume-flips-199-root/skin-after198-run001']
 batch200=H/'fixed-boundary-volume-flips-200-root/skin-batch001';suffix=[H/'fixed-boundary-volume-flips-202-root/skin-run002']
else:prefix=[];batch200=H/'fixed-boundary-volume-flips-200-root/fat-batch001';suffix=[]
def batch_steps(path):
 br=json.loads((path/'receipt.json').read_text())
 return [(path/('pass-%03d'%h['pass']),h) for h in br['history']]
batch=H/('fixed-boundary-volume-flips-203-root/'+a.role.lower()+'-batch001')
br=json.loads((batch/'receipt.json').read_text())
steps=[(d,None) for d in prefix]+batch_steps(batch200)+[(d,None) for d in suffix]+batch_steps(batch)
checked=[];operations=0
for d,hist in steps:
 isdelta=hist is not None
 r=json.loads((d/'report.json').read_text())
 if isdelta:
  delta=d/'exact-connectivity-delta.npz'
  assert sha(delta)==hist['delta_sha256'] and sha(d/'report.json')==hist['report_sha256']
  with np.load(delta) as z:
   if 'replacement_parent_offsets' in z.files:
    oldflat=z['replacement_parent_cell_rows'];po=z['replacement_parent_offsets'];newflat=z['replacement_tetrahedra'];no=z['replacement_tet_offsets']
   else:
    oldrows=z['replaced_parent_cell_rows'];new=z['replacement_tetrahedra']
    oldflat=oldrows.ravel();po=np.arange(len(oldrows)+1)*oldrows.shape[1];newflat=new.reshape(-1,4);no=np.arange(len(new)+1)*new.shape[1]
   newmat=z['replacement_material_indices']
 else:
  f=d/'fixed-boundary-volume-proposal.npz'
  with np.load(f) as z:
   assert np.array_equal(p.view(np.uint32),z['points_world_m'].view(np.uint32))
   proposal=z['tetrahedra'];proposalmat=z['material_indices']
   if 'replacement_parent_offsets' in z.files:
    oldflat=z['replacement_parent_cell_rows'];po=z['replacement_parent_offsets'];newflat=z['replacement_tetrahedra'];no=z['replacement_tet_offsets']
   else:
    oldrows=z['replacement_parent_cell_rows'];new=z['replacement_tetrahedra']
    oldflat=oldrows.ravel();po=np.arange(len(oldrows)+1)*oldrows.shape[1];newflat=new.reshape(-1,4);no=np.arange(len(new)+1)*new.shape[1]
  newmat=np.concatenate([np.full(no[k+1]-no[k],mat[oldflat[po[k]]],np.int32) for k in range(len(po)-1)]) if len(oldflat) else np.empty(0,np.int32)
 nops=len(po)-1
 assert nops==r['passing_operations'] and len(no)==len(po) and po[0]==no[0]==0 and po[-1]==len(oldflat) and no[-1]==len(newflat)
 assert len(np.unique(oldflat))==oldflat.size
 seen=set()
 for k in range(nops):
  rows=oldflat[po[k]:po[k+1]];nt=newflat[no[k]:no[k+1]]
  before=t[rows];vertices=set(map(int,before.ravel()))
  assert not vertices&seen;seen.update(vertices)
  assert len(set(map(int,mat[rows])))==1
  ok,why=op.certify(p,before,nt)
  if not ok:raise ValueError(why)
  bd,bq=op.measurements(p,before);ad,aq=op.measurements(p,nt)
  assert np.all(ad>0) and sum(~op.passes(ad,aq))<sum(~op.passes(bd,bq))
 mask=np.ones(len(t),bool);mask[oldflat]=False
 t=np.vstack([t[mask],newflat]);mat=np.r_[mat[mask],newmat];operations+=nops
 if not isdelta:assert np.array_equal(t,proposal) and np.array_equal(mat,proposalmat)
 assert quality(p,t)['union_failures']==r['bad_cells_after']
 checked.append({'path':str(d),'operations':nops,'exact_F32_positive_volume_and_boundary':'PASS'})
final=Path(br['final_proposal']);assert sha(final)==br['final_proposal_sha256']
with np.load(final) as z:
 assert np.array_equal(p.view(np.uint32),z['points_world_m'].view(np.uint32))
 assert np.array_equal(t,z['tetrahedra']) and np.array_equal(mat,z['material_indices'])
fb=boundary(t);assert fb==baseboundary
canonical=np.sort(t,axis=1);ix=np.lexsort(tuple(canonical[:,k] for k in (3,2,1,0)));assert not np.any(np.all(canonical[ix][1:]==canonical[ix][:-1],axis=1))
report={'status':'PASS_INDEPENDENT_FIXED_BOUNDARY_VOLUME_CONNECTIVITY_REPLAY','physical_admission':False,'native_steps':0,'role':a.role,'coordinates_bit_identical':True,'complete_oriented_boundary_identical':True,'boundary_face_count':fb[0],'oriented_boundary_sha256':fb[1],'nonmanifold_faces':0,'inconsistent_internal_face_orientation':0,'duplicate_tets':0,'all_local_cavity_F32_volume_equalities_exact':True,'material_per_cavity_preserved':True,'operations':operations,'baseline_quality':basequality,'final_quality':quality(p,t),'source_node_sha256':sha(H/'combined-child-handoff-142'/(a.role.lower()+'.node')),'source_element_sha256':sha(H/'combined-child-handoff-142'/(a.role.lower()+'.ele')),'final_proposal':str(final),'final_proposal_sha256':sha(final),'checked_steps':checked,'verifier_sha256':sha(Path(__file__)),'cavity_predicate_source_sha256':sha(H/'fixed_boundary_volume_flip_198_root.py'),'elapsed_s':time.monotonic()-start,'limitation':'Connectivity proposal still fails the native geometry gate; new cell ancestry and physical mass mapping are not yet generated. No motion or runtime claim.'}
(a.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='checked_steps'}))
