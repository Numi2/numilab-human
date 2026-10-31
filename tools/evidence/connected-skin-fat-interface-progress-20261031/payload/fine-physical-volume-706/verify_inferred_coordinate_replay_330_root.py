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


ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--proposal',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--registration-map',type=Path,required=True);ap.add_argument('--source-node',type=Path,required=True);a=ap.parse_args()
a.output.mkdir(parents=True,exist_ok=False);st=time.monotonic()
r=json.loads(a.report.read_text());assert r['input_sha256']==sha(a.input) and r['output_sha256']==sha(a.proposal)
with np.load(a.input) as z:v={k:z[k] for k in z.files}
with np.load(a.proposal) as z:w={k:z[k] for k in z.files}
assert set(v)==set(w)
for k in v:
 if k!='points_world_m':assert np.array_equal(v[k],w[k]),k
p=v['points_world_m'].copy();t=v['tetrahedra'];initial=p.copy();final=w['points_world_m']
basequality=quality(p,t);baseboundary=boundary(t)
s=importlib.util.spec_from_file_location('identity330',H/'interior-quality-conditioning-309-banks/condition_registered_interior_309.py');reg=importlib.util.module_from_spec(s);s.loader.exec_module(reg)
source=np.loadtxt(a.source_node,skiprows=1,usecols=(1,2,3)).astype(np.float32)
registered,counts=reg.registered_source_rows_from_map(a.registration_map,source)
bb=reg.top.boundary(t);fixed=np.zeros(len(p),bool);fixed[np.unique(bb)]=True;fixed[:len(registered)]|=registered
assert np.array_equal(p[fixed].view(np.uint32),final[fixed].view(np.uint32))
nodes=[x['node'] for x in r['operations']];assert len(set(nodes))==len(nodes),'this verifier requires unique nodes per sweep'
changed=np.flatnonzero(np.any(p.view(np.uint32)!=final.view(np.uint32),axis=1));assert set(changed)==set(nodes)
flat=t.ravel();order=np.argsort(flat,kind='stable');offset=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(p)))]
for o in r['operations']:
 n=o['node'];assert not fixed[n]
 ir=np.unique(order[offset[n]:offset[n+1]]//4);star=t[ir]
 olddet,oldq=op.measurements(p,star);good=op.passes(olddet,oldq)
 before=sum(reg.exact_star_determinants(p,star));p[n]=final[n]
 dd,qq=op.measurements(p,star);ng=op.passes(dd,qq)
 assert np.all(dd>0) and all(x>0 for x in reg.exact_star_determinants(p,star))
 assert sum(reg.exact_star_determinants(p,star))==before
 assert int(np.sum(~good))==o['bad_before'] and int(np.sum(~ng))==o['bad_after']<o['bad_before']
 assert np.all(ng[good]),'formerly-good incident cell regressed'
assert np.array_equal(p.view(np.uint32),final.view(np.uint32))
newquality=quality(p,t);assert newquality['union_failures']==r['native_quality_defects_after']
assert boundary(t)==baseboundary
assert np.array_equal(p[:len(source)][registered].view(np.uint32),source[registered].view(np.uint32))
bd,bq=op.measurements(initial,t);ad,aq=op.measurements(p,t)
assert np.all(op.passes(ad,aq)[op.passes(bd,bq)])
rr={'status':'PASS_INDEPENDENT_INFERRED_INTERIOR_COORDINATE_REPLAY','physical_admission':False,'native_steps':0,
 'input':str(a.input),'input_sha256':sha(a.input),'proposal':str(a.proposal),'proposal_sha256':sha(a.proposal),
 'report_sha256':sha(a.report),'verifier_sha256':sha(Path(__file__)),'exact_per_star_F32_volume_and_sign':True,
 'all_registered_surface_identity_and_boundary_points_fixed':True,'nonpoint_npz_members_identical':True,
 'complete_oriented_boundary_sha256':baseboundary[1],'boundary_face_count':baseboundary[0],
 'unique_changed_inferred_nodes':len(nodes),'registered_identity_counts':counts,
 'baseline_quality':basequality,'final_quality':newquality,'no_formerly_good_cell_regressed':True,
 'operation_incident_row_note':'producer incident_cell_rows are local star rows; replay independently reconstructs global incidence from each point identity',
 'elapsed_s':time.monotonic()-st}
(a.output/'receipt.json').write_text(json.dumps(rr,indent=2)+'\n');print(json.dumps(rr))
