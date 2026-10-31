"""Independent exact replay of construction-only LP interior relocations."""
from pathlib import Path
import json,hashlib,importlib.util,argparse,numpy as np,time
H=Path(__file__).resolve().parent
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
T=mod('volume456',H/'transfer_surface_contractions_indexed_379_root.py');R=mod('registration456',H/'interior-quality-conditioning-309-banks/condition_registered_interior_309.py')
a=argparse.ArgumentParser();a.add_argument('--report',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();args.output.mkdir(exist_ok=False);start=time.monotonic()
r=json.loads(args.report.read_text());ip=Path(r['input']);op=Path(r['candidate']);assert sha(ip)==r['input_sha256'] and sha(op)==r['candidate_sha256']
with np.load(ip) as z:old={k:z[k] for k in z.files}
with np.load(op) as z:new={k:z[k] for k in z.files}
for key in old:
 if key!='points_world_m':assert np.array_equal(old[key],new[key]),key
p=old['points_world_m'].copy();t=old['tetrahedra'];mat=old['material_indices'];d,q=T.m.measurements(p,t);good=T.m.passes(d,q)
boundary=T.boundary(t);fixed=np.zeros(len(p),bool);fixed[np.unique(boundary)]=True
source=H/'combined-child-handoff-142/skin.node';sp=np.loadtxt(source,skiprows=1,usecols=(1,2,3)).astype(np.float32)
assert sha(r['registration_map'])==r['registration_map_sha256']
reg,counts=R.registered_source_rows_from_map(Path(r['registration_map']),sp);fixed[:len(sp)]|=reg
used=set();signs=0;exact_delta=0;maxd=0.
for entry in r['operations']:
 node=int(entry['node']);assert not fixed[node]
 rows=np.asarray(entry['cell_rows'],np.int64);assert np.array_equal(rows,np.flatnonzero(np.any(t==node,axis=1)))
 assert not used&set(map(int,rows));used.update(map(int,rows))
 assert len(np.unique(mat[rows]))==1 and np.array_equal(p[node].view(np.uint32),np.asarray(entry['old_f32'],np.float32).view(np.uint32))
 oldsum=sum(R.exact_star_determinants(p,t[rows]));assert str(oldsum)==entry['exact_det6_sum']
 oldpass=T.m.passes(*T.m.measurements(p,t[rows]));before=p[node].copy();p[node]=np.asarray(entry['new_f32'],np.float32)
 nd,nq=T.m.measurements(p,t[rows]);newpass=T.m.passes(nd,nq)
 assert np.all(nd>0) and not np.any(oldpass&~newpass)
 assert int((~oldpass).sum())==entry['old_bad'] and int((~newpass).sum())==entry['new_bad'] and entry['new_bad']<entry['old_bad']
 ex=R.exact_star_determinants(p,t[rows]);assert min(ex)>0 and sum(ex)==oldsum;signs+=len(ex)
 displacement=float(np.linalg.norm(p[node].astype(float)-before.astype(float)));assert displacement==entry['displacement_m'] and displacement<=entry['maximum_displacement_m']
 maxd=max(maxd,displacement)
assert np.array_equal(p.view(np.uint32),new['points_world_m'].view(np.uint32))
assert np.array_equal(p[fixed].view(np.uint32),old['points_world_m'][fixed].view(np.uint32))
assert len(np.unique(p,axis=0))==len(np.unique(old['points_world_m'],axis=0))
nd,nq=T.m.measurements(p,t);ng=T.m.passes(nd,nq)
assert np.all(nd>0) and not np.any(good&~ng) and int((~ng).sum())==r['bad_after']
assert np.array_equal(T.boundary_identity(boundary),T.boundary_identity(new['desired_boundary_faces']))
receipt={'status':'PASS_EXACT_FIXED_BOUNDARY_INTERIOR_LP_REPLAY','physical_admission':False,'native_steps':0,'input':str(ip),'input_sha256':sha(ip),'candidate':str(op),'candidate_sha256':sha(op),'construction_report':str(args.report),'construction_report_sha256':sha(args.report),'bad_before':r['bad_before'],'bad_after':r['bad_after'],'operations':len(r['operations']),'exact_positive_signs_checked':signs,'exact_local_and_global_volume_delta':0,'complete_boundary_source_geometry_material_connectivity_preserved':True,'all_registered_points_bit_identical':True,'maximum_inferred_interior_displacement_m':maxd,'source330_mass':'IMMUTABLE_CURRENT_MAPPING_REQUIRED','producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start}
(args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
