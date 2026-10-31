"""Independent typed and exact volume replay of optional construction-only net-quality exchange. Native admission remains false."""
#!/usr/bin/env python3
"""Independent paired I quotient replay and existing typed-face-ancestry binding."""
from pathlib import Path
import numpy as np,json,hashlib,importlib.util,sys,time,argparse
H=Path(__file__).resolve().parent
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
A=mod('typed353',H/'fixed-boundary-quality-topology-196/prepare_current_earclip_adapter_024.py')
B=mod('base353',H/'fixed-boundary-quality-topology-196/run_fixed_boundary_dp_batch_197.py')
J=mod('apex353',H/'audit_joint_native_apex_272_root.py')
T=mod('transfer353',H/'transfer_surface_contractions_to_volume_209_root.py')
R=mod('exact353',H/'interior-quality-conditioning-309-banks/condition_registered_interior_309.py')
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bad(p,t):
 d,q=T.m.measurements(p,t);assert np.all(d>0);return int(sum(~T.m.passes(d,q)))
ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
rpath=args.report
r=json.loads(rpath.read_text());assert r['status']=='PASS_PAIRED_I_QUOTIENT_PHYSICAL_PREFLIGHT'
out=args.output;out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
ip=Path(r['source_I']);assert sha(ip)==r['source_I_sha256'];par=A.load(ip);p=par['points_world_m_f32'];f=par['faces']
raw=A.load(Path(r['candidate_I']));assert sha(Path(r['candidate_I']))==r['candidate_I_sha256']
F=raw['faces'];fm=raw['source_input_face_rows'];mp=raw['source_input_vertex_quotient']
assert np.array_equal(raw['points_world_m'],p) and np.array_equal(F,mp[f][fm])
beforeq=J.score(p,f);newq=J.score(p,F)
badrows=np.flatnonzero(newq<J.GATE)
if len(badrows):
 (out/'source-apex-failure.json').write_text(json.dumps({'status':'FAIL_SOURCE_APEX_PREFLIGHT','bad_output_face_rows':badrows.tolist(),'q':newq[badrows].tolist(),'source_input_face_rows':fm[badrows].tolist(),'native_steps':0,'physical_admission':False},indent=2)+'\n')
 raise ValueError(('source apex obstruction',len(badrows)))
b0,t0=B.topology(f);b1,t1=B.topology(F);assert np.array_equal(b0,b1)
assert all(t0[k]==t1[k] for k in ('boundary_edges','nonmanifold_edges','same_direction_edges'))
def chi(f):
 e=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
 return len(np.unique(f))-len(np.unique(e,axis=0))+len(f)
assert chi(f)==chi(F)
ops=r['operations'];op_by_parent={};docs=[]
for oi,op in enumerate(ops):
 rows=op['retired_parent_face_rows'];U=B.union_tuples(par,rows);C=B.ancestry_union(par,rows)
 assert U and C
 for j in rows:assert j not in op_by_parent;op_by_parent[j]=(oi,U,C)
 docs.append({'operation':oi,'role':'I','retired_parent_face_rows':rows,'retained_input_point_row':op['retained_input_point_row'],'collapsed_input_point_row':op['collapsed_input_point_row'],'complete_typed_parent_union':U,'source_face_ancestry_union':C,'new_output_face_rows':[]})
us=[];anc=[];opindex=np.full(len(F),-1,np.int64)
for i,j in enumerate(fm):
 j=int(j)
 if np.array_equal(F[i],f[j]):us.append(A.rows_union(par,j));anc.append(A.rows_anc(par,j))
 else:
  oi,U,C=op_by_parent[j];us.append(U);anc.append(C);opindex[i]=oi;docs[oi]['new_output_face_rows'].append(i)
uo,ud=A.csr(us,7);ao,ad=A.csr(anc,3);typ=np.asarray([u[0] for u in us],np.int64)
faceid=par['quality_star_face_output_to_input_row'][fm].copy();faceid[opindex>=0]=-1
payload={'points_world_m':par['points_world_m'],'points_world_m_f32':p,'faces':F,
 'quality_star_point_output_to_input_row':par['quality_star_point_output_to_input_row'],
 'quality_star_face_output_to_input_row':faceid,'quality_star_patch_union_offsets':uo,'face_ancestry_offsets':ao,
 'source_current_candidate_path':np.asarray([r['candidate_I']]),'source_current_candidate_sha256':np.asarray([r['candidate_I_sha256']]),
 'typed_parent_candidate_path':np.asarray([str(ip)]),'typed_parent_candidate_sha256':np.asarray([sha(ip)]),
 'source_input_face_rows':fm,'source_input_vertex_quotient':mp,'source_collapse_operation_rows':opindex}
payload.update({k:typ[:,j] for j,k in enumerate(A.T)})
payload.update({k:ud[:,j] for j,k in enumerate(A.U)})
payload.update({k:ad[:,j] for j,k in enumerate(A.A)})
target=out/'candidate-I-typed-current.npz';np.savez_compressed(target,**payload)
pp=A.load(target);assert all(np.array_equal(pp[k],v) for k,v in payload.items())
prot=json.loads(A.PROT.read_text());routes=0
for route in prot['surface_routes']:
 if route['role']=='I':
  z=np.asarray(route['target_world_m_f32'],np.float32)
  ids=np.flatnonzero(np.all(p==z,axis=1));assert len(ids)==1 and int(ids[0]) in set(map(int,F.ravel()));routes+=1
roleproof=[];volume_delta=0
for rr in r['roles']:
 source=Path(rr['input']);dest=Path(rr['output'])
 assert sha(source)==rr['input_sha256'] and sha(dest)==rr['output_sha256']
 a=A.load(source);b=A.load(dest);v=a['points_world_m'];t=a['tetrahedra'].copy();active=np.ones(len(t),bool)
 assert np.array_equal(v,b['points_world_m'])
 occupied=set();changed=0;localdelta=0
 for entry,op in zip(rr['operations'],ops):
  assert entry['operation']==changed;changed+=1
  rows=np.asarray(entry['source_cell_rows'],np.int64);keep=entry['kept_volume_node'];drop=entry['retired_volume_node']
  expectedrows=np.flatnonzero(np.any(a['tetrahedra']==drop,axis=1));assert np.array_equal(rows,expectedrows)
  assert T.key(v[keep])==T.key(p[op['retained_input_point_row']]) and T.key(v[drop])==T.key(p[op['collapsed_input_point_row']])
  old=t[rows].copy();new=np.where(old==drop,keep,old);degen=np.any(np.diff(np.sort(new,axis=1),axis=1)==0,axis=1)
  ids=set(map(int,old.ravel()));assert not(ids&occupied);occupied.update(ids)
  od,oq=T.m.measurements(v,old);nd,nq=T.m.measurements(v,new[~degen]);assert np.all(nd>0)
  oldpass=T.m.passes(od,oq);newpass=T.m.passes(nd,nq);newfail=oldpass[~degen]&~newpass
  if not r.get('construction_quality_exchange',False):assert not np.any(newfail)
  else:
   assert int(np.count_nonzero(newfail))==entry['new_retained_quality_failures']
   assert np.array_equal(rows[~degen][newfail],entry['new_retained_quality_failure_rows'])
   assert np.count_nonzero(~newpass)<=np.count_nonzero(~oldpass)
   for material in np.unique(a['material_indices'][rows]):
    assert np.count_nonzero((~newpass)&(a['material_indices'][rows][~degen]==material))<=np.count_nonzero((~oldpass)&(a['material_indices'][rows]==material))
  assert sum(~T.m.passes(od,oq))==entry['bad_before'] and sum(~T.m.passes(nd,nq))==entry['bad_after']
  assert all(x>0 for x in R.exact_star_determinants(v,new[~degen]))
  localdelta+=sum(R.exact_star_determinants(v,new[~degen]))-sum(R.exact_star_determinants(v,old))
  t[rows]=new;active[rows[degen]]=False
 assert np.array_equal(t[active],b['tetrahedra']) and np.array_equal(a['material_indices'][active],b['material_indices'])
 assert np.array_equal(np.flatnonzero(active),b['retained_input_cell_rows'])
 bd=T.boundary(b['tetrahedra']);assert np.array_equal(T.boundary_identity(bd),T.boundary_identity(b['boundary_faces']))
 assert len(np.unique(np.sort(b['tetrahedra'],axis=1),axis=0))==len(b['tetrahedra'])
 vm=np.arange(len(v));look={T.key(x):i for i,x in enumerate(v)}
 for op in ops:vm[look[T.key(p[op['collapsed_input_point_row']])]]=look[T.key(p[op['retained_input_point_row']])]
 expected=vm[T.boundary(a['tetrahedra'])];degen=np.any(np.diff(np.sort(expected,axis=1),axis=1)==0,axis=1)
 assert np.array_equal(T.boundary_identity(expected[~degen]),T.boundary_identity(bd))
 assert bad(v,b['tetrahedra'])==rr['bad_after']
 volume_delta+=localdelta
 roleproof.append({'role':rr['role'],'input':str(source),'input_sha256':sha(source),'output':str(dest),'output_sha256':sha(dest),'operations_replayed':changed,'bad_before':rr['bad_before'],'bad_after':rr['bad_after'],'exact_signed_integer_volume_delta':str(localdelta),'point_material_connectivity_boundary_replay':True})
assert volume_delta==0
(out/'typed-operations.json').write_text(json.dumps(docs,indent=2)+'\n')
receipt={'status':'PASS_PAIRED_I_TYPED_ANCESTRY_AND_INDEPENDENT_VOLUME_REPLAY','physical_admission':False,'native_steps':0,'source':str(ip),'source_sha256':sha(ip),'candidate':str(target),'candidate_sha256':sha(target),'input_report':str(rpath),'input_report_sha256':sha(rpath),'construction_quality_exchange':r.get('construction_quality_exchange',False),'new_retained_quality_failures':sum(e.get('new_retained_quality_failures',0) for v in r['roles'] for e in v['operations']),'native_quality_gates_unchanged':True,'source_points_bit_identical':True,'rim_edges_bit_identical':True,'protected_routes_unchanged':routes,'minimum_joint_apex_bound':float(newq.min()),'joint_apex_obstructions':0,'typed_tuples':len(ud),'ancestry_tuples':len(ad),'roles':roleproof,'exact_combined_integer_volume_delta':str(volume_delta),'producer_sha256':sha(Path(__file__)),'limitations':'Complete anatomical collision context and global mass transfer not yet run; no native or physical admission','elapsed_s':time.monotonic()-start}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

