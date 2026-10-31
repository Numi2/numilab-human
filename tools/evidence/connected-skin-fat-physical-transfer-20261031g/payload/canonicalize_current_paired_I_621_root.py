"""Bind and canonicalize the independently replayed current paired-I batch."""
from pathlib import Path
import numpy as np,json,hashlib,importlib.util,time
H=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def imp(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 start=time.monotonic();root=H/'paired-I-current620-root';rp=root/'batch001/report.json';r=json.loads(rp.read_text())
 proof=root/'typed-replay001/receipt.json';pr=json.loads(proof.read_text());assert pr['input_report_sha256']==sha(rp) and pr['status']=='PASS_PAIRED_I_TYPED_ANCESTRY_AND_INDEPENDENT_VOLUME_REPLAY'
 ctx=root/'context001/report.json';assert sha(ctx)=='368dda36536ee24fa1d332597c8367a4fd5d8f738a1c86c35229bce26f26a677'
 cr=json.loads(ctx.read_text());assert cr['status']=='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
 cf=Path(cr['config']['path']);assert sha(cf)==cr['config']['sha256'];cfg=json.loads(cf.read_text())
 ip=Path(pr['candidate']);assert sha(ip)==pr['candidate_sha256']==cfg['candidate_surfaces']['I']['sha256'];ii=load(ip)
 U=imp('canon055',H/'paired-I-local-cavity-527-root/run055-current-I50-D264-source-I-baseline-context/canonicalize_run055_current_parent_unions_v3.py')
 T=imp('target209',H/'transfer_surface_contractions_to_volume_209_root.py')
 out=root/'canonical621';out.mkdir(exist_ok=False);reports=[]
 for rr in r['roles']:
  role=rr['role'];basep=Path(rr['input']);rawp=Path(rr['output']);assert sha(basep)==rr['input_sha256'] and sha(rawp)==rr['output_sha256']
  base=load(basep);raw=load(rawp);p=base['points_world_m'];ot=base['tetrahedra'];om=base['material_indices']
  assert p.dtype==np.float32 and np.array_equal(p.view('u4'),raw['points_world_m'].view('u4'))
  csr=U.incidence_csr(ot,len(p));q=np.arange(len(p),dtype=np.int64);stars=[]
  for op in rr['operations']:
   drop=op['retired_volume_node'];keep=op['kept_volume_node'];assert drop!=keep and q[drop]==drop
   assert np.array_equal(np.sort(op['source_cell_rows']),U.full_star(csr,[drop]))
   stars.append(U.full_star(csr,[keep,drop]));q[q==drop]=q[keep]
  mapped=q[ot];deg=np.any(np.diff(np.sort(mapped,axis=1),axis=1)==0,axis=1);origin=np.flatnonzero(~deg);t=mapped[origin];mat=om[origin]
  assert np.array_equal(t,raw['tetrahedra']) and np.array_equal(mat,raw['material_indices'])
  oldtarget=base['desired_boundary_faces'];assert U.counter_faces(T,oldtarget)==U.counter_faces(T,T.boundary(ot))
  target=q[oldtarget];target=target[~np.any(np.diff(np.sort(target,axis=1),axis=1)==0,axis=1)];bd=T.boundary(t)
  assert U.counter_faces(T,target)==U.counter_faces(T,bd) and len(target)==len(oldtarget)-2*len(stars)
  lookup={T.key(x):i for i,x in enumerate(p)};assert len(lookup)==len(p)
  sf=ii['faces'][:,::-1] if role=='Skin' else ii['faces']
  iface=np.asarray([[lookup[T.key(ii['points_world_m_f32'][v])] for v in f] for f in sf],np.int64)
  assert set(map(tuple,T.boundary_identity(iface))).issubset(set(map(tuple,T.boundary_identity(target))))
  changed=np.any(t!=ot[origin],axis=1);changed_out=np.flatnonzero(changed)
  _,po,parents,ng,ncg,total=U.make_parent_unions(len(ot),stars,origin[changed])
  oi=origin.copy();oi[changed]=-1;io=np.full(len(ot),-1,np.int64);io[origin[~changed]]=np.flatnonzero(~changed)
  payload=dict(base);payload.update(tetrahedra=t,material_indices=mat,boundary_faces=bd,desired_boundary_faces=target,
   current_geometry_parent_path=np.asarray([str(basep)]),current_geometry_parent_sha256=np.asarray([sha(basep)]),
   output_cell_to_input_cell=oi,input_cell_to_output_cell=io,retained_input_cell_rows=origin[~changed],
   replacement_parent_offsets=po,replacement_parent_cell_rows=parents,replacement_tet_offsets=np.asarray([0,len(changed_out)],np.int64),
   replacement_tetrahedra=t[changed],input_node_to_output_node=q,output_node_to_input_node=np.arange(len(p),dtype=np.int64),
   output_point_to_input_point_row=np.arange(len(p),dtype=np.int64),output_point_parent_offsets=np.zeros(len(p)+1,np.int64),
   output_point_parent_input_cell_rows=np.empty(0,np.int64),source_I_vertex_quotient=q,
   canonical_parent_union_schema=np.asarray('numi.run055.current-immediate-parent-unions.v1'),
   canonical_parent_union_operation_order=np.asarray('current620_I430'))
  for role2,e in cfg['candidate_surfaces'].items():
   payload['typed_'+role2+'_candidate_path']=np.asarray([e['path']]);payload['typed_'+role2+'_candidate_sha256']=np.asarray([e['sha256']])
  payload.pop('source_cell_operation_rows',None)
  d,qual=T.m.measurements(p,t);bad=int((~T.m.passes(d,qual)).sum());assert np.all(d>0) and bad==rr['bad_after']
  dest=out/(role.lower()+'-current-I430-canonical-volume-proposal.npz');np.savez_compressed(dest,**payload)
  written=load(dest);assert set(written)==set(payload)
  for k,v in payload.items():assert np.array_equal(written[k],v),k
  reports.append({'role':role,'candidate':pin(dest),'input':pin(basep),'raw':pin(rawp),'cells':len(t),'points':len(p),'native_quality_failures':bad,'nonpositive':0,'complete_target_faces':len(target),'missing':0,'extra':0,'changed_output_cells':len(changed_out),'full_endpoint_star_sum':total,'parent_union_components':ng,'used_parent_components':ncg,'parent_entries':len(parents)})
 result={'schema':'numi.run055.current-immediate-parent-unions.v1','status':'PASS_CANONICAL_CURRENT_PAIRED_I430_FULL_TARGET_AND_PARENT_UNIONS','roles':reports,'source_report':pin(rp),'typed_replay':pin(proof),'context':pin(ctx),'producer':pin(Path(__file__)),'physical_admission':False,'native_steps':0,'mass_mapping':'CURRENT_SOURCE330_ALLOCATION_REQUIRED','elapsed_s':time.monotonic()-start}
 (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
