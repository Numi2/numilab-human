"""Rebuild current O endpoint-star lineage without editing the verified geometry."""
from pathlib import Path
import json,hashlib,importlib.util,time,numpy as np
H=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def imp(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 tic=time.monotonic()
 root=H/'outer-O-current612-witness-pruned-root'
 rp=root/'run001/report.json';r=json.loads(rp.read_text());rr=r['roles'][0]
 ctx=H/'joint-current-O612-D597-context-613-root/audit354/report.json'
 assert sha(ctx)=='4f457bbbab1d858061ac141b383cff5648ff5d2080a2a39cc4b39853b6fdb73c'
 cr=json.loads(ctx.read_text());assert cr['status']=='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
 bp=H/'outer-O-current612-target614-root/receipt.json';br=json.loads(bp.read_text())
 assert br['status']=='PASS_EXACT_BOUNDED_O_QUOTIENT_PHYSICAL_TARGET_BIND_ONLY' and br['exact_context_report_sha256']==sha(ctx)
 basep=Path(rr['input']);boundp=Path(br['candidate']);assert sha(basep)==rr['input_sha256'] and sha(boundp)==br['candidate_sha256']
 base=load(basep);bound=load(boundp);p=base['points_world_m'];ot=base['tetrahedra'];om=base['material_indices']
 U=imp('canonical055',H/'paired-I-local-cavity-527-root/run055-current-I50-D264-source-I-baseline-context/canonicalize_run055_current_parent_unions_v3.py')
 T=imp('transfer209',H/'transfer_surface_contractions_to_volume_209_root.py')
 csr=U.incidence_csr(ot,len(p));q=np.arange(len(p),dtype=np.int64);stars=[]
 for op in rr['operations']:
  drop=op['retired_volume_node'];keep=op['kept_volume_node']
  assert drop!=keep and q[drop]==drop
  rows=np.asarray(op['source_cell_rows'],np.int64)
  assert np.array_equal(np.sort(rows),U.full_star(csr,[drop]))
  stars.append(U.full_star(csr,[keep,drop]));q[q==drop]=q[keep]
 mapped=q[ot];deg=np.any(np.diff(np.sort(mapped,axis=1),axis=1)==0,axis=1);origin=np.flatnonzero(~deg)
 t=mapped[origin];mat=om[origin]
 assert np.array_equal(t,bound['tetrahedra']) and np.array_equal(mat,bound['material_indices']) and np.array_equal(p.view('u4'),bound['points_world_m'].view('u4'))
 target=q[base['desired_boundary_faces']];target=target[~np.any(np.diff(np.sort(target,axis=1),axis=1)==0,axis=1)]
 bd=T.boundary(t)
 assert U.counter_faces(T,target)==U.counter_faces(T,bound['desired_boundary_faces'])==U.counter_faces(T,bd)
 changed=np.any(t!=ot[origin],axis=1);changed_out=np.flatnonzero(changed)
 rowowner,po,pr,ng,ncg,total=U.make_parent_unions(len(ot),stars,origin[changed])
 out_to_in=origin.copy();out_to_in[changed]=-1
 in_to_out=np.full(len(ot),-1,np.int64);in_to_out[origin[~changed]]=np.flatnonzero(~changed)
 payload=dict(base)
 for k in ('points_world_m','points_world_m_f32','tetrahedra','material_indices','boundary_faces','desired_boundary_faces'):payload[k]=bound[k]
 payload.update(current_geometry_parent_path=np.asarray([str(basep)]),current_geometry_parent_sha256=np.asarray([sha(basep)]),
  output_cell_to_input_cell=out_to_in,input_cell_to_output_cell=in_to_out,retained_input_cell_rows=origin[~changed],
  replacement_parent_offsets=po,replacement_parent_cell_rows=pr,replacement_tet_offsets=np.asarray([0,len(changed_out)],np.int64),
  replacement_tetrahedra=t[changed],input_node_to_output_node=q,output_node_to_input_node=np.arange(len(p),dtype=np.int64),
  output_point_to_input_point_row=np.arange(len(p),dtype=np.int64),output_point_parent_offsets=np.zeros(len(p)+1,np.int64),
  output_point_parent_input_cell_rows=np.empty(0,np.int64),source_O_vertex_quotient=q,
  canonical_parent_union_schema=np.asarray('numi.run055.current-immediate-parent-unions.v1'),
  canonical_parent_union_operation_order=np.asarray('O612_six_witness_pruned_operations'))
 config=json.loads(Path(cr['config']['path']).read_text())
 for role,e in config['candidate_surfaces'].items():
  payload['typed_'+role+'_candidate_path']=np.asarray([e['path']]);payload['typed_'+role+'_candidate_sha256']=np.asarray([e['sha256']])
 payload.pop('source_cell_operation_rows',None)
 det,qual=T.m.measurements(p,t);bad=int((~T.m.passes(det,qual)).sum());assert np.all(det>0) and bad==rr['bad_after']==873
 out=H/'outer-O-current612-canonical615-root';out.mkdir(exist_ok=False)
 cp=out/'skin-current-O612-canonical-parent-unions.npz';np.savez_compressed(cp,**payload)
 written=load(cp);assert set(written)==set(payload)
 for k,v in payload.items():assert np.array_equal(written[k],v),k
 result={'schema':'numi.run055.current-immediate-parent-unions.v1','status':'PASS_CANONICAL_CURRENT_O612_IMMEDIATE_PARENT_UNIONS','physical_admission':False,'native_steps':0,'mass_mapping':'CURRENT_SOURCE330_ALLOCATION_REQUIRED',
  'input':pin(basep),'candidate':pin(cp),'bound_target':pin(boundp),'source_report':pin(rp),'context':pin(ctx),'producer':pin(Path(__file__)),
  'input_cells':len(ot),'cells':len(t),'removed_cells':int(deg.sum()),'native_quality_failures':bad,'nonpositive':0,'operations':len(stars),
  'changed_output_cells':len(changed_out),'full_endpoint_star_rows_sum':total,'parent_union_components':ng,'components_used_by_changed_cells':ncg,
  'replacement_parent_row_entries':len(pr),'all_point_coordinate_bits_fixed':True,'unrelated_retained_cell_identity_exact':True,'complete_target_quotient_exact':True,
  'source330_mass':'IMMUTABLE_CURRENT_MAPPING_REQUIRED','elapsed_s':time.monotonic()-tic}
 (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
