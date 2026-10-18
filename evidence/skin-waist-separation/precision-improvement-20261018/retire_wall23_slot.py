from pathlib import Path
import json,hashlib,sys,numpy as np
sys.path.insert(0,'/Users/n/numi-human-closed-outer-skin-20261017/src')
from numilab_human.resting_pleura_proxy import _parse_payload
from numilab_human.resting_intercostal_mesh_arrangement import encode_payload_with_row_replacements
c=Path('/Users/n/numi-human-separation-20261018');base=c/'C/improve-wall23-003';out=c/'C/improve-wall23-map-retired-004';out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
raw=(base/'resting-thorax.nhanatomy').read_bytes();rows=_parse_payload(raw)[-1];r=rows[23];drop=26654;assert drop not in r['faces'];v=np.delete(r['vertices6'],drop,axis=0);f=r['faces'].copy();f[f>drop]-=1;assert np.array_equal(v[f],r['vertices6'][r['faces']]);assert len(np.unique(f))==len(v)
payload,abi=encode_payload_with_row_replacements(raw,{23:{**r,'vertices6':v,'faces':f}});p=out/'resting-thorax.nhanatomy';p.write_bytes(payload)
receipt=json.loads((base/'resting-anatomy-receipt.json').read_text());co=receipt['provenance']['cardiac_geometry_binding']['common_field'];mr=co['map'];source_map=base/mr['path'];assert sha(source_map)==mr['sha256'];m=np.fromfile(source_map,dtype='<f4').reshape(-1,7,4);vr=next(x for x in co['vertex_ranges'] if x['stable_id']==23);record=vr['first_vertex']+drop;newmap=np.delete(m,record,axis=0);mp=out/mr['path'];mp.write_bytes(newmap.tobytes());mr['record_count']-=1;mr['sha256']=sha(mp)
for group in ('vertex_ranges','passive_attachment_ranges'):
 for x in co[group]:
  if x['stable_id']==23 and group=='vertex_ranges':x['vertex_count']-=1
  elif x['first_vertex']>record:x['first_vertex']-=1
for key in ('domain_boxes','polynomials'):
 a=co[key];src=base/a['path'];assert sha(src)==a['sha256'];(out/a['path']).symlink_to(src.resolve())
old=sha(base/'resting-thorax.nhanatomy');new=sha(p);paths=[]
def walk(x,path=()):
 if isinstance(x,dict):
  for k,y in list(x.items()):
   if y==old:x[k]=new;paths.append('.'.join(path+(k,)))
   else:walk(y,path+(k,))
 elif isinstance(x,list):
  for i,y in enumerate(x):walk(y,path+(str(i),))
# Preserve historical repair provenance; update seven explicitly active payload identities only.
receipt['functional_bindings']['anatomy_payload_sha256']=new
for rib in (48,49,50):receipt['functional_bindings'][f'intercostal_313_rib{rib}_motion_attachment']['anatomy_payload_sha256']=new
receipt['passive_tissue_parent_bindings']['anatomy_payload_sha256']=new;receipt['payload'].update(path=str(p),sha256=new,bytes=len(payload),vertex_count=receipt['payload']['vertex_count']-1);co['anatomy_payload_sha256']=new
report=dict(base_payload_sha256=old,candidate_payload_sha256=new,source_map_sha256=sha(source_map),candidate_map_sha256=sha(mp),retired_source_vertex=drop,retired_common_map_record=record,source_allocated_vertices_before=len(r['vertices6']),source_allocated_vertices_after=len(v),all_emitted_triangle_vertex_attributes_identical=True,all_referenced_affine_displacement_records_identical=True,all_other_map_records_byte_exact=np.array_equal(np.delete(m,record,axis=0),newmap),all_referenced_vertices_contiguous=True,abi=abi,volume_bound_proof=str(base/'volume-perturbation.json'),volume_bound_proof_sha256=sha(base/'volume-perturbation.json'),no_force_mass_controller_or_tolerance_change=True,native_admission=False)
# Independently compare every triangle's common map coefficients before/after remapping.
old_coeff=m[vr['first_vertex']+r['faces']];new_coeff=newmap[vr['first_vertex']+f];assert np.array_equal(old_coeff,new_coeff)
(out/'allocation-retirement.json').write_text(json.dumps(report,indent=2)+'\n');receipt['provenance']['cardiac_wall23_unused_allocation_retirement']={**report,'report_sha256':sha(out/'allocation-retirement.json')};rp=out/'resting-anatomy-receipt.json';rp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
plan=json.loads((base/'trial-plan.json').read_text());plan.update(candidate_sha256=new,method_change='Retire unsupported unused source/map allocation together; emitted geometry and affine coefficients identical to source-checked candidate003. No native guard changes.');(out/'trial-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
for kind in ('held','released'):
 inv=json.loads((base/f'invocation-{kind}-001.json').read_text());argv=inv['argv'];argv[argv.index('--torso-anatomy-payload')+1]=str(p);argv[argv.index('--resting-anatomy-receipt')+1]=str(rp)
 for asset in (p,rp,mp,out/'allocation-retirement.json',out/'trial-plan.json'):inv['asset_sha256'][str(asset)]=sha(asset)
 inv['mechanical_trial']['allocation_retirement_trial_plan']=str(out/'trial-plan.json');(out/f'invocation-{kind}-001.json').write_text(json.dumps(inv,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2))
