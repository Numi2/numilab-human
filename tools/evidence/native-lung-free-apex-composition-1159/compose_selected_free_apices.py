"""Compose the selected microscopic free-apex repairs through the existing owner."""
from pathlib import Path
import sys,copy,json,hashlib,importlib.util,argparse
import numpy as np
E=Path('/Users/n/numi-human-resting-evidence-20261005')
PUB=Path('/Users/n/numi-human-final-compose-publication-001/tools/evidence/native-lung-final-selected-composition-1113/compose_candidate_v8.py')
spec=importlib.util.spec_from_file_location('frozen_v8_composer',PUB);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
from numilab_human.resting_pleura_proxy import build_candidate
BASE=E/'native-lung-final-metadata-refresh-1116/final'
V8=E/'native-lung-final-selected-composition-dryrun-1113/provisional-1105-1106-v8'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--operations',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 specs=c.read(a.operations);out=a.out;out.mkdir(exist_ok=False)
 pins={str(Path(__file__).resolve()):c.sha(Path(__file__)),str(PUB):c.sha(PUB),str(a.operations):c.sha(a.operations),str(c.OWNER):c.PINS[c.OWNER],str(c.RESP):c.PINS[c.RESP]}
 for name in ['resting-thorax.nhanatomy','resting-anatomy-receipt.json','resting-anatomy-manifest.json','resting-reference-respiration.json']:
  p=BASE/name;pins[str(p)]=c.sha(p)
 for name in ['composition-report.json','final/resting-thorax.nhanatomy','final-exact-D311-to-lobes-map.jsonl','current-reciprocal-map-report-v2.json','current-reciprocal-face-map-v2.jsonl','current-reciprocal-edge-map-v2.jsonl']:
  p=V8/name;pins[str(p)]=c.sha(p)
 for key in ['base_1078_payload']:
  pin=c.read(BASE/'resting-anatomy-receipt.json')['provenance']['lung_final_shared_interface_rebuild'][key];pins[pin['path']]=pin['sha256']
 p=E/'native-lung-conditioned-final-compose-1078/final-exact-D311-to-lobes-map.jsonl';pins[str(p)]=c.sha(p)
 for name in ['resting_pleura_proxy.py','resting_anatomy_interface_patch.py','resting_lung_edge_repair.py','cardiac_cavity_geometry.py']:
  p=c.SRC/'numilab_human'/name;pins[str(p)]=c.sha(p)
 for item in specs:
  for pin in item['pins']:
   p=Path(pin['path'])
   if c.sha(p)!=pin['sha256']:raise ValueError('operation input pin mismatch '+str(p))
   pins[str(p)]=pin['sha256']
 for p,h in pins.items():
  if c.sha(p)!=h:raise ValueError('input changed '+p)
 if c.sha(BASE/'resting-thorax.nhanatomy')!='1b62f569f218cf8ba45f26c393bfaa4c8ed2ff499ab500c002dbbb006747b8bc':raise ValueError('unexpected source')
 c.write(out/'input-pins.json',pins)
 head,base=c.parse_payload(BASE/'resting-thorax.nhanatomy')
 raw=(BASE/'resting-thorax.nhanatomy').read_bytes()
 order=[int(c.RECORD.unpack_from(raw,c.HEADER.size+i*c.RECORD.size)[5]) for i in range(int(head[2]))]
 prepared={s:(r['vertices6'].copy(),r['faces'].copy(),list(range(len(r['faces']))),{}) for s,r in base.items()}
 ops=[];cumulative={};groups={}
 for i,item in enumerate(specs):
  args=copy.deepcopy(item['arguments'])
  # Apply the original owner constraints to the cumulative composed state.
  args['cumulative_volume_change_m3']=cumulative
  args['cumulative_volume_group_change_m3']=groups
  op=c.open_registered_shared_vertex_star(prepared,**args)
  if op['owner_surface_ids']!=item['expected_owner_ids']:raise ValueError('unexpected coordinate owners')
  if np.asarray(op['new_point'],dtype='<f4').tobytes()!=np.asarray(item['expected_new_point'],dtype='<f4').tobytes():raise ValueError('owner replay differs selected candidate')
  for sid,v in op['per_surface_signed_volume_delta_m3'].items():cumulative[int(sid)]=cumulative.get(int(sid),0)+v
  groups=op['cumulative_volume_group_change_m3']
  ops.append({'label':item['label'],'arguments':item['arguments'],'result':op,'selection_evidence':item['pins']})
  c.write(out/'operation-progress.json',ops);print('owner operation',i,item['label'],'accepted',flush=True)
 rows=copy.deepcopy(base)
 for s,(v,f,_,_) in prepared.items():rows[s]['vertices6']=v;rows[s]['faces']=f
 changed={}
 for s in rows:
  if not np.array_equal(rows[s]['faces'],base[s]['faces']):raise ValueError('selected operation changed face topology')
  ids=np.flatnonzero(np.any(rows[s]['vertices6'][:,:3].view(np.uint32)!=base[s]['vertices6'][:,:3].view(np.uint32),axis=1))
  if len(ids):changed[str(s)]={'vertices':ids.tolist(),'faces':np.flatnonzero(np.any(np.isin(rows[s]['faces'],ids),axis=1)).tolist(),'maximum_move_m':float(np.max(np.linalg.norm(rows[s]['vertices6'][ids,:3].astype('f8')-base[s]['vertices6'][ids,:3].astype('f8'),axis=1)))}
 if set(map(int,changed))-{305,306,307,308,310}:raise ValueError('unintended owner changed')
 mapchecks={}
 for name,kind in [('current-reciprocal-face-map-v2.jsonl','ll'),('final-exact-D311-to-lobes-map.jsonl','d')]:
  records=[json.loads(x) for x in (V8/name).read_text().splitlines()]
  for m in records:
   pair=m['pair'] if kind=='ll' else [311,m['lobe_stable_id']]
   faces=m['face_rows'] if kind=='ll' else [m['d_face_row'],m['l_face_row']]
   pts=[]
   for s,f in zip(pair,faces):
    before=base[s]['vertices6'][base[s]['faces'][f],:3];after=rows[s]['vertices6'][rows[s]['faces'][f],:3]
    if not np.array_equal(before.view(np.uint32),after.view(np.uint32)):raise ValueError('registered interface moved')
    pts.append(c.ori(rows[s],rows[s]['faces'][f]))
   if not c.opposite(*pts):raise ValueError('interface winding mismatch')
  mapchecks[kind]={'count':len(records),'source_registered_triangles_exact':True,'opposite_winding':True,'map':c.meta(V8/name)}
 pre=out/'pre-pleura';final=out/'final';pre.mkdir();final.mkdir()
 prenha=pre/'resting-thorax.nhanatomy';prenha.write_bytes(c._serialize_payload(head,order,rows))
 rec0=c.read(BASE/'resting-anatomy-receipt.json');rec=copy.deepcopy(rec0)
 rec['provenance']['retained_parent_lung_interface_rebuild']=copy.deepcopy(rec['provenance']['lung_final_shared_interface_rebuild'])
 rec['provenance']['lung_final_shared_interface_rebuild'].update(status='source candidate; native acceptance pending',parent_payload=c.meta(BASE/'resting-thorax.nhanatomy'),parent_composition=c.meta(V8/'composition-report.json'),selected_free_apex_operations=ops)
 rec.setdefault('qualification',{})['source_geometry_candidate']=str(len(ops))+' bounded external free-apex operations on inferred reference anatomy; source checks only until actual native acceptance.'
 rec['provenance']['lung_final_shared_interface_rebuild']['operation_308_first_cluster']['parent_coverage_scope']='Historical intermediate pre-pleura collapse lineage retained from V8; synthetic stage parents are superseded for current row310 by the complete current row310_face_lineage_path copy proof.'
 c.update_bind(rec,prenha,rows,rec0);c.copy_sidecars(rec,pre,BASE)
 pr=pre/'resting-anatomy-receipt.json';c.write(pr,rec)
 # Reuse the existing pleura union owner; independently check complete row310 identity.
 pleura=build_candidate(prenha,pr,final)
 fnha=final/'resting-thorax.nhanatomy';frp=final/'resting-anatomy-receipt.json';fr=c.read(frp);_,frs=c.parse_payload(fnha)
 for s in rows:
  if not np.array_equal(rows[s]['vertices6'].view(np.uint32),frs[s]['vertices6'].view(np.uint32)) or not np.array_equal(rows[s]['faces'],frs[s]['faces']):raise ValueError('recooked exterior differs atomic owner row '+str(s))
 lineage=c._pleura_face_lineage(frs);lp=out/'row310-face-lineage.npy';np.save(lp,lineage,allow_pickle=False)
 c.update_bind(fr,fnha,frs,rec0);c.copy_sidecars(fr,final,pre)
 fr['provenance']['lung_final_shared_interface_rebuild'].update(output_payload_sha256=c.sha(fnha),pre_pleura_payload_sha256=c.sha(prenha),row310_face_lineage_path=str(lp),row310_face_lineage_sha256=c.sha(lp),pleura_owner_derivation=pleura['derivation'])
 # Existing owner derives the runtime area on exact emitted geometry. If it changes,
 # update the one derived parameter explicitly and retain all other physical values.
 c.CFG=BASE/'resting-reference-respiration.json'
 cfg0=c.read(c.CFG);cfg=copy.deepcopy(cfg0)
 area64,_=c._derive_basal_effective_area(frs,c._load_pinned_respiratory_owner(c.RESP).kuhn_basis)
 cfg['diaphragm_area_m2']=float(np.float32(area64))
 refresh=c.refresh_respiration_geometry_config(cfg,fnha,frs,final,'bounded-free-apex successor')
 actualcfg=c.read(refresh['config_path'])
 same=np.float32(cfg0['diaphragm_area_m2']).tobytes()==np.float32(cfg['diaphragm_area_m2']).tobytes()
 actualcfg['parameter_scope']['diaphragm_area_refinement_note']='Runtime-matching analytic signed-volume derivative recomputed from emitted Float32 geometry; parent Float32 equality: '+str(same)+'. Inferred reference parameter, not measured subject data.'
 actualcfg['parameter_scope']['source_retriangulated_reference_geometry']='Bounded external star conditioning; all registered interface coordinates unchanged. Only the derived diaphragm area may change.'
 c.write(refresh['config_path'],actualcfg);refresh['config_sha256']=c.sha(refresh['config_path'])
 # refresh accepted cfg as its input; record the real unchanged parent identity and value.
 deriv=c.read(refresh['derivation_path']);deriv['input_runtime_area_m2']=float(cfg0['diaphragm_area_m2']);deriv['runtime_area_f32_equal_to_parent']=same;c.write(refresh['derivation_path'],deriv);refresh['derivation_sha256']=c.sha(refresh['derivation_path'])
 actualcfg['source_geometry_candidate']['derivation_sha256']=refresh['derivation_sha256'];c.write(refresh['config_path'],actualcfg);refresh['config_sha256']=c.sha(refresh['config_path'])
 refresh['binding_check']=c.validate_current_respiration_geometry_binding(actualcfg,c.sha(fnha),refresh['geometry_sha256'],refresh['area32'])
 before=c._physical_config_without_geometry_metadata(cfg0);after=c._physical_config_without_geometry_metadata(actualcfg);before['diaphragm_area_m2']=after['diaphragm_area_m2']
 if before!=after:raise ValueError('unrelated physical parameter changed')
 c.bind_receipt_respiration(fr,refresh);fr['provenance']['respiratory_configuration']['effective_area_f32_equal_to_parent']=same
 c.write(frp,fr)
 fm=c.read(BASE/'resting-anatomy-manifest.json')
 for k in ['payload','functional_bindings','qualification','thorax_source_volume_m3']:fm[k]=copy.deepcopy(fr[k])
 fm['receipt'].update(path=str(frp),sha256=c.sha(frp))
 fm.setdefault('source_receipt_lineage',{})['current_respiratory_configuration']={'path':str(refresh['config_path']),'sha256':refresh['config_sha256'],'derivation_path':str(refresh['derivation_path']),'derivation_sha256':refresh['derivation_sha256'],'bound_geometry_path':str(fnha),'bound_geometry_sha256':c.sha(fnha)}
 c.write(final/'resting-anatomy-manifest.json',fm)
 lr=c.read(V8/'current-reciprocal-map-report-v2.json');lr['final_nha']=c.meta(fnha);lr['status']='source-validated current successor; native pending';lr['parent_v8_report']=c.meta(V8/'current-reciprocal-map-report-v2.json');lr['mapped_triangle_identity_transfer']=mapchecks['ll'];c.write(out/'current-reciprocal-map-report-v2.json',lr)
 normals={str(s):c.verify_normals(frs[s],s) for s in range(305,312)}
 topology={}
 for s in range(305,312):
  t=c.analyze_topology(frs[s]['vertices6'][:,:3].tolist(),frs[s]['faces'].tolist())
  # Function signature and complete closed-manifold result must remain authoritative.
  defects={k:len(t.get(k,[])) for k in ['degenerate_face_ids','duplicate_face_ids','nonmanifold_edges','orientation_defect_edges','unused_vertex_ids','vertex_manifold_defect_ids','repeated_vertex_face_ids']}
  if not t.get('closed_oriented_manifold_candidate') or t.get('boundary_edge_count')!=0 or any(defects.values()):raise ValueError('invalid closed topology '+str(s))
  if t.get('euler_characteristic')!=(-32 if s==310 else -2 if s==311 else 2) or t.get('face_component_count')!=(2 if s==310 else 1):raise ValueError('changed Euler/components '+str(s))
  topology[str(s)]={k:t.get(k) for k in ['vertex_count','face_count','euler_characteristic','face_component_count','boundary_edge_count','closed_oriented_manifold_candidate']}
 report={'schema':'numi.human.final-lung-selected-composition-dryrun-v1','status':'source candidate complete; actual native geometry and physiology pending','inputs':pins,'operations':ops,'changed_rows':changed,'map_identity_transfer':mapchecks,'row310_recook_all_arrays_exact':True,'row310_lineage_count':len(lineage),'final_topology':topology,'final_normal_checks':normals,'runtime_area_f32_unchanged':same,'area64':refresh['area64'],'area32':refresh['area32'],'respiration_geometry_refresh':refresh['binding_check'],'outputs':{k:c.meta(p) for k,p in [('pre_nha',prenha),('pre_receipt',pr),('final_nha',fnha),('final_receipt',frp),('final_manifest',final/'resting-anatomy-manifest.json'),('final_respiration_config',refresh['config_path']),('final_area_derivation',refresh['derivation_path']),('d_map',V8/'final-exact-D311-to-lobes-map.jsonl'),('ll_map',V8/'current-reciprocal-face-map-v2.jsonl'),('ll_edges',V8/'current-reciprocal-edge-map-v2.jsonl'),('row310_lineage',lp)]}}
 c.write(out/'composition-report.json',report)
 print(json.dumps({'report':c.meta(out/'composition-report.json'),'changed_rows':changed,'area_f32_same':same,'area64':refresh['area64']}),flush=True)
if __name__=='__main__':main()
