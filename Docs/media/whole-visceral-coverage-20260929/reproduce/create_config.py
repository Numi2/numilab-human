from pathlib import Path
import json,copy,hashlib
from numilab_human.physiology import load_anatomy
repo=Path(__file__).resolve().parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda p,d:p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
old=repo/'config/source-organ-family-composite.v1.json';config=json.loads(old.read_text())
an=load_anatomy(repo/'Sources')
base=json.loads((repo/'Build/organ-family-coverage-20260929/payload.final/source-organ-family-anatomy.manifest.json').read_text())
members=list(base['selection']['baseline_source_members'].values())+base['selection']['added']
baseline_path=repo/'config/source-organ-family-baseline-members.v2.json'
write(baseline_path,{'schema':'numi.human.source-organ-family-baseline-members.v2','source_payload_sha256':base['payload']['sha256'],'entries':members})
graph=json.loads((repo/config['template']['file']).read_text());bindings=copy.deepcopy(config['family_bindings'])
targets=[('brain','brain','part_of','head'),('small_intestine','small intestine','part_of','Abdomen'),
 ('large_intestine','large intestine','part_of','Abdomen'),('rectum','rectum','part_of','pelvis'),
 ('esophagus','esophagus','part_of','torso'),('trachea','trachea','part_of','neck'),
 ('gallbladder','gallbladder','part_of','Abdomen'),('spleen','spleen','is_a','Abdomen'),
 ('thymus','thymus','part_of','torso'),('right_adrenal','right adrenal gland','part_of','Abdomen'),
 ('left_adrenal','left adrenal gland','part_of','Abdomen'),('right_ureter','right ureter','part_of','Abdomen'),
 ('left_ureter','left ureter','part_of','Abdomen'),('urinary_bladder','urinary bladder','part_of','pelvis'),
 ('urethra','urethra','part_of','pelvis'),('prostate','prostate','part_of','pelvis'),
 ('right_testis','right testis','is_a','pelvis'),('left_testis','left testis','is_a','pelvis'),
 ('epididymides','epididymis','is_a','pelvis'),('seminal_vesicles','seminal vesicle','is_a','pelvis'),
 ('penile_spongiosum','corpus spongiosum of penis','is_a','pelvis'),
 ('penile_cavernosum','corpus cavernosum of penis','is_a','pelvis'),('glans_penis','glans penis','is_a','pelvis'),
 ('salivary_glands','salivary gland','is_a','head'),('right_lacrimal_gland','right lacrimal gland','part_of','head'),
 ('left_lacrimal_gland','left lacrimal gland','part_of','head'),('right_eye','right eye','part_of','head'),
 ('left_eye','left eye','part_of','head')]
for ident,name,hierarchy,owner in targets:
 matches=[(key,ms) for key,ms in an['tables'][hierarchy].items() if key[1]==name];assert len(matches)==1,name
 (concept,_),ms=matches[0]
 spec={'id':ident,'concept_id':concept,'source_name':name,'hierarchy':hierarchy,'selection':'complete_membership','member_ids':sorted(ms)}
 graph['regions'].append(spec);bindings[ident]={'concept_id':concept,'source_name':name,'hierarchy':hierarchy,'myosim_body':owner}
graph['schema']='numi.human.source-organ-family-template.v2'
graph['boundary']='Cumulative exact source families. Reference geometry only; not exhaustive whole-body, connected tissue, mechanics or physiological qualification.'
graph_path=repo/'config/source-organ-family-template.v2.json';write(graph_path,graph)
registry=config['layer_registry']
duct_type=next((c,n) for c,n in an['tables']['is_a'] if n=='segment of lacrimal duct')
registry['duct']['source_types'].append(list(duct_type))
registry['neural_region_reference']={'layer_code':11,'semantic':51027,'required_family':'brain',
 'source_types':[['FMA67165','material anatomical entity']],
 'excluded_source_types':[['FMA78447','region of ventricular system of brain']]}
registry['ventricular_region_reference']={'layer_code':12,'semantic':51028,'required_family':'brain',
 'source_types':[['FMA78447','region of ventricular system of brain']]}
registry['junction_reference']={'layer_code':13,'semantic':51029,
 'source_types':[['FMA5898','anatomical junction']]}
registry['ocular_muscle_reference']={'layer_code':15,'semantic':51031,'required_families':['right_eye','left_eye'],
 'source_types':[['FMA5022','muscle organ']]}
registry['ocular_region_reference']={'layer_code':14,'semantic':51030,'required_families':['right_eye','left_eye'],
 'source_types':[['FMA61775','physical anatomical entity']],
 'excluded_source_types':registry['cavity_reference']['source_types']+registry['duct']['source_types']+
 registry['vessel']['source_types']+registry['junction_reference']['source_types']+registry['ocular_muscle_reference']['source_types']}
rectum=next(ms for (c,n),ms in an['tables']['part_of'].items() if n=='rectum');assert len(rectum)==1
config['member_owner_overrides'][next(iter(rectum))]='pelvis'
by_member={r['member_id']:r for r in members}
for region in graph['regions']:
 for member in region['member_ids']:
  if member in by_member and by_member[member]['myosim_body'] != bindings[region['id']]['myosim_body']:
   config['member_owner_overrides'][member]=by_member[member]['myosim_body']
config.update(schema='numi.human.source-organ-family-composite.v2',
 configuration_file='config/source-organ-family-composite.v2.json',family_bindings=bindings,
 base_surface_count=387,base_bodyparts_surface_count=381,base_payload_abi=4,payload_abi=5,
 base_payload_sha256=base['payload']['sha256'],
 baseline_map={'file':str(baseline_path.relative_to(repo)),'sha256':sha(baseline_path)},
 template={'file':str(graph_path.relative_to(repo)),'sha256':sha(graph_path)},
 prefix_configuration={'file':str(old.relative_to(repo)),'sha256':sha(old)},
 boundary='Cumulative source-family reference geometry with explicit source muscle, neural, ventricular, ocular, duct, cavity and junction distinctions. Single-link kinematic bindings retain source atlas positions and fixed source head/neck behavior. Not exhaustive whole-body coverage, clinical registration, disjoint tissue, connected lumen, fluid or mechanical volume, material, independent eye/cervical motion, physiology or control.')
write(repo/config['configuration_file'],config)
print('families',len(bindings),'duct_type',duct_type)
