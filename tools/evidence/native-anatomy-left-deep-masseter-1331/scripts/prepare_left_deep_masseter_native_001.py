from pathlib import Path
import sys,json,hashlib,struct,argparse,copy,numpy as np
R=Path('/Users/n/numi-human-retained-delivery-20261009');A=R/'anatomy-completion-1276'
B=A/'sartorius50-precision-native-001';O=A/'left-deep-masseter-native-001';O.mkdir(exist_ok=False)
D=R/'head-muscle-coverage-1291/alternate-head-source-inventory-001/z-anatomy-range/head-bone-triangle-registration-001/deep-temporalis-correction-001/deep-left-rib-shell-subtraction-003/subtract-superficial-160-004'
C=D/'result-003/stable-161-deep-left-bone-and-superficial-clearance.npz';RP=D/'final-recheck-002/report.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(C)=='e0670f41250b0d3c9495f195159589c56b27c8b10c3ab6db00fdeaa1012c95ca'
assert sha(RP)=='5c20857dd1480978b7b3e7854646e40342f64c10b7243cd209bb4d357201791c'
proof=json.loads(RP.read_text());assert proof['inputs_unchanged'] and proof['external_checks']['all_target_pairs_zero'] and proof['pose_checks']['all_self_topology_pass']
for p,d in proof['input_pins_sha256'].items():assert sha(p)==d,p
sys.path.insert(0,'/Users/n/numi-human-anatomy-publication-1308/src')
from numilab_human import passive_attachment_composition as pc
T=B/'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue';parent=pc._read_nhtiss4(T)
pm=json.loads(T.with_suffix('.manifest.json').read_text());assert parent['surface_count']==154
row=next(r for r in parent['records'] if int(r[6])==160);local=pc._row_slices(parent,row)
assert sha(B/'native-verification-001/report.json')
assert hashlib.sha256(local['binding_bytes']).hexdigest()==proof['candidate']['binding_sha256']
z=np.load(C,allow_pickle=False);p=np.asarray(z['positions_m'],dtype='<f4');f=np.asarray(z['faces'],dtype='<u4')
assert int(z['stable_id'][0])==161 and int(z['core_body_index'][0])==23
assert np.all(z['weights'][:,0]==1) and np.all(z['weights'][:,1:]==0) and np.all(z['binding_indices'][:,0]==0)
tri=p.astype(float)[f];acc=np.zeros_like(p,dtype=float);n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
for j in range(3):np.add.at(acc,f[:,j],n)
length=np.linalg.norm(acc,axis=1);assert np.all(length>1e-20);norm=np.asarray(acc/length[:,None],dtype='<f4')
assert np.sum(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])))/6>0
vb=b''.join(struct.pack('<6f4I4f',*p[i],*norm[i],0,0xffffffff,0xffffffff,0xffffffff,1,0,0,0) for i in range(len(p)))
raw=struct.pack('<8s6I32s',b'NHTISS4\0',5,1,1,len(p),f.size,parent['fingerprint'],parent['source_digest'])+struct.pack('<8I',0,1,0,len(p),0,f.size,162,1)+local['binding_bytes']+vb+f.tobytes()
sub=O/'subset';sub.mkdir();payload=sub/'left-deep-masseter.nhtissue';payload.write_bytes(raw)
meta=copy.deepcopy(next(r for r in pm['source']['surfaces'] if r['stable_id']==160))
meta.update(stable_id=162,member_id='FMA49005',source_name='left deep masseter',vertex_count=len(p),triangle_count=len(f))
meta.pop('source_geometry_bounds',None)
ext=meta['external_source']
for k in list(ext):
 if k.startswith('harmonic_source_'):ext.pop(k)
ext.update(source_object_name='Deep part of masseter.l',source_member_id='FMA49005',source_correction='Registered mixed-source boundary reconciliation against temporal bone, mandible and left superficial masseter with inferred 10 micrometre precision reserve; total volume removed 1.389 percent. Passive visual reference, not measured anatomy.',clearance_report_path=str(RP),clearance_report_sha256=sha(RP),candidate_npz_path=str(C),candidate_npz_sha256=sha(C),identity_mapping={'historical_offline_trial_stable_id':161,'canonical_stable_id':162,'reason':'Trial assigned left source object to right reserved ID. Canonical left deep masseter162 restored before first native composition; original trial files retained. Geometry and weights unchanged.'})
meta['source_geometry_bounds']={'candidate':proof['candidate'],'cut':proof['cut'],'attachment_proxies':proof['attachment_proxies']}
sm={'schema':pm['schema'],'payload':{'file':payload.name,'sha256':sha(payload),'bytes':len(raw),'surface_count':1,'binding_count':1,'vertex_count':len(p),'index_count':f.size,'registration_fingerprint32':f'{parent["fingerprint"]:08x}'},'source':{k:copy.deepcopy(pm['source'][k]) for k in ('bodyparts','myosim_source_archive_sha256','registration','myosim_manifest')},'coverage':{'configured_surface_count':1,'emitted_surface_count':1,'muscle_surface_count':1,'tendon_surface_count':0},'runtime_binding':'Existing single head core23 passive support; no active jaw mechanics.','evidence_boundary':'Inferred registered reference anatomy. Not measured individual or certified attachment footprints.'}
sm['source']['surfaces']=[meta];payload.with_suffix('.manifest.json').write_text(json.dumps(sm,indent=2,sort_keys=True)+'\n')
composition=pc.append_passive_surfaces(T,payload,O/'assets',stable_ids=(162,));child=O/'assets'/T.name
receipt=O/'resting-anatomy-receipt.json';bound=pc.bind_anatomy_receipt(B/receipt.name,child,receipt)
assert bound['mass_geometry_accounting']==json.loads((B/receipt.name).read_text())['mass_geometry_accounting']
(O/'composition-result.json').write_text(json.dumps(composition,indent=2,sort_keys=True)+'\n')
(O/'source-resolution-bounds.json').write_text(json.dumps({'162':meta,'all154_parent_rows_exact':True,'physical_mass_unchanged':True},indent=2,sort_keys=True)+'\n')
# The prepared run retains the frozen CLI/runtime; the published composer is asset preparation only.
import importlib.util
spec=importlib.util.spec_from_file_location('frozen_resting_run','/Users/n/numi-human-local-clearance-preservation-1280/src/numilab_human/resting_run.py')
# Package imports resolve byte-identical resting_run helpers; explicit module source pinned.
from numilab_human import resting_run as rr
bp=B/'baseline/run-declaration.json';base=json.loads(bp.read_text());idx=base['argv'].index('numilab_human.resting_run');av=base['argv'][idx+1:]
for flag,val in (('--output',str(O/'baseline/native-run')),('--anatomy-receipt',str(receipt))):av[av.index(flag)+1]=val
parser=argparse.ArgumentParser();rr.add_arguments(parser);args=parser.parse_args(av);native,assets=rr.command(args)
env=dict(x.split('=',1) for x in base['argv'][2:idx-2]);env['NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT']=str(args.output/'common-field-failure.json')
argv=['/usr/bin/env','-i']+[k+'='+v for k,v in sorted(env.items())]+['/Users/n/numi-human-prep-venv-20261005/bin/python3.13','-m','numilab_human.resting_run']+av
launch=O/'launch_arm.py';launch.write_text((B/'launch_arm.py').read_text())
pins={**assets,**proof['input_pins_sha256'],**{str(x):sha(x) for x in (Path(__file__),Path(pc.__file__),Path(rr.__file__),bp,launch,C,RP,payload,payload.with_suffix('.manifest.json'),O/'source-resolution-bounds.json')}}
decl={'schema':'numi.human.native-smoke.run-declaration.v1','seconds':16.0,'accepted_steps':8000,'capture_steps':[0,4767,5023,5599,6207,6815,7423,8000],'immutable_assets':pins,'argv':argv,'launch_adapter':{'path':str(launch),'sha256':sha(launch)},'owner_cli_preview':{'native_argv':native,'asset_sha256':assets,'recorded_invocation_environment':rr.invocation_environment(env)},'qualification':'Left deep masseter canonical162 added to current154 rows. Historical trial161 mapped explicitly to162; sourcegeometry/bindings unchanged. Eight offlineposes self/targets clear. Native geometry validation pending; passivejaw reference only. All priorrows/mass/routes/physics unchanged. Wholebody completion and final five-minute pair remain outstanding. Performance deferred.'}
(O/'baseline').mkdir();dp=O/'baseline/run-declaration.json';dp.write_text(json.dumps(decl,indent=2,sort_keys=True)+'\n')
(O/'launch-guard.json').write_text(json.dumps({'launcher_sha256':sha(launch),'arms':{'baseline':{'run_declaration_sha256':sha(dp),'native_output':str(args.output)}}},indent=2,sort_keys=True)+'\n')
print(json.dumps({'output':str(O),'tissue_sha256':sha(child),'rows':155,'canonical_id':162,'candidate_geometry_unchanged':True,'pins':len(pins)}))
