from pathlib import Path
import sys,json,hashlib,argparse,numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276');B=A/'temporalis163-native-m6-001';O=A/'temporalis164-native-m6-002';P=Path('/Users/n/numi-human-complete-reference-inventory-1335')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
S,RP=map(Path,sys.argv[1:3]);proof=json.loads(RP.read_text())
assert proof['status']=='native_capture_local_repair_passed' and proof['inputs_unchanged'] and len(proof['poses'])==8
assert all(not r['local_bone32_external_pairs_after_repair'] and not r['local_self_pairs_after_repair'] for r in proof['poses'])
for p,v in proof['input_sha256'].items():assert sha(p)==v,p
C=Path(proof['candidate041_path']);assert sha(C)==proof['candidate041_sha256']
sys.path.insert(0,str(P/'src'));from numilab_human import resting_run as rr,passive_attachment_composition as pc
extra=pc._read_nhtiss4(S);record=next(r for r in extra['records'] if int(r[6])==164);local=pc._row_slices(extra,record);z=dict(np.load(C))
assert np.array_equal(np.asarray(local['local_faces']),z['faces'])
assert hashlib.sha256(local['binding_bytes']).hexdigest()==str(z['binding_record_sha256'][0])
vb=local['vertex_bytes'];n=int(record[3]);assert np.array_equal(np.ndarray((n,6),dtype='<f4',buffer=vb,offset=0,strides=(56,4)),z['vertices6'])
assert np.array_equal(np.ndarray((n,4),dtype='<u4',buffer=vb,offset=24,strides=(56,4)),z['binding_indices'])
assert np.array_equal(np.ndarray((n,4),dtype='<f4',buffer=vb,offset=40,strides=(56,4)),z['weights'])
O.mkdir(exist_ok=False);T=B/'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
composition=pc.append_passive_surfaces(T,S,O/'assets',stable_ids=(164,));payload=O/'assets'/T.name;receipt=O/'resting-anatomy-receipt.json';bound=pc.bind_anatomy_receipt(B/'resting-anatomy-receipt.json',payload,receipt)
assert bound['mass_geometry_accounting']==json.loads((B/'resting-anatomy-receipt.json').read_text())['mass_geometry_accounting']
parent=pc._read_nhtiss4(T);child=pc._read_nhtiss4(payload);assert parent['surface_count']==157 and child['surface_count']==158
for pr,cr in zip(parent['records'],child['records'][:157],strict=True):assert pc._row_local_equal(pc._row_slices(parent,pr),pc._row_slices(child,cr))
(O/'composition-result.json').write_text(json.dumps(composition,indent=2,sort_keys=True)+'\n')
bounds=dict(added_rows=[164],all157_parent_rows_exact=True,mass_accounting_exact=True,physical_routes_tendons_support_transforms_and_controller_unchanged=True,source_candidate_sha256=sha(C),local_eight_pose_repair_report_sha256=sha(RP),scope='Source-derived left temporalis passive reference surface, closed against registered skull and skin with retained ancestry. Single head attachment; no active jaw mechanics. Mixed-source inferred anatomy, not measurements of one individual.',qualification_limit='Experimental repair: local eight-pose reconstruction clear. Full25 offline screening and fresh native checks pending; no anatomy admission.',operating_point='Unchanged2Nm/rad aggregate passive MTP, real flat-plane contact and coupled GPU physiology.')
(O/'source-resolution-bounds.json').write_text(json.dumps(bounds,indent=2,sort_keys=True)+'\n')
bp=B/'baseline/run-declaration.json';base=json.loads(bp.read_text());idx=base['argv'].index('numilab_human.resting_run');av=base['argv'][idx+1:]
for flag,value in [('--output',str(O/'baseline/native-run')),('--anatomy-receipt',str(receipt))]:av[av.index(flag)+1]=value
if '--inspection-tour' not in av:av.append('--inspection-tour')
if '--inspection-period-seconds' not in av:av.extend(['--inspection-period-seconds','2.0'])
parser=argparse.ArgumentParser();rr.add_arguments(parser);args=parser.parse_args(av);native,assets=rr.command(args)
env=dict(x.split('=',1) for x in base['argv'][2:idx-2]);env['NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT']=str(args.output/'common-field-failure.json');env['DYLD_LIBRARY_PATH']=':'.join(str(Path(p).resolve()) for p in env['DYLD_LIBRARY_PATH'].split(':'))
env['NUMI_HUMAN_RESTING_INSPECTION_TOUR']='1';env['NUMI_HUMAN_RESTING_INSPECTION_PERIOD_SECONDS']='2.0'
runtime_rr=Path(env['PYTHONPATH'])/'numilab_human/resting_run.py';assert sha(runtime_rr)==sha(Path(rr.__file__))
argv=['/usr/bin/env','-i']+[k+'='+v for k,v in sorted(env.items())]+['/Users/n/numi-human-prep-venv-20261005/bin/python3.13','-m','numilab_human.resting_run']+av
launch=O/'launch_arm.py';launch.write_bytes((B/'launch_arm.py').read_bytes())
paths=[Path(__file__),bp,launch,RP,C,S,S.with_suffix('.manifest.json'),Path(rr.__file__),runtime_rr,Path(pc.__file__),O/'source-resolution-bounds.json',A/'verify_temporalis164_native_m6_002.py'];pins={**assets,**{str(p):sha(p) for p in paths}}
decl=dict(schema='numi.human.native-smoke.run-declaration.v1',seconds=16.,accepted_steps=8000,capture_steps=[0,4767,5023,5599,6207,6815,7423,8000],immutable_assets=pins,argv=argv,launch_adapter=dict(path=str(launch),sha256=sha(launch)),owner_cli_preview=dict(native_argv=native,asset_sha256=assets,recorded_invocation_environment=rr.invocation_environment(env)),execution_host='numi',hardware='Apple M6,12GPU cores,16GiB,macOS27.0(26A425)',qualification='Integrated native temporalis164 passive anatomy increment. All157 parent tissue rows and physical mass/routes/tendons/support/GPU physiology unchanged. Local eight-pose repair screen passed; full25 offline and native geometry and physiology parity require verification. No five-minute or whole-anatomy completion claim.')
(O/'baseline').mkdir();dp=O/'baseline/run-declaration.json';dp.write_text(json.dumps(decl,indent=2,sort_keys=True)+'\n');(O/'launch-guard.json').write_text(json.dumps(dict(launcher_sha256=sha(launch),arms=dict(baseline=dict(run_declaration_sha256=sha(dp),native_output=str(args.output)))),indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(path=str(O),tissue_sha256=sha(payload),pins=len(pins))),flush=True)

