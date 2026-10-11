from pathlib import Path
import sys,json,hashlib,argparse
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276');B=A/'right-pq-native-mtp-two-001';O=A/'left-fds-native-m6-001';P=Path('/Users/n/numi-human-complete-reference-inventory-1335');M=A/'m6-native-current-anatomy-baseline-004'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
CP=Path(sys.argv[1]);LP=Path('/Users/n/numi-human-fds132-m6-1350/attempt-049/audit-all25/result-002/report.json');SP=Path('/Users/n/numi-human-fds132-m6-1350/attempt-049/result-001/proposal-report.json')
l=json.loads(LP.read_text());c=json.loads(CP.read_text())
for path,d in [(LP,l),(CP,c)]:
 assert d['complete'] and d['inputs_unchanged'],path
 for p,v in d['pins'].items():assert sha(p)==v,p
assert l['source_and_all25_sampled_pose_self_clear'] and l['all25_retained_targets_no_added_pairs'] and len(l['poses'])==25
assert c['all_components_positive'] and c['faces_exact_source_reference'] and c['face_origins_exact_source_reference'] and c['lineage_parent_face_range_valid'] and c['binding_indices_exact_source_reference'] and c['weights_exact_source_reference'] and c['pure_onehot_proxy_coordinates_and_weights_exact_source_reference']
assert c['max_vertex_displacement_from_source_mm']<2 and max(abs(c[k]) for k in ('source_relative_area_change','source_relative_volume_change'))<.01
C=Path(l['candidate_path']);assert sha(C)==l['candidate_sha256']==c['candidate_sha256']=='7d072ccbadbb13fdcad255b10c6ab1bf692c442b47fb12b917a8e68e80d7d0cf'
sys.path.insert(0,str(P/'src'));from numilab_human import resting_run as rr,passive_attachment_composition as pc
O.mkdir(exist_ok=False);T=B/'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue';composition=pc.compose(T,O/'assets',[(132,C,LP)],reference_surface_rows=(132,));payload=O/'assets'/T.name;receipt=O/'resting-anatomy-receipt.json';bound=pc.bind_anatomy_receipt(B/'resting-anatomy-receipt.json',payload,receipt)
assert bound['mass_geometry_accounting']==json.loads((B/'resting-anatomy-receipt.json').read_text())['mass_geometry_accounting']
parent=pc._read_nhtiss4(T);child=pc._read_nhtiss4(payload);assert parent['surface_count']==child['surface_count']==156
for pr,cr in zip(parent['records'],child['records'],strict=True):
 if int(pr[6])!=132:assert pc._row_local_equal(pc._row_slices(parent,pr),pc._row_slices(child,cr))
(O/'composition-result.json').write_text(json.dumps(composition,indent=2,sort_keys=True)+'\n')
bounds=dict(changed_rows=[132],all155_other_rows_exact=True,mass_accounting_exact=True,physical_routes_tendons_support_transforms_and_controller_unchanged=True,reference_surface_scope='Left flexor digitorum superficialis source-boundary lift with bounded inferred source-coordinate correction. Source-lift topology, raw-parent ancestry and every display attachment weight remain fixed. Passive anatomical representation, not measured entheses or active forearm function.',source_characterization= str(CP),max_total_source_correction_m=c['max_vertex_displacement_from_source_mm']/1000,binding_changes=[],source_relative_area_change=c['source_relative_area_change'],source_relative_volume_change=c['source_relative_volume_change'],all_components_positive=True,qualification_limit='Source and25 sampled poses self-clear, no new exact parent neighbor pair identities against865 current or859 older targets. Existing external interfaces remain unresolved; no continuous intersection proof or full-anatomy completion claim.',operating_point='Unchanged2Nm/rad aggregate passive MTP, real flat-plane contact and coupled GPU physiology; no runtime performance change.')
(O/'source-resolution-bounds.json').write_text(json.dumps(bounds,indent=2,sort_keys=True)+'\n')
bp=M/'baseline/run-declaration.json';base=json.loads(bp.read_text());idx=base['argv'].index('numilab_human.resting_run');av=base['argv'][idx+1:]
for flag,value in [('--output',str(O/'baseline/native-run')),('--anatomy-receipt',str(receipt))]:av[av.index(flag)+1]=value
parser=argparse.ArgumentParser();rr.add_arguments(parser);args=parser.parse_args(av);native,assets=rr.command(args)
env=dict(x.split('=',1) for x in base['argv'][2:idx-2]);env['NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT']=str(args.output/'common-field-failure.json');env['DYLD_LIBRARY_PATH']=':'.join(str(Path(p).resolve()) for p in env['DYLD_LIBRARY_PATH'].split(':'))
runtime_rr=Path(env['PYTHONPATH'])/'numilab_human/resting_run.py';assert sha(runtime_rr)==sha(Path(rr.__file__))
argv=['/usr/bin/env','-i']+[k+'='+v for k,v in sorted(env.items())]+['/Users/n/numi-human-prep-venv-20261005/bin/python3.13','-m','numilab_human.resting_run']+av
launch=O/'launch_arm.py';launch.write_bytes((M/'launch_arm.py').read_bytes())
paths=[Path(__file__),bp,launch,LP,CP,SP,C,Path(rr.__file__),runtime_rr,Path(pc.__file__),O/'source-resolution-bounds.json'];pins={**assets,**{str(p):sha(p) for p in paths}}
decl=dict(schema='numi.human.native-smoke.run-declaration.v1',seconds=16.,accepted_steps=8000,capture_steps=[0,4767,5023,5599,6207,6815,7423,8000],immutable_assets=pins,argv=argv,launch_adapter=dict(path=str(launch),sha256=sha(launch)),owner_cli_preview=dict(native_argv=native,asset_sha256=assets,recorded_invocation_environment=rr.invocation_environment(env)),execution_host='numi',hardware='Apple M6,12GPU cores,16GiB,macOS27.0(26A425)',qualification='Integrated native left FDS132 inferred boundary increment. Other155 tissue rows and physical mass/routes/tendons/support/GPU physiology unchanged. Offline25-pose qualification passed; native geometry and physiology parity still require verification. No five-minute or whole-anatomy completion claim.')
(O/'baseline').mkdir();dp=O/'baseline/run-declaration.json';dp.write_text(json.dumps(decl,indent=2,sort_keys=True)+'\n');(O/'launch-guard.json').write_text(json.dumps(dict(launcher_sha256=sha(launch),arms=dict(baseline=dict(run_declaration_sha256=sha(dp),native_output=str(args.output)))),indent=2,sort_keys=True)+'\n');print(json.dumps(dict(path=str(O),tissue_sha256=sha(payload),pins=len(pins))),flush=True)
