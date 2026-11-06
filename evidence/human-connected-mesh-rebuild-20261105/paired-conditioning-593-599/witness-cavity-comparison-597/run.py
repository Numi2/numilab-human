from pathlib import Path
import importlib.util,json,subprocess,sys
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');H=Path('/Users/numi/Work/human-checkpoint-owner-20261105');O=Path(__file__).resolve().parent
control=W/'early-state-native-572';candidate=W/'witness-cavity-native-596'
spec=importlib.util.spec_from_file_location('owner',H/'tools/compare_human_native_mesh_trials.py');owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
a=owner.summarize(candidate);b=owner.summarize(control)
cp=json.loads((candidate/'install-motion.plan.json').read_text());bp=json.loads((control/'install-motion.plan.json').read_text());diff=owner.control_differences(cp,bp)
assert not diff,diff
assert cp['runtime_dependencies']==bp['runtime_dependencies']
assert cp['required_transitive_input_pins'][:-1]==bp['required_transitive_input_pins']
assert len(cp['required_transitive_input_pins'])==len(bp['required_transitive_input_pins'])+1
before=bp['stages'][0]['environment'];after=cp['stages'][0]['environment']
new_keys={'NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ','NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256'}
assert set(after)-set(before)==new_keys
(O/'native-comparison.json').write_text(json.dumps(dict(candidate=a,matched_geometry_control=b,control_differences=diff,allowed_comparison_delta='two exact fixed-coordinate cavity compositions through the existing native initialization owner',runtime_dt_tolerances_budgets_unchanged=True,geometry_coordinates_unchanged=True,mechanical_topology_changed=True,physical_admission=False,analysis_owner=owner.ref(__file__)),indent=2,allow_nan=False)+'\n')
with (O/'matched-endpoint.log').open('w') as f:r=subprocess.run([sys.executable,str(W/'matched-endpoint-pair-509/compare_endpoint_channels.py'),str(control/'install-motion.stdout.log.gz'),str(candidate/'install-motion.stdout.log.gz'),str(O/'matched-endpoint-channels.json')],stdout=f,stderr=subprocess.STDOUT)
assert r.returncode==0
r=json.loads((O/'matched-endpoint-channels.json').read_text());force=[(k,c,n,x[n]) for k,y in r['regional_force_channels'].items() for c,x in y.items() for n in ['relative_resultant_difference_over_reference_nodal_l1','relative_nodal_l1_change'] if x[n] is None or x[n]>.01];strain=[(k,n,v) for k,x in r['regional_strain'].items() for n,v in x['relative_change'].items() if v is None or v>.05];phys=[(k,x) for k,x in r['coupled_trace'].items() if x['relative_max_abs_change'] is None or x['relative_max_abs_change']>.01]
summary=dict(force_failures=force,strain_failures=strain,strict_relative_physiology_outliers=phys,body_throughput_ratio=r['body_throughput_ratio'],body_wall_reduction_fraction=r['body_wall_reduction_fraction'],candidate_active_work=a['active_work_all_attempts'],control_active_work=b['active_work_all_attempts'],candidate_body_wall_s=a['body']['wall_s'],control_body_wall_s=b['body']['wall_s'],candidate_accepted_roots=a['accepted_roots'],candidate_body_rtf=a['body_real_time_factor'],full_human_qualified=False,single_short_pair_only=True)
(O/'gate-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(force_failures=len(force),strain_failures=len(strain),physiology_outliers=len(phys),**{k:v for k,v in summary.items() if k not in ['force_failures','strain_failures','strict_relative_physiology_outliers']}),indent=2))
