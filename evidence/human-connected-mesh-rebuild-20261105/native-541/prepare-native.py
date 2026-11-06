from pathlib import Path
import json,hashlib,shutil,sys
W=Path('/Users/numi/Work/human-integrated-gaps-20261104')
O=W/'joint-response-native-546';O.mkdir(exist_ok=False)
raw=W/'joint-response-plan-545/install-motion.plan.json'
controlpath=W/'targeted-fat-native-499/install-motion.plan.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
p=json.loads(raw.read_text());control=json.loads(controlpath.read_text())
generated=p['stages'][0]['environment'];ce=control['stages'][0]['environment']
geometry=lambda k:k.startswith('NUMI_H706_FRACTIONAL_') or k.startswith('NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_')
e={k:v for k,v in ce.items() if not geometry(k)};e.update({k:v for k,v in generated.items() if geometry(k)})
stage=json.loads(json.dumps(control['stages'][0]));oldroot=stage['argv'][4];newroot=str(O/'install-motion');stage['argv'][4]=newroot
for k,v in list(e.items()):
 if v.startswith(oldroot):e[k]=newroot+v[len(oldroot):]
stage['environment']=e;p['stages']=[stage]
for k in ['native_binary','runtime_dependencies','ccd_repair','current_native_source_overlays','full_execution_source_pins','build_manifest','reused_immutable_preparation_dump','optional_driver_archive_disabled','routine_observation_policy','runtime_precision_binding','replay_policy_binding','precision_repair','reference_edge_certificate','endpoint_potential_change']:
 if k in control:p[k]=control[k]
for path,r in {p['native_binary']['path']:p['native_binary'],**p['runtime_dependencies']}.items():
 if path in p['input_pins']:
  old=p['input_pins'][path];p['input_pins'][path]=r['sha256'] if isinstance(old,str) else r
p['runtime_launcher']=ref(W/'endpoint-potential-native-481/run_native_plan.py')
p['compiled_plan_before_runtime_binding']=ref(raw)
p['native_comparison_reference']=ref(controlpath)
p['geometry_baseline']=ref(W/'multifield-response-interior-541/proposal/report.json')
p['cap_geometric_bound']=ref(W/'multifield-response-interior-541/proposal/report.json')
p['analysis_owner']=ref('/Users/numi/Work/human-checkpoint-owner-20261105/tools/compare_human_native_mesh_trials.py')
p['prior_mesh_native_reference']=ref(controlpath)
p['measurement_scope']='Accepted physical time, GPU/root work, retained numerical certificates and coupled trace; initialization transient only.'
p['mesh_generation_owner']=ref('/Users/numi/Work/human-checkpoint-owner-20261105/tools/simplify_human_skin_fat_boundary.py')
p['mesh_compilation_receipts']=[ref(W/'joint-response-compiler-544'/n) for n in ['compiler-execution.json','compiler-execution-middle.json','compiler-execution-mass.json','compiler-execution-final.json']]
p['runtime_binding_owner']=ref(__file__)
p['purpose']='Complete native Human scene test of jointly screened fixed-boundary skin/fat conditioning541. Same499 equations, contact, material laws, solver controls, timestep and acceptance gates.'
p['trial_scope']='64 roots at unchanged15.625us; release initialization transient, not supported rest or sustained physiology qualification.'
p['expected_accepted_steps']=64
p['predeclared_checks']['reference']=ref(controlpath);p['predeclared_checks']['accepted_roots']=64
p['predeclared_checks']['prediction']='Improve conditioning through interior-only connectivity repairs, preserving exact surfaces, local conservative mass and all physics/solver controls. Compare64 accepted roots with495/499; no throughput gain assumed.'
p['predeclared_checks']['failure_rule']='Retain failed or slower result and deterministic witnesses; do not relax physics or fidelity gates.'
sys.path.insert(0,'/Users/numi/Work/human-checkpoint-owner-20261105/tools')
from compare_human_native_mesh_trials import control_differences
assert control_differences(p,control)==[],control_differences(p,control)
fail=[]
for r in [p['native_binary'],*p['runtime_dependencies'].values(),*p['required_transitive_input_pins']]:
 if not Path(r['path']).is_file() or sha(r['path'])!=r['sha256']:fail.append(r['path'])
for path,r in p['input_pins'].items():
 if not Path(path).is_file() or sha(path)!=(r if isinstance(r,str) else r['sha256']):fail.append(path)
for k,v in e.items():
 if k.endswith('SHA256'):
  for pk in [k[:-7],k[:-7]+'_CSV',k[:-7]+'_ASSET',k[:-7]+'_NPZ',k[:-7]+'_JSON']:
   if pk in e and Path(e[pk]).is_file():
    if sha(e[pk])!=v:fail.append(k)
    break
assert not fail,fail
launcher_source=W/'endpoint-potential-native-481/run_native_plan.py'
launcher_text=launcher_source.read_text()
assert launcher_text.count('stage_name=sys.argv[1]')==1
launcher_text=launcher_text.replace('stage_name=sys.argv[1]',"stage_name=(sys.argv[1] if len(sys.argv)>1 else 'install-motion')")
compile(launcher_text,'run_native_plan.py','exec')
(O/'run_native_plan.py').write_text(launcher_text)
p['runtime_launcher_owner']=ref(launcher_source)
p['runtime_launcher']=ref(O/'run_native_plan.py')
p['runtime_launcher_extension']='Only default missing stage argument to install-motion; all owner checks and scheduling unchanged.'
(O/'install-motion.plan.json').write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')
(O/'prelaunch-checks.json').write_text(json.dumps(dict(non_geometry_control_differences=[],input_failures=fail,available_bytes=shutil.disk_usage(O).free,physical_admission=False,native_steps=0),indent=2)+'\n')
print('CURRENT541_MATCHED499_PLAN_READY',O,flush=True)
