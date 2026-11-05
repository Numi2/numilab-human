"""Bind compiled364 geometry to the exact351 runtime and solver controls."""
from pathlib import Path
import hashlib,json,shutil,subprocess
W=Path('/Users/numi/Work/human-integrated-gaps-20261104')
N=W/'flat-cap-native-365'
D=W/'flat-cap-batch-rebuild-364/proposal'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def put(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
target=N/'install-motion.plan.json'
assert target.exists() and not (N/'install-motion.execution.json').exists()
p=json.loads(target.read_text())
prior=N/'compiler-produced.plan.json'
if prior.exists():
 assert sha(prior)==sha(target),'Refuse replacing a changed plan after prior binding'
else:
 shutil.copy2(target,prior)
control_path=W/'skin-backtracking-witness-351/install-motion.plan.json'
control=json.loads(control_path.read_text())
ce=control['stages'][0]['environment'];generated=p['stages'][0]['environment']
# Only compiler-owned geometry and mass values may replace the matched controls.
geometry=lambda k:k.startswith('NUMI_H706_FRACTIONAL_') or k.startswith('NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_')
env={k:v for k,v in ce.items() if not geometry(k)}
env.update({k:v for k,v in generated.items() if geometry(k)})
oldroot=control['stages'][0]['argv'][4];newroot=str(N/'install-motion')
for k,v in list(env.items()):
 if v.startswith(oldroot):env[k]=newroot+v[len(oldroot):]
assert 'NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ' not in env
assert 'NUMI_MATTER_PIPELINE_ARCHIVE' not in env
stage=dict(control['stages'][0])
stage['argv']=list(stage['argv']);stage['argv'][4]=newroot;stage['environment']=env
p['stages']=[stage]
for k in ['native_binary','runtime_dependencies','ccd_repair','current_native_source_overlays','full_execution_source_pins','build_manifest','reused_immutable_preparation_dump','optional_driver_archive_disabled','routine_observation_policy']:
 p[k]=control[k]
p['runtime_launcher']=ref(W/'run_native_plan_compact.py')
# The307 template also lists runtime libraries inside its generic input inventory.
# Rebind only those runtime entries to the verified351 execution dependencies;
# keep the compiler-produced original and record the historical inventory delta.
rebindings={}
for path,r in {p['native_binary']['path']:p['native_binary'],**p['runtime_dependencies']}.items():
 if path in p['input_pins']:
  old=p['input_pins'][path];oldsha=old if isinstance(old,str) else old['sha256']
  assert sha(path)==r['sha256'],path
  if oldsha!=r['sha256']:
   rebindings[path]={'template':old,'matched351_execution_dependency':r}
   p['input_pins'][path]=r['sha256'] if isinstance(old,str) else r
p['runtime_inventory_rebinding']=rebindings

p['native_comparison_reference']=ref(control_path)
p['geometry_baseline']=ref(D/'report.json')
p['prior_mesh_native_reference']=ref(W/'witness-cluster-native-357/install-motion.plan.json')
p.pop('screened_reference_field',None)
p['purpose']='Matched full native Human scene test of 38 bounded outer-skin cap contractions and four retained deep-fat contractions; current351 equations, material bounds and solver controls unchanged.'
p['trial_scope']='32 roots at unchanged15.625us; initialization transient, not supported rest or sustained motion qualification.'
p['measurement_scope']='Accepted simulated time, body wall time, per-root nonlinear progress, deterministic element witness, force and strain comparisons.'
p['predeclared_checks']['reference']=ref(control_path)
p['predeclared_checks']['unchanged_mesh_control']=ref(W/'matched-ccd-baseline-335/install-motion.plan.json')
p['predeclared_checks']['prediction']='Must exceed15 accepted roots at unchanged351 physical gates; speed/fidelity require actual native measurements and can fail.'
p['predeclared_checks']['failure_rule']='Retain any physics rejection or fidelity mismatch. No loosened material J, collision, mass, force, strain or residual gates.'
p['compiled_plan_before_runtime_binding']=ref(prior)
p['runtime_binding_owner']=ref(__file__)
p['mesh_generation_owner']=ref('/Users/numi/Work/human-checkpoint-owner-20261105/tools/simplify_human_skin_fat_boundary.py')
p['mesh_compilation_receipts']=[ref(W/'compiler-flat-cap-batch-364'/name) for name in ['compiler-execution-prefix.json','compiler-execution-resumed-middle-367.json','compiler-execution-final.json']]
p['cap_geometric_bound']=ref(D/'cap-replay-geometric-bound.json')
p['analysis_owner']=ref('/Users/numi/Work/human-checkpoint-owner-20261105/tools/compare_human_native_mesh_trials.py')
p['fidelity_analysis_owners']={key:ref(W/name) for key,name in [('force_strain','compare_remeshed_native_310.py'),('matched_field','compare_matched_deformation_339.py')]}
differences={k:{'candidate':v,'control':ce.get(k)} for k,v in env.items() if not geometry(k) and v!=ce.get(k) and not str(v).startswith(newroot)}
assert not differences,differences
failures=[]
for x in [p['native_binary'],*p['runtime_dependencies'].values(),*p['required_transitive_input_pins']]:
 if not Path(x['path']).is_file() or sha(x['path'])!=x['sha256']:failures.append(x['path'])
for raw,x in p['input_pins'].items():
 if not Path(raw).is_file():failures.append(raw)
 elif isinstance(x,str) and sha(raw)!=x:failures.append(raw)
 elif isinstance(x,dict) and sha(raw)!=x['sha256']:failures.append(raw)
for k,v in env.items():
 if k.endswith('SHA256'):
  for pk in [k[:-7],k[:-7]+'_CSV',k[:-7]+'_ASSET',k[:-7]+'_NPZ',k[:-7]+'_JSON']:
   if pk in env and Path(env[pk]).is_file():
    if sha(env[pk])!=v:failures.append(k)
    break
assert not failures,failures
assert not any('reference-screened-interior-341/compiler/' in v for v in env.values())
put(target,p)
shutil.copy2(W/'run_native_plan_compact.py',N/'run_native_plan.py')
free=shutil.disk_usage(N).free
put(N/'current-runtime-prelaunch-checks.json',dict(plan=ref(target),runtime_matches351=True,non_geometry_control_differences=differences,input_failures=failures,available_bytes=free,minimum_launch_available_bytes=16*2**30,disk_launch_eligible=free>=16*2**30,physical_admission=False,native_steps=0,device=subprocess.run(['system_profiler','SPDisplaysDataType'],capture_output=True,text=True).stdout))
print('COMPILED365_BOUND_TO_MATCHED351_CONTROLS',free/2**30,'GiB available; launch requires >=16GiB')
