from pathlib import Path
import json,hashlib,shutil,sys
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');O=Path(__file__).resolve().parent
sys.path.insert(0,'/Users/numi/Work/human-checkpoint-owner-20261105/tools')
from compare_human_native_mesh_trials import control_differences
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
plans={}
for name,source in [('control','endpoint-potential-native-481'),('candidate','targeted-fat-native-499')]:
 d=O/name;d.mkdir(exist_ok=False);p=json.loads((W/source/'install-motion.plan.json').read_text());stage=p['stages'][0];root=stage['argv'][4];newroot=str(d/'install-motion');stage['argv'][4]=newroot
 argv=stage['argv'];argv[argv.index('--muscle-step-count')+1]='58'
 stage['environment']['NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP']='58'
 p['audit_horizon_rebind']={'old_last_step':64,'new_last_step':58,'scope':'Observation window truncated to requested horizon. Same observation and physical policy on all58 executed steps.'}
 for k,v in list(stage['environment'].items()):
  if v.startswith(root):stage['environment'][k]=newroot+v[len(root):]
 p['purpose']='Matched successful58-root endpoint capture for targeted495 force and strain diagnostics. No physical/numerical policy changes.'
 p['trial_scope']='Matched58-step release-initialization endpoints, not supported rest or sustained physiology.';p['expected_accepted_steps']=58;p['predeclared_checks']['accepted_roots']=58
 p['predeclared_checks']['prediction']='Both existing certified58-step prefixes complete and emit matched-time full FEM captures. Preserve1percent force and5percent deformation gates; report failures without relaxation. Material energy remains unmeasured by this exporter.'
 p['predeclared_checks']['failure_rule']='No missing endpoints substituted; failed or mismatched trial invalidates endpoint comparison.'
 p['matched_endpoint_source_plan']=ref(W/source/'install-motion.plan.json');p['runtime_binding_owner']=ref(Path(__file__))
 p['native_comparison_reference']=ref(O/'control/install-motion.plan.json') if name=='candidate' else ref(W/source/'install-motion.plan.json')
 for r in [p['native_binary'],*p['runtime_dependencies'].values(),*p['required_transitive_input_pins']]:assert Path(r['path']).is_file() and sha(r['path'])==r['sha256'],r['path']
 for path,r in p['input_pins'].items():assert Path(path).is_file() and sha(Path(path))==(r if isinstance(r,str) else r['sha256']),path
 (d/'install-motion.plan.json').write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')
 shutil.copy2(W/'endpoint-potential-native-481/run_native_plan.py',d/'run_native_plan.py');plans[name]=p
assert not control_differences(plans['candidate'],plans['control'])
(O/'prelaunch.json').write_text(json.dumps({'matched_non_geometry_controls':[],'horizon_steps':58,'physical_admission':False,'bounded_next_check':'Existing endpoint channels and regional Green strain; energy and complete force-owner coverage remain open.'},indent=2)+'\n')
print('MATCHED58_ENDPOINT_PAIR_READY',flush=True)
