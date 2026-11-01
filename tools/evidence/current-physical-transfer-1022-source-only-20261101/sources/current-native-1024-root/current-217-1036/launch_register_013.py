from pathlib import Path
import hashlib, json, subprocess, sys, time
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
TASK=ROOT/'current-native-1024-root/current-217-1036'
PREV=ROOT/'current-native-1024-root/current-217-1034/actual-006/preflight-receipt.json'
PREFLIGHT=ROOT/'current-native-1024-root/current-217-1034/actual-006/preflight-receipt.json'
SIDE=ROOT/'current-registered-sidecars-1034/actual-004'
LI=SIDE/'launch-inputs.json'; REG=SIDE/'registered-sidecars'
OUT=TASK/'actual-013'; OUT.mkdir(parents=True,exist_ok=False)
PY=Path('/Users/numi/numi-human-prep-venv-20261005/bin/python')
WRAPPER=ROOT.parent/'run_geometry_serial.py'
DRIVER=TASK/'register_current_combined_volume_nodes_217_1036.py'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
old=json.loads(PREV.read_text()); li=json.loads(LI.read_text()); pre=json.loads(PREFLIGHT.read_text())
if pre.get('status')!='PASS_217_INPUT_READERS_AND_CURRENT_230_PROOF': raise SystemExit('fresh current-006 reader preflight not PASS')
proof=li['source_proof']; state=li['state']; crosswalk=li['crosswalk']; handoff=li['handoff']; shell=li['shell_receipt']; binding=li['shell_binding']
input_pins=old['input_pins']; role_inputs=li['current_surfaces']; side_receipt=REG/'receipt.json'
checks={str(DRIVER):sha(DRIVER),str(PREFLIGHT):sha(PREFLIGHT),str(LI):sha(LI),str(side_receipt):sha(side_receipt)}
for group in (proof,state,crosswalk,handoff,shell,binding,*role_inputs.values()): checks[group['path']]=group['sha256']
for group in input_pins.values(): checks[group['path']]=group['sha256']
for p,expected in checks.items():
 actual=sha(p)
 if actual!=expected: raise SystemExit(f'PIN_MISMATCH {p}: expected {expected}, actual {actual}')
argv=[str(PY),str(WRAPPER),str(PY),str(DRIVER),
 '--node-map',input_pins['node_map']['path'],'--node',input_pins['node']['path'],
 '--ele',input_pins['ele']['path'],'--labels',input_pins['labels']['path'],
 '--parent-base',input_pins['parent_base']['path'],'--registered-dir',str(REG),
 '--poses',input_pins['poses']['path'],'--maximum-parent-distance-m','0.02',
 '--ambiguity-tolerance-m','1e-10','--capacity-profile','full-reference-500k-nodes-2m-cells',
 '--current-typed-sidecars']
for role in ('O','I','D'):
 x=role_inputs[role]; argv += ['--current-typed-role',role,x['path'],x['sha256']]
argv += ['--current-230-source-proof',proof['path'],proof['sha256'],
 '--current-230-receipt-sha256',sha(side_receipt),'--current-volume-state',state['path'],state['sha256'],
 '--output',str(OUT/'candidate-node-map.csv'),'--receipt',str(OUT/'receipt.json')]
launch={'schema':'numi.h706.current-217-rich-registration-launch.v1','started_unix':time.time(),
 'driver':{'path':str(DRIVER),'sha256':sha(DRIVER)},'preflight':{'path':str(PREFLIGHT),'sha256':sha(PREFLIGHT)},
 'input_manifest':{'path':str(LI),'sha256':sha(LI)},'sidecar_receipt':{'path':str(side_receipt),'sha256':sha(side_receipt)},
 'argv':argv,'authenticated_pins':checks,'physical_admission':False,'native_steps':0}
(OUT/'launch-inputs.json').write_text(json.dumps(launch,sort_keys=True,indent=2)+'\n')
proc=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
(OUT/'stdout.log').write_text(proc.stdout); (OUT/'stderr.log').write_text(proc.stderr)
execution={'schema':'numi.h706.current-217-rich-registration-execution.v1','returncode':proc.returncode,
 'ended_unix':time.time(),'stdout_sha256':sha(OUT/'stdout.log'),'stderr_sha256':sha(OUT/'stderr.log'),
 'candidate_path':str(OUT/'candidate-node-map.csv') if (OUT/'candidate-node-map.csv').exists() else None,
 'candidate_sha256':sha(OUT/'candidate-node-map.csv') if (OUT/'candidate-node-map.csv').exists() else None,
 'receipt_path':str(OUT/'receipt.json') if (OUT/'receipt.json').exists() else None,
 'receipt_sha256':sha(OUT/'receipt.json') if (OUT/'receipt.json').exists() else None,
 'physical_admission':False,'native_steps':0}
(OUT/'execution.json').write_text(json.dumps(execution,sort_keys=True,indent=2)+'\n')
print(proc.stdout,end='')
if proc.stderr: print(proc.stderr,file=sys.stderr,end='')
print(json.dumps(execution,sort_keys=True))
raise SystemExit(proc.returncode)
