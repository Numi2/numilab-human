from pathlib import Path
import json,hashlib,subprocess,time,csv
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
D=P/'conforming-fit-006-native-assets-545/run-002-operational-reference'
D.mkdir(parents=True,exist_ok=False)
plan=json.loads((P/'conforming-fit-006-native-assets-545/run-001/plan.json').read_text())
cmd=plan['command'].copy()
script=P/'operational-reference-map-501/export_refined_coarse_registered_volume_433.py'
cmd[1]=str(script);cmd[cmd.index('--out')+1]=str(D/'registered-assets');cmd.append('--candidate-operational-reference')
plan['command']=cmd;plan['export_script_sha256']=hashlib.sha256(script.read_bytes()).hexdigest()
plan['registration_contract']='existing54column separated source330 ancestry and exact candidate operational self registration; conforming530proof still required'
(D/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
t=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True)
(D/'export.stdout').write_text(r.stdout);(D/'export.stderr').write_text(r.stderr)
out={'exit_code':r.returncode,'wall_s':time.monotonic()-t,'physical_admission':False,'runtime':0}
if r.returncode==0:
 p=D/'registered-assets/volume/combined-node-map.csv'
 with p.open(newline='') as f:
  reader=csv.DictReader(f);rows=list(reader)
 out['map_column_count']=len(reader.fieldnames);out['map_row_count']=len(rows)
 assert len(reader.fieldnames)==54
 receipt=json.loads((D/'registered-assets/receipt.json').read_text())
 assert receipt['registration_domain']=='candidateOperationalReference'
 out['registration_domain']=receipt['registration_domain']
 out['combined_node_map_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
(D/'result.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out));raise SystemExit(r.returncode)
